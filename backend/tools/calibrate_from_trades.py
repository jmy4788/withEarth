#!/usr/bin/env python3
from __future__ import annotations
import argparse, csv, os, sys
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import List, Optional, Tuple

import numpy as np
import pandas as pd

# 내부 모듈
# PYTHONPATH에 프로젝트 루트가 잡혀 있어야 합니다. (예: `python -m tools.calibrate_from_trades`)
from helpers.data_fetch import fetch_ohlcv  # 시간범위 인자 지원 필요 :contentReference[oaicite:8]{index=8}
from helpers.calibration import ProbCalibrator  # 보정 곡선 저장/적용 :contentReference[oaicite:9]{index=9}

LOG_DIR = Path(os.getenv("LOG_DIR", "./logs"))
TRADES_CSV = LOG_DIR / "trades.csv"
HORIZON_MIN_DEFAULT = int(os.getenv("HORIZON_MIN", "30"))  # signals와 일치 권장 :contentReference[oaicite:10]{index=10}

@dataclass
class Sample:
    prob: float
    label: int
    symbol: str
    ts_iso: str

def _parse_iso(ts: str) -> Optional[datetime]:
    if not ts: return None
    try:
        return datetime.fromisoformat(ts)
    except Exception:
        try:
            return datetime.fromisoformat(ts.replace("Z", "+00:00"))
        except Exception:
            return None

def _ms(dt: datetime) -> int:
    return int(dt.timestamp() * 1000)

def _label_from_ohlcv(df: pd.DataFrame, side: str, tp: float, sl: float,
                      t0_ms: int, t1_ms: int) -> Optional[int]:
    """
    캔들 타임스탬프는 '종가 시각(closetime)' 기준(helpers/data_fetch.py 설계)입니다. :contentReference[oaicite:11]{index=11}
    - 동일 캔들에서 TP/SL 모두 터치 시 모호 → None 반환(샘플 제외)
    - 호라이즌 내 아무것도 터치 안하면 0(실패) 반환
    """
    if df is None or df.empty: return None
    ts = pd.to_datetime(df["timestamp"], utc=True, errors="coerce")
    ts_ms = (ts.view("int64") // 1_000_000).astype("int64")
    mask = (ts_ms >= t0_ms) & (ts_ms <= t1_ms)
    sub = df.loc[mask]
    if sub.empty:
        return None
    highs = pd.to_numeric(sub["high"], errors="coerce").values
    lows  = pd.to_numeric(sub["low"], errors="coerce").values

    if side == "long":
        for hi, lo in zip(highs, lows):
            hit_tp = (hi >= tp)
            hit_sl = (lo <= sl)
            if hit_tp and hit_sl:
                return None  # ambiguous
            if hit_sl:
                return 0
            if hit_tp:
                return 1
        return 0
    else:
        for hi, lo in zip(highs, lows):
            hit_sl = (hi >= sl)
            hit_tp = (lo <= tp)
            if hit_tp and hit_sl:
                return None  # ambiguous
            if hit_sl:
                return 0
            if hit_tp:
                return 1
        return 0

def _maybe_float(x) -> Optional[float]:
    try:
        return float(x)
    except Exception:
        return None

def load_samples(csv_path: Path, horizon_min: int,
                 only_symbols: Optional[List[str]] = None) -> List[Sample]:
    rows: List[Sample] = []
    if not csv_path.exists():
        print(f"[WARN] trades.csv not found at {csv_path}")
        return rows

    with open(csv_path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            sym = str(row.get("symbol","")).upper()
            if only_symbols and sym not in set(s.upper() for s in only_symbols):
                continue
            side = str(row.get("side","")).lower()
            ts = _parse_iso(row.get("timestamp",""))
            entry = _maybe_float(row.get("entry"))
            tp = _maybe_float(row.get("tp"))
            sl = _maybe_float(row.get("sl"))
            prob = (_maybe_float(row.get("prob")) or _maybe_float(row.get("prob_at_entry")))

            # 필수 필드 체크
            if ts is None or entry is None or tp is None or sl is None or prob is None:
                continue

            # 너무 최근인 경우(호라이즌이 아직 안 지남) 제외
            if datetime.now(tz=timezone.utc) < ts + timedelta(minutes=horizon_min):
                continue

            # 캔들 가져오기(버퍼 2바 포함)
            interval = "5m"
            int_ms = 300_000
            start_ms = _ms(ts) - 2*int_ms
            end_ms   = _ms(ts + timedelta(minutes=horizon_min)) + 2*int_ms
            df = fetch_ohlcv(sym, interval=interval, limit=500, start_ms=start_ms, end_ms=end_ms)  # :contentReference[oaicite:12]{index=12}
            label = _label_from_ohlcv(df, side=side, tp=tp, sl=sl, t0_ms=_ms(ts), t1_ms=_ms(ts + timedelta(minutes=horizon_min)))
            if label is None:
                continue
            rows.append(Sample(prob=float(np.clip(prob,0,1)), label=int(label), symbol=sym, ts_iso=ts.isoformat()))
    return rows

def main():
    ap = argparse.ArgumentParser(description="Build probability calibration from trades.csv")
    ap.add_argument("--csv", type=str, default=str(TRADES_CSV), help="path to trades.csv")
    ap.add_argument("--symbols", type=str, default="", help="comma-separated symbols filter, e.g., BTCUSDT,ETHUSDT")
    ap.add_argument("--horizon-min", type=int, default=HORIZON_MIN_DEFAULT)
    ap.add_argument("--min-samples", type=int, default=int(os.getenv("CALIB_MIN_SAMPLES", "150")))
    ap.add_argument("--bins", type=int, default=int(os.getenv("CALIB_BINS", "10")))
    args = ap.parse_args()

    only_symbols = [s.strip().upper() for s in args.symbols.split(",") if s.strip()] if args.symbols else None
    samples = load_samples(Path(args.csv), args.horizon_min, only_symbols=only_symbols)
    print(f"[INFO] collected samples: {len(samples)} (min required: {args.min_samples})")

    if len(samples) < args.min_samples:
        print("[WARN] not enough samples; aborting calibration.")
        return

    probs = [s.prob for s in samples]
    labels = [s.label for s in samples]

    calib = ProbCalibrator(bins=args.bins, min_samples=args.min_samples)  # :contentReference[oaicite:13]{index=13}
    ok = calib.fit_from_arrays(probs, labels)
    if not ok:
        print("[WARN] calibration fit failed (insufficient or unsuitable data).")
        return
    calib.save()
    print("[OK] calibration saved.")
    # 간략 리포트
    # bin_edges는 x좌표(평균 예측확률), bin_means는 경험적 승률(단조 보정 후)
    xs, ys = calib.bin_edges, calib.bin_means
    print("=== Reliability Table ===")
    for x, y in zip(xs, ys):
        print(f"  p̂≈{x:.2f} -> P(TP first)≈{y:.2f}")

if __name__ == "__main__":
    sys.exit(main())
