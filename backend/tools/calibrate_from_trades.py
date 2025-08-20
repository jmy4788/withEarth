# tools/calibrate_from_trades.py
from __future__ import annotations
from dataclasses import dataclass
from typing import List, Optional, Iterable, Dict, Any
from pathlib import Path
from datetime import datetime, timezone, timedelta
import csv
import os

# --- flexible imports (root or helpers.*) ---
try:
    from data_fetch import fetch_ohlcv  # 1m 구간 조회가 중요
except Exception:
    from helpers.data_fetch import fetch_ohlcv  # type: ignore

try:
    from utils import LOG_DIR, log_event
except Exception:
    from helpers.utils import LOG_DIR, log_event  # type: ignore

# ---- Config fallbacks ----
DEFAULT_LOG_DIR = LOG_DIR if isinstance(LOG_DIR, (str, Path)) else "./logs"

@dataclass
class Sample:
    prob: float
    label: int   # 1=TP선행, 0=SL선행/미도달
    symbol: str

def _iso_to_ms(ts: str) -> Optional[int]:
    if not ts:
        return None
    # ISO8601 다양한 포맷 수용
    for cand in (ts, ts.replace("Z", "+00:00")):
        try:
            dt = datetime.fromisoformat(cand)
            if dt.tzinfo is None:
                dt = dt.replace(tzinfo=timezone.utc)
            return int(dt.timestamp() * 1000)
        except Exception:
            continue
    return None

def _first_touch_times(df, *, tp: float, sl: float, direction: str) -> Dict[str, Optional[int]]:
    """
    1분봉 고가/저가를 사용해 선행 터치 시각(ms)을 판정.
    - 같은 봉에서 TP/SL이 동시에 터치 가능한 모호 케이스는 'ambiguous'로 간주(샘플 스킵 권장).
    """
    import pandas as pd  # lazy
    if df is None or len(df) == 0:
        return {"tp": None, "sl": None, "ambiguous": None}
    H = pd.to_numeric(df["high"], errors="coerce")
    L = pd.to_numeric(df["low"], errors="coerce")
    # timestamp는 close_time(UTC ISO)로 들어옴
    T = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)

    for i in range(len(df)):
        hi = float(H.iloc[i]); lo = float(L.iloc[i])
        t_ms = int(T.iloc[i].timestamp() * 1000) if pd.notna(T.iloc[i]) else None
        if direction == "long":
            tp_hit = (hi >= tp)
            sl_hit = (lo <= sl)
        else:  # short
            tp_hit = (lo <= tp)
            sl_hit = (hi >= sl)
        if tp_hit and sl_hit:
            return {"tp": t_ms, "sl": t_ms, "ambiguous": t_ms}
        if tp_hit:
            return {"tp": t_ms, "sl": None, "ambiguous": None}
        if sl_hit:
            return {"tp": None, "sl": t_ms, "ambiguous": None}
    return {"tp": None, "sl": None, "ambiguous": None}

def _row_prob(row: Dict[str, Any]) -> Optional[float]:
    for key in ("prob", "calibrated_prob", "p"):
        v = row.get(key)
        if v is None or v == "":
            continue
        try:
            x = float(v)
            if 0.0 <= x <= 1.0:
                return x
        except Exception:
            continue
    return None

def _row_horizon(row: Dict[str, Any], default_min: int) -> int:
    v = row.get("horizon_min")
    try:
        hv = int(v)
        return hv if hv > 0 else default_min
    except Exception:
        return int(default_min)

def load_samples(
    trades_csv: Path | str,
    horizon_min: int,
    only_symbols: Optional[Iterable[str]] = None,
    *,
    max_rows: Optional[int] = None,
) -> List[Sample]:
    """
    CSV→(prob,label) 샘플 생성.
    - ambiguous(동일 봉에서 TP,SL 터치) → 보수적으로 스킵
    - neither(호라이즌 내 미터치) → label=0(보수적)
    """
    path = Path(trades_csv or (Path(DEFAULT_LOG_DIR) / "trades.csv"))
    if not path.exists():
        return []

    only = {s.upper() for s in (only_symbols or []) if str(s).strip()}
    out: List[Sample] = []

    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        rows = list(r)

    if not rows:
        return []

    if isinstance(max_rows, int) and max_rows > 0:
        rows = rows[-max_rows:]

    for row in rows:
        try:
            sym = str(row.get("symbol", "")).upper().strip()
            if only and sym not in only:
                continue

            side = str(row.get("side", "")).lower().strip()  # 'long' | 'short'
            if side not in ("long", "short"):
                continue

            ts_entry = _iso_to_ms(row.get("timestamp", ""))
            entry = float(row.get("entry", 0.0) or 0.0)
            tp = float(row.get("tp", 0.0) or 0.0)
            sl = float(row.get("sl", 0.0) or 0.0)
            if not ts_entry or entry <= 0 or tp <= 0 or sl <= 0:
                continue

            # 확률
            prob = _row_prob(row)
            if prob is None:
                # prob가 없으면 데이터 오염 방지를 위해 스킵(캘리브레이션 의미 퇴색 방지)
                continue

            # 호라이즌: 행 단위 우선 → 없으면 파라미터 기본
            hz_min = _row_horizon(row, default_min=int(horizon_min))
            start_ms = int(ts_entry)
            end_ms = int(ts_entry + hz_min * 60_000)

            # 1분봉으로 판정(더 정밀)
            df = fetch_ohlcv(sym, interval="1m", limit=hz_min + 5, start_ms=start_ms, end_ms=end_ms)

            touch = _first_touch_times(df, tp=tp, sl=sl, direction=side)
            if touch.get("ambiguous") is not None:
                # 같은 봉 내 동시 터치는 방향성 불확실 → 샘플 제외
                continue

            if touch.get("tp") is not None:
                label = 1
            elif touch.get("sl") is not None:
                label = 0
            else:
                # 호라이즌 내 미터치 → 보수적으로 패배(0) 처리
                label = 0

            out.append(Sample(prob=float(prob), label=int(label), symbol=sym))
        except Exception:
            continue

    try:
        log_event("calibrate.samples_built", total=len(out))
    except Exception:
        pass
    return out

__all__ = ["Sample", "load_samples"]
