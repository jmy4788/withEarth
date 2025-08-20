#!/usr/bin/env python3
"""
scripts/report.py — trading journal & decision diagnostics (2025-08-19)

- Summarize trades from logs/trades.csv
- (Best-effort) Join decisions from logs/payloads/*_decision.json (if available)
- Optionally fit a simple calibration curve and save to logs/calibration.json
"""
from __future__ import annotations
import argparse, json, os, sys, glob
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

LOG_DIR = os.getenv("LOG_DIR", "./logs")
TRADES_CSV = str(Path(LOG_DIR) / "trades.csv")
PAYLOAD_DIR = str(Path(LOG_DIR) / "payloads")
CALIB_PATH = os.getenv("PROB_CALIBRATION_PATH", str(Path(LOG_DIR) / "calibration.json"))

@dataclass
class Summary:
    total: int = 0
    closed: int = 0
    open: int = 0
    wins: int = 0
    losses: int = 0
    pnl_sum: float = 0.0
    win_rate: float = 0.0
    avg_pnl: float = 0.0

def _read_trades() -> pd.DataFrame:
    if not Path(TRADES_CSV).exists():
        return pd.DataFrame(columns=["timestamp","symbol","side","qty","entry","tp","sl","exit","pnl","status","id"])
    df = pd.read_csv(TRADES_CSV)
    for c in ["entry","tp","sl","exit","pnl"]:
        if c in df.columns:
            df[c] = pd.to_numeric(df[c], errors="coerce")
    return df

def _scan_decisions() -> pd.DataFrame:
    # Expect files: logs/payloads/YYYYMMDD/*_decision.json
    if not Path(PAYLOAD_DIR).exists():
        return pd.DataFrame(columns=["timestamp","symbol","prob","direction"])
    files = []
    for d in glob.glob(str(Path(PAYLOAD_DIR) / "*" )):
        files.extend(glob.glob(str(Path(d) / "*_decision.json")))
    rows: List[Dict[str, Any]] = []
    for fp in sorted(files):
        try:
            # Timestamp prefix in name: HHMMSS_symbol_decision.json (see helpers/predictor._dump_debug)
            ts_part = Path(fp).name.split("_", 1)[0]
            # date from parent dir
            date_part = Path(fp).parent.name
            ts = None
            try:
                ts = datetime.strptime(date_part + ts_part, "%Y%m%d%H%M%S").replace(tzinfo=timezone.utc).isoformat()
            except Exception:
                ts = None
            obj = json.loads(Path(fp).read_text(encoding="utf-8"))
            symbol = obj.get("pair") or obj.get("symbol") or ""
            prob = float(obj.get("prob", np.nan))
            direction = str(obj.get("direction",""))
            rows.append({"timestamp": ts, "symbol": symbol, "prob": prob, "direction": direction, "file": Path(fp).name})
        except Exception:
            continue
    return pd.DataFrame(rows)

def _summarize(df: pd.DataFrame) -> Summary:
    if df is None or len(df) == 0:
        return Summary()
    total = len(df)
    closed = int((df["status"].str.lower() == "closed").sum()) if "status" in df.columns else 0
    open_cnt = int((df["status"].str.lower() == "open").sum()) if "status" in df.columns else 0
    pnl_sum = float(df.get("pnl", pd.Series(dtype=float)).sum()) if "pnl" in df.columns else 0.0
    wins = int((df.get("pnl", 0) > 0).sum()) if "pnl" in df.columns else 0
    losses = int((df.get("pnl", 0) < 0).sum()) if "pnl" in df.columns else 0
    win_rate = float(wins / max(1, wins + losses)) if (wins + losses) > 0 else 0.0
    avg_pnl = float(pnl_sum / max(1, total))
    return Summary(total, closed, open_cnt, wins, losses, pnl_sum, win_rate, avg_pnl)

def _join_decisions(trades: pd.DataFrame, decs: pd.DataFrame) -> pd.DataFrame:
    if trades is None or len(trades) == 0 or decs is None or len(decs) == 0:
        return pd.DataFrame(columns=["timestamp","symbol","prob","pnl"])
    # Approximate join: for each trade, pick the latest decision for same symbol not later than trade time (+2m tolerance)
    try:
        trades = trades.copy()
        trades["t_dt"] = pd.to_datetime(trades["timestamp"], errors="coerce", utc=True)
        decs = decs.copy()
        decs["d_dt"] = pd.to_datetime(decs["timestamp"], errors="coerce", utc=True)
        out_rows = []
        for i, r in trades.iterrows():
            sym = str(r.get("symbol","")).upper()
            t = r.get("t_dt")
            if pd.isna(t): 
                continue
            df_sym = decs[decs["symbol"].str.upper() == sym]
            if len(df_sym) == 0:
                continue
            df_sym = df_sym[df_sym["d_dt"] <= t + pd.Timedelta(minutes=2)].sort_values("d_dt")
            if len(df_sym) == 0:
                continue
            last = df_sym.iloc[-1]
            out_rows.append({
                "timestamp": r.get("timestamp"), "symbol": sym,
                "prob": float(last.get("prob", np.nan)),
                "direction": str(last.get("direction","")),
                "pnl": float(r.get("pnl", 0.0)),
                "status": str(r.get("status",""))
            })
        return pd.DataFrame(out_rows)
    except Exception:
        return pd.DataFrame(columns=["timestamp","symbol","prob","pnl"])

def _fit_calibration(joined: pd.DataFrame, min_samples: int = 150) -> Optional[Dict[str, List[float]]]:
    joined = joined.dropna(subset=["prob"]).copy()
    if len(joined) < min_samples:
        return None
    # Label: pnl>0 → 1 else 0
    joined["label"] = (joined["pnl"] > 0).astype(int)
    p = np.clip(joined["prob"].astype(float).to_numpy(), 0.0, 1.0)
    y = joined["label"].astype(int).to_numpy()
    # 10-bin isotonic-ish monotone smoothing
    bins = 10
    edges = np.linspace(0, 1, bins + 1)
    xs, ys = [], []
    for i in range(bins):
        lo, hi = edges[i], edges[i+1]
        mask = (p >= lo) & (p < hi if i < bins - 1 else p <= hi)
        if mask.sum() < 5:
            continue
        xs.append(float(p[mask].mean()))
        ys.append(float(y[mask].mean()))
    if len(xs) < 3:
        return None
    # monotone
    mono = []
    last = 0.0
    for v in ys:
        last = max(last, float(v))
        mono.append(last)
    return {"bin_edges": xs, "bin_means": mono}

def _save_calibration(obj: Dict[str, List[float]]) -> None:
    try:
        p = Path(CALIB_PATH); p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
        print(f"[+] calibration saved to {p}")
    except Exception as e:
        print(f"[!] failed to save calibration: {e}")

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--save-calibration", action="store_true", help="fit and save isotonic-like calibration to logs/calibration.json")
    args = ap.parse_args()

    trades = _read_trades()
    decs = _scan_decisions()
    summary = _summarize(trades)

    print("== Trades summary ==")
    print(f"total={summary.total} closed={summary.closed} open={summary.open} wins={summary.wins} losses={summary.losses}")
    print(f"win_rate={summary.win_rate:.3f} pnl_sum={summary.pnl_sum:.2f} avg_pnl/trade={summary.avg_pnl:.4f}")
    if len(trades) > 0:
        print("\nLast 5 trades:")
        print(trades.tail(5).to_string(index=False))

    joined = _join_decisions(trades, decs)
    if len(joined) > 0:
        print("\n== Decision vs Outcome ==")
        print(joined.tail(10).to_string(index=False))
        if args.save_calibration:
            cal = _fit_calibration(joined)
            if cal:
                _save_calibration(cal)
            else:
                print("[i] not enough samples for calibration")

if __name__ == "__main__":
    main()
