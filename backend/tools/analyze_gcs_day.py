# tools/analyze_gcs_day.py
from __future__ import annotations
import argparse, io, sys, os
from dataclasses import dataclass
from typing import Dict, List, Optional, Tuple
from datetime import datetime, timezone
import pandas as pd
import numpy as np

# ---- GCS ----
try:
    from google.cloud import storage
    _GCS_OK = True
except Exception:
    _GCS_OK = False

# ---- Defaults (fees & env fallbacks) ----
FEE_MAKER_BPS = float(os.getenv("FEE_MAKER_BPS", "2.0"))
FEE_TAKER_BPS = float(os.getenv("FEE_TAKER_BPS", "4.0"))

def _to_df(rows: List[Dict]) -> pd.DataFrame:
    try:
        df = pd.DataFrame(rows)
    except Exception:
        df = pd.DataFrame()
    return df

def _read_csv_bytes(b: bytes) -> pd.DataFrame:
    try:
        return pd.read_csv(io.StringIO(b.decode("utf-8")), dtype=str)
    except Exception:
        return pd.DataFrame()

def _list_day_objects(client: "storage.Client", bucket: str, prefix: str, dataset: str, yyyymmdd: str) -> List[str]:
    # objects under: {prefix}/{dataset}/{YYYYMMDD}/HHMMSS_*.csv
    it = client.list_blobs(bucket, prefix=f"{prefix}/{dataset}/{yyyymmdd}/")
    return [obj.name for obj in it]

def _load_day_csvs(client: "storage.Client", bucket: str, keys: List[str]) -> pd.DataFrame:
    frames = []
    for k in keys:
        try:
            blob = client.bucket(bucket).blob(k)
            b = blob.download_as_bytes()
            df = _read_csv_bytes(b)
            if not df.empty:
                df["_gcs_path"] = k
                frames.append(df)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    out = pd.concat(frames, ignore_index=True)
    # normalize column names (strip spaces)
    out.columns = [str(c).strip() for c in out.columns]
    return out

def _ensure_cols(df: pd.DataFrame, cols: List[str]) -> pd.DataFrame:
    if df is None or df.empty:
        return pd.DataFrame({c: [] for c in cols})
    for c in cols:
        if c not in df.columns:
            df[c] = np.nan
    return df

def _num(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")

def _parse_timestamp_iso(s: pd.Series) -> pd.Series:
    try:
        return pd.to_datetime(s, errors="coerce", utc=True)
    except Exception:
        return pd.Series([pd.NaT]*len(s))

def _latest_per_id(df: pd.DataFrame) -> pd.DataFrame:
    """trades(=open) 스냅샷에서 id별 최신 행만 선택."""
    need = ["id","timestamp"]
    df = _ensure_cols(df, need)
    if df.empty:
        return df
    ts = _parse_timestamp_iso(df["timestamp"])
    df = df.assign(_ts=ts)
    df = df.sort_values("_ts").drop_duplicates(subset=["id"], keep="last")
    return df

def _realized_metrics(df_close: pd.DataFrame) -> Dict[str, float]:
    df = _ensure_cols(df_close, ["pnl"])
    if df.empty:
        return dict(closed=0, wins=0, win_rate=float("nan"), profit_factor=float("nan"),
                    pnl_sum=0.0, max_drawdown=0.0)
    pnl = _num(df["pnl"]).fillna(0.0)
    closed = len(pnl)
    wins = int((pnl > 0).sum())
    win_rate = wins / closed if closed else float("nan")
    gains = pnl[pnl > 0].sum()
    losses = -pnl[pnl < 0].sum()
    profit_factor = (gains / losses) if losses > 0 else float("nan")
    # equity curve & max DD
    eq = pnl.cumsum()
    peak = eq.cummax()
    dd = (peak - eq).max() if len(eq) else 0.0
    return dict(closed=closed, wins=wins, win_rate=win_rate,
                profit_factor=profit_factor, pnl_sum=float(pnl.sum()),
                max_drawdown=float(dd))

def _execution_metrics(df_open_latest: pd.DataFrame) -> Dict[str, float]:
    df = _ensure_cols(df_open_latest, ["entry_maker","used_market_fallback","spread_bps"])
    if df.empty:
        return dict(entry_maker_ratio=0.0, market_fallback_ratio=0.0, avg_spread_bps=0.0)
    maker = _num(df["entry_maker"]).fillna(0.0)
    fallback = _num(df["used_market_fallback"]).fillna(0.0)
    sp = _num(df["spread_bps"]).fillna(0.0)
    n = max(1, len(df))
    return dict(
        entry_maker_ratio=float(maker.mean()),
        market_fallback_ratio=float(fallback.mean()),
        avg_spread_bps=float(sp.mean()),
    )

def _rr_up_dn_net(row: pd.Series) -> Tuple[float, float]:
    """진입·TP·SL과 수수료로 순수익률(상승/하락)을 계산."""
    entry = float(row.get("entry", 0.0) or 0.0)
    tp    = float(row.get("tp", 0.0) or 0.0)
    sl    = float(row.get("sl", 0.0) or 0.0)
    side  = str(row.get("side","")).lower()
    if entry <= 0 or tp <= 0 or sl <= 0 or side not in ("long","short"):
        return 0.0, 0.0

    if side == "long":
        up_gross = (tp - entry) / entry
        dn_gross = (entry - sl) / entry
    else:
        up_gross = (entry - tp) / entry
        dn_gross = (sl - entry) / entry

    # fees: entry (maker? taker?), tp (LIMIT=maker else taker), sl (taker 가정)
    maker = FEE_MAKER_BPS / 1e4
    taker = FEE_TAKER_BPS / 1e4
    entry_maker = None
    try:
        val = str(row.get("entry_maker","")).strip()
        entry_maker = (val == "1") or (val.lower() == "true")
    except Exception:
        pass
    fee_e  = maker if entry_maker else taker

    tp_type = str(row.get("tp_type","")).upper()
    fee_tp  = maker if tp_type == "LIMIT" else taker
    fee_sl  = taker  # STOP_MARKET가 일반적

    up_net = max(0.0, up_gross - (fee_e + fee_tp))
    dn_net = max(1e-12, dn_gross + (fee_e + fee_sl))
    return float(up_net), float(dn_net)

def _ev_row_usd(row: pd.Series) -> float:
    """사전 기대 PnL(USD). notional * (p*up_net - (1-p)*dn_net)"""
    try:
        p_raw = row.get("prob_cal", row.get("prob", 0.5))
        p = float(p_raw)
    except Exception:
        p = 0.5
    p = min(1.0, max(0.0, p))
    up_net, dn_net = _rr_up_dn_net(row)
    notional = 0.0
    try:
        notional = float(row.get("notional", 0.0))
    except Exception:
        pass
    return float(notional) * (p * up_net - (1.0 - p) * dn_net)

def _ev_summary(df_open_latest: pd.DataFrame) -> Dict[str, float]:
    if df_open_latest is None or df_open_latest.empty:
        return dict(n=0, s_notional=0.0, ev_usd=0.0, ev_perc=float("nan"))
    df = df_open_latest.copy()
    # ensure needed columns exist
    for c in ["prob","prob_cal","entry","tp","sl","side","notional","tp_type","entry_maker"]:
        if c not in df.columns: df[c] = np.nan
    # numeric coercions where applicable
    for c in ["entry","tp","sl","notional"]:
        df[c] = _num(df[c]).fillna(0.0)
    evs = df.apply(_ev_row_usd, axis=1)
    s_notional = float(_num(df["notional"]).fillna(0.0).sum())
    ev_total = float(evs.sum())
    ev_perc = (ev_total / s_notional) if s_notional > 0 else float("nan")
    return dict(n=int(len(df)), s_notional=s_notional, ev_usd=ev_total, ev_perc=ev_perc)

def _print_header(title: str):
    print(title)
    print("=" * len(title))

def _save_out(df: pd.DataFrame, outdir: str, fname: str):
    if df is None or df.empty: return
    os.makedirs(outdir, exist_ok=True)
    path = os.path.join(outdir, fname)
    df.to_csv(path, index=False)
    return path

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--bucket", required=True)
    p.add_argument("--prefix", required=True)  # e.g., trading_bot
    p.add_argument("--date", required=True)    # YYYYMMDD
    p.add_argument("--tz", default="UTC")
    args = p.parse_args()

    if not _GCS_OK:
        print("google-cloud-storage가 설치되어야 합니다: pip install google-cloud-storage")
        sys.exit(2)

    client = storage.Client()
    bucket = args.bucket
    prefix = args.prefix
    day = args.date
    tz = args.tz

    # 1) 목록
    keys_open = _list_day_objects(client, bucket, prefix, "trades", day)
    keys_close = _list_day_objects(client, bucket, prefix, "trades_close", day)

    # 2) 적재
    df_open = _load_day_csvs(client, bucket, keys_open)
    df_close = _load_day_csvs(client, bucket, keys_close)

    # 3) 최신 스냅샷(id별) + 실현손익
    open_latest = _latest_per_id(df_open)
    realized = _realized_metrics(df_close)
    execq = _execution_metrics(open_latest)
    evsum_all = _ev_summary(open_latest)

    # 4) 심볼별 EV
    per_symbol = []
    if not open_latest.empty:
        for sym, g in open_latest.groupby(open_latest.get("symbol", pd.Series([], dtype=str))):
            per_symbol.append({"symbol": sym, **_ev_summary(g)})
    df_ev_by_sym = pd.DataFrame(per_symbol)

    # 5) 출력
    print(f"GCS Day Analysis  | bucket={bucket}  prefix={prefix}  date={day}  tz={tz}")
    print(f"open_snapshots={len(open_latest)}  close_snapshots={len(df_close)}  symbols=ALL")
    print("-"*72)
    print("[ Realized PnL Today (trades_close) ]")
    print(f"  closed={realized['closed']}  wins={realized['wins']}  win_rate={realized['win_rate']:.3f}  "
          f"profit_factor={realized['profit_factor']:.3f}  pnl_sum={realized['pnl_sum']:.6f}  max_drawdown={realized['max_drawdown']:.6f}")
    print("-"*72)
    print("[ Ex-ante Expected Value (latest snapshot per trade id) ]")
    if evsum_all["n"] == 0:
        print("  (no open snapshots)")
    else:
        print(f"  n={evsum_all['n']}  notional_sum={evsum_all['s_notional']:.2f}  "
              f"EV_total=${evsum_all['ev_usd']:.2f}  EV_rate={evsum_all['ev_perc']*100 if np.isfinite(evsum_all['ev_perc']) else float('nan'):.3f}%")
    print("-"*72)
    print("[ Execution Quality (latest snapshot per trade id) ]")
    print(f"  entry_maker_ratio={execq['entry_maker_ratio']:.3f}  market_fallback_ratio={execq['market_fallback_ratio']:.3f}  "
          f"avg_spread_bps={execq['avg_spread_bps']:.2f}")

    outdir = f"./out_{day}"
    p1 = _save_out(open_latest, outdir, "open_latest.csv")
    p2 = _save_out(df_close, outdir, "closed_today.csv")
    p3 = _save_out(df_ev_by_sym, outdir, "ev_by_symbol.csv")
    print("-"*72)
    print(f"[OK] CSVs saved under: {os.path.abspath(outdir)}")
    for pth in (p1, p2, p3):
        if pth: print("  -", pth)

if __name__ == "__main__":
    main()
