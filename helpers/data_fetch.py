from __future__ import annotations

"""
helpers/data_fetch.py — refactor #2 (2025-08-11, KST)

Why this patch?
- SDK klines/depth가 환경/버전마다 이름이 달라 실패 → DF 비어짐 → payload 0.0
- SDK 실패시 공식 REST로 폴백 (klines/depth/fundingRate)

Env(옵션)
- BINANCE_USE_TESTNET=true → testnet REST 도메인 사용
- BINANCE_FAPI_BASE, BINANCE_FAPI_TESTNET_BASE 커스터마이즈 가능
"""

import logging
import os
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# Optional: requests only when REST fallback is needed
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

# ----------------------------------------------------------------------------
# Binance client glue (SDK)
# ----------------------------------------------------------------------------
try:
    from .binance_client import get_client as _get_client  # type: ignore
except Exception as e:  # pragma: no cover
    _get_client = None  # type: ignore
    logging.warning("binance_client.get_client import failed: %s", e)

# Base URLs for REST fallback
FAPI_BASE = os.getenv("BINANCE_FAPI_BASE", "https://fapi.binance.com")  # prod
FAPI_TESTNET_BASE = os.getenv("BINANCE_FAPI_TESTNET_BASE", "https://testnet.binancefuture.com")
USE_TESTNET = os.getenv("BINANCE_USE_TESTNET", "false").strip().lower() in ("1", "true", "yes", "y", "on")

def _rest_base() -> str:
    return FAPI_TESTNET_BASE if USE_TESTNET else FAPI_BASE

# ----------------------------------------------------------------------------
# Small utilities
# ----------------------------------------------------------------------------
def _is_df(x) -> bool:
    return isinstance(x, pd.DataFrame) and not x.empty

def _normalize_ohlcv_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Ensure [timestamp, open, high, low, close, volume], numeric, sorted."""
    df = df.copy()
    rename_map = {
        "open_time": "timestamp",
        "openTime": "timestamp",
        "t": "timestamp",
        "o": "open",
        "h": "high",
        "l": "low",
        "c": "close",
        "v": "volume",
        "Open": "open",
        "High": "high",
        "Low": "low",
        "Close": "close",
        "Volume": "volume",
    }
    df.rename(columns=rename_map, inplace=True)
    if "timestamp" in df.columns:
        try:
            df["timestamp"] = pd.to_datetime(df["timestamp"], unit="ms", errors="coerce")
        except Exception:
            df["timestamp"] = pd.to_datetime(df["timestamp"], errors="coerce")
    for col in ("open", "high", "low", "close", "volume"):
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    keep = [c for c in ["timestamp", "open", "high", "low", "close", "volume"] if c in df.columns]
    df = df[keep]
    df.dropna(how="any", inplace=True)
    if "timestamp" in df.columns:
        df.sort_values("timestamp", inplace=True)
    df.reset_index(drop=True, inplace=True)
    return df

def _from_sdk_klines(raw: Any) -> Optional[pd.DataFrame]:
    """Accept diverse SDK return shapes and normalize."""
    try:
        if raw is None:
            return None
        if hasattr(raw, "data"):
            raw = raw.data()
        # list[dict]
        if isinstance(raw, dict) and "data" in raw:
            raw = raw["data"]
        if isinstance(raw, list) and raw and isinstance(raw[0], dict):
            rows = []
            for r in raw:
                ts = r.get("openTime") or r.get("open_time") or r.get("t") or r.get("timestamp")
                rows.append({
                    "timestamp": ts,
                    "open": r.get("open") or r.get("o"),
                    "high": r.get("high") or r.get("h"),
                    "low": r.get("low") or r.get("l"),
                    "close": r.get("close") or r.get("c"),
                    "volume": r.get("volume") or r.get("v"),
                })
            return _normalize_ohlcv_columns(pd.DataFrame(rows))
        # list[list]
        if isinstance(raw, list) and raw and isinstance(raw[0], (list, tuple)):
            rows = []
            for r in raw:
                if not isinstance(r, (list, tuple)) or len(r) < 6:
                    continue
                rows.append({
                    "timestamp": r[0],
                    "open": r[1],
                    "high": r[2],
                    "low": r[3],
                    "close": r[4],
                    "volume": r[5],
                })
            return _normalize_ohlcv_columns(pd.DataFrame(rows))
    except Exception as e:
        logging.error("_from_sdk_klines error: %s", e)
    return None

def _call(obj: Any, names: List[str], /, **kwargs):
    for name in names:
        fn = getattr(obj, name, None)
        if callable(fn):
            return fn(**kwargs)
    raise AttributeError(f"None of {names} is available on {type(obj).__name__}")

# ----------------------------------------------------------------------------
# Public: fetch_ohlcv / orderbook / funding (with REST fallbacks)
# ----------------------------------------------------------------------------
def fetch_ohlcv(symbol: str, interval: str = "5m", limit: int = 600) -> Optional[pd.DataFrame]:
    """Try SDK; if it fails/empty, use REST GET /fapi/v1/klines."""
    # 1) SDK try
    try:
        client = _get_client() if _get_client else None
        if client is not None:
            try:
                resp = _call(
                    client.rest_api,
                    ["kline_candlestick_data", "kline_candlestick", "klines", "continuous_klines"],
                    symbol=symbol.upper(),
                    interval=interval.lower(),
                    limit=int(limit),
                )
            except Exception:
                resp = _call(
                    client.rest_api,
                    ["kline_candlestick_data", "kline_candlestick", "klines", "continuous_klines"],
                    symbol=symbol.upper(),
                    interval=interval.upper(),
                    limit=int(limit),
                )
            df = _from_sdk_klines(resp)
            if df is not None and not df.empty:
                logging.info("fetch_ohlcv SDK ok: %s %s rows=%d", symbol, interval, len(df))
                return df
            logging.warning("fetch_ohlcv SDK empty: %s %s", symbol, interval)
    except Exception as e:
        logging.warning("fetch_ohlcv SDK failed: %s %s err=%s", symbol, interval, e)

    # 2) REST fallback
    if requests is None:
        logging.error("requests not available; cannot REST-fallback klines")
        return None
    try:
        url = _rest_base().rstrip("/") + "/fapi/v1/klines"
        params = {"symbol": symbol.upper(), "interval": interval, "limit": int(limit)}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        data = r.json()
        df = _from_sdk_klines(data)
        if df is not None and not df.empty:
            logging.info("fetch_ohlcv REST ok: %s %s rows=%d", symbol, interval, len(df))
            return df
        logging.error("fetch_ohlcv REST empty: %s %s", symbol, interval)
    except Exception as e:
        logging.error("fetch_ohlcv REST failed: %s %s err=%s", symbol, interval, e)
    return None

def fetch_orderbook(symbol: str, limit: int = 50) -> Optional[Dict[str, Any]]:
    """Return {'bids': [[p,q],...], 'asks': [[p,q],...]} with SDK→REST fallback."""
    # 1) SDK try
    try:
        client = _get_client() if _get_client else None
        if client is not None:
            resp = _call(client.rest_api, ["order_book", "depth"], symbol=symbol.upper(), limit=int(limit))
            if hasattr(resp, "data"):
                resp = resp.data()
            if isinstance(resp, dict):
                bids = resp.get("bids") or []
                asks = resp.get("asks") or []
            elif isinstance(resp, (list, tuple)) and len(resp) >= 2:
                bids, asks = resp[0], resp[1]
            else:
                bids, asks = [], []
            ob = {"bids": [[float(x[0]), float(x[1])] for x in bids[:limit]], "asks": [[float(x[0]), float(x[1])] for x in asks[:limit]]}
            if ob["bids"] and ob["asks"]:
                logging.info("fetch_orderbook SDK ok: %s", symbol)
                return ob
            logging.warning("fetch_orderbook SDK empty: %s", symbol)
    except Exception as e:
        logging.warning("fetch_orderbook SDK failed: %s err=%s", symbol, e)

    # 2) REST fallback
    if requests is None:
        logging.error("requests not available; cannot REST-fallback depth")
        return None
    try:
        url = _rest_base().rstrip("/") + "/fapi/v1/depth"
        params = {"symbol": symbol.upper(), "limit": int(limit)}
        r = requests.get(url, params=params, timeout=10)
        r.raise_for_status()
        j = r.json()
        bids = [[float(p), float(q)] for p, q in j.get("bids", [])[:limit]]
        asks = [[float(p), float(q)] for p, q in j.get("asks", [])[:limit]]
        if bids and asks:
            logging.info("fetch_orderbook REST ok: %s", symbol)
            return {"bids": bids, "asks": asks}
        logging.error("fetch_orderbook REST empty: %s", symbol)
    except Exception as e:
        logging.error("fetch_orderbook REST failed: %s err=%s", symbol, e)
    return None

def compute_orderbook_stats(ob: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """Compute imbalance and spread (bps) from orderbook.
    Returns {"imbalance": float, "spread": float}
    """
    try:
        if not ob or not ob.get("bids") or not ob.get("asks"):
            return {"imbalance": 0.0, "spread": 0.0}
        bids = ob["bids"]
        asks = ob["asks"]
        best_bid = float(bids[0][0])
        best_ask = float(asks[0][0])
        mid = (best_bid + best_ask) / 2.0
        spread_bps = ((best_ask - best_bid) / mid) * 1e4 if mid > 0 else 0.0
        bid_qty = sum(float(x[1]) for x in bids[:10])
        ask_qty = sum(float(x[1]) for x in asks[:10])
        total = bid_qty + ask_qty
        imbalance = (bid_qty - ask_qty) / total if total > 0 else 0.0
        return {"imbalance": float(imbalance), "spread": float(spread_bps)}
    except Exception as e:
        logging.error("compute_orderbook_stats error: %s", e)
        return {"imbalance": 0.0, "spread": 0.0}

def fetch_funding_rate(symbol: str) -> float:
    """Return last funding rate (%) using SDK→REST fallback. Returns percent value."""
    # 1) SDK try
    try:
        client = _get_client() if _get_client else None
        if client is not None:
            try:
                resp = _call(client.rest_api, ["funding_rate", "fundingRate"], symbol=symbol.upper(), limit=1)
            except Exception:
                resp = _call(client.rest_api, ["funding_rate_history", "fundingRateHistory"], symbol=symbol.upper(), limit=1)
            if hasattr(resp, "data"):
                resp = resp.data()
            x = resp[0] if isinstance(resp, list) and resp else resp
            if isinstance(x, dict):
                for k in ("fundingRate", "funding_rate", "lastFundingRate", "r"):
                    if k in x and x[k] not in (None, ""):
                        return float(x[k]) * 100.0
            if isinstance(x, list) and x and len(x) >= 2:
                return float(x[1]) * 100.0
    except Exception as e:
        logging.warning("fetch_funding_rate SDK failed: %s err=%s", symbol, e)

    # 2) REST fallback
    if requests is None:
        return 0.0
    try:
        url = _rest_base().rstrip("/") + "/fapi/v1/fundingRate"
        r = requests.get(url, params={"symbol": symbol.upper(), "limit": 1}, timeout=10)
        r.raise_for_status()
        arr = r.json()
        if isinstance(arr, list) and arr:
            fr = arr[-1].get("fundingRate")
            if fr not in (None, ""):
                return float(fr) * 100.0
    except Exception as e:
        logging.error("fetch_funding_rate REST failed: %s err=%s", symbol, e)
    return 0.0

# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def add_indicators(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """
    Adds RSI(14), SMA_20, volatility (rolling std of returns).
    Returns df even if input is None (empty fallback with expected columns).
    """
    if df is None or df.empty:
        base = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        base["RSI"] = np.nan
        base["SMA_20"] = np.nan
        base["volatility"] = np.nan
        return base

    out = df.copy()
    out["returns"] = out["close"].pct_change()
    out["volatility"] = out["returns"].rolling(20).std(ddof=0)
    delta = out["close"].diff()
    up = delta.clip(lower=0)
    down = (-delta).clip(lower=0)
    gain = up.rolling(14).mean()
    loss = down.rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    out["RSI"] = 100 - (100 / (1 + rs))
    out["SMA_20"] = out["close"].rolling(20).mean()
    out.drop(columns=["returns"], inplace=True, errors="ignore")
    return out

def compute_atr(df: Optional[pd.DataFrame], window: int = 14) -> pd.Series:
    """Average True Range. Returns a Series aligned with df index; empty Series if df invalid."""
    if df is None or df.empty:
        return pd.Series(dtype=float)
    tr1 = (df["high"] - df["low"]).abs()
    tr2 = (df["high"] - df["close"].shift(1)).abs()
    tr3 = (df["low"] - df["close"].shift(1)).abs()
    tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
    return tr.rolling(window).mean()

def compute_support_resistance(df: Optional[pd.DataFrame], lookback: int = 50) -> Dict[str, float]:
    if df is None or df.empty:
        return {"recent_high": 0.0, "recent_low": 0.0}
    sub = df.tail(lookback)
    return {"recent_high": float(sub["high"].max()), "recent_low": float(sub["low"].min())}

def compute_recent_price_sequence(df: Optional[pd.DataFrame], n: int = 10) -> List[float]:
    if df is None or df.empty:
        return [0.0] * n
    return [float(x) for x in df["close"].tail(n).tolist()]

def compute_trend_filter(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """Simple daily uptrend filter: compare 20-day SMA slope."""
    if df is None or df.empty or "timestamp" not in df.columns:
        return {"daily_uptrend": False, "trend_strength": 0.0}
    df_daily = df.resample("1D", on="timestamp").agg({"open": "first", "high": "max", "low": "min", "close": "last", "volume": "sum"})
    if df_daily is None or df_daily.empty:
        return {"daily_uptrend": False, "trend_strength": 0.0}
    sma = df_daily["close"].rolling(20).mean()
    slope = (sma - sma.shift(5)) / 5.0
    last = float(slope.iloc[-1]) if slope.size else 0.0
    return {"daily_uptrend": bool(last > 0), "trend_strength": float(last)}

def compute_relative_volume(df: Optional[pd.DataFrame], lookback: int = 20) -> float:
    if df is None or df.empty:
        return 1.0
    vol = df["volume"].tail(lookback + 1)
    if vol.size < lookback + 1:
        return 1.0
    recent = float(vol.iloc[-1])
    avg = float(vol.iloc[:-1].mean()) if vol.size > 1 else 1.0
    return float(recent / avg) if avg > 0 else 1.0

# ----------------------------------------------------------------------------
# Aggregated fetch
# ----------------------------------------------------------------------------
def fetch_data(symbol: str, tf_entry: str = "5m", include_orderbook: bool = True, **kwargs) -> Dict[str, Any]:
    """
    Compatibility: also accepts legacy keyword ``interval`` and maps it to ``tf_entry``.
    """
    interval_kw = kwargs.pop("interval", None)
    if interval_kw is not None:
        tf_entry = str(interval_kw)

    ohlcv = fetch_ohlcv(symbol, interval=tf_entry, limit=600)
    if not _is_df(ohlcv):
        ohlcv = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    ob = fetch_orderbook(symbol, limit=50) if include_orderbook else None
    funding = fetch_funding_rate(symbol)

    # enrich entry frame
    if _is_df(ohlcv):
        ohlcv = add_indicators(ohlcv)
        if not ohlcv.empty:
            ohlcv.loc[ohlcv.index[-1], "funding_rate_pct"] = float(funding)
            ohlcv.loc[ohlcv.index[-1], "x_sentiment"] = 0.0

    # daily frame for trend filter
    df_day = fetch_ohlcv(symbol, interval="1d", limit=300)
    if not _is_df(df_day):
        df_day = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    return {
        "ohlcv": ohlcv,
        "orderbook": ob,
        "funding_rate_pct": float(funding),
        "times": {"tf_entry": tf_entry},
        "daily": df_day,
    }

# ----------------------------------------------------------------------------
# Multi-timeframe raw fetch (for signals.py extra section)
# ----------------------------------------------------------------------------
def fetch_mtf_raw(symbol: str, tfs: Optional[List[str]] = None, limit_by_tf: Optional[Dict[str, int]] = None) -> Dict[str, pd.DataFrame]:
    """
    Returns dict like {"5m": df, "1h": df, "4h": df, "1d": df}
    """
    out: Dict[str, pd.DataFrame] = {}
    tfs = tfs or ["5m", "1h", "4h", "1d"]
    lim_default = 600
    limits = limit_by_tf or {tf: (600 if tf == "5m" else 500) for tf in tfs}
    for tf in tfs:
        try:
            lim = int(limits.get(tf, lim_default))
            df = fetch_ohlcv(symbol, interval=tf, limit=lim)
            if not _is_df(df):
                out[tf] = pd.DataFrame()
            else:
                keep = [c for c in df.columns if c in {"timestamp", "open", "high", "low", "close", "volume"}]
                out[tf] = df[keep].copy()
        except Exception as e:
            logging.error("[fetch_mtf_raw] %s %s: %s", symbol, tf, e)
            out[tf] = pd.DataFrame()
    return out

__all__ = [
    "fetch_ohlcv",
    "add_indicators",
    "compute_atr",
    "fetch_orderbook",
    "compute_orderbook_stats",
    "fetch_funding_rate",
    "compute_recent_price_sequence",
    "compute_support_resistance",
    "compute_relative_volume",
    "compute_trend_filter",
    "fetch_data",
    "fetch_mtf_raw",
]
