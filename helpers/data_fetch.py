# =============================================
# data_fetch.py  (2025-08-03 refactor)
# =============================================
"""Data-layer helpers for Gemini 2.5 Pro crypto trading pipeline.

Key improvements
----------------
* Funding-rate (%/8h) is now fetched via ``fetch_funding_rate`` and returned to
  the caller so that the LLM sees market-bias information.
* Order-book statistics (imbalance, absolute spread) were already computed, but
  the caller didn’t receive them consistently.  We standardise the keys so
  later stages can consume them.
* Duplicate 5-minute OHLCV upload removed – we now treat 5 m as *micro* context
  and pass only one row of summarised indicators to the model (handled in
  ``predictor.py``).
* Default multi-time-frame (MTF) layout changed to   ``1h:240,4h:200,1d:120``
  (can be overridden via the ``MTF_LAYOUT`` env-var).
"""
from __future__ import annotations

import logging
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd
from binance.um_futures import UMFutures

from .binance_client import client as _get_client

__all__ = [
    "fetch_ohlcv",
    "add_indicators",
    "fetch_orderbook",
    "fetch_data",
    "fetch_multitime_indicators",
    "fetch_mtf_raw",
    "compute_orderbook_stats",
    "fetch_funding_rate",
]

# Single, shared client instance (avoids handshake spam)
client: UMFutures = _get_client()

# ---------------------------------------------------------------------------
# OHLCV helpers
# ---------------------------------------------------------------------------


def fetch_ohlcv(
    symbol: str = "BTCUSDT",
    *,
    interval: str = "1h",
    limit: int = 1000,
) -> Optional[pd.DataFrame]:
    """Return **raw** klines for a given symbol/interval."""
    try:
        res = client.klines(symbol=symbol, interval=interval, limit=limit)
        klines = res.get("data") if isinstance(res, dict) else res  # type: ignore[attr-defined]
        if not klines:
            raise RuntimeError("Empty klines")

        cols = [
            "open_time",
            "open",
            "high",
            "low",
            "close",
            "volume",
            "close_time",
            "quote_asset_volume",
            "number_of_trades",
            "taker_buy_base_asset_volume",
            "taker_buy_quote_asset_volume",
            "ignore",
        ]
        df = pd.DataFrame(klines, columns=cols)
        df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
        df = df.astype(
            {
                "open": float,
                "high": float,
                "low": float,
                "close": float,
                "volume": float,
            }
        )
        # Lightweight MA used by a few downstream features
        df["SMA_20"] = df["close"].rolling(20).mean()
        return df
    except Exception as exc:  # pragma: no cover
        logging.error(f"[fetch_ohlcv] {symbol} {interval}: {exc}")
        return None


# ---------------------------------------------------------------------------
# Indicator helpers
# ---------------------------------------------------------------------------


def add_indicators(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """Compute basic volatility + RSI features needed by the model."""
    if df is None or df.empty:
        return pd.DataFrame(
            {
                "close": [0.0],
                "volatility": [0.0],
                "RSI": [50.0],
                "SMA_20": [0.0],
                "x_sentiment": [0.0],
                "timestamp": [pd.Timestamp.utcnow()],
            }
        )

    delta = df["close"].diff()
    gain = delta.where(delta > 0, 0).rolling(14).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
    rs = gain / loss.replace(0, np.nan)
    df["RSI"] = 100 - (100 / (1 + rs))
    df["RSI"] = df["RSI"].replace([np.inf, -np.inf], 50).fillna(50)
    df["volatility"] = df["close"].rolling(20).std().fillna(0)
    df["x_sentiment"] = 0.0  # Placeholder – replace with real sentiment if available
    return df


def compute_atr(df: pd.DataFrame, window: int = 14) -> pd.Series:
    high = df["high"].astype(float)
    low = df["low"].astype(float)
    close = df["close"].astype(float)
    prev_close = close.shift(1).fillna(close)

    tr = pd.concat([
        high - low,
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    return tr.rolling(window).mean()


# ---------------------------------------------------------------------------
# Micro-structure & funding
# ---------------------------------------------------------------------------


def fetch_funding_rate(symbol: str) -> float:
    """Return *latest* funding-rate for ``symbol`` (decimal, e.g. 0.00025)."""
    try:
        res = client.funding_rate(symbol=symbol, limit=1)
        # 일부 게이트웨이는 {"data": [...]} 형태를 반환
        if isinstance(res, dict) and "data" in res:
            res = res["data"]
        rate = float(res[0]["fundingRate"]) if res else 0.0
        if rate == 0.0:
            logging.warning(f"[funding_rate] empty/zero for {symbol}")
        return rate
    except Exception as exc:  # pragma: no cover
        logging.error(f"[funding_rate] {symbol}: {exc}")
        return 0.0

def fetch_orderbook(symbol: str = "BTCUSDT", *, limit: int = 50, retries: Tuple[int, ...] = (1, 3, 5)) -> Optional[Dict[str, Any]]:
    for attempt, delay in enumerate(retries, 1):
        try:
            res = client.depth(symbol=symbol, limit=limit)
            ob = res.get("data") if isinstance(res, dict) else res  # type: ignore[attr-defined]
            if ob and "bids" in ob and "asks" in ob:
                return ob
        except Exception as exc:
            logging.error(f"[orderbook] attempt {attempt}: {exc}")
        sleep(delay)
    logging.warning("Orderbook fetch failed – returning None")
    return None


def compute_orderbook_stats(orderbook: dict) -> Tuple[float, float]:
    bids = orderbook.get("bids", [])
    asks = orderbook.get("asks", [])
    if not bids or not asks:
        return 0.0, 0.0
    try:
        total_bid_qty = sum(float(b[1]) for b in bids)
        total_ask_qty = sum(float(a[1]) for a in asks)
        imbalance = 0.0
        denom = total_bid_qty + total_ask_qty
        if denom > 0:
            imbalance = (total_bid_qty - total_ask_qty) / denom
        spread = float(asks[0][0]) - float(bids[0][0])
        return imbalance, spread
    except Exception:  # pragma: no cover
        return 0.0, 0.0


# ---------------------------------------------------------------------------
# High-level data fetch orchestration
# ---------------------------------------------------------------------------


def fetch_data(
    symbol: str = "BTCUSDT",
    *,
    interval: str = "5m",
    ohlcv_limit: int = 200,
    orderbook_limit: int = 50,
    include_orderbook: bool = True,
) -> Dict[str, Any]:
    """Return latest OHLCV (+basic indicators) *and* micro-structure extras."""
    ohlcv = add_indicators(fetch_ohlcv(symbol, interval=interval, limit=ohlcv_limit))

    # Order-book --------------------------------------------------------
    ob: Optional[Dict[str, Any]] = None
    if include_orderbook:
        ob = fetch_orderbook(symbol, limit=orderbook_limit)
        if ob:
            imbalance, spread = compute_orderbook_stats(ob)
            ohlcv.at[ohlcv.index[-1], "orderbook_imbalance"] = imbalance
            ohlcv.at[ohlcv.index[-1], "orderbook_spread"] = spread
            ob["imbalance"] = imbalance
            ob["spread"] = spread
        else:
            ohlcv.at[ohlcv.index[-1], "orderbook_imbalance"] = 0.0
            ohlcv.at[ohlcv.index[-1], "orderbook_spread"] = 0.0

    # Funding-rate ------------------------------------------------------
    funding = fetch_funding_rate(symbol)
    ohlcv.at[ohlcv.index[-1], "funding_rate_pct"] = funding * 100

    # Timestamps (ISO8601 strings keep payload JSON-serialisable)
    times = {
        "ohlcv": str(ohlcv["timestamp"].iloc[-1]) if not ohlcv.empty else None,
        "orderbook": str(pd.Timestamp.utcnow()) if ob else None,
        "funding": str(pd.Timestamp.utcnow()),
    }

    return {
        "ohlcv": ohlcv,
        "orderbook": ob,
        "funding_rate_pct": funding * 100,
        "times": times,
    }


# ---------------------------------------------------------------------------
# Cross-time-frame helpers
# ---------------------------------------------------------------------------

def fetch_multitime_indicators(symbol: str, intervals: List[str]) -> Dict[str, float]:
    """Return a flat ``{feature_name: value}`` dict across *intervals*."""
    out: Dict[str, float] = {}
    for interval in intervals:
        df = add_indicators(fetch_ohlcv(symbol, interval=interval, limit=200))
        if df is None or df.empty:
            out[f"RSI_{interval}"] = 50.0
            out[f"volatility_{interval}"] = 0.0
            out[f"SMA20_{interval}"] = 0.0
            out[f"ATR_{interval}"] = 0.0
            continue
        atr = compute_atr(df).iloc[-1]
        last = df.iloc[-1]
        out[f"RSI_{interval}"] = float(last.get("RSI", 50.0))
        out[f"volatility_{interval}"] = float(last.get("volatility", 0.0))
        out[f"SMA20_{interval}"] = float(last.get("SMA_20", 0.0))
        out[f"ATR_{interval}"] = float(atr) if pd.notna(atr) else 0.0
    return out


def fetch_mtf_raw(symbol: str = "BTCUSDT", layouts: List[Tuple[str, int]] | None = None) -> Dict[str, pd.DataFrame]:
    """Return *raw* OHLCV DFs for multiple TFs (token-friendly subset)."""
    if layouts is None:
        layouts = [("1h", 240), ("4h", 200), ("1d", 120)]
    res: Dict[str, pd.DataFrame] = {}
    for tf, lim in layouts:
        try:
            df = fetch_ohlcv(symbol, interval=tf, limit=lim)
            if df is None or df.empty:
                res[tf] = pd.DataFrame()
            else:
                keep = [c for c in df.columns if c in {"timestamp", "open", "high", "low", "close", "volume"}]
                res[tf] = df[keep].copy()
        except Exception as exc:
            logging.error(f"[fetch_mtf_raw] {symbol} {tf}: {exc}")
            res[tf] = pd.DataFrame()
    return res
