# =============================================
# data_fetch.py  (2025-08-04 - SDK-v1 migration)
# =============================================
"""Data-layer helpers for Gemini-based crypto trading pipeline (USDS-Futures).

Changes vs. pre-SDK-v1 version
------------------------------
* 구(舊) ``UMFutures`` → ``DerivativesTradingUsdsFutures`` 싱글턴으로 교체
* 모든 REST 호출은 ``client.rest_api.<method>(...).data()`` 패턴 사용
* 심볼을 소문자로 전달하도록 통일(Binance가 대소문자 무시하긴 하나 일부 엔드포인트에서 권장)
"""

from __future__ import annotations

import logging
from time import sleep
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# NEW: lazily-instantiated singleton wrapping DerivativesTradingUsdsFutures
from .binance_client import get_client as _get_client  # 수정: client 대신 get_client 사용

# Bring in internal helpers from binance_client for adaptive method resolution.
# These helpers are not part of the public API but provide fallback logic for
# different SDK versions.  If import fails (e.g. due to refactor), they
# gracefully default to None.
try:
    from .binance_client import _try_methods as _client_try_methods  # type: ignore
    from .binance_client import _data as _client_data  # type: ignore
except Exception:
    _client_try_methods = None  # type: ignore
    _client_data = None  # type: ignore

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

# ---------------------------------------------------------------------------
# Binance client (shared)
# ---------------------------------------------------------------------------

client = _get_client()  # DerivativesTradingUsdsFutures singleton

# ---------------------------------------------------------------------------
# OHLCV helpers
# ---------------------------------------------------------------------------
# helpers/data_fetch.py  – fetch_ohlcv()
def fetch_ohlcv(
    symbol: str = "BTCUSDT",
    *, interval: str = "1h", limit: int = 1000
) -> Optional[pd.DataFrame]:
    """
    Get raw klines for `symbol`/`interval`, trying several SDK method names.
    Handles enum‐style intervals required by kline_candlestick_data().
    """
    sym = symbol.lower()
    ivl = interval.lower()

    # ---------------- enum 변환 ----------------
    enum_map = {
        "1m": "INTERVAL_1m", "3m": "INTERVAL_3m", "5m": "INTERVAL_5m",
        "15m": "INTERVAL_15m", "30m": "INTERVAL_30m",
        "1h": "INTERVAL_1h", "2h": "INTERVAL_2h", "4h": "INTERVAL_4h",
        "6h": "INTERVAL_6h", "8h": "INTERVAL_8h", "12h": "INTERVAL_12h",
        "1d": "INTERVAL_1d", "3d": "INTERVAL_3d",
        "1w": "INTERVAL_1w", "1M": "INTERVAL_1M",
    }
    enum_ivl = enum_map.get(ivl, ivl)  # fallback 그대로

    # -------- candidate method list --------
    candidates = [
        "klines",                    # (구버전)
        "kline_candlestick_data",    # (신버전)
        "get_klines",
        "kline_candlestick_data_v2",
    ]
    fn = (_client_try_methods(client.rest_api, candidates)
          if _client_try_methods else None)
    if not fn:
        for name in candidates:
            attr = getattr(client.rest_api, name, None)
            if callable(attr):
                fn = attr; break
    if not fn:
        logging.error(f"[fetch_ohlcv] no kline method found for {symbol}")
        return None

    # ------------- 호출 파라미터 -------------
    params: Dict[str, Any]
    if fn.__name__ in ("klines", "get_klines"):
        params = {"symbol": sym, "interval": ivl, "limit": limit}
    else:  # kline_candlestick_data*
        # 새 SDK도 문자열 `"1h"`·`"5m"` 그대로 받습니다.
        params = {"symbol": sym.upper(), "interval": ivl}
        if "limit" in fn.__code__.co_varnames:
            params["limit"] = limit

    # ------------- 호출 -------------
    try:
        resp = fn(**params)
    except TypeError:
        # positional fallback
        resp = fn(sym, enum_ivl, limit) if "limit" in params else fn(sym, enum_ivl)

    raw = _client_data(resp) if _client_data else (resp.data() if hasattr(resp, "data") else resp)
    if isinstance(raw, dict) and "data" in raw:
        raw = raw["data"]
    if not isinstance(raw, list) or not raw:
        raise RuntimeError("Empty klines")

    # ------------- DataFrame 변환 -------------
    cols = [
        "open_time","open","high","low","close","volume","close_time",
        "quote_asset_volume","number_of_trades",
        "taker_buy_base_asset_volume","taker_buy_quote_asset_volume","ignore",
    ]
    rows = []
    for item in raw:
        if isinstance(item, (list, tuple)):
            rows.append(list(item))
        elif hasattr(item, "to_dict"):
            d = item.to_dict()
            rows.append([d.get(c) for c in cols])
        elif isinstance(item, dict):
            rows.append([item.get(c) for c in cols])

    df = pd.DataFrame(rows, columns=cols)
    df["timestamp"] = pd.to_datetime(df["open_time"], unit="ms")
    for c in ("open","high","low","close","volume"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df["SMA_20"] = df["close"].rolling(20).mean()
    return df

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

    tr = pd.concat(
        [
            high - low,
            (high - prev_close).abs(),
            (low - prev_close).abs(),
        ],
        axis=1,
    ).max(axis=1)
    return tr.rolling(window).mean()


# ---------------------------------------------------------------------------
# Micro-structure & funding
# ---------------------------------------------------------------------------

def fetch_funding_rate(symbol: str) -> float:
    """
    Return the most recent funding rate for ``symbol``.

    This helper is resilient to SDK changes.  Earlier versions exposed a
    simple ``funding_rate()`` method; version 1.0.0 of the modular SDK
    introduces new names (e.g. ``get_funding_rate_history``, ``mark_price``)
    and versioned variants.  We attempt multiple candidate method names via
    the internal `_try_methods` helper if available.  The first available
    method is called and the funding rate is extracted from the response.

    Args:
        symbol (str): Trading pair, e.g. ``"BTCUSDT"``.

    Returns:
        float: Latest funding rate (e.g. 0.00025).  Returns 0.0 if no data.
    """
    if not symbol:
        return 0.0
    sym = symbol.lower()
    # List of method names to try, ordered by priority.  Legacy names first,
    # followed by v1-specific and camelCase variants, then mark price endpoints.
    candidates = [
        "funding_rate",
        "funding_rate_v1",
        "fundingRate",
        "get_funding_rate_history",
        "funding_rate_history",
        "get_funding_rate_info",
        "funding_rate_info",
        "mark_price",
        "premium_index",
    ]
    try:
        fn = None
        # Use helper from binance_client if available; otherwise fallback to direct lookup.
        if _client_try_methods is not None:
            fn = _client_try_methods(client.rest_api, candidates)
        else:
            for name in candidates:
                attr = getattr(client.rest_api, name, None)
                if callable(attr):
                    fn = attr
                    break
        if not fn:
            logging.warning(f"[funding_rate] no method found for {symbol}")
            return 0.0
        # Attempt to call the selected method.  Some history endpoints accept
        # ``limit=1``; mark price endpoints usually accept only ``symbol``.
        try:
            if "limit" in fn.__code__.co_varnames:
                resp = fn(symbol=sym, limit=1)
            else:
                resp = fn(symbol=sym)
        except TypeError:
            resp = fn(sym)
        # Unwrap the response into raw data
        if _client_data is not None:
            raw = _client_data(resp)
        else:
            raw = resp.data() if hasattr(resp, "data") else resp
        rate = 0.0
        # If response is list-like, take first element
        if isinstance(raw, list) and raw:
            first = raw[0]
            if not isinstance(first, dict) and hasattr(first, "to_dict"):
                try:
                    first = first.to_dict()
                except Exception:
                    pass
            if isinstance(first, dict):
                for key in ("fundingRate", "lastFundingRate", "funding_rate"):
                    if key in first and first[key] not in (None, ""):
                        try:
                            rate = float(first[key])
                            break
                        except Exception:
                            pass
        elif isinstance(raw, dict):
            # For mark_price or premium_index responses
            if hasattr(raw, "to_dict"):
                try:
                    raw = raw.to_dict()
                except Exception:
                    pass
            for key in ("fundingRate", "lastFundingRate", "funding_rate"):
                if key in raw and raw[key] not in (None, ""):
                    try:
                        rate = float(raw[key])
                        break
                    except Exception:
                        pass
            if rate == 0.0 and "data" in raw and isinstance(raw["data"], list):
                for entry in raw["data"]:
                    if hasattr(entry, "to_dict"):
                        try:
                            entry = entry.to_dict()
                        except Exception:
                            pass
                    if isinstance(entry, dict):
                        for key in ("fundingRate", "lastFundingRate", "funding_rate"):
                            if key in entry and entry[key] not in (None, ""):
                                try:
                                    rate = float(entry[key])
                                    break
                                except Exception:
                                    pass
                        if rate != 0.0:
                            break
        if rate == 0.0:
            logging.warning(f"[funding_rate] empty/zero for {symbol} – raw={raw}")
        return rate
    except Exception as exc:
        logging.error(f"[funding_rate] {symbol}: {exc}")
        return 0.0


def fetch_orderbook(
    symbol: str = "BTCUSDT",
    *, limit: int = 50,
    retries: Tuple[int, ...] = (1, 3, 5, 10, 20),  # 재시도 증가 및 백오프
) -> Optional[Dict[str, Any]]:
    """
    Fetch the current order-book snapshot for ``symbol``.
    """
    sym = symbol.upper()
    client = _get_client()  # binance_client.py의 get_client() 사용
    for attempt, delay in enumerate(retries, 1):
        try:
            resp = client.rest_api.order_book(symbol=sym, limit=limit)
            raw = resp.data() if hasattr(resp, "data") else resp
            
            # 새: 모델 객체 처리 (to_dict() 또는 vars()로 dict 변환)
            if hasattr(raw, "to_dict"):
                try:
                    raw = raw.to_dict()
                except Exception as exc:
                    logging.warning(f"[orderbook] to_dict failed for {sym}: {exc}")
            
            # 또는 직접 bids/asks 접근 (모델 속성일 경우)
            elif hasattr(raw, "bids") and hasattr(raw, "asks"):
                # bids/asks가 리스트면 그대로 사용, 아니면 변환
                bids = [(item.root[0], item.root[1]) for item in raw.bids] if isinstance(raw.bids, list) else []
                asks = [(item.root[0], item.root[1]) for item in raw.asks] if isinstance(raw.asks, list) else []
                raw = {"bids": bids, "asks": asks}  # dict로 재구성
            
            if isinstance(raw, dict) and "bids" in raw and "asks" in raw:
                if not raw["bids"] or not raw["asks"]:
                    logging.warning(f"[orderbook] Empty bids/asks for {sym} - raw={raw}")
                return raw
            else:
                logging.warning(f"[orderbook] Invalid response format for {sym} - raw={raw}")
        except Exception as exc:
            logging.error(f"[orderbook] Fetch error for {sym}: {exc} (attempt {attempt})")
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


def fetch_mtf_raw(
    symbol: str = "BTCUSDT",
    layouts: List[Tuple[str, int]] | None = None,
) -> Dict[str, pd.DataFrame]:
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
                keep = [
                    c
                    for c in df.columns
                    if c in {"timestamp", "open", "high", "low", "close", "volume"}
                ]
                res[tf] = df[keep].copy()
        except Exception as exc:
            logging.error(f"[fetch_mtf_raw] {symbol} {tf}: {exc}")
            res[tf] = pd.DataFrame()
    return res