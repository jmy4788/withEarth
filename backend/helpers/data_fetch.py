from __future__ import annotations
"""
helpers/data_fetch.py — refactor (2025-08-12, KST)

설계 원칙
- Binance USDⓈ-M Futures 데이터 수집은 **SDK 우선 → REST 폴백**.
- 캔들 데이터는 표준 컬럼으로 정규화: timestamp, open, high, low, close, volume
- 마지막 '미종가(진행중) 봉'은 제거하여 부분봉 혼입 방지.
- 신호용 파생지표(RSI, SMA_20, 변동성, ATR 등) 계산 유틸 제공.
- 오더북/펀딩/MTF(1h/4h/1d) 원천 수집 지원.

환경변수
- BINANCE_USE_TESTNET=true           → REST 폴백에서 테스트넷 도메인 사용
- BINANCE_FAPI_BASE                  → REST 폴백 프로덕션 베이스(URL)
- BINANCE_FAPI_TESTNET_BASE          → REST 폴백 테스트넷 베이스(URL)
- BINANCE_HTTP_TIMEOUT_MS, ...       → 타임아웃/재시도(요청 라이브러리 레벨에서 활용 가능)

호환성
- signals.py가 기대하는 공개 함수/시그니처를 유지합니다:
  fetch_data(symbol, interval="5m", ohlcv_limit=200, orderbook_limit=50, include_orderbook=True) -> dict
  fetch_mtf_raw(symbol) -> dict[str, pd.DataFrame]
  add_indicators(df) -> pd.DataFrame
  compute_atr(df, window=14) -> pd.Series
  compute_orderbook_stats(ob) -> dict
  compute_recent_price_sequence(df, n=10) -> list[float]
  compute_support_resistance(df, lookback=50) -> dict
  compute_relative_volume(df, lookback=20) -> float
  fetch_orderbook(symbol, limit=50) -> dict
  fetch_funding_rate(symbol) -> float
"""

from typing import Any, Dict, List, Optional, Tuple
import logging
import os
import time
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# requests는 런타임 환경에 따라 없을 수 있으므로 방어적으로 임포트
try:
    import requests  # type: ignore
except Exception:  # pragma: no cover
    requests = None  # type: ignore

# ----------------------------------------------------------------------------
# Binance modular SDK glue (optional)
# ----------------------------------------------------------------------------
_SDK_READY = False
try:
    from .binance_client import get_client as _get_client  # type: ignore
    _SDK_READY = True
except Exception as e:  # pragma: no cover
    _get_client = None  # type: ignore
    logging.warning("binance_client.get_client import failed: %s", e)

# ----------------------------------------------------------------------------
# REST base URLs (fallback)
# ----------------------------------------------------------------------------
FAPI_BASE = os.getenv("BINANCE_FAPI_BASE", "https://fapi.binance.com")  # prod
FAPI_TESTNET_BASE = os.getenv("BINANCE_FAPI_TESTNET_BASE", "https://testnet.binancefuture.com")  # testnet
USE_TESTNET = os.getenv("BINANCE_USE_TESTNET", "").lower() in ("1", "true", "yes")

HTTP_TIMEOUT = int(os.getenv("BINANCE_HTTP_TIMEOUT_MS", "10000")) / 1000.0

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ----------------------------------------------------------------------------
# Interval map & helpers
# ----------------------------------------------------------------------------
_INTERVAL_MS = {
    "1m": 60_000,
    "3m": 180_000,
    "5m": 300_000,
    "15m": 900_000,
    "30m": 1_800_000,
    "1h": 3_600_000,
    "2h": 7_200_000,
    "4h": 14_400_000,
    "6h": 21_600_000,
    "8h": 28_800_000,
    "12h": 43_200_000,
    "1d": 86_400_000,
    "3d": 259_200_000,
    "1w": 604_800_000,
    "1M": 2_592_000_000,  # 30d approx
}


def _now_ms() -> int:
    return int(time.time() * 1000)


def _to_float(x: Any) -> float:
    try:
        return float(x)
    except Exception:
        return 0.0


def _kline_to_df(rows: List[List[Any]], *, use_close_time: bool = True) -> pd.DataFrame:
    """
    REST /fapi/v1/klines 응답 → 표준화 DF
    rows[i] = [
        0 open_time, 1 open, 2 high, 3 low, 4 close, 5 volume,
        6 close_time, 7 quote_asset_volume, 8 number_of_trades,
        9 taker_buy_base_volume, 10 taker_buy_quote_volume, 11 ignore
    ]
    """
    if not rows:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])

    ot = [int(r[0]) for r in rows]
    ct = [int(r[6]) for r in rows]
    ts = ct if use_close_time else ot

    df = pd.DataFrame(
        {
            "timestamp": [datetime.fromtimestamp(t / 1000, tz=timezone.utc).isoformat() for t in ts],
            "open": [_to_float(r[1]) for r in rows],
            "high": [_to_float(r[2]) for r in rows],
            "low": [_to_float(r[3]) for r in rows],
            "close": [_to_float(r[4]) for r in rows],
            "volume": [_to_float(r[5]) for r in rows],
            "_open_time_ms": ot,
            "_close_time_ms": ct,
        }
    )
    return df


def _trim_to_closed(df: pd.DataFrame, interval: str) -> pd.DataFrame:
    """미종가(진행 중) 봉 제거: close_time이 현재보다 크거나, 차주기 미만이면 마지막 봉 drop."""
    if df is None or df.empty:
        return df
    try:
        now = _now_ms()
        int_ms = _INTERVAL_MS.get(interval, 0)
        # close_time 기준으로 판정
        last_close_ms = int(df["_close_time_ms"].iloc[-1])
        last_open_ms = int(df["_open_time_ms"].iloc[-1])
        # close_time이 미래거나, (now - open) < 간격 → 진행중으로 간주
        if (last_close_ms > now) or (int_ms and (now - last_open_ms < int_ms)):
            df = df.iloc[:-1].copy()
        # 내부용 컬럼 제거
        if "_open_time_ms" in df.columns:
            df.drop(columns=["_open_time_ms"], inplace=True)
        if "_close_time_ms" in df.columns:
            df.drop(columns=["_close_time_ms"], inplace=True)
    except Exception:
        # 실패시 내부컬럼만 제거
        for c in ["_open_time_ms", "_close_time_ms"]:
            if c in df.columns:
                try:
                    df.drop(columns=[c], inplace=True)
                except Exception:
                    pass
    return df


# ----------------------------------------------------------------------------
# Fetchers (SDK first → REST fallback)
# ----------------------------------------------------------------------------
def _sdk_fetch_klines(symbol: str, interval: str, limit: int = 200) -> Optional[pd.DataFrame]:
    """모듈형 SDK 시도. 실패 시 None 반환."""
    if not _SDK_READY:
        return None
    try:
        client = _get_client()
        # SDK 메서드 네이밍 가드: 여러 후보 시도
        rest = getattr(client, "rest_api")
        for name in ["kline_candlestick_data", "kline_candlestick", "klines", "continuous_klines"]:
            fn = getattr(rest, name, None)
            if callable(fn):
                resp = fn(symbol=symbol, interval=interval, limit=int(limit))
                data = resp.data() if hasattr(resp, "data") else resp
                # data가 이미 list[list] 형태일 수 있음
                if isinstance(data, list) and data and isinstance(data[0], (list, tuple)):
                    df = _kline_to_df(data)
                    return df
                # 일부 SDK는 객체 리스트를 제공 → 속성 추출
                rows: List[List[Any]] = []
                for r in data or []:
                    try:
                        rows.append([
                            int(getattr(r, "open_time", 0)),
                            getattr(r, "open", "0"),
                            getattr(r, "high", "0"),
                            getattr(r, "low", "0"),
                            getattr(r, "close", "0"),
                            getattr(r, "volume", "0"),
                            int(getattr(r, "close_time", 0)),
                            getattr(r, "quote_asset_volume", "0"),
                            getattr(r, "number_of_trades", 0),
                            getattr(r, "taker_buy_base_volume", "0"),
                            getattr(r, "taker_buy_quote_volume", "0"),
                            0,
                        ])
                    except Exception:
                        continue
                if rows:
                    return _kline_to_df(rows)
        return None
    except Exception as e:
        logger.info("SDK klines fetch failed: %s", e)
        return None


def _rest_fetch_klines(symbol: str, interval: str, limit: int = 200) -> pd.DataFrame:
    """REST /fapi/v1/klines"""
    if requests is None:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
    base = FAPI_TESTNET_BASE if USE_TESTNET else FAPI_BASE
    url = f"{base}/fapi/v1/klines"
    try:
        r = requests.get(url, params={"symbol": symbol, "interval": interval, "limit": int(limit)}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        return _kline_to_df(data)
    except Exception as e:
        logger.info("REST klines fetch failed: %s", e)
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])


def fetch_ohlcv(symbol: str, interval: str = "5m", limit: int = 200) -> pd.DataFrame:
    """OHLCV 수집(SDK 우선 → REST 폴백) + 부분봉 제거 + 표준화 컬럼 반환."""
    df = _sdk_fetch_klines(symbol, interval, limit)
    if df is None or df.empty:
        df = _rest_fetch_klines(symbol, interval, limit)
    df = _trim_to_closed(df, interval)
    return df


def fetch_orderbook(symbol: str, limit: int = 50) -> Dict[str, Any]:
    """오더북(depth) 수집(REST). SDK 경로는 버전 의존성이 커서 REST를 기본 사용."""
    if requests is None:
        return {"bids": [], "asks": [], "timestamp": _now_ms()}
    base = FAPI_TESTNET_BASE if USE_TESTNET else FAPI_BASE
    url = f"{base}/fapi/v1/depth"
    try:
        r = requests.get(url, params={"symbol": symbol, "limit": int(limit)}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        bids = [[_to_float(p), _to_float(q)] for p, q in data.get("bids", [])]
        asks = [[_to_float(p), _to_float(q)] for p, q in data.get("asks", [])]
        return {"bids": bids, "asks": asks, "timestamp": _now_ms()}
    except Exception as e:
        logger.info("REST depth fetch failed: %s", e)
        return {"bids": [], "asks": [], "timestamp": _now_ms()}


def fetch_funding_rate(symbol: str) -> float:
    """최근 펀딩비율(%) 하나를 반환. REST /fapi/v1/fundingRate?limit=1"""
    if requests is None:
        return 0.0
    base = FAPI_TESTNET_BASE if USE_TESTNET else FAPI_BASE
    url = f"{base}/fapi/v1/fundingRate"
    try:
        r = requests.get(url, params={"symbol": symbol, "limit": 1}, timeout=HTTP_TIMEOUT)
        r.raise_for_status()
        data = r.json()
        if isinstance(data, list) and data:
            rate = float(data[-1].get("fundingRate", 0.0))
            return rate * 100.0  # percentage
    except Exception as e:
        logger.info("REST fundingRate fetch failed: %s", e)
    return 0.0


# ----------------------------------------------------------------------------
# Indicators
# ----------------------------------------------------------------------------
def add_indicators(df: Optional[pd.DataFrame]) -> pd.DataFrame:
    """RSI(14), SMA_20, 변동성(20) 컬럼을 추가. df가 None/empty면 빈 DF를 반환."""
    if df is None or len(df) == 0:
        return pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume", "RSI", "SMA_20", "volatility"])

    out = df.copy()
    try:
        close = pd.to_numeric(out["close"], errors="coerce")
        # RSI(14) — Wilder
        delta = close.diff()
        gain = (delta.where(delta > 0, 0.0)).rolling(window=14, min_periods=14).mean()
        loss = (-delta.where(delta < 0, 0.0)).rolling(window=14, min_periods=14).mean()
        rs = gain / (loss.replace(0, np.nan))
        rsi = 100.0 - (100.0 / (1.0 + rs))
        out["RSI"] = rsi.bfill().fillna(50.0)

        # SMA_20
        out["SMA_20"] = close.rolling(window=20, min_periods=1).mean()

        # 변동성(20) — 수익률 표준편차
        ret = close.pct_change()
        out["volatility"] = ret.rolling(window=20, min_periods=5).std().fillna(0.0)
    except Exception as e:
        logger.info("add_indicators failed: %s", e)
        # 최소 컬럼 보장
        for c in ["RSI", "SMA_20", "volatility"]:
            if c not in out.columns:
                out[c] = 0.0
    return out


def compute_atr(df: Optional[pd.DataFrame], window: int = 14) -> pd.Series:
    """ATR(14) 계산. df가 비면 빈 시리즈 반환."""
    if df is None or len(df) == 0:
        return pd.Series(dtype=float)
    high = pd.to_numeric(df["high"], errors="coerce")
    low = pd.to_numeric(df["low"], errors="coerce")
    close = pd.to_numeric(df["close"], errors="coerce")
    prev_close = close.shift(1)
    tr = pd.concat([
        (high - low),
        (high - prev_close).abs(),
        (low - prev_close).abs(),
    ], axis=1).max(axis=1)
    atr = tr.rolling(window=window, min_periods=1).mean()
    return atr


def compute_orderbook_stats(ob: Optional[Dict[str, Any]]) -> Dict[str, float]:
    """오더북 요약: 스프레드(bps), 수량 불균형."""
    if not isinstance(ob, dict):
        return {"spread": 0.0, "imbalance": 0.0}
    bids = ob.get("bids") or []
    asks = ob.get("asks") or []
    if not bids or not asks:
        return {"spread": 0.0, "imbalance": 0.0}

    best_bid = float(bids[0][0])
    best_ask = float(asks[0][0])
    mid = (best_bid + best_ask) / 2.0 if (best_bid > 0 and best_ask > 0) else 0.0
    spread_bps = ((best_ask - best_bid) / mid * 10_000.0) if mid > 0 else 0.0

    bid_qty = float(sum(q for _, q in bids[:10]))
    ask_qty = float(sum(q for _, q in asks[:10]))
    tot = bid_qty + ask_qty
    imbalance = (bid_qty - ask_qty) / tot if tot > 0 else 0.0
    return {"spread": float(spread_bps), "imbalance": float(imbalance)}


def compute_recent_price_sequence(df: Optional[pd.DataFrame], n: int = 10) -> List[float]:
    """최근 n개 종가 시퀀스(리스트)"""
    if df is None or len(df) == 0:
        return [0.0] * n
    closes = pd.to_numeric(df["close"], errors="coerce").tail(n).tolist()
    if len(closes) < n:
        closes = [0.0] * (n - len(closes)) + closes
    return [float(x) for x in closes]


def compute_support_resistance(df: Optional[pd.DataFrame], lookback: int = 50) -> Dict[str, float]:
    """최근 고저점(lookback) 기반 지지/저항 레벨"""
    if df is None or len(df) == 0:
        return {"recent_high": 0.0, "recent_low": 0.0}
    window = max(2, int(lookback))
    recent = df.tail(window)
    return {
        "recent_high": float(pd.to_numeric(recent["high"], errors="coerce").max()),
        "recent_low": float(pd.to_numeric(recent["low"], errors="coerce").min()),
    }


def compute_relative_volume(df: Optional[pd.DataFrame], lookback: int = 20) -> float:
    """현재봉(or 최근봉) 거래량 / 과거 lookback 기간 중앙값 → 상대거래량(배)"""
    if df is None or len(df) == 0:
        return 1.0
    vol = pd.to_numeric(df["volume"], errors="coerce")
    ref = vol.tail(lookback + 1)
    if len(ref) < 2:
        return 1.0
    cur = float(ref.iloc[-1])
    hist = ref.iloc[:-1]
    med = float(hist.median()) if len(hist) else 1.0
    if med <= 0:
        return 1.0
    return float(cur / med)


def compute_trend_filter(df: Optional[pd.DataFrame]) -> Dict[str, Any]:
    """
    일봉 추세 필터
    - 5m 데이터가 들어와도 내부에서 일단 시간열을 datetime으로 변환 후 일 단위로 다운샘플.
    - MA_fast(5), MA_slow(20) 교차로 추세 판정.
    """
    if df is None or len(df) == 0:
        return {"daily_uptrend": False, "trend_strength": 0.0}

    try:
        ts = pd.to_datetime(df["timestamp"], errors="coerce", utc=True)
        tmp = pd.DataFrame({"close": pd.to_numeric(df["close"], errors="coerce")}, index=ts)
        daily = tmp.resample("1D").last().dropna()
        if len(daily) < 5:
            return {"daily_uptrend": False, "trend_strength": 0.0}

        ma_fast = daily["close"].rolling(window=5, min_periods=5).mean()
        ma_slow = daily["close"].rolling(window=20, min_periods=5).mean()
        up = bool(ma_fast.iloc[-1] > (ma_slow.iloc[-1] if not np.isnan(ma_slow.iloc[-1]) else ma_fast.iloc[-1]))
        # 간단한 강도: (fast - slow) / slow (abs 0~)
        denom = ma_slow.iloc[-1] if not np.isnan(ma_slow.iloc[-1]) and ma_slow.iloc[-1] != 0 else ma_fast.iloc[-1]
        strength = float((ma_fast.iloc[-1] - (denom)) / (denom)) if denom else 0.0
        return {"daily_uptrend": up, "trend_strength": strength}
    except Exception as e:
        logger.info("compute_trend_filter failed: %s", e)
        return {"daily_uptrend": False, "trend_strength": 0.0}


# ----------------------------------------------------------------------------
# Composite fetch (for payload)
# ----------------------------------------------------------------------------
def fetch_data(
    symbol: str,
    interval: str = "5m",
    ohlcv_limit: int = 200,
    orderbook_limit: int = 50,
    include_orderbook: bool = True,
) -> Dict[str, Any]:
    """
    신호 엔트리용 기본 데이터 묶음 수집.
    반환:
      {
        "ohlcv": DataFrame(with indicators + funding_rate_pct),
        "orderbook": {bids, asks, timestamp} | None,
        "times": {"t0": iso, "tN": iso}
      }
    """
    df = fetch_ohlcv(symbol, interval=interval, limit=ohlcv_limit)
    df = add_indicators(df)

    # 최신 펀딩 비율(%) 부착
    try:
        fr_pct = fetch_funding_rate(symbol)
        if df is not None and not df.empty:
            df = df.copy()
            df["funding_rate_pct"] = float(fr_pct)
    except Exception as e:
        logger.info("funding rate attach failed: %s", e)

    ob = fetch_orderbook(symbol, limit=orderbook_limit) if include_orderbook else None

    times = {}
    try:
        if df is not None and not df.empty:
            times = {"t0": str(df["timestamp"].iloc[0]), "tN": str(df["timestamp"].iloc[-1])}
    except Exception:
        times = {}

    return {"ohlcv": df, "orderbook": ob, "times": times}


def fetch_mtf_raw(symbol: str) -> Dict[str, pd.DataFrame]:
    """MTF 원천: 1h/4h/1d OHLCV dict 반환. 키는 {"1h","4h","1d"}"""
    out: Dict[str, pd.DataFrame] = {}
    for tf in ("1h", "4h", "1d"):
        try:
            df = fetch_ohlcv(symbol, interval=tf, limit=200)
            out[tf] = df if df is not None else pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
        except Exception as e:
            logger.info("fetch_mtf_raw %s failed: %s", tf, e)
            out[tf] = pd.DataFrame(columns=["timestamp", "open", "high", "low", "close", "volume"])
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
