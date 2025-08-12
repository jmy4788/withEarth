from __future__ import annotations

"""
helpers/signals.py – refactor #1 (2025-08-11, KST)

Goals
- Stable with binance-sdk-derivatives-trading-usds-futures==1.0.0
- No import mistakes (e.g., importing get_overview from wrong module)
- Deterministic payload → Gemini → decision → trade pipeline
- Clear ENV knobs for risk and thresholds

Public API (kept compatible with app.py):
- get_overview() -> Tuple[pd.DataFrame, pd.DataFrame]
- generate_signal(symbol: str) -> Dict[str, Any]
- manage_trade(symbol: str) -> Dict[str, Any]
"""

import json
import logging
import math
import os
from dataclasses import dataclass
from decimal import Decimal
from typing import Any, Dict, List, Optional, Tuple

import numpy as np
import pandas as pd

# --- Internal helpers (robust imports with fallbacks) ---
try:  # GCS/local logging helpers
    from .utils import LOG_DIR, gcs_enabled, gcs_append_csv_row
except Exception:  # pragma: no cover
    LOG_DIR = os.path.join(os.getcwd(), "logs")

    def gcs_enabled() -> bool:
        return False

    def gcs_append_csv_row(*args, **kwargs):  # noqa
        return None

# Data layer
try:
    from .data_fetch import (
        fetch_data,
        fetch_mtf_raw,
        add_indicators,
        compute_atr,
        compute_support_resistance,
        compute_recent_price_sequence,
        compute_orderbook_stats,
        compute_trend_filter,
        compute_relative_volume,
        fetch_orderbook,
    )
except Exception as e:  # pragma: no cover
    logging.exception("data_fetch import failed: %s", e)
    raise

# Trading client wrappers (SDK v1.0.0 friendly)
try:
    from .binance_client import (
        get_overview as _bn_get_overview,  # re-exported below
        cancel_open_orders,
        set_position_mode,
        set_margin_type,
        set_leverage,
        load_symbol_filters,
        ensure_min_notional,
        normalize_price_with_mode,
        place_market_order,
        place_limit_order,
        place_bracket_orders,
        build_entry_and_brackets,
        get_position,
    )
except Exception as e:  # pragma: no cover
    logging.exception("binance_client import failed: %s", e)
    raise

# Gemini predictor
try:
    from .predictor import get_gemini_prediction, should_predict
except Exception as e:  # pragma: no cover
    logging.exception("predictor import failed: %s", e)
    raise

# -----------------
# Config knobs
# -----------------
MIN_PROB = float(os.getenv("MIN_PROB", "0.60"))
RR_MIN = float(os.getenv("RR_MIN", "1.20"))
ATR_MULT_TP = float(os.getenv("ATR_MULT_TP", "1.8"))
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.0"))
DEFAULT_LEVERAGE = int(os.getenv("LEVERAGE", "5"))
MARGIN_TYPE = os.getenv("MARGIN_TYPE", "ISOLATED").upper()
POSITION_MODE = os.getenv("POSITION_MODE", "ONEWAY").upper()
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", "2.0"))
ENTRY_MAX_RETRIES = int(os.getenv("ENTRY_MAX_RETRIES", "2"))
RISK_USDT = float(os.getenv("RISK_USDT", "100"))
HORIZON_MIN = int(os.getenv("HORIZON_MIN", "60"))
ENTRY_MODE = os.getenv("ENTRY_MODE", "LIMIT").upper()  # MARKET | LIMIT
LIMIT_TTL_SEC = int(os.getenv("LIMIT_TTL_SEC", "15"))   # 한 주문당 대기 시간
LIMIT_POLL_SEC = float(os.getenv("LIMIT_POLL_SEC", "1.0"))
LIMIT_MAX_REPRICES = int(os.getenv("LIMIT_MAX_REPRICES", "3"))
LIMIT_MAX_SLIPPAGE_BPS = float(os.getenv("LIMIT_MAX_SLIPPAGE_BPS", "2.0"))  # entry 대비 허용 슬리피지 상한(bps)

# --------------
# Data classes
# --------------
@dataclass
class Signal:
    symbol: str
    direction: str  # long|short|hold
    prob: float
    entry: float
    tp: float
    sl: float
    risk_scalar: float
    reasoning: str = ""

    def as_dict(self) -> Dict[str, Any]:
        return {
            "symbol": self.symbol,
            "direction": self.direction,
            "prob": self.prob,
            "entry": self.entry,
            "tp": self.tp,
            "sl": self.sl,
            "risk_scalar": self.risk_scalar,
            "reasoning": self.reasoning,
        }

# --------------------
# Utility helpers
# --------------------
def _df_ok(x) -> bool:
    return isinstance(x, pd.DataFrame) and not x.empty

def _ob_stats_to_dict(stats) -> dict:
    try:
        if isinstance(stats, dict):
            return {"imbalance": float(stats.get("imbalance", 0.0)), "spread": float(stats.get("spread", 0.0))}
        if isinstance(stats, (tuple, list)):
            imb = float(stats[0]) if len(stats) >= 1 else 0.0
            spr = float(stats[1]) if len(stats) >= 2 else 0.0
            return {"imbalance": imb, "spread": spr}
    except Exception:
        pass
    return {"imbalance": 0.0, "spread": 0.0}

def _mid_from_orderbook(ob) -> Optional[float]:
    try:
        if ob is None:
            return None
        # accept dict or (bids, asks) or (dict, ts)
        bids = asks = None
        if isinstance(ob, dict):
            bids, asks = ob.get("bids"), ob.get("asks")
        elif isinstance(ob, (tuple, list)):
            if len(ob) >= 2 and isinstance(ob[0], (list, tuple)) and isinstance(ob[1], (list, tuple)):
                bids, asks = ob[0], ob[1]
            elif len(ob) >= 1 and isinstance(ob[0], dict):
                bids, asks = ob[0].get("bids"), ob[0].get("asks")
        if not bids or not asks:
            return None
        bp, ap = float(bids[0][0]), float(asks[0][0])
        return (bp + ap) / 2.0
    except Exception:
        return None
def _sr_levels(df: pd.DataFrame) -> Dict[str, float]:
    if df is None or df.empty:
        return {"recent_high": 0.0, "recent_low": 0.0}
    try:
        sr = compute_support_resistance(df)
        return {"recent_high": float(sr.get("recent_high", 0.0)), "recent_low": float(sr.get("recent_low", 0.0))}
    except Exception:
        return {"recent_high": 0.0, "recent_low": 0.0}
    
def _best_quotes(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    """Return (best_bid, best_ask)."""
    try:
        ob = fetch_orderbook(symbol, limit=5)
        if ob and ob.get("bids") and ob.get("asks"):
            bid = float(ob["bids"][0][0])
            ask = float(ob["asks"][0][0])
            return bid, ask
    except Exception:
        pass
    return None, None

def _limit_price_for_side(symbol: str, side: str, desired_entry: float) -> Optional[float]:
    """현재 호가와 슬리피지 예산을 고려해 리밋 가격 산출."""
    bid, ask = _best_quotes(symbol)
    if bid is None or ask is None:
        return None
    budget = LIMIT_MAX_SLIPPAGE_BPS / 1e4
    if side.upper() == "BUY":
        # 롱: bid 기준으로 시작, 너무 높게 쫓지 않도록 desired_entry*(1+budget) 상한
        raw = min(ask, max(bid, desired_entry * (1.0 - 0.0000)))  # desired_entry 아래로는 굳이 낮추지 않음
        capped = min(raw, desired_entry * (1.0 + budget))
        return normalize_price_with_mode(symbol, capped)
    else:
        # 숏: ask 기준으로 시작, 너무 낮게 던지지 않도록 desired_entry*(1-budget) 하한
        raw = max(bid, min(ask, desired_entry * (1.0 + 0.0000)))
        capped = max(raw, desired_entry * (1.0 - budget))
        return normalize_price_with_mode(symbol, capped)

def _position_qty_after_fill(symbol: str, side: str) -> float:
    """체결 후 포지션 수량(절대값)을 읽어온다. 없으면 0."""
    p = get_position(symbol) or {}
    amt = float(p.get("positionAmt") or p.get("positionAmount") or 0.0)
    # ONEWAY 기준: 롱>0, 숏<0
    if side.upper() == "BUY" and amt > 0:
        return abs(amt)
    if side.upper() == "SELL" and amt < 0:
        return abs(amt)
    return 0.0

def _enter_limit_then_brackets(symbol: str, side: str, qty: float, desired_entry: float, tp: float, sl: float) -> Dict[str, Any]:
    """
    1) 최우선호가 기반 리밋 제출 (GTC)
    2) TTL 동안 폴링하며 포지션 증가 감지
    3) 미체결이면 재호가(최대 LIMIT_MAX_REPRICES 회)
    4) 체결 시 reduce-only TP/SL 제출
    """
    attempt = 0
    last_order = None
    filled_qty = 0.0
    while attempt <= LIMIT_MAX_REPRICES:
        attempt += 1
        price = _limit_price_for_side(symbol, side, desired_entry)
        if price is None or price <= 0:
            raise RuntimeError("orderbook unavailable for limit entry")

        # 기존 미체결 리밋은 정리(간단화)
        try:
            cancel_open_orders(symbol)
        except Exception as e:
            logging.info("cancel_open_orders(%s) failed (non-fatal): %s", symbol, e)

        # 리밋 제출
        last_order = place_limit_order(symbol, side, qty, price, time_in_force="GTC", reduce_only=False)

        # 폴링하며 체결 감시
        deadline = time.time() + LIMIT_TTL_SEC
        while time.time() < deadline:
            time.sleep(LIMIT_POLL_SEC)
            filled_qty = _position_qty_after_fill(symbol, side)
            if filled_qty > 0:
                # 체결 확인 → 브래킷(RO) 생성
                br = place_bracket_orders(symbol, side, filled_qty, take_profit=tp, stop_loss=sl)
                return {"entry_order": last_order, "brackets": br, "filled_qty": filled_qty, "entry_price": price}
        # TTL 내 미체결 → 다음 시도(재호가)
    # 모든 시도 실패
    raise TimeoutError(f"limit entry not filled within TTL×reprices (ttl={LIMIT_TTL_SEC}s, n={LIMIT_MAX_REPRICES})")

def _latest_asof(df: pd.DataFrame, ref_ts: pd.Timestamp) -> pd.Series:
    """ref_ts(5m 마지막 '닫힌' 봉 기준) 이전/동일 시각 중 가장 최근 1행만 반환."""
    if df is None or df.empty: 
        return pd.Series(dtype="float64")
    d = df[df["timestamp"] <= ref_ts]
    if d.empty:
        return pd.Series(dtype="float64")
    return d.iloc[-1]

# -----------------------------
# Re-export account overview
# -----------------------------

def get_overview():
    return _bn_get_overview()

# ---------------------------------
# Payload builder & signal generator
# ---------------------------------

def _build_payload(symbol: str) -> Tuple[Dict[str, Any], pd.DataFrame, Optional[Dict[str, Any]]]:
    base = fetch_data(symbol, interval="5m", ohlcv_limit=200, orderbook_limit=50, include_orderbook=True)
    ohlcv = base.get("ohlcv") if isinstance(base.get("ohlcv"), pd.DataFrame) else pd.DataFrame()
    if not _df_ok(ohlcv):
        # placeholder DF to keep upstream logic stable
        ohlcv = add_indicators(None)

    last = ohlcv.iloc[-1] if _df_ok(ohlcv) else pd.Series({})
    ob = base.get("orderbook") if isinstance(base, dict) else None

    # MTF extras
    mtf = fetch_mtf_raw(symbol)
    extra: Dict[str, Any] = {}
    for tf, df in mtf.items():
        ind = add_indicators(df)
        atr = float(compute_atr(df, window=14).iloc[-1]) if _df_ok(df) else 0.0
        key = {"1h": "_1h", "4h": "_4h", "1d": "_1d"}.get(tf, "")
        if _df_ok(ind):
            extra.update({
                f"RSI{key}": float(ind["RSI"].iloc[-1]) if "RSI" in ind else 50.0,
                f"volatility{key}": float(ind["volatility"].iloc[-1]) if "volatility" in ind else 0.0,
                f"SMA20{key}": float(ind["SMA_20"].iloc[-1]) if "SMA_20" in ind else 0.0,
                f"ATR{key}": float(atr),
                f"relative_volume{key}": float(compute_relative_volume(df)) if _df_ok(df) else 1.0,
                f"recent_high{key}": float(df["high"].tail(50).max()) if _df_ok(df) else 0.0,
                f"recent_low{key}": float(df["low"].tail(50).min()) if _df_ok(df) else 0.0,
            })

    price_seq = compute_recent_price_sequence(ohlcv, n=10) if _df_ok(ohlcv) else [0.0] * 10
    sr5 = _sr_levels(ohlcv)
    raw_stats = compute_orderbook_stats(ob) if isinstance(ob, dict) else {"imbalance": 0.0, "spread": 0.0}
    ob_stats = _ob_stats_to_dict(raw_stats)
    trend = compute_trend_filter(ohlcv) if _df_ok(ohlcv) else {"daily_uptrend": False, "trend_strength": 0.0}
    atr5_series = compute_atr(ohlcv, window=14) if _df_ok(ohlcv) else None
    payload = {
        "pair": symbol,
        "entry_5m": {
            "close": float(last.get("close", 0.0)),
            "rsi": float(ohlcv["RSI"].iloc[-1]) if _df_ok(ohlcv) and "RSI" in ohlcv else 50.0,
            "volatility": float(ohlcv["volatility"].iloc[-1]) if _df_ok(ohlcv) and "volatility" in ohlcv else 0.0,
            "sma20": float(ohlcv["SMA_20"].iloc[-1]) if _df_ok(ohlcv) and "SMA_20" in ohlcv else 0.0,
            "funding_rate_pct": float(last.get("funding_rate_pct", 0.0)),
            "sentiment": float(last.get("x_sentiment", 0.0)),
            "timestamp": str(last.get("timestamp", "")),
        },
        "extra": {
            "ATR_5m": float(atr5_series.iloc[-1]) if getattr(atr5_series, "size", 0) else 0.0,
            "relative_volume_5m": float(compute_relative_volume(ohlcv)) if _df_ok(ohlcv) else 1.0,
            "recent_high_5m": float(ohlcv["high"].tail(50).max()) if _df_ok(ohlcv) else 0.0,
            "recent_low_5m": float(ohlcv["low"].tail(50).min()) if _df_ok(ohlcv) else 0.0,
            "orderbook_imbalance": float(ob_stats.get("imbalance", 0.0)),
            "orderbook_spread": float(ob_stats.get("spread", 0.0)),
            **extra,
        },
        "times": base.get("times", {}),
        "price_sequence": price_seq,
        "sr_levels": sr5,
        "relative_volume": float(compute_relative_volume(ohlcv)) if _df_ok(ohlcv) else 1.0,
        "trend_filter": trend,
    }
    payload["horizon_min"] = HORIZON_MIN
    return payload, ohlcv, ob


def _propose_levels(direction: str, entry: float, atr5: float, sr: Dict[str, float]) -> Tuple[float, float]:
    if entry <= 0:
        return 0.0, 0.0
    atr5 = float(max(atr5, 1e-8))
    if direction == "long":
        tp = max(entry + ATR_MULT_TP * atr5, sr.get("recent_high", 0.0))
        sl = min(entry - ATR_MULT_SL * atr5, sr.get("recent_low", entry * 0.98))
    elif direction == "short":
        tp = min(entry - ATR_MULT_TP * atr5, sr.get("recent_low", entry * 0.98))
        sl = max(entry + ATR_MULT_SL * atr5, sr.get("recent_high", 0.0))
    else:
        return 0.0, 0.0
    return float(tp), float(sl)

def _risk_ok(entry: float, tp: float, sl: float) -> Tuple[bool, float]:
    try:
        if entry <= 0 or tp <= 0 or sl <= 0:
            return False, 0.0
        if tp <= entry and sl >= entry:
            return False, 0.0
        if tp > entry:
            rr = (tp - entry) / max(entry - sl, 1e-8)
        else:
            rr = (entry - tp) / max(sl - entry, 1e-8)
        return rr >= RR_MIN, float(rr)
    except Exception:
        return False, 0.0
    
def generate_signal(symbol: str) -> Dict[str, Any]:
    """Build payload → Gemini prediction → TP/SL proposal with guards."""
    payload, ohlcv, ob = _build_payload(symbol)

    # Gate: allow prediction only if data is sufficiently volatile
    if isinstance(ohlcv, pd.DataFrame) and not should_predict(ohlcv.tail(50)):
        preview = {k: payload[k] for k in ("pair", "entry_5m", "extra") if k in payload}
        # 게이트 상황에서도 미리보기/entry 기본값 제공
        entry0 = float(preview.get("entry_5m", {}).get("close", 0.0)) if isinstance(preview.get("entry_5m"), dict) else 0.0
        return {
            "symbol": symbol,
            "direction": "hold",
            "reason": "low_volatility_gate",
            "prob": 0.5,
            "entry": entry0,
            "tp": 0.0,
            "sl": 0.0,
            "risk_ok": False,
            "rr": 0.0,
            "payload_preview": preview,
        }

    pred = get_gemini_prediction(payload, symbol=symbol)
    logging.info("gemini_pred %s -> %s", symbol, pred)
    direction = str(pred.get("direction", "hold")).lower()
    prob = float(pred.get("prob", 0.5))
    reasoning = str(pred.get("reasoning", ""))

    # Mid price & ATR for levels
    mid = _mid_from_orderbook(ob)  # payload dict엔 orderbook을 넣지 않으므로 ob 사용
    entry = float(mid if mid is not None else payload["entry_5m"].get("close", 0.0))
    ex = payload.get("extra", {}) if isinstance(payload.get("extra"), dict) else {}
    atr5 = float(ex.get("ATR_5m", 0.0))
    atr1h = float(ex.get("ATR_1h", 0.0))

    # 5분봉 → HORIZON_MIN 분으로의 √시간 스케일 (대략적 변동성 시간 스케일링)
    scale = max(1.0, (HORIZON_MIN / 5.0) ** 0.5)
    atr_for_horizon = atr1h if (HORIZON_MIN >= 45 and atr1h > 0) else (atr5 * scale)

    tp, sl = _propose_levels(direction, entry, atr_for_horizon, payload.get("sr_levels", {}))
    ok_rr, rr = _risk_ok(entry, tp, sl)
    # Conditionally relax RR if the model is confident
    relax_th = float(os.getenv("HIGH_PROB_RELAX_RR", "0.70"))     # 확률 임계
    relax_rr = float(os.getenv("RR_MIN_HIGH_PROB", "1.05"))       # 높은 확률일 때 허용 RR
    if not ok_rr and prob >= relax_th:
        ok_rr = rr >= relax_rr
    # === end paste ===

    return {
        "symbol": symbol,
        "direction": direction,
        "prob": prob,
        "entry": float(entry),
        "tp": float(tp),
        "sl": float(sl),
        "risk_ok": bool(ok_rr),
        "risk_scalar": float(max(0.2, min(1.5, prob * (1.0 if ok_rr else 0.5)))),
        "rr": float(rr),
        "reasoning": reasoning,
        "payload_preview": {k: payload[k] for k in ("pair", "entry_5m", "extra") if k in payload},
    }

# -----------------
# Execution layer
# -----------------

def _guard_spread(symbol: str, max_bps: float = MAX_SPREAD_BPS) -> Tuple[bool, float, Dict[str, Any]]:
    ob = fetch_orderbook(symbol, limit=25)
    stats = _ob_stats_to_dict(compute_orderbook_stats(ob) if ob else {"spread": 0.0})
    spr = float(stats.get("spread", 0.0))
    ok = spr <= max_bps
    # ob는 dict가 아닐 수 있으므로 그대로 반환
    return ok, spr, ob or {}



def _size_from_balance(symbol: str, entry: float, risk_scalar: float) -> float:
    # Use simple notional target (RISK_USDT * risk_scalar); ensure step/minNotional via client helper
    notional = max(RISK_USDT * max(risk_scalar, 0.2), 5.0)
    qty = notional / max(entry, 1e-8)
    try:
        filters = load_symbol_filters(symbol)
        qty = float(ensure_min_notional(symbol, qty, entry, filters))
    except Exception:
        qty = float(qty)
    return max(qty, 0.0)

def manage_trade(symbol: str, sig: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """
    - sig 없으면 generate_signal(symbol) 사용
    - 조건 통과 시 ENTRY_MODE에 따라 MARKET or LIMIT 진입
    - LIMIT은 체결 모니터링 후 reduce-only TP/SL 자동 세팅
    """
    try:
        if sig is None:
            sig = generate_signal(symbol)

        direction = str(sig.get("direction", "hold")).lower()
        prob = float(sig.get("prob", 0.5))
        entry = float(sig.get("entry", 0.0))
        tp = float(sig.get("tp", 0.0))
        sl = float(sig.get("sl", 0.0))
        risk_ok = bool(sig.get("risk_ok", False))
        rr = float(sig.get("rr", 0.0))
        preview = sig.get("payload_preview", {})
        reason = sig.get("reason") or sig.get("reasoning", "")

        # 거래 조건 미충족 → 요약만 반환
        if direction not in ("long", "short") or not risk_ok or entry <= 0 or tp <= 0 or sl <= 0:
            return {
                "symbol": symbol,
                "action": "hold",
                "direction": direction,
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "prob": prob,
                "risk_ok": False,
                "rr": rr,
                "reason": reason if reason else "no_trade_conditions",
                "payload_preview": preview,
            }

        # 계정 설정
        try: set_position_mode(POSITION_MODE)
        except Exception as e: logging.warning("set_position_mode: %s", e)
        try: set_margin_type(symbol, MARGIN_TYPE)
        except Exception as e: logging.warning("set_margin_type: %s", e)
        try: set_leverage(symbol, DEFAULT_LEVERAGE)
        except Exception as e: logging.warning("set_leverage: %s", e)

        # 위험금액 기반 수량
        risk_scalar = float(sig.get("risk_scalar", 1.0))
        qty = _size_from_balance(symbol, entry, risk_scalar)

        side = "BUY" if direction == "long" else "SELL"

        if ENTRY_MODE == "MARKET":
            # 마켓 → 진입 + 브래킷 (SDK 래퍼 제공)
            entry_res, br_res = build_entry_and_brackets(symbol, side, qty, tp, sl)  # entry_price 인자 없음
            return {
                "symbol": symbol,
                "action": direction,
                "direction": direction,
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "prob": prob,
                "risk_ok": True,
                "rr": rr,
                "payload_preview": preview,
                "order": {"entry": entry_res, "brackets": br_res},
                "mode": "MARKET",
            }

        # LIMIT 모드
        exec_res = _enter_limit_then_brackets(symbol, side, qty, entry, tp, sl)
        return {
            "symbol": symbol,
            "action": direction,
            "direction": direction,
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "prob": prob,
            "risk_ok": True,
            "rr": rr,
            "payload_preview": preview,
            "order": exec_res,
            "mode": "LIMIT",
        }

    except Exception as e:
        logging.exception("manage_trade failed for %s", symbol)
        return {"symbol": symbol, "error": str(e)}
    
__all__ = [
    "get_overview",
    "generate_signal",
    "manage_trade",
]
