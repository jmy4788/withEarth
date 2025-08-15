from __future__ import annotations

"""
helpers/signals.py — refactor (2025-08-12, KST)

역할
- 데이터 수집(5m 엔트리 + 1h/4h/1d 컨텍스트) → 페이로드 생성
- Gemini로 방향/확률 의사결정 수신(스키마 강제는 predictor.py에서 처리)
- ATR×계수 및 S/R 기반 TP/SL 제안, 손익비(RR) 검증
- (옵션) 주문 실행: MARKET 또는 LIMIT(+TTL/재호가) → 리듀스온리 브래킷(TP/SL)

호환성
- app.py 및 기존 모듈이 기대하는 공개 API를 유지합니다:
    get_overview() -> dict
    generate_signal(symbol: str) -> Dict[str, Any]
    manage_trade(symbol: str) -> Dict[str, Any]
"""

from typing import Any, Dict, List, Optional, Tuple
import json
import logging
import math
import os
from dataclasses import dataclass
from datetime import datetime, timezone

import numpy as np
import pandas as pd

# ----------------------------
# 프로젝트 공용 유틸 (선택적)
# ----------------------------
try:
    from .utils import LOG_DIR, gcs_enabled, gcs_append_csv_row, log_event  # type: ignore
except Exception:  # pragma: no cover
    LOG_DIR = os.path.join(os.getcwd(), "logs")

    def gcs_enabled() -> bool:
        return False

    def gcs_append_csv_row(*args, **kwargs) -> None:  # noqa: D401
        return None

# ----------------------------
# 데이터 레이어
# ----------------------------
try:
    from .data_fetch import (
        fetch_data,
        fetch_mtf_raw,
        add_indicators,
        compute_atr,
        compute_orderbook_stats,
        compute_recent_price_sequence,
        compute_support_resistance,
        compute_relative_volume,
        compute_trend_filter,
        fetch_orderbook,
    )  # type: ignore
except Exception as e:  # pragma: no cover
    raise

# ----------------------------
# 모델(의사결정) 레이어
# ----------------------------
try:
    from .predictor import get_gemini_prediction  # type: ignore
except Exception as e:  # pragma: no cover
    raise

# ----------------------------
# 트레이딩 클라이언트 (모듈형 SDK)
# ----------------------------
try:
    from .binance_client import (
        get_overview as _bn_get_overview,
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
        get_position,
    )  # type: ignore
except Exception as e:  # pragma: no cover
    raise

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ================================
# 환경 변수(리스크/실행 파라미터)
# ================================
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

HORIZON_MIN = int(os.getenv("HORIZON_MIN", "60"))  # 모델 의사결정 유효 시간(분)

ENTRY_MODE = os.getenv("ENTRY_MODE", "LIMIT").upper()  # MARKET | LIMIT
LIMIT_TTL_SEC = float(os.getenv("LIMIT_TTL_SEC", "15"))
LIMIT_POLL_SEC = float(os.getenv("LIMIT_POLL_SEC", "1.0"))
LIMIT_MAX_REPRICES = int(os.getenv("LIMIT_MAX_REPRICES", "3"))
LIMIT_MAX_SLIPPAGE_BPS = float(os.getenv("LIMIT_MAX_SLIPPAGE_BPS", "2.0"))

# 높은 확률일 때 RR 완화 옵션
HIGH_PROB_RELAX_RR = float(os.getenv("HIGH_PROB_RELAX_RR", "0.70"))
RR_MIN_HIGH_PROB = float(os.getenv("RR_MIN_HIGH_PROB", "1.05"))

TRADES_CSV = os.path.join(LOG_DIR, "trades.csv")


# =====================
# 데이터 구조
# =====================
@dataclass
class SignalOut:
    symbol: str
    direction: str = "hold"
    prob: float = 0.5
    entry: float = 0.0
    tp: float = 0.0
    sl: float = 0.0
    risk_scalar: float = 1.0
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


# ==============
# 유틸리티
# ==============
def _df_ok(df: Optional[pd.DataFrame]) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty and all(c in df.columns for c in ("timestamp", "open", "high", "low", "close", "volume"))


def _ob_stats_to_dict(stats: Any) -> Dict[str, float]:
    try:
        return {
            "imbalance": float(stats.get("imbalance", 0.0)),
            "spread": float(stats.get("spread", 0.0)),
        }
    except Exception:
        return {"imbalance": 0.0, "spread": 0.0}


def _mid_from_orderbook(ob: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(ob, dict):
        return None
    bids = ob.get("bids") or []
    asks = ob.get("asks") or []
    if not bids or not asks:
        return None
    try:
        bb = float(bids[0][0])
        ba = float(asks[0][0])
        if bb <= 0 or ba <= 0:
            return None
        return (bb + ba) / 2.0
    except Exception:
        return None


def _sr_levels(df: Optional[pd.DataFrame], lookback: int = 50) -> Dict[str, float]:
    if not _df_ok(df):
        return {"recent_high": 0.0, "recent_low": 0.0}
    recent = df.tail(max(2, lookback))
    return {
        "recent_high": float(pd.to_numeric(recent["high"], errors="coerce").max()),
        "recent_low": float(pd.to_numeric(recent["low"], errors="coerce").min()),
    }


def _best_quotes(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    """현재 베스트 bid/ask (없으면 None)."""
    try:
        ob = fetch_orderbook(symbol, limit=5)
        bids = ob.get("bids") if isinstance(ob, dict) else None
        asks = ob.get("asks") if isinstance(ob, dict) else None
        if bids and asks:
            bid = float(bids[0][0])
            ask = float(asks[0][0])
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
        raw = min(ask, max(bid, desired_entry))
        capped = min(raw, desired_entry * (1.0 + budget))
        return normalize_price_with_mode(symbol, capped)
    else:
        # 숏: ask 기준으로 시작, 너무 낮게 던지지 않도록 desired_entry*(1-budget) 하한
        raw = max(bid, min(ask, desired_entry))
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
    LIMIT 진입: TTL 동안 체결 감시 → 미체결 시 재호가(최대 LIMIT_MAX_REPRICES) → 체결되면 브래킷 생성.
    반환 dict 예:
      {
        "entry_order": {...} | None,
        "brackets": {"take_profit": {...}|None, "stop_loss": {...}|None},
        "filled_qty": 0.01,
        "reprices": 1
      }
    """
    side = side.upper()
    price = _limit_price_for_side(symbol, side, desired_entry)
    if price is None or price <= 0:
        raise RuntimeError("Could not determine limit price")

    # 최초 리밋 주문
    entry_res = place_limit_order(symbol, side, quantity=qty, price=price, time_in_force="GTC", reduce_only=False)

    filled = 0.0
    reprices = 0

    # TTL 동안 체결 감시
    deadline = datetime.now(tz=timezone.utc).timestamp() + float(LIMIT_TTL_SEC)
    while datetime.now(tz=timezone.utc).timestamp() < deadline:
        q = _position_qty_after_fill(symbol, side)
        if q >= (qty * 0.95):  # 95% 이상 체결
            filled = q
            break
        time_sleep = max(0.1, float(LIMIT_POLL_SEC))
        try:
            import time as _t
            _t.sleep(time_sleep)
        except Exception:
            pass

    # 미체결 → 재호가 시도
    while filled <= 0 and reprices < LIMIT_MAX_REPRICES:
        reprices += 1
        try:
            cancel_open_orders(symbol)
        except Exception:
            pass
        # 새 기준가 산출
        price = _limit_price_for_side(symbol, side, desired_entry)
        if price is None or price <= 0:
            break
        entry_res = place_limit_order(symbol, side, quantity=qty, price=price, time_in_force="GTC", reduce_only=False)
        deadline = datetime.now(tz=timezone.utc).timestamp() + float(LIMIT_TTL_SEC)
        while datetime.now(tz=timezone.utc).timestamp() < deadline:
            q = _position_qty_after_fill(symbol, side)
            if q >= (qty * 0.95):
                filled = q
                break
            try:
                import time as _t
                _t.sleep(max(0.1, float(LIMIT_POLL_SEC)))
            except Exception:
                pass
        if filled > 0:
            break

    brackets = {"take_profit": None, "stop_loss": None}
    if filled > 0:
        try:
            brackets = place_bracket_orders(symbol, side, filled, take_profit=tp, stop_loss=sl)
        except Exception as e:
            logger.info("placing brackets failed: %s", e)

    return {
        "entry_order": entry_res,
        "brackets": brackets,
        "filled_qty": float(filled),
        "reprices": reprices,
    }


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
    """호환용 래퍼 — 현재는 binance_client.get_overview()를 그대로 재노출."""
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
            "high": float(last.get("high", 0.0)),
            "low": float(last.get("low", 0.0)),
            "open": float(last.get("open", 0.0)),
            "volume": float(last.get("volume", 0.0)),
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
    """
    손익비(RR) 계산:
      long  : (tp - entry) / (entry - sl)
      short : (entry - tp) / (sl - entry)
    방향 모를 때는 long 공식 기준으로 양수/음수만 판정.
    """
    try:
        entry = float(entry)
        tp = float(tp)
        sl = float(sl)
        if entry <= 0 or tp <= 0 or sl <= 0:
            return False, 0.0
        rr_up = (tp - entry)
        rr_dn = (entry - sl)
        if rr_dn <= 0:
            return False, 0.0
        rr = float(rr_up / rr_dn)
        return rr >= RR_MIN, rr
    except Exception:
        return False, 0.0


def _guard_spread(spread_bps: float) -> bool:
    """스프레드 상한 게이트. True면 통과, False면 홀드."""
    try:
        return float(spread_bps) <= float(MAX_SPREAD_BPS)
    except Exception:
        return True


def _size_from_balance(symbol: str, entry: float, sl: float, risk_scalar: float = 1.0) -> float:
    """
    간단 사이징: RISK_USDT / entry (레버리지는 거래소에서 관리)
    - 심볼 필터(minNotional/stepSize)로 최종 보정
    """
    entry = float(entry or 0.0)
    if entry <= 0:
        return 0.0
    notional = max(1.0, float(RISK_USDT) * float(risk_scalar))
    qty = notional / entry
    # 필터 보정
    try:
        f = load_symbol_filters(symbol)
        qty = ensure_min_notional(symbol, qty, price=entry, filters=f)
    except Exception:
        pass
    return float(qty)


# =====================
# 공개 API
# =====================#

def generate_signal(symbol: str) -> Dict[str, Any]:
    """
    심볼별 payload 생성 → Gemini 예측 → RR/게이트 적용 → 실행 or HOLD 요약 반환.
    INFO 이벤트:
      - signal.decision: {symbol, direction, prob, entry, tp, sl, rr, risk_ok}
    """
    payload, ohlcv, ob = _build_payload(symbol)

    # 모델 호출
    res = get_gemini_prediction(payload, symbol=symbol)

    # 가격 및 RR 계산 (기존 로직과 동일하게 해석)
    direction = str(res.get("direction") or "").lower()
    entry = float(res.get("entry") or payload.get("entry_5m") or 0.0)
    tp = float(res.get("tp") or 0.0)
    sl = float(res.get("sl") or 0.0)
    prob = float(res.get("prob", 0.0))
    rr = float(res.get("rr", 0.0))
    risk_ok = bool(res.get("risk_ok", False))

    # 게이트 조건을 통과 못하면 hold 요약 반환(기존 패턴 유지)
    if direction not in ("long", "short") or not risk_ok or entry <= 0 or tp <= 0 or sl <= 0:
        preview = res.get("payload_preview", {})
        out = {
            "symbol": symbol,
            "action": "hold",
            "direction": direction,
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "prob": prob,
            "risk_ok": False,
            "rr": rr,
            "reason": res.get("reasoning", "") or "no_trade_conditions",
            "payload_preview": preview,
        }
        # 의사결정 요약 로깅
        log_event("signal.decision",
                  symbol=symbol, direction=direction, prob=prob,
                  entry=entry, tp=tp, sl=sl, rr=rr, risk_ok=False)
        return out

    # 확률 게이트
    if prob < float(os.getenv("MIN_PROB", "0.60")):
        preview = res.get("payload_preview", {})
        out = {
            "symbol": symbol,
            "action": "hold",
            "direction": direction,
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "prob": prob,
            "risk_ok": False,
            "rr": rr,
            "reason": "prob_below_threshold",
            "payload_preview": preview,
        }
        log_event("signal.decision",
                  symbol=symbol, direction=direction, prob=prob,
                  entry=entry, tp=tp, sl=sl, rr=rr, risk_ok=False)
        return out

    # 여기서부터는 manage_trade() 쪽을 쓰는 루트와 동일하게 동작합니다.
    # (app.py에서 EXECUTE_TRADES=false면 이 결과만 반환합니다.)
    out = {
        "symbol": symbol,
        "action": "enter",
        "direction": direction,
        "entry": entry,
        "tp": tp,
        "sl": sl,
        "prob": prob,
        "rr": rr,
        "risk_ok": True,
        "payload_preview": res.get("payload_preview", {}),
        "result": {
            "entry": entry, "tp": tp, "sl": sl
        }
    }

    # 의사결정 요약 로깅 (성사 조건)
    log_event("signal.decision",
              symbol=symbol, direction=direction, prob=prob,
              entry=entry, tp=tp, sl=sl, rr=rr, risk_ok=True)

    return out


def manage_trade(symbol: str) -> Dict[str, Any]:
    """
    주문 실행(옵션). ENTRY_MODE=MARKET|LIMIT 지원.
    - 포지션 모드/마진 타입/레버리지 세팅
    - 수량 보정(거래 필터) → 진입 → reduce-only TP/SL 브래킷
    """
    try:
        sig = generate_signal(symbol)
        if "result" not in sig or not isinstance(sig["result"], dict):
            return {"symbol": symbol, "error": "no_signal"}

        res = sig["result"]
        direction = res.get("direction", "hold")
        entry = float(res.get("entry", 0.0))
        tp = float(res.get("tp", 0.0))
        sl = float(res.get("sl", 0.0))
        prob = float(res.get("prob", 0.5))
        rr = float(res.get("rr", 0.0))
        risk_ok = bool(res.get("risk_ok", False))
        reason = res.get("reasoning", "")

        # 거래 조건 미충족 → 요약만 반환
        if direction not in ("long", "short") or not risk_ok or entry <= 0 or tp <= 0 or sl <= 0:
            preview = res.get("payload_preview", {})
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

        # 확률 게이트: 낮은 확률이면 중단
        if float(res.get("prob", 0.0)) < float(os.getenv("MIN_PROB", "0.60")):
            preview = res.get("payload_preview", {})
            return {
                "symbol": symbol,
                "action": "hold",
                "direction": direction,
                "entry": entry,
                "tp": tp,
                "sl": sl,
                "prob": float(res.get("prob", 0.0)),
                "risk_ok": False,
                "rr": float(res.get("rr", 0.0)),
                "reason": "prob_below_threshold",
                "payload_preview": preview,
            }

        # 계정 설정


        try:
            set_position_mode(POSITION_MODE)
        except Exception as e:
            logger.warning("set_position_mode: %s", e)
        try:
            set_margin_type(symbol, MARGIN_TYPE)
        except Exception as e:
            logger.warning("set_margin_type: %s", e)
        try:
            set_leverage(symbol, DEFAULT_LEVERAGE)
        except Exception as e:
            logger.warning("set_leverage: %s", e)

        # 위험금액 기반 수량
        risk_scalar = float(res.get("risk_scalar", 1.0))
        qty = _size_from_balance(symbol, entry, sl, risk_scalar)
        side = "BUY" if direction == "long" else "SELL"

        # 진입 & 브래킷
        exec_res: Dict[str, Any]
        if ENTRY_MODE == "MARKET":
            entry_res = place_market_order(symbol, side, quantity=qty)
            bracket_res = place_bracket_orders(symbol, side, quantity=qty, take_profit=tp, stop_loss=sl)
            exec_res = {"entry_order": entry_res, "brackets": bracket_res}
            mode = "MARKET"
        else:
            exec_res = _enter_limit_then_brackets(symbol, side, qty, desired_entry=entry, tp=tp, sl=sl)
            mode = "LIMIT"

        log_event("signal.decision", symbol=symbol, direction=("long" if side=="BUY" else "short"),
          prob=float(res.get("prob", 0.0)), entry=float(entry), tp=float(tp), sl=float(sl),
          rr=float(res.get("rr", 0.0)), risk_ok=True)
        
        # 저널 기록(선택)
        try:
            row = {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "symbol": symbol,
                "side": "long" if side == "BUY" else "short",
                "qty": float(exec_res.get("filled_qty", qty) if mode == "LIMIT" else qty),
                "entry": float(entry),
                "tp": float(tp),
                "sl": float(sl),
                "exit": 0.0,
                "pnl": 0.0,
                "status": "open",
                "id": "",
            }
            headers = ["timestamp", "symbol", "side", "qty", "entry", "tp", "sl", "exit", "pnl", "status", "id"]
            # 로컬 CSV append
            try:
                os.makedirs(os.path.dirname(TRADES_CSV), exist_ok=True)
                new_file = not os.path.exists(TRADES_CSV)
                import csv
                with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
                    w = csv.DictWriter(f, fieldnames=headers)
                    if new_file:
                        w.writeheader()
                    w.writerow(row)
            except Exception as e:
                logger.info("local trades.csv append failed: %s", e)
            # GCS 스냅샷(옵션)
            if gcs_enabled():
                gcs_append_csv_row("trades", headers, row)
        except Exception as e:
            logger.info("journal append failed: %s", e)

        preview = res.get("payload_preview", {})
        return {
            "symbol": symbol,
            "action": "enter",
            "direction": direction,
            "entry": entry,
            "tp": tp,
            "sl": sl,
            "prob": prob,
            "risk_ok": True,
            "rr": rr,
            "payload_preview": preview,
            "order": exec_res,
            "mode": mode,
        }

    except Exception as e:
        logger.exception("manage_trade failed for %s", symbol)
        return {"symbol": symbol, "error": str(e)}


__all__ = [
    "get_overview",
    "generate_signal",
    "manage_trade",
]
