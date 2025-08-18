# helpers/signals.py
from __future__ import annotations
"""
helpers/signals.py — profitability-focused refactor (2025-08-18, KST)

- LLM pre-gating (volatility) + spread gate
- Payload: funding_rate_pct + microstructure features
- Limit price 'nudge' & >=1 tick diff on repricing
- TTL expiry fallback to MARKET (optional)
- Volatility-weighted sizing (risk_scalar)
- Prob calibration hook
- Default horizon 30m
"""

from typing import Any, Dict, List, Optional, Tuple
import json, logging, math, os
from dataclasses import dataclass
from datetime import datetime, timezone
import numpy as np
import pandas as pd

try:
    from .utils import LOG_DIR, gcs_enabled, gcs_append_csv_row, log_event  # type: ignore
except Exception:  # pragma: no cover
    LOG_DIR = os.path.join(os.getcwd(), "logs")
    def gcs_enabled() -> bool: return False
    def gcs_append_csv_row(*args, **kwargs) -> None: return None
    def log_event(*args, **kwargs): pass

try:
    from .data_fetch import (
        fetch_data, fetch_mtf_raw, add_indicators, compute_atr,
        compute_orderbook_stats, compute_recent_price_sequence,
        compute_support_resistance, compute_relative_volume,
        compute_trend_filter, fetch_orderbook,
    )  # type: ignore
except Exception as e:  # pragma: no cover
    raise

try:
    from .predictor import get_gemini_prediction, should_predict  # type: ignore
except Exception as e:  # pragma: no cover
    raise

try:
    from .binance_client import (
        get_overview as _bn_get_overview,
        cancel_open_orders, set_position_mode, set_margin_type, set_leverage,
        load_symbol_filters, ensure_min_notional, normalize_price_for_side, normalize_price_with_mode,
        place_market_order, place_limit_order, place_bracket_orders,
        get_position, get_open_orders, get_last_price, replace_stop_loss_to_price,
    )  # type: ignore
except Exception as e:  # pragma: no cover
    raise

# --- optional prob calibrator ---
try:
    from .calibration import calibrate_prob  # type: ignore
except Exception:
    def calibrate_prob(p: float) -> float: return float(p)

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

# === horizon: shorten default to 30 for 5m strategy ===
HORIZON_MIN = int(os.getenv("HORIZON_MIN", "30"))

TIME_BARRIER_ENABLED = str(os.getenv("TIME_BARRIER_ENABLED", "true")).lower() in ("1","true","yes")
ENTRY_MODE = os.getenv("ENTRY_MODE", "LIMIT").upper()  # MARKET | LIMIT
LIMIT_TTL_SEC = float(os.getenv("LIMIT_TTL_SEC", "15"))
LIMIT_POLL_SEC = float(os.getenv("LIMIT_POLL_SEC", "1.0"))
LIMIT_MAX_REPRICES = int(os.getenv("LIMIT_MAX_REPRICES", "3"))
LIMIT_MAX_SLIPPAGE_BPS = float(os.getenv("LIMIT_MAX_SLIPPAGE_BPS", "2.0"))
LIMIT_TTL_FALLBACK_TO_MARKET = str(os.getenv("LIMIT_TTL_FALLBACK_TO_MARKET", "true")).lower() in ("1","true","yes")

# relax rule: only these two keys
PROB_RELAX_THRESHOLD = float(os.getenv("PROB_RELAX_THRESHOLD", "0.75"))
RR_MIN_HIGH_PROB = float(os.getenv("RR_MIN_HIGH_PROB", "1.05"))

TP_ORDER_TYPE = os.getenv("TP_ORDER_TYPE", "LIMIT").upper()

# --- BE trailing ---
BE_TRAILING_ENABLED = str(os.getenv("BE_TRAILING_ENABLED", "true")).lower() in ("1","true","yes")
BE_TRIGGER_R_MULT = float(os.getenv("BE_TRIGGER_R_MULT", "1.0"))
BE_OFFSET_TICKS = int(os.getenv("BE_OFFSET_TICKS", "1"))

# --- volatility-weighted sizing ---
VOL_SIZE_SCALING = str(os.getenv("VOL_SIZE_SCALING", "true")).lower() in ("1","true","yes")
VOL_LOOKBACK = int(os.getenv("VOL_LOOKBACK", "60"))         # bars for median ATR
VOL_SCALAR_MIN = float(os.getenv("VOL_SCALAR_MIN", "0.50")) # clamp lower
VOL_SCALAR_MAX = float(os.getenv("VOL_SCALAR_MAX", "1.25")) # clamp upper

# --- prob calibration toggle ---
USE_CALIBRATED_PROB = str(os.getenv("USE_CALIBRATED_PROB", "true")).lower() in ("1","true","yes")

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
            "symbol": self.symbol, "direction": self.direction, "prob": self.prob,
            "entry": self.entry, "tp": self.tp, "sl": self.sl,
            "risk_scalar": self.risk_scalar, "reasoning": self.reasoning,
        }

# ==============
# 유틸리티
# ==============
def _df_ok(df: Optional[pd.DataFrame]) -> bool:
    return isinstance(df, pd.DataFrame) and not df.empty and all(c in df.columns for c in ("timestamp","open","high","low","close","volume"))

def _ob_stats_to_dict(stats: Any) -> Dict[str, float]:
    try:
        return {
            "imbalance": float(stats.get("imbalance", 0.0)),
            "spread": float(stats.get("spread", 0.0)),
            "microprice": float(stats.get("microprice", 0.0)),
            "mid": float(stats.get("mid", 0.0)),
            "micro_dislocation_bps": float(stats.get("micro_dislocation_bps", 0.0)),
        }
    except Exception:
        return {"imbalance": 0.0, "spread": 0.0, "microprice": 0.0, "mid": 0.0, "micro_dislocation_bps": 0.0}

def _mid_from_orderbook(ob: Optional[Dict[str, Any]]) -> Optional[float]:
    if not isinstance(ob, dict): return None
    bids = ob.get("bids") or []; asks = ob.get("asks") or []
    if not bids or not asks: return None
    try:
        bb = float(bids[0][0]); ba = float(asks[0][0])
        if bb <= 0 or ba <= 0: return None
        return (bb + ba) / 2.0
    except Exception:
        return None

def _best_quotes(symbol: str) -> Tuple[Optional[float], Optional[float]]:
    try:
        ob = fetch_orderbook(symbol, limit=5)
        bids = ob.get("bids") if isinstance(ob, dict) else None
        asks = ob.get("asks") if isinstance(ob, dict) else None
        if bids and asks:
            return float(bids[0][0]), float(asks[0][0])
    except Exception:
        pass
    return None, None

def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)

def _parse_iso(ts: str) -> Optional[datetime]:
    try: return datetime.fromisoformat(str(ts))
    except Exception:
        try: return datetime.fromisoformat(str(ts).replace("Z","+00:00"))
        except Exception: return None

def _last_open_trade_timestamp(symbol: str) -> Optional[datetime]:
    try:
        import csv
        if not os.path.exists(TRADES_CSV): return None
        last_ts: Optional[datetime] = None
        with open(TRADES_CSV, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if str(row.get("symbol","")).upper() != symbol.upper(): continue
                if str(row.get("status","")).lower() != "open": continue
                dt = _parse_iso(row.get("timestamp",""))
                if dt and (last_ts is None or dt > last_ts): last_ts = dt
        return last_ts
    except Exception:
        return None

def _time_barrier_due(symbol: str) -> bool:
    if not TIME_BARRIER_ENABLED: return False
    start = _last_open_trade_timestamp(symbol)
    if not start: return False
    try:
        import pandas as _pd
        return _now_utc() >= (start + _pd.to_timedelta(HORIZON_MIN, unit="m"))
    except Exception:
        from datetime import timedelta
        return _now_utc() >= (start + timedelta(minutes=HORIZON_MIN))

def _close_position_market(symbol: str) -> Optional[dict]:
    try:
        p = get_position(symbol) or {}
        amt = float(p.get("positionAmt") or p.get("positionAmount") or 0.0)
        if abs(amt) <= 1e-12: return None
        side = "SELL" if amt > 0 else "BUY"
        res = place_market_order(symbol, side, quantity=abs(amt), reduce_only=True)
        try: cancel_open_orders(symbol)
        except Exception: pass
        log_event("time_exit", symbol=symbol, positionAmt=amt, side=side)
        return res if isinstance(res, dict) else {"raw": res}
    except Exception as e:
        logger.info("time_exit close failed for %s: %s", symbol, e)
        return None

def _has_reduce_only_or_bracket_orders(symbol: str) -> bool:
    try: orders = get_open_orders(symbol)
    except Exception: orders = []
    for o in orders or []:
        t = str((o.get("type") if isinstance(o, dict) else getattr(o, "type","")) or "").upper()
        ro = ((o.get("reduceOnly") if isinstance(o, dict) else getattr(o, "reduceOnly", None)) or
              (o.get("reduce_only") if isinstance(o, dict) else getattr(o, "reduce_only", None)))
        if t in ("TAKE_PROFIT","TAKE_PROFIT_MARKET","STOP","STOP_MARKET"): return True
        if isinstance(ro, bool) and ro: return True
    return False

def _size_from_balance(symbol: str, entry: float, sl: float, risk_scalar: float = 1.0) -> float:
    entry = float(entry or 0.0)
    if entry <= 0: return 0.0
    notional = max(1.0, float(RISK_USDT) * float(risk_scalar))
    qty = notional / entry
    try:
        f = load_symbol_filters(symbol)
        qty = ensure_min_notional(symbol, qty, price=entry, filters=f)
    except Exception:
        pass
    return float(qty)

def get_overview():
    return _bn_get_overview()

# ---------------------------------
# Payload & proposal
# ---------------------------------
def _build_payload(symbol: str) -> Tuple[Dict[str, Any], pd.DataFrame, Optional[Dict[str, Any]]]:
    base = fetch_data(symbol, interval="5m", ohlcv_limit=200, orderbook_limit=50, include_orderbook=True)
    ohlcv = base.get("ohlcv") if isinstance(base.get("ohlcv"), pd.DataFrame) else pd.DataFrame()
    if not _df_ok(ohlcv): ohlcv = add_indicators(None)
    last = ohlcv.iloc[-1] if _df_ok(ohlcv) else pd.Series({})
    ob = base.get("orderbook") if isinstance(base, dict) else None

    mtf = fetch_mtf_raw(symbol)
    extra: Dict[str, Any] = {}
    for tf, df in mtf.items():
        ind = add_indicators(df)
        atr = float(compute_atr(df, window=14).iloc[-1]) if _df_ok(df) else 0.0
        key = {"1h":"_1h","4h":"_4h","1d":"_1d"}.get(tf, "")
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

    price_seq = compute_recent_price_sequence(ohlcv, n=10) if _df_ok(ohlcv) else [0.0]*10

    atr5_series = compute_atr(ohlcv, window=14) if _df_ok(ohlcv) else None
    atr5 = float(atr5_series.iloc[-1]) if getattr(atr5_series, "size", 0) else 0.0
    sr5 = {"recent_high": float(ohlcv["high"].tail(50).max()) if _df_ok(ohlcv) else 0.0,
           "recent_low": float(ohlcv["low"].tail(50).min()) if _df_ok(ohlcv) else 0.0}
    raw_stats = compute_orderbook_stats(ob) if isinstance(ob, dict) else {"imbalance":0.0,"spread":0.0,"microprice":0.0,"mid":0.0,"micro_dislocation_bps":0.0}
    ob_stats = _ob_stats_to_dict(raw_stats)
    trend = compute_trend_filter(ohlcv) if _df_ok(ohlcv) else {"daily_uptrend": False, "trend_strength": 0.0}

    # funding(%): DF 마지막값
    try:
        funding_pct = float(ohlcv["funding_rate_pct"].iloc[-1]) if _df_ok(ohlcv) and "funding_rate_pct" in ohlcv.columns else 0.0
    except Exception:
        funding_pct = 0.0

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
            "ATR_5m": float(atr5),
            "relative_volume_5m": float(compute_relative_volume(ohlcv)) if _df_ok(ohlcv) else 1.0,
            "recent_high_5m": sr5["recent_high"],
            "recent_low_5m": sr5["recent_low"],
            "orderbook_imbalance": float(ob_stats.get("imbalance", 0.0)),
            "orderbook_spread": float(ob_stats.get("spread", 0.0)),
            "microprice": float(ob_stats.get("microprice", 0.0)),
            "micro_dislocation_bps": float(ob_stats.get("micro_dislocation_bps", 0.0)),
            "funding_rate_pct": float(funding_pct),
            **extra,
        },
        "times": base.get("times", {}),
        "price_sequence": price_seq,
        "sr_levels": sr5,
        "relative_volume": float(compute_relative_volume(ohlcv)) if _df_ok(ohlcv) else 1.0,
        "trend_filter": trend,
        "horizon_min": HORIZON_MIN,
    }
    return payload, ohlcv, ob

def _propose_levels(direction: str, entry: float, atr5: float, sr: Dict[str, float]) -> Tuple[float, float]:
    if entry <= 0: return 0.0, 0.0
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

def _spread_ok(spread_bps: float) -> bool:
    try: return float(spread_bps) <= float(MAX_SPREAD_BPS)
    except Exception: return False

def _risk_ok(entry: float, tp: float, sl: float) -> Tuple[bool, float]:
    try:
        entry = float(entry); tp = float(tp); sl = float(sl)
        if entry <= 0 or tp <= 0 or sl <= 0: return False, 0.0
        rr_up = (tp - entry); rr_dn = (entry - sl)
        if rr_dn <= 0: return False, 0.0
        rr = float(rr_up / rr_dn)
        return rr >= RR_MIN, rr
    except Exception:
        return False, 0.0

def _rule_backup(ohlcv: pd.DataFrame, trend: Dict[str, Any]) -> Tuple[str, float]:
    if not _df_ok(ohlcv): return "hold", 0.5
    close = float(ohlcv["close"].iloc[-1])
    sma20 = float(ohlcv["SMA_20"].iloc[-1]) if "SMA_20" in ohlcv else close
    rsi = float(ohlcv["RSI"].iloc[-1]) if "RSI" in ohlcv else 50.0
    up = bool(trend.get("daily_uptrend", False))
    if close > sma20 and rsi > 55 and up: return "long", 0.61
    if close < sma20 and rsi < 45 and (not up): return "short", 0.61
    return "hold", 0.5

# =====================
# 공개 API
# =====================
def generate_signal(symbol: str) -> Dict[str, Any]:
    payload, ohlcv, ob = _build_payload(symbol)

    # ---- PRE-GATING: 변동성 낮거나 스프레드 과대면 모델 호출 스킵 ----
    spread_bps_gate = float((payload.get("extra") or {}).get("orderbook_spread", 0.0))
    proceed = should_predict(payload, min_vol_frac_env="MIN_VOL_FRAC") and _spread_ok(spread_bps_gate)
    if not proceed:
        trend = payload.get("trend_filter") or {}
        direction_rb, prob_rb = _rule_backup(ohlcv, trend)
        if direction_rb in ("long","short"):
            direction, prob = direction_rb, prob_rb
        else:
            return {
                "symbol": symbol, "action": "hold", "direction": "hold",
                "entry": float((payload.get("entry_5m") or {}).get("close") or 0.0),
                "tp": 0.0, "sl": 0.0, "prob": 0.5, "risk_ok": False, "rr": 0.0,
                "reason": "pre_gate_block",
            }
    else:
        res = get_gemini_prediction(payload, symbol=symbol)
        direction = str(res.get("direction") or "").lower()
        prob = float(res.get("prob", 0.0))
        # --- 확률 보정(옵션) ---
        if USE_CALIBRATED_PROB:
            prob = float(calibrate_prob(prob))

    support = float(0.0); resistance = float(0.0)
    entry = float((payload.get("entry_5m") or {}).get("close") or 0.0)
    atr5 = float((payload.get("extra") or {}).get("ATR_5m") or 0.0)
    rh5 = float((payload.get("extra") or {}).get("recent_high_5m") or 0.0)
    rl5 = float((payload.get("extra") or {}).get("recent_low_5m") or 0.0)
    spread_bps = float((payload.get("extra") or {}).get("orderbook_spread") or 0.0)

    if direction not in ("long","short") or entry <= 0:
        out = {
            "symbol": symbol, "action": "hold",
            "direction": "hold", "entry": entry, "tp": 0.0, "sl": 0.0,
            "prob": prob, "risk_ok": False, "rr": 0.0, "reason": "invalid_direction_or_entry",
        }
        log_event("signal.decision", symbol=symbol, direction="hold", prob=prob,
                  entry=entry, tp=0.0, sl=0.0, rr=0.0, risk_ok=False)
        return out

    # TP/SL 제안
    k_tp = float(os.getenv("ATR_MULT_TP", str(ATR_MULT_TP)))
    k_sl = float(os.getenv("ATR_MULT_SL", str(ATR_MULT_SL)))
    if direction == "long":
        tp_candidates = [x for x in [resistance, rh5, entry + k_tp * atr5] if x and x > entry]
        sl_candidates = [x for x in [support, rl5, entry - k_sl * atr5] if x and x < entry]
        tp = max(tp_candidates) if tp_candidates else entry + max(1e-8, k_tp * atr5)
        sl = min(sl_candidates) if sl_candidates else entry - max(1e-8, k_sl * atr5)
        rr = (tp - entry) / max(1e-8, entry - sl)
    else:
        tp_candidates = [x for x in [support, rl5, entry - k_tp * atr5] if x and x < entry]
        sl_candidates = [x for x in [resistance, rh5, entry + k_sl * atr5] if x and x > entry]
        tp = min(tp_candidates) if tp_candidates else entry - max(1e-8, k_tp * atr5)
        sl = max(sl_candidates) if sl_candidates else entry + max(1e-8, k_sl * atr5)
        rr = (entry - tp) / max(1e-8, sl - entry)

    # RR 요구치(확률 높으면 완화)
    rr_req = RR_MIN_HIGH_PROB if prob >= PROB_RELAX_THRESHOLD else RR_MIN
    reasons = []
    if prob < MIN_PROB: reasons.append("prob_below_threshold")
    if rr <= 0 or rr < rr_req: reasons.append(f"rr_below_min({rr:.2f}<{rr_req:.2f})")
    if not _spread_ok(spread_bps): reasons.append(f"wide_spread({spread_bps:.2f}bps)")
    risk_ok = (len(reasons) == 0)

    # --- 변동성 가중 사이징 ---
    risk_scalar = 1.0
    try:
        if VOL_SIZE_SCALING and _df_ok(ohlcv):
            atr_series = compute_atr(ohlcv, window=14)
            cur = float(atr_series.iloc[-1]) if len(atr_series) else 0.0
            med = float(atr_series.tail(VOL_LOOKBACK).median()) if len(atr_series) else 0.0
            if cur > 0 and med > 0:
                # 변동성 높을수록 사이즈 ↓ : median/cur
                risk_scalar = max(VOL_SCALAR_MIN, min(VOL_SCALAR_MAX, med / cur))
    except Exception:
        risk_scalar = 1.0

    out = {
        "symbol": symbol,
        "action": "enter" if risk_ok else "hold",
        "direction": direction,
        "entry": float(entry),
        "tp": float(tp),
        "sl": float(sl),
        "prob": float(prob),
        "rr": float(rr),
        "risk_ok": bool(risk_ok),
        "reason": "ok" if risk_ok else ";".join(reasons) or "no_trade_conditions",
        "result": {
            "direction": direction, "entry": float(entry), "tp": float(tp), "sl": float(sl),
            "prob": float(prob), "rr": float(rr), "risk_ok": bool(risk_ok),
            "risk_scalar": float(risk_scalar),
        },
    }
    log_event("signal.decision", symbol=symbol, direction=direction, prob=prob,
              entry=entry, tp=tp, sl=sl, rr=rr, risk_ok=risk_ok)
    return out

# --- Limit then Brackets with TTL/reprice ---
def _position_qty_after_fill(symbol: str, side: str) -> float:
    p = get_position(symbol) or {}
    amt = float(p.get("positionAmt") or p.get("positionAmount") or 0.0)
    if side.upper() == "BUY" and amt > 0: return abs(amt)
    if side.upper() == "SELL" and amt < 0: return abs(amt)
    return 0.0

def _limit_price_for_side(symbol: str, side: str, desired_entry: float, prev_submitted: Optional[float]) -> Optional[float]:
    """
    - 슬리피지 예산 내에서 best bid/ask와 desired_entry 사이 캡
    - 사이드별 틱 라운딩(normalize_price_for_side)
    - 이전 제출가와 최소 1 tick 이상 차이가 나도록 '넛지'
    """
    bid, ask = _best_quotes(symbol)
    if bid is None or ask is None: return None
    budget = LIMIT_MAX_SLIPPAGE_BPS / 1e4
    # base cap
    if side.upper() == "BUY":
        raw = min(ask, max(bid, desired_entry))
        capped = min(raw, desired_entry * (1.0 + budget))
        px = normalize_price_for_side(symbol, capped, side="BUY")
    else:
        raw = max(bid, min(ask, desired_entry))
        capped = max(raw, desired_entry * (1.0 - budget))
        px = normalize_price_for_side(symbol, capped, side="SELL")

    # tick nudge vs previous price
    try:
        f = load_symbol_filters(symbol)
        tick = float(f["tickSize"])
    except Exception:
        tick = 0.0
    if prev_submitted is not None and tick > 0:
        # 보수적으로 사이드 방향으로 1틱 이동
        if side.upper() == "BUY":
            if abs(px - prev_submitted) < tick:
                px = prev_submitted + tick
        else:
            if abs(px - prev_submitted) < tick:
                px = prev_submitted - tick
        # 최종 라운딩
        px = normalize_price_for_side(symbol, px, side=side)
    return float(px)

def _enter_limit_then_brackets(symbol: str, side: str, qty: float, desired_entry: float, tp: float, sl: float) -> Dict[str, Any]:
    side = side.upper()
    last_submitted: Optional[float] = None

    # 1st attempt
    price = _limit_price_for_side(symbol, side, desired_entry, prev_submitted=None)
    if price is None or price <= 0:
        raise RuntimeError("Could not determine limit price")
    last_submitted = price

    try:
        entry_res = place_limit_order(symbol, side, quantity=qty, price=price, time_in_force="GTC", reduce_only=False)
    except Exception:
        entry_res = {"type": "LIMIT_FAILOVER_MARKET"}

    filled = 0.0
    reprices = 0
    from time import sleep, time as _t
    deadline = _t() + float(LIMIT_TTL_SEC)
    while _t() < deadline:
        q = _position_qty_after_fill(symbol, side)
        if q >= (qty * 0.95):
            filled = q
            break
        sleep(max(0.1, float(LIMIT_POLL_SEC)))

    # reprices with tick-nudge
    while filled <= 0 and reprices < LIMIT_MAX_REPRICES:
        reprices += 1
        try: cancel_open_orders(symbol)
        except Exception: pass
        price = _limit_price_for_side(symbol, side, desired_entry, prev_submitted=last_submitted)
        if price is None or price <= 0: break
        last_submitted = price
        try:
            entry_res = place_limit_order(symbol, side, quantity=qty, price=price, time_in_force="GTC", reduce_only=False)
        except Exception:
            entry_res = {"type": "LIMIT_FAILOVER_MARKET"}
        deadline = _t() + float(LIMIT_TTL_SEC)
        while _t() < deadline:
            q = _position_qty_after_fill(symbol, side)
            if q >= (qty * 0.95):
                filled = q
                break
            sleep(max(0.1, float(LIMIT_POLL_SEC)))
        if filled > 0: break

    # TTL & reprices done → MARKET fallback (옵션)
    if filled <= 0 and LIMIT_TTL_FALLBACK_TO_MARKET:
        try:
            entry_res = place_market_order(symbol, side, quantity=qty, reduce_only=False)
            filled = _position_qty_after_fill(symbol, side)
        except Exception as e:
            logger.info("market fallback after TTL failed: %s", e)

    brackets = {"take_profit": None, "stop_loss": None}
    if filled > 0:
        try:
            brackets = place_bracket_orders(symbol, side, filled, take_profit=tp, stop_loss=sl)
        except Exception as e:
            logger.info("placing brackets failed: %s", e)

    return {"entry_order": entry_res, "brackets": brackets, "filled_qty": float(filled), "reprices": reprices}

# ---------------------------------
# Maintain (time barrier + BE trailing + cleanup)
# ---------------------------------
def _current_stop_price(symbol: str) -> Optional[float]:
    try: orders = get_open_orders(symbol)
    except Exception: orders = []
    sl_prices: List[float] = []
    for o in orders or []:
        t = str((o.get("type") if isinstance(o, dict) else getattr(o, "type","")) or "").upper()
        if t in ("STOP","STOP_MARKET"):
            sp = (o.get("stopPrice") if isinstance(o, dict) else getattr(o, "stopPrice", None))
            try:
                if sp is not None: sl_prices.append(float(sp))
            except Exception:
                pass
    if not sl_prices: return None
    sl_prices.sort()
    mid = sl_prices[len(sl_prices)//2]
    return float(mid)

def maintain_positions(symbol: str) -> Dict[str, Any]:
    try:
        if _time_barrier_due(symbol):
            od = _close_position_market(symbol)
            return {"action": "time_exit", "order": od or {}}

        if BE_TRAILING_ENABLED:
            p = get_position(symbol) or {}
            amt = float(p.get("positionAmt") or p.get("positionAmount") or 0.0)
            if abs(amt) > 1e-12:
                is_long = amt > 0
                entry = float(p.get("entryPrice") or 0.0)
                last = get_last_price(symbol) or 0.0
                cur_sl = _current_stop_price(symbol)
                if entry > 0 and last > 0 and cur_sl:
                    R = (entry - cur_sl) if is_long else (cur_sl - entry)
                    if R > 0:
                        trigger = entry + BE_TRIGGER_R_MULT * R if is_long else entry - BE_TRIGGER_R_MULT * R
                        hit = (last >= trigger) if is_long else (last <= trigger)
                        already_be = (cur_sl >= entry) if is_long else (cur_sl <= entry)
                        if hit and not already_be:
                            try:
                                f = load_symbol_filters(symbol); tick = float(f["tickSize"])
                            except Exception: tick = 0.0
                            new_sl = (entry + BE_OFFSET_TICKS * tick) if is_long else (entry - BE_OFFSET_TICKS * tick)
                            try:
                                replaced = replace_stop_loss_to_price(symbol, is_long=is_long, quantity=abs(amt), new_stop_price=new_sl)
                                log_event("be_trail.move_to_be", symbol=symbol, side=("long" if is_long else "short"),
                                          entry=entry, old_sl=cur_sl, new_sl=new_sl, last=last, R=R)
                                return {"action": "be_move", "new_stop": new_sl, "details": replaced}
                            except Exception as e:
                                logger.info("be_trailing replace SL failed for %s: %s", symbol, e)

        p = get_position(symbol) or {}
        amt = float(p.get("positionAmt") or p.get("positionAmount") or 0.0)
        if abs(amt) <= 1e-12 and _has_reduce_only_or_bracket_orders(symbol):
            try:
                cancel_open_orders(symbol)
                log_event("cleanup.brackets_cancelled", symbol=symbol)
                return {"action": "cleanup"}
            except Exception as e:
                logger.info("cleanup cancel_open_orders failed for %s: %s", symbol, e)
        return {"action": "none"}
    except Exception as e:
        logger.info("maintain_positions error for %s: %s", symbol, e)
        return {"action": "error", "error": str(e)}

def manage_trade(symbol: str) -> Dict[str, Any]:
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
        risk_scalar = float(res.get("risk_scalar", 1.0))

        if direction not in ("long","short") or not risk_ok or entry <= 0 or tp <= 0 or sl <= 0:
            return {
                "symbol": symbol, "action": "hold", "direction": direction,
                "entry": entry, "tp": tp, "sl": sl, "prob": prob,
                "risk_ok": False, "rr": rr, "reason": reason or "no_trade_conditions",
            }

        if float(res.get("prob", 0.0)) < float(os.getenv("MIN_PROB", "0.60")):
            return {
                "symbol": symbol, "action": "hold", "direction": direction,
                "entry": entry, "tp": tp, "sl": sl, "prob": float(res.get("prob", 0.0)),
                "risk_ok": False, "rr": float(res.get("rr", 0.0)), "reason": "prob_below_threshold",
            }

        # 계정 설정
        try: set_position_mode(POSITION_MODE)
        except Exception as e: logger.warning("set_position_mode: %s", e)
        try: set_margin_type(symbol, MARGIN_TYPE)
        except Exception as e: logger.warning("set_margin_type: %s", e)
        try: set_leverage(symbol, DEFAULT_LEVERAGE)
        except Exception as e: logger.warning("set_leverage: %s", e)

        qty = _size_from_balance(symbol, entry, sl, risk_scalar)
        side = "BUY" if direction == "long" else "SELL"

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

        try:
            row = {
                "timestamp": datetime.now(tz=timezone.utc).isoformat(),
                "symbol": symbol,
                "side": "long" if side == "BUY" else "short",
                "qty": float(exec_res.get("filled_qty", qty) if mode == "LIMIT" else qty),
                "entry": float(entry), "tp": float(tp), "sl": float(sl),
                "exit": 0.0, "pnl": 0.0, "status": "open", "id": "",
            }
            headers = ["timestamp","symbol","side","qty","entry","tp","sl","exit","pnl","status","id"]
            os.makedirs(os.path.dirname(TRADES_CSV), exist_ok=True)
            new_file = not os.path.exists(TRADES_CSV)
            import csv
            with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
                w = csv.DictWriter(f, fieldnames=headers)
                if new_file: w.writeheader()
                w.writerow(row)
            if gcs_enabled():
                gcs_append_csv_row("trades", headers, row)
        except Exception as e:
            logger.info("journal append failed: %s", e)

        return {
            "symbol": symbol, "action": "enter", "direction": direction,
            "entry": entry, "tp": tp, "sl": sl, "prob": prob, "risk_ok": True, "rr": rr,
            "order": exec_res, "mode": mode,
        }
    except Exception as e:
        logger.exception("manage_trade failed for %s", symbol)
        return {"symbol": symbol, "error": str(e)}
