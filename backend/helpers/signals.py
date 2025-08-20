from __future__ import annotations
"""
helpers/signals.py — profitability-focused refactor (2025-08-19, KST) — patched

- LLM pre-gating (volatility) + spread gate
- NEW: Early pre-gating **before** LLM call: cooldown + shock guard (direction-agnostic),
       and optional MTF alignment using a rule-based hint.
- Payload: funding_rate_pct + microstructure features
- Limit price 'nudge' & >=1 tick diff on repricing
- TTL expiry fallback to MARKET (optional)
- Volatility-weighted sizing (risk_scalar)
- Prob calibration hook
- Default horizon 30m
- NEW: Use LLM-provided support/resistance to build TP/SL (Step 1 fix)
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

# ==== [ENV 추가] 파일 상단 ENV 블록 근처에 다음을 추가 ====
ENTRY_POST_ONLY = str(os.getenv("ENTRY_POST_ONLY", "false")).lower() in ("1","true","yes")

# fee-aware RR gate
RR_EVAL_WITH_FEES = str(os.getenv("RR_EVAL_WITH_FEES", "true")).lower() in ("1","true","yes")
FEE_MAKER_BPS = float(os.getenv("FEE_MAKER_BPS", "2.0"))
FEE_TAKER_BPS = float(os.getenv("FEE_TAKER_BPS", "4.0"))

# shock guard & MTF alignment & cooldown
SHOCK_BPS = float(os.getenv("SHOCK_BPS", "30.0"))           # 0.30%
SHOCK_ATR_MULT = float(os.getenv("SHOCK_ATR_MULT", "1.5"))  # 1.5x ATR(14,5m)
MTF_ALIGN_ENABLED = str(os.getenv("MTF_ALIGN_ENABLED", "true")).lower() in ("1","true","yes")
MTF_RSI_LONG_MIN = float(os.getenv("MTF_RSI_LONG_MIN", "48"))
MTF_RSI_SHORT_MAX = float(os.getenv("MTF_RSI_SHORT_MAX", "52"))
ENTRY_COOLDOWN_MIN = int(os.getenv("ENTRY_COOLDOWN_MIN", "10"))

RR_GATE_MODE = os.getenv("RR_GATE_MODE", "worst").lower()  # worst | expected | best
MAKER_PROB_LOOKBACK = int(os.getenv("MAKER_PROB_LOOKBACK", "200"))

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
def _estimate_p_maker_from_journal() -> float:
    """
    최근 trades.csv와 limit 엔트리 메타(maintain/exec 로그)를 바탕으로
    메이커 체결률 근사치를 추정한다. 정보가 부족하면 0.5로.
    """
    try:
        path = os.path.join(LOG_DIR if isinstance(LOG_DIR, str) else str(LOG_DIR), "trades.csv")
        if not os.path.exists(path):
            return 0.5
        import csv
        rows = []
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                rows.append(row)
        if not rows:
            return 0.5
        rows = rows[-MAKER_PROB_LOOKBACK:]
        # 힌트: LIMIT 체결(폴백X) 사례를 메이커로 인정
        # exec_res에 대한 정밀 표시는 trades.csv엔 없으므로, 보수적 휴리스틱 사용:
        # - qty가 0이 아닌 LIMIT 모드 엔트리이며 TTL 이후 MARKET 폴백으로 기록된 특이 로그가 드문 경우 메이커 가정↑
        # - 데이터 부족 시 0.5
        # 더 나은 해법: manage_trade에서 maker/taker 플래그를 trades.csv에 직접 기록(아래 5‑D 패치 참조)
        cnt = len(rows)
        return 0.5 if cnt < 20 else 0.6  # 데이터 없으면 중립, 임시 상수
    except Exception:
        return 0.5

def _rr_with_fee_mode(entry: float, tp: float, sl: float) -> float:
    """
    RR_EVAL_WITH_FEES=true일 때 RR_GATE_MODE에 따라 순RR을 계산.
    - worst: 현재와 동일(엔트리=테이커 가정 가능)
    - best: 엔트리=메이커 가정
    - expected: p_maker로 기대 수수료 적용
    """
    if not RR_EVAL_WITH_FEES:
        # 기존 gross RR
        if entry <= 0 or tp <= 0 or sl <= 0: return 0.0
        up_gross = (tp - entry) / entry
        dn_gross = (entry - sl) / entry
        return up_gross / max(1e-12, dn_gross)

    # leg별 수수료(bps) 설정
    maker = FEE_MAKER_BPS / 1e4
    taker = FEE_TAKER_BPS / 1e4

    def _rr_net_local(e_is_maker: bool, tp_is_maker: bool, sl_is_taker: bool) -> float:
        e = entry; t = tp; s = sl
        if e <= 0 or t <= 0 or s <= 0: return 0.0
        up_gross = (t - e) / e
        dn_gross = (e - s) / e
        fee_e  = maker if e_is_maker else taker
        fee_tp = maker if tp_is_maker else taker
        fee_sl = taker if sl_is_taker else maker
        up_net = max(0.0, up_gross - (fee_e + fee_tp))
        dn_net = max(1e-12, dn_gross + (fee_e + fee_sl))
        return up_net / dn_net

    # 기존 worst-case 로직
    entry_is_maker_worst = (ENTRY_MODE == "LIMIT" and ENTRY_POST_ONLY and (not LIMIT_TTL_FALLBACK_TO_MARKET))
    rr_worst = _rr_net_local(entry_is_maker_worst, TP_ORDER_TYPE == "LIMIT", True)
    if RR_GATE_MODE == "worst":
        return rr_worst
    if RR_GATE_MODE == "best":
        return _rr_net_local(True, TP_ORDER_TYPE == "LIMIT", True)
    # expected
    p_maker = _estimate_p_maker_from_journal()
    rr_maker = _rr_net_local(True, TP_ORDER_TYPE == "LIMIT", True)
    rr_taker = _rr_net_local(False, TP_ORDER_TYPE == "LIMIT", True)
    return max(0.0, p_maker * rr_maker + (1.0 - p_maker) * rr_taker)
    
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
        return {"imbalance": 0.0, "spread": 0.0, "mid": 0.0, "microprice": 0.0, "micro_dislocation_bps": 0.0}

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

def _shock_guard_block(direction: str, ohlcv: pd.DataFrame, atr5: float) -> Tuple[bool, float, float, str]:
    if not _df_ok(ohlcv) or direction not in ("long","short"):
        return False, 0.0, 0.0, ""
    last = ohlcv.iloc[-1]
    close = float(last.get("close", 0.0))
    openp = float(last.get("open", 0.0))
    chg = close - openp
    chg_bps = abs(chg) / max(1e-8, close) * 1e4
    atr = float(atr5) if atr5 else 0.0
    atr_mult = abs(chg) / max(1e-8, atr)
    candle_up = (chg > 0)
    counter = (direction == "long" and not candle_up) or (direction == "short" and candle_up)
    block = counter and ((chg_bps >= SHOCK_BPS) or (atr_mult >= SHOCK_ATR_MULT))
    reason = f"shock_guard(countertrend,{chg_bps:.1f}bps,{atr_mult:.2f}ATR)"
    return block, float(chg_bps), float(atr_mult), reason

def _mtf_align_ok(direction: str, extra: Dict[str, Any]) -> Tuple[bool, str]:
    if not MTF_ALIGN_ENABLED or direction not in ("long","short"):
        return True, ""
    r1h = float(extra.get("RSI_1h", 50.0))
    r4h = float(extra.get("RSI_4h", 50.0))
    if direction == "long":
        ok = (r1h >= MTF_RSI_LONG_MIN) and (r4h >= MTF_RSI_LONG_MIN)
    else:
        ok = (r1h <= MTF_RSI_SHORT_MAX) and (r4h <= MTF_RSI_SHORT_MAX)
    return ok, "mtf_rsi_mismatch"

def _cooldown_active(symbol: str) -> Tuple[bool, int]:
    mins = int(max(0, ENTRY_COOLDOWN_MIN))
    if mins <= 0:
        return False, 0
    last_open = _last_open_trade_timestamp(symbol)
    if not last_open:
        return False, 0
    from datetime import timedelta
    left = mins - int((_now_utc() - last_open).total_seconds() // 60)
    return (left > 0), max(0, left)

def _rr_net(entry: float, tp: float, sl: float,
            entry_is_maker: bool, tp_is_maker: bool, sl_is_taker: bool,
            maker_bps: float = FEE_MAKER_BPS, taker_bps: float = FEE_TAKER_BPS) -> float:
    """
    순RR = (상방수익률 - (엔트리+TP)수수료) / (하방손실률 + (엔트리+SL)수수료)
    수수료는 bps→비율로 변환하여 엔트리 가격 기준의 수익률 스케일에 합산.
    """
    e = float(entry)
    if e <= 0 or tp <= 0 or sl <= 0:
        return 0.0
    up_gross = (tp - e) / e
    dn_gross = (e - sl) / e
    fee_entry = maker_bps if entry_is_maker else taker_bps
    fee_tp    = maker_bps if tp_is_maker    else taker_bps
    fee_sl    = taker_bps if sl_is_taker    else maker_bps  # SL은 보통 테이커
    up_net = max(0.0, up_gross - (fee_entry + fee_tp) / 1e4)
    dn_net = max(1e-12, dn_gross + (fee_entry + fee_sl) / 1e4)
    return float(up_net / dn_net)

def _tp_sl_with_sr_clamp(
    direction: str,
    entry: float,
    atr5: float,
    sr_high: float,
    sr_low: float,
    llm_support: Optional[float],
    llm_resistance: Optional[float],
    k_tp: float = ATR_MULT_TP,
    k_sl: float = ATR_MULT_SL,
) -> Tuple[float, float]:
    """
    롱: SL을 엔트리에 가장 가까운 '지지'로 클램프(너무 깊지 않게), TP는 보수적으로 상방 후보 중 최댓값.
    쇼트: SL을 엔트리에 가장 가까운 '저항'으로 클램프, TP는 하방 후보 중 최솟값.
    """
    e = float(entry or 0.0)
    a = float(max(atr5, 1e-12))
    hi = float(sr_high or 0.0)
    lo = float(sr_low or 0.0)
    sup = float(llm_support) if (llm_support is not None) else None
    res = float(llm_resistance) if (llm_resistance is not None) else None

    if direction == "long":
        base_tp = e + k_tp * a
        base_sl = e - k_sl * a
        # TP: 과도하게 낮추지 않음(상방 후보의 최대)
        tp_candidates = [x for x in [base_tp, hi, res] if (x and x > e)]
        tp = max(tp_candidates) if tp_candidates else base_tp
        # SL: 엔트리 아래 후보 중 '가장 가까운' 값으로 클램프(너무 깊지 않게)
        sl_candidates_below = [x for x in [base_sl, lo, sup] if (x and x < e)]
        sl = max(sl_candidates_below) if sl_candidates_below else base_sl
    else:
        base_tp = e - k_tp * a
        base_sl = e + k_sl * a
        # TP: 하방 후보의 최소
        tp_candidates = [x for x in [base_tp, lo, sup] if (x and x < e)]
        tp = min(tp_candidates) if tp_candidates else base_tp
        # SL: 엔트리 위 후보 중 '가장 가까운' 값
        sl_candidates_above = [x for x in [base_sl, hi, res] if (x and x > e)]
        sl = min(sl_candidates_above) if sl_candidates_above else base_sl

    return float(tp), float(sl)

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

    # ---- PRE-GATING #0: 변동성/스프레드 (LLM 호출 전) ----
    spread_bps_gate = float((payload.get("extra") or {}).get("orderbook_spread", 0.0))
    proceed_basic = should_predict(payload, min_vol_frac_env="MIN_VOL_FRAC") and _spread_ok(spread_bps_gate)

    # ---- PRE-GATING #1: 쿨다운 & 쇼크가드(방향 비종속) ----
    #   - 쿨다운: 최근 오픈 트레이드 후 대기시간 존재 → 즉시 hold
    #   - 쇼크가드: 최근 5m 변동이 기준 초과이면 보수적으로 LLM 호출 자체를 스킵
    dir_hint, _prob_hint = _rule_backup(ohlcv, payload.get("trend_filter") or {})
    atr5 = float((payload.get("extra") or {}).get("ATR_5m") or 0.0)

    cd_active, cd_left = _cooldown_active(symbol)
    if cd_active:
        return {
            "symbol": symbol, "action": "hold", "direction": "hold",
            "entry": float((payload.get("entry_5m") or {}).get("close") or 0.0),
            "tp": 0.0, "sl": 0.0, "prob": 0.5, "risk_ok": False, "rr": 0.0,
            "reason": f"pre_gate_cooldown({cd_left}m_left)",
        }

    # shock against either possible trade direction → block if either countertrend shock exists
    # 기존: if sg_long or sg_short: ...  → 교체
    sg_long, sg_bps_L, sg_mult_L, _ = _shock_guard_block("long", ohlcv, atr5)
    sg_short, sg_bps_S, sg_mult_S, _ = _shock_guard_block("short", ohlcv, atr5)

    if (sg_long or sg_short):
        # 규칙 힌트가 존재하고, 그 힌트 방향과 '쇼크 캔들 방향'이 일치하면 통과(모멘텀 허용)
        candle_up = float(ohlcv.iloc[-1]["close"]) - float(ohlcv.iloc[-1]["open"]) > 0 if _df_ok(ohlcv) else False
        shock_dir = "long" if candle_up else "short"
        if dir_hint in ("long","short") and dir_hint == shock_dir:
            pass  # allow
        else:
            bps = max(sg_bps_L, sg_bps_S)
            mult = max(sg_mult_L, sg_mult_S)
            return {
                "symbol": symbol, "action": "hold", "direction": "hold",
                "entry": float((payload.get("entry_5m") or {}).get("close") or 0.0),
                "tp": 0.0, "sl": 0.0, "prob": 0.5, "risk_ok": False, "rr": 0.0,
                "reason": f"pre_gate_shock({bps:.1f}bps,{mult:.2f}ATR)",
            }

    # Optional pre-check: if we already have a confident rule-based hint, ensure MTF alignment before LLM
    if MTF_ALIGN_ENABLED and dir_hint in ("long","short"):
        mtf_ok, _mtf_reason = _mtf_align_ok(dir_hint, payload.get("extra") or {})
        if not mtf_ok:
            return {
                "symbol": symbol, "action": "hold", "direction": "hold",
                "entry": float((payload.get("entry_5m") or {}).get("close") or 0.0),
                "tp": 0.0, "sl": 0.0, "prob": 0.5, "risk_ok": False, "rr": 0.0,
                "reason": "pre_gate_mtf_mismatch",
            }

    # If basic pre-gates fail, fall back to rule backup; else LLM
    if not proceed_basic:
        trend = payload.get("trend_filter") or {}
        direction_rb, prob_rb = _rule_backup(ohlcv, trend)  # 규칙 폴백
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

    entry = float((payload.get("entry_5m") or {}).get("close") or 0.0)
    extra = payload.get("extra") or {}
    rh5 = float(extra.get("recent_high_5m") or 0.0)
    rl5 = float(extra.get("recent_low_5m") or 0.0)
    spread_bps = float(extra.get("orderbook_spread") or 0.0)

    if direction not in ("long","short") or entry <= 0:
        out = {
            "symbol": symbol, "action": "hold",
            "direction": "hold", "entry": entry, "tp": 0.0, "sl": 0.0,
            "prob": prob, "risk_ok": False, "rr": 0.0, "reason": "invalid_direction_or_entry",
        }
        log_event("signal.decision", symbol=symbol, direction="hold", prob=prob,
                  entry=entry, tp=0.0, sl=0.0, rr=0.0, risk_ok=False)
        return out

    # ---- TP/SL 제안 (ATR/SR + LLM SR) ----
    k_tp = float(os.getenv("ATR_MULT_TP", str(ATR_MULT_TP)))
    k_sl = float(os.getenv("ATR_MULT_SL", str(ATR_MULT_SL)))
    atr5 = float((payload.get("extra") or {}).get("ATR_5m") or 0.0)

    # NEW: merge LLM SR with 5m SR
    support_m = None
    resistance_m = None
    try:
        res_locals = locals().get("res")
        if isinstance(res_locals, dict):
            if res_locals.get("support") is not None:
                support_m = float(res_locals.get("support") or 0.0)
            if res_locals.get("resistance") is not None:
                resistance_m = float(res_locals.get("resistance") or 0.0)
    except Exception:
        pass

    sr_high = float((payload.get("extra") or {}).get("recent_high_5m") or 0.0)
    sr_low  = float((payload.get("extra") or {}).get("recent_low_5m") or 0.0)

    tp, sl = _tp_sl_with_sr_clamp(
        direction=direction,
        entry=entry,
        atr5=atr5,
        sr_high=sr_high,
        sr_low=sr_low,
        llm_support=support_m,
        llm_resistance=resistance_m,
        k_tp=k_tp,
        k_sl=k_sl,
    )

    # RR(gross) 산출
    if direction == "long":
        rr_gross = (tp - entry) / max(1e-8, entry - sl)
    else:
        rr_gross = (entry - tp) / max(1e-8, sl - entry)
    rr_req = RR_MIN_HIGH_PROB if prob >= PROB_RELAX_THRESHOLD else RR_MIN
    reasons: List[str] = []
    if prob < MIN_PROB:
        reasons.append("prob_below_threshold")
    if not _spread_ok(spread_bps):
        reasons.append(f"wide_spread({spread_bps:.2f}bps)")

    # ---- 신규 게이트: MTF 정합 / 쇼크가드 / 쿨다운 ----
    mtf_ok, mtf_reason = _mtf_align_ok(direction, extra)
    if not mtf_ok:
        reasons.append(mtf_reason)

    sg_block, sg_bps, sg_mult, sg_reason = _shock_guard_block(direction, ohlcv, atr5)
    if sg_block:
        reasons.append(sg_reason)

    cd_active2, cd_left2 = _cooldown_active(symbol)
    if cd_active2:
        reasons.append(f"entry_cooldown({cd_left2}m_left)")

    # ---- 수수료 인지형 RR 평가 ----
    rr_for_report = _rr_with_fee_mode(entry, tp, sl)
    rr_req = RR_MIN_HIGH_PROB if prob >= PROB_RELAX_THRESHOLD else RR_MIN
    if rr_for_report <= 0 or rr_for_report < rr_req:
        reasons.append(f"rr_net_below_min({rr_for_report:.2f}<{rr_req:.2f})")

    # --- 변동성 가중 사이징(기존 구현 유지) ---
    risk_scalar = 1.0
    try:
        if VOL_SIZE_SCALING and _df_ok(ohlcv):
            atr_series = compute_atr(ohlcv, window=14)
            cur = float(atr_series.iloc[-1]) if len(atr_series) else 0.0
            med = float(atr_series.tail(VOL_LOOKBACK).median()) if len(atr_series) else 0.0
            if cur > 0 and med > 0:
                risk_scalar = max(VOL_SCALAR_MIN, min(VOL_SCALAR_MAX, med / cur))
    except Exception:
        risk_scalar = 1.0

    risk_ok = (len(reasons) == 0)
    
    log_event(
    "signal.gate",
    symbol=symbol,
    direction=direction,
    prob=float(prob),
    spread_bps=float(spread_bps),
    rr=float(rr_for_report),
    reasons=";".join(reasons) if reasons else "ok",
)
    out = {
        "symbol": symbol,
        "action": "enter" if risk_ok else "hold",
        "direction": direction,
        "entry": float(entry),
        "tp": float(tp),
        "sl": float(sl),
        "prob": float(prob),
        "rr": float(rr_for_report),
        "risk_ok": bool(risk_ok),
        "reason": "ok" if risk_ok else ";".join(reasons) or "no_trade_conditions",
        "result": {
            "direction": direction, "entry": float(entry), "tp": float(tp), "sl": float(sl),
            "prob": float(prob), "rr": float(rr_for_report), "risk_ok": bool(risk_ok),
            "risk_scalar": float(risk_scalar),
        },
    }
    log_event("signal.decision", symbol=symbol, direction=direction, prob=prob,
              entry=entry, tp=tp, sl=sl, rr=rr_for_report, risk_ok=risk_ok)
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

def _enter_limit_then_brackets(symbol: str, side: str, qty: float,
                               desired_entry: float, tp: float, sl: float) -> Dict[str, Any]:
    """
    ENTRY_POST_ONLY=true → time_in_force="GTX"(POST-ONLY), 가격은 BUY=bestBid / SELL=bestAsk, 절대 교차 금지.
    그렇지 않으면 기존 GTC 슬리피지 예산/틱 넛지 로직 사용.
    """
    side = side.upper()
    last_submitted: Optional[float] = None

    def _post_only_px() -> Optional[float]:
        bid, ask = _best_quotes(symbol)
        if bid is None or ask is None:
            return None
        try:
            f = load_symbol_filters(symbol)
            tick = float(f["tickSize"])
        except Exception:
            tick = 0.0
        if side == "BUY":
            px = normalize_price_for_side(symbol, bid, side="BUY")  # ≤ bid
            # 안전: 혹시라도 ask와 같거나 넘으면 한 틱 낮춤
            if ask and tick > 0 and px >= ask:
                px = normalize_price_for_side(symbol, ask - tick, side="BUY")
            return float(px)
        else:
            px = normalize_price_for_side(symbol, ask, side="SELL")  # ≥ ask
            # 안전: bid 이하로 내려가면 한 틱 올림
            if bid and tick > 0 and px <= bid:
                px = normalize_price_for_side(symbol, bid + tick, side="SELL")
            return float(px)

    def _next_px(prev: Optional[float]) -> Optional[float]:
        if ENTRY_POST_ONLY:
            return _post_only_px()
        return _limit_price_for_side(symbol, side, desired_entry, prev_submitted=prev)

    tif = "GTX" if ENTRY_POST_ONLY else "GTC"
    price = _next_px(None)
    if price is None or price <= 0:
        raise RuntimeError("Could not determine limit price")
    last_submitted = price

    # 1st attempt
    try:
        entry_res = place_limit_order(symbol, side, quantity=qty, price=price,
                                      time_in_force=tif, reduce_only=False, post_only=ENTRY_POST_ONLY)
    except Exception:
        entry_res = {"type": ("LIMIT_POST_ONLY_REJECTED" if ENTRY_POST_ONLY else "LIMIT_FAILOVER_MARKET")}

    filled = 0.0
    reprices = 0
    from time import sleep, time as _t
    deadline = _t() + float(LIMIT_TTL_SEC)

    # fill poll
    while _t() < deadline:
        q = _position_qty_after_fill(symbol, side)
        if q >= (qty * 0.95):
            filled = q
            break
        sleep(max(0.1, float(LIMIT_POLL_SEC)))

    # reprices
    while filled <= 0 and reprices < LIMIT_MAX_REPRICES:
        reprices += 1
        try:
            cancel_open_orders(symbol)
        except Exception:
            pass
        price = _next_px(last_submitted)
        if price is None or price <= 0:
            break
        last_submitted = price
        try:
            entry_res = place_limit_order(symbol, side, quantity=qty, price=price,
                                          time_in_force=tif, reduce_only=False, post_only=ENTRY_POST_ONLY)
        except Exception:
            entry_res = {"type": ("LIMIT_POST_ONLY_REJECTED" if ENTRY_POST_ONLY else "LIMIT_FAILOVER_MARKET")}
        deadline = _t() + float(LIMIT_TTL_SEC)
        while _t() < deadline:
            q = _position_qty_after_fill(symbol, side)
            if q >= (qty * 0.95):
                filled = q
                break
            sleep(max(0.1, float(LIMIT_POLL_SEC)))
        if filled > 0:
            break

    # TTL done → MARKET fallback (옵션)
    if filled <= 0 and LIMIT_TTL_FALLBACK_TO_MARKET:
        try:
            # POST-ONLY라도 TTL 이후엔 체결 보장을 위해 MARKET 폴백 허용(운영 정책)
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
        # after exec_res prepared
        entry_fill_is_maker = (mode == "LIMIT" and ENTRY_POST_ONLY and exec_res.get("reprices", 0) >= 0 and exec_res.get("filled_qty", 0) > 0)
        row = {
            "timestamp": datetime.now(tz=timezone.utc).isoformat(),
            "symbol": symbol,
            "side": "long" if side == "BUY" else "short",
            "qty": float(exec_res.get("filled_qty", qty) if mode == "LIMIT" else qty),
            "entry": float(entry), "tp": float(tp), "sl": float(sl),
            "exit": 0.0, "pnl": 0.0, "status": "open", "id": "",
            "entry_maker": int(1 if entry_fill_is_maker else 0),  # <-- 추가
            "tp_type": TP_ORDER_TYPE,                              # <-- 추가(리포팅용)
        }
        headers = ["timestamp","symbol","side","qty","entry","tp","sl","exit","pnl","status","id","entry_maker","tp_type"]
                
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
