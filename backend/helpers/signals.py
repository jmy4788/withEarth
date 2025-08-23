# helpers/signals.py — 3rd pass (balance-% sizing, journaling/telemetry extended)
# Hotfix-1: export get_overview() for app.py import
# Hotfix-2: restore _last_open_trade_timestamp() used by cooldown/time-barrier
from __future__ import annotations
from typing import Any, Dict, List, Optional, Tuple
import logging, os, uuid
from dataclasses import dataclass
from datetime import datetime, timezone, timedelta
import pandas as pd
import numpy as np

# --- utils & io ---
try:
    from .utils import LOG_DIR, gcs_enabled, gcs_append_csv_row, log_event  # type: ignore
except Exception:  # pragma: no cover
    LOG_DIR = os.path.join(os.getcwd(), "logs")
    def gcs_enabled() -> bool: return False
    def gcs_append_csv_row(*args, **kwargs) -> None: return None
    def log_event(*args, **kwargs): pass

# --- data & indicators ---
try:
    from .data_fetch import (
        fetch_data, fetch_mtf_raw, add_indicators, compute_atr,
        compute_orderbook_stats, compute_recent_price_sequence,
        compute_support_resistance, compute_relative_volume,
        compute_trend_filter, fetch_orderbook,
    )  # type: ignore
except Exception as e:  # pragma: no cover
    raise

# --- predictor (Gemini) ---
try:
    from .predictor import get_gemini_prediction, should_predict  # type: ignore
except Exception as e:  # pragma: no cover
    raise

# --- Binance client wrapper ---
try:
    from .binance_client import (
        get_overview as _bn_get_overview,
        cancel_open_orders, set_position_mode, set_margin_type, set_leverage,
        load_symbol_filters, ensure_min_notional, normalize_price_for_side,
        place_market_order, place_limit_order, place_bracket_orders,
        get_position, get_open_orders, get_last_price,
        replace_stop_loss_to_price, find_recent_exit_fill,
    )  # type: ignore
except Exception as e:  # pragma: no cover
    raise

# --- optional probability calibrator ---
try:
    from .calibration import calibrate_prob  # type: ignore
except Exception:
    def calibrate_prob(p: float) -> float: return float(p)

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# ================================
# ENV (risk/execution parameters)
# ================================
MIN_PROB = float(os.getenv("MIN_PROB", "0.60"))
RR_MIN = float(os.getenv("RR_MIN", "1.20"))

# ATR-based levels (respect current user values)
ATR_MULT_TP = float(os.getenv("ATR_MULT_TP", "2.2"))
ATR_MULT_SL = float(os.getenv("ATR_MULT_SL", "1.2"))

DEFAULT_LEVERAGE = int(os.getenv("LEVERAGE", "5"))
MARGIN_TYPE = os.getenv("MARGIN_TYPE", "ISOLATED").upper()
POSITION_MODE = os.getenv("POSITION_MODE", "ONEWAY").upper()
MAX_SPREAD_BPS = float(os.getenv("MAX_SPREAD_BPS", "1.5"))

HORIZON_MIN = int(os.getenv("HORIZON_MIN", "30"))
TIME_BARRIER_ENABLED = str(os.getenv("TIME_BARRIER_ENABLED", "true")).lower() in ("1","true","yes")

ENTRY_MODE = os.getenv("ENTRY_MODE", "LIMIT").upper()  # MARKET | LIMIT
LIMIT_TTL_SEC = float(os.getenv("LIMIT_TTL_SEC", "20"))
LIMIT_POLL_SEC = float(os.getenv("LIMIT_POLL_SEC", "1.0"))
LIMIT_MAX_REPRICES = int(os.getenv("LIMIT_MAX_REPRICES", "2"))
LIMIT_MAX_SLIPPAGE_BPS = float(os.getenv("LIMIT_MAX_SLIPPAGE_BPS", "2.0"))
LIMIT_TTL_FALLBACK_TO_MARKET = str(os.getenv("LIMIT_TTL_FALLBACK_TO_MARKET", "true")).lower() in ("1","true","yes")

PROB_RELAX_THRESHOLD = float(os.getenv("PROB_RELAX_THRESHOLD", "0.78"))
RR_MIN_HIGH_PROB = float(os.getenv("RR_MIN_HIGH_PROB", "1.08"))
TP_ORDER_TYPE = os.getenv("TP_ORDER_TYPE", "LIMIT").upper()
SL_ORDER_TYPE = os.getenv("SL_ORDER_TYPE", "STOP_MARKET").upper()

# --- Break-even trailing ---
BE_TRAILING_ENABLED = str(os.getenv("BE_TRAILING_ENABLED", "true")).lower() in ("1","true","yes")
BE_TRIGGER_R_MULT = float(os.getenv("BE_TRIGGER_R_MULT", "1.0"))
BE_OFFSET_TICKS = int(os.getenv("BE_OFFSET_TICKS", "2"))

# --- volatility-weighted sizing ---
VOL_SIZE_SCALING = str(os.getenv("VOL_SIZE_SCALING", "true")).lower() in ("1","true","yes")
VOL_LOOKBACK = int(os.getenv("VOL_LOOKBACK", "60"))
VOL_SCALAR_MIN = float(os.getenv("VOL_SCALAR_MIN", "0.50"))
VOL_SCALAR_MAX = float(os.getenv("VOL_SCALAR_MAX", "1.25"))

# --- prob calibration toggle ---
USE_CALIBRATED_PROB = str(os.getenv("USE_CALIBRATED_PROB", "true")).lower() in ("1","true","yes")

# journaling
TRADES_CSV = os.path.join(LOG_DIR, "trades.csv")

# post-only & fee-aware RR gate
ENTRY_POST_ONLY = str(os.getenv("ENTRY_POST_ONLY", "true")).lower() in ("1","true","yes")
RR_EVAL_WITH_FEES = str(os.getenv("RR_EVAL_WITH_FEES", "true")).lower() in ("1","true","yes")
FEE_MAKER_BPS = float(os.getenv("FEE_MAKER_BPS", "2.0"))
FEE_TAKER_BPS = float(os.getenv("FEE_TAKER_BPS", "4.0"))

# shock guard & MTF & cooldown
SHOCK_BPS = float(os.getenv("SHOCK_BPS", "30.0"))
SHOCK_ATR_MULT = float(os.getenv("SHOCK_ATR_MULT", "1.8"))
MTF_ALIGN_ENABLED = str(os.getenv("MTF_ALIGN_ENABLED", "true")).lower() in ("1","true","yes")
MTF_RSI_LONG_MIN = float(os.getenv("MTF_RSI_LONG_MIN", "49"))
MTF_RSI_SHORT_MAX = float(os.getenv("MTF_RSI_SHORT_MAX", "51"))
ENTRY_COOLDOWN_MIN = int(os.getenv("ENTRY_COOLDOWN_MIN", "10"))

RR_GATE_MODE = os.getenv("RR_GATE_MODE", "expected").lower()  # worst | expected | best
MAKER_PROB_LOOKBACK = int(os.getenv("MAKER_PROB_LOOKBACK", "200"))

# --- sizing mode (NEW) ---
SIZE_MODE = os.getenv("SIZE_MODE", "USDT").upper()              # USDT | BALANCE_PCT
RISK_USDT = float(os.getenv("RISK_USDT", "100"))                # legacy (SIZE_MODE=USDT)
RISK_BAL_PCT = float(os.getenv("RISK_BAL_PCT", "1.0"))          # % of wallet balance
SIZE_BAL_INCLUDE_UPNL = str(os.getenv("SIZE_BAL_INCLUDE_UPNL", "false")).lower() in ("1", "true", "yes")
SIZE_BAL_ASSET_OVERRIDE = os.getenv("SIZE_BAL_ASSET_OVERRIDE", "").upper().strip()

# =====================
# Dataclass
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

# ============== helpers ==============
def _now_utc() -> datetime:
    return datetime.now(tz=timezone.utc)

def _parse_iso(ts: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(ts))
    except Exception:
        try:
            return datetime.fromisoformat(str(ts).replace("Z", "+00:00"))
        except Exception:
            return None

def _iso_to_ms(ts: str) -> Optional[int]:
    dt = _parse_iso(ts); return int(dt.timestamp() * 1000) if dt else None

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

# maker-prob estimate (for RR expected-mode)
def _estimate_p_maker_from_journal() -> float:
    try:
        path = TRADES_CSV
        if not os.path.exists(path): return 0.5
        import csv
        rows: List[Dict[str, Any]] = []
        with open(path, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r: rows.append(row)
        if not rows: return 0.5
        rows = rows[-MAKER_PROB_LOOKBACK:]
        vals = []
        for row in rows:
            try:
                vals.append(int(row.get("entry_maker", 0)))
            except Exception:
                pass
        n = len(vals)
        if n == 0:
            return 0.5
        s = sum(1 for v in vals if v == 1)
        a = float(os.getenv("MAKER_PROB_PRIOR_A", "3"))
        b = float(os.getenv("MAKER_PROB_PRIOR_B", "3"))
        p = (s + a) / (n + a + b)
        return float(min(1.0, max(0.0, p)))
    except Exception:
        return 0.5

# RR with fees (worst/best/expected)
def _rr_with_fee_mode(direction: str, entry: float, tp: float, sl: float) -> float:
    e = float(entry); t = float(tp); s = float(sl)
    if e <= 0 or t <= 0 or s <= 0: return 0.0
    if not RR_EVAL_WITH_FEES:
        if direction == "long":
            up_gross = (t - e) / e; dn_gross = (e - s) / e
        else:
            up_gross = (e - t) / e; dn_gross = (s - e) / e
        if dn_gross <= 0: return 0.0
        return max(0.0, up_gross) / max(1e-12, dn_gross)
    maker = FEE_MAKER_BPS / 1e4
    taker = FEE_TAKER_BPS / 1e4
    def _rr_net_local(e_is_maker: bool, tp_is_maker: bool, sl_is_taker: bool) -> float:
        if direction == "long":
            up_gross = (t - e) / e; dn_gross = (e - s) / e
        else:
            up_gross = (e - t) / e; dn_gross = (s - e) / e
        fee_e  = maker if e_is_maker else taker
        fee_tp = maker if tp_is_maker else taker
        fee_sl = taker if sl_is_taker else maker
        up_net = max(0.0, up_gross - (fee_e + fee_tp))
        dn_net = max(1e-12, dn_gross + (fee_e + fee_sl))
        return float(up_net / dn_net)
    entry_is_maker_worst = (ENTRY_MODE == "LIMIT" and ENTRY_POST_ONLY and (not LIMIT_TTL_FALLBACK_TO_MARKET))
    if RR_GATE_MODE == "worst":
        return _rr_net_local(entry_is_maker_worst, TP_ORDER_TYPE == "LIMIT", True)
    if RR_GATE_MODE == "best":
        return _rr_net_local(True, TP_ORDER_TYPE == "LIMIT", True)
    p_maker = _estimate_p_maker_from_journal()
    rr_maker = _rr_net_local(True, TP_ORDER_TYPE == "LIMIT", True)
    rr_taker = _rr_net_local(False, TP_ORDER_TYPE == "LIMIT", True)
    return max(0.0, p_maker * rr_maker + (1.0 - p_maker) * rr_taker)

# --- shock guard / MTF / cooldown ---
def _shock_guard_block(direction: str, ohlcv: pd.DataFrame, atr5: float) -> Tuple[bool, float, float, str]:
    if not _df_ok(ohlcv) or direction not in ("long","short"):
        return False, 0.0, 0.0, ""
    last = ohlcv.iloc[-1]
    close = float(last.get("close", 0.0)); openp = float(last.get("open", 0.0))
    chg = close - openp
    chg_bps = abs(chg) / max(1e-8, close) * 1e4
    atr_mult = abs(chg) / max(1e-8, float(atr5))
    candle_up = (chg > 0)
    counter = (direction == "long" and not candle_up) or (direction == "short" and candle_up)
    block = counter and ((chg_bps >= SHOCK_BPS) or (atr_mult >= SHOCK_ATR_MULT))
    reason = f"shock_guard(countertrend,{chg_bps:.1f}bps,{atr_mult:.2f}ATR)"
    return block, float(chg_bps), float(atr_mult), reason

def _mtf_align_ok(direction: str, extra: Dict[str, Any]) -> Tuple[bool, str]:
    if not MTF_ALIGN_ENABLED or direction not in ("long","short"): return True, ""
    r1h = float(extra.get("RSI_1h", 50.0)); r4h = float(extra.get("RSI_4h", 50.0))
    if direction == "long":
        ok = (r1h >= MTF_RSI_LONG_MIN) and (r4h >= MTF_RSI_LONG_MIN)
    else:
        ok = (r1h <= MTF_RSI_SHORT_MAX) and (r4h <= MTF_RSI_SHORT_MAX)
    return ok, "mtf_rsi_mismatch"

# === last-open timestamp (restored) ===
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

def _cooldown_active(symbol: str) -> Tuple[bool, int]:
    mins = int(max(0, ENTRY_COOLDOWN_MIN))
    if mins <= 0: return False, 0
    last_open = _last_open_trade_timestamp(symbol)
    if not last_open: return False, 0
    left = mins - int((_now_utc() - last_open).total_seconds() // 60)
    return (left > 0), max(0, left)

# === SR/ATR 기반 레벨 산출 (LLM SR clamp) ===
def _tp_sl_with_sr_clamp(direction: str, entry: float, atr5: float, sr_high: float, sr_low: float,
                         llm_support: Optional[float], llm_resistance: Optional[float],
                         k_tp: float = ATR_MULT_TP, k_sl: float = ATR_MULT_SL) -> Tuple[float, float]:
    e = float(entry or 0.0); a = float(max(atr5, 1e-12))
    hi = float(sr_high or 0.0); lo = float(sr_low or 0.0)
    sup = float(llm_support) if (llm_support is not None) else None
    res = float(llm_resistance) if (llm_resistance is not None) else None
    if direction == "long":
        base_tp = e + k_tp * a; base_sl = e - k_sl * a
        tp_candidates = [x for x in [base_tp, hi, res] if (x and x > e)]
        sl_candidates = [x for x in [base_sl, lo, sup] if (x and x < e)]
        tp = max(tp_candidates) if tp_candidates else base_tp
        sl = max(sl_candidates) if sl_candidates else base_sl
    else:
        base_tp = e - k_tp * a; base_sl = e + k_sl * a
        tp_candidates = [x for x in [base_tp, lo, sup] if (x and x < e)]
        sl_candidates = [x for x in [base_sl, hi, res] if (x and x > e)]
        tp = min(tp_candidates) if tp_candidates else base_tp
        sl = min(sl_candidates) if sl_candidates else base_sl
    return float(tp), float(sl)

# ---------------------------------
# Payload & signal
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
        },
        "times": base.get("times", {}),
        "price_sequence": price_seq,
        "sr_levels": sr5,
        "relative_volume": float(compute_relative_volume(ohlcv)) if _df_ok(ohlcv) else 1.0,
        "trend_filter": trend,
        "horizon_min": HORIZON_MIN,
    }
    return payload, ohlcv, ob

def _spread_ok(spread_bps: float) -> bool:
    try:
        return float(spread_bps) <= float(MAX_SPREAD_BPS)
    except Exception:
        return False

def _rule_backup(ohlcv: pd.DataFrame, trend: Dict[str, Any]) -> Tuple[str, float]:
    if not _df_ok(ohlcv): return "hold", 0.5
    close = float(ohlcv["close"].iloc[-1])
    sma20 = float(ohlcv["SMA_20"].iloc[-1]) if "SMA_20" in ohlcv else close
    rsi = float(ohlcv["RSI"].iloc[-1]) if "RSI" in ohlcv else 50.0
    up = bool(trend.get("daily_uptrend", False))
    if close > sma20 and rsi > 55 and up:   return "long", 0.61
    if close < sma20 and rsi < 45 and not up: return "short", 0.61
    return "hold", 0.5

# ---------------------------------
# Balance-% sizing helpers (NEW)
# ---------------------------------
def _infer_quote_asset(symbol: str) -> str:
    try:
        f = load_symbol_filters(symbol)
        raw = f.get("raw") if isinstance(f, dict) else {}
        q = (raw.get("quoteAsset") if isinstance(raw, dict) else None) or ""
        q = str(q).upper().strip()
        if q: return q
    except Exception:
        pass
    for cand in ("FDUSD","USDT","USDC","BUSD","TUSD","DAI","BIDR","EUR","TRY","BRL","USD"):
        if symbol.upper().endswith(cand):
            return cand
    return "USDT"

def _wallet_balance(asset: str, include_upnl: bool = False) -> float:
    try:
        ov = _bn_get_overview() or {}
        bals = ov.get("balances") or []
        assetU = (asset or "").upper()
        for b in bals:
            if str(b.get("asset","")).upper() == assetU:
                bal = float(b.get("balance", 0.0))
                if include_upnl:
                    bal += float(b.get("unrealizedPnL", 0.0))
                return max(0.0, float(bal))
    except Exception as e:
        logger.info("wallet_balance read failed: %s", e)
    return 0.0

def _compute_size(symbol: str, entry: float, sl: float, risk_scalar: float = 1.0) -> Tuple[float, Dict[str, Any]]:
    entry = float(entry or 0.0)
    if entry <= 0:
        return 0.0, {"size_mode": SIZE_MODE}
    size_mode = (SIZE_MODE or "USDT").upper()
    bal_asset = (SIZE_BAL_ASSET_OVERRIDE or _infer_quote_asset(symbol)).upper()
    include_upnl = bool(SIZE_BAL_INCLUDE_UPNL)
    if size_mode.startswith("BAL"):
        wallet = _wallet_balance(bal_asset, include_upnl=include_upnl)
        notional = max(1.0, float(wallet) * max(0.0, float(RISK_BAL_PCT)) / 100.0) * float(risk_scalar)
    else:
        notional = max(1.0, float(RISK_USDT) * float(risk_scalar))
    qty = notional / entry
    try:
        f = load_symbol_filters(symbol)
        qty = ensure_min_notional(symbol, qty, price=entry, filters=f)
    except Exception:
        pass
    meta = {
        "size_mode": size_mode,
        "bal_asset": bal_asset,
        "wallet_balance": None,
        "bal_pct": float(RISK_BAL_PCT) if size_mode.startswith("BAL") else 0.0,
        "include_upnl": include_upnl,
        "notional": float(qty * entry),
        "risk_scalar": float(risk_scalar),
    }
    if size_mode.startswith("BAL"):
        try:
            meta["wallet_balance"] = float(_wallet_balance(bal_asset, include_upnl=include_upnl))
        except Exception:
            meta["wallet_balance"] = None
    return float(qty), meta

# ---------------------------------
# Signal generation
# ---------------------------------
def generate_signal(symbol: str) -> Dict[str, Any]:
    payload, ohlcv, ob = _build_payload(symbol)
    spread_bps_gate = float((payload.get("extra") or {}).get("orderbook_spread", 0.0))
    proceed_basic = should_predict(payload, min_vol_frac_env="MIN_VOL_FRAC") and _spread_ok(spread_bps_gate)
    dir_hint, _ = _rule_backup(ohlcv, payload.get("trend_filter") or {})
    atr5 = float((payload.get("extra") or {}).get("ATR_5m") or 0.0)
    cd_active, cd_left = _cooldown_active(symbol)
    if cd_active:
        return {"symbol": symbol, "action":"hold","direction":"hold","entry": float((payload.get("entry_5m") or {}).get("close") or 0.0),
                "tp":0.0,"sl":0.0,"prob":0.5,"risk_ok":False,"rr":0.0,"reason": f"pre_gate_cooldown({cd_left}m_left)"}
    sg_long, bpsL, multL, _ = _shock_guard_block("long", ohlcv, atr5)
    sg_short, bpsS, multS, _ = _shock_guard_block("short", ohlcv, atr5)
    if (sg_long or sg_short):
        candle_up = float(ohlcv.iloc[-1]["close"]) - float(ohlcv.iloc[-1]["open"]) > 0 if _df_ok(ohlcv) else False
        shock_dir = "long" if candle_up else "short"
        if not (dir_hint in ("long","short") and dir_hint == shock_dir):
            bps = max(bpsL, bpsS); mult = max(multL, multS)
            return {"symbol":symbol,"action":"hold","direction":"hold","entry": float((payload.get("entry_5m") or {}).get("close") or 0.0),
                    "tp":0.0,"sl":0.0,"prob":0.5,"risk_ok":False,"rr":0.0,"reason": f"pre_gate_shock({bps:.1f}bps,{mult:.2f}ATR)"}
    if not proceed_basic:
        trend = payload.get("trend_filter") or {}
        direction_rb, prob_rb = _rule_backup(ohlcv, trend)
        if direction_rb in ("long","short"):
            direction, prob = direction_rb, prob_rb
            prob_raw = float(prob_rb)
            prob_cal = float(calibrate_prob(prob_raw)) if USE_CALIBRATED_PROB else float(prob_raw)
            prob = prob_cal
            llm_support = None; llm_resistance = None
        else:
            return {"symbol":symbol,"action":"hold","direction":"hold","entry": float((payload.get("entry_5m") or {}).get("close") or 0.0),
                    "tp":0.0,"sl":0.0,"prob":0.5,"risk_ok":False,"rr":0.0,"reason":"pre_gate_block"}
    else:
        llm_decision = get_gemini_prediction(payload, symbol=symbol)
        direction = str(llm_decision.get("direction") or "").lower()
        prob_raw = float(llm_decision.get("prob", 0.0))
        prob_cal = float(calibrate_prob(prob_raw)) if USE_CALIBRATED_PROB else float(prob_raw)
        prob = prob_cal
        llm_support = llm_decision.get("support")
        llm_resistance = llm_decision.get("resistance")
    entry = float((payload.get("entry_5m") or {}).get("close") or 0.0)
    extra = payload.get("extra") or {}
    spread_bps = float(extra.get("orderbook_spread") or 0.0)
    if direction not in ("long","short") or entry <= 0:
        log_event("signal.decision", symbol=symbol, direction="hold", prob=prob, entry=entry, tp=0.0, sl=0.0, rr=0.0, risk_ok=False)
        return {"symbol":symbol,"action":"hold","direction":"hold","entry":entry,"tp":0.0,"sl":0.0,"prob":prob,"risk_ok":False,"rr":0.0,"reason":"invalid_direction_or_entry"}
    sr_high = float(extra.get("recent_high_5m") or 0.0)
    sr_low  = float(extra.get("recent_low_5m") or 0.0)
    tp, sl = _tp_sl_with_sr_clamp(direction, entry, float(extra.get("ATR_5m") or 0.0), sr_high, sr_low, llm_support, llm_resistance,
                                  k_tp=float(os.getenv("ATR_MULT_TP", str(ATR_MULT_TP))), k_sl=float(os.getenv("ATR_MULT_SL", str(ATR_MULT_SL))))
    rr_net = _rr_with_fee_mode(direction, entry, tp, sl)
    rr_req = RR_MIN_HIGH_PROB if prob >= PROB_RELAX_THRESHOLD else RR_MIN
    reasons: List[str] = []
    if prob < MIN_PROB: reasons.append("prob_below_threshold")
    if not _spread_ok(spread_bps): reasons.append(f"wide_spread({spread_bps:.2f}bps)")
    mtf_ok, mtf_reason = _mtf_align_ok(direction, extra)
    if not mtf_ok: reasons.append(mtf_reason)
    sg_block, sg_bps, sg_mult, sg_reason = _shock_guard_block(direction, ohlcv, float(extra.get("ATR_5m") or 0.0))
    if sg_block: reasons.append(sg_reason)
    cd_active2, cd_left2 = _cooldown_active(symbol)
    if cd_active2: reasons.append(f"entry_cooldown({cd_left2}m_left)")
    if rr_net <= 0 or rr_net < rr_req: reasons.append(f"rr_net_below_min({rr_net:.2f}<{rr_req:.2f})")
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
    log_event("signal.gate", symbol=symbol, direction=direction, prob=float(prob), spread_bps=float(spread_bps),
              rr=float(rr_net), rr_req=float(rr_req), rr_mode=RR_GATE_MODE, reasons=";".join(reasons) if reasons else "ok")
    telemetry = {
        "spread_bps": float(spread_bps),
        "atr_now": float(extra.get("ATR_5m") or 0.0),
        "funding_pct": float(extra.get("funding_rate_pct") or 0.0),
        "maker_prob_est": _estimate_p_maker_from_journal(),
        "rr_gate_mode": RR_GATE_MODE,
        "sizing_mode": SIZE_MODE,
    }
    out = {
        "symbol": symbol,
        "action": "enter" if risk_ok else "hold",
        "direction": direction,
        "entry": float(entry),
        "tp": float(tp),
        "sl": float(sl),
        "prob": float(prob),             # calibrated if USE_CALIBRATED_PROB else raw
        "prob_raw": float(prob_raw if 'prob_raw' in locals() else prob),
        "prob_cal": float(prob if 'prob' in locals() else prob),
        "rr": float(rr_net),
        "risk_ok": bool(risk_ok),
        "reason": "ok" if risk_ok else ";".join(reasons) or "no_trade_conditions",
        "result": {
            "direction": direction, "entry": float(entry), "tp": float(tp), "sl": float(sl),
            "prob": float(prob), "prob_raw": float(prob_raw if 'prob_raw' in locals() else prob),
            "prob_cal": float(prob if 'prob' in locals() else prob),
            "rr": float(rr_net), "risk_ok": bool(risk_ok), "risk_scalar": float(risk_scalar),
        },
        "telemetry": telemetry,
    }
    log_event("signal.decision", symbol=symbol, direction=direction, prob=float(prob), entry=float(entry),
              tp=float(tp), sl=float(sl), rr=float(rr_net), risk_ok=bool(risk_ok))
    return out

# --- Limit-first execution with TTL/reprice ---
def _position_qty_after_fill(symbol: str, side: str) -> float:
    p = get_position(symbol) or {}
    amt = float(p.get("positionAmt") or p.get("positionAmount") or 0.0)
    if side.upper() == "BUY" and amt > 0: return abs(amt)
    if side.upper() == "SELL" and amt < 0: return abs(amt)
    return 0.0

def _limit_price_for_side(symbol: str, side: str, desired_entry: float, prev_submitted: Optional[float]) -> Optional[float]:
    bid, ask = _best_quotes(symbol)
    if bid is None or ask is None: return None
    budget = LIMIT_MAX_SLIPPAGE_BPS / 1e4
    if side.upper() == "BUY":
        raw = min(ask, max(bid, desired_entry))
        capped = min(raw, desired_entry * (1.0 + budget))
        px = normalize_price_for_side(symbol, capped, side="BUY")
    else:
        raw = max(bid, min(ask, desired_entry))
        capped = max(raw, desired_entry * (1.0 - budget))
        px = normalize_price_for_side(symbol, capped, side="SELL")
    try:
        f = load_symbol_filters(symbol); tick = float(f["tickSize"])
    except Exception:
        tick = 0.0
    if prev_submitted is not None and tick > 0:
        if side.upper() == "BUY":
            if abs(px - prev_submitted) < tick: px = prev_submitted + tick
        else:
            if abs(px - prev_submitted) < tick: px = prev_submitted - tick
        px = normalize_price_for_side(symbol, px, side=side)
    return float(px)

def _enter_limit_then_brackets(symbol: str, side: str, qty: float,
                               desired_entry: float, tp: float, sl: float) -> Dict[str, Any]:
    side = side.upper()
    tif = "GTX" if ENTRY_POST_ONLY else "GTC"
    last_submitted: Optional[float] = None
    def _post_only_px() -> Optional[float]:
        bid, ask = _best_quotes(symbol)
        if bid is None or ask is None: return None
        try:
            f = load_symbol_filters(symbol); tick = float(f["tickSize"])
        except Exception:
            tick = 0.0
        if side == "BUY":
            px = normalize_price_for_side(symbol, bid, side="BUY")
            if ask and tick > 0 and px >= ask:
                px = normalize_price_for_side(symbol, ask - tick, side="BUY")
            return float(px)
        else:
            px = normalize_price_for_side(symbol, ask, side="SELL")
            if bid and tick > 0 and px <= bid:
                px = normalize_price_for_side(symbol, bid + tick, side="SELL")
            return float(px)
    def _next_px(prev: Optional[float]) -> Optional[float]:
        return _post_only_px() if ENTRY_POST_ONLY else _limit_price_for_side(symbol, side, desired_entry, prev)
    price = _next_px(None)
    if price is None or price <= 0: raise RuntimeError("Could not determine limit price")
    try:
        entry_res = place_limit_order(symbol, side, quantity=qty, price=price, time_in_force=tif, reduce_only=False, post_only=ENTRY_POST_ONLY)
    except Exception:
        entry_res = {"type": ("LIMIT_POST_ONLY_REJECTED" if ENTRY_POST_ONLY else "LIMIT_FAILOVER_MARKET")}
    filled = 0.0; reprices = 0; used_market_fallback = False
    from time import sleep, time as _t
    deadline = _t() + float(LIMIT_TTL_SEC)
    while _t() < deadline:
        q = _position_qty_after_fill(symbol, side)
        if q >= (qty * 0.95): filled = q; break
        sleep(max(0.1, float(LIMIT_POLL_SEC)))
    while filled <= 0 and reprices < LIMIT_MAX_REPRICES:
        reprices += 1
        try: cancel_open_orders(symbol)
        except Exception: pass
        price = _next_px(price)
        if price is None or price <= 0: break
        try:
            entry_res = place_limit_order(symbol, side, quantity=qty, price=price, time_in_force=tif, reduce_only=False, post_only=ENTRY_POST_ONLY)
        except Exception:
            entry_res = {"type": ("LIMIT_POST_ONLY_REJECTED" if ENTRY_POST_ONLY else "LIMIT_FAILOVER_MARKET")}
        deadline = _t() + float(LIMIT_TTL_SEC)
        while _t() < deadline:
            q = _position_qty_after_fill(symbol, side)
            if q >= (qty * 0.95): filled = q; break
            sleep(max(0.1, float(LIMIT_POLL_SEC)))
        if filled > 0: break
    if filled <= 0 and LIMIT_TTL_FALLBACK_TO_MARKET:
        try:
            entry_res = place_market_order(symbol, side, quantity=qty, reduce_only=False)
            used_market_fallback = True
            filled = _position_qty_after_fill(symbol, side)
        except Exception as e:
            logger.info("market fallback after TTL failed: %s", e)
    brackets = {"take_profit": None, "stop_loss": None}
    if filled > 0:
        try:
            brackets = place_bracket_orders(symbol, side, filled, take_profit=tp, stop_loss=sl)
        except Exception as e:
            logger.info("placing brackets failed: %s", e)
    fill_px = 0.0
    try:
        pos = get_position(symbol) or {}
        amt = float(pos.get("positionAmt") or pos.get("positionAmount") or 0.0)
        if (side == "BUY" and amt > 0) or (side == "SELL" and amt < 0):
            fill_px = float(pos.get("entryPrice") or 0.0)
    except Exception:
        pass
    if fill_px <= 0:
        try:
            for k in ("avgPrice","price"):
                v = entry_res.get(k) if isinstance(entry_res, dict) else None
                if v is not None:
                    fill_px = float(v); break
            if fill_px <= 0:
                fills = entry_res.get("fills") if isinstance(entry_res, dict) else []
                if fills and isinstance(fills, list):
                    px = fills[0].get("price")
                    fill_px = float(px) if px is not None else 0.0
        except Exception:
            pass
    return {"entry_order": entry_res, "brackets": brackets, "filled_qty": float(filled),
            "reprices": reprices, "used_market_fallback": bool(used_market_fallback),
            "entry_price": float(fill_px)}

# ---------------------------------
# Maintain (time barrier + BE trailing + cleanup)
# ---------------------------------
# helpers/signals.py

def _current_stop_price(symbol: str) -> Optional[float]:
    try:
        orders = get_open_orders(symbol)
    except Exception:
        orders = []
    sl_prices: List[float] = []
    for o in orders or []:
        d = o if isinstance(o, dict) else {}
        t = str((d.get("type") or "")).upper()
        if t in ("STOP", "STOP_MARKET"):
            for key in ("stopPrice", "stop_price", "price"):
                v = d.get(key)
                if v is None:
                    continue
                try:
                    sl_prices.append(float(v))
                    break
                except Exception:
                    continue
    if not sl_prices:
        return None
    sl_prices.sort()
    mid = sl_prices[len(sl_prices)//2]
    return float(mid)

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

def _last_open_row_index_and_ts(symbol: str) -> Tuple[Optional[int], Optional[int], list]:
    import csv
    if not os.path.exists(TRADES_CSV): return None, None, []
    rows = []
    with open(TRADES_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r: rows.append(row)
    idx = None; ts_ms = None
    for i in range(len(rows) - 1, -1, -1):
        if str(rows[i].get("symbol","")).upper() == symbol.upper() and str(rows[i].get("status","")).lower() == "open":
            idx = i; ts_ms = _iso_to_ms(rows[i].get("timestamp","")); break
    return idx, ts_ms, rows

def _rewrite_trades_csv(rows: list, pref_headers: Optional[list] = None) -> None:
    keys = set()
    for r in rows: keys.update(r.keys())
    base = [
        "timestamp","symbol","side","qty","entry","entry_intent","tp","sl",
        "exit","pnl","status","id","prob","rr","entry_maker","tp_type","mode",
        "reprices","used_market_fallback","post_only","spread_bps","atr_now",
        "funding_pct","maker_prob_est","rr_gate_mode","reasons","close_reason",
        "size_mode","bal_asset","notional","bal_pct","exit_ts"  # NEW
    ]
    headers = [k for k in base if k in keys] + [k for k in sorted(keys) if k not in base]
    import csv, os
    os.makedirs(os.path.dirname(TRADES_CSV), exist_ok=True)
    with open(TRADES_CSV, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=headers)
        w.writeheader()
        for r in rows: w.writerow(r)

def _journal_close_last(symbol: str, exit_price: float, reason: str) -> bool:
    idx, ts_ms, rows = _last_open_row_index_and_ts(symbol)
    if idx is None or not rows: return False
    r = rows[idx]
    try:
        entry = float(r.get("entry", 0.0)); qty = float(r.get("qty", 0.0))
        side = str(r.get("side", "long"))
        entry_maker = None
        if "entry_maker" in r and str(r.get("entry_maker","")).strip() != "":
            try: entry_maker = bool(int(r.get("entry_maker"))); 
            except Exception: entry_maker = None
        tp_type = str(r.get("tp_type") or TP_ORDER_TYPE)
        fee_e = (FEE_MAKER_BPS if (entry_maker is True) else FEE_TAKER_BPS) / 1e4
        exit_is_maker = (str(tp_type).upper() == "LIMIT")
        fee_x = (FEE_MAKER_BPS if exit_is_maker else FEE_TAKER_BPS) / 1e4
        sgn = 1.0 if side.lower() == "long" else -1.0
        gross = (float(exit_price) - entry) * sgn * qty
        fees = entry * qty * fee_e + float(exit_price) * qty * fee_x
        pnl = float(gross - fees)
        now_iso = _now_utc().isoformat()
        r.update({
            "exit": f"{float(exit_price):.10f}",
            "pnl": f"{pnl:.10f}",
            "status": reason,
            "close_reason": reason,
            "exit_ts": now_iso,     # NEW
        })
        rows[idx] = r
        _rewrite_trades_csv(rows)
        log_event("journal.close", symbol=symbol, exit=float(exit_price), pnl=float(pnl), reason=reason)
        if gcs_enabled():
            try: gcs_append_csv_row("trades_close", list(r.keys()), r)
            except Exception: pass
        return True
    except Exception as e:
        logger.info("journal close failed: %s", e); return False

def _settle_by_orders(symbol: str) -> bool:
    idx, ts_ms, rows = _last_open_row_index_and_ts(symbol)
    if idx is None or ts_ms is None: return False
    info = find_recent_exit_fill(symbol, since_ms=int(ts_ms))
    if not info or not info.get("price"): return False
    typ = str(info.get("type","")).upper()
    rsn = "closed_tp" if "TAKE_PROFIT" in typ else ("closed_sl" if "STOP" in typ else "closed")
    return _journal_close_last(symbol, float(info["price"]), reason=rsn)

def maintain_positions(symbol: str) -> Dict[str, Any]:
    try:
        if _time_barrier_due(symbol):
            od = _close_position_market(symbol)
            exitp = 0.0
            try:
                exitp = float((od or {}).get("avgPrice") or (od or {}).get("price") or (od or {}).get("fills", [{}])[0].get("price", 0.0))
            except Exception:
                exitp = 0.0
            if not exitp:
                try: exitp = float(get_last_price(symbol) or 0.0)
                except Exception: exitp = 0.0
            if exitp > 0:
                _journal_close_last(symbol, exitp, reason="time_exit")
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
                            except Exception:
                                tick = 0.0
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
        if abs(amt) <= 1e-12:
            if _has_reduce_only_or_bracket_orders(symbol):
                try:
                    cancel_open_orders(symbol); log_event("cleanup.brackets_cancelled", symbol=symbol)
                except Exception as e:
                    logger.info("cleanup cancel_open_orders failed for %s: %s", symbol, e)
            settled = _settle_by_orders(symbol)
            if settled: return {"action": "settled"}
            return {"action": "none"}
        return {"action": "none"}
    except Exception as e:
        logger.info("maintain_positions error for %s: %s", symbol, e)
        return {"action": "error", "error": str(e)}

def _time_barrier_due(symbol: str) -> bool:
    if not TIME_BARRIER_ENABLED: return False
    start = _last_open_trade_timestamp(symbol)
    if not start: return False
    try:
        import pandas as _pd
        return _now_utc() >= (start + _pd.to_timedelta(HORIZON_MIN, unit="m"))
    except Exception:
        return _now_utc() >= (start + timedelta(minutes=HORIZON_MIN))

# ---------------------------------
# Manage trade (entry + journaling)
# ---------------------------------
def manage_trade(symbol: str) -> Dict[str, Any]:
    try:
        sig = generate_signal(symbol)
        if "result" not in sig or not isinstance(sig["result"], dict):
            return {"symbol": symbol, "error": "no_signal"}
        res = sig["result"]
        telemetry = sig.get("telemetry") or {}
        direction = res.get("direction", "hold")
        entry_intent = float(res.get("entry", 0.0))
        tp = float(res.get("tp", 0.0))
        sl = float(res.get("sl", 0.0))
        prob = float(res.get("prob", 0.5))
        prob_raw = float(res.get("prob_raw", prob))
        prob_cal = float(res.get("prob_cal", prob))
        rr = float(res.get("rr", 0.0))
        risk_ok = bool(res.get("risk_ok", False))
        reason = sig.get("reason", "")
        risk_scalar = float(res.get("risk_scalar", 1.0)) if "risk_scalar" in res else float(res.get("risk_scalar", 1.0))
        if direction not in ("long","short") or not risk_ok or entry_intent <= 0 or tp <= 0 or sl <= 0:
            return {"symbol": symbol, "action": "hold", "direction": direction,
                    "entry": entry_intent, "tp": tp, "sl": sl, "prob": prob,
                    "risk_ok": False, "rr": rr, "reason": reason or "no_trade_conditions"}
        if prob < float(os.getenv("MIN_PROB", "0.60")):
            return {"symbol": symbol, "action": "hold", "direction": direction,
                    "entry": entry_intent, "tp": tp, "sl": sl, "prob": prob,
                    "risk_ok": False, "rr": rr, "reason": "prob_below_threshold"}
        try: set_position_mode(POSITION_MODE)
        except Exception as e: logger.warning("set_position_mode: %s", e)
        try: set_margin_type(symbol, MARGIN_TYPE)
        except Exception as e: logger.warning("set_margin_type: %s", e)
        try: set_leverage(symbol, DEFAULT_LEVERAGE)
        except Exception as e: logger.warning("set_leverage: %s", e)
        qty, size_meta = _compute_size(symbol, entry_intent, sl, risk_scalar)
        side = "BUY" if direction == "long" else "SELL"
        if ENTRY_MODE == "MARKET":
            entry_res = place_market_order(symbol, side, quantity=qty)
            bracket_res = place_bracket_orders(symbol, side, quantity=qty, take_profit=tp, stop_loss=sl)
            exec_res = {"entry_order": entry_res, "brackets": bracket_res, "filled_qty": float(qty),
                        "reprices": 0, "used_market_fallback": True, "entry_price": float(0.0)}
            mode = "MARKET"
        else:
            exec_res = _enter_limit_then_brackets(symbol, side, qty, desired_entry=entry_intent, tp=tp, sl=sl)
            mode = "LIMIT"
        entry_actual = float(exec_res.get("entry_price") or 0.0)
        if entry_actual <= 0:
            try:
                pos = get_position(symbol) or {}
                entry_actual = float(pos.get("entryPrice") or 0.0) or entry_intent
            except Exception:
                entry_actual = entry_intent
        try:
            row = {
                "timestamp": _now_utc().isoformat(),
                "symbol": symbol,
                "side": "long" if side == "BUY" else "short",
                "qty": f"{float(exec_res.get('filled_qty') or qty):.10f}",
                "entry": f"{float(entry_actual):.10f}",
                "entry_intent": f"{float(entry_intent):.10f}",
                "tp": f"{float(tp):.10f}",
                "sl": f"{float(sl):.10f}",
                "exit": "",
                "pnl": "",
                "status": "open",
                "id": str(uuid.uuid4())[:8],
                "prob": f"{float(prob):.6f}",
                "prob_raw": f"{float(prob_raw):.6f}",
                "prob_cal": f"{float(prob_cal):.6f}",
                "rr": f"{float(rr):.6f}",
                "entry_maker": "1" if (mode == "LIMIT" and ENTRY_POST_ONLY and not exec_res.get("used_market_fallback", False)) else "0",
                "tp_type": str(TP_ORDER_TYPE),
                "mode": mode,
                "reprices": int(exec_res.get("reprices", 0)),
                "used_market_fallback": "1" if exec_res.get("used_market_fallback", False) else "0",
                "post_only": "1" if ENTRY_POST_ONLY else "0",
                "spread_bps": f"{float(telemetry.get('spread_bps', 0.0)):.6f}",
                "atr_now": f"{float(telemetry.get('atr_now', 0.0)):.10f}",
                "funding_pct": f"{float(telemetry.get('funding_pct', 0.0)):.6f}",
                "maker_prob_est": f"{float(telemetry.get('maker_prob_est', _estimate_p_maker_from_journal())):.6f}",
                "rr_gate_mode": str(telemetry.get("rr_gate_mode", RR_GATE_MODE)),
                "reasons": str(sig.get("reason","")),
                "close_reason": "",
                "size_mode": str(size_meta.get("size_mode","")),
                "bal_asset": str(size_meta.get("bal_asset","")),
                "notional": f"{float(size_meta.get('notional', float(qty*entry_intent))):.10f}",
                "bal_pct": f"{float(size_meta.get('bal_pct', 0.0)):.6f}",
            }
            _journal_append_open(row)
            if gcs_enabled():
                gcs_append_csv_row("trades", list(row.keys()), row)
        except Exception as e:
            logger.info("journal append failed: %s", e)
        try:
            log_event("order.size_meta", symbol=symbol, **{k: v for k, v in size_meta.items() if k in ("size_mode","bal_asset","wallet_balance","notional","bal_pct","include_upnl","risk_scalar")})
        except Exception:
            pass
        return {"symbol": symbol, "action": "enter", "direction": direction,
                "entry": entry_actual, "tp": tp, "sl": sl, "prob": prob,
                "risk_ok": True, "rr": rr, "order": exec_res, "mode": mode}
    except Exception as e:
        logger.exception("manage_trade failed for %s", symbol)
        return {"symbol": symbol, "error": str(e)}
    
def _journal_append_open(row: Dict[str, Any]) -> None:
    import csv, os
    os.makedirs(os.path.dirname(TRADES_CSV), exist_ok=True)
    headers = [
        "timestamp","symbol","side","qty","entry","entry_intent","tp","sl","exit","pnl","status","id",
        "prob","prob_raw","prob_cal","rr","entry_maker","tp_type","mode","reprices","used_market_fallback","post_only",
        "spread_bps","atr_now","funding_pct","maker_prob_est","rr_gate_mode","reasons","close_reason",
        "size_mode","bal_asset","notional","bal_pct","exit_ts"  # NEW
    ]
    if "exit_ts" not in row: row["exit_ts"] = ""
    if not os.path.exists(TRADES_CSV):
        with open(TRADES_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers); w.writeheader(); w.writerow(row); return
    with open(TRADES_CSV, "r", encoding="utf-8") as f:
        r = csv.DictReader(f); old_rows = list(r); old_headers = r.fieldnames or []
    if set(old_headers) == set(headers):
        with open(TRADES_CSV, "a", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers); w.writerow(row)
    else:
        old_rows.append(row); _rewrite_trades_csv(old_rows, pref_headers=headers)

# === NEW: export to app.py ===
def get_overview() -> Dict[str, Any]:
    """Pass-through for /api/overview import in app.py."""
    try:
        return _bn_get_overview()
    except Exception as e:
        logger.info("get_overview passthrough failed: %s", e)
        return {"balances": [], "positions": []}
