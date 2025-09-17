# helpers/signals.py — 3rd pass + checkpointing & EV logging (balance-% sizing, journaling/telemetry extended)
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
    from .utils import (
        LOG_DIR, gcs_enabled, gcs_append_csv_row, log_event,
        gcs_list, gcs_download_text, GCS_PREFIX,
        gcs_upload_file, gcs_download_file,
    )  # type: ignore
except Exception:  # pragma: no cover
    LOG_DIR = os.path.join(os.getcwd(), "logs")
    def gcs_enabled() -> bool: return False
    def gcs_append_csv_row(*args, **kwargs) -> None: return None
    def gcs_list(prefix: str): return []
    def gcs_download_text(path: str, encoding: str = "utf-8"): return ""
    GCS_PREFIX = os.getenv("GCS_PREFIX", "trading_bot")
    def gcs_upload_file(*args, **kwargs) -> bool: return False
    def gcs_download_file(*args, **kwargs) -> bool: return False
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

# --- predictor (router) ---
try:
    from .predictor_router import get_prediction  # type: ignore
except Exception:
    def get_prediction(payload, symbol=""):
        from .predictor import get_gemini_prediction  # type: ignore
        return get_gemini_prediction(payload, symbol=symbol)
from .predictor import should_predict  # type: ignore

# --- Binance client wrapper ---
try:
    from .binance_client import (
        get_overview as _bn_get_overview,
        cancel_open_orders, set_position_mode, set_margin_type, set_leverage,
        load_symbol_filters, ensure_min_notional, normalize_price_for_side,
        place_market_order, place_limit_order, place_bracket_orders,
        get_position, get_open_orders, get_last_price,
        replace_stop_loss_to_price, find_recent_exit_fill, find_recent_exit_trade,
        cancel_orders_by_type,
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
MIN_VOL_FRAC_ENV = float(os.getenv("MIN_VOL_FRAC", "0.0005"))

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

# Bracket reset toggle (default on)
BRACKETS_RESET_ON_FILL = str(os.getenv("BRACKETS_RESET_ON_FILL", "true")).lower() in ("1","true","yes")

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

# risk kill-switch (off by default)
MAX_DAILY_LOSS_USD = float(os.getenv("MAX_DAILY_LOSS_USD", "0"))  # 0=off
MAX_CONSEC_LOSSES  = int(os.getenv("MAX_CONSEC_LOSSES", "0"))     # 0=off
MAX_MDD_USD        = float(os.getenv("MAX_MDD_USD", "0"))          # 0=off

# post-only & fee-aware RR gate
ENTRY_POST_ONLY = str(os.getenv("ENTRY_POST_ONLY", "true")).lower() in ("1","true","yes")
RR_EVAL_WITH_FEES = str(os.getenv("RR_EVAL_WITH_FEES", "true")).lower() in ("1","true","yes")
FEE_MAKER_BPS = float(os.getenv("FEE_MAKER_BPS", "2.0"))
FEE_TAKER_BPS = float(os.getenv("FEE_TAKER_BPS", "4.0"))
# Economic viability: enforce minimum TP distance in bps (net)
MIN_TP_BPS_NET = float(os.getenv("MIN_TP_BPS_NET", "12.0"))

# shock guard & MTF & cooldown
SHOCK_BPS = float(os.getenv("SHOCK_BPS", "30.0"))
SHOCK_ATR_MULT = float(os.getenv("SHOCK_ATR_MULT", "1.8"))
MTF_ALIGN_ENABLED = str(os.getenv("MTF_ALIGN_ENABLED", "true")).lower() in ("1","true","yes")
MTF_RSI_LONG_MIN = float(os.getenv("MTF_RSI_LONG_MIN", "49"))
MTF_RSI_SHORT_MAX = float(os.getenv("MTF_RSI_SHORT_MAX", "51"))
ENTRY_COOLDOWN_MIN = int(os.getenv("ENTRY_COOLDOWN_MIN", "10"))

RR_GATE_MODE = os.getenv("RR_GATE_MODE", "expected").lower()  # worst | expected | best
MAKER_PROB_LOOKBACK = int(os.getenv("MAKER_PROB_LOOKBACK", "200"))

# Safer defaults (can still be opened via env)
EV_OVERRIDE_ENABLED = str(os.getenv("EV_OVERRIDE_ENABLED", "false")).lower() in ("1","true","yes")
EV_OVERRIDE_MIN_PROB = float(os.getenv("EV_OVERRIDE_MIN_PROB", "0.58"))
EV_OVERRIDE_MIN_PERC = float(os.getenv("EV_OVERRIDE_MIN_PERC", "0.0015"))
MTF_RELAX_WITH_EV   = str(os.getenv("MTF_RELAX_WITH_EV", "false")).lower() in ("1","true","yes")
OVR_SPREAD_MAX_BPS  = float(os.getenv("OVR_SPREAD_MAX_BPS", "1.8"))
OVR_RR_EXTRA        = float(os.getenv("OVR_RR_EXTRA", "0.05"))

# --- sizing mode (NEW) ---
SIZE_MODE = os.getenv("SIZE_MODE", "USDT").upper()              # USDT | BALANCE_PCT
RISK_USDT = float(os.getenv("RISK_USDT", "100"))                # legacy (SIZE_MODE=USDT)
RISK_BAL_PCT = float(os.getenv("RISK_BAL_PCT", "1.0"))          # % of wallet balance
SIZE_BAL_INCLUDE_UPNL = str(os.getenv("SIZE_BAL_INCLUDE_UPNL", "false")).lower() in ("1", "true", "yes")
SIZE_BAL_ASSET_OVERRIDE = os.getenv("SIZE_BAL_ASSET_OVERRIDE", "").upper().strip()

# --- Journal checkpoint paths (NEW) ---
_JOURNAL_LATEST = os.getenv("JOURNAL_LATEST_PATH", f"{GCS_PREFIX}/journal/latest/trades.csv")
_JOURNAL_DAILY_PREFIX = os.getenv("JOURNAL_DAILY_PREFIX", f"{GCS_PREFIX}/journal")
JOURNAL_SYNC_ON_START = str(os.getenv("JOURNAL_SYNC_ON_START","true")).lower() in ("1","true","yes")

# --- NEW EV gate & dynamic ATR levels ---
EV_MIN_PERC = float(os.getenv("EV_MIN_PERC", "0.0"))  # EV_perc < EV_MIN_PERC 이면 진입 금지
DYN_ATR_LEVELS = str(os.getenv("DYN_ATR_LEVELS", "true")).lower() in ("1","true","yes")
VOL_TRANQ = float(os.getenv("VOL_TRANQ_RATIO", "0.70"))   # 저변동 경계
VOL_TURB = float(os.getenv("VOL_TURB_RATIO", "1.30"))     # 고변동 경계
TP_FUDGE_TRANQ = float(os.getenv("TP_FUDGE_TRANQ", "1.10"))
SL_FUDGE_TRANQ = float(os.getenv("SL_FUDGE_TRANQ", "0.90"))
TP_FUDGE_TURB = float(os.getenv("TP_FUDGE_TURB", "0.90"))
SL_FUDGE_TURB = float(os.getenv("SL_FUDGE_TURB", "1.10"))

# === TP 최소폭(순이익 하한) 게이트 ===
def _tp_bps_gate(direction: str, entry: float, tp: float, spread_bps: float) -> Tuple[bool, str, Dict[str, float]]:
    """
    True -> 통과, False -> 차단.
    기준: TP_bps >= max(MIN_TP_BPS_NET, fee_roundtrip_exp_bps + 2*spread_bps + 2bps)
    - fee 기대값은 저널 기반 maker 확률로 혼합(코드 일관성 유지).
    """
    try:
        min_tp_bps_net = float(os.getenv("MIN_TP_BPS_NET", "12.0"))
    except Exception:
        min_tp_bps_net = 12.0

    e = float(entry); t = float(tp)
    if e <= 0 or t <= 0:
        return False, "tp_bps_invalid", {"tp_delta_bps": 0.0, "tp_threshold_bps": min_tp_bps_net, "fee_roundtrip_bps": 0.0,
                                          "maker_prob_est": float(_estimate_p_maker_from_journal()),
                                          "spread_bps": float(spread_bps), "min_tp_bps_net": float(min_tp_bps_net)}

    if str(direction).lower() == "long":
        tp_bps = (t - e) / max(1e-12, e) * 1e4
    else:
        tp_bps = (e - t) / max(1e-12, e) * 1e4

    # 기대 수수료(bps)
    p_maker = float(_estimate_p_maker_from_journal())
    maker = float(FEE_MAKER_BPS)
    taker = float(FEE_TAKER_BPS)
    entry_fee = p_maker*maker + (1.0 - p_maker)*taker
    tp_fee = maker if TP_ORDER_TYPE == "LIMIT" else taker
    fee_roundtrip = entry_fee + tp_fee

    try:
        sbps = float(spread_bps)
    except Exception:
        sbps = 0.0
    threshold = max(min_tp_bps_net, fee_roundtrip + 2.0*sbps + 2.0)

    ok = tp_bps >= threshold
    reason = ("" if ok else f"tp_bps_too_small(tp={tp_bps:.2f}bps<th={threshold:.2f}bps)")
    meta = {
        "tp_delta_bps": float(tp_bps),
        "tp_threshold_bps": float(threshold),
        "fee_roundtrip_bps": float(fee_roundtrip),
        "maker_prob_est": float(p_maker),
        "spread_bps": float(sbps),
        "min_tp_bps_net": float(min_tp_bps_net),
    }
    return ok, reason, meta
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

def _quantize_prob(p: float, decimals_env_primary: str = "PROB_DECIMALS_POST") -> float:
    """
    캘리브레이션 이후 게이트에 들어가는 확률을 지정 소수점 자리까지 확정.
    - PROB_DECIMALS_POST 가 없으면 PROB_DECIMALS, 그마저 없으면 2(=0.01) 사용
    """
    try:
        d = int(os.getenv(decimals_env_primary, os.getenv("PROB_DECIMALS", "2")))
    except Exception:
        d = 2
    try:
        return float(f"{float(p):.{d}f}")
    except Exception:
        return round(float(p), d)
    
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
    # Observability: distinguish missing vs mismatch while keeping conservative logic
    if not MTF_ALIGN_ENABLED or direction not in ("long","short"):
        return True, ""
    has1 = "RSI_1h" in extra
    has4 = "RSI_4h" in extra
    if not (has1 or has4):
        # No MTF context at all -> block and mark explicitly
        return False, "mtf_rsi_missing"
    r1h = float(extra.get("RSI_1h", 50.0)) if has1 else 50.0
    r4h = float(extra.get("RSI_4h", 50.0)) if has4 else 50.0
    if direction == "long":
        ok = (r1h >= MTF_RSI_LONG_MIN) and (r4h >= MTF_RSI_LONG_MIN if has4 else True)
    else:
        ok = (r1h <= MTF_RSI_SHORT_MAX) and (r4h <= MTF_RSI_SHORT_MAX if has4 else True)
    return ok, ("mtf_rsi_mismatch" if (has1 or has4) else "mtf_rsi_missing")

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
    if not _df_ok(ohlcv):
        ohlcv = add_indicators(None)
    last = ohlcv.iloc[-1] if _df_ok(ohlcv) else pd.Series({})
    ob = base.get("orderbook") if isinstance(base, dict) else None

    # --- MTF / extra ---
    mtf = fetch_mtf_raw(symbol)
    extra: Dict[str, Any] = {}
    for tf, df in mtf.items():
        ind = add_indicators(df)
        atr = float(compute_atr(df, window=14).iloc[-1]) if _df_ok(df) else 0.0
        key = {"1h": "_1h", "4h": "_4h", "1d": "_1d"}.get(tf, "")
        if _df_ok(ind):
            extra.update({
                f"RSI{key}": float(ind.get("RSI").iloc[-1]) if "RSI" in ind else 50.0,
                f"volatility{key}": float(ind.get("volatility").iloc[-1]) if "volatility" in ind else 0.0,
                f"SMA20{key}": float(ind.get("SMA_20").iloc[-1]) if "SMA_20" in ind else 0.0,
                f"ATR{key}": float(atr),
                f"relative_volume{key}": float(compute_relative_volume(df)) if _df_ok(df) else 1.0,
                f"recent_high{key}": float(df.get("high").tail(50).max()) if _df_ok(df) else 0.0,
                f"recent_low{key}": float(df.get("low").tail(50).min()) if _df_ok(df) else 0.0,
            })

    price_seq = compute_recent_price_sequence(ohlcv, n=10) if _df_ok(ohlcv) else [0.0] * 10
    atr5_series = compute_atr(ohlcv, window=14) if _df_ok(ohlcv) else None
    atr5 = float(atr5_series.iloc[-1]) if getattr(atr5_series, "size", 0) else 0.0

    sr5 = {
        "recent_high": float(ohlcv["high"].tail(50).max()) if _df_ok(ohlcv) else 0.0,
        "recent_low": float(ohlcv["low"].tail(50).min()) if _df_ok(ohlcv) else 0.0,
    }
    raw_stats = compute_orderbook_stats(ob) if isinstance(ob, dict) else {
        "imbalance": 0.0, "spread": 0.0, "microprice": 0.0, "mid": 0.0, "micro_dislocation_bps": 0.0
    }
    ob_stats = _ob_stats_to_dict(raw_stats)
    trend = compute_trend_filter(ohlcv) if _df_ok(ohlcv) else {"daily_uptrend": False, "trend_strength": 0.0}

    try:
        funding_pct = float(ohlcv["funding_rate_pct"].iloc[-1]) if _df_ok(ohlcv) and "funding_rate_pct" in ohlcv.columns else 0.0
    except Exception:
        funding_pct = 0.0

    entry = float(last.get("close", 0.0))
    extra_common = {
        "ATR_5m": float(atr5),
        "relative_volume_5m": float(compute_relative_volume(ohlcv)) if _df_ok(ohlcv) else 1.0,
        "recent_high_5m": sr5["recent_high"],
        "recent_low_5m": sr5["recent_low"],
        "orderbook_imbalance": float(ob_stats.get("imbalance", 0.0)),
        "orderbook_spread": float(ob_stats.get("spread", 0.0)),
        "microprice": float(ob_stats.get("microprice", 0.0)),
        "micro_dislocation_bps": float(ob_stats.get("micro_dislocation_bps", 0.0)),
        "funding_rate_pct": float(funding_pct),
    }
    # Merge MTF-derived features into extra for downstream checks (e.g., RSI_1h/RSI_4h)
    extra_common.update(extra)

    # ---- LLM용 브래킷 계산: tick 라운딩 + (선택) 표기 소수 고정 ----
    # normalize_price_for_side: SELL=올림, BUY=내림 (tickSize 준수)
    def _llm_px(px: float, exit_side: str) -> float:
        v = normalize_price_for_side(symbol, float(px), side=exit_side)
        dec = os.getenv("LLM_PRICE_DECIMALS", "")
        if dec.strip() != "":
            try:
                v = float(f"{v:.{int(dec)}f}")
            except Exception:
                pass
        return float(v)

    # 동적 ATR 보정(현재/중앙값 비율로 fudge) — 실행 로직과 동일한 파라미터 사용
    k_tp_env = float(os.getenv("ATR_MULT_TP", str(ATR_MULT_TP)))
    k_sl_env = float(os.getenv("ATR_MULT_SL", str(ATR_MULT_SL)))
    k_tp, k_sl = k_tp_env, k_sl_env
    try:
        if _df_ok(ohlcv):
            atr_series = compute_atr(ohlcv, window=14)
            cur_atr = float(atr_series.iloc[-1]) if len(atr_series) else 0.0
            med_atr = float(atr_series.tail(VOL_LOOKBACK).median()) if len(atr_series) else 0.0
            if DYN_ATR_LEVELS and cur_atr > 0 and med_atr > 0:
                ratio = cur_atr / med_atr
                if ratio >= VOL_TURB:
                    k_tp = k_tp_env * TP_FUDGE_TURB
                    k_sl = k_sl_env * SL_FUDGE_TURB
                elif ratio <= VOL_TRANQ:
                    k_tp = k_tp_env * TP_FUDGE_TRANQ
                    k_sl = k_sl_env * SL_FUDGE_TRANQ
    except Exception:
        k_tp, k_sl = k_tp_env, k_sl_env

    # LLM 입력용: long/short 양방향 브래킷(정규화)
    sr_high = float(extra_common["recent_high_5m"])  # type: ignore[index]
    sr_low  = float(extra_common["recent_low_5m"])   # type: ignore[index]

    def _mk_bracket(direction: str) -> Dict[str, float]:
        tp_raw, sl_raw = _tp_sl_with_sr_clamp(direction, entry, atr5, sr_high, sr_low,
                                              llm_support=None, llm_resistance=None,
                                              k_tp=k_tp, k_sl=k_sl)
        exit_side = "SELL" if direction == "long" else "BUY"
        tp_q = _llm_px(tp_raw, exit_side)
        sl_q = _llm_px(sl_raw, exit_side)
        if direction == "long":
            d_tp_bps = (tp_q - entry) / max(1e-12, entry) * 1e4
            d_sl_bps = (entry - sl_q) / max(1e-12, entry) * 1e4
        else:
            d_tp_bps = (entry - tp_q) / max(1e-12, entry) * 1e4
            d_sl_bps = (sl_q - entry) / max(1e-12, entry) * 1e4
        rr_net = _rr_with_fee_mode(direction, entry, tp_q, sl_q)
        return {
            "tp": float(tp_q),
            "sl": float(sl_q),
            "tp_delta_bps": float(d_tp_bps),
            "sl_delta_bps": float(d_sl_bps),
            "rr_net": float(rr_net),
        }

    brackets = {
        "entry": float(entry),
        "long": _mk_bracket("long") if entry > 0 and atr5 > 0 else {"tp": 0.0, "sl": 0.0},
        "short": _mk_bracket("short") if entry > 0 and atr5 > 0 else {"tp": 0.0, "sl": 0.0},
    }

    # legacy bracket for backward-compat with existing code paths
    legacy_bracket = {
        "entry": float(entry),
        "k_tp": float(k_tp),
        "k_sl": float(k_sl),
        "atr": float(atr5),
        "sr_high": float(sr_high),
        "sr_low": float(sr_low),
        "long": {"tp": float((brackets.get("long") or {}).get("tp", 0.0)), "sl": float((brackets.get("long") or {}).get("sl", 0.0))},
        "short": {"tp": float((brackets.get("short") or {}).get("tp", 0.0)), "sl": float((brackets.get("short") or {}).get("sl", 0.0))},
    }

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
        "extra": extra_common,
        "times": base.get("times", {}),
        "price_sequence": price_seq,
        "sr_levels": sr5,
        "relative_volume": float(compute_relative_volume(ohlcv)) if _df_ok(ohlcv) else 1.0,
        "trend_filter": trend,
        "horizon_min": HORIZON_MIN,
        # LLM이 사용할 결정적 수치 피처: 브래킷
        "brackets": brackets,
        # keep legacy key for compatibility
        "bracket": legacy_bracket,
        # informational
        "fees": {"maker_bps": float(FEE_MAKER_BPS), "taker_bps": float(FEE_TAKER_BPS)},
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

def preview_size(symbol: str, entry: float, sl: float, risk_scalar: float = 1.0) -> Dict[str, Any]:
    qty, meta = _compute_size(symbol, entry, sl, risk_scalar)
    return {"qty": float(qty), **meta}

# ---------------------------------
# Journal checkpoint helpers (NEW)
# ---------------------------------
def _journal_restore_from_gcs_if_needed() -> bool:
    """
    로컬 trades.csv가 없거나 너무 작으면 GCS 최신본으로 복원.
    """
    try:
        need = (not os.path.exists(TRADES_CSV)) or (os.path.getsize(TRADES_CSV) < 256)
    except Exception:
        need = True
    if (not need) or (not gcs_enabled()):
        return False
    ok = gcs_download_file(_JOURNAL_LATEST, TRADES_CSV)
    if ok:
        log_event("journal.restore", source=_JOURNAL_LATEST, local=TRADES_CSV)
    return ok

def _journal_backup_to_gcs(tag: str = "auto") -> bool:
    """
    로컬 trades.csv 전체본을 GCS latest + 일자 보관 경로로 업로드.
    """
    if (not gcs_enabled()) or (not os.path.exists(TRADES_CSV)):
        return False
    from datetime import datetime, timezone
    now = datetime.now(tz=timezone.utc)
    date_dir = now.strftime("%Y%m%d"); time_tag = now.strftime("%H%M%S")
    daily_path = f"{_JOURNAL_DAILY_PREFIX}/{date_dir}/trades_{time_tag}_{tag}.csv"
    ok1 = gcs_upload_file(TRADES_CSV, _JOURNAL_LATEST, content_type="text/csv")
    ok2 = gcs_upload_file(TRADES_CSV, daily_path, content_type="text/csv")
    if ok1 or ok2:
        log_event("journal.backup", latest=_JOURNAL_LATEST, daily=daily_path, tag=tag)
    return bool(ok1 or ok2)

def journal_sync(mode: str = "backup") -> Dict[str, Any]:
    if mode == "restore":
        ok = _journal_restore_from_gcs_if_needed()
        return {"action":"restore","ok":bool(ok)}
    ok = _journal_backup_to_gcs(tag="cron")
    return {"action":"backup","ok":bool(ok)}

from pathlib import Path
from zoneinfo import ZoneInfo

def _journal_headers() -> list[str]:
    # _journal_append_open() header order to keep downstream safe
    return [
        "timestamp","symbol","side","qty","entry","entry_intent","tp","sl","exit","pnl","status","id",
        "prob","prob_raw","prob_cal","rr","entry_maker","tp_type","mode","reprices","used_market_fallback","post_only",
        "spread_bps","atr_now","funding_pct","maker_prob_est","rr_gate_mode","reasons","close_reason",
        "size_mode","bal_asset","notional","bal_pct","exit_ts",
        "ev_perc","ev_usd","ev_ex_ante_perc","ev_ex_ante_usd"
    ]

def _any_open_rows_in_journal() -> bool:
    try:
        import csv, os
        if not os.path.exists(TRADES_CSV):
            return False
        with open(TRADES_CSV, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if str(row.get("status","")) .strip().lower() == "open":
                    return True
    except Exception:
        return False
    return False

def _dt_start_of_today_utc() -> "datetime":
    # Respect TZ env (default UTC). e.g., Asia/Seoul
    tz = os.getenv("TZ", "UTC")
    local = datetime.now(ZoneInfo(tz))
    sod_local = local.replace(hour=0, minute=0, second=0, microsecond=0)
    return sod_local.astimezone(timezone.utc)

def _risk_circuit_tripped() -> tuple[bool, str, dict]:
    """
    일일 손익, 연속 손실, 누적 MDD 기준으로 회로차단.
    활성화: 각 임계값이 >0일 때만 검사.
    Returns: (tripped, reason, stats)
    """
    import csv
    stats = {"daily_pnl": 0.0, "consec_losses": 0, "mdd": 0.0}
    try:
        if not os.path.exists(TRADES_CSV):
            return (False, "", stats)
        with open(TRADES_CSV, "r", encoding="utf-8") as f:
            rows = list(csv.DictReader(f))

        # 1) Daily PnL
        try:
            sod_utc = _dt_start_of_today_utc()
            dp = 0.0
            for row in rows:
                ts = _parse_iso(row.get("timestamp", ""))
                if not ts or ts < sod_utc:
                    continue
                if str(row.get("status", "")).lower() == "open":
                    continue
                dp += float(row.get("pnl", 0.0) or 0.0)
            stats["daily_pnl"] = float(dp)
            if MAX_DAILY_LOSS_USD > 0 and dp <= -abs(MAX_DAILY_LOSS_USD):
                return True, f"daily_loss({dp:.2f}≤-{abs(MAX_DAILY_LOSS_USD):.2f})", stats
        except Exception:
            pass

        # 2) 연속 손실
        try:
            consec = 0
            for row in reversed(rows):
                if str(row.get("status", "")).lower() == "open":
                    continue
                pnl = float(row.get("pnl", 0.0) or 0.0)
                if pnl <= 0:
                    consec += 1
                else:
                    break
            stats["consec_losses"] = int(consec)
            if MAX_CONSEC_LOSSES > 0 and consec >= int(MAX_CONSEC_LOSSES):
                return True, f"consec_losses({consec}≥{MAX_CONSEC_LOSSES})", stats
        except Exception:
            pass

        # 3) 최대 낙폭(MDD)
        try:
            s = 0.0
            peak = 0.0
            mdd = 0.0
            for row in rows:
                if str(row.get("status", "")).lower() == "open":
                    continue
                s += float(row.get("pnl", 0.0) or 0.0)
                peak = max(peak, s)
                mdd = max(mdd, peak - s)
            stats["mdd"] = float(mdd)
            if MAX_MDD_USD > 0 and mdd >= abs(MAX_MDD_USD):
                return True, f"max_drawdown({mdd:.2f}≥{abs(MAX_MDD_USD):.2f})", stats
        except Exception:
            pass

        return (False, "", stats)
    except Exception as e:
        logger.info("risk_circuit check failed: %s", e)
        return (False, "", stats)

def _parse_keep_from(keep_from: Optional[str]) -> Optional["datetime"]:
    if not keep_from:
        return None
    k = keep_from.strip().lower()
    if k in ("today", "오늘"):
        return _dt_start_of_today_utc()
    # YYYY-MM-DD or ISO-like
    try:
        # date-only -> interpret as local TZ midnight then to UTC
        if len(keep_from.strip()) == 10:
            tz = os.getenv("TZ", "UTC")
            local = datetime.fromisoformat(keep_from.strip() + "T00:00:00")
            local = local.replace(tzinfo=ZoneInfo(tz))
            return local.astimezone(timezone.utc)
        # otherwise parse ISO
        dt = datetime.fromisoformat(keep_from.replace("Z", "+00:00"))
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        else:
            dt = dt.astimezone(timezone.utc)
        return dt
    except Exception:
        return None

def journal_reset(
    *,
    confirm: bool = False,
    backup: bool = True,
    require_no_open: bool = True,
    keep_from: Optional[str] = None
) -> Dict[str, Any]:
    """
    Journal reset/trim:
      - confirm=False -> dry-run (no changes)
      - keep_from=None -> full reset (headers only)
      - keep_from='today' or 'YYYY-MM-DD' -> keep rows since that (local TZ 00:00) in UTC
      - backup=True -> pre-backup to GCS latest and daily dir
      - require_no_open=True -> block if any 'open' rows present
    Returns: {"action": str, "path": TRADES_CSV, "backup_ok": bool, "gcs_latest_ok": bool, "kept": int, "dropped": int}
    """
    p = Path(TRADES_CSV)
    exists = p.exists()
    size = int(p.stat().st_size) if exists else 0
    has_open = _any_open_rows_in_journal()

    cut_utc = _parse_keep_from(keep_from)

    plan = {
        "exists": bool(exists),
        "size_bytes": size,
        "open_rows_present": bool(has_open),
        "keep_from": (cut_utc.isoformat() if cut_utc else None),
        "gcs_enabled": bool(gcs_enabled()),
        "path": str(p),
    }

    if not confirm:
        return {"action": "reset_dryrun", **plan}
    if require_no_open and has_open:
        return {"action": "blocked", "reason": "open_rows_present", **plan}

    backed = False
    if backup and exists:
        try:
            backed = _journal_backup_to_gcs(tag="pre_reset")
        except Exception:
            backed = False

    kept_rows: list = []
    dropped = 0
    headers = _journal_headers()

    if cut_utc is not None and exists:
        import csv
        with open(TRADES_CSV, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                ts = _parse_iso(row.get("timestamp",""))
                if ts is None or ts.tzinfo is None:
                    dropped += 1
                    continue
                if ts >= cut_utc:
                    kept_rows.append(row)
                else:
                    dropped += 1
        _rewrite_trades_csv(kept_rows, pref_headers=headers)
        log_event("journal.trim", kept=len(kept_rows), dropped=int(dropped), since=cut_utc.isoformat())
    else:
        import csv, os
        os.makedirs(os.path.dirname(TRADES_CSV), exist_ok=True)
        with open(TRADES_CSV, "w", newline="", encoding="utf-8") as f:
            w = csv.DictWriter(f, fieldnames=headers)
            w.writeheader()
        kept_rows = []
        dropped = -1

    gcs_latest_ok = False
    try:
        gcs_latest_ok = _journal_backup_to_gcs(tag=("reset_trim" if cut_utc else "reset_empty"))
    except Exception:
        gcs_latest_ok = False

    out = {
        "action": ("reset_trim" if cut_utc else "reset_empty"),
        "backup_ok": bool(backed),
        "gcs_latest_ok": bool(gcs_latest_ok),
        "kept": len(kept_rows),
        "dropped": int(dropped) if dropped >= 0 else None,
        "path": str(p),
    }
    log_event("journal.reset", **out)
    return out


# ---------------------------------
# Signal generation
# ---------------------------------# helpers/signals.py — replace this whole function# helpers/signals.py — generate_signal() REPLACE WHOLE FUNCTION
def generate_signal(symbol: str) -> Optional[Dict[str, Any]]:
    payload, ohlcv, ob = _build_payload(symbol)
    spread_bps_gate = float((payload.get("extra") or {}).get("orderbook_spread", 0.0))
    proceed_basic = should_predict(payload, min_vol_frac_env="MIN_VOL_FRAC") and _spread_ok(spread_bps_gate)
    # pre-gate diagnostics (observability): capture current vol/spread
    vol_last = float((payload.get("entry_5m") or {}).get("volatility", 0.0))
    spread_bps = float((payload.get("extra") or {}).get("orderbook_spread", 0.0))
    dir_hint, _ = _rule_backup(ohlcv, payload.get("trend_filter") or {})
    atr5 = float((payload.get("extra") or {}).get("ATR_5m") or 0.0)

    cd_active, cd_left = _cooldown_active(symbol)
    if cd_active:
        return {"symbol": symbol, "action":"hold","direction":"hold",
                "entry": float((payload.get("entry_5m") or {}).get("close") or 0.0),
                "tp":0.0,"sl":0.0,"prob":0.5,"risk_ok":False,"rr":0.0,
                "reason": f"pre_gate_cooldown({cd_left}m_left)"}

    sg_long, bpsL, multL, _ = _shock_guard_block("long", ohlcv, atr5)
    sg_short, bpsS, multS, _ = _shock_guard_block("short", ohlcv, atr5)
    if (sg_long or sg_short):
        candle_up = float(ohlcv.iloc[-1]["close"]) - float(ohlcv.iloc[-1]["open"]) > 0 if _df_ok(ohlcv) else False
        shock_dir = "long" if candle_up else "short"
        if not (dir_hint in ("long","short") and dir_hint == shock_dir):
            bps = max(bpsL, bpsS); mult = max(multL, multS)
            return {"symbol":symbol,"action":"hold","direction":"hold",
                    "entry": float((payload.get("entry_5m") or {}).get("close") or 0.0),
                    "tp":0.0,"sl":0.0,"prob":0.5,"risk_ok":False,"rr":0.0,
                    "reason": f"pre_gate_shock({bps:.1f}bps,{mult:.2f}ATR)"}

    # ---- LLM or rule-backup path
    if not proceed_basic:
        trend = payload.get("trend_filter") or {}
        direction_rb, prob_rb = _rule_backup(ohlcv, trend)
        if direction_rb in ("long","short"):
            direction, prob_raw = direction_rb, float(prob_rb)
        else:
            reason = (
                f"pre_gate_block(vol={vol_last:.6f}<{MIN_VOL_FRAC_ENV:.6f},"
                f"spread_bps={spread_bps:.2f}<=max({MAX_SPREAD_BPS:.2f}))"
            )
            return {"symbol":symbol,"action":"hold","direction":"hold",
                    "entry": float((payload.get("entry_5m") or {}).get("close") or 0.0),
                    "tp":0.0,"sl":0.0,"prob":0.5,"risk_ok":False,"rr":0.0,"reason": reason}
        llm_support = None; llm_resistance = None
    else:
        llm_decision = get_prediction(payload, symbol=symbol)
        if not llm_decision:
            log_event("predictor.timeout", symbol=symbol, reason="no_prediction")
            return None
        if not isinstance(llm_decision, dict):
            log_event("predictor.timeout", symbol=symbol, reason="invalid_prediction_schema")
            return None
        if llm_decision.get("error"):
            log_event("predictor.timeout", symbol=symbol, reason=str(llm_decision.get("error")))
            return None
        direction = str(llm_decision.get("direction") or "").lower()
        prob_raw = float(llm_decision.get("prob", 0.0) or 0.0)
        llm_support = llm_decision.get("support")
        llm_resistance = llm_decision.get("resistance")

    # ---- calibration & quantize
    prob_cal = float(calibrate_prob(prob_raw)) if USE_CALIBRATED_PROB else float(prob_raw)
    prob = _quantize_prob(prob_cal)

    entry = float((payload.get("entry_5m") or {}).get("close") or 0.0)
    extra = payload.get("extra") or {}
    br = payload.get("bracket") or {}
    spread_bps = float(extra.get("orderbook_spread") or 0.0)

    if direction not in ("long","short") or entry <= 0:
        log_event("signal.decision", symbol=symbol, direction="hold", prob=prob, entry=entry, tp=0.0, sl=0.0, rr=0.0, risk_ok=False)
        return {"symbol":symbol,"action":"hold","direction":"hold","entry":entry,"tp":0.0,"sl":0.0,"prob":prob,"risk_ok":False,"rr":0.0,"reason":"invalid_direction_or_entry"}

    # ---- dynamic ATR levels (tranq/turb) same as execution side
    k_tp_env = float(os.getenv("ATR_MULT_TP", str(ATR_MULT_TP)))
    k_sl_env = float(os.getenv("ATR_MULT_SL", str(ATR_MULT_SL)))
    k_tp, k_sl = k_tp_env, k_sl_env
    cur_atr, med_atr = 0.0, 0.0
    try:
        if _df_ok(ohlcv):
            atr_series = compute_atr(ohlcv, window=14)
            cur_atr = float(atr_series.iloc[-1]) if len(atr_series) else 0.0
            med_atr = float(atr_series.tail(VOL_LOOKBACK).median()) if len(atr_series) else 0.0
            if DYN_ATR_LEVELS and cur_atr > 0 and med_atr > 0:
                ratio = cur_atr / med_atr
                if ratio >= VOL_TURB:
                    k_tp = k_tp_env * TP_FUDGE_TURB
                    k_sl = k_sl_env * SL_FUDGE_TURB
                elif ratio <= VOL_TRANQ:
                    k_tp = k_tp_env * TP_FUDGE_TRANQ
                    k_sl = k_sl_env * SL_FUDGE_TRANQ
    except Exception:
        k_tp, k_sl = k_tp_env, k_sl_env

    sr_high = float(extra.get("recent_high_5m") or 0.0)
    sr_low  = float(extra.get("recent_low_5m") or 0.0)
    tp, sl = _tp_sl_with_sr_clamp(
        direction, entry, float(extra.get("ATR_5m") or 0.0),
        sr_high, sr_low, llm_support, llm_resistance,
        k_tp=k_tp, k_sl=k_sl
    )

    # align with explicit bracket if present
    try:
        if direction == "long" and isinstance(br.get("long"), dict):
            tp = float((br.get("long") or {}).get("tp") or tp)
            sl = float((br.get("long") or {}).get("sl") or sl)
        elif direction == "short" and isinstance(br.get("short"), dict):
            tp = float((br.get("short") or {}).get("tp") or tp)
            sl = float((br.get("short") or {}).get("sl") or sl)
    except Exception:
        pass

    rr_net = _rr_with_fee_mode(direction, entry, tp, sl)
    rr_req = RR_MIN_HIGH_PROB if prob >= PROB_RELAX_THRESHOLD else RR_MIN

    # ---- MTF check with numeric exposure
    r1h = float(extra.get("RSI_1h", 50.0))
    r4h = float(extra.get("RSI_4h", 50.0))
    mtf_ok, mtf_reason = _mtf_align_ok(direction, extra)

    reasons: List[str] = []
    if prob < MIN_PROB: reasons.append("prob_below_threshold")
    if not _spread_ok(spread_bps): reasons.append(f"wide_spread({spread_bps:.2f}bps)")
    if not mtf_ok:
        reasons.append(f"{mtf_reason}(r1h={r1h:.1f},r4h={r4h:.1f},long_min={MTF_RSI_LONG_MIN:.0f},short_max={MTF_RSI_SHORT_MAX:.0f})")
    sg_block, sg_bps, sg_mult, sg_reason = _shock_guard_block(direction, ohlcv, float(extra.get("ATR_5m") or 0.0))
    if sg_block: reasons.append(sg_reason)
    cd_active2, cd_left2 = _cooldown_active(symbol)
    if cd_active2: reasons.append(f"entry_cooldown({cd_left2}m_left)")
    if rr_net <= 0 or rr_net < rr_req: reasons.append(f"rr_net_below_min({rr_net:.2f}<{rr_req:.2f})")

    # ---- EV gate
    ev_perc = _compute_ev_perc(prob, direction, entry, tp, sl)
    if ev_perc < EV_MIN_PERC:
        reasons.append(f"ev_below_threshold({ev_perc:.4f}<{EV_MIN_PERC:.4f})")

    # ---- Economic viability gate (expected fees + 2*spread + buffer)
    meta_tp: Dict[str, float] = {"tp_delta_bps": 0.0, "tp_threshold_bps": 0.0, "fee_roundtrip_bps": 0.0}
    try:
        ok_tp, r_tp, meta_tp = _tp_bps_gate(direction, entry, tp, spread_bps)
        if not ok_tp and r_tp:
            reasons.append(r_tp)
    except Exception:
        pass

    # volatility-weighted sizing (risk_scalar)
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

    # --- EV-based override (NEW) ---
    if EV_OVERRIDE_ENABLED:
        override_ok = (
            (prob < MIN_PROB) and (prob >= EV_OVERRIDE_MIN_PROB) and
            (rr_net >= rr_req) and (ev_perc >= EV_OVERRIDE_MIN_PERC) and
            _spread_ok(spread_bps) and (not sg_block) and (not cd_active2)
        )
        if override_ok:
            # 확률 컷만 제거(+선택적으로 MTF 불일치도 제거)
            reasons = [r for r in reasons if r != "prob_below_threshold"]
            if MTF_RELAX_WITH_EV:
                # our reasons include details like f"{mtf_reason}(...)"; relax by prefix match
                reasons = [r for r in reasons if not (str(mtf_reason) and str(r).startswith(str(mtf_reason)))]
            log_event("signal.override_ev",
                      symbol=symbol, prob=float(prob), rr=float(rr_net),
                      ev_perc=float(ev_perc), reasons=";".join(reasons) or "ok")
    # Enforce stricter EV override guard
    override_ok_strict = (
        EV_OVERRIDE_ENABLED
        and (prob < MIN_PROB) and (prob >= EV_OVERRIDE_MIN_PROB)
        and (ev_perc >= EV_OVERRIDE_MIN_PERC)
        and (rr_net >= (rr_req + OVR_RR_EXTRA))
        and (spread_bps <= min(MAX_SPREAD_BPS, OVR_SPREAD_MAX_BPS))
        and (not sg_block)
        and (not cd_active2)
        and mtf_ok
    )
    if (prob < MIN_PROB) and (not override_ok_strict):
        if "prob_below_threshold" not in reasons:
            reasons.append("prob_below_threshold")
    risk_ok = (len(reasons) == 0)

    # gate log with MTF numeric fields (observability)
    log_event("signal.gate", symbol=symbol, direction=direction, prob=float(prob), spread_bps=float(spread_bps),
              rr=float(rr_net), rr_req=float(rr_req), rr_mode=RR_GATE_MODE, ev_perc=float(ev_perc),
              rsi_1h=float(r1h), rsi_4h=float(r4h),
              reasons=";".join(reasons) if reasons else "ok")

    telemetry = {
        "spread_bps": float(spread_bps),
        "atr_now": float(extra.get("ATR_5m") or 0.0),
        "funding_pct": float(extra.get("funding_rate_pct") or 0.0),
        "maker_prob_est": _estimate_p_maker_from_journal(),
        "rr_gate_mode": RR_GATE_MODE,
        "sizing_mode": SIZE_MODE,
        "ev_perc": float(ev_perc),
        "k_tp": float((br.get("k_tp") if isinstance(br, dict) else None) or k_tp),
        "k_sl": float((br.get("k_sl") if isinstance(br, dict) else None) or k_sl),
        "atr_ratio": (cur_atr/med_atr if (cur_atr>0 and med_atr>0) else 0.0),
        "rsi_1h": float(r1h),
        "rsi_4h": float(r4h),
        "mtf_ok": bool(mtf_ok),
        "tp_delta_bps": float(meta_tp.get("tp_delta_bps", 0.0)),
        "tp_threshold_bps": float(meta_tp.get("tp_threshold_bps", 0.0)),
        "fee_roundtrip_bps": float(meta_tp.get("fee_roundtrip_bps", 0.0)),
    }

    out = {
        "symbol": symbol,
        "action": "enter" if risk_ok else "hold",
        "direction": direction,
        "entry": float(entry),
        "tp": float(tp),
        "sl": float(sl),
        "prob": float(prob),
        "prob_raw": float(prob_raw),
        "prob_cal": float(prob),
        "rr": float(rr_net),
        "risk_ok": bool(risk_ok),
        "reason": "ok" if risk_ok else ";".join(reasons) or "no_trade_conditions",
        "result": {
            "direction": direction, "entry": float(entry), "tp": float(tp), "sl": float(sl),
            "prob": float(prob), "prob_raw": float(prob_raw), "prob_cal": float(prob),
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

def _reset_brackets_old(symbol: str, side: str, tp: float, sl: float) -> Dict[str, Any]:
    """
    기존 TP/SL(RO) 전부 취소 후, '현재 포지션 잔고' 전량 기준으로 브래킷(TP/SL) 1쌍만 재배치.
    side: 엔트리 방향("BUY"/"SELL") 그대로 전달.
    """
    # 1) 기존 브래킷류 취소
    try:
        cancel_orders_by_type(symbol, ["TAKE_PROFIT", "TAKE_PROFIT_MARKET", "STOP", "STOP_MARKET"])
        log_event("brackets.reset.cancelled", symbol=symbol)
    except Exception as e:
        logger.info("brackets cancel failed for %s: %s", symbol, e)

    # 2) 현재 포지션 잔고 조회 → 전량
    qty = 0.0
    try:
        pos = get_position(symbol) or {}
        amt = float(pos.get("positionAmt") or pos.get("positionAmount") or 0.0)
        qty = abs(amt)
    except Exception:
        qty = 0.0

    if qty <= 1e-12:
        log_event("brackets.reset.skip", symbol=symbol, reason="no_position")
        return {"take_profit": None, "stop_loss": None, "skipped": True}

    # 3) 브래킷 1쌍 재배치
    out = place_bracket_orders(symbol, side, qty, take_profit=float(tp), stop_loss=float(sl))
    log_event("brackets.reset.placed", symbol=symbol, qty=float(qty), tp=float(tp), sl=float(sl))
    return out if isinstance(out, dict) else {"raw": out}

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
            if BRACKETS_RESET_ON_FILL:
                # 전량 기준 1쌍만 배치
                b = _reset_brackets(symbol, side, tp, sl)
                if isinstance(b, dict) and b.get("skipped"):
                    brackets = place_bracket_orders(symbol, side, quantity=float(filled), take_profit=float(tp), stop_loss=float(sl))
                    log_event("brackets.reset.fallback_qty", symbol=symbol, qty=float(filled), tp=float(tp), sl=float(sl))
                else:
                    brackets = b
            else:
                # 기존 동작(추가 누적) 유지 옵션
                brackets = place_bracket_orders(symbol, side, quantity=float(filled), take_profit=float(tp), stop_loss=float(sl))
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

def _close_position_market_old(symbol: str) -> Optional[dict]:
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

def _journal_has_row_id(row_id: Optional[str]) -> bool:
    if not row_id: return False
    try:
        import csv
        if not os.path.exists(TRADES_CSV): return False
        with open(TRADES_CSV, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if str(row.get("id","")) == str(row_id):
                    return True
    except Exception:
        return False
    return False

def _journal_status_by_id(row_id: Optional[str]) -> Optional[str]:
    """해당 id의 현행 status(open/closed/None)를 반환."""
    if not row_id: return None
    try:
        import csv
        if not os.path.exists(TRADES_CSV): return None
        with open(TRADES_CSV, "r", encoding="utf-8") as f:
            r = csv.DictReader(f)
            for row in r:
                if str(row.get("id","")) == str(row_id):
                    st = str(row.get("status","")).lower()
                    return st
    except Exception:
        return None
    return None

def _reconcile_open_from_gcs(symbol: str, max_scan: int = 500) -> bool:
    """
    GCS trades/ 스냅샷에서 최신 'open' 행(해당 심볼)을 찾아 로컬 trades.csv에 복구.
    """
    if not gcs_enabled():
        return False
    try:
        prefix = f"{GCS_PREFIX}/trades/"
        names = [n for n in (gcs_list(prefix) or []) if n.endswith(".csv")]
        if not names:
            return False
        names.sort(reverse=True)  # 최신 우선
        import csv, io
        scanned = 0
        for name in names:
            if "/trades_close/" in name:
                continue
            scanned += 1
            if scanned > max_scan:
                break
            text = gcs_download_text(name) or ""
            if not text:
                continue
            try:
                row = next(csv.DictReader(io.StringIO(text), skipinitialspace=True), None)
            except Exception:
                row = None
            if not row:
                continue
            if str(row.get("symbol","")).upper() != symbol.upper():
                continue
            if str(row.get("status","")).lower() != "open":
                continue
            if _journal_has_row_id(row.get("id")):
                st = _journal_status_by_id(row.get("id"))
                if st == "open":
                    log_event("reconcile.skip", symbol=symbol, reason="already_present_open", id=row.get("id"))
                    return True
                else:
                    # 이미 로컬에 존재하되 open이 아님(=close 처리됨). 복구 불필요이며 정산 대상도 아님.
                    log_event("reconcile.skip", symbol=symbol, reason="already_present_but_closed", id=row.get("id"))
                    return False
            _journal_append_open(row)
            log_event("reconcile.gcs_open_restored", symbol=symbol, source=name, id=row.get("id"))
            return True
        return False
    except Exception as e:
        logger.info("reconcile_from_gcs failed for %s: %s", symbol, e)
        return False

def _atomic_write(path: str, data: str) -> None:
    from pathlib import Path
    import tempfile
    dst = Path(path)
    dst.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.NamedTemporaryFile("w", delete=False, encoding="utf-8", dir=str(dst.parent)) as tf:
        tf.write(data)
        tmp = tf.name
    Path(tmp).replace(dst)

def _lock_file(path: str):
    """문맥 관리자: POSIX(fcntl)/Windows(msvcrt) 파일락. 실패 시 no-op."""
    try:
        import fcntl  # type: ignore
        class _Lock:
            def __init__(self, p): self.p = p; self.f = None
            def __enter__(self):
                self.f = open(self.p, "a+", encoding="utf-8")
                fcntl.flock(self.f.fileno(), fcntl.LOCK_EX)
                self.f.seek(0)
                return self.f
            def __exit__(self, exc_type, exc, tb):
                try: fcntl.flock(self.f.fileno(), fcntl.LOCK_UN)
                finally: self.f.close()
        return _Lock(path)
    except Exception:
        try:
            import msvcrt  # type: ignore
        except Exception:
            msvcrt = None  # type: ignore
        class _Lock:
            def __init__(self, p): self.p = p; self.f = None
            def __enter__(self):
                self.f = open(self.p, "a+", encoding="utf-8")
                if msvcrt is not None:
                    try: msvcrt.locking(self.f.fileno(), msvcrt.LK_LOCK, 1)
                    except Exception: pass
                self.f.seek(0)
                return self.f
            def __exit__(self, exc_type, exc, tb):
                if msvcrt is not None:
                    try: msvcrt.locking(self.f.fileno(), msvcrt.LK_UNLCK, 1)
                    except Exception: pass
                self.f.close()
        return _Lock(path)

def _rewrite_trades_csv(rows: list, pref_headers: Optional[list] = None) -> None:
    """
    파일락 하에서 CSV 전체를 원자적으로 재기록한다.
    """
    keys = set()
    for r in rows: keys.update(r.keys())
    base = [
        "timestamp","symbol","side","qty","entry","entry_intent","tp","sl",
        "exit","pnl","status","id","prob","rr","entry_maker","tp_type","mode",
        "reprices","used_market_fallback","post_only","spread_bps","atr_now",
        "funding_pct","maker_prob_est","rr_gate_mode","reasons","close_reason",
        "size_mode","bal_asset","notional","bal_pct","exit_ts",
        "prob_raw","prob_cal","ev_perc","ev_usd","ev_ex_ante_perc","ev_ex_ante_usd"
    ]
    headers = [k for k in (pref_headers or base) if k in keys] + [k for k in sorted(keys) if k not in (pref_headers or base)]
    import csv, io
    buf = io.StringIO()
    w = csv.DictWriter(buf, fieldnames=headers)
    w.writeheader()
    for r in rows: w.writerow(r)
    text = buf.getvalue()
    with _lock_file(TRADES_CSV):
        _atomic_write(TRADES_CSV, text)
# helpers/signals.py — replace _journal_close_last() body with the following edits near 'r.update({...})'
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

        # NEW: if reason=='closed', infer TP/SL by entry vs exit
        final_reason = reason
        if final_reason == "closed":
            if side.lower() == "long":
                final_reason = "closed_tp" if float(exit_price) >= entry else "closed_sl"
            else:
                final_reason = "closed_tp" if float(exit_price) <= entry else "closed_sl"

        now_iso = _now_utc().isoformat()
        r.update({
            "exit": f"{float(exit_price):.10f}",
            "pnl": f"{pnl:.10f}",
            "status": final_reason,        # <- changed
            "close_reason": final_reason,  # <- changed
            "exit_ts": now_iso,
        })
        rows[idx] = r
        _rewrite_trades_csv(rows)
        log_event("journal.close", symbol=symbol, exit=float(exit_price), pnl=float(pnl), reason=final_reason)
        if gcs_enabled():
            try: gcs_append_csv_row("trades_close", list(r.keys()), r)
            except Exception: pass
            try: _journal_backup_to_gcs(tag=final_reason)
            except Exception: pass
        return True
    except Exception as e:
        logger.info("journal close failed: %s", e); return False

def _find_recent_exit(symbol: str, since_ms: int, back_ms: Optional[int]) -> Optional[Dict[str, Any]]:
    """
    1차: 주문 히스토리(find_recent_exit_fill) → 2차: 체결 이력(find_recent_exit_trade) 폴백
    """
    try:
        info = find_recent_exit_fill(symbol, since_ms=int(since_ms), back_ms=back_ms)
        if info:
            return info
    except Exception:
        pass
    try:
        return find_recent_exit_trade(symbol, since_ms=int(since_ms), back_ms=back_ms)
    except Exception:
        return None

def _settle_by_orders(symbol: str) -> bool:
    idx, ts_ms, rows = _last_open_row_index_and_ts(symbol)
    if idx is None or ts_ms is None:
        restored = _reconcile_open_from_gcs(symbol)
        if restored:
            idx, ts_ms, rows = _last_open_row_index_and_ts(symbol)
        if idx is None or ts_ms is None:
            log_event("settle.skip", symbol=symbol, reason="no_open_row")
            return False
    # 🔽 명시적으로 back_ms 적용 (기본 3시간 = 10,800,000ms 권장)
    try:
        back_ms = int(os.getenv("EXIT_SEARCH_BACK_MS", "10800000"))
    except Exception:
        back_ms = 10800000
    info = _find_recent_exit(symbol, since_ms=int(ts_ms), back_ms=back_ms)
    if not info or not info.get("price"):
        log_event("settle.no_exit_found", symbol=symbol, since_ms=int(ts_ms), back_ms=back_ms)
        return False
    typ = str(info.get("type","")).upper()
    rsn = "closed_tp" if "TAKE_PROFIT" in typ else ("closed_sl" if "STOP" in typ else "closed")
    return _journal_close_last(symbol, float(info["price"]), reason=rsn)


def maintain_positions(symbol: str) -> Dict[str, Any]:
    try:
        # NEW: 주기 태스크에서도 필요 시 복원
        if JOURNAL_SYNC_ON_START:
            _journal_restore_from_gcs_if_needed()

        # 위험 회로차단: 일중 손실/연속 손실/누적 MDD 초과 시 거래 중지
        tripped, reason_kill, kstats = _risk_circuit_tripped()
        if tripped:
            entry_guess = 0.0
            try:
                entry_guess = float(get_last_price(symbol) or 0.0)
            except Exception:
                pass
            return {
                "symbol": symbol, "action": "hold", "direction": "hold",
                "entry": entry_guess, "tp": 0.0, "sl": 0.0, "prob": 0.5,
                "risk_ok": False, "rr": 0.0,
                "reason": f"risk_circuit_tripped:{reason_kill}",
                "risk_stats": kstats,
            }

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
        # --- brackets de-dupe on maintain ---
        try:
            orders = get_open_orders(symbol) or []
        except Exception:
            orders = []
        tp_cnt = sum(1 for o in orders if str((o.get("type") if isinstance(o, dict) else getattr(o, "type",""))).upper() in ("TAKE_PROFIT","TAKE_PROFIT_MARKET"))
        sl_cnt = sum(1 for o in orders if str((o.get("type") if isinstance(o, dict) else getattr(o, "type",""))).upper() in ("STOP","STOP_MARKET"))
        if tp_cnt > 1 or sl_cnt > 1:
            # try restore tp/sl from journal's last open row
            tp_val, sl_val = None, None
            try:
                idx, _, rows = _last_open_row_index_and_ts(symbol)
                if idx is not None:
                    last = rows[idx]
                    tp_val = float(last.get("tp") or 0.0) or None
                    sl_val = float(last.get("sl") or 0.0) or None
            except Exception:
                pass
            # estimate current side from position
            try:
                p = get_position(symbol) or {}
                amt2 = float(p.get("positionAmt") or p.get("positionAmount") or 0.0)
                side2 = "BUY" if amt2 > 0 else "SELL"
            except Exception:
                side2 = "BUY"
            if tp_val and sl_val:
                _reset_brackets(symbol, side2, tp_val, sl_val)
                log_event("cleanup.brackets_deduped", symbol=symbol, tp=tp_val, sl=sl_val, tp_cnt=tp_cnt, sl_cnt=sl_cnt)
            else:
                # if no tp/sl, cancel current brackets once; they will be re-created by next entry
                try:
                    cancel_orders_by_type(symbol, ["TAKE_PROFIT","TAKE_PROFIT_MARKET","STOP","STOP_MARKET"])
                    log_event("cleanup.brackets_cancelled", symbol=symbol, reason="dedupe_no_levels")
                except Exception as e:
                    logger.info("cleanup cancel failed for %s: %s", symbol, e)

        # --- heal: TP 또는 SL 누락 시 복구 ---
        if (tp_cnt == 0 or sl_cnt == 0):
            tp_val, sl_val = None, None
            try:
                idx, _, rows = _last_open_row_index_and_ts(symbol)
                if idx is not None:
                    last = rows[idx]
                    tp_val = float(last.get("tp") or 0.0) or None
                    sl_val = float(last.get("sl") or 0.0) or None
            except Exception:
                tp_val, sl_val = None, None

            # 현재 포지션 방향 추정
            try:
                p = get_position(symbol) or {}
                amt2 = float(p.get("positionAmt") or p.get("positionAmount") or 0.0)
                side2 = "BUY" if amt2 > 0 else "SELL"
            except Exception:
                side2 = "BUY"

            if tp_val and sl_val:
                _reset_brackets(symbol, side2, tp_val, sl_val)
                log_event("heal.brackets_missing_placed", symbol=symbol, tp=tp_val, sl=sl_val, tp_cnt=tp_cnt, sl_cnt=sl_cnt)
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

# === NEW: EV components & EV_perc ===
def _ev_components(direction: str, entry: float, tp: float, sl: float) -> tuple[float,float]:
    """
    fee-aware 순수 R_up_net, R_dn_net 계산(ENTRY/TP/SL maker/taker 기대 혼합).
    """
    e, t, s = float(entry), float(tp), float(sl)
    if e<=0 or t<=0 or s<=0: return 0.0, 0.0
    maker = FEE_MAKER_BPS / 1e4; taker = FEE_TAKER_BPS / 1e4
    if direction == "long":
        up_g = (t-e)/e; dn_g = (e-s)/e
    else:
        up_g = (e-t)/e; dn_g = (s-e)/e
    if not RR_EVAL_WITH_FEES:
        return max(0.0, up_g), max(1e-12, dn_g)
    p_maker = _estimate_p_maker_from_journal()
    up_net_m = max(0.0, up_g - (maker + (maker if TP_ORDER_TYPE=="LIMIT" else taker)))
    up_net_t = max(0.0, up_g - (taker + (maker if TP_ORDER_TYPE=="LIMIT" else taker)))
    up_net   = p_maker*up_net_m + (1.0-p_maker)*up_net_t
    dn_net_m = max(1e-12, dn_g + (maker + taker))
    dn_net_t = max(1e-12, dn_g + (taker + taker))
    dn_net   = p_maker*dn_net_m + (1.0-p_maker)*dn_net_t
    return float(up_net), float(dn_net)

def _compute_ev_perc(prob: float, direction: str, entry: float, tp: float, sl: float) -> float:
    """
    EV_perc = p·R_up_net − (1−p)·R_dn_net
    """
    p = max(0.0, min(1.0, float(prob)))
    up, dn = _ev_components(direction, entry, tp, sl)
    return float(p*up - (1.0-p)*dn)

# ---------------------------------
# Manage trade (entry + journaling)
# ---------------------------------# helpers/signals.py — REPLACE WHOLE FUNCTION manage_trade()
# helpers/signals.py — REPLACE WHOLE FUNCTION manage_trade()
def manage_trade(symbol: str) -> Dict[str, Any]:
    try:
        # 필요 시 부팅 직후 복원
        if JOURNAL_SYNC_ON_START:
            _journal_restore_from_gcs_if_needed()

        # 위험 회로차단
        tripped, reason_kill, kstats = _risk_circuit_tripped()
        if tripped:
            entry_guess = 0.0
            try:
                entry_guess = float(get_last_price(symbol) or 0.0)
            except Exception:
                pass
            return {
                "symbol": symbol, "action": "hold", "direction": "hold",
                "entry": entry_guess, "tp": 0.0, "sl": 0.0, "prob": 0.5,
                "risk_ok": False, "rr": 0.0,
                "reason": f"risk_circuit_tripped:{reason_kill}",
                "risk_stats": kstats,
            }

        sig = generate_signal(symbol)
        if not sig:
            log_event("trade.skip", symbol=symbol, reason="predictor_timeout")
            return {"symbol": symbol, "action": "skip", "reason": "predictor_timeout"}
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

        # 기본 유효성
        if direction not in ("long","short") or not risk_ok or entry_intent <= 0 or tp <= 0 or sl <= 0:
            return {"symbol": symbol, "action": "hold", "direction": direction,
                    "entry": entry_intent, "tp": tp, "sl": sl, "prob": prob,
                    "risk_ok": False, "rr": rr, "reason": reason or "no_trade_conditions"}

        # MIN_PROB 재검사 + EV override 정합
        MIN_PROB_LOCAL = float(os.getenv("MIN_PROB", "0.60"))
        EV_OVERRIDE_ENABLED = str(os.getenv("EV_OVERRIDE_ENABLED", "true")).lower() in ("1","true","yes")
        EV_OVERRIDE_MIN_PERC = float(os.getenv("EV_OVERRIDE_MIN_PERC", "0.0005"))
        EV_OVERRIDE_MIN_PROB = float(os.getenv("EV_OVERRIDE_MIN_PROB", "0.54"))

        ev_perc = telemetry.get("ev_perc", None)
        override_ok = False
        try:
            if EV_OVERRIDE_ENABLED and (prob < MIN_PROB_LOCAL) and (prob >= EV_OVERRIDE_MIN_PROB):
                if ev_perc is not None:
                    ep = float(ev_perc)
                    override_ok = (ep >= EV_OVERRIDE_MIN_PERC)
        except Exception:
            override_ok = False

        if (prob < MIN_PROB_LOCAL) and (not override_ok):
            return {"symbol": symbol, "action": "hold", "direction": direction,
                    "entry": entry_intent, "tp": tp, "sl": sl, "prob": prob,
                    "risk_ok": False, "rr": rr, "reason": "prob_below_threshold"}

        # 포지션/레버리지/마진 모드
        try: set_position_mode(POSITION_MODE)
        except Exception as e: logger.warning("set_position_mode: %s", e)
        try: set_margin_type(symbol, MARGIN_TYPE)
        except Exception as e: logger.warning("set_margin_type: %s", e)
        try: set_leverage(symbol, DEFAULT_LEVERAGE)
        except Exception as e: logger.warning("set_leverage: %s", e)

        # 사이징
        qty, size_meta = _compute_size(symbol, entry_intent, sl, risk_scalar)
        side = "BUY" if direction == "long" else "SELL"

        # 집행 (LIMIT 우선 → TTL/reprice → MARKET 폴백)
        if ENTRY_MODE == "MARKET":
            entry_res = place_market_order(symbol, side, quantity=qty)
            brackets = {"take_profit": None, "stop_loss": None}
            try:
                if BRACKETS_RESET_ON_FILL:
                    b = _reset_brackets(symbol, side, tp, sl)
                    if isinstance(b, dict) and b.get("skipped"):
                        # 가시성 레이스 시 즉시 '요청 수량' 폴백
                        brackets = place_bracket_orders(symbol, side, quantity=qty, take_profit=tp, stop_loss=sl)
                        log_event("brackets.reset.fallback_qty", symbol=symbol, qty=float(qty), tp=float(tp), sl=float(sl))
                    else:
                        brackets = b
                else:
                    brackets = place_bracket_orders(symbol, side, quantity=qty, take_profit=tp, stop_loss=sl)
            except Exception as e:
                logger.info("placing brackets failed: %s", e)
            exec_res = {"entry_order": entry_res, "brackets": brackets, "filled_qty": float(qty),
                        "reprices": 0, "used_market_fallback": True, "entry_price": float(0.0)}
            mode = "MARKET"
        else:
            exec_res = _enter_limit_then_brackets(symbol, side, qty, desired_entry=entry_intent, tp=tp, sl=sl)
            mode = "LIMIT"

        # === ABORT_IF_NO_FILL: 미체결이면 저널에 open 행 쓰지 않고 종료 ===
        filled_qty = 0.0
        try:
            filled_qty = float(exec_res.get("filled_qty", 0.0) or 0.0)
        except Exception:
            filled_qty = 0.0

        if filled_qty <= 0.0:
            try:
                cancel_open_orders(symbol)
            except Exception:
                pass
            log_event("entry.aborted_not_filled", symbol=symbol, mode=mode,
                      entry_intent=float(entry_intent), tp=float(tp), sl=float(sl))
            return {"symbol": symbol, "action": "hold", "direction": direction,
                    "entry": entry_intent, "tp": tp, "sl": sl, "prob": prob,
                    "risk_ok": False, "rr": rr, "reason": "entry_not_filled_no_journal"}

        # 엔트리 체결가 보정
        entry_actual = float(exec_res.get("entry_price") or 0.0)
        if entry_actual <= 0:
            try:
                pos = get_position(symbol) or {}
                entry_actual = float(pos.get("entryPrice") or 0.0) or entry_intent
            except Exception:
                entry_actual = entry_intent

        # EV 로깅(Ex-ante)
        try:
            ev_perc2 = _compute_ev_perc(prob_cal, direction, entry_intent, tp, sl)
            ev_usd  = ev_perc2 * float(size_meta.get('notional', float(filled_qty*entry_intent)))
            row = {
                "timestamp": _now_utc().isoformat(),
                "symbol": symbol,
                "side": "long" if side == "BUY" else "short",
                "qty": f"{float(filled_qty):.10f}",
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
                "notional": f"{float(size_meta.get('notional', float(filled_qty*entry_intent))):.10f}",
                "bal_pct": f"{float(size_meta.get('bal_pct', 0.0)):.6f}",
                "ev_perc": f"{float(ev_perc2):.10f}",
                "ev_usd":  f"{float(ev_usd):.10f}",
                "ev_ex_ante_perc": f"{float(ev_perc2):.10f}",
                "ev_ex_ante_usd":  f"{float(ev_usd):.10f}",
            }
            _journal_append_open(row)
            if gcs_enabled():
                gcs_append_csv_row("trades", list(row.keys()), row)
                try: _journal_backup_to_gcs(tag="entry")
                except Exception: pass
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
        "size_mode","bal_asset","notional","bal_pct","exit_ts",
        "ev_perc","ev_usd","ev_ex_ante_perc","ev_ex_ante_usd"
    ]
    row.setdefault("ev_ex_ante_perc", row.get("ev_perc",""))
    row.setdefault("ev_ex_ante_usd",  row.get("ev_usd",""))
    if "exit_ts" not in row: row["exit_ts"] = ""

    # 파일락 범위에서 현재 파일을 읽고 행을 추가한 뒤 전체 재기록
    with _lock_file(TRADES_CSV):
        old_rows = []
        if os.path.exists(TRADES_CSV):
            with open(TRADES_CSV, "r", encoding="utf-8") as f:
                r = csv.DictReader(f)
                old_rows = list(r)
        old_rows.append(row)
        _rewrite_trades_csv(old_rows, pref_headers=headers)

# === NEW: export to app.py ===
def get_overview() -> Dict[str, Any]:
    """Pass-through for /api/overview import in app.py."""
    try:
        return _bn_get_overview()
    except Exception as e:
        logger.info("get_overview passthrough failed: %s", e)
        return {"balances": [], "positions": []}

# --- Safe helpers appended: TP/SL race hardening and time-exit close ---
def _position_qty_with_retry(symbol: str, attempts: int = 8, sleep_ms: int = 250) -> float:
    from time import sleep
    for _ in range(max(1, int(attempts))):
        try:
            pos = get_position(symbol) or {}
            amt = float(pos.get("positionAmt") or pos.get("positionAmount") or 0.0)
            q = abs(amt)
        except Exception:
            q = 0.0
        if q > 1e-12:
            return q
        sleep(max(1, int(sleep_ms)) / 1000.0)
    return 0.0

def _reset_brackets(symbol: str, side: str, tp: float, sl: float) -> Dict[str, Any]:
    """
    기존 TP/SL(RO) 전부 취소 후, '현재 포지션 잔고' 전량 기준으로 브래킷(TP/SL) 1쌍만 재배치.
    side: 엔트리 방향("BUY"/"SELL") 그대로 전달.
    """
    # 1) 기존 브래킷류 취소
    try:
        cancel_orders_by_type(symbol, ["TAKE_PROFIT", "TAKE_PROFIT_MARKET", "STOP", "STOP_MARKET"])
        log_event("brackets.reset.cancelled", symbol=symbol)
    except Exception as e:
        logger.info("brackets cancel failed for %s: %s", symbol, e)

    # 2) 현재 포지션 잔고 조회(재시도 포함)
    qty = _position_qty_with_retry(symbol, attempts=8, sleep_ms=250)

    if qty <= 1e-12:
        # 관측성 강화: 재시도 후에도 포지션 미가시 → 상위에서 '요청 수량' 폴백을 선택할 수 있게 신호 반환
        log_event("brackets.reset.skip", symbol=symbol, reason="no_position_after_retry")
        return {"take_profit": None, "stop_loss": None, "skipped": True, "reason": "no_position_after_retry"}

    # 3) 브래킷 1쌍 재배치
    out = place_bracket_orders(symbol, side, qty, take_profit=float(tp), stop_loss=float(sl))
    log_event("brackets.reset.placed", symbol=symbol, qty=float(qty), tp=float(tp), sl=float(sl))
    return out if isinstance(out, dict) else {"raw": out}

def _close_position_market(symbol: str) -> Optional[dict]:
    try:
        p = get_position(symbol) or {}
        amt = float(p.get("positionAmt") or p.get("positionAmount") or 0.0)
        if abs(amt) <= 1e-12:
            return None
        side = "SELL" if amt > 0 else "BUY"
        try:
            # 1차: 현 설정값 그대로
            res = place_market_order(symbol, side, quantity=abs(amt), reduce_only=True)
        except Exception as e1:
            # 2차: 모드 의심 → position_side='BOTH' 강제
            logger.info("time_exit primary close failed for %s: %s; retry with position_side=BOTH", symbol, e1)
            try:
                res = place_market_order(symbol, side, quantity=abs(amt), reduce_only=True, position_side_override="BOTH")
            except Exception as e2:
                logger.info("time_exit close failed for %s: %s", symbol, e2)
                return None
        try:
            cancel_open_orders(symbol)
        except Exception:
            pass
        log_event("time_exit", symbol=symbol, positionAmt=amt, side=side)
        return res if isinstance(res, dict) else {"raw": res}
    except Exception as e:
        logger.info("time_exit close failed for %s: %s", symbol, e)
        return None

# Public exports
__all__ = [
    "generate_signal",
    "manage_trade",
    "get_overview",
    "maintain_positions",
    "journal_sync",
    "preview_size",
    "journal_reset",
]



