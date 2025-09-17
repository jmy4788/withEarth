# helpers/predictor_router.py
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import os, json, logging, time
import requests

try:
    from .utils import log_event  # type: ignore
except Exception:
    def log_event(event: str, **fields): logging.getLogger("event").info(json.dumps({"event":event, **fields}))

# --- env knobs ---
BACKEND = os.getenv("PREDICTOR_BACKEND", os.getenv("PREDICTOR_IMPL", "GEMINI")).upper()
TSFM2_URL = os.getenv("TSFM2_URL", "").rstrip("/")
TSFM2_API_KEY = os.getenv("TSFM2_API_KEY", "")
TSFM2_TIMEOUT = int(os.getenv("TSFM2_TIMEOUT_MS", "8000"))/1000.0
SIGMA_MODE = os.getenv("TSFM2_SIGMA_MODE", "DATA_ATR").upper()  # DATA_ATR | FEATURE_AWARE | HYBRID
SIGMA_CAP = float(os.getenv("TSFM2_SIGMA_CAP", "0.06"))          # per-step sigma upper bound
SIGMA_MIN = float(os.getenv("TSFM2_SIGMA_MIN", "1e-6"))          # lower bound
USE_GEMINI_FALLBACK = str(os.getenv("USE_GEMINI_FALLBACK", "false")).lower() in ("1", "true", "yes")


def _atr_frac(entry: float, atr: float) -> float:
    try:
        e = float(entry); a = float(atr)
        return (a / e) if (e > 0 and a > 0) else 0.0
    except Exception:
        return 0.0


def _log_req_meta(symbol: str, req: Dict[str, Any], entry: float, atr: float) -> None:
    try:
        from .utils import log_event  # type: ignore
    except Exception:
        def log_event(event: str, **fields): return
    log_event("tsfm2.req_meta",
              symbol=symbol,
              sigma_mode=SIGMA_MODE,
              override_sigma=(req.get("override_sigma") or 0.0),
              sigma_cap=SIGMA_CAP,
              atr_frac=_atr_frac(entry, atr),
              n_paths=req.get("n_paths"))


# Fallback: Gemini
def _fallback_gemini(payload: Dict[str, Any], symbol: str="") -> Dict[str, Any]:
    from .predictor import get_gemini_prediction  # type: ignore
    return get_gemini_prediction(payload, symbol=symbol)

def _extract_context(payload: Dict[str, Any]) -> Tuple[list[float], int, int]:
    dt_sec = 300
    hz_min = int(payload.get("horizon_min", 30))
    horizon_steps = max(1, int(hz_min * 60 // dt_sec))
    closes = payload.get("price_sequence") or []
    try:
        if len(closes) >= 64:
            return [float(x) for x in closes[-256:]], dt_sec, horizon_steps
        from .data_fetch import fetch_ohlcv  # lazy import
        pair = str(payload.get("pair", ""))
        if pair:
            df = fetch_ohlcv(pair, interval="5m", limit=256)
            if df is not None and not df.empty:
                ohlc = [float(x) for x in df["close"].tolist()]
                return ohlc[-256:], dt_sec, horizon_steps
    except Exception:
        pass
    entry = float((payload.get("entry_5m") or {}).get("close", 0.0) or 0.0)
    base_len = max(64, len(closes))
    closes = (closes + [entry] * (base_len - len(closes)))[:base_len]
    return [float(x) for x in closes[-256:]], dt_sec, horizon_steps

def _sigma_from_features(payload: Dict[str, Any], entry: float, atr: float) -> Optional[float]:
    """Feature-aware sigma adjustment using spread, imbalance, relative volume, volatility, and micro dislocation."""
    try:
        if entry <= 0:
            return None
        extra = payload.get("extra") or {}
        rv = float(extra.get("relative_volume_5m", 1.0))
        spread_bps = float(extra.get("orderbook_spread", 0.0))
        imbalance = abs(float(extra.get("orderbook_imbalance", 0.0)))
        micro_disp = abs(float(extra.get("micro_dislocation_bps", 0.0)))
        vol5 = float((payload.get("entry_5m") or {}).get("volatility", 0.0))

        base = max((atr / entry) if atr > 0 else 0.0, SIGMA_MIN)

        scale = 1.0
        scale += 0.50 * (spread_bps / 1e4)
        scale += 0.30 * imbalance
        scale += 0.20 * (micro_disp / 1e4)
        scale += 0.30 * max(0.0, rv - 1.0)
        scale += 0.20 * min(1.0, max(0.0, vol5 * 10.0))

        sigma = max(SIGMA_MIN, min(SIGMA_CAP, base * scale))
        return float(sigma)
    except Exception:
        return None


def _build_prob_gate_req(payload: Dict[str, Any]) -> Dict[str, Any]:
    closes, dt_sec, H = _extract_context(payload)
    br = payload.get("brackets") or payload.get("bracket") or {}
    entry = float((br.get("entry") or (payload.get("entry_5m") or {}).get("close") or 0.0))
    long_b = br.get("long") or {}
    short_b = br.get("short") or {}
    extra = payload.get("extra") or {}
    atr = float(extra.get("ATR_5m") or 0.0)

    req = {
        "closes": closes,
        "freq": 0,               # 5m frequency (<= daily)
        "dt_sec": dt_sec,
        "horizon_steps": H,
        "bracket": {
            "entry": entry,
            "long": {"tp": float(long_b.get("tp", 0.0)), "sl": float(long_b.get("sl", 0.0))} if long_b else None,
            "short": {"tp": float(short_b.get("tp", 0.0)), "sl": float(short_b.get("sl", 0.0))} if short_b else None,
        },
        "atr_now": (atr if atr > 0 else None),
        "n_paths": int(os.getenv("TSFM2_MC_PATHS", "4000")),
    }

    override_sigma = None
    try:
        if SIGMA_MODE in ("FEATURE_AWARE", "HYBRID"):
            override_sigma = _sigma_from_features(payload, entry=entry, atr=atr)
    except Exception:
        override_sigma = None
    if override_sigma is not None:
        req["override_sigma"] = float(override_sigma)

    _log_req_meta(str(payload.get("pair", "")), req, entry, atr)
    return req

def _call_tsfm2_prob_gate(req: Dict[str, Any]) -> Optional[Dict[str, Any]]:
    headers = {"Content-Type":"application/json"}
    if TSFM2_API_KEY:
        headers["X-Api-Key"] = TSFM2_API_KEY
    try:
        t0 = time.time()
        r = requests.post(f"{TSFM2_URL}/v1/prob_gate", data=json.dumps(req), headers=headers, timeout=TSFM2_TIMEOUT)
        t1 = time.time()
        out = None
        if r.status_code == 200:
            out = r.json()
        log_event("tsfm2.response",
                  status=r.status_code,
                  elapsed_ms=int((t1-t0)*1000),
                  direction=(out or {}).get("direction"),
                  prob=(out or {}).get("prob"),
                  sigma_step=((out or {}).get("diagnostics") or {}).get("sigma_step"))
        return out if r.status_code == 200 else None
    except Exception as e:
        log_event("tsfm2.error", error=str(e)[:200])
        return None

def get_prediction(payload: Dict[str, Any], symbol: str="") -> Optional[Dict[str, Any]]:
    """
    signals.generate_signal()에서 호출. 반환 사양은 helpers/predictor.get_gemini_prediction()과 동일:
    {direction, prob, support, resistance, reasoning}
    """
    if BACKEND not in ("TSFM2", "TIMESFM", "TSFM"):
        return _fallback_gemini(payload, symbol=symbol) if USE_GEMINI_FALLBACK else None

    if not TSFM2_URL:
        if USE_GEMINI_FALLBACK:
            return _fallback_gemini(payload, symbol=symbol)
        log_event("tsfm2.unavailable", reason="missing_url")
        return None

    # 로깅(요약)
    try:
        br = (payload.get("brackets") or payload.get("bracket") or {})
        log_event("tsfm2.request", symbol=(symbol or payload.get("pair")), entry=(br.get("entry") or (payload.get("entry_5m") or {}).get("close")))
    except Exception:
        pass

    req = _build_prob_gate_req(payload)
    out = _call_tsfm2_prob_gate(req)
    if not out:
        if USE_GEMINI_FALLBACK:
            return _fallback_gemini(payload, symbol=symbol)
        log_event("tsfm2.timeout", symbol=(symbol or payload.get("pair")), reason="no_response")
        return None

    # TimesFM 서비스는 SR을 추정하지 않으므로 0으로 채움(상위 로직이 SR clamp/ATR 기반 재계산 수행)
    return {
        "direction": str(out.get("direction", "hold")),
        "prob": float(out.get("prob", 0.5)),
        "support": None,
        "resistance": None,
        "reasoning": "tsfm2 probabilistic first-passage via MC",
    }
