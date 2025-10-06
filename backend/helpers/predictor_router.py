# helpers/predictor_router.py  (drop-in)
from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import json
import logging
import os
import time

import requests

logger = logging.getLogger(__name__)

try:
    from .utils import log_event  # type: ignore
except Exception:  # pragma: no cover
    def log_event(event: str, **fields) -> None:  # type: ignore[override]
        logging.getLogger("event").info(json.dumps({"event": event, **fields}, ensure_ascii=False))


# --- env knobs ---
BACKEND = os.getenv("PREDICTOR_BACKEND", os.getenv("PREDICTOR_IMPL", "GEMINI")).upper()
TSFM2_URL = os.getenv("TSFM2_URL", "").rstrip("/")
TSFM2_API_KEY = os.getenv("TSFM2_API_KEY", "")
TSFM2_TIMEOUT_MS = int(os.getenv("TSFM2_TIMEOUT_MS", "8000"))
TSFM2_TIMEOUT = TSFM2_TIMEOUT_MS / 1000.0
TSFM2_CONNECT_TIMEOUT = int(os.getenv("TSFM2_CONNECT_TIMEOUT_MS", "3000")) / 1000.0
_default_read_ms = max(60000, TSFM2_TIMEOUT_MS)
TSFM2_READ_TIMEOUT = int(os.getenv("TSFM2_READ_TIMEOUT_MS", str(_default_read_ms))) / 1000.0

SIGMA_MODE = os.getenv("TSFM2_SIGMA_MODE", "DATA_ATR").upper()  # DATA_ATR | FEATURE_AWARE | HYBRID
SIGMA_CAP = float(os.getenv("TSFM2_SIGMA_CAP", "0.06"))
SIGMA_MIN = float(os.getenv("TSFM2_SIGMA_MIN", "1e-6"))
USE_GEMINI_FALLBACK = str(os.getenv("USE_GEMINI_FALLBACK", "false")).lower() in ("1", "true", "yes")


def _parse_aux_tf_mins(raw: str | None) -> tuple[int, ...]:
    if not raw:
        return (5,)
    tokens = []
    for part in raw.replace(";", ",").split(","):
        part = part.strip()
        if not part:
            continue
        try:
            tokens.append(int(part))
        except ValueError:
            continue
    return tuple(tokens) if tokens else (5,)


AUX_TF_MINS: tuple[int, ...] = _parse_aux_tf_mins(os.getenv("AUX_TF_MINS", "5"))

_BASE_TF_RAW = os.getenv("TSFM2_BASE_TF_MIN", "5")
try:
    BASE_TF_MIN = int(_BASE_TF_RAW)
except Exception:  # pragma: no cover
    BASE_TF_MIN = 5


def _base_tf_min() -> int:
    """Sanitize base TF env (supports 5 or 15, clamps 5-60)."""
    try:
        raw = os.getenv("TSFM2_BASE_TF_MIN", str(BASE_TF_MIN))
        return max(5, min(60, int(raw)))
    except Exception:
        return 5


def _safe_log_event(event: str, **fields) -> None:
    try:
        log_event(event, **fields)
    except Exception:
        logger.info("%s %s", event, json.dumps(fields, ensure_ascii=False, default=str))


def _atr_frac(entry: float, atr: float) -> float:
    try:
        e = float(entry)
        a = float(atr)
        return (a / e) if (e > 0 and a > 0) else 0.0
    except Exception:
        return 0.0


def _log_req_meta(symbol: str, req: Dict[str, Any], entry: float, atr: float) -> None:
    closes_len = len(req.get("closes") or [])
    _safe_log_event(
        "tsfm2.req_meta",
        symbol=symbol,
        base_tf_min=_base_tf_min(),
        dt_sec=req.get("dt_sec"),
        horizon_steps=req.get("horizon_steps"),
        closes_len=closes_len,
        entry=entry,
        atr_frac=_atr_frac(entry, atr),
        sigma_mode=SIGMA_MODE,
        override_sigma=req.get("override_sigma") or 0.0,
        sigma_cap=SIGMA_CAP,
        n_paths=req.get("n_paths"),
        aux_tf_mins=list(AUX_TF_MINS),
        connect_timeout_s=float(TSFM2_CONNECT_TIMEOUT),
        read_timeout_s=float(TSFM2_READ_TIMEOUT),
    )


# Fallback: Gemini

def _fallback_gemini(payload: Dict[str, Any], symbol: str = "") -> Dict[str, Any]:
    from .predictor import get_gemini_prediction  # type: ignore

    return get_gemini_prediction(payload, symbol=symbol)


def _extract_context(payload: Dict[str, Any]) -> Tuple[list[float], int, int]:
    """Build TSFM context from OHLC with unified base TF (5m/15m)."""

    base_tf = _base_tf_min()
    dt_sec = base_tf * 60
    hz_min = int(payload.get("horizon_min", 30))
    horizon_steps = max(1, int(hz_min * 60 // max(1, dt_sec)))

    closes = payload.get("price_sequence") or []
    if len(closes) < (256 if base_tf <= 5 else 192):
        try:
            from .data_fetch import fetch_ohlcv  # lazy import

            pair = str(payload.get("pair", "") or "")
            if pair:
                interval = "5m" if base_tf <= 5 else "15m"
                limit = 256 if interval == "5m" else 192
                df = fetch_ohlcv(pair, interval=interval, limit=limit)
                if df is not None and not df.empty:
                    closes = [float(x) for x in df["close"].tolist()]
        except Exception:
            pass

    entry = float(
        (payload.get("brackets") or payload.get("bracket") or {}).get("entry")
        or (payload.get("entry_15m") or {}).get("close")
        or (payload.get("entry_5m") or {}).get("close")
        or 0.0
    )
    if len(closes) < 64:
        pad = max(64 - len(closes), 0)
        closes = (closes + [entry] * pad)[:64]

    closes = [float(x) for x in closes[-256:]]
    return closes, dt_sec, horizon_steps


def _sigma_from_features(payload: Dict[str, Any], entry: float, atr: float) -> Optional[float]:
    """Feature-aware sigma. Prefer ATR_15m if available when base TF>=15."""

    try:
        extra = payload.get("extra") or {}
        atr5 = float(extra.get("ATR_5m", 0.0))
        atr15 = float(extra.get("ATR_15m", 0.0))
        use_atr = atr15 if (_base_tf_min() >= 15 and atr15 > 0) else (atr if atr > 0 else atr5)
        if entry <= 0 or use_atr <= 0:
            return None

        rv = float(extra.get("relative_volume_5m", 1.0))
        spread_bps = float(extra.get("orderbook_spread", 0.0))
        imbalance = abs(float(extra.get("orderbook_imbalance", 0.0)))
        micro_disp = abs(float(extra.get("micro_dislocation_bps", 0.0)))
        vol5 = float((payload.get("entry_5m") or {}).get("volatility", 0.0))

        base = max((use_atr / entry), SIGMA_MIN)
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
    closes, dt_sec, horizon_steps = _extract_context(payload)
    br = payload.get("brackets") or payload.get("bracket") or {}
    entry = float(
        br.get("entry")
        or (payload.get("entry_15m") or {}).get("close")
        or (payload.get("entry_5m") or {}).get("close")
        or 0.0
    )
    long_b = br.get("long") or {}
    short_b = br.get("short") or {}
    extra = payload.get("extra") or {}
    base_tf = _base_tf_min()
    atr_key = "ATR_15m" if base_tf >= 15 else "ATR_5m"
    atr = float(extra.get(atr_key) or 0.0)

    req: Dict[str, Any] = {
        "closes": closes,
        "freq": 0,
        "dt_sec": dt_sec,
        "horizon_steps": horizon_steps,
        "bracket": {
            "entry": entry,
            "long": {
                "tp": float(long_b.get("tp", 0.0)),
                "sl": float(long_b.get("sl", 0.0)),
            }
            if long_b
            else None,
            "short": {
                "tp": float(short_b.get("tp", 0.0)),
                "sl": float(short_b.get("sl", 0.0)),
            }
            if short_b
            else None,
        },
        "atr_now": atr if atr > 0 else None,
        "n_paths": int(os.getenv("TSFM2_MC_PATHS", "2000")),
    }

    override_sigma: Optional[float] = None
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
    headers = {"Content-Type": "application/json"}
    if TSFM2_API_KEY:
        headers["X-Api-Key"] = TSFM2_API_KEY
    try:
        t0 = time.time()
        r = requests.post(
            f"{TSFM2_URL}/v1/prob_gate",
            data=json.dumps(req),
            headers=headers,
            timeout=(TSFM2_CONNECT_TIMEOUT, TSFM2_READ_TIMEOUT),
        )
        elapsed_ms = int((time.time() - t0) * 1000)
        out: Optional[Dict[str, Any]] = None
        if r.status_code == 200:
            out = r.json()
        _safe_log_event(
            "tsfm2.response",
            status=r.status_code,
            elapsed_ms=elapsed_ms,
            direction=(out or {}).get("direction"),
            prob=(out or {}).get("prob"),
            sigma_step=((out or {}).get("diagnostics") or {}).get("sigma_step"),
        )
        return out if r.status_code == 200 else None
    except Exception as e:
        _safe_log_event("tsfm2.error", error=str(e)[:200])
        return None


def get_prediction(payload: Dict[str, Any], symbol: str = "") -> Optional[Dict[str, Any]]:
    if BACKEND not in ("TSFM2", "TIMESFM", "TSFM"):
        return _fallback_gemini(payload, symbol=symbol) if USE_GEMINI_FALLBACK else None

    if not TSFM2_URL:
        if USE_GEMINI_FALLBACK:
            return _fallback_gemini(payload, symbol=symbol)
        _safe_log_event("tsfm2.unavailable", reason="missing_url")
        return None

    try:
        br = (payload.get("brackets") or payload.get("bracket") or {})
        _safe_log_event(
            "tsfm2.request",
            symbol=(symbol or payload.get("pair")),
            entry=(
                br.get("entry")
                or (payload.get("entry_15m") or {}).get("close")
                or (payload.get("entry_5m") or {}).get("close")
            ),
            base_tf_min=_base_tf_min(),
            aux_tf_mins=list(AUX_TF_MINS),
            connect_timeout_s=float(TSFM2_CONNECT_TIMEOUT),
            read_timeout_s=float(TSFM2_READ_TIMEOUT),
        )
    except Exception:
        pass

    req = _build_prob_gate_req(payload)
    out = _call_tsfm2_prob_gate(req)
    if not out:
        if USE_GEMINI_FALLBACK:
            return _fallback_gemini(payload, symbol=symbol)
        _safe_log_event(
            "tsfm2.timeout",
            symbol=(symbol or payload.get("pair")),
            reason="no_response",
        )
        return None

    return {
        "direction": str(out.get("direction", "hold")),
        "prob": float(out.get("prob", 0.5)),
        "support": None,
        "resistance": None,
        "reasoning": "tsfm2 probabilistic first-passage via MC",
    }
