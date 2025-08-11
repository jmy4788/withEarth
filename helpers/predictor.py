from __future__ import annotations

import base64
import json
import logging
import os
from datetime import datetime
from typing import Any, Dict, Optional, List

from google import genai
from google.genai import types

# --- Safe imports for shared utils (standalone-friendly fallbacks) ---
try:
    from .utils import get_secret, LOG_DIR  # type: ignore
except Exception:  # pragma: no cover
    def get_secret(name: str) -> Optional[str]:
        return os.getenv(name)

    LOG_DIR = os.path.join(os.getcwd(), "logs")

# =====================
# Config & Constants
# =====================
# Default to a stable model; env can override
_DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-1.5-flash").strip()
TEMPERATURE: float = float(os.getenv("G_TEMPERATURE", "0.0"))
# Give the model enough room; you can raise via env to 768/1024 if needed
MAX_TOKENS: int = int(os.getenv("G_MAX_TOKENS", "512"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

HOLD: Dict[str, Any] = {
    "direction": "hold",
    "prob": 0.5,
    "support": 0.0,
    "resistance": 0.0,
    "reasoning": "",
}

# Enforce structured JSON back from Gemini (when supported)
RESPONSE_SCHEMA = types.Schema(
    type=types.Type.OBJECT,
    properties={
        "direction": types.Schema(type=types.Type.STRING, enum=["long", "short", "hold"]),
        "prob": types.Schema(type=types.Type.NUMBER),
        "support": types.Schema(type=types.Type.NUMBER),
        "resistance": types.Schema(type=types.Type.NUMBER),
        "reasoning": types.Schema(type=types.Type.STRING),
    },
    required=["direction"],
)

# =====================
# Debug helpers
# =====================

def _mk_debug_dir() -> str:
    d = os.path.join(LOG_DIR, "payloads", datetime.now().strftime("%Y%m%d"))
    os.makedirs(d, exist_ok=True)
    return d


def _save_json(obj: Any, fname: str) -> None:
    try:
        base = _mk_debug_dir()
        path = os.path.join(base, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
        logger.info("Saved debug json -> %s", path)
    except Exception as e:
        logger.warning("Failed to save debug json: %s", e)


def _save_payload(payload: Dict[str, Any], symbol: Optional[str]) -> None:
    try:
        name = f"{datetime.now().strftime('%H%M%S')}_{(symbol or payload.get('pair') or 'NA').replace('/', '_')}.json"
        _save_json(payload, name)
    except Exception:
        pass


def _dump_candidates_debug(resp: Any, fname: str) -> None:
    """Write a compact view of candidate parts + usage for troubleshooting."""
    try:
        out: List[Dict[str, Any]] = []
        for cand in getattr(resp, "candidates", []) or []:
            row: Dict[str, Any] = {
                "finish": getattr(cand, "finish_reason", ""),
                "safety": getattr(cand, "safety_ratings", []),
                "parts": [],
            }
            c = getattr(cand, "content", None)
            if c is not None:
                for p in getattr(c, "parts", []) or []:
                    item: Dict[str, Any] = {}
                    if hasattr(p, "text") and p.text:
                        item["text"] = p.text[:512]
                    idata = getattr(p, "inline_data", None)
                    if idata:
                        item["inline_mime"] = getattr(idata, "mime_type", "")
                        item["inline_size"] = len(getattr(idata, "data", b"") or b"")
                    fcall = getattr(p, "function_call", None)
                    if fcall:
                        item["function_call"] = {
                            "name": getattr(fcall, "name", ""),
                            "has_args": bool(getattr(fcall, "args", None)),
                        }
                    row["parts"].append(item)
            out.append(row)
        usage = {}
        try:
            usage = {
                "input_tokens": getattr(getattr(resp, "usage_metadata", None), "prompt_token_count", None),
                "output_tokens": getattr(getattr(resp, "usage_metadata", None), "candidates_token_count", None),
                "total_tokens": getattr(getattr(resp, "usage_metadata", None), "total_token_count", None),
            }
        except Exception:
            pass
        _save_json({"candidates": out, "usage": usage}, fname)
    except Exception as e:
        logger.warning("failed to dump candidates: %s", e)

# =====================
# Parse helpers
# =====================

def _normalize_model_name(name: str) -> str:
    # google-genai expects bare names like "gemini-1.5-flash"
    return name.split("/")[-1].strip()


def _find_json_in_text(text: str) -> Optional[str]:
    if not text:
        return None
    s = text
    n = len(s)
    i = 0
    while i < n and s[i] != '{':
        i += 1
    if i >= n:
        return None
    start = i
    depth = 0
    in_str = False
    escape = False
    while i < n:
        ch = s[i]
        if in_str:
            if escape:
                escape = False
            elif ch == '\\':
                escape = True
            elif ch == '"':
                in_str = False
        else:
            if ch == '"':
                in_str = True
            elif ch == '{':
                depth += 1
            elif ch == '}':
                depth -= 1
                if depth == 0:
                    end = i + 1
                    return s[start:end]
        i += 1
    return None


def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    try:
        j = _find_json_in_text(text)
        return json.loads(j) if j else None
    except Exception:
        return None


def _extract_from_parts(resp: Any) -> Optional[Dict[str, Any]]:
    try:
        if not getattr(resp, "candidates", None):
            return None
        parts = getattr(resp.candidates[0].content, "parts", []) or []
        for p in parts:
            if hasattr(p, "text") and p.text and p.text.strip():
                obj = _safe_json_extract(p.text)
                if obj:
                    return obj
            idata = getattr(p, "inline_data", None)
            if idata and getattr(idata, "mime_type", "") == "application/json":
                try:
                    raw = base64.b64decode(getattr(idata, "data", b"") or b"").decode("utf-8", "ignore")
                    return json.loads(raw)
                except Exception:
                    pass
        return None
    except Exception:
        return None

# =====================
# Payload shaping
# =====================

def _round_floats(obj: Any, ndigits: int = 6) -> Any:
    try:
        if isinstance(obj, dict):
            return {k: _round_floats(v, ndigits) for k, v in obj.items()}
        if isinstance(obj, list):
            return [_round_floats(v, ndigits) for v in obj]
        if isinstance(obj, float):
            return round(obj, ndigits)
        return obj
    except Exception:
        return obj


def _minimize_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {
        "pair": payload.get("pair") or payload.get("symbol") or "",
    }
    e5 = payload.get("entry_5m") or {}
    if isinstance(e5, dict):
        keep_e5 = {
            k: e5.get(k)
            for k in [
                "close",
                "rsi",
                "volatility",
                "sma20",
                "funding_rate_pct",
                "sentiment",
                "timestamp",
            ]
            if k in e5
        }
        out["entry_5m"] = keep_e5

    extra = payload.get("extra") or {}
    if isinstance(extra, dict):
        for k in [
            "RSI_1h",
            "ATR_1h",
            "RSI_4h",
            "ATR_4h",
            "RSI_1d",
            "ATR_1d",
            "orderbook_imbalance",
            "orderbook_spread",
            "recent_high_5m",
            "recent_low_5m",
            "recent_high_1h",
            "recent_low_1h",
        ]:
            if k in extra:
                out[k] = extra[k]
    if "sr_levels" in payload and isinstance(payload["sr_levels"], dict):
        out["sr_levels"] = {
            k: payload["sr_levels"].get(k)
            for k in ("recent_high", "recent_low")
            if k in payload["sr_levels"]
        }
    return _round_floats(out, 6)


def _micro_payload(payload: Dict[str, Any]) -> Dict[str, Any]:
    # ultra-thin: just a few decisive signals
    e5 = (payload or {}).get("entry_5m", {})
    ex = (payload or {}).get("extra", {})
    return _round_floats(
        {
            "pair": payload.get("pair") or payload.get("symbol") or "",
            "entry_5m": {
                "close": e5.get("close", 0.0),
                "rsi": e5.get("rsi", 50.0),
            },
            "extra": {
                "ATR_5m": ex.get("ATR_5m", 0.0),
                "orderbook_imbalance": ex.get("orderbook_imbalance", 0.0),
            },
        },
        6,
    )

# =====================
# Public API
# =====================

def get_gemini_prediction(payload: Dict[str, Any], symbol: Optional[str] = None) -> Dict[str, Any]:
    """Ask Gemini for a single structured decision. Returns HOLD on any hard failure."""
    _save_payload(payload, symbol)

    api_key = get_secret("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY") or ""
    if not api_key:
        logger.warning("GOOGLE_API_KEY not provided; returning HOLD.")
        return HOLD.copy()

    client = genai.Client(api_key=api_key)
    model_name = _normalize_model_name(_DEFAULT_MODEL)

    # Prompt designed to minimize HOLD bias but still allow when noisy/flat
    system_note = (
        "You are an ultra-concise crypto futures signaler. Use ONLY the provided numeric features. "
        "Choose LONG or SHORT when there is directional evidence; otherwise HOLD."
    )
    user_rule = (
        "Return ONLY one JSON object with keys: direction, prob, support, resistance, reasoning.\n"
        "direction ∈ {long, short, hold}; prob is 0..1.\n"
        "If any field is unknown, still output all keys with defaults (prob≈0.5, support=0, resistance=0)."
    )

    content = _minimize_payload(payload)
    msgs = [
        types.Content(
            role="user",
            parts=[
                types.Part.from_text(text=user_rule),
                types.Part.from_text(text=json.dumps(content, ensure_ascii=False)),
            ],
        )
    ]

    # 1) Schema-enforced call first (most reliable)
    cfg_schema = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        top_p=0,
        top_k=1,
        max_output_tokens=min(640, MAX_TOKENS),
        response_mime_type="application/json",
        response_schema=RESPONSE_SCHEMA,
        system_instruction=types.Content(role="system", parts=[types.Part.from_text(text=system_note)]),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="NONE")
        ),
    )

    try:
        resp1 = client.models.generate_content(model=model_name, contents=msgs, config=cfg_schema)
        _dump_candidates_debug(resp1, f"{datetime.now().strftime('%H%M%S')}_{symbol or content.get('pair')}_debug_parts_1st.json")
        obj = getattr(resp1, "parsed", None) or _extract_from_parts(resp1) or _safe_json_extract(getattr(resp1, "text", "") or "")
        if obj:
            d = str(obj.get("direction", "hold")).lower()
            if d not in ("long", "short", "hold"):
                d = "hold"
            return {
                "direction": d,
                "prob": float(obj.get("prob", 0.5)),
                "support": float(obj.get("support", 0.0)),
                "resistance": float(obj.get("resistance", 0.0)),
                "reasoning": str(obj.get("reasoning", "")),
            }
    except Exception as e:
        logger.warning(f"gemini first-call failed: {e}")

    # 2) Fallback: JSON MIME without schema
    cfg_json = types.GenerateContentConfig(
        temperature=TEMPERATURE,
        top_p=0,
        top_k=1,
        max_output_tokens=min(512, MAX_TOKENS),
        response_mime_type="application/json",
        system_instruction=types.Content(role="system", parts=[types.Part.from_text(text=system_note)]),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="NONE")
        ),
    )

    try:
        resp2 = client.models.generate_content(model=model_name, contents=msgs, config=cfg_json)
        _dump_candidates_debug(resp2, f"{datetime.now().strftime('%H%M%S')}_{symbol or content.get('pair')}_debug_parts_2nd.json")
        obj = getattr(resp2, "parsed", None) or _extract_from_parts(resp2) or _safe_json_extract(getattr(resp2, "text", "") or "")
        if obj:
            d = str(obj.get("direction", "hold")).lower()
            if d not in ("long", "short", "hold"):
                d = "hold"
            prob = obj.get("prob")
            p = float(prob) if isinstance(prob, (int, float, str)) and str(prob) not in ("", "nan") else 0.5
            return {
                "direction": d,
                "prob": p,
                "support": float(obj.get("support", 0.0)),
                "resistance": float(obj.get("resistance", 0.0)),
                "reasoning": str(obj.get("reasoning", "")),
            }
    except Exception as e:
        logger.warning(f"gemini second-call failed: {e}")

    # 3) Minimal single-text fallback (still JSON MIME)
    mini_prompt = (
        "Return ONLY one JSON object with keys: direction, prob, support, resistance, reasoning.\n"
        "direction ∈ {long, short, hold}; prob 0..1.\n"
        f"DATA: {json.dumps(content, ensure_ascii=False)}"
    )
    mini_cfg = types.GenerateContentConfig(
        temperature=0,
        top_p=0,
        top_k=1,
        max_output_tokens=min(384, MAX_TOKENS),
        response_mime_type="application/json",
        system_instruction=types.Content(role="system", parts=[types.Part.from_text(text=system_note)]),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="NONE")
        ),
    )

    try:
        resp3 = client.models.generate_content(
            model=model_name,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=mini_prompt)])],
            config=mini_cfg,
        )
        _dump_candidates_debug(resp3, f"{datetime.now().strftime('%H%M%S')}_{symbol or content.get('pair')}_debug_parts_3rd.json")
        obj = getattr(resp3, "parsed", None) or _extract_from_parts(resp3) or _safe_json_extract(getattr(resp3, "text", "") or "")
        if obj:
            d = str(obj.get("direction", "hold")).lower()
            if d not in ("long", "short", "hold"):
                d = "hold"
            return {
                "direction": d,
                "prob": float(obj.get("prob", 0.5)),
                "support": float(obj.get("support", 0.0)),
                "resistance": float(obj.get("resistance", 0.0)),
                "reasoning": str(obj.get("reasoning", "")),
            }
    except Exception as e:
        logger.warning(f"gemini third-call failed: {e}")

    # 4) Micro-fallback (ultra-thin payload, JSON MIME)
    micro = _micro_payload(payload)
    micro_rule = (
        "Respond ONLY with this JSON: {\"direction\":one of [long,short,hold], \"prob\":number 0..1, \"support\":number, \"resistance\":number, \"reasoning\":string}."
    )
    micro_cfg = types.GenerateContentConfig(
        temperature=0,
        top_p=0,
        top_k=1,
        max_output_tokens=min(256, MAX_TOKENS),
        response_mime_type="application/json",
        system_instruction=types.Content(role="system", parts=[types.Part.from_text(text=system_note)]),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        tool_config=types.ToolConfig(
            function_calling_config=types.FunctionCallingConfig(mode="NONE")
        ),
    )
    try:
        resp4 = client.models.generate_content(
            model=model_name,
            contents=[types.Content(role="user", parts=[
                types.Part.from_text(text=micro_rule),
                types.Part.from_text(text=json.dumps(micro, ensure_ascii=False)),
            ])],
            config=micro_cfg,
        )
        _dump_candidates_debug(resp4, f"{datetime.now().strftime('%H%M%S')}_{symbol or micro.get('pair')}_debug_parts_4th.json")
        obj = getattr(resp4, "parsed", None) or _extract_from_parts(resp4) or _safe_json_extract(getattr(resp4, "text", "") or "")
        if obj:
            d = str(obj.get("direction", "hold")).lower()
            if d not in ("long", "short", "hold"):
                d = "hold"
            return {
                "direction": d,
                "prob": float(obj.get("prob", 0.5)),
                "support": float(obj.get("support", 0.0)),
                "resistance": float(obj.get("resistance", 0.0)),
                "reasoning": str(obj.get("reasoning", "")),
            }
    except Exception as e:
        logger.warning(f"gemini fourth-call failed: {e}")

    # 5) Nano fallback — plain text (no JSON MIME, no schema). Robust against rare JSON-mode stalls.
    nano_prompt = (
        "Return exactly one JSON object with keys: direction, prob, support, resistance, reasoning.\n"
        "direction in [long, short, hold]; prob 0..1.\n"
        f"DATA: {json.dumps(_micro_payload(payload), ensure_ascii=False)}"
    )
    try:
        resp5 = client.models.generate_content(
            model=model_name,
            contents=[types.Content(role="user", parts=[types.Part.from_text(text=nano_prompt)])],
            config=types.GenerateContentConfig(
                temperature=0,
                top_p=0,
                top_k=1,
                max_output_tokens=min(128, MAX_TOKENS),
                # intentionally no response_mime_type / no schema / no system_instruction
            ),
        )
        _dump_candidates_debug(resp5, f"{datetime.now().strftime('%H%M%S')}_{symbol or content.get('pair')}_debug_parts_5th.json")
        obj = _extract_from_parts(resp5) or _safe_json_extract(getattr(resp5, "text", "") or "")
        if obj:
            d = str(obj.get("direction", "hold")).lower()
            if d not in ("long", "short", "hold"):
                d = "hold"
            return {
                "direction": d,
                "prob": float(obj.get("prob", 0.5)),
                "support": float(obj.get("support", 0.0)),
                "resistance": float(obj.get("resistance", 0.0)),
                "reasoning": str(obj.get("reasoning", "")),
            }
    except Exception as e:
        logger.warning(f"gemini fifth-call failed: {e}")

    # Hard-fail fallback → HOLD
    return HOLD.copy()


def should_predict(df) -> bool:
    """
    Gate: use returns-based volatility directly.
    df['volatility'] is std of returns (dimensionless). Compare to MIN_VOL_FRAC directly.
    """
    try:
        if df is None or getattr(df, "empty", True):
            return False
        cols = set(getattr(df, "columns", []))
        if not {"close", "volatility"}.issubset(cols):
            return False

        # last_vol is std of returns over a rolling window (e.g., 20 bars)
        last_vol = float(df["volatility"].astype("float64").iloc[-1])

        # threshold as a pure fraction (e.g., 0.0005 = 0.05%)
        min_vol_frac = float(os.getenv("MIN_VOL_FRAC", "0.0005"))

        # if volatility >= threshold → proceed to predict
        return last_vol > max(1e-6, min_vol_frac)
    except Exception:
        # Be permissive if anything odd: allow prediction rather than blocking.
        return True


__all__ = ["get_gemini_prediction", "should_predict"]
