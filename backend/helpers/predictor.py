from __future__ import annotations

"""
helpers/predictor.py — Gemini migrate fix (2025-08-15, KST)

주요 변경점
- contents 배열에서 role:"system" 제거 (user/model만 허용)
- system_instruction은 config.system_instruction 문자열로 전달
- dict-contents 방식/스키마 강제/폴백/수선/디버그 덤프 로직 유지

공개 API
- get_gemini_prediction(payload: dict, symbol: str = "") -> dict
- should_predict(payload_or_df, *, min_vol_frac_env="MIN_VOL_FRAC") -> bool
"""

from typing import Any, Dict, Optional, List, Union
import base64
import json
import logging
import os
from datetime import datetime, timezone

# --- google-genai (1.28.0) ---
_GENAI_OK = True
try:
    from google import genai
    from google.genai import types
except Exception:
    _GENAI_OK = False

# --- shared utils ---
try:
    from .utils import get_secret, LOG_DIR, log_event  # type: ignore
except Exception:  # pragma: no cover
    def get_secret(name: str) -> Optional[str]:
        return os.getenv(name)
    LOG_DIR = os.path.join(os.getcwd(), "logs")

# =====================
# Config
# =====================
MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash-lite").strip()
TEMPERATURE: float = float(os.getenv("G_TEMPERATURE", "0.0"))
MAX_TOKENS: int = int(os.getenv("G_MAX_TOKENS", "512"))

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# =====================
# Client bootstrap
# =====================
def _get_client() -> Optional["genai.Client"]:
    if not _GENAI_OK:
        return None
    api_key = get_secret("GOOGLE_API_KEY") or os.getenv("GOOGLE_API_KEY")
    if not api_key:
        logger.warning("GOOGLE_API_KEY not configured.")
        return None
    try:
        return genai.Client(api_key=api_key)
    except Exception as e:
        logger.info("genai.Client init failed: %s", e)
        return None

# =====================
# Debug helpers
# =====================
def _mk_debug_dir() -> str:
    d = os.path.join(LOG_DIR, "payloads", datetime.now(tz=timezone.utc).strftime("%Y%m%d"))
    os.makedirs(d, exist_ok=True)
    return d

def _save_json(obj: Any, fname: str) -> None:
    try:
        base = _mk_debug_dir()
        path = os.path.join(base, fname)
        with open(path, "w", encoding="utf-8") as f:
            json.dump(obj, f, ensure_ascii=False, indent=2)
    except Exception as e:
        logger.info("save_json failed: %s", e)

def _dump_debug(name: str, content: Any) -> None:
    ts = datetime.now(tz=timezone.utc).strftime("%H%M%S")
    try:
        _save_json(content, f"{ts}_{name}.json")
    except Exception:
        pass

# =====================
# Prompt & schema
# =====================
def _system_note() -> str:
    return (
        "You are a cautious crypto futures signal assistant. "
        "Use ONLY the numeric features provided in the JSON. "
        "Output MUST be a single JSON object strictly matching the schema. "
        "Never include extra keys or text."
    )

def _user_intro(payload: Dict[str, Any]) -> str:
    pair = payload.get("pair", "")
    entry = payload.get("entry_5m", {}).get("close", 0.0)
    spread = payload.get("extra", {}).get("orderbook_spread", 0.0)
    rsi = payload.get("entry_5m", {}).get("rsi", 50.0)
    return (
        f"Pair={pair}, entry_close_5m={entry}, spread_bps={spread}, rsi_5m={rsi}. "
        "Decide using ONLY numeric features from the JSON below."
    )

def _response_schema() -> "types.Schema":
    return types.Schema(
        type=types.Type.OBJECT,
        properties={
            "direction": types.Schema(type=types.Type.STRING, enum=["long", "short", "hold"]),
            "prob": types.Schema(type=types.Type.NUMBER),
            "reasoning": types.Schema(type=types.Type.STRING),
            "support": types.Schema(type=types.Type.NUMBER),
            "resistance": types.Schema(type=types.Type.NUMBER),
        },
        required=["direction", "prob"],
    )


# =====================
# Response parsing utils
# =====================
def _safe_json_extract(text: str) -> Optional[Dict[str, Any]]:
    """느슨한 중괄호 매칭으로 텍스트에서 첫 번째 JSON 오브젝트 추출."""
    if not text:
        return None
    start = text.find("{")
    if start < 0:
        return None
    depth = 0
    for i in range(start, len(text)):
        ch = text[i]
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(text[start:i + 1])
                except Exception:
                    return None
    return None

def _parts_to_json(resp: Any) -> Optional[Dict[str, Any]]:
    """
    candidates[*].content.parts[*]에서 text/inline_data(application/json) 파싱.
    dict/typed object 둘 다 지원.
    """
    try:
        cands = getattr(resp, "candidates", []) or []
        if not cands and isinstance(resp, dict):
            cands = resp.get("candidates", []) or []

        for c in cands:
            content = getattr(c, "content", None)
            if content is None and isinstance(c, dict):
                content = c.get("content")
            parts = getattr(content, "parts", None)
            if parts is None and isinstance(content, dict):
                parts = content.get("parts", [])

            for p in parts or []:
                # 1) text
                text = None
                if hasattr(p, "text"):
                    text = getattr(p, "text", None)
                elif isinstance(p, dict):
                    text = p.get("text")
                if text and str(text).strip():
                    obj = _safe_json_extract(str(text))
                    if obj:
                        return obj
                # 2) inline_data (base64 json)
                inline = None
                if hasattr(p, "inline_data"):
                    inline = getattr(p, "inline_data", None)
                    mime = getattr(inline, "mime_type", "")
                    data = getattr(inline, "data", b"")
                    if mime == "application/json" and data:
                        try:
                            raw = base64.b64decode(data).decode("utf-8", "ignore")
                            return json.loads(raw)
                        except Exception:
                            pass
                elif isinstance(p, dict) and "inline_data" in p:
                    idata = p["inline_data"] or {}
                    mime = idata.get("mime_type", "")
                    data = idata.get("data", "")
                    if mime == "application/json" and data:
                        try:
                            raw = base64.b64decode(data).decode("utf-8", "ignore")
                            return json.loads(raw)
                        except Exception:
                            pass
        return None
    except Exception:
        return None

# =====================
# Contents (dict-based)
# =====================
def _contents(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """
    messages는 user/model만. system 지침은 config.system_instruction로 전달.
    """
    user_msg = {
        "role": "user",
        "parts": [
            {"text": _user_intro(payload)},
            {"text": "DATA_JSON:\n" + json.dumps(payload, ensure_ascii=False, separators=(",", ":"))},
        ],
    }
    return [user_msg]

# =====================
# Config factories
# =====================
def _cfg_json_schema() -> "types.GenerateContentConfig":
    return types.GenerateContentConfig(
        temperature=TEMPERATURE,
        max_output_tokens=MAX_TOKENS,
        top_p=0,
        top_k=1,
        response_mime_type="application/json",
        response_schema=_response_schema(),
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE")),
        system_instruction=_system_note(),  # ← 문자열 권장
    )

def _cfg_json_plain() -> "types.GenerateContentConfig":
    return types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=min(384, MAX_TOKENS),
        top_p=0,
        top_k=1,
        response_mime_type="application/json",
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE")),
        system_instruction=_system_note(),
    )

def _cfg_text_plain() -> "types.GenerateContentConfig":
    return types.GenerateContentConfig(
        temperature=0,
        max_output_tokens=min(384, MAX_TOKENS),
        top_p=0,
        top_k=1,
        response_mime_type="text/plain",
        automatic_function_calling=types.AutomaticFunctionCallingConfig(disable=True),
        tool_config=types.ToolConfig(function_calling_config=types.FunctionCallingConfig(mode="NONE")),
        system_instruction=_system_note(),
    )

# =====================
# Generation steps
# =====================
def _generate_with_schema(client: "genai.Client", payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = client.models.generate_content(model=MODEL, contents=_contents(payload), config=_cfg_json_schema())
    raw = getattr(resp, "output_text", "") or ""
    if raw.strip():
        try:
            return json.loads(raw)
        except Exception:
            pass
    obj = _parts_to_json(resp)
    return obj or {}

def _generate_json_plain(client: "genai.Client", payload: Dict[str, Any]) -> Dict[str, Any]:
    resp = client.models.generate_content(model=MODEL, contents=_contents(payload), config=_cfg_json_plain())
    raw = getattr(resp, "output_text", "") or ""
    if raw.strip():
        try:
            return json.loads(raw)
        except Exception:
            pass
    obj = _parts_to_json(resp)
    return obj or {}

def _generate_text_then_parse(client: "genai.Client", prompt: str) -> Dict[str, Any]:
    resp = client.models.generate_content(
        model=MODEL,
        contents=[{"role": "user", "parts": [{"text": prompt}]}],
        config=_cfg_text_plain(),
    )
    raw = getattr(resp, "output_text", "") or ""
    obj = _safe_json_extract(raw)
    return obj or {}

def _repair_via_model(client: "genai.Client", bad_json: Dict[str, Any]) -> Dict[str, Any]:
    prm = (
        "Repair this object to match the schema strictly and return JSON only.\n"
        "Schema keys: direction∈{long,short,hold}, prob∈[0,1], support, resistance, reasoning.\n"
        f"Object: {json.dumps(bad_json, ensure_ascii=False)}"
    )
    return _generate_text_then_parse(client, prm)

# =====================
# Sanitization
# =====================
def _sanitize_decision(d: Dict[str, Any]) -> Dict[str, Any]:
    out: Dict[str, Any] = {}
    try:
        direction = str(d.get("direction", "hold")).lower().strip()
    except Exception:
        direction = "hold"
    if direction not in ("long", "short", "hold"):
        direction = "hold"
    try:
        prob = float(d.get("prob", 0.5))
    except Exception:
        prob = 0.5
    prob = max(0.0, min(1.0, prob))

    out["direction"] = direction
    out["prob"] = prob
    out["reasoning"] = str(d.get("reasoning", ""))[:800]
    for k in ("support", "resistance"):
        try:
            out[k] = float(d.get(k)) if d.get(k) is not None else 0.0
        except Exception:
            out[k] = 0.0
    return out

# =====================
# Public API
# =====================
# --- predictor.py: top imports 주변에 추가 ---
def get_gemini_prediction(payload: Dict[str, Any], symbol: str = "") -> Dict[str, Any]:
    """
    구조화 JSON 의사결정 반환.
    - 우선 스키마 강제 호출(_cfg_json_schema)
    - 실패 시 JSON/plain → 텍스트 후 수선 순으로 폴백
    - Cloud Logging에 gemini.request / gemini.response 요약을 INFO로 남김
    """
    # 요청 요약 로그 (민감정보/대용량 방지)
    try:
        preview = {
            "pair": payload.get("pair"),
            "entry_5m": payload.get("entry_5m"),
            "mtf_keys": sorted(list((payload.get("extra") or {}).keys()))[:8],
        }
        log_event(
            "gemini.request",
            symbol=(symbol or payload.get("pair")),
            model=MODEL,
            payload_hint="payloads/YYYYMMDD/*_request.json",
            payload_preview=preview,
        )
    except Exception:
        pass

    # 디버그 덤프(파일): 전체 payload는 여기로
    _dump_debug(f"{symbol or 'unknown'}_payload", {"payload": payload, "model": MODEL})

    client = _get_client()
    if client is None:
        decision = {"direction": "hold", "prob": 0.5, "reasoning": "client_unavailable"}
        _dump_debug(f"{symbol or 'unknown'}_decision", decision)
        # 응답 요약 로그
        log_event(
            "gemini.response",
            symbol=(symbol or payload.get("pair")),
            direction=decision.get("direction"),
            prob=float(decision.get("prob", 0.0)),
            support=None,
            resistance=None,
            entry=(payload.get("entry_5m") or {}).get("close"),
        )
        return decision

    # 1) 스키마 강제 호출
    d: Dict[str, Any] = {}
    try:
        d = _generate_with_schema(client, payload)  # 내부에서 config=_cfg_json_schema() 사용
    except Exception as e:
        logging.getLogger(__name__).info("schema gen failed: %s", e)

    # 2) JSON/plain 폴백
    if not d:
        try:
            d = _generate_json_plain(client, payload)  # config=_cfg_json_plain()
        except Exception as e:
            logging.getLogger(__name__).info("json/plain gen failed: %s", e)

    # 3) 텍스트+파서 폴백
    if not d:
        try:
            mini = (
                "Return ONLY one JSON object with keys: direction, prob, support, resistance, reasoning.\n"
                "direction ∈ {long, short, hold}; prob ∈ [0,1].\n"
                f"DATA: {json.dumps(payload, ensure_ascii=False)}"
            )
            d = _generate_text_then_parse(client, mini)  # config=_cfg_text_plain()
        except Exception as e:
            logging.getLogger(__name__).info("text/plain gen failed: %s", e)

    # 4) 모델로 수선
    if not isinstance(d, dict) or "direction" not in d or "prob" not in d:
        try:
            d = _repair_via_model(client, d if isinstance(d, dict) else {"raw": str(d)})
        except Exception as e:
            logging.getLogger(__name__).info("repair failed: %s", e)

    decision = _sanitize_decision(d)
    _dump_debug(f"{symbol or 'unknown'}_decision", decision)

    # 응답 요약 로그
    try:
        log_event(
            "gemini.response",
            symbol=(symbol or payload.get("pair")),
            direction=decision.get("direction"),
            prob=float(decision.get("prob", 0.0)),
            support=decision.get("support"),
            resistance=decision.get("resistance"),
            entry=(payload.get("entry_5m") or {}).get("close"),
        )
    except Exception:
        pass

    return decision


def should_predict(payload_or_df: Union[Dict[str, Any], "pd.DataFrame"], *, min_vol_frac_env: str = "MIN_VOL_FRAC") -> bool:
    """
    간단 게이트: 변동성 기준으로 예측 진행 여부 판단.
    - payload(dict) → payload["entry_5m"]["volatility"] 또는 payload["extra"]["ATR_5m"]/close 근사
    - DataFrame → 'volatility' 컬럼(20-window 표준편차) 기준
    ENV:
      MIN_VOL_FRAC (default 0.0005 = 0.05%)
    """
    try:
        thr = float(os.getenv(min_vol_frac_env, "0.0005"))
    except Exception:
        thr = 0.0005

    # payload dict
    if isinstance(payload_or_df, dict):
        try:
            vol = float(payload_or_df.get("entry_5m", {}).get("volatility", 0.0))
            if vol and vol > max(1e-8, thr):
                return True
            close = float(payload_or_df.get("entry_5m", {}).get("close", 0.0))
            atr5 = float(payload_or_df.get("extra", {}).get("ATR_5m", 0.0))
            ratio = (atr5 / close) if (close and close > 0) else 0.0
            return ratio > max(1e-8, thr)
        except Exception:
            return True

    # pandas.DataFrame 분기
    try:
        import pandas as pd  # lazy
        df = payload_or_df  # type: ignore
        if df is None or len(df) == 0 or "volatility" not in df.columns:
            return True
        last_vol = float(pd.to_numeric(df["volatility"], errors="coerce").iloc[-1])
        return last_vol > max(1e-8, thr)
    except Exception:
        return True


__all__ = ["get_gemini_prediction", "should_predict"]
