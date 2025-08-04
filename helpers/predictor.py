"""
predictor.py – Google GenAI SDK inference helper (migrated to google-genai)
"""

from __future__ import annotations

import json
import logging
import os
from typing import Any, Dict, List, Optional

import pandas as pd
from google import genai  # 변경: from google import genai (새 SDK import)
from google.genai import types  # 변경: types.GenerateContentConfig를 위한 import

from .utils import get_secret, LOG_DIR

logger = logging.getLogger(__name__)

# Constants
DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")  # 변경: 새 모델 이름으로 업데이트 (가이드 예시 참조, 필요 시 gemini-2.0-flash 등으로 변경)
MIN_MODEL_PROB = float(os.getenv("MIN_MODEL_PROB", "0.55"))

SYSTEM_INSTRUCTION = (
    "You are a trading signal generator. Return ONLY a strict JSON object: "
    '{"direction":"long|short|hold","prob":0..1,"support":number,"resistance":number,"reasoning":"..."}'
)

# 변경: types.GenerationConfig → types.GenerateContentConfig (이름 변경)
GEN_CFG = types.GenerateContentConfig(
    temperature=float(os.getenv("GEMINI_TEMPERATURE", "0.25")),
    top_p=float(os.getenv("GEMINI_TOP_P", "0.9")),
    top_k=int(os.getenv("GEMINI_TOP_K", "40")),
    max_output_tokens=int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024")),
)

# Lazy initialization for client (변경: 모델 대신 클라이언트 초기화)
_client: Optional[genai.Client] = None  # 변경: 모델 대신 Client 사용

def _init_client() -> Optional[genai.Client]:
    global _client
    if _client is None:
        api_key = get_secret("GOOGLE_API_KEY")
        if api_key:
            # 변경: genai.configure(api_key=...) 대신 genai.Client(api_key=...) 사용
            # 환경 변수 GEMINI_API_KEY를 우선 사용 가능 (가이드 추천)
            _client = genai.Client(api_key=api_key)  # 환경 변수가 설정되어 있으면 api_key 생략 가능
        else:
            logger.warning("GOOGLE_API_KEY not set – predictor will always return HOLD")
    return _client

# Helpers (기존과 유사)
def _to_content(payload: Dict[str, Any]) -> List[Dict[str, Any]]:
    """Build [system, user] messages for genai."""
    return [
        {"role": "model", "parts": [{"text": SYSTEM_INSTRUCTION}]},
        {"role": "user", "parts": [{"text": "Analyze the JSON below and respond with the required JSON only:\n\n" + json.dumps(payload, ensure_ascii=False)}]},
    ]

def _safe_json_extract(text: str) -> Dict[str, Any]:
    """Safely extract JSON from text."""
    try:
        start = text.find("{")
        end = text.rfind("}")
        if start >= 0 and end > start:
            return json.loads(text[start:end + 1])
    except json.JSONDecodeError as exc:
        logger.debug(f"_safe_json_extract failed: {exc}")
    return {}

def _normalize(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize the raw prediction dictionary."""
    direction = str(raw.get("direction", "hold")).lower()
    if direction not in ("long", "short", "hold"):
        direction = "hold"
    prob = max(0.0, min(1.0, float(raw.get("prob", 0.5))))
    return {
        "direction": direction if prob >= MIN_MODEL_PROB else "hold",
        "prob": prob,
        "support": float(raw.get("support", 0.0)),
        "resistance": float(raw.get("resistance", 0.0)),
        "reasoning": str(raw.get("reasoning", ""))[:2000],
    }

def _dump_payload(payload: Dict[str, Any]) -> None:
    """Dump payload to log file."""
    try:
        ts = pd.Timestamp.utcnow().strftime("%Y%m%dT%H%M%S%f")
        fn = os.path.join(LOG_DIR, f"gemini_payload_{ts}.json")
        with open(fn, "w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)
    except Exception as exc:
        logger.debug(f"Payload dump failed: {exc}")

# Main functions
def get_gemini_prediction(
    *,
    symbol: str,
    df_5m: Optional[pd.DataFrame] = None,
    extra_indicators: Optional[Dict[str, Any]] = None,
    sentiment_score: float = 0.0,
    funding_rate_pct: Optional[float] = None,
    times: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build JSON payload and call Gemini; return normalized dict.
    On any failure → direction=hold.
    """
    hold = {"direction": "hold", "prob": 0.0, "support": 0.0, "resistance": 0.0, "reasoning": ""}

    client = _init_client()  # 변경: 클라이언트 초기화 (모델 대신)
    if client is None:
        return hold

    # Build payload (기존과 동일)
    last_row = df_5m.iloc[-1].to_dict() if df_5m is not None and not df_5m.empty else {}
    payload = {
        "pair": symbol,
        "entry_5m": {
            "close": float(last_row.get("close", 0.0)),
            "rsi": float(last_row.get("RSI", 50.0)),
            "volatility": float(last_row.get("volatility", 0.0)),
            "sma20": float(last_row.get("SMA_20", 0.0)),
            "funding_rate_pct": float(funding_rate_pct or 0.0),
            "sentiment": float(sentiment_score or 0.0),
            "timestamp": str((times or {}).get("ohlcv") or last_row.get("timestamp") or ""),
        },
        "extra": dict(extra_indicators or {}),
        "times": times or {},
    }

    _dump_payload(payload)

    # Call GenAI (변경: model.generate_content → client.models.generate_content)
    try:
        contents = _to_content(payload)  # 기존 contents를 'contents' 매개변수로 사용
        # 변경: 모델 인스턴스 대신 클라이언트 메서드 호출, config 전달
        resp = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=contents,  # 변경: 'contents' 키워드 사용 (기존 'contents' 매개변수)
            config=GEN_CFG  # 변경: config=types.GenerateContentConfig
        )
        txt = ""
        if resp and resp.candidates:
            parts = resp.candidates[0].content.parts
            if parts and hasattr(parts[0], "text"):
                txt = parts[0].text
        raw = _safe_json_extract(txt)
        return _normalize(raw)
    except Exception as exc:
        logger.error(f"Gemini prediction error: {exc}")
        return hold

def should_predict(df: Optional[pd.DataFrame]) -> bool:
    """Skip if trailing volatility is negligible."""
    if df is None or df.empty:
        return False
    if not {"close", "volatility"}.issubset(df.columns):
        return False
    mean_close = pd.to_numeric(df["close"], errors="coerce").mean()
    last_vol = pd.to_numeric(df["volatility"], errors="coerce").iloc[-1]
    return mean_close > 0 and last_vol > mean_close * 0.001