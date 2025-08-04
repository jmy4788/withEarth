import json
import logging
import os
from typing import Any, Dict, Optional

import pandas as pd
from google import genai
from .utils import get_secret

logger = logging.getLogger(__name__)

DEFAULT_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-pro")
TEMPERATURE = float(os.getenv("GEMINI_TEMPERATURE", "0.25"))
TOP_P = float(os.getenv("GEMINI_TOP_P", "0.9"))
TOP_K = int(os.getenv("GEMINI_TOP_K", "40"))
MAX_OUTPUT_TOKENS = int(os.getenv("GEMINI_MAX_OUTPUT_TOKENS", "1024"))

SYSTEM_INSTRUCTION = (
    "You are a trading signal generator. "
    "Read the compact JSON payload and respond with a strict JSON object: "
    '{"direction":"long|short|hold","prob":0..1,"support":number,"resistance":number,"reasoning":"..."} '
    "No extra text outside JSON."
)

def _build_user_content(payload: Dict[str, Any]) -> list[dict]:
    """
    Return a list of 'Content' for generate_content(). **Only** 'user' role.
    """
    return [
        {
            "role": "user",
            "parts": [
                {
                    "text": (
                        "Analyze the following JSON and output a strict JSON decision.\n\n"
                        + json.dumps(payload, ensure_ascii=False)
                    )
                }
            ],
        }
    ]

def _extract_json(text: str) -> Dict[str, Any]:
    try:
        # 가장 마지막 JSON 오브젝트를 파싱(모델이 앞에 텍스트를 붙였을 경우 대비)
        start = text.rfind("{")
        end = text.rfind("}")
        if start >= 0 and end >= start:
            return json.loads(text[start : end + 1])
    except Exception as e:
        logger.warning(f"JSON parse failed: {e}; raw: {text[:200]}")
    return {}

def _postprocess(raw: Dict[str, Any]) -> Dict[str, Any]:
    direction = str(raw.get("direction", "hold")).lower()
    prob = float(raw.get("prob", 0.5))
    support = float(raw.get("support", 0.0))
    resistance = float(raw.get("resistance", 0.0))
    reasoning = str(raw.get("reasoning", ""))[:2000]
    if direction not in ("long", "short", "hold"):
        direction = "hold"
    prob = max(0.0, min(1.0, prob))
    return {
        "direction": direction,
        "prob": prob,
        "support": support,
        "resistance": resistance,
        "reasoning": reasoning,
    }

def get_gemini_prediction(
    *,
    symbol: str,
    df_5m: Optional[pd.DataFrame],
    extra_indicators: Optional[Dict[str, Any]] = None,
    sentiment_score: float = 0.0,
    orderbook_data: Optional[Dict[str, Any]] = None,
    funding_rate_pct: Optional[float] = None,
    times: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Build a compact payload and call Gemini 2.5 Pro **with user-role only** contents.
    Returns normalized dict: {direction, prob, support, resistance, reasoning}.
    """
    try:
        last = (df_5m.iloc[-1].to_dict() if (df_5m is not None and not df_5m.empty) else {})
        entry_row = {
            "close": float(last.get("close", 0.0) or 0.0),
            "rsi": float(last.get("RSI", 50.0) or 50.0),
            "volatility": float(last.get("volatility", 0.0) or 0.0),
            "sma20": float(last.get("SMA_20", 0.0) or 0.0),
            "orderbook_imbalance": float(last.get("orderbook_imbalance", 0.0) or 0.0),
            "orderbook_spread": float(last.get("orderbook_spread", 0.0) or 0.0),
            "funding_rate_pct": float(funding_rate_pct or 0.0),
            "sentiment": float(sentiment_score or 0.0),
            "timestamp": str((times or {}).get("ohlcv") or last.get("timestamp") or ""),
        }

        payload = {
            "pair": symbol,
            "entry_5m": entry_row,
            "extra": dict(extra_indicators or {}),
            "times": times or {},
            "constraints": {
                "spread_cap_bps": float(os.getenv("PROMPT_SPREAD_CAP_BPS", "15")),
                "slippage_cap_bps": float(os.getenv("PROMPT_SLIPPAGE_CAP_BPS", "25")),
                "min_model_prob": float(os.getenv("MIN_MODEL_PROB", "0.55")),
            },
        }

        api_key = get_secret("GOOGLE_API_KEY")
        if not api_key:
            logger.warning("GOOGLE_API_KEY missing → hold")
            return {"direction": "hold", "prob": 0.5, "support": 0.0, "resistance": 0.0, "reasoning": "no_api_key"}

        client = genai.Client(api_key=api_key)
        contents = _build_user_content(payload)

        # **중요**: contents는 user/model만 허용. system 프롬프트는 system_instruction로 전달.
        response = client.models.generate_content(
            model=DEFAULT_MODEL,
            contents=contents,
            generation_config={
                "temperature": TEMPERATURE,
                "top_p": TOP_P,
                "top_k": TOP_K,
                "max_output_tokens": MAX_OUTPUT_TOKENS,
            },
            system_instruction=SYSTEM_INSTRUCTION,
        )

        text = getattr(response, "text", None) or ""
        raw = _extract_json(text)
        out = _postprocess(raw)

        if out["prob"] < float(os.getenv("MIN_MODEL_PROB", "0.55")):
            out["direction"] = "hold"
        return out

    except Exception as exc:
        logger.error(f"gemini_prediction error: {exc}")
        return {"direction": "hold", "prob": 0.5, "support": 0.0, "resistance": 0.0, "reasoning": "exception"}


def should_predict(df: Optional[pd.DataFrame]) -> bool:
    """
    간단 휴리스틱: 최근 변동성이 너무 작으면 예측 생략.
    - df가 None이거나 비어 있으면 False.
    - 'close'와 'volatility' 컬럼이 없으면 False.
    - 마지막 volatility가 평균 close 대비 0.1% 미만이면 False.
    """
    if df is None or df.empty:
        return False
    required_cols = {"close", "volatility"}
    if not required_cols.issubset(df.columns):
        return False
    mean_close = float(pd.to_numeric(df["close"], errors="coerce").dropna().mean())
    last_vol = float(pd.to_numeric(df["volatility"], errors="coerce").dropna().iloc[-1])
    if mean_close <= 0:
        return False
    return last_vol > mean_close * 0.001


