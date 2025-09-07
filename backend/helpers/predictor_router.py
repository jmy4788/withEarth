from __future__ import annotations
from typing import Dict, Any
import os, logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# 기존 LLM 예측기
from .predictor import get_gemini_prediction as _llm_predict  # type: ignore


def get_prediction(payload: Dict[str, Any], symbol: str = "") -> Dict[str, Any]:
    """
    ENV PREDICTOR_BACKEND ∈ {GEMINI, TSFM}
    - TSFM 선택 시 helpers/tsfm_predictor.get_tsfm_prediction 호출
    - 실패/미가용 시 LLM으로 폴백
    """
    backend = (os.getenv("PREDICTOR_BACKEND", os.getenv("PREDICTOR_IMPL", "GEMINI")) or "GEMINI").upper()
    if backend.startswith("TSFM") or backend == "TSFM":
        try:
            from .tsfm_predictor import get_tsfm_prediction as _tsfm_predict  # type: ignore
            d = _tsfm_predict(payload, symbol=symbol)
            if not isinstance(d, dict) or "direction" not in d or "prob" not in d:
                raise RuntimeError("tsfm bad schema")
            return d
        except Exception as e:
            logger.info("TSFM backend failed, falling back to GEMINI: %s", e)
    return _llm_predict(payload, symbol=symbol)

