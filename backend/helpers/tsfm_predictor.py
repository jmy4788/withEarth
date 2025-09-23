from __future__ import annotations
from typing import Any, Dict, Optional, Tuple
import os, math, json, random
import logging

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

# --------------- CONFIG ---------------
# 선택지 1) 원격 엔드포인트(권장: SageMaker/Vertex)
TSFM_ENDPOINT_URL   = os.getenv("TSFM_ENDPOINT_URL", "").strip()
TSFM_ENDPOINT_FIELD = os.getenv("TSFM_ENDPOINT_FIELD", "predictions").strip()

# 선택지 2) 로컬 HuggingFace (개발/별도 워커 권장)
TSFM_MODEL_NAME     = os.getenv("TSFM_MODEL_NAME", "amazon/chronos-bolt-tiny").strip()
TSFM_BACKEND        = os.getenv("TSFM_BACKEND", "CHRONOS").upper()  # CHRONOS | TIMESFM
TSFM_DEVICE         = os.getenv("TSFM_DEVICE", "cpu").strip()
TSFM_MAX_H          = int(os.getenv("TSFM_PRED_LEN", "30"))

# 경로 샘플링 수
N_PATHS = int(os.getenv("TSFM_N_PATHS", "512"))


def _safe_float(x, d=0.0):
    try:
        return float(x)
    except Exception:
        return float(d)


def _entry_and_brackets(payload: Dict[str, Any]):
    br = payload.get("brackets") or payload.get("bracket") or {}
    entry = _safe_float((br.get("entry") or (payload.get("entry_5m") or {}).get("close")))
    longb = br.get("long") or {}
    shortb = br.get("short") or {}
    hz_min = int(payload.get("horizon_min", 30))
    return entry, longb, shortb, hz_min


def _h_steps(hz_min: int, base_tf_min: int = 5) -> int:
    return max(1, int(round(hz_min / base_tf_min)))


def _normal_from_quantiles(q05: float, q50: float, q95: float) -> Tuple[float, float]:
    # 분위 5/50/95에서 N(μ,σ^2) 근사
    sigma = max(1e-12, (q95 - q05) / (2.0 * 1.645))
    mu = q50
    return mu, sigma


def _path_hits(entry: float, tp: float, sl: float, step_returns: list[float]) -> int:
    p = entry
    for r in step_returns:
        p = max(1e-12, p * (1.0 + r))
        if p >= tp:
            return 1
        if p <= sl:
            return 0
    return -1


def _simulate_first_hit(mu: float, sigma: float, entry: float, tp: float, sl: float, steps: int, n: int) -> float:
    tp_win = 0
    undecided = 0
    for _ in range(n):
        step_mu = mu / steps
        step_sigma = sigma / (steps ** 0.5)
        step_returns = [random.gauss(step_mu, step_sigma) for _ in range(steps)]
        res = _path_hits(entry, tp, sl, step_returns)
        if res == 1:
            tp_win += 1
        elif res == -1:
            undecided += 1
    total = max(1, n)
    return float(tp_win) / float(total)


def _predict_quantiles_remote(inputs, params) -> Optional[Dict[str, Any]]:
    """
    Call Cloud Run TSFM endpoint using helpers.tsfm_remote_client,
    and normalize to legacy shape: {TSFM_ENDPOINT_FIELD: [[q05],[q50],[q95]]}
    """
    if not TSFM_ENDPOINT_URL:
        return None
    try:
        from .tsfm_remote_client import call_tsfm_remote  # type: ignore
        # Inputs: [{"target": series}], params: {prediction_length, quantiles}
        if not inputs or not isinstance(inputs, list):
            return None
        series = inputs[0].get("target") if isinstance(inputs[0], dict) else None
        if not isinstance(series, list) or len(series) == 0:
            return None
        H = int((params or {}).get("prediction_length", 1))
        qs = (params or {}).get("quantiles", [0.05, 0.5, 0.95])
        res = call_tsfm_remote(series, horizon=H, quantiles=qs)
        q05 = res.get("q05") or []
        q50 = res.get("q50") or []
        q95 = res.get("q95") or []
        return {TSFM_ENDPOINT_FIELD: [[q05, q50, q95]]}
    except Exception as e:
        logger.info("TSFM remote error: %s", e)
        return None


def _predict_quantiles_local_chronos(context: list[float], pred_len: int) -> Optional[list[list[float]]]:
    try:
        import torch  # type: ignore
        from chronos import BaseChronosPipeline  # type: ignore
    except Exception as e:
        logger.info("chronos import failed: %s", e)
        return None
    try:
        pipe = BaseChronosPipeline.from_pretrained(
            TSFM_MODEL_NAME,
            device_map=TSFM_DEVICE,
            torch_dtype=(torch.bfloat16 if TSFM_DEVICE != "cpu" else torch.float32),
        )
        out = pipe.predict(context=torch.tensor(context), prediction_length=int(pred_len))
        if hasattr(out, "tolist"):
            out = out.tolist()
        return out
    except Exception as e:
        logger.info("chronos predict failed: %s", e)
        return None


def get_tsfm_prediction(payload: Dict[str, Any], symbol: str = "") -> Dict[str, Any]:
    """
    TS‑FM을 이용해 horizon 내 TP 선행 돌파 확률을 근사.
    출력 스키마는 LLM 결과와 동일: {direction, prob, support, resistance, reasoning}
    실패 시 안전 폴백: hold/0.5
    """
    try:
        entry, longb, shortb, hz = _entry_and_brackets(payload)
        if entry <= 0:
            return {"direction": "hold", "prob": 0.5, "reasoning": "invalid_entry"}
        steps = min(_h_steps(hz), TSFM_MAX_H)

        # 1) 과거 종가 시퀀스 (payload에서)
        seq = payload.get("price_sequence") or []
        seq = [float(x) for x in seq if x is not None]
        if len(seq) < max(10, steps):
            seq = [entry] * max(10, steps)

        # 2) quantile 예측: 원격 → 로컬
        q_fore = None
        if TSFM_ENDPOINT_URL:
            obj = _predict_quantiles_remote(
                [{"target": seq}], {"prediction_length": steps, "quantiles": [0.05, 0.50, 0.95]}
            )
            if isinstance(obj, dict) and TSFM_ENDPOINT_FIELD in obj:
                q_fore = obj[TSFM_ENDPOINT_FIELD]
                # Normalize shape: allow [[q05,q50,q95]] or [q05,q50,q95]
                if isinstance(q_fore, list) and q_fore and all(isinstance(x, list) for x in q_fore) and len(q_fore) == 3:
                    q_fore = [q_fore]
        if q_fore is None and TSFM_BACKEND == "CHRONOS":
            out = _predict_quantiles_local_chronos(seq, pred_len=steps)
            if out and isinstance(out, list):
                q_fore = out

        if not q_fore:
            return {"direction": "hold", "prob": 0.5, "reasoning": "tsfm_unavailable"}

        # 단일 시계열 가정
        try:
            q05 = [float(x) for x in q_fore[0][0]]
            q50 = [float(x) for x in q_fore[0][1]] if len(q_fore[0]) > 1 else q05
            q95 = [float(x) for x in q_fore[0][2]] if len(q_fore[0]) > 2 else q50
        except Exception:
            preds = q_fore if isinstance(q_fore, dict) else {}
            q05 = preds.get("q05") or preds.get("p05") or preds.get("q_0.05")
            q50 = preds.get("q50") or preds.get("p50") or preds.get("median")
            q95 = preds.get("q95") or preds.get("p95") or preds.get("q_0.95")
            if not (q05 and q50 and q95):
                return {"direction": "hold", "prob": 0.5, "reasoning": "tsfm_quantiles_missing"}

        mu_T, sigma_T = _normal_from_quantiles(q05[-1], q50[-1], q95[-1])

        def prob_hit(tp, sl) -> float:
            if tp <= 0 or sl <= 0:
                return 0.5
            return _simulate_first_hit(mu_T, sigma_T, entry, float(tp), float(sl), steps, N_PATHS)

        p_long = prob_hit(longb.get("tp", 0.0), longb.get("sl", 0.0))
        p_short = prob_hit(shortb.get("tp", 0.0), shortb.get("sl", 0.0))

        if max(p_long, p_short) < 0.52:
            return {"direction": "hold", "prob": 0.5, "reasoning": "weak_edge"}

        if p_long >= p_short:
            return {
                "direction": "long",
                "prob": float(p_long),
                "support": 0.0,
                "resistance": 0.0,
                "reasoning": f"tsfm_long p={p_long:.2f} (h={steps})",
            }
        else:
            return {
                "direction": "short",
                "prob": float(p_short),
                "support": 0.0,
                "resistance": 0.0,
                "reasoning": f"tsfm_short p={p_short:.2f} (h={steps})",
            }
    except Exception as e:
        logger.info("get_tsfm_prediction error: %s", e)
        return {"direction": "hold", "prob": 0.5, "reasoning": "tsfm_exception"}
