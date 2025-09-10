from __future__ import annotations
from typing import Dict, Any, List
import os, json, logging, urllib.request

logger = logging.getLogger(__name__)
logger.setLevel(logging.INFO)

TSFM_ENDPOINT_URL   = os.getenv("TSFM_ENDPOINT_URL", "").strip()
TSFM_ENDPOINT_FIELD = os.getenv("TSFM_ENDPOINT_FIELD", "predictions").strip()
TSFM_API_KEY        = os.getenv("TSFM_API_KEY", "").strip()

def call_tsfm_remote(series: List[float], horizon: int, quantiles=(0.05, 0.5, 0.95)) -> Dict[str, Any]:
    if not TSFM_ENDPOINT_URL:
        raise RuntimeError("TSFM_ENDPOINT_URL not set")
    payload = {
        "inputs": [{"target": series}],
        "parameters": {"prediction_length": int(horizon), "quantiles": list(quantiles)}
    }
    data = json.dumps(payload).encode("utf-8")
    headers = {"Content-Type": "application/json"}
    if TSFM_API_KEY:
        headers["X-API-Key"] = TSFM_API_KEY  # 서버에서 동일 키로 검증 가능

    req = urllib.request.Request(TSFM_ENDPOINT_URL.rstrip("/") + "/predict", data=data, headers=headers, method="POST")
    try:
        with urllib.request.urlopen(req, timeout=20) as resp:
            if resp.status // 100 != 2:
                raise RuntimeError(f"TSFM predict HTTP {resp.status}")
            res = json.loads(resp.read().decode("utf-8", errors="ignore"))
    except Exception as e:
        logger.info("TSFM remote call failed: %s", e)
        raise

    # 허용 스키마: {"predictions":[[q05],[q50],[q95]]} 또는 {"q05":[...],"q50":[...],"q95":[...]}
    if isinstance(res, dict):
        if TSFM_ENDPOINT_FIELD in res and isinstance(res[TSFM_ENDPOINT_FIELD], list):
            preds = res[TSFM_ENDPOINT_FIELD]
            if len(preds) >= 3 and all(isinstance(preds[i], list) for i in range(3)):
                return {"q05": preds[0], "q50": preds[1], "q95": preds[2]}
        if all(k in res for k in ("q05", "q50", "q95")):
            return {"q05": res["q05"], "q50": res["q50"], "q95": res["q95"]}
    raise RuntimeError("TSFM bad response schema")

