import os
import math
import json
from typing import Dict, Any, List, Tuple
from fastapi import FastAPI, Request, Response
from fastapi.responses import JSONResponse
import uvicorn

# TimesFM 2.0 (PyTorch) - requires: pip install 'timesfm[torch]==1.2.6'
import timesfm

APP_PORT = int(os.getenv("PORT", "8080"))
MODEL_REPO = os.getenv("TIMESFM_REPO", "google/timesfm-2.0-500m-pytorch")
BACKEND = os.getenv("TIMESFM_BACKEND", "cpu")  # "cpu" | "gpu"
CONTEXT_LEN = int(os.getenv("TIMESFM_CONTEXT_LEN", "2048"))

# Build model once on startup
# Horizon_len은 "최대 필요 길이"로 잡아도 되고, 요청마다 더 짧게 예측해도 됨.
DEFAULT_H = int(os.getenv("TIMESFM_DEFAULT_H", "128"))

tfm = timesfm.TimesFm(
    hparams=timesfm.TimesFmHparams(
        backend=BACKEND,
        context_len=CONTEXT_LEN,
        horizon_len=DEFAULT_H,
        per_core_batch_size=4,
        input_patch_len=32,
        output_patch_len=128,
        num_layers=50,
        model_dims=1280,
        use_positional_embedding=False,
    ),
    checkpoint=timesfm.TimesFmCheckpoint(huggingface_repo_id=MODEL_REPO),
)

app = FastAPI()

@app.get("/")
def root():
    return {"ok": True, "model": MODEL_REPO, "backend": BACKEND}

@app.get("/health")
@app.get("/healthz")
def health():
    return {"status": "ok"}

def _nearest_quantile_indices(target_qs: List[float], model_qs: List[float]) -> List[int]:
    # TimesFM 2.0은 "10개 experimental quantile heads"를 제공(범위는 릴리스에 따라 변동 가능)
    # 실제 사용 가능한 모델 quantile 순서를 timesfm 라이브러리에서 노출하지 않으므로,
    # forecast 결과 텐서의 마지막 축을 0..n-1로 가정하고 선형 보간 대신 "근접 인덱스" 선택.
    # 보수적: 사후 캘리브레이션 전제로 사용.
    idxs = []
    for tq in target_qs:
        # 모델이 반환하는 quantile 리스트가 명시되지 않으면 균등격자 가정(예: 0.05..0.95)
        # 10개라면 [0.05,0.15,...,0.95]로 간주
        if len(model_qs) == 0:
            grid = [0.05 + 0.10*i for i in range(10)]
        else:
            grid = model_qs
        nearest = min(range(len(grid)), key=lambda i: abs(grid[i] - tq))
        idxs.append(nearest)
    return idxs

@app.post("/predict")
async def predict(req: Request):
    """
    입력 스키마(우리 클라 호환):
    {
      "inputs": [{"target":[...]}],
      "parameters": {"prediction_length": H, "quantiles":[0.05,0.5,0.95]}
    }
    출력 스키마(우리 클라 호환, TSFM_ENDPOINT_FIELD="predictions"):
    {"predictions":[ [q05_list], [q50_list], [q95_list] ]}
    """
    body = await req.json()
    inputs = body.get("inputs") or []
    params = (body.get("parameters") or {})
    if not inputs or not isinstance(inputs, list):
        return JSONResponse({"error": "inputs must be a non-empty list"}, status_code=400)

    series = inputs[0].get("target")
    if not isinstance(series, list) or len(series) < 16:
        return JSONResponse({"error": "inputs[0].target must be a numeric list (len>=16)"}, status_code=400)

    H = int(params.get("prediction_length", DEFAULT_H))
    req_quantiles = params.get("quantiles", [0.05, 0.5, 0.95])

    # TimesFM 주파수 인디케이터: 0(high: T, MIN, H, D), 1(weekly/monthly), 2(quarterly+)
    # 우리는 5분봉이므로 0 사용
    freq_ind = [0]

    # horizon_len은 hparams의 최대치와 달라도 forecast에서 처리 가능
    # point_forecast: (B, H), experimental_quantile_forecast: (B, H, Q)
    point_forecast, quantile_forecast = tfm.forecast(
        [series], freq=freq_ind, horizon_len=H
    )

    if quantile_forecast is None:
        # Quantile heads가 비활성인 경우: 정규 근사 생성 (보수적, 후속 캘리브레이션 권장)
        # sigma 추정은 간략화를 위해 역사적 1-step 절대변동의 IQR 기반으로 근사
        import numpy as np
        arr = np.asarray(series, dtype=float)
        diffs = np.abs(np.diff(arr))
        if len(diffs) < 8:
            diffs = np.ones(8) * (arr[-1] * 0.001)  # 긴급 폴백
        iqr = np.subtract(*np.percentile(diffs, [75, 25]))
        sigma = (iqr / 1.349) if iqr > 0 else max(np.std(diffs), 1e-6)
        mu_path = np.repeat(point_forecast[0], H)
        from scipy.stats import norm
        out = []
        for q in req_quantiles:
            out.append((mu_path + norm.ppf(q) * sigma).tolist())
        return {"predictions": out}

    # Quantile heads 사용: 마지막 축이 Q(개수≈10). 타겟 분위수에 근접한 인덱스를 고름.
    # 모델이 제공하는 실제 q그리드를 timesfm에서 노출하지 않으므로 균등격자 가정.
    Q = quantile_forecast.shape[-1]
    model_q_grid = [0.05 + 0.10*i for i in range(Q)]  # 예: 10개면 0.05..0.95
    idxs = _nearest_quantile_indices(req_quantiles, model_q_grid)

    # shape: (B, H, Q) -> (len(req_quantiles), H)
    selected = []
    for qi in idxs:
        selected.append(quantile_forecast[0, :H, qi].tolist())

    return {"predictions": selected}

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=APP_PORT)

