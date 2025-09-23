# main.py
import os, time
from typing import List, Optional, Literal, Dict, Any
from fastapi import FastAPI, HTTPException, Header
from pydantic import BaseModel, Field
import numpy as np
import pandas as pd
import torch
import timesfm  # from google-research/timesfm (torch)

API_KEY = os.getenv("TSFM2_API_KEY", "").strip()
DEVICE = "cpu"  # Cloud Run CPU default
DT_DEFAULT_SEC = int(os.getenv("TSFM2_DT_SEC", "300"))  # 5m
MAX_CONTEXT = int(os.getenv("TSFM2_MAX_CONTEXT", "2048"))
MAX_HORIZON = int(os.getenv("TSFM2_MAX_HORIZON", "512"))
N_PATHS = int(os.getenv("TSFM2_MC_PATHS", "4000"))
SEED = int(os.getenv("TSFM2_SEED", "42"))
HF_REPO = os.getenv("TSFM2_HF_REPO", "google/timesfm-2.5-200m-pytorch")
PER_CORE_BATCH = int(os.getenv("TSFM2_PER_CORE_BATCH", "32"))

class ForecastIn(BaseModel):
    closes: List[float]
    freq: int = Field(0, description="legacy field; ignored by TimesFM 2.5")
    horizon_steps: int
    return_quantiles: bool = False

class Bracket(BaseModel):
    entry: float
    long: Optional[Dict[str, float]] = None  # {tp, sl}
    short: Optional[Dict[str, float]] = None

class ProbGateIn(BaseModel):
    closes: List[float]
    freq: int = 0
    dt_sec: int = DT_DEFAULT_SEC
    horizon_steps: int
    bracket: Bracket
    use_quantiles: bool = False  # TimesFM 2.0 quantiles (legacy compatibility)
    override_sigma: Optional[float] = None  # per-step sigma override
    atr_now: Optional[float] = None         # ATR for slippage guard
    n_paths: int = N_PATHS

class ForecastOut(BaseModel):
    point: List[float]
    quantiles: Optional[Dict[str, List[float]]] = None

class ProbGateOut(BaseModel):
    direction: Literal["long","short","hold"]
    prob: float
    diagnostics: Dict[str, Any]

app = FastAPI(title="TimesFM2 Cloud Run Service", version="1.0")

# ---- Load model at startup
def load_tsfm():
    model = timesfm.TimesFM_2p5_200M_torch()
    model.load_checkpoint(hf_repo_id=HF_REPO)
    fc = timesfm.ForecastConfig(
        max_context=MAX_CONTEXT,
        max_horizon=MAX_HORIZON,
        normalize_inputs=True,
        per_core_batch_size=PER_CORE_BATCH,
        use_continuous_quantile_head=True,
        force_flip_invariance=True,
        infer_is_positive=True,
        fix_quantile_crossing=True,
    )
    model.compile(fc)
    return model

TSFM = load_tsfm()
np.random.seed(SEED)
torch.manual_seed(SEED)

# Precompute quantile names once to avoid per-request loops
_QUANT_NAMES = [f"q{i}" for i in range(10)]

def _auth(x_api_key: Optional[str]):
    if API_KEY and (x_api_key or "").strip() != API_KEY:
        raise HTTPException(status_code=403, detail="forbidden")

def _ensure_horizon(horizon_steps: int) -> int:
    if horizon_steps <= 0:
        raise HTTPException(400, "horizon_steps must be > 0")
    max_supported = getattr(TSFM, "forecast_config", None)
    if max_supported and horizon_steps > max_supported.max_horizon:
        raise HTTPException(400, f"horizon_steps must be <= {max_supported.max_horizon}")
    return horizon_steps

def _point_forecast(closes: List[float], freq: int, horizon_steps: int, want_q=False):
    horizon = _ensure_horizon(horizon_steps)
    series = [np.array(closes, dtype=np.float32)]
    point_arr, quant_arr = TSFM.forecast(horizon, series)
    point = point_arr[0].astype(float).tolist()
    quants = None
    if want_q and quant_arr is not None and quant_arr.ndim == 3:
        q_count = quant_arr.shape[2]
        names = _QUANT_NAMES[:q_count]
        quants = {}
        for idx, name in enumerate(names):
            quants[name] = quant_arr[0, :horizon, idx].astype(float).tolist()
    return point, quants

def _sigma_from_data(closes: np.ndarray) -> float:
    rets = np.diff(closes) / np.maximum(1e-12, closes[:-1])
    if len(rets) < 20:
        return float(np.std(rets)) if len(rets) else 0.0
    return float(np.std(rets[-200:]))

def _sigma_fallback(atr_now: Optional[float], last_price: float) -> float:
    if atr_now and last_price > 0:
        return float((atr_now / last_price))
    return 0.0

def _first_passage_prob(direction: str, entry: float, tp: float, sl: float,
                        point_path: np.ndarray, sigma_step: float, n_paths: int) -> float:
    """Monte Carlo first-passage probability using point forecast as drift."""
    H = len(point_path)
    if H == 0 or entry <= 0 or tp <= 0 or sl <= 0:
        return 0.5
    means = np.empty(H, dtype=np.float32)
    prev = entry
    for t in range(H):
        means[t] = point_path[t] - prev
        prev = point_path[t]
    wins = 0
    for _ in range(max(1, n_paths)):
        px = entry
        hit_tp = False
        hit_sl = False
        for t in range(H):
            inc = np.random.normal(loc=means[t], scale=sigma_step * px)
            px = px + inc
            if direction == "long":
                if px >= tp:
                    hit_tp = True
                    break
                if px <= sl:
                    hit_sl = True
                    break
            else:
                if px <= tp:
                    hit_tp = True
                    break
                if px >= sl:
                    hit_sl = True
                    break
        if hit_tp and (not hit_sl):
            wins += 1
    return float(wins / max(1, n_paths))

@app.get("/health")
def health():
    cfg = getattr(TSFM, "forecast_config", None)
    horizon = cfg.max_horizon if cfg else MAX_HORIZON
    return {"status": "ok", "model": "timesfm-2.5-200m", "max_horizon": horizon}

@app.post("/v1/forecast", response_model=ForecastOut)
def v1_forecast(inp: ForecastIn, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    if len(inp.closes) < 16:
        raise HTTPException(400, "need >=16 closes")
    closes = inp.closes[-MAX_CONTEXT:]
    point, quants = _point_forecast(closes, inp.freq, inp.horizon_steps, inp.return_quantiles)
    return {"point": point, "quantiles": quants}

@app.post("/v1/prob_gate", response_model=ProbGateOut)
def v1_prob_gate(inp: ProbGateIn, x_api_key: Optional[str] = Header(None)):
    _auth(x_api_key)
    closes = np.array(inp.closes[-MAX_CONTEXT:], dtype=np.float32)
    if closes.size < 32:
        raise HTTPException(400, "need >=32 closes")
    H = int(inp.horizon_steps)
    point, _ = _point_forecast(closes.tolist(), inp.freq, H, want_q=False)

    sigma_data = _sigma_from_data(closes)
    sigma_atr = _sigma_fallback(inp.atr_now, float(closes[-1]))
    sigma = float(inp.override_sigma if inp.override_sigma is not None else max(sigma_data, sigma_atr, 1e-6))

    entry = float(inp.bracket.entry)
    res: Dict[str, Any] = {"sigma_step": sigma, "horizon_steps": H}

    cand = []
    if inp.bracket.long and inp.bracket.long.get("tp") and inp.bracket.long.get("sl"):
        pL = _first_passage_prob("long", entry, float(inp.bracket.long["tp"]), float(inp.bracket.long["sl"]),
                                 np.array(point, dtype=np.float32), sigma, int(inp.n_paths))
        cand.append(("long", pL))
        res["prob_long"] = pL
    if inp.bracket.short and inp.bracket.short.get("tp") and inp.bracket.short.get("sl"):
        pS = _first_passage_prob("short", entry, float(inp.bracket.short["tp"]), float(inp.bracket.short["sl"]),
                                 np.array(point, dtype=np.float32), sigma, int(inp.n_paths))
        cand.append(("short", pS))
        res["prob_short"] = pS

    if not cand:
        return {"direction": "hold", "prob": 0.5, "diagnostics": res}

    cand.sort(key=lambda x: x[1], reverse=True)
    direction, prob = cand[0]
    return {"direction": direction, "prob": float(max(0.0, min(1.0, prob))), "diagnostics": res}
