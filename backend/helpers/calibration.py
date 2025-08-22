# helpers/calibration.py
from __future__ import annotations
import json, os
from pathlib import Path
from typing import Dict, List
import numpy as np

# ---- LOG_DIR 일원화: helpers.utils의 LOG_DIR 사용 ----
try:
    from .utils import LOG_DIR as _LOG_DIR_PATH  # Path or str
    _LOG_DIR_STR = str(_LOG_DIR_PATH)
except Exception:
    _LOG_DIR_STR = os.getenv("LOG_DIR", "/tmp/trading_bot")  # 마지막 폴백

# 환경
_CAL_PATH = os.getenv("PROB_CALIBRATION_PATH", str(Path(_LOG_DIR_STR) / "calibration.json"))
_MIN_SAMPLES = int(os.getenv("CALIB_MIN_SAMPLES", "150"))
_BINS = int(os.getenv("CALIB_BINS", "10"))

def _safe_load(path: str) -> Dict:
    try:
        p = Path(path)
        if p.exists():
            return json.loads(p.read_text(encoding="utf-8"))
    except Exception:
        pass
    return {}

def _safe_save(path: str, obj: Dict) -> None:
    try:
        p = Path(path)
        p.parent.mkdir(parents=True, exist_ok=True)
        p.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")
    except Exception:
        pass

def _interp(x: float, xs: List[float], ys: List[float]) -> float:
    if not xs or not ys or len(xs) != len(ys):
        return x
    x = max(0.0, min(1.0, float(x)))
    for i in range(1, len(xs)):
        if x <= xs[i]:
            x0, x1 = xs[i-1], xs[i]
            y0, y1 = ys[i-1], ys[i]
            if x1 == x0:
                return float(y0)
            w = (x - x0) / (x1 - x0)
            return float(y0 + w * (y1 - y0))
    return float(ys[-1])

class ProbCalibrator:
    """
    간단 신뢰도 곡선 기반 보정.
    파일에 {bin_edges:[...], bin_means:[...]} 저장/로드.
    """
    def __init__(self, path: str = _CAL_PATH, bins: int = _BINS, min_samples: int = _MIN_SAMPLES):
        self.path = path
        self.bins = max(4, int(bins))
        self.min_samples = max(50, int(min_samples))
        self.bin_edges: List[float] = []
        self.bin_means: List[float] = []
        self._load()

    def _load(self):
        obj = _safe_load(self.path)
        be = obj.get("bin_edges") or []
        bm = obj.get("bin_means") or []
        if isinstance(be, list) and isinstance(bm, list) and len(be) == len(bm):
            self.bin_edges = [float(x) for x in be]
            self.bin_means = [float(y) for y in bm]

    def save(self):
        if self.bin_edges and self.bin_means:
            _safe_save(self.path, {"bin_edges": self.bin_edges, "bin_means": self.bin_means})

    def fit_from_arrays(self, probs: List[float], labels: List[int]) -> bool:
        """
        labels: 1=승리(TP 우선), 0=패배(SL/타임아웃 등).
        데이터 부족/단조 위반 시 False.
        """
        n = min(len(probs), len(labels))
        if n < self.min_samples:
            return False
        p = np.clip(np.asarray(probs[:n], dtype=float), 0.0, 1.0)
        y = np.asarray(labels[:n], dtype=int)

        # 등폭 bin
        edges = np.linspace(0, 1, self.bins + 1)
        means_x, means_y = [], []
        for i in range(self.bins):
            lo, hi = edges[i], edges[i+1]
            mask = (p >= lo) & (p < hi if i < self.bins - 1 else p <= hi)
            if mask.sum() < 5:
                continue
            px = float(p[mask].mean())
            py = float(y[mask].mean())
            means_x.append(px)
            means_y.append(py)
        if len(means_x) < 3:
            return False

        # 단조 비감소 강제
        mono = []
        last = 0.0
        for v in means_y:
            last = max(last, float(v))
            mono.append(last)

        self.bin_edges = means_x
        self.bin_means = mono
        self.save()
        return True

    def calibrate(self, prob: float) -> float:
        if not self.bin_edges or not self.bin_means:
            return float(np.clip(prob, 0.0, 1.0))
        return float(np.clip(_interp(prob, self.bin_edges, self.bin_means), 0.0, 1.0))

# 전역 인스턴스(로드만)
_CAL = ProbCalibrator()

def calibrate_prob(prob: float) -> float:
    return _CAL.calibrate(prob)

def reload_global() -> None:
    """cron 캘리브레이션 직후 메모리 테이블을 핫리로드."""
    try:
        _CAL._load()
    except Exception:
        pass

__all__ = ["ProbCalibrator", "calibrate_prob", "reload_global"]
