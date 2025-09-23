# tools/calibrate_from_trades.py
from __future__ import annotations
from dataclasses import dataclass
from pathlib import Path
from typing import List, Optional, Sequence, Union
import csv
from datetime import datetime, timezone

@dataclass
class Sample:
    prob: float
    label: int  # 1=TP 우선 승리, 0=SL/타임아웃/패배

def _to_float(x) -> float:
    try: return float(x)
    except Exception: return 0.0

def _norm(s: Optional[str]) -> str:
    return (s or "").strip().lower()

def _parse_iso(s: str) -> Optional[datetime]:
    try:
        return datetime.fromisoformat(str(s).replace("Z", "+00:00"))
    except Exception:
        return None

def load_samples(
    trades_csv_path: Union[str, Path],
    horizon_min: int = 30,
    only_symbols: Optional[Sequence[str]] = None,
    max_rows: int = 100000,
) -> List[Sample]:
    """
    trades.csv → (prob, label) 리스트.
    - label 판정 우선순위:
      1) status에 'closed_tp' 또는 'tp' 포함 → 1
      2) status에 'closed_sl'/'sl'/'time_exit' 포함 → 0
      3) 불분명하면 pnl>0 → 1 else 0
    - 'open' 상태 행은 제외.
    - exit_ts가 있으면 (exit_ts - timestamp) <= horizon_min 조건을 만족하는 표본만 채택.
      (없으면 보수적으로 포함)
    """
    only = set([s.upper() for s in (only_symbols or [])])
    path = Path(trades_csv_path)
    if not path.exists():
        return []

    rows: List[Sample] = []
    buf = []
    with open(path, "r", encoding="utf-8") as f:
        r = csv.DictReader(f)
        for row in r:
            buf.append(row)
            if len(buf) >= max_rows:
                break

    for row in buf:
        sym = (row.get("symbol") or "").upper()
        if only and sym not in only:
            continue

        status = _norm(row.get("status"))
        if status == "open":
            continue  # 미결제

        # 시간 호라이즌 필터
        t0 = _parse_iso(row.get("timestamp",""))
        t1 = _parse_iso(row.get("exit_ts",""))
        if t0 and t1:
            dt_min = (t1 - t0).total_seconds() / 60.0
            if horizon_min > 0 and dt_min > float(horizon_min):
                continue  # 호라이즌 초과 표본 제외 (TP/SL '먼저' 의미 근사)
        # prob
        prob = _to_float(row.get("prob_raw") or row.get("prob"))
        if not (0.0 <= prob <= 1.0):
            continue

        # 라벨
        if "closed_tp" in status or ("tp" in status and "stop" not in status):
            label = 1
        elif "closed_sl" in status or "stop" in status or "time_exit" in status:
            label = 0
        else:
            pnl = _to_float(row.get("pnl"))
            label = 1 if pnl > 0 else 0

        rows.append(Sample(prob=float(prob), label=int(1 if label else 0)))

    return rows

__all__ = ["Sample", "load_samples"]
