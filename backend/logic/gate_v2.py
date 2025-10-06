from __future__ import annotations
"""Gate decision helper that relaxes MTF RSI mismatch for high-confidence signals."""

from dataclasses import dataclass
import os

GATE_PROB_HARD = float(os.getenv("GATE_PROB_HARD", "0.55"))
GATE_PROB_SUPER = float(os.getenv("GATE_PROB_SUPER", "0.75"))
RSI_SHORT_MAX = float(os.getenv("RSI_SHORT_MAX", "48"))
RSI_LONG_MIN = float(os.getenv("RSI_LONG_MIN", "52"))


@dataclass(slots=True)
class GateInputs:
    symbol: str
    direction: str
    prob: float
    ev_perc: float
    rr: float
    rr_req: float
    rsi_1h: float
    rsi_4h: float


@dataclass(slots=True)
class GateDecision:
    ok: bool
    reasons: list[str]


def _mtf_rsi_ok(direction: str, rsi_1h: float, rsi_4h: float, allow_mismatch: bool) -> bool:
    if direction == "short":
        c1 = rsi_1h <= RSI_SHORT_MAX
        c4 = rsi_4h <= RSI_SHORT_MAX
    else:
        c1 = rsi_1h >= RSI_LONG_MIN
        c4 = rsi_4h >= RSI_LONG_MIN
    return c1 if allow_mismatch else (c1 and c4)


def gate_decision(inp: GateInputs) -> GateDecision:
    reasons: list[str] = []
    if inp.prob < GATE_PROB_HARD:
        reasons.append("prob_below_threshold")

    allow_mismatch = (
        inp.prob >= GATE_PROB_SUPER and inp.ev_perc >= 0 and inp.rr >= inp.rr_req
    )

    if not _mtf_rsi_ok(inp.direction, inp.rsi_1h, inp.rsi_4h, allow_mismatch):
        reasons.append(f"mtf_rsi_mismatch(allow_mismatch={allow_mismatch})")

    if inp.ev_perc < 0:
        reasons.append(f"ev_below_threshold({inp.ev_perc:.4f}<0)")

    return GateDecision(ok=(len(reasons) == 0), reasons=reasons)
