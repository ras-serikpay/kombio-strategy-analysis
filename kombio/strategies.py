"""Player strategies: how a player calls, uses abilities and remembers cards."""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True, slots=True)
class Strategy:
    """Parameters of a simulated player.

    threshold : call KOMBIO when the estimated hand total <= threshold.
    forget    : per-turn probability of forgetting each remembered card (0 = perfect memory).
    p_call    : if set, call when the Monte Carlo estimate of P(strictly lowest) >= p_call
                (replaces ``threshold``).
    sabotage  : use 11/12 swaps to dump your worst card on the opponent closest to going out.
    scout     : aim 9/10 peeks at the opponent closest to going out.
    samples   : Monte Carlo samples for the ``p_call`` estimate.
    name      : label used in reports.
    """

    threshold: float = 8
    forget: float = 0.0
    p_call: float | None = None
    sabotage: bool = False
    scout: bool = False
    samples: int = 120
    name: str = ""

    def label(self) -> str:
        if self.name:
            return self.name
        rule = f"P>={self.p_call}" if self.p_call is not None else f"T<={self.threshold}"
        extras = "+sabotage" if self.sabotage else ""
        return f"{rule}{extras} (forget {self.forget:.0%})"


BASELINE_OPPONENT = Strategy(threshold=8, name="baseline (T<=8)")
BEST = Strategy(threshold=1, name="patient caller (T<=1)")
