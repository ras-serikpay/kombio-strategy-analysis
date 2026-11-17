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
    final_turn: use the calibrated ``p_call`` estimate: it simulates the final turn every
                opponent gets after a call, and models opponents' hidden cards as the lowest of
                ``opp_draws`` random cards (they have been keeping low cards all round).
                False reproduces the original, overconfident estimate.
    opp_draws : see ``final_turn``. 2 is what the calibration experiment supports.
    sabotage  : use 11/12 swaps to dump your worst card on the opponent closest to going out.
    scout     : aim 9/10 peeks at the opponent closest to going out.
    samples   : Monte Carlo samples for the ``p_call`` estimate.
    misremember : per-turn probability that a remembered card is remembered as the *wrong* value
                (off by 1-2). Wrong beliefs lead to wrong matches and penalty cards.
    take_discard: if set, take the top card of the discard pile instead of drawing when its value
                is <= this number and lower than your worst card. None = never.
    name      : label used in reports.
    """

    threshold: float = 8
    forget: float = 0.0
    p_call: float | None = None
    final_turn: bool = False
    opp_draws: int = 2
    sabotage: bool = False
    scout: bool = False
    samples: int = 120
    misremember: float = 0.0
    take_discard: int | None = None
    name: str = ""

    def label(self) -> str:
        if self.name:
            return self.name
        if self.p_call is not None:
            rule = f"P>={self.p_call}" + (" calibrated" if self.final_turn else "")
        else:
            rule = f"T<={self.threshold}"
        extras = "+sabotage" if self.sabotage else ""
        return f"{rule}{extras} (forget {self.forget:.0%})"


BASELINE_OPPONENT = Strategy(threshold=8, name="baseline (T<=8)")
BEST = Strategy(threshold=1, name="patient caller (T<=1)")
