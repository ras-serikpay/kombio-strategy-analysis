"""Deck composition and scoring rules."""
from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

#: 70 cards: -1 x3, 0 x3, 1..12 x5, 13 x2, 14 x2
DECK_VALUES: tuple[int, ...] = tuple(
    [-1] * 3 + [0] * 3 + [v for v in range(1, 13) for _ in range(5)] + [13] * 2 + [14] * 2
)
DECK_TOTAL = sum(DECK_VALUES)

#: Cards drawn from the deck and discarded directly trigger these abilities.
ABILITIES = {
    7: "peek_own", 8: "peek_own",
    9: "peek_opponent", 10: "peek_opponent",
    11: "blind_swap", 12: "blind_swap",
    13: "look_and_swap",
    14: "look_two_and_swap",
}

CALLER_FAIL_PENALTY = 15
OTHERS_PENALTY_WHEN_CALLER_WINS = 10

#: Throwing a card that turns out not to match: the card stays where it was (now face up for a
#: moment, so everyone learns it) and the thrower draws this many penalty cards face down.
#: Assumption -- check the rulebook and adjust.
WRONG_MATCH_PENALTY_CARDS = 1


@dataclass(slots=True)
class Card:
    value: int
    uid: int  # unique id so knowledge can follow a card when it is swapped


def hand_score(values: Sequence[int]) -> int:
    """Hand total including the two 14-card bonuses from the rulebook."""
    vals = list(values)
    if sorted(vals) == [14, 14]:
        return -15
    if vals.count(14) == 1 and len(vals) > 1 and sum(vals) - 14 == 25:
        return -10
    return sum(vals)


def round_scores(totals: Sequence[int], caller: int | None) -> list[int]:
    """Apply the KOMBIO calling bonus/penalty to raw hand totals."""
    if caller is None:
        return list(totals)
    caller_wins = all(totals[caller] < t for j, t in enumerate(totals) if j != caller)
    if caller_wins:
        return [t + (0 if j == caller else OTHERS_PENALTY_WHEN_CALLER_WINS) for j, t in enumerate(totals)]
    return [t + (CALLER_FAIL_PENALTY if j == caller else 0) for j, t in enumerate(totals)]
