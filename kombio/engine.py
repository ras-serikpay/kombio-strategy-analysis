"""Game engine: simulates one KOMBIO round (and full games) between strategy-driven players.

Rules modeled (official rulebook):
* 4 face-down cards each; every player peeks at 2 at the start.
* A turn: draw from the deck, then either swap the card into your hand (the replaced card is
  discarded with no ability) or discard it straight away and optionally use its ability (7-14).
* Matching: after any discard, a player who *knows* a card of the same value -- in their own hand
  or an opponent's -- may discard it. Matching an opponent's card means giving them one of yours.
* Calling KOMBIO replaces your turn; every opponent gets one final turn; the caller's hand is locked.
  Emptying your hand is an automatic call.
* Scoring: see :func:`kombio.cards.round_scores`.

Simplifications: no drawing from the discard pile, no bluffing, and players only act on cards they
remember, so wrong-match penalties never occur. Forgetting is modeled instead.
"""
from __future__ import annotations

import random
from dataclasses import dataclass
from typing import Sequence

from .cards import DECK_TOTAL, DECK_VALUES, Card, hand_score, round_scores
from .strategies import Strategy

MAX_TURNS = 100


@dataclass(slots=True)
class RoundResult:
    scores: list[int]
    totals: list[int]
    caller: int | None
    caller_won: bool | None
    caller_known_sum: int | None  # sum of the caller's *known* cards at the moment of calling


class Round:
    def __init__(self, strategies: Sequence[Strategy], rng: random.Random):
        self.s = list(strategies)
        self.n = len(self.s)
        self.rng = rng
        self.deck = [Card(v, uid) for uid, v in enumerate(DECK_VALUES)]
        rng.shuffle(self.deck)
        self.hands: list[list[Card]] = [[self.deck.pop() for _ in range(4)] for _ in range(self.n)]
        self.know: list[set[int]] = [{c.uid for c in h[:2]} for h in self.hands]
        self.discards: list[Card] = []
        self.discard_sum = 0
        self.caller: int | None = None

    # ------------------------------------------------------------------ knowledge
    def unseen_mean(self, i: int) -> float:
        """Average value of the cards player ``i`` has not seen (card counting)."""
        known = self.know[i]
        s, k = self.discard_sum, len(self.discards)
        for h in self.hands:
            for c in h:
                if c.uid in known:
                    s += c.value
                    k += 1
        left = len(DECK_VALUES) - k
        return (DECK_TOTAL - s) / left if left > 0 else DECK_TOTAL / len(DECK_VALUES)

    def ev(self, i: int, c: Card, mu: float) -> float:
        return c.value if c.uid in self.know[i] else mu

    def estimate(self, i: int) -> float:
        mu = self.unseen_mean(i)
        return sum(self.ev(i, c, mu) for c in self.hands[i])

    def worst_slot(self, i: int, mu: float) -> int:
        h = self.hands[i]
        return max(range(len(h)), key=lambda k: self.ev(i, h[k], mu))

    def opponent_slots(self, i: int) -> list[tuple[int, int]]:
        return [(j, k) for j in range(self.n) if j not in (i, self.caller) for k in range(len(self.hands[j]))]

    def leader(self, i: int) -> int | None:
        """Opponent who looks closest to going out: fewest cards, then lowest estimated total."""
        mu = self.unseen_mean(i)
        cands = [j for j in range(self.n) if j not in (i, self.caller) and self.hands[j]]
        if not cands:
            return None
        return min(cands, key=lambda j: (len(self.hands[j]), sum(self.ev(i, c, mu) for c in self.hands[j])))

    def reveal(self, card: Card) -> None:
        for k in self.know:
            k.add(card.uid)

    def forget(self, i: int) -> None:
        f = self.s[i].forget
        if f <= 0:
            return
        rng, known = self.rng, self.know[i]
        for h in self.hands:
            for c in h:
                if c.uid in known and rng.random() < f:
                    known.discard(c.uid)

    def p_lowest(self, i: int, samples: int) -> float:
        """Monte Carlo estimate, from player i's knowledge, that i is strictly lowest right now."""
        known = self.know[i]
        seen = {c.uid for c in self.discards} | {c.uid for h in self.hands for c in h if c.uid in known}
        pool = [v for uid, v in enumerate(DECK_VALUES) if uid not in seen] or [6]
        n_unknown = sum(1 for h in self.hands for c in h if c.uid not in known)
        base = [sum(c.value for c in h if c.uid in known) for h in self.hands]
        counts = [sum(1 for c in h if c.uid not in known) for h in self.hands]
        wins = 0
        for _ in range(samples):
            draw = iter(self.rng.sample(pool, min(len(pool), n_unknown)))
            tot = [base[j] + sum(next(draw, 6) for _ in range(counts[j])) for j in range(self.n)]
            wins += all(tot[i] < tot[j] for j in range(self.n) if j != i)
        return wins / samples

    # ------------------------------------------------------------------ actions
    def discard(self, card: Card, who: int) -> None:
        self._to_pile(card)
        v = card.value
        for step in range(1, self.n + 1):
            i = (who + step) % self.n
            if i == self.caller:
                continue
            own = next((c for c in self.hands[i] if c.uid in self.know[i] and c.value == v), None)
            if own is not None:
                self.hands[i].remove(own)
                self._to_pile(own)
                continue
            for j, k in self.opponent_slots(i):
                c = self.hands[j][k]
                if c.uid in self.know[i] and c.value == v:
                    self.hands[j].pop(k)
                    self._to_pile(c)
                    if self.hands[i]:  # give the opponent our worst card to fill the gap
                        give = self.hands[i].pop(self.worst_slot(i, self.unseen_mean(i)))
                        self.hands[j].append(give)
                    break

    def _to_pile(self, card: Card) -> None:
        self.discards.append(card)
        self.discard_sum += card.value
        self.reveal(card)

    def use_ability(self, i: int, v: int) -> None:
        h, rng, st = self.hands[i], self.rng, self.s[i]
        mu = self.unseen_mean(i)
        opps = self.opponent_slots(i)
        unknown_own = [c for c in h if c.uid not in self.know[i]]
        if v in (7, 8):
            if unknown_own:
                self.know[i].add(unknown_own[0].uid)
            return
        if not opps:
            return
        if v in (9, 10):
            target = self.leader(i) if st.scout else None
            cand = [(j, k) for j, k in opps if self.hands[j][k].uid not in self.know[i] and target in (None, j)]
            cand = cand or [(j, k) for j, k in opps if self.hands[j][k].uid not in self.know[i]]
            if cand:
                j, k = rng.choice(cand)
                self.know[i].add(self.hands[j][k].uid)
            return
        if not h:
            return
        w = self.worst_slot(i, mu)
        wv = self.ev(i, h[w], mu)
        if v in (11, 12):
            known_low = sorted((self.hands[j][k].value, j, k) for j, k in opps if self.hands[j][k].uid in self.know[i])
            if known_low and known_low[0][0] < wv:
                _, j, k = known_low[0]
            elif st.sabotage and wv > mu and (L := self.leader(i)) is not None:
                j, k = L, min(range(len(self.hands[L])), key=lambda kk: self.ev(i, self.hands[L][kk], mu))
            elif wv > mu + 1:
                j, k = rng.choice(opps)
            else:
                return
            h[w], self.hands[j][k] = self.hands[j][k], h[w]
            return
        # 13: look at one card; 14: look at two (one of ours, one of theirs). Take it if better.
        if v == 14 and unknown_own:
            self.know[i].add(unknown_own[0].uid)
            mu = self.unseen_mean(i)
            w = self.worst_slot(i, mu)
        cand = [(j, k) for j, k in opps if self.hands[j][k].uid not in self.know[i]] or opps
        j, k = rng.choice(cand)
        self.know[i].add(self.hands[j][k].uid)
        if self.hands[j][k].value < self.ev(i, h[w], mu):
            h[w], self.hands[j][k] = self.hands[j][k], h[w]

    def take_turn(self, i: int) -> None:
        self.forget(i)
        if not self.deck:  # reshuffle the discard pile, keeping the top card
            top = self.discards.pop() if self.discards else None
            self.deck = self.discards
            self.rng.shuffle(self.deck)
            self.discards = [top] if top else []
            self.discard_sum = top.value if top else 0
            if not self.deck:
                return
        d = self.deck.pop()
        self.know[i].add(d.uid)
        h = self.hands[i]
        if h:
            mu = self.unseen_mean(i)
            w = self.worst_slot(i, mu)
            if d.value < self.ev(i, h[w], mu):
                old, h[w] = h[w], d
                self.discard(old, i)  # discarded from hand: no ability
                return
        self.discard(d, i)
        if d.value >= 7:
            self.use_ability(i, d.value)

    def wants_to_call(self, i: int, turn: int) -> bool:
        if turn < self.n:  # nobody calls before everyone has had a turn
            return False
        st = self.s[i]
        if st.p_call is not None:
            return self.estimate(i) <= 14 and self.p_lowest(i, st.samples) >= st.p_call
        return self.estimate(i) <= st.threshold

    # ------------------------------------------------------------------ round
    def play(self, force_first_call: bool = False) -> RoundResult:
        i, known_sum = 0, None
        for turn in range(MAX_TURNS):
            calling = (force_first_call and turn == 0) or self.wants_to_call(i, turn) or not self.hands[i]
            if not calling:
                self.take_turn(i)
                calling = not self.hands[i]
            if calling:
                self.caller = i
                known_sum = sum(c.value for c in self.hands[i] if c.uid in self.know[i])
                for step in range(1, self.n):
                    self.take_turn((i + step) % self.n)
                break
            i = (i + 1) % self.n
        totals = [hand_score([c.value for c in h]) for h in self.hands]
        scores = round_scores(totals, self.caller)
        won = None if self.caller is None else scores[self.caller] == totals[self.caller]
        return RoundResult(scores, totals, self.caller, won, known_sum)


def play_round(strategies: Sequence[Strategy], rng: random.Random, force_first_call: bool = False) -> RoundResult:
    return Round(strategies, rng).play(force_first_call)


def play_game(strategies: Sequence[Strategy], rng: random.Random, limit: int = 100) -> list[int]:
    """Play rounds until someone reaches ``limit`` points; return cumulative scores (lowest wins)."""
    totals = [0] * len(strategies)
    while max(totals) < limit:
        r = play_round(strategies, rng)
        totals = [a + b for a, b in zip(totals, r.scores)]
    return totals
