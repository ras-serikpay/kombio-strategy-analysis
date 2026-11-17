"""Game engine: simulates one KOMBIO round (and full games) between strategy-driven players.

Rules modeled (official rulebook):
* 4 face-down cards each; every player peeks at 2 at the start.
* A turn: draw from the deck -- or take the top card of the discard pile -- then either swap the
  card into your hand (the replaced card is discarded with no ability) or discard it straight
  away and optionally use its ability (7-14). A card taken from the discard pile must be swapped
  in and has no ability.
* Matching: after any discard, a player who *believes* they know a card of the same value -- in
  their own hand or an opponent's -- may discard it. Matching an opponent's card means giving
  them one of yours. A wrong match costs penalty cards (see ``WRONG_MATCH_PENALTY_CARDS``).
* Calling KOMBIO replaces your turn; every opponent gets one final turn; the caller's hand is locked.
  Emptying your hand is an automatic call.
* Scoring: see :func:`kombio.cards.round_scores`.

Memory: each player remembers card values by card id. Remembered cards can be forgotten
(``Strategy.forget``) or misremembered as a wrong value (``Strategy.misremember``).

Simplification: no bluffing.
"""
from __future__ import annotations

import random
from collections import Counter
from dataclasses import dataclass, field
from typing import Sequence

from .cards import DECK_TOTAL, DECK_VALUES, WRONG_MATCH_PENALTY_CARDS, Card, hand_score, round_scores
from .strategies import Strategy

MAX_TURNS = 100
_DECK_COUNTS = Counter(DECK_VALUES)


@dataclass(slots=True)
class RoundResult:
    scores: list[int]
    totals: list[int]
    caller: int | None
    caller_won: bool | None
    caller_known_sum: int | None  # sum of the caller's *known* cards at the moment of calling
    caller_p: float | None = None  # caller's estimated P(lowest) when it called (p_call strategies)
    wrong_matches: list[int] = field(default_factory=list)  # wrong matches per player


class Round:
    def __init__(self, strategies: Sequence[Strategy], rng: random.Random):
        self.s = list(strategies)
        self.n = len(self.s)
        self.rng = rng
        self.deck = [Card(v, uid) for uid, v in enumerate(DECK_VALUES)]
        rng.shuffle(self.deck)
        self.hands: list[list[Card]] = [[self.deck.pop() for _ in range(4)] for _ in range(self.n)]
        self.know: list[set[int]] = [{c.uid for c in h[:2]} for h in self.hands]
        self.wrong: list[dict[int, int]] = [{} for _ in range(self.n)]  # uid -> wrongly believed value
        self.discards: list[Card] = []
        self.discard_sum = 0
        self.caller: int | None = None
        self.call_p: float | None = None
        self.wrong_matches = [0] * self.n
        # The probability estimate gets its own random stream so that turning it on does not change
        # the cards everyone draws. Only created when needed, so old seeds reproduce old results.
        uses_p = any(st.p_call is not None for st in self.s)
        self.est_rng = random.Random(rng.getrandbits(64)) if uses_p else rng

    # ------------------------------------------------------------------ knowledge
    def believes(self, i: int, c: Card) -> int | None:
        """The value player ``i`` thinks card ``c`` has, or None if they don't remember it."""
        if c.uid not in self.know[i]:
            return None
        return self.wrong[i].get(c.uid, c.value)

    def learn(self, i: int, c: Card) -> None:
        """Player ``i`` sees card ``c``: they now know its true value."""
        self.know[i].add(c.uid)
        self.wrong[i].pop(c.uid, None)

    def unseen_mean(self, i: int) -> float:
        """Average value of the cards player ``i`` has not seen (card counting)."""
        s, k = self.discard_sum, len(self.discards)
        for h in self.hands:
            for c in h:
                b = self.believes(i, c)
                if b is not None:
                    s += b
                    k += 1
        left = len(DECK_VALUES) - k
        return (DECK_TOTAL - s) / left if left > 0 else DECK_TOTAL / len(DECK_VALUES)

    def ev(self, i: int, c: Card, mu: float) -> float:
        b = self.believes(i, c)
        return mu if b is None else b

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
        for i in range(self.n):
            self.learn(i, card)

    def forget(self, i: int) -> None:
        st = self.s[i]
        f, m = st.forget, st.misremember
        if f <= 0 and m <= 0:
            return
        rng, known, wrong = self.rng, self.know[i], self.wrong[i]
        for h in self.hands:
            for c in h:
                if c.uid not in known:
                    continue
                r = rng.random()
                if r < f:
                    known.discard(c.uid)
                    wrong.pop(c.uid, None)
                elif r < f + m:
                    wrong[c.uid] = self._confuse(self.believes(i, c))

    def _confuse(self, v: int) -> int:
        """A plausible wrong memory of value ``v``: off by 1 or 2, kept inside the deck's range."""
        shift = self.rng.choice((-2, -1, 1, 2))
        w = v + shift
        return w if -1 <= w <= 14 else v - shift

    def p_lowest(self, i: int, samples: int, final_turn: bool = False, opp_draws: int = 2) -> float:
        """Monte Carlo estimate, from player i's beliefs, that i ends the round strictly lowest.

        Original (``final_turn=False``): fill every card ``i`` doesn't remember with a random unseen
        card and compare totals as they are now.

        Calibrated (``final_turn=True``) fixes the two reasons that estimate is overconfident:
        1. Opponents' hidden cards are not random -- they have been swapping in low cards all round.
           Each one is modeled as the lowest of ``opp_draws`` random unseen cards.
        2. Every opponent gets a final turn after the call: they draw one more card and keep it if
           it beats their highest card.
        """
        if final_turn:
            return self._p_lowest_calibrated(i, samples, opp_draws)
        base, hidden, pool = self._belief_view(i)
        rng = self.est_rng
        wins = 0
        for _ in range(samples):
            draw = iter(rng.sample(pool, min(len(pool), sum(hidden))))
            tot = [hand_score(base[j] + [next(draw, 6) for _ in range(hidden[j])]) for j in range(self.n)]
            wins += all(tot[i] < tot[j] for j in range(self.n) if j != i)
        return wins / samples

    def _belief_view(self, i: int) -> tuple[list[list[int]], list[int], list[int]]:
        """(remembered values per hand, number of hidden cards per hand, unseen pool) for player i."""
        seen = Counter(c.value for c in self.discards)
        base, hidden = [], []
        for h in self.hands:
            vals, u = [], 0
            for c in h:
                b = self.believes(i, c)
                if b is None:
                    u += 1
                else:
                    vals.append(b)
                    seen[b] += 1
            base.append(vals)
            hidden.append(u)
        return base, hidden, list((_DECK_COUNTS - seen).elements()) or [6]

    def _p_lowest_calibrated(self, i: int, samples: int, opp_draws: int) -> float:
        base, hidden, pool = self._belief_view(i)
        rng = self.est_rng
        wins = 0
        for _ in range(samples):
            hands = []
            for j in range(self.n):
                k = 1 if j == i else opp_draws  # our own hidden cards are just random
                hands.append(base[j] + [min(rng.choice(pool) for _ in range(k)) for _ in range(hidden[j])])
            for j in range(self.n):  # every opponent's final turn
                hj = hands[j]
                if j == i or not hj:
                    continue
                d = rng.choice(pool)
                w = max(range(len(hj)), key=hj.__getitem__)
                if d < hj[w]:
                    hj[w] = d
            tot = [hand_score(h) for h in hands]
            wins += all(tot[i] < tot[j] for j in range(self.n) if j != i)
        return wins / samples

    # ------------------------------------------------------------------ deck & discards
    def _draw(self) -> Card | None:
        """Pop the top of the deck, reshuffling the discard pile (minus its top card) if empty."""
        if not self.deck:
            top = self.discards.pop() if self.discards else None
            self.deck = self.discards
            self.rng.shuffle(self.deck)
            # Shuffled cards are face down again: nobody knows where they are.
            for c in self.deck:
                for i in range(self.n):
                    self.know[i].discard(c.uid)
                    self.wrong[i].pop(c.uid, None)
            self.discards = [top] if top else []
            self.discard_sum = top.value if top else 0
            if not self.deck:
                return None
        return self.deck.pop()

    def discard(self, card: Card, who: int) -> None:
        self._to_pile(card)
        v = card.value
        for step in range(1, self.n + 1):
            i = (who + step) % self.n
            if i == self.caller:
                continue
            own = next((c for c in self.hands[i] if self.believes(i, c) == v), None)
            if own is not None:
                if own.value == v:
                    self.hands[i].remove(own)
                    self._to_pile(own)
                else:
                    self._wrong_match(i, own)
                continue
            for j, k in self.opponent_slots(i):
                c = self.hands[j][k]
                if self.believes(i, c) != v:
                    continue
                if c.value != v:
                    self._wrong_match(i, c)
                    break
                self.hands[j].pop(k)
                self._to_pile(c)
                if self.hands[i]:  # give the opponent our worst card to fill the gap
                    give = self.hands[i].pop(self.worst_slot(i, self.unseen_mean(i)))
                    self.hands[j].append(give)
                break

    def _wrong_match(self, i: int, card: Card) -> None:
        """Player ``i`` threw ``card`` but it doesn't match: it goes back, everyone saw it, i draws penalties."""
        self.wrong_matches[i] += 1
        self.reveal(card)
        for _ in range(WRONG_MATCH_PENALTY_CARDS):
            p = self._draw()
            if p is not None:
                self.hands[i].append(p)  # face down: i doesn't know it

    def _to_pile(self, card: Card) -> None:
        self.discards.append(card)
        self.discard_sum += card.value
        self.reveal(card)

    # ------------------------------------------------------------------ actions
    def use_ability(self, i: int, v: int) -> None:
        h, rng, st = self.hands[i], self.rng, self.s[i]
        mu = self.unseen_mean(i)
        opps = self.opponent_slots(i)
        unknown_own = [c for c in h if c.uid not in self.know[i]]
        if v in (7, 8):
            if unknown_own:
                self.learn(i, unknown_own[0])
            return
        if not opps:
            return
        if v in (9, 10):
            target = self.leader(i) if st.scout else None
            cand = [(j, k) for j, k in opps if self.hands[j][k].uid not in self.know[i] and target in (None, j)]
            cand = cand or [(j, k) for j, k in opps if self.hands[j][k].uid not in self.know[i]]
            if cand:
                j, k = rng.choice(cand)
                self.learn(i, self.hands[j][k])
            return
        if not h:
            return
        w = self.worst_slot(i, mu)
        wv = self.ev(i, h[w], mu)
        if v in (11, 12):
            known_low = sorted((b, j, k) for j, k in opps if (b := self.believes(i, self.hands[j][k])) is not None)
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
            self.learn(i, unknown_own[0])
            mu = self.unseen_mean(i)
            w = self.worst_slot(i, mu)
        cand = [(j, k) for j, k in opps if self.hands[j][k].uid not in self.know[i]] or opps
        j, k = rng.choice(cand)
        self.learn(i, self.hands[j][k])
        if self.hands[j][k].value < self.ev(i, h[w], mu):
            h[w], self.hands[j][k] = self.hands[j][k], h[w]

    def take_turn(self, i: int) -> None:
        self.forget(i)
        h, st = self.hands[i], self.s[i]
        if st.take_discard is not None and h and self.discards:
            top = self.discards[-1]
            mu = self.unseen_mean(i)
            w = self.worst_slot(i, mu)
            if top.value <= st.take_discard and top.value < self.ev(i, h[w], mu):
                self.discards.pop()
                self.discard_sum -= top.value
                old, h[w] = h[w], top  # everyone saw `top`, so everyone knows it's in our hand now
                self.discard(old, i)
                return
        d = self._draw()
        if d is None:
            return
        self.learn(i, d)
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
        self.call_p = None
        if turn < self.n:  # nobody calls before everyone has had a turn
            return False
        st = self.s[i]
        if st.p_call is not None:
            if self.estimate(i) > 14:
                return False
            self.call_p = self.p_lowest(i, st.samples, st.final_turn, st.opp_draws)
            return self.call_p >= st.p_call
        return self.estimate(i) <= st.threshold

    # ------------------------------------------------------------------ round
    def play(self, force_first_call: bool = False) -> RoundResult:
        i, known_sum, call_p = 0, None, None
        for turn in range(MAX_TURNS):
            forced = force_first_call and turn == 0
            wants = not forced and self.wants_to_call(i, turn)
            calling = forced or wants or not self.hands[i]
            call_p = self.call_p if wants else None
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
        return RoundResult(scores, totals, self.caller, won, known_sum, call_p, list(self.wrong_matches))


def play_round(strategies: Sequence[Strategy], rng: random.Random, force_first_call: bool = False) -> RoundResult:
    return Round(strategies, rng).play(force_first_call)


def play_game(strategies: Sequence[Strategy], rng: random.Random, limit: int = 100) -> list[int]:
    """Play rounds until someone reaches ``limit`` points; return cumulative scores (lowest wins)."""
    totals = [0] * len(strategies)
    while max(totals) < limit:
        r = play_round(strategies, rng)
        totals = [a + b for a, b in zip(totals, r.scores)]
    return totals
