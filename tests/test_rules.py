import random
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from kombio.cards import DECK_VALUES, hand_score, round_scores  # noqa: E402
from kombio.engine import Round, play_game, play_round  # noqa: E402
from kombio.strategies import Strategy  # noqa: E402


def test_deck_composition():
    c = Counter(DECK_VALUES)
    assert len(DECK_VALUES) == 70
    assert c[-1] == c[0] == 3
    assert all(c[v] == 5 for v in range(1, 13))
    assert c[13] == c[14] == 2


def test_hand_score_bonuses():
    assert hand_score([3, 4]) == 7
    assert hand_score([14, 14]) == -15
    assert hand_score([14, 12, 13]) == -10   # one 14 + others totalling 25
    assert hand_score([14]) == 14


def test_round_scores_caller_wins_and_loses():
    assert round_scores([2, 5, 7, 9], caller=0) == [2, 15, 17, 19]
    assert round_scores([5, 5, 7, 9], caller=0) == [20, 5, 7, 9]   # tie = caller loses
    assert round_scores([1, 2, 3, 4], caller=None) == [1, 2, 3, 4]


def test_cards_are_conserved():
    rng = random.Random(1)
    for _ in range(200):
        r = Round([Strategy(threshold=4)] * 4, rng)
        r.play()
        uids = [c.uid for h in r.hands for c in h] + [c.uid for c in r.discards] + [c.uid for c in r.deck]
        assert sorted(uids) == list(range(70))


def test_start_knowledge_is_two_cards():
    r = Round([Strategy()] * 4, random.Random(0))
    assert all(len(k) == 2 for k in r.know)


def test_reproducible_with_seed():
    a = [play_round([Strategy()] * 4, random.Random(5)).scores for _ in range(1)]
    b = [play_round([Strategy()] * 4, random.Random(5)).scores for _ in range(1)]
    assert a == b


def test_game_ends_at_limit():
    totals = play_game([Strategy()] * 4, random.Random(3), limit=100)
    assert max(totals) >= 100
