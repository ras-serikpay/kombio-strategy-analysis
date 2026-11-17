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


# --------------------------------------------------------------------------- new features
def _all_uids(r):
    return sorted([c.uid for h in r.hands for c in h] + [c.uid for c in r.discards] + [c.uid for c in r.deck])


def test_cards_conserved_with_discard_pile_and_memory_errors():
    rng = random.Random(7)
    s = Strategy(threshold=3, forget=0.2, misremember=0.2, take_discard=4)
    wrong = 0
    for _ in range(300):
        r = Round([s] * 4, rng)
        res = r.play()
        assert _all_uids(r) == list(range(70))
        wrong += sum(res.wrong_matches)
    assert wrong > 0  # misremembering players do make wrong matches


def test_no_wrong_matches_with_perfect_memory():
    rng = random.Random(8)
    for _ in range(200):
        res = play_round([Strategy(threshold=3)] * 4, rng)
        assert sum(res.wrong_matches) == 0


def test_wrong_match_gives_penalty_card_and_reveals():
    from kombio.cards import Card, WRONG_MATCH_PENALTY_CARDS
    r = Round([Strategy()] * 4, random.Random(0))
    card = r.hands[1][0]
    r.know[1].add(card.uid)
    r.wrong[1][card.uid] = card.value + 1  # player 1 misremembers it
    before = len(r.hands[1])
    r.discard(Card(card.value + 1, 999), who=0)  # a card with the believed value is discarded
    assert len(r.hands[1]) == before + WRONG_MATCH_PENALTY_CARDS
    assert card in r.hands[1]
    assert all(r.believes(p, card) == card.value for p in range(4))  # everyone saw the real value


def test_final_turn_estimate_is_more_cautious():
    rng = random.Random(11)
    naive, cal = [], []
    for _ in range(40):
        r = Round([Strategy(p_call=0.9)] * 4, rng)
        naive.append(r.p_lowest(0, 400, final_turn=False))
        cal.append(r.p_lowest(0, 400, final_turn=True))
    assert sum(cal) < sum(naive)
    assert all(0.0 <= p <= 1.0 for p in naive + cal)


def test_tournament_runs_and_seats_rotate():
    from kombio.experiments import tournament
    pool = [Strategy(threshold=t, name=f"T<={t}") for t in (1, 3, 5, 8, 10)]
    standings, h2h = tournament(pool, games=2, seed=1, workers=1)
    assert len(standings) == 5 and len(h2h) == 5
    # 5 tables of 4, each strategy in 4 of them, 4 rotations x 2 games each
    assert all(row["games"] == 4 * 4 * 2 for row in standings)
    assert abs(sum(row["win_rate"] * row["games"] for row in standings) - 5 * 4 * 2) < 1e-9
