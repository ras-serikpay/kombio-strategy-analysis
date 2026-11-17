"""Experiments behind the README results. Each returns plain rows (list of dicts)."""
from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from itertools import combinations
from statistics import mean
from typing import Sequence

from .engine import play_game, play_round
from .strategies import BASELINE_OPPONENT, Strategy


def _bucket(s: int) -> str:
    for hi, name in ((2, "<=2"), (5, "3-5"), (8, "6-8"), (11, "9-11"), (14, "12-14")):
        if s <= hi:
            return name
    return "15+"


def early_call(rounds: int, seed: int = 0) -> list[dict]:
    """A) Knowing 2 cards and calling immediately: how often does the call win, by known sum?"""
    rng = random.Random(seed)
    players = [BASELINE_OPPONENT] * 4
    by: dict[str, list[bool]] = {}
    for _ in range(rounds):
        r = play_round(players, rng, force_first_call=True)
        by.setdefault(_bucket(r.caller_known_sum), []).append(bool(r.caller_won))
    order = ["<=2", "3-5", "6-8", "9-11", "12-14", "15+"]
    return [{"known_sum": b, "share_of_deals": len(by[b]) / rounds, "call_win_rate": mean(by[b])} for b in order if b in by]


def evaluate(me: Strategy, opponents: list[Strategy], rounds: int, games: int, seed: int) -> dict:
    """Round-level and game-level metrics for player 0 (``me``) against ``opponents``."""
    rng = random.Random(seed)
    players = [me, *opponents]
    calls = wins = round_wins = 0
    pts = []
    for _ in range(rounds):
        r = play_round(players, rng)
        pts.append(r.scores[0])
        round_wins += r.scores[0] < min(r.scores[1:])
        if r.caller == 0:
            calls += 1
            wins += bool(r.caller_won)
    game_wins = sum(t[0] < min(t[1:]) for t in (play_game(players, rng) for _ in range(games)))
    return {
        "strategy": me.label(),
        "threshold": me.threshold if me.p_call is None else None,
        "p_call": me.p_call,
        "sabotage": me.sabotage,
        "opp_forget": opponents[0].forget,
        "call_rate": calls / rounds,
        "call_win_rate": wins / calls if calls else 0.0,
        "round_win_rate": round_wins / rounds,
        "points_per_round": mean(pts),
        "game_win_rate": game_wins / games,
    }


def _run(args):
    return evaluate(*args)


def strategy_grid(rounds: int, games: int, seed: int = 0, workers: int | None = None) -> list[dict]:
    """B) Threshold sweep x opponent memory, plus probability-based and sabotage variants."""
    jobs = []
    k = 0
    for forget in (0.0, 0.15, 0.30):
        opps = [replace(BASELINE_OPPONENT, forget=forget)] * 3
        mine = [Strategy(threshold=t) for t in (0, 1, 2, 3, 4, 6, 8, 10, 12)]
        mine += [Strategy(threshold=1, sabotage=True, scout=True)]
        mine += [Strategy(p_call=p) for p in (0.75, 0.9)]
        mine += [Strategy(p_call=p, final_turn=True) for p in (0.75, 0.9)]
        for me in mine:
            jobs.append((me, opps, rounds, games, seed + k))
            k += 1
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_run, jobs))


# ---------------------------------------------------------------------------- C) calibration
CALIBRATION_BINS = (0.5, 0.6, 0.7, 0.8, 0.9, 1.0001)


def _calibration_job(args) -> list[dict]:
    final_turn, rounds, seed = args
    rng = random.Random(seed)
    # p_call=0.5 makes the player call across a wide range of confidence levels, so every bin fills.
    me = Strategy(p_call=0.5, final_turn=final_turn)
    players = [me, *[BASELINE_OPPONENT] * 3]
    preds: dict[int, list[tuple[float, bool]]] = {}
    for _ in range(rounds):
        r = play_round(players, rng)
        if r.caller == 0 and r.caller_p is not None:
            b = next(k for k in range(len(CALIBRATION_BINS) - 1) if r.caller_p < CALIBRATION_BINS[k + 1])
            preds.setdefault(b, []).append((r.caller_p, bool(r.caller_won)))
    rows = []
    for b in sorted(preds):
        ps, ws = zip(*preds[b])
        rows.append({
            "estimator": "calibrated (final turns + kept-low cards)" if final_turn else "original (current totals only)",
            "bin": f"{CALIBRATION_BINS[b]:.1f}-{min(CALIBRATION_BINS[b + 1], 1.0):.1f}",
            "predicted": mean(ps),
            "actual": mean(ws),
            "calls": len(ps),
        })
    return rows


def calibration(rounds: int, seed: int = 0, workers: int | None = None) -> list[dict]:
    """C) Does a predicted P(win) of x% actually win x% of calls? Original vs calibrated estimator."""
    jobs = [(False, rounds, seed), (True, rounds, seed)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return [row for rows in ex.map(_calibration_job, jobs) for row in rows]


# ---------------------------------------------------------------------------- D) memory errors & discard pile
def memory_errors(rounds: int, games: int, seed: int = 0, workers: int | None = None) -> list[dict]:
    """D) Opponents who misremember (and so make wrong matches); does taking from the discard pile help?"""
    jobs, k = [], 0
    for mis in (0.0, 0.05, 0.10):
        opps = [replace(BASELINE_OPPONENT, forget=0.15, misremember=mis)] * 3
        for me in (Strategy(threshold=1, name="T<=1"), Strategy(threshold=1, take_discard=3, name="T<=1 +discard pile")):
            jobs.append((me, opps, rounds, games, seed + k, mis))
            k += 1
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_memory_job, jobs))


def _memory_job(args) -> dict:
    me, opps, rounds, games, seed, mis = args
    row = evaluate(me, opps, rounds, games, seed)
    rng = random.Random(seed + 10_000)
    wrong = [play_round([me, *opps], rng).wrong_matches for _ in range(rounds)]
    row["opp_misremember"] = mis
    row["opp_wrong_matches_per_round"] = mean(sum(w[1:]) / 3 for w in wrong)
    return row


# ---------------------------------------------------------------------------- E) tournament
TOURNAMENT_POOL = (
    Strategy(threshold=0, name="T<=0"),
    Strategy(threshold=1, name="T<=1"),
    Strategy(threshold=2, name="T<=2"),
    Strategy(threshold=4, name="T<=4"),
    Strategy(threshold=8, name="T<=8"),
    Strategy(threshold=1, sabotage=True, scout=True, name="T<=1 +sabotage"),
    Strategy(p_call=0.9, name="P>=0.9 original"),
    Strategy(p_call=0.9, final_turn=True, name="P>=0.9 calibrated"),
)


def _table_job(args) -> list[list[tuple[int, int]]]:
    idx, pool, games, seed = args
    out = []
    for rot in range(len(idx)):  # every strategy sits in every seat (seat 0 moves first)
        seats = idx[rot:] + idx[:rot]
        players = [pool[s] for s in seats]
        for g in range(games):
            # Same seed for the same (seat rotation, game) at every table: common random numbers.
            rng = random.Random(seed * 1_000_003 + rot * 10_007 + g)
            out.append(list(zip(seats, play_game(players, rng))))
    return out


def tournament(pool: Sequence[Strategy] = TOURNAMENT_POOL, games: int = 50, seed: int = 0,
               forget: float = 0.0, workers: int | None = None) -> tuple[list[dict], list[dict]]:
    """E) Every 4-strategy table from ``pool``, every seat rotation, ``games`` games each.

    Returns (standings, head_to_head). A game is won by the lowest final score (ties split the win).
    head_to_head[a][b] = share of shared games where a finished below b (ties count half).
    """
    pool = [replace(s, forget=forget) for s in pool]
    names = [s.label() for s in pool]
    jobs = [(list(t), pool, games, seed) for t in combinations(range(len(pool)), 4)]
    with ProcessPoolExecutor(max_workers=workers) as ex:
        results = [g for table in ex.map(_table_job, jobs) for g in table]

    n = len(pool)
    played, wins, pts = [0] * n, [0.0] * n, [0] * n
    beat = [[0.0] * n for _ in range(n)]
    met = [[0] * n for _ in range(n)]
    for game in results:
        low = min(score for _, score in game)
        winners = [s for s, score in game if score == low]
        for s, score in game:
            played[s] += 1
            pts[s] += score
            if s in winners:
                wins[s] += 1 / len(winners)
            for t, other in game:
                if t != s:
                    met[s][t] += 1
                    beat[s][t] += 1.0 if score < other else 0.5 if score == other else 0.0
    standings = sorted(
        ({"strategy": names[s], "games": played[s], "win_rate": wins[s] / played[s],
          "avg_final_score": pts[s] / played[s]} for s in range(n)),
        key=lambda r: -r["win_rate"],
    )
    h2h = [{"strategy": names[s], **{names[t]: (beat[s][t] / met[s][t] if met[s][t] else None) for t in range(n)}}
           for s in range(n)]
    return standings, h2h
