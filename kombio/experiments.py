"""Experiments behind the README results. Each returns plain rows (list of dicts)."""
from __future__ import annotations

import random
from concurrent.futures import ProcessPoolExecutor
from dataclasses import replace
from statistics import mean

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
        for me in mine:
            jobs.append((me, opps, rounds, games, seed + k))
            k += 1
    with ProcessPoolExecutor(max_workers=workers) as ex:
        return list(ex.map(_run, jobs))
