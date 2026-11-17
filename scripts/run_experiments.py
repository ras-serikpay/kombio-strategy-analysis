"""Reproduce every table and chart in the README.

    python scripts/run_experiments.py            # full run (a few minutes)
    python scripts/run_experiments.py --quick    # smoke test
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from kombio.experiments import calibration, early_call, memory_errors, strategy_grid, tournament  # noqa: E402

RESULTS = ROOT / "results"


def write_csv(rows: list[dict], path: Path) -> None:
    with path.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


def plot(early: list[dict], grid: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib not installed; skipping charts")
        return
    plt.rcParams.update({"font.size": 11, "axes.spines.top": False, "axes.spines.right": False})

    # Chart 1: early call win rate by known sum
    fig, ax = plt.subplots(figsize=(7, 4))
    xs = [r["known_sum"] for r in early]
    ys = [100 * r["call_win_rate"] for r in early]
    ax.bar(xs, ys, color="#3b6ea5")
    ax.axhline(50, ls="--", lw=1, color="#888")
    for x, y in zip(xs, ys):
        ax.text(x, y + 1, f"{y:.0f}%", ha="center")
    ax.set(xlabel="Sum of your two known cards", ylabel="Call wins (%)",
           title="Calling right away rarely works, even with a great start")
    fig.tight_layout()
    fig.savefig(RESULTS / "early_call.png", dpi=150)

    # Chart 2: game win rate vs threshold for each opponent-memory level
    fig, ax = plt.subplots(figsize=(7, 4))
    colors = {0.0: "#1b4965", 0.15: "#5fa8d3", 0.30: "#9cc5a1"}
    for forget, c in colors.items():
        rows = [r for r in grid if r["opp_forget"] == forget and r["threshold"] is not None and not r["sabotage"]]
        rows.sort(key=lambda r: r["threshold"])
        ax.plot([r["threshold"] for r in rows], [100 * r["game_win_rate"] for r in rows], marker="o", color=c,
                label=f"opponents forget {forget:.0%}/turn")
    ax.axhline(25, ls=":", lw=1, color="#888")
    ax.text(12, 26, "random baseline 25%", ha="right", color="#666", fontsize=9)
    ax.axhline(70, ls="--", lw=1, color="#c0392b")
    ax.text(12, 71, "70% target", ha="right", color="#c0392b", fontsize=9)
    ax.set(xlabel="Your calling threshold T (call when estimated hand <= T)", ylabel="Games won (%)",
           title="Patience wins games; memory decides by how much", ylim=(0, 100))
    ax.legend(frameon=False, loc="upper right")
    fig.tight_layout()
    fig.savefig(RESULTS / "game_win_vs_threshold.png", dpi=150)
    print("charts written to results/")


def plot_calibration(rows: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot([50, 100], [50, 100], ls="--", lw=1, color="#888", label="perfect calibration")
    for est, c in (("original (current totals only)", "#c0392b"), ("calibrated (final turns + kept-low cards)", "#1b4965")):
        pts = [r for r in rows if r["estimator"] == est]
        ax.plot([100 * r["predicted"] for r in pts], [100 * r["actual"] for r in pts], marker="o", color=c, label=est)
    ax.set(xlabel="Predicted chance the call wins (%)", ylabel="Calls actually won (%)",
           title="The calibrated estimate means what it says", xlim=(50, 100), ylim=(30, 100))
    ax.legend(frameon=False, loc="upper left")
    fig.tight_layout()
    fig.savefig(RESULTS / "calibration.png", dpi=150)


def plot_tournament(standings: list[dict]) -> None:
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        return
    rows = standings[::-1]
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.barh([r["strategy"] for r in rows], [100 * r["win_rate"] for r in rows], color="#3b6ea5")
    ax.axvline(25, ls=":", lw=1, color="#888")
    for y, r in enumerate(rows):
        ax.text(100 * r["win_rate"] + 0.5, y, f"{100 * r['win_rate']:.0f}%", va="center")
    ax.set(xlabel="Games won (%) across all tables and seats", title="Round-robin tournament: strategies vs each other")
    fig.tight_layout()
    fig.savefig(RESULTS / "tournament.png", dpi=150)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="tiny run to check everything works")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    early_n, rounds, games = (2000, 300, 50) if a.quick else (40000, 3000, 1000)
    cal_rounds, t_games = (2000, 2) if a.quick else (40000, 60)

    RESULTS.mkdir(exist_ok=True)
    print(f"A) early call: {early_n:,} rounds")
    early = early_call(early_n, a.seed)
    write_csv(early, RESULTS / "early_call.csv")
    for r in early:
        print(f"   known sum {r['known_sum']:>6}: call wins {r['call_win_rate']:6.1%}")

    print(f"B) strategy grid: {rounds:,} rounds + {games:,} games per cell")
    grid = strategy_grid(rounds, games, a.seed, a.workers)
    write_csv(grid, RESULTS / "strategy_grid.csv")
    for r in grid:
        print(f"   {r['strategy']:<42} opp forget {r['opp_forget']:>4.0%} | call win {r['call_win_rate']:6.1%}"
              f" | round win {r['round_win_rate']:6.1%} | game win {r['game_win_rate']:6.1%}")
    plot(early, grid)

    print(f"C) calibration: {cal_rounds:,} rounds per estimator")
    cal = calibration(cal_rounds, a.seed, a.workers)
    write_csv(cal, RESULTS / "calibration.csv")
    for r in cal:
        print(f"   {r['estimator']:<36} {r['bin']}: predicted {r['predicted']:6.1%} | actual {r['actual']:6.1%} ({r['calls']} calls)")
    plot_calibration(cal)

    print(f"D) memory errors & discard pile: {rounds:,} rounds + {games:,} games per cell")
    mem = memory_errors(rounds, games, a.seed, a.workers)
    write_csv(mem, RESULTS / "memory_errors.csv")
    for r in mem:
        print(f"   {r['strategy']:<20} opp misremember {r['opp_misremember']:>4.0%} | wrong matches/opp/round "
              f"{r['opp_wrong_matches_per_round']:.2f} | game win {r['game_win_rate']:6.1%}")

    print(f"E) tournament: 70 tables x 4 seat rotations x {t_games} games")
    standings, h2h = tournament(games=t_games, seed=a.seed, workers=a.workers)
    write_csv(standings, RESULTS / "tournament.csv")
    write_csv(h2h, RESULTS / "tournament_head_to_head.csv")
    for r in standings:
        print(f"   {r['strategy']:<20} win {r['win_rate']:6.1%} | avg final score {r['avg_final_score']:6.1f}")
    plot_tournament(standings)


if __name__ == "__main__":
    main()
