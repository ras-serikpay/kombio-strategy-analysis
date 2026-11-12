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

from kombio.experiments import early_call, strategy_grid  # noqa: E402

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


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--quick", action="store_true", help="tiny run to check everything works")
    ap.add_argument("--seed", type=int, default=2026)
    ap.add_argument("--workers", type=int, default=None)
    a = ap.parse_args()
    early_n, rounds, games = (2000, 300, 50) if a.quick else (40000, 3000, 1000)

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


if __name__ == "__main__":
    main()
