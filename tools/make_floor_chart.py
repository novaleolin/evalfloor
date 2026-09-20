#!/usr/bin/env python3
"""Regenerate docs/floor.png -- the figure in the README.

Kept out of the package on purpose: `import overtuned` must not pull in a
plotting library. Run this only when the numbers change.

    pip install matplotlib && python3 tools/make_floor_chart.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from overtuned import selection_floor

TRIES = [2, 3, 5, 8, 12, 20, 30, 50, 75, 100]
EVAL_SIZES = [50, 100, 200, 500, 2000]
BASELINE = 0.60
# Theme-neutral: a dark-on-light figure that still reads when a reader's
# viewer inverts it, rather than one tuned to either background.
COLORS = ["#c0392b", "#d68910", "#2e86c1", "#1e8449", "#7d3c98"]


def main():
    fig, ax = plt.subplots(figsize=(8.2, 4.8), dpi=170)
    for c, n in zip(COLORS, EVAL_SIZES):
        ys = [100 * selection_floor(k, n, BASELINE, reps=3000, seed=1)
              for k in TRIES]
        # End labels only, no legend: the lines never cross, so labelling
        # each where it ends is read in one pass, while a legend makes the
        # eye bounce between two places for the same fact.
        ax.plot(TRIES, ys, marker="o", ms=3.4, lw=1.9, color=c)
        ax.annotate(f"{n} examples", (TRIES[-1], ys[-1]), xytext=(8, 0),
                    textcoords="offset points", color=c, fontsize=9.5,
                    fontweight="bold", va="center")

    y = 100 * selection_floor(30, 200, BASELINE, reps=3000, seed=1)

    ax.set_xscale("log")
    ax.set_xticks(TRIES)
    ax.set_xticklabels([str(t) for t in TRIES])
    ax.set_xlabel("variants tried, then keep the best", fontsize=10.5)
    ax.set_ylabel("points gained for free\n(percentage points)", fontsize=10.5)
    ax.set_title("What a search reports when NO variant is actually better",
                 fontsize=13, fontweight="bold", pad=18)
    ax.text(0.5, 1.015, f"circled: 200 examples and 30 variants "
            f"= +{y:.1f} points from nothing",
            transform=ax.transAxes, ha="center", fontsize=10,
            color="#2e86c1", fontweight="bold")
    ax.grid(alpha=0.22, lw=0.7)
    ax.set_axisbelow(True)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    ax.set_xlim(1.8, 260)
    ax.set_ylim(0, None)

    # The cell most readers are standing in, called out rather than left to
    # be found: a couple of hundred eval examples and a few dozen variants
    # is an ordinary afternoon, and it is the cell where the floor is large.
    # No callout arrow. The point worth noticing sits in the middle of the
    # plot, so a leader line to it crosses two or three curves whichever
    # corner it starts from -- both earlier attempts did, in different
    # directions. Ringing the point and letting the axes name it costs the
    # reader one glance and costs the chart nothing.
    ax.plot([30], [y], marker="o", ms=13, mfc="none", mec="#2e86c1", mew=2.2,
            zorder=5)
    ax.plot([30], [y], marker="o", ms=4.2, color="#2e86c1", zorder=6)
    fig.tight_layout()
    out = Path(__file__).resolve().parents[1] / "docs" / "floor.png"
    fig.savefig(out, bbox_inches="tight", facecolor="white")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
