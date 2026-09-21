#!/usr/bin/env python3
"""Regenerate docs/logo.png. Run only when the mark changes.

    pip install matplotlib && python3 tools/make_logo.py

The mark is the concept, not decoration: a horizontal floor with candidate
scores scattered around it, the ones under the line greyed out and the one
lucky sample above it picked out. That is what the library computes.
"""
import random
import sys
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

INK, FLOOR, HIT = "#1b2733", "#c0392b", "#2e86c1"


def main():
    rng = random.Random(4)
    fig, ax = plt.subplots(figsize=(7.4, 1.9), dpi=260)
    fig.patch.set_alpha(0)
    ax.set_position([0, 0, 1, 1])
    ax.axis("off")

    # The mark sits inside the wordmark's cap height, not above it. A first
    # version spread the dots over twice that and put the floor line well
    # clear of them, which read as two unrelated objects stacked up rather
    # than one idea: scores under a line, one lucky sample over it.
    # Axes units are not square here -- the figure is ~4x wider than tall, so
    # a vertical range that looks small in data units renders tall. The mark
    # is therefore wide and shallow: widening it and compressing the scatter
    # is what makes it read as one compact object next to the word.
    x0, w = 0.030, 0.235
    lo, hi, line = 0.455, 0.545, 0.575
    ys = [lo + (hi - lo) * rng.random() for _ in range(18)]
    ys[11] = line + 0.038
    for i, y in enumerate(ys):
        px = x0 + w * i / (len(ys) - 1)
        above = y > line
        ax.plot([px], [y], "o", ms=6.4 if above else 4.0,
                color=HIT if above else "#a8b6c2", zorder=3)
    ax.plot([x0 - 0.010, x0 + w + 0.010], [line, line], lw=2.6, color=FLOOR,
            zorder=2, solid_capstyle="round")

    # Display name is CamelCase; the package, import and CLI stay lowercase
    # `evalfloor`, following PEP 8 for module names.
    ax.text(x0 + w + 0.045, 0.50, "EvalFloor", fontsize=44, color=INK,
            fontweight="bold", va="center", family="DejaVu Sans")
    ax.set_xlim(0, 1)
    ax.set_ylim(0.40, 0.64)

    out = Path(__file__).resolve().parents[1] / "docs" / "logo.png"
    fig.savefig(out, transparent=True, bbox_inches="tight", pad_inches=0.06)
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
