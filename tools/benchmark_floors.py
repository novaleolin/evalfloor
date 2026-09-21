#!/usr/bin/env python3
"""Selection floor for benchmarks people actually tune against.

    python3 tools/benchmark_floors.py

`p` is a plausible current accuracy on each benchmark; the floor moves
slowly with it, so the numbers stay useful even if your model sits a few
points away. Item counts are the public test-split sizes.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalfloor import selection_floor

BENCHMARKS = [
    ("AIME 2025",            30, 0.30),
    ("MT-Bench",             80, 0.70),
    ("HumanEval",           164, 0.75),
    ("GPQA Diamond",        198, 0.45),
    ("MBPP",                378, 0.70),
    ("Arena-Hard",          500, 0.50),
    ("SWE-bench Verified",  500, 0.40),
    ("MMLU-Pro (1 subject)", 800, 0.60),
    ("GSM8K",              1319, 0.85),
    ("MMLU (full)",       14042, 0.70),
]
TRIES = (10, 30, 100)


def main(reps=2000):
    print(f"{'benchmark':23s}{'items':>7s}{'p':>6s}" +
          "".join(f"{k:>7d} tries" for k in TRIES))
    for name, n, p in BENCHMARKS:
        cells = "".join(f"{100 * selection_floor(k, n, p, reps=reps, seed=1):>+13.1f}"
                        for k in TRIES)
        print(f"{name:23s}{n:>7d}{p:>6.2f}{cells}")
    print("\npoints of accuracy a search gains when no variant is better")


if __name__ == "__main__":
    main()
