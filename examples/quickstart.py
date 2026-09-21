#!/usr/bin/env python3
"""Thirty seconds, no downloads, no API key.

    python3 examples/quickstart.py

Simulates two tuning sessions that look identical from the outside -- same
number of variants, same eval set, both ending on a higher score than they
started. In one, every variant is genuinely identical and the gain is
entirely the search finding the luckiest sample. In the other, one variant
really is better.

If you cannot tell them apart from the final score, that is the point.
"""
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalfloor import check, confirm

N_EVAL = 200          # how many examples each variant was scored on
N_TRIES = 30          # how many prompt variants were tried
TRUE_BASE = 0.62      # the accuracy the starting prompt really has


def score(true_acc, rng):
    """One measurement of a variant: n_eval Bernoulli draws, as in real life."""
    return sum(rng.random() < true_acc for _ in range(N_EVAL)) / N_EVAL


def session(gains, rng):
    """Run a tuning session. `gains` is the TRUE improvement of each variant."""
    return [score(min(0.99, TRUE_BASE + g), rng) for g in gains]


def banner(title):
    print(f"\n{'=' * 64}\n{title}\n{'=' * 64}")


def main():
    # A fixed seed, chosen so both sessions start from a baseline near the
    # true 0.62. In a real run the baseline is itself one noisy measurement,
    # which is a second reason the reported delta wobbles -- but showing that
    # on top of the selection effect would muddle a demo about the selection
    # effect.
    rng = random.Random(17)

    banner("SESSION A -- 30 prompt variants, none of them actually better")
    a = session([0.0] * N_TRIES, rng)
    print(f"  you would report: {a[0]:.3f} -> {max(a):.3f}"
          f"   (+{max(a) - a[0]:.3f})\n")
    print(check(a, N_EVAL))

    banner("SESSION B -- same setup, but one variant is genuinely +8 points")
    b = session([0.0] * (N_TRIES - 1) + [0.08], rng)
    print(f"  you would report: {b[0]:.3f} -> {max(b):.3f}"
          f"   (+{max(b) - b[0]:.3f})\n")
    print(check(b, N_EVAL))

    banner("The part that settles it: data the search never saw")
    print("""  check() tells you when a search proved nothing. It cannot tell you
  that a search proved something -- the scores it reads were all measured
  on the data you searched over. For that you need held-out examples and
  a per-example comparison.""")
    held = random.Random(11)
    base_hits = [held.random() < TRUE_BASE for _ in range(N_EVAL)]
    # the genuinely better variant, re-measured on examples it never saw
    new_hits = [h or (held.random() < 0.21) for h in base_hits]
    print()
    print(confirm(base_hits, new_hits))

    print(f"""
{'=' * 64}
  Your own run, in two lines:

      from evalfloor import check, confirm
      print(check(my_scores, n_examples=len(my_eval_set)))
      print(confirm(baseline_correct, winner_correct))

  `my_scores` is every variant you tried, not just the winner -- the
  number of variants is half of what determines the floor.
{'=' * 64}""")


if __name__ == "__main__":
    main()
