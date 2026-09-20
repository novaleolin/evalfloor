"""Was that improvement real?

The whole library in two functions, neither of which needs a model, a
backend, or anything but the numbers your existing tuning loop already
produced.

    check(scores, n_examples)          -- did the search prove anything?
    confirm(baseline_hits, new_hits)   -- does the winner survive held-out data?

The problem they address is not exotic. Try k variants of a prompt, a
schema, a threshold, a retrieval config; keep the one that scored best; the
number went up. It goes up even when none of the variants is any better than
the others, because the maximum of k noisy estimates is biased upward, and
the bias grows with k. Reporting that maximum as an improvement is the most
common way a tuning session overstates itself, and almost nothing in the
LLM tooling stack checks for it.
"""
from __future__ import annotations

import math
import random
import statistics as st
from dataclasses import dataclass


@dataclass
class Check:
    n_candidates: int
    n_examples: int
    baseline: float
    best: float
    floor: float
    shrunk: float

    @property
    def apparent_gain(self) -> float:
        return self.best - self.baseline

    @property
    def beats_floor(self) -> bool:
        return self.apparent_gain > self.floor

    def __str__(self) -> str:
        v = ("the gain is larger than selection alone produces, but it was "
             "still measured on the data you searched over -- confirm it on "
             "held-out data with confirm()"
             if self.beats_floor else
             "this search has not shown anything: a search this size reports "
             "at least this much on data with no real differences at all")
        return "\n".join([
            f"  candidates tried      {self.n_candidates}",
            f"  examples each         {self.n_examples}",
            "",
            f"  baseline              {self.baseline:.3f}",
            f"  best                  {self.best:.3f}   apparent gain "
            f"{self.apparent_gain:+.3f}",
            f"  selection floor       {self.floor:+.3f}   <- what a "
            f"{self.n_candidates}-candidate search reports on pure noise",
            f"  best, de-biased       {self.shrunk:.3f}",
            "",
            f"  {'ABOVE THE FLOOR' if self.beats_floor else 'BELOW THE FLOOR'}"
            f" -- {v}",
        ])


def selection_floor(k: int, n: int, p: float, reps: int = 4000, seed: int = 0) -> float:
    """Expected gain of the best of k candidates that are all equally good.

    Each candidate's score is an n-example estimate of the same true p, so
    the spread between them is sampling noise and nothing else. The mean of
    the maximum, minus p, is what a search of this size reports for free.
    """
    rng = random.Random(seed)
    p = min(max(p, 1e-3), 1 - 1e-3)
    return st.mean(max(sum(rng.random() < p for _ in range(n)) / n
                       for _ in range(max(1, k))) for _ in range(reps)) - p


def eb_shrink(scores: list[float], n: int) -> float:
    """De-bias the winner by shrinking every candidate toward their mean.

    Splits the spread between candidates into sampling noise and signal, and
    pulls each score toward the mean in proportion to how much of its
    deviation noise alone explains. When candidates genuinely differ it
    barely moves them; when they do not, it collapses them. Nothing is
    fitted -- the shrink factor comes from the scores you already have.
    """
    if len(scores) < 3:
        return max(scores) if scores else 0.0
    m = st.mean(scores)
    sig2 = max(m * (1 - m), 1e-6) / max(1, n)
    signal = max(st.pvariance(scores) - sig2, 0.0)
    b = sig2 / (sig2 + signal) if (sig2 + signal) > 0 else 1.0
    return max(m + (1 - b) * (s - m) for s in scores)


def check(scores, n_examples: int, baseline: float | None = None,
          seed: int = 0) -> Check:
    """How much of your best score is just the search finding noise?

    `scores` is every candidate you evaluated, `n_examples` how many
    examples each was scored on. `baseline` defaults to the first score,
    which is usually the configuration you started from.
    """
    scores = list(scores)
    if not scores:
        raise ValueError("no scores given")
    base = scores[0] if baseline is None else baseline
    return Check(n_candidates=len(scores), n_examples=n_examples,
                 baseline=base, best=max(scores),
                 floor=selection_floor(len(scores), n_examples, base, seed=seed),
                 shrunk=eb_shrink(scores, n_examples))


@dataclass
class Confirm:
    wins: int
    losses: int
    p_value: float

    @property
    def underpowered(self) -> bool:
        """Could this comparison have reached significance at all?

        An exact sign test over d discordant pairs cannot go below 2^(1-d)
        two-sided, so at five or fewer it can never reach 0.05 however
        one-sided they are. Calling that "not significant" states a fact
        about the size of your held-out split while sounding like a fact
        about your change. They are different claims and only one of them
        is about the thing you changed.
        """
        d = self.wins + self.losses
        return d > 0 and 2.0 ** (1 - d) > 0.05 and self.wins > self.losses

    @property
    def confirmed(self) -> bool:
        return self.wins > self.losses and self.p_value < 0.05

    def __str__(self) -> str:
        d = self.wins + self.losses
        if self.confirmed:
            tail = "CONFIRMED -- the winner is better on data it was not selected on"
        elif self.underpowered:
            tail = (f"UNDERPOWERED -- every disagreement favours the winner "
                    f"({self.wins}-{self.losses}), but {d} of them cannot reach "
                    f"p<0.05: the floor for {d} pairs is {2.0 ** (1 - d):.4f}. "
                    f"Your held-out split is too small to settle this. Add "
                    f"examples; about {6 - d} more disagreements would decide it")
        else:
            tail = "NOT CONFIRMED -- not distinguishable from chance"
        return (f"  held-out   {self.wins} fixed / {self.losses} broken"
                f"   sign test p={self.p_value:.4f}\n  {tail}")


def confirm(baseline_hits, new_hits) -> Confirm:
    """Paired comparison on data the search never saw.

    Takes two equal-length sequences of per-example outcomes -- booleans, or
    any numbers where higher is better. Only the examples where the two
    disagree carry information; an aggregate difference hides whether a
    change fixed a handful or fixed many and broke nearly as many.
    """
    a, b = list(baseline_hits), list(new_hits)
    if len(a) != len(b):
        raise ValueError(f"paired inputs must match: {len(a)} vs {len(b)}")
    wins = sum(1 for x, y in zip(a, b) if y > x)
    losses = sum(1 for x, y in zip(a, b) if x > y)
    d = wins + losses
    if d == 0:
        return Confirm(0, 0, 1.0)
    lo = min(wins, losses)
    p = min(1.0, 2 * sum(math.comb(d, i) for i in range(lo + 1)) / (2 ** d))
    return Confirm(wins, losses, p)
