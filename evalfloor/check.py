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
        v = ("larger than selection alone produces. It was still measured on "
             "the data you searched over, so confirm it with confirm()"
             if self.beats_floor else
             "a search this size reports at least this much on data with no "
             "real differences at all")
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
            f"  {'ABOVE THE FLOOR' if self.beats_floor else 'BELOW THE FLOOR'}: {v}",
        ])


def selection_floor(k: int, n: int, p: float, reps: int = 4000, seed: int = 0) -> float:
    """Expected gain of the best of k candidates that are all equally good.

    Each candidate's score is an n-example estimate of the same true p, so
    the spread between them is sampling noise and nothing else. The mean of
    the maximum, minus p, is what a search of this size reports for free.

    WHAT THIS ASSUMES, and when the number it returns is too low:

    * **Independent candidates.** Real variants are often edits of each
      other, so their errors correlate. Correlation usually reduces the
      spread of the maximum, which makes this estimate conservative -- but
      when every candidate is scored on the SAME fixed eval set, part of the
      per-candidate noise becomes a run-level term (that eval set can itself
      be lucky) which does not average out at all.
    * **One evaluation per candidate.** Staged or cascaded evaluation --
      score everything cheaply, promote a few, re-score the survivors -- is
      a different and usually worse object, because the reported maximum
      tends to come from whichever rung most candidates stopped at, which is
      the cheapest and noisiest one. If your loop has promotion thresholds,
      treat this as a lower bound.
    * **A binomial metric.** Scores here are means of per-example 0/1
      outcomes. For an unbounded or heavy-tailed metric, the floor is
      typically higher than this returns.

    In every one of those cases the error is in the same direction: the real
    floor is higher, so a gain that fails this test fails it for certain.
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
            tail = ("CONFIRMED: the winner is better on data it was not "
                    "selected on")
        elif self.underpowered:
            tail = (f"UNDERPOWERED: every disagreement favours the winner "
                    f"({self.wins}-{self.losses}), but {d} of them cannot reach "
                    f"p<0.05: the floor for {d} pairs is {2.0 ** (1 - d):.4f}. "
                    f"Your held-out split is too small to settle this. Add "
                    f"examples; about {6 - d} more disagreements would decide it")
        else:
            tail = "NOT CONFIRMED: not distinguishable from chance"
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


# ------------------------------------------------------- staged evaluation
#
# Everything above assumes each candidate was evaluated once. Many real
# loops are staged instead: score everything on something cheap, promote the
# survivors to something dearer, report the best. That is a different and
# usually worse object, for a reason that is not obvious.

@dataclass
class StagedFloor:
    floor: float
    reported: float
    baseline: float
    stages: list
    #: fraction of runs whose reported maximum came from each stage
    from_stage: dict

    @property
    def headline_stage(self) -> tuple:
        return self.stages[max(self.from_stage, key=self.from_stage.get)]

    def __str__(self) -> str:
        n, _t = self.headline_stage
        rows = [
            f"  stages                {' -> '.join(str(n) for n, _ in self.stages)}"
            f" examples",
            f"  baseline              {self.baseline:.3f}",
            f"  reported best         {self.reported:.3f}",
            f"  selection floor       {self.floor:+.3f}   <- with NO real "
            f"difference between candidates",
            "",
            "  the reported best came from:",
        ]
        for i, (sn, st_) in enumerate(self.stages):
            f = self.from_stage.get(i, 0.0)
            thr = "" if st_ is None else f", promote above {st_:.0%}"
            rows.append(f"    stage {i + 1}: {sn:>4} examples{thr:<22}  {f:>5.0%}")
        rows += [
            "",
            f"  Your headline number is coming from the {n}-example stage "
            f"{self.from_stage[max(self.from_stage, key=self.from_stage.get)]:.0%}"
            f" of the time.",
        ]
        if n != self.stages[-1][0]:
            rows.append(
                f"  That is not the {self.stages[-1][0]}-example stage you pay for. "
                f"A strict\n  promotion threshold means few candidates ever reach "
                f"it, so the max\n  is taken over scores from a cheaper, noisier "
                f"stage.")
        return "\n".join(rows)


def staged_floor(stages, k: int, p: float, nested: bool = False,
                 reps: int = 1500, seed: int = 0) -> StagedFloor:
    """Selection floor for a loop that evaluates in cheap-then-dear stages.

    `stages` is [(n_examples, promote_above), ...] with the last threshold
    None -- e.g. [(10, 0.0), (60, 0.4), (200, None)] for "smoke test on 10,
    estimate on 60, confirm the promising ones on 200". `k` is how many
    candidates went through, `p` their common true score.

    Set `nested=True` if the confirmation set CONTAINS the earlier one --
    200 examples of which 60 are the ones selection was made on. Re-using
    them means confirmation can only dilute the selection noise, never
    remove it.

    The number that matters here is usually not the floor itself but the
    breakdown underneath it. A promotion threshold well above the true score
    means almost nothing is promoted, so the reported maximum is a maximum
    over scores from whichever stage candidates actually stopped at -- the
    cheapest and noisiest one. The expensive stage the loop pays for is not
    where its headline number comes from.
    """
    rng = random.Random(seed)
    stages = [(int(n), t) for n, t in stages]
    best_vals, best_stage = [], []
    for _ in range(reps):
        recorded = []
        for _c in range(max(1, k)):
            prev_n, prev_hits, idx = 0, 0, 0
            for idx, (n, thr) in enumerate(stages):
                if nested and prev_n:
                    new = max(0, n - prev_n)
                    hits = prev_hits + sum(rng.random() < p for _ in range(new))
                else:
                    hits = sum(rng.random() < p for _ in range(n))
                score = hits / n
                prev_n, prev_hits = n, hits
                if thr is None or score <= thr:
                    break
            recorded.append((score, idx))
        top = max(recorded)
        best_vals.append(top[0])
        best_stage.append(top[1])
    mean_best = st.mean(best_vals)
    counts = {i: best_stage.count(i) / len(best_stage)
              for i in range(len(stages)) if best_stage.count(i)}
    return StagedFloor(floor=mean_best - p, reported=mean_best, baseline=p,
                       stages=stages, from_stage=counts)
