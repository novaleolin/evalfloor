"""The search loop, and an honest account of what it found.

Any search that keeps the best-scoring candidate reports its own luck. With
k candidates scored on n examples, the maximum is biased upward even when
every candidate is equally good, and the bias grows with k. Reporting the
winner's score without that context is the single most common way a tuning
loop overstates itself.

So this optimiser reports four numbers, not one:

  reported    the winner's score on the data the search ran over
  held-out    the same schema on data the search never saw
  null        what a search of this size would have reported with NO real
              difference between candidates -- the selection floor
  shrunk      the winner's score after empirical-Bayes shrinkage toward the
              archive mean, which removes most of the selection bias without
              needing a held-out split at all

A gain worth believing clears the null and survives on held-out data.
"""
from __future__ import annotations

import math
import random
import statistics as st
from dataclasses import dataclass, field
from typing import Callable

from .check import confirm as _confirm, eb_shrink, selection_floor
from .schema import Mutation, Schema
from .task import Task, predict


@dataclass
class Candidate:
    schema: Schema
    train: float
    n_train: int
    note: str = ""


@dataclass
class Result:
    best: Schema
    baseline_train: float
    reported: float
    heldout_baseline: float
    heldout_best: float
    null_gain: float
    shrunk: float
    wins: int = 0            # held-out examples the winner fixed
    losses: int = 0          # held-out examples the winner broke
    p_value: float = 1.0     # exact sign test on those discordant pairs
    archive: list[Candidate] = field(default_factory=list)
    evaluations: int = 0

    @property
    def real_gain(self) -> float:
        return self.heldout_best - self.heldout_baseline

    @property
    def underpowered(self) -> bool:
        """Could this comparison have reached significance at all?

        An exact sign test over d discordant pairs cannot go below 2^(1-d)
        two-sided, so with five or fewer it cannot reach 0.05 however
        one-sided they are. Reporting that as "not credible" states a fact
        about the size of the held-out split and dresses it as a fact about
        the schema. They are not the same, and conflating them is how a real
        improvement gets discarded -- so the two are reported separately and
        the remedy, more held-out examples, is named.
        """
        d = self.wins + self.losses
        return d > 0 and 2.0 ** (1 - d) > 0.05 and self.real_gain > 0

    @property
    def credible(self) -> bool:
        """Is the held-out gain more than a coin flip would give?

        Note what this does NOT do: it does not require the TRAIN gain to
        clear the selection floor. An earlier version did, and called a run
        not credible whose held-out gain was more than twice its train gain.
        That was a category error. The floor exists to discount the score the
        search SELECTED ON; the held-out score was never selected on, so the
        floor has no claim on it. The two numbers answer different questions
        and only one of them is evidence.
        """
        return self.real_gain > 0 and self.p_value < 0.05

    def report(self) -> str:
        d = self.wins + self.losses
        if self.credible:
            verdict, why = "CREDIBLE", (
                "the held-out gain is significant; the train number is inflated "
                "by the selection floor and should not be quoted")
        elif self.underpowered:
            need = 6 - d
            verdict, why = "UNDERPOWERED", (
                f"every discordant pair favours the winner ({self.wins}-{self.losses}), "
                f"but {d} of them cannot reach p<0.05 -- the floor for {d} pairs is "
                f"{2.0 ** (1 - d):.4f}. This says the held-out split is too small, "
                f"not that the schema failed. Add held-out examples "
                f"(about {need} more discordant pairs would settle it)")
        else:
            verdict, why = "NOT CREDIBLE", (
                "the held-out gain is not distinguishable from chance")
        L = [
            f"  candidates evaluated      {len(self.archive)}",
            f"  backend calls             {self.evaluations}",
            "",
            f"  baseline (train)          {self.baseline_train:.3f}",
            f"  winner   (train)          {self.reported:.3f}   "
            f"apparent gain {self.reported - self.baseline_train:+.3f}",
            f"  selection floor (null)    {self.null_gain:+.3f}   "
            f"<- what a {len(self.archive)}-candidate search reports on pure noise",
            f"  winner, EB-shrunk         {self.shrunk:.3f}",
            "",
            f"  baseline (held-out)       {self.heldout_baseline:.3f}",
            f"  winner   (held-out)       {self.heldout_best:.3f}   "
            f"real gain {self.real_gain:+.3f}",
            f"  held-out paired           {self.wins} fixed / {self.losses} broken"
            f"   sign test p={self.p_value:.4f}",
            "",
            f"  verdict: {verdict} -- {why}",
        ]
        return "\n".join(L)


def evaluate(task: Task, schema: Schema, backend) -> float:
    total = 0.0
    for ex in task.examples:
        state = ({k: v for k, v in ex.state.items() if k in schema.state_fields}
                 if schema.state_fields is not None else ex.state)
        ans = backend.decide(state, schema.questions)
        total += task.metric(predict(ans, schema), ex.labels)
    return total / max(1, len(task.examples))






def optimize(task: Task, schema: Schema, backend, mutations: list[Mutation],
             rounds: int = 30, train_frac: float = 0.5, seed: int = 0,
             verbose: bool = True) -> Result:
    """Hill-climb over schemas on the train split, gate on held-out."""
    rng = random.Random(seed)
    train, held = task.split(train_frac, seed)
    calls = [0]

    class Counting:
        def decide(self, state, questions):
            calls[0] += 1
            return backend.decide(state, questions)

    b = Counting()
    base_train = evaluate(train, schema, b)
    cur, cur_score = schema, base_train
    archive = [Candidate(schema, base_train, len(train.examples), "baseline")]
    if verbose:
        print(f"  baseline {base_train:.3f}  (train n={len(train.examples)})")

    for i in range(rounds):
        op = rng.choice(mutations)
        cand = op(cur, rng)
        if cand is None:
            continue
        s = evaluate(train, cand, b)
        archive.append(Candidate(cand, s, len(train.examples), cand.note))
        if verbose:
            mark = "+" if s > cur_score else " "
            print(f"  [{i+1:3d}] {mark} {s:.3f}  {cand.note}")
        if s > cur_score:
            cur, cur_score = cand, s

    n = len(train.examples)
    w, l, p = paired_heldout(held, schema, cur, b)
    return Result(
        best=cur, baseline_train=base_train, reported=cur_score,
        heldout_baseline=evaluate(held, schema, b),
        heldout_best=evaluate(held, cur, b),
        wins=w, losses=l, p_value=p,
        null_gain=selection_floor(len(archive), n, base_train, seed=seed),
        shrunk=eb_shrink([c.train for c in archive], n),
        archive=archive, evaluations=calls[0],
    )


def paired_heldout(task: Task, base: Schema, best: Schema, backend):
    """Per-example comparison on data the search never saw.

    An aggregate difference hides whether a schema fixed a few examples or
    fixed many and broke nearly as many. The discordant pairs are the whole
    story, and an exact sign test over them needs no distributional
    assumption. With few discordant pairs it cannot reach significance no
    matter how one-sided they are -- which is information about the
    experiment's size, not about the schema.
    """
    wins = losses = 0
    for ex in task.examples:
        def sc(s):
            state = ({k: v for k, v in ex.state.items() if k in s.state_fields}
                     if s.state_fields is not None else ex.state)
            return task.metric(predict(backend.decide(state, s.questions), s),
                               ex.labels)
        a, b = sc(base), sc(best)
        if b > a:
            wins += 1
        elif a > b:
            losses += 1
    # the statistics live in check.py, which is the public surface and has
    # no dependencies; duplicating them here would let the two drift
    c = _confirm([0] * wins + [1] * losses, [1] * wins + [0] * losses)
    return wins, losses, c.p_value
