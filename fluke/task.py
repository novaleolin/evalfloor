"""A task is a list of examples plus how to score a prediction.

Deliberately thin. Anything shaped like "given some state, pick zero or more
labels from a bounded set" fits: support-ticket routing, tool selection,
moderation, retrieval relevance, lead qualification. Keeping the abstraction
at that level is what makes the optimiser domain-agnostic without needing a
plugin per domain.
"""
from __future__ import annotations

import json
import random
from dataclasses import dataclass, field
from typing import Any, Callable, Iterable


@dataclass
class Example:
    state: dict[str, Any]
    labels: set[str]
    eid: str = ""


def exact(pred: set[str], gold: set[str]) -> float:
    return float(pred == gold)


def contains(pred: set[str], gold: set[str]) -> float:
    """Gold is a subset of the prediction: nothing missed, over-calling free."""
    return float(gold <= pred)


def f1(pred: set[str], gold: set[str]) -> float:
    if not pred and not gold:
        return 1.0
    if not pred or not gold:
        return 0.0
    tp = len(pred & gold)
    if not tp:
        return 0.0
    p, r = tp / len(pred), tp / len(gold)
    return 2 * p * r / (p + r)


METRICS: dict[str, Callable[[set, set], float]] = {
    "exact": exact, "contains": contains, "f1": f1,
}


@dataclass
class Task:
    examples: list[Example]
    metric: Callable[[set, set], float] = exact
    name: str = "task"
    #: every state key any example offers -- the search space for state_fields
    state_keys: list[str] = field(default_factory=list)

    def __post_init__(self):
        if not self.state_keys:
            seen: list[str] = []
            for e in self.examples:
                for k in e.state:
                    if k not in seen:
                        seen.append(k)
            self.state_keys = seen

    def split(self, train=0.5, seed=0) -> tuple["Task", "Task"]:
        """Train / held-out split.

        The held-out half is not for measuring; it is for GATING. A search
        that keeps whatever scored best on the data it searched over reports
        its own luck. Splitting is the cheapest defence and the one most
        often skipped.
        """
        xs = list(self.examples)
        random.Random(seed).shuffle(xs)
        k = int(len(xs) * train)
        return (Task(xs[:k], self.metric, self.name + ":train", self.state_keys),
                Task(xs[k:], self.metric, self.name + ":heldout", self.state_keys))

    def subsample(self, n: int, seed=0) -> "Task":
        xs = list(self.examples)
        random.Random(seed).shuffle(xs)
        return Task(xs[:n], self.metric, self.name, self.state_keys)


def from_jsonl(path: str, state_key="state", label_key="labels",
               metric="exact", name="") -> Task:
    """Load a task from JSONL.

    Each line needs a state (an object, or a string that becomes {"text": ...})
    and labels (a list, or a single string).
    """
    xs = []
    for i, line in enumerate(open(path, encoding="utf-8")):
        d = json.loads(line)
        st = d[state_key]
        if isinstance(st, str):
            st = {"text": st}
        lb = d[label_key]
        if isinstance(lb, str):
            lb = [lb]
        xs.append(Example(st, set(lb), d.get("id", str(i))))
    return Task(xs, METRICS[metric], name or path)


def predict(answers: dict, schema, gold_labels: Iterable[str] | None = None) -> set[str]:
    """Turn backend answers into a label set, applying the schema's thresholds.

    choice: the argmax, plus any option close enough to it -- "close enough"
    being a per-question threshold expressed as a fraction of the argmax's
    probability, so it stays meaningful when the model is globally
    under- or over-confident.
    noul: every question whose probability clears its own threshold.
    """
    out: set[str] = set()
    for name, a in (answers or {}).items():
        q = schema.questions.get(name, {})
        thr = schema.thresholds.get(name, 0.5)
        if "noul" in a:
            if a["noul"] >= thr:
                out.add(name)
            continue
        probs = a.get("probabilities") or {}
        am = a.get("choice")
        if not am or am == "__none__":
            continue
        out.add(am)
        top = probs.get(am, 0.0)
        out |= {k for k, v in probs.items()
                if k != am and k != "__none__" and top and v >= thr * top}
    return out
