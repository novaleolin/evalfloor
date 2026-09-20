"""The thing being optimised: a set of typed questions, not a model.

Every open System One project optimises the MODEL -- faster, smaller, local,
retrained. None of them optimises the QUESTIONS. That is what this library
searches over: the instruction text, the per-option criteria, how a decision
is decomposed into questions, what goes into the state, and the thresholds
applied to the returned probabilities.
"""
from __future__ import annotations

import copy
import random
from dataclasses import dataclass, field
from typing import Any, Callable


@dataclass
class Schema:
    """A System One decision, expressed as typed questions plus thresholds.

    `questions` follows the wire format the hosted and open implementations
    share: a dict of name -> {type: choice|noul|score, instructions, ...}.
    A `choice` carries `criteria`, a mapping of option name -> the text that
    describes it; that text is a first-class search dimension, because it is
    what the model actually reads to tell options apart.

    `state_fields` names which fields of an example are passed as state. It
    is searchable because withholding a field is a real design choice with a
    real cost -- and one that is easy to get wrong in the confident direction.
    """

    questions: dict[str, dict[str, Any]]
    thresholds: dict[str, float] = field(default_factory=dict)
    state_fields: list[str] | None = None
    note: str = ""

    def copy(self) -> "Schema":
        return Schema(copy.deepcopy(self.questions), dict(self.thresholds),
                      None if self.state_fields is None else list(self.state_fields),
                      self.note)

    def fingerprint(self) -> str:
        import hashlib
        import json
        blob = json.dumps({"q": self.questions, "t": self.thresholds,
                           "s": self.state_fields}, sort_keys=True,
                          ensure_ascii=False)
        return hashlib.sha256(blob.encode()).hexdigest()[:12]

    @property
    def kind(self) -> str:
        types = {q.get("type") for q in self.questions.values()}
        if types == {"choice"}:
            return "choice"
        if types == {"noul"}:
            return "noul"
        return "mixed"


# --------------------------------------------------------------- operators
#
# A mutation takes a Schema and returns a new one, or None when it does not
# apply. `rewrite` is any callable that improves a piece of text -- an LLM, a
# template, or a human-written list. Keeping it injected means the search
# works with no LLM at all, which is what lets the whole loop run locally.

Mutation = Callable[[Schema, random.Random], "Schema | None"]

#: A rewriter takes (text, slot, rng) and returns a replacement. The rng is
#: part of the contract, not a convenience: a rewriter that reaches for the
#: global `random` makes the whole search unreproducible, and a tool whose
#: business is telling you how much of a gain is real cannot hand you a
#: different answer each time you ask.
Rewriter = Callable[[str, str, random.Random], str]


def mutate_instructions(rewrite: "Rewriter") -> Mutation:
    """Rewrite one question's instruction text."""
    def op(s: Schema, rng: random.Random):
        names = [n for n, q in s.questions.items() if q.get("instructions")]
        if not names:
            return None
        out = s.copy()
        n = rng.choice(names)
        new = rewrite(out.questions[n]["instructions"], f"instructions for {n}", rng)
        if not new or new == out.questions[n]["instructions"]:
            return None
        out.questions[n]["instructions"] = new
        out.note = f"instructions:{n}"
        return out
    return op


def mutate_criteria(rewrite: "Rewriter") -> Mutation:
    """Rewrite the descriptive text of ONE option.

    Per-option rather than wholesale: options compete against each other, so
    the useful edit is usually sharpening the boundary between two of them,
    and a whole-block rewrite makes it impossible to see which edit paid.
    """
    def op(s: Schema, rng: random.Random):
        cands = [(n, k) for n, q in s.questions.items()
                 for k in (q.get("criteria") or {})]
        if not cands:
            return None
        out = s.copy()
        n, k = rng.choice(cands)
        new = rewrite(out.questions[n]["criteria"][k], f"option {k} of {n}", rng)
        if not new or new == out.questions[n]["criteria"][k]:
            return None
        out.questions[n]["criteria"][k] = new
        out.note = f"criteria:{n}.{k}"
        return out
    return op


def mutate_threshold(grid=(0.05, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9)) -> Mutation:
    """Move one threshold. Cheap, and often worth more than any text edit.

    The grid reaches down to 0.05 because a scorer's absolute probabilities
    are only meaningful relative to its own training, and one that puts 0.15
    on everything it likes is not broken -- it is differently calibrated. A
    grid that starts at 0.1 reports such a model as hopeless.
    """
    def op(s: Schema, rng: random.Random):
        if not s.thresholds:
            return None
        out = s.copy()
        n = rng.choice(list(out.thresholds))
        choices = [g for g in grid if g != out.thresholds[n]]
        out.thresholds[n] = rng.choice(choices)
        out.note = f"threshold:{n}={out.thresholds[n]}"
        return out
    return op


def mutate_state_fields(available: list[str]) -> Mutation:
    """Add or drop one state field.

    Included because it is the operator with the largest measured effect in
    our own prior work and the one most often decided by assumption: a field
    judged "irrelevant to a decision model" and dropped by hand turned out,
    when measured, to be worth a large accuracy gain. Vendors warn that large
    irrelevant state costs accuracy, which makes dropping feel safe; whether
    a specific field is irrelevant is an empirical question, not a stylistic
    one, so it belongs in the search.
    """
    def op(s: Schema, rng: random.Random):
        cur = list(s.state_fields if s.state_fields is not None else available)
        out = s.copy()
        missing = [f for f in available if f not in cur]
        if missing and (not cur or rng.random() < 0.5):
            f = rng.choice(missing)
            cur.append(f)
            out.note = f"state:+{f}"
        elif len(cur) > 1:
            f = rng.choice(cur)
            cur.remove(f)
            out.note = f"state:-{f}"
        else:
            return None
        out.state_fields = cur
        return out
    return op


def mutate_criteria_from_examples(examples_by_label, k=2) -> Mutation:
    """Describe an option by real utterances that belong to it.

    The default anyone writes first is the label name itself -- `card_arrival`
    as its own description -- which tells the model nothing the label did not
    already say, and leaves neighbouring labels indistinguishable. Two real
    examples usually separate them immediately.

    This operator needs no LLM, which matters: it keeps the whole search loop
    runnable offline and free, so the cost of being honest about a schema is
    not a reason to skip it.
    """
    def op(s: Schema, rng: random.Random):
        cands = [(n, k2) for n, q in s.questions.items()
                 for k2 in (q.get("criteria") or {})
                 if k2 in examples_by_label and examples_by_label[k2]]
        if not cands:
            return None
        out = s.copy()
        n, lab = rng.choice(cands)
        pool = examples_by_label[lab]
        picks = rng.sample(pool, min(k, len(pool)))
        new = "Examples: " + " | ".join(f'"{p}"' for p in picks)
        if new == out.questions[n]["criteria"][lab]:
            return None
        out.questions[n]["criteria"][lab] = new
        out.note = f"examples:{lab}"
        return out
    return op
