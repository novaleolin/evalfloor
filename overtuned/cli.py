"""askfit <data.jsonl> -- optimise a decision schema and report honestly."""
from __future__ import annotations

import argparse
import json
import random
import sys

from .backends import JevBackend, LocalBackend
from .optimize import optimize
from .schema import (Schema, mutate_criteria_from_examples, mutate_instructions,
                     mutate_threshold)
from .task import METRICS, from_jsonl


def _wording(text, slot, rng):
    """Default LLM-free rewriter: a handful of standard framings.

    Deliberately not an LLM call. A tool that needs an API key to take its
    first step does not get taken; and the operators that need no model at
    all -- thresholds, option descriptions drawn from labelled examples --
    are usually where the first large gain is anyway.
    """
    v = [text,
         f"{text.rstrip('?.')}? Answer based only on what the state says.",
         f"{text.rstrip('?.')}? Choose the option the text is actually about, "
         f"not one it merely mentions.",
         f"{text.rstrip('?.')}? If the state does not settle it, prefer the "
         f"more conservative option."]
    return rng.choice([x for x in v if x != text])


def build_schema(task, kind: str) -> Schema:
    labels = sorted({l for e in task.examples for l in e.labels})
    if kind == "choice":
        return Schema(
            questions={"label": {"type": "choice",
                                 "instructions": "Which label fits the state?",
                                 "criteria": {l: l.replace("_", " ") for l in labels}}},
            thresholds={"label": 0.9})
    return Schema(
        questions={l: {"type": "noul",
                       "instructions": f"Does the state match: {l.replace('_',' ')}?"}
                   for l in labels},
        thresholds={l: 0.5 for l in labels})


def main(argv=None):
    ap = argparse.ArgumentParser(prog="overtuned")
    ap.add_argument("data", help="JSONL with a state object and a labels list")
    ap.add_argument("--kind", choices=["choice", "noul"], default="choice")
    ap.add_argument("--metric", choices=sorted(METRICS), default="exact")
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--limit", type=int, default=0, help="subsample the data")
    ap.add_argument("--backend", choices=["local", "jev"], default="local")
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    ap.add_argument("--state-key", default="state")
    ap.add_argument("--label-key", default="labels")
    ap.add_argument("--save", default="", help="write the winning schema here")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args(argv)

    task = from_jsonl(a.data, a.state_key, a.label_key, a.metric)
    if a.limit:
        task = task.subsample(a.limit, a.seed)
    print(f"{a.data}: {len(task.examples)} examples, metric={a.metric}, kind={a.kind}")

    backend = (JevBackend(a.model if a.backend == "jev" else "typesafe/jev-1.13")
               if a.backend == "jev" else LocalBackend(a.model))
    by_label: dict[str, list[str]] = {}
    for e in task.examples:
        text = next((str(v) for v in e.state.values() if isinstance(v, str)), "")
        for l in e.labels:
            by_label.setdefault(l, []).append(text)

    res = optimize(task, build_schema(task, a.kind), backend,
                   mutations=[mutate_criteria_from_examples(by_label),
                              mutate_instructions(_wording),
                              mutate_threshold()],
                   rounds=a.rounds, seed=a.seed)
    print("\n" + "=" * 68)
    print(res.report())
    print("=" * 68)
    if a.save:
        with open(a.save, "w", encoding="utf-8") as f:
            json.dump({"questions": res.best.questions,
                       "thresholds": res.best.thresholds,
                       "state_fields": res.best.state_fields}, f,
                      ensure_ascii=False, indent=2)
        print(f"\nwinning schema -> {a.save}")
    return 0 if res.credible else 1


if __name__ == "__main__":
    sys.exit(main())
