#!/usr/bin/env python3
"""Support-ticket intent routing, optimised end to end, offline and free.

    python3 examples/banking77_intent.py

Starts from the schema everyone writes first -- each intent described by its
own label name -- and searches over the description text, the instruction
wording and the threshold. Nothing here calls a hosted API: the decisions
come from a small open model scoring option logits locally, so the whole run
costs nothing and can be repeated.

The point of the example is not the accuracy number. It is the last block of
the output, which separates the gain the search found from the gain a search
of that size reports on pure noise.
"""
import argparse
import random
import sys
from collections import defaultdict
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from evalfloor.backends import LocalBackend
from evalfloor.optimize import optimize
from evalfloor.schema import (Schema, mutate_criteria_from_examples,
                           mutate_instructions, mutate_threshold)
from evalfloor.task import Example, Task, exact

#: A confusable cluster, not a random sample. Routing is easy when intents
#: are far apart; the schema is what earns its keep when they are close, and
#: an example that hides that is not showing the reader anything.
INTENTS = [
    "card_arrival", "card_delivery_estimate", "card_not_working",
    "card_payment_fee_charged", "declined_card_payment",
    "lost_or_stolen_card", "pending_card_payment", "card_swallowed",
]


def load(n_per_intent=12, seed=0):
    from datasets import load_dataset
    ds = load_dataset("legacy-datasets/banking77", split="test")
    names = ds.features["label"].names
    keep = {names.index(i) for i in INTENTS}
    by = defaultdict(list)
    for r in ds:
        if r["label"] in keep:
            by[names[r["label"]]].append(r["text"])
    rng = random.Random(seed)
    xs, pool = [], {}
    for lab, texts in by.items():
        rng.shuffle(texts)
        pool[lab] = texts[:6]                       # only for the operator
        for t in texts[6:6 + n_per_intent]:
            xs.append(Example({"customer_message": t}, {lab}))
    rng.shuffle(xs)
    return Task(xs, exact, "banking77"), pool


def naive_schema():
    """What a first implementation looks like: the label as its own criteria."""
    return Schema(
        questions={"intent": {
            "type": "choice",
            "instructions": "Which intent matches the customer message?",
            "criteria": {i: i.replace("_", " ") for i in INTENTS}}},
        thresholds={"intent": 0.9})


def wording(text, _slot, rng):
    """A free, LLM-free rewriter: try a few standard instruction phrasings."""
    variants = [
        "Which intent matches the customer message?",
        "Read the customer message and pick the single intent it is about.",
        "Classify the customer's message into exactly one intent below. "
        "Choose the one the customer is actually asking about, not one merely mentioned.",
        "You are routing a support ticket. Select the one intent that best "
        "describes what the customer wants resolved.",
    ]
    return rng.choice([v for v in variants if v != text])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=24)
    ap.add_argument("--per-intent", type=int, default=12)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    a = ap.parse_args()

    task, pool = load(a.per_intent)
    print(f"banking77 / {len(INTENTS)} confusable intents / {len(task.examples)} examples")
    backend = LocalBackend(a.model)
    print(f"backend: {a.model} on {backend.device}\n")

    res = optimize(
        task, naive_schema(), backend,
        mutations=[mutate_criteria_from_examples(pool),
                   mutate_instructions(wording),
                   mutate_threshold()],
        rounds=a.rounds, seed=0)

    print("\n" + "=" * 68)
    print(res.report())
    print("=" * 68)
    print("\nwinning schema changes:")
    for c in res.archive:
        if c.schema.fingerprint() == res.best.fingerprint():
            print(f"  last accepted edit: {c.note}")
            break


if __name__ == "__main__":
    main()
