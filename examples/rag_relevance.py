#!/usr/bin/env python3
"""Which retrieved passages are worth sending to the LLM?

    python3 examples/rag_relevance.py

The other example routes a ticket with one `choice`. This one keeps or drops
each retrieved passage with a `noul` per passage, which is the decision a
retrieval-augmented pipeline makes hundreds of times per query -- and the
place a fast decision model actually pays for itself, because the volume is
there. It also makes the threshold a first-class knob rather than a detail:
on this task the threshold IS the precision/recall trade-off.

MS MARCO ships candidate passages alongside each query with the relevant one
marked, so the example is self-contained -- no separate corpus download, no
index to build. Everything runs against a small local model at zero cost.
"""
import argparse
import random
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from fluke.backends import LocalBackend
from fluke.optimize import optimize
from fluke.schema import Schema, mutate_instructions, mutate_threshold
from fluke.task import Example, Task, f1


def load(n_queries=40, max_passages=6, seed=0):
    from datasets import load_dataset
    # Sequential iteration, not ds[i] over a shuffled index list. Random
    # access into a large Arrow file re-reads a block per row and turned a
    # few-second load into one that never finished; streaming through it in
    # order and stopping early is what the format is built for.
    ds = load_dataset("ms_marco", "v1.1", split="validation")
    skip = random.Random(seed).randrange(0, max(1, len(ds) - n_queries * 20))
    xs = []
    for r in ds.select(range(skip, min(len(ds), skip + n_queries * 20))):
        texts = r["passages"]["passage_text"][:max_passages]
        sel = r["passages"]["is_selected"][:max_passages]
        if not any(sel) or len(texts) < 3:
            continue          # a query with nothing to find teaches nothing
        state = {"query": r["query"]}
        for j, t in enumerate(texts):
            state[f"passage_{j}"] = t
        xs.append(Example(state, {f"passage_{j}" for j, s in enumerate(sel) if s}))
        if len(xs) >= n_queries:
            break
    return Task(xs, f1, "ms_marco_relevance")


def naive_schema(max_passages=6):
    """One yes/no per passage, asked the way most pipelines ask it."""
    q = {f"passage_{j}": {
            "type": "noul",
            "instructions": f"Is passage_{j} relevant to the query?"}
         for j in range(max_passages)}
    return Schema(questions=q, thresholds={k: 0.5 for k in q})


def wording(text, slot, rng):
    """Free instruction variants. `slot` names the question being rewritten."""
    name = slot.split()[-1]
    variants = [
        f"Is {name} relevant to the query?",
        f"Does {name} contain information that answers the query? "
        f"Being on the same topic is not enough.",
        f"Would a careful assistant cite {name} when answering the query? "
        f"Answer yes only if it carries the specific fact the query asks for.",
        f"Judge {name} against the query. Say yes if dropping it would make "
        f"the answer worse, no if the answer would be unchanged.",
    ]
    return rng.choice([v for v in variants if v != text])


def shared_threshold():
    """Move every passage's threshold together.

    Per-question thresholds are the default the library offers, but here the
    questions are interchangeable -- passage_3 is not a different KIND of
    decision from passage_1 -- so tuning them independently spends the search
    budget fitting which slot happened to be easy in the training split. One
    shared knob is both the right model of the task and far cheaper to fit.
    """
    # Spans an order of magnitude at the low end on purpose. A model's noul
    # probabilities are only calibrated relative to how it was trained, and a
    # small open scorer put 0.10-0.19 on every passage here -- so a grid
    # starting at 0.2 silently evaluates every candidate at F1 zero and makes
    # the task look impossible rather than mis-thresholded. The default 0.5
    # that pipelines ship with is a guess, and on this task it is the single
    # most costly one.
    grid = (0.05, 0.08, 0.12, 0.15, 0.18, 0.22, 0.3, 0.4, 0.5, 0.7)

    def op(s, rng):
        cur = next(iter(s.thresholds.values()), 0.5)
        out = s.copy()
        t = rng.choice([g for g in grid if g != cur])
        out.thresholds = {k: t for k in out.thresholds}
        out.note = f"threshold(all)={t}"
        return out
    return op


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rounds", type=int, default=20)
    ap.add_argument("--queries", type=int, default=15)
    ap.add_argument("--passages", type=int, default=6)
    ap.add_argument("--model", default="Qwen/Qwen2.5-0.5B-Instruct")
    a = ap.parse_args()

    task = load(a.queries, a.passages)
    n_dec = sum(len(e.state) - 1 for e in task.examples)
    print(f"ms_marco / {len(task.examples)} queries / {n_dec} keep-or-drop decisions")
    print("(each decision is its own prefill of the whole passage set, so the "
          "default is small; threshold edits are free -- they reuse the cache -- "
          "while instruction edits pay for a full re-evaluation)")
    backend = LocalBackend(a.model)
    print(f"backend: {a.model} on {backend.device}\n")

    res = optimize(task, naive_schema(a.passages), backend,
                   mutations=[mutate_instructions(wording), shared_threshold()],
                   rounds=a.rounds, seed=0)
    print("\n" + "=" * 68)
    print(res.report())
    print("=" * 68)


if __name__ == "__main__":
    main()
