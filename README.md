# overtuned

**Your eval score went up. Did anything actually get better?**

You tried 30 prompt variants and kept the best. The number moved from 0.62 to
0.70. Here is the thing nobody tells you: **a 30-candidate search reports about
+0.07 on data where every candidate is equally good.** Not sometimes — on
average.

```python
import overtuned

print(overtuned.check(scores=my_30_scores, n_examples=200))
```
```
  candidates tried      30
  examples each         200

  baseline              0.620
  best                  0.702   apparent gain +0.082
  selection floor       +0.069   <- what a 30-candidate search reports on pure noise
  best, de-biased       0.632

  ABOVE THE FLOOR -- but it was still measured on the data you searched over;
                     confirm it on held-out data with confirm()
```

That took two numbers you already have. No model, no API key, no rerun.

---

## Where this came from

I was measuring something else: whether self-improving AI systems actually
improve. Several publish a curve — best-in-archive score, rising over
iterations — and I wanted to know how much of the rise was real.

So I reimplemented one system's published selection protocol exactly, and fed
it candidates that were **all equally good, by construction**. The curve went
up anyway. Under a strict null of zero real improvement, that protocol still
reported a gain of double digits, purely from selecting the luckiest candidate
out of a growing pool.

Then the obvious thought: this is not a quirk of self-improving systems. It is
what happens to **anyone who tries several things against an eval set and
keeps the best one** — which is everyone tuning a prompt, a threshold, a
retrieval config, an agent scaffold. The statistics are textbook. The tooling
to check for it, in this ecosystem, did not exist.

So here it is.

---

## Two functions

### `check()` — did the search prove anything?

Takes the scores your loop already produced.

```python
overtuned.check(scores, n_examples)
```

It reports three things your loop does not:

- **the selection floor** — what a search of that size reports on pure noise.
  If your gain is under it, your search has shown nothing.
- **the de-biased best** — every candidate shrunk toward their mean in
  proportion to how much of its lead sampling noise alone explains. Nothing
  is fitted.
- a verdict that says which of those two you are in.

### `confirm()` — does the winner survive data it was not chosen on?

```python
overtuned.confirm(baseline_correct, winner_correct)   # per-example outcomes
```

Paired, because an aggregate difference hides whether a change fixed a handful
or fixed many and broke nearly as many. Only the disagreements carry
information, and an exact sign test over them assumes nothing.

It also refuses to lie to you in the other direction:

```
  held-out   5 fixed / 0 broken   sign test p=0.0625
  UNDERPOWERED -- every disagreement favours the winner (5-0), but 5 of them
                  cannot reach p<0.05: the floor for 5 pairs is 0.0625. Your
                  held-out split is too small to settle this. Add examples;
                  about 1 more disagreement would decide it
```

Five one-sided disagreements out of five, and it *still* cannot be
significant — because an exact sign test over `d` pairs cannot go below
`2^(1-d)`. "Not significant" there is a fact about the size of your split
wearing the costume of a fact about your change. Most tools would just print
"no improvement."

---
## The optimiser that cannot lie to you

Once you have the checker, the obvious next thing is a tuning loop that runs
it on itself. That ships too.

`overtuned.optimize()` hill-climbs over a **typed decision schema** — the
instruction text, the per-option descriptions, what goes into the state, the
thresholds — and reports the floor and the held-out confirmation as part of
its output, not as an afterthought.

```bash
overtuned mydata.jsonl --kind choice --metric exact
```

No API key. The search runs against a small local model by default, so it
costs nothing and you can run it twice.

### What it searches, and why each one is in there

| dimension | why |
| :--- | :--- |
| instruction text | the wording that tells the model what the decision is |
| per-option criteria | options compete; this text is what separates them |
| option descriptions from labelled examples | needs no LLM — two real examples usually beat any hand-written gloss |
| **state fields** | "this field is irrelevant to the decision" is an empirical claim, and a confident one. Withholding a field should be measured, not assumed |
| thresholds | the cheapest edit and often the largest. A scorer that puts 0.15 on everything it likes is differently calibrated, not broken — and a pipeline shipping `0.5` is shipping a guess |

### Two worked examples, both on public data, both offline

```bash
python3 examples/banking77_intent.py    # support-ticket routing, one `choice`
python3 examples/rag_relevance.py       # keep-or-drop retrieved passages, `noul` each
```

**Ticket routing** starts from the schema everyone writes first — each intent
described by its own label name — on a deliberately *confusable* cluster of
intents, because that is where schema design earns its keep:

```
  baseline (train)          0.438
  winner   (train)          0.521   apparent gain +0.083
  selection floor (null)    +0.141   <- 25-candidate search, on pure noise
  winner, EB-shrunk         0.453

  baseline (held-out)       0.333
  winner   (held-out)       0.542   real gain +0.208
  held-out paired           13 fixed / 3 broken   sign test p=0.0213

  verdict: CREDIBLE
```

Read the first block and the second block against each other. **The apparent
gain of +0.083 is below the floor of +0.141** — on the split it searched, this
run proved nothing. The evidence is entirely in the held-out block. A tool
that printed only the first block would have reported a win.

**RAG relevance** makes the threshold point concrete. The starting schema is
the one pipelines ship — one yes/no per passage at `0.5` — and it scores
**F1 = 0.000**, because the scorer puts 0.10–0.19 on everything it likes and
nothing clears the bar. Nothing is wrong with the model. The bar is a guess.

```
  baseline (held-out)       0.000
  winner   (held-out)       0.286   real gain +0.286
  held-out paired           7 fixed / 0 broken   sign test p=0.0156

  verdict: CREDIBLE
```

### Backends

Anything that turns state plus typed questions into probabilities.

- **`LocalBackend`** (default) — reads option logits from any causal LM in one
  prefill. Offline, free, no account. This is what the search runs on.
- **`JevBackend`** — a hosted System One model, for a final validation pass.
- **Bring your own** — implement `decide(state, questions) -> answers`.

## Install

```bash
pip install overtuned                 # check() and confirm(), zero dependencies
pip install "overtuned[local]"        # + the local backend and the optimiser
```

`check()` and `confirm()` have **no dependencies at all** — they are arithmetic
on numbers you already have.

## Reproducibility

Every run is seeded end to end, including the text rewriters: the same
`--seed` gives the same schema and the same numbers. This is not a nicety. A
tool whose business is telling you how much of a gain is real cannot hand you
a different answer each time you ask — an early version reached for the global
`random` inside a rewriter and drifted by several points between identical
runs.

```bash
python3 -m pytest tests/ -q      # every test is an attack on a claim above
```

## What this is not

- **Not a model.** It does not train, distil, or serve anything.
- **Not a replacement for a held-out set.** The floor tells you when a search
  proved nothing; only held-out data tells you when it proved something.
- **Not new statistics.** Dodge et al. (2019), *Show Your Work*, argued for
  reporting expected best-found performance as a function of search budget.
  The winner's-curse and selective-inference literatures supply the
  correction. Evolutionary computation has been re-evaluating noisy archive
  elites for decades. What is new here is none of the maths — it is that you
  can get the number for your own tuning run in one line.

MIT licensed.
