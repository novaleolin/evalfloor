<div align="center">

<img src="docs/logo.png" alt="evalfloor" width="330">

### is your LLM eval improvement real?

[![PyPI](https://img.shields.io/pypi/v/evalfloor)](https://pypi.org/project/evalfloor/)
[![Python](https://img.shields.io/pypi/pyversions/evalfloor)](https://pypi.org/project/evalfloor/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-23%20passing-brightgreen)](tests/)

**[Quickstart](#quickstart) · [Why](#why-this-happens) · [API](#api) · [FAQ](#faq) · [简体中文](README.zh-CN.md)**

</div>

---

## What is evalfloor?

Tune a prompt, a threshold, a retrieval config or an agent scaffold against an
eval set; try k variants; keep the best. **The score goes up even when none of
the variants is better than the others**, because the maximum of k noisy
measurements is biased upward, and the bias grows with k.

evalfloor computes that bias — the *floor* your search clears for free — so you
can tell an improvement from a lucky sample. One line, zero dependencies, on
numbers your tuning loop already produced.

| | |
|---|---|
| **+7.0 points free** | 200 eval examples, 30 variants tried, no real difference between any of them |
| **2 functions** | `check()` for the floor, `confirm()` for a paired held-out test |
| **0 dependencies** | no model, no API key, no rerun — it reads scores you already have |
| **Catches both errors** | tells you when a gain is fake *and* when your split is too small to say |

## Quickstart

```bash
pip install evalfloor
```

You tried 30 prompts and kept the best one. The score went 0.62 → 0.69.

```python
import evalfloor
print(evalfloor.check(scores=my_30_scores, n_examples=200))
```

```
  baseline              0.620
  best                  0.685   apparent gain +0.065
  selection floor       +0.069   <- a 30-candidate search scores this much on pure noise
  best, de-biased       0.616

  BELOW THE FLOOR -- this search has not shown anything
```

All 6.5 points were luck. In that run every one of the 30 prompts was
**identical by construction** — the spread was sampling noise, and the search
found the luckiest sample.

```bash
python3 examples/quickstart.py     # 30 seconds, no downloads, no API key
```

Two tuning sessions that look the same from outside. In one, every variant is
identical and the gain is pure luck. In the other, one variant is genuinely
better. **From the final score alone you cannot tell them apart.**

## Why this happens

Take the max of 30 noisy scores and you get a high number even when all 30
options are identical. The more you try, the higher it goes.

![points a search gains when no variant is actually better](docs/floor.png)

| eval set | 5 tries | 10 tries | 30 tries | 100 tries |
| ---: | ---: | ---: | ---: | ---: |
| 50 | +8.0 | +10.4 | +13.9 | +16.8 |
| 100 | +5.6 | +7.5 | +10.0 | +12.1 |
| **200** | +4.1 | **+5.3** | **+7.0** | +8.6 |
| 500 | +2.6 | +3.4 | +4.5 | +5.5 |
| 2000 | +1.3 | +1.7 | +2.2 | +2.7 |

200 eval examples and 30 variants is a normal Tuesday. That row is **+7.0**.

## Usage

### `check` — how much of your best score is luck

```python
evalfloor.check(scores, n_examples)          # scores = every variant you tried
```

Pass **every** variant, not just the winner. The number of variants is half
of what sets the floor.

### `confirm` — does the winner hold up on data it wasn't chosen on

```python
evalfloor.confirm(baseline_correct, winner_correct)   # per-example, True/False
```
```
  held-out   13 fixed / 3 broken   sign test p=0.0213
  CONFIRMED -- the winner is better on data it was not selected on
```

It also tells you when you simply don't have enough data:

```
  held-out   5 fixed / 0 broken   sign test p=0.0625
  UNDERPOWERED -- 5 disagreements can never reach p<0.05, no matter how
                  one-sided. Your held-out split is too small. Add examples.
```

Most tools print "no improvement" there. That's wrong — it isn't that the
change failed, it's that you can't tell yet.

### `staged_floor` — for cheap-then-dear loops

Score everything on something cheap, promote survivors to something dearer,
report the best:

```python
evalfloor.staged_floor(stages=[(10, 0.0), (60, 0.40), (200, None)],
                   k=30, p=0.20, nested=True)
```
```
  selection floor       +0.110

  the reported best came from:
    stage 2:   60 examples, promote above 40%      100%
    stage 3:  200 examples                           0%

  Your headline number is coming from the 60-example stage 100% of the time.
  That is not the 200-example stage you pay for.
```

The gate is 40% and the candidates are worth 20%, so **almost nothing is ever
promoted**. The floor is the cheap stage's **+0.110**, not the 200-example
stage's +0.059. The stricter your gate, the more this bites.

### API

```python
from evalfloor import check, confirm, selection_floor, staged_floor, eb_shrink

check(scores, n_examples, baseline=None)   # .apparent_gain .floor .shrunk .beats_floor
confirm(baseline_hits, new_hits)           # .wins .losses .p_value .confirmed .underpowered
selection_floor(k, n, p)                   # points a k-candidate search gets free
staged_floor(stages, k, p, nested=False)   # same, for staged loops
eb_shrink(scores, n)                       # de-biased best
```

## A tuning loop that runs this on itself

```bash
pip install "evalfloor[local]"
evalfloor mydata.jsonl --kind choice --metric exact
```

Optimizes a typed decision schema — instruction text, option descriptions,
which fields go into the state, thresholds — and prints the floor and the
held-out test as part of its output. Runs on a small local model by default:
no API key, no cost.

```bash
python3 examples/banking77_intent.py    # ticket routing
python3 examples/rag_relevance.py       # keep-or-drop retrieved passages
```

Ticket routing, real output:

```
  winner   (train)          0.521   apparent gain +0.083
  selection floor (null)    +0.141   <- the gain is BELOW the floor

  winner   (held-out)       0.542   real gain +0.208
  held-out paired           13 fixed / 3 broken   p=0.0213

  verdict: CREDIBLE
```

The training number proved nothing; the held-out number is the whole case. A
tool printing only the first block would have called this a win.

RAG relevance starts at **F1 = 0.000** — the default `0.5` threshold everyone
ships sits above every score the model produces. Search finds 0.286.

## Limits

`selection_floor` assumes independent candidates, one evaluation each, and a
binomial metric. Correlated variants or a heavy-tailed metric push the real
floor **higher**; staged loops have their own function above. The error is
always in the same direction: **a gain that fails this test fails it for
certain.** A gain that passes still needs `confirm()`.

## FAQ

**"I tuned my prompt 30 times and accuracy went up 5 points. Is that real?"**
Run `check()` on all 30 scores. At 200 eval examples the floor is +7.0, so a
5-point gain is below what the search gets for free.

**"How is this different from a held-out set?"**
It is not a replacement — it is the step before. The floor tells you when a
search has proved nothing, using only the data you already have. A held-out
set tells you when it has proved something; `confirm()` runs that test.

**"Is this just overfitting to the eval set?"**
Same family, different mechanism. Overfitting usually means a model memorising
examples. This is selection bias: you never fit anything, you just picked the
maximum of several noisy measurements.

**"My metric isn't accuracy."**
`selection_floor` assumes a binomial metric. For an unbounded or heavy-tailed
one the real floor is higher than it reports, so a failing gain still fails.

**"My loop promotes candidates between cheap and expensive stages."**
Use `staged_floor()`. The floor is usually the *cheap* stage's, not the
expensive one you pay for — see above.

## Notes

The statistics are old — winner's curse, selective inference, expected
best-of-k. What's here is one line to get the number for your own run.

```bash
pytest tests/ -q     # 23 tests, each an attack on a claim above
```

MIT.
