<div align="center">

<img src="docs/logo.png" alt="Eval Floor" width="270">

# Eval Floor: is your LLM eval improvement real?

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-23%20passing-brightgreen)](tests/)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)](pyproject.toml)

**[Quickstart](#quickstart) · [Why](#why-this-happens) · [API](#api) · [FAQ](#faq) · [简体中文](README.zh-CN.md)**

</div>

---

## What is Eval Floor?

When you try k variants against an eval set and keep the best score, that
score is biased upward. The maximum of k noisy measurements exceeds the true
value even when all k variants are equally good, and the bias grows with k.

Eval Floor computes that bias from the scores your tuning loop already
produced. No model, no rerun, no dependencies.

| | |
|---|---|
| **+7.0 points free** | 200 eval examples, 30 variants tried, no real difference between any of them |
| **2 functions** | `check()` for the floor, `confirm()` for a paired held-out test |
| **0 dependencies** | no model, no API key, no rerun. It reads scores you already have |
| **Catches both errors** | when a gain is fake, and when your split is too small to say |

## Quickstart

```bash
pip install git+https://github.com/novaleolin/evalfloor.git
```

*(PyPI release pending. `pip install evalfloor` once it lands.)*

You tried 30 prompts and kept the best one. The score went 0.62 to 0.69.

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

All 30 prompts in this example have the same true accuracy. The 6.5 point
gain is sampling noise, and the floor says so.

```bash
python3 examples/quickstart.py     # 30 seconds, no downloads, no API key
```

Runs two tuning sessions. In the first, all 30 variants have the same true
accuracy. In the second, one variant is 8 points better. Both report a gain.

## Why this happens

Each score is an estimate measured on a finite eval set, so each carries
sampling error. Taking the maximum selects for positive error. Trying more
variants raises the expected maximum.

![points a search gains when no variant is actually better](docs/floor.png)

| eval set | 5 tries | 10 tries | 30 tries | 100 tries |
| ---: | ---: | ---: | ---: | ---: |
| 50 | +8.0 | +10.4 | +13.9 | +16.8 |
| 100 | +5.6 | +7.5 | +10.0 | +12.1 |
| **200** | +4.1 | **+5.3** | **+7.0** | +8.6 |
| 500 | +2.6 | +3.4 | +4.5 | +5.5 |
| 2000 | +1.3 | +1.7 | +2.2 | +2.7 |

At 200 examples and 30 variants the floor is +7.0 points.

## Usage

### `check`: how much of your best score is luck

```python
evalfloor.check(scores, n_examples)          # scores = every variant you tried
```

`scores` must contain every variant you evaluated, not just the winner. The
floor depends on how many were tried.

### `confirm`: does the winner hold up on data it wasn't chosen on

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

An exact sign test over `d` disagreements cannot return a p-value below
`2^(1-d)`, so five or fewer can never reach 0.05. `confirm()` reports this as
UNDERPOWERED rather than as a negative result.

### `staged_floor`: for cheap-then-dear loops

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

With a 40% gate and candidates at 20%, few candidates reach stage 3. The
reported maximum comes from stage 2, so the floor is +0.110 rather than the
+0.059 of a 200-example stage. A stricter gate raises the floor.

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
pip install "evalfloor[local] @ git+https://github.com/novaleolin/evalfloor.git"
evalfloor mydata.jsonl --kind choice --metric exact
```

Optimizes a typed decision schema: instruction text, option descriptions,
which fields go into the state, and thresholds. The floor and the held-out
test are part of the output. Uses a local model by default, so no API key is
required.

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

The train gain of +0.083 is below the floor of +0.141, so the train split
supports nothing. The verdict comes from the held-out split.

The RAG example starts at F1 = 0.000: the scorer assigns 0.10 to 0.19 to
every passage, so a `0.5` threshold returns an empty set. The search reaches
0.286.

## Limits

`selection_floor` assumes independent candidates, one evaluation per
candidate, and a binomial metric. Correlated variants and heavy-tailed metrics
both raise the true floor above what it returns; staged loops use
`staged_floor` instead. All three violations push in the same direction, so a
gain below the reported floor is below the true floor as well. A gain above it
still requires `confirm()`.

## FAQ

**"I tuned my prompt 30 times and accuracy went up 5 points. Is that real?"**
Run `check()` on all 30 scores. At 200 examples the floor is +7.0, so a
5-point gain is within it.

**"How is this different from a held-out set?"**
They answer different questions. The floor is computed from the scores you
already have and identifies searches that support nothing. A held-out set is
needed to establish that a variant is better, which is what `confirm()`
tests.

**"Is this just overfitting to the eval set?"**
Related but distinct. Overfitting refers to a model fitting noise in its
training data. This is selection bias in the reporting step: no parameters are
fitted, the maximum of several noisy measurements is simply biased upward.

**"My metric isn't accuracy."**
`selection_floor` assumes a binomial metric. Unbounded and heavy-tailed
metrics have a higher true floor than it reports, so a gain below the reported
floor is still below the true one.

**"My loop promotes candidates between cheap and expensive stages."**
Use `staged_floor()`. When the promotion gate is strict, the reported
maximum usually comes from an early stage, and the floor follows that stage
rather than the final one.

## Notes

The underlying results are standard: winner's curse, selective inference,
and the expected maximum of k order statistics.

```bash
pytest tests/ -q     # 23 tests, each an attack on a claim above
```

MIT.
