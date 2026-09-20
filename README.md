# overtuned

**Check if your eval improvement is real.**

You tried 30 prompts and kept the best one. Score went 0.62 → 0.69.

```python
import overtuned
overtuned.check(scores=my_30_scores, n_examples=200)
```

```
  baseline              0.620
  best                  0.685   apparent gain +0.065
  selection floor       +0.069   <- a 30-candidate search scores this much on pure noise
  best, de-biased       0.616

  BELOW THE FLOOR -- this search has not shown anything
```

All 6.5 points were luck. In that run every one of the 30 prompts was
**exactly as good as the others** — the spread was sampling noise, and the
search found the luckiest sample.

```bash
pip install overtuned
```

Zero dependencies. Works on numbers you already have.

---

## Why this happens

Pick the max of 30 noisy scores and you get a high number even when all 30
options are identical. The more you try, the higher it goes.

![points a search gains when no variant is actually better](docs/floor.png)

**The same thing as a table** (baseline 0.60, no real difference between candidates):

| eval set | 5 tries | 10 tries | 30 tries | 100 tries |
| ---: | ---: | ---: | ---: | ---: |
| 50 | +8.0 | +10.4 | +13.9 | +16.8 |
| 100 | +5.6 | +7.5 | +10.0 | +12.1 |
| **200** | +4.1 | **+5.3** | **+7.0** | +8.6 |
| 500 | +2.6 | +3.4 | +4.5 | +5.5 |
| 2000 | +1.3 | +1.7 | +2.2 | +2.7 |

200 eval examples and 30 variants is a normal Tuesday. That row is +7.0.

---

## Two functions

**`check(scores, n_examples)`** — how much of your best score is luck.

**`confirm(baseline_hits, new_hits)`** — does the winner hold up on held-out data?

```python
overtuned.confirm(baseline_correct, winner_correct)   # per-example, True/False
```
```
  held-out   13 fixed / 3 broken   sign test p=0.0213
  CONFIRMED -- the winner is better on data it was not selected on
```

It also tells you when you don't have enough data to know:

```
  held-out   5 fixed / 0 broken   sign test p=0.0625
  UNDERPOWERED -- 5 disagreements can never reach p<0.05, no matter how
                  one-sided. Your held-out split is too small. Add examples.
```

Most tools print "no improvement" there. That's wrong — it's not that the
change failed, it's that you can't tell yet.

---

## Also: a tuning loop that runs this on itself

```bash
pip install "overtuned[local]"
overtuned mydata.jsonl --kind choice --metric exact
```

Optimizes a typed decision schema — instruction text, option descriptions,
which fields go into the state, thresholds — and prints the floor and the
held-out test as part of its output.

Runs on a small local model by default. No API key, no cost.

**Two examples, public data, offline:**

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

The training number proved nothing. The held-out number is the whole case.
A tool that printed only the first block would have called this a win.

RAG relevance starts at **F1 = 0.000** — because the default `0.5` threshold
everyone ships is above every score the model produces. Search finds 0.286.

---

## API

```python
from overtuned import check, confirm, selection_floor, eb_shrink

check(scores, n_examples, baseline=None)   # -> .apparent_gain .floor .shrunk .beats_floor
confirm(baseline_hits, new_hits)           # -> .wins .losses .p_value .confirmed .underpowered
selection_floor(k, n, p)                   # points a k-candidate search gets free
staged_floor(stages, k, p, nested=False)   # same, for cheap-then-dear loops
eb_shrink(scores, n)                       # de-biased best
```

## Try it in 30 seconds

```bash
python3 examples/quickstart.py     # no downloads, no key
```

Two tuning sessions that look identical from the outside. In one, every
variant is the same and the gain is pure luck. In the other, one variant is
genuinely better. Same number of tries, same eval set, both end higher.

## If your loop evaluates in stages

Score everything on something cheap, promote the survivors to something
dearer, report the best. Common, sensible, and worse than it looks:

```python
overtuned.staged_floor(
    stages=[(10, 0.0), (60, 0.40), (200, None)],   # (examples, promote above)
    k=30, p=0.20, nested=True)                     # 30 candidates, true score 0.20
```
```
  stages                10 -> 60 -> 200 examples
  reported best         0.310
  selection floor       +0.110   <- with NO real difference between candidates

  the reported best came from:
    stage 1:   10 examples, promote above 0%         0%
    stage 2:   60 examples, promote above 40%      100%
    stage 3:  200 examples                           0%

  Your headline number is coming from the 60-example stage 100% of the time.
  That is not the 200-example stage you pay for.
```

The gate is set at 40% and the candidates are worth 20%, so **almost nothing
is ever promoted**. The reported maximum is a maximum over 60-example scores.
Its floor is **+0.110** — the floor of the cheap stage, not the +0.059 of the
200-example stage the loop is paying for.

The stricter your promotion threshold, the more this bites.

## Limits

`selection_floor` assumes independent candidates, one evaluation each, and a
binomial metric. Correlated variants or a heavy-tailed metric push the real
floor **higher** than it reports; staged evaluation has its own function
above. The error is always in the same direction: **a gain that fails this
test fails it for certain.** A gain that passes still needs `confirm()`.

## Notes

The statistics are old — winner's curse, selective inference, expected
best-of-k. What's here is one line to get the number for your own run.

MIT. Tests: `pytest tests/ -q`
