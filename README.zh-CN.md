<div align="center">

<img src="docs/logo.png" alt="EvalFloor" width="270">

# EvalFloor：你的 LLM 评测提升是真的吗？

[![PyPI](https://img.shields.io/pypi/v/evalfloor)](https://pypi.org/project/evalfloor/)
[![Python](https://img.shields.io/pypi/pyversions/evalfloor)](https://pypi.org/project/evalfloor/)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)](pyproject.toml)

**[五秒上手](#五秒上手) · [为什么会这样](#为什么会这样) · [API](#api) · [常见问题](#常见问题) · [English](README.md)**

</div>

---

## EvalFloor 是什么

对着评测集试 k 个变体、留下分数最高的那个，这个分数是有偏的。
k 个带噪测量的最大值高于真实值，哪怕这 k 个变体一样好，而且偏差随 k 增长。

EvalFloor 从你的调参循环已经产出的分数里算出这个偏差。
不需要模型，不需要重跑，零依赖。

| | |
|---|---|
| **白涨 7.0 个点** | 评测集 200 条、试了 30 个变体，而它们之间毫无真实差异 |
| **两个函数** | `check()` 算地板，`confirm()` 做留出集配对检验 |
| **零依赖** | 不要模型、不要 API key、不用重跑，只读你已有的分数 |
| **两类错误都管** | 既告诉你提升是假的，也告诉你数据还不够、判不了 |

## 五秒上手

```bash
pip install evalfloor
```

你试了 30 个 prompt，留下分数最高的那个。分数从 0.62 涨到 0.69。

```python
import evalfloor
print(evalfloor.check(scores=my_30_scores, n_examples=200))
```

```
  baseline              0.620
  best                  0.685   apparent gain +0.065
  selection floor       +0.069   <- 30 个候选的搜索，在纯噪声上就能拿到这么多
  best, de-biased       0.616

  BELOW THE FLOOR -- this search has not shown anything
```

这个例子里 30 个 prompt 的真实准确率相同。6.5 个点的涨幅是抽样误差，落在地板之下。

```bash
python3 examples/quickstart.py     # 不下载任何东西，不需要 key
```

跑两次调参。第一次 30 个变体真实准确率相同；第二次有一个高 8 个点。两次都报出涨幅。

## 为什么会这样

每个分数都是在有限评测集上得到的估计，带抽样误差。取最大值等于挑选正误差。
试的变体越多，最大值的期望越高。

![一次搜索在没有任何真实差异时能拿到多少分](docs/floor.png)

| 评测集 | 试 5 次 | 试 10 次 | 试 30 次 | 试 100 次 |
| ---: | ---: | ---: | ---: | ---: |
| 50 | +8.0 | +10.4 | +13.9 | +16.8 |
| 100 | +5.6 | +7.5 | +10.0 | +12.1 |
| **200** | +4.1 | **+5.3** | **+7.0** | +8.6 |
| 500 | +2.6 | +3.4 | +4.5 | +5.5 |
| 2000 | +1.3 | +1.7 | +2.2 | +2.7 |

评测集 200 条、试 30 个变体时，地板是 +7.0 个点。

## 用法

### `check`：你的最高分里有多少是运气

```python
evalfloor.check(scores, n_examples)          # scores = 你试过的每一个变体
```

`scores` 要包含你评测过的每一个变体，不只是赢的那个。地板取决于试了多少个。

### `confirm`：赢家在没被用来挑选的数据上还站得住吗

```python
evalfloor.confirm(baseline_correct, winner_correct)   # 逐样本，True/False
```
```
  held-out   13 fixed / 3 broken   sign test p=0.0213
  CONFIRMED -- the winner is better on data it was not selected on
```

它也会告诉你"数据还不够、判不了"：

```
  held-out   5 fixed / 0 broken   sign test p=0.0625
  UNDERPOWERED -- 5 disagreements can never reach p<0.05, no matter how
                  one-sided. Your held-out split is too small. Add examples.
```

大多数工具在这里会打印"没有提升"。`d` 个不一致对的精确符号检验，p 值下限是 `2^(1-d)`，所以 5 个及以下永远到不了 0.05。
`confirm()` 把这种情况报成 UNDERPOWERED，而不是报成负结果。

### `staged_floor`：给分级评测用

先在小集合上把所有候选评一遍，晋级的再评大集合，报告最好的那个：

```python
evalfloor.staged_floor(stages=[(10, 0.0), (60, 0.40), (200, None)],
                   k=30, p=0.20, nested=True)
```
```
  selection floor       +0.110

  the reported best came from:
    stage 2:   60 examples, promote above 40%      100%
    stage 3:  200 examples                           0%
```

门槛 40%、候选真实水平 20% 时，很少有候选能进到第三级。
报告的最大值来自第二级，所以地板是 +0.110，而不是 200 题那级的 +0.059。门槛越严，地板越高。

### API

```python
from evalfloor import check, confirm, selection_floor, staged_floor, eb_shrink

check(scores, n_examples, baseline=None)   # .apparent_gain .floor .shrunk .beats_floor
confirm(baseline_hits, new_hits)           # .wins .losses .p_value .confirmed .underpowered
selection_floor(k, n, p)                   # k 个候选的搜索白拿多少分
staged_floor(stages, k, p, nested=False)   # 同上，分级循环版
eb_shrink(scores, n)                       # 去偏后的最优值
```

## Schema 优化器（可选）

```bash
pip install "evalfloor[local]"
evalfloor mydata.jsonl --kind choice --metric exact
```

在类型化决策 schema 上做搜索：指令文本、选项描述、哪些字段进 state、阈值。
每次运行都会打印自己的地板和留出检验。默认后端是本地模型，不需要 API key。

```bash
python3 examples/banking77_intent.py    # 工单意图路由
python3 examples/rag_relevance.py       # 检索段落的留/弃判断
```

工单路由的真实输出：

```
  winner   (train)          0.521   apparent gain +0.083
  selection floor (null)    +0.141   <- 表观增益在地板之下

  winner   (held-out)       0.542   real gain +0.208
  held-out paired           13 fixed / 3 broken   p=0.0213

  verdict: CREDIBLE
```

这里训练集的 +0.083 低于地板 +0.141，所以判定只能靠留出集。

RAG 示例从 F1 = 0.000 起步。打分器给每个段落 0.10 到 0.19，默认阈值 `0.5` 会返回空集。
搜索把阈值降下来之后到 0.286。

## 适用边界

`selection_floor` 假设候选相互独立、每个只评一次、指标是 0/1 型。
候选之间相关、指标重尾，都会让真实地板比它报的更高。分级循环改用 `staged_floor`。

这几种情况都让真实地板变高，所以它报的是个下界。
低于这个地板的涨幅，也低于真实地板；高于它的仍然要跑 `confirm()`。

## 常见问题

**我调了 30 次 prompt，准确率涨了 5 个点，这是真的吗？**
大概率不是。评测集 200 条时，30 个变体的地板是 +7.0。
把 30 个分数全传进 `check()`，可以算出你自己那组的地板。

**这和留出集有什么区别？**
留出集用来确认某个变体更好。地板用来判断你手上的分数能不能支持任何结论，
在你把留出数据花掉之前。先跑 `check()`，活下来的再跑 `confirm()`。

**这不就是过拟合评测集吗？**
这里没有拟合任何东西。偏差来自"在几个带噪分数里挑最大的"这个动作本身，
跟有没有训练模型无关。

**我的指标不是准确率。**
`selection_floor` 假设指标是 0/1 型。其他指标下它会报低，
所以低于它的涨幅同样低于真实地板。

**我的循环是分级的，先在小集合上筛，再评大集合。**
用 `staged_floor()`。它还会告诉你最大值来自哪一级，通常不是最大的那一级。

**超参搜索、A/B 测试、模型选型能用吗？**
能。只要是"评测 k 个选项、留下最好的"，就有这个偏差。

MIT 协议。
