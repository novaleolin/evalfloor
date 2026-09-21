<div align="center">

<img src="docs/logo.png" alt="Eval Floor" width="270">

# Eval Floor：你的 LLM 评测提升是真的吗？

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-23%20passing-brightgreen)](tests/)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)](pyproject.toml)

**[五秒上手](#五秒上手) · [为什么会这样](#为什么会这样) · [API](#api) · [常见问题](#常见问题) · [English](README.md)**

</div>

---

## Eval Floor 是什么

对着评测集试 k 个变体、留下分数最高的那个，这个分数是有偏的。
k 个带噪测量的最大值高于真实值，哪怕这 k 个变体一样好，而且偏差随 k 增长。

Eval Floor 从你的调参循环已经产出的分数里算出这个偏差。
不需要模型，不需要重跑，零依赖。

| | |
|---|---|
| **白涨 7.0 个点** | 评测集 200 条、试了 30 个变体，而它们之间毫无真实差异 |
| **两个函数** | `check()` 算地板，`confirm()` 做留出集配对检验 |
| **零依赖** | 不要模型、不要 API key、不用重跑，只读你已有的分数 |
| **两类错误都管** | 既告诉你提升是假的，也告诉你数据还不够、判不了 |

## 五秒上手

```bash
pip install git+https://github.com/novaleolin/evalfloor.git
```

*（PyPI 发布中，上线后可直接 `pip install evalfloor`。）*

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

这个例子里 30 个 prompt 的真实准确率相同。6.5 个点的涨幅是抽样噪声，地板把它标了出来。

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

### `staged_floor`：给先便宜后昂贵的分级循环用

先用便宜的评测筛一遍，晋级的再用贵的评，报告最好的那个：

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

## 附带：一个会对自己做这套检查的调参循环

```bash
pip install "evalfloor[local] @ git+https://github.com/novaleolin/evalfloor.git"
evalfloor mydata.jsonl --kind choice --metric exact
```

优化一个类型化决策 schema：指令文本、选项描述、哪些字段进 state、阈值。
地板和留出检验是输出的一部分。默认用本地模型，不需要 API key。

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

训练集的 +0.083 低于地板 +0.141，所以训练集不支持任何结论。判定来自留出集。

RAG 示例从 F1 = 0.000 起步：打分器给每个段落 0.10 到 0.19，阈值 `0.5` 会返回空集。
搜索最后到 0.286。

## 适用边界

`selection_floor` 假设候选相互独立、每个候选只评测一次、指标是二项型。
候选相关与指标重尾都会让真实地板高于它返回的值；分级循环用 `staged_floor`。
三种违反的方向一致，所以低于报告地板的涨幅也低于真实地板。高于它的仍然需要 `confirm()`。

## 常见问题

**「我调了 30 次 prompt，准确率涨了 5 个点，这是真的吗？」**
把 30 个分数全传进 `check()`。200 条时地板是 +7.0，5 个点的涨幅在地板之内。

**「这和留出集有什么区别？」**
两者回答的问题不同。地板只用你已有的分数计算，用来识别不支持任何结论的搜索。
要确认某个变体确实更好，需要留出集，那是 `confirm()` 做的事。

**「这不就是过拟合评测集吗？」**
相关但不同。过拟合指模型拟合了训练数据里的噪声。
这里是报告环节的选择偏差：没有拟合任何参数，只是几个带噪测量的最大值本身有偏。

**「我的指标不是准确率。」**
`selection_floor` 假设指标是二项型。无界或重尾指标的真实地板高于它返回的值，
所以低于报告地板的涨幅也低于真实地板。

**「我的循环是分级的，便宜档筛完再上贵档。」**
用 `staged_floor()`。晋级门槛严时，报告的最大值通常来自靠前的那一级，地板跟随那一级而不是最后一级。

## 说明

底层结果都是标准的：胜者诅咒、选择性推断、k 个次序统计量最大值的期望。

```bash
pytest tests/ -q     # 23 个测试，每个都攻击上面的一句话
```

MIT 协议。
