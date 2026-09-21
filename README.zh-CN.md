<div align="center">

<img src="docs/logo.png" alt="evalfloor" width="270">

# evalfloor：你的 LLM 评测提升是真的吗？

[![Python](https://img.shields.io/badge/python-3.9%2B-blue)](pyproject.toml)
[![License](https://img.shields.io/badge/license-MIT-green)](LICENSE)
[![Tests](https://img.shields.io/badge/tests-23%20passing-brightgreen)](tests/)
[![Dependencies](https://img.shields.io/badge/dependencies-none-brightgreen)](pyproject.toml)

**[五秒上手](#五秒上手) · [为什么会这样](#为什么会这样) · [API](#api) · [常见问题](#常见问题) · [English](README.md)**

</div>

---

## evalfloor 是什么

对着评测集调 prompt、调阈值、调检索配置、调 agent 脚手架——试 k 个变体，留最好的那个。
**哪怕这些变体之间没有任何真实差异，分数也会涨**，因为对 k 个带噪测量取最大值本身就有偏，
而且偏差随 k 增长。

evalfloor 把这个偏差算出来——也就是你的搜索**白拿的那条地板线**——
让你能把"真的提升"和"运气好的那次采样"分开。
一行代码，零依赖，只用你的调参循环已经产出的数字。

| | |
|---|---|
| **白涨 7.0 个点** | 评测集 200 条、试了 30 个变体，而它们之间毫无真实差异 |
| **两个函数** | `check()` 算地板，`confirm()` 做留出集配对检验 |
| **零依赖** | 不要模型、不要 API key、不用重跑——只读你已有的分数 |
| **两类错误都管** | 既告诉你提升是假的，也告诉你数据还不够、判不了 |

## 五秒上手

```bash
pip install git+https://github.com/novaleolin/evalfloor.git
```

*（PyPI 发布中——上线后可直接 `pip install evalfloor`。）*

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

这 6.5 个点全是运气。那次模拟里 30 个 prompt **在构造上完全一样好**——
它们之间的差异只是抽样噪声，搜索找到的是运气最好的那次采样。

```bash
python3 examples/quickstart.py     # 不下载任何东西，不需要 key
```

跑两次从外面看完全一样的调参：一次所有变体都一样好、涨幅纯属运气，
另一次真有一个更好。**光看最终分数你分不出哪个是哪个。**

## 为什么会这样

对 30 个带噪分数取最大值，哪怕这 30 个选项完全一样，你也会得到一个高的数。
试得越多，这个数越高。

![一次搜索在没有任何真实差异时能拿到多少分](docs/floor.png)

| 评测集 | 试 5 次 | 试 10 次 | 试 30 次 | 试 100 次 |
| ---: | ---: | ---: | ---: | ---: |
| 50 | +8.0 | +10.4 | +13.9 | +16.8 |
| 100 | +5.6 | +7.5 | +10.0 | +12.1 |
| **200** | +4.1 | **+5.3** | **+7.0** | +8.6 |
| 500 | +2.6 | +3.4 | +4.5 | +5.5 |
| 2000 | +1.3 | +1.7 | +2.2 | +2.7 |

评测集两百条、试三十个变体，是再普通不过的一个下午。那一行是 **+7.0**。

## 用法

### `check` —— 你的最高分里有多少是运气

```python
evalfloor.check(scores, n_examples)          # scores = 你试过的每一个变体
```

要传**全部**变体，不是只传赢的那个。候选数量是决定地板高低的一半因素。

### `confirm` —— 赢家在"没被用来挑选"的数据上还站得住吗

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

大多数工具在这里会打印"没有提升"。**那是错的**——不是这个改动失败了，
是你现在还判不出来。5 个不一致对的精确符号检验下限就是 0.0625，
无论多一边倒都到不了 0.05。

### `staged_floor` —— 给"先便宜后昂贵"的分级循环用

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

门槛设在 40%，候选只值 20%，所以**几乎没有人能晋级**。
报告的最高分实际是在 60 题那一级取的，地板是 **+0.110**，
而不是你花钱买的 200 题那级的 +0.059。**门槛越严，这个问题越重。**

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

优化一个类型化决策 schema —— 指令文本、选项描述、哪些字段进 state、阈值 ——
并把地板和留出检验作为输出的一部分打印出来。默认跑本地小模型：
**不需要 API key，不花钱。**

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

训练集那个数什么也没证明，全部证据在留出集那一块。
**只打印第一块的工具会把这次判成成功。**

RAG 相关性那个示例从 **F1 = 0.000** 起步 —— 因为大家默认发的 `0.5` 阈值，
高于这个模型给出的每一个分数。搜索最后找到 0.286。

## 适用边界

`selection_floor` 假设候选相互独立、每个只评测一次、指标是二项型。
候选相关或指标重尾都会让真实地板**更高**；分级循环用上面那个专门的函数。
误差方向始终一致：**过不了这个检验的提升，是确定性地过不了。**
过了的仍然需要 `confirm()`。

## 常见问题

**「我调了 30 次 prompt，准确率涨了 5 个点，这是真的吗？」**
把 30 个分数全传进 `check()`。评测集 200 条时地板是 +7.0，
所以 5 个点的涨幅**在搜索白拿的范围之内**。

**「这和留出集有什么区别？」**
它不是替代品，是**留出集之前的那一步**。地板只用你手上已有的数据，
就能告诉你这次搜索什么也没证明；留出集才能告诉你它证明了什么——那是 `confirm()` 做的事。

**「这不就是过拟合评测集吗？」**
同一类，但机制不同。过拟合通常指模型记住了样本；
这里是**选择偏差**——你什么都没拟合，只是在几个带噪测量里挑了最大的那个。

**「我的指标不是准确率。」**
`selection_floor` 假设指标是二项型。指标无界或重尾时真实地板**更高**，
所以过不了检验的提升仍然过不了。

**「我的循环是分级的，便宜档筛完再上贵档。」**
用 `staged_floor()`。地板通常是**便宜那一级**的，不是你花钱买的贵档——见上文。

## 说明

这里的统计不新 —— 胜者诅咒、选择性推断、best-of-k 的期望值，都是老东西。
新的只是：你能用一行拿到你自己那次运行的数。

```bash
pytest tests/ -q     # 23 个测试，每个都攻击上面的一句话
```

MIT 协议。
