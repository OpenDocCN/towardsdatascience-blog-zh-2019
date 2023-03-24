# 作为强化学习问题的电子商务搜索重新排序

> 原文：<https://towardsdatascience.com/e-commerce-search-re-ranking-as-a-reinforcement-learning-problem-a9d1561edbd0?source=collection_archive---------12----------------------->

![](img/77d95cc5494e90c23b423c97f7d7af30.png)

Photo by [Clem Onojeghuo](https://unsplash.com/@clemono2?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/search?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

> 作为一个术语，搜索是超负荷的:它既描述了寻找某物的愿望，也描述了寻找的过程。

任何搜索工作流程都有三个主要部分:

**搜索预处理**:涉及 [*查询理解*](https://queryunderstanding.com/introduction-c98740502103) 的整个工作流程。从 [*语言识别*](https://queryunderstanding.com/language-identification-c1d2a072eda) → [*字符过滤*](https://queryunderstanding.com/character-filtering-76ede1cf1a97) → [*标记化*](https://queryunderstanding.com/tokenization-c8cdd6aef7ff) → [*拼写纠正*](https://queryunderstanding.com/spelling-correction-471f71b19880) *→* [*词干化和词条化*](https://queryunderstanding.com/stemming-and-lemmatization-6c086742fe45) 到 [*查询重写*](https://queryunderstanding.com/query-rewriting-an-overview-d7916eb94b83) (在查询理解之后，查询被传递给搜索引擎用于信息检索和排序。

**信息检索&排名:**它涉及检索在搜索引擎中被编入索引的信息(文档)，并根据传递给搜索引擎的查询对它们进行排名。

**搜索后处理:**根据外部信号对从引擎检索到的搜索结果进行重新排序，并显示搜索结果。

# 搜索后处理—根据点击流数据重新排序

搜索最容易获得的外部信号/反馈之一是用户点击数据。我们试图将用户点击建模为相关性的函数，并对搜索结果进行重新排序。

聚集和处理的点击流数据(点击归一化的时间和位置)将具有以下形式:

在我们深入研究重新排序的方法之前，让我们定义一下本文中使用的一些术语。

**确定性过程**:指系统未来状态不涉及随机性的过程。如果我们从*开始，相同的初始状态*确定性模型将总是给*相同的结果*。

**随机过程:**指随机确定的过程。随机优化产生并使用随机变量。

**马尔可夫决策过程:**指离散时间随机控制过程。它为决策过程提供了一个数学框架，其中结果部分是随机的，部分是由决策者控制的。它包含以下内容:

一个马尔可夫决策过程(MDP)是一个元组 M =(状态空间，动作空间，报酬，状态转移函数，折扣率)。MDP 的目标是找到一个从任意状态 s 开始最大化期望累积回报的策略。如果回报未知，这是一个*强化学习*问题。

> 强化学习侧重于在探索(未知领域)和利用(现有知识)之间找到平衡。

MDP 遵循**马尔可夫性质**，这意味着状态能够以这样一种方式简洁地总结过去的感觉，即所有相关信息都被保留，因此未来状态仅取决于当前状态。

**多臂土匪**:是 MDP 的特例，那里只有一个州。术语“多臂强盗”来自一个假设的实验，在该实验中，一个人必须在多个动作(即吃角子老虎机，即“单臂强盗”)中进行选择，每个动作都有未知的支付。目标是通过一系列选择来确定最佳或最有利可图的结果。在实验开始时，当赔率和支出未知时，赌徒必须确定要拉动的机器、其顺序和次数。

**MDP 和多臂强盗的区别**

为了区分 MDP 和多臂强盗，我们以餐馆为例:我们可能经常去我们最喜欢的餐馆，有时尝试一些新的，以找到一个潜在的更好的。如果说我们对每一家餐厅的偏好从未改变，土匪就是一个非常合适的提法。如果在吃了日本食物后，我们的偏好改变到中国食物，那么建模和探索状态转换是更合适的，因此 MDP。

## 回到重新排序的问题

学习排名方法是基于外部反馈的搜索结果重新排名的黄金标准。他们的工作原理是优化注释数据上的 [NDCG](https://en.wikipedia.org/wiki/Discounted_cumulative_gain) 或 [MRR](https://en.wikipedia.org/wiki/Mean_reciprocal_rank) 或[地图](https://en.wikipedia.org/wiki/Evaluation_measures_%28information_retrieval%29#Mean_average_precision)。这些方法有两个明显的缺点:

*   它需要大量带注释的数据来训练，并且在索引/目录中引入新的文档/产品会使事情变得更加复杂。
*   在获得大量反馈之前，它不会探索新的文档/产品，因此会引入偏见。

强化学习就是为了解决上述两个问题而出现的。

在搜索领域，如果我们有无限长时间的反馈数据，我们可以使用 frequentist 方法。尽管如此，因为那是不可能的，我们总是用当前的知识更新我们先前的信念(贝叶斯方法)。

![](img/f799e444b6bff465db1b27a219dc4124.png)

[https://xkcd.com/1132/](https://xkcd.com/1132/)

我们可以将搜索会话建模为 MDP，其中的流程如下:搜索查询 1、添加到购物车、搜索查询 2、添加到购物车……，结账。有一篇很棒的[论文](https://arxiv.org/pdf/1803.00710.pdf)描述了这种方法。我们采用一种更直接的方法，其中我们为单个查询(单个状态)的点击建模，即多臂 Bandit。

## 作为多臂强盗问题的搜索重新排序

我们将针对搜索查询 q 检索到的搜索结果视为不同的手臂，将点击/添加到购物车的任何结果的概率视为拉动手臂的机会。在这种情况下，奖励将是最大化搜索查询 q 的点击/添加到购物车的总次数。

有多种方法可以解决多臂土匪问题。我们分析了实现它的不同方式。

## 1)贪婪

我们选择数量为ε，ε的用户来显示随机结果(拉随机臂)，同时显示其余用户的最佳结果(拉得分最高的臂)。这有助于我们探索所有可能的组合。这里的主要挑战是选择正确的ε。如果太小，我们将无法探索所有的武器/产品。如果它太大，许多用户会发现自己的排名结果很差。

## **2)政策梯度**

在该方法中，我们选择一个策略，例如 softmax 策略，并且从 softmax 分布中随机选择 arm。我们维持每个手臂/文档/产品的平均奖励，梯度代理向最大奖励方向移动。如果出现以下情况，这将是一种理想的方法:

*   自观察开始以来，所有产品/武器均已推出&
*   环境中没有新条目(搜索结果页面)。

## 3) **置信上限(UCB)**

这种方法基于 ***面对不确定性的乐观主义原则*** 。我们取置信区间的上界，并据此选择产品/拉臂。如果能实时更新上面提到的表格，这是最好的方法之一。

## 4)汤姆逊取样

多臂 bandit 的最有效方法之一，它考虑了每个臂的概率分布(汇总表图 1)。这里的基本思想是，从每个 arms 元组(good_samples，total_samples)，可以生成置信区间，并且可以在 LCB 和 UCB 之间给定随机值。

> 当总样本数较低时，间隔较宽。尽管如此，随着总样本数的增加，置信区间变紧，最好的结果开始持续获得更高的分数。

![](img/95e83bafbaca934b797389700cb34994.png)

[Chris Stucchio’s article on bandit algorithms](https://www.chrisstucchio.com/blog/2013/bayesian_bandit.html) illustrates sharpening of the probability distribution with increase sample size

这在开发和探索之间取得了完美的平衡。在这里，如果我们使用威尔逊的置信区间，而不是贝叶斯区间，我们会得到更好的结果，因为它们更好地模拟了数据中的不对称。

Tensorflow 刚刚发布了他们深度上下文土匪[论文](https://arxiv.org/pdf/1802.09127.pdf)的[实现](https://github.com/tensorflow/models/tree/master/research/deep_contextual_bandits)。将尝试在其中实现的方法，并将其与随机抽样进行比较。请关注此空间了解更多信息。

# 参考资料:

1.  [https://medium . com/sajari/reinforcement-learning-assisted-search-ranking-a 594 CDC 36 c 29](https://medium.com/sajari/reinforcement-learning-assisted-search-ranking-a594cdc36c29)
2.  [https://medium . com/@ dtunkelang/search-the-all-the-story-599 F5 d 9 c 20 c](https://medium.com/@dtunkelang/search-the-whole-story-599f5d9c20c)
3.  [https://www . chriss tuccio . com/blog/2013/Bayesian _ bandit . html](https://www.chrisstucchio.com/blog/2013/bayesian_bandit.html)
4.  [https://medium . com/hacking-and-gonzo/how-Reddit-ranking-algorithms-work-ef 111 e 33 d0 d 9](https://medium.com/hacking-and-gonzo/how-reddit-ranking-algorithms-work-ef111e33d0d9)
5.  [https://www . Evan miller . org/how-not-to-sort-by-average-rating . html](https://www.evanmiller.org/how-not-to-sort-by-average-rating.html)
6.  [https://towards data science . com/how-not-to-sort-by-population-92745397 a7ae](/how-not-to-sort-by-popularity-92745397a7ae)
7.  [https://medium . com/@ sym 0920/the-多臂-土匪-问题-bba9ea35a1e4](https://medium.com/@SYM0920/the-multi-armed-bandits-problem-bba9ea35a1e4)
8.  [https://towards data science . com/13-非数学家多臂强盗问题解决方案-1b88b4c0b3fc](/13-solutions-to-multi-arm-bandit-problem-for-non-mathematicians-1b88b4c0b3fc)