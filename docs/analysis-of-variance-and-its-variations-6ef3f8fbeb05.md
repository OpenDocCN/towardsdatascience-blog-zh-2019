# 方差及其变异分析

> 原文：<https://towardsdatascience.com/analysis-of-variance-and-its-variations-6ef3f8fbeb05?source=collection_archive---------10----------------------->

在统计学中，当试图比较样本时，我们首先想到的是进行学生的 t 检验。它比较两个样本(或一个样本和总体)的平均值相对于平均值或合并标准差的标准误差。虽然 t 检验是一个稳健而有用的实验，但它仅限于一次比较两组。

为了一次比较多个组，我们可以看 ANOVA，或方差分析。与 t 检验不同，它比较每个样本内的方差与样本间的方差。罗纳德·费雪在 1918 年引入了术语**方差**及其正式分析，方差分析在 1925 年因费希尔的*研究人员统计方法*而广为人知。学生 t 检验遵循 t 分布，遵循正态分布的形状，但是它具有较厚的尾部，以说明样本中更多远离平均值的值。

![](img/be46600bf8a4965c0172df3180b84d9a.png)

Source: Wikipedia

然而，方差分析遵循 f 分布，这是一个具有长尾的右偏分布。对于仅有的两组，我们可以直接使用 f 分布来比较方差。

如果 U1 和 U2 是独立变量，分别由自由度为 d1 和 d2 的卡方分布分布，则随机变量 X(其中 X = (U1/d1)/(U2/d2))遵循参数为 X~F(d1，d2)的 F 分布，其 PDF 由下式给出:

![](img/8662514b22bbb0c4058484654702cbcc.png)

Source: Wikipedia; (B=Beta Function)

Anova 使用相同的分布，但是它计算 f 值的方式因所执行的 ANOVA 测试的类型而异。最简单的方差分析形式是单向方差分析测试，它允许我们通过评估一个自变量和一个因变量来比较多个组。一般而言，方差分析遵循三个主要假设:

因变量的分布应该是连续的和近似正态的

样本的独立性

方差的同质性

Anova 随后评估组间方差与组内方差的比值，以计算其 f 值。下表给出了单因素方差分析:

![](img/dc144ef16b864c2c1e826bb38784b6de.png)

Source: Analytics Buddhu

一旦我们计算出我们的 F 比率，我们就可以将其与我们的 F 临界进行比较，以确定我们是否可以拒绝我们的零假设。对于 Anova 检验，我们的替代假设是**组中至少有一个**组彼此不同，因此可以进行特别检验，如最小显著性平方或 Tukey 的 HSD(诚实显著性检验)。这些过滤器通过所有组合来确定哪些样本组彼此不同。只有当我们的 Anova 返回具有统计学意义的结果时，才执行这些测试。

**方差分析的变化**

正如我上面所说的，单向方差分析只能解释一个独立变量和因变量。单向方差分析有一些扩展，允许我们避开这些限制。第一个是双向方差分析。这个测试仍然要求我们只有一个因变量，但是我们能够包括多个自变量来分析组间的方差。由于我们有多个变量，两个计算发生，主要影响和相互作用的影响。主效应分别考虑每个独立变量，而交互效应同时考虑所有变量。

双向方差分析实际上是一种因子方差分析，这意味着测试将包含多个独立变量水平(也称为因子)。简单地说，双向方差分析是水平为 2 的因子方差分析。所以三向方差分析有三个自变量，四向方差分析有四个自变量，依此类推。最常见的水平是 2 和 3，因为在单向方差分析中很难解释以上水平。变异性在组内和组间进行比较，而阶乘方差分析将每个因素的水平与其他因素进行比较。

单向方差分析的另一个限制是通过执行方差分析来解决的，它允许我们比较一个以上因变量的组间方差。与返回单变量 f 值的 anova 相比，Manova 将返回多变量 f 值。多变量 f 值将仅指示检验是否显著，它不提供关于组间哪个特定变量不同的信息。为了了解一个或多个因变量中哪些是显著的，需要进行单向 Anova 测试以获取每个变量的单变量 f 值，然后进行特别测试。

以上所有测试的一个主要假设是我们的样本是相互独立的。这种假设意味着我们无法随着时间的推移对群体进行评估，或者测量同一受试者的多个结果。对于仅有的两组，这可以很容易地通过使用相关 t 检验来解决。重复测量 Anova 是从属 t 检验的扩展，它允许我们在多个类别或时间内评估相同的从属受试者。

重复测量方差分析的计算非常类似于单向方差分析。它不是在组之间和组内划分，而是在时间/条件之间划分，在条件/时间内划分为 2 个更小的类别，其中 SSw 等于 ss subjects+s error。由于我们在每组中使用相同的受试者，我们可以消除受试者的可变性，从而使组内误差更小。

![](img/cbfc3c22ddce8edf312b853313ee0d3a.png)

Source: statistics.laerd.com

这种划分最终增加了我们的 F 统计量，这意味着重复测量方差分析在发现统计差异方面具有更大的能力。然而，如果我们组内可变性的减少超过自由度的减少，这只会导致更强有力的测试。

我最近做了自己的项目，比较了几十年来每年的平均气温。我最初的错误是进行单向方差分析，忘记了我的不同“组”是相互依赖的。我无法摆脱这种感觉，即我进行了错误的测试，我想更深入地挖掘，因此我写了这篇文章。我回到我的数据，使用重复测量 Anova 在 Python 中重新运行测试，得到了以下结果。

![](img/d185a97260956bd1828b51945274fa2b.png)

给定我的结果和 0.05 的α，我发现 F 临界值为 2.77，因此我能够拒绝我的零假设，即不同年代的温度保持不变。理解测试的假设和需求总是很重要的，这样才能确保我们执行准确和正确的测试。作为一名数据科学家，我们的工作是确保我们总是执行正确的测试，而不仅仅是那些给我们想要的答案的测试。

来源:

[](https://en.wikipedia.org/wiki/Analysis_of_variance) [## 方差分析

### 方差分析(ANOVA)是统计模型及其相关估计程序的集合(例如…

en.wikipedia.org](https://en.wikipedia.org/wiki/Analysis_of_variance) [](https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/hypothesis-testing/anova/#targetText=There%20are%20two%20main%20types,double%2Dtesting%20that%20same%20group.) [## 方差分析检验:定义、类型、实例

### 统计学定义>方差分析内容:方差分析测试单因素方差分析双因素方差分析什么是方差分析？什么是阶乘…

www.statisticshowto.datasciencecentral.com](https://www.statisticshowto.datasciencecentral.com/probability-and-statistics/hypothesis-testing/anova/#targetText=There%20are%20two%20main%20types,double%2Dtesting%20that%20same%20group.) [](https://en.wikipedia.org/wiki/F-distribution#targetText=In%20probability%20theory%20and%20statistics,%2C%20e.g.%2C%20F%2Dtest.) [## f 分布

### 在概率论和统计学中，F 分布，也称为斯奈德尔的 F 分布或…

en.wikipedia.org](https://en.wikipedia.org/wiki/F-distribution#targetText=In%20probability%20theory%20and%20statistics,%2C%20e.g.%2C%20F%2Dtest.) [](https://statistics.laerd.com/statistical-guides/repeated-measures-anova-statistical-guide.php) [## 重复测量方差分析

### 重复测量方差分析相当于单向方差分析，但对于相关的，而不是独立的群体，是…

statistics.laerd.com](https://statistics.laerd.com/statistical-guides/repeated-measures-anova-statistical-guide.php)