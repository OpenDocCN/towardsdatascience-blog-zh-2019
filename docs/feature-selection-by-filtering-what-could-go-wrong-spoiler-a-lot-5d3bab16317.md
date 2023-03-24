# 通过过滤选择特征:什么会出错？(剧透:很多)

> 原文：<https://towardsdatascience.com/feature-selection-by-filtering-what-could-go-wrong-spoiler-a-lot-5d3bab16317?source=collection_archive---------23----------------------->

## 如何通过单变量过滤正确(和不正确)执行特征选择的说明

从计算机科学到心理学，处理具有比观察数量多得多的特征的数据集现在是许多领域的共同之处。这通常被称为“p > n”问题(其中 p =特征数，n =观察数)，“维数灾难”，或者我个人最喜欢的“短胖数据”。

传统方法，如线性回归，在这种情况下会失效。我们可以在一个玩具示例中观察到这一点，在这个示例中，我们删减了“mtcars”数据集，以生成#行< # of columns. When trying to calculate beta coefficients, we run into singularity problems.

## Feature Selection

A common strategy to deal with this issue is to perform feature selection where the aim is to find a meaningful subset of features to use in model construction. There are many different ways to carry out feature selection (for a summary check out this [概览](https://bookdown.org/max/FES/selection.html)。这里，我们将关注最简单的变体，即单变量过滤器的特征选择。与其他特征选择策略一样，单变量过滤器的目的是找到与结果相关的特征子集。这可以通过将特征与结果相关联并仅选择那些满足特定阈值的特征(例如， *r* >的绝对值)以非常直接的方式来实现。

## 为什么专注于单变量过滤器进行特征选择？

正如在统计学习的[元素](https://web.stanford.edu/~hastie/Papers/ESLII.pdf) (ESL)和 2002 年的[论文](https://www.pnas.org/content/99/10/6562.full)、**中指出的，许多**发表的作品都错误地执行了特征选择。毫无疑问，这将继续是一个问题，因为随着研究人员努力处理更大的数据集，p > n 数据在生物、社会科学和健康领域继续变得更加常见。这并不奇怪为什么过滤通常是不正确的，乍看起来，简单地删除数据集中与结果无关的特征似乎并没有什么错。

为了说明通过过滤进行的特征选择是如何错误地和正确地完成的，我们将首先生成一个简短、丰富的数据集，其中我们的预测器在很大程度上与我们的因变量“y”无关。关于模型拟合，我们将基本上按照第 245–247 页的 [ESL](https://web.stanford.edu/~hastie/Papers/ESLII.pdf) 进行，并通过一个例子进行编码，看看这些数字是如何产生的。首先，让我们生成一些基本正交的数据。

## 单变量过滤错误的方式

现在我们有了数据，我将通过筛选交叉验证的之外的**数据来说明如何错误地过滤数据。也就是说，通过获取**整个**数据集，保持与结果相关的特征等于或高于 *r* = .1，并且**随后**建立模型。对于 100 次迭代，我们将把这个数据集分成训练集和测试集，并检查观察值和预测值之间的相关性分布。**

![](img/ba94f8130726689e52ef6f88fbc1ac8f.png)

Distribution of outcomes for the incorrectly filtered model. The thick red line indicates the mean.

总的来说，这些模型看起来很棒，观察值和预测值之间的平均相关性大约为 0.5。唯一的问题是，模拟数据在很大程度上是独立的，平均相关性应该在 0 左右！我们通过选择基于所有数据的变量引入了偏差。在变量选择后实施验证方案**并不能准确描述模型在真正独立的测试数据集上的应用。现在让我们观察如何通过正确过滤来执行选择。**

## 正确的单变量过滤方法

当正确执行单变量过滤时，特征选择发生在外部交叉验证循环中，而模型调整发生在内部循环中。如果我前面提到的样本有偏差是真的，我们应该会在内部验证循环中看到类似的高相关性，但在外部循环和后续遗漏的样本中完全缺乏泛化能力。

我们将在 SBF 的 caret 中使用一个函数来实现这一点。在“sbfControl”中对控制滤波的外部验证回路进行说明，而在“trainControl”中对参数调整进行正常说明。对于连续的结果和特征，caret 利用广义加法模型将结果与每个特征联系起来——这是一个很好的接触，可以发现非线性关系。这些模型中的每一个的 p 值被用作过滤标准。默认值设置为 p = .05，这意味着只有当特征与该级别的结果有显著关系时，才会保留这些特征。

![](img/3d2a88d582374425539f2d8567dd1cad.png)

A closer at whats happening during the nested validation for filtering. Thick lines indicate the distribution mean.

正如预期的那样，我们在内部训练循环中看到了很强的关系，但在外部训练循环中以及当我们最终将我们的模型拟合到遗漏的数据时，我们观察到了本质上正交的预测(平均值 *r* 正好在 0 附近)。如果你查看第 246 页的 [ESL](https://web.stanford.edu/~hastie/Papers/ESLII.pdf) ，你会注意到类似的结果模式。

## 与 Glmnet 的比较

最后，作为质量保证检查，我们将把我们的结果与正则化模型(一种添加了参数以防止过度拟合的模型)进行比较，正则化模型应该给出与正确的单变量过滤方法几乎相同的结果。

![](img/40ef28542c46d1d49d489d8af2238fb4.png)

Distribution of outcomes for the glmnet models. The thick green line indicates the mean.

不出所料，平均 *r* right 在 0 左右。

在结束之前，我们将最后快速浏览一次，并排查看正确的过滤、不正确的过滤和 glmnet 模型，注意不正确的过滤会对预测产生多大的偏差。

![](img/813b06ea10fb6aaf8053361f5d7f39c3.png)

Resulting distributions of the three approaches

**包装完毕**

总之，如果您决定使用单变量过滤进行特征筛选，请确保在交叉验证方案中执行特征选择**，而不是在整个数据集上执行**。**这样做将导致特征已经“见证”了遗漏的数据，因此不能有效地模拟将模型应用到真实的测试集。需要注意的一点是，如果没有考虑结果，这并不一定适用于筛选特征*。例如，删除以下要素:彼此高度相关、不包含方差或者是其他要素集的线性组合。最后，根据您的需要，还值得记住的是，有许多防止过度拟合和/或本质上执行特征选择的正则化方法可能是比任何用于特征选择的包装器或过滤器方法更好的选择(并且更快)。***