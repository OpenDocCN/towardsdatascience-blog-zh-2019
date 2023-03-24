# 机器学习可解释性

> 原文：<https://towardsdatascience.com/machine-learning-explainability-d6a3d198fd95?source=collection_archive---------14----------------------->

## kaggle.com 微课综述

最近，我在 kaggle.com 上做了微课[机器学习可解释性。我强烈推荐这门课程，因为我学到了很多有用的方法来分析一个训练过的 ML 模型。为了对所涉及的主题有一个简要的概述，这篇博文将总结我的学习。](https://www.kaggle.com/learn/machine-learning-explainability)

以下段落将解释排列重要性、部分相关图和 SHAP 值等方法。我将使用著名的[泰坦尼克号](https://www.kaggle.com/c/titanic/data)数据集来说明这些方法。

![](img/72a16d863a8635e2194ce00615b0bacb.png)

Photo by [Maximilian Weisbecker](https://unsplash.com/@maximilianweisbecker?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

## 开始之前

我将要描述的方法可以用于任何模型，并且在模型适合数据集之后应用。以下回答的问题分别涉及上述在线课程中的一个部分。泰坦尼克号数据集可以用来训练一个分类模型，该模型预测泰坦尼克号上的乘客是幸存还是死亡。我在这个任务中使用了一个简单的决策树，没有优化它，也没有平衡数据，等等。它只是为了说明方法而被训练。分析代码可以在这个 [GitHub 库](https://github.com/wenig/mlx-titanic)上找到。

# 哪些变量对存活率影响最大？

对于这个问题，本课程建议采用排列重要性法。这种方法只是简单地取一列数据，并改变它的值，而保持其他列不变。利用改变的数据，该方法计算模型的性能。与原始数据相比，性能下降得越多，混洗列中的特征对于模型就越重要。为了找到所有特性的重要性，对所有列逐一进行这种操作。使用 *eli5* python 包，我为我们的数据集找出了以下数字。

![](img/a2e9edc84927953787419009c75b3481.png)

Importance of each feature for the survival on the Titanic

正如我们所见，泰坦尼克号幸存的最重要特征是性别，其次是年龄和票价。

这些数字只告诉我们哪些特征是重要的，而不是它们如何影响预测，例如，我们不知道是女性还是男性更好。为了找出特征是如何影响结果的，本课程建议使用部分相关图。

# 这些变量是如何影响存活率的？

找出变量如何影响结果的一个方法是在线课程建议的部分相关图。此方法获取数据集的一行，并重复更改一个要素的值。对不同的行执行多次，然后进行汇总，以找出该特性如何在很大范围内影响目标。python 包 *pdpbox* 可用于创建一个图表，显示使用不同值时的结果。对于我们的数据集，绘制“年龄”对目标的部分依赖关系如下所示。

![](img/e3db0ef24a273ecc22cce6ea2d166610.png)

Partial dependence plot for age on survival on Titanic

这个情节以一种非常好的方式表明，随着年龄的增长，在泰坦尼克号上幸存的概率会降低。尤其是 20 岁到 30 岁之间，在泰坦尼克号上可不是个好年龄。线周围的蓝色区域是线的置信度。这表明其他因素也对某人是否幸存起着很大的作用。幸运的是 *pdpbox* 库提供了二维图来显示两个特征对结果的相互作用。让我们看看年龄是如何与乘客乘坐的舱位相互作用的。

![](img/97324f927b552049c0e03538d268c043.png)

Partial dependence plot for age-class interaction on survival on Titanic

交互图显示，具有特定年龄的班级确实会对存活率产生影响，例如，20 至 30 岁之间的三等生的存活率低于 0.3，而拥有一等票的同龄人的存活率约为 0.5。

让我们看看三等舱的那个 30 岁的乘客，看看她的情况如何影响她的生存。为了分析特定的数据样本，本课程建议使用 SHAP 值。

# 变量如何影响特定乘客的生存？

SHAP 值用于显示单个用户的特征的效果。这里，该方法也采用一个特征并将该值与该特征的基线值进行比较，而不改变其他特征。所有特征都是如此。最后，该方法返回总和为 1 的所有 SHAP 值。有些价值观对结果有正面影响，有些价值观对结果有负面影响。我们特定的 30 岁乘客具有以下 SHAP 值。

![](img/69a0bf212b804fc35c597a96d723d499.png)

Showing parameters of a specific passenger influencing the output

影响最大的有她是女性的事实，票价价格等等。蓝色值显示她乘坐的是三等舱，对结果有负面影响。

## **概要剧情**

为了显示所有乘客的 SHAP 值汇总，我使用了如下所示的汇总图。

![](img/523eacbfa1589e41788cd4bbaf410cd8.png)

Summary plot of features importance towards the survival on the Titanic

该图按颜色显示特征值，红色为最大值，蓝色为最小值。因此，分类特征“性”只有两种颜色。水平位置是对样本结果的特定影响，垂直位置显示特征重要性。可以看出，就生存而言，女性大多具有积极影响，而男性大多具有消极影响。然而，例如,“费用”特征就不太明显。它的大部分值分布在 x 轴上。

## 依赖贡献图

为了说明在给定乘客性别的情况下，年龄对实际结果的影响，依赖贡献图就派上了用场。下面的图显示了这种贡献。

![](img/a1db7e9f252c76a19e4e747d14ffd04c.png)

Interaction of Age and Sex towards contribution to survivals on the Titanic

随着女性乘客年龄的增加，我们看到了一个小的下降趋势(红点)。不过，这些点主要出现在正值 0 以上。如前所述，男性乘客大多低于 0，尤其是 20 到 30 之间的值。

# 结论

出于多种原因，我真的很喜欢上这门课。然而，对于每个数据分析师和科学家来说，分析数据或模型本身也是非常重要的。特别是，当考虑深度神经网络或其他难以遵循决策的黑盒方法时，引入的方法肯定会增加价值。