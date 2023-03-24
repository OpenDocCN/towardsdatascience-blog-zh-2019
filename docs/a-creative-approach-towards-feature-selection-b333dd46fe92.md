# 一种创造性的特征选择方法

> 原文：<https://towardsdatascience.com/a-creative-approach-towards-feature-selection-b333dd46fe92?source=collection_archive---------31----------------------->

## 品格如树，名誉如影。影子就是我们对它的看法。这棵树是真的。——亚伯拉罕·林肯

![](img/3af1cf99c31c65fae4071700367d9bf9.png)

# **简介**

谈到特征工程，特征选择是最重要的事情之一。我们需要减少特征的数量，以便我们可以更好地解释模型，使训练模型的计算压力更小，消除冗余效应，并使模型更好地一般化。在某些情况下，特征选择变得极其重要，否则输入维度空间太大，使得模型难以训练。

**执行功能选择的各种方式**

有多种方法来执行特征选择。我将使用我自己在实践中使用的方法。

1.  卡方检验(分类变量对连续变量)
2.  相关矩阵(连续变量与连续变量)
3.  领域知识
4.  线性正则模型的系数。
5.  树方法的特征重要性(最喜欢这个)

# **我的创作方法**

在这里，我提出了我的创造性观点，如何使用以上我最喜欢的方法来进行特征选择，而不是仅仅局限于其中一种。核心思想是结合相关矩阵(每个变量与目标变量比较)、线性正则化模型的系数和各种树方法的特征重要性，使用随机加权平均将它们结合起来。使用随机加权是为了避免对某一种单一方法的固有偏见。所以，最后，我们有了一个最终的特征重要性矩阵，它综合了我最喜欢的特征选择方法。与研究各种方法相比，在我们面前有一个单一的产出数字总是一件好事。

# 概念证明

为了展示概念证明，我将使用波士顿住房数据集。这是一个回归案例，所有输入变量都是连续的。

1.  首先计算整个数据集的相关矩阵。这里只做这个，因为所有变量都是连续的。在有一些分类变量的情况下，我们需要使用卡方统计而不是相关矩阵。

![](img/fbbdc402e679e75546395c017a408a18.png)

Natural values of linear correlation with respect to target variable

![](img/2f37a3f609b36f7fa86b99e0f82dbb5e.png)

Absolute Linear correlation with respect to the target variable ‘MEDV’

2.选择套索和山脊计算变量系数。

![](img/212702f0612cc8123233d37d0c54ec17.png)

Lasso natural coefficients

![](img/fdcee803a53be0056b02ea0ee8099dd5.png)

Ridge natural coefficients

![](img/5f37fcb49e80dad1f0bbed7113c46cf7.png)

Normalised and absolute values of lasso coefficients

![](img/8a3a67f31d77b18bfcaf5fbb17c2ef99.png)

Normalised and absolute values of ridge coefficients

3.选择 GradientBoostingRegressor、RandomForestRegressor、DecisionTreeRegressor 和 ExtraTreeRegressor 来计算特征重要性。在实践中，在此步骤之前，分类变量的所有转换都应该完成(例如，使用一个热编码或标签编码)。

![](img/133632c9291ad45fb000b0e30633496d.png)

Decision Tree Feature Importance

![](img/35f6712f50aaea04030d81a2ad47e820.png)

Random Forest Feature Importance

![](img/8386725b6cb5ceff39fa3b393b6d3856.png)

Extra Trees Feature Importance

![](img/85aac95cc15e77b2deb855d230f58210.png)

Gradient Boosted Trees Feature Importance

![](img/19da5144cfde8c4e0fa633b89a082264.png)

4.使用 numpy 进行随机采样，为每种方法赋予权重，然后取加权平均值。

这里，我们使用了随机权重生成方案，其中，对于任何形式的特征选择机制，我们都不使用 0，并且为了简单起见，值或权重以 0.1 为步长。

5.然后，我们返回合并的结果，可以使用条形图显示。

![](img/4a89332813015db9bdc3e9c07eb9c6e9.png)

# 结论

还有各种其他的特征选择技术，但我提出的这个可以提供一个非常不同的视角，使用一种我最喜欢的特征选择技术的集合来比较特征与目标变量的相关性。你可以摆弄各种参数，让这个想法更有创意。还可以使用具有更具体设置的模型，例如使用树模型对象，根据具体的使用情况，用 n_estimators 或 max_depth 的一些具体值初始化，但这里只是为了展示我使用默认模型对象的想法。

这一思考过程不仅有利于得出最终的一次性特征重要性输出，而且我们可以直观地看到数字和图表，以比较不同方法如何独立执行，然后就输入变量的选择做出明智的决定。

就像爱因斯坦的名言 ***一样，我相信直觉和灵感。我有时觉得我是对的。我不知道我是。***