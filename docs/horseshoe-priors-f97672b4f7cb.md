# 马蹄修道院

> 原文：<https://towardsdatascience.com/horseshoe-priors-f97672b4f7cb?source=collection_archive---------12----------------------->

## 比 L1 和 L2 有更多的正规化选择。

![](img/f6f0829f0e089bdd9d945fd5a8846d06.png)

Photo by [Jeff Smith](https://unsplash.com/photos/0iK-l7Z3sgU?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/horseshoe?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

正规化是一个迷人的话题，困扰了我很长时间。首先在机器学习课程中介绍，它总是提出一个问题，为什么它会工作。然后，我开始揭示正则化与潜在模型的统计属性之间的联系。

事实上，如果我们考虑线性回归模型，很容易表明，L2 正则化等价于将高斯噪声添加到输入中。事实上，如果我们考虑特征交互，后者是优选的(或者我们必须使用非平凡的 [Tikhonov 矩阵](https://en.wikipedia.org/wiki/Tikhonov_regularization)，例如，与单位矩阵不成比例)。从贝叶斯的角度来看，L2 正则化相当于在我们的模型中使用高斯先验，正如这里的[所解释的](https://en.wikipedia.org/wiki/Bayesian_linear_regression)。从贝叶斯线性回归模型开始，通过取最大后验概率，你将得到 L2 正则化项。一般来说，正态先验使模型变得简单，因为正态分布是正态分布的一个[共轭先验](https://en.wikipedia.org/wiki/Conjugate_prior)，这使得后验分布也是正态的！如果你选择一个不同的先验分布，那么你的后验分布会更复杂。

高斯先验和 L2 正则化的问题在于，它没有将参数设置为零，而只是减少了它们。当我们有许多功能时，这尤其成问题，大约 6 个月前我不得不处理这种情况，当时在所有功能处理之后，我有超过 100 万个功能。在这种情况下，我们想使用一种不同的正则化，L1 或套索正则化。它不是添加参数平方和的惩罚，而是基于参数绝对值的和进行惩罚。正则化模型将许多参数设置为零，实际上为我们完成了特征选择的任务。不幸的是，每个使用过它的人都可以证实，它大大降低了精度，因为它引入了偏差。

在贝叶斯术语中，L1 正则化等价于双指数先验:

![](img/06c5d546b433cfd8ad2ea6a6990c4e57.png)

在这里以及更远的地方，我会跟踪这个[案例研究](https://betanalpha.github.io/assets/case_studies/bayes_sparse_regression.html)。理想的先验分布将把概率质量设置为零以减少方差，并具有厚尾以减少偏差。L1 和 L2 都没有通过测试，也就是说，双指数分布和正态分布都有细尾，他们把概率质量设为 0。请注意，为了使概率质量为 0，概率密度函数必须发散。

卡瓦略等人[在此](http://proceedings.mlr.press/v5/carvalho09a/carvalho09a.pdf)、[在此](http://faculty.chicagobooth.edu/nicholas.polson/research/papers/Horse.pdf)以及其他一些论文提出了一个非常优雅的解决方案。该解决方案称为马蹄形先验，定义如下:

![](img/f15f1b0577a35ac5d60a55149267267f.png)

让我来帮你破译这个。如果你记得，L2 正则化相当于有一个正态先验，这是一个均值为 0 的正态分布，方差是一个超参数，我们必须调整。(方差是 L2 正则化常数 **λ** 的倒数)。本着真正的贝叶斯精神，我们希望定义一个关于 **λ** 的先验分布，然后将其边缘化。在这种方法中，建议使用半柯西分布作为先验分布。已经证明它把非零概率质量放在 0，也有一个胖尾。如果我们使用 s *hrinkage weight* k 的转换来重新参数化分布的形状，那么“马蹄”这个名称就来自于它:

![](img/be89b28375812ec877e77e4aac085ad2.png)

在这种情况下，分布支持变为[0，1],并且当与其他先前的分布候选进行比较时:

![](img/05e90005f3decf2f9c050668759d12c2.png)

我们可以看到马蹄先验满足我们的两个条件。

# 结论

在上述论文中，该方法在各种合成数据集上进行了测试，并且从那时起，它成为贝叶斯线性回归正则化方法的标准之一。从那时起，它已经过多次改进，并针对其他情况进行了调整。当然，由于[免费的午餐定理](https://en.wikipedia.org/wiki/No_free_lunch_theorem)，不可能排斥所有的超参数，所以很多讨论都致力于调整算法或克服它的一些缺点。

# 参考

卡瓦略，C. M .，波尔森，N. G .和斯科特，J. G. (2009 年)。通过马蹄处理稀疏性。《第 12 届人工智能和统计国际会议论文集》(D. van Dyk 和 M. Welling 编辑。).机器学习研究会议录 573–80。PMLR。

卡瓦略，C. M .，波尔森，N. G .和斯科特，J. G. (2010 年)。稀疏信号的马蹄形估计器。生物计量学 97 465–480。