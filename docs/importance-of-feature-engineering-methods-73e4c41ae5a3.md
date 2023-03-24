# 特征工程方法的重要性

> 原文：<https://towardsdatascience.com/importance-of-feature-engineering-methods-73e4c41ae5a3?source=collection_archive---------11----------------------->

## 特征工程方法及其对不同机器学习算法的影响分析

![](img/5ba48cfe72176b1e30b1bfc0ef357174.png)

Photo by [Fabrizio Verrecchia](https://unsplash.com/@fabrizioverrecchia?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

机器学习算法接受一些输入，然后生成一个输出。例如，这可能是一只股票第二天的价格，使用的是前几天的价格。从给定的输入数据，它建立一个数学模型。该模型以自动化的方式发现数据中的模式。这些模式提供了洞察力，这些洞察力被用于做出决策或预测。为了提高这种算法的性能，特征工程是有益的。特征工程是一个数据准备过程。一种是修改数据，使机器学习算法识别更多的模式。这是通过将现有特征组合并转换成新特征来实现的。

应用机器学习从根本上来说是特征工程。但是这也是困难和费时的。它需要大量的经验和领域知识。评估哪些类型的工程特征性能最佳可以节省大量时间。如果一个算法可以自己学习一个特性，就没有太大的必要提供。

# 特征工程方法的前期分析

杰夫·希顿在[1]中研究了公共特征工程方法。他评估了机器学习算法综合这些特征的能力。起初，他采样均匀随机值作为输入特征 *x* 。接下来，他用特定的方法设计了一个功能。他用这个工程特征作为标签 *y* 。然后使用预测性能来测量该能力。研究的特征工程方法有:

![](img/6ffceae381bd8f00a4c24e283cc97d71.png)

Counts

![](img/c495c2ca2fa8b9d3bc5711ac70e43567.png)

Differences

![](img/9b9e5895e61c7a171be39fb158ab5dd2.png)

Logarithms

![](img/4cec1174e1f1e2f4aac402c2f28fc598.png)

Polynomials

![](img/f13de66ad7f0a70171681b3c014d731b.png)

Powers (square)

![](img/c3c7a6adb23e54037d0367763c0271c4.png)

Ratios

![](img/6da19f8fad6c1bff5172838747aaf42d.png)

Ratio differences

![](img/68edda1e0b9cc627f864252091d82fd9.png)

Ratio polynomials

![](img/63f3eacdfc2778adf8935f49e3f0d4a3.png)

Root distance (quadratic)

![](img/8d4b1a8f77ca4977eb08bc13ab7cc7ea.png)

Square root

他一次只测试一种特征工程方法。他对输入要素进行采样的范围各不相同。在每种情况下，他评估了四种不同模型的均方根误差(RMSE)。其中包括:

*   神经网络
*   高斯核支持向量机(SVM)
*   随机森林
*   梯度推进决策树(BDT)

他在自己的 GitHub 页面上发布了 [RMSE 结果](https://github.com/jeffheaton/papers/blob/master/2016/ieee-feature-eng/results.csv)。希顿得出以下结论:

*   所有的算法都不能综合比值差特征。
*   基于决策树的算法不能综合计数特征。
*   神经网络和 SVM 不能合成比率特征。
*   SVM 未能综合二次特征。

因此，结果表明，算法可以综合一些特征，但不是全部。尽管根据算法的类型有一些不同。

然而，均方根误差对标签的绝对值很敏感。并且实验的标签值在不同的数量级上变化。此外，分布也不相同。在某些情况下，当从稍微不同的范围对输入要素进行采样时，可能会得出不同的结论。因此，不同实验中的 RMSE 值不具有可比性。只是同一实验中不同模型的 RMSE 值。为了说明这一点，我用改进的评估方法重复了这个实验。这揭示了一些有趣的差异。

# 改进的分析设置和结果

我改变了评估方法，使标签值均匀随机抽样。我选择的范围在 1 到 10 之间。所有实验都使用相同的取样标签。之后，我在标注上使用逆变换创建了输入要素。例如，我没有使用输入要素的平方来创建标注，而是使用标注的平方根来创建输入要素。

![](img/c36e413cb993e7259ef7707d315c61fc.png)

Inversion

在反演有多个解的情况下，只使用一个解。在多输入的实验中，我随机采样了除一个输入特征之外的所有输入特征。我选择了采样范围，以便所有输入要素都在同一范围内。此外，我使用所有模型的标准化来缩放数据。最后，我在每个实验中尝试了神经网络和 BDT 的不同学习速率。取学习率，即各个模型表现最好的学习率。我用了希顿分析中剩下的装置。结果如图 1 所示。

![](img/00e7ed30ed7abcc6aa8c69b1699bb9eb.png)

Figure 1: Capacity of Machine Learning algorithms to synthesize common Feature Engineering methods. Label values ranged uniformly between 1 and 10\. Simply predicting the mean of the labels, would result in an RMSE value of 2.6.

与希顿的分析一致，所有算法都无法合成比率差异特征。此外，计数和比率功能也有相似之处。基于决策树的算法比其他具有计数特征的算法性能更差。决策树没有总结不同特征的固有方式。他们需要为每种可能的组合创建一片叶子，以便很好地合成计数特征。此外，神经网络和 SVM 在比率特征方面表现较差。最后，实验结果证实了决策树可以综合单调变换特征。正如所料。

与希顿的分析相反，我的分析显示了二次特征的巨大差异。所有模型在综合这些特征时都有问题。在希顿的二次特征实验中，许多标签的值为零。其余的价值很小。这就解释了区别。此外，线性模型在综合对数特征方面存在更多问题。这个实验中的标记比其他实验中的标记小一个数量级。这导致了差异。

# 结论

该分析评估了特征工程方法在建模过程中的重要性。这是通过测量机器学习算法合成特征的能力来完成的。所有被测试的模型都可以学习简单的转换，比如只包含一个特征的转换。因此，专注于基于多个特征的工程特征可以更有效地利用时间。尤其是二次特征和包含除法的特征非常有用。

此外，这解释了为什么集合不同的模型通常会导致显著的性能提高。基于相似算法的集成模型则不会。例如，在具有类似计数和对数模式的数据集中，组合随机森林和增强决策树仅受益于对数特征。而组合神经网络和随机森林从两种特征类型中获益。

[1] J .希顿，[预测建模的特征工程实证分析](https://arxiv.org/pdf/1701.07852.pdf) (2016)，arXiv