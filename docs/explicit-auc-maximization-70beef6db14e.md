# 显式 AUC 最大化

> 原文：<https://towardsdatascience.com/explicit-auc-maximization-70beef6db14e?source=collection_archive---------15----------------------->

## 如何在 ROC 下显式优化最大面积

![](img/a146a0e586da9a955f6bb5b5d7ded2d6.png)

Photo by [André Sanano](https://unsplash.com/@andresanano?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

我刚开始参加[“IEEE-CIS 欺诈检测”Kaggle 竞赛](https://www.kaggle.com/c/ieee-fraud-detection/overview/evaluation)，有件事引起了我的注意:

![](img/632ff60b4f9b574842a60f931a903281.png)

基于 AUC 评估结果的事实对于欺诈检测任务是有意义的，原因如下:

1.  数据集通常是不平衡的，这使得很难针对召回或其他简单指标进行优化，除非您使用数据的过采样或欠采样。
2.  在现实中，假阴性和假阳性的成本是不同的，实际任务应该包括这些信息并优化实际成本，而不是像召回或 F1 分数这样的数学量。
3.  高 AUC 增加了我们能够找到满足这些要求的最佳阈值的机会。

通常，AUC 仅在超参数调整期间优化，而在训练期间，它们使用交叉熵损失。为什么不首先优化 AUC，而不是交叉熵？

这篇论文提出了同样的问题:

论文的引言很好地概括了这个问题:

![](img/e02c13de5c742b3b0eeb21539683ec38.png)

如果我们使用 AUC 代替交叉熵作为优化函数会怎么样？可以使用 Wilcoxon-Mann-Whitney 统计来计算 AUC 曲线:

![](img/023bf6770751619631c0b406921c6e2b.png)

如果 x 和 y 是概率，则该度量惩罚任何实际肯定情况的概率小于否定情况的概率的情况。这不是传统的优化函数，因为它缺乏 MSE 或交叉熵成本函数的可加性。正如文件中指出的那样，

![](img/6a018121922093f0def5546023720e92.png)

上式中的 AUC 不是概率的平滑函数，因为它是使用阶跃函数构建的。我们可以尝试平滑它，以确保函数总是可微的，这样我们就可以使用传统的优化算法。如本文所述，解决这一问题的方法之一是 RankBoost 损失函数:

这里 x 和 y 分别是正类和负类的概率。例如，在逻辑回归模型中，x 和 y 都可以表示为 sigmoid 函数:

这里ξ是一个预测变量。

不得不承认，这些车型并没有什么新意。它们被称为*成对比较模型*，由瑟斯通于 20 世纪 20 年代首创。更多信息请看[这本书](http://web4.cs.ucl.ac.uk/staff/D.Barber/pmwiki/pmwiki.php?n=Brml.HomePage)。这种分析的基础是布拉德利-特里-卢斯模型，但最广为人知的变化是 Elo 算法。模型本身非常简单，是通过最大化似然函数得出的:

![](img/9ee7a858a253dac499bf1692900e7c49.png)

在我们的上下文中，如果 I 来自正类，j 来自负类，则 M_ij 为 1。在所有其他情况下，它是 0。这里的能力值α是对数，而不是概率。这使得模型比 RankBoost 更简单，并且还允许我们将其用作神经网络的最后一层(在这种情况下，值α将只是前一层的激活)。损失函数是:

其中 I 和 j 分别是正例和反例。

我们将使用一个简单的分类问题来演示这种方法。我们将使用 TensorFlow 来利用自动微分。

![](img/0f81e819c0eb10256281bc263d608195.png)

如你所见，这是一个使用合成数据的二元分类问题。我们将尝试最大化 AUC，而不是尝试对结果进行正确分类。

```
Epoch 0 Cost = -0.12107406 
Epoch 100 Cost = -0.24906836 
Epoch 200 Cost = -0.2497198 
Epoch 300 Cost = -0.24984348 
Epoch 400 Cost = -0.24989356
```

现在我们画出我们的预测:

![](img/de942471f45512e67c295153aff6febb.png)

## 讨论

下图基于阈值 0.945。如果我选择常规阈值 0.5，我们将在一个类中得到所有观察值。这是为什么呢？原因是 AUC 对偏差值不敏感。事实上，因为在这个模型中，我们减去成对激活，偏差值抵消。我们可以在成本函数中增加额外的项，使偏差更接近 0.5。或者选择做我们已经做过的事情，手动选择适合我们的阈值。我想提醒你，我们正在优化 AUC，所以如果所有的观察结果都被预测为属于一个类，只要 AUC 具有最优值，我们就不会介意。

## 结论

人们可能会想，除了使用 AUC 作为评估标准的数据科学竞赛之外，这种方法对任何真正的机器学习项目是否有用。但我认为，作为机器学习的第一步，它在不平衡数据集的情况下可能是有用的。第二步是学习偏差参数，并通过使用基于业务的优化指标对模型进行微调。

AUC 成本函数的一个缺点是，它在观察值之间使用成对比较，因此函数的复杂性随着观察值数量的平方而增长，随着观察值数量的增长，这可能成为性能瓶颈。此外，还必须想出一个好的分批策略，根据一些较小的样本(批次)正确估计 AUC。

所有代码都可以在[我的 github 库](https://github.com/mlarionov/machine_learning_POC/blob/master/auc/pairwise.ipynb)中找到。也可以在下面找到更多关于 RankBoost 的信息。

[](https://mitpress.mit.edu/sites/default/files/titles/content/boosting_foundations_algorithms/chapter011.html) [## 助推

### 我们接下来考虑如何学习对一组对象进行排序，这是一个在各种领域中自然出现的问题。对于…

mitpress.mit.edu](https://mitpress.mit.edu/sites/default/files/titles/content/boosting_foundations_algorithms/chapter011.html)