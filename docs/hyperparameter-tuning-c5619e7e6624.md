# 超参数调谐

> 原文：<https://towardsdatascience.com/hyperparameter-tuning-c5619e7e6624?source=collection_archive---------1----------------------->

## 探索 Kaggle 的不要过度拟合 II 竞赛中超参数调整方法

![](img/0599fa6dcc93c8169ab93dafbbb191c8.png)

Photo by [rawpixel](https://unsplash.com/photos/BeDcRuoBzzw?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/collection?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

卡格尔的不要过度适应 II 竞赛提出了一个有趣的问题。我们有 20，000 行连续变量，其中只有 250 行属于训练集。

挑战在于不要吃太多。

对于如此小的数据集，甚至更小的训练集，这可能是一项艰巨的任务！

在本文中，我们将探索超参数优化作为一种防止过度拟合的方法。

完整的笔记本可以在这里找到[。](https://www.kaggle.com/tboyle10/hyperparameter-tuning)

## 超参数调谐

[维基百科](https://en.wikipedia.org/wiki/Hyperparameter_optimization)声明“超参数调优就是为一个学习算法选择一组最优的超参数”。那么什么是[超参数](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))？

> 超参数是在学习过程开始之前设置其值的参数。

超参数的一些例子包括逻辑回归中的惩罚和随机梯度下降中的损失。

在 [sklearn](https://scikit-learn.org/stable/modules/grid_search.html#grid-search) 中，超参数作为参数传递给模型类的构造函数。

## 调优策略

我们将探讨优化超参数的两种不同方法:

*   网格搜索
*   随机搜索

我们将从准备数据开始，用默认的超参数尝试几种不同的模型。我们将从中选择两种最佳的超参数调节方法。

然后，我们找到平均交叉验证分数和标准偏差:

```
Ridge
CV Mean:  0.6759762475523124
STD:  0.1170461756924883

Lasso
CV Mean:  0.5
STD:  0.0

ElasticNet
CV Mean:  0.5
STD:  0.0

LassoLars
CV Mean: 0.5
STD:  0.0

BayesianRidge
CV Mean:  0.688224616492365
STD:  0.13183095412112777

LogisticRegression
CV Mean:  0.7447916666666667
STD:  0.053735373404660246

SGDClassifier
CV Mean:  0.7333333333333333
STD:  0.03404902964480909
```

我们这里表现最好的模型是逻辑回归和随机梯度下降。让我们看看是否可以通过超参数优化来提高它们的性能。

## 网格搜索

网格搜索是执行超参数优化的传统方式。它通过彻底搜索超参数的指定子集来工作。

使用 sklearn 的`GridSearchCV`，我们首先定义要搜索的参数网格，然后运行网格搜索。

```
Fitting 3 folds for each of 128 candidates, totalling 384 fitsBest Score:  0.7899186582809224
Best Params:  {'C': 1, 'class_weight': {1: 0.6, 0: 0.4}, 'penalty': 'l1', 'solver': 'liblinear'}
```

我们将交叉验证分数从 0.744 提高到了 0.789！

网格搜索的好处是可以保证找到所提供参数的最佳组合。缺点是非常耗时且计算量大。

我们可以用随机搜索来解决这个问题。

## 随机搜索

随机搜索不同于网格搜索，主要在于它随机地而不是穷尽地搜索超参数的指定子集。主要的好处是减少了处理时间。

然而，减少处理时间是有代价的。我们不能保证找到超参数的最佳组合。

我们用 sklearn 的`RandomizedSearchCV`来试试随机搜索吧。与上面的网格搜索非常相似，我们在运行搜索之前定义了要搜索的超参数。

这里需要指定的一个重要的附加参数是`n_iter`。这指定了随机尝试的组合数量。

选择太低的数字会降低我们找到最佳组合的机会。选择太大的数字会增加我们的处理时间。

```
Fitting 3 folds for each of 1000 candidates, totalling 3000 fitsBest Score:  0.7972911250873514
Best Params:  {'penalty': 'elasticnet', 'loss': 'log', 'learning_rate': 'optimal', 'eta0': 100, 'class_weight': {1: 0.7, 0: 0.3}, 'alpha': 0.1}
```

在这里，我们将交叉验证分数从 0.733 提高到 0.780！

## 结论

在这里，我们探索了两种超参数化的方法，并看到了模型性能的改善。

虽然这是建模中的一个重要步骤，但绝不是提高性能的唯一方法。

在以后的文章中，我们将探索防止过度拟合的其他方法，包括特征选择和集合。