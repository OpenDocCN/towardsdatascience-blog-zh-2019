# 学习聚合函数

> 原文：<https://towardsdatascience.com/learning-aggregate-functions-5b4102383433?source=collection_archive---------14----------------------->

## 利用关系数据的机器学习

![](img/53b740419911847cab37d94c7c3b1c08.png)

[https://pixabay.com](https://pixabay.com)

本文灵感来源于 Kaggle 大赛[https://www . ka ggle . com/c/elo-merchant-category-recommendation](https://www.kaggle.com/c/elo-merchant-category-recommendation)。虽然我没有参加比赛，但我用这些数据探索了另一个在使用真实数据时经常出现的问题。所有的机器学习算法都可以很好地处理表格数据，但实际上很多数据都是关系型的。在这个数据集中，我们试图根据交易历史得出关于用户(由卡 id 标识)的结论。

![](img/dea45332d149d889e707521d57aafd71.png)

The first two rows of the historical transactions

事务表列出了事务，因此包含每个用户的多行。让我们看看对数标度的直方图。

![](img/72a841273e2f3a631761ec6f6ef72e79.png)

Histogram in the logarithmic scale

正如我们从这个图中看到的，大多数用户有不止一个交易。如果我们想对用户进行分类，我们会遇到每个用户有多条记录的问题，而机器学习算法需要表格数据。

![](img/7987db265741bcde3314884081e43fd7.png)

Illustration of the problem we have with machine learning with relational data

这类似于 SQL `GROUP BY`语句，它帮助我们根据键(在我们的例子中是卡 id)对数据进行分组。SQL 还定义了许多聚合函数，这些函数的完整列表由定义并可以找到，例如，[此处为](https://www.oreilly.com/library/view/sql-in-a/9780596155322/ch04s02.html)。在 Pandas 中，聚合函数由 [agg](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.agg.html) 函数支持。

许多机器学习实践者为所有可能的聚合函数定义了额外的特征，如`count`、`sum`、`mean`等。问题是我们通常不知道哪些聚合函数是重要的。例如，我们可以添加总量作为一个特性，但是如果标签受到特定类别的总量的影响呢？我们最终会添加许多额外的特性，其中大部分是无用的，并且很可能会错过真正重要的东西。

如果我们可以用算法来学习它们，而不是手工制作集合特征，会怎么样？虽然过去研究过这种方法(例如，参见[级联相关方法](https://pdfs.semanticscholar.org/7919/a8a4cb2e628b7f8e316265d36f84c7714c17.pdf))，但是当事务之间没有相关性时，我们可以使用一个简单的神经网络，该网络将使用一些标准张量流函数来学习聚合函数。

在我们的例子中，神经网络的结构非常简单。我们有一个完全连接的层，后面是`segment_sum`函数，它实际上执行分组操作，并对第一层的输出求和。第二个完全连接的层连接到输出，在我们的例子中，这是一个聚合函数，我们正在尝试学习。为简单起见，在这个概念验证中，我们使用线性单位，但对于更复杂的情况，我们必须引入非线性。

![](img/80d83699475f010fd1c31dcf6369e5be.png)

Sample neural network architecture

第一层中单元的数量给了我们它可以同时学习的集合函数的数量。

让我们考虑两个例子:学习计数和求和。

如果我们将第一层中的所有权重设置为零，并将所有偏差设置为 1，那么(假设线性单位)，输出的总和将为我们提供每个用户的事务计数。

然而，如果我们将偏差和所有权重设置为零，但是将`purchase_amount`的权重设置为 1，我们将获得每个用户的总购买量。让我们在 TensorFlow 中演示我们的想法。

函数 [segment_sum](https://www.tensorflow.org/api_docs/python/tf/math/segment_sum) 的工作方式如下:

![](img/f8c8f480190123cf08e884e1d08a7678.png)

The image taken from TensorFlow documentation

它接受具有段 id 的单独张量，并且数据必须用相同的段 id 标记。它按段 id 对数据进行分组，并在零的维度上进行求和缩减。

```
Cost after epoch 0: 187.700562
Cost after epoch 100: 0.741461
Cost after epoch 200: 0.234625
Cost after epoch 300: 0.346947
Cost after epoch 400: 0.082935
Cost after epoch 500: 0.197804
Cost after epoch 600: 0.059093
Cost after epoch 700: 0.057192
Cost after epoch 800: 0.036180
Cost after epoch 900: 0.037890
Cost after epoch 1000: 0.048509
Cost after epoch 1100: 0.034636
Cost after epoch 1200: 0.023873
Cost after epoch 1300: 0.052844
Cost after epoch 1400: 0.024490
Cost after epoch 1500: 0.021363
Cost after epoch 1600: 0.018440
Cost after epoch 1700: 0.016469
Cost after epoch 1800: 0.018164
Cost after epoch 1900: 0.016391
Cost after epoch 2000: 0.011880
```

![](img/df034bc4c3187ca2ea0f0a9fe80d897e.png)

MSE loss vs. iterations

在这里，我们绘制了每次迭代后的成本函数。我们看到算法学习计数功能相当快。通过调整 Adam 优化器的超参数，我们可以获得更高的精度。

```
Cost after epoch 0: 8.718903
Cost after epoch 100: 0.052751
Cost after epoch 200: 0.097307
Cost after epoch 300: 0.206612
Cost after epoch 400: 0.060864
Cost after epoch 500: 0.209325
Cost after epoch 600: 0.458591
Cost after epoch 700: 0.807105
Cost after epoch 800: 0.133156
Cost after epoch 900: 0.026491
Cost after epoch 1000: 3.841630
Cost after epoch 1100: 0.423557
Cost after epoch 1200: 0.209481
Cost after epoch 1300: 0.054792
Cost after epoch 1400: 0.031808
Cost after epoch 1500: 0.053614
Cost after epoch 1600: 0.024091
Cost after epoch 1700: 0.111102
Cost after epoch 1800: 0.026337
Cost after epoch 1900: 0.024871
Cost after epoch 2000: 0.155583
```

![](img/5bf84c0797f424b3d5dbdfd33745887a.png)

MSE vs. iterations

我们看到成本也下降了，但随着我们向算法提供新数据，成本会出现峰值，这可以用高梯度来解释。也许我们可以调整超参数来改善学习过程的收敛性。

# 结论和下一步措施

我们演示了一个简单的神经网络，它可以学习基本的聚合函数。虽然我们的演示使用了线性单位，但实际上我们必须对第 1 层使用非线性单位，以便能够学习更复杂的聚合函数。例如，如果我们想知道`category2 = 5`的总量，那么线性单位就不起作用。但是如果我们使用例如 sigmoid 函数，那么我们可以将偏差设置为-100，然后将`category2 = 5`的权重设置为+100，并将`purchase_amount`的权重设置为小的正值 *ω* ω。在第二层中，我们可以将偏差设置为零，权重设置为 1 *ω* 1ω。

这种架构不学习功能`mean`。但是它学习它两个组件:`sum`和`count`。如果我们的决策边界取决于平均销售额，这与它取决于交易数量和总金额是一样的。

这种架构也不会学习更复杂的函数，如方差和标准差。这在金融领域可能很重要，因为你可能想根据市场波动来做决定。可能需要聚合前附加层来实现这一点。

最后，在这个例子中，学习速度很慢，因为我们必须以汇总的形式显示数据。可以通过预聚合数据，然后重新采样来提高速度。

本文中使用的所有代码都可以在[我的 github repo](https://github.com/mlarionov/machine_learning_POC/blob/master/aggregate_functions/basic_aggregate_function_learning.ipynb) 中找到。