# 机器学习的特征选择(1/2)

> 原文：<https://towardsdatascience.com/feature-selection-for-machine-learning-1-2-1597d9ccb54a?source=collection_archive---------18----------------------->

![](img/454a7d58e8f15426ec68eded92d9a1ff.png)

[https://images.app.goo.gl/JMcX4RLACqLSuhMw8](https://images.app.goo.gl/JMcX4RLACqLSuhMw8)

特征选择，也称为变量选择，是一个强大的想法，对您的机器学习工作流有重大影响。

你为什么需要它？
嗯，你喜欢把你的功能数量减少 10 倍吗？或者如果做 NLP，甚至 1000x。除了更小的特征空间，导致更快的训练和推理，还能在准确性上有可观察到的改进，或者你为你的模型使用的任何度量？如果这还不能引起你的注意，我不知道还有什么可以。

不相信我？几天前我工作时就遇到过这种事。

所以，这是一篇由两部分组成的博文，我将解释并展示如何在 Python 中进行自动特征选择，这样你就可以升级你的 ML 游戏。

将只介绍过滤器方法，因为它们比包装器方法更通用，计算量更小，而嵌入特征选择方法嵌入在模型中，不如过滤器方法灵活。

留在原处等候😉

> 第二部分可以访问[这里](https://medium.com/@alexburlacu1996/feature-selection-for-machine-learning-2-2-1a5a5b822581)

# 首先，最基本的

所以，你需要为你的模型找到最强大的，也就是重要的特性。我们将断言在一个重要的特征和目标变量之间有一个有意义的关系(不一定是线性的，但稍后会有更多的介绍)，这类似于`target ~ f(feature)`。最简单的关系是线性关系，识别这种关系的强大工具是**相关性**。相关性意味着两个变量之间的关联或依赖，这正是我们所需要的。有很多方法来计算它，但是考虑到这篇博文的目的是密集地提供实用的建议，这里有“最简单的”2: Spearman 和 Pearson 方法，在熊猫身上

```
>>> # shamelessly taken from here: [https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.corr.html)
>>> df = pd.DataFrame([(.2, .3), (.0, .6), (.6, .0), (.2, .1)],
...                   columns=['dogs', 'cats'])
>>> # df.corr(method="spearman")
>>> df.corr(method="pearson")
      dogs  cats
dogs   1.0   0.3
cats   0.3   1.0
```

因此，一旦计算出来，我们就可以得到指数，并使用它们来选择高度相关的特征，这些特征将进一步用于模型训练。在实践中，如果变量之间确实有显著的线性相关性，它应该在 0.7 以上。这些相关性测试的好处在于，在实践中，这些测试非常稳健，有时甚至可以识别非线性相关性，如果可能的话，可以用一条线进行局部近似，例如，二阶多项式相关性，或对数和平方根相关性，甚至指数相关性。相关系数会更小，可能在 0.5 到 0.7 之间。当得到这样的值时，打开 EDA 模式并绘制这些值，也许你可以发现一些依赖关系。我有时会。

```
pearson = pd.concat([features_df, target_df], axis=1).corr(method="pearson")
indices = pearson[abs(pearson["prediction"]) > 0.55].index
```

另一种方法是使用 chi2 测试。大概是这样的:

```
chi_sq = feature_selection.chi2(X, y)
corr_table = pd.DataFrame(zip(*chi_sq), columns = ("Correlation", "P-value"))

top_features = corr_table.sort_values("Correlation", ascending=False).head()["Correlation"]
```

最后一点，您需要记住:通常，数据集中的要素不仅与目标变量相关，而且它们之间也相关。**你不要这个！**在选择特征时，您(或算法)应尽可能选择最少数量的最重要的特征，这些特征之间尽可能正交/不相关。在第二篇博文中，将会介绍一些实现这一点的方法，敬请关注。

# 更大的枪:特征重要性和基于模型的选择

好的，很多这些方法都是基于统计学的，这很好，但是有时候，你只需要少一点形式主义，多一点科学知识。

scikit-learn 中的一些模型具有`coef_`或`feature_importances_`属性。一旦这些模型被训练，属性就被填充了对于特征选择非常有价值的信息。这里有两个例子，使用决策树和 L1 正则化。

## 基于决策树的方法

```
feature_importance_tree = tree.DecisionTreeClassifier()
feature_importance_tree.fit(X, y)

feature_importance_list = feature_importance_tree.feature_importances_.tolist()
indices = zip(*sorted(enumerate(feature_importance_list), key=lambda x: x[1], reverse=True)[:5])[0]

X_tree = X[:, indices]

scores = [model.fit(X_tree[train], y[train]).score(X_tree[test], y[test]) for train, test in kfcv]
```

既然我引起了你的注意，让我解释一下。决策树是非常好的工具，具有高度的可解释性，事实证明，它不仅仅在分类/回归问题上有用。在这个例子中，一个`DecisionTreeClassifier`被快速拟合到一个数据集上，然后`feature_importances_`被用来挑选最相关的特征，并训练一个更大、更复杂、更慢的模型。在实践中，如果您有大量数据，您可能会选择 next 方法的变体，但是对于较小的数据，这种方法非常好，能够捕获具有非线性依赖关系的特征。此外，`ExtraTreesClassifier`也可以很好地处理更大的数据，如果正则化(更浅的树，每片叶子更多的样本)甚至更好。永远要实验。

## 基于 L1 方法

对于那些还不知道的人来说，L1 正则化，由于它的性质，在模型中引入了稀疏…这正是我们所需要的，真的！

```
clf = linear_model.LassoCV()

sfm = feature_selection.SelectFromModel(clf, threshold=0.002)
sfm.fit(X, y)

X_l1 = sfm.transform(X)

scores = [model.fit(X_l1[train], y[train]).score(X_l1[test], y[test]) for train, test in kfcv]
```

与上面的树示例一样，训练了一个小模型，但与示例不同的是，`coef_`是驱动 sklearn 实现的特征选择的因素。因为使用了 L1，模型中的大多数系数将是 0 或接近于 0，所以任何更大的系数都可以算作一个重要特征。这适用于线性依赖关系。对于非线性模型，可以尝试 SVM，或者在线性模型之前使用 sklearn 的`RBFSampler`。

## 关于性能和大数据集的重要说明。

说你有 NLP 问题，用 TF-IDF。在一个合理的数据集上，你会有一个巨大的输出矩阵，类似于几千行(文档)和几百万列(n 元语法)。在这样的矩阵上运行任何模型都是耗时耗内存的，所以你最好使用一些速度快的东西。在这种情况下，我肯定会推荐 L1 的方法，但是用`SGDClassifier(penalty="l1")`代替`LassoCV`。这两种方法几乎是等效的，但是在大型数据集上，后者的运行速度几乎快一个数量级。所以要牢记在心。

此外，请记住，在收敛之前，您不需要训练您的要素选择模型，您不会将它们用于预测，并且最相关的要素将在模型中首先被选择。

# 收场白

这里大部分代码来自我在 GitHub 上的一个老项目，[这里](https://github.com/AlexandruBurlacu/MLExperiments/blob/master/machine-learning-and-a-bit-of-data-science/Breast_Cancer_feature_selection.ipynb)。这应该行得通，但如果行不通，不要害羞——HMU。

请注意，这篇博文和下一篇博文中的所有特征选择都适用于特征向量。这意味着，这里没有方法可以应用于视觉问题，例如，除非你想减少 CNN 最后一层的“功能”，这可能是一个好主意，idk。

此外，记住这一点，没有免费的午餐——想一想你准备做出哪些取舍，选择几个方法，进行实验，选择最适合你的问题的方法。

如果你正在读这篇文章，我想感谢你，并希望上面写的对你有很大的帮助，就像对我一样。请在评论区让我知道你的想法。你的反馈对我很有价值。