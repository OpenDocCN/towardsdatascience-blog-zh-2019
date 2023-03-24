# 随机森林和额外树分类器的直观解释

> 原文：<https://towardsdatascience.com/an-intuitive-explanation-of-random-forest-and-extra-trees-classifiers-8507ac21d54b?source=collection_archive---------6----------------------->

## 利用群众的智慧来提高绩效

![](img/476135a8aa4ee4c0c8c38044979a7359.png)

Photo by [Aperture Vintage](https://unsplash.com/@aperturevintage?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

**目的:**本文的目的是让读者对[随机森林](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)和[多余树](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.ExtraTreesClassifier.html)量词有一个直观的了解。

**材料和方法:**我们将使用包含描述三种花的特征的 Iris 数据集。总共有 150 个实例，每个实例包含四个特征，并标有一种花。我们将调查并报告决策树、随机森林和额外树的准确性。

**硬件**:我们在配备 8 GB 1600 MHz DDR3、带 2 个 CPUs @ 2.9 Ghz 的 Inter(R)Core(TM) i7 和英特尔 HD Graphics 4000 卡的苹果工作站上训练和评估我们的模型。让我们踢一些尾巴。

**注意:**如果您是从零开始，我建议您按照这篇[文章](/i-want-to-be-the-very-best/installing-keras-tensorflow-using-anaconda-for-machine-learning-44ab28ff39cb)安装所有必要的库。最后，假设读者熟悉 Python、Pandas、Scikit-learn 和决策树。关于 Scikit-learn 决策树的详细解释可以在[这里](/scikit-learn-decision-trees-explained-803f3812290d)找到。

![](img/77d0ace87ccb9e9a1bf8579dbe563807.png)

# 一个树桩对 1000 个树桩

假设我们有一个弱学习器，一个准确率略好于随机决策的分类器，分类准确率为 51 %。这可能是一个[决策树桩](https://en.wikipedia.org/wiki/Decision_stump)，一个深度设置为 1 的决策树分类器。在第一个例子中，它似乎不应该麻烦这样一个弱分类器；然而，如果我们考虑将 1000 个略有不同的决策树桩(系综)放在一起，每个都有 51 %的准确性来做出我们的最终预测，会怎么样呢？直观地，我们可以看到，平均来说，510 个分类器会正确地分类一个测试用例，而 490 个会错误地分类。如果我们收集每个分类器的[硬票](https://stats.stackexchange.com/questions/320156/hard-voting-versus-soft-voting-in-ensemble-based-methods)，我们可以看到平均会有大约 20 个以上的正确预测；因此，我们的系综将趋向于具有高于 51 %的准确度。让我们在实践中看到这一点。

在这里，我们将构建一个决策树桩，并将其预测性能与 1000 个决策树桩进行比较。决策树的集合是使用 sci kit-learn[bagging classifier](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html)创建的。决策树桩和集成将在包含四个特征和三个类别的 Iris 数据集上训练。数据被随机分割以创建训练和测试集。

每个决策树桩将根据以下标准构建:

1.  训练集中可用的所有数据都用于构建每个树桩。
2.  为了形成根节点或任何节点，通过搜索所有可用特征来确定最佳分割。
3.  决策树桩的最大深度是 1。

首先，我们导入将用于本文的所有库。

**Script 1** — Importing the libraries.

然后，我们加载数据，分割数据，训练并比较单个树桩和整体。结果被打印到控制台。

Script 2— Stump vs Ensemble of 1000 Stumps

```
The accuracy of the stump is 55.0 %
The accuracy of the ensemble is 55.0 %
```

结果表明，1000 个决策树桩的集合获得了 55 %的准确率，表明它们并不比单个决策树桩好。发生了什么事？为什么我们没有得到更好的结果？我们基本上创造了 1000 个完全相同的决策树桩。这就像我们问一个人 1000 次他们最喜欢的食物是什么，毫不奇怪，得到了 1000 次相同的答案。

![](img/77d0ace87ccb9e9a1bf8579dbe563807.png)

# 随机森林分类器

在上一节中，我们了解到在我们的系综中有 1000 个相同的决策树桩就像有一个决策树桩。因此，我们将改变构建每个树桩的标准，以引入变化。

每个决策树桩将根据以下标准构建:

1.  将通过用替换对训练集进行随机采样来创建引导。引导的大小被设置为等于训练集的大小。
2.  为了形成根节点或任何节点，通过在大小为 sqrt(特征数量)的随机选择的特征的子集中进行搜索来确定最佳分割。在我们的例子中，每个决策树桩被允许检查四个特征中的两个。
3.  决策树桩的最大深度是 1。

我们刚刚描述的是创建随机森林的标准。但是，随机森林使用深度为 1 或更大的决策树。术语 *random* 来源于我们对训练集进行随机采样的事实，因为我们有一个树的集合，所以很自然地称之为森林——因此称为随机森林。为了构建树中的根节点或任何节点，选择特征的随机子集。对于这些选定的特征中的每一个，该算法搜索最佳切割点，以确定给定特征的分割。然后，从随机选择的子集产生最纯粹分裂的特征被用于创建根节点。树增长到深度 1，并且对树中的所有其他节点重复相同的过程，直到达到树的期望深度。最后，需要注意的是，每棵树都是使用不同的 bootstrap 单独构建的，这会在树之间引入差异。因此，每棵树都会犯不同的错误，当组合在一起时，可以建立一个强大的分类器。如果你被所有的术语搞糊涂了，[阅读这篇文章](/scikit-learn-decision-trees-explained-803f3812290d)，它用一段话解释了我刚刚描述的大部分内容。

Script 3 — Stump vs Random Forest. Notice how in line 5, we set splitter = “best” and in line 9 bootstrap = True. Your results may slightly vary since we did not fixed the seeds in the stump.

```
The accuracy of the stump is 55.0 %
The accuracy of the Random Forest is 95.0 %
```

什么？！因此，通过简单地引入变量，我们能够获得 95 %的准确率。换句话说，低精度的决策树桩被用来建造森林。通过在 bootstraps 上构建树桩，在树桩之间引入了变化-通过使用替换对训练集进行采样来创建，并允许树桩仅搜索随机选择的特征子集来分割根节点。单独地，每个树桩将获得低精度。然而，当在合奏中使用时，我们发现它们的准确性直线上升！朋友们，这就是通常所说的群体智慧。

为了让它深入人心，我们可以使用弱分类器作为集成的基础，以获得高性能的集成。

![](img/77d0ace87ccb9e9a1bf8579dbe563807.png)

# **额外的树分类器**

与随机森林分类器类似，我们有额外的树分类器，也称为极度随机树。为了在整体中引入更多的变化，我们将改变构建树的方式。

每个决策树桩将根据以下标准构建:

1.  训练集中可用的所有数据都用于构建每个树桩。
2.  为了形成根节点或任何节点，通过在大小为 sqrt(特征数量)的随机选择的特征的子集中进行搜索来确定最佳分割。每个选定特征的分割是随机选择的。
3.  决策树桩的最大深度是 1。

注意，在额外的树分类器中，特征和分裂是随机选择的；因此，“极度随机化的树”。由于额外树分类器中的每个特征都是随机选择分裂的，因此它的计算开销比随机森林要少。

Script 4— Stump vs Extra Trees. Notice how in line 5 splitter = “random” and the bootstrap is set to false in line 9\. Your results may slightly vary since we did not fixed the seeds in the stump.

```
The accuracy of the stump is 55.0 %
The accuracy of the Extra Trees is 95.0 %
```

额外树分类器的表现类似于随机森林。但是，我想提一下性能差异。即:决策树表现为高方差，随机森林表现为中方差，额外树表现为低方差。

![](img/77d0ace87ccb9e9a1bf8579dbe563807.png)

# 结束语

如果您已经阅读完本文，现在您应该了解集成方法的强大功能，并且知道随机森林和额外树分类器是如何构建的。我想提一下，你不应该使用 Bagging 分类器来构建你的随机森林或额外树木分类器。这两个分类器的更有效版本已经内置到 Scikit-learn 中。

集成方法不限于将弱学习者作为它们的基本估计器。例如，您可以为给定的任务确定最佳的三个分类器，并使用 Scikit-learn [投票分类器](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.VotingClassifier.html)来形成一个集成。您可以优化集合的权重，并使用它们进行预测。假设您已经对分类器进行了微调，并且所有分类器都具有相似的性能，那么投票分类器将比它们中的任何一个都有优势。为了缩小要使用的分类器的搜索范围，您可以阅读这篇讨论使用 Scikit-learn 进行[模型设计和选择的文章。](https://medium.com/i-want-to-be-the-very-best/model-design-and-selection-with-scikit-learn-18a29041d02a?source=post_stats_page---------------------------)

你可以在 [LinkedIn](https://www.linkedin.com/in/frank-ceballos/) 上找到我，或者访问我的[个人博客](https://www.frank-ceballos.com/)。

[](https://www.frank-ceballos.com/) [## 弗兰克·塞瓦洛斯

### 图表

www.frank-ceballos.com](https://www.frank-ceballos.com/)  [## Frank Ceballos -威斯康星医学院博士后| LinkedIn

### 我是威斯康星医学院的博士后研究员，在那里我分析高维复杂的临床数据…

www.linkedin.com](https://www.linkedin.com/in/frank-ceballos/)