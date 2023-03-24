# 机器学习完全入门指南:简单线性回归四行代码！

> 原文：<https://towardsdatascience.com/simple-linear-regression-in-four-lines-of-code-d690fe4dba84?source=collection_archive---------1----------------------->

## 一个清晰而全面的蓝图，绝对适合任何想要构建简单机器学习模型的人

![](img/5a22e41eb1d31d07438dcf58483a60ae.png)

甚至你可以建立一个机器学习模型。

说真的！

仅有好的数据并不能说明全部情况。你是想根据一个人多年的经验来计算他的工资应该是多少吗？你需要检查相对于你的年销售额你在广告上花了多少钱吗？线性回归可能正是你所需要的！

GIF via [GIPHY](https://giphy.com/gifs/beyonce-nicki-minaj-mK8jj0AHECrHW)

## 什么是线性回归？

> 线性回归着眼于您拥有的数据和您想要预测的数据之间的关系。

线性回归是一种基本且常用的预测分析类型。这是所有统计技术中应用最广泛的。它量化了一个或多个**预测变量**和一个**结果变量**之间的关系。

**线性回归模型**用于显示(或预测)两个变量或因素之间的关系。**回归分析**常用于显示两个变量之间的相关性。

例如，您可以查看棒球队球员的一些信息，并预测他们在该赛季的表现。你可能想研究一家公司的一些变量，并预测他们的股票走势。你甚至可能只想检查人们学习的小时数和他们在考试中的表现，或者你可以查看学生的家庭作业总成绩与他们在考试中表现的关系。这是一个非常有用的技术！

![](img/ae318e8c28d3267a6fddfb0722ea9c24.png)

Photo by StockSnap via [Pixabay](https://pixabay.com/photos/baseball-bat-athlete-sports-2617310/)

> 只要记住:**相关不等于因果**！两个变量之间存在关系并不意味着一个变量导致了另一个变量！回归分析不用于预测因果关系。它可以观察变量之间的关系。它可以检查变量之间的关联程度。这取决于你来仔细看看那些关系。

## 几个重要术语:

您的线性回归模型中的方程预测的变量称为**因变量**。我们称之为 y。用于预测因变量的变量称为**自变量。**我们称之为 **X** 。

你可以认为预测( **y** )依赖于其他变量( **X** )。这使得 **y** 成为因变量！

在**简单线性回归分析**中，每个观察值由两个变量组成。这些是自变量和因变量。**多元回归分析**着眼于两个或更多的自变量，以及它们如何与自变量相关联。描述 **y** 如何与 **X** 相关的等式被称为**回归模型**！

回归最早是由兴趣广泛的弗朗西斯·高尔顿爵士深入研究的。虽然他是一个非常有问题的角色，有很多值得反对的信仰，但他确实写了一些关于治疗枪伤和让你的马从流沙中脱离的很酷的信息的书。他还在指纹、听力测试方面做了一些有益的工作，甚至设计了第一张天气图。他于 1909 年被封为爵士。

在研究植物和动物中父母和孩子相对大小的数据时，他观察到，比一般人大的父母有比一般人大的孩子，但这些孩子在他们自己的一代中的相对位置会小一些。他称之为**回归平庸。**用现代术语来说，这将是向平均值的**回归。**

(然而，我不得不说，“回归平庸”这句话有一定的闪光点，我需要把它融入我的日常生活...)

GIF via [GIPHY](https://giphy.com/gifs/sam-elliott-UXlE9tMhPaJJC)

不过，要明确的是，我们谈论的是**期望**(预测)而不是绝对确定性！

## 回归模型有什么好处？

回归模型用于预测真实值，例如工资或身高。如果你的自变量是**时间**，那么你就是在预测未来值。否则，您的模型预测的是当前未知的值。回归技术的例子包括:

*   简单回归
*   多次回归
*   多项式回归
*   支持向量回归

假设您正在查看一些数据，其中包括员工的工作年限和工资。你应该看看这两个数字之间的相关性。也许你正在经营一家新企业或小公司，这些数字是随机设定的。

那么你如何找到这两个变量之间的相关性呢？为了搞清楚这一点，我们将创建一个模型，告诉我们什么是这种关系的最佳拟合线。

## 直觉

这里有一个简单的线性回归公式:

![](img/917c830a18a45c7827fac593587775a6.png)

(你可能认为这是高中代数中的斜率或趋势线方程。)

在这个等式中， **y** 是因变量，这就是你要解释的。在本文的其余部分， **y** 将是员工在一定工作年限后的工资。

可以看到上面的自变量。这是与你的预测值变化相关的变量。自变量可能引起变化，或者只是与变化有关。记住，**线性回归不能证明因果关系**！

这个系数就是你如何解释自变量的变化可能不完全等于 y 的变化。

现在我们想看看证据。我们想在我们的数据中画一条最适合我们的数据的线。回归线可以显示正线性关系(该线看起来向上倾斜)、负线性关系(该线向下倾斜)或完全没有关系(一条平线)。

![](img/e2dbaf43ff543311aa2db2a5bec9394e.png)![](img/2a1a371abbc7a5afb7c1ff4cb85e1823.png)![](img/4487887d5023c6cd1024aa9a36532145.png)

该常数是直线与纵轴相交的点。例如，如果你在下图中看到 0 年的工作经验，你的工资大约是 30，000 美元。所以下图中的常数大约是 30，000 美元。

![](img/9c0a84f7175a6c5de9df346c07ef30f6.png)

坡度越陡，你多年的经验得到的钱就越多。例如，也许多一年的经验，你的工资(y)会额外增加 10，000 美元，但如果斜率更大，你可能会得到 15，000 美元。如果斜率为负，随着经验的积累，你实际上会赔钱，但我真的希望你不要在那家公司工作太久...

## 简单线性回归怎么找到那条线？

当我们看一张图表时，我们可以从这条线到我们的实际观察值画出垂直线。您可以将实际观察值视为点，而线条显示模型观察值(预测值)。

![](img/d7a227c5f24814189b5bdefc8fe6649a.png)

我们画的这条线是员工实际收入和他的模型(预测)收入之间的差异。我们将查看**最小平方和**以找到最佳直线，这意味着您将获得所有平方差的总和并找到最小值。

那叫**普通最小二乘法**法！

## 那么我们该怎么做呢？

首先是进口货！

```
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
```

现在让我们预处理我们的数据！如果你不太了解数据清理和预处理，你可能想看看这篇文章。它将引导您完成导入库、准备数据和要素缩放。

[](/the-complete-beginners-guide-to-data-cleaning-and-preprocessing-2070b7d4c6d) [## 数据清理和预处理完全初学者指南

### 如何在几分钟内为机器学习模型成功准备数据

towardsdatascience.com](/the-complete-beginners-guide-to-data-cleaning-and-preprocessing-2070b7d4c6d) 

我们将复制并粘贴那篇文章中的代码，并做两处微小的修改。当然，我们需要更改数据集的名称。然后我们来看一下数据。在我们的例子中，假设我们的员工有一列年资和一列薪水，仅此而已。请记住，我们的索引从 0 开始，我们将继续从因变量的数据中分离出最后一列，就像我们已经设置的那样。然而，这一次，我们将获取自变量的第二列，所以我们将做一个小的改变来获取它。

```
dataset = pd.read_csv('salary.csv')
X = dataset.iloc[:, :-1].values
y = dataset.iloc[:, 1].values
```

现在 X 是一个特征矩阵(我们的自变量), y 是因变量的向量。完美！

是时候将我们的数据分成训练集和测试集了。通常，我们会对我们的训练和测试数据进行 80/20 分割。然而，在这里，我们使用的是一个只有 30 个观察值的小数据集。也许这一次我们将拆分我们的数据，这样我们就有 20 个训练观察和 10 个测试规模。

```
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 1/3, random_state = 0)
```

你有 X_train，X_test，y_train，y_test！你已经准备好了！(永远不要忘记，在这个过程的每一步，都有大约一百万件事情需要学习、改变和改进。你的模型的力量取决于你和你投入的一切！)

![](img/e793af6656b66626c2b9dbd659e10894.png)

Photo by [Thomas William](https://unsplash.com/@thomasw?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

我们将随机状态设置为 0，这样我们可以得到相同的结果。(计算中可能会有随机因素，我想确保我们都在同一页上，这样就没有人会紧张。)

我们将在训练集上训练我们的模型，然后根据我们的信息预测结果。我们的模型将**学习**训练集上的相关性。然后，我们将通过让它用我们的测试集预测值来测试它学到了什么。我们可以将我们的结果与测试集上的实际结果进行比较，看看我们的模型做得如何！

A **总是把你的数据分成训练集和测试集**！如果你用你用来训练它的相同数据来测试你的结果，你可能会有非常好的结果，但是你的模型并不好！它只是记住了你想让它做的事情，而不是学习任何可以用于未知数据的东西。那叫过度拟合，也就是说你**没有造好模型**！

## 特征缩放

我们实际上不需要在这里做任何功能缩放！

![](img/3e2b82f3d6f75c8f6bef1e12423368cf.png)

Photo by [Gift Habeshaw](https://unsplash.com/@gift?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

## 线性回归

现在我们可以让模型适合我们的训练集了！

为此，我们将使用 [Scikit-learn](https://scikit-learn.org/stable/index.html) learn。首先，我们将导入线性模型库和线性回归类。然后，我们将创建该类的一个对象—回归量。我们将使用一种方法(fit 方法)来使我们创建的回归对象适合训练集。为了创建对象，我们给它命名，然后用括号调用它。我们可以用大约三行代码完成所有这些工作！

让我们从 Scikit-Learn 导入线性回归，以便我们可以继续使用它。在括号之间，我们将指定我们想要使用的数据，以便我们的模型确切地知道我们想要拟合什么。我们希望获得 X_train 和 y_train，因为我们正在处理所有的训练数据。

如果你想了解更多的细节，你可以看看文档。

现在，我们准备创建我们的回归变量，并使其适合我们的训练数据。

```
from sklearn.linear_model import LinearRegression
regressor = LinearRegression()
regressor.fit(X_train, y_train)
```

在那里！我们对数据使用简单的线性回归，并准备在测试集上测试我们的预测能力！

这是机器学习！我们创造了一台机器，回归器，我们让它在训练集上学习多年经验和薪水之间的关系。

现在，它可以根据已有的信息预测未来的数据。我们的机器已经准备好根据一个新员工的工作经验来预测他的工资了！

让我们用回归量来预测新的观察结果。我们想通过观察新的观察来了解机器是如何学习的。

我们将创建一个预测值的向量。这是因变量的预测向量，我们称之为 y_pred。为此，我们可以使用我们创建和训练的回归变量，并使用 predict 方法。我们需要指定要进行哪些预测，因此我们希望确保包含测试集。对于 regressor.predict 中的输入参数，我们希望指定新观测值的特征矩阵，因此我们将指定 X_test。

```
y_pred = regressor.predict(X_test)
```

说真的。那只需要一行代码！

现在，y_test 是测试集中 10 个观察值的实际工资，y_pred 是我们的模型预测的这 10 个雇员的预测工资。

你做到了！四行代码的线性回归！

GIF via [GIPHY](https://giphy.com/gifs/snoop-dogg-drop-it-like-its-hot-ScZzMlETdv9mg)

## 形象化

让我们把结果可视化吧！我们需要看看我们的预测和实际结果之间有什么区别。

我们可以绘制图表来解释结果。首先，我们可以使用 plt.scatter 绘制真实的观察值，以制作散点图。(我们之前导入 matplotlib.pyplot 为 plt)。

我们将首先查看训练集，因此我们将在 X 坐标上绘制 X_train，在 y 坐标上绘制 y_train。那么我们可能想要一些颜色。我们用蓝色表示我们的观察，用红色表示我们的回归线(预测)。对于回归线，我们将再次对 X 坐标使用 X_train，然后使用 X_train 观测值的预测。

让我们用 x 轴和 y 轴的标题和标签来稍微想象一下。

```
plt.scatter(X_train, y_train, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Salary vs Experience (Training set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

现在我们可以看到我们的蓝点，它们是我们的真实值和我们沿着红线的预测值！

![](img/d5db726d18eb23b6a3dcbb5dd8f2d96a.png)

让我们对测试集做同样的事情！我们将更改测试集的标题，并将代码中的“train”改为“test”。

```
plt.scatter(X_test, y_test, color = 'blue')
plt.plot(X_train, regressor.predict(X_train), color = 'red')
plt.title('Salary vs Experience (Test set)')
plt.xlabel('Years of Experience')
plt.ylabel('Salary')
plt.show()
```

请务必注意，我们没有在第二行中将 X_train 更改为 X_test。我们的回归变量已经被训练集训练过了。当我们训练时，我们得到了一个唯一的模型方程。如果我们替换它，我们会得到相同的线，我们可能会建立相同回归线的新点。

![](img/9c0a84f7175a6c5de9df346c07ef30f6.png)

这是一个相当不错的模型！

我们的模型很好地预测了这些新员工的工资。一些实际观察和预测是一样的，这很好。在 **y** 和 **X** 变量之间没有 100%的相关性，所以一些预测不会完全准确。

你做到了！您导入了库，清理并预处理了数据，构建并训练了一个简单的线性回归器，用它来进行预测，甚至还可视化了结果！

恭喜你！！！

![](img/719a31e08a33323b12830c259248cc8e.png)

Photo by Free-Photos via [Pixabay](https://pixabay.com/photos/girls-sparklers-fireworks-984154/)

## 想要更多吗？

[接下来是多元线性回归](/multiple-linear-regression-in-four-lines-of-code-b8ba26192e84)！

[](/multiple-linear-regression-in-four-lines-of-code-b8ba26192e84) [## 机器学习完全初学者指南:4 行代码中的多元线性回归！

### 征服多元线性回归的基础(和向后消除！)并用你的数据预测未来！

towardsdatascience.com](/multiple-linear-regression-in-four-lines-of-code-b8ba26192e84) 

## 继续学习！

机器学习建立在统计学的基础上，如果没有简单的线性回归等概念，你就无法开始理解机器学习。但这并不意味着统计学和机器学习是一回事！除了作为机器学习的基本构件的一部分之外，线性回归器在很大程度上是统计学(和数据科学)的工具。

如果你有兴趣了解更多关于统计学和机器学习之间的差异，我推荐你看看马修·斯图尔特博士研究员写的这篇精彩的文章！他很好地阐明了这些概念，我强烈推荐你花些时间通读他的文章。

[](/the-actual-difference-between-statistics-and-machine-learning-64b49f07ea3) [## 统计学和机器学习的实际区别

### 不，它们不一样。如果机器学习只是被美化的统计学，那么建筑学只是被美化了…

towardsdatascience.com](/the-actual-difference-between-statistics-and-machine-learning-64b49f07ea3) 

和往常一样，如果你用这些信息做了什么很酷的事情，请在下面的回复中让人们知道，或者随时在 LinkedIn 上联系。

你可能也想看看这些文章:

[](/getting-started-with-git-and-github-6fcd0f2d4ac6) [## Git 和 GitHub 入门:完全初学者指南

### Git 和 GitHub 基础知识，供好奇和完全困惑的人使用(加上最简单的方法来为您的第一次公开…

towardsdatascience.com](/getting-started-with-git-and-github-6fcd0f2d4ac6) [](/how-to-create-a-free-github-pages-website-53743d7524e1) [## 如何用 GitHub 毫不费力地免费创建一个网站

### GitHub Pages 入门:创建和发布免费作品集的快速简便指南…

towardsdatascience.com](/how-to-create-a-free-github-pages-website-53743d7524e1) [](/getting-started-with-google-colab-f2fff97f594c) [## Google Colab 入门

### 沮丧和困惑的基本教程

towardsdatascience.com](/getting-started-with-google-colab-f2fff97f594c) [](/intro-to-deep-learning-c025efd92535) [## 深度学习简介

### 新手、新手和新手的神经网络。

towardsdatascience.com](/intro-to-deep-learning-c025efd92535) [](/wtf-is-image-classification-8e78a8235acb) [## WTF 是图像分类？

### 为好奇和困惑征服卷积神经网络

towardsdatascience.com](/wtf-is-image-classification-8e78a8235acb) [](https://medium.freecodecamp.org/how-to-build-the-best-image-classifier-3c72010b3d55) [## 如何构建准确率高于 97%的图像分类器

### 清晰完整的成功蓝图

medium.freecodecamp.org](https://medium.freecodecamp.org/how-to-build-the-best-image-classifier-3c72010b3d55) [](https://heartbeat.fritz.ai/brilliant-beginners-guide-to-model-deployment-133e158f6717) [## 出色的模型部署初学者指南

### 一个清晰简单的路线图，让你的机器学习模型在互联网上，做一些很酷的事情

heartbeat.fritz.ai](https://heartbeat.fritz.ai/brilliant-beginners-guide-to-model-deployment-133e158f6717) 

感谢阅读！