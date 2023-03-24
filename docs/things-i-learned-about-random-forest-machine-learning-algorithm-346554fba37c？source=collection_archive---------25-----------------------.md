# 我学到的关于随机森林机器学习算法的东西

> 原文：<https://towardsdatascience.com/things-i-learned-about-random-forest-machine-learning-algorithm-346554fba37c?source=collection_archive---------25----------------------->

## 第一课概述:来自 [Fast.ai](https://www.fast.ai/) 的机器学习课程随机森林介绍

![](img/9ae043c3e713fae2d686a4595c675be8.png)

Forest in Autumn (Image by [Valentin Sabau](https://pixabay.com/users/Valiphotos-1720744/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=1072821) from [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=1072821))

几个月前，我在悉尼参加了一个聚会，会上， [fast.ai](https://www.fast.ai/) 向我介绍了一个在线机器学习课程。那时我从未注意过它。本周，在参加一个卡格尔竞赛，寻找提高分数的方法时，我又碰到了这个课程。我决定试一试。

这是我从第一堂课中学到的东西，这是一个 1 小时 17 分钟的关于 [**随机森林**](http://course18.fast.ai/lessonsml1/lesson1.html) 介绍的视频。

虽然讲座的主题是随机森林，但 Jeremy(讲师)提供了一些一般信息以及使用 Jupyter 笔记本的提示和技巧。

![](img/09bbde56284b20da835f6517ba3ded82.png)

Laptop Screen with code (Photo by [Clément H](https://unsplash.com/@clemhlrdt?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText))

杰里米谈到的一些重要的事情是，数据科学不等同于软件工程。在数据科学中，我们建立模型的原型。虽然软件工程有自己的一套实践，但是数据科学也有自己的一套最佳实践。

模型构建和原型制作需要一个交互式的环境，并且是一个迭代的过程。我们建立一个模型。然后，我们采取措施改善它。重复，直到我们对结果满意为止。

# 随机森林

![](img/4b28ff3939cc405eafd161b53ba228b3.png)

Photo of Forest (Photo by [Sebastian Unrau](https://unsplash.com/@sebastian_unrau?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText))

我听说过随机森林这个术语，我知道它是现有的机器学习技术之一，但说实话，我从未费心去了解更多关于它的知识。相反，我总是渴望了解更多关于深度学习技术的知识。

从这个讲座中，我了解到随机森林确实很牛逼。

它就像一种通用的机器学习技术，可用于回归和分类目的。这意味着你可以使用随机森林来预测股票价格，并对给定的医疗数据样本进行分类。

一般来说，随机森林模型不会过拟合，即使它过拟合，也很容易阻止它过拟合。

随机森林模型不需要单独的验证集。

随机森林只有很少的统计假设。它既不假设您的数据是正态分布的，也不假设关系是线性的。

它只需要很少的特征工程。

因此，如果你是机器学习的新手，这可能是一个很好的起点。

# 其他概念

**维数灾难**是一个概念，数据的特征越多，数据点就越分散。这意味着两点之间的距离意义要小得多。

杰里米保证，在实践中，情况并非如此，事实上，你的数据拥有的特征越多，就越有利于训练模型。

**没有免费的午餐定理**是一个概念，没有一个模型可以完美地适用于任何类型的数据。

# 提示和技巧

![](img/f23e52fdb4190e7b14047dd2da8da7e6.png)

Little girl doing a magic trick (Photo by [Paige Cody](https://unsplash.com/@paige_cody?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText))

1.  你可以通过包含**来痛击 Jupyter 笔记本中的命令！**在命令面前。

```
!ls
!mkdir new_dr
```

2.Python 3.6 中追加字符串的新方法。

```
name = 'Sabina'
print(f'Hello {name}')no_of_new_msg = 11
print(f'Hello {name}, you have {no_of_new_msg} new messages')
```

3.不用离开 Jupyter 笔记本就可以了解 python 函数。用**？**在函数名前获取其文档。

```
from sklearn.ensemble import RandomForestClassifier?RandomForestClassifier.fit()
```

4.如果想看源代码，可以用 double **？**在函数名前面。

```
from sklearn.ensemble import RandomForestClassifier??RandomForestClassifier.fit()
```

5.使用 ***to_feather*** 方法保存处理过的数据集，将数据集以其存储在 RAM 中的格式保存到磁盘中。您可以使用 ***read_feather*** 方法从保存的文件中读回数据。注意，为了使用这些方法，你需要安装 ***羽毛格式*** 库。

```
import pandasdf = pd.DataFrame()
df.to_feather('filename')saved_df= pd.read_feather('filename')
```

如果你有兴趣深入这个话题，这里有一个到讲座视频的链接。

请在下面留下你的想法。如果你确实想去看看这门课程，请告诉我你是怎么去的。

[**点击这里**](https://medium.com/@sabinaa.pokhrel) 阅读我其他关于 AI/机器学习的帖子。

以下是该主题的一些进一步阅读材料:

 [## 面向程序员的深度学习——36 小时免费课程

### 点按视频右下角的方块以全屏观看。有关本课的更多信息，请访问…

course18.fast.ai](http://course18.fast.ai/lessonsml1/lesson1.html) [](/understanding-random-forest-58381e0602d2) [## 了解随机森林

### 该算法如何工作以及为什么如此有效

towardsdatascience.com](/understanding-random-forest-58381e0602d2)