# 最被低估的标准 Python 库

> 原文：<https://towardsdatascience.com/the-most-undervalued-standard-python-library-14021632f692?source=collection_archive---------1----------------------->

## [视频教程](https://towardsdatascience.com/tagged/video-tutorial)

## 面向数据科学家的集合

[YouTube](https://www.youtube.com/watch?v=Cp2cduNpHY8)

Python 有许多现成的优秀库。其中之一就是 [**收藏**](https://docs.python.org/2/library/collections.html) 。集合模块提供了“高性能容器数据类型”,它提供了通用容器 dict、list、set 和 tuple 的替代方案。我很乐意向你介绍其中的三种数据类型，最后，**你会想知道没有它们你是如何生活的。**

# 命名元组

我不能夸大[命名的组合](https://docs.python.org/2/library/collections.html#collections.namedtuple)对数据科学家的用处。让我知道这种情况听起来是否熟悉:你正在做特征工程，因为你喜欢列表，你只是不断地将特征添加到列表中，然后将列表输入到你的机器学习模型中。很快，您可能会有数百个功能，这时候事情就变得混乱了。您不再记得哪个特性引用了列表中的哪个索引。更糟糕的是，当别人看你的代码时，他们根本不知道这个庞大的特性列表是怎么回事。

> 输入 NamedTuples 以保存该天。

**只需几行额外的代码，你那疯狂杂乱的列表就会恢复到有序**。让我们来看看

如果您运行这个代码，它将打印出“22”，即您存储在您的行中的年龄。这太神奇了！现在，您不必使用索引来访问您的要素，而是可以使用人类可以理解的名称。这使得你的代码更容易维护，更简洁。

# 计数器

[计数器](https://docs.python.org/2/library/collections.html#collections.Counter)名副其实——它的主要功能是计数。这听起来很简单，但事实证明**数据科学家经常要计算事情**，所以它可以非常方便。

有几种方法可以初始化它，但我通常有一个值列表，并按如下方式输入该列表

如果您要运行上面的代码(您可以使用这个[棒极了的工具](https://www.pythonanywhere.com/gists/))，您将得到下面的输出:

```
[(22, 5), (25, 3), (24, 2), (30, 2), (35, 1), (40, 1), (11, 1), (45, 1), (15, 1), (16, 1), (52, 1), (26, 1)]
```

按最常见的排序的元组列表，其中元组首先包含值，然后包含计数。因此，我们现在可以很快看到，22 岁是最常见的年龄，出现 5 次，还有一个长尾年龄，只有 1 次计数。不错！

# 默认字典

这是我的最爱之一。 [DefaultDict](https://docs.python.org/2/library/collections.html#collections.defaultdict) 是一个字典，当第一次遇到每个键时，用默认值初始化。这里有一个例子

这返回

```
defaultdict(<type 'int'>, {'a': 4, ' ': 8, 'c': 1, 'e': 2, 'd': 2, 'f': 2, 'i': 1, 'h': 1, 'l': 1, 'o': 2, 'n': 1, 's': 3, 'r': 2, 'u': 1, 't': 3, 'x': 1})
```

通常，当你试图访问一个不在字典中的值时，它会抛出一个错误。还有其他方法来处理这个问题，但是当您有一个想要采用的默认值时，它们会增加不必要的代码。在上面的例子中，我们用 int 初始化 defauldict。这意味着在第一次访问时，它将假设一个零，所以我们可以很容易地继续增加所有字符的计数。**简单干净**。另一个常见的初始化是 list，它允许您在第一次访问时立即开始追加值。

![](img/298f5bb788046ecf37d0c5c92583a5df.png)

Photo by [Hitesh Choudhary](https://unsplash.com/@hiteshchoudhary?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 去写更干净的代码

既然您已经了解了 collections library 和它的一些令人敬畏的特性，那就去使用它们吧！你会惊讶地发现它们是如此的有用，你的代码会变得更好。尽情享受吧！

这篇文章也可以在[这里](https://learningwithdata.com/posts/tylerfolkman/the-most-undervalued-standard-python-library-14021632f692/)找到。