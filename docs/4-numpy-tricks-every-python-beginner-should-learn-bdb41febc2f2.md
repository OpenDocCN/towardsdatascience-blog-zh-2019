# 每个 Python 初学者都应该学习的 4 个 NumPy 技巧

> 原文：<https://towardsdatascience.com/4-numpy-tricks-every-python-beginner-should-learn-bdb41febc2f2?source=collection_archive---------4----------------------->

## Python 初学者

## 编写可读代码的技巧

![](img/4b5280959db0aa44b7fc44784eccde2b.png)

Photo by [Pierre Bamin](https://unsplash.com/@bamin?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

**NumPy** 是 **Python** 中最流行的库之一，鉴于它的优点，几乎每个 Python 程序员都用它来进行算术计算。Numpy 数组比 Python 列表更紧凑。这个库也非常方便，许多常见的矩阵运算以非常高效的计算方式实现。

在帮助同事和朋友解决数字问题后，我总结了 4 个数字技巧，Python 初学者应该学习。这些技巧会帮助你写出更加整洁和可读的代码。

在学习 numpy 技巧之前，请确保您熟悉以下文章中的一些 Python 内置特性。

[](/5-python-features-i-wish-i-had-known-earlier-bc16e4a13bf4) [## 我希望我能早点知道的 5 个 Python 特性

### 超越 lambda、map 和 filter 的 Python 技巧

towardsdatascience.com](/5-python-features-i-wish-i-had-known-earlier-bc16e4a13bf4) 

# 1.参数函数—位置

对于数组`arr`、`np.argmax(arr)`、`np.argmin(arr)`和`np.argwhere(condition(arr))`，分别返回最大值、最小值和满足用户定义条件的值的索引。虽然这些 arg 函数被广泛使用，但我们经常忽略函数`np.argsort()`,它返回对数组排序的**索引。**

我们可以使用`np.argsort`到**根据另一个数组**对数组的值进行排序。下面是一个使用考试成绩对学生姓名进行排序的示例。排序后的名称数组也可以使用`np.argsort(np.argsort(score))`转换回原来的顺序。

它的性能比使用内置 Python 函数`sorted(zip())`更快，并且更具可读性。

![](img/0fca49145e9cb17f68dcc6eb1edcbafd.png)

Photo by [Kamen Atanassov](https://unsplash.com/@katanassov?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 2.广播-形状

广播是一个初学者可能无意中尝试过的事情。许多 numpy 算术运算在逐个元素的基础上应用于具有**相同形状**的数组对。广播对数组操作进行矢量化**，而不会产生不必要的数据副本**。这导致了高效的算法实现和更高的代码可读性。

例如，您可以使用`arr + 1`将数组中的所有值递增 1，而不考虑`arr`的维度。还可以通过`arr > 2`检查数组中的所有值是否都大于 2。

但是我们怎么知道两个数组是否兼容广播呢？

```
Argument 1  (4D array): 7 × 5 × 3 × 1
Argument 2  (3D array):     1 × 3 × 9
Output      (4D array): 7 × 5 × 3 × 9
```

两个数组的每个维度必须是**等于**，或者**其中一个是 1** 。它们不需要有相同的维数。上面的例子说明了这些规则。

# 3.省略和新轴——维度

分割 numpy 数组的语法是`i:j`，其中 *i，j* 分别是起始索引和停止索引。比如上一篇文章中提到的——5 个我希望早点知道的 Python 特性，对于一个 numpy 数组`arr = np.array(range(10))`，调用`arr[:3]`给出`[0, 1, 2]`。

![](img/a1e3b56880f14361272cfa171806a1e5.png)

Photo by [Daryan Shamkhali](https://unsplash.com/@daryan?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

当处理高维数组时，我们使用`:`来选择每个轴上的所有索引。我们也可以使用`…`选择多个轴上的所有指标**。展开的轴的确切数量是从**推断出来的**。**

另一方面，如上图所示使用`np.newaxis`**在用户定义的轴位置插入一个新轴**。此操作将数组的形状扩展一个单位的维度。虽然这也可以通过`np.expand_dims()`来完成，但是使用`np.newaxis`可读性更好，也更优雅。

# 4.屏蔽阵列—选择

数据集是不完美的。它们总是包含缺少或无效条目的数组，我们经常想忽略那些条目。例如，由于传感器故障，气象站的测量值可能包含缺失值。

Numpy 有一个子模块`numpy.ma`，支持带有掩码的数据数组**。被屏蔽的数组包含一个普通的 numpy 数组和一个掩码**指示无效条目的位置**。**

```
np.ma.MaskedArray(data=arr, mask=invalid_mask)
```

![](img/5d6f0519bdc91c0f12e4fda30359edef.png)

Photo by [Nacho Bilbao](https://unsplash.com/@nachoscense?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

数组中的无效条目有时用负值或字符串标记。如果我们知道屏蔽值，比如说`-999`，我们也可以使用`np.ma.masked_values(arr, value=-999)`创建一个屏蔽数组。任何以掩码数组作为参数的 numpy 操作都会自动忽略那些无效的条目，如下所示。

## 相关文章

感谢您的阅读。你可以[注册我的时事通讯](http://edenau.mailchimpsites.com/)来接收我的新文章的更新。如果您对 Python 感兴趣，以下文章可能会有用:

[](/5-python-features-i-wish-i-had-known-earlier-bc16e4a13bf4) [## 我希望我能早点知道的 5 个 Python 特性

### 超越 lambda、map 和 filter 的 Python 技巧

towardsdatascience.com](/5-python-features-i-wish-i-had-known-earlier-bc16e4a13bf4) [](/6-new-features-in-python-3-8-for-python-newbies-dc2e7b804acc) [## Python 3.8 中针对 Python 新手的 6 项新特性

### 请做好准备，因为 Python 2 不再受支持

towardsdatascience.com](/6-new-features-in-python-3-8-for-python-newbies-dc2e7b804acc) 

*最初发表于*[*edenau . github . io*](https://edenau.github.io/)*。*