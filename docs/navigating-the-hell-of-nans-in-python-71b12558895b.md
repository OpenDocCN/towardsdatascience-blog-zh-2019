# 在 Python 中导航 NaNs 的地狱

> 原文：<https://towardsdatascience.com/navigating-the-hell-of-nans-in-python-71b12558895b?source=collection_archive---------0----------------------->

了解 nan 并在您的数据中轻松处理它们的摘要。

![](img/0c4acbb2954b0db99ade80077e234f62.png)

Photo by [Chris Ried](https://unsplash.com/@cdr6934?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

![I](img/372a85df590327f646ff2d6a8b3607e9.png) 我 最近有很多 NaNs 引起的头痛。每个程序员都知道它们是什么，以及它们为什么会发生，但在我的情况下，我并不知道它们的所有特征，或者说还不够好来阻止我的努力。为了找到解决方案并避免令人头疼的问题，我进一步研究了 Python 中 NaNs 值的行为。在 Jupyter Notebook 里摆弄了几个语句之后，我的结果相当令人吃惊，也极其混乱。这是我用 Numpy 的 np.nan 得到的。

`np.nan in [np.nan]`是`True`

到目前为止还好，但是…

`np.nan == np.nan`是`False`

啊？还有…

`np.nan is np.nan`是`True`

> 那么 Python 中的 NaNs 到底是怎么回事？

# 简短介绍

NaN 代表**不是一个数字，是一种常见的缺失数据表示。**它是一个特殊的浮点值，不能转换为除 float 以外的任何其他类型。甚至在 Python 存在之前，它就由用于算术运算的二进制浮点 I [EEE 标准](https://www.python-course.eu/dealing_with_NaN_in_python.php) (IEEE 754)引入，并在遵循该标准的所有系统中使用。NaN 可以被视为某种数据病毒，会感染它所涉及的所有操作。

## 无对南

None 和 NaN 听起来相似，看起来相似，但实际上很不一样。None 是一个 Python 内部类型，可以认为它等同于 NULL。`[None](https://www.w3schools.com/python/ref_keyword_none.asp)`[关键字用于定义空值，或者根本没有值。None 不同于 0、False 或空字符串。它是自己的数据类型(NoneType ),并且只有 None 可以是… None。](https://www.w3schools.com/python/ref_keyword_none.asp)数值数组中的缺失值为 NaN，而对象数组中的缺失值为 None。最好通过使用`foo is None`而不是`foo == None which brings`来检查无。我们回到上一个问题，我在 NaN 操作中发现了特殊的结果。

# 南不等于南

起初，读到`np.nan == np.nan`是`False`会引发困惑和沮丧的反应。这看起来很奇怪，听起来真的很奇怪，但如果你稍微思考一下，逻辑就会开始出现，甚至开始变得有意义。

> 尽管我们不知道每个 NaN 是什么，但不是每个 NaN 都是一样的。

让我们想象一下，我们看到的不是 nan 值，而是一群我们不认识的人。他们对我们来说是完全陌生的人。陌生人对我们来说都是一样的，也就是说我们把他们都描述成陌生人。但是，现实中并不意味着一个不认识的人就等于另一个不认识的人。

离开我这个奇怪的比喻，回到 Python， **NaN 不能等于它自己，因为 NaN 是失败的结果**，但是失败可以以多种方式发生。一次失败的结果不能等于任何其他失败的结果，未知值也不能彼此相等。

# 平等与身份

现在，要理解`np.nan in [np.nan]`为什么是`True`，我们得看看*相等*和*相同*的区别。

## 平等

等式指的是大多数 Python 程序员都知道的“==”这个概念。这用于询问 Python 该变量的内容是否与另一个变量的内容相同。

```
num = 1
num2 = 1num == num2 
```

最后一行将导致`True`。**两个变量的内容相同**。如前所述，一个 NaN 的内容永远不等于另一个 NaN 的内容。

## 身份

当你问 Python 一个变量**是否与另一个变量**相同时，你就是在问 Python 这两个变量是否共享**相同的标识**。Python 为每个创建的变量分配一个 **id** ，当 Python 在一个操作中查看变量的身份时，会比较 id。然而，`np.**nan**` **是一个单一的对象，它总是有相同的 id，不管你把它赋给哪个变量。**

```
import numpy as np
one = np.nan
two = np.nan
one is two
```

`np.nan is np.nan`是`True``one is two`也是`True`。

如果使用`id(one)`和`id(two)`检查`one`和`two`的 id，将显示相同的 id。

`np.nan in [np.nan]`之所以是`True`是因为 Python 中的 list 容器在检查**相等**之前先检查**身份** **。然而，根据它们是如何被创造出来的，它们有不同的“味道”。`float(‘nan’)`用不同的 id 创建不同的对象，所以`float('nan') is float('nan')`实际上给了**假！！稍后我们将再次提到这些差异。****

# 对付南而不头痛

最初，完整的 nan 概念可能很难理解，也很难处理。令人欣慰的是， **pandas** 和 **numpy** 在处理 nan 值方面非常出色，它们提供了几个函数，可以轻松地选择、替换或删除变量中的 nan 值。

## 测试值是否为 nan

正如我所说的，每当你想知道一个值是否是一个 nan，你不能检查它是否等于 nan。然而，有许多其他的选择可以做到这一点，我提出的并不是唯一的选择。

```
import numpy as np
import pandas as pdvar = float('nan')var is np.nan #results in True
#or
np.isnan(var) #results in True
#or
pd.isna(var) #results in True
#or
pd.isnull(var)#results in True
```

`pd.isnull` & `pd.isna()`表现一致。熊猫提供了。isnull()函数，因为它是 Python 中 R 数据帧的改编版。在 R 中，null 和 na 是两种不同的类型，具有不同的行为。

除了 numpy 和从 **Python** 3.5 开始，你也可以使用`math.**nan**` 。我在本文中同时写 nan 和 NaN 的原因(除了我缺乏一致性)是值不区分大小写的事实。`float(‘nan’)`或`float(‘NAN’)`都会产生相同的结果。

```
import math
var = float('nan')
math.isnan(var) #results in True
```

**小小警告:**

```
import math
import numpy as np
math.nan is math.nan #results in True
math.nan is np.nan #results in False
math.nan is float('nan') #results in False
```

因为`math.nan`、`np.nan`和`float('nan')`都有不同的 id，所以这些语句为假。他们没有相同的身份。

## **对于数据帧**

```
import pandas as pddf = pd.DataFrame(some_data)df.dropna()
#will drop all rows of your dataset with nan values. 
#use the subset parameter to drop rows with nan values in specific columnsdf.fillna()
#will fill nan values with the value of your choicedf.isnull()
#same as pd.isnull() for dataframesdf.isna()
#same as pd.isna() for dataframes
```

不幸的是，我并不认为熊猫文档在丢失数据文档方面非常有用。然而，我真的很欣赏这篇摘自 *Python 数据科学手册* 的[，它很好地概述了如何处理 Pandas 中的缺失数据。](https://jakevdp.github.io/PythonDataScienceHandbook/03.04-missing-values.html)

# 需要注意什么

> type error:“float”对象不可迭代

虽然非类型错误非常清楚，但是由 nan 值引起的错误可能会有点混乱。Nan 值经常会导致错误(更具体地说是 **TypeErrors** )，这将涉及到它们的类型“ **float** ”。错误消息可能会令人惊讶，尤其是当您认为您的数据绝对没有浮点数时。您的数据帧可能看起来不包含任何浮点，但实际上，它确实包含。它可能有你不知道的 nan 值，你只需要去掉你的 NaN 值就可以消除这个错误！

作为一名数据科学家和 Python 程序员，我喜欢分享我在这一领域的经验，并将继续撰写关于 Python、机器学习或任何有趣的发现的文章，这些发现可能会使其他程序员的生活和任务变得更容易。在[*Medium*](https://medium.com/me)*或*[*Twitter*](https://twitter.com/DIRUSSOJulia)*上关注我，以获得关于 Python &数据科学的任何未来文章的通知！*

[](/explaining-data-science-to-your-grandma-f8345621483d) [## 向你的祖母解释数据科学

### 或者如何向你的家人或任何与技术世界脱节的人解释数据科学。

towardsdatascience.com](/explaining-data-science-to-your-grandma-f8345621483d)