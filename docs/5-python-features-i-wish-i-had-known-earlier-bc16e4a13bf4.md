# 我希望我能早点知道的 5 个 Python 特性

> 原文：<https://towardsdatascience.com/5-python-features-i-wish-i-had-known-earlier-bc16e4a13bf4?source=collection_archive---------0----------------------->

## Python 初学者

## 超越 lambda、map 和 filter 的 Python 技巧

![](img/27503c7254cd37d60c30512078795af0.png)

Photo by [Kirill Sharkovski](https://unsplash.com/@sharkovski?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

Python 可以说是近十年来崛起的编程语言，并且被证明是一种非常强大的语言。我已经用 Python 开发了很多应用，从[交互式地图](/visualizing-bike-mobility-in-london-using-interactive-maps-for-absolute-beginners-3b9f55ccb59)到[区块链](/building-a-minimal-blockchain-in-python-4f2e9934101d)。Python 的特性那么多，初学者一开始很难掌握所有的东西。

即使你是一个从 C 或 MATLAB 等其他语言转换过来的程序员，用更高抽象层次的 Python 编码绝对是一种不同的体验。我希望我能早点知道 Python 的一些特性，并强调其中最重要的五个。

# 1.列表理解—紧凑代码

许多人会将 **lambda** 、 **map** 和 **filter** 作为每个初学者都应该学习的 Python“技巧”。虽然我认为它们是我们应该知道的功能，但我发现它们在大多数时候并不特别有用，因为它们缺乏灵活性。

`Lambda`是在一行**中组合一个函数供一次性使用**的方法。如果多次调用这些函数，性能会受到影响。另一方面，`map`将一个函数应用于列表中的所有元素，而`filter`获取满足用户定义条件的集合中的元素子集。

![](img/5ffaa7070bf8e9fc6c638d8b26733386.png)

Photo by [Anastase Maragos](https://unsplash.com/@visualsbyroyalz?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

**列表理解**是一种简洁而灵活的方法，通过灵活的表达式和条件从其他列表中创建列表。它由一个方括号构成，带有一个表达式或函数，仅当元素满足特定条件时，该表达式或函数才应用于列表中的每个元素。它还可以嵌套处理嵌套列表，比使用 map 和 filter 灵活得多。

```
# Syntax of list comprehension **[** expression(x) **for** x **in** aList **if** optional_condition(x) **]**
```

# 2.列表操作—循环列表

Python 允许**负索引** where `aList[-1] == aList[len(aList)-1]`。因此，我们可以通过调用`aList[-2]`等等来获得列表中的倒数第二个元素。

我们还可以使用语法`aList[start:end:step]`对列表进行**切片，其中包含开始元素，但不包含结束元素。因此，调用`aList[2:5]`就产生了`[2, 3, 4]`。我们也可以简单地通过调用`aList[::-1]`来**反转一个列表**，我发现这种技术非常优雅。**

![](img/c230129bbaea15ca0a415f4baf1587bb.png)

Photo by [Martin Shreder](https://unsplash.com/@martinshreder?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

列表也可以**解包**成单独的元素，或者使用星号混合元素和子列表。

# 3.压缩和枚举循环

`Zip`函数创建了一个**迭代器**，它聚集了多个列表中的元素。它允许**在 for 循环中并行遍历列表**和**并行排序**。可以用星号解压。

![](img/a42b457032d8a394bea576175ba487b6.png)

Photo by [Erol Ahmed](https://unsplash.com/@erol?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

`Enumerate`一开始可能看起来有点吓人，但在许多情况下会变得非常方便。它是一个**自动计数器**，经常在 for 循环中使用，因此**不再需要通过`counter = 0`和`counter += 1`在 for 循环中创建和初始化计数器**变量。Enumerate 和 zip 是构造 for 循环时两个最强大的工具。

# 4.生成器—内存效率

**当我们打算计算一大组结果，但希望**避免同时分配所有结果所需的内存**时，使用生成器**。换句话说，它们动态地生成值**并且不在内存中存储先前的值，因此我们只能对它们迭代一次。**

它们通常在读取大文件或**使用关键字`yield`生成无限序列**时使用。我经常发现它在我的大部分数据科学项目中很有用。

# 5.虚拟环境—隔离

如果您只能记住本文中的一件事，那么它应该是使用虚拟环境。

![](img/9f81a8cf44b66e6454c84de27dbaa51a.png)

Photo by [Matthew Kwong](https://unsplash.com/@mattykwong1?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

Python 应用程序经常使用来自不同开发者的许多不同的**包**，具有复杂的**依赖性**。使用特定的库设置开发不同的应用程序，其中**结果不能使用其他库版本再现**。这里**不存在满足所有应用要求的单一安装**。

```
conda create -n venv pip python=3.7  # select python version
source activate venv
...
source deactivate
```

因此，为每个应用程序创建独立的自包含虚拟环境`venv`至关重要，这可以使用`pip`或`conda`来完成。

## 相关文章

感谢您的阅读。你可以[注册我的简讯](http://edenau.mailchimpsites.com/)来接收我的新文章的更新。如果您对数据科学感兴趣，以下文章可能会有用:

[](/4-numpy-tricks-every-python-beginner-should-learn-bdb41febc2f2) [## 每个 Python 初学者都应该学习的 4 个 NumPy 技巧

### 编写可读代码的技巧

towardsdatascience.com](/4-numpy-tricks-every-python-beginner-should-learn-bdb41febc2f2) [](/6-new-features-in-python-3-8-for-python-newbies-dc2e7b804acc) [## Python 3.8 中针对 Python 新手的 6 项新特性

### 请做好准备，因为 Python 2 不再受支持

towardsdatascience.com](/6-new-features-in-python-3-8-for-python-newbies-dc2e7b804acc) [](/4-common-mistakes-python-beginners-should-avoid-89bcebd2c628) [## Python 初学者应该避免的 4 个常见错误

### 我很艰难地学会了，但你不需要

towardsdatascience.com](/4-common-mistakes-python-beginners-should-avoid-89bcebd2c628) 

*最初发布于*[*edenau . github . io*](https://edenau.github.io/)*。*