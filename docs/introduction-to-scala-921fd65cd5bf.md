# Scala 简介

> 原文：<https://towardsdatascience.com/introduction-to-scala-921fd65cd5bf?source=collection_archive---------11----------------------->

![](img/27632fa093a9a8c91f6f1c37647cd865.png)

[source](https://www.slideshare.net/ViyaanJhiingade/introduction-to-scala-80328270)

WScala 是什么帽子？Scala 是一种高级语言，结合了函数式和面向对象编程以及高性能运行时。那么为什么要用 Scala 而不是 Python 呢？Spark 通常用于处理大数据的大多数情况。由于 Spark 是使用 Scala 构建的，因此学习它对于任何数据科学家来说都是一个很好的工具。

![](img/bc803604cf875e597d9e6eb70e72225a.png)

[source](http://Photo by Markus Spiske on Unsplash)

Scala 是一种强大的语言，可以利用许多与 Python 相同的功能，例如构建机器学习模型。我们将深入介绍 Scala，熟悉基本语法，并学习如何使用循环、映射和过滤器。在本指南中，我将使用社区版的[databricks.com](https://community.cloud.databricks.com/)。

# 变量

在 Scala 中，我们用`val`或`var`声明变量，我们将讨论两者之间的区别。值得注意的是，Scala 的语法约定是 camelCase，而 Python 在声明变量时使用 snake_case。让我们在 Scala 中创建第一个变量:

```
**val** counterVal = 0counterVal: Int = 0
```

现在让我们用`var`声明一个变量:

```
**var** counterVar = 0counterVar: Int = 0
```

当我们运行细胞时，它们看起来一样，输出也一样，但它们本质上是不同的。当我们试图估算每个变量的值时，我们可以看到这一点:

```
counterVar = 1counterVar: Int = 1
```

当我们试图赋予反补贴一个新的价值时，会发生这样的情况:

笔记本:1:错误:重新分配到 val count val = 1 ^

任何使用`val`声明的变量都是**不可变的**，，因此我们不能改变它的值。当我们不希望一个变量被改变时，不管是有意还是无意，使用`val`是非常好的。例如，我们可能想用一个名字作为我们的`val`并存储它，这样就没有人能更改一个人的名字。

```
**val** firstName = "John"**val** lastName = "Doe"firstName: String = John lastName: String = Doe
```

注意，Scala 显示了我们的变量(字符串)在创建时的类型。

# 用线串

在 Scala 中，我们可以像在 Python 中一样处理字符串。字符串的一个常见用途是插入，这意味着在短语或句子中插入一个注释或单词。字符串插值如下所示:

```
s"Hello, $firstName $lastName"res0: String = Hello, John Doe
```

类似于 Python 如何使用 *f* 字符串，这里我们使用 *s* 。双引号在 Scala 中很重要，因为如果我们试图传递单引号，它会返回一个错误。Scala 使用 *$* 对传递到插值中的变量进行排序。请注意它是如何自动在单词之间添加空格的:

```
s"Hello, ${firstName + lastName}"res1: String = Hello, JohnDoe
```

在上面的方法中，名字和姓氏之间没有分隔。为了获得空间，我们必须明确使用:`${firstName + " " + lastName}"`。我更喜欢对每个变量使用`$`而不使用`{}`——你可以使用任何一种方法进行插值。

## 字符串索引

数据科学中最常用的技术可能是索引和使用范围。在这两种情况下，Scala 都使用了`.slice()`方法，其中第一个数字是包含性的，而最后一个数字是排他性的。让我们来看一些例子。

首先我创建了一个名为“fullName”的新变量。

`val fullName = firstName + " " + lastName`

在 Scala 中，我们可以简单地在变量后使用`()`来调用变量的第一个索引。

这将在我们的全名变量中返回`J`。为了在我们的变量中索引一个索引范围，我们需要调用`.slice()`并传入这个索引范围。

运行`fullName.slice(3, 6)`将从名字返回`n D`。Scala 包含 3，并将空间计为 4。在`D`处停止，因为设置范围时 6 是唯一的。这和其他编程语言类似。熟悉这一概念需要时间，但仍会有不正确设置范围的时候。这里需要注意的一点是，您不能索引负数。对于熟悉 Python 的人来说，使用`[-1]`会返回索引的结尾，而 Scala 会给出一个错误。超出变量范围的索引将只给出最后一个。要获得字符串的长度，请使用:`.length()`。

# 数组

![](img/a6e1bada94371974e79a8dab3c67c96a.png)

[source](http://Photo by Glenn Carstens-Peters on Unsplash)

rrays 基本上就是 Scala 处理列表的方式。数组有类似 Python 的方法，但也有细微的差别，这使得 Scala 数组独一无二。创建数组时，请小心选择`var`,因为您可能想要更改数组值。因为数组只能改变它们的值，而不能改变数组本身的大小，所以我们将使用 ArrayBuffer 来演示。

```
**var** myArr = ArrayBuffer**(**2, 3 , 4, 5, 6**)**myArr: scala.collection.mutable.ArrayBuffer[Int] = ArrayBuffer(2, 3, 4, 5, 6)
```

注意，Scala 检测到我们的数组包含所有整数类型。现在我们有了一个数组，让我们来看一些例子，看看我们可以用它们做些什么。像字符串一样，我们可以对数组进行索引和切片，以查看任何给定位置的值。

```
myArr(3)res0: Int = 5myArr.slice(3, 5)res1: scala.collection.mutable.ArrayBuffer[Int] = ArrayBuffer(5, 6)
```

要向数组添加元素，使用`+=`添加一个值:

`myArr += 10`

```
myArrres3: scala.collection.mutable.ArrayBuffer[Int] = ArrayBuffer(2, 3, 4, 5, 6, 10)
```

您可以看到 10 作为最后一个元素被添加到数组中。我们也可以用类似的方式删除一个项目:

```
myArr -= 10myArrres7: scala.collection.mutable.ArrayBuffer[Int] = ArrayBuffer(2, 3, 4, 5, 6)
```

要从列表中删除多个元素，我们需要像这样使用`()`:

```
myArr -= (2, 4)myArrres8: scala.collection.mutable.ArrayBuffer[Int] = ArrayBuffer(3, 5, 6)
```

通过索引删除元素可以使用`.remove(x)`方法完成，只需用`x`输入想要删除的索引。您还可以向该方法传递一个范围来移除该范围内的索引:`.remove(0, 3)`将移除索引元素 0 和 2。

## 映射和过滤

![](img/6da31d4faa02b63e443e8a715ff65ea4.png)

[source](http://Photo by Tyler Nix on Unsplash)

我们经常想要过滤列表中的元素或者映射它们。在我们看这些之前，先看看我们如何在 Scala 中使用循环来遍历数组。

```
**for** (n <- myArr) {
  **println**(n)
}3 5 6
```

上面的代码将运行一个 for 循环，遍历数组中的每个元素，并打印出数组中的每个元素。`<-`的使用告诉 Scala 我们想要迭代`myArr` print `n`中的`myArr`和`for`每个`n`(元素)。缩进是不必要的，因为使用`{}`将表示代码块的开始和结束。

Mapping 将遍历列表中的每一项，并将其转换为不同的元素。如果我们想一次改变数组中的所有值，这很好。

这里我们将使用`.map()`方法将`myArr`中的每个元素乘以 5:

```
myArr.map**(**n => n * 5**)**res22: scala.collection.mutable.ArrayBuffer[Int] = ArrayBuffer(15, 25, 30)
```

如果满足我们设定的标准或条件，过滤将返回原始数据或数组的子集。很多时候，我们希望使用 filter 来获取数据或在数组或数据集中找到某些元素。我想把`myArr`过滤掉，这样它只会返回能被 2 整除的数字。

```
myArr.filter(n => n % 2 == 0)res26: scala.collection.mutable.ArrayBuffer[Int] = ArrayBuffer(6)
```

上面的代码迭代`myArr`，返回能被 2 整除的数(基本上是偶数)。我们还可以将映射和过滤结合在一起，以增加我们的列表并检查偶数。实际上，我会在`myArr`上随机添加一些数字，这样我们就可以让它变得有趣！

在这里，我们将多个元素追加到数组中:

```
myArr += **(**10, 3, 7, 5, 12, 20**)**res30: scala.collection.mutable.ArrayBuffer[Int] = ArrayBuffer(3, 5, 6, 10, 3, 7, 5, 12, 20)
```

现在我们将把映射和过滤结合起来返回`myArr`中的偶数:

```
myArr.map(n => n * 5).filter**(**n => n % 2 == 0**)**res31: scala.collection.mutable.ArrayBuffer[Int] = ArrayBuffer(30, 50, 60, 100)
```

我们也可以对奇数做同样的事情，把`n % 2 == 0`改成`n % 3 == 0`。

映射和过滤对于数据科学工作流至关重要，这项技术在我们每次处理数据集时都会用到。

Scala 是我们作为数据科学家的一个很好的工具。我们可以用它来处理数据和建立机器学习模型。对 Scala 的介绍只涵盖了最基本的内容。现在该由您来深入研究这种语言了。