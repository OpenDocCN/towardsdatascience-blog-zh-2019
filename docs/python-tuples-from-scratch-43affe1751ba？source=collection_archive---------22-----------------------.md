# Python 元组从零开始！！！

> 原文：<https://towardsdatascience.com/python-tuples-from-scratch-43affe1751ba?source=collection_archive---------22----------------------->

## 让我们了解一下 python 元组的基本概念。

![](img/5a9741ab3a047bd13a3ee8d13965a9e7.png)

Image credits: [GangBoard](https://www.gangboard.com/blog/wp-content/uploads/2019/05/Tuples-in-Python.jpg)

在开始之前，我想告诉你，我将以问答的形式来写这篇教程，因为它会有所帮助，尤其是在面试的时候。在整个教程中，我将只研究一个例子来保持一致性。本教程的完整代码可以在我的 [GitHub](https://github.com/Tanu-N-Prabhu/Python/blob/master/Tuples/%20Tuples.ipynb) 页面上找到。

## 1)什么是元组，为什么使用元组？

元组是一种数据结构，用于存储异构数据，即不同类型的数据。一个元组用于将相关数据分组在一起，比如狗的名字、身高、体重、品种和颜色(我喜欢狗)。一个元组包含几个用逗号分隔的值。

## 2)如何创建元组？

可以通过将值放在一对括号内并用逗号分隔来创建元组，例如:

**让我们创建一个称为“元组”的元组，并放入一些值，包括字符串和整数数据。这可以如下所示:**

```
**# Creating a tuple and storing the values**tuple = (“Today’s”, “Date”, “Is”, 15, “August”, 2019)
print(tuple)**(“Today’s”, ‘Date’, ‘Is’, 15, ‘August’, 2019)**print(type(tuple))**<class 'tuple'>**
```

如上所述，一个元组可以保存异构数据(因此证明:)

## 3)我们如何访问一个元组的值？

索引可以做到这一点。元组的值可以通过它们的索引来获取。要做到这一点，您只需将数字(索引值)和元组的名称一起放在一对方括号中。

```
**# Accessing the values of a tuple**print(tuple[2])**'Is'**print(tuple[5])**2019**
```

## 4)如何嵌套或合并两个元组？

只要将旧元组放置在由逗号分隔的新创建的元组旁边，就可以实现两个或更多元组的嵌套。

```
**# Nesting two tuples as one tuple**tuple1 = (“And”, “its”, “Thursday”)
print(tuple1)**(‘And’, ‘its’, ‘Thursday’)**nest = tuple, tuple1
print(nest)**((“Today’s”, ‘Date’, ‘Is’, 15, ‘August’, 2019), (‘And’, ‘its’, ‘Thursday’))**
```

## 5)元组是不可变的还是可变的？

在回答这个问题之前，你应该知道不可变的值是不可变的，可变的值是可变的。现在让我们来回答这个问题，答案是元组是不可变的，是的，一旦元组被创建，我们就永远不能改变它们的值。不信，我证明给你看。

```
**# Tuples are immutable in nature**print(tuple[0])**“Today’s”****# Changing the value of the 0th index to "Hi"**tuple[0] = ("Hi")
print(tuple)**--------------------------------------------------------------------****TypeError: Traceback (most recent call last)**[**<ipython-input-12-18297fa5df7e>**](/<ipython-input-12-18297fa5df7e>) **in <module>()
----> 1 tuple[0] = ("Hi")
      2 tuple****TypeError: 'tuple' object does not support item assignment**
```

因此，以上证明了元组在本质上是不可变的。

## **6)元组能在其中存储相同的数据吗？**

是的，元组可以在其中存储相同的数据，我们可以在一个元组中存储许多相同的值。例如:

```
**# Storing identical data with a tuple**tuple = (“Today’s”, “Date”, “Is”, 15, “August”, 2019)
print(tuple)**(“Today’s”, ‘Date’, ‘Is’, 15, ‘August’, 2019)**tuple = [(“Today’s”, “Date”, “Is”, 15, “August”, 2019), (“Today’s”, “Date”, “Is”, 15, “August”, 2019)]
print(tuple)**[(“Today’s”, ‘Date’, ‘Is’, 15, ‘August’, 2019), (“Today’s”, ‘Date’, ‘Is’, 15, ‘August’, 2019)]**
```

## 如何循环遍历一个元组？

这个问题很直接，使用循环结构，我们可以循环遍历一个元组。下面我将使用 for 循环并遍历元组中的值，您可以类似地使用其他循环结构并获得结果。

```
tuple = (“Today’s”, “Date”, “Is”, 15, “August”, 2019)
print(tuple)**(“Today’s”, ‘Date’, ‘Is’, 15, ‘August’, 2019)**for i in tuple:
   print(i)**Today’s 
Date 
Is 
15 
August 
2019**
```

## 8)如何使用循环访问元组的索引？

我们可以使用带有枚举函数的 for 循环来实现这一点。枚举是 Python 的内置函数。它允许我们循环一些东西，并有一个自动计数器，了解更多关于枚举的信息，阅读它的完整文档[这里](http://book.pythontips.com/en/latest/enumerate.html)。例如:

```
**# Accessing the index of the tuple using enumerate function.**tuple = (“Today’s”, “Date”, “Is”, 15, “August”, 2019)
print(tuple)**(“Today’s”, ‘Date’, ‘Is’, 15, ‘August’, 2019)**for counter, value in enumerate(tuple):
    print(counter, value)**0 Today's 
1 Date 
2 Is 
3 15 
4 August 
5 2019**
```

## 我们能从元组中移除值或项吗？

我想现在你可以轻松回答这个问题了。答案是否定的，不能从元组中删除值或项，但是可以完全删除元组。这是因为元组是不可变的。例如:

```
**# Deleting an entire tuple using del**tuple = (“Today’s”, “Date”, “Is”, 15, “August”, 2019)
tuple**(“Today’s”, ‘Date’, ‘Is’, 15, ‘August’, 2019)**del tuple
print(tuple)**<class 'tuple'>**
```

如上所述，可以使用 del 删除整个元组，所以当我们打印元组时，可以看到其中没有任何元素。

## 10)如何统计值在元组中出现的次数？

这可以通过使用元组的 count 方法来完成，count 方法返回一个值在元组中出现的次数。例如:

```
**# Counting the number of times a value has appeared in the tuple**tuple = (“Today’s”, “Date”, “Is”, 15, “August”, 2019, “And”, “Day”, “Is”, “Thursday”)
print(tuple)**("Today's", 'Date', 'Is', 15, 'August', 2019, 'And', 'Day', 'Is', 'Thursday')**print(tuple.count("Is"))
**2**print(tuple.count(15))
**1**
```

所以在上面的例子中，值“Is”出现了 2 次，类似地，15 出现了 1 次。

## 11)如何获取元组中某个值的索引？

这可以通过使用元组的索引方法来完成，索引方法搜索值的第一次出现，并返回它的位置。例如:

```
**# Counting the number of times a value has appeared in the tuple**tuple = (“Today’s”, “Date”, “Is”, 15, “August”, 2019, “And”, “Day”, “Is”, “Thursday”)
print(tuple)**("Today's", 'Date', 'Is', 15, 'August', 2019, 'And', 'Day', 'Is', 'Thursday')**print(tuple.index("Date"))
**1**print(tuple.index("August"))
**4**print(tuple.index("Is"))
**2**
```

如上所述，index 方法返回值的位置，但是在“是”的最后一种情况下，index 方法返回元组中第一次出现的“是”并返回其位置。

## 13)如何检查值是否存在于元组中？

这可以使用 if 语句中的关键字中的**来检查。例如，让我们检查 tuple 中是否存在 August。**

```
**# Checking for the values present in the tuple**tuple = (“Today’s”, “Date”, “Is”, 15, “August”, 2019)
print(tuple)**("Today's", 'Date', 'Is', 15, 'August', 2019)****# Case 1:**if "August" in tuple:
    print(True)
    print(tuple.index("August"))
else:
    print(False)**True
4**--------------------------------------------------------------------**#** **Case 2:**if "September" in tuple:
    print(True)
    print(tuple.index("September"))
else:
    print(False)**False**
```

如上所述,“八月”出现在元组的位置 4，但是“九月”不出现在元组中。

因此，以上是以问答格式编写的 Python 中元组的非常重要的技术或功能(通常在编程面试中非常有用)。我引用了一些来自 [Python 元组](https://docs.python.org/3/tutorial/datastructures.html)的例子。我以一种简单的方式编写了本教程，这样每个人都可以理解和掌握 Python 中元组的概念，而无需事先具备编程知识或经验。如果你们对代码有什么疑问，评论区就是你们的了。

## 谢谢你。