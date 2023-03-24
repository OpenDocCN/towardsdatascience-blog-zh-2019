# 面向数据科学家的 3 个高级 Python 函数

> 原文：<https://towardsdatascience.com/3-advanced-python-functions-for-data-scientists-f869016da63a?source=collection_archive---------8----------------------->

Python 可以带来很多乐趣。重新发明一些你一开始不知道存在的内置函数并不是一件困难的事情，但是**为什么**你会想这么做呢？。今天，我们来看看其中的三个函数，我每天或多或少都会用到它们，但在我的数据科学职业生涯中，有很长一段时间我都没有意识到。

![](img/e1adfa77b6b21c56a155b0fe23141edf.png)

Photo by [Drew Beamer](https://unsplash.com/@drew_beamer?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

虽然它们可能不会节省大量的时间(*，如果你理解*背后的逻辑)，你的代码看起来会干净得多。也许对你来说这听起来没什么大不了的，但是未来你会感激的。

几周前，我发布了一篇关于一些**基本纯 Python 技能的文章，**，这篇文章涵盖了一些其他很酷的内置函数，所以一定要看看:

[](/3-essential-python-skills-for-data-scientists-b642a1397ae3) [## 数据科学家的 3 项基本 Python 技能

### 你不需要图书馆做所有的事情。纯 Python 有时候是绰绰有余的。

towardsdatascience.com](/3-essential-python-skills-for-data-scientists-b642a1397ae3) 

事不宜迟，就从第一个开始吧！

# 地图()

***map()*** 是一个内置的 Python 函数，用于将一个函数应用于一系列元素，如列表或字典。这可能是对数据进行某种操作的最干净、最易读的方式。

在下面的例子中，目标是对列表中的数字求平方。首先，必须声明这样做的函数，然后我将展示在有和没有 ***map()*** 函数的情况下如何实现，因此是以*非 python 化*和*python 化*的方式。

```
nums = [1, 2, 3, 4, 5]# this function will calculate square
def square_num(x): 
    return x**2 **# non-pythonic approach**
squares = []
for num in nums:
    squares.append(square_num(num))

print('Non-Pythonic Approach: ', squares) **# pythonic approach**
x = map(square_num, nums)
print('Pythonic Approach: ', list(x))
```

输出基本上是一样的，但是花一点时间来体会一下*python 式*方法看起来有多干净。也不需要循环。

# zip()

***zip()*** 是我的最爱之一。它使你能够同时遍历两个或更多的列表。这在处理日期和时间时会派上用场。

例如，当我有一个属性表示某个事件的开始时间，第二个属性表示该事件的结束时间时，我每天在工作中使用它。为了进一步分析，几乎总是需要计算它们之间的时间差，而目前为止最简单的方法就是使用 ***zip*** 来完成。

在这个例子中，我创建了两个包含数字的列表，任务是**对相应的元素**求和:

```
first = [1, 3, 8, 4, 9]
second = [2, 2, 7, 5, 8]# Iterate over two or more list at the same time
for x, y in zip(first, second):
    print(x + y)
```

如此简单干净。

# 过滤器()

***filter()*** 函数类似于 ***map()*** —它也将函数应用于某些序列，不同的是 ***filter()*** 将只返回那些被评估为 **True** 的元素。

在下面的例子中，我创建了一个任意的数字列表和一个函数，如果数字是偶数，该函数将返回 True。我将再次演示如何以*非 python 化*和*python 化*的方式执行操作。

```
numbers = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10]# Will return true if input number is even
def even(x):
    return x % 2 == 0**# non-pythonic approach**
even_nums = []
for num in numbers:
    if even(num):
        even_nums.append(num)

print('Non-Pythonic Approach: ', even_nums)**# pythonic approach**
even_n = filter(even, numbers)
print('Pythonic Approach: ', list(even_n))
```

再说一次，*python 式的*方式要干净得多，可读性也更好——这是你将来会喜欢的。

# 在你走之前

在 Python 中有更多类似于 3 的函数，但是我不认为它们在数据科学中有太多的适用性。实践这三条，当你在工作或大学中面临任何挑战时，记住它们。重新发明轮子太容易了，但是没有意义。

***你日常使用的内置函数有哪些？随意分享。***

*喜欢这篇文章吗？成为* [*中等会员*](https://medium.com/@radecicdario/membership) *继续无限制学习。如果你使用下面的链接，我会收到你的一部分会员费，不需要你额外付费。*

[](https://medium.com/@radecicdario/membership) [## 通过我的推荐链接加入 Medium-Dario rade ci

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

medium.com](https://medium.com/@radecicdario/membership)