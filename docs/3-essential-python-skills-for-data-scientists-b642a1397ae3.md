# 数据科学家的 3 项基本 Python 技能

> 原文：<https://towardsdatascience.com/3-essential-python-skills-for-data-scientists-b642a1397ae3?source=collection_archive---------9----------------------->

学习熊猫很棒。 [Numpy](https://numpy.org/) 也非常有趣。但是你可能很早就开始使用图书馆**了吗？**也许你还没有意识到 pure Python 所提供的一切。

如果这听起来像你，你会喜欢这篇文章。

![](img/b14bd031d94b375b687a095ca2a92696.png)

Photo by [fabio](https://unsplash.com/@fabioha?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

本文将介绍我在日常数据科学工作中最常用的一些非常酷的纯 Python 功能。我在整个数据准备阶段都在使用它们(*大量用于数据清理*)，甚至后来在绘图前汇总数据。

我希望你也能把这些融入到你的项目中。虽然没有运行时速度或性能方面的好处，但是当您从头开始实现这个逻辑时，您将节省大量时间。那么事不宜迟，让我们跳入第一点！

By [GIPHY](https://giphy.com/gifs/l0HlHFRbmaZtBRhXG/html5)

# λ函数

Lambda 函数非常强大。是的，当您必须以相同的方式清理多个列时，您不会使用它们——但这不是我经常遇到的事情——通常情况下，每个属性都需要自己的清理逻辑。

Lambda 函数允许你创建'*匿名*函数。这基本上意味着您可以快速创建特定的函数，而不需要使用 python***def***来正确定义函数。

话虽如此，请记住，lambdas 主要被设计成一行程序**——因此应该用于更简单的东西。对于更复杂的逻辑，你需要使用常规函数。**

好了，说的够多了，我现在给你看两个具体的例子，通过它们你可以看到在你的下一个项目中，仅仅通过不为每件事定义一个函数，你可以节省多少时间。第一个例子可能不会在现实世界中经常用到，但是值得一提。一切都是为了平方数字。

```
# regular function
def square_number(x):
    res = x ** 2
    return res# lambda function
square = lambda x: x ** 2# results
print('square_number(4): {}'.format(square_number(4)))
print('square lambda: {}'.format(square(4)))**>>> square_number(4): 16
>>> square lambda: 16**
```

上面的代码片段以常规方式和 lambda 方式包含了相同逻辑的实现。显然，结果是一样的，但是看看这个一行程序的美妙之处吧！

第二个例子将涵盖检查数字是否偶数的过程:

```
# regular function
def is_even(x):
    if x % 2 == 0:
        return True
    else:
        return False

# lambda function
even = lambda x: x % 2 == 0# results
print('is_even(4): {}'.format(is_even(4)))
print('is_even(3): {}'.format(is_even(3)))
print('even(4): {}'.format(even(4)))
print('even(3): {}'.format(even(3)))**>>> is_even(4): True
>>> is_even(3): False
>>> even(4): True
>>> even(3): False**
```

同样的逻辑以两种方式实现。你决定你喜欢哪一个。

# 列出理解

用最简单的方式解释，列表理解允许你使用不同的符号创建列表。你可以认为它本质上是一个建立在括号内的**单行 for 循环。**

在做特征工程的时候，我经常使用列表理解。例如，如果我正在分析垃圾邮件检测的电子邮件标题，我很好奇问号是否更经常出现在垃圾邮件中。这是一个用列表理解完成的非常简单的任务。

也就差不多了，不需要进一步的理论解释。例子是最重要的。

我选择声明一个常规函数，它将检查列表中以某个字符开始的条目，在本例中是' *a* 。一旦实现，我会做同样的事情，但用列表理解。*猜猜哪一个会写得更快。*

```
lst = ['Acer', 'Asus', 'Lenovo', 'HP']# regular function
def starts_with_a(lst):
    valids = []

    for word in lst:
        if word[0].lower() == 'a':
            valids.append(word)

    return valids

# list comprehension
lst_comp = [word for word in lst if word[0].lower() == 'a']# results
print('starts_with_a: {}'.format(starts_with_a(lst)))
print('list_comprehension: {}'.format(lst_comp))**>>> starts_with_a: ['Acer', 'Asus']
>>> list_comprehension: ['Acer', 'Asus']**
```

如果你第一次看到这个，语法可能会有点混乱。但是当你每天写它们的时候，它们开始吸引你，让你看看你能把多少复杂性放进去。

# 活力

这是我在实践中很少见到的许多内置 Python 方法之一。从数据科学家的角度来看，它使您能够同时**迭代两个或更多列表**。这在处理日期和时间时会派上用场。

例如，当我有一个属性表示某个事件的开始时间，第二个属性表示该事件的结束时间时，我每天在工作中使用它。为了进一步分析，几乎总是需要计算它们之间的时间差，而 ***zip*** 是迄今为止最简单的方法。

例如，我决定比较一些虚构公司和虚构地区的一周销售数据:

```
sales_north = [350, 287, 550, 891, 241, 653, 882]
sales_south = [551, 254, 901, 776, 105, 502, 976]for s1, s2 in zip(sales_north, sales_south):
    print(s1 — s2)>>> -201
    33
    -351
    115
    136
    151
    -94
```

看看这有多简单。您可以应用相同的逻辑来同时迭代 3 个数组，您只需要在括号中添加' *s3* '和其他一些列表名。

# 最后的话

纯 Python 就是这么厉害。确保你知道它的能力。你不需要一个专门的库来处理所有的事情。我的意思是这很有帮助，但是**这会让你成为更好的程序员**。

练习这些技能，掌握它们，并把它们应用到你的日常工作中，不管是为了娱乐，为了大学，还是为了工作。你不会后悔的。

***你有什么想法？你认为纯 Python 之外的东西对数据科学家来说是必不可少的吗？让我知道。***

*喜欢这篇文章吗？成为* [*中等会员*](https://medium.com/@radecicdario/membership) *继续无限制学习。如果你使用下面的链接，我会收到你的一部分会员费，不需要你额外付费。*

[](https://medium.com/@radecicdario/membership) [## 通过我的推荐链接加入 Medium-Dario rade ci

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

medium.com](https://medium.com/@radecicdario/membership)