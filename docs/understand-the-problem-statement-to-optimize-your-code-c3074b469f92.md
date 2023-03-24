# O(n)是改进你的算法的唯一方法吗？

> 原文：<https://towardsdatascience.com/understand-the-problem-statement-to-optimize-your-code-c3074b469f92?source=collection_archive---------19----------------------->

## [蟒蛇短裤](https://towardsdatascience.com/tagged/python-shorts)

## 理解问题陈述如何帮助您优化代码

![](img/92ae75282f038c496c901af81841855d.png)

Photo by [Patrick Tomasso](https://unsplash.com/@impatrickt?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

每当我们谈论优化代码时，我们总是讨论代码的计算复杂性。

是 O(n)还是 O(n 的平方)？

***但是，有时候我们需要超越算法，看看算法在现实世界中是如何被使用的。***

在这篇文章中，我将谈谈类似的情况以及我是如何解决的。

# 问题是:

最近我在处理一个 NLP 问题，不得不从文本中清除数字。

从表面上看，这似乎是一个微不足道的问题。一个人可以用许多方法做到这一点。

我用下面给出的方法做了这件事。该函数将问题字符串中的数字替换为#s。

```
def clean_numbers(x):    
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x
```

这些都是干净的代码，但是我们能优化它的运行时间吗？在继续下一步之前思考。

# 诀窍是:

![](img/f14e727d7c6fdc47fde500b84ff16d17.png)

优化的诀窍在于理解代码的目标。

虽然我们大多数人都理解 O(n)范式，并且总是可以去寻找更好的算法，但优化代码的一个被遗忘的方法是回答一个问题:

***这段代码将如何使用？***

你知道吗，如果我们在替换字符串之前只检查一个数字，我们可以获得 10 倍的运行时间改进！！！

由于大多数字符串没有数字，我们不会评估对`re.sub.`的 4 次函数调用

它只需要一个简单的 if 语句。

```
def clean_numbers(x):
    if bool(re.search(r'\d', x)):
        x = re.sub('[0-9]{5,}', '#####', x)
        x = re.sub('[0-9]{4}', '####', x)
        x = re.sub('[0-9]{3}', '###', x)
        x = re.sub('[0-9]{2}', '##', x)
    return x
```

***这样做有什么必要？***

对我们使用的数据的理解和对问题陈述的理解。

这些改进当然是隐藏的，你可能需要考虑这段代码要运行多少次？或者谁会触发功能？做这样的改进。

但这是值得的。

> 把你写的每一个函数都当成你的孩子，尽你所能去改进它。

我们可以使用 Jupyter 笔记本中的`%%timeit`魔法功能来查看性能结果。

```
%%timeit
clean_numbers(“This is a string with 99 and 100 in it”)
clean_numbers(“This is a string without any numbers”)
```

使用第一个函数调用上述简单函数的结果是:

```
17.2 µs ± 332 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
```

使用第二个函数:

```
12.2 µs ± 324 ns per loop (mean ± std. dev. of 7 runs, 100000 loops each)
```

***即使我们 50%的句子都有数字，我们还是提高了两倍。***

由于我们在数据库中不会有那么多包含数字的句子，我们可以这样做并得到很好的改进。

# 结论

有不同的方法来改进一个算法，甚至上面的问题可以用一些我想不到的逻辑来进一步改进。

但我想说的是，在封闭的盒子里思考并想出解决方案永远不会帮助你。

如果您理解了问题陈述和您试图实现的目标，您将能够编写更好、更优化的代码。

所以，下次你的老板让你降低算法的计算复杂度时，告诉他，虽然你无法降低复杂度，但你可以使用一些定制的业务逻辑让算法变得更快。我猜他会印象深刻。

另外，如果你想了解更多关于 Python 3 的知识，我想向你推荐一门来自密歇根大学的关于学习 T2 中级 Python 3 的优秀课程。一定要去看看。

将来我也会写更多初学者友好的帖子。让我知道你对这个系列的看法。在 [**媒体**](https://medium.com/@rahul_agarwal) 关注我，或者订阅我的 [**博客**](https://mlwhiz.com/) 了解他们。

一如既往，我欢迎反馈和建设性的批评，可以通过 Twitter [@mlwhiz](https://twitter.com/MLWhiz) 联系到我。