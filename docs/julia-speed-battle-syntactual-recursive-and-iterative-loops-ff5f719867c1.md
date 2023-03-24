# 朱莉娅速度战:合成、递归和迭代循环

> 原文：<https://towardsdatascience.com/julia-speed-battle-syntactual-recursive-and-iterative-loops-ff5f719867c1?source=collection_archive---------20----------------------->

![](img/e05eef026518651c333a05850d6d9f46.png)

随着我为机器学习模块[车床](http://lathe.emmettboudreau.com/)所做的所有写作，以及我在 0.0.4 中实现的新算法，我的思想一直集中在优化上，以真正为快速的机器学习设置标杆。几周前，在做一个时间敏感的项目时，我试图用 Python 从一个巨大的数据集构建一个算法，Jupyter 内核会因为试图读取它而崩溃。另一方面，我想到了我的 Julia 经验，甚至不需要使用 Scala，我就可以很好地读取超过 3500 万次观察的数据集，并且使用 Lathe 最多只需要几秒钟就可以做出预测。这是一个很难理解的概念，与其说我遇到了运行的硬件的限制，不如说我遇到了语言的限制。

这让我想到了低级算法以及它们在循环形式的计算速度中的位置。经过深思熟虑，我决定测试众所周知的关于处理和语言的计算机科学思想的有效性和严重性，特别是在 Julia 中。我们都知道朱莉娅不遵循任何人的规则，在这方面她确实是一种独特的语言。我有一篇文章[在这里](/things-to-know-before-using-julia-for-machine-learning-487744c0b9b2)关于你在使用它之前应该知道的关于朱莉娅的事情，你可以查看更多信息。

# 假设

在早期的计算领域，迭代通常被认为是处理和个性化数组中的索引的最快的循环类型。考虑到这一点，由于明显的原因(mb 内存对 gb 内存),许多较老的系统使用递归传输的堆栈比现代计算机少，然而许多 80 年代的遗留语言仍然更适合于在语法循环和递归循环上进行迭代。还应该注意的是，递归经常不被接受，被认为是完成快速任务最慢的方式。至于语法表达式，通常结果完全取决于你的语言及其编译器。

![](img/631f2b4fb67fc917d95457808517be9f.png)

在那个特别的音符上；我将向您介绍 Julia 的(股票)编译器:

> 提前(AOT)

考虑到 AOT，我们可以假设许多其他编译器和语言作为一个整体可能不会产生相同的结果，甚至可能产生完全不同的结果。我想指出这一点，因为我个人被这个测试的结果带了回来，它真的让我对他们如何能够操纵语言和编译器到如此程度以使 Julia 成为如此甜美的语言感兴趣。我想那是麻省理工学院给你的。

# 设置我们的测试

我们的数据集将是我在[这个项目](/deep-sea-correlation-27245959ccfa)中使用的数据集，因为它非常大。数据清理后的形状是 864，863 行× 3 列，这对于一个像样的速度测试来说应该足够了。

我将使用线性回归函数作为控制来进行速度测试，唯一要做的更改是循环(最多一行)和循环本身所必需的部分。线性回归的公式为:

y^ = a+(b*x^)哪里

a =(∑y)(∑x)-(∑x)(∑xy))/(n(∑x)-(∑x))

和
b =(x(∑xy)—(∑x)(∑y))/n(∑x)—(∑x)

因此，让我们编写一个简单线性回归函数的迭代版本:

```
function pred_LinearRegression(x,y,xt)
    # a = ((∑y)(∑x^2)-(∑x)(∑xy)) / (n(∑x^2) - (∑x)^2)
    # b = (x(∑xy) - (∑x)(∑y)) / n(∑x^2) - (∑x)^2
    # Get our Summatations:
    Σx = sum(x)
    Σy = sum(y)
    # dot x and y
    xy = x .* y
    # ∑dot x and y
    Σxy = sum(xy)
    # dotsquare x
    x2 = x .^ 2
    # ∑ dotsquare x
    Σx2 = sum(x2)
    # n = sample size
    n = length(x)
    # Calculate a
    a = (((Σy) * (Σx2)) - ((Σx * (Σxy)))) / ((n * (Σx2))-(Σx^2))
    # Calculate b
    b = ((n*(Σxy)) - (Σx * Σy)) / ((n * (Σx2)) - (Σx ^ 2))
    # Empty array:
    ypred = []
    for i in xt
        yp = a+(b*i)
        append!(ypred,yp)
    end
    return(ypred)
end
```

我们将使用 Julia 的计时器来为我们的预测计时。我认为这将是理想的方式，因为我们可以消除任何中间件，因为我们可以在没有任何额外开销计时器的情况下计算值。

```
[@time](http://twitter.com/time) pred_LinearRegression(trainX,trainy,testX)
```

现在我们将对一个递归循环做同样的事情:

```
function pred_LinearRegression(m,xt)
    # a = ((∑y)(∑x^2)-(∑x)(∑xy)) / (n(∑x^2) - (∑x)^2)
    # b = (x(∑xy) - (∑x)(∑y)) / n(∑x^2) - (∑x)^2
    # Get our Summatations:
    x = m.x
    y = m.y
    Σx = sum(x)
    Σy = sum(y)
    # dot x and y
    xy = x .* y
    # ∑dot x and y
    Σxy = sum(xy)
    # dotsquare x
    x2 = x .^ 2
    # ∑ dotsquare x
    Σx2 = sum(x2)
    # n = sample size
    n = length(x)
    # Calculate a
    a = (((Σy) * (Σx2)) - ((Σx * (Σxy)))) / ((n * (Σx2))-(Σx^2))
    # Calculate b
    b = ((n*(Σxy)) - (Σx * Σy)) / ((n * (Σx2)) - (Σx ^ 2))
    [i = a+(b*i) for i in xt]
    return(xt)
end
```

实际上:

```
function pred_LinearRegression(m,xt)
    # a = ((∑y)(∑x^2)-(∑x)(∑xy)) / (n(∑x^2) - (∑x)^2)
    # b = (x(∑xy) - (∑x)(∑y)) / n(∑x^2) - (∑x)^2
    # Get our Summatations:
    x = m.x
    y = m.y
    Σx = sum(x)
    Σy = sum(y)
    # dot x and y
    xy = x .* y
    # ∑dot x and y
    Σxy = sum(xy)
    # dotsquare x
    x2 = x .^ 2
    # ∑ dotsquare x
    Σx2 = sum(x2)
    # n = sample size
    n = length(x)
    # Calculate a
    a = (((Σy) * (Σx2)) - ((Σx * (Σxy)))) / ((n * (Σx2))-(Σx^2))
    # Calculate b
    b = ((n*(Σxy)) - (Σx * Σy)) / ((n * (Σx2)) - (Σx ^ 2))
    f(X) = a.+b.*X
    ypred = f(xt)
    return(ypred)
end
```

# 结果

这些结果可能会让你震惊:

1.  平均值为 0.114379399999999999 秒的递归
2.  平均为 0.1685546 秒
3.  平均值为 0.4208722 秒的迭代(用于循环)

> 迭代是最后一次！？

最重要的是，我认为这确实说明了 Julia 非常适合作为函数式语言。递归是函数式编程的一个奇妙的构建块，并且是开创这一概念的基本算法之一。

有了这些结果，我想最小化任何不受控制的变量，所以我的第一个目标是内存，因为递归非常依赖内存。

![](img/f396ca0b5cf8b9a07f08e38e59533f4f.png)

如左边的系统监视器所示，我们当然没有处理器速度或内存不足的问题，资源远远没有被过度使用。

因此，我决定做的下一件事是拓宽频谱，看看较慢的运行时间是否对结果有显著影响，为此，我将处理单元的时钟频率降低到 1824 MHz，并将调节器设置为仅允许使用两个内核。这说起来容易做起来难，因为完成所有的数据计算需要相当长的时间(太慢了。)我最初选择单核，但那对我来说太远了，感觉像是在 90 年代等待二十分钟来切换窗口。

![](img/71bccacdea4a20191d30f41a0000c302.png)

这产生了明显更高的速度，但趋势完全继续，这没有对结果产生影响…

1.  速度为. 33 的递归
2.  速度为 0.39 的合成事实
3.  速度为 1.37 的迭代

# 概观

没错，差异实际上非常显著，考虑到这一点，从现在开始，我一定会尽可能地在 Julia 中使用递归！我不知道这种对速度的渴望来自哪里，但我对朱莉娅轻松处理如此大量数据的能力印象深刻。我真的很喜欢做这些测试，因为结果使我成为一个更好的程序员，可以更有效地操作语言和编译器。希望这对于在 Julia 工作的任何其他数据科学家来说也是非常有益的，我知道从现在开始我肯定会将这些结果应用到我的所有功能中。