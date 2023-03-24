# 理解最大似然估计

> 原文：<https://towardsdatascience.com/understanding-maximum-likelihood-estimation-fa495a03017a?source=collection_archive---------12----------------------->

 [## 想在数据科学方面变得更好吗？

### 当我在我发布独家帖子的媒体和个人网站上发布新内容时，请单击此处获得通知。](https://bobbywlindsey.ck.page/5dca5d4310) 

假设您从某个发行版收集了一些数据。正如你可能知道的，每个分布只是一个有一些输入的函数。如果您改变这些输入的值，输出也会改变(如果您用不同的输入集绘制分布图，您可以清楚地看到这一点)。

碰巧您收集的数据是具有一组特定输入的分布的输出。最大似然估计(MLE)的目标是估计产生数据的输入值。有点像逆向工程你的数据是从哪里来的。

实际上，你并不真的对数据进行采样来估计参数，而是从理论上求解；分布的每个参数都有自己的函数，该函数是参数的估计值。

# 这是怎么做到的

首先，假设数据的分布。例如，如果您正在观看 YouTube 并跟踪哪些视频有 clickbaity 标题，哪些没有，您可能会假设二项分布。

接下来，来自这个分布的“样本”数据，你仍然不知道它的输入。请记住，您是在理论上解决这个问题，所以不需要实际获取数据，因为样本数据的值在下面的推导中并不重要。

现在问，得到你得到的样本的可能性有多大？这个可能性就是得到你的样本的概率。假设每个样本彼此独立，我们可以将似然函数定义为:

![](img/4685409f56a40059a5c93fffbdaca2cd.png)

Likelihood function

现在您已经有了可能性函数，您希望找到使可能性最大化的分布参数值。这样思考这个问题可能会有帮助:

![](img/f41794ee42029904bbb5f786c6797c41.png)

What you’re trying to do with MLE

如果你熟悉微积分，找到一个函数的最大值需要对它求微分，并使它等于零。如果你真的对函数的 log 求导，这会使求导更容易，你会得到相同的最大值。

一旦你区分了对数似然，就可以求解参数了。如果您正在查看伯努利、二项式或泊松分布，您将只有一个参数需要求解。高斯分布将有两个，等等…

# YouTube 视图建模

假设你一年前创办了一个 YouTube 频道。到目前为止，你做得很好，并且收集了一些数据。你想知道在给定的一段时间内，至少有 *x* 个访问者访问你的频道的概率。分布中最明显的选择是泊松分布，它只依赖于一个参数λ，λ是每个区间出现的平均次数。我们想用最大似然估计来估计这个参数。

我们从泊松分布的似然函数开始:

![](img/86aaf133fdc35dbd0e3ca1215533e1d6.png)

Likelihood function for Poisson distribution

现在看看它的日志:

![](img/2711f8d3ac40e32b511a3bf5a84338b5.png)

Log likelihood for Poisson distribution

然后求微分，把整件事设为 0:

![](img/d8c79a0d13e14ad23d3fb2c33e2a79bf.png)

Finding the maximum of the log likelihood for Poisson distribution

现在你有了一个λ的函数，只要插入你的数据，你就会得到一个实际值。然后，您可以将此λ值用作泊松分布的输入，以便对一段时间内的收视率进行建模。很酷吧。

# 最大化积极因素和最小化消极因素是一样的

现在从数学上来说，最大化对数似然和最小化负对数似然是一样的。我们可以通过类似上面的推导来说明这一点:

取负对数可能性:

![](img/3e4ff3b4ce4fca0b6537dd6240d92008.png)

Negative log likelihood for Poisson distribution

然后求微分，把整件事设为 0:

![](img/6cfa7f9de1a7f1b1df8458cb6cb0ddc6.png)

Finding the maximum of the negative log likelihood for Poisson distribution

现在，是最大化对数似然还是最小化负对数似然取决于你。但是通常你会发现对数似然的最大化更常见。

# 结论

现在你知道如何使用最大似然估计了！概括地说，您只需要:

1.  求对数可能性
2.  区分它
3.  将结果设置为零
4.  然后求解你的参数

*原载于 2019 年 11 月 6 日*[*【https://www.bobbywlindsey.com】*](https://www.bobbywlindsey.com/2019/11/06/understanding-maximum-likelihood-estimation/)*。*