# 卷积层中的反向传播

> 原文：<https://towardsdatascience.com/backpropagation-in-a-convolutional-layer-24c8d64d8509?source=collection_archive---------0----------------------->

![](img/df9b37b2fb0d77182930962adaa556ed.png)

Backpropagation in a convolutional layer

# 介绍

## 动机

这篇文章的目的是详细说明梯度反向传播是如何在神经网络的卷积层中工作的。典型地，该层的输出将是所选激活功能的输入(例如`relu`)。我们假设给定了从该激活函数反向传播的梯度`dy`。因为我无法在网上找到一个完整的，详细的，和“简单”的解释它是如何工作的。我决定做数学，试图在归纳之前一步一步地理解简单的例子是如何工作的。在进一步阅读之前，你应该熟悉神经网络，尤其是计算图形中梯度的前向传递、后向传播和带有张量的基本线性代数。

![](img/2d6fda8ae02ee5a4dc705105fcc13650.png)

Convolution layer — Forward pass & BP

## 记号

`*`在神经网络的情况下是指 2 个张量的卷积(一个输入`x`和一个滤波器`w`)。

*   当`x`和`w`为矩阵时:
*   如果`x`和`w`共享相同的形状，`x*w`将是一个标量，等于数组之间的元素级乘法结果之和。
*   如果`w`小于`x`，我们将获得一个激活图`y`，其中每个值是 x 的一个子区域与 w 的大小的预定义卷积运算。由滤波器激活的这个子区域在输入阵列`x`上滑动。
*   如果`x`和`w`有 2 个以上的维度，我们考虑将后 3 个维度用于卷积，后 2 个维度用于高亮显示的滑动区域(我们只是给矩阵增加了一个深度)

符号和变量与[斯坦福优秀课程](http://cs231n.stanford.edu/)中关于视觉识别的卷积神经网络中使用的符号和变量相同，尤其是[作业 2](http://cs231n.github.io/assignments2019/assignment2/) 中的符号和变量。关于卷积层和前向传递的细节将在这个[视频](https://www.youtube.com/watch?v=bNb2fEVKeEo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=5)和前向传递[帖子](https://neodelphis.github.io/convnet/python/2019/07/02/convnet-forward-pass.html)的一个简单实现的实例中找到。

![](img/283ef1edb53afd51ce1817ceac9175a9.png)

Convolution layer notations

## 目标

我们的目标是找出梯度是如何在卷积层中向后传播的。向前传球是这样定义的:

输入由 N 个数据点组成，每个数据点有 C 个通道，高 H，宽 w，我们用 F 个不同的滤波器对每个输入进行卷积，其中每个滤波器跨越所有 C 个通道，高 HH，宽 WW。

输入:

*   x:形状(N，C，H，W)的输入数据
*   w:形状的过滤器权重(F，C，HH，WW)
*   b:形状为(F，)的偏差
*   conv _ 参数:带有以下键的字典:
*   “步幅”:水平和垂直方向上相邻感受野之间的像素数。
*   ' pad ':将用于对输入进行零填充的像素数。

在填充过程中，“填充”零应沿着输入的高度和宽度轴对称放置(即两边相等)。

返回一个元组:

*   out:形状为(N，F，H’，W’)的输出数据，其中 H’和 W’由下式给出

H' = 1 + (H + 2 * pad — HH) /步幅

W' = 1 + (W + 2 * pad — WW) /步幅

*   缓存:(x，w，b，conv 参数)

# 前进传球

## 一般情况(简化为 N=1，C=1，F=1)

N=1 个输入，C=1 个通道，F=1 个滤波器。

![](img/b19eb2f746cc8a983f254b1c6d368f53.png)

Convolution 2D

x:H×W
x′=带填充的 x
W:hh×WW
b 偏移:标量
y:H′×W′
步幅 s

![](img/6c22142cd891eb89e525dea70ff00b67.png)

## 具体情况:stride=1，pad=0，无偏差。

![](img/17e1eed2dc16e0d4c95c512ec70a1770.png)

# 反向传播

我们知道:

![](img/0e9caefe17fad7f0a7e898fedc0f6060.png)

我们要计算成本函数 l 的偏导数 *dx* 、 *dw* 和 *db* ，我们假设这个函数的梯度已经反向传播到 y。

# 平凡的例子:输入 x 是一个向量(一维)

我们正在寻找一种直觉，它是如何在一个简单的设置上工作的，稍后我们将尝试概括。

## 投入

![](img/aecc3fc9d8cac0ae1555a81c689af244.png)

## 输出

![](img/a8eac6cdf03307c695b48abf19882da8.png)

## 前向通过—与一个滤波器卷积，步长= 1，填充= 0

![](img/60a90d4d2f73877a4869ecbd4f22b604.png)

## 反向传播

我们知道成本函数 L 相对于 y 的梯度:

![](img/e786ddaa95e5755d95d8191664c76904.png)

这可以用雅可比符号来表示:

![](img/b0dbf52ba8c587d2dc8be15d770153fa.png)

dy 和 y 具有相同的形状:

![](img/9f0799bbb69a1f669ad430e8b08fddfd.png)

我们正在寻找

![](img/78f3ef9331287df3c5e550b8f397fe8f.png)

## 

![](img/e6f8613781235a6fc6a8d9e6d3ebfe58.png)

使用链式法则和正向传递公式(1)，我们可以写出:

![](img/023e7771651a136db702f8d59a105f17.png)

## 发展的宽度（Developed Width 的缩写）

![](img/964a6eb8f33029923bab263605e1a962.png)![](img/c39b4c1388d736c8bae6f80f9dadd872.png)![](img/b31b7ae537e10c3fb515634021aa70d8.png)![](img/9c14a0687c0c40a40b42e7fb023c52dd.png)![](img/4d6abbb4db1517f40eb9a22a3da7e9f6.png)

我们可以注意到，dw 是输入 x 与滤波器 dy 的卷积。让我们看看它在增加了一个维度后是否仍然有效。

![](img/43290a0ca67418f41a91a83d11ec0d26.png)

## 高级的（deluxe 的简写）

![](img/f1517f1f40994918ff47afc7b0a80694.png)![](img/d740dc72f8013fe74e8da53ed5e881ad.png)![](img/d93a4f96f887dc8c0121783449311f1d.png)![](img/0b441f8a120db0f1a74ddf3855311b74.png)

再一次，我们有一个卷积。这次有点复杂。我们应该考虑一个输入 dy，其大小为 1 的 0 填充与一个“反向”滤波器 w 卷积，如( *w* 2， *w* 1)

![](img/28a5f23a1b961d5c882d7b067bddc76b.png)

下一步将是看看它如何在小矩阵上工作。

# 输入 x 是一个矩阵(二维)

## 投入

![](img/775aa5ea1e433a298b1308068d47bdff.png)

## 输出

我们将再次选择最简单的情况:stride = 1 并且没有填充。y 的形状将是(3，3)

![](img/d1917ebbe53188c52d90f919b574868b.png)

## 向前传球

我们将拥有:

![](img/0fb3e8e77300abde2cbfaf33cc068b26.png)

用下标写的:

![](img/90ac16ce97327f74e26eb9d286b68ac1.png)

## 反向传播

我们知道:

![](img/3034601037fe59b0ba85e3f3d9cf2d4a.png)

## 

使用爱因斯坦约定来简化公式(当一个指数变量在乘法中出现两次时，它意味着该指数的所有值的总和)

![](img/5614610f11ebe6125036d69407c14edb.png)

对 I 和 j 求和，我们得到:

![](img/997c56adc43e493875250e301193b425.png)

## 发展的宽度（Developed Width 的缩写）

![](img/d2eff9da13e3f753f5fb13b07398c971.png)![](img/8d09e7875cdabbab2b956bc5a2c6b3b4.png)

我们正在寻找

![](img/11ef622c8e702c2d09b49c2a063a455e.png)

使用公式(4 ),我们得到:

![](img/f0ac7313484fbc39305e77bfd4166c5e.png)

所有术语

![](img/0c84e94d746e39927a6be6fc30d91904.png)

除了( *k* ， *l* )=( *m* ， *n* )为 1 的情况，double sum 中只出现一次。因此:

![](img/3b7e82d04398fb6eeb4a937c3536e9f2.png)

使用公式(3 ),我们现在有:

![](img/c4c07693b91aca9b7082124f685eb5ec.png)

如果我们将该等式与给出卷积结果的公式(1)进行比较，我们可以区分类似的模式，其中 dy 是应用于输入 x 的滤波器。

![](img/cd5a7f4ab260f3447a3c62c22e972df4.png)

## 高级的（deluxe 的简写）

使用我们对(5)所做的链式法则，我们有:

![](img/256276922b07f1d25a362989d7f2ce2c.png)

这一次，我们要寻找

![](img/e58fda2ba6e2414d6c1633f589c7bcb9.png)

使用等式(4):

![](img/e87f605c9c5b2165e4c73a3b544badbf.png)

我们现在有:

![](img/34ea7eddca4fb69da389101a4e45f906.png)

在我们的示例中，索引的范围集是:

![](img/2058d0c08b2481213394e66798236f95.png)

当我们设置*k*=*m*-*I*+1 时，我们将超出定义的边界:(*m*-*I*+1)∈[1，4]

为了保持上述公式的可信度，一旦指数超出定义的范围，我们选择用 0 值扩展矩阵 *w* 的定义。

在二重和中，我们只有一次 x 的偏导数等于 1。所以:

![](img/5e6368d5f1127ce5145d972c16f03b97.png)

其中 *w* 是我们的 0 扩展初始滤波器，因此:

![](img/ea7340953f58b5e845052d1bbdf996bf.png)

让我们用几个选定的指数值来形象化它。

![](img/e1def291a4c4a50afebbf3ff4c742a7d.png)

使用∫符号进行卷积，我们得到:

![](img/15fecc7af48dd9a05f680cbdac446c32.png)

由于 *dy* 保持不变，我们将只查看 *dx* 22 的 w .的索引值，范围为 w:3-I，3-j

![](img/9e5aadd61cda3596e1e338ac6d042794.png)

现在我们有了 dy 和 w’矩阵之间的卷积，定义如下:

![](img/beb46f1405c5a38c05af10f12e64b0a9.png)

另一个例子来看看发生了什么。 *dx* 43，w:4—*I*，3—*j*

![](img/22a15437166acacb127fa6019e3b7990.png)

最后一个 *dx* 44

![](img/bbfd7dc555bd5ab27ecaccfe28f9c0aa.png)

我们确实看到弹出一个“反向过滤器”w’。这一次，我们在具有大小为 1 的 0 填充边界的输入 *dy* 和步长为 1 的滤波器 w’slidding 之间进行卷积。

![](img/03d9a7f9ad92f9371e7e3f961628d87e.png)

## 反投影方程综述

![](img/b9d19f739977c9d8bbe9ce57e454e611.png)

# 考虑到深度

当我们试图考虑深度时，事情变得稍微复杂一些(C 通道用于输入 x，F 不同的过滤器用于输入 w)

输入:

*   x:形状(C，H，W)
*   w:过滤器的权重形状(F，C，HH，WW)
*   形状(F，)

产出:

*   y:形状(F，H '，W ')

数学公式中出现了许多指数，使得它们更难阅读。我们示例中的正向传递公式是:

![](img/40f24362c9f4da37e69a92957a938c07.png)

## 

db 计算仍然很容易，因为每个 *b_f* 都与一个激活图 *y_f* 相关:

![](img/2c219ec63f618160869274ea3c903fa0.png)

## 发展的宽度（Developed Width 的缩写）

![](img/78b8911bdff8f9633fd400eac1104629.png)

使用向前传递公式，由于双和不使用 dy 索引，我们可以写为:

![](img/84ad3b4097502f181135cc7d427a80b4.png)

# 算法

既然我们对它是如何工作的有了直觉，我们选择不写整个方程组(这可能是相当乏味的)，但是我们将使用已经为正向传递编码的内容，并且通过玩维度来尝试为每个梯度编码反向投影。幸运的是，我们可以计算梯度的数值来检查我们的实现。这种实现只对步幅=1 有效，对于不同步幅，事情变得稍微复杂一些，需要另一种方法。也许是为了另一个职位！

## 梯度数值检验

```
Testing conv_backward_naive function
dx error:  7.489787768926947e-09
dw error:  1.381022780971562e-10
db error:  1.1299800330640326e-10
```

几乎每次都是 0，一切似乎都很好！:)

# 参考

*   [我博客上的这篇关于 mathjax 方程的文章](https://neodelphis.github.io/convnet/maths/python/english/2019/07/10/convnet-bp-en.html):)
*   [斯坦福视觉识别卷积神经网络课程](http://cs231n.stanford.edu/)
*   [斯坦福 CNN 作业 2](http://cs231n.github.io/assignments2019/assignment2/)
*   [卷积神经网络，正向传递](https://www.youtube.com/watch?v=bNb2fEVKeEo&list=PL3FW7Lu3i5JvHM8ljYj-zLfQRF3EO8sYv&index=5)
*   [卷积层:正向传递的朴素实现](https://neodelphis.github.io/convnet/python/2019/07/02/convnet-forward-pass.html)。
*   [卷积神经网络中的反向传播](https://www.jefkine.com/general/2016/09/05/backpropagation-in-convolutional-neural-networks/)
*   [Cet 法语文章](https://neodelphis.github.io/convnet/maths/python/2019/07/08/convnet-bp.html)

欢迎评论改进本帖，随时联系我！