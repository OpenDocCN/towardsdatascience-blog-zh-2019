# Pytorch 中的稀疏矩阵

> 原文：<https://towardsdatascience.com/sparse-matrices-in-pytorch-be8ecaccae6?source=collection_archive---------10----------------------->

## [预定义稀疏度](https://towardsdatascience.com/tagged/predefined-sparsity)

## 第 1 部分:CPU 运行时

这是分析 Pytorch 中稀疏矩阵及其密集矩阵的执行时间的系列文章的第 1 部分。第 1 部分处理 CPU 执行时间，而[第 2 部分](https://medium.com/@souryadey/sparse-matrices-in-pytorch-part-2-gpus-fd9cc0725b71)扩展到 GPU。在深入讨论之前，让我先简单介绍一下概念。

[Pytorch](https://pytorch.org/) 是用 Python 编程语言编写的深度学习库。深度学习是科学的一个分支，近年来由于它为自动驾驶汽车、语音识别等“智能”技术提供了动力，因此越来越受到重视。深度学习的核心是大量的矩阵乘法，这非常耗时，也是深度学习系统需要大量计算能力才能变好的主要原因。不足为奇的是，研究的一个关键领域是简化这些系统，以便它们可以快速部署。简化它们的一种方法是使矩阵**稀疏**，这样它们的大部分元素都是 0，在计算时可以忽略。例如，这里有一个稀疏矩阵，我们称之为 *S* :

![](img/8a050dde19a7ad520683cee7a17e0cd4.png)

您可能想知道这种矩阵在哪里以及如何出现。矩阵通常用于描述实体之间的交互。例如，S 的行可能表示不同的人，而列可能表示不同的地方。这些数字表明每个人在上周去过每个地方多少次。有几个 0 是可以解释的，因为每个人只去了一两个特定的地方。稀疏矩阵的**密度**是它的非零元素的分数，比如 s 中的 1/3，现在的问题是，有没有更好的方法来存储稀疏矩阵以避免所有的 0？

有几种稀疏格式，Pytorch 使用的一种叫做**首席运营官**坐标格式。它在一个稀疏矩阵中存储非零元素(nnz)的索引、值、大小和数量。下面是在 Pytorch 中构造 S 的一种方法(输出以粗体显示，注释以斜体显示):

```
S = torch.sparse_coo_tensor(indices = torch.tensor([[0,0,1,2],[2,3,0,3]]), values = torch.tensor([1,2,1,3]), size=[3,4])
#*indices has x and y values separately along the 2 rows*print(S)
**tensor(indices=tensor([[0, 0, 1, 2],
                       [2, 3, 0, 3]]),
       values=tensor([1, 2, 1, 3]),
       size=(3, 4), nnz=4, layout=torch.sparse_coo)**print(S.to_dense()) #*this shows S in the regular (dense) format*
**tensor([[0, 0, 1, 2],
        [1, 0, 0, 0],
        [0, 0, 0, 3]])**
```

Pytorch 有处理稀疏矩阵的`torch.sparse` API。这包括一些与常规数学函数相同的函数，例如用于将稀疏矩阵与密集矩阵相乘的`mm`:

```
D = torch.ones(3,4, dtype=torch.int64)torch.sparse.mm(S,D) #*sparse by dense multiplication*
**tensor([[3, 3],
        [1, 1],
        [3, 3]])**torch.mm(S.to_dense(),D) #*dense by dense multiplication*
**tensor([[3, 3],
        [1, 1],
        [3, 3]])**
```

现在我们来看这篇文章的要点。Pytorch 中使用稀疏矩阵和函数是否节省时间？换句话说，torch.sparse API 到底有多好？答案取决于 a)矩阵大小，和 b)密度。我用来测量运行时间的 CPU 是我的【2014 年年中 Macbook Pro，配有 2.2 GHz 英特尔酷睿 i7 处理器和 16 GB 内存。所以，让我们开始吧！

# 大小和密度都不同

对角矩阵是稀疏的，因为它只包含沿对角线的非零元素。密度将总是 1/ *n* ，其中 *n* 是行数(或列数)。这是我的两个实验案例:

*   稀疏:稀疏格式的对角矩阵乘以密集的方阵
*   密集:使用`to_dense()` *将相同的对角矩阵转换为密集格式，然后用*乘以相同的密集方阵

所有元素均取自随机正态分布。输入`torch.randn`就可以得到这个。以下是 *n* 随 2 的幂变化的运行时间:

![](img/c30ade3d368714670903ff447b129f64.png)![](img/21e62d4bd4a85681194886fd15cde2b9.png)

Left: Complete size range from 2²=4 to 2¹³=8192\. Right: Zoomed in on the x-axis up to 2⁸=256

密集情况下的计算时间增长大约为 O( *n* )。这并不奇怪，因为矩阵乘法是 O( *n* )。计算稀疏情况下的增长顺序更加棘手，因为我们将 2 个矩阵乘以不同的元素增长顺序。每次 *n* 翻倍，密集矩阵的非零元素的数量翻两番，但是稀疏矩阵的非零元素的数量翻倍。这给出了在 O( *n* )和 O( *n* )之间的顺序的总计算时间。

从右边的图中，我们看到稀疏情况下的初始增长很慢。这是因为访问开销在实际计算中占主导地位。然而，超过 n=64(即密度≤ 1.5%)标志的*是稀疏矩阵比密集矩阵计算速度更快的时候*。

# 密度固定，尺寸变化

请记住，稀疏对角矩阵的密度随着大小的增长而下降，因为密度= 1/ *n* 。更公平的比较是保持密度不变。下面的图再次比较了两种情况，只是现在稀疏矩阵的密度固定为 1/8，即 12.5%。因此，例如， *n* =2 稀疏情况将有 2 x 2 /8 = 2 个元素。

![](img/dc5fd189b2ac9ca2dc8a37c5f704be88.png)![](img/cfa14ac4cd98d4eeff0c9ff246393e7a.png)

Left: Complete size range from 2²=4 to 2¹³=8192\. Right: Zoom in on the x-axis up to 2⁷=128

这一次，当 n 增加一倍时，稀疏矩阵和密集矩阵的元素数量都增加了四倍。首席运营官格式需要一些时间来访问基于独立索引-值对的元素。这就是为什么*稀疏矩阵计算时间以大于 O(n )* 的速度增长，导致稀疏计算时间总是比密集计算时间差。

# 尺寸固定，密度变化

最后，让我们研究在保持大小固定在不同值时，密度对稀疏矩阵计算时间的影响。这里的伪代码是:

```
cases = [2**4=16, 2**7=128, 2**10=1024, 2**13=8192]
for n in cases:
    Form a dense random square matrix of size n
    for nnz = powers of 2 in range(1 to n**2):
        Form sparse square matrix of size n with nnz non-zero values
        Compute time to multiply these matrices
```

我将 *n* 固定在 4 种不同的情况下——16、128、1024 和 8192，并绘制了每种情况下的计算时间与密度的关系。密度可以用 nnz 除以 *n* 得到。我将之前的两个实验——对角线矩阵和 12.5%密度——标记为每个图中的垂直虚线。将 2 个密集矩阵相乘的时间是红色水平虚线。

![](img/267a7054ae1f452ce97057345712e17e.png)![](img/6522c75557296f2a7688ef25abc55e50.png)![](img/035fe9b5e3937c13e64044792d51b557.png)![](img/247be5837dbd204faae15bf04f15d07d.png)

n=16 的情况有点不同，因为它足够小，访问开销足以支配计算时间。对于其他 3 种情况，计算时间随着 nnz 加倍而加倍，即 O(nnz)。这并不奇怪，因为矩阵大小是相同的，所以唯一的增长来自 nnz。

**主要结论是，2 个密集矩阵总是比稀疏和密集矩阵相乘更快，除非稀疏矩阵具有非常低的密度。‘很低’好像是 1.5%及以下。**

所以你有它。在 Pytorch 中使用当前状态的稀疏库并不会带来太多好处，除非您正在处理非常稀疏的情况(比如大小大于 100 的对角矩阵)。我的研究涉及神经网络中预定义的稀疏性( [IEEE](https://ieeexplore.ieee.org/document/8689061) ， [arXiv](https://arxiv.org/abs/1812.01164) )，其中权重矩阵是稀疏的，输入矩阵是密集的。然而，尽管预定义的稀疏度在低至 20%的密度下给出了有希望的结果，但在 1.5%及以下的密度下性能确实会下降。所以不幸的是 Pytorch 稀疏的库目前并不适合。话虽如此，Pytorch sparse API 仍处于试验阶段，正在积极开发中，因此希望新的 pull 请求能够提高稀疏库的性能。

本文的代码和图片可以在 Github [这里](https://github.com/souryadey/speed-tests/tree/master/pytorch_sparse/part1_cpu_macbookpro)找到。

# 附录:存储稀疏矩阵

Pytorch 中的张量可以使用`torch.save()`保存。结果文件的大小是单个元素的大小乘以元素的数量。张量的`dtype`给出了单个元素的位数。例如，数据类型为 *float32* 的密集 1000x1000 矩阵的大小为(32 位 x 1000 x 1000) = 4 MB。(回想一下，8 位=1 字节)

不幸的是，稀疏张量不支持`.save()`特性。有两种方法可以保存它们——(a)转换为密集的并存储它们，或者(b)将`indices()`、`values()`和`size()`存储在单独的文件中，并从这些文件中重建稀疏张量。例如，假设`spmat`是一个大小为 1000x1000 的稀疏对角矩阵，即它有 1000 个非零元素。假设数据类型为 *float32* 。使用(a)，存储的矩阵的文件大小= (32 位 x 1000 x 1000) = 4 MB。使用(b)，`indices()`是数据类型 *int64* 的整数，有 2000 个索引(1000 个非零元素的每一个有 1 行 1 列)。这 1000 个非零`values()`都是*浮动 32* 。`size()`是一种叫做`torch.Size`的特殊数据类型，它是一个由两个整数组成的元组。因此，总文件大小大约为=(64 x 2000)+(32 x 1000)+(64 x 2)= 20.2 KB。这远远小于 4 MB。更一般地，文件大小对于(a)增长为 O( *n* ),对于(b)增长为 O(nnz)。但是你每次从(b)加载的时候都需要重构稀疏张量。

> Sourya Dey 正在南加州大学攻读博士学位。他的研究涉及探索深度学习中的复杂性降低。你可以在他的网站上读到更多关于他的信息。