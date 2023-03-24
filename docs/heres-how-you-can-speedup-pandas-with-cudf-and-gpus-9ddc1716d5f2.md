# 以下是你如何用 cuDF 和 GPU 加速熊猫的方法

> 原文：<https://towardsdatascience.com/heres-how-you-can-speedup-pandas-with-cudf-and-gpus-9ddc1716d5f2?source=collection_archive---------7----------------------->

## 轻松加速超过 10 倍

![](img/0ac74d8ff0b4747d116b291f4e5aa6ba.png)

> 想获得灵感？快来加入我的 [**超级行情快讯**](https://www.superquotes.co/?utm_source=mediumtech&utm_medium=web&utm_campaign=sharing) 。😎

Pandas 是数据科学家和机器学习实践者的首选数据存储库。它允许轻松管理、探索和操作许多不同类型的数据，包括数字和文本。

# 熊猫和速度

即使就其本身而言，Pandas 在速度上已经比 Python 有了很大的进步。每当你发现你的 Python 代码运行缓慢，特别是如果你看到许多 for 循环，改变执行[数据探索和过滤的代码到熊猫函数](/how-to-use-pandas-the-right-way-to-speed-up-your-code-4a19bd89926d)中总是一个好主意。熊猫函数是专门开发的矢量化运算，运行速度最快！

尽管如此，即使有这样的加速，熊猫也只是在 CPU 上运行。由于消费类 CPU 通常只有 8 个内核或更少，因此并行处理的数量以及所能实现的加速是有限的。现代数据集可能有多达数百万、数十亿甚至数万亿的数据点需要处理——8 个内核是不够的。

幸运的是，随着 GPU 加速在机器学习中的普遍成功，有一股强大的推动力将数据分析库引入 GPU。cuDF 库是朝着这个方向迈出的一步。

# 带 cuDF 的 GPU 上的熊猫

cuDF 是一个基于 Python 的 GPU 数据框架库，用于处理数据，包括加载、连接、聚合和过滤数据。由于 GPU 拥有比 CPU 多得多的内核，因此向 GPU 的转移允许大规模加速。

cuDF 的 API 是 Pandas 的一面镜子，在大多数情况下可以用来直接替换。这使得数据科学家、分析师和工程师很容易将其集成到他们的工作流程中。

所有需要做的就是把你的熊猫数据帧转换成 cuDF 帧，瞧，你有 GPU 加速！cuDF 将支持 Pandas 所做的大部分常见数据帧操作，因此许多常规的 Pandas 代码可以毫不费力地加速。

为了开始使用 cuDF 的例子，我们可以通过 conda 安装这个库:

```
conda install -c nvidia -c rapidsai -c numba -c conda-forge -c defaults cudf
```

请记住，对于以下实验，我们测试的机器具有以下规格:

*   i7–8700k CPU
*   1080 Ti GPU
*   32 GB DDR 4 3000 MHz 内存
*   CUDA 9.2

# 获得 GPU 加速

我们将加载一个随机数字的大数据集，并比较各种 Pandas 操作的速度与使用 cuDF 在 GPU 上做同样事情的速度。

我们可以在 Python Jupyter 笔记本中完成所有这些工作。让我们从初始化数据帧开始:一个用于熊猫，一个用于 cuDF。数据框有超过 1 亿个点！

对于我们的第一个测试，让我们测量一下在 Pandas 和 cuDF 中计算我们的数据中‘a’变量的平均值需要多长时间。 [*%timeit*](https://docs.python.org/2/library/timeit.html) 命令允许我们在 Jupyter 笔记本上测量 Python 命令的速度。

平均运行时间显示在上面的代码注释中。我们就这样获得了 16 倍的加速！

现在，做一些更复杂的事情怎么样，比如做一个巨大的合并？！让我们在数据帧的“b”列上合并数据帧本身。

这里的合并是一个**非常**大的操作，因为熊猫将不得不寻找和匹配公共值——对于一个有 1 亿行的数据集来说，这是一个耗时的操作！GPU 加速将使这更容易，因为我们有更多的并行进程可以一起工作。

下面是代码和结果:

即使使用相当强大的 i7–8700k CPU，熊猫平均也需要 39.2 秒才能完成合并。另一方面，我们在 GPU 上的朋友 cuDF 只用了 2.76 秒，这是一个更容易管理的时间！总共加速了 14 倍以上。

# 结论

所以你有它！这就是你如何使用 cuDF 在 GPU 上加速熊猫。

如果你想吃更多，不用担心！cuDF 和 RAPIDS API 的其他部分可以提供更快的速度。最好的起点是[官方 GitHub 页面](https://github.com/rapidsai)！

# 喜欢学习？

在推特[上关注我，在那里我会发布所有最新最棒的人工智能、技术和科学！也在 LinkedIn](https://twitter.com/GeorgeSeif94) 上与我联系！