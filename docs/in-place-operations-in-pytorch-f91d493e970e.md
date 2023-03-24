# PyTorch 的就地操作

> 原文：<https://towardsdatascience.com/in-place-operations-in-pytorch-f91d493e970e?source=collection_archive---------7----------------------->

![](img/30cda8e4bddbd2afd205446157e76c58.png)

Photo by [Fancycrave.com](https://www.pexels.com/@fancycrave?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) from [Pexels](https://www.pexels.com/photo/green-ram-card-collection-825262/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)

## 它们是什么，为什么要避开它们

今天先进的深度神经网络有数百万个可训练参数(例如，参见本文中的比较)，试图在 Kaggle 或 Google Colab 等免费 GPU 上训练它们往往会导致 GPU 上的内存耗尽。有几种简单的方法可以减少模型占用的 GPU 内存，例如:

*   考虑改变模型的架构或使用具有较少可训练参数的模型类型(例如，选择 [DenseNet-121 而不是 DenseNet-169](https://arxiv.org/pdf/1608.06993.pdf) )。这种方法会影响模型的性能指标。
*   减少批处理大小或手动设置数据加载器工作进程的数量。在这种情况下，模型需要更长的训练时间。

在神经网络中使用就地操作可能有助于避免上述方法的缺点，同时节省一些 GPU 内存。但是，由于几个原因，不建议使用就地操作。

在这篇文章中，我想:

*   描述什么是就地操作，并演示它们如何帮助节省 GPU 内存。
*   告诉我们为什么应该避免就地操作或非常谨慎地使用它们。

# 就地操作

> “就地操作是直接改变给定的线性代数、向量、矩阵(张量)的内容，而不制作副本的操作。”—定义摘自[本 Python 教程](https://www.tutorialspoint.com/inplace-operator-in-python)。

根据定义，就地操作不会复制输入。这就是为什么在处理高维数据时，它们可以帮助减少内存使用。

我想演示就地操作如何帮助消耗更少的 GPU 内存。为此，我将使用这个简单的函数来测量 PyTorch 中的[非适当位置 ReLU 和适当位置 ReLU 的分配内存:](https://pytorch.org/docs/stable/nn.html#relu)

Function to measure the allocated memory

调用函数来测量为不合适的 ReLU 分配的内存:

Measure the allocated memory for the out-of-place ReLU

我收到如下输出:

```
Allocated memory: 382.0
Allocated max memory: 382.0
```

然后调用就地 ReLU，如下所示:

Measure the allocated memory for the in-place ReLU

我收到的输出如下:

```
Allocated memory: 0.0
Allocated max memory: 0.0
```

看起来使用就地操作可以帮助我们节省一些 GPU 内存。**但是，在使用就地操作时要极其谨慎，要检查两遍。在下一部分，我将告诉你为什么。**

# 就地操作的缺点

就地操作的主要缺点是，**它们可能会覆盖计算梯度所需的值，**这意味着会破坏模型的训练过程。这就是 PyTorch 官方亲笔签名文件所说的:

> 在亲笔签名中支持就地操作是一件困难的事情，我们不鼓励在大多数情况下使用它们。Autograd 的积极的缓冲区释放和重用使其非常有效，并且很少有就地操作实际上降低内存使用量的情况。除非你的内存压力很大，否则你可能永远都不需要使用它们。
> 
> 有两个主要原因限制了就地作业的适用性:
> 
> 1.就地操作可能会覆盖计算梯度所需的值。
> 
> 2.每个就地操作实际上都需要实现重写计算图。不在位置的版本只是分配新的对象并保持对旧图的引用，而在位置操作中，需要改变表示该操作的函数的所有输入的创建者。

小心就地操作的另一个原因是它们的实现非常棘手。这就是为什么我会推荐使用 PyTorch 标准的就地操作(就像上面的就地 ReLU ),而不是手动实现。

让我们看一个[路斯](https://arxiv.org/pdf/1606.08415.pdf)(或 Swish-1)激活函数的例子。这是路斯的不合时宜的实现:

Out-of-place SiLU implementation

让我们尝试使用 `torch.[sigmoid](https://en.wikipedia.org/wiki/Sigmoid_function)_` 就地函数实现就地路斯:

Incorrect implementation of in-place SiLU

上面的代码**错误地实现了**就地路斯。我们可以通过比较两个函数返回的值来确定这一点。实际上，函数`silu_inplace_1`返回`sigmoid(input) * sigmoid(input)`！使用`torch.sigmoid_`就地实施路斯的工作示例如下:

这个小例子演示了为什么我们在使用就地操作时应该小心谨慎并检查两次。

# 结论

在本文中:

*   我描述了就地操作及其目的。演示了就地操作如何帮助减少 GPU 内存消耗。
*   我描述了就地作业的**重大缺点**。人们应该非常小心地使用它们，并检查两次结果。