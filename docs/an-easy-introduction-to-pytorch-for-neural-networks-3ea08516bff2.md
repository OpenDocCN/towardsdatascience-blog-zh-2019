# 神经网络 Pytorch 的简单介绍

> 原文：<https://towardsdatascience.com/an-easy-introduction-to-pytorch-for-neural-networks-3ea08516bff2?source=collection_archive---------8----------------------->

## 感受 Pytorch 之火！

![](img/56e4710e9fcf34eb980e8b675f0e2472.png)

> 想获得灵感？快来加入我的 [**超级行情快讯**](https://www.superquotes.co/?utm_source=mediumtech&utm_medium=web&utm_campaign=sharing) 。😎

深度学习重新点燃了公众对人工智能的兴趣。原因很简单:深度学习*就是管用*。它让我们有能力建立以前无法建立的技术。它创造了新的商业机会，从整体上改善了技术世界。

为了进行深度学习，你需要知道如何编码，尤其是用 Python。从那里，有一个不断增长的深度学习库可供选择:TensorFlow，Keras，MXNet，MatConvNet，以及最近的 Pytorch！

Pytorch 发行后不久就迅速流行起来。人们称它为 *TensorFlow 杀手*，因为它更加用户友好和易于使用。事实上，您将看到使用 Pytorch 启动和运行深度学习是多么容易。

# Pytorch 入门

Pytorch 开发的核心目标是尽可能地与 Python 的 Numpy 相似。这样做可以让常规 Python 代码、Numpy 和 Pytorch 之间的交互变得简单流畅，从而实现更快更简单的编码。

首先，我们可以通过 pip 安装 Pytorch:

```
pip3 install torch torchvision
```

如果你对具体的特性感兴趣，Pytorch 文档非常棒。

## 张量

任何深度学习库最基本的构建块都是*张量*。张量是类似矩阵的数据结构，在功能和属性上非常类似于 Numpy 数组。事实上，在大多数情况下，您可以将它们想象成 Numpy 数组。两者最重要的区别在于，现代深度学习库中张量的实现可以在 CPU 或 GPU 上运行(非常快)。

在 PyTorch 中，可以使用简单的张量对象来声明张量:

```
import torch 
x = torch.Tensor(3, 3)
```

上面的代码创建了一个大小为(3，3)的张量，即 3 行 3 列，用浮点零填充:

```
0\.  0\.  0.
0\.  0\.  0.
0\.  0\.  0.
[torch.FloatTensor of size 3x3]
```

我们还可以创建张量填充的随机浮点值:

```
x = torch.rand(3, 3)
print(x)"""
Prints out:tensor([[0.5264, 0.1839, 0.9907],
        [0.0343, 0.9839, 0.9294],
        [0.6938, 0.6755, 0.2258]])
"""
```

使用 Pytorch，张量相乘、相加和其他基本数学运算非常简单:

```
x = torch.ones(3,3)
y = torch.ones(3,3) * 4
z = x + y
print(z)"""
Prints out:tensor([[5., 5., 5.],
        [5., 5., 5.],
        [5., 5., 5.]])
"""
```

Pytorch 张量甚至提供了类似 Numpy 的切片功能！

```
x = torch.ones(3,3) * 5
y = x[:, :2]
print(y)"""
Prints out:tensor([[5., 5.],
        [5., 5.],
        [5., 5.]])
"""
```

所以 Pytorch 张量可以像 Numpy 数组一样被使用和处理。现在，我们将看看如何使用这些简单的 Pytorch 张量作为构建模块来构建深度网络！

# 用 Pytorch 构建神经网络

在 Pytorch 中，神经网络被定义为 Python 类。定义网络的类从 torch 库中扩展了 *torch.nn.Module* 。让我们为卷积神经网络(CNN)创建一个类，我们将应用于 MNIST 数据集。

查看下面定义我们网络的代码！

Pytorch 网络类中最重要的两个函数是 *__init__()* 和 *forward()* 函数。 *__init__()* 用于定义您的模型将使用的任何网络层。在 *forward()* 函数中，您实际上是通过将所有层堆叠在一起来建立模型。

对于我们的模型，我们在 init 函数中定义了 2 个卷积层，其中一个我们将重复使用几次(conv2)。我们有一个最大池层和一个全局平均池层，将在最后应用。最后，我们有我们的全连接(FC)层和一个 softmax 来获得最终的输出概率。

在 forward 函数中，我们确切地定义了我们的层如何堆叠在一起以形成完整的模型。这是一个标准网络，具有堆叠的 conv 层、池层和 FC 层。Pytorch 的美妙之处在于，我们可以在 *forward()* 函数中的任何地方，通过简单的 print 语句打印出中间层中任何张量的形状和结果！

# 培训、测试和保存

## 加载数据

是时候为训练准备好我们的数据了！我们将开始，但准备好必要的导入，初始化参数，并确保 Pytorch 设置为使用 GPU。下面使用`torch.device()`的一行检查 Pytorch 是否安装了 CUDA 支持，如果是，则使用 GPU！

我们可以直接从 Pytroch 检索 MNIST 数据集。我们将下载数据，并将训练集和测试集放入单独的张量中。一旦数据被加载，我们将把它传递给 torch *DataLoader* ，它只是准备好以特定的批量和可选的混洗传递给模型。

## 培养

训练时间到了！

optimzer(我们将使用 Adam)和 loss 函数(我们将使用交叉熵)的定义与其他深度学习库非常相似，如 TensorFlow、Keras 和 MXNet。

在 Pytorch 中，所有的网络模型和数据集都必须明确地从 CPU 转移到 GPU。我们通过将`.to()`函数应用于下面的模型来实现这一点。稍后，我们将对图像数据进行同样的操作。

最后，我们可以写出我们的训练循环。查看下面的代码，看看它是如何工作的！

1.  所有 Pytorch 训练循环将在训练数据加载器中经历每个时期和每个批次。
2.  在每次循环迭代中，图像数据和标签都被传输到 GPU。
3.  每个训练循环还明确应用向前传递、向后传递和优化步骤。
4.  将该模型应用于该批中的图像，然后计算该批的损失。
5.  计算梯度并通过网络反向传播

## 测试和保存

在 Pytorch 中测试网络的性能会建立一个与训练阶段类似的循环。主要的区别是我们不需要做梯度的反向传播。我们仍将进行前向传递，只在网络的输出端获取具有最大概率的标签。

在这种情况下，经过 10 个时期后，我们的网络在测试集上获得了 99.06%的准确率！

要将模型保存到磁盘以备后用，只需使用`torch.save()`功能，瞧！

# 喜欢学习？

在 [twitter](https://twitter.com/GeorgeSeif94) 上关注我，我会在那里发布所有最新最棒的人工智能、技术和科学！也在 LinkedIn 上与我联系！