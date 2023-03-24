# 用 Pytorch 实现神经网络的批量归一化和丢失

> 原文：<https://towardsdatascience.com/batch-normalization-and-dropout-in-neural-networks-explained-with-pytorch-47d7a8459bcd?source=collection_archive---------1----------------------->

![](img/5cd18626395df3310304d5b70431281d.png)

Photo by [Wesley Caribe](https://unsplash.com/@wesleycaribe?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

在本文中，我们将讨论为什么我们需要深度神经网络中的批处理规范化和剔除，然后在标准数据集上使用 Pytorch 进行实验，以查看批处理规范化和剔除的效果。本文基于我对 PadhAI 深度学习讲座的理解。

> *引用注:本文内容和结构基于四分之一实验室深度学习讲座——*[*帕德海*](https://padhai.onefourthlabs.in/) *。*

# 为什么要标准化输入？

在讨论批处理规范化之前，我们将了解为什么规范化输入会加快神经网络的训练。

考虑这样一种情况，我们将具有特征 x_1 和 x_2 的 2D 数据输入神经网络。这些特征之一 x_1 具有从-200 到 200 的较宽范围，而另一个特征 x_2 具有从-10 到 10 的较窄范围。

![](img/76078d1c1d230a5332a250ceb87112a0.png)

Before Normalization

一旦我们将数据标准化，两个特征的数据分布集中在一个区域，即从-2 到 2。传播看起来像这样，

![](img/b2047a401ccc64ca36aff49d89f2023c.png)

After Normalization

> 让我们讨论一下为什么规范化输入会有帮助？

在对输入进行归一化之前，与这些输入相关联的权重会有很大变化，因为输入要素存在于从-200 到 200 以及从-2 到 2 的不同范围内。为了适应特征之间的这种范围差异，一些权重必须较大，而一些权重必须较小。如果我们具有更大的权重，那么与反向传播相关联的更新也将很大，反之亦然。由于输入权重的这种不均匀分布，学习算法在找到全局最小值之前一直在平台区域振荡。

为了避免学习算法花费大量时间在平台上振荡，我们对输入特征进行归一化，使得所有特征都在相同的尺度上。因为我们的输入是在相同的尺度上，所以与它们相关联的权重也将是在相同的尺度上。从而帮助网络更快地训练。

# 批量标准化

> 我们已经对输入进行了规范化，但是隐藏的代表呢？

通过对输入进行归一化，我们能够将所有的输入特征置于相同的尺度上。在神经网络中，我们需要计算第一层 a₁₁.的第一个神经元的预激活我们知道预激活只不过是输入加上偏差的加权和。换句话说，它是权重矩阵 W₁的第一行和输入矩阵 **X** 加上偏差 b₁₁.的点积

![](img/a70bf571e1bd4c4b5cb802aaef48effe.png)

在每一层“I”上预激活的数学方程由下式给出:

![](img/0f9c3a816d80b75be8fbc0124c1b5c29.png)

每一层的激活等于将激活函数应用于该层的预激活的输出。在每一层“I”上激活的数学方程由下式给出，

![](img/8c23fac1ad73c4f7319cf07a663e4b59.png)

类似地，需要计算网络中存在的“n”个隐藏层的激活值。激活值将作为网络中下一个隐藏层的输入。因此，不管我们对输入做了什么，不管我们是否对它们进行了归一化，激活值都会随着我们基于与相应神经元相关联的权重对网络越来越深入而发生很大变化。

为了使所有的激活值达到相同的尺度，我们对激活值进行了归一化，这样隐藏的表示就不会发生剧烈的变化，同时也有助于我们提高训练速度。

> W 为什么叫批量规范化？

因为我们计算的是单个批次的平均值和标准差，而不是整个数据。批量标准化在网络中的每个隐藏神经元上单独完成。

![](img/6e17221c1cdaeff1bae3dfd2c60875ae.png)

# 学习γγ和ββ

由于我们正在规范网络中的所有激活，我们是否正在实施一些可能会降低网络性能的约束？

为了保持隐藏神经网络的代表性，批处理规范化引入了两个额外的参数 Gamma 和 Beta。一旦我们规范化了激活，我们需要再执行一个步骤来获得最终的激活值，它可以作为另一个层的输入。

![](img/efaea5fe44e195506c8dd3ae10546ab9.png)

参数γ和β与网络的其他参数一起被学习。如果γ(γ)等于平均值(μ),β(β)等于标准偏差(σ),则激活 h_final 等于 h_norm，从而保持网络的代表能力。

# 在 Colab 中运行此笔记本

文章中讨论的所有代码都在我的 GitHub 上。你可以通过谷歌虚拟机上运行的 Colab 在 Github 上直接打开我的 [Jupyter 笔记本](https://colab.research.google.com/github/Niranjankumar-c/DeepLearning-PadhAI/blob/master/DeepLearning_Materials/7_BatchNormalization/BatchNorm_Dropout.ipynb)，用任何设置打开 code 笔记本。如果您只想快速打开笔记本并按照本教程进行操作，请单击此处的。

 [## niranjankumar-c/deep learning-PadhAI

### 来自 pad hai-Niranjankumar-c/deep learning-pad hai 的深度学习课程相关的所有代码文件

github.com](https://github.com/Niranjankumar-c/DeepLearning-PadhAI/tree/master/DeepLearning_Materials/7_BatchNormalization) 

# 使用 Pytorch 进行批量标准化

为了了解批量标准化是如何工作的，我们将使用 Pytorch 构建一个神经网络，并在 MNIST 数据集上进行测试。

![](img/f1c8a00fe27486b5bffc9ec95826327d.png)

## 批量标准化— 1D

在本节中，我们将建立一个完全连接的神经网络(DNN)来分类 MNIST 数据，而不是使用 CNN。使用 DNN 的主要目的是解释在 1D 输入(如数组)的情况下批处理规范化是如何工作的。在将大小为 28x28 的 MNIST 图像提供给网络之前，我们将它们展平成大小为 784 的一维输入数组。

我们将创建两个深度神经网络，具有三个完全连接的线性层，并在它们之间交替重新激活。在具有批处理规范化的网络的情况下，我们将在 ReLU 之前应用批处理规范化，如原始论文中所提供的。由于我们的输入是一个 1D 数组，我们将使用 Pytorch nn 模块中的`BatchNorm1d`类。

```
import torch.nn as nn
nn.BatchNorm1d(48) #48 corresponds to the number of input features it is getting from the previous layer.
```

为了更好地了解批处理规范化如何帮助网络更快地收敛，我们将在训练阶段查看网络中多个隐藏层的值分布。

为了保持一致，我们将绘制两个网络中第二个线性图层的输出，并比较该图层的输出在网络中的分布。结果看起来像这样:

![](img/042230fd58ac2c768dc13e6e3af9ede0.png)

The first row indicates first epoch and second row for second epoch

从图表中，我们可以得出结论，在每个时段**内，没有批量归一化的值的分布在输入的迭代之间发生了显著变化，这意味着没有批量归一化的网络中的后续层看到了输入数据**的变化分布。但是批量标准化模型的值分布的变化似乎可以忽略不计。

此外，我们可以看到，由于协方差移动，即每批输入的隐藏值移动，批量标准化网络的损耗比正常网络降低得更快。这有助于网络更快地收敛并减少训练时间。

![](img/cc7fd0cb1a2c4340fde3b3f95455fa42.png)

## 批量标准化— 2D

在上一节中，我们已经看到了如何为以 1D 数组作为输入的前馈神经网络编写线性层之间的批量规范化。在本节中，我们将从语法的角度讨论如何实现卷积神经网络的批量规范化。

我们将采用相同的 MNIST 数据图像，并编写一个实现批量标准化的网络。该批 RGB 图像有四个维度——batch _ size**x**通道 **x** 高度 **x** 宽度。在图像的情况下，我们对每个通道的批次进行标准化。类别`BatchNorm2d`对 4D 输入(带有额外通道维度的小型 2D 输入)应用批量标准化。

类`BatchNorm2d`将它从前一层的输出中接收的通道数作为参数。

# 拒绝传统社会的人

在文章的这一部分，我们将讨论神经网络中的丢失概念，特别是它如何帮助减少过拟合和泛化错误。之后，我们将使用 Pytorch 实现一个有和没有丢失的神经网络，看看丢失如何影响网络的性能。

放弃是一种正则化技术，它随机“放弃”或“停用”神经网络中的少数神经元，以避免过拟合问题。

# 辍学的想法

用数据上的大参数训练一个深度神经网络可能导致过度拟合。能否在同一个数据集上训练多个不同配置的神经网络，取这些预测的平均值？。

![](img/b9d02936c266860bccd5700ec3c94be6.png)

但创建一个具有不同架构的神经网络集合并训练它们在实践中是不可行的。辍学去救援。

丢弃在每个训练步骤中随机地去激活神经元，而不是在原始网络上训练数据，我们在具有丢弃节点的网络上训练数据。在训练步骤的下一次迭代中，由于其概率性行为，被丢弃去激活的隐藏神经元发生变化。以这种方式，通过应用 dropout，即…在训练期间随机停用某些单个节点，我们可以模拟具有不同架构的神经网络的整体。

## 培训时辍学

![](img/2be261f429a25114e7b53bd04d70df9f.png)

在每次训练迭代中，网络中的每个节点都与概率 p 相关联，是保留在网络中还是以概率 1-p 将其停用(退出)网络。这意味着与节点相关联的权重仅更新了 p 次，因为节点在训练期间仅活动 p 次。

## 测试时掉线

![](img/98967fc77e7a67bc181bdcf82499568f.png)

在测试期间，我们考虑所有激活都存在的原始神经网络，并通过值 p 缩放每个节点的输出。

# 使用 Pytorch 辍学

为了直观地显示 dropout 如何减少神经网络的过度拟合，我们将使用 Pytorch `torch.unsqueeze`生成一个简单的随机数据点。遗漏的效用在可能过度拟合的自定义数据上表现得最为明显。

一旦我们生成了数据，我们可以使用如下所示的`matplotlib`散点图来可视化张量。

![](img/7ea0d13d33c7ec34ed3eb159cd2c42d6.png)

为了显示过度拟合，我们将训练两个网络——一个没有辍学，另一个有辍学。没有丢失的网络具有 3 个完全连接的隐藏层，ReLU 作为隐藏层的激活函数，而有丢失的网络也具有类似的架构，但是在第一和第二线性层之后应用了丢失。

在 Pytorch 中，我们可以使用`torch.nn`模块申请退学。

```
import torch.nn as nn
nn.Dropout(0.5) #apply dropout in a neural network
```

在本例中，我在第一个线性图层后使用了 0.5 的漏失分数，在第二个线性图层后使用了 0.2 的漏失分数。一旦我们训练了两个不同的模型，即…一个没有辍学，另一个有辍学，并绘制测试结果，它看起来像这样:

![](img/5a93cb714ee42c0cfda047b6a143ffe7.png)

从上面的图表中，我们可以得出结论，随着我们增加历元的数量，没有丢失的模型正在过度拟合数据。无漏失模型是学习与数据相关的噪声，而不是对数据进行归纳。我们可以看到，与没有丢弃的模型相关联的损耗随着时段数量的增加而增加，这与具有丢弃的模型相关联的损耗不同。

 [## niranjankumar-c/deep learning-PadhAI

### 来自 pad hai-Niranjankumar-c/deep learning-pad hai 的深度学习课程相关的所有代码文件

github.com](https://github.com/Niranjankumar-c/DeepLearning-PadhAI/tree/master/DeepLearning_Materials/7_BatchNormalization) 

# 继续学习

如果你想用 Keras & Tensorflow 2.0(Python 或者 R)学习更多关于人工神经网络的知识。查看来自 [Starttechacademy](https://courses.starttechacademy.com/full-site-access/?coupon=NKSTACAD) 的 Abhishek 和 Pukhraj 的[人工神经网络](https://courses.starttechacademy.com/full-site-access/?coupon=NKSTACAD)。他们以一种简单化的方式解释了深度学习的基础。

# 结论

在本文中，我们已经讨论了为什么我们需要批量归一化，然后我们继续使用 MNIST 数据集来可视化批量归一化对隐藏层输出的影响。在此之后，我们讨论了辍学的工作，它防止数据过拟合的问题。最后，我们可视化了两个网络在有丢包和无丢包情况下的性能，以观察丢包的影响。

*推荐阅读*

[](https://www.marktechpost.com/2019/06/30/building-a-feedforward-neural-network-using-pytorch-nn-module/) [## 利用 Pytorch 神经网络模块构建前馈神经网络

### 前馈神经网络也称为多层神经元网络(MLN)。这些模型网络是…

www.marktechpost.com](https://www.marktechpost.com/2019/06/30/building-a-feedforward-neural-network-using-pytorch-nn-module/) [](/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e) [## 用 Pytorch 可视化卷积神经网络

### 可视化 CNN 过滤器并对输入进行遮挡实验

towardsdatascience.com](/visualizing-convolution-neural-networks-using-pytorch-3dfa8443e74e) 

如果你在实现我的 GitHub 库中的代码时遇到任何问题，请随时通过 LinkedIn 或 twitter 联系我。

直到下次和平:)

NK。

**免责声明** —这篇文章中可能有一些相关资源的附属链接。你可以以尽可能低的价格购买捆绑包。如果你购买这门课程，我会收到一小笔佣金。