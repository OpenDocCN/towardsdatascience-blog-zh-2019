# UNet

> 原文：<https://towardsdatascience.com/u-net-b229b32b4a71?source=collection_archive---------2----------------------->

## 在分段中引入对称性

# 介绍

视觉是人类拥有的最重要的感官之一。但是你有没有想过这个任务的复杂性？捕捉反射光线并从中获取意义的能力是一项非常复杂的任务，但我们却轻而易举地做到了。由于数百万年的进化，我们开发了它。那么如何才能在极短的时间内赋予机器同样的能力呢？对于计算机来说，这些图像只不过是矩阵，理解这些矩阵背后的细微差别多年来一直是许多数学家的困扰。但是在人工智能，特别是 CNN 架构出现之后，研究取得了前所未有的进展。许多以前被认为是不可触及的问题现在显示出惊人的结果。

一个这样的问题是图像分割。在图像分割中，机器必须将图像分割成不同的片段，每个片段代表一个不同的实体。

![](img/9150665c9d64eb5e7b6cd11a17e58f6b.png)

Image Segmentation Example

正如你在上面看到的，图像是如何变成两段的，一段代表猫，另一段代表背景。从自动驾驶汽车到卫星，图像分割在许多领域都很有用。也许其中最重要的是医学成像。医学图像中的微妙之处非常复杂，有时甚至对训练有素的医生来说也是一个挑战。能够理解这些细微差别并识别必要区域的机器可以在医疗保健领域产生深远的影响。

卷积神经网络在简单的图像分割问题上给出了不错的结果，但在复杂的问题上没有取得任何进展。这就是 UNet 出现的原因。UNet 最初是专门为医学图像分割而设计的。它显示出如此好的效果，以至于后来它被用于许多其他领域。在本文中，我们将讨论 UNet 为什么以及如何工作。如果你不知道 CNN 背后的直觉，请先阅读[这个](https://ujjwalkarn.me/2016/08/11/intuitive-explanation-convnets/)。你可以在这里查看 UNet 在行动[。](https://www.kaggle.com/hsankesara/unet-image-segmentation)

# UNet 背后的直觉

CNN 背后的主要思想是学习图像的特征映射，并利用它来进行更细致的特征映射。这在分类问题中很有效，因为图像被转换成进一步用于分类的向量。但是在图像分割中，我们不仅需要将特征映射转换成一个向量，还需要从这个向量中重建图像。这是一项艰巨的任务，因为将矢量转换成图像比反过来要困难得多。UNet 的整个理念就是围绕这个问题展开的。

在将图像转换为矢量时，我们已经学习了图像的特征映射，那么为什么不使用相同的映射将其再次转换为图像呢？这是 UNet 背后的配方。使用用于收缩的相同特征映射将向量扩展为分割图像。这将保持图像的结构完整性，从而极大地减少失真。让我们更简单地了解一下架构。

# UNet 架构

## UNet 如何工作

![](img/342f2c88bc2f14bea027a44026df8a50.png)

UNet Architecture

该建筑看起来像一个“U”形，名副其实。这个架构由三个部分组成:收缩部分、瓶颈部分和扩展部分。收缩段由许多收缩块组成。每个块接受一个输入，应用两个 3×3 卷积层，然后是一个 2×2 最大池。每个块之后的核或特征图的数量加倍，以便架构可以有效地学习复杂的结构。最底层介于收缩层和膨胀层之间。它使用两个 3X3 CNN 层，然后是 2X2 up 卷积层。

但是这个架构的核心在于扩展部分。类似于收缩层，也是由几个膨胀块组成。每个模块将输入传递到两个 3X3 CNN 层，然后是一个 2×2 上采样层。此外，在每个块之后，卷积层使用的特征图的数量减半以保持对称性。然而，每次输入也得到相应收缩层的附加特征图。这个动作将确保在收缩图像时学习的特征将被用于重建图像。扩展块的数量与收缩块的数量相同。之后，所得到的映射通过另一个 3X3 CNN 层，其特征映射的数量等于期望的分段数量。

## UNet 中的损失计算

在这种固有图像分割中，人们会使用哪种损失？好吧，它在论文中被简单地定义了。

> 结合交叉熵损失函数，通过在最终特征图上的逐像素软最大值来计算能量函数

UNet 为每个**像素**使用了一种相当新颖的损失加权方案，使得在分割对象的边界处具有较高的权重。这种损失加权方案有助于 U-Net 模型以*不连续的方式*分割生物医学图像中的细胞，从而可以在二进制分割图中轻松识别单个细胞。

首先对合成图像应用逐像素的 softmax，然后应用交叉熵损失函数。所以我们将每个像素分为一类。这个想法是，即使在分割中，每个像素都必须位于某个类别中，我们只需要确保它们确实如此。因此，我们只是将一个分割问题转化为一个多类分类问题，与传统的损失函数相比，它表现得非常好。

# 联合国网络的实施

我使用 Pytorch 框架实现了 UNet 模型。你可以在这里 查看 UNet 模块 [*。使用用于具有糖尿病性黄斑水肿的光学相干断层扫描图像的分割的图像。你可以在这里*](https://github.com/Hsankesara/DeepResearch)查看正在运行的 UNet[。](https://www.kaggle.com/hsankesara/unet-image-segmentation)

上面代码中的 UNet 模块代表了 UNet 的整个架构。`contraction_block`和`expansive_block`分别用于创建收缩段和扩张段。函数`crop_and_concat`将收缩层的输出与新的扩展层输入相加。训练部分可以写成

```
unet = Unet(in_channel=1,out_channel=2)
#out_channel represents number of segments desired
criterion = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(unet.parameters(), lr = 0.01, momentum=0.99)
optimizer.zero_grad()       
outputs = unet(inputs)
# permute such that number of desired segments would be on 4th dimension
outputs = outputs.permute(0, 2, 3, 1)
m = outputs.shape[0]
# Resizing the outputs and label to caculate pixel wise softmax loss
outputs = outputs.resize(m*width_out*height_out, 2)
labels = labels.resize(m*width_out*height_out)
loss = criterion(outputs, labels)
loss.backward()
optimizer.step()
```

# 结论

图像分割是一个重要的问题，每天都有新的研究论文发表。联合国教育网在此类研究中做出了重大贡献。许多新的建筑都受到了 UNet 的启发。但是仍然有太多的东西需要探索。行业中有如此多的这种架构的变体，因此有必要理解第一种变体，以便更好地理解它们。因此，如果您有任何疑问，请在下面评论或参考资源页面。

# 资源

*   [UNet 原稿](https://arxiv.org/pdf/1505.04597.pdf)
*   [UNet Pytorch 实施](https://github.com/Hsankesara/DeepResearch/tree/master/UNet)
*   [UNet Tensorflow 实施](https://github.com/jakeret/tf_unet)
*   [关于语义分割的更多信息](https://www.jeremyjordan.me/semantic-segmentation/)
*   [实用图像分割](https://tuatini.me/practical-image-segmentation-with-unet/)

# 作者说明

这篇教程是我的系列文章中的第二篇。如果你喜欢这个教程，请在评论中告诉我，如果你不喜欢，请在评论中简单告诉我。如果你有任何疑问或批评，请在评论中大量发表。我会尽快回复你。如果你喜欢这个教程，请与你的同伴分享。