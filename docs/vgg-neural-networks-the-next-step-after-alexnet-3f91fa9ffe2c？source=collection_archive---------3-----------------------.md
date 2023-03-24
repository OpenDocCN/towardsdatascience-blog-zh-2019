# VGG 神经网络:AlexNet 之后的下一步

> 原文：<https://towardsdatascience.com/vgg-neural-networks-the-next-step-after-alexnet-3f91fa9ffe2c?source=collection_archive---------3----------------------->

2012 年 AlexNet 问世，是革命性的进步；它改进了传统的卷积神经网络(CNN)，成为图像分类的最佳模型之一……直到 [VGG](https://arxiv.org/pdf/1409.1556.pdf) 问世。

![](img/086e2e1eaabc8aeb9dc584bb5647897a.png)

Photo by [Taylor Vick](https://unsplash.com/@tvick?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

**AlexNet。**当 AlexNet 发布时，它轻松赢得了 ImageNet 大规模视觉识别挑战赛(ILSVRC ),并证明了自己是最有能力的对象检测模型之一。它的主要特性包括使用 ReLU 代替 tanh 函数、针对多个 GPU 的优化以及重叠池。它通过使用数据扩充和删除来解决过拟合问题。那么 AlexNet 到底出了什么问题？嗯，没有什么，比如说，特别“错误”的。人们只是想要更精确的模型。

**数据集。**图像识别的一般基线是 ImageNet，这是一个由超过 1500 万张图像组成的数据集，标记有超过 22，000 个类别。通过网络抓取图像和众包人类贴标机，ImageNet 甚至举办了自己的竞争:前面提到的 ImageNet 大规模视觉识别挑战(ILSVRC)。来自世界各地的研究人员面临着创新方法的挑战，以产生最低的前 1 名和前 5 名错误率(前 5 名错误率将是正确标签不是模型的五个最可能标签之一的图像的百分比)。比赛给出 120 万幅图像的 1000 类训练集，5 万幅图像的验证集，15 万幅图像的测试集；数据是丰富的。AlexNet 在 2012 年赢得了这场比赛，基于其设计的模型在 2013 年赢得了比赛。

![](img/09ef96da03dbc08aef0262d0afc57b1c.png)

Configurations of VGG; depth increases from left to right and the added layers are bolded. The convolutional layer parameters are denoted as “conv<receptive field size> — <number of channels>”. Image credits to Simonyan and Zisserman, the original authors of the VGG paper.

**VGG 神经网络。虽然 AlexNet 以前的衍生产品专注于第一卷积层的较小窗口大小和步长，但 VGG 解决了 CNN 的另一个非常重要的方面:深度。让我们来看看 VGG 的建筑:**

*   **输入。** VGG 接受 224x224 像素的 RGB 图像。对于 ImageNet 竞赛，作者在每个图像中裁剪出中心 224x224 的补丁，以保持输入图像大小一致。
*   **卷积层。**VGG 的卷积层使用非常小的感受域(3x3，仍然可以捕捉左/右和上/下的最小可能尺寸)。还有 1x1 卷积滤波器，它充当输入的线性变换，后跟一个 ReLU 单元。卷积步距固定为 1 个像素，以便在卷积后保持空间分辨率。
*   **全连接层。** VGG 有三个全连接层:前两层各有 4096 个通道，第三层有 1000 个通道，每个类别一个通道。
*   **隐藏层。**VGG 的所有隐藏层都使用 ReLU(Alex net 的一项巨大创新，减少了训练时间)。VGG 通常不使用本地响应标准化(LRN)，因为 LRN 增加了内存消耗和训练时间，但准确性没有特别提高。

**区别。** VGG 虽然基于 AlexNet，但它与其他竞争车型有几个不同之处:

*   VGG 没有使用像 AlexNet 这样的大感受野(11x11，步幅为 4)，而是使用非常小的感受野(3x3，步幅为 1)。因为现在有三个 ReLU 单元，而不是只有一个，所以决策函数更具区分性。参数也更少(27 倍通道数而不是 AlexNet 的 49 倍通道数)。
*   VGG 合并了 1x1 卷积层，以在不改变感受野的情况下使决策函数更加非线性。
*   小尺寸卷积滤波器允许 VGG 具有大量的权重层；当然，层数越多，性能越好。不过，这并不是一个不常见的特性。GoogLeNet 是另一个使用深度 CNN 和小型卷积滤波器的模型，也在 2014 年的 ImageNet 竞赛中亮相。

![](img/2cf92f041a8e8b2e68a05d430dd6de1b.png)

Performance of VGG at multiple test scales. Image credits to Simonyan and Zisserman, the original authors of the VGG paper.

**战果。**在单个测试量表上，VGG 取得了 25.5%的前 1 名误差和 8.0%的前 5 名误差。在多个测试量表中，VGG 得到的前 1 名误差为 24.8%，前 5 名误差为 7.5%。VGG 还在 2014 年 ImageNet 竞赛中以 7.3%的前五名误差获得了第二名，提交后该误差降至 6.8%。

**现在怎么办？** VGG 是一种创新的物体识别模型，支持多达 19 层。作为一个深度 CNN，VGG 在 ImageNet 之外的许多任务和数据集上也优于基线。VGG 现在仍然是最常用的图像识别架构之一。

我在下面附上了一些有趣的资源

*   [原创论文](https://arxiv.org/pdf/1409.1556.pdf)
*   [Alex net 上的博文](/alexnet-the-architecture-that-challenged-cnns-e406d5297951)
*   [CNN 上的维基百科文章](https://en.wikipedia.org/wiki/Convolutional_neural_network)