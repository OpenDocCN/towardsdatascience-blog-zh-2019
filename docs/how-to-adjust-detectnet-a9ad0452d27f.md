# 如何调整检测网

> 原文：<https://towardsdatascience.com/how-to-adjust-detectnet-a9ad0452d27f?source=collection_archive---------25----------------------->

## 一种由 NVIDIA 创建的对象检测架构

![](img/caffd187e5deca8c482b04fa36b6d138.png)

Photo by [Christian Wiediger](https://unsplash.com/@christianw?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

DetectNet 是由 NVIDIA 创建的对象检测架构。它可以从 NVIDIA 的 Deep Learning 图形用户界面 DIGITS 运行，通过该界面，您可以快速设置和开始训练分类、物体检测、分割和其他类型的模型。

NVIDIA 提供了两个基本的 DetectNet prototxt 文件:

1.  单个类一个(为原件)，可在[这里](https://github.com/NVIDIA/caffe/blob/v0.15.9/examples/kitti/detectnet_network.prototxt)找到，并且
2.  两个等级中的一个可以在这里找到。

DetectNet 最初的架构是用 Caffe 编写的。除了在[英伟达网站](https://devblogs.nvidia.com/)上发表的两篇博客文章和一些(主要)重申博客内容的教程之外，我没有找到太多关于该架构的文档。我确实发现，在 NVIDIA/DIGITS 存储库中，有一个特定的 GitHub 问题[问题#980](https://github.com/NVIDIA/DIGITS/issues/980) 已经积累了大量信息。

以下是我从 GitHub 问题中收集的重点:

*   训练集中的图像大小不应该不同。如果是，您应该填充它们或调整它们的大小，使其大小相等。调整大小或填充可以在数字数据集创建步骤中完成。
*   DetectNet 对大小在 50x50 像素到 400x400 像素之间的边界框很敏感。它很难识别超出这个范围的边界框。
*   如果您想检测比检测网敏感的尺寸小的对象，您可以调整图像的大小，使大部分边界框适合检测网的首选范围，或者您可以将模型的步距更改得更小。
*   图像尺寸必须可被步幅整除。例如，1248 和 384(检测网的默认图像大小)可以被 16 整除。
*   如果您正在使用不同于原始架构的图像分辨率来训练模型(原始架构期望图像的宽度为 1248，高度为 384)，您需要在架构内的行 [57](https://github.com/NVIDIA/caffe/blob/94c600c56e726deed2aebe954f9ec390a8e4f9f3/examples/kitti/detectnet_network.prototxt#L57) 、 [58](https://github.com/NVIDIA/caffe/blob/94c600c56e726deed2aebe954f9ec390a8e4f9f3/examples/kitti/detectnet_network.prototxt#L58) 、 [79](https://github.com/NVIDIA/caffe/blob/94c600c56e726deed2aebe954f9ec390a8e4f9f3/examples/kitti/detectnet_network.prototxt#L79) 、 [80](https://github.com/NVIDIA/caffe/blob/94c600c56e726deed2aebe954f9ec390a8e4f9f3/examples/kitti/detectnet_network.prototxt#L80) 、 [118](https://github.com/NVIDIA/caffe/blob/94c600c56e726deed2aebe954f9ec390a8e4f9f3/examples/kitti/detectnet_network.prototxt#L118) 、 [119](https://github.com/NVIDIA/caffe/blob/94c600c56e726deed2aebe954f9ec390a8e4f9f3/examples/kitti/detectnet_network.prototxt#L119) 、 [2504](https://github.com/NVIDIA/caffe/blob/94c600c56e726deed2aebe954f9ec390a8e4f9f3/examples/kitti/detectnet_network.prototxt#L2504) 、 [2519 上更改指定的图像尺寸](https://github.com/NVIDIA/caffe/blob/94c600c56e726deed2aebe954f9ec390a8e4f9f3/examples/kitti/detectnet_network.prototxt#L2519)
*   要更改模型步幅，您必须在行 [73](https://github.com/NVIDIA/caffe/blob/94c600c56e726deed2aebe954f9ec390a8e4f9f3/examples/kitti/detectnet_network.prototxt#L73) 、 [112](https://github.com/NVIDIA/caffe/blob/94c600c56e726deed2aebe954f9ec390a8e4f9f3/examples/kitti/detectnet_network.prototxt#L112) 、 [2504](https://github.com/NVIDIA/caffe/blob/94c600c56e726deed2aebe954f9ec390a8e4f9f3/examples/kitti/detectnet_network.prototxt#L2504) 、 [2519](https://github.com/NVIDIA/caffe/blob/94c600c56e726deed2aebe954f9ec390a8e4f9f3/examples/kitti/detectnet_network.prototxt#L2519) 和 [2545](https://github.com/NVIDIA/caffe/blob/94c600c56e726deed2aebe954f9ec390a8e4f9f3/examples/kitti/detectnet_network.prototxt#L2545) 中将默认步幅值(16)更改为您想要的值(这些行指的是单个类 DetectNet prototxt)。
*   如果指定一个较小的步幅，则需要减少网络中的层数来调整维度。降低维数的一种方法是将 pool3/3x3_s2 层的内核和步幅参数更改为 1。该层存在于从[行 826 到 836](https://github.com/NVIDIA/caffe/blob/94c600c56e726deed2aebe954f9ec390a8e4f9f3/examples/kitti/detectnet_network.prototxt#L826-L836) (这些行指的是单个类 DetectNet prototxt)。

对于**多类**目标检测，其中您想要检测两个以上的类，您可以更改 2 类检测网络协议[ [5](https://www.coria.com/insights/blog/computer-vision/training-a-custom-mutliclass-object-detection-model) ]。取决于类别数量的行有:

*   [第 82 行到第 83 行](https://github.com/NVIDIA/caffe/blob/caffe-0.15/examples/kitti/detectnet_network-2classes.prototxt#L82-%23L83):为每个额外的类添加一行，递增地或基于数据集标签文本文件中的类值更改“src”和“dst”。
*   [第 2388 行](https://github.com/NVIDIA/caffe/blob/caffe-0.15/examples/kitti/detectnet_network-2classes.prototxt#L2388):更改你的模型将识别的类别数量。
*   第 2502 至 2503 行:每增加一个等级增加一行
*   [第 2507 行](https://github.com/NVIDIA/caffe/blob/caffe-0.15/examples/kitti/detectnet_network-2classes.prototxt#L2507):将最后一个数字更改为您的模型将识别的类的数量。
*   [2518-2519](https://github.com/NVIDIA/caffe/blob/caffe-0.15/examples/kitti/detectnet_network-2classes.prototxt#L2518-#L2519)行:每增加一节课就增加一行。
*   [2523](https://github.com/NVIDIA/caffe/blob/caffe-0.15/examples/kitti/detectnet_network-2classes.prototxt#L2523)行:将最后一个数字更改为您的模型将识别的类的数量。
*   [2527-2579 行](https://github.com/NVIDIA/caffe/blob/caffe-0.15/examples/kitti/detectnet_network-2classes.prototxt#L2527-L2579):这里有 4 层，每层 2 层。每个类别有一个对应于分数的层(该层对指定类别的检测进行评分)和一个对应于 mAP 的层(该层计算指定类别的 mAP)。为每一个额外的类别添加一个分数和类别图层，并确保在图层的顶部和底部 blobs 中指定正确的类别号。

综上所述:(I)确保数据集中的所有图像大小相同(如果不相同，请调整大小或填充它们)；(ii)如果使用的是自定义大小的数据集，则必须修改 DetectNet prototxt 文件中的多行内容；(iii)确保数据中的大部分边界框的大小在 50x50 和 400x400 像素之间；(iv)如果要将跨步更改为较小的值， 您必须减少网络中的层，并且(v)您可以调整 DetectNet 体系结构，通过将一些数字更改为您要识别的类的数目，并在 prototxt 文件中添加额外的行和层来进行多类对象检测。

我建议您阅读下面提到的所有参考资料，以便更好地了解 DetectNet 以及如何针对您的问题对其进行修改。

参考文献:

1.  [检测网:数字量物体检测的深层神经网络](https://devblogs.nvidia.com/detectnet-deep-neural-network-object-detection-digits/)
2.  [1 级检测网网络原型](https://github.com/NVIDIA/caffe/blob/v0.15.9/examples/kitti/detectnet_network.prototxt)
3.  [2 级检测网网络原型](https://github.com/NVIDIA/caffe/blob/caffe-0.15/examples/kitti/detectnet_network-2classes.prototxt)
4.  [GitHub NVIDIA/DIGITS 问题#980:如何使用 DetectNet 获取自定义大小的数据？](https://github.com/NVIDIA/DIGITS/issues/980)
5.  [数字用户:将初始步幅从 16 改为 8](https://groups.google.com/forum/m/#!topic/digits-users/zx_UYu3jlt8)
6.  [使用 Nvidia 数字训练自定义多类对象检测模型](https://www.coria.com/insights/blog/computer-vision/training-a-custom-mutliclass-object-detection-model)