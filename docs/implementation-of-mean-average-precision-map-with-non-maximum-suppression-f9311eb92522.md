# 使用非最大抑制(NMS)实现平均精度(mAP)

> 原文：<https://towardsdatascience.com/implementation-of-mean-average-precision-map-with-non-maximum-suppression-f9311eb92522?source=collection_archive---------10----------------------->

## 实现对象检测的度量

写完你的 CNN 物体检测模型后，你可能会认为最艰难的部分已经过去了。衡量你的物体探测器表现如何的标准呢？衡量异议检测的标准是映射。为了实现 mAP 计算，工作从来自 CNN 对象检测模型的预测开始。

## 非最大抑制

诸如 Yolov3 或更快的 RCNN 的 CNN 对象检测模型产生比实际需要更多的边界框(bbox)预测。第一步是通过非最大值抑制来清理预测。

![](img/4c4dd660f01574df022a9412b8843921.png)

ground truth bbox (Blue), predicted bbox (light pink), averaged predicted bbox (red)

上图显示了一个图像，其中蓝色矩形是基本事实边界框。浅粉色矩形是预测的边界框，其具有超过 0.5 的客观性，即边界框中存在对象的置信度得分。红色边界框是由浅粉色边界框平均得到的最终预测边界框。浅粉色包围盒到红色包围盒的平均称为非最大值抑制。

[https://github.com/eriklindernoren/PyTorch-YOLOv3](https://github.com/eriklindernoren/PyTorch-YOLOv3)提供了 Yolov3 预测后的非最大值抑制和 mAP 计算的详细实现，如本文所述。每个 Yolov3 的预测由右上边界框坐标(x1，y1)、左下边界框坐标(x2，y2)、对象置信度(Objectness)和每个类的分类置信度(C1，..C60，如果边界框内容可以被分类为 60 个类别)。对于每个图像，假设预测了 10654 个初始边界框，仅保留 6 个具有高于 0.5 的客观置信度的预测。在 6 个边界框中，彼此具有高重叠(高 IOU)并且预测相同类别的边界框被一起平均。这些步骤的详细说明解释如下:

![](img/b7c1a728b672302149e6461de64a68e1.png)

## 真阳性检测

一旦最终预测被确定，预测的边界框可以相对于地面真实接地框被测量，以产生 mAP 来查看对象检测器做得有多好。为此，需要确定真正阳性的数量。如果预测的边界以 IOU 阈值(0.5)与基本真实边界框重叠，则认为是成功的检测，并且预测的边界框是真的正的。如果预测的边界框与基本事实的重叠小于阈值，则认为是不成功的检测，并且预测的边界框是假阳性。精确度和召回率可以通过真阳性和假阳性来计算，如下所示:

![](img/48cf5f55a075429a4e0bba13ea77b5c0.png)

详细的实现如下所示。对于一批中的每个图像，对于图像中的每个预测边界框，如果边界框的预测类别不是图像中的目标类别之一，则将边界框记录为假阳性，否则，检查预测边界框与图像中的所有目标框，并获得与目标框的最高重叠。如果最高重叠大于 IOU 阈值，则认为目标框被成功检测到，并且预测的边界框被记录为真阳性。否则边界框被记录为假阳性。隐藏成功检测到的目标框，并继续循环以检查其他预测的边界框。返回每个预测的对象、其预测的类别以及它是否为真阳性。这些步骤的详细说明如下所示:

![](img/d9381902044644a99bd721ba580cc274.png)

## 地图计算

上述步骤的输出用于计算 mAP。按照反对程度的降序对预测进行排序。从具有最高客观性的预测开始，在每次增量预测之后，测量召回率(真阳性的计数/全局所有目标框的计数)和精确度(真阳性的计数/到目前为止的预测计数),并绘制召回率对精确度曲线。曲线下的区域就是地图。计算的面积为矩形，因此图形的三角化部分被忽略。可以为每一类预测计算 mAP，然后对所有类进行平均。感谢[https://medium . com/@ Jonathan _ hui/mAP-mean-average-precision-for-object-detection-45c 121 a 31173](https://medium.com/@jonathan_hui/map-mean-average-precision-for-object-detection-45c121a31173)对地图的详细讲解。召回率对精确度曲线的绘制如下所示:

![](img/04552041310d4742bdbb86eab7c36986.png)

dataframe for Recall-Precision graph (left), Recall-Precision graph (right)

详细的实现如下所示。按照客观性降序排列真阳性记录、客观性记录和类别记录。对于每个类，按照排序顺序，在每次增量预测后，从真阳性记录中找出累积的真阳性和假阳性。通过将累积的真阳性分别除以基础真值和预测的数量(真阳性+假阳性)，找到相应的召回率和精确度值。计算每一类的曲线下面积。这些步骤的详细说明如下所示:

![](img/eff1c068c26816288ea5b3dedea702be.png)

https://github.com/eriklindernoren/PyTorch-YOLOv3 中[的 test.py 和 utils/utils.py 可以参考 NMS 和 mAP 的完整实现。](https://github.com/eriklindernoren/PyTorch-YOLOv3)