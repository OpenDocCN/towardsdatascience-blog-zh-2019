# 卷积神经网络用于乳腺癌分类

> 原文：<https://towardsdatascience.com/convolutional-neural-network-for-breast-cancer-classification-52f1213dcc9?source=collection_archive---------3----------------------->

## 深度学习用于解决女性中最常诊断的**癌症**

![](img/26b2dfba500e54e6e0603426cb3485f9.png)

Photo by [Tamara Bellis](https://unsplash.com/@tamarabellis?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/pink-ribbon?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

被困在付费墙后面？点击[这里](https://medium.com/p/convolutional-neural-network-for-breast-cancer-classification-52f1213dcc9?source=email-c3f5233f3441--writer.postDistributed&sk=cdc740178a784a00ec62a51aa87201b3)阅读完整故事与我的朋友链接！

乳腺癌是全世界女性和男性中第二常见的癌症。2012 年，它占所有新癌症病例的 12%，占所有女性癌症的 25%。

当乳房中的细胞开始不受控制地生长时，乳腺癌就开始了。这些细胞通常会形成一个肿瘤，通常可以在 x 射线上看到或摸到一个肿块。如果细胞可以生长(侵入)周围组织或扩散(转移)到身体的远处区域，则肿瘤是恶性的(癌症)。

# 挑战

建立一种算法，通过查看活检图像自动识别患者是否患有乳腺癌。算法必须非常精确，因为人命关天。

# 数据

数据集可以从[这里](https://web.inf.ufpr.br/vri/databases/breast-cancer-histopathological-database-breakhis/)下载。这是一个二元分类问题。我把数据分开，如图所示-

```
dataset train
  benign
   b1.jpg
   b2.jpg
   //
  malignant
   m1.jpg
   m2.jpg
   //  validation
   benign
    b1.jpg
    b2.jpg
    //
   malignant
    m1.jpg
    m2.jpg
    //...
```

训练文件夹在每个类别中有 1000 个图像，而验证文件夹在每个类别中有 250 个图像。

![](img/6c82f20a8df867d983e3f808791f8283.png)![](img/6d1f84a14472be6cb6c7ded9fd8f0e33.png)

Benign sample

![](img/950455085c3d1d872a38dd7bc208a3b2.png)![](img/d7be5e96b60fb293f6b473fbf73cab49.png)

Malignant sample

# CNN 架构

让我们一步一步来分析卷积神经网络中的每一层。

## 投入

形状为[宽度、高度、通道]的像素值矩阵。假设我们的输入是[32x32x3]。

## **卷积**

该图层的目的是接收要素地图。通常，我们从低数量的滤波器开始进行低级特征检测。我们越深入 CNN，我们使用越多的过滤器来检测高级特征。特征检测是基于用给定尺寸的过滤器“扫描”输入，并应用矩阵计算来导出特征图。

![](img/4d6903bd354ef12f811cf9a344cab69a.png)

Convolution Operation

## **联营**

这一层的目标是提供空间变化，这简单地意味着系统将能够识别一个对象，即使它的外观以某种方式变化。Pooling layer 将沿空间维度(宽度、高度)执行缩减采样操作，从而产生输出，例如 pooling_size=(2，2)的[16x16x12]。

![](img/aab85f576dccfb5de12da0aa4d6d5c06.png)

Pooling Operation

## **完全连接**

在完全连接的层中，我们展平最后一个卷积层的输出，并将当前层的每个节点与下一层的其他节点连接起来。完全连接层中的神经元与前一层中的所有激活都有完全连接，正如在常规神经网络中看到的那样，并以类似的方式工作。

![](img/28712ae7e0b425500424c0b58addf3bd.png)

CNN Overview

# 图像分类

完整的图像分类管道可以形式化如下:

*   我们的输入是一个由 *N* 幅图像组成的训练数据集，每幅图像都被标记为两个不同类别中的一个。
*   然后，我们使用这个训练集来训练一个分类器，以学习每个类的样子。
*   最后，我们通过要求分类器预测一组它以前从未见过的新图像的标签来评估分类器的质量。然后，我们将这些图像的真实标签与分类器预测的标签进行比较。

# 代码在哪里？

事不宜迟，让我们从代码开始吧。github 上的完整项目可以在[这里](https://github.com/abhinavsagar/Breast-cancer-classification)找到。

让我们从加载所有的库和依赖项开始。

接下来，我将图片加载到各自的文件夹中。

之后，我创建了一个由 0 组成的 numpy 数组来标记良性图像，同样也创建了一个由 1 组成的 numpy 数组来标记恶性图像。我还重组了数据集，并将标签转换成分类格式。

然后，我将数据集分成两组——分别包含 80%和 20%图像的训练集和测试集。让我们看一些良性和恶性图像的样本。

![](img/dd14c4883edf49e930bfc72119984984.png)

Benign vs malignant samples

我使用的批量值是 16。批量大小是深度学习中要调整的最重要的超参数之一。我更喜欢使用较大的批量来训练我的模型，因为它允许 GPU 并行性的计算加速。然而，众所周知，批量太大会导致泛化能力差。在一个极端情况下，使用等于整个数据集的批次保证了收敛到目标函数的全局最优。然而，这是以较慢地收敛到最优值为代价的。另一方面，使用较小的批量已被证明具有更快的收敛到良好的结果。这可以通过以下事实直观地解释，即较小的批量允许模型在必须看到所有数据之前就开始学习。使用较小批量的缺点是模型不能保证收敛到全局最优。因此，通常建议从小批量开始，获得更快的训练动态的好处，并通过训练稳步增加批量。

我也做了一些数据扩充。数据扩充**的做法**是增加训练集规模的有效方法。增加训练示例允许网络在训练期间看到更多样化但仍有代表性的数据点。

然后，我创建了一个数据生成器，以自动方式从我们的文件夹中获取数据并导入 Keras。为此，Keras 提供了方便的 python 生成器函数。

下一步是构建模型。这可以通过以下 3 个步骤来描述:

1.  我使用 DenseNet201 作为预训练的重量，它已经在 Imagenet 比赛中训练过。学习率被选择为 0.0001。
2.  在此基础上，我使用了一个 globalaveragepooling 层，然后是 50%的辍学，以减少过度拟合。
3.  我使用了批量标准化和一个具有 2 个神经元的密集层，用于 2 个输出类别，即良性和恶性，使用 softmax 作为激活函数。
4.  我用 Adam 作为优化器，用二元交叉熵作为损失函数。

让我们看看输出的形状和每一层所涉及的参数。

![](img/7dbcc064c8131ac98376b80bdb5a6bca.png)

Model summary

在定型模型之前，定义一个或多个回调是有用的。相当方便的一个，有:ModelCheckpoint 和 ReduceLROnPlateau。

*   **ModelCheckpoint** :当训练需要大量时间来达到一个好的结果时，通常需要多次迭代。在这种情况下，最好仅在改善度量的时期结束时保存最佳执行模型的副本。
*   **ReduceLROnPlateau** :当指标停止改善时，降低学习率。一旦学习停滞，模型通常会受益于将学习速度降低 2-10 倍。这种回调监控一个数量，如果在“耐心”次数内没有看到改进，则学习率降低。

![](img/e463448b9f03bb2d6edd72df89113a91.png)

ReduceLROnPlateau.

我训练了 20 个纪元的模型。

# 性能指标

评估模型性能的最常见指标是精确度。然而，当只有 2%的数据集属于一个类别(恶性)而 98%属于其他类别(良性)时，错误分类分数实际上没有意义。你可以有 98%的准确率，但仍然没有发现任何恶性病例，这可能是一个可怕的分类器。

![](img/39c1ca41eda43dc6497f370209ce2837.png)

Loss vs epoch

![](img/151dd82e1babd75818350c0ff17c6b53.png)

Accuracy vs epoch

# 精确度、召回率和 F1 分数

为了更好地了解错误分类，我们经常使用以下指标来更好地了解真阳性(TP)、真阴性(TN)、假阳性(FP)和假阴性(FN)。

**精度**是正确预测的正观测值与总预测正观测值的比率。

**召回**是正确预测的正面观察值与实际类中所有观察值的比率。

**F1-Score** 是准确率和召回率的加权平均值。

![](img/7ea290f14dc075fbec2a354d2bfc0032.png)![](img/5d070ffaba36a13196ac0f882015fdc3.png)

F1 分数越高，模型越好。对于所有三个指标，0 是最差的，而 1 是最好的。

# 混淆矩阵

在分析误分类时，混淆矩阵是一个非常重要的度量。矩阵的每一行代表预测类中的实例，而每一列代表实际类中的实例。对角线代表已被正确分类的类别。这很有帮助，因为我们不仅知道哪些类被错误分类，还知道它们被错误分类为什么。

![](img/4d5e238f0c39a994cdc9dd35a9976492.png)

Confusion matrix

# ROC 曲线

45 度线是随机线，其中曲线下面积或 AUC 是 0.5。曲线离这条线越远，AUC 越高，模型越好。一个模型能得到的最高 AUC 是 1，其中曲线形成一个直角三角形。ROC 曲线也可以帮助调试模型。例如，如果曲线的左下角更接近随机线，则暗示模型在 Y=0 处分类错误。然而，如果右上角是随机的，则意味着误差发生在 Y=1 处。

![](img/a5d8f11c3c9fcc65700ee35a9f504162.png)

ROC-AUC curve

# 结果

![](img/4a67dbeb9d51d99b3ee30753d479f3a7.png)

Final results

# 结论

虽然这个项目还远未完成，但在如此多样的现实世界问题中看到深度学习的成功是令人瞩目的。在这篇博客中，我展示了如何使用卷积神经网络和迁移学习从一组显微图像中对良性和恶性乳腺癌进行分类。

# 参考资料/进一步阅读

[](/transfer-learning-for-image-classification-in-keras-5585d3ddf54e) [## 基于迁移学习的 Keras 图像分类

### 迁移学习的一站式指南

towardsdatascience.com](/transfer-learning-for-image-classification-in-keras-5585d3ddf54e) [](/predicting-invasive-ductal-carcinoma-using-convolutional-neural-network-cnn-in-keras-debb429de9a6) [## 在 Keras 中使用卷积神经网络(CNN)预测浸润性导管癌

### 使用卷积神经网络将组织病理学切片分类为恶性或良性

towardsdatascience.com](/predicting-invasive-ductal-carcinoma-using-convolutional-neural-network-cnn-in-keras-debb429de9a6) [](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0214587) [## 乳腺癌组织病理学图像分类使用卷积神经网络与小…

### 尽管从组织病理学图像中成功检测恶性肿瘤在很大程度上取决于长期的…

journals.plos.org](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0214587) [](https://becominghuman.ai/deep-learning-for-image-classification-with-less-data-90e5df0a7b8e) [## 基于深度学习的少数据图像分类

### 深度学习确实可以用更少的数据实现

becominghuman.ai](https://becominghuman.ai/deep-learning-for-image-classification-with-less-data-90e5df0a7b8e) 

# 在你走之前

相应的源代码可以在这里找到。

[](https://github.com/abhinavsagar/Breast-cancer-classification) [## abhinavsagar/乳腺癌-分类

### 使用卷积神经网络的良性与恶性分类器数据集可以从这里下载。pip 安装…

github.com](https://github.com/abhinavsagar/Breast-cancer-classification) 

# 联系人

如果你想了解我最新的文章和项目[，请关注我的媒体](https://medium.com/@abhinav.sagar)。以下是我的一些联系人详细信息:

*   [个人网站](https://abhinavsagar.github.io)
*   [领英](https://in.linkedin.com/in/abhinavsagar4)
*   [中等轮廓](https://medium.com/@abhinav.sagar)
*   [GitHub](https://github.com/abhinavsagar)
*   [卡格尔](https://www.kaggle.com/abhinavsagar)

快乐阅读，快乐学习，快乐编码！