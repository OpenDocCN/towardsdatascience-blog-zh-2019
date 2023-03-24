# 什么是深度迁移学习，为什么它变得如此流行？

> 原文：<https://towardsdatascience.com/what-is-deep-transfer-learning-and-why-is-it-becoming-so-popular-91acdcc2717a?source=collection_archive---------10----------------------->

![](img/bad94ad10b03409229d7348b86779e7b.png)

A man sitting on a bridge in Austria

# 介绍

正如我们已经知道的，大型有效的深度学习模型是数据饥渴的。他们需要用数千甚至数百万个数据点进行训练，然后才能做出合理的预测。

培训在时间和资源上都非常昂贵。例如，由谷歌开发的流行语言表示模型 BERT 已经在 **16 个云 TPU(总共 64 个 TPU 芯片)上训练了 4 天**。客观地说，这是大约 60 台台式计算机连续运行 4 天。

然而，最大的问题是，像这样的模型只能在单一任务中执行。未来的任务需要一组新的数据点以及等量或更多的资源。

![](img/747cf5726c4e8fc16a8177c74fc8064b.png)

Photo by [Rachel](https://unsplash.com/@noguidebook?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/learning?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

然而，人类的大脑并不是这样工作的。当解决一个新任务时，它不会丢弃先前获得的知识。相反，它根据从过去学到的东西做出决定。

*迁移学习旨在模仿这种行为。*

# 什么是迁移学习？

*迁移学习是深度学习(和机器学习)中的一种方法，其中知识从一个模型转移到另一个模型。*

> Def: *模型 A 使用大型数据集 D.a 成功训练以解决源任务 T.a。但是，目标任务 T.b 的数据集 D.b 太小，妨碍了模型 B 的有效训练。因此，我们使用部分模型 A 来预测任务 T.b.* 的结果

一个常见的误解是，训练和测试数据应该来自同一个来源或具有相同的分布。

使用迁移学习，我们能够在不同的任务中使用全部或部分已经预先训练好的模型来解决特定的任务。

著名的人工智能领袖吴恩达在下面的视频中很好地解释了这个概念。

# 什么时候使用迁移学习？

迁移学习正在成为使用深度学习模型的首选方式。原因解释如下。

## 缺乏数据

深度学习模型需要大量数据来有效地解决一项任务。然而，并不是经常有这么多数据可用。例如，一家公司可能希望为其内部通信系统构建一个非常特定的垃圾邮件过滤器，但并不拥有大量带标签的数据。

在这种情况下，可以使用类似源任务的预训练模型来解决特定的目标任务。

> **任务可以不同，但它们的领域应该相同。**

换句话说，你不能在语音识别和图像分类任务之间进行迁移学习，因为输入数据集的类型不同。

你可以做的是在狗的照片上使用预先训练好的图像分类器来预测猫的照片。

![](img/a2bd95fe82f3aff95edf723c1a7eea36.png)

Source: “How to build your own Neural Network from scratch in Python” by James Loy

## 速度

迁移学习减少了很大一部分培训时间，并允许立即构建各种解决方案。此外，它还可以防止设置复杂且昂贵的云 GPU/TPU。

## 社会公益

使用迁移学习对环境有积极的影响。

根据[麻省理工科技评论](https://www.technologyreview.com/s/613630/training-a-single-ai-model-can-emit-as-much-carbon-as-five-cars-in-their-lifetimes/)的一项研究，在云 TPU 上训练的大型神经网络(200M+参数)在其生命周期内产生的二氧化碳相当于 6 辆汽车。迁移学习可以防止这些强大的处理单元的广泛使用。

# 深度迁移学习策略

迁移学习可以通过几种不同的策略应用于深度学习和机器学习领域。在这篇文章中，我将只涉及深度学习技术，称为*深度迁移学习策略*。

在深度学习模型上进行迁移学习有 3 种主要策略。

## 直接使用预先训练的模型

最简单的策略是通过直接应用来自源任务的模型来解决目标任务。

这种模型通常是大型(数百万个参数)神经网络，在最先进的机器上训练数天甚至数周。

大公司(公司、大学等。)倾向于向公众发布此类模型，旨在增强该领域的发展。

一些直接使用的预训练模型包括前面提到的[伯特](https://arxiv.org/abs/1810.04805)以及 [YOLO(你只看一次)](https://pjreddie.com/darknet/yolo/)、[手套](https://nlp.stanford.edu/projects/glove/)、[未监督的](https://github.com/facebookresearch/UnsupervisedMT)等等。

## 利用从预训练模型中提取的特征

我们可以通过丢弃最后一个**完全连接的输出层**，将预训练的神经网络视为特征提取器，而不是像前面的例子那样使用端到端的模型。

这种方法允许我们直接应用新的数据集来解决一个完全不同的问题。

它带来了两个主要优势:

*   **允许指定最后一个完全连接层的尺寸。**

例如，预训练网络可能具有来自最后完全连接之前的层的*7×7×512*输出。我们可以将其拉平为 *21，055* ，这将产生一个新的 *N x 21，055* 网络输出( *N* —数据点的数量)。

*   **允许使用轻量级线性模型(如线性 SVM、逻辑回归)。**

因为预训练的复杂神经网络模型被用作新任务的特征，所以我们被允许训练更简单和更快速的线性模型，以基于新的数据集修改输出。

特征提取策略最适合于目标任务数据集非常小的情况。

## 微调预训练模型的最后几层

我们可以更进一步，不仅训练输出分类器，而且在预训练模型的一些层中微调权重。

典型地，网络的早期层(尤其是 CNN)被冻结，而最后的层被释放用于调谐。

这允许我们在现有模型上执行完整的训练，并在最后一层修改参数。

> 我们选择仅修改最后的图层，因为已经观察到网络中较早的图层捕获更多的通用要素，而较晚的图层则非常特定于数据集。

假设我们最初的预训练模型以非常高的准确度识别奔驰汽车。该模型的初始层倾向于捕捉关于车轮位置、汽车形状、曲线等的信息。我们可以在下一个识别法拉利汽车的任务中保留这些信息。然而，对于更具体的法拉利功能，我们应该用新的数据集重新训练最后几层。

*话虽如此，当目标任务数据集非常大，并且与源任务数据集共享一个相似的域时，最好使用微调策略。*

# 资源

本文的灵感来自一系列论文和教程，其中包括:

*   [转移学习](http://cs231n.github.io/transfer-learning/)作者 Andrej Karpathy @ Stanford。
*   [由](/a-comprehensive-hands-on-guide-to-transfer-learning-with-real-world-applications-in-deep-learning-212bf3b2f27a) [Dipanjan (DJ) Sarkar](https://towardsdatascience.com/@dipanzan.sarkar?source=post_page-----212bf3b2f27a----------------------) 撰写的综合实践指南，将学习与深度学习中的真实世界应用。
*   [Keras:利用深度学习对大型数据集进行特征提取](https://www.pyimagesearch.com/2019/05/27/keras-feature-extraction-on-large-datasets-with-deep-learning/)作者 Adrian Rosebrock。
*   [使用 Keras 进行微调和深度学习](https://www.pyimagesearch.com/2019/06/03/fine-tuning-with-keras-and-deep-learning/)Adrian rose Brock。
*   [迁移学习研究综述。](https://ieeexplore.ieee.org/stamp/stamp.jsp?tp=&arnumber=5288526)
*   [深度学习迁移学习的温和介绍](https://machinelearningmastery.com/transfer-learning-for-deep-learning/)作者 Jason Brownlee。
*   Jason Brownlee 的《用计算机视觉模型在 Keras 中转移学习》。

# 感谢您的阅读。希望你喜欢这篇文章。❤️