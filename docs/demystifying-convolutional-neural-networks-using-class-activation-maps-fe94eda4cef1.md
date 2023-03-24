# 使用类激活图解密卷积神经网络。

> 原文：<https://towardsdatascience.com/demystifying-convolutional-neural-networks-using-class-activation-maps-fe94eda4cef1?source=collection_archive---------9----------------------->

机器学习每天都在获得指数级的动力，其应用在每个领域都在增加，无论是金融领域中琐碎的股票价格预测，还是计算机视觉领域中复杂的任务，如对象的检测和分割。没有哪个领域没有受到人工智能革命的影响，在某些领域，机器学习算法甚至超过了人类水平的性能。例如，ImageNet challenge 每年都举办各种计算机视觉任务，如图像分类、对象检测、图像定位等，每年表现最好的算法的错误率都在不断下降，2017 年，38 个参赛团队中有 29 个团队的准确率超过 95%。据报道，大规模 ImageNet 数据集上的人类前 5 名分类错误率为 5.1%，而最先进的 CNN 达到了约 3.57%的准确度。

随着机器学习系统性能的提高，系统的可解释性逐渐降低。这种趋势在深度学习算法中更为常见，这些算法包括数百万个参数和数百层，与线性回归、K 近邻、决策树等基本机器学习算法相比，解释这些算法极其困难。这些算法已经变得像黑匣子一样，从用户那里获取输入，并给出超出预期的输出，但对导致这种输出的因果信息却不提供直觉。这些算法可能适用于准确性是主要要求的任务，如 Kaggle 竞赛，数据科学家不必向各种利益相关者解释和说明结果。但是在结果可解释性非常重要的应用中，黑盒性质会导致各种障碍。例如，如果某个图像识别系统被训练来检测图像中的肿瘤，并且在验证和测试集的准确性方面表现得非常好。但是，当你向利益相关者展示结果时，他们会问你的模型是从图像的哪个部分学习的，或者这个输出的主要原因是什么，你最可能的答案是“我不知道”，无论你的模型有多完美，利益相关者都不会接受，因为这关系到人的生命。

随着机器学习特别是深度学习领域研究的增加，正在进行各种努力来解决可解释性问题并达到可解释 AI 的阶段。

在 CNN 的情况下，已经发现了各种可视化技术，其中之一是类激活图(CAM)。

在论文[中介绍的类激活图或 CAM 通过使用 CNN 中的全局平均池来学习用于区别性定位的深度特征](https://arxiv.org/abs/1512.04150)。特定类别的类别激活图指示 CNN 用来识别类别的区分区域。

# 建筑:

这篇论文的作者在 Network 中使用了类似于 [GoogLeNet](https://arxiv.org/abs/1409.4842) 和 [Network 的网络架构。网络主要由大量卷积层组成，在最终输出层之前，我们执行全局平均池。如此获得的特征被馈送到具有 softmax 激活的完全连接的层，该层产生期望的输出。我们可以通过将输出层的权重反投影到从最后一个卷积层获得的卷积特征图上来识别图像区域的重要性。这种技术被称为类激活映射。](https://arxiv.org/abs/1512.04150)

![](img/2c7a2f4b41ebe9c5d92696248e1b784f.png)

**Architecture and Working**

全局平均池层(GAP)优于在 [Oquab 等人](https://www.di.ens.fr/~josef/publications/Oquab15.pdf)中使用的全局最大池层(GMP ),因为与仅识别一个区别部分的 GMP 层相比，GAP 层有助于识别对象的完整范围。这是因为在 GAP 中，我们对所有激活取平均值，这有助于找到所有有区别的区域，而 GMP 层仅考虑最有区别的区域。

间隙层产生最后卷积层中每个单元的特征图的空间平均值，然后加权求和以产生最终输出。类似地，我们产生最后卷积层的加权和，以获得我们的类激活图。

# 简单工作

在定义了架构之后，让我们看看如何产生类激活图。让我们考虑一个图像，它包含一个我们已经为其训练了网络的对象。softmax 层输出模型被训练的各种类的概率。通过使用所有概率的 Argmax，我们找到最有可能是图像中存在的对象的类别。提取对应于该类别的最终层的权重。此外，提取来自最后一个卷积层的特征图。

![](img/ac9471bb5edf26bdfbafa05935146c0f.png)

Extracting weights from final layer and Building a Model to Output the feature maps as well as the final predicted class.

![](img/0f8526d84ace737cf2cf143597421cae.png)

Features represents the Feature Map obtained from the last layer and results represents the class probabilities for each class.

最后，计算从最终层提取的权重和特征图的点积，以产生类别激活图。通过使用双线性插值对类别激活图进行上采样，并将其叠加在输入图像上，以显示 CNN 模型正在查看的区域。

![](img/8c12d334f0a303c78a67abe790f620b4.png)

The code shows iterating through ten images, upsampling the extracted feature maps for the predicted class and finally performing dot product between the feature maps and final layer weights.

# 实施和结果:

我按照论文中的指导使用 Keras 实现了类激活映射。

最初，以下模型架构使用 3 个卷积层，每个卷积层之后是 max-pooling 层，最后一个卷积层之后是 GAP 层，最后一个输出层激活 softmax。

![](img/8c0711fae60595103b7fa517ec6afa54.png)

Architecture of First Model

![](img/b8ed8a070fee696b3c93d2e45eee27ae.png)

**Model Architecture**

![](img/a8c0e509fb1dda24bc12f971f9e42a71.png)

Model 1 Summary

上述模型在 MNIST 数据集上进行训练，并在训练、验证和最终测试集上产生了大约 99 %的准确度。

执行了上面提到的以下步骤，并且获得了如下所示的输出:

![](img/73715e9fd3ffdee921851db138499b12.png)![](img/2134179941d9388ddbeb1c2ebbdb92ba.png)![](img/8b0804d485d500e53b2f2b5d3d975809.png)![](img/a3ad268c9bd5fc311859509fe9e0054e.png)![](img/167a4f54d611b6ae6779065078c78533.png)![](img/f51c474ca78ddcff9b635a0571ee7cfb.png)

**Outputs**

使用原始方法获得的输出是令人满意的，但它仍然令人困惑，因为由于每个卷积层之后的后续最大池层丢失了空间信息，它并不完全清楚。

为了解决这个问题，我稍微调整了一下架构，删除了所有 max-pooling 层，从而保留了空间信息，这将有助于提高模型的本地化能力。但是，由于移除了最大池层，特征图的维数较大，因此训练时间也较长。

以下架构用于第二部分:

![](img/e43864146a53b2407ab89b49aac7c65e.png)

Architecture of Model 2

![](img/f92b713f1bb8bf4f918751a102a3d8f9.png)

Tweaked Model

![](img/133761cc003b08b7822cd5a9b49c235c.png)

Model 2 Summary

第二个模型在训练、验证和最终测试集上给出了 97%准确度。

以下输出是从调整后的模型中获得的:

注意:*在第二个模型中，没有应用最大池，因此在向前传递期间没有维度的减少。因此，我们不需要对从最后一个卷积层获得的特征图进行上采样。*

![](img/0fc7bc7991d08d5e07679f6cec7a984c.png)![](img/cf416ebe1448a47395e89245bbfca14d.png)![](img/f7ff7fc30829918e48fcce39581355bc.png)![](img/1e68543ece6c8b9d61d5e6a7e83f7b44.png)![](img/cc2ba72b72870fc260a84a8ff30aadb9.png)![](img/73972184243f108e5019481284faf634.png)

**The output of the Tweaked version**

# 结论:

类激活图或 cam 是查看 CNN 模型在训练时所见内容的好方法，并为开发人员和利益相关者提供见解，这对模型的可用性和生命周期至关重要。在 CAMs 的帮助下，开发人员可以看到模型正在查看图像中的哪个区域，并可以用于检查模型中偏差的原因、缺乏概括能力或许多其他原因，这些原因在修复后可以使模型在现实世界中更加健壮和可部署。

随着计算机视觉和深度学习领域研究的不断发展，各种深度学习框架，如 Tensorflow、Pytorch 等，不断向其环境中添加新的包，以方便用户，并专注于可解释性。Tensorflow 在可解释模型领域迈出了良好的一步，推出了一个名为 tf.explain 的特殊包，说明使用了哪些技术(如类激活映射等)为您实现和优化，您只需导入库，创建其对象，传递输入并在几秒钟内接收结果，而无需花费时间从头开始实现算法。

要了解如何使用 tf.explain 库，可以参考下面的 [*教程*](https://medium.com/google-developer-experts/interpreting-deep-learning-models-for-computer-vision-f95683e23c1d.) 作者[迪潘坚(DJ)萨卡尔](https://medium.com/@dipanzan.sarkar?source=post_page-----f95683e23c1d----------------------)。

希望你喜欢这篇文章，如果想了解更多信息和相关内容，请随时在 Linkedin 上联系我，或者在 twitter 上关注我。我的 Github 上有这篇文章的代码。