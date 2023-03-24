# 利用高斯混合变分自动编码器抵抗对抗性攻击

> 原文：<https://towardsdatascience.com/resisting-adversarial-attacks-using-gaussian-mixture-variational-autoencoders-be98e69b5070?source=collection_archive---------14----------------------->

深度神经网络太神奇了！他们能够通过查看 100 多万张图像来学习将图像分为不同的类别，在众多语言对之间进行翻译，将我们的语音转换为文本，制作艺术品(甚至可以在拍卖会上出售！)，并擅长大量其他令人兴奋和有用的应用程序。人们很容易被深度学习的成功故事所陶醉，但它们不会犯错吗？

事实证明，他们实际上是**极易被愚弄的**！:-(

针对深度神经网络的对抗性攻击的研究越来越多。已经提出了许多防御方法来保护模型免受这种攻击。不幸的是，几乎所有的方法都被证明是无效的。该领域的发展速度显而易见，因为在 ICLR 2018 年成果宣布后的几周内，一篇研究论文发表了，显示作者如何能够绕过被接受论文中提出的 8 种防御技术中的 7 种。2018 年 CVPR 奥运会上接受的关于防御对抗性攻击的论文遭遇了类似的命运！

![](img/a0f75031ed300fe5bab0250984aa0410.png)

We are all too familiar with this cute *gibbon…*errrr panda by now! Image courtesy: [http://www.cleverhans.io/assets/adversarial-example.png](http://www.cleverhans.io/assets/adversarial-example.png)

一项平行的研究表明，深度神经网络也可以很容易地被*【愚弄】*样本欺骗，这些样本是分布外的样本，与模型训练的任何类别都不相似。事实上，已经表明，简单地提供随机高斯噪声作为输入足以欺骗深度网络以高置信度预测其中一个类作为输出。正如“GAN 之父”伊恩·古德菲勒所说，*“深度神经网络几乎在任何地方都是错误的！”*

![](img/1347c10a89dd821f6bfd408730a1fd2b.png)

Each of these images are classified into the class mentioned with >99% confidence! Image courtesy: [http://www.evolvingai.org/fooling](http://www.evolvingai.org/fooling)

虽然这在一开始看起来非常令人惊讶，但让我们仔细检查一下。首先，我们来看看糊弄的样本。

基于深度神经网络的图像分类器的任务是学习将每个输入分类到 *K* 允许类的**1 中。然而，有一个问题。当我们提供一个不属于任何允许的类的输入图像时，模型应该做什么？在典型的机器学习设置中，潜在的假设是，训练和测试样本是从原始数据分布 *P(x，y)* 中抽取的。然而，**这个假设在当前场景中**被打破。**

理想情况下，我们希望网络预测类的均匀分布(当输出层是 softmax 层时)，以及每个类的接近 *0* 的概率得分(当输出层由 sigmoid 激活组成时)。然而，我们需要在这里停下来，问问我们自己——考虑到我的培训目标，我应该期望网络以这种方式运行吗？

在训练阶段，模型应该优化的唯一目标是由训练样本的预测误差确定的*经验风险函数*，这意味着目标函数中没有迫使网络学习非分布样本的期望行为的项。因此，毫不奇怪，在模型的输入空间中很容易找到导致对某些类别标签的高置信度预测的样本，尽管事实上它们实际上不属于任何类别——我们根本没有为这样的任务训练模型！解决问题的办法似乎是引入一个**【拒绝类】**的概念。

接下来，我们将通过几个图像来分析愚弄样本和敌对样本同时存在的原因。

![](img/8c9f632025e8c71d70a3ee17413068c6.png)

Let’s consider the simplistic scenario of an image in a 2-D space. If we randomly perturb the image in some direction, the classifier usually does a good job of handling the noise.

![](img/3d8bb54ce37652fd591e2fd0b4292bbc.png)

However, adversarial attacks find certain specific directions in the input space, such that the classifier is fooled into predicting an incorrect class label after perturbing the input in that direction.

![](img/e20a35bc06dd277703e9aa23479f7aad.png)

Similarly, fooling images can be found much more easily, such that they do not belong to any of the classes, but lead to high confidence predictions when provided as inputs to the classifier.

![](img/4e440ab1f89165c2cbb7b0cd3885d2f9.png)

If we try to analyze the reason for existence of such fooling samples more closely, it becomes clear that any input which does not lie in the region of the input space representing any of the classes, but lies sufficiently far away from the decision boundaries of the classifier, will lead to high confidence predictions corresponding to one of the classes. Further, if we look closely at the adversarial image depicted in the above figure, it becomes clear that unless the training data provides a very good representation of the true boundary of a given class, it might be possible to find certain adversarial directions for perturbing an input image. A 2-D depiction is provided here just for an idea - imagine how the possibility of finding such directions will increase as the input dimensionality increases! Now think of the input dimensionality in the case of image classifiers - it is probably not so surprising that adversarial samples exist!

如果我们可以为输入空间中的每个类设置某种**信封，那么我们可以将属于该信封的每个输入分类为属于该类，并拒绝不属于任何信封的每个输入，这不是很好吗？我们可以使用任何所需形状的信封，或者如果需要的话，对每一类输入使用多个信封。但问题是，我们如何得到这样的信封？事实证明，这是一个在原始输入空间中很难解决的问题。**

如果我们从概率论的角度来看，这个问题归结为给定训练数据的密度估计。如果我们知道 *P(x)* ，我们可以简单地拒绝任何输入样本 *x* ，对于这些样本 *P(x)* 可以忽略不计。但是，高维密度估计面临维数灾难！

在我们的建议中，我们使用变分自动编码器(VAE)来实现上述目标-同时降低维度并在较低的维度中执行密度估计。我们使用高斯混合分布作为潜在变量先验，混合分量的数量等于数据集中的类的数量。所提出的框架还允许我们在半监督场景中训练模型。请参考我们的[论文](https://arxiv.org/pdf/1806.00081.pdf)了解如何修改 ELBO 以说明分类标签，从而获得用于训练模型的目标函数。

![](img/c4e8da07d603c952207ed45d55ae6f96.png)

Training phase of our model is represented in this figure. f and g represent the encoder and decoder networks respectively, and they output the means of the two distributions Q(z|x) and P(x|z). *ϵ represents the noise added corresponding to the reparameterization trick used for training VAEs.*

一旦模型被训练，我们使用模型的潜在(编码器阈值)和输出(解码器阈值)空间中的阈值来拒绝输入样本。基于训练数据和训练模型来计算阈值，其中编码器阈值对应于不可忽略的 *P(z)* ，解码器阈值对应于不可忽略的 *P(x|z)* (在某些假设下)。这些阈值对应于我们之前讨论的使用信封的想法。这两个阈值的组合确保任何被分类的输入 *x* 在训练数据分布下具有不可忽略的概率 *P(x)* 。

![](img/cfe0fdb3ad4cf787f1f7c5547120d920.png)

Testing phase of the model is represented in this figure. The input image is passed through the encoder, and if the latent encoding lies within the encoder threshold from the mean of the predicted class, then the encoding is passed through the decoder network. If the reconstruction error is also within the the decoder threshold of the predicted class, then the predicted label is accepted. If any of the thresholds is not satisfied, then the input is rejected. It is important to note here that each of the circles in the latent space in the figure represent the Gaussian mixture component for the corresponding class.

由于我们已经在模型管道中引入了拒绝机制，现在让我们来看看使用和不使用阈值的模型的性能。鉴于现在这是一个**选择性分类模型**，我们必须确保引入拒绝机制后，测试数据的精度以及召回率都很高。

![](img/b6de80932e8b6aa8abe9002f4ba82d82.png)

It is noteworthy here that although there is a certain drop in accuracy after thresholding, the error percentages go down in each of the cases. For example, for MNIST data, the error percentage goes down from 0.33% to 0.22%, while 1.81% samples now move into the reject class.

现在，让我们看看一些被我们的模型错误分类的样本。

![](img/a9973737520d2be55829dc1b1a46e99b.png)

The first label indicates the true label from the test dataset, and the second label indicates our model’s prediction. In most of the cases, it is clear that the labels predicted cannot be claimed to be “wrong”, since it is not clear even to a human which of the two labels should be “correct”.

接下来，我们将深入我们的模型，探索有助于拒绝敌对和愚弄输入的关键特征，而不是对它们进行错误分类。首先需要注意的是，该模型的编码器部分非常类似于通常的基于 CNN 的分类器。因此，我们可以使用现有的敌对/愚弄攻击来欺骗编码器，类似于对 CNN 的攻击。然而，解码器网络是模型鲁棒性的主要来源。重要的是要注意，解码器的输入维数明显小于编码器的输入维数(例如，对于 MNIST 数据，10 比 28×28 = 784)。此外，因为我们在潜在空间中执行阈值处理，所以我们只允许解码器接受来自该空间非常有限的部分的输入。在训练时，解码器输入空间的这个区域被密集采样，即，给定来自这个区域的输入，可以期望解码器仅生成有效的输出图像。这是我们的模型的对抗性鲁棒性的据点。

> 给定训练数据中的类在视觉上是不同的，在给定来自我们的模型的潜在空间中的两个不同高斯混合分量的两个输入的情况下，可以期望解码器生成具有高欧几里德距离的图像。只要该距离大于相应类别的解码器阈值，解码器将总是能够检测到对模型的对抗性攻击。

![](img/2679301b8b18598b818417a477fe62ac.png)

This figure represents the evolution of the latent space structure for the model for a single class. Existence of “holes” in the input space of the decoder is highly improbable, given the training objective. Image courtesy: [https://arxiv.org/pdf/1711.01558.pdf](https://arxiv.org/pdf/1711.01558.pdf)

现在，让我们来看看如何通过阈值处理来抵御敌对样本。

![](img/d545d2d3a06e467f3afa26104b2b5e98.png)

Suppose the input image (from class 1) has been adversarially perturbed to fool the encoder into believing that it belongs to class 0\. This implies that the latent encoding of the input image must lie within the cluster for 0’s. Although it is easy for an adversary to achieve this objective, the decoder is where it becomes tricky. Once the latent encoding is passed through the decoder network, the output image now resembles a 0, and thus, the reconstruction error is invariably high. This results in rejection of the input image!

接下来，我们提供在编码器网络上运行 FGSM 攻击的结果。随着我们增加噪声幅度，模型的精度持续下降，但是在整个ϵ值范围内(对于监督模型)*，误差百分比保持在 4%以下。如所期望的，随着ϵ的增加，拒绝百分比从 0%上升到 100%，而不是敌对输入被错误分类。对于半监督的情况也观察到类似的趋势，但是，如预期的，模型性能不如监督情况下的好。*

![](img/1b58d5380a88331488e91de9dee334b2.png)

Results corresponding to the FGSM attack with varying *ϵ on the encoder of the model trained on MNIST dataset.*

我们还使用 Cleverhans 库尝试了许多其他强攻击，如 PGD、BIM、MIM 等。使用默认参数，但是在每种情况下，所有生成的对立样本都被拒绝。由于这些攻击缺乏解码器网络的知识，我们接下来将注意力转移到完全白盒攻击，其中敌对目标是利用两个阈值的知识专门设计的。我们还设计了类似的白盒愚弄目标，并在随机选择的测试数据集子集上运行这些攻击(对抗性攻击)，或在多个开始随机图像上运行这些攻击(愚弄攻击)。然而，在每种情况下，优化过程收敛的图像被我们的模型基于两个阈值成功地拒绝。有关这些敌对目标的详细信息，请参考我们的[文件](https://arxiv.org/pdf/1806.00081.pdf)。

![](img/7fbd53ad20301b2ab9452207bb603712.png)

White-box adversarial attack objective

![](img/db72625e5e7c6402f19cbe0e43d96381.png)

White-box fooling attack objective

最后，为了解决敌对样本进入拒绝类而不是被正确分类的问题，我们引入了仅基于解码器网络的重新分类方案。该方法背后的基本思想是找到最大化 *P(x|z)* 的 *z* ，即，我们对解码器网络的输入执行梯度下降，以最小化给定对抗图像的重建误差。然后，我们选择与最优 *z* 所在的集群相对应的标签。如果这个 *z* 超出相应类别均值的编码器阈值，我们再次将输入作为非分布样本拒绝。按照这种重新分类方案，我们在通过对具有变化ϵ.的编码器进行 FGSM 攻击而生成的敌对图像上获得了以下结果

![](img/4f8d4b8e904727a80bfb353ef18ea6ba.png)

这些结果意味着重新分类方案能够以非常高的准确度成功地将检测到的对抗性输入分类到正确的类别中。即使对于 0.3 的最高ϵ，解码器网络也能够正确地对 87%的样本进行分类，而所有这些样本先前都被拒绝。

有趣的是，在拒绝的测试数据样本上遵循相同的重新分类方案，我们也获得了先前报告的准确度值的改进。例如，对于 MNIST 数据集，精度值从 97.97%上升到 99.07%，因为 181 个被拒绝的样本中的 110 个被重新分类方案正确分类。因此，该方法进一步提高了测试数据集的召回率。然而，重要的是要注意，由于重分类方案涉及优化步骤，因此直接在所有输入样本上使用它是不实际的。因此，我们建议在通常情况下使用编码器网络进行分类，并且只对被拒绝的输入样本使用重新分类方案。

有趣的是，我们最近看到了一篇关于非常相似的想法的论文，该论文使用了 Geoffrey Hinton 教授团队的胶囊网络以及检测敌对样本，可以在这里找到。在贝斯吉实验室的这个[链接](https://arxiv.org/pdf/1805.09190.pdf)上也可以找到另一个类似的工作。

再一次，这篇文章对应的论文可以在下面的[链接](https://arxiv.org/pdf/1806.00081.pdf)找到。这项工作是由 MPI-IS 和 IBM Research AI 合作完成的。该论文已被第 33 届 AAAI 人工智能会议(2019 年，AAAI)接受发表。请在下面留下任何问题/建议/反馈，我们将很高兴与您联系！感谢您的时间和兴趣！