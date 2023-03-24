# 具有深度生成模型的插值

> 原文：<https://towardsdatascience.com/interpolation-with-generative-models-eb7d288294c?source=collection_archive---------3----------------------->

## 生成模型如何学习创造新事物

在这篇文章中，我将写关于生成模型。它将涵盖生成模型和判别模型之间的二分法，以及生成模型如何通过能够执行插值来真正了解感兴趣对象的本质。

# 0.生成模型(G)与判别模型(D)

老实说，在生成对抗网络(GANs)起飞后，我才开始思考统计和机器学习模型的本质。在 GAN 的原始版本中，让我们称之为香草 GAN，你有一个生成网络(G ),它从高斯噪声中生成合成数据，还有一个鉴别网络(D ),它试图区分真假。显然，香草甘中的 G 和 D 分别是生成模型和判别模型。事实上，GAN 可能是第一个协调生成和判别模型的 ML 算法，它通过创新的对抗训练来学习两个模型的参数。

![](img/a89a6ed9f32741a5f34cbf2f2f654621.png)

*Image source:* [*https://www.slideshare.net/ckmarkohchang/generative-adversarial-networks*](https://www.slideshare.net/ckmarkohchang/generative-adversarial-networks)

我自己的经验就说这么多，什么是生成式和判别式模型？直觉上，生成模型试图抽象出一些对象集合的一些可概括的模式，而鉴别模型试图发现集合之间的差异。具体来说，在分类问题的背景下，例如，生成模型将学习每个类的特征，而判别模型将找到最好地分离类的决策边界。更正式地说，让我们将实例表示为由一些标量值 *y、*标记的特征向量***【x】***生成模型学习**联合**概率分布*p(****x****，y)、*而判别模型学习**条件**概率分布*(y |****x***

还有一些有趣的生成器-鉴别器对可以考虑:

*   二元分类:朴素贝叶斯与逻辑回归
*   序列建模:隐马尔可夫模型与条件随机场

值得一提的是，大多数传统的最大似然分类器都是判别模型，包括逻辑回归、SVM、决策树、随机森林、LDA。判别模型在需要学习的参数方面很少，并且已经被证明在许多分类任务中比它们的生成模型具有更好的性能。

但是我想说的是，学习区分一个类和另一个类并不是真正的学习，因为当处于另一个环境中时，它通常不起作用。例如，当一个看不见的类(狗)被添加到测试集中时，被训练来以异常的准确度区分猫和鸟的鉴别分类器可能会悲惨地失败，因为鉴别分类器可能简单地知道有四条腿的东西是猫，否则是鸟。

为了进一步说明生成性和判别性模型*真正*学到了什么，让我们考虑一下最简单的分类模型，朴素贝叶斯和逻辑回归。下图显示了朴素贝叶斯和逻辑回归分类器在二元分类问题上获得的“知识”。

![](img/bf309d0b51605ec6cf99fc4a1db2b85a.png)

朴素贝叶斯分类器学习两个类的均值和方差向量，而逻辑回归学习线性边界的斜率和截距，以最佳方式分隔两个类。利用从朴素贝叶斯分类器中学习的均值和方差，我们可以通过从多元高斯分布中采样来为每个类生成合成样本。这类似于使用 GANs 生成合成样本，但显然朴素贝叶斯无法生成任何高质量的高维图像，因为它太幼稚，无法独立地对特征进行建模。

# 1.生成模型

我简单地提到了朴素贝叶斯算法，它可以说是生成模型的最简单形式。现代生成模型通常涉及深度神经网络架构，因此称为深度生成模型。有三种类型的深度生成模型:

*   可变自动编码器(VAE)
*   开始
*   基于流程的生成模型([一个关于这类模型的优秀博客](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)

## 1.1.VAE

VAE 由[金玛&韦林，2014](https://arxiv.org/abs/1312.6114) 推出，作为自动编码器(AE)的概率扩展。与 vanilla AE 相比，它有以下三个附加功能:

1.  概率编码器 qϕ( **z** | **x** 和解码器 pθ( **x** | **z**
2.  潜在空间(AE 的瓶颈层)的先验概率分布:pθ( **z**
3.  由 [Kullback-Leibler 散度](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence)定义的潜在损失:d(qϕ(**z**|**x**)‖pθ(**z**|**x**))来量化这两个概率分布之间的距离

![](img/0a417bf76b314909cfcd1a94020fdf11.png)

*VAE illustration from* [*https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html*](https://lilianweng.github.io/lil-log/2018/08/12/from-autoencoder-to-beta-vae.html)

## 1.2.甘斯

GAN 是由 [Goodfellow et al .，2014](https://arxiv.org/abs/1406.2661) 介绍的，由一对生成器和鉴别器网络组成，彼此进行一场极小极大博弈。GAN 的许多变体已经被开发出来，例如双向 GAN(甘比)、CycleGAN、InfoGAN、Wasserstein GAN 和[这个列表还在继续增长](https://medium.com/@jonathan_hui/gan-a-comprehensive-review-into-the-gangsters-of-gans-part-1-95ff52455672)。

**特别吸引人，因为它明确地学习一个编码器网络，*E(****)x****)*将输入映射回潜在空间:**

**![](img/68335d45bc1b84174cd98df9932ffd9d.png)**

**Figure source: [*Donahue et al, 2016 Adversarial Feature Learning*](https://arxiv.org/abs/1605.09782)**

# **2.生成模型插值**

**有了一些深层生成模型的知识，我们将检查它们的能力。生成模型能够学习来自不同类别的样本的低维概率分布。这种概率分布可用于监督学习和生成合成样本。虽然这些功能非常有用，但让我印象更深的是生成模型能够沿着任意轴对真实样本执行插值，以生成不存在的操纵样本。例如，深度生成模型可以[沿着年龄、性别、头发颜色等轴操纵人脸图像](https://blog.openai.com/glow/)。在我看来，这表明深度生成模型能够获得想象的能力，因为[想象是产生心理图像的过程](https://en.wikipedia.org/wiki/Imagination)。接下来让我们深入研究如何执行插值。**

**插值通过在生成模型学习的潜在空间( ***z*** )中执行简单的线性代数来工作。首先，我们想在潜在空间中找到一个轴来进行插值，它可以是类似生物性别的东西。生物性别的插值向量可以简单地计算为潜在空间中从雄性质心指向雌性质心的向量。**

**更一般地，我们首先需要在潜在空间中找到两类质心( *a* ， *b* ):**

**![](img/bb24d7ef7a8ac7300aa6de8848c83470.png)****![](img/382aea287a1c5c702615d5daca7ddff2.png)**

**从类别 *b* 指向类别 *a* 的潜在空间中的插值向量为:**

**![](img/e09cff2c4c5234fa24b25fc4853587ea.png)**

**给定任何类别***x _****c*的任何看不见的样本，我们可以通过以下方式用插值向量操纵看不见的样本:1)将样本编码到潜在空间中；2)在潜在空间中执行线性插值；以及 3)将内插样本解码回原始空间:**

**![](img/dc6b75677d24ad3f998b0714496b9dbe.png)****![](img/6cee12de27ab4a3dd2b85c438d1ecb04.png)**

**上式中的 *α* 是决定插值大小和方向的标量。接下来，我将围绕 *α* 沿着不同的插值向量滑动。以下 Python 函数可使经过训练的创成式模型执行此类插值:**

# **3.基于 MNIST 数据的生成模型实验**

**我在 MNIST 手写数字数据集上训练了一些生成模型，包括朴素贝叶斯、VAE 和甘比，以实验插值。下图显示了瓶颈层只有两个神经元的 VAE 的潜在空间。虽然不同的数字有不同的模式，但重建质量很差。也许将 784 维空间压缩到 2 维空间是一个挑战。我发现瓶颈层有 20 个神经元的 VAE 可以重建质量不错的 MNIST 数据。**

**![](img/edaac7a8c169a4a4162242fef19eb868.png)**

**Latent space learned by a VAE with 2 neurons at the bottleneck layer**

**还值得指出的是，生成模型的训练是不受监督的。因此，学习的潜在空间不知道类别标签。插值向量是在模型完成学习后计算的。**

**为了进行插值，我首先可视化了 10 个数字的所有 45 个可能对之间的插值向量:**

**![](img/0aa0ff9e7aeb5da4af68041e02a5242b.png)**

**Visualization of the interpolation vectors of MNIST digits in the latent space of a VAE with 20 neurons at the bottleneck layer**

**在上图中，每行对应一个从一个数字指向另一个数字的插值向量，而每列对应一个 alpha 值。从左到右观察潜在空间产生的数字，看一个数字如何逐渐变成另一个数字，这很有趣。由此，我们还可以找到位于 10 位数字的两个质心之间的模糊数字。**

**接下来，我做了另一个有趣的插值实验:我问我们是否可以通过沿着 6->0 向量移动数字 7 来将它变成数字 6 或 0。以下是生成图像的结果。它在右边显示了一些看起来相对 0 的图像，而左边的看起来一点也不像 6。**

**![](img/c5e8c63318e83b3def47c2833ea0c728.png)**

**也可以使用在 MNIST 上训练的逻辑回归分类器来量化这些图像，以预测标记的概率。分类器非常符合我们目测图像的感觉。**

**![](img/ec971251b29a36128c7620994955c7ed.png)**

**Predicted probability for the images of interpolated digits from a Logit classifier**

**用 MNIST 数据集进行的看似无聊的概念验证实验展示了深度生成模型的想象能力。我可以想象插值的许多实际应用。**

**如果你想了解更多的技术细节，这篇文章基于我的 GitHub repo:**

**[](https://github.com/wangz10/Generative-Models) [## Wang 10/创成式模型

### MNIST -王 10/生成模型实验深度生成模型教程

github.com](https://github.com/wangz10/Generative-Models) 

这篇文章的笔记本版本在 Ma'ayan 实验室会议上发表:

[](https://nbviewer.jupyter.org/github/wangz10/Generative-Models/blob/master/Main.ipynb) [## Jupyter 笔记本浏览器

### 看看这个 Jupyter 笔记本！

nbviewer.jupyter.org](https://nbviewer.jupyter.org/github/wangz10/Generative-Models/blob/master/Main.ipynb) 

# 参考

*   [Ng AY & Jordan MI:论区别量词与生成量词](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf)
*   金玛&韦林:自动编码变分贝叶斯
*   [古德菲勒·IJ 等人:生成敌对网络](https://arxiv.org/abs/1406.2661)
*   [多纳休等人:对抗性特征学习](https://arxiv.org/abs/1605.09782)
*   [基于流程的深度生成模型](https://lilianweng.github.io/lil-log/2018/10/13/flow-based-deep-generative-models.html)
*   [辉光:更好的可逆生成模型](https://blog.openai.com/glow/)**