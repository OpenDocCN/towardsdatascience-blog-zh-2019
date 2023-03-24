# Wasserstein GAN 在 Swift for TensorFlow

> 原文：<https://towardsdatascience.com/wasserstein-gan-in-swift-for-tensorflow-61b557bd8c63?source=collection_archive---------12----------------------->

![](img/2e8dbaafe1d0cd6b74c69ddd65886d88.png)

Vanilla Generative Adversarial Network (GAN) as explained by [Ian Goodfellow](https://en.wikipedia.org/wiki/Ian_Goodfellow) in [original paper](https://papers.nips.cc/paper/5423-generative-adversarial-nets.pdf).

我是[苹果 Swift](https://swift.org/) 和[深度神经网络](https://en.wikipedia.org/wiki/Deep_learning)的*大粉丝*。而最近即将推出的深度学习框架是 TensorFlow 的[Swift](https://www.tensorflow.org/swift/)。所以，很明显，我必须马上投入进去！我已经在 TensorFlow 或 PyTorch 中写过 Wasserstein GAN 和其他 GAN，但这个 Swift for TensorFlow 的东西超级酷。在后端，从编译器的角度来看，这是使 Swift 成为机器学习语言的最终努力。在本帖中，我将分享我在 Swift 为 TensorFlow 编写和培训 Wasserstein GAN 的工作。代码在 GitHub 上是开源的，现在就可以在 Google Colab 上运行！

# 生成对抗网的历史

G 生成对抗网络(GAN)是由 [Ian Goodfellow](https://en.wikipedia.org/wiki/Ian_Goodfellow) 在 2014 年发明的。GAN 通常有两个神经网络，即生成器 **G** 和评价器 **C** 。唯一可用的数据是真实世界&实值数据(来自自然)的未标记集合，可以是图像、音频等。GANs 是为改进真实数据建模而设计的，这样当模型被要求生成图像时，它应该能够这样做，这就是 **G** 的用途。这里， **C** 帮助 **G** 学习生成更真实的数据，自己学习预测 **G** 生成的图像是假的。它也接受真实的图像，并学会称之为真实的图像。这是一个迭代过程，它提高了 **C** 预测虚假和真实数据的能力，并反过来帮助 **G** 调整其参数，从而生成更真实的数据。

这些香草甘不产生非常好的图像质量。因此，提高图像生成质量的工作仍在继续，该子领域中最重要的方向之一是将 Critic 网络约束在函数空间的 1-Lipschitz 集合中，并最小化 **G** 分布(假)和 **P** 分布(真)之间的 Wasserstein 距离。查看[维基百科页面](https://en.wikipedia.org/wiki/Lipschitz_continuity)了解 Lipschitz 连续性。现在我们继续用 Swift 为 TensorFlow 编码 WGAN！

# 数据

数据是神经网络学习的第一步。因此，我使用了 CIFAR-10 数据集，它包含以下 10 个类别的图像:

*   飞机
*   汽车
*   伯德
*   猫
*   鹿
*   狗
*   青蛙
*   马
*   船
*   卡车

每个图像都是一个`32x32`大小的 RGB 图像。大约有 50k 的训练图像和 10k 的测试图像。哇，我从来没有注意到图像类的数量接近 MNIST 数据集中的数量🤔。总之，我用这些数据来训练我的 Wasserstein GAN 生成这样的图像。

```
import TensorFlow
import Python
PythonLibrary.useVersion(3)// Import some Python libraries
let plt = Python.import("atplotlib.pyplot")
```

Data downloading and loading

首先使用 Swift for TensorFlow toolchain 导入 TensorFlow 和 Python(3 . x 版)。然后通过 Python 互用性特性导入 Python 库！现在定义一些 CIFAR-10 下载、加载和预处理函数。最后，加载数据集。

## 配置

为了训练网络，需要设置一些重要的配置。我尽量保持配置与 [WGAN 纸](https://arxiv.org/abs/1701.07875)相似。因此，我将批量大小设置为 64，图像大小调整为 64x64，通道数(RGB)为 3。WGAN 被训练了 5 个纪元(如 PyTorch 教程中所建议的)。 **G** 的潜在空间被设定为 128 维。每个 **G** 的 **C** 的迭代次数被设置为 5，这是为了很好地近似 1-Lipschitz 函数，如论文中所建议的。另外， **C** 的可训练参数值必须限制在极限值[-0.01，0.01]。

Configurations

# 瓦瑟斯坦生成对抗网络

如上所述，WGAN 的模型包含一个 **C** 和 **G** 网络。 **C** 包含多个卷积层，而 **G** 由顺序转置卷积层组成，这些卷积层有时也被错误地称为反卷积层。

## 自定义图层

在 Swift 中为 TensorFlow 定制神经层，使你的结构符合`[Layer](https://www.tensorflow.org/swift/api_docs/Protocols/Layer)`协议。Swift 中 TensorFlow 的参数是通过使您的神经结构符合`[KeyPathIterable](https://www.tensorflow.org/swift/api_docs/Protocols/KeyPathIterable.html)`协议来访问的，这是默认的，但我写它是为了记住 Swift 中类型属性的迭代是如何发生的。目前，用于 TensorFlow 的 Swift 中的`[TransposedConv2D](https://www.tensorflow.org/swift/api_docs/Structs/TransposedConv2D)`实现工作不太好，所以我决定按照 [Odena 等人 2016](https://distill.pub/2016/deconv-checkerboard/) 建议的方式，在`Conv2D`层之后使用`[UpSampling2D](https://www.tensorflow.org/swift/api_docs/Structs/UpSampling2D)` op。使用`[Conv2D](https://www.tensorflow.org/swift/api_docs/Structs/Conv2D)`结构，因为它跟随`[BatchNorm](https://www.tensorflow.org/swift/api_docs/Structs/BatchNorm)` op，该 op 也在连续的`UpSampling2D`和`Conv2D`操作之后使用，代替`TranposedConv2D`。我写的这些自定义层用 Swift 代码显示如下。

Custom neural layers

## WGAN 架构

**G** 网络从高斯分布中取形状为【128】的随机向量。这将在`BatchNorm`和`relu`激活功能之后通过一个密集(全连接)层。产生的转换输出被重新整形，使得它可以通过一系列的`UpSampling2D`、`Conv2D`、`BatchNorm`层和`relu`激活。最后的层块简单地对卷积层和`tanh`激活函数之后的输出进行上采样。

网络 **C** 的架构是这样的，它有 4 个块`Conv2D`，后面跟着`BatchNorm`。每个块后面都有一个负斜率为 0.2 的`leakyReLU`激活函数。在撰写本文时，`leakyReLU`功能尚未在 Swift for TensorFlow 中实现，因此我通过使其可区分来实现自己的功能。最后，输出被展平，并通过产生[1]维输出的密集层，以给出图像是真/假的概率。

Wasserstein GAN architecture

请注意，在 **G** 和 **C** 中，步幅为(2，2)，填充设置为相同。两者中的内核大小都被设置为(4，4)大小。在上面的两个网络中，`call(_:)`函数使用`@differentiable`属性变得可微。用类似的方法创建可微的 LeakyReLU 激活函数。

Differentiable Leaky ReLU

# 培养

网络被训练 5 个时期。在每次迭代中, **C** 比 **G** 的单个训练步长多训练 5 次。每个网络使用的批次大小为 64。我使用 RMSProp 优化器，两个网络的学习率都是 0.00005。 **G** 的`zInput`取自均匀分布。我还记录了 1 个纪元的训练时间，对我来说大约是 12 分钟。使用 [Matplotlib](https://matplotlib.org) 还绘制了 Wasserstein 距离和几个损失(如 **G** 损失&T21C 损失)的图表，这是因为 Swift for TensorFlow 支持 Python 互操作性。关于 TensorFlow 的 Swift，我学到的另一件事是，您可以迭代任何任意类型的属性，这些属性符合在 [apple/swift](https://github.com/apple/swift) 的 [tensorflow/swift](https://github.com/tensorflow/swift) fork 中实现的[keypathiable](https://www.tensorflow.org/swift/api_docs/Protocols/KeyPathIterable)协议。这是一个超级酷的想法，但许多工作仍然需要做，如访问特定层的参数和修改需要应用每层激活。

不管怎样，下面是一个更深入的自我解释的 Swift 代码，用于训练我的 Wasserstein GAN！

注意，需要通过下面几行代码来设置训练/推理模式。

```
// For setting to training mode
Context.local.learningPhase = .training
// If you want to perform only inference
Context.local.learningPhase = .inference
```

# 讨论

用 Swift 语言接触深度神经网络是一种很好的体验。我认为 Swift for TensorFlow 将成长为一个新的主流机器学习框架，很可能取代 PyTorch 的大部分功能，就像 PyTorch 过去对 vanilla TensorFlow 所做的那样。我真的很喜欢它支持作为默认行为的急切执行，而且我不必手动设置设备进行训练&我还可以在 Google Colab 中使用云 TPU。很高兴看到 TensorFlow 社区通过提供基于 Jupyter 的 Swift 环境，努力使 Swift for TensorFlow 可在 Colab 上使用。也可以使用`Raw`类型的名称空间(我想它更像结构，因为 Swift 中不存在名称空间的概念)来访问一些基本的操作，如`mul(_:_:)`、`add(_:_:)`等。在 Swift for TensorFlow。这是谷歌 TensorFlow 团队[克里斯·拉特纳](https://en.wikipedia.org/wiki/Chris_Lattner) &斯威夫特的一项令人敬畏的努力。

以下是我对 Swift for TensorFlow 目前在研究原型方面提供灵活性的方式的部分担忧。

## 访问每层的参数

Swift for TensorFlow 采用了一种新颖的方法来访问一种类型的属性以进行修改。这是访问和更新神经网络参数所必需的。该设备提供给符合[按键可变](https://www.tensorflow.org/swift/api_docs/Protocols/KeyPathIterable)协议的类型。默认情况下，符合[层](https://www.tensorflow.org/swift/api_docs/Protocols/Layer)协议的神经网络结构存在这种行为。所以，你不必一遍又一遍地写，因为很可能你会想要访问神经网络的属性。这工作得很好，正如我在 Wasserstein GAN 代码中所做的那样，但我仍然不能灵活地访问每一层的参数，但我只是迭代一次所有属性，而不知道它们当前属于哪一层。这实际上不是我的要求，但在其他情况下，比如你想进行迁移学习时，这可能是一个要求。它需要访问特定的层来更新参数。希望 Swift 能很快为 TensorFlow 提供这种灵活性。

## 每层激活函数

我认为谷歌将会并且可能需要改变激活应用到每一层的方式(这是我不用的，因为 LeakyReLU 不能那样应用)。它类似于`Conv2D(…, activation: relu)`，其中 relu 也可以被线性激活代替，但 LeakyReLU 不太适合这种设计，因为它没有给定的斜率值。这里，函数没有被调用，它只是一个作为参数传递给 Conv2D 层或任何其他层的函数。我能想到的最好的解决方案是使用枚举而不是像`Conv2D(…, activation: .relu)`那样传递函数，或者对 LeakyReLU 做类似`Conv2D(…, activation: .leakyrelu(0.2))`的事情，其中 0.2 是激活函数的枚举情况`leakyrelu`的关联值。

*我希望你喜欢这本书。如果你真的喜欢我的文章，请与你的朋友分享，关注我吧！除了写文章，我还积极地在推特上发关于机器学习、区块链、量子计算的微博*[***@ Rahul _ Bhalley***](https://twitter.com/Rahul_Bhalley)*！*