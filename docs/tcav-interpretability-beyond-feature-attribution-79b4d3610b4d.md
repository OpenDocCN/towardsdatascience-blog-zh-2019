# TCAV:特征归因之外的可解释性

> 原文：<https://towardsdatascience.com/tcav-interpretability-beyond-feature-attribution-79b4d3610b4d?source=collection_archive---------9----------------------->

## GoogleAI 的模型可解释性技术在人性化概念方面的概述。

![](img/f7c0daa216ca7b66021230252441ee3f.png)

[How convolutional neural networks see the world](https://blog.keras.io/how-convolutional-neural-networks-see-the-world.html)

> 仅仅知道一个模型是否有效是不够的，我们需要知道它是如何工作的:桑德尔·皮帅

今天的重点正慢慢转向模型的可解释性，而不仅仅是模型预测。然而，可解释性的真正本质应该是让机器学习模型更容易被人类理解，特别是对于那些不太了解机器学习的人。机器学习是一种强有力的工具，伴随这种能力而来的是确保公平等价值观在模型中得到很好反映的责任。确保人工智能模型不会强化现实世界中存在的偏见也是至关重要的。为了解决这些问题，谷歌人工智能研究人员正在研究一种叫做 **TCAV** 的解决方案(用概念激活向量进行测试)，以了解神经网络模型使用什么信号进行预测。

# 目标

![](img/b456671866ac1b17eb4054f4ea191986.png)

[Google Keynote (Google I/O’19)](https://www.youtube.com/watch?v=lyRPyRKHO8M&t=3408s)

在**谷歌 I/O 2019** 、**、**的[主题演讲](https://www.youtube.com/watch?v=lyRPyRKHO8M&t=3408s)中，桑德尔·皮帅谈到了他们如何试图为每个人建立一个更有帮助的谷歌，包括为每个人建立人工智能。他重申，机器学习中的偏见是一个令人担忧的问题，当涉及到人工智能时，风险甚至很高。为了让人工智能更加负责和透明，他讨论了 TCAV 方法，通过这篇文章，我将概述这一方法以及它打算如何解决偏见和公平的问题。这篇文章在数学方面会很轻，所以如果你想更深入地了解，你可以阅读[原始研究论文](https://arxiv.org/abs/1711.11279)或者访问 TCAV 的 [Github 知识库](https://github.com/tensorflow/tcav)。

# 需要另一种可解释性技术

在 ML 领域，主要有三种可解释性技术:

![](img/e14260a29d8dbb165284a98554b8c3ae.png)

Types of Interpretability Techniques

大多数情况下，你会得到一个由多年工程和专业知识创建的模型，你不能改变它的架构，也不能重新训练它。那么，你如何着手解释一个你毫无头绪的模型呢？ **TCAV** 是一种旨在处理此类场景的技术。

大多数机器学习模型被设计成对底层特征进行操作，比如图片中的边缘和线条，或者说单个像素的颜色。这与人类更熟悉的**高层概念**非常不同，就像斑马身上的条纹。例如，如果您有一幅图像，该图像的每个像素都是一个输入要素。尽管可以观察每个像素并推断出它们的数值，但它们对人类来说毫无意义。我们不会说这个图像的第 5 个像素的值是 28；作为人类，我们总说图中有一条蓝色的河。TCAV 试图解决这个问题。

此外，典型的可解释性方法要求你有一个你有兴趣理解的特定图像。TCAV 解释说，这通常是真实的一类利益超过一个图像(全球解释)。

# TCAV 方法

假设我们有一个模型，它被训练从图像中检测出**斑马**。我们想知道哪些变量在决定图像是否是斑马时起了作用。TCAV 可以帮助我们理解条纹的概念是否对模型的预测至关重要，在这种情况下实际上是肯定的。

![](img/38efa21b69687c568b53744cf44b4931.png)

[TCAV shows that stripes are a critical ‘concept’ when deciding if an image contains a zebra or not](https://www.youtube.com/watch?v=lyRPyRKHO8M&t=3408s)

类似地，考虑一个根据医生图像训练的分类器。如果训练数据主要由穿着白大褂和听诊器的男性组成，则模型会假设穿着白大褂的男性是成为医生的一个重要因素。这对我们有什么帮助？这将带来训练数据中的偏见，其中女性的图像较少，我们可以很容易地纠正这一点。

![](img/3121fbbfe0cc6c88c069e6b6d8986d71.png)

[TCAV shows that being male is an important ‘concept’ when deciding if an image belongs to a doctor or no](https://www.youtube.com/watch?v=lyRPyRKHO8M&t=3408s)t

# 那么什么是 TCAV 呢？

用概念激活向量(TCAV)进行测试是来自谷歌人工智能团队的一项新的可解释性倡议。概念激活向量(CAV)根据人类友好的概念提供了对神经网络内部状态的解释。TCAV 使用方向导数来量化用户定义的想法对分类结果的重要程度——例如，“斑马”的预测对条纹的存在有多敏感。

由 Been Kim 和 Martin Wattenberg、Justin Gilmer、Carrie Cai、James Wexler、Fernanda Viegas 和 Rory Sayres 开创的团队旨在让机器学习赋予人类的能力被它淹没。这是他对可解释性的看法。

[Source: Quanta Magazine](https://youtu.be/8Bi-EhFPSLk)

# 工作

TCAV 本质上是从例子中学习概念。例如，TCAV 需要一些“女性”和“非女性”的例子来学习“性别”概念。TCAV 的目标是确定某个概念(如性别、种族)对已训练模型中的预测有多必要，即使该概念不是训练的一部分。

继续“斑马分类器”，考虑神经网络由输入 x ∈ R^ n 和具有 **m** 个神经元的前馈层 **l** 组成，从而输入推理及其层 **l** 激活可以被视为一个函数:

![](img/c26fe637c2d175e24d943174be8864e7.png)![](img/d0377c3bd238319a323bfec129aeeb6a.png)

[Testing with Concept Activation Vectors](https://arxiv.org/pdf/1711.11279.pdf)

*   **定义兴趣概念**

对于代表这一概念的一组给定的例子(例如，条纹)( **a** )或带有概念标签( **b** )和训练过的网络( **c** )的独立数据集，TCAV 可以量化模型对该类概念的敏感度。

*   **寻找概念激活向量**

我们需要在层 l 的激活空间中找到一个代表这个概念的向量。CAV 通过训练线性分类器来学习，以区分由概念的示例和任何层中的示例产生的激活( **d** )。然后，我们定义一个“概念激活向量”(或 CAV)作为超平面的法线，将模型激活中没有概念的例子和有概念的例子分开

![](img/39e4b173eaeff811054b23ca2258d54e.png)

*   **计算方向导数**

对于感兴趣的类别(斑马)，TCAV 使用方向导数 SC，k，l(x)来量化概念敏感度( **e** )。这个 SC，k，l(x)可以定量地测量模型预测对任何模型层上的概念的敏感性

以下是在工作流程中使用 TCAV 的分步指南:

[](https://github.com/tensorflow/tcav/blob/master/Run%20TCAV.ipynb) [## 张量流/tcav

### 通过在 GitHub 上创建帐户，为 TensorFlow/tcav 开发做出贡献。

github.com](https://github.com/tensorflow/tcav/blob/master/Run%20TCAV.ipynb) 

# 洞察力和偏见

TCAV 用于两个广泛使用的图像预测模型，即 InceptionV3 和 GoogleNet。

![](img/3e7d175f3214dde5ea4999f83cfff479.png)

[Source](https://beenkim.github.io/slides/TCAV_ICML_pdf.pdf)

虽然这些结果显示了 red 概念对于消防车的重要性，但一些结果也证实了模型中对性别和种族的固有偏见，尽管没有明确受过这些类别的训练。例如:

*   乒乓球和橄榄球与特定的比赛高度相关
*   手臂概念比其他概念更能预测哑铃级别。

# 结论

TCAV 是朝着创建深度学习模型内部状态的人类友好线性解释迈出的一步，以便关于模型决策的问题可以根据自然的高级概念来回答。