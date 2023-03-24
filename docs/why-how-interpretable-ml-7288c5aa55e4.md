# 为什么&如何:可解释的 ML

> 原文：<https://towardsdatascience.com/why-how-interpretable-ml-7288c5aa55e4?source=collection_archive---------25----------------------->

![](img/9c083fce572d29f1d2d342f2cd00c945.png)

Explanation of classification ‘tabby’ as produced by LRP. Photo by author, heatmap implementation available at [https://lrpserver.hhi.fraunhofer.de/image-classification](https://lrpserver.hhi.fraunhofer.de/image-classification).

可解释的机器学习(或可解释的人工智能)技术最近受到了很多关注，因为它试图打开现代预测算法(主要是神经网络)的黑匣子。不仅仅是在学术界，决策者和企业也已经意识到，可解释性是避免在企业、公共卫生、刑事司法等领域部署众所周知的[不稳定的 ML 模型](https://docs.google.com/spreadsheets/u/1/d/e/2PACX-1vRPiprOaC3HsCf5Tuum8bRfzYUiKLRqJmbOoC-32JorNdfyTiRRsR7Ea5eWtvsWzuxo8bjOxCG84dAg/pubhtml)所带来的潜在危险的关键。例如，美国军方致力于开发他们所谓的[可解释的人工智能系统(XAI)](https://www.darpa.mil/program/explainable-artificial-intelligence) ，欧盟 [*通用数据保护条例*(GDPR)2016](https://en.wikipedia.org/wiki/General_Data_Protection_Regulation)包含算法决定的“解释权”，美国的 [*平等信贷机会法案*](https://en.wikipedia.org/wiki/Equal_Credit_Opportunity_Act) 主张拒绝信贷的具体原因的权利。可解释的 ML 也催生了众多初创企业，比如[可解释的 AI](https://www.interpretable.ai/) 、 [Clarifai](https://www.clarifai.com/) 和 [Aignostics](http://www.aignostics.com/) 等等。

这篇文章的目的是给你一个关于可解释 ML 的概述，为什么它如此有用，你可能从中受益的地方，并给你一些现有方法的初步指示。在这篇文章中，*不会*涉及任何具体技术的细节。

# *那么什么是可解释性 ML 呢？*

简而言之，可解释的 ML 意味着你的算法做出的决定可以以某种方式被翻译成人类可以理解的领域。例如，如果该算法对猫的图像进行分类，那么一种方法是简单地突出显示该算法已经使用了哪些像素来达成分类决定。

如果你想精确一点，你应该[区分*解释*和*解释*之间的](https://www.sciencedirect.com/science/article/pii/S1051200417302385)。解释将预测的类别转换到人类可以理解的领域，例如图像、文本或规则。*解释*只是负责模型输出的输入特征的集合，它可以是你输入到算法中的任何东西。通常，你会发现这些术语可以互换使用。

# *可解释 ML 有什么好处？*

除了上述的法规要求，可解释的 ML 在许多情况下是有用的，但并不相互排斥:

*   **建立信任**:当必须做出安全关键决策时，例如在医疗应用中，提供解释非常重要，以便相关领域的专家能够理解模型是如何做出决策的，从而决定是否信任模型。([这里](https://arxiv.org/abs/1610.02391)一篇考虑信任的论文。)
*   **故障分析**:自动驾驶等其他应用在部署时可能不涉及专家。但是，如果出现问题，可解释的方法可以帮助回顾性地检查在哪里做出了错误的决策，并了解如何改进系统。
*   **发现**:想象一下，你有一种算法可以准确地检测出早期癌症，除此之外，还可以开出最佳治疗方案。能够将它作为一个黑匣子已经很棒了，但是如果专家们能够检查*为什么*算法做得这么好，并随后深入了解癌症的机制和治疗的功效，那就更好了。
*   **验证**:在训练 ML 模型的时候，往往很难说出模型有多健壮(即使测试误差很大)，以及为什么它在某些情况下做得很好，而在其他情况下却不行。尤其令人发指的是所谓的*虚假相关性*:与你想在训练数据中预测的类别相关的特征，但不是这个类别正确的真正潜在原因。有很多虚假关联的例子，也有学术文献中记载的案例。
*   **模型改进**:如果你的模型做得不好，你不知道为什么，有时可以看看模型决策的解释，确定问题是出在[数据](https://www.nature.com/articles/s41467-019-08987-4http://www.stat.ucla.edu/~sczhu/papers/Conf_2018/AAAI_2018_DNN_Learning_Bias.pdf)还是[模型结构](https://link.springer.com/chapter/10.1007/978-3-319-10590-1_53)。

# 方法动物园

值得注意的是，有许多不同的方法来实现可解释的 ML，并且不是每个方法都适合每个问题。

基于规则的系统，例如 70 年代和 80 年代成功的[专家系统](https://www.laits.utexas.edu/~anorman/long.extra/Info.S98/Exp/intro.html)，通常是可解释的，因为它们通常将决策表示为一系列简单的 if-then 规则。在这里，整个，可能是复杂的，决策过程可以被追踪。一个主要的缺点是，它们通常依赖于手动定义的规则，或者依赖于以可以从中导出规则的参数形式(即，以符号形式)准备数据。

决策树也遵循 if-then 模式，但是它们可以直接用于许多数据类型。不幸的是，如果决策是复杂的，甚至它们也是不可理解的，而树木肯定不是我们所能使用的最有力的工具。如果您的问题不太复杂，并且您希望模型天生易于解释和交流，那么它们是一个很好的选择。

线性模型(如线性回归)非常有用，因为它们能以与每个输入变量相关的权重形式立即给出解释。由于其性质，线性模型只能捕捉简单的关系，对于高维输入空间，解释可能是不可理解的。

如果你的问题很简单或者符合专家系统的设置，那么上面的一种方法就足够了。如果你想要更强大的最新模型的解释(例如，深度神经网络或内核方法)，请继续阅读。

一些模型具有内置的可解释性。例如，[神经注意力架构](https://arxiv.org/abs/1409.0473)学习可以直接视为解释的输入的权重。解开模型，如[贝塔-VAE](https://openreview.net/forum?id=Sy2fzU9gl) ，也可以被视为[解释产生模型](https://ieeexplore.ieee.org/document/8631448)，因为它们为我们提供了数据中有意义的变异因素。

与上面的方法相反，*事后可解释性*方法处理在模型已经被训练之后提供解释*。这意味着您不必改变您的模型或培训管道，或者重新培训现有的模型来引入可解释性。这些方法中的一些具有与*模型无关的巨大优势，*这意味着你可以将它们应用于任何先前训练过的模型。这意味着您也可以轻松地比较不同模型提供的解释。*

最著名的事后可解释性技术之一是[局部可解释的模型不可知解释](https://arxiv.org/pdf/1602.04938v1.pdf) (LIME)。基本思想是，为了局部地解释算法对于特定输入的决定，学习线性模型来仅针对输入周围的小区域仿真算法。这个线性模型本质上是可以解释的，它告诉我们，如果我们稍微改变任何输入特征，输出会如何变化。

[SHapley Additive explaints](https://arxiv.org/abs/1705.07874)(SHAP)是一种建立在 SHapley 分析基础上的方法，该方法本质上是通过在所有可用特征的多个子集上训练模型来判断特征的重要性，并评估特征的省略会产生什么影响。在 SHAP 的论文中，对石灰和[深度提升](https://arxiv.org/abs/1704.02685)也进行了连接。

可解释性方法的另外两个密切相关的分支是基于传播和基于梯度的方法。它们或者通过模型传播算法的决定，或者利用损失梯度提供的灵敏度信息。突出的代表有[反卷积网络](https://www.cv-foundation.org/openaccess/content_iccv_2015/papers/Noh_Learning_Deconvolution_Network_ICCV_2015_paper.pdf)、[导向反向传播](https://arxiv.org/abs/1412.6806)、 [Grad-CAM](https://arxiv.org/abs/1610.02391) 、[综合梯度](https://arxiv.org/abs/1703.01365)、[逐层相关传播](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140) (LRP)。

这些方法中有许多是专为(卷积)神经网络设计的。一个显著的例外是 LRP，它也已经被应用于例如[内核方法](https://journals.plos.org/plosone/article?id=10.1371/journal.pone.0130140)和[lstm](https://arxiv.org/abs/1706.07206)。2017 年，LRP 获得了额外的理论基础和扩展，即[深度泰勒分解](https://www.sciencedirect.com/science/article/pii/S0031320316303582?via%3Dihub)。一些基于传播/梯度的方法已经在[工具箱](https://github.com/albermax/innvestigate)中实现并准备好使用。

我试图捕捉最重要的技术，但是当然还有更多的技术，而且这个数字随着每次相关会议的召开而增加。

总结一下，以下是一些要点:

*   可解释的 ML 扮演着越来越重要的角色，并且已经是一个(监管)需求。
*   这在许多情况下都很有帮助，例如，与用户建立信任，或者更好地理解数据和模型。
*   有大量的方法，从有悠久传统的非常简单的工具(基于规则的系统，或线性回归)，到可以用于现代模型的技术，如神经网络。

希望你学到了有用的东西。如果您有任何意见或反馈，请告诉我！