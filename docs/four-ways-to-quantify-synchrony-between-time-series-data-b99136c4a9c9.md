# 量化时间序列数据之间同步性的四种方法

> 原文：<https://towardsdatascience.com/four-ways-to-quantify-synchrony-between-time-series-data-b99136c4a9c9?source=collection_archive---------0----------------------->

## 数据科学教程

## 计算同步度量的样本代码和数据，包括皮尔逊相关、时间滞后交叉相关、动态时间弯曲和瞬时相位同步。

![](img/5ae94160074d3afb04c0a734d443867c.png)

Airplanes flying in synchrony, photo by [Gabriel Gusmao](https://unsplash.com/@gcsgpp?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

在心理学中，个体之间的同步可以是一个重要的信号，它提供了关于社会动态和社会互动的潜在结果的信息。人们已经在许多领域观察到了个体之间的同步，包括身体运动( [Ramseyer & Tschacher，2011](https://s3.amazonaws.com/academia.edu.documents/32743702/Nonverbal_synchrony_in_psychotherapy_Coordinated_body_movement_reflects_relationship_quality_and_out.pdf?AWSAccessKeyId=AKIAIWOWYYGZ2Y53UL3A&Expires=1557726122&Signature=bv5tXK0IqERSosMJUm4Rz9%2F71w4%3D&response-content-disposition=inline%3B%20filename%3DNonverbal_synchrony_in_psychotherapy_Coo.pdf) )、面部表情( [Riehle，Kempkensteffen，& Lincoln，2017](https://link.springer.com/article/10.1007/s10919-016-0246-8) )、瞳孔扩张( [Kang & Wheatley，2015](https://www.dropbox.com/s/8sfzjaqqkb6996h/Pupils_ConscAttn_CC2015.pdf?dl=0) )，以及神经信号( [Stephens，Silbert，& Hasson，2010](https://docs.wixstatic.com/ugd/b75639_82d46e0fa03a4f9290835c5db3888b8c.pdf) )。然而，术语*同步*可以有多种含义，因为有多种方法来量化两个信号之间的同步。

在本文中，我调查了一些最常见的同步指标和测量技术的优缺点，包括皮尔逊相关、时间滞后互相关(TLCC)和加窗 TLCC、动态时间弯曲和瞬时相位同步。举例来说，这些指标是使用样本数据计算的，其中微笑的面部表情是从两个参与者进行 3 分钟对话的视频镜头中提取的(如下截图)。要跟进，请随意下载[样本提取的面部数据](https://gist.github.com/jcheong0428/c6d6111ee1b469cf39683bd70fab1c93)和包含所有示例代码的 [Jupyter 笔记本](https://gist.github.com/jcheong0428/4a74f801e770c6fdb08e81a906902832)。

## 概述

1.  皮尔逊相关
2.  时间滞后互相关(TLCC)和加窗 TLCC
3.  动态时间扭曲(DTW)
4.  瞬时相位同步

![](img/4f76ffbc24d6207aa41c9ee764e897ca.png)

Sample data is the smiling facial expression between two participants having a conversation.

# 1.皮尔逊相关——简单最好

[Pearson correlation](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient) 测量两个连续信号如何随时间共同变化，并以-1(负相关)到 0(不相关)到 1(完全相关)之间的数字表示线性关系。它很直观，容易理解，也容易解释。使用 Pearson correlation 时需要注意两件事:1)异常值会扭曲相关估计的结果，2)它假设数据是[同方差的](https://en.wikipedia.org/wiki/Homoscedasticity)，因此数据的方差在整个数据范围内是同质的。一般来说，相关性是全局同步性的一个快照度量。因此，它不提供关于两个信号之间的方向性的信息，例如哪个信号在前，哪个信号在后。

Pearson 关联在多个包中实现，包括 Numpy、Scipy 和 Pandas。如果数据中有空值或缺失值，Pandas 中的关联函数会在计算前删除这些行，而如果使用 Numpy 或 Scipy 的实现，则需要手动删除这些数据。

以下代码加载的是样本数据(在同一个文件夹中)，使用 Pandas 和 Scipy 计算 Pearson 相关性，并绘制过滤后的中值数据。

![](img/c426253af9b23b4198299c765630bbc7.png)

再一次，总体皮尔逊 r 是对*全局*同步性的度量，它将两个信号之间的关系简化为单个值。尽管如此，还是有一种方法可以利用皮尔逊相关来观察瞬间的*局部*同步。一种计算方法是通过测量一小部分信号中的皮尔逊相关，并沿着滚动窗口重复该过程，直到覆盖整个信号。这可能有点主观，因为它需要任意定义窗口大小，你想重复这个过程。在下面的代码中，我们使用 120 帧(~4 秒)的窗口大小，并在下图中绘制时刻同步。

![](img/aecee779a59df42b6b1b4b810b26c9b6.png)

Sample data on top, moment-to-moment synchrony from moving window correlation on bottom.

总的来说，皮尔逊相关是一个很好的起点，因为它提供了一种非常简单的方法来计算全局和局部同步性。然而，这仍然不能提供对信号动态的洞察，例如哪个信号首先出现，这可以通过互相关来测量。

# 2.时间滞后互相关——评估信号动态

时间滞后互相关(TLCC)可以识别两个信号之间的方向性，例如领导者-追随者关系，其中领导者发起由追随者重复的响应。有几种方法可以研究这种关系，包括经济学中使用的[格兰杰因果关系](https://en.wikipedia.org/wiki/Granger_causality)，但请注意，这些仍然不一定反映真正的因果关系。尽管如此，我们仍然可以通过观察互相关来判断哪个信号先出现。

![](img/4b790009fc47aec56ac473619b466da9.png)

[http://robosub.eecs.wsu.edu/wiki/ee/hydrophones/start](http://robosub.eecs.wsu.edu/wiki/ee/hydrophones/start)

如上所示，TLCC 是通过递增移动一个时间序列向量(红色)并重复计算两个信号之间的相关性来测量的。如果峰值相关性位于中心(offset=0)，这表明两个时间序列在该时间最同步。然而，如果一个信号领先于另一个信号，则峰值相关可能处于不同的偏移。下面的代码使用 pandas 功能实现了一个互相关函数。它还可以*包装*数据，以便仍然通过添加来自信号另一侧的数据来计算边缘上的相关值。

![](img/06f4617c21ff3f0f69011f7635f0ecc5.png)

Peak synchrony is not at the center, suggesting a leader-follower signal dynamic.

在上面的图中，我们可以从负偏移推断出受试者 1 (S1)正在引导交互(当 S2 向前推进 46 帧时，相关性最大化)。但这又一次评估了全球水平的信号动态，例如在整个 3 分钟期间谁领先。另一方面，我们可能会认为互动甚至可能*更加*动态，这样领导者和追随者的角色会随时变化。

为了评估更细粒度的动态，我们可以计算*加窗*时间滞后互相关(WTLCC)。这个过程在信号的多个窗口中重复时间滞后互相关。然后，我们可以分析每个窗口，或者对窗口求和，这将提供一个分数，比较两个个体之间领导者与追随者之间的互动差异。

![](img/a51c6e34fcea6bbc783e4e2915aa3977.png)

Windowed time lagged cross correlation for discrete windows

上面的图将时间序列分成 20 个均匀的块，并计算每个窗口中的互相关。这让我们对交互中发生的事情有了更细粒度的了解。例如，在第一个窗口(第一行)中，右边的红色峰值表示 S2 最初领导交互。然而，在第三或第四个窗口(行)，我们可以看到 S1 开始更多地引导互动。我们也可以连续计算，得到更平滑的图，如下所示。

![](img/1a5da15a3b9cf9031a5bbeef104c8ea9.png)

Rolling window time lagged cross correlation for continuous windows

时间滞后互相关和窗口时间滞后互相关是可视化两个信号之间的细粒度动态交互的好方法，例如领导者-追随者关系以及它们如何随时间变化。然而，这些信号是在假设事件同时发生且持续时间相似的情况下计算的，这将在下一节讨论。

# 3.动态时间弯曲——长度不同的信号的同步

动态时间弯曲(DTW)是一种计算两个信号之间的路径的方法，该方法可以最小化两个信号之间的距离。这种方法最大的优点是还可以处理不同长度的信号。最初是为语音分析而设计的(在[这个视频](https://www.youtube.com/watch?v=_K1OsqCicBY)中了解更多)，DTW 计算每帧之间的欧几里德距离，以计算将匹配两个信号的最小路径。一个缺点是它不能处理丢失的值，所以如果你有丢失的数据点，你需要预先插值。

![](img/d40d3795e99cc5c0ba9250b816b7f47d.png)

XantaCross [CC BY-SA 3.0 ([https://creativecommons.org/licenses/by-sa/3.0](https://creativecommons.org/licenses/by-sa/3.0))]

为了计算 DTW，我们将使用`dtw` Python 包来加速计算。

![](img/40629df5c85de63f6509c91e0d32562b.png)

这里我们可以看到白色凸线中显示的最小路径。换句话说，较早的主题 2 数据与较晚的主题 1 数据的同步相匹配。最小路径代价为 *d* =.33，可与其他信号相比。

# 4.瞬时相位同步。

最后，如果您有一个您认为可能具有振荡特性的时间序列数据(例如 EEG、fMRI)，您也可以测量瞬时相位同步。这种方法还可以测量两个信号之间的瞬间同步。这可能有些主观，因为您需要将数据过滤到感兴趣的波长，但您可能有确定此类波段的理论原因。为了计算相位同步，我们需要提取信号的相位，这可以通过使用希尔伯特变换来完成，希尔伯特变换将信号分解为相位和功率([点击](https://www.youtube.com/watch?v=VyLU8hlhI-I)了解关于希尔伯特变换的更多信息)。这使我们能够评估两个信号是同相(一起上下移动)还是异相。

![](img/22c0e106c800c8c1fe78139da060e0f7.png)

Gonfer at English Wikipedia [CC BY-SA 3.0 ([https://creativecommons.org/licenses/by-sa/3.0](https://creativecommons.org/licenses/by-sa/3.0))]

![](img/1e982ee0120e2818f14eae595da57409.png)

Filtered time series (top), angle of each signal at each moment in time (middle row), and instantaneous phase synchrony measure (bottom).

瞬时相位同步测量是计算两个信号之间瞬间同步的好方法，无需像滚动窗口相关那样任意决定窗口大小。如果你想知道瞬时相位同步与窗口相关相比如何，[点击这里查看我之前的博客文章](http://jinhyuncheong.com/jekyll/update/2017/12/10/Timeseries_synchrony_tutorial_and_simulations.html)。

# 结论

这里我们介绍了四种测量时间序列数据同步性的方法:皮尔逊相关、时间滞后交叉相关、动态时间弯曲和瞬时相位同步。确定同步度量标准将基于您拥有的信号类型、您对数据的假设以及您希望从数据中获得什么同步信息的目标。欢迎在下面留下任何问题或评论！

查看下面 Jupyter 笔记本中的所有代码，并使用此处提供的[样本数据](https://gist.github.com/jcheong0428/c6d6111ee1b469cf39683bd70fab1c93/archive/b2546c195e6793e00ed23c97a982ce439f4f95aa.zip)。

**新增 2021.05.12:在 Google Colab 中直接打开教程！*

如果您觉得本教程很有用，并将在您的研究项目中使用，请考虑引用这篇文章，并附上以下参考:

> Cheong，J. H. (2020 年 12 月 8 日)。量化时间序列数据同步性的四种方法。【https://doi.org/10.17605/OSF.IO/BA3NY 

## 谢谢引用！

西蒙，O. B .，布阿尔，I .，罗哈斯，D. C .，霍尔登，S. K .，克鲁格，B. M .，&戈什，D. (2021)。使用最小生成树、边缘切割和脑磁图来理解帕金森认知衰退的新方法。*科学报道*， *11* (1)，1–17。

Pagay，V. (2021)。用微型张力计揭示植物水势的动态方面。 *bioRxiv* 。

Galvis，d .，Zavala，e .，Walker，J. J .，Upton，t .，Lightman，S. L .，Angelini，G. D .，… & Gibbison，B. (2021)。大手术期间和术后全身炎症和下丘脑-垂体-肾上腺(HPA)轴的动态相互作用。 *bioRxiv* 。

Bente，g .，Novotny，e .，Roth，d .，和 Al-Issa，A. (2020 年)。超越刻板印象:分析非言语融洽中的性别和文化差异。*心理学前沿*， *11* 。