# 面向大规模偏好聚合的数据集融合

> 原文：<https://towardsdatascience.com/dataset-fusion-sushi-age-and-image-quality-and-what-the-hell-do-they-have-in-common-814e8dae7cf7?source=collection_archive---------14----------------------->

![](img/12f47c72429d5a2edcbdca624d0dded3.png)

Image by Author

对于很多问题，我们有强大的机器学习算法，在给定足够数据的情况下，可以达到前所未有的性能。然而，拥有大量通用和高质量数据的条件并不容易满足。

对于需要人工输入的问题来说尤其困难。让人们直接参与数据采集过程既费钱又费时。此外，对于某些问题，需要专业知识(如医学样本评估)，或受控环境(如食品质量、视觉实验)。由于这些要求和约束，创建的数据集可能非常小且非常具体。

一个很大的问题领域，其中很难建立数据集是引发规模，因为人类的判断。结果数据集包含我们正在标记的对象和感知比例之间的对应关系。例如食物的味道(酸奶有多甜？)、图像质量(有噪声的图像离参考有多远？)或者甚至是人的年龄(照片中的人的感知年龄是多少？).

## 怎么才能解决问题？

有两种方法可以解决数据不足的问题——投入大量资金从零开始收集数据集，或者更聪明一点，将已经收集的数据集融合在一起——重用收集的知识。

在这篇文章中，我将讨论分级和评级，以及如何在一个图像质量评估的例子中将这两种协议混合在一起。收集的数据通常用于训练[客观图像质量指标](/deep-image-quality-assessment-30ad71641fac)，这些指标应与人类感知密切相关。

文章的 MATLAB 代码在这里[可用](https://github.com/gfxdisp/pwcmp_rating_unified)，Python 代码在这里[可用](https://github.com/mikhailiuk/medium/blob/master/Dataset-fusion.ipynb)。

这篇文章是基于[这篇论文](https://www.cl.cam.ac.uk/~rkm38/pdfs/perezortiz2019unified_quality_scale.pdf)——如果你想看更多的细节和对这个问题的严谨方法，不要错过。

# 如何引出一个尺度？

在构建量表时，我们试图恢复潜在的分数。这种量表有两种构建方式——评级或排名。

![](img/05f54feb7c1af0494c27474f807c3d3c.png)

Figure 1: Rating and ranking protocols. Image by Author.

## 等级

我们可以成对或成组地排列对象。在这里，我将集中讨论成对比较，因为它们简单并且能够将集合比较的结果转换成成对比较。

在成对比较实验中，受试者根据某种标准选择两个条件中的一个。答案被记录在矩阵 ***C*** 中，其中每个条目 c *ij* 是条件 *Ai* 被选择超过条件 *Aj 的次数。*

为了将这个比较矩阵转换成一维尺度，我们可以使用[布拉德利-特里](https://www.jstor.org/stable/pdf/2334029.pdf)或[瑟斯通](https://www.anishathalye.com/media/2015/03/07/thurstone1927.pdf)模型。在实践中，这两个模型产生了类似的规模，但布拉德利-特里使用不对称 Gumbel 分布(累积逻辑函数)和 Thurstone 使用对称高斯分布。这里我就说说瑟斯通案 V 模式。其他病例(I、II、III、IV)的描述可在原始论文中找到。

![](img/82af32aadfdcf64607074c1ed4f15b7c.png)

Figure 2: From conditions to the quality scale. Image by Author.

瑟斯通案例五模型首先将观察者的答案映射到一个条件比另一个条件更好的概率上。然后概率被转换成距离。这种从概率到距离的映射是通过逆正态累积分布实现的。该分布的标准偏差(sigma)定义了映射。通常将 0.75 的一个条件被选为更好的概率映射到一个单位距离的条件之间的差异(图 3)，然后构建的尺度被称为恰好不良差异(JOD)尺度。

![](img/e4263b0dab6d5f1a0169438f9d04705f.png)

Figure 3: Mapping from distance to probability of a condition selected as better. Image by Author.

然后，构建量表的问题转化为降维问题。在这里，对于每一个条件 *Ai* 和 *Aj* ，我们通过二项式分布将它们的质量分数差异与 *Ai* 被选择超过 *Aj* 的次数联系起来(反之亦然):

![](img/07c3a139c77fc4a9f2e938958d83b43a.png)

其中 n *ij —* 在 *i* 和*j*之间的比较总数然后我们使用最大似然估计来推断质量分数。由于质量是相对的，我们将第一个条件的质量设为 0 ( *q1* = 0)。关于成对比较的更多细节，请看这里的[和这里的](http://mayagupta.org/publications/PairedComparisonTutorialTsukidaGupta.pdf)和。图 2 给出了从成对比较中得出比例的途径。

## 评级

分级实验可以是:(I)分类的——受试者选择一个条件所属的类别；(ii)基数—为条件分配数值的主体。然后汇总所有科目的分数，得出平均值。这个平均值被称为平均意见得分(MOS)。

我们现在通过评级转向建模质量 **q** 。评级实验中使用的标度范围由实验指挥设定，可以是从 0 到 10、从 1 到 100 等任何值。为了结合范围和偏差，我们引入两个变量 *a* 和*b*。我们进一步假设从评级测量中得出的质量遵循正态分布。对于条件 i (q *i* )的每个潜在质量，我们有:

![](img/edce51a12f4e91ac27151c663d507a8b.png)

其中 m *ik* 是由第 k *个*观测器分配给第 i *个*条件的分数，而 *c* 定义了相对于固定观测器噪声σ的标准偏差的大小。将以上展开，代入正态分布公式:

![](img/1f614552ec909c9a2e1dc248bbcb4bcc.png)

观察评级矩阵的可能性，条目 m *ik* 如上则由下式给出:

![](img/7741e951c3d93e5e174016e8271620b5.png)

# 融合数据集

融合成对比较数据集很简单——在数据集中选择一些条件，将它们与成对比较联系起来(进行一些实验，并输入 Thurstone/Bradley-Terry 模型)。类似地，对于评级分数，从不相交的数据集中选择一些条件，在联合实验中测量这些条件的投票，并基于新测量的条件的相对质量来重新调整原始数据。但是我们如何继续融合具有成对比较和评级分数的数据集呢？

## 就不能把收视率数据扔在一起吗？

嗯…人类参与者可能会被问及稍微不同的问题，或者实验可以在稍微不同的环境中进行，所以一个数据集中的寿司多汁性 4 可能对应于另一个数据集中的 3，仅仅因为它是相对于当天品尝的其他寿司进行评级的。

## 模型

我们定义了一个最大化问题，其中我们试图找到潜在的质量分数 q 和参数 a、b 和 C，在给定矩阵 M 和 C 以及观察者模型 sigma 的标准偏差的情况下，将两两比较和评级测量相关联。

![](img/bc084d9027c929e8a64f9cb52bf7297a.png)

我们可以看到一些熟悉的术语——即 P(C|q)和 P(M|q)的定义如上。然而，这里我们也有 P(q)-高斯先验，包括以加强凸性。

![](img/5e1de3b2110cb7a3f994b6f86ea1b74b.png)

请注意，现在潜在的质量分数是使用来自平均意见分数和成对比较的信息找到的。参数 *c* 有它的含义——如果 c 大于 1，成对比较对实验更好，如果小于 1，则更差。然后可以用最大似然估计来找到模型的参数。

# 测试模型

让我们考虑一个玩具的例子。这里我们有两个数据集，DS1 和 DS2，每个都有成对比较和评级测量

DS1 有四个条件。成对比较矩阵 C1 因此是 4x4。请注意，条件 3 在成对比较中没有与其余条件相关联，但是这不是问题，因为它是在评级实验中测量的。DS1 的其他条件进行了 6 次比较。评级测量结果收集在 4x4 矩阵 M1 中，即由 4 名观察者对条件进行评级。

![](img/9daed64c26883bcf79c284260859adcb.png)

Image by Author.

DS2 有 5 个条件，在评级实验中由 5 个受试者测量。在这个数据集中，条件 2 也很突出—它没有被评级。然而，通过成对比较，它与其余的联系在一起。

![](img/69037a3d54bf0157d45d1172d40f915c.png)

Image by Author.

有两个不相交的数据集，我们希望通过成对比较将它们连接在一起。下面是一个矩阵 **C** 。 **C** 包括原始数据集的成对比较数据(红色和绿色)，以及为将两个数据集链接在一起而收集的附加比较数据(蓝色)。类似地，矩阵 M 包含来自 DS1 和 DS2 的评级实验的组合数据。

![](img/f0f351e6cea82bb259d3fed9b8656ccf.png)

Image by Author.

我们现在可以将 DS1 和 DS2 一起进行缩放，以获得最终的缩放比例。

![](img/10ce01d3e4a9961fe72e6a190af4e51f.png)

Image by Author.

这里，真实分数是用于生成矩阵 **C** 和 **M** 中的分数，而预测分数是通过将来自 **C** 和 **M** 的数据混合在一起而获得的分数。MOS 分数是平均等级测量，而成比例的成对比较是仅从成对比较中获得的质量分数。请注意，结果的准确性取决于数据的质量和数量。为了获得更好的结果，我们可以收集更多的成对比较或评级测量。

# 进一步阅读

这是文章中提到的来源的汇总:[原始论文](https://www.cl.cam.ac.uk/research/rainbow/projects/unified_quality_scale/perezortiz2019unified_quality_scale-large.pdf)、[代号](https://github.com/gfxdisp/pwcmp_rating_unified)、[特斯通模型原始论文](https://www.anishathalye.com/media/2015/03/07/thurstone1927.pdf)、[布拉德利-特里原始论文](https://www.jstor.org/stable/pdf/2334029.pdf)，缩放两两比较数据:[论文 1](http://mayagupta.org/publications/PairedComparisonTutorialTsukidaGupta.pdf) 和[论文 2](https://arxiv.org/pdf/1712.03686.pdf) 。如果你想从另一个角度来看融合评级和排名测量，这两篇论文将会很有帮助:[论文 1](http://openaccess.thecvf.com/content_cvpr_2014/papers/Ye_Active_Sampling_for_2014_CVPR_paper.pdf) ，[论文 2](http://dbgroup.cs.tsinghua.edu.cn/ligl/papers/sigmod18-crowdtopk.pdf) 。

## 喜欢作者？保持联系！

我错过了什么吗？不要犹豫，直接在 [LinkedIn](https://www.linkedin.com/in/aliakseimikhailiuk/) 或 [Twitter](https://twitter.com/mikhailiuka) 上给我留言、评论或发消息吧！

[](/active-sampling-for-pairwise-comparisons-476c2dc18231) [## 成对比较的主动采样

### 如何配对玩家，以便在尽可能少的游戏中知道排名，同时游戏体验质量…

towardsdatascience.com](/active-sampling-for-pairwise-comparisons-476c2dc18231) [](/bayesian-optimization-or-how-i-carved-boats-from-wood-examples-and-code-78b9c79b31e5) [## 贝叶斯优化的超参数调整或者我如何用木头雕刻船

### 超参数调整通常是不可避免的。对于一个参数，网格搜索可能就足够了，但如何处理…

towardsdatascience.com](/bayesian-optimization-or-how-i-carved-boats-from-wood-examples-and-code-78b9c79b31e5) [](/can-you-do-better-sampling-strategies-with-an-emphasis-on-gibbs-sampling-practicals-and-code-c97730d54ebc) [## 你能做得更好吗？抽样策略，重点是吉布斯抽样，实践和代码

### 提供了通用采样策略的概述，重点是 Gibbs 采样、示例和 python 代码。

towardsdatascience.com](/can-you-do-better-sampling-strategies-with-an-emphasis-on-gibbs-sampling-practicals-and-code-c97730d54ebc)