# 了解数据集转换

> 原文：<https://towardsdatascience.com/understanding-dataset-shift-f2a5a262a766?source=collection_archive---------0----------------------->

如何不被数据捉弄？

> 数据集转移是一种具有挑战性的情况，其中输入和输出的联合分布在训练和测试阶段之间不同。 **—** [***数据集转移，麻省理工学院出版社。***](https://cs.nyu.edu/~roweis/papers/invar-chapter.pdf)

[数据集转移](https://cs.nyu.edu/~roweis/papers/invar-chapter.pdf)是其中一个简单的主题，也许简单到被认为是显而易见的。在我自己的数据科学课上，这个想法被简单地讨论过，但是，我认为对数据集转移的原因和表现的更深入的讨论对数据科学社区是有益的。

这篇文章的主题可以用一句话来概括:

**数据集移位是指训练和测试分布不同时。**

![](img/b0fd00e8406ff9abe476652c8ee6772a.png)

An example of differing training and test distributions.

![](img/73b367df3864982ed996e51121a16453.png)

虽然你可能会嘲笑这种说法的琐碎，但这可能是我在查看 Kaggle 挑战的解决方案时看到的最常见的问题。在某些方面，对数据集转移的深刻理解是赢得 Kaggle 竞赛的关键。

数据集偏移不是一个标准术语，有时被称为**概念偏移**或**概念漂移**、**分类的变化**、**环境的变化**、**分类学习中的对比挖掘**、**断点**和**数据间的断裂。**

数据集转移主要发生在监督学习的机器学习范式和半监督学习的混合范式中。

数据集偏移的问题可能源于输入特征的利用方式、训练和测试集的选择方式、数据稀疏性、由于非稳定环境导致的数据分布偏移，以及深层神经网络层内激活模式的变化。

为什么数据集转移很重要？

它依赖于应用程序，因此很大程度上依赖于数据科学家的技能来检查和解决。例如，如何确定数据集何时发生了足够大的变化，从而给我们的算法带来了问题？如果只有某些特征开始发散，我们如何确定通过移除特征的准确度损失和通过错误的数据分布的准确度损失之间的权衡？

在本文中，我将讨论不同类型的数据集转移、它们的出现可能带来的问题，以及可以用来避免它们的当前最佳实践。本文不包含代码示例，纯粹是概念性的。为了便于演示，将使用分类示例。

我们将研究数据集转移的多种表现形式:

*   协变量移位
*   先验概率转移
*   观念转变
*   内部协变移位(协变移位的一个重要亚型)

这是机器学习中一个巨大而重要的话题，所以不要期望对这个领域有一个全面的概述。如果读者对这一主题感兴趣，那么会有大量关于这一主题的研究文章——其中绝大多数集中在协变量转换上。

# 协变量移位

在数据集移位的所有表现形式中，最容易理解的是协变量移位。

> 协变量移位是指*协变量*的分布变化，具体来说，就是自变量的分布变化。这通常是由于潜在变量状态的变化，可能是时间的(甚至是时间过程的平稳性的变化)，或空间的，或不太明显的。——[Quora](https://www.quora.com/What-is-Covariate-shift)

协变量移位是一个学术术语，指数据分布(即我们的输入要素)发生变化的情况。

![](img/e206039fae538fea0c282f416a8f1124.png)![](img/c44b8b0371f8b45f1d440c02bc603445.png)

以下是协变量移位可能导致问题的一些例子:

*   人脸识别算法主要针对较年轻的人脸进行训练，但数据集中有更大比例的老年人脸。
*   预测预期寿命，但在吸烟者的训练集中样本很少，而在训练集中样本更多。
*   将图像分类为猫或狗，并从训练集中省略在测试集中看到的某些物种。

在这种情况下，输入和输出之间的基本关系没有变化(回归线仍然相同)，但是该关系的一部分是数据稀疏的、省略的或错误表示的，使得测试集和训练集不反映相同的分布。

在执行交叉验证时，协方差变化会导致许多问题。交叉验证在没有协变量转移的情况下几乎是无偏的，但在协变量转移的情况下会有很大的偏差！

# 先验概率转移

协变量移位关注特征( ***x*** )分布的变化，先验概率移位关注类别变量 ***y*** 分布的变化。

![](img/fe1c17f0945cbaa63fe2695e418882ff.png)![](img/81cc35dda52fc75692bc4623ae38d803.png)

这种类型的转换可能看起来更令人困惑，但它本质上是协变量转换的反向转换吗？一种直观的思考方式可能是考虑一个不平衡的数据集。

如果训练集对您收到的垃圾邮件的数量具有相同的先验概率(即，一封电子邮件是垃圾邮件的概率是 0.5)，那么我们预计 50%的训练集包含垃圾邮件，50%包含非垃圾邮件。

如果在现实中，只有 90%的电子邮件是垃圾邮件(也许不是不可能的)，那么我们的类别变量的先验概率已经改变。这种想法与数据稀疏性和有偏差的特征选择有关，它们是导致协方差偏移的因素，但它们不是影响我们的输入分布，而是影响我们的输出分布。

这个问题只出现在 Y → X 问题中，通常与朴素贝叶斯有关(因此出现了垃圾邮件示例，因为朴素贝叶斯通常用于过滤垃圾邮件)。

下图中的先验概率转移取自机器学习一书中的[数据集转移，很好地说明了这种情况。](http://www.acad.bg/ebook/ml/The.MIT.Press.Dataset.Shift.in.Machine.Learning.Feb.2009.eBook-DDU.pdf)

![](img/dd447b9570c433a3c83ba9488b42148a.png)

# **概念漂移**

概念漂移不同于协变量和先验概率转移，因为它与数据分布或类别分布无关，而是与两个变量之间的关系有关。

思考这个观点的一个直观方法是看时间序列分析。

在时间序列分析中，通常在执行任何分析之前检查时间序列是否是平稳的，因为平稳时间序列比非平稳时间序列更容易分析。

![](img/e95783cac930929054d3ca33a75de9cf.png)

为什么会这样呢？

这更容易，因为输入和输出之间的关系并没有持续变化！有多种方法可以消除时间序列的趋势，使其保持平稳，但这并不总是有效的(例如股票指数通常包含很少的自相关或长期变化)。

![](img/ebce8a54c4e20c69a168a4186da7bef1.png)

举一个更具体的例子，假设我们考察了 2008 年金融危机前公司的利润，并根据行业、员工数量、产品信息等因素制作了一个算法来预测利润。如果我们的算法是根据 2000 年至 2007 年的数据训练的，但在金融危机后没有用它来预测同样的信息，它可能表现不佳。

那么是什么改变了呢？很明显，由于新的社会经济环境，投入和产出之间的整体关系发生了变化，如果这些没有反映在我们的变量中(例如，金融危机发生日期的虚拟变量和该日期前后的培训数据)，那么我们的模型将遭受概念转变的后果。

在我们的具体案例中，我们希望在金融危机后的几年里看到利润发生显著变化(这是一个[中断时间序列](https://en.wikipedia.org/wiki/Interrupted_time_series)的例子)。

# 内部协变量移位

这个话题最近引起兴趣的一个原因是由于深度神经网络(因此有“内部”一词)的隐藏层中的协方差偏移的可疑影响。

研究人员发现，由于来自给定隐藏层的输出(用作后续层的输入)的激活分布的变化，网络层可能遭受协变量移位，这可能妨碍深度神经网络的训练。

![](img/9310d18977847301accc4d1d683573eb.png)

The situation without batch normalization, network activations are exposed to varying data input distributions that propagate through the network and distort the learned distributions.

这个想法就是[批量归一化](https://en.wikipedia.org/wiki/Batch_normalization)的刺激，由 Christian Szegedy 和 Sergey Ioffe 在他们 2015 年的论文 [*“批量归一化:通过减少内部协变量移位加速深度网络训练”*](https://arxiv.org/pdf/1502.03167.pdf) *中提出。*

作者提出，隐藏层中的内部协变量移位会减慢训练，并且需要较低的学习速率和仔细的参数初始化。他们通过添加一个批处理规范化层来规范化隐藏层的输入，从而解决了这个问题。

该批次规范层获取一批样本的平均值和标准差，并使用它们来标准化输入。这也给输入增加了一些噪声(因为不同批次之间的平均值和标准偏差中固有的噪声),这有助于调整网络。

![](img/0a675a40949d1758e22838ff0e113629.png)

How batch normalization fits within the network architecture of deep neural networks.

该问题用于将变化的分布转化为更稳定的内部数据分布(更少的漂移/振荡),这有助于稳定学习。

![](img/50725e20becc84430ee2bea6fde79da2.png)

Varying data distributions across batches are normalized via a batch normalization layer in order to stabilize the data distribution used as input to subsequent layers in a deep neural network.

批处理规范化现在在深度学习社区中被很好地采用，尽管最近的一篇论文[暗示，从该技术获得的改进结果可能不纯粹是由于内部协变量偏移的抑制，而可能是平滑网络损失前景的结果。](https://arxiv.org/pdf/1805.11604.pdf)

对于那些不熟悉批处理规范化、其目的和实现的人，我建议看看吴恩达的相关 Youtube 视频，其中一个视频链接如下。

# 数据集转移的主要原因

数据集偏移的两个最常见原因是(1) **样本选择偏差**和(2) **非稳定环境**。

需要注意的是，这些都不是数据集移位的类型，并不总是会导致数据集移位。它们只是我们的数据中可能发生数据集偏移的潜在原因。

**样本选择偏差:**分布中的差异是由于通过有偏差的方法获得的训练数据，因此不能可靠地代表分类器将被部署的操作环境(用机器学习的术语来说，将构成测试集)。

**非稳定环境:**当训练环境不同于测试环境时，无论是由于时间还是空间变化。

## 样本选择偏差

样本选择偏差不是任何算法或数据处理的缺陷。这纯粹是数据收集或标记过程中的系统缺陷，导致从总体中选择的训练样本不一致，从而导致训练过程中形成偏差。

样本选择偏差是协方差变动的一种形式，因为我们正在影响我们的数据分布。

这可以被认为是对操作环境的曲解，因此我们的模型将其训练环境优化为一个人为的或精选的操作环境。

![](img/4eae4a60f3d3932b218259fdfce15b63.png)

当处理不平衡分类时，由样本选择偏差导致的数据集偏移尤其相关，因为在高度不平衡的域中，少数类对单一分类错误特别敏感，这是因为它所呈现的样本数量通常很少。

![](img/fe7c4a40eab0a091095589f2d7ea61ee.png)

Example of the impact of dataset shift in imbalanced domains.

在最极端的情况下，少数类的一个错误分类示例就可能导致性能的显著下降。

## 不稳定的环境

在现实世界的应用 it 世界的应用中，数据通常不是(时间或空间)静态的。

最相关的非平稳场景之一涉及对立的分类问题，例如垃圾邮件过滤和网络入侵检测。

这种类型的问题在机器学习领域受到越来越多的关注，并且通常处理非静态环境，因为存在试图绕过现有分类器的学习概念的对手。就机器学习任务而言，这个对手扭曲了测试集，使其变得不同于训练集，从而引入任何可能的数据集转移。

# 识别数据集转移

有多种方法可用于确定数据集中是否存在移位及其严重程度。

![](img/472c6d8009b832249fb5b062fbe750ce.png)

Tree diagram showing the methods of identifying dataset shift.

无监督的方法可能是识别数据集变化的最有用的方法，因为它们不需要进行事后分析，这种延迟在一些生产系统中是无法承受的。存在监督方法，其本质上观察模型运行时不断增加的错误以及外部维持(验证集)的性能。

**统计距离**
[s*统计距离*](https://en.wikipedia.org/wiki/Statistical_distance) 方法对于检测您的模型预测是否随时间变化非常有用。这是通过创建和使用直方图来实现的。通过制作直方图，您不仅能够检测模型预测是否随时间变化，还可以检查最重要的特征是否随时间变化。简而言之，你形成训练数据的直方图，随着时间的推移跟踪它们，并比较它们以查看任何变化。金融机构通常在信用评分模型中使用这种方法。

![](img/fc595c5a0cb1f9a2214b198324ed55be.png)

Two distributions are their KL-divergence (effectively the ‘distance’ between the two distributions). If the two distributions overlap, they are effectively the same distribution and the KL-divergence is zero.

有几个指标可用于监控模型预测随时间的变化。其中包括 [**人口稳定指数**](https://www.quora.com/What-is-population-stability-index)**【PSI】**[**柯尔莫哥洛夫-斯米尔诺夫统计**](https://en.wikipedia.org/wiki/Kolmogorov%E2%80%93Smirnov_test)[**库尔贝克-勒布勒散度**](https://en.wikipedia.org/wiki/Kullback%E2%80%93Leibler_divergence) (或其他[*-*散度](https://en.wikipedia.org/wiki/F-divergence))[**直方图交集**](http://blog.datadive.net/histogram-intersection-for-change-detection/) 。

![](img/040709a40340bc2a48e9ae0c48a6f9b8.png)

Data plotted along one feature axis for a training and test set. There is ~72% intersection of the distributions which indicates a reasonable level of covariate shift between the distributions.

这种方法的主要缺点是对于高维或稀疏特征来说效果不是很好。然而，它可能非常有用，并且在我看来应该是处理这个问题时首先要尝试的。

![](img/1277b97e73dfcba57442747f388da0fa.png)

A comparison between KL-divergence, KS statistic, PSI, and histogram intersection for two examples. The left example shows little to no covariate shift, whilst the right example shows a substantial covariate shift. Notice how it affects the expected values of the statistical distances.

**2)新颖性检测**
一种更适合计算机视觉等相当复杂领域的方法是 [*新颖性检测*](https://en.wikipedia.org/wiki/Novelty_detection) 。这个想法是创建一个模型来模拟源分布。给定一个新的数据点，您尝试测试该数据点从源分布中提取的可能性。对于这种方法，您可以使用各种技术，例如大多数常见库中可用的单类支持向量机。

![](img/df686cf359380d2e174f079f1b52102e.png)

如果您处于同质但非常复杂的交互状态(例如，视觉、听觉或遥感)，那么这是一种您应该研究的方法，因为在这种情况下，统计距离(直方图方法)将不是一种有效的方法。

这种方法的主要缺点是它不能明确地告诉你发生了什么变化，只能告诉你已经发生了变化。

**3)判别距离**
[*判别距离*](https://www.sciencedirect.com/science/article/abs/pii/S0031320313000307) 的方法不太常见，尽管如此，它也可以是有效的。直觉上，你想训练一个分类器来检测一个例子是来自源领域还是目标领域。您可以使用训练误差作为这两个分布之间的距离的代理。误差越大，它们越接近(即分类器不能区分源域和目标域)。

判别距离具有广泛的适用性和高维性。虽然这需要时间并且可能非常复杂，但是如果您正在进行领域适应，这种方法是一种有用的技术(对于一些深度学习方法，这可能是唯一可行的技术)。

这种方法对高维稀疏数据有很好的效果，适用范围广。然而，它只能离线完成，并且比以前的方法实现起来更复杂。

# 处理数据集移位

如何纠正数据集偏移？如果可能的话，你应该经常再培训。当然，在某些情况下，这可能是不可能的，例如，如果再培训存在延迟问题。在这种情况下，有几种技术可以纠正数据集偏移。

**1)特征移除**

通过利用上面讨论的用于识别协变量偏移的统计距离方法，我们可以使用这些方法作为偏移程度的度量。我们可以为可接受的偏移水平设定一个界限，通过分析单个特征或消融研究，我们可以确定哪些特征对偏移负有最大责任，并将其从数据集中移除。

如您所料，在移除导致协变量偏移的特征和增加额外特征并容忍某些协变量偏移之间存在权衡。这种权衡是数据科学家需要根据具体情况进行评估的。

一个在训练和测试过程中差别很大，但是不能给你很多预测能力的特性，应该总是被放弃。

例如，PSI 用于风险管理，0.25 的任意值用作限制，超过该值则视为重大变化。

**2)重要性重新加权**
重要性重新加权的主要思想是你想得到与你的测试实例非常相似的训练实例。本质上，您尝试更改您的训练数据集，使其看起来像是从测试数据集提取的。这个方法唯一需要的是测试域的未标记的例子。这可能导致测试集的数据泄漏。

![](img/01cfd7cc6fb06cdb6d36f716b6d7dffa.png)

On the left, we have our typical training set and in the center our test set. We estimate the data probability of the training and test sets and use this to rescale our training set to produce the training set on the right (notice the size of the points has got larger, this represents the ‘weight’ of the training example).

为了弄清楚这是如何工作的，我们基本上通过训练和测试集的相对概率来重新加权每个训练示例。我们可以通过密度估计、通过核方法(如核均值匹配)或通过区别性重新加权来做到这一点。

**3)对抗性搜索**

*对抗搜索*方法使用对抗模型，其中学习算法试图构建一个预测器，该预测器对测试时的特征删除具有鲁棒性。

该问题被公式化为寻找相对于对手的最优极大极小策略，该对手删除了特征，并且表明该最优策略可以通过求解二次规划或者使用有效的最优化束方法来找到。

文献中已经对协变量移位进行了广泛的研究，并且已经发表了许多关于协变量移位的建议。其中最重要的包括:

*   对数似然函数的加权(Shimodaira，2000 年)
*   重要性加权交叉验证(杉山等人，2007 年 JMLR)
*   集成优化问题。辨别学习。(比克尔等人，2009 年 JMRL)
*   核均值匹配( [Gretton 等人，2009](http://www.gatsby.ucl.ac.uk/~gretton/papers/covariateShiftChapter.pdf) )
*   对抗性搜索( [Globerson 等人，2009](http://www.acad.bg/ebook/ml/The.MIT.Press.Dataset.Shift.in.Machine.Learning.Feb.2009.eBook-DDU.pdf)
*   Frank-Wolfe 算法(【文】等，2015 )

# 最终意见

在我看来，数据集转移是一个极其重要的话题，但却被数据科学和机器学习领域的人们低估了。

鉴于它可能对我们算法的性能产生的影响，我建议花一些时间研究如何正确处理数据，以便给你的模型更多的信心，并希望有更好的性能。

## 时事通讯

关于新博客文章和额外内容的更新，请注册我的时事通讯。

[](https://mailchi.mp/6304809e49e7/matthew-stewart) [## 时事通讯订阅

### 丰富您的学术之旅，加入一个由科学家，研究人员和行业专业人士组成的社区，以获得…

mailchi.mp](https://mailchi.mp/6304809e49e7/matthew-stewart) 

# 参考

[1][http://iwann . ugr . es/2011/pdf/invited talk-FHerrera-iwann 11 . pdf](http://iwann.ugr.es/2011/pdf/InvitedTalk-FHerrera-IWANN11.pdf)

[2] J.G .莫雷诺-托雷斯、t .雷德尔、r .阿莱兹-罗德里格斯、N.V .舒拉、f .埃雷拉。分类中数据转移的统一观点。模式识别，2011，出版中。

[3] J .基诺内罗·坎德拉、m .杉山、a .施瓦格霍夫和 N. D .劳伦斯。机器学习中的数据集转移。麻省理工学院出版社，2009。

[4]雷德尔、霍恩斯和舒拉出版公司。分类器性能评估中分类器可变性的后果。，ICDM 2010 年 IEEE 数据挖掘国际会议论文集。

[5]莫雷诺-托雷斯，J. G .，&埃雷拉，F. (2010 年)。基于遗传规划的特征提取在不平衡领域中重叠和数据断裂的初步研究。《第十届智能系统设计与应用国际会议论文集》(ISDA，2010)(第 501–506 页)。

[6][https://www . NCBI . NLM . NIH . gov/PMC/articles/PMC 5070592/pdf/f 1000 research-5-10228 . pdf](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC5070592/pdf/f1000research-5-10228.pdf)