# 每个人都应该知道的三个关键误差区间

> 原文：<https://towardsdatascience.com/three-key-error-intervals-everyone-should-know-658e2d8bfe?source=collection_archive---------31----------------------->

说“推断统计”这个词，人们通常会惊恐地睁大眼睛，但他们不必如此可怕，一旦你知道他们在做什么，他们就可以成为有用的工具。

不管你喜不喜欢，数据科学是围绕着解释数据和讲述数据给我们带来的故事而建立的。为此，对于人们要有信心( [p 值](https://en.wikipedia.org/wiki/P-value)有人吗？)在我们所说的话中，我们经常需要量化我们给出的价值。这要求我们对统计学和误差估计有所了解和理解。例如，能够辨别一个图表上的一个大的突然变化是真实的还是不真实的，可能是一个公司花费资源去纠正它的区别。

出于这个原因，当结果被呈现时，你能理解、解释和说明你可能看到或使用的三种主要类型的区间是什么是至关重要的。因此，我将讨论置信区间和它的另外两个兄弟姐妹(预测性和耐受性)，以及置信水平。

![](img/02b3298550f6ee839674ba8555456051.png)

Photo by [Isaac Smith](https://unsplash.com/@isaacmsmith?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 我们为什么关心？

很常见的是，当我们研究某一特定人群时，我们收集的数据只是该总人群的一个样本(例如，因为它太耗时或太大而无法收集)，然而，我们经常希望使用该样本来“推断”他们所来自人群的一些情况。每个人最熟悉的最常见的统计数据是平均值。

样本的均值是总体均值的无偏估计量，但它是一个“[点估计](https://en.wikipedia.org/wiki/Point_estimation)，它只给出一个要处理的值，如果我们从总体中随机选择另一个数据样本，它很可能会给出不同的均值。因此，计算一个“区间估计”通常是很好的，这是我们为感兴趣的总体参数构造一个似是而非的值的地方。我今天要讨论的三个区间涵盖了不同的区间估计，对回答不同的问题很有用。但是我们的区间估计有多宽呢？一般来说，这是由我们采集的样本数量和我们希望区间估计具有的置信度决定的(请查看下面的链接以了解更多信息)。

这些区间估计可以总结为:

*   **置信区间**——我想要一个总体参数的可信值范围，比如它的平均值
*   **预测区间** —我想知道未来的观察值可能是多少，给定我从先前的观察中所知道的
*   **容差区间** —我想知道包含一定比例/百分比人口的数值范围

# 置信级别

![](img/df21ce3849f54e5a713d72015400e97c.png)

Photo by [Chris Liverani](https://unsplash.com/@chrisliverani?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

阅读任何科学论文，你都可能听到关于置信水平的讨论(不要与置信区间混淆)。这通常以百分比的形式引用，常见值为 90%、95%和 99%(尤其是对生命至关重要的应用)。但是它们是什么意思呢？一个常见的误解是，它们告诉您真实(或真实)总体参数包含在该区间内的概率，而它真正的意思是，如果您重复采样并无限次计算该区间，置信水平是包含真实值的区间的百分比。这似乎不是一个很大的区别，但这是一个微妙的区别，它基于频率统计的基础。在大多数情况下，这种微妙之处并不重要，但如果有人向你挑战，知道这一点是有好处的。你会发现，对于给定的样本量，置信水平越高，区间估计覆盖的似是而非的值的范围就越大。在极端情况下，如果您使用 100%的置信水平，则计算的区间将包含参数可能取的所有可能值。当有人问生产过程中要破坏性测试多少个器件，以确保该批产品 100%可靠时，这通常会引起惊讶。

**注意:**如果您确实想要一个区间来说明真实值包含在这些值中的概率，那么我会查看基于贝叶斯统计的可信区间(我在这里不涉及这些，但是这里的、这里的和这里的是一些介绍链接)。

# 置信区间

这可以针对几个群体参数(中值、平均值等)进行计算。)但最常用于总体均值。这将给出总体均值的一系列似是而非的值(给定置信水平和数据样本)，上限和下限称为置信上限和置信下限。这些的确切公式取决于被计算的人口参数，但是如果你想知道更多，我在这里提供了一些链接( [1](https://www.mathsisfun.com/data/confidence-interval.html) ， [2](http://www.usablestats.com/tutorials/CI) )。这个区间没有给出从总体中的单个样本中可能得到的值的范围。这就是预测区间可以发挥作用的地方。

# 预测区间

这种情况通常出现在回归分析中，此时您对数据进行了直线拟合，并想知道另一个样本的值可能在哪里。它使用你先前样本的信息来构建它。它们可用于计算参考范围，参考范围用于确定测量值的正常范围。例如，给定一条生产线上的测试样品，下一个样品应在此范围内(如果预测间隔超过工程公差，那么您可以在样品生产或测试前就如何处理做出决定，等等。).你可以在这里阅读更多关于预测区间的内容( [1](http://www.oswego.edu/~srp/stats/pi.htm) 、 [2](https://robjhyndman.com/hyndsight/intervals/) 、 [3](https://newonlinecourses.science.psu.edu/stat501/node/274/) )。

# 公差区间

这是一个没有多少人听说过，但在工业上有很多有趣的用途和应用。这个区间是一个似是而非的数值范围，包含了一部分人口。我个人曾在[可靠性工程](https://accendoreliability.com/tolerance-intervals-for-normal-distribution-based-set-of-data/)中见过它的使用，在那里它被用来获得正在制造的产品的使用值，以确定实际的上限值和下限值来测试设备的可靠性(即 95%的人会在两年保修期内至少打开这个开关两次，最多 20，000 次)。我第一次在一本旧的海军条例手册中读到这些(他们想确定一定比例的射击落在目标的一定范围内)，但如果你环顾四周，你可以在网上找到更多现代来源( [1](https://machinelearningmastery.com/statistical-tolerance-intervals-in-machine-learning/) 、 [2](https://www.qualitydigest.com/inside/statistics-column/010416-statistical-tolerance-intervals.html) 、 [3](https://support.minitab.com/en-us/minitab/18/help-and-how-to/quality-and-process-improvement/quality-tools/how-to/tolerance-intervals-normal-distribution/methods-and-formulas/methods-and-formulas/) )。

关于公差区间的最后一节还提出了这样一个事实，区间不必同时有上限和下限。你可以计算它们只有一个上限或下限。在容差区间的例子中，这在可靠性方面很有用，因为您可能想知道 0 到 95%的客户使用值，并测试上限的极值。

我希望这是一个有趣的，如果短期的高层次的看那种在你的数据科学工具包中有好处的统计。