# 抽样分布，重点是吉布斯抽样，实践和代码

> 原文：<https://towardsdatascience.com/can-you-do-better-sampling-strategies-with-an-emphasis-on-gibbs-sampling-practicals-and-code-c97730d54ebc?source=collection_archive---------14----------------------->

![](img/67c342465b837d1e13530f3b5851196f.png)

Image by Author

继[上一篇关于用 bootstrap 重采样估计置信区间的文章](https://medium.com/@aliaksei.mikhailiuk/a-note-on-parametric-and-non-parametric-bootstrap-resampling-72069b2be228)之后，我想简要回顾一下如何估计没有封闭解的复杂后验分布。贝叶斯推理就是一个很好的例子。这里，一个提示帮助来自采样。

# 我们要解决什么？

可以说，贝叶斯推理最难的部分是后验估计，多年来，贝叶斯推理一直把这个领域蒙在鼓里。在早期(在高计算能力出现之前)，计算后验概率的方法通常依赖于某种近似(例如拉普拉斯近似)或数值积分。然而，随着计算机的发展及其增强的内存和速度，采样方法开始获得动力:从实践中落后的非常简单的方法(反转采样)到更复杂的方法，如 Metropolis-Hasting (MH)算法，该算法被列为 20 世纪影响科学和工程发展的十大算法之一。

当开始阅读该主题时，最常见的问题是采样如何帮助估计后验概率，以及它与积分有什么关系？

抽样的关键思想是从感兴趣的分布中生成值，然后应用离散公式来计算必要的统计数据。根据[大数定律:](https://en.wikipedia.org/wiki/Law_of_large_numbers)，考虑一个计算分布期望值的例子

![](img/3e682cc59bbccc8627556356bf1c9a3e.png)

这里一个复积分被一个离散的公式所代替，只包括原始函数 f(x)。这让我们想到了**蒙特卡罗方法**(或**蒙特卡罗实验**)——一类依靠重复随机采样来获得数值结果的计算算法。这里讨论的采样算法属于蒙特卡罗方法的范畴

在写这篇文章的时候，我有机会修改了最常见的采样算法，并找到了几个链接，感兴趣的读者可能会觉得有用。下一章是对采样算法的快速回顾。那些只对吉布斯采样感兴趣的人可以跳过它。

# 取样策略，概述

这里我简单解释一下常用的抽样方法:倒置抽样、拒绝抽样和重要性抽样。那些只对吉布斯采样感兴趣的人可以跳过这一节。

## 倒置取样

给定任何累积分布函数 F(x ),可以按照以下步骤从其生成随机变量:

1.  从均匀 U~[0，1]生成样本
2.  为 x 求解 F(X)=U，或者等价地求出 X = F^(-1)(U).
3.  返回 x。

![](img/f74ae040427eb7ced0309cc402c75d6c.png)

Figure 1: Inversion sampling. Image by Author.

这里，我们按照均匀分布对 y 轴进行采样。当 x，f(x)的概率较低时，cdf F(x)将是平坦的，即从均匀分布中抽取的样本不太可能落在曲线上。x，f(x)概率较高的区域倾向于平行于 y 轴，这意味着均匀分布的样本很可能落在 y 轴上。

**为什么不在实践中使用？**

求 cdf 的逆可能是复杂的，如果 cdf 不可用，积分仍然是找到它所必需的，所以我们回到原始问题。

**延伸阅读**

反转采样的另一个例子在[这篇](https://www.scratchapixel.com/lessons/mathematics-physics-for-computer-graphics/monte-carlo-methods-mathematical-foundations/inverse-transform-sampling-method)文章中给出。

## 拒绝抽样

拒绝采样允许从分布中采样，已知比例常数，但是太复杂而不能直接采样。

为了使用拒绝抽样，需要有一个简单的建议分布 q(x ),从中可以抽取样本。还选择常数 k，使得分布 k*q(x)总是包围或覆盖所有 x 的分布 p(x)，我们从分布 k*q(x)中采样，如果 u*k*q(x(i)) < p(x(i)) accept the sample, here u is coming from a uniform distribution.

![](img/0f3da89d4fd6fd4c0084fca7f9446915.png)

Figure 2: illustration of rejection sampling. Image by [G. Pilikos](https://scholar.google.com/citations?hl=en&user=WALTgpEAAAAJ&view_op=list_works&sortby=pubdate).

**为什么可能不工作？**

使用剔除取样时会面临几个问题。首先，由于 k 控制建议分布对感兴趣的分布的覆盖，所以需要找到 k，它不需要过多的采样，并且 q(x)几乎不覆盖 p(x)。因此，在采样的覆盖和冗余之间有一个由 k 控制的折衷。通常，甚至不可能找到正确的 k 或建议分布来覆盖兴趣分布。

**延伸阅读**

我发现[这篇](https://wiseodd.github.io/techblog/2015/10/21/rejection-sampling/)文章在回顾剔除抽样时非常有用。它有代码和很好的图形来显示正在发生的事情。

## 重要性抽样

与前面讨论的方法不同，重要性抽样用于直接逼近函数 f(x)的期望值。该方法与拒绝抽样非常相似，也需要一个建议分布。我不会说太多的细节，但是，更多的细节你可以看看[这篇](http://dept.stat.lsa.umich.edu/~jasoneg/Stat406/lab7.pdf)文章，我发现它非常有用。

# 吉布斯采样

上述所有方法在低维时都是有效的，然而，当维数增加时，用这些方法采样就成问题了。需要使用不同种类的方法。这类方法基于马尔可夫链蒙特卡罗(MCMC)。马尔可夫链是一个随机过程，在给定当前状态的情况下，未来状态独立于过去状态。当不可能从 p(x)中提取样本，而只能评估 p(x)直到一个归一化常数时，使用 MCMC。关于马尔可夫链我就不赘述了，不过，有兴趣的读者可以在这里找到很好的解释[。](http://setosa.io/ev/markov-chains/)

**Gibbs sampling** 是马尔可夫链蒙特卡罗采样器，是一族**Metropolis**-**Hasting(MH)算法的特例(简化情况)。Metropolis-Hastings (MH)算法是最流行的 MCMC 采样器，许多实际的 MCMC 采样器可以解释为 MH 算法的特例。以与拒绝和重要性采样类似的方式，它需要一个建议分布 q(x ),并涉及给定当前状态的候选 x 的采样。也就是说，从 q(x(i+1)|x(i))中抽取新的样本，即给定当前状态。然而，该样品不能保证被接受。类似于拒绝采样，只有当条件满足时，我们才移动到 *x* ，否则我们停留在当前状态并尝试新的样本。在区间[0；1]，然后与 A(x(i+1)，x(i))进行比较，其定义为:**

![](img/e2e3c206041b8c95f947aa2f42d9dfdc.png)

Acceptance probability for the MH algorithms.

我就讲到这里，不过，如果你对 MH 算法的更多细节感兴趣，你可以看看这里的。这一步要注意的是，对于吉布斯抽样，接受概率总是 1。除此之外，它与 MH 采样器非常相似。

Gibbs 抽样的思想是将高维联合分布的抽样问题分解为低维条件分布的一系列样本。这里，我们通过扫描每个变量(或变量块)来生成后验样本，以从条件分布中采样，并将剩余变量设置为其当前值。显然，我们能否抽样取决于我们能否导出条件后验分布。下面给出了算法。

![](img/de5af750bc26597b5888273b212bf056.png)

Alg 1: Gibbs sampling.

这个过程一直持续到收敛(样本值具有相同的分布，就好像它们是从真实的后验联合分布中采样的一样)。因为我们用随机值初始化算法，所以在早期迭代中基于该算法模拟的样本不一定代表实际的后验分布。这被称为预烧期。

## 为什么会有用？

假设我们想从 p(x，y)的联合分布中取样，然后，应用概率规则，我们可以在 p(x|y)或 p(y|x)的条件下导出 p(x，y)的关系。

![](img/a0c464e6eaf30ee4379e3b22673ff34e.png)

通过在 x 和 y 之间交替，我们可以从原始的联合分布中取样。

# 例子

这里我考虑两个例子，在第一个例子中，我们想使用 Gibbs 抽样来估计 1D 高斯的矩。然后，我将考虑同一个例子，但有一个更复杂的情况——多元高斯分布。对于这两种情况，我都提供了一个 [jupyter 笔记本](https://github.com/mikhailiuk/medium/blob/master/Gibbs-sampling.ipynb) 和代码。

## 1D 高斯

考虑一个经常发生的例子，我们有一些数据，比如说工人上班迟到的时间。这些数据看起来与正态分布的数据非常相似。作为一个贝叶斯主义者，你决定开发一个模型。这里你要估计均值和标准差。给定数据的平均值和方差的全概率模型如下所示:

![](img/f45703857f8eb99d70a9207596ce8203.png)

这里我们有 X 给定均值和方差乘以它们的联合先验的可能性。假设独立，先验可以拆分。理想情况下，我们希望有一些信息性的先验，但是事先没有信息的情况下，采用参数 *a* 和 *b* 都趋向于零(这将产生 1/方差的先验)的方差和均值方差较大的正态分布。

![](img/8bd18fc80ac4c448597eb925f6e86972.png)![](img/20fa3a66409cabdaa7608180b746b9b7.png)

Inverse gamma prior for variance (top) and normal for mean

然后我们可以展开高斯分布。因此，似然项可以写成如下形式:

![](img/8ea29584f46f37bbcebd538c7908e55c.png)

要获得平均值的后验概率，首先，将所有观察值相乘，然后展开二次项，消除相对于恒定的项，并首先用均方项重新排列这些项。对方差采取类似的程序。这产生了均值的后验概率，以方差和数据为条件，方差也类似，以均值和数据为条件:

![](img/674f565f48d7ff04b944250194be5e71.png)

Posterior of the mean

![](img/74bcab0032af562168b67a0905aa1cc1.png)

Posterior of the variance

在示例中，我们从平均值为 2、标准差为 5 的正态分布中生成数据。经验平均值为，标准偏差为 2.1，标准偏差为 4.959。我们可以通过在两者之间交替来应用吉布斯采样，得到下面的估计值 mean = 2.094，std = 4.932。

![](img/16e79dcf924d5119bf95ce4e5bd3c5ce.png)![](img/0c3308350f47c290b8f8c598ec17e4a9.png)![](img/97f9b439a3d21a85577d6674637558ae.png)

Figure 3: Mean and Variance evolving over time for 1D Gaussian, distribution of the data and estimated model. Image by Author.

## 2D 高斯

使用 1D 高斯扩展该示例，我们可以从联合分布开始，导出给定数据的均值和协方差矩阵的后验概率:

![](img/1a1aecd555215a5da9320aa7b83f7468.png)

以均值和数据为条件的协方差矩阵的后验分布遵循具有 n 个自由度的逆 Wishart 分布，并且给定协方差矩阵和数据的均值遵循正态分布:

![](img/71ae27b3d3f69f29d0236919cd2f7fb6.png)

我们用 1 和 3 个均值、2 和 10 个标准差和 0.9 个协方差对多元高斯数据进行采样。经验均值分别为 1.01 和 2.98，吉布斯抽样估计值分别为 1.00 和 2.98。经验标准差和协方差分别为 1.95、10.07 和 0.85。吉布斯抽样估计那些有 1.95，10.22 和 0.95 的。

![](img/280aed91eb3a84c08e2638151044068c.png)![](img/ef686d46778810f13617377027c55090.png)

Figure 5: Estimated distribution and the data, evolution of covariance. Image by Author.

![](img/0743c49126415b6405dbd1652cfc96e0.png)![](img/e5557fb16340f00a546a6645f5e4fdde.png)

Figure 4: Evolution of means standard deviations. Image by Author.

## 利弊

使用吉布斯采样的缺点包括:

1.  收敛时间长，尤其是随着数据维数的增加。收敛时间也取决于分布的形状。
2.  很难找到每个变量的后验概率。
3.  一般而言，来自链起点的样本(*预烧期*)可能无法准确代表期望的分布。因此它经常被丢弃。也可以使用稀疏化——即保持马尔可夫链的每第 k 个*样本。*

然而这些往往被压倒，记住，MH 算法被评为 20 世纪影响科学与工程发展的十大算法。

# 进一步阅读

与许多其他抽样策略不同，吉布斯抽样需要了解几个领域，因此，可能需要进一步阅读概率的基础知识。在准备这篇文章的时候，我从林奇的[社会科学家应用贝叶斯统计和评估简介](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.177.857&rep=rep1&type=pdf)中找到了有用的第 4 章。这本书是用通俗易懂的语言写的，第一章解释了基本知识。罗切斯特大学的笔记中还有另一个吉布斯抽样的实际例子。你可能也想在 stats stackexchange 上阅读[这个回复](https://stats.stackexchange.com/questions/185631/what-is-the-difference-between-metropolis-hastings-gibbs-importance-and-rejec)。

本文的 jupyter 笔记本可从[这里](https://github.com/mikhailiuk/medium/blob/master/Gibbs-sampling.ipynb)获得。

## 喜欢作者？保持联系！

我错过了什么吗？不要犹豫，直接在 [LinkedIn](https://www.linkedin.com/in/aliakseimikhailiuk/) 或 [Twitter](https://twitter.com/mikhailiuka) 上给我留言、评论或发消息吧！

[](/a-note-on-parametric-and-non-parametric-bootstrap-resampling-72069b2be228) [## 置信区间:参数和非参数自助重采样

### 制作一个好模型需要花费数天时间。但是一旦模型在那里，一个狡猾的问题潜伏在里面:“它能告诉我什么…

towardsdatascience.com](/a-note-on-parametric-and-non-parametric-bootstrap-resampling-72069b2be228) [](/dataset-fusion-sushi-age-and-image-quality-and-what-the-hell-do-they-have-in-common-814e8dae7cf7) [## 面向大规模偏好聚合的数据集融合

### 融合评分和排序数据集的高效简单的概率框架。示例和代码。

towardsdatascience.com](/dataset-fusion-sushi-age-and-image-quality-and-what-the-hell-do-they-have-in-common-814e8dae7cf7) [](/active-sampling-for-pairwise-comparisons-476c2dc18231) [## 成对比较的主动采样

### 如何配对玩家，以便在尽可能少的游戏中知道排名，同时游戏体验质量…

towardsdatascience.com](/active-sampling-for-pairwise-comparisons-476c2dc18231)