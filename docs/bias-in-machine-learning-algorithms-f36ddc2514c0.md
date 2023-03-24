# 机器学习算法中的偏差

> 原文：<https://towardsdatascience.com/bias-in-machine-learning-algorithms-f36ddc2514c0?source=collection_archive---------10----------------------->

## 算法何时出现偏差以及如何控制。

![](img/003e1d033ae5450f9f7743df2c9546c4.png)

source: [google](https://www.google.com/imgres?imgurl=https%3A%2F%2Fcdn.drawception.com%2Fimages%2Fpanels%2F2018%2F5-5%2F6EanFXXCnn-14.png&imgrefurl=https%3A%2F%2Fdrawception.com%2Fgame%2F6EanFXXCnn%2Fi-love-you-panel-11%2F&docid=g5JGjNQHIqkysM&tbnid=rJumnv8raIWcqM%3A&vet=10ahUKEwj1g5399rvhAhWYHDQIHeymAAQQMwg_KAEwAQ..i&w=300&h=250&itg=1&client=safari&bih=737&biw=1280&q=robot%20saying%20not%20allowed&ved=0ahUKEwj1g5399rvhAhWYHDQIHeymAAQQMwg_KAEwAQ&iact=mrc&uact=8)

# 机器学习模型中的偏差

我参加了 Sharad Goyal 教授关于我们机器学习模型中各种类型的偏见的演讲，以及他最近在斯坦福计算政策实验室的一些工作的见解。这真的让我很兴奋，我做了一些研究，写了这篇关于机器学习偏见的笔记。让我们来谈谈偏见，以及为什么我们需要关心它。

# 机器学习中的偏差是什么？

偏见这个术语是由 Tom Mitchell 在 1980 年他的论文中首次提出的，这篇论文的标题是“[学习归纳中对偏见的需要](http://dml.cs.byu.edu/~cgc/docs/mldm_tools/Reading/Need%20for%20Bias.pdf)”。有偏差的想法是模型重视一些特征，以便更好地对具有各种其他属性的较大数据集进行概化。ML 中的偏差确实有助于我们更好地进行归纳，并使我们的模型对某些单个数据点不那么敏感。

# 对谁有影响？

但是，当我们对更一般化算法的假设产生系统性偏见的结果时，问题就出现了。很多时候，算法可能在某些特征上有偏差，即使我们忽略了我们的模型不需要加权的特征。他们通过从其他提供的特征中学习那些特征的潜在表示来做到这一点。

这是令人担忧的，因为机器学习模型已经开始在我们生活的各种关键决策中发挥更大的作用，例如；贷款申请、医疗诊断、信用卡欺诈检测、来自 CCTV 的可疑活动检测等。因此，机器学习中的偏见不仅会给出基于社会刻板印象和信仰的结果，还会在社会中放大它们。

# 如何识别和消除偏见？

你们中的许多人一定听说过亚马逊试图建立一个[简历过滤工具](https://www.reuters.com/article/us-amazon-com-jobs-automation-insight/amazon-scraps-secret-ai-recruiting-tool-that-showed-bias-against-women-idUSKCN1MK08G)，以及它最终是如何对女性产生偏见的。每个人都希望有一个能从数百份简历中筛选出五份简历的系统。当模型开始根据性别拒绝申请者时，它就出了问题，因此从未投入生产。另一个简单的例子可以在[单词嵌入](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)中看到。单词嵌入的引入对于各种自然语言理解问题来说是一场革命，因为它们能够捕捉单词之间的简单关系:

> 国王——男人+女人=王后

相同的词嵌入也捕捉到以下关系:

> 电脑程序员——男人+女人=家庭主妇

从数学上来说，这些单词嵌入可能没有错，并且正确地编码了数据，但是我们愿意在我们的机器学习算法中编码这些偏差吗？因此，我们需要消除这些偏差，使之成为我们算法中不可或缺的一部分。

对于研究机器学习各个方面的研究人员来说，这是一个非常活跃有趣的话题。我还查看了[Sharad Goel](https://profiles.stanford.edu/sharad-goel)教授和 [James Zou](https://profiles.stanford.edu/james-zou) 教授分别对我们的机器学习算法和[在单词嵌入](https://arxiv.org/abs/1607.06520)中的偏差进行的[公平性测量。两者都试图解决同一个问题，但方法不同。詹姆斯教授最近的工作是量化和消除单词嵌入中的性别和种族偏见。在他题为《男人对于电脑程序员就像女人对于家庭主妇一样？去偏置单词嵌入](https://5harad.com/papers/fair-ml.pdf)”，他们与托尔加·博鲁克巴斯提出了一种方法，以几何方式修改嵌入，以消除性别刻板印象。最终的嵌入保留了女王和女性之间的关联，丢弃了接待员和女性之间的关联。鉴于 Sharad 教授最近的工作，“[公平的衡量和错误衡量:公平机器学习的批判性评论](https://arxiv.org/abs/1808.00023)”讨论了各种公平定义(如反分类、分类奇偶校验以及校准)如何受到显著的统计限制，并可能损害它们旨在保护的群体。他们还认为，基于对一个人可能产生的风险的准确估计，我们应该以同样的方式对待同样有风险的人。对机器学习中的公平和偏见的研究刚刚开始，看看它如何发展和塑造下一代的政策将是有趣的。