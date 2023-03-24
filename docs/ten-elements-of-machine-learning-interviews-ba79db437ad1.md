# 机器学习面试的十个要素

> 原文：<https://towardsdatascience.com/ten-elements-of-machine-learning-interviews-ba79db437ad1?source=collection_archive---------36----------------------->

## 非常有用的在线资源列表。

作为一名博士生，我对 ML 算法有相当好的理解，但仍然发现机器学习面试具有挑战性。挑战来自于这样一个事实:在一个 ML 项目中，有比模型拟合更多的东西。大多数教科书都涵盖了 model.train()背后的技术细节。从一个 ML 教科书中推导出所有的东西，并为一个用例找到合适的 ML 解决方案，这需要两套非常不同的技能。此外，分解一个 ML 项目的复杂性，并在压力下连贯地讨论它不是开玩笑的。在准备科技公司的机器学习面试时，我试图保持一个广泛的范围，并对最广泛使用的 ML 用例进行逆向工程。

![](img/5930e286a652b8e247c4755d5e8cfce0.png)

Photo by [Kaleidico](https://unsplash.com/@kaleidico?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/whiteboard?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

1.  除了“训练验证和测试”之外，这个脸书指南还是一个很好的通用框架这里的重点是:寻找数据来源和实验。ML 项目通常关注如何处理给定的数据，但更重要的问题是如何找到相关的数据。实验是另一个很少被讨论的方面，因为大多数 ML 模型都没有被部署。

[](https://research.fb.com/blog/2018/05/the-facebook-field-guide-to-machine-learning-video-series/) [## 介绍机器学习视频系列的脸书现场指南

### 《脸书机器学习指南》是由脸书广告机器学习公司开发的六集视频系列

research.fb.com](https://research.fb.com/blog/2018/05/the-facebook-field-guide-to-machine-learning-video-series/) 

2.Andrej Karpathy 的博客文章非常棒，因为它概述了进行深度学习的高度实用的过程，即从最简单的基线开始，并从那里逐步迭代。我非常相信这个过程。这与许多人的做法相反，他们有立即使用最复杂模型的冲动。此外，从面试的角度来看，说出漂亮模特的名字比给人留下深刻印象更危险，除非你真的很了解自己。

 [## 训练神经网络的方法

### 几周前，我发表了一篇关于“最常见的神经网络错误”的推文，列出了一些常见的问题，涉及…

karpathy.github.io](http://karpathy.github.io/2019/04/25/recipe/) 

3.这篇 Airbnb 博客文章涵盖了他们如何在几个不同的阶段建立体验搜索。他们分享了这么多关于他们如何解决这个问题的细节，这太棒了。当你对问题的正确方面进行优先排序时，解决方案就会自然而然地出现。在 ML 面试中，确定主要目标并意识到限制因素是至关重要的。

[](https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789) [## 基于机器学习的 Airbnb 体验搜索排名

### 我们如何为一个新的双边市场构建和迭代机器学习搜索排名平台，以及我们如何…

medium.com](https://medium.com/airbnb-engineering/machine-learning-powered-search-ranking-of-airbnb-experiences-110b4b1a0789) 

4.我认为凯文·墨菲的书对于有研究生水平数学背景并有耐心读完它的人来说是一个很好的参考。然而，Hal Daumé III 的书要简洁得多，下面的备忘单包含了大多数突击测验的概念。你不想在一个基本的问题上空白或者在不必要的数学上花费时间(为了面试)。

 [## 机器学习课程

### 机器学习是对从数据和经验中学习的算法的研究。它被广泛应用于…

ciml.info](http://ciml.info/) [](https://stanford.edu/~shervine/teaching/cs-229/) [## 教学- CS 229

### 您想看看这套母语的备忘单吗？你可以在 GitHub 上帮我们翻译！我的…

stanford.edu](https://stanford.edu/~shervine/teaching/cs-229/) 

5.如果像大多数 Kaggle 竞赛中一样有足够多的高质量标记数据，使用随机森林或梯度提升就可以达到目的。最有可能的是，你不会击败 Kaggle 赢家，但在实践中，谁会关心 0.1%的改善？《统计学习元素》有关于随机森林和梯度推进的精彩章节，但下面是一篇较短的文章。此外，了解 scikit-learn 中特性的重要性是如何计算的也是值得的。

[](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d) [## 从零开始的渐变提升

### 简化复杂的算法

medium.com](https://medium.com/mlreview/gradient-boosting-from-scratch-1e317ae4587d) [](https://stats.stackexchange.com/questions/311488/summing-feature-importance-in-scikit-learn-for-a-set-of-features) [## 总结 Scikit 中的功能重要性-了解一组功能

### 是的，这是对特征集合的重要性求和的完全正确的方法。在 scikit-learn 中，重要性…

stats.stackexchange.com](https://stats.stackexchange.com/questions/311488/summing-feature-importance-in-scikit-learn-for-a-set-of-features) 

6.西蒙·芬克为网飞奖设计的矩阵分解算法出奇的简单，其背后的逻辑也很直观。我们只是简单地将用户向量和项目向量线性投影到一个潜在空间中，在这个空间中，它们的相似性决定了推荐的概率。有人会说学习合适的潜在表征是最大似然法的本质。

[](https://sifter.org/~simon/journal/20061211.html) [## 网飞更新:在家试试这个

### >]网飞更新:在家里试试这个[后续]好了，我在这里告诉大家我(现在我们)是如何成为…

sifter.org](https://sifter.org/~simon/journal/20061211.html) 

7.然而在实践中，与网飞竞赛相比，有许多用例特定的考虑事项。网飞竞赛基本上有固定的数据，但如果新的数据不断出现呢？此外，如果我们可以合并关于项目本身的数据，会怎么样呢？这个谷歌课程提供了更全面的指导。

[](https://developers.google.com/machine-learning/recommendation/overview) [## 建议:什么和为什么？推荐系统

### YouTube 怎么知道你接下来可能想看什么视频？谷歌 Play 商店如何挑选一款适合你的应用程序…

developers.google.com](https://developers.google.com/machine-learning/recommendation/overview) 

8.这篇 Instagram 博文无疑将推荐系统设计提升到了一个新的高度。它建立在上面提到的东西之上，并针对它们的特定用例进行了大量优化。

[](https://instagram-engineering.com/core-modeling-at-instagram-a51e0158aa48) [## Instagram 上的核心建模

### 在 Instagram，我们有许多机器学习团队。虽然他们都在产品的不同部分工作，但他们都…

instagram-engineering.com](https://instagram-engineering.com/core-modeling-at-instagram-a51e0158aa48) 

9.这个脸书视频从模型部署的角度提供了一个视角，并令人惊讶地揭示了他们的 ML 解决方案在高层次上的几个用例。任何想用深度神经网络解决每个问题的人都应该看看这个视频。

10.回到实验这个话题，每个人都听说过多臂强盗，但它实际上是如何工作的呢？

[https://peterroelants . github . io/posts/multi-armed-bandit-implementation/](https://peterroelants.github.io/posts/multi-armed-bandit-implementation/)