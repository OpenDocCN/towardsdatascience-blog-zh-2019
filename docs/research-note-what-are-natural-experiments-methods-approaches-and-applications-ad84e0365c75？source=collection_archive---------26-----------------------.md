# 什么是自然实验？

> 原文：<https://towardsdatascience.com/research-note-what-are-natural-experiments-methods-approaches-and-applications-ad84e0365c75?source=collection_archive---------26----------------------->

## 实验和因果推理

## 方法、途径和应用

![](img/602a6054d5d9a1bff50e7a132273f9eb.png)

Photo by [Andreas Gücklhorn](https://unsplash.com/@draufsicht?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/nature?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

## 介绍

我喜欢阅读 [Craig 等人(2017](https://www.annualreviews.org/doi/pdf/10.1146/annurev-publhealth-031816-044327) )关于自然实验的综述文章([关于方法、途径和对公共卫生干预研究的贡献的概述](https://www.annualreviews.org/doi/pdf/10.1146/annurev-publhealth-031816-044327))。在这篇文章中，我想总结它的要点，并附上我对因果推理发展的一些思考。这篇综述文章介绍了什么是 NE 以及对 NE 数据可用的方法和途径。通常情况下，年度综述，比如由[克雷格等人](https://www.annualreviews.org/doi/pdf/10.1146/annurev-publhealth-031816-044327)撰写的这一篇，提供了对该领域近期发展状况和未来方向的快速回顾。这是学习数据科学的好方法。强烈推荐！

## 什么是 NE？

根据英国医学研究委员会的说法，任何不受研究人员控制的事件，将人群分为暴露组和未暴露组。

由于缺乏对分配过程的直接控制，研究人员不得不依靠统计工具来确定操纵暴露于治疗条件的变化的因果影响。

NEs 的关键挑战是排除选择进入治疗组的可能性，这将违反**忽略假设**。这种违反也使得治疗组和对照组不具有可比性，我们不能将结果变量的差异归因于干预的存在。

为了解决这个问题，数据科学家提出了潜在结果框架。POF 代表如果一个人暴露于和不暴露于干预时会发生的结果。

然而，棘手的是，这两种结果中只有一种是可观察到的，我们必须依靠反事实来推断单位之间的平均治疗效果。

如果分配是随机的，如在随机对照试验(RCTs)中，那么治疗组和对照组是可交换的**。我们可以将这两组之间的差距归因于干预的存在。**

如果分配不是随机的，如在 NEs 中，数据科学家必须依靠分配机制和统计方法的领域知识来实现**有条件交换**。

这是定性研究和领域知识发挥作用的时候，并确定在分配过程背后是否有一个因果故事。

我想说，由于实践和伦理的原因，NEs 在现实世界中比 RCT 有更广泛的应用范围。因此，正如 [Craig et al. (2017](https://www.annualreviews.org/doi/pdf/10.1146/annurev-publhealth-031816-044327) )建议的那样，选择合适的方法/设计对 NE 数据进行因果推断变得至关重要。

这样做主要有八种技术。我将在这里用一些研究笔记和实际应用的链接来介绍每种方法。请参考原始文章(这里是)来更全面地讨论每种技术。

## **方法**

1.  [**前后分析**](/a-data-scientists-journey-into-program-evaluation-78cffa5b2ccc) 。就我个人而言，如果没有更好的选择，这将是我的最后手段。没有多个数据点的单个案例比较。我们如何控制混杂因素？不是因果推理的理想选择。
2.  [**回归调整**](https://ijbnpa.biomedcentral.com/track/pdf/10.1186/s12966-016-0356-z) 。当我们试图比较案例时，它有很多应用。
3.  [**倾向得分匹配**](https://www.thelancet.com/action/showPdf?pii=S0140-6736%2808%2961687-6) 。也适用于观测数据，但是 Gary King 最近否定了使用 PSM 的想法。
4.  [**差异中的差异**](/does-minimum-wage-decrease-employment-a-difference-in-differences-approach-cb208ed07327) 。这是一种强有力的因果推理技术，具有简单明了的研究思想。
5.  [**中断的时间序列**](/interrupted-time-series-f59fe5b00b31) 。具有多个数据项的因果方法。
6.  [**合成控件**](https://economics.mit.edu/files/11859) 。这是工业界和学术界的一种流行方法，政治学家为此做出了巨大贡献。简而言之，如果对照组中没有与治疗组匹配的病例，我们可以人为地创建对照组的加权平均值作为基点。例如，我们使用其他案例的加权值创建一个人工控制场景，并比较这两组之间的差异。这是一个如此巧妙的想法，但有潜在的陷阱，对此我将在另一篇文章中详细阐述。
7.  [T5【回归不连续设计】T6](/the-crown-jewel-of-causal-inference-regression-discontinuity-design-rdd-bad37a68e786)。强大的因果技术与一个伟大的视觉插图。
8.  [工具变量 。IV 方法包含很强的推理能力，但是众所周知很难找到。因此，它的应用有限。](http://data.nber.org/ens/feldstein/Papers/_Paper__Economic_Shocks_and_Civil_Conflict.pdf)

## **如何让 NEs 在因果推理上更强？**

这篇综述文章提供了三种解决方案:

1.  加入定性成分以理解工作机制。我的两点看法是，在大数据和机器学习的时代，我们不应忘记定性研究的重要性。足够好地理解这个过程或领域知识，有助于我们开发更好的统计模型。
2.  多种定量方法和目视检查相结合，检查 RDD 和 ITS 的不连续性。目视检查对于识别异常情况至关重要且简单明了。如果可能的话，更多更明智地使用它们。
3.  引入伪造/安慰剂测试来评估因果归因的合理性。例如，我们可以使用不等价的因变量来测试未受干预影响的结果与受干预影响的结果的变化。这里，潜在的想法是使用多个 dv 交叉检查结果，这是一种广泛用于社会科学的研究思想。

*Medium 最近进化出了它的* [*作家伙伴计划*](https://blog.medium.com/evolving-the-partner-program-2613708f9f3c) *，支持像我这样的普通作家。如果你还不是订户，通过下面的链接注册，我会收到一部分会员费。*

[](https://leihua-ye.medium.com/membership) [## 阅读叶雷华博士研究员(以及其他成千上万的媒体作家)的每一个故事

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

leihua-ye.medium.com](https://leihua-ye.medium.com/membership) 

## 实验和因果推理

[](/online-controlled-experiment-8-common-pitfalls-and-solutions-ea4488e5a82e) [## 运行 A/B 测试的 8 个常见陷阱

### 如何不让你的在线控制实验失败

towardsdatascience.com](/online-controlled-experiment-8-common-pitfalls-and-solutions-ea4488e5a82e) [](/the-turf-war-between-causality-and-correlation-in-data-science-which-one-is-more-important-9256f609ab92) [## 相关性并不意味着因果关系。现在怎么办？

### 数据科学中因果关系和相关性之间的地盘之争

towardsdatascience.com](/the-turf-war-between-causality-and-correlation-in-data-science-which-one-is-more-important-9256f609ab92) [](/why-do-we-do-and-how-can-we-benefit-from-experimental-studies-a3bbdab313fe) [## 数据科学家应该有充分的理由进行更多的实验

### 好的，坏的，丑陋的

towardsdatascience.com](/why-do-we-do-and-how-can-we-benefit-from-experimental-studies-a3bbdab313fe) 

# 喜欢读这本书吗？

> 请在 [LinkedIn](https://www.linkedin.com/in/leihuaye/) 和 [Youtube](https://www.youtube.com/channel/UCBBu2nqs6iZPyNSgMjXUGPg) 上找到我。
> 
> 还有，看看我其他关于人工智能和机器学习的帖子。