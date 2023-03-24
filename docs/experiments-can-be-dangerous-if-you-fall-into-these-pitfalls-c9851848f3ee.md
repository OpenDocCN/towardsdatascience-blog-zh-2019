# 受控实验的 3 个常见陷阱

> 原文：<https://towardsdatascience.com/experiments-can-be-dangerous-if-you-fall-into-these-pitfalls-c9851848f3ee?source=collection_archive---------27----------------------->

## **实验和因果推断**

## **不合规、外部有效性和伦理考虑**

![](img/923a8a67f3751dd1f1951f8b17548cc1.png)

Photo by [Michal Matlon](https://unsplash.com/@michalmatlon?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

**2021 年 2 月 21 日更新**

在两篇文章中([为什么要进行更多的实验](/why-do-we-do-and-how-can-we-benefit-from-experimental-studies-a3bbdab313fe)和[双向因果关系](/the-chicken-or-the-egg-experiments-can-help-determine-two-way-causality-723a06c37db7))，我提出了我们应该进行更多受控实验并做出明智的商业决策的想法。实验擅长于提高内部效度，区分因果关系和相关性，以及产生和检验假设。

有一个心理陷阱经常绊倒数据科学家:他们选择最熟悉的方法，而不是选择最能回答问题的方法。

> "如果你只有一把锤子，所有的东西看起来都像钉子."
> 
> 伯纳德·巴鲁克

为了谨慎起见，我在今天的帖子中详细阐述了实验的陷阱。采用这种方法时，数据科学家应该首先考虑研究问题，然后检查方法的适用性和局限性。也就是检查你要回答什么，检查你有什么信息和约束。

## 何时不使用

如果这三种情况中的一种或多种发生了，你可能要对实验三思:

## **1。高度不合规**

采用实验的全部意义在于它的高内部效度，这源于一个随机化过程([我的解释)。](/the-chicken-or-the-egg-experiments-can-help-determine-two-way-causality-723a06c37db7)也就是说，研究参与者被随机分配到治疗组和对照组。

然而，高不符合率或退出率表明设计有问题，并损害了研究的可信度。

这里有一个假设的场景。数据科学和 UX 团队在谷歌合作，试图在搜索算法更新后跟踪用户的行为变化。一位名叫约翰的用户被选中接触新设计，但他在研究中途退出了。例如，用户在一段时间后清空他们的 cookies 后会流失。研究人员争先恐后地寻找与约翰的用户属性相似的替代品。然而，替换不是约翰，并且在实验中间的用户替换对统计假设提出了挑战(例如，缺失数据、依赖性)。

此外，实验组之间的交叉污染会产生有偏见的估计。约翰可能无意中向对照组暴露了治疗状况(例如，更新他的脸书状态并让他的朋友知道该研究)。

## **2。社会实验的外部效度很低**

实验环境并不是真实的世界，一旦用户意识到自己被监视，他们就会改变自己的行为。这两个世界之间存在差距，研究人员应该缩小差距，并确定他们设计的范围和限制。

在社会科学中，有一种现象叫做社会合意性。也就是说，如果可能的话，人们以社会可接受的方式表达自己和行为，并避免对抗。每个人都想成为社区的一部分。

作为一名政治学博士生，这是我最喜欢的现实生活中的例子。在 2016 年的总统选举中，多波民调研究表明希拉里将以较大优势获胜。最终结果出来时，真是一个巨大的惊喜。许多受访者隐藏了他们真实的政治偏好，给出了符合社会需求的回答。

此外，一个地区的实验发现在其他地区并不成立。欧洲用户的认知和行为模式与美国用户不同。任何来自欧洲用户的实验结果在应用于美国的案例时都应该持保留态度。

换句话说，你的研究适用性范围是什么？你的研究有高/低的外部效度吗？当你试图推广到更广泛的案例时，请格外小心，因为实验案例是独特的。

## **3。道德考量**

一些实验给研究参与者带来压力和伤害。例如，研究人员曾经用积极和消极的情绪操纵社交媒体订阅，试图看看人类的情绪如何在虚拟环境中传播。

在这种情况下，研究人员应该告知研究参与者任何潜在的风险和危害。在用户明确同意参与之前，研究不应继续。

权衡的结果是，如果用户知道他们正在参与一项研究，他们可能会以一种意想不到的方式行事。此外，研究参与者和非参与者之间可能存在根本不同的属性。这两种情况都对我们的结果产生了有偏见的估计。

## 外卖食品

1.  **不合规**

*   我们需要多少个实验组？
*   如何控制交叉污染？
*   我们如何更好地构建潜在的结果框架(例如，提供积极的激励，如亚马逊礼品卡)？

**2。外部有效性**

*   我们的研究条件看起来是真的还是假的？
*   我们研究的局限性是什么？有更好的选择吗？
*   如何获得真诚的回应？

**3。道德考量**

*   讲道德！
*   通知用户
*   明确同意参与

## 进一步阅读:

[](/the-turf-war-between-causality-and-correlation-in-data-science-which-one-is-more-important-9256f609ab92) [## 相关性并不意味着因果关系。现在怎么办？

### 数据科学中因果关系和相关性之间的地盘之争

towardsdatascience.com](/the-turf-war-between-causality-and-correlation-in-data-science-which-one-is-more-important-9256f609ab92) [](/why-do-we-do-and-how-can-we-benefit-from-experimental-studies-a3bbdab313fe) [## 数据科学家应该有充分的理由进行更多的实验

### 好的，坏的，丑陋的

towardsdatascience.com](/why-do-we-do-and-how-can-we-benefit-from-experimental-studies-a3bbdab313fe) 

# 喜欢读这本书吗？

> 请在 [LinkedIn](https://www.linkedin.com/in/leihuaye/) 和 [Twitter](https://twitter.com/leihua_ye) 上找到我。
> 
> 还有，看看我其他关于人工智能和机器学习的帖子。