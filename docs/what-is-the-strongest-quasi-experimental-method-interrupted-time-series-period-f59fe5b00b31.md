# 中断时间序列实践指南

> 原文：<https://towardsdatascience.com/what-is-the-strongest-quasi-experimental-method-interrupted-time-series-period-f59fe5b00b31?source=collection_archive---------5----------------------->

## 实验和因果推理

## 基础、假设、优点、限制和应用

![](img/63c97969de7adc4a221e917239c8ca19.png)

Photo by [ahmadreza sajadi](https://unsplash.com/@ahmadreza_sajadi?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/time?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

# **背景**

在因果推断的世界里，随机对照试验(RCT)被认为是黄金标准，因为它在干预前排除了任何协变量差异。然而，由于多种原因(例如，太昂贵、无效假设、太长、不道德等)，运行 RCT 不是一个选项。).

在这种情况下，间断时间序列(ITS)的设计就派上了用场(参见[网飞](https://medium.com/netflix-techblog/quasi-experimentation-at-netflix-566b57d2e362))。作为一种准实验方法，它具有很强的推理能力，在流行病学、药物研究和项目评估中有广泛的应用。

可以说，ITS 是因果推理中最强的准实验方法(【彭福】和张，2013 )。

在这篇文章中，我们将学习这种方法的基础，以及如何在现实生活中应用它。

# **什么是 ITS？**

作为一种准实验设计，ITS 是对干预前后单个时间序列数据的分析( [Bernal，et al. 2017](https://www.ncbi.nlm.nih.gov/pubmed/27283160) )。从研究设计的角度来看，它建立在一个相当简单的设计理念上:如果没有干预，结果变量不会改变。

然而，棘手的是:

> 我们如何从单一时间序列数据中推导出因果论证？
> 
> 怎样才能消除混杂因素？

换句话说，创建作为基线点的“反事实”至关重要。我们可以把“改变的”轨迹归因于干预的存在。

幸运的是，正如它的名字所暗示的，有一个时间成分，允许我们假设如果没有干预，结果变量不会改变。

此外，如果有多个数据条目(参见[网飞](https://medium.com/netflix-techblog/quasi-experimentation-at-netflix-566b57d2e362)的示例)，我们可以检查去除治疗条件后，结果变量是否回到基线。

此外，我们必须控制随时间变化的混杂因素，包括可能干扰结果的季节性趋势和并发事件。

例如，研究人员质疑并否定了之前的发现，即 2008 年的大衰退导致了美国更多的自杀事件，认为之前的研究没有考虑季节性和社会群体([哈珀和布鲁克纳](https://www.sciencedirect.com/science/article/abs/pii/S1047279716303568))。

# **它的优点和局限性**

【彭福和张(2013 )已经提供了一个完整的优势和局限性列表，我将在下面总结其中的要点。

# **强项**

1.  以控制数据的长期时间趋势。它提供了一个更长时期的长期分析框架，可以更好地解释任何数据趋势。
2.  解释个体水平的偏倚并评估人群水平的结果变量。个体水平的数据可能会引入偏倚，但群体数据不会。**老实说，这既是福也是祸。我们将在接下来的部分详细阐述后一个方面。**
3.  评估干预的预期和非预期后果。我们可以很容易地扩大分析和纳入更多的结果变量与最低限度或没有适应。
4.  对个体亚群进行分层分析，并得出不同的因果关系。**这很关键。我们可以根据不同的标准将总人口分成不同的子群体，并检查每个子群体的不同表现。社会群体是不同的，将他们归类在一起可能会稀释或隐藏关键信息，因为积极和消极的影响混合在一起并相互抵消(参见[哈珀和布鲁克纳](https://www.sciencedirect.com/science/article/abs/pii/S1047279716303568)的例子)。**
5.  以提供清晰可辨的视觉效果。目视检查总是受欢迎的，应该认真对待([更多解释见我的另一篇文章)。](/research-note-what-are-natural-experiments-methods-approaches-and-applications-ad84e0365c75)

# **限制**

1.  **多轮数据录入**。干预前和干预后至少 8 个周期，以评估变化。因此，我们总共需要 16 个数据条目，这可能并不总是可行的。我认为彭福和张(2013 )对数据条目的数量持谨慎态度。仍然可以通过几轮数据输入来应用 ITS。只是因果力量可能没有多回合的强大。
2.  **时间滞后。程序需要一些未知的时间来达到预期的结果，这使得很难确定几个同时发生的事件的因果关系。假设美国交通部在两年时间内采取了三项政策来遏制高速公路超速。扮演上帝的角色，我们不知何故知道政策 A 需要 1 年的时间才会有效果，政策 B 需要 1.5 年，政策 c 需要 3 年。**
3.  **推理水平。**这是总体水平的数据，所以我们不能对每个个体进行推断。

# **应用**

它使用分段回归来检验干预的效果。它需要两个部分:干预前的部分和干预后的部分。每一段都有自己的斜率和截距，我们比较这两个分段回归模型得出的影响。

我们将这两个分段回归模型之间的方向(例如，从正到负)和/或程度(从大影响到小影响)的任何变化归因于干预变量。

实际上，这就是它如何克服只有一个案例的限制，仍然具有强大的推理能力。

这是一个使用模拟数据进行分析的例子。

```
# simulated data # data preparation
set.seed(1)
CaseID = rep(1:100,6)# intervention
Intervention = c(rep(0,300), rep(1,300))
Outcome_Variable = c(rnorm(300), abs(rnorm(300)*4))
mydata = cbind(CaseID, Intervention, Outcome_Variable)
mydata = as.data.frame(mydata)#construct a simple OLS model
model = lm(Outcome_Variable ~ Intervention, data = mydata)
summary(model)Call:
lm(formula = Outcome_Variable ~ Intervention, data = mydata)Residuals:
    Min      1Q  Median      3Q     Max 
-3.3050 -1.2315 -0.1734  0.8691 11.9185Coefficients:
             Estimate Std. Error t value Pr(>|t|)    
(Intercept)   0.03358    0.11021   0.305    0.761    
Intervention  3.28903    0.15586  21.103   <2e-16 ***
---
Signif. codes:  0 ‘***’ 0.001 ‘**’ 0.01 ‘*’ 0.05 ‘.’ 0.1 ‘ ’ 1Residual standard error: 1.909 on 598 degrees of freedom
Multiple R-squared:  0.4268, Adjusted R-squared:  0.4259 
F-statistic: 445.3 on 1 and 598 DF,  p-value: < 2.2e-16
```

可以看出，干预变量的回归结果具有统计学意义。

这是一个使用模拟数据的快速入门课程。事实上，它在因果推理方面可以做得更多，我将在后续文章中详细阐述。希望如此~~

*Medium 最近进化出了它的* [*作家伙伴计划*](https://blog.medium.com/evolving-the-partner-program-2613708f9f3c) *，支持像我这样的普通作家。如果你还不是订户，通过下面的链接注册，我会收到一部分会员费。*

[](https://leihua-ye.medium.com/membership) [## 阅读叶雷华博士研究员(以及其他成千上万的媒体作家)的每一个故事

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

leihua-ye.medium.com](https://leihua-ye.medium.com/membership) 

# 喜欢读这本书吗？

> 请在 [LinkedIn](https://www.linkedin.com/in/leihuaye/) 和 [Youtube](https://www.youtube.com/channel/UCBBu2nqs6iZPyNSgMjXUGPg) 找到我。
> 
> 还有，看看我其他关于人工智能和机器学习的帖子。