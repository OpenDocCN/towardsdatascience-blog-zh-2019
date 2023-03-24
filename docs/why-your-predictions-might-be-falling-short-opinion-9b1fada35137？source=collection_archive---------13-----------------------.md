# 从预测到决策

> 原文：<https://towardsdatascience.com/why-your-predictions-might-be-falling-short-opinion-9b1fada35137?source=collection_archive---------13----------------------->

## 为什么你的预测可能会失败——观点

![](img/1763cdbb20f571d4276317d5d3855656.png)

Photo by [Mika Baumeister](https://unsplash.com/photos/Wpnoqo2plFA?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/numbers?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

> “做预测和做决定之间有很多差距”苏珊·艾希[1]

# **相关性并不意味着因果关系**

这是统计测试中重复次数最多的短语之一。我认为这样做是有原因的，那就是这两个概念的抽象层次。这是混乱和复杂的，我们需要提醒自己在解释结果时的区别。然而，这种区别似乎在机器学习、大数据分析或数据科学领域并不明显，这可能并不太令人惊讶。对该领域的明显强调是通过更多数据和微调现有模型来提高预测准确性。根据我对在线课程和教程的经验，在之后的**步骤上投入的工作似乎更少，预测的准确性达到了预期的水平。**

在为我的应用程序研究相关文献时，我可以看到一些错误，从术语*预测、关联、原因*和*解释*之间相对无害的混淆，到一个虚假的声明，即一个变量(或一组特征或协变量)正在解释一个模型的行为，而不仅仅是表明它们之间的关联。这在图像识别、自然语言处理、情感分析等方面都不是问题。在这种情况下，从业者关心的是特性之间的映射。当同样的技术和哲学应用于决策或政策制定领域时，问题就来了。

# 为什么这是个问题

绘制相关性并不意味着我们理解了手头的问题。为了说明这一点，我们可以看一个简化的例子:假设有酒店价格和入住率的历史数据。酒店价格由软件设定，该软件根据入住率提高价格(入住率越高，价格越高)。现成的 ML 算法将识别价格和入住率之间的正相关关系。如果你想预测任何给定价格下的入住率，这是正确的方法。然而，如果你想知道价格上涨对入住率的影响，我们的预测模型可能会说价格上涨会卖出更多的房间。这是非常不可能的情况，需要一套不同的统计技术来回答这样的问题[1]。

# 我们能做什么

理解这种局限性并将其传达给决策者将会让我们走很长的路。确定并使用适当的方法来解决这个问题是我们想要的途径。人们一直有兴趣结合计量经济学和机器学习的经验来扩展我们对预测和因果推理的理解[2]。和往常一样，我们能做的最好的事情就是不断学习和了解相关领域的发展。这里是一个很好的起点。

[1] Athey，s .，[超越预测:利用大数据解决政策问题](http://science.sciencemag.org/content/355/6324/483.abstract?casa_token=Jb2YqGWt7ZQAAAAA:oL5JIvQSohfzIMHgx-xKsQhn556dqYiJoI3Q-r38Qno5E5FKjK7Qjzv2ysu3u-7S9K6do0_OXDOolA)，(2017)，《科学》第 355 卷，第 6324 期，第 483-485 页

[2]艾希，s .，[，](https://www.nber.org/chapters/c14009)，机器学习对经济学的影响，(2017)，即将出版的 NBER 著作[中的章节《人工智能的经济学:一个议程](https://www.nber.org/books/agra-1)，阿贾伊·k·阿格拉瓦尔，约书亚·甘斯和阿维·戈德法布

[3] Imbens，g .，& Rubin，d .，[统计、社会和生物医学科学的因果推理:导论](https://www.cambridge.org/core/books/causal-inference-for-statistics-social-and-biomedical-sciences/71126BE90C58F1A431FE9B2DD07938AB)，(2015)，剑桥大学出版社