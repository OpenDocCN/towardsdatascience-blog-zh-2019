# 问答:数据科学中的常见问题

> 原文：<https://towardsdatascience.com/q-a-common-questions-in-data-science-7cd7f9d82a8d?source=collection_archive---------19----------------------->

## [问我们任何事情](https://towardsdatascience.com/tagged/ask-us-anything)

## 回答在“走向数据科学”团队中最常见的一些问题。

![](img/3e78f250116ead21b2d2f556dd509c49.png)

Photo by [Ian Schneider](https://unsplash.com/@goian?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 介绍

最近，走向数据科学(TDS)推出了一个名为[“向我们提问”](/ask-us-anything-70617fc7eccf)的新专栏，让 TDS 的读者有机会向团队提出任何有关数据科学的问题。在这篇文章中，我将回答一些到目前为止我们收到的最常见的问题。

人们对数据科学如此感兴趣的一个主要原因是它的各种应用。数据一直是人类历史上非常重要的组成部分，根据过去的经验和提供的信息做出明智的决策对个人或组织都至关重要。

由于这一主题的跨学科性(图 1)，数据科学可能既是一个真正令人兴奋的工作领域，也是一个令人望而生畏的起点。这就是为什么在这个栖息地提问可以自然而然的到来。

![](img/926f8d68b138c9dc4356f14fd82b011e.png)

Figure 1: Data Science in summary. [1]

# 读者提问

## 数据科学如何入门？

开始一个新的领域总是出乎意料地让人既兴奋又害怕。幸运的是，在过去的几年中，为了增加公众对数据科学的兴趣，已经创建了许多免费资源。一些例子是:

*   在线课程:[谷歌 AI 教育](https://ai.google/education/)、 [Coursera:吴恩达机器学习](https://www.coursera.org/learn/machine-learning)等...
*   书籍:[Christopher Bishop 著的《模式识别与机器学习》](http://users.isr.ist.utl.pt/~wurmd/Livros/school/Bishop%20-%20Pattern%20Recognition%20And%20Machine%20Learning%20-%20Springer%20%202006.pdf)，[Gareth James 等人著的《统计学习导论》](http://faculty.marshall.usc.edu/gareth-james/ISL/)等…
*   在线出版物:[走向数据科学！](https://towardsdatascience.com/)、[提取](https://distill.pub/)、[无自由预感](http://blog.kaggle.com/)等…
*   YouTube 频道: [3Blue1Brown](https://www.youtube.com/channel/UCYO_jab_esuFRV4b17AJtAw) ， [Arxiv Insights](https://www.youtube.com/channel/UCNIkB2IeJ-6AmZv7bQ1oBYg) ，[走向数据科学](https://www.youtube.com/channel/UCuHZ1UYfHRqk3-5N5oc97Kw)，[两分钟论文](https://www.youtube.com/channel/UCbfYPyITQ-7l4upoX8nvctg)等…
*   播客: [DeepMind 播客](https://deepmind.com/blog/article/welcome-to-the-deepmind-podcast)， [TDS 播客](/our-podcast-c5c1129bc5cf)等…

获得必要的数据科学背景知识后，对于初学者来说，开始实际项目非常重要。如果你正在寻找一些灵感，Kaggle 可能是一个很好的起点。

## 我如何写一篇伟大的数据科学文章？

一旦你获得了一些数据科学的知识，尝试展示一些你研究过的项目和主题是一个很好的主意。最简单的方法之一是在你自己的个人网站或在线出版物上写文章(以获得更广泛的受众)。如果你对撰写《走向数据科学》感兴趣，你可以在[这篇我以前的文章中找到更多信息和技巧。](/writing-for-towards-data-science-more-than-a-community-6c9f0452b280)撰写数据科学相关文章的一些主要指导原则是:

*   使用 GitHub Gists 来共享您的代码。
*   利用免费的开放数据源。这样，读者测试演练示例将变得更加容易。
*   利用[交互式数据可视化](/interactive-data-visualization-167ae26016e8)工具提供更多见解。
*   遵循[策展人指南！](https://help.medium.com/hc/en-us/articles/360006362473-Medium-s-Curation-Guidelines-everything-writers-need-to-know)

## 确定数据集中哪些变量对响应变量影响最大的最佳选择是什么？

了解哪些特性会在数据分析中带来更多信息是一项至关重要的任务(尤其是在处理大量数据时！).减少数据集中要素数量的一些主要好处是:

*   精度提高。
*   过度拟合风险降低。
*   在训练中加速。
*   改进的数据可视化。
*   增加我们模型的可解释性。

为了减少数据集中的要素数量，有两种主要方法:特征选择和特征提取。

*   **特征选择**:在特征选择中，我们利用不同的技术，如过滤器/包装器/嵌入式方法，来选择从数据集中保留哪些特征，丢弃哪些特征。
*   **特征提取**:在特征提取中，我们通过从现有特征中创建新的特征(然后丢弃原始特征)来减少数据集中的特征数量。这可以通过使用诸如主成分分析(PCA)、局部线性嵌入(LLE)等技术来实现

如果你有兴趣了解更多关于特征选择和特征提取技术的信息，更多信息请点击[这里](/feature-selection-techniques-1bfab5fe0784)和[这里](/feature-extraction-techniques-d619b56e31be)。

## 你如何在你的模型中考虑专家知识？

当试图决定将哪些特征输入用于预测任务的机器学习模型时，对正在分析的数据具有良好的背景知识会有很大的帮助。

例如，假设我们有一个数据集，其中包含不同出租车行程的出发、到达时间和行驶距离，我们希望预测每次不同行程的出租车费用。将出发和到达时间直接输入机器学习模型可能不是最好的主意，因为我们必须让 ML 模型来计算这两个特征之间的关系对预测出租车费用是有用的。使用专业知识，我们可以首先尝试计算到达和离开之间的时间差(通过简单地减去这两列)，然后将这一新列(与行驶距离一起)输入到我们的模型中。这样，我们的模型更有可能表现得更好。为机器学习分析准备原始特征的技术通常被称为特征工程。

如果您有兴趣了解更多关于不同特征工程技术的信息，更多信息请点击[这里](/feature-engineering-techniques-9a57e4901545)。

## 如何入门机器人流程自动化(RPA)？

机器人流程自动化(RPA)是一项正在开发的用于自动化手动任务的技术。RPA 与传统编程的区别在于其图形用户界面(GUI)。RPA 自动化是通过分解用户执行的任务并重复它们来执行的。这项技术可以使基于图形的过程的自动化编程变得更加容易。平均而言，机器人执行同等过程的速度比人快三倍，而且它们能够全年 24 小时不间断地工作。RPA 公司的一些常见例子有 Automation Anywhere、UIPath 和 blueprism [2]。

如果您打算开始使用 RPA，可以在 Windows 上免费下载 ui path Community Edition[。使用 UIPath，可以实现复杂的工作流，创建序列和基于流程图的架构。然后，每个序列/流程图可以由许多子活动组成，例如记录机器人要执行的一组动作或筛选和数据采集指令。此外，UIPath 还支持错误处理机制，以防决策情况或不同流程之间出现意外延迟。](https://www.uipath.com/developers/community-edition-download)

如果你想了解更多关于 RPA 的信息，你可以看看我的 [GitHub 库](https://github.com/pierpaolo28/UIPath)里面有一些例子或者这个[我以前的文章](/robotic-process-automation-rpa-using-uipath-7b4645aeea5a)。

# 联系人

如果你想了解我最新的文章和项目[，请通过媒体](https://medium.com/@pierpaoloippolito28?source=post_page---------------------------)关注我，并订阅我的[邮件列表](http://eepurl.com/gwO-Dr?source=post_page---------------------------)。以下是我的一些联系人详细信息:

*   [领英](https://uk.linkedin.com/in/pier-paolo-ippolito-202917146?source=post_page---------------------------)
*   [个人博客](https://pierpaolo28.github.io/blog/?source=post_page---------------------------)
*   [个人网站](https://pierpaolo28.github.io/?source=post_page---------------------------)
*   [中等轮廓](https://towardsdatascience.com/@pierpaoloippolito28?source=post_page---------------------------)
*   [GitHub](https://github.com/pierpaolo28?source=post_page---------------------------)
*   [卡格尔](https://www.kaggle.com/pierpaolo28?source=post_page---------------------------)

# 文献学

[1]您需要了解的数据科学概念！迈克尔·巴伯。访问:[https://towardsdatascience . com/introduction-to-statistics-e9d 72d 818745](/introduction-to-statistics-e9d72d818745)

[2] Analytics Insight，[Kamalika Some](https://www.analyticsinsight.net/author/kamalika/)=[https://www . Analytics Insight . net/top-10-robotic-process-automation-companies-of-2018/](https://www.analyticsinsight.net/top-10-robotic-process-automation-companies-of-2018/)