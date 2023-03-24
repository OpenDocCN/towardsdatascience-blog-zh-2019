# 生产中的数据科学

> 原文：<https://towardsdatascience.com/data-science-in-production-13764b11d68e?source=collection_archive---------8----------------------->

![](img/0b0473a43070f9fca9a2f319b1788110.png)

Source: [https://pixabay.com/photos/factory-industry-sugar-3713310/](https://pixabay.com/photos/factory-industry-sugar-3713310/)

## 用 Python 构建可扩展的模型管道

作为一名数据科学家，我最大的遗憾之一是我太久没有学习 Python 了。我一直认为其他语言在完成数据科学任务方面提供了同等的优势，但是现在我已经跨越到 Python，没有回头路了。我已经接受了一种语言，这种语言可以帮助数据科学家快速地将想法从概念变为原型并投入生产。最后一个术语， *production* ，可能是不断发展的数据科学学科中最重要的方面。

知道如何建立机器学习模型是有用的，但像 [AutoML](https://cloud.google.com/automl/) 这样的新工具正开始将数据科学家的工作商品化。现在，您可以构建健壮的模型，扩展到产品组合，而不是让数据科学家为单个产品构建定制模型。随着新角色的出现，如*应用科学家*，混合了人工智能工程和数据科学的能力，数据科学有了新的机会。

在我目前的职位上，我是数据产品开发的先锋，这些产品实现了数据科学的承诺，我们构建了[组合规模的](https://medium.com/zynga-engineering/portfolio-scale-machine-learning-at-zynga-bda8e29ee561)系统来提供预测信号。我们希望建立我们的产品团队可以使用的生产质量系统，我希望为下一批数据科学家提供学习材料。这就是为什么我决定写一本关于 Python 的书，因为我想提供一个文本，清楚地阐明从产品到应用数据科学的过渡需要学习哪些技能。我们的目标是为分析从业者和数据科学毕业生提供参考，通过实践经验提升他们的技能。为了交付数据产品，这本书将通过大量的例子从你的本地机器到云再到无服务器。

## 自助出版

虽然我有机会与大出版商合作，但我已经决定为这本书寻求自助出版。我完全承认，这本书的质量不会与出版商支持的质量相同，但我觉得这将是一个更加开放的创作体验。我的计划是用 [bookdown](https://bookdown.org/yihui/bookdown/) 来创作和设计这本书，用 [Leanpub](https://leanpub.com/ProductionDataScience/) 来发布并获得反馈。

从经济角度来看，写书对作家来说通常不是一笔大的支出。自助出版有可能获得更高的版税，但你可能会失去更多的读者。除了版税，我想自己出版的原因如下:

*   **时间线:**你想写就写，根据需要回复社区反馈。你决定哪些问题是最重要的。
*   **所有权:**对我来说，与传统出版商打交道最大的恐惧就是写完一本书的大部分内容，然后把它打包。我想写博客并获得反馈，拥有我写的所有东西。
*   **内容:**各大出版社希望书籍能涵盖相关主题，比如 GDPR，但这超出了我希望涵盖的范围。
*   也许对我来说最重要的一点是，我希望能够为我的书开展营销活动。然而，像亚马逊赞助这样的事情通常不会在合同中提及。
*   工具:我可以使用我认为最适合工作的工具来写我的书，并避免格式之间的翻译问题。

虽然与出版商合作是一种选择，但我决定单干，安德烈·布尔科夫是我追求这一选择的最大灵感之一。有许多开放的工具可以用来设计你的书，在出版前获得反馈，建立一个社区，并重复你的工作。

我出版这本书的计划是在 [Leanpub](https://leanpub.com/ProductionDataScience/) 上提供早期访问，在 [GitHub](https://github.com/bgweber/DS_Production) 上发布代码示例，并在媒体上分享摘录。

[](https://leanpub.com/ProductionDataScience/) [## 生产中的数据科学

### 从初创公司到数万亿美元的公司，数据科学在帮助组织最大化…

leanpub.com](https://leanpub.com/ProductionDataScience/) 

## 书籍内容

这本书的主题是采用简单的机器学习模型，并在多个云环境的不同配置中扩展它们。这本书假设读者已经具备 Python 和 pandas 的知识，以及一些使用 scikit-learn 等建模包的经验。这是一本关注广度而非深度的书，目标是让读者接触到许多不同的工具。虽然我将探索一些我以前在博客上写过的内容，但这将是所有新的写作。以下是我打算在文中涉及的内容:

1.  **简介:**这一章将[推动](/data-science-for-startups-r-python-2ca2cd149c5c)Python 的使用，并讨论应用数据科学的学科，展示全书中使用的数据集、模型和云环境，并概述[自动化特征工程](/automated-feature-engineering-for-predictive-modeling-d8c9fa4e478b)。
2.  **模型作为 web 端点:**本章展示了如何使用 Web 端点来消费数据，并使用 Flask 和 Gunicorn 库将机器学习模型作为端点。我们将从 scikit-learn 模型开始，并使用 [Keras](/deploying-keras-deep-learning-models-with-flask-5da4181436a2) 设置深度学习端点。
3.  **模型作为无服务器功能:**本章将建立在前一章的基础上，并展示如何使用 [AWS Lambda](/data-science-for-startups-model-services-2facf2dde81d) 和 GCP 云功能将模型端点设置为无服务器功能。
4.  **可重现模型的容器:**本章将展示如何使用容器来部署带有 [Docker](/data-science-for-startups-containers-d1d785bfe5b) 的模型。我们还将探索使用 ECS 和 Kubernetes 进行扩展，以及使用 Plotly Dash 构建 web 应用程序。
5.  **模型管道的工作流工具:**本章重点介绍使用 Airflow 和 Luigi 安排自动化工作流。我们将建立一个模型，从 BigQuery 中提取数据，应用模型，并保存结果。
6.  **批量建模的 PySpark:**本章将[向](/a-brief-introduction-to-pyspark-ff4284701873)读者介绍使用社区版 Databricks 的 PySpark。我们将构建一个批处理模型管道，从数据湖中提取数据，[生成特性](/scalable-python-code-with-pandas-udfs-a-data-science-application-dd515a628896)，应用模型，并将结果存储到一个非 SQL 数据库中。
7.  **批量建模的云数据流:**本章将介绍 GCP 云数据流的核心组件。我们将实现一个批处理模型管道，使用这个工具获得与前一章相同的结果。
8.  **模型工作流的消息系统:**本章将向读者介绍 Kafka 和 PubSub 在云环境中的消息流。阅读完这些材料后，读者将准备好使用 Python 来创建流数据管道。
9.  **使用 PySpark 和 Dataflow 的流式工作流:**本章将展示如何结合使用消息系统和第 6 章& 7 中介绍的批处理模型管道来创建低延迟的流式模型管道。
10.  **模型部署:**本章将讨论使用模型存储时将模型产品化的不同方法，并将提供 Jenkins 用于持续集成和 Chef 用于以编程方式支持模型服务的例子。
11.  **模型生命周期:**本章将讨论对已部署模型的监控和变更选项，并提供检查数据沿袭和模型漂移的例子。它还将涵盖模型剧本和事后分析的主题，以及发生故障时的通信处理。

阅读完这些材料后，读者应该对构建数据产品所需的许多工具有了实践经验，并对如何在云环境中构建可扩展的机器学习管道有了更好的理解。

## 结论

应用科学是 ML 工程和数据科学交叉的一个成长领域。这个领域的需求正在增长，因为投资组合规模的数据产品可以为公司提供巨大的价值。为了帮助满足这种需求，我正在写一本书，重点是用许多承担应用科学角色所需的工具来构建 Python 的实践经验。我正在使用 Leanpub 自行发布这篇文章，并启用社区反馈。

本·韦伯是 Zynga 杰出的数据科学家。我们正在[招聘](https://www.zynga.com/job-listing-category/data-analytics-user-research/)！