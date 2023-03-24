# 给有抱负的数据科学家的六条建议

> 原文：<https://towardsdatascience.com/six-recommendations-for-aspiring-data-scientists-93d12aeb9b9?source=collection_archive---------2----------------------->

![](img/10c78ee0845afa4b0141e3e25ea87e8c.png)

Source: [https://www.maxpixel.net/Art-Colourful-Gears-Creativity-Cogs-Colorful-1866468](https://www.maxpixel.net/Art-Colourful-Gears-Creativity-Cogs-Colorful-1866468)

## 找到工作前积累经验

数据科学是一个需求巨大的领域，部分原因是它似乎需要作为数据科学家的经验才能被聘为数据科学家。但我合作过的许多最优秀的数据科学家都有从人文科学到神经科学的不同背景，要脱颖而出需要经验。作为一名即将进入数据科学职业生涯的新毕业生或分析专业人士，构建一个工作组合来展示该领域的专业知识可能是一项挑战。我在数据科学职位的招聘过程中经历过两种情况，我想列举一些能够帮助我找到数据科学家工作的关键经历:

1.  亲身体验云计算
2.  创建新的数据集
3.  把东西粘在一起
4.  支持一项服务
5.  创造惊人的视觉效果
6.  写白皮书

我将详细阐述这些主题，但数据科学的关键主题是能够构建为公司增加价值的数据产品。能够构建这些端到端数据产品的数据科学家是一笔宝贵的财富，在从事数据科学职业时展示这些技能是有用的。

## 亲身体验云计算

许多公司都在寻找过去在云计算环境中有经验的数据科学家，因为这些平台提供了支持数据工作流和预测模型扩展到海量数据的工具。你还可能在日常工作中使用云平台，比如亚马逊网络服务(AWS)或谷歌云平台(GCP)。

好消息是，这些平台中的许多都提供了免费的层来熟悉平台。例如，AWS 有免费的 EC2 实例，并免费使用 Lambda 等服务来满足少量请求，GCP 提供 300 美元的免费积分来试用该平台的大部分内容，Databricks 提供了一个社区版，您可以使用它来接触该平台。有了这些免费选项，你将无法处理大量数据集，但你可以在这些平台上积累经验。

我的一个建议是在这些平台上尝试不同的特性，看看是否可以使用一些工具来训练和部署模型。例如，在我的模型服务文章中，我利用了我已经熟悉的工具 SKLearn，并研究了如何将模型包装成 Lambda 函数。

[](/data-science-for-startups-model-services-2facf2dde81d) [## 创业数据科学:模型服务

### 我的创业数据科学系列的第二部分主要关注 Python。

towardsdatascience.com](/data-science-for-startups-model-services-2facf2dde81d) 

## 创建新的数据集

在学术课程和数据科学竞赛中，您通常会得到一个干净的数据集，其中项目的重点是探索性数据分析或建模。然而，对于大多数现实世界的项目，您需要执行一些数据整理，以便将原始数据集整理成对分析或建模任务更有用的转换数据集。通常，数据管理需要收集额外的数据集来转换数据。例如，为了更好地理解美国富裕家庭的资产配置，我过去曾在美联储数据公司工作过。

[](https://medium.freecodecamp.org/clustering-the-top-1-asset-analysis-in-r-6c529b382b42) [## 聚类前 1%:R 中的资产分析

### 美国最近通过的税改法案引发了许多关于该国财富分配的问题…

medium.freecodecamp.org](https://medium.freecodecamp.org/clustering-the-top-1-asset-analysis-in-r-6c529b382b42) 

这是一个有趣的项目，因为我使用第三方数据来衡量第一方数据的准确性。我的第二个建议是更进一步，建立一个数据集。这可以包括抓取网站、从端点采样数据(例如 [steamspy](https://steamspy.com/) )或将不同的数据源聚合到新的数据集。例如，我在研究生学习期间创建了一个自定义的星际争霸回放数据集，它展示了我在一个新的数据集上执行数据管理的能力。

[](/reproducible-research-starcraft-mining-ea140d6789b9) [## 可再生研究:星际采矿

### 2009 年，我发表了一篇关于预测《星际争霸:育雏战争》建造顺序的论文，使用了不同的分类…

towardsdatascience.com](/reproducible-research-starcraft-mining-ea140d6789b9) 

## 把东西粘在一起

我喜欢看到数据科学家展示的技能之一是让不同的组件或系统协同工作以完成任务的能力。在数据科学的角色中，可能没有一个清晰的模型产品化的路径，您可能需要构建一些独特的东西来启动和运行系统。理想情况下，数据科学团队将获得工程支持来启动和运行系统，但原型制作是数据科学家快速行动的一项重要技能。

我的建议是尝试将不同的系统或组件集成到数据科学工作流中。这可能涉及到动手使用工具，如[气流](https://airflow.apache.org/)，以原型数据管道。它可以包括在不同系统之间建立一座桥梁，比如我开始用 Java 连接星际争霸育雏战争 API 库的 [JNI-BWAPI](https://github.com/JNIBWAPI/JNIBWAPI) 项目。或者它可以涉及在一个平台内将不同的组件粘合在一起，例如使用 GCP 数据流从 BigQuery 中提取数据，应用预测模型，并将结果存储到云数据存储中。

[](/data-science-for-startups-model-production-b14a29b2f920) [## 创业公司的数据科学:模型生产

### 我正在进行的关于在创业公司建立数据科学学科系列的第七部分。您可以找到所有……

towardsdatascience.com](/data-science-for-startups-model-production-b14a29b2f920) 

## 支持一项服务

作为一名数据科学家，您经常需要提供公司内其他团队可以使用的服务。例如，这可能是一个提供深度学习模型结果的 Flask 应用程序。能够原型化服务意味着其他团队将能够更快地使用您的数据产品。

[](/deploying-keras-deep-learning-models-with-flask-5da4181436a2) [## 使用 Flask 部署 Keras 深度学习模型

### 这篇文章演示了如何使用 Keras 构建的深度学习模型来设置端点以服务于预测。它…

towardsdatascience.com](/deploying-keras-deep-learning-models-with-flask-5da4181436a2) 

我的建议是获得使用诸如 [Flask](http://flask.pocoo.org/) 或 [Gunicorn](https://gunicorn.org/) 等工具的实践经验，以便设置 web 端点，以及 [Dash](https://dash.plot.ly/) 以便用 Python 创建交互式 web 应用程序。尝试在一个 [Docker](https://www.docker.com/) 实例中设置这些服务之一也是一个有用的实践。

## 创造惊人的视觉效果

虽然伟大的作品应该是独立的，但在解释为什么一个分析或模型是重要的之前，通常有必要首先引起你的观众的注意。我在这里的建议是学习各种可视化工具，以创建引人注目的突出可视化。

[](/visualizing-professional-starcraft-with-r-598b5e7a82ac) [## 用 R 可视化职业星际争霸

### 自从我开始为《星际争霸:育雏战争》进行数据挖掘专业回放以来，已经过去了将近十年。最近之后…

towardsdatascience.com](/visualizing-professional-starcraft-with-r-598b5e7a82ac) 

创建可视化也是建立作品组合的一种有用方式。下面的博文展示了我作为数据科学家 10 多年来探索的不同工具和数据集的样本。

[](/10-years-of-data-science-visualizations-af1dd8e443a7) [## 10 年的数据科学可视化

### 我在数据科学领域的职业生涯始于十年前，当时我在加州大学圣克鲁斯分校(UC Santa Cruz)上了第一门机器学习课程。自从…

towardsdatascience.com](/10-years-of-data-science-visualizations-af1dd8e443a7) 

## 写白皮书

我最近一直提倡的数据科学技能之一是以白皮书的形式解释项目的能力，该白皮书提供了执行摘要，讨论了如何使用该工作，提供了有关方法和结果的详细信息。我们的目标是让你的研究能够被广泛的受众所理解，并且能够自我解释，以便其他数据科学家可以在此基础上进行研究。

写博客和其他形式的写作是获得提高书面交流经验的好方法。我在这里的建议是，尝试为广大受众撰写数据科学文章，以便获得在不同细节层次传达想法的经验。

[](/data-science-for-startups-blog-book-bf53f86ca4d5) [## 创业数据科学:博客->书籍

### 数据科学家写书有很多令人信服的理由。我想更好地理解新工具，而且…

towardsdatascience.com](/data-science-for-startups-blog-book-bf53f86ca4d5) 

## 结论

数据科学需要大量工具的实践经验。幸运的是，这些工具越来越容易获得，构建数据科学组合也变得越来越容易。

本·韦伯是 Zynga 公司的首席数据科学家，也是 T2 恶作剧公司的顾问。