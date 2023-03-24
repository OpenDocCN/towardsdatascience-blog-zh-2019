# 用于数据科学的 SQL

> 原文：<https://towardsdatascience.com/sql-for-data-science-f9a42ae66bbb?source=collection_archive---------1----------------------->

## SQL 是数据科学中最需要的技能之一。让我们使用 BigQuery 来了解它如何用于数据处理和机器学习。

![](img/0c0b322235d3aa66e7a475974be829fb.png)

(Source: [https://wallpapercave.com/database-wallpapers](https://wallpapercave.com/database-wallpapers))

# 介绍

SQL(结构化查询语言)是一种用于在关系数据库中查询和管理数据的编程语言。关系数据库由二维表的集合构成(例如数据集、Excel 电子表格)。然后，这些表中的每一个都由固定数量的列和任何可能数量的行组成。

举个例子，让我们考虑一下汽车制造商。每个汽车制造商可能有一个由许多不同的表组成的数据库(例如，每个不同的汽车模型都有一个表)。在这些表中的每一个中，将存储关于不同国家的每种汽车型号销售的不同指标。

与 Python 和 R 一起，SQL 现在被认为是数据科学中最需要的技能之一(图 1)。如今如此需要 SQL 的一些原因是:

*   [每天大约产生 2.5 万亿字节的数据](/big-data-analysis-spark-and-hadoop-a11ba591c057)。为了存储如此大量的数据，利用数据库是绝对必要的。
*   公司现在越来越重视数据的价值。例如，数据可以用于:分析和解决业务问题，预测市场趋势，了解客户需求。

使用 SQL 的主要优势之一是，当对数据执行操作时，可以直接访问数据(无需事先复制)。这可以大大加快工作流的执行速度。

![](img/12c22e732b7704b828168080f09b7a33.png)

Figure 1: Most Requested Data Science Skills, June 2019 [1]

存在许多不同的 SQL 数据库，如:SQLite、MySQL、Postgres、Oracle 和 Microsoft SQL Server。

在本文中，我将向您介绍如何使用 Google big query ka ggle integration 免费开始使用 SQL。从数据科学的角度来看，SQL 既可以用于预处理，也可以用于机器学习。本教程中使用的所有代码都将使用 Python 运行。

如 BigQuery 文档中所述:

> BigQuery 是一个企业数据仓库，通过使用 Google 基础设施的处理能力实现超快速的 SQL 查询来解决问题。只需将您的数据转移到 BigQuery 中，让我们来处理困难的工作。
> 
> — BigQuery 文档[2]

# MySQL 入门

当使用 Kaggle 内核(嵌入在 Kaggle 系统中的 Jupyter 笔记本的在线版本)时，可以选择启用 Google BigQuery(图 2)。事实上，Kaggle 为每个用户每月提供高达 5TB 的免费 BigQuery 服务(如果你用完了每月的限额，你必须等到下个月)。

为了使用 BigQuery ML，我们首先需要在我们的 Google 服务上创建一个免费的 Google 云平台账户和一个项目实例。你可以在这里找到[关于如何在几分钟内开始的指南](https://www.youtube.com/watch?v=_YYqfS7rLUo&utm_medium=email&utm_source=intercom&utm_campaign=sql-summer-camp)。

![](img/419d13f3acfe48633c12fb9863e8446b.png)

Figure 2: Enabling BigQuery on Kaggle Kernels

一旦在谷歌账户平台上创建了一个 BigQuery 项目，我们就会得到一个项目 ID。我们现在可以将 Kaggle 内核与 BigQuery 连接起来，只需运行下面几行代码。

在这个演示中，我将使用 [OpenAQ 数据集](https://www.kaggle.com/open-aq/openaq)(图 3)。该数据集包含有关世界各地空气质量数据的信息。

![](img/d541b12e54c68f51ddc013d09a2d0d8e.png)

Figure 3: OpenAQ Dataset

# 数据预处理

我现在将向您展示一些基本的 SQL 查询，它们可用于预处理我们的数据。

让我们先来看看一个国家有多少个不同的城市进行了空气质量测量。我们只需在 SQL 中选择 Country 列，并在 location 列中计算所有不同的位置。最后，我们将结果按国家分组，并按降序排列。

前十个结果如图 4 所示。

![](img/065f3658be9ac62213f2b6d045c502c2.png)

Figure 4: Number of measure stations in each country

之后，我们可以尝试检查该值的一些统计特征，并在小时列中取平均值，仅将 g/m 作为单位。通过这种方式，我们可以快速检查是否有任何异常。

“数值”栏代表污染物的最新测量值，而“小时平均值”栏代表该值的平均小时数。

![](img/72b2d9c169c1637a6beccd9589ca5a37.png)

Figure 5: Value and Averaged Over In Hours columns statistical summary

最后，为了总结我们的简要分析，我们可以计算每个不同国家臭氧气体的平均值，并使用 Matplotlib 创建一个条形图来总结我们的结果(图 6)。

![](img/c275dcdc47137f3a4ed2f897dac3d753.png)

Figure 6: Average value of Ozone in each different country

# 机器学习

此外，谷歌云还提供了另一项名为 BigQuery ML 的服务，专门用于直接使用 SQL 查询来执行机器学习任务。

> BigQuery ML 使用户能够使用标准的 SQL 查询在 BigQuery 中创建和执行机器学习模型。BigQuery ML 通过让 SQL 从业者使用现有的 SQL 工具和技能来构建模型，使机器学习民主化。BigQuery ML 通过消除移动数据的需要提高了开发速度。
> 
> — BigQuery ML 文档[3]

使用 BigQuery ML 可以带来几个好处，例如:我们不必在本地内存中读取我们的数据，我们不需要使用多种编程语言，我们的模型可以在训练后直接提供。

BigQuery ML 支持的一些机器学习模型有:

*   线性回归。
*   逻辑回归。
*   k-表示集群。
*   预训练张量流模型的导入。

首先，我们需要导入所有需要的依赖项。在这种情况下，我还决定将 BigQuery magic command 集成到我们的笔记本中，以使我们的代码更具可读性。

我们现在可以创建我们的机器学习模型。对于这个例子，我决定使用逻辑回归(只对前 800 个样本进行回归，以减少内存消耗)，根据纬度、经度和污染程度来预测国家名称。

一旦训练了我们的模型，我们就可以使用下面的命令来查看训练总结(图 7)。

![](img/55649732df7c3eaf708b2a2cda8ad511.png)

Figure 7: Logistic Regression Training Summary

最后，我们可以使用 BigQuery ML 来评估模型性能的准确性。EVALUETE 函数(图 8)。

![](img/dc005ce06fb227c72f69f5201dbc8227.png)

Figure 8: BigQuery ML model evaluation

# 结论

这是一个关于如何开始使用 SQL 解决数据科学问题的简单介绍，如果你有兴趣了解更多关于 SQL 的知识，我强烈建议你遵循 [Kaggle 介绍 SQL](https://www.kaggle.com/learn/intro-to-sql) 和 [SQLBolt](https://sqlbolt.com/) 免费课程。相反，如果你正在寻找更实际的例子，这些可以在这个[我的 GitHub 库](https://github.com/pierpaolo28/Artificial-Intelligence-Projects/tree/master/SQL%20for%20Data%20Science)中找到。

*希望您喜欢这篇文章，感谢您的阅读！*

# 联系人

如果你想了解我最新的文章和项目[，请在 Medium](https://medium.com/@pierpaoloippolito28?source=post_page---------------------------) 上关注我，并订阅我的[邮件列表](http://eepurl.com/gwO-Dr?source=post_page---------------------------)。以下是我的一些联系人详细信息:

*   [领英](https://uk.linkedin.com/in/pier-paolo-ippolito-202917146?source=post_page---------------------------)
*   [个人博客](https://pierpaolo28.github.io/blog/?source=post_page---------------------------)
*   [个人网站](https://pierpaolo28.github.io/?source=post_page---------------------------)
*   [中等轮廓](https://towardsdatascience.com/@pierpaoloippolito28?source=post_page---------------------------)
*   GitHub
*   [卡格尔](https://www.kaggle.com/pierpaolo28?source=post_page---------------------------)

# 文献学

[1]如何成为更有市场的数据科学家，KDnuggets。访问网址:[https://www . kdnugges . com/2019/08/market able-data-scientist . html](https://www.kdnuggets.com/2019/08/marketable-data-scientist.html)

[2]什么是 BigQuery？谷歌云。访问地点:【https://cloud.google.com/bigquery/what-is-bigquery 

[3]big query ML 简介。谷歌云。访问地点:【https://cloud.google.com/bigquery-ml/docs/bigqueryml-intro 