# 在 Azure Databricks 上开始使用 Apache Spark 和 TensorFlow

> 原文：<https://towardsdatascience.com/get-started-with-apache-spark-and-tensorflow-on-azure-databricks-163eb3fdb8f3?source=collection_archive---------15----------------------->

## 现在在 Apache Spark framework 上可以使用 TensorFlow，但是如何开始呢？它叫做张量框架

# TL；速度三角形定位法(dead reckoning)

这是一个关于如何在 Azure Databricks 上运行新的 Spark TensorFrame 库的分步教程。

![](img/6f5873f9f79b9c19e7ddb7e0bf878d0a.png)

大数据是一个庞大的主题，包含许多领域和专业知识。一路从 DevOps，数据工程师到数据科学家，AI，机器学习，算法开发者等等。我们都在与海量数据作斗争。当我们处理大量数据时，我们需要最好的头脑和工具。这就是**阿帕奇火花**和**张量流**的 ***神奇*** 组合发生的地方，我们称之为**张量框架**。

Apache Spark 接管了大数据世界，提供答案并支持数据工程师变得更加成功，而数据科学家则不得不设法绕过 Spark 提供的机器学习库 Spark MLlib 的限制。

但是现在，Apache Spark 用户可以使用 TensorFlow 了。这些工具的结合使数据科学家的工作更高效、更准确、更快速。从研究到开发再到生产，比以往任何时候都要快。

在我们开始之前，让我们先整理一下术语:

*   **Tensor Flow** 是 Google 创建的用于高性能数值计算的开源机器学习框架。它附带了对 AI 的强大支持:机器学习和深度学习。
*   **Azure Databricks** 是一个基于 Apache Spark 的分析平台，针对**微软 Azure** 云服务平台进行了优化。Azure Databricks 还充当软件即服务(SaaS) /大数据即服务(BDaaS)。
*   TensorFrames 是一个 Apache Spark 组件，它使我们能够在 Spark 集群上创建自己的可扩展 TensorFlow 学习算法。

# -1-工作区:

首先，我们需要创建工作空间，我们正在使用 Databricks 工作空间，这里有一个[教程](https://docs.microsoft.com/azure/azure-databricks/quickstart-create-databricks-workspace-portal?WT.mc_id=online-medium-adpolak)用于创建它。

# -2-集群:

有了工作空间之后，我们需要创建集群本身。让我们使用[本](https://docs.microsoft.com/azure/azure-databricks/quickstart-create-databricks-workspace-portal?WT.mc_id=online-medium-adpolak)教程创建我们的 spark 集群，确保您的集群中有以下配置:

*   Apache Spark 的工作版本(2.4 或更高版本)
*   Java 8+
*   (可选)python 2.7+/3.6+如果要使用 python 接口。
*   (可选)python TensorFlow 包(如果要使用 python 接口)。关于如何获得 TensorFlow 的最新版本，请参见[官方说明](https://www.tensorflow.org/install/)。
*   (可选)如果要使用 python 接口，pandas >= 0.19.1

配置:

![](img/e2fada2fefbe9c4fe67ffd56cdbbc2ed.png)

使用 Databricks 运行时版本或更高版本:

![](img/c5189ca59a6473f491b90bc5b795723b.png)

按 start 启动集群

![](img/9326e9bbc29c9b624f32b4728a36933e.png)

# -3-导入库:

在 **Azure Databricks** 下，进入**常见任务**，点击**导入库**:

![](img/50a715c824f5766344d2965a152075fc.png)

TensorFrame 可以在 [maven 资源库](https://mvnrepository.com/artifact/databricks/tensorframes/0.2.9-s_2.11)上找到，所以选择 maven 标签。在坐标下，插入您选择的库，现在，它将是:

```
databricks:tensorframes:0.6.0-s_2.11
```

![](img/64e18c70055b006189b3b0ab8e018a6b.png)

点击**创建**按钮。

![](img/b87af81d5f7d806fe012178d8324c5e2.png)

点击**安装**。

你会看到这个:

![](img/7d28e69fc036ec0d2d8d807a3cb6fb51.png)

嘣。您的 Databricks 集群上有 TensorFrame。

# -4-笔记本:

我们使用笔记本作为我们的代码笔记本，在那里我们可以编写代码并直接在我们的 Spark 集群上运行它。

现在我们有了一个正在运行的集群，让我们运行一个笔记本:

点击**新笔记本**并选择您喜欢的编程语言(这里我们选择了 Scala)

![](img/f16dd119b309956348c035814b01dd12.png)

这是 scala 代码在带有 TensorFrames 的笔记本门户上的外观:

![](img/820f2f59fa640407a42e2070accc3962.png)

代码示例来自 D [atabricks git 仓库](https://github.com/databricks/tensorframes/blob/master/README.md)。

您也可以在这里查看:

# -结尾-

现在一切都已启动并运行，在 Apache Spark 集群中创建自己的使用 TensorFrame 的触发/调度作业。

还有…

既然你已经走到这一步了，请点击鼓掌按钮👏和[在 Medium](https://medium.com/@adipolak) 和 [Twitter](https://twitter.com/adipolak) 上关注我，获取更多关于 Scala、Kotlin、大数据、干净代码和软件工程师挑战的帖子。干杯！