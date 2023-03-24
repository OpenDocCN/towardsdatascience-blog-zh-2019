# omega|ml:以简单的方式部署数据管道和机器学习模型

> 原文：<https://towardsdatascience.com/omega-ml-deploying-data-machine-learning-pipelines-the-easy-way-a3d281569666?source=collection_archive---------22----------------------->

## 部署数据管道和机器学习模型可能需要几周到几个月的时间，涉及许多学科的工程师。我们的开源数据科学平台 omega|ml 是用 Python 编写的，利用了 MongoDB，在几秒钟内完成了这项任务——只需要一行代码。

![](img/af6c12788826b72822beb393d1ce9669.png)

Photo by [Ashkan Forouzani](https://unsplash.com/@ashkfor121) on [Unsplash](https://unsplash.com/)

**部署数据&机器学习管道很难——不应该是**

当谈到部署数据和机器学习管道时，有许多选项可供选择——其中大多数都相当复杂。截至 2019 年春天，最佳实践包括构建自己的应用服务器，以某种方式使用训练有素的机器学习模型的序列化版本，或使用部分自动化的工具构建 docker 映像，以及使用拥有一切的全面复杂的商业数据科学平台。

无论采用何种方法部署机器学习模型，这通常都不是数据科学家的核心技能。要做好这件事，需要对更接近软件工程和分布式系统，而不是机器学习和统计学的学科充满热情。

> 无论采用何种方法部署机器学习模型，这通常都不是数据科学家的核心技能

此外，当基础设施现成可用时，为什么还要构建新的基础设施呢？

**omega|ml:你只需要一行代码**

进入开源包 [omega|ml](https://github.com/omegaml/omegaml) ，这是一个生产就绪的数据科学框架，可以从笔记本电脑扩展到云。使用 omega|ml 部署机器学习模型就像这个 Python 一行程序一样简单，可以直接从您的 Jupyter 笔记本或任何其他 Python 程序中获得:

```
om.models.put(model, ‘mymodel’) # model is e.g. a scikit-learn model
```

无需更多麻烦，该模型可立即从 omega|ml 的 REST API 中获得，并可用于任何客户端以任何语言编写的预测:

```
GET [http://host/v1/model/mymodel/predict/](http://host/v1/model/mymodel/predict/`) + <json body>
```

这里的 *< json body >* 是一个包含列名和相应数据值的字典。一个*模型*是任何 scikit-learn 模型，其他框架如 Keras、Tensorflow、Pytorch 都可以通过使用扩展 API 轻松添加。

**还有更多:数据接收、数据管理、调度&报告发布**

模型部署不是从模型开始到结束，还需要数据、模型再训练和监控。为此，omega|ml 涵盖了完整的数据科学管道，从数据接收、数据管理、作业调度，到模型构建、选择和验证，再到在线报告和应用发布。所有这些都来自于一个易于使用的 API，它只需要一行代码就可以完成大多数任务，直接来自于 Jupyter Notebook。

> 模型部署不是从模型开始到结束，还需要数据、模型再训练和监控。omega|ml 涵盖了完整的数据科学管道—大多数任务只需要一行代码

例如，从 Python 获取新数据:

```
om.datasets.put(data, ‘mydata’) 
# data is a container object, e.g. list, dict or a pandas DataFrame
```

类似地，可以使用 REST API 获取数据:

```
PUT [http://host/v1/dataset/mydata](http://host/v1/dataset/mydata') + <json body+
```

也可以从 REST API 或 Python 客户端查询数据:

```
# Python 
om.datasets.get(‘mydata’, column=value)# REST API
GET [http://host/v1/dataset/mydata?column=value'](http://host/v1/dataset/mydata?column=value')
```

类似熊猫的数据帧:大于内存的数据帧

除了部署模型和数据管道之外，omega|ml 还为任何规模的列数据集提供了一个类似熊猫的 API，称为 MDataFrame ("M "一般代表*大规模**，*，具体代表 *MongoDB* )。虽然 MDataFrame 还不是 Pandas 的替代产品，但它提供了 Pandas API 的许多最有用的部分，如索引、过滤、合并和数据管理的行功能，以及描述性统计、相关性和协方差。

借助 MDataFrame，甚至比内存更大的数据集也可以轻松分析，使用 omega|ml 的集成数据和计算集群在核外训练和执行模型。默认情况下，利用 MongoDB & Python 的优秀分布式任务框架 Celery，它与 scikit-learn 的 joblib 很好地集成在一起，它还可以随时利用 Dask 分布式或 Apache Spark 集群
(托管版和商业版)。

> 借助 MDataFrame，使用 omega|ml 的集成数据和计算集群，甚至可以轻松分析大于内存的数据集、训练模型并在核外执行

请注意，存储在 omega|ml 中的数据集只受其存储层可用的物理磁盘空间的限制，而不受内存大小的限制。与 Spark 不同，omega|ml 的 MDataFrame 在处理开始之前不会产生预加载延迟，因为所有繁重的工作都由 MongoDB 完成。默认情况下，新数据是持久的，其他数据科学家可以随时使用。此外，多个用户可以同时利用集群，消耗许多不同的数据集，每个数据集或其组合都大于集群中可用的物理内存。

使用 MDataFrame，例如汇总数据，

```
mdf = om.datasets.getl('mydata')
mdf.groupby(['division', 'region']).sales.mean()
```

通过筛选、索引或列对数据进行子集化，

```
# by filter
query = mdf['column'] == value
mdf.loc[query]# by index
mdf.loc[‘value’]
mdf.loc[['value1', 'value2', ...]# by column
mdf['column']
mdf['column1', 'column2']
```

要执行由 MongoDB 执行的延迟过滤操作和计算，

```
# query on sales
mdf.query(sales__gte=5000).value# apply row-wise calculation
mdf.apply(lambda v: v * 2).value 
```

合并数据帧，

```
mdf1.merge(mdf2)
```

MDataFrames 也可以很容易地与存储在 omega|ml 中的 scikit-learn 模型一起使用，例如在使用计算集群时。

```
# mydata is a MDataFrame previously stored using om.datasets.put
om.runtime.model('mymodel').fit('mydata[^Y]', 'mydata[Y]')
```

omega|ml 的 Python 和 REST API 的完整文档可以在[https://omegaml.github.io/omegaml/index.html](https://omegaml.github.io/omegaml/index.html)获得。

**特性&附加组件**

omega|ml 包括一系列数据科学家工作区通常缺少的现成功能:

*   使用类似熊猫的 API 的核外数据集
*   直接从 Python 和通过 REST API 进行异步和预定模型训练、优化和验证
*   集成的、可扩展的数据集群(基于 MongoDB)
*   利用 Celery、Dask Distributed 或 Spark 的集成计算集群。

> 虽然 omega|ml 通过易于使用的 API 在笔记本电脑上运行良好，但其架构是为云可伸缩性和可扩展性而构建的，集成了 scikit-learn 和 Spark MLLib

许多附加组件使 omega|ml 在团队和组织中的协作变得可行:

*   直接从 Jupyter 笔记本电脑(附件)中向商业用户分发 Plotly Dash & Jupyter 笔记本电脑
*   类似于 Spark Streaming 的纯 Python 小批量框架(附加组件，没有 Scala/JVM/Hadoop 设置的复杂性)
*   多用户角色和安全性(托管版和企业版中提供的附加组件)

**一个可扩展的&可伸缩的架构**

虽然 omega|ml 通过易于使用的 API 在笔记本电脑上运行良好，但其架构是为云的可伸缩性和可扩展性而构建的。它的核心 API 集成了 scikit-learn 和 Spark MLLib，使开发人员能够只用几行代码为任何机器学习框架(如 Keras、Tensorflow 或 Pytorch)构建扩展，同时保持 Python 和 REST API 的稳定 API。对于外部数据源，如亚马逊 S3 或其他对象存储，以及数据库，如 MySQL 或 Oracle，也是如此，它们可以很容易地作为扩展添加。

![](img/dd287819c6c1039bad45a8168a34dfcb.png)

omega|ml architecture

利用 MongoDB 作为其存储层，omega|ml 可以水平扩展到任何大小的数据集，分布到任何数量的存储/计算节点，同时它没有内存需求，也没有所有内存堆栈(例如 Spark 或 Dask)的数据加载延迟，有效地结合了 MongoDB 的高性能混合架构，用于内存处理和分布式存储。

由于其集成的纯 Python RabbitMQ/Celery 计算集群，它提供了 Python 原生的无服务器功能，同时可以利用任何计算集群，如 Apache Spark 或 Dask Distributed。

**欧米茄|ml 入门**

从 docker 直接运行 omega|ml 开始(这是开源社区版):

```
$ wget https://raw.githubusercontent.com/omegaml/omegaml/master/docker-compose.yml
$ docker-compose up -d
```

接下来在[打开浏览器 http://localhost:8899](http://localhost:8899/) 打开 Jupyter 笔记本。您创建的任何笔记本都将自动存储在 omega|ml 数据库中，从而便于与同事合作。REST API 可从 [http://localhost:5000](http://localhost:5000/) 获得。

您还可以使用 omega|ml 作为现有 Python 发行版的附加包(例如来自 Anaconda)。在这种情况下，您还必须运行 MongoDB 和 RabbitMQ。

```
pip install omegaml
```

> 利用 MongoDB 的高性能聚合，omega|ml 可以水平扩展到任何规模的数据集。然而，它没有像 Apache Spark 或 Dask 那样的全内存堆栈的内存需求和数据加载延迟

**了解更多信息**

omega|ml (Apache License)建立在广泛使用的 Python 包之上，如 scikit-learn、Pandas 和 PyMongo。对 TensorFlow 等其他机器学习框架的扩展很容易通过一个定义良好的 API 来实现。

omega|ml 作为 docker-compose 的现成部署 docker 映像和软件即服务在 [https://omegaml.io](https://omegaml.io) (目前处于测试阶段)提供。在私有或公共云上部署到 Kubernetes 的本地版本可以通过商业许可获得。有一个[入门指南](https://drive.google.com/file/d/1Ao6o1QnzH_7EmMQLMRs9ehfZHWjABf-v/view?usp=sharing)和一个[指南笔记本](https://gist.github.com/omegaml/14e08ea74d413834ced695a98839d6df)来帮助你入门。

**关于作者**

Patrick Senti 是一名自由职业的高级数据科学家和 Fullstack 软件工程师，拥有近 30 年的专业经验。他最初建立了 omega|ml 的核心，作为他在 2014 年推出的智能城市和下一代移动创业公司的内部数据科学平台，其中的挑战是在分布式数据科学家团队之间协作处理大型核外数据集，并部署数百个机器学习模型，以便在云中运行并集成到智能手机旅行应用程序中。