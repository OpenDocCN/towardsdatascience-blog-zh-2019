# 从熊猫到有考拉的派斯帕克

> 原文：<https://towardsdatascience.com/from-pandas-to-pyspark-with-koalas-e40f293be7c8?source=collection_archive---------7----------------------->

![](img/26455d3ec084a4747b02a79c6605e8a5.png)

Photo by [Ozgu Ozden](https://unsplash.com/@ozgut?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

对于那些熟悉[熊猫](https://pandas.pydata.org)数据帧的人来说，切换到 PySpark 可能会相当混乱。API 是不一样的，当切换到分布式特性时，由于该特性所施加的限制，有些事情的处理方式会非常不同。

我最近在一个非常有趣的关于 Apache Spark 3.0、Delta Lake 和考拉的 Databricks 演示中偶然发现了[考拉](https://github.com/databricks/koalas)，我想探索一下会很不错。

> 考拉项目通过在 Apache Spark 上实现 pandas DataFrame API，使数据科学家在与大数据交互时更有效率。
> 
> pandas 是 Python 中事实上的标准(单节点)DataFrame 实现，而 Spark 是大数据处理的事实上的标准。使用此软件包，您可以:
> 
> -如果您已经熟悉熊猫，使用 Spark 可以立即提高工作效率，无需学习曲线。
> 
> -拥有一个既适用于 pandas(测试，较小的数据集)又适用于 Spark(分布式数据集)的单一代码库。

来源:[https://koalas.readthedocs.io/en/latest/index.html](https://koalas.readthedocs.io/en/latest/index.html)

# **如何入门**

考拉支持≥ **Python 3.5** ，从我从文档中看到的来看， **PySpark 2.4.x.** 依赖项包括 pandas ≥ 0.23.0，pyarrow ≥ 0.10 用于使用柱状内存格式以获得更好的矢量操作性能，matplotlib ≥ 3.0.0 用于绘图。

## 装置

下面列出了安装考拉的不同方法:

 [## 安装-考拉 0.20.0 文档

### 正式 Python 3.5 及以上。首先你需要安装 Conda。此后，我们应该创造一个新的环境

考拉. readthedocs.io](https://koalas.readthedocs.io/en/latest/getting_started/install.html) 

但是让我们从简单的开始:

`pip install koalas`和`pip install pyspark`

请记住上面提到的依赖性。

## 使用

给定以下数据:

```
**import** pandas **as** pd
**from** databricks **import** koalas **as** ks
**from** pyspark.sql **import** SparkSession

data = {**'a'**: [1, 2, 3, 4, 5, 6],
        **'b'**: [100, 200, 300, 400, 500, 600],
        **'c'**: [**"one"**, **"two"**, **"three"**, **"four"**, **"five"**, **"six"**]}

index = [10, 20, 30, 40, 50, 60]
```

你可以从熊猫的数据框架开始:

```
pdf = pd.DataFrame(data, index=index)*# from a pandas dataframe* kdf = ks.from_pandas(pdf)
```

来自考拉的数据框架:

```
*# start from raw data* kdf = ks.DataFrame(data, index=index)
```

或来自火花数据帧(单向):

```
# creating a spark dataframe from a pandas dataframe
sdf2 = spark_session.createDataFrame(pdf)# and then converting the spark dataframe to a koalas dataframe
kdf = sdf.to_koalas('index')
```

一个完整的简单输出示例:

A simple comparison of pandas, Koalas, pyspark Dataframe API

熊猫和考拉的 API 差不多。官方文档中的更多示例:

[](https://koalas.readthedocs.io/en/latest/getting_started/10min.html) [## 10 分钟到考拉-考拉 0.20.0 文档

### 这是对考拉的简短介绍，主要面向新用户。本笔记本向您展示了一些关键的不同之处…

考拉. readthedocs.io](https://koalas.readthedocs.io/en/latest/getting_started/10min.html) 

# 谨记在心

关于考拉项目的一些说明:

*   如果你是从零开始，没有任何关于熊猫的知识，那么直接进入 PySpark 可能是一个更好的学习方法。
*   一些功能可能**丢失** —丢失的功能在[这里](https://github.com/databricks/koalas/tree/master/databricks/koalas/missing)记录
*   一些行为可能**不同**(例如，Null 与 NaN，NaN 用于考拉，更适合熊猫，Null 用于 Spark)
*   请记住，由于它是在幕后使用 Spark，s **ome 操作是懒惰的**，这意味着在有 Spark 动作之前，它们不会真正被评估和执行，比如打印出前 20 行。
*   我确实对一些操作的效率有些担心，例如`*.to_koalas()*` 给出了一个`*No Partition Defined for Window operation! Moving all data to a single partition, this can cause serious performance degradation.*` 警告，这似乎是因为索引操作和**可能会有相当大的问题，这取决于您拥有的数据量。**请注意，当您在`.to_koalas('index')`中指定索引列名称时，警告并不存在，这是合理的，因为 spark/koalas 知道使用哪一列作为索引，并且不需要将所有数据放入一个分区来计算全局排名/索引。更多细节请看这里:[https://towards data science . com/adding-sequential-ids-to-a-spark-data frame-fa 0 df 5566 ff 6](/adding-sequential-ids-to-a-spark-dataframe-fa0df5566ff6)

# **结论**

**免责声明:**我真的没有怎么使用它，因为当我开始学习 spark 时，这还不可用，但我真的认为了解可用的工具是很好的，并且它可能对那些来自熊猫环境的人有所帮助——我仍然记得我从熊猫数据帧切换到 Spark 数据帧时的困惑。

我希望这是有帮助的，并且知道考拉将会节省你一些时间和麻烦。任何想法，问题，更正和建议都非常欢迎:)

如果您想了解更多关于 Spark 的工作原理，请访问:

[](/explaining-technical-stuff-in-a-non-techincal-way-apache-spark-274d6c9f70e9) [## 用非技术性的方式解释技术性的东西——Apache Spark

### 什么是 Spark 和 PySpark，我可以用它做什么？

towardsdatascience.com](/explaining-technical-stuff-in-a-non-techincal-way-apache-spark-274d6c9f70e9) 

关于在 Spark 中添加索引:

[](/adding-sequential-ids-to-a-spark-dataframe-fa0df5566ff6) [## 向 Spark 数据帧添加顺序 id

### 怎么做，这是个好主意吗？

towardsdatascience.com](/adding-sequential-ids-to-a-spark-dataframe-fa0df5566ff6)