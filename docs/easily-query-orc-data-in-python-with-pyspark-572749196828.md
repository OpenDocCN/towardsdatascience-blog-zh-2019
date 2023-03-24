# 使用 PySpark 轻松查询 Python 中的 ORC 数据

> 原文：<https://towardsdatascience.com/easily-query-orc-data-in-python-with-pyspark-572749196828?source=collection_archive---------25----------------------->

![](img/032692317d31cc3ff4038b0dbc72e4cd.png)

Photo by [Eric Han](https://unsplash.com/@madeyes?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

优化的行列(ORC)是一种面向列的数据存储格式，是 Apache Hadoop 家族的一部分。虽然 ORC 文件和处理它们通常不在数据科学家的工作范围内，但有时您需要提取这些文件，并使用您选择的数据管理库来处理它们。

最近，我遇到了一种情况，我想处理一些以 ORC 格式存储的对象数据。当我把它读出来时，我不能直接把它写到数据帧中。我想分享一些以 ORC 格式获取数据并将其转换成更容易接受的格式的技巧，比如熊猫数据帧或 CSV。

为此，我们将利用 Pyspark。如果你刚刚开始使用 Pyspark，这里有一个[很棒的介绍](/a-brief-introduction-to-pyspark-ff4284701873)。在本教程中，我们将快速浏览 PySpark 库，并展示如何读取 ORC 文件，并将其读出到 Pandas 中。

我们将从终端内部安装 PySpark 库

```
Pip install pyspark
```

从这里，我们将引入 PySpark 库的两个部分，SparkContext 和 SQLContext。如果你是 Spark 新手，那么[我推荐这个教程。](https://jaceklaskowski.gitbooks.io/mastering-apache-spark/spark-SparkContext.html)您可以将 SparkContext 视为所有 Apache Spark 服务的入口点，也是我们 Spark 应用程序的核心。SQLContext 被认为是 Spark SQL 功能的[入口点](https://spark.apache.org/docs/1.6.1/sql-programming-guide.html)，使用 SQLContext 允许您以一种熟悉的、类似 SQL 的方式查询 Spark 数据。

```
from pyspark import SparkContext, SQLContext
sc = SparkContext(“local”, “SQL App”)
sqlContext = SQLContext(sc)
```

你可以在上面的代码中看到，我们还为 SparkContext 声明了一些细节。在这种情况下，我们说我们的代码在本地运行，我们给它一个 appName，在这种情况下我们称之为“SQL App”。

一旦我们创建了 SparkContext(这里称为 sc ),我们就将它传递给 SQLContext 类来初始化 SparkSQL。

至此，我们已经安装了 PySpark 并创建了 Spark 和 SQL 上下文。现在到了重要的一点，读取和转换 ORC 数据！假设我们将数据存储在与 python 脚本相同的文件夹中，它被称为“objectHolder”。要将它读入 PySpark 数据帧，我们只需运行以下命令:

```
df = sqlContext.read.format(‘orc’).load(‘objectHolder’)
```

如果我们想把这个数据帧转换成熊猫数据帧，我们可以简单地做以下事情:

```
pandas_df = df.toPandas()
```

综上所述，我们的代码如下:

```
from pyspark import SparkContext, SQLContext
sc = SparkContext(“local”, “SQL App”)
sqlContext = SQLContext(sc)
df = sqlContext.read.format(‘orc’).load(‘objectHolder’)
pandas_df = df.toPandas()
```

现在我们有了。只用几行代码，我们就可以读取一个本地 orc 文件，并将其转换成我们更习惯的格式，在这个例子中，是一个熊猫数据帧。