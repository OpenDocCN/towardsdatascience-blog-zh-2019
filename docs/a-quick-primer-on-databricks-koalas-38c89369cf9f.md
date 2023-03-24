# 树袋熊数据快速入门

> 原文：<https://towardsdatascience.com/a-quick-primer-on-databricks-koalas-38c89369cf9f?source=collection_archive---------26----------------------->

与 Spark 数据框互动熊猫词汇

![](img/8854751c9f0007a6313ef1cfc276db38.png)

Photo by [Jordan Whitt](https://unsplash.com/@jwwhitt?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/koala?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

在我的一个项目中，我广泛使用 Spark 来管理一些大型数据文件。尽管它通常以使用大型分布式系统的诸多好处而闻名，但它在本地同样适用于处理大量信息的项目。

我经常使用熊猫进行数据处理，但有时 Spark 会让它相形见绌。你可以对 Pandas 使用 chunksize，但是根据我的经验，即使你找到了合适的大小来管理，它也比 Spark 数据帧要慢得多，而且要消耗更多的内存。从 Pandas 迁移到 PySpark(使用 Spark 的 Python API)的挑战是导航数据帧的词汇是不同的。我真的不介意摸索着与包交互，但是 PySpark 的错误消息通常很隐晦，文档也不是很好。在某些情况下，它就像编写一个快速的 SQL 查询一样简单，但是通常情况下，您需要进行更新和创建函数，并且这些部分所需的语言变得更加复杂。

考拉入场！[今年早些时候由 Databricks 推出的](https://databricks.com/blog/2019/04/24/koalas-easy-transition-from-pandas-to-apache-spark.html)，考拉很容易将熊猫的相同知识应用到 Spark 数据框架中。从类似的导入开始，创建一个 Spark 会话:

```
import pandas as pd
import numpy as np
import databricks.koalas as ks
from pyspark.sql import SparkSession
```

树袋熊数据框可以通过多种不同方式创建:

```
# Dataframe from scratch
koala_df = ks.DataFrame(
    {'type': ['panda', 'koala', 'grizzly', 'brown'],
     'color': ['black/white', 'grey', 'brown', 'brown'],
      'score': 5, 1, 5, np.nan},
    index=[1, 2, 3, 4])# From a file
koala_df = ks.read_csv("all_the_bears.csv", header=0)# From an existing Spark dataframe
koala_df = spark_df.to_koalas()# From an existing Pandas dataframe
koala_df = ks.from_pandas(pandas_df)
```

作为一个树袋熊的数据框架，你可以接触到和熊猫一样的语言:

```
# Get first rows of dataframe
koala_df.head()# See the column names
koala_df.columns# Quick summary stats
koala_df.describe()# Convert to Numpy
koala_df.to_numpy()# Group rows
koala_df.groupby('score')
```

它还提供了一些在 PySpark 中很麻烦的数据清理特性:

```
# Drop rows with missing values
koala_df.dropna(how='any')# Fill missing values
koala_df.fillna(value=5)
```

我最喜欢的功能之一是轻松导出，这在 Spark 中肯定会很时髦:

```
# Export to csv
koala_df.to_csv("bear_bears.csv")
```

作为一个不断学习和寻求提高技能的人，对我来说最重要的是，它使使用 Pandas 的本地项目更容易过渡到使用 Spark 的更具可扩展性的项目。我建议你用它弄脏你的手，看看你有什么想法。现在去下载一个 10GB 的 json 文件，尽情享受吧！