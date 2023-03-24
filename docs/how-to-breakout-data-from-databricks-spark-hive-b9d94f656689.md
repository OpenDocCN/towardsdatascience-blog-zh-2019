# 如何从 Databricks-Spark-Hive 中获取数据

> 原文：<https://towardsdatascience.com/how-to-breakout-data-from-databricks-spark-hive-b9d94f656689?source=collection_archive---------28----------------------->

![](img/3e8e23cc93a6015afb1b6db75861bab6.png)

Fingers Trying to break out of jail, [Pixabay](https://pixabay.com/photos/thieves-theft-robbery-swag-money-2012538/).

## 简单的方法。

这篇文章是为那些使用 Databricks (DB)笔记本并希望通过使用 Pyspark 将基于 Hive 的数据集导出到外部机器的科学家而写的，以便使用 Pandas 获得更高效的工作流。

有很多方法可以做到以下几点，但这个方法对我很有效。

首先，在 DB 笔记本中创建一个 SQL 查询，并等待结果。以下查询是从 table_x 中选择所有列并将结果分配给 spark 数据框的简单示例。

> df = spark . SQL(" " SELECT * FROM table _ x " " " ")

Spark 不会参与，直到您要求它物化数据，因此，您必须执行以下命令。

> df.show()

此时，您应该会看到一个非常难看的打印输出，其中包含一些数据。

在打印输出之后，您需要安装 DB 客户机，对其进行配置，并在 DB 文件存储上创建一个目录。

> pip3 安装数据块-cli
> 
> 设置和配置 DB 客户端，我假设您可以遵循文档或者熟悉设置过程。[点击此处](https://docs.databricks.com/dev-tools/databricks-cli.html#set-up-the-cli)获取文件。
> 
> dbfs mkdirs dbfs:/file store/my/path/is/here

按照这些步骤，在 DB 笔记本中执行 write-to-JSON 命令，数据帧将保存在预定义路径的多个 JSON 文件中。请注意，路径应该根据您的配置进行更改。

> df . write . JSON(
> f " dbfs:/FileStore/my/path/is/here "，
> mode="overwrite "，
> compression="gzip"
> )

执行这些命令后，我们将在我们的机器上创建一个新目录，转到该目录，检查文件是否在我们的云目录中等待，最后使用 DB 客户端复制文件。

> mkdir/data/our data
> CD/data/our data
> dbfs ls dbfs:/file store/my/path/is/here
> dbfs CP-r dbfs:/file store/my/path/is/here。

按照这些步骤，您可以将 location 指向您机器的数据目录并处理文件，将所有下载的 JSON.GZ 文件连接到一个 Pandas 数据框中。

> 进口熊猫作为 pd
> 
> location = '/data/ourData '
> 
> 从 pathlib 导入路径
> files = list(路径(位置)。glob(" * . JSON . gz ")
> l =[]
> for f in files:
> l . append(PD . read _ JSON(f，lines = True))
> df = PD . concat(l)

现在，您可以使用 Pandas，处理数据的速度比使用 DB 笔记本更快。唯一的条件是，你实际上可以把所有的数据都放进你的机器的内存里。

我要感谢巴拉克·亚伊尔·赖夫的宝贵帮助，感谢阿黛尔·古尔审阅本文。

Ori Cohen 博士拥有计算机科学博士学位，主要研究机器学习。他是 TLV 新遗迹公司的首席数据科学家，从事 AIOps 领域的机器和深度学习研究。