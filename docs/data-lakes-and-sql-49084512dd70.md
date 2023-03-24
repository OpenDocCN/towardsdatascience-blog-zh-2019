# 数据湖和 SQL？？？

> 原文：<https://towardsdatascience.com/data-lakes-and-sql-49084512dd70?source=collection_archive---------9----------------------->

## 不是悖论。SQL 正被用于分析和转换数据湖中的大量数据。

随着数据量越来越大，推动因素是更新的技术和模式变化。与此同时，SQL 仍然是主流。在这里，我将探讨 SQL 如何用于数据湖和新的数据生态系统。

*TL；灾难恢复版本:随着数据和复杂性的增长，SQL 比以往任何时候都更适合分析和转换数据湖中的数据。*

![](img/e7c8c7b2a5e1260413be3b1f0c25efb5.png)

SQL Code — Photo by Caspar Camille Rubin on Unsplash

# 记得 NoSQL 吗？

NoSQL 数据库出现了，承诺了巨大的可伸缩性和简单性。

如果我们必须高速处理种类繁多、数量庞大的大量数据，我们被告知 NoSQL 是唯一的出路。供应商一直在喋喋不休地谈论 SQL 和中间件代码之间的阻抗不匹配。

我们现在发现大多数 NoSQL 供应商在多年贬低连接之后引入了 SQL 层。一些供应商引入了 SQL 的方言，使事情变得更糟。

看起来，在 NoSQL 引入这个 SQL 层是因为害怕像 Google Spanner 这样的新一代数据库，以及提供 JSON、XML 作为一级数据类型的数据库供应商。

# Hadoop 呢？

Hadoop 为开发人员提供了 map-reduce 接口，带来了一些巨大的进步，但也带来了许多恐慌。(参见— [MapReduce:一个重大的退步— DeWitt 和 Stonebraker](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.701.5795&rep=rep1&type=pdf) )。

使用 map-reduce 在 Hadoop 上进行数据处理，还有很多不足之处。性能调优、处理数据不对称、获得最佳吞吐量，所有这些都需要修改太多的裸机代码。

尝试了多种受 SQL 启发的方法

*   Apache Pig:类似 SQL 的语法，FOREACH 代替 FROM，GENERATE 代替 SELECT
*   Hive:类似 MySQL 的 SQL-in-Hadoop 语法，将 SQL 转换为 map-reduce
*   Drill、Impala、Presto 和 Pivotal 的 HAWQ:Hadoop 上的 SQL，绕过 map-reduce
*   Spark SQL:Spark 上的 SQL
*   Apache Phoenix:h base 上的 SQL
*   Hadoop 作为现有数据库的外部表:Oracle 大数据 SQL、Teradata SQL-H

在经历了许多“大数据年”以及一些 Hadoop 合并和破产之后，我们现在看到了这些技术的幸存者。Hadoop 技术现在更多地存在于云中，而不是本地。现在在组织中很少看到完整的 Cloudera 或 HortonWorks 堆栈。取而代之的是，一些精选的技术蓬勃发展，现在在云数据栈中广泛使用。

# 数据湖上的 SQL

Stonebraker 很久以前就指出，数据库的性能问题和可伸缩性与 SQL 关系不大，更多的是与数据库本身的设计有关([NoSQL 的讨论与 SQL](https://cacm.acm.org/blogs/blog-cacm/50678-the-no-sql-discussion-has-nothing-to-do-with-sql/fulltext) 无关)。

SQL 的最大优势是它提供了熟悉性和分析数据的表现力。SQL 的健壮性来自关系代数和集合论的基础。

有了数据湖，这就是我们看到的技术。

*   Hive metastore 是最受欢迎的数据目录。
*   在 SQL 层中，Presto 作为查询层胜出，并在 Amazon Athena、Google Cloud DataProc、Qubole 中广泛使用。
*   Spark 和 Spark SQL 也被广泛使用。
*   Hadoop 文件系统(HDFS)用得不多，云存储(Azure Blob、谷歌云存储、AWS S3)更受欢迎，有 CSV、Avro 和 Parquet 文件格式。

# 云数据仓库和数据湖

原始文件系统上存储的经济性鼓励了数据湖的创建。SQL 用于分析这些数据。

亚马逊红移光谱可以查询 S3 数据。

Snowflake DB 可以使用 VARIANT 列在数据库内部存储 XML、JSON 或 ORC 数据，还可以使用外部表指向 S3 中的数据。

Google BigQuery 和 Azure SQL 数据仓库也支持外部表。

# SQL 和 ELT(提取负载转换)

数据处理的 ELT(提取-加载-转换)范式将数据转换步骤放在了最后。首先从源系统中提取并加载到数据库中。

RBAR(逐行处理)的旧的 **ETL** 方式与关系数据库执行的基于集合的处理形成了直接对比，而基于集合的处理构成了 SQL 的基础。

在 **ELT** 中，我们现在从源数据库中提取数据，并将其放入数据湖中。

SQL 转换在云数据仓库中或使用 Presto 完成，转换后的数据被加载到目标表中。

通过 GoldenGate、AWS DMS 或使用 Workato/Jitterbit/StitchData 之类的工具或 Kafka 之类的健壮事件管道的涓涓细流都被输入到数据湖或数据仓库中。源系统和装载区之间的转换最小。然后使用 SQL 将这些数据转换并加载到仓库和分析层。

这个 **ELT** 工具链使用 DAG(有向无环图)工具，如 Apache AirFlow 和无服务器函数，而不是旧的 **ETL** 工具链的类似 AutoSys 的调度程序。

DBT 是另一个正在转型领域流行的工具。像 FiveTran 和 Matillion 这样的云数据处理工具也使用 SQL 和 ELT。Domo 对 SQL 进行排序以创建转换管道。Looker 基于 LookML 生成 SQL。

# 参考

1.  德威特博士和斯通布雷克博士(2008 年)。MapReduce:一大退步。*数据库列*， *1* ，23。
2.  斯通布雷克，M. (2009 年)。“NoSQL”的讨论与 SQL 无关。*ACM 的通信*。