# AWS 弹性 MapReduce(EMR)——你不应该忽视的 6 个警告

> 原文：<https://towardsdatascience.com/aws-elastic-mapreduce-emr-6-caveats-you-shouldnt-ignore-7a3e260e19c1?source=collection_archive---------9----------------------->

![](img/03086ffc72ed8cec6a89696cd7498f03.png)

如果您从事数据和分析行业，您一定听说过新兴趋势“数据湖”，简单来说，它代表一种存储策略，允许组织在一个地方存储来自不同来源、具有不同特征(大小、格式和速度)的数据。然后，Data-lake 成为许多使用情形的推动者，例如高级分析或数据仓库等，通常数据会移动到更专业的存储中，例如 MPP 关系引擎或 NoSQL，以更好地满足特定使用情形的要求。如果在 AWS、Azure 或 GCP 等云环境中提供平台，那么对象存储(例如 AWS 的 S3，Azure 的 Azure 数据湖存储 Gen2)通常是提供数据湖基础物理层的最强候选。一旦数据进入数据湖，它将经过一系列区域/阶段/层，以在数据中建立语义一致性，并使其符合最佳消费。

通常，根据数据湖平台的参考架构，它与您选择的云提供商无关，Hadoop(特别是 Spark)被用作处理引擎/组件来处理数据层中的数据，因为它会经过不同的层。这些处理框架与数据湖服务很好地集成在一起，提供了水平可伸缩性、内存计算和非结构化数据处理等功能，这使它们成为这种环境下的可行选项。在云中使用 Hadoop 发行版通常有多种选择，例如，可以着手调配基于 IaaS 的基础架构(即 AWS EC2 或 Azure 虚拟机，并安装 Hadoop 发行版，如 vanilla Hadoop、Cloudera/Hortonworks)。或者，几乎所有的云提供商都在原生提供 Hadoop 作为托管服务(例如 AWS 中的 ElasticMapReduce (EMR)、Azure 中的 HDInsight/Databricks、GCP 的 Cloud Dataproc)。每种选择都有其利弊。例如，对于基于 IaaS 的实施，自行调配、配置和维护集群的开销成为许多人的一大担忧。此外，云的内在主张，如弹性和可扩展性，对基于 IaaS 的实施构成了挑战。

另一方面，在减少支持和管理开销方面，像 EMR 这样的托管产品确实提供了有力的主张。但是从功能的角度来看，在使用领先的云提供商提供的托管 Hadoop 产品时，仍然有许多需要注意的地方。虽然根据我的经验，我已经与三家云提供商的三大托管 Hadoop 产品合作过，但在这篇文章中，我将特别强调 AWS EMR 的一些注意事项。这背后的动机是让读者能够更好地利用 EMR 的潜力，同时避免潜在的问题，如果在您的开发中没有考虑这些影响，您可能会遇到这些问题。

# AWS 粘合数据目录:

AWS Glue 是一个托管数据目录和 ETL 服务。具体来说，当用于数据目录目的时，它为传统 Hadoop 集群过去依赖于 Hive 表元数据管理的 Hive metastore 提供了一个替代品。

![](img/b5fc4f7a65d2d65f19153dee0f57ce24.png)

Conceptual view of how Glue integrated with AWS services eco-system. (Source: [https://docs.aws.amazon.com/athena/latest/ug/glue-athena.htm)l](https://docs.aws.amazon.com/athena/latest/ug/glue-athena.html)

## 1.粘附数据库位置:

当使用 Glue Catalog 时，人们通常会创建提供目录中表的逻辑分组的数据库。现在，当您通常使用以下命令创建粘合数据库时:

```
CREATE DATABASE <database_name>
```

然后，如果您执行以下任一操作:

1.  计算此数据库表的统计数据

```
ANALYZE TABLE <database_name.table_name> COMPUTE STATISTICS
```

1.  使用 Spark SQL dataframe 的 saveAsTable 函数:

```
df.write.saveAsTable(“database_name.table_name”)
```

您肯定会遇到异常(其中一个会声明“它不能从空字符串创建路径”，另一个会声明“没有到主机的路由”)

这一警告的原因是，默认情况下，它将数据库位置设置为对创建数据库的集群有效的 HDFS 位置。但是，如果您使用多个集群或以临时方式使用多个集群(即，您启动一个集群，使用，然后终止)，那么 HDFS 位置将不再有效，从而带来问题。

解决方法是在创建数据库时明确指定 S3 位置:

```
CREATE DATABASE <database_name> LOCATION 's3://<bucket_name>/<folder_name>'
```

或者，您也可以在创建数据库后，在 Glue Catalog 中编辑数据库位置。

## 2.重命名粘附表列:

如果您已经创建了一个表，并且想要重命名一个列，方法之一是您可以通过 AWS Glue 来实现。然而，我所看到的是，即使你可以通过 Glue 做到这一点，有时也会导致元数据不一致。例如，如果您重命名一个列，然后通过 Athena 和/或 EMR 查询该表，两者可能会显示不同的视图，即一个显示重命名的列，另一个显示原始的列。因此，建议避免这样做，用正确的列名创建一个新表(或者求助于其他方法)。

## 3.使用外部表进行数据操作:

这个问题不是 AWS EMR 独有的，但它是值得警惕的。胶表，投影到 S3 桶是外部表。因此，如果删除一个表，底层数据不会被删除。但是如果您删除一个表，再次创建它并覆盖它(通过 spark.sql()或通过 dataframe APIs)，它将按预期覆盖内容。但是，如果您删除表，创建它，然后插入，因为原始数据仍然在那里，因此您实际上得到的是追加结果，而不是覆盖结果。如果您也想删除表中的内容，一种方法是在 S3 删除文件(通过 AWS CLI/SDK 或控制台),但是请注意，在您这样做时，请确保在此之后运行 *emrfs sync* (如下所示)。

# EMR 文件系统(EMRFS):

根据计算层和存储层保持分离的最新架构模式，使用 EMR 群集的方式之一是将 S3 用作存储层(或数据湖),即 EMR 群集从 S3 读取/写入持久数据。由于 S3 本质上是一种对象存储，并且这种对象存储通常具有一致性约束(即，最终在某些方面保持一致)，因此当与 Hadoop 等计算引擎一起使用时，这可能会带来挑战。AWS 在这种情况下采用的方法是 EMRFS 形式，这是 EMR 集群用于从 Amazon EMR 直接向 S3 读写文件的 HDFS 的实现。这提供了在亚马逊 S3 中存储持久数据的能力，以便与 Hadoop 一起使用，同时还提供了一致视图等功能。

![](img/71a9776a824e721a2471829f278c61a3.png)

Source: [https://www.slideshare.net/AmazonWebServices/deep-dive-amazon-elastic-map-reduce](https://www.slideshare.net/AmazonWebServices/deep-dive-amazon-elastic-map-reduce)

这里需要注意的是，EMRFS 使用了另一个存储，即 dynamo db(AWS 的 NoSQL 产品),来保存关于 S3 的元数据。然后，EMR 集群利用 DynamoDB 来确保 S3 的元数据(如对象键)是否一致。这有几个含义:

## 1.定价:

因为您将使用 DynamoDB，所以会有与之相关的成本。当 DynamoDB 中的一个表被配置为存储 EMRFS 元数据时，默认情况下，它会使用配置的读/写容量进行部署。虽然 DynamoDB 非常经济(相对而言)，所以成本不会很高，但还是要注意这一点，并确保在您的平台 OPEX 估计中考虑到这一成本。

## 2.大量小文件的含义:

如果您正在处理大量数据和大量小文件(如果您希望 Hadoop 处理管道能够执行，那么您应该尽一切可能避免这种情况)，并且如果多个集群正在从/向 S3 读取/写入数据，那么 DynamoDB 要管理的元数据会增加，并且有时会超过所配置的读取和写入吞吐量。这会限制读/写请求，并导致延迟(在某些情况下还会导致作业失败)。在这种情况下，您可以增加 DynamoDB 提供的读/写容量，或者将其设置为自动伸缩。此外，您还可以在 DynamoDB 中监控消耗，如果遇到此类问题，您还可以设置警报，并且可以自动完成整个过程(通过多种方式，例如使用 AWS Lambda)

## 3.正在同步元数据:

如果通过 EMR 从/向 S3 读取/写入数据，那么 EMRFS 元数据将保持一致，一切正常。但是，如果您通过任何外部机制操作 S3 数据，例如使用 AWS CLI/SDK 以及出于任何原因删除/添加文件(例如，在删除表之后)，那么元数据往往会变得不一致。因此，EMR 作业可能会停滞不前。在这种情况下，解决方案是同步元数据，即在 S3 中发生的变化，例如，新对象的添加/删除需要注册到 EMRFS 元数据持久化的 DynamoDB 中。其中一种方法是通过 Linux bash shell。通常，emrfs 实用程序随 EMR 节点一起安装，因此您可以通过 SSH 进入 EMR 主节点并运行:

```
 emrfs sync s3://<bucket_name>/<folder_where_you_manipulated_data>
```

在运行时，DynamoDB 中的 EMRFS 元数据将得到更新，不一致性将被消除，您的 EMR 作业应该可以顺利运行。

总之，每个产品都有一些细微差别，电子病历也是如此。这篇博文的目的不是强调它的消极方面，而是教育你，EMR 的潜在用户，让你能够最好地利用 AWS 的这项出色服务。这句话出自一位在企业级生产环境中成功处理万亿字节数据的人之手。

如果你有兴趣提升自己在这些技术方面的技能，一定要看看我的关于 Spark 的最畅销课程和我的关于大数据分析的书