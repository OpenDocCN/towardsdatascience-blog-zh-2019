# 利用气流在 AWS 中转换拼花地板(第 1 部分)

> 原文：<https://towardsdatascience.com/parquet-conversion-in-aws-using-airflow-part-1-66adc0485405?source=collection_archive---------17----------------------->

这篇文章将探讨云计算服务中围绕拼花地板的一切，优化的 S3 文件夹结构，分区的适当大小，何时，为什么以及如何使用分区，以及随后如何使用气流来编排一切。

![](img/6cfbc3c611494a422d355fe9eda94673.png)

我不想浪费这个空间来详述气流的特性和它是一个多么完整的产品。网上有很多帖子探索它的功能。我宁愿将这篇文章分成以下四个部分:

1.  拼花地板文件格式概述
2.  s3 文件夹结构的类型以及正确的 S3 结构“如何”节省成本
3.  为外部表(红移谱、雅典娜、ADLA 等)提供足够的分区大小和数量
4.  用气流片段总结(下一篇文章)

# **拼花文件格式和压缩类型**

选择完美的文件格式和压缩有很多因素，但以下 5 个因素涵盖了相当多的领域:

*   **基于列 vs 基于行** : 每个人都想使用 CSV，直到你的数据量达到几乎无法查看的程度，或者它占用了你的数据湖中的大量空间。如果您的数据大于，通常会有过多的列。正如我们所知，并不是所有的列都是信息性的，为了查询一些列，我们需要一种针对数据的列选择进行优化的格式。换句话说，当您将它导入外部表(Athena、Redshift Spectrum、Azure data lake Analytics 等)并从 parquet 数据中选择一列时，它不会也不应该在选择所需列之前加载整个文件。除非你在 Hadoop 集群中使用像 Hive 这样的开源工具，否则你将不得不为扫描的数据量买单。
*   **模式进化** : Parquet 支持模式进化。现在，大多数数据源都会在某个时候演化它们的模式(添加/删除列、数据类型更改等)，如果您将它接收到一种不支持模式演化的格式，那么您将无法使用以前接收的文件读取该文件。
*   **支持压缩**:现在，当你选择了一个文件格式，为什么不包含一个压缩格式，但选择正确的压缩格式是棘手的，因为你需要在速度或压缩率之间做出选择。到目前为止，Snappy 赢得了这场战斗，因为它在两个世界之间取得了巨大的平衡。此外，镶木地板支持它。
*   **存储空间**:我相信很多读者已经知道了，但是 parquet 是一种经过优化的格式，它消耗的空间是 CSV 的 1/10。我知道 S3，Azure blob 或 GCS 都不贵，你可以把它作为一个着陆区，没有任何成本，但几年后，当你最终使用数十兆字节的空间时，你会意识到成本确实上升了。
*   **自描述**:除了每个文件之外，Parquet 还给出了它的元数据，比如每个列的数据类型，这在列解析或者数据被发送到另一个系统的场景中总是需要的。

我知道文件格式取决于用例。例如:如果你想读取所有的值，那么 Avro 是最适合的，因为它是基于行的格式(读取键值对)，但并不是每个用例都有这种需求。Avro 也非常适合写密集型操作。

# **S3 文件夹结构及其如何节约成本**

现在拼花地板和隔墙是如何联系在一起的。因此，到目前为止，我们已经确定了 parquet 是大多数用例的正确文件格式。为了简单起见，我们假设我们定期接收数据，并确定它来自哪个时间段，我们在 s3 存储桶“生产”中给它一个这样的文件名:

> 数字营销/yyyy-mm-dd.snappy.parquet

一切都进行得很好，直到有人要求你处理这些关键点/斑点，或者将其放入他们的数据仓库(DWH)进行分析，或者用于机器学习(读取特征生成)任务。为什么很难，因为你的代码，你的逻辑，必须首先扫描桶中的所有文件，然后选择那些你需要处理的文件。现在，由于数据是周期性的，我们只想选择那些我们之前没有处理的键/blob。因此，让我们将这些键重新命名如下:

> yyyy/mm/DD/yyyy-mm-DD . snappy . parquet

换句话说，我们已经将数据划分为 yyyy/mm/dd 格式。到目前为止一切顺利。代码选择正确的密钥为 DWH 进行处理。几年后，我们收到一封电子邮件，说我们的 DWH 成本正在拍摄，我们需要管理它。但是数据科学家告诉我们，他们至少需要最近 5 年的数据来进行历史分析，如果不是更多的话。此外，他们还担心，由于 DWH 中的所有数据，进行汇总变得过于缓慢。外部表来了。

因此，Redshift spectrum 的以下数据定义语言(DDL)将把数据驻留在文件夹“S3://production/digital-marketing/Facebook/accounts/engagement/”下。将仓库结构(数据库/模式/表)与数据湖的结构对齐总是好的。你可以想象它的好处。

> 创建外部表 Facebook . accounts . engagement(
> page _ fans _ new BIGINT，
> page_fans BIGINT，
> account VARCHAR(80)，
> DATE DATE
> )
> partition by(year char(4)，month char(2)，day char(2))
> 行格式分隔的
> 字段以“|”终止
> location ' S3://production/Facebook/accounts/engagement/'
> 表属性(' skip.header.line.count'='1 '，' has _ has

并且密钥/blob/文件将使用这个名称。

> Facebook/accounts/engagement/yyyy/mm/DD/yyyy-mm-DD . snappy . parquet

要在*外部*表中添加数据，必须运行以下命令。

> alter table Facebook . accounts . engagement
> 添加分区(year='2019 '，month='01 '，day='01')
> 位置 S3://production/Facebook/accounts/engagement/year = 2019/month = 01/day = 01/'；

因此，当您运行上面的命令时，它将在其元数据中添加 3 个分区键，指定特定行来自哪个分区。最好的部分是，现在我们不需要扫描整个表来获取所需的数据。我们将只使用“where”子句来选择我们需要分析数据的年、月和日。但最糟糕的是，它将在元数据中添加 3 个分区键，这是反高潮的。因此，让我们把三个合并成一个。下面的 DDL 将用诸如‘2019–01–01’的值对 *batch_date* 上的数据进行分区。

> 创建外部表 Facebook . accounts . engagement(
> page _ fans _ new BIGINT，
> page_fans BIGINT，
> account VARCHAR(80)，
> DATE DATE
> )
> partition by(batch _ DATE char(10))
> 行格式分隔的
> 字段以“|”终止
> location ' S3://production/Facebook/accounts/engagement/'
> 表属性(' skip.header.line.count'='1 '，' has_encrypted_data'='false

为了添加分区，让我们使用下面的代码片段:

> alter table Facebook . accounts . engagement
> 添加分区(batch _ date = ' 2019–01–01 ')
> 位置 S3://production/Facebook/accounts/engagement/batch _ date = 2019–01–01/'；

现在，这将只在元数据中添加 1 个分区键，即 batch_date，而不是 3 个。

此外，s3/Blob/GCS 文件夹结构很大程度上取决于您在外部表上使用的查询类型。如果每天进行查询，那么 batch_date=YYYY-MM-DD 是最佳选择，其中每个分区至少应为 50–100 MB，但不能超过 2GB。如果您每小时查询一次，则 YYYY/MM/DD/HH。因为如果您每小时查询一次，并且您的结构是每日 YYYY/MM/DD，那么在一天结束之前，不能添加任何分区。如果您每天以小时为单位进行查询，即使文件非常大，也可能是一种大材小用，但是以小时为单位进行查询是有好处的，因为将来分析师需要逐小时分析，因为 1 小时内生成的数据非常大。因此，它可以是每日分区或每小时分区，但不能是每周分区或每月分区，因为分析师不希望等待一个月或一周才能获得新数据。

**注意**:删除和添加相同的分区(以及更新的文件)成本很高，因为 Redshift Spectrum 或 Athena 是根据扫描的数据量计费的。

外部表最多可以有 20，000 个分区，否则就是一个新表。

# 分区的大小和数量

到目前为止，我们已经对 batch_date 上的数据进行了分区，但是如果您正在使用 Redshift、Spark 或任何类似的工具，您可以并行卸载该文件。因此，问题是一个分区中有多少文件是最佳的，理想的分区大小应该是多少。

分区数据的文件大小可以从 1MB 到 10GB，但建议分区的最小大小应为 50-100MB 或更大，否则与未分区的数据相比，读取和解析外部表将花费大量时间。

如果数据的总大小小于 10Gbs，则根本不需要分区。在这种情况下，由于数据不是很大，您可以将所有文件放在一个目录中，因为向下钻取目录结构也会影响性能。但是问题是，尽量将文件数量保持在 100 以下，否则查询将再次花费大量时间，因为它必须找到文件，打开并读取它，这比处理结果更耗时。

# **总结**

因此，调查您的数据源的摄取率(MBs/天)。根据经验，当您的摄取率超过 500Mbs 天时，开始考虑分区，并以分区大小在 100 MB-2gb 左右的分区方案为目标。如果数据有时间成分，那么把它作为你的分区键，然后根据查询决定是需要按天分区还是按小时分区。

在下一篇文章中，我们将深入研究定制的气流操作符，看看如何轻松地处理气流中的拼花转换。

下面是第二部分:[https://towardsdatascience . com/parquet-conversion-in-AWS-using-air flow-part-2-8898029 c 49 db](/parquet-conversion-in-aws-using-airflow-part-2-8898029c49db)

 [## Gagandeep Singh -数据工程师-澳大利亚广播公司(ABC) | LinkedIn

### 加入 LinkedIn Gagandeep 曾参与涉及以下领域的结构化和非结构化数据的项目

www.linkedin.com](https://www.linkedin.com/in/gagandeepsingh8/)