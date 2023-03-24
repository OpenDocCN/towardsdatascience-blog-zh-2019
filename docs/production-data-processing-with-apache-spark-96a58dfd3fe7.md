# 用 PySpark 在自动气象站电子病历上处理生产数据

> 原文：<https://towardsdatascience.com/production-data-processing-with-apache-spark-96a58dfd3fe7?source=collection_archive---------9----------------------->

## 使用 AWS CLI 在集群上提交 PySpark 应用程序，分步指南

**带有 PySpark 和 AWS EMR 的数据管道**是一个多部分系列。这是第二部分。如果你需要 AWS EMR 的入门知识，请查看[第 1 部分](/getting-started-with-pyspark-on-amazon-emr-c85154b6b921)。

1.  [在 AWS EMR 上开始使用 PySpark](/getting-started-with-pyspark-on-amazon-emr-c85154b6b921)
2.  用 PySpark 在 AWS EMR 上处理生产数据**(本文)**

![](img/d7fab68eaeabe1a12ffb7d56a5601e38.png)

# 动机

Apache Spark 在大规模数据处理和分析领域风靡一时，这是有充分理由的。借助 Spark，组织能够从不断增长的数据堆中提取大量价值。正因为如此，能够构建 Spark 应用的数据科学家和工程师受到企业的高度重视。本文将向您展示如何从命令行在 Amazon EMR 集群上运行 Spark 应用程序。

大多数 PySpark 教程都使用 Jupyter 笔记本来演示 Spark 的数据处理和机器学习功能。原因很简单。在集群上工作时，笔记本通过在 UI 中提供快速反馈和显示错误消息，使得测试语法和调试 Spark 应用程序变得更加容易。否则，您将不得不挖掘日志文件来找出哪里出错了——这对于学习来说并不理想。

一旦您确信您的代码可以工作，您可能想要将您的 Spark 应用程序集成到您的系统中。在这里，笔记本就没那么有用了。要按计划运行 PySpark，我们需要将代码从笔记本转移到 Python 脚本中，并将该脚本提交给集群。在本教程中，我将向您展示如何操作。

# 我们开始吧

一开始，从命令行向集群提交 Spark 应用程序可能会令人生畏。我的目标是揭开这个过程的神秘面纱。本指南将向您展示如何使用 AWS 命令行界面来:

1.  创建一个能够处理比本地计算机容量大得多的数据集的集群。
2.  向集群提交 Spark 应用程序，集群读取数据、处理数据并将结果存储在可访问的位置。
3.  该步骤完成后自动终止集群，因此您只需在使用集群时付费。

# Spark 开发工作流程

当开发 Spark 应用程序来处理数据或运行机器学习模型时，我的首选是从使用 Jupyter 笔记本开始，原因如上所述。这里有一个关于[创建一个亚马逊 EMR 集群并用 Jupyter 笔记本](/getting-started-with-pyspark-on-amazon-emr-c85154b6b921)连接到它的指南。

一旦我知道我的代码起作用了，我可能想把这个过程作为一个预定的工作来进行。我会把代码放在一个脚本中，这样我就可以用 [Cron](https://www.ostechnix.com/a-beginners-guide-to-cron-jobs/) 或 [Apache Airflow](https://airflow.apache.org/start.html) 把它放在一个时间表中。

# 生产火花应用

*重要更新:本指南使用 AWS CLI 版本 1 —以下命令需要进行一些调整才能与版本 2 配合使用。*

[创建您的 AWS 帐户](https://aws.amazon.com/)如果您还没有。[安装](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html#install-tool-pip)和[配置](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-install.html#post-install-configure)AWS 命令行界面。要配置 AWS CLI，您需要添加您的凭据。您可以按照这些说明创建凭证[。您还需要指定您的默认区域。对于本教程，我们使用`us-west-2`。您可以使用任何您想要的区域。只是要确保所有的资源都使用相同的区域。](https://docs.aws.amazon.com/IAM/latest/UserGuide/getting-started_create-admin-group.html)

## 定义 Spark 应用程序

对于这个例子，我们将从 S3 加载亚马逊书评数据，执行基本处理，并计算一些聚合。然后，我们将把聚合数据帧写回 S3。

这个例子很简单，但这是 Spark 的一个常见工作流。

1.  从源(本例中为 S3)读取数据。
2.  使用 Spark ML 处理数据或执行模型工作流。
3.  将结果写在我们的系统可以访问的地方(在这个例子中是另一个 S3 桶)。

如果您还没有，[现在就创建一个 S3 存储桶](https://docs.aws.amazon.com/quickstarts/latest/s3backup/step-1-create-bucket.html)。 ***确保你创建桶的区域与你在本教程剩余部分使用的区域相同。我将使用地区“美国西部(俄勒冈州)”。*** 复制下面的文件。确保编辑`main()`中的`output_path`来使用你的 S3 桶。然后把`pyspark_job.py`上传到你的桶里。

```
# pyspark_job.pyfrom pyspark.sql import SparkSession
from pyspark.sql import functions as Fdef create_spark_session():
    """Create spark session.Returns:
        spark (SparkSession) - spark session connected to AWS EMR
            cluster
    """
    spark = SparkSession \
        .builder \
        .config("spark.jars.packages",
                "org.apache.hadoop:hadoop-aws:2.7.0") \
        .getOrCreate()
    return sparkdef process_book_data(spark, input_path, output_path):
    """Process the book review data and write to S3.Arguments:
        spark (SparkSession) - spark session connected to AWS EMR
            cluster
        input_path (str) - AWS S3 bucket path for source data
        output_path (str) - AWS S3 bucket for writing processed data
    """
    df = spark.read.parquet(input_path)
    # Apply some basic filters and aggregate by product_title.
    book_agg = df.filter(df.verified_purchase == 'Y') \
        .groupBy('product_title') \
        .agg({'star_rating': 'avg', 'review_id': 'count'}) \
        .filter(F.col('count(review_id)') >= 500) \
        .sort(F.desc('avg(star_rating)')) \
        .select(F.col('product_title').alias('book_title'),
                F.col('count(review_id)').alias('review_count'),
                F.col('avg(star_rating)').alias('review_avg_stars'))
    # Save the data to your S3 bucket as a .parquet file.
    book_agg.write.mode('overwrite')\
        .save(output_path)def main():
    spark = create_spark_session()
    input_path = ('s3://amazon-reviews-pds/parquet/' +
                  'product_category=Books/*.parquet')
    output_path = 's3://spark-tutorial-bwl/book-aggregates'
    process_book_data(spark, input_path, output_path)if __name__ == '__main__':
    main()
```

## 使用 AWS 命令行界面

是时候创建集群并提交应用程序了。一旦我们的应用程序完成，我们将告诉集群终止。自动终止允许我们仅在需要时才支付资源费用。

根据我们的使用案例，我们可能不想在完成时终止集群。例如，如果您有一个依赖 Spark 来完成数据处理任务的 web 应用程序，那么您可能希望有一个一直运行的专用集群。

运行下面的命令。确保用你自己的文件替换 ***粗体斜体*** 部分。关于`--ec2-attributes`和`--bootstrap-actions`以及所有其他参数的细节包括在下面。

```
aws emr create-cluster --name "Spark cluster with step" \
    --release-label emr-5.24.1 \
    --applications Name=Spark \
    --log-uri ***s3://your-bucket/logs/*** \
    --ec2-attributes KeyName=***your-key-pair*** \
    --instance-type m5.xlarge \
    --instance-count 3 \
    --bootstrap-actions Path=***s3://your-bucket/emr_bootstrap.sh*** \
    --steps Type=Spark,Name="Spark job",ActionOnFailure=CONTINUE,Args=[--deploy-mode,cluster,--master,yarn,***s3://your-bucket/pyspark_job.py***] \
    --use-default-roles \
    --auto-terminate
```

**`**aws emr create-cluster**`重要**论据:****

*   **`--steps`告诉您的集群在集群启动后要做什么。确保将`--steps`参数中的`s3://your-bucket/pyspark_job.py`替换为 Spark 应用程序的 S3 路径。您还可以将应用程序代码放在 S3 上，并传递一个 S3 路径。**
*   **`--bootstrap-actions`允许您指定要安装在所有集群节点上的软件包。只有当您的应用程序使用非内置 Python 包而不是`pyspark`时，这一步才是必需的。要使用这样的包，使用下面的例子作为模板创建您的`emr_bootstrap.sh`文件，并将其添加到您的 S3 桶中。在`aws emr create-cluster`命令中包含`--bootstrap-actions Path=s3://your-bucket/emr_bootstrap.sh`。**

```
#!/bin/bash
sudo pip install -U \
    matplotlib \
    pandas \
    spark-nlp
```

*   **`--ec2-attributes`允许您指定许多不同的 EC2 属性。使用以下语法设置您的密钥对`--ec2-attributes KeyPair=your-key-pair`。 ***注意:这只是你的密钥对的名字，不是文件路径。*** 你可以在这里了解更多关于创建密钥对文件[的信息。](https://medium.com/@brent_64035/create-a-key-pair-file-for-aws-ec2-b71c6badb16)**
*   **`--log-uri`需要一个 S3 桶来存储你的日志文件。**

****其他** `**aws emr create-cluster**` **论据解释:****

*   **`--name`给你正在创建的集群一个标识符。**
*   **`--release-label`指定使用哪个版本的 EMR。我推荐使用最新版本。**
*   **`--applications`告诉 EMR 您将在集群上使用哪种类型的应用程序。要创建火花簇，使用`Name=Spark`。**
*   **`--instance-type`指定要为集群使用哪种类型的 EC2 实例。**
*   **`--instance-count`指定集群中需要多少个实例。**
*   **`--use-default-roles`告诉集群使用 EMR 的默认 IAM 角色。如果这是你第一次使用 EMR，你需要运行`aws emr create-default-roles`才能使用这个命令。如果您已经在配置了 AWS CLI 的区域中的 EMR 上创建了一个集群，那么您应该已经准备好了。**
*   **`--auto-terminate`告诉集群在`--steps`中指定的步骤完成后立即终止。如果您想让您的集群保持运行，请排除此命令—请注意，只要您保持集群运行，您就是在为它付费。**

## **检查 Spark 应用程序的进度**

**在您执行了`aws emr create-cluster`命令之后，您应该会得到一个响应:**

```
{
    "ClusterId": "j-xxxxxxxxxx"
}
```

**登录到 AWS 控制台并导航到 EMR 仪表板。您的集群状态应该是“正在启动”。您的集群启动、引导和运行您的应用程序大约需要 10 分钟(如果您使用了我的示例代码)。一旦该步骤完成，您应该会在 S3 存储桶中看到输出数据。**

**就是这样！**

# **最后的想法**

**您现在知道了如何创建 Amazon EMR 集群并向其提交 Spark 应用程序。该工作流是使用 Spark 构建生产数据处理应用程序的重要组成部分。我希望您现在对使用所有这些工具更有信心了。**

**一旦你的工作顺利进行，考虑在亚马逊上建立一个气流环境来安排和监控你的管道。**

# **取得联系**

**感谢您的阅读！请让我知道你是否喜欢这篇文章，或者你是否有任何批评。如果你觉得这个指南有用，一定要关注我，这样你就不会错过我以后的文章。**

**如果你在一个数据项目上需要帮助或者想打个招呼，**在**[**LinkedIn**](https://www.linkedin.com/in/brent-lemieux/?source=post_page---------------------------)上联系我。干杯！**