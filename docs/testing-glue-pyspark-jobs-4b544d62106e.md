# 测试胶水 Pyspark 作业

> 原文：<https://towardsdatascience.com/testing-glue-pyspark-jobs-4b544d62106e?source=collection_archive---------10----------------------->

## 设置你的环境，这样你的 [Glue](https://aws.amazon.com/glue/) PySpark 作业就可以读取和写入一个模仿的 S3 桶，这要感谢 [moto 服务器](https://github.com/spulec/moto/blob/master/docs/docs/server_mode.rst)。

![](img/af147c165a711187d658700efcbf9054.png)

Photo by [Scott Sanker](https://unsplash.com/@scottsanker?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/glue?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

# 挑战

胶合作业的典型用例是:

*   你阅读来自 S3 的数据；
*   你对这些数据进行一些转换；
*   你把转换后的数据传回 S3。

当编写 PySpark 作业时，用 Python 编写代码和测试，并使用 [PySpark 库](https://pypi.org/project/pyspark/)在 Spark 集群上执行代码。但是我如何让 Python 和 Spark 用同一个被嘲笑的 S3 桶进行通信呢？

> 在本文中，我将向您展示如何设置一个模拟的 S3 桶，您可以从 python 进程以及 Spark 集群访问它。

# 测试环境

## 创建粘合作业的先决条件

我们用的是 [Glue 1.0](https://docs.aws.amazon.com/glue/latest/dg/release-notes.html) ，也就是 Python 3.6.8，Spark/PySpark 2.4.3，Hadoop 2.8.5。
确定；

*   你已经安装了[python 3 . 6 . 8](https://www.python.org/downloads/release/python-368/)；
*   你已经安装了 Java[JDK 8](https://www.oracle.com/technetwork/java/javase/downloads/jdk8-downloads-2133151.html)；
*   您已经安装了用于 hadoop 2.7 的 spark 2.4.3。

> **注意** : Glue 使用 Hadoop 2.8.5，但是为了简单起见，我们使用 Hadoop 2.7，因为它是 Spark 2.4.3 附带的。

## Python 依赖性

```
pipenv --python 3.6
pipenv install moto[server]
pipenv install boto3
pipenv install pyspark==2.4.3
```

# PySpark 代码使用了一个模仿的 S3 桶

如果您遵循了上述步骤，您应该能够成功运行以下脚本:

```
import os
import signal
import subprocessimport boto3
from pyspark.sql import DataFrame
from pyspark.sql import SparkSession **# start moto server, by default it runs on localhost on port 5000.** process = subprocess.Popen(
    "moto_server s3", stdout=subprocess.PIPE,
    shell=True, preexec_fn=os.setsid
)**# create an s3 connection that points to the moto server.** s3_conn = boto3.resource(
    "s3", endpoint_url="http://127.0.0.1:5000"
)
**# create an S3 bucket.** s3_conn.create_bucket(Bucket="bucket")**# configure pyspark to use hadoop-aws module.
# notice that we reference the hadoop version we installed.**
os.environ[
    "PYSPARK_SUBMIT_ARGS"
] = '--packages "org.apache.hadoop:hadoop-aws:2.7.3" pyspark-shell'**# get the spark session object and hadoop configuration.** spark = SparkSession.builder.getOrCreate()
hadoop_conf = spark.sparkContext._jsc.hadoopConfiguration()**# mock the aws credentials to access s3.** hadoop_conf.set("fs.s3a.access.key", "dummy-value")
hadoop_conf.set("fs.s3a.secret.key", "dummy-value")
**# we point s3a to our moto server.** hadoop_conf.set("fs.s3a.endpoint", "http://127.0.0.1:5000")
**# we need to configure hadoop to use s3a.** hadoop_conf.set("fs.s3.impl", "org.apache.hadoop.fs.s3a.S3AFileSystem")**# create a pyspark dataframe.** values = [("k1", 1), ("k2", 2)]
columns = ["key", "value"]
df = spark.createDataFrame(values, columns)**# write the dataframe as csv to s3.** df.write.csv("s3://bucket/source.csv")**# read the dataset from s3** df = spark.read.csv("s3://bucket/source.csv")**# assert df is a DataFrame** assert isinstance(df, DataFrame)**# shut down the moto server.** os.killpg(os.getpgid(process.pid), signal.SIGTERM)print("yeeey, the test ran without errors.")
```

将上述代码复制粘贴到一个名为“py spark-mocked-S3 . py”*的文件中，并执行:*

```
*pipenv shell
python pyspark-mocked-s3.py*
```

*输出将类似于:*

```
*(glue-test-1) bash-3.2$ python pyspark-mocked-s3.py
* Running on [http://127.0.0.1:5000/](http://127.0.0.1:5000/) (Press CTRL+C to quit)...127.0.0.1 - - [28/Nov/2019 20:54:59] "HEAD /bucket/source.csv/part-00005-0f74bb8c-599f-4511-8bcf-8665c6c77cc3-c000.csv HTTP/1.1" 200 -
127.0.0.1 - - [28/Nov/2019 20:54:59] "GET /bucket/source.csv/part-00005-0f74bb8c-599f-4511-8bcf-8665c6c77cc3-c000.csv HTTP/1.1" 206 -yeeey, the test ran without errors.*
```

# *编写测试用例*

*上面脚本中显示的原则在我的 repo[testing-glue-py spark-jobs](https://github.com/vincentclaes/testing-glue-pyspark-jobs)中以更结构化的方式应用。*

*在这个 repo 中，你会发现一个 Python 文件， **test_glue_job.py** 。这个文件是 Glue PySpark 作业的一个测试用例的例子。它结合了上述逻辑和我写的一篇关于[测试无服务器服务](/testing-serverless-services-59c688812a0d)的文章中概述的原则。查看测试用例，并按照自述文件中的步骤运行测试。为了方便起见，我在下面添加了测试用例。*

*祝你好运！*

*[https://gist.github.com/vincentclaes/5d78f3890a117c613f23e3539e5e3e5d](https://gist.github.com/vincentclaes/5d78f3890a117c613f23e3539e5e3e5d)*

*【https://stackoverflow.com/a/50242383/1771155
[https://gist.github.com/tobilg/e03dbc474ba976b9f235](https://gist.github.com/tobilg/e03dbc474ba976b9f235)
[https://github . com/spulec/moto/issues/1543 # issue comment-429000739](https://github.com/spulec/moto/issues/1543#issuecomment-429000739)*