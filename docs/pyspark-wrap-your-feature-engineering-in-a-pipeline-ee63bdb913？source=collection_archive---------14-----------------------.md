# Pyspark —将您的特征工程包装在管道中

> 原文：<https://towardsdatascience.com/pyspark-wrap-your-feature-engineering-in-a-pipeline-ee63bdb913?source=collection_archive---------14----------------------->

## 将变量创建集成到 spark 管道的指南

![](img/c642dff3a6b7d10764cc9b447e82d151.png)

[https://upload.wikimedia.org/wikipedia/commons/f/f3/Apache_Spark_logo.svg](https://upload.wikimedia.org/wikipedia/commons/f/f3/Apache_Spark_logo.svg)

为了有一个更干净和更工业化的代码，创建一个处理特征工程的管道对象可能是有用的。假设我们有这种类型的数据帧:

```
df.show()+----------+-----+
|      date|sales|
+----------+-----+
|2018-12-22|   17|
|2017-01-08|   22|
|2015-08-25|   48|
|2015-03-12|  150|
+----------+-----+
```

然后我们想创建从日期派生的变量。大多数时候，我们会做这样的事情:

现在，我们希望将这些变量的创建集成到 spark 的管道中，此外，在它们的计算之前采取一些保护措施。为此，我们将创建一个继承 spark 管道的`[Transformer](https://spark.apache.org/docs/latest/ml-pipeline.html#transformers)`方法的**的类，并且我们将添加一个在计算前检查输入的函数。**

*   我们的类继承了 Spark `[Transforme](https://spark.apache.org/docs/latest/ml-pipeline.html#transformers)r`的属性，这允许我们将其插入到管道中。
*   `[this](https://spark.apache.org/docs/latest/api/java/org/apache/spark/ml/util/Identifiable.html)`函数允许我们通过给对象分配一个唯一的 ID 来使我们的对象在管道中可识别和不可变
*   `[defaultCopy](https://spark.apache.org/docs/1.5.1/api/java/org/apache/spark/ml/param/Params.html#defaultCopy(org.apache.spark.ml.param.ParamMap))`尝试创建一个具有相同 UID 的新实例。然后，它复制嵌入的和额外的参数，并返回新的实例。
*   然后使用 check_input_type 函数检查输入字段的格式是否正确，最后我们简单地实现`[SQL Functions](https://spark.apache.org/docs/2.3.0/api/sql/index.html)` F.day of month 来返回我们的列。

我们将重用这个框架来创建我们需要的其他变量，他唯一要做的改变就是 ID 和 _transform 函数

我们已经定义了具有相同框架的 MonthQuarterExtractor，与`[withColumn](https://spark.apache.org/docs/1.6.1/api/java/org/apache/spark/sql/DataFrame.html#withColumn(java.lang.String,%20org.apache.spark.sql.Column))` 方法相比，它可能看起来有点冗长，但是要干净得多！

最后，我们对年份变量做同样的处理。

现在让我们将它们集成到我们的管道中。

```
df.show()+----------+-----+----------+----+------------+
|      date|sales|dayofmonth|year|monthquarter|
+----------+-----+----------+----+------------+
|2018-12-22|   17|        22|2018|           2|
|2017-01-08|   22|         8|2017|           0|
|2015-08-25|   48|        25|2015|           3|
|2015-03-12|  150|        12|2015|           1|
+----------+-----+----------+----+------------+
```

诀窍是，我们创建了一个“海关”变压器，并将其插入火花管道。也可以将它们与其他对象、向量汇编器、字符串索引器或其他对象一起插入管道。

# 最后

Spark 管道是一个非常强大的工具，我们可以在一个管道中管理几乎整个数据科学项目，同时保持每个对象的可追溯性，并允许更简单的代码工业化，不要犹豫滥用它！

感谢您阅读我，如果您对 Pyspark 的更多技巧|教程感兴趣，请不要犹豫，留下您的评论。 *(Y* ou 可以在这里找到代码*:*[](https://github.com/AlexWarembourg/Medium)**)**