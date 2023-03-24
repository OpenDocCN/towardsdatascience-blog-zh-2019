# 在 Apache PySpark 中引入时间序列数据的端到端插值

> 原文：<https://towardsdatascience.com/end-to-end-time-series-interpolation-in-pyspark-filling-the-gap-5ccefc6b7fc9?source=collection_archive---------5----------------------->

![](img/2b86e9e7315098a5d0de01055ec454b7.png)

Photo by [Steve Halama](https://unsplash.com/@steve3p_0?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/spark?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

任何处理数据的人都知道现实世界中的数据通常是不完整的，清理数据会占用你大量的时间(有人知道 80/20 法则吗？).最近从 Pandas 搬到 Pyspark，我已经习惯了 Pandas 提供的便利，而 Pyspark 由于它的分布式特性有时会缺乏这些便利。我一直特别缺少的一个特性是插值(或填充)时间序列数据的直接方式。虽然填充缺失值的问题已经被讨论过几次(例如，这里的[[]这里的](https://johnpaton.net/posts/forward-fill-spark) ])，但是我找不到一个来源，它详细描述了生成底层时间网格然后填充缺失值的端到端过程。这篇文章试图填补这个空白。从缺失条目的时间序列开始，我将展示我们如何利用 PySpark 首先生成缺失的时间戳，然后使用三种不同的插值方法(向前填充、向后填充和插值)填充缺失的值。使用在一组房屋中收集的传感器读取数据的例子来演示这一点。这篇文章的完整代码可以在我的 github 中的[找到。](https://github.com/walkenho/tales-of-1001-data/blob/master/timeseries-interpolation-in-spark/interpolating_timeseries_p2_pyspark.ipynb)

# 准备数据并可视化问题

为了演示这个过程，首先，我们生成一些测试数据。数据集包含两所房子的数据，并使用 sin()和 cos()函数为一组日期生成一些传感器读取数据。为了生成缺失值，我们随机丢弃一半的条目。

Creating the sensor data test set

Raw data set

下图显示了缺失值清晰可见的数据。

![](img/da298f974c0788df1f9556ce790244b4.png)

Read Data with Missing Entries

为了并行化数据集，我们将 Pandas 数据帧转换为 Spark 数据帧。注意，我们需要用 10⁹除日期时间，因为熊猫日期时间和火花的时间单位是不同的。我们还添加了列‘read time _ exist’来跟踪哪些值丢失了。

```
import pyspark.sql.functions as func
from pyspark.sql.functions import coldf = spark.createDataFrame(df0)
df = df.withColumn("readtime", col('readtime')/1e9)\
       .withColumn("readtime_existent", col("readtime"))
```

现在我们的数据框架看起来像这样:

Read Data prepared for interpolation with PySpark

# 插入文字

## 对读取的日期时间重新采样

第一步是对读取时间数据进行重新采样。如果我们和熊猫一起工作，这将是直接的，我们可以使用`resample()`方法。然而，Spark 在分布式数据集上工作，因此没有提供等效的方法。在 PySpark 中获得相同的功能需要三个步骤。在第一步中，我们按“房子”对数据进行分组，并为每栋房子生成一个包含等距时间网格的数组。在第二步中，我们使用 spark SQL 函数`explode()`为数组的每个元素创建一行。在第三步中，使用所得结构作为基础，使用外部左连接将现有读取值信息连接到该基础。下面的代码展示了如何做到这一点。

注意，这里我们使用的是 spark 用户自定义函数(如果你想了解更多关于如何创建 UDF 的知识，你可以看看这里的。从 Spark 2.3 开始，Spark 提供了一个 pandas udf，它利用 Apache Arrow 的性能来分布计算。如果您使用 Spark 2.3，我建议您研究一下这个，而不是使用(性能很差的)内置 UDF。

结果表的摘录如下所示:

可以看到，readtime _ existent 列中的 null 表示缺少读取值。

## 使用窗口函数的向前填充和向后填充

当使用向前填充时，我们用最新的已知值填充缺失的数据。相反，当使用向后填充时，我们用下一个已知值填充数据。这可以通过结合使用 SQL 窗口函数和`last()`和`first()`来实现。为了确保我们不会用另一个缺失值填充缺失值，可以使用`ignorenulls=True`参数。我们还需要确保设置了正确的窗口范围。对于向前填充，我们将窗口限制为负无穷大和现在之间的值(我们只查看过去，不查看未来)，对于向后填充，我们将窗口限制为现在和无穷大之间的值(我们只查看未来，不查看过去)。下面的代码展示了如何实现这一点。

请注意，如果我们想使用插值而不是向前或向后填充，我们需要知道前一个现有读取值和下一个现有读取值之间的时间差。因此，我们需要保留 readtime _ existent 列。

```
from pyspark.sql import Window
import syswindow_ff = Window.partitionBy('house')\
               .orderBy('readtime')\
               .rowsBetween(-sys.maxsize, 0)

window_bf = Window.partitionBy('house')\
               .orderBy('readtime')\
               .rowsBetween(0, sys.maxsize)

# create series containing the filled values
read_last = func.last(df_all_dates['readvalue'],  
                      ignorenulls=True)\
                .over(window_ff)
readtime_last = func.last(df_all_dates['readtime_existent'],
                          ignorenulls=True)\
                    .over(window_ff)read_next = func.first(df_all_dates['readvalue'],
                       ignorenulls=True)\
                .over(window_bf)
readtime_next = func.first(df_all_dates['readtime_existent'],
                           ignorenulls=True)\
                    .over(window_bf)# add columns to the dataframe
df_filled = df_all_dates.withColumn('readvalue_ff', read_last)\
                        .withColumn('readtime_ff', readtime_last)\
                        .withColumn('readvalue_bf', read_next)\
                        .withColumn('readtime_bf', readtime_next)
```

## 插入文字

在最后一步中，我们使用向前填充和向后填充的数据，通过一个简单的样条来插值读取日期时间和读取值。这也可以使用用户定义的函数来完成。

这给我们留下了一个包含所有插值方法的单一数据帧。它的结构是这样的:

Interpolated read data

# 形象化

最后，我们可以将结果可视化，以观察插值技术之间的差异。不透明点显示插值。

![](img/1a3d15bb7a4505ab48589580c83868d0.png)

Original data (dark) and interpolated data (light), interpolated using (top) forward filling, (middle) backward filling and (bottom) interpolation.

我们可以清楚地看到，在上面的图中，间隙已被上一个已知值填充，在中间的图中，间隙已被下一个值填充，而在下面的图中，差异已被插值。

# 总结和结论

在这篇文章中，我们看到了如何使用 PySpark 来执行时间序列数据的端到端插值。我们已经演示了如何使用重采样时间序列数据，以及如何将`Window`函数与`first()`和`last()`函数结合使用来填充生成的缺失值。然后，我们看到了如何使用用户定义的函数来执行简单的样条插值。

希望这篇帖子有助于填补 PySpark 中关于端到端时间序列插值的文献空白。

*原发布于*[*https://walken ho . github . io*](https://walkenho.github.io/interpolating-time-series-p1-pandas/)*。*