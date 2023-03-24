# 在 Spark 上部署 Python 模型的不同方式

> 原文：<https://towardsdatascience.com/a-different-way-to-deploy-a-python-model-over-spark-2da4d625f73e?source=collection_archive---------15----------------------->

## 将预测方法与 Python 类的其余部分分开，然后在 Scala 中实现

![](img/8c882a5c83ddf87d8728c03497900185.png)

Instead of using the whole thing, just take the pieces you need.

前阵子，我写了一篇关于[如何在 Spark](/deploy-a-python-model-more-efficiently-over-spark-497fc03e0a8d) 上部署 Python 模型的帖子。方法大致如下:

1.  根据全部数据的样本在 Python 中训练模型。
2.  将测试数据收集到任意大小的组中——大约 500，000 条记录对我来说似乎很好。
3.  广播训练好的模型，然后使用用户定义的函数对每组记录整体调用模型的`predict`方法，而不是对每条单独的记录(如果对未分组的数据帧调用 UDF，Spark 就会这样做)。

该方法利用了支持 scikit-learn 的支持 numpy 的优化，并减少了您必须经历序列化和反序列化模型对象的昂贵过程的次数。

我最近采用了一种不同的方式在 Spark 上部署 Python 模型，不需要对大量数据进行分组和分解。我仍然在 Python 中对全部数据的样本进行模型训练，但是我将调用 predict 方法所需的一切存储在一个 JSON 文件中，然后该文件可以被调用到一个可以实现 predict 方法的 Scala 函数中。

例如，以 sci kit-learn[RandomForestRegressor](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestRegressor.html)为例。`predict`方法是一堆个体决策树的`predict`方法结果的平均值，但是方法本身的实现根本不使用树结构。它将所有内容编码成一系列列表。

以下代码创建了一个纯 Python 函数，该函数将精确再现经过训练的 RandomForestRegressor 的预测:

看一下`tree_template`字符串。我们从五个列表开始——每个列表都有树中的节点。`features`列表中的唯一值与模型训练所依据的特征一样多。我们从列表的第一个值开始—索引 0。如果索引 0 处的值是，比如说，3，那么我们取出模型被训练所基于的第三个特征的值。然后我们从`thresholds`列表中取出索引零值。如果所选特征的值小于或等于相应阈值的值，那么我们查看`children_left`列表的零索引值。否则，我们查看`children_right`列表的零索引值。不管怎样，这个值就是我们的新索引，然后我们重新开始这个过程。我们一直这样做，直到子列表中的值成为“您已经到达了树的末尾”的占位符。在 scikit-learn 中，这个占位符的默认值是-2。此时，无论您当前在哪个索引上，您都可以从`values`列表中查找该索引的值。那是你的预测。

所以，是的，这是一个很大的数据量——拥有几十个特征的决策树回归器通常有大约 100，000 个节点。但是导航树以获得预测的逻辑非常简单。因此，您所要做的就是创建一个包含每棵树的列表的函数，以及从一个索引跳到另一个索引的逻辑。然后取一组特征，从每棵树上得到预测值，并取平均值。那是你的随机森林。

很容易将所有这些信息转储到 JSON。下面的函数就是这样做的。它只需要一个经过训练的 RandomForestRegressor 对象、按照训练中使用的顺序排列的特性列表，以及一个将 JSON 文件转储到的文件路径。

单棵树的输出如下所示:

下一段代码来自我的同事 Sam Hendley，他已经忘记了比我所知道的更多的 Scala 知识。它从 JSON 文件中读入树信息，实现每棵树的预测逻辑，然后对整个森林进行平均。

在 Scala 中实现预测避免了对函数的 Python 表示进行序列化和反序列化的过程——一切都可以直接在 JVM 上完成。随机森林是这里最复杂的用例之一。在任何产生系数的模型中，将预测函数转换成 JSON 和 Scala 甚至更容易。