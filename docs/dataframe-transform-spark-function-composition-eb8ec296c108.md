# DataFrame.transform —火花函数合成

> 原文：<https://towardsdatascience.com/dataframe-transform-spark-function-composition-eb8ec296c108?source=collection_archive---------9----------------------->

![](img/84746e785a19eac2c11a085be7a079d7.png)

Photo by 嘉淇 徐 from Pexels

## 让您的 Spark 代码更具功能性，可读性更好

## 如何从转换方法中返回容易组合的函数

随着组织迁移到 Spark 上并在 Spark 上创建新的数据处理逻辑，最终的软件会变得非常大，大到需要考虑我们应用于其他软件项目的所有**可维护性**。

虽然有许多关于编写性能逻辑的全面而有价值的资源，但关于**结构化项目**的资源就不一样了，这些资源创建了可重用的 Spark 代码，并最终降低了长期维护这些项目的成本。

在这篇文章中，我们来具体看看 Spark Scala DataFrame API，以及如何利用 [**数据集[T]。转换**](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.Dataset@transform[U](t:org.apache.spark.sql.Dataset[T]=%3Eorg.apache.spark.sql.Dataset[U]):org.apache.spark.sql.Dataset[U]) 函数来编写可组合代码。

注意:[data frame 是 Dataset[Row]](https://spark.apache.org/docs/latest/api/scala/index.html#org.apache.spark.sql.package@DataFrame=org.apache.spark.sql.Dataset[org.apache.spark.sql.Row]) 的类型别名。

# 这个例子

有一些特定金额的交易，包含描述付款人和受益人的“详细信息”列:

Note that this DataFrame could be a Dataset[Transaction], but it’s not useful to the examples

# 没有。改变

让我们创建两个函数来处理事务:

*   **sumAmounts** :对一列或多列的合计值求和
*   **extractPayerBeneficiary**:将付款人和受益人从一列分离成两个新列

使用这些方法来回答以下问题:“哪些受益人在哪些天的总金额超过 25？”

这是一个简单的例子，但是读起来不太好。将其与`Dataset`功能的典型用法进行比较:

```
df.select(...).filter(...).withColumn(...)...
```

我们将一些逻辑分解到方法中，这有助于我们分别对每一部分进行推理，但是代码的可读性变得更差了。

# 使用。改变

transform 函数是 Dataset 类的一个方法，它的目的是添加一个“*简洁的语法来链接定制的转换*

```
*def* transform[U](t: Dataset[T] => Dataset[U]): Dataset[U] = t(*this*)
```

它采用了一个函数，从`Dataset[T]`， *T(数据集中的行类型*)到`Dataset[U]`， *U(结果数据集中的行类型)——*U 可以与 T 相同

一个函数`DataFrame => DataFrame`符合这个签名——如果我们解开类型别名，我们得到`Dataset[Row] => Dataset[Row]`,其中 T 和 U 都是`Row`。

使用您之前定义的方法并简单地切换到使用`.transform`是一个很好的起点:

# 更进一步

`sumAmounts`和`extractPayerBeneficiary`方法不太适合`.transform`。这是因为这些方法返回的是一个**数据帧**，而不是一个函数`DataFrame => DataFrame`，所以为了返回一个可以在`.transform`中使用的函数，你需要不断地使用下划线来代替数据帧参数。

您可以重写这些方法来返回签名的函数:`DataFrame => DataFrame`，以精确匹配`.transform` 参数类型:

Only the signature had to be changed and a “df =>” added!

现在，您不再需要“下划线”,可以用不同的方式组合这些功能:

您所有的自定义转换现在都返回`DataFrame => DataFrame`，因此您可以使用类型别名来更好地描述返回值:

> `type Transform = DataFrame => DataFrame`

例如`def sumAmounts(by: Column*): Transform`

# 摘要

*   自定义转换方法可以重新排列以返回类型为`DataFrame => DataFrame`的函数。
*   返回函数使得组合转换和使用它们变得更加容易。
*   类型别名可用于显式定义“转换”。

你可以在这个[要点](https://gist.github.com/dmateusp/e738a9647ffe2fa432457460d1c0c445)里找到我的 **build.sbt** 和上面的代码