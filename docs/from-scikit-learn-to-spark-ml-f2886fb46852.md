# 从 sci kit-学会火花 ML

> 原文：<https://towardsdatascience.com/from-scikit-learn-to-spark-ml-f2886fb46852?source=collection_archive---------8----------------------->

## 从 Python 到 Scala 的机器学习项目

![](img/774f06bfe7492059af4cbae293715e69.png)

在[之前的一篇文章](/data-cleaning-and-feature-engineering-in-python-b4d448366022)中，我展示了如何获取房屋销售的原始数据集，并在 Python 中对熊猫应用特征工程技术。这使我们能够使用 scikit-learn 机器学习模型来产生和改进对房屋销售价格的预测。

但是，当你想把这种项目投入生产，而不是 10，000 个数据点，也许有几十或几百千兆字节的数据来训练时，会发生什么呢？在这种情况下，脱离 Python 和 scikit 是值得的——学习可以处理大数据的框架。

## 输入 Scala 和 Spark

Scala 是一种基于 Java 虚拟机(JVM)的编程语言，使用函数式编程技术。Scala 中有无数非常复杂的特性可以学习，但是开始使用基本的 Scala 并不比用 Java 甚至 Python 编写代码难多少。

另一方面，Spark 是一个基于 Hadoop 技术的框架，它提供了比传统 Hadoop 更多的灵活性和可用性。它可能是管理和分析大型数据集(也称为大数据)的最佳工具。Spark ML 框架允许开发人员在构建机器学习模型时使用 Spark 进行大规模数据处理。

为什么不首先在 Spark ML 中完成所有的数据探索和模型训练呢？当然可以，但事实是，Python 更容易进行开放式探索，尤其是当您在 Jupyter 笔记本上工作时。但是话虽如此，Scala 和 Spark 并不需要比 Python 复杂太多，因为 pandas 和 Spark 都使用数据帧结构进行数据存储和操作。我们的目标是展示这种简单性，而不是纠结于困难。

这篇文章不会深入探讨 Scala 的复杂性，它假设你对之前的项目有所了解。我们将重现 Python 示例的结果，但我们不会重复我们这样做的所有原因，因为它们在前面已经解释过了。

首先，我们将从加载数据集开始，与之前完全相同的 CSV 文件。

```
val data_key = “housing_data_raw.csv”val df = spark.read
.format(“csv”)
.option(“header”, “true”)
.option(“inferSchema”, “true”)
.load(s”./$data_key”)
```

接下来，我们将删除我们在上一篇文章中发现的异常值。

```
import org.apache.spark.sql.DataFrame
import org.apache.spark.sql.functions._def drop_outliers(data: DataFrame) = {
    val drop = List(1618, 3405,10652, 954, 11136, 5103, 916, 10967, 7383, 1465, 8967, 8300, 4997)

    data.filter(not($"_c0".isin(drop:_*)))
}val housing = drop_outliers(df)
```

现在，我们将`lastsolddate`字段从文本转换为数值，回归模型可以对其进行训练。

```
val housing_dateint = housing.withColumn("lastsolddateint", unix_timestamp($"lastsolddate","MM/dd/yy"))
```

我们还想删除一些列，反正现在我们不想在这些列上进行训练。

```
def drop_geog(data: DataFrame, keep: List[String] = List()) = {
    val removeList = List("info","address","z_address","longitude","latitude","neighborhood",
                          "lastsolddate","zipcode","zpid","usecode", "zestimate","zindexvalue")
    .filter(!keep.contains(_))

    data.drop(removeList: _*)
}val housing_dropgeo = drop_geog(housing_dateint)
```

我们通过一个函数来执行这个操作，以防我们以后想要再次运行相同的代码，我们将会这样做。还要注意，该函数有一个`keep`参数，以防我们想要实际保留我们要删除的这些值中的一个。我们稍后会讲到。

注意，除了两个主要的例外，这个语法的大部分非常接近 Python。首先，Scala 是一种类型化语言。变量实际上不需要显式类型化，但是函数参数需要。第二，您将看到一行类似于:

```
data.drop(removeList: _*)
```

这意味着从数据 DataFrame 中删除`removeList`中的所有内容。`_*`是一个通配符类型定义，只是一些必要的语法，告诉 Scala 编译器如何完成工作。

## 在 Spark ML 中使用 VectorAssembler

现在，我们希望将数据分为训练集和测试集。

```
import org.apache.spark.ml.feature.VectorAssemblerdef train_test_split(data: DataFrame) = {

    val assembler = new VectorAssembler().
       setInputCols(data.drop("lastsoldprice").columns).
       setOutputCol("features")

    val Array(train, test) = data.randomSplit(Array(0.8, 0.2), seed = 30) (assembler.transform(train), assembler.transform(test))
}val (train, test) = train_test_split(housing_dropgeo)
```

这有点棘手，需要一些解释。

在 scitkit-learn 中，你可以获取整个熊猫数据帧，并将其发送给机器学习算法进行训练。Spark ML 也有一个数据帧结构，但模型训练总体上有点挑剔。您必须通过提取每行值并将它们打包到一个向量中，将您想要训练的每一列中的所有要素打包到一个列中。这意味着 Spark ML 只训练一列数据，而这恰好是一个实际包含多列数据的数据结构。一旦你有了代码(见上)，设置起来并不复杂，但这是你必须意识到的一个额外的步骤。

使用 VectorAssembler 创建这些向量列之一。我们创建 VectorAssembler，表示我们想要使用所有的特征列(除了我们的标签/目标列，`lastsoldprice`)，然后给新的向量列命名，通常是`features`。然后，我们使用这个新的组装器来转换两个数据帧，即测试和训练数据集，然后将每个转换后的数据帧作为一个元组返回。

现在，我们可以做一些机器学习。先说线性回归。

```
import org.apache.spark.ml.regression.LinearRegression
import org.apache.spark.ml.evaluation.RegressionEvaluatorval lr = new LinearRegression()
    .setLabelCol("lastsoldprice")
    .setFeaturesCol("features")val lrModel = lr.fit(train)
val predictions = lrModel.transform(test)val rmse = new RegressionEvaluator()
  .setLabelCol("lastsoldprice")
  .setPredictionCol("prediction")
  .setMetricName("rmse")val r2 = new RegressionEvaluator()
  .setLabelCol("lastsoldprice")
  .setPredictionCol("prediction")
  .setMetricName("r2")println("Root Mean Squared Error (RMSE) on test data = " + rmse.evaluate(predictions))
println("R^2 on test data = " + r2.evaluate(predictions))
```

## 我们的第一个预测

![](img/ec041b9b1847659a1bfd25ec299e3509.png)

这应该是不言自明的。下面是上面代码的输出。

```
Root Mean Squared Error (RMSE) on test data = 857356.2890199891
R^2 on test data = 0.31933500943383086
```

这不太好，但这不是重点。这里真正的优势是我们现在有了一种格式的数据，我们*可以*使用 Spark ML，所以从这里开始尝试其他算法是很容易的一步。但是首先，我们将在一个函数中重新创建上述内容，我们可以使用不同的算法调用这个函数。在此过程中，我们将添加一个交叉验证选项，以便我们可以测试多个不同的超参数，并选择最佳结果模型。请注意，这一步并不是绝对必要的——前面的代码可以很容易地被重构以使用不同的算法——但这是一个更好的实践。

```
import org.apache.spark.ml.Predictor
import org.apache.spark.ml.PredictionModel
import org.apache.spark.ml.linalg.Vector
import org.apache.spark.ml.tuning.{CrossValidator, ParamGridBuilder}
import org.apache.spark.ml.param.ParamMapdef train_eval[R <: Predictor[Vector, R, M],
               M <: PredictionModel[Vector, M]](
    predictor: Predictor[Vector, R, M],
    paramMap: Array[ParamMap],
    train: DataFrame, 
    test: DataFrame) = {val cv = new CrossValidator()
      .setEstimator( predictor    
                    .setLabelCol("lastsoldprice")
                    .setFeaturesCol("features"))
      .setEvaluator(new RegressionEvaluator()
          .setLabelCol("lastsoldprice")
          .setPredictionCol("prediction")
          .setMetricName("rmse"))
      .setEstimatorParamMaps(paramMap)
      .setNumFolds(5)
      .setParallelism(2) val cvModel = cv.fit(train)
    val predictions = cvModel.transform(test)

    println("Root Mean Squared Error (RMSE) on test data = " + rmse.evaluate(predictions))
    println("R^2 on test data = " + r2.evaluate(predictions)) val bestModel = cvModel.bestModel

    println(bestModel.extractParamMap)

    bestModel
}
```

大部分内容应该是不言自明的— *除了*函数定义，它实际上是难以理解的。这是这一部分:

```
def train_eval[R <: Predictor[Vector, R, M],
               M <: PredictionModel[Vector, M]](
    predictor: Predictor[Vector, R, M],
    paramMap: Array[ParamMap],
    train: DataFrame, 
    test: DataFrame) = {
```

不幸的是，Spark ML 似乎没有一个通用的“模型”类型，例如，我们可以传入一个 RegressionModel 对象，然后我们的函数可以接受这个“模型”对象并调用`.fit`方法。这将使我们避免这种混乱。相反，我们必须提出这个复杂的定义，这样我们就可以创建一个接受通用“模型”类型的方法，因为该类型也依赖于输入格式和隐式赋值器类型。想出这种定义是相当棘手的——在环顾四周和反复试验之后，我发现直接从 Spark ML 源代码中提取它就可以了。

好消息是，一旦你写了这样的东西，呼叫者不需要真正了解或理解它(老实说，他们不太可能会了解)。相反，这种复杂类型定义允许调用者编写真正干净的代码，如下所示:

```
val lr = new LinearRegression()val lrParamMap = new ParamGridBuilder()
    .addGrid(lr.regParam, Array(10, 1, 0.1, 0.01, 0.001))
    .addGrid(lr.elasticNetParam, Array(0.0, 0.5, 1.0))
    .addGrid(lr.maxIter, Array(10000, 250000))
    .build()train_eval(lr, lrParamMap, train, test)
```

现在我们回到一些简单的东西:一个算法(线性回归)和一个交叉验证的参数网格。然后函数调用变得像我们在 Python 中看到的一样简单。这实际上是相同的代码。注意，我甚至为`train_eval`和其他变量保留了嵌入式下划线语法。在 Scala 和 Java 中，我们通常会使用 camel case，即`trainEval`，这也是我在编写产品代码时会做的事情，但是由于本教程最初是用 Python 编写的，所以为了比较，保持一些一致性是值得的。

我们现在来看看其他一些算法，看看它们是如何工作的。注意 Lasso 和 Ridge 没有显式类，因为`elasticNetParam`分别设置为 1 和 0 时指定这两种算法。

决策树:

```
import org.apache.spark.ml.regression.DecisionTreeRegressorval decisionTree = new DecisionTreeRegressor()
val dtParamMap = new ParamGridBuilder().build()
train_eval(decisionTree, dtParamMap, train, test)
```

结果:

```
Root Mean Squared Error (RMSE) on test data = 759685.8395738212
R^2 on test data = 0.46558480196241925
```

随机森林:

```
import org.apache.spark.ml.regression.RandomForestRegressorval randomForest = new RandomForestRegressor()val rfParamMap = new ParamGridBuilder()
    .addGrid(randomForest.maxBins, Array(4, 16, 32, 64))
    .addGrid(randomForest.numTrees, Array(1, 10, 100))
    .addGrid(randomForest.maxDepth, Array(2, 5, 10))
    .build()train_eval(randomForest, rfParamMap, train, test)
```

结果:

```
Root Mean Squared Error (RMSE) on test data = 647133.830611256
R^2 on test data = 0.6122079099308858
```

梯度增强:

```
import org.apache.spark.ml.regression.GBTRegressorval gradientBoost = new GBTRegressor()val gbParamMap = new ParamGridBuilder()
    .addGrid(randomForest.maxBins, Array(16, 32))
    .addGrid(randomForest.numTrees, Array(5, 10, 100))
    .addGrid(randomForest.maxDepth, Array(5, 10))
    .addGrid(randomForest.minInfoGain, Array(0.0, 0.1, 0.5))
    .build()train_eval(gradientBoost, gbParamMap, train, test)
```

结果:

```
Root Mean Squared Error (RMSE) on test data = 703037.6456894034
R^2 on test data = 0.5423137139558296
```

虽然确切的结果不同(特别是对于线性回归)，但我们也看到了随机森林和梯度增强比更简单的算法工作得更好的类似趋势。同样，我们可以通过使用更好的数据来改进我们的结果。我们通过重新整合`neighborhood`数据来做到这一点，但采用算法可以使用的数字格式。我们通过使用 one-hot 编码将像`Mission`和`South Beach`这样的类别更改为 1 和 0 的列来实现这一点。

## 一键编码

首先，我们用这些数据重建数据框架:

```
val housing_neighborhood = drop_geog(housing_dateint, List("neighborhood"))
```

![](img/66c4d58290a11278af432d3bf4b83255.png)

然后，我们使用一个`Pipeline`通过一键编码来转换数据:

```
import org.apache.spark.ml.Pipeline
import org.apache.spark.ml.feature.OneHotEncoderEstimator
import org.apache.spark.ml.feature.StringIndexerval indexer = new StringIndexer().setInputCol("neighborhood").setOutputCol("neighborhoodIndex")val encoder = new OneHotEncoderEstimator()
  .setInputCols(Array(indexer.getOutputCol))
  .setOutputCols(Array("neighborhoodVector"))val pipeline = new Pipeline().setStages(Array(indexer, encoder))val housingEncoded = pipeline.fit(housing_neighborhood).transform(housing_neighborhood)
.drop("neighborhoodIndex")
.drop("neighborhood")
```

首先，我们必须使用一个`StringIndexer`，然后我们可以使用`OneHotEncoderEstimator`得到我们想要的结果。因为这是两个步骤，我们可以使用阶段的`Pipeline`将它们放在一起。管道可以用于许多多阶段的清理和修改操作，特别是当你需要一遍又一遍地做这些相同的阶段时，例如需要重新训练的不断发展的数据集。这在这里并不是真正必要的，但它值得展示，因为它非常有用。

更新数据后，我们可以像以前一样重新进行实验。我们只是传递我们的新数据(在创建训练和测试集之后):

```
val (train_neighborhood, test_neighborhood) = train_test_split(housingEncoded)
```

例如，对于线性回归，我们使用这些新变量调用完全相同的函数:

```
train_eval(lr, lrParamMap, train_neighborhood, test_neighborhood)
```

结果:

```
Root Mean Squared Error (RMSE) on test data = 754869.9632285038
R^2 on test data = 0.4723389619596349
```

这仍然不是很好，但比以前好多了。现在，让我们看看在传递转换后的数据帧后其他算法的结果。

决策树:

```
Root Mean Squared Error (RMSE) on test data = 722171.2606321493
R^2 on test data = 0.5170622654844328
```

随机森林:

```
Root Mean Squared Error (RMSE) on test data = 581188.983582857
R^2 on test data = 0.6872153115815951
```

梯度增强:

```
Root Mean Squared Error (RMSE) on test data = 636055.9695573623
R^2 on test data = 0.6253709908240936
```

在每种情况下，我们都看到了相当大的改进，现在我们有了一个模型和一个训练管道，可以用更大的数据集投入生产。