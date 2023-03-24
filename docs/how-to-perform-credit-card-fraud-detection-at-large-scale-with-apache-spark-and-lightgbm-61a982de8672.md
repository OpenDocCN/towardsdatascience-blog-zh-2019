# 如何用 Apache Spark 和 LightGBM 大规模构建机器学习模型？

> 原文：<https://towardsdatascience.com/how-to-perform-credit-card-fraud-detection-at-large-scale-with-apache-spark-and-lightgbm-61a982de8672?source=collection_archive---------22----------------------->

![](img/0bb75401ab1280e926a82f8f9436bcdc.png)

Image credit iStockPhoto.com

虽然在 Jupyter 笔记本电脑上用相对较小的静态数据集训练机器学习模型很容易，但当它部署在金融机构内部的真实环境中时，会出现挑战，因为海量的交易数据驻留在 Hadoop 或数据湖中。在本文中，我将向您展示如何利用 Apache Spark 来实现使用 LightGBM 在 Spark 环境中构建和训练模型的核心部分，light GBM 被认为是在数据科学社区中广泛使用的梯度推进算法的快速实现。

Apache Spark 是一个内存集群运行时环境，存储和处理数据的速度比从 SSD 或硬盘访问数据快几千倍。假设我们要处理的数据集位于 Hadoop、S3、数据库或本地文件中，那么第一个任务就是在 Spark cluster 中以 Spark dataframe 的形式访问它。

欺诈检测是每个发行信用卡的金融机构目前都在进行的一项活动，该活动使用大规模数据集，这些数据集是每天在其平台上发生的数百万笔交易的结果。在我们的例子中，我将 ULB 机器学习小组提供的 CSV 文件格式的数据集作为 Spark dataframe 导入到由 Databricks community edition 托管的具有 6gb RAM 的 Spark 集群上。[这是关于向数据块添加数据的指南](https://docs.databricks.com/data/tables.html)。

```
*# File location and type*
file_location = "/FileStore/tables/creditcard.csv"
file_type = "csv"

*# CSV options*
infer_schema = "true"
first_row_is_header = "true"
delimiter = ","

df = spark.read.format(file_type) \
  .option("inferSchema", infer_schema) \
  .option("header", first_row_is_header) \
  .option("sep", delimiter) \
  .load(file_location)
```

该数据集包含约 300，000 行，由 31 个与欧洲信用卡持卡人交易相关的变量组成，其中 28 个是通过对一些未披露的原始参数进行主成分分析而得出的数值变量。剩下的三个变量是交易量、相对于第一次交易的交易时间(以秒计)以及指示交易是真实还是欺诈的交易标签。当我们谈论大规模时，数据集可能看起来很小，但只要选择使用 Apache Spark 来训练模型，即使是小数据集，也可以放心，相同的代码将无缝扩展到任意大的数据集，这些数据集可能托管在 HDFS、S3、NoSQL 数据库或数据湖中，只要它可以以 Spark 数据帧的形式访问模型。

就标记为欺诈和真实的样本数量而言，数据集严重失衡。与数据集中的总行数相比，它只有少量的 492 个样本与被识别为欺诈的交易相关。你可以在 Kaggle 上访问这个[数据集。](https://www.kaggle.com/mlg-ulb/creditcardfraud)

下一步是从这个数据帧中选择我们希望在训练模型时用作输入变量和目标变量的列。在构建生产级 ML 模型时，执行数据转换步骤通常是一种很好的做法，例如使用标签编码器或一个热编码器作为管道阶段将分类列转换为数字列，然后可以用于转换训练、验证或测试数据，而不会将测试用例的知识泄露给训练数据(同时执行标准化缩放等转换)和代码的高维护性。

```
feature_cols = ["V" + str(i) **for** i **in** range(1,29)] + ["Amount"]
assembler = VectorAssembler(inputCols=feature_cols, outputCol="features")
stages = [assembler]
```

接下来，我们可以向管道添加 LightGBMClassifier 实现的实例。LightGBMClassifier 类在由 Microsoft Azure 团队作为开源项目维护的 MMLSpark 库中可用。按照本页中建议的[步骤，可以将它作为库包含到集群中。我通过从 Maven option 中搜索名为 mmlspark 的包添加了这个库，选择了它的 0.17 版本，并将其安装在集群中。之后用 Python 从 mmlspark 导入 LightGBMClassifier 就奏效了。](https://mmlspark.blob.core.windows.net/website/index.html#install)

在使用 LightGBM 时，使用超参数的最佳值对其进行调优是非常重要的，例如叶子的数量、最大深度、迭代次数等。如果用不同的超参数值训练，在相同数据上训练的模型的性能可以有很大不同。

通常，这需要设置交叉验证实验，并使用一些超参数值空间探索策略，如全网格搜索、随机搜索、贝叶斯优化或树形结构 Parzen Estimators 算法，以找出超参数的最佳值，通过最小化一些预定义的误差度量(如二进制误差)或最大化一些分数(如 ROC 或 F1 测量下的面积)来最大化模型在验证数据集上的性能。

这种探索可能会持续很长时间，这取决于我们想要探索的参数值空间的大小以及我们对模型的预期性能水平。在这种情况下，我决定调优 LightGBM 模型的 7 个超参数。这些参数中有许多是实值，这意味着要探索的参数空间是无限深的。

为了获得不间断的计算时间，通过评估超参数值的不同组合来探索这个超参数空间，同时使用一些优化策略在相对较少的评估轮次中找出超参数的接近最优的值，我使用了具有相同数据集的 Python 中 Kaggle 的执行环境内的 Hyperopt 库。在对超参数值的组合进行 200 次评估后，我使用树形结构的 Parzen 估计算法来找出超参数的最优值。[这是我在 Spark 内部训练这个模型时确定要使用的超参数值的笔记本](https://www.kaggle.com/patelatharva/credit-card-transaction-fraud-detection)。

在 Databricks 的社区版中，我可以不间断地访问集群 2 个小时。如果您可以访问 Spark cluster 更长的不间断时间，您还可以使用 Spark 中的 ParamGridBuilder 和 CrossValidator 类来探索和评估超参数值，如下面的示例代码所示。如果您想在这里以更有效的方式进行整个超参数调整，而不是执行整个网格搜索，您可以参考由 [Hyperopt](http://hyperopt.github.io/hyperopt/scaleout/spark/) 和 [DataBricks](https://docs.databricks.com/applications/machine-learning/automl/hyperopt/hyperopt-spark-mlflow-integration.html) 提供的指南，这些指南展示了如何使用 Spark cluster 和 Hyperopt。

在这里，我使用了我从 Kaggle 的 python 环境中进行的模型调整中获得的大多数最佳超参数值，以及 lambda L2 正则化参数的调整值作为 Spark 内部使用 CrossValidator 执行网格搜索的示例。

请注意，我在 LightGBMClassifier 中设置了标志 isUnbalance=True，这样它就可以处理我们前面讨论的数据集中的不平衡。

```
best_params = {        
	'bagging_fraction': 0.8,
     	'bagging_freq': 1,
     	'eval_metric': 'binary_error',
     	'feature_fraction': 0.944714847210862,
	'lambda_l1': 1.0,
     	'lambda_l2': 45.0,
     	'learning_rate': 0.1,
     	'loss_function': 'binary_error',
     	'max_bin': 60,
     	'max_depth': 58,
     	'metric': 'binary_error',
     	'num_iterations': 379,
     	'num_leaves': 850,
	'objective': 'binary',
     	'random_state': 7,
     	'verbose': **None** }lgb = LightGBMClassifier(
 	learningRate=0.1,
 	earlyStoppingRound=100,
       	featuresCol='features',
        labelCol='label',
        isUnbalance=**True**,
  	baggingFraction=best_params["bagging_fraction"],
	baggingFreq=1,
	featureFraction=best_params["feature_fraction"],
	lambdaL1=best_params["lambda_l1"],
	# lambdaL2=best_params["lambda_l2"],
	maxBin=best_params["max_bin"],
	maxDepth=best_params["max_depth"],
	numIterations=best_params["num_iterations"],
	numLeaves=best_params["num_leaves"],
	objective="binary",
	baggingSeed=7                  
)paramGrid = ParamGridBuilder().addGrid(
  lgb.lambdaL2, list(np.arange(1.0, 101.0, 10.0))
).build()evaluator = BinaryClassificationEvaluator(labelCol="label",metricName="areaUnderROC")crossValidator = CrossValidator(estimator=lgb,
                          estimatorParamMaps=paramGrid, 
                          evaluator=evaluator, 
                          numFolds=2)   
stages += [crossValidator]
pipelineModel = Pipeline(stages=stages)
```

下一步是将数据集分成训练数据集和测试数据集。我们将使用训练数据集来拟合我们刚刚创建的管道，该管道由特征组装和模型训练等步骤组成。然后，我们将使用此管道来转换测试数据集，以生成预测。

```
train, test = df.randomSplit([0.8, 0.2], seed=7)
model = pipelineModel.fit(train)
preds = model.transform(test)
```

一旦我们有了来自测试数据集的预测，我们就可以用它们来衡量我们的模型的性能。Spark 提供了 BinaryClassificationEvaluator，可以计算 ROC 度量下的面积。

为了计算其他相关指标，如精确度、召回率和 F1 分数，我们可以利用测试数据集的预测标签和实际标签。

```
binaryEvaluator = BinaryClassificationEvaluator(labelCol="label")print ("Test Area Under ROC: " + str(binaryEvaluator.evaluate(preds, {binaryEvaluator.metricName: "areaUnderROC"})))#True positives
tp = preds[(preds.label == 1) & (preds.prediction == 1)].count() #True negatives
tn = preds[(preds.label == 0) & (preds.prediction == 0)].count()#False positives
fp = preds[(preds.label == 0) & (preds.prediction == 1)].count()#Falsel negatives
fn = preds[(preds.label == 1) & (preds.prediction == 0)].count()print ("True Positives:", tp)
print ("True Negatives:", tn)
print ("False Positives:", fp)
print ("False Negatives:", fn)
print ("Total", preds.count()) r = float(tp)/(tp + fn) print ("recall", r) p = float(tp) / (tp + fp)print ("precision", p)f1 = 2 * p * r /(p + r) print ("f1", f1)
```

在这种情况下，其 AUC-ROC 得分为 0.93，F1 得分为 0.70。在 Kaggle notebook 中，我在使用它进行训练之前还使用了 [SMOTE](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.over_sampling.SMOTE.html) 来平衡数据集，它获得了 0.98 的 AUC-ROC 分数和接近 0.80 的 F1 分数。我在 Kaggle 环境中对超参数值的组合进行了 200 次评估。如果我们想进一步提高模型的性能，那么接下来要做的显然是在 Spark 中实现 SMOTE 来平衡训练数据集，然后再用它来拟合管道。我们还可以通过对参数值组合进行大量评估来更深入地探索超参数空间。

除了在 Spark 上训练 LightGBM 等高性能模型之外，数据科学团队通常面临的另一个重大挑战是管理准备数据、选择模型、训练、调整、保存最佳参数值、部署训练好的模型以及以 API 形式访问输出预测的生命周期。 [MLFlow](https://mlflow.org/) 是一个试图解决这个问题的开源解决方案。建议有兴趣的读者多了解一下。我也可能会写一篇文章，介绍如何将它集成到建立 ML 模型构建、培训和部署项目的工作流中，就像我在 Databricks 平台上使用 Spark 演示的那个项目一样。

你可以在 [GitHub repo](https://github.com/patelatharva/Credit_card_fraud_detection_ULB_with_Apache_Spark_on_Databricks) 上托管的这个笔记本中参考我在数据块[上执行的确切步骤。这里](https://nbviewer.jupyter.org/github/patelatharva/Credit_card_fraud_detection_ULB_with_Apache_Spark_on_Databricks/blob/master/credit%20card%20fraud%20detection.ipynb)[是另一个笔记本，](https://www.kaggle.com/patelatharva/credit-card-transaction-fraud-detection)我在 Kaggle 上对这个数据集进行了参数调优实验。

我希望这篇文章对你使用 LightGBMClassifier 在 Apache Spark 上实现机器学习有所启发。如果你在自己尝试的时候有什么问题，欢迎在评论中提问。