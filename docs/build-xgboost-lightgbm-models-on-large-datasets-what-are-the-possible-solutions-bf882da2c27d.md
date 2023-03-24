# 在大型数据集上构建 XGBoost / LightGBM 模型——可能的解决方案是什么？

> 原文：<https://towardsdatascience.com/build-xgboost-lightgbm-models-on-large-datasets-what-are-the-possible-solutions-bf882da2c27d?source=collection_archive---------4----------------------->

## XGBoost 和 LightGBM 已经在许多表格数据集上被证明是性能最好的 ML 算法。但是当数据庞大时，我们如何使用它们呢？

![](img/ac4bafa7deb0e90e2cdd5dfce3d9da01.png)

Photo by [Sean O.](https://unsplash.com/photos/KMn4VEeEPR8?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/beach?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

XGBoost 和 LightGBM 在最近所有的 kaggle 表格数据竞争中占据了主导地位。只需进入任何竞赛页面(表格数据)并检查内核，你就会看到。在这些比赛中，数据并不“庞大”——好吧，不要告诉我你正在处理的数据是巨大的，如果它可以在你的笔记本电脑上训练的话。对于这些情况，Jupyter notebook 足够用于 XGBoost 和 LightGBM 模型构造。

当数据变得更大，但不是超级大，而你仍然想坚持使用 Jupyter 笔记本，比方说，来构建模型——一种方法是使用一些内存减少技巧(例如，ArjanGroen 的代码:[https://www . ka ggle . com/arjanso/reducing-data frame-memory-size-by-65](https://www.kaggle.com/arjanso/reducing-dataframe-memory-size-by-65))；或者使用云服务，比如在 AWS 上租一台 EC2。例如，r5.24xlarge 实例拥有 768 GiB 内存，成本为 6 美元/小时，我认为它已经可以处理大量数据，您的老板认为它们真的很“大”。

但是如果数据更大呢？

我们需要分布式机器学习工具。据我所知，如果我们想使用可伸缩的 XGBoost 或 LightGBM，我们有以下几种选择:

1 XGBoost 4j on Scala-Spark
2 light GBM on Spark(py Spark/Scala/R)
3 XGBoost with H2O . ai
4 XGBoost on Amazon SageMaker

我想根据我的个人经验指出每种工具的一些问题，如果您想使用它们，我会提供一些资源。如果你也有类似的问题/你是如何解决的，我也很高兴向你学习！请在下面评论。

在我开始使用这些工具之前，有一些事情需要事先了解。

# XGBoost 与 LightGBM

XGBoost 是一种非常快速和准确的 ML 算法，但它现在受到了 LightGBM 的挑战，light GBM 运行得更快(对于某些数据集，根据其基准测试，它快了 10 倍)，具有相当的模型准确性，并且有更多的超参数供用户调整。速度上的关键差异是因为 XGBoost 一次将树节点拆分一层，而 LightGBM 一次只拆分一个节点。

因此，XGBoost 开发人员后来改进了他们的算法，以赶上 LightGBM，允许用户也在逐叶分割模式下运行 XGBoost(grow _ policy = ' loss guide ')。现在，XGBoost 在这一改进下速度快了很多，但根据我在几个数据集上的测试，LightGBM 的速度仍然是 XGB 的 1.3-1.5 倍。(欢迎分享你的测试结果！)

读者可以根据自己的喜好选择任何一个选项。这里再补充一点:XGBoost 有一个 LightGBM 没有的特性——“单调约束”。这将牺牲一些模型精度并增加训练时间，但可以提高模型的可解释性。(参考:[https://xgboost . readthe docs . io/en/latest/tutorials/monotonic . html](https://xgboost.readthedocs.io/en/latest/tutorials/monotonic.html)和【https://github.com/dotnet/machinelearning/issues/1651】T2)

# 在渐变提升树中找到“甜蜜点”

对于随机森林算法，建立的树越多，模型的方差越小。但是在某种程度上，你不能通过添加更多的树来进一步改进这个模型。

XGBoost 和 LightGBM 不是这样工作的。当树的数量增加时，模型精度不断提高，但是在某个点之后，性能开始下降——过度拟合的标志；随着树越建越多，性能越差。

为了找到‘甜蜜点’，你可以做交叉验证或者简单的做训练-验证集分裂，然后利用提前停止时间找到它应该停止训练的地方；或者，你可以用不同数量的树(比如 50、100、200)建立几个模型，然后从中选出最好的一个。

如果你不在乎极端的性能，你可以设置一个更高的学习率，只构建 10-50 棵树(比如说)。它可能有点不合适，但你仍然有一个非常准确的模型，这样你可以节省时间找到最佳的树的数量。这种方法的另一个好处是模型更简单(构建的树更少)。

# 1.Scala-Spark 上的 XGBoost4j

如果读者打算选择这个选项，[https://xgboost . readthedocs . io/en/latest/JVM/xgboost 4j _ spark _ tutorial . html](https://xgboost.readthedocs.io/en/latest/jvm/xgboost4j_spark_tutorial.html)是一个很好的起点。我想在这里指出几个问题(在本文发布时):

1.  XGBoost4j 不支持 Pyspark。
2.  XGBoost4j 不支持逐叶分割模式，因此速度较慢。[https://github.com/dmlc/xgboost/issues/3724](https://github.com/dmlc/xgboost/issues/3724)
3.  因为是在 Spark 上，所以所有缺失值都要进行插补(vector assembler 不允许缺失值)。这可能会降低模型精度。[http://docs . H2O . ai/H2O/latest-stable/H2O-docs/data-science/GBM-FAQ/missing _ values . html](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/data-science/gbm-faq/missing_values.html)
4.  早期停止可能仍然包含 bug。如果你关注他们在 https://github.com/dmlc/xgboost/releases[的最新发布，你会发现他们最近仍在修复这些漏洞。](https://github.com/dmlc/xgboost/releases)

# 2.Spark 上的 light GBM(Scala/Python/R)

基于我个人经验的主要问题:

1.  缺少文档和好的例子。[https://github . com/Azure/mmspark/blob/master/docs/light GBM . MD](https://github.com/Azure/mmlspark/blob/master/docs/lightgbm.md)
2.  所有缺失的值都必须进行估算(类似于 XGBoost4j)
3.  我在 spark cross validator 的“提前停止”参数上也有问题。(为了测试它是否正常工作，选择一个较小的数据集，选择一个非常大的回合数，提前停止= 10，并查看训练模型需要多长时间。训练完成后，将模型精度与使用 Python 构建的模型进行比较。如果过拟合得很差，很可能早期停止根本不起作用。)

一些示例代码(不包括矢量汇编程序):

```
from mmlspark import LightGBMClassifier
from pyspark.ml.evaluation import BinaryClassificationEvaluator
from pyspark.ml.tuning import CrossValidator, ParamGridBuilderlgb_estimator = LightGBMClassifier(learningRate=0.1, 
                                   numIterations=1000,
                                   earlyStoppingRound=10,
                                   labelCol="label")paramGrid = ParamGridBuilder().addGrid(lgb_estimator.numLeaves, [30, 50]).build()eval = BinaryClassificationEvaluator(labelCol="label",metricName="areaUnderROC")crossval = CrossValidator(estimator=lgb_estimator,
                          estimatorParamMaps=paramGrid, 
                          evaluator=eval, 
                          numFolds=3) cvModel  = crossval.fit(train_df[["features", "label"]])
```

# 3.H2O.ai 上的 XGBoost

这是我个人最喜欢的解决方案。该模型可以使用 H2O.ai 构建，集成在 py sparking Water(H2O . ai+py spark)管道中:
[https://www . slide share . net/0x data/productionizing-H2O-models-using-sparking-Water-by-jakub-hava](https://www.slideshare.net/0xdata/productionizing-h2o-models-using-sparkling-water-by-jakub-hava)

很容易建立一个带有交叉验证的优化轮数的模型

```
# binary classificationfeatures = ['A', 'B', 'C']
train['label'] = train['label'].asfactor() # train is an H2O framecv_xgb = H2OXGBoostEstimator(
    ntrees = 1000,
    learn_rate = 0.1,
    max_leaves = 50,
    stopping_rounds = 10,
    stopping_metric = "AUC",
    score_tree_interval = 1,
    tree_method="hist",
    grow_policy="lossguide",
    nfolds=5, 
    seed=0)cv_xgb.train(x = features, y = 'label', training_frame = train)
```

并且 XGBoost 模型可以用`cv_xgb.save_mojo()`保存在 Python 中使用。如果您想以 h2o 格式保存模型，请使用`h2o.save_model()`。

我唯一的抱怨是保存的模型(用`save.mojo`保存的那个)不能和 SHAP 包一起使用来生成 SHAP 特性重要性(但是 XGBoost 特性重要性，`.get_fscore()`，工作正常)。似乎最初的 XGBoost 包有一些问题。
https://github.com/slundberg/shap/issues/464
https://github.com/dmlc/xgboost/issues/4276

(更新:似乎他们刚刚在其最新版本中实现了 SHAP-[https://github . com/h2oai/H2O-3/blob/373 ca 6 B1 BC 7d 194 C6 c 70 e 1070 F2 F6 f 416 f 56 B3 d 0/changes . MD](https://github.com/h2oai/h2o-3/blob/373ca6b1bc7d194c6c70e1070f2f6f416f56b3d0/Changes.md))

# 4.SageMaker 上的 XGBoost

这是 AWS 的一个非常新的解决方案。两个主要特性是使用贝叶斯优化的自动超参数调整，并且该模型可以作为端点部署。在他们的 Github 上可以找到几个例子:[https://Github . com/aw slabs/Amazon-sage maker-examples/tree/master/introduction _ to _ applying _ machine _ learning](https://github.com/awslabs/amazon-sagemaker-examples/tree/master/introduction_to_applying_machine_learning)。以下是我对此的一些担忧:

1.  与其他解决方案相比，参数调整工具对用户(数据科学家)不太友好:
    [https://github . com/aw slats/Amazon-sagemaker-examples/blob/master/hyperparameter _ tuning/xgboost _ direct _ marketing/hpo _ xgboost _ direct _ marketing _ sagemaker _ APIs . ipynb](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/xgboost_direct_marketing/hpo_xgboost_direct_marketing_sagemaker_APIs.ipynb)和
    [https://github . com/aw slats/Amazon-sagemaker-examples/blob/master/hyperparameter _ tuning/Analyze _ results/HPO](https://github.com/awslabs/amazon-sagemaker-examples/blob/master/hyperparameter_tuning/analyze_results/HPO_Analyze_TuningJob_Results.ipynb)
2.  贝叶斯优化是否是调优 XGB 参数的最佳选择还是未知数。如果你检查了文件，梯度推进树没有被提及/测试。[https://docs . AWS . Amazon . com/sage maker/latest/DG/automatic-model-tuning-how-it-works . html](https://docs.aws.amazon.com/sagemaker/latest/dg/automatic-model-tuning-how-it-works.html)
3.  该参数通过单个验证集进行调整，而不是交叉验证。
4.  我还没想好如何在 Python XGBoost 中使用其内置的 XGBoost 算法训练出来的模型神器。

但是除了这些问题，我们仍然可以利用它的端点特性。你可以在任何地方训练你的 XGB 模型，从 Amazon ECR(Elastic Container Registry)把它放在 XGBoost 映像中，然后把它部署成一个端点。

* * * * *

XGBoost / LightGBM 是相当新的 ML 工具，它们都有潜力变得更强。开发人员已经做了出色的工作，创造了这些工具，使人们的生活变得更容易。我在这里指出我的一些观察和分享我的经验，希望它们能成为更好、更易用的工具。