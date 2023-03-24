# 使用 Spark 进行用户流失预测

> 原文：<https://towardsdatascience.com/user-churn-prediction-using-spark-22ff8dafb5c?source=collection_archive---------28----------------------->

## Udacity 数据科学家纳米学位计划顶点项目

![](img/d0bff288ae5c19ea062a5f3895b45f1f.png)

该项目是 [**Udacity**](https://eu.udacity.com/) **数据科学家纳米学位项目:数据科学家顶点计划的最终项目。**目标是预测用户是否会从虚拟的数字音乐服务中流失 **Sparkify**

流失预测是商业中最受欢迎的**大数据**用例之一。正如这篇[帖子](https://neilpatel.com/blog/improve-by-predicting-churn/)中更好地解释的那样，它的目标是确定客户是否会取消他的服务订阅

让我们从使用 **CRISP-DM 流程**(数据挖掘的跨行业流程)开始:

1.  **业务理解**
2.  **数据理解**
3.  **准备资料**
4.  **数据建模**
5.  **评估结果**
6.  **展开**

**业务理解**

Sparkify 是一项数字音乐服务，可以免费使用，方法是在歌曲之间收听一些广告，或者支付每月订阅费以获得无广告体验。在任何时候，用户都可以决定从高级降级到免费，从免费升级到高级或者取消服务。

![](img/b3e4a03d1201cb8a921003385f2a5935.png)

[https://www.udacity.com/course/data-scientist-nanodegree--nd025](https://www.udacity.com/course/data-scientist-nanodegree--nd025)

**数据理解**

所提供的数据集基本上由平台上每个用户操作的日志组成。每个动作都标有时间戳 **ts**

![](img/d94256b466506a8177e0e4db75f72bb8.png)

Dataset attributes

![](img/53170e1f8a6122303d15e7f10262989e.png)

First 5 records as example

在这个小数据集中，我们有来自 225 个用户的 286500 条记录:

46%的女性和 54%的男性

![](img/a90c0670518063e8fa1bc8f57d2ffdde.png)

Gender distribution in the small dataset

54%的互动来自免费用户，46%来自高级用户

![](img/e067bb43780c9faa3f509853aed86f83.png)

Level distribution in the small dataset

![](img/42d21c57f62aaa0602cfb577c69aa0f8.png)

Page distribution in the small dataset

![](img/b3e76bb79016d34f5232833773a7060e.png)

% page distribution in the small dataset

这些记录的时间跨度从 2018 年 10 月到 2018 年 12 月

**准备数据**

第一步是删除所有用户 Id 为空的记录。空字符串 **userId** 很可能指的是尚未注册的用户，或者已经注销并即将登录的用户，因此我们可以删除这些记录。

然后我定义了一个新的`Churn`列，它将被用作模型的标签。基本上，如果用户曾经访问过`Cancellation Confirmation`页面，我们会将其标记为搅动。当然，这个事件对于付费和免费用户都可能发生。

我们获得了 23%的流失用户和 77 %的未流失用户，因此数据集非常不平衡。正如这篇精彩的[帖子](http://www.davidsbatista.net/blog/2018/08/19/NLP_Metrics/)中所解释的，当我们在**评估结果**部分讨论指标时，我们必须记住这一点

![](img/ee35ad10919255bcaa4bb180a3fcbbd4.png)

Churn distribution in the small dataset

然后，我对一些特性进行了比较，同时也考虑了`Churn`值:

![](img/863619e8fcfeb8fe1914d2d4de94c8b9.png)

Churn Gender distribution in the small dataset

![](img/2c891ccb546ec915c36ddf7e163604e1.png)

Churn Level distribution in the small dataset

![](img/d5d87c82d72c1a17cb4e44634a56cf9e.png)

Churn Page distribution in the small dataset

![](img/5a2b3f5d071c436a0ea77e84907568c4.png)

% churn page distribution in the small dataset

**数据建模**

所有对我们的任务有用的分类特征都已经通过**用户 Id** :
-性别
-级别
-页面进行了一次性编码和聚合

![](img/aa3ae8dd840147cbb2feff1f908436d2.png)

Engineered dataset attributes

然后，我们通过提出以下问题添加了一些有趣的工程特性:
-用户订阅该服务多久了？流失与此有关吗？
-上个月的活动(对于不满意的用户，取消是取消前的最后一个月)以周为单位划分
-一个用户听了多少艺术家的音乐？

![](img/a25b0a4f2d712c9ffcd2f7c4f2457223.png)

Churn Registration days distribution in the small dataset

![](img/f75b4b6b39e8ab87870ca094c081b251.png)

Churn Last week average activity distribution in the small dataset

![](img/a833b91f18c0c5e15c6e5f401337c290.png)

Churn Artist distribution in the small dataset

为 ML 准备的最终数据集如下所示:

![](img/e8ffb3e2df5e8c07a1775a8e5e93c7f0.png)

Engineered dataset attributes in input to the model

重要的是不要考虑像`page_Cancellation_Confirmation`或`page_Cancel`这样的属性，因为它们精确地映射了标签列，所以准确率总是 100%，因为我们基本上是在学习我们想要预测的值

**评估结果**

**混淆矩阵**是一个表格，通常用于描述一个分类模型对一组真实值已知的测试数据的性能。

![](img/3e6f4fcbcf895a43b31b015d7b339a04.png)

**准确性**衡量分类器做出正确预测的频率。它是正确预测数与总预测数的比率:

`Accuracy = (True Positives + True Negative) / (True Positives + False Positives + True Negatives + False Negatives)`

精度告诉我们正确的预测中有多少是正确的。它是真阳性与所有阳性的比率:

`Precision = True Positives / (True Positives + False Positives)`

回忆(敏感度)告诉我们实际上正确预测中有多少被我们归类为正确的。它是真阳性与所有实际阳性预测的比率:

`Recall = True Positives / (True Positives + False Negative)`

**F-beta 评分**是一个同时考虑精确度和召回率的指标:

![](img/4199dad8f00e50e0c4499a0066c01d34.png)

生成**朴素预测器**的目的只是为了显示没有任何智能的基础模型是什么样子。如前所述，通过观察数据的分布，很明显大多数用户不会流失。因此，总是预测`'0'`(即用户不访问页面`Cancellation Confirmation`)的模型通常是正确的。

![](img/7c721fb88a61ff3af2892b6825b2be04.png)

将所有用户标记为流失= 0 的朴素模型在测试集上做得很好，准确率为 81.2%，F1 分数为 0.7284

数据集不平衡的事实也意味着**精确度**没有太大帮助，因为即使我们获得高精确度，实际预测也不一定那么好。在这种情况下，通常建议使用**精度**和**召回**

让我们比较 3 个模型的结果:

*   **逻辑回归**
*   **梯度提升树**
*   **支持向量机**

第一步是删除培训中不必要的列

```
colonne = df.columns[1:-1]
colonne
```

![](img/24ea5715d2d17f081e7ae1b65d13a598.png)

Features used for ML training

然后所有的特征都被矢量化(不需要转换，因为所有的特征都是数字)

```
assembler = VectorAssembler(inputCols = colonne, outputCol = ‘features’)data = assembler.transform(df)
```

`StandarScaler()`用于缩放数据

```
scaler = StandardScaler(inputCol = 'features', outputCol = 'scaled_features', withStd = True)scaler_model = scaler.fit(data)data = scaler_model.transform(data)
```

然后，我将数据分为训练、测试和验证数据集

```
train, rest = data.randomSplit([0.6, 0.4], seed = 42)validation, test = rest.randomSplit([0.5, 0.5], seed = 42)
```

对于所有车型，我都使用了 F1 分数作为衡量标准

```
f1_evaluator = MulticlassClassificationEvaluator(metricName = ‘f1’)
```

以及`ParamGridBuilder()`和 3 倍`CrossValidator()`来确定考虑所有参数的模型的最佳超参数

```
param_grid = ParamGridBuilder().build()
```

**逻辑回归**

```
logistic_regression = LogisticRegression(maxIter = 10)crossvalidator_logistic_regression = CrossValidator(  
estimator = logistic_regression,                                 evaluator = f1_evaluator,                                                    estimatorParamMaps = param_grid,                                                    numFolds = 3)cv_logistic_regression_model = crossvalidator_logistic_regression.fit(train)
```

![](img/917605b92f9e6acfc825cebd327b01c5.png)

Best parameters

**梯度提升树**

```
gradient_boosted_trees = GBTClassifier(maxIter = 10, seed = 42)crossvalidator_gradient_boosted_trees = CrossValidator(
estimator = gradient_boosted_trees,                                                       evaluator = f1_evaluator,                                                       estimatorParamMaps = param_grid,                                                       numFolds = 3)cv_gradient_boosted_trees_model = crossvalidator_gradient_boosted_trees.fit(train)
```

![](img/a96c8c3255f7e8abd5d999368fac86e9.png)

Best parameters

GBT 也允许看到特性的重要性:

![](img/edcbf3929d8094f1cf7f11320660c6dd.png)

Feature importances from GBT

我们可以看到`registration_days`和`count_page_last_week`具有最高的重要性

**支持向量机**

```
linear_svc = LinearSVC(maxIter = 10)crossvalidator_linear_svc = CrossValidator(
estimator = linear_svc,                                           evaluator = f1_evaluator,                                           estimatorParamMaps = param_grid,                                           numFolds = 3)cv_linear_svc_model = crossvalidator_linear_svc.fit(train)
```

![](img/47b175f5f675593686dfe15c92bcc702.png)

Best parameters

总的来说，**逻辑回归**具有最好的结果，在测试数据集上 **F-1 得分**为 0.8218，在验证数据集上为 0.7546

![](img/f12348050650c0be4c29b0b716c3e7d3.png)

Results on test and validation datasets

**细化**

我第一次尝试手动调整模型的一些参数，但是通过让`ParamGridBuilder()`和`CrossValidator()`搜索所有参数获得了最佳结果

**部署**

根据 DSND 顶点项目的**云部署说明的建议，我已经用 [**AWS**](https://aws.amazon.com) 创建了一个集群**

![](img/019eee5a9b99ba6ad3e6eed70e4ce89c.png)

My cluster configuration

正如这里更好地解释的**m3 . xlarge**是第二代通用 **EC2** 实例，配备了高频**英特尔至强 E5–2670**和 2 个基于 40 GB 固态硬盘的实例存储

![](img/3c1079d44d129b25c6e44ba20eee7fcc.png)

My cluster summary

然后我创建了一个笔记本并复制粘贴了必要的代码

![](img/92f25c760bab48b37037a4b788aa75d9.png)

sparkify notebook summary

在真实数据集上，我们有来自 22278 个用户的 26259199 条记录:

47%的女性和 53%的男性

![](img/a1ddd547adb17ea9f2392bfd87ca93f8.png)

Gender distribution in the full dataset

21%的互动来自免费用户，79%来自高级用户

![](img/4d3af7fe8b1b589e31876bf591263894.png)

Level distribution in the full dataset

![](img/04dfaac0b9aaa85f642fcdde7adbdb8a.png)

Page distribution in the full dataset

22%的用户感到不适，78 %没有

这个小数据集很好地代表了真实数据集，这意味着它似乎没有偏见

**结论**

我们的目标是预测用户是否会取消服务，以使公司能够为他提供优惠或折扣，从而留住这些用户。在清理数据并将它们建模为准备用于 ML 训练的数据集之后，我们测试了三个不同模型的性能。所有产生的模型都成功地预测了用户是否会离开服务，比给出总是答案的`'0'`(用户不会流失)的**天真预测器**好不了多少。考虑到 **F-1 得分**最好的模型是**逻辑回归。**尽管结果很好，但该模型可以通过精心设计更多的工程特征来捕捉一些可能与用户对服务的满意度相关的行为模式来改进:推荐引擎好吗？意思是推荐给用户的歌真的符合他们的口味。从 **GBT** 的功能重要性来看，原始功能`page_Thumbs_Up`和`page_Thumbs_Down`相当重要，因此捕捉用户音乐品味的新功能确实可以改善模型

这个项目的代码可以在这个 github [资源库](https://github.com/simonerigoni/data_scientist_capstone)中找到，在我的博客上有一个意大利语的[帖子](https://simonerigoni01.blogspot.com/2023/01/previsione-dellabbandono-degli-utenti.html)。