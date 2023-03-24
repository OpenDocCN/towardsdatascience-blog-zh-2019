# 使用 Google Cloud BigQuery ML 的经验教训

> 原文：<https://towardsdatascience.com/lessons-learned-using-google-cloud-bigquery-ml-dfd4763463c?source=collection_archive---------11----------------------->

## 云自动机器学习工具(第 1 部分)

## 使用德国信贷数据的全程 ML 演示

![](img/2d092cbb26d0b3075be7829f12cae204.png)

## 动机

我经常听到的一个新流行语是“为大众民主化人工智能”。通常接下来是一个建议的云机器学习工具。这些工具的统称似乎是 AML，或自动机器学习。作为一名数据科学家，我很想研究这些工具。一些问题浮现在我的脑海:AML 到底能做什么？我可以在我通常的建模工作流程中使用这些工具吗？如果有，如何实现，为了什么利益？我作为一个拥有 ML 技能的人类的有用性会很快消失吗？

我的计划是演示最流行的 AML 工具，每次都使用相同的“真实”数据集，比较运行时间、拟合指标、生产步骤以及一些用户界面截图。

在我最近的工作中，我需要建立和部署金融欺诈检测模型。我不能在公开的博客文章中使用专有数据，所以我将使用众所周知的公开数据集[德国信用数据](https://archive.ics.uci.edu/ml/datasets/statlog+(german+credit+data))。注意，成本矩阵是每个假阴性 5 DM，每个假阳性 1 DM。尽管数据很小，但我将介绍所有用于真实数据的技术。所有演示的代码将在我的 github 上[。](https://github.com/christy/MachineLearningTools)

我的计划是涵盖这些流行的反洗钱工具:

*   第一部分(本帖):[谷歌云平台上的谷歌大查询(GCP)](https://cloud.google.com/bigquery) 。

第 2，3，4 部分即将推出，希望(！) :

*   第二部分: [H2O 的 AutoML](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/automl.html) ，以及[思科的 Kubeflow 超参数调优](https://github.com/kubeflow/katib)。重点是算法对比和超参数选择。
*   第三部分:[在谷歌云平台(GCP)上使用 Keras 的 tensor flow 2.0](https://www.tensorflow.org/)。
*   第 4 部分:[使用 Scala 的 AWS 上的数据块 MLFlow](https://docs.databricks.com/applications/mlflow)

## BigQuery ML —步骤 1)创建数据

让我们开始吧。谷歌的 BigQuery 提供了大量免费的公共数据集。你可以在这里搜索它们。我没有看到任何信用数据，所以我用谷歌的 [BigQuery](https://cloud.google.com/bigquery/docs/quickstarts/quickstart-web-ui) 沙箱上传了我的。BigQuery sandbox 是谷歌的 GCP 自由层云 SQL 数据库。这是免费的，但你的数据每次只能保存 60 天。

![](img/3c6c20d78c971b44d4a8bde257cbd565.png)

[https://cloud.google.com/free/](https://cloud.google.com/free/)

![](img/c28e79101b52e4f66846cd3514a3fe1a.png)

[https://cloud.google.com/bigquery/docs/sandbox](https://cloud.google.com/bigquery/docs/sandbox)

对于所有的谷歌云工具，你必须首先创建一个谷歌云账户和一个项目。接下来，分配沙盒资源给项目[按照这些指示](https://cloud.google.com/bigquery/docs/sandbox)。现在转到[https://console.cloud.google.com/bigquery](https://console.cloud.google.com/bigquery)，这是网络用户界面。接下来，固定您的沙盒启用项目(见左下截图)。接下来，[使用 Web UI](https://cloud.google.com/bigquery/docs/quickstarts/quickstart-web-ui#load_data_into_a_table) 加载您的数据。我把我的表叫做“allData”。

德国信贷数据是一个非常小的数据集，只有 998 行。通常，最佳实践是将您的数据拆分为 train/valid/test，但由于这很小，我将拆分为 80/20 train/test，并使用[交叉验证技术](https://machinelearningmastery.com/k-fold-cross-validation/)将 train 扩展为 train/valid。注意:我创建了一个假的“uniqueID”列(见右下截图)，以帮助随机拆分。

![](img/f4e92014a0aeb0c9d36517e89c51d689.png)

Create train and test tables

下面是我用来创建“trainData”和“testData”的 SQL，它们是从“allData”中随机抽取的行。

```
CREATE TABLE cbergman.germanCreditData.testData AS
SELECT *
FROM `cbergman.germanCreditData.allData`
WHERE MOD(ABS(FARM_FINGERPRINT(CAST(uniqueID AS STRING))), 5) = 0;

CREATE OR REPLACE TABLE cbergman.germanCreditData.trainData AS
SELECT *
FROM `cbergman.germanCreditData.allData`
WHERE NOT uniqueID IN (
  SELECT DISTINCT uniqueID FROM `cbergman.germanCreditData.testData`
);
```

现在，我已经在 BigQuery 中加载了分别包含 199 行和 798 行的表。检查随机样本做了正确的事情:

```
SELECT count (distinct uniqueID)
FROM `cbergman.germanCreditData.allData`
where response = 2;

# repeat for train, test data...
```

请参见下面的最后一行，我最终得出每个所有/训练/测试数据集的负(响应=1)与正(响应=2)比率分别为 30%、29%、32%，这看起来是一个很好的采样:

![](img/f59d937dff9a060178446d259f2659e5.png)

## 大查询 ML —步骤 2)训练一个 ML 模型

在写这篇文章的时候，API 给了你 3 个算法的选择:回归，逻辑回归，或者 K-最近邻。

> 更新 2019 年 12 月:自撰写本文以来，谷歌已向 BigQuery ML 添加了 2 种新算法:多类逻辑回归和 Tensorflow！

在创建你的模型之前，通常你必须:1)确保业务目标明确:“尽可能自动地抓住最多的欺诈资金”。在我的工作中，欺诈资金被进一步分为不同类型的欺诈，例如第三方欺诈。2)接下来，您通常必须将其转化为数学模型算法:在这种情况下，分类选择 API 方法逻辑回归。3)接下来，你定义损失函数，这在真实的公司中是很棘手的，因为一个部门可能会说新用户更重要；而风险部说所有的欺诈都更重要。例如，在我工作的一家公司，他们无法就一项损失指标达成一致。然而，这个问题在这里得到了解决，这个公共数据的创建者给出了一个成本矩阵，每个假阴性 5 DM，每个假阳性 1 DM。

完成业务定义后，请看下面的屏幕截图，如何通过从查询编辑器运行 SQL 查询来训练基本模型。在训练细节下，每次迭代的损失是递减的，这是我们预期的；由于这是一个逻辑回归，损失是对数损失。

![](img/025a0fb3c8045b4c623013c0579c0201.png)

Train a model in BigQuery ML

下面是创建模型并检查模型是否适合训练和测试数据的 SQL。

```
# Create the base model
CREATE OR REPLACE MODEL cbergman.germanCreditData.baseModel OPTIONS(input_label_cols=['response'], model_type='logistic_reg') AS 
SELECT * EXCEPT (uniqueID) 
FROM `cbergman.germanCreditData.trainData`;# Model fit on train data
SELECT *
FROM ML.EVALUATE(MODEL `cbergman.germanCreditData.baseModel`, 
(
  SELECT * EXCEPT (uniqueID)
  FROM `cbergman.germanCreditData.trainData`);# Model fit on test data
SELECT *
FROM ML.EVALUATE(MODEL `cbergman.germanCreditData.baseModel`, 
(
  SELECT * EXCEPT (uniqueID)
  FROM `cbergman.germanCreditData.testData`);# To view your linear beta-values
SELECT * from ML.WEIGHTS(MODEL cbergman.germanCreditData.baseModel);# To get test data confusion matrix
SELECT *
FROM ML.CONFUSION_MATRIX(MODEL `cbergman.germanCreditData.baseModel`,
(
  SELECT* EXCEPT (uniqueID)
  FROM `cbergman.germanCreditData.testData`);
```

**我的基线逻辑回归在 1 分 36 秒内训练，在 0.7 秒内运行测试，训练 ROC_AUC = 0.84，测试 ROC_AUC = 0.75，回忆= 0.41，成本 38 * 5 + 12 = 202 DM。**

我注意到火车 AUC 是 0.84，所以发生了一些轻微的过度拟合。我并不感到惊讶，因为所有的变量都被使用了(选择*)，而且逻辑回归是一个线性模型，它要求所有的输入都是独立的。创建基线线性模型的一部分是清理共线输入。我知道如何在 Python 中做到这一点，所以这是我稍后将转向的地方。

同时，我们可以检查并保存逻辑回归模型系数的β值，以及排除验证数据的混淆矩阵和性能指标。注意:除非您另外指定，否则默认阈值将是 0.5。我计算了标准化和非标准化的默认混淆矩阵，并将它们保存在模型比较表中(我备份了 Google Sheets 中的系数和性能统计表，WebUI 中非常简单的菜单导航，因为我知道我的免费数据将在 60 天后消失)。

## 步骤 3)特征工程

下面是一个有经验的从业者如何获得更好的模式的“秘制酱”。我现在切换到 [Jupyter 笔记本](https://jupyter.org)，因为我知道如何在 Python 中清理共线输入。我首先检查数字列的相关性。下面，右边突出显示的红框，我们看到数量和持续时间相关 63%。这是有道理的，贷款的金额和期限都可能一起增加，我们可以在 pair-plots 中证实这一点(左边红色轮廓的小图)。下面，底部一行显示了所有的数字变量对响应 1 或 2 绘制，显然没有什么看起来相关。我将*删除持续时间字段。*

![](img/3bc09d7920eb15c4d662a346873d674b.png)

Top left: pair-plots amount and duration. Top right: Numeric variables only correlations. Bottom: Plot response vs each numeric variable.

相当多的变量是字符串类别。我想对字符串变量进行 [VIF 分析](https://en.wikipedia.org/wiki/Variance_inflation_factor)，为此我已经用 Python 实现了[信息 R 包](https://cran.r-project.org/web/packages/Information/index.html)，有时称为“使用信息值的证据权重”。信息或 WOE 宁滨的想法不仅仅是通过[一键编码](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f)添加新变量，而是实际上按照响应变量的线性升序对类别进行分类和编号。例如，如果您有类别为“汽车”、“船”、“卡车”的变量“车辆”，那么将选择转换为数字 1、2、3(例如代表汽车、船、卡车)来表示它们给出的关于响应变量的“信息增益”。所以，你的回答不会是 1，2，3，而是一个信息对齐的数字。

> 我在本文底部的参考资料部分添加了几个关于“信息宁滨”的链接。

在我做了信息线性宁滨后，我可以计算波动率指数并重新运行相关性，见下面的结果。

![](img/33df239302721efab5807c0b485853d7.png)

Left: VIF factors. Right: correlation heatmap

VIF 因子在左上方。有趣的是，就方差解释而言，工作和电话看起来密切相关。也许失业/非居民不太可能有电话？另一个方差-覆盖率对看起来像性别与其他债务人相关。嗯，这表明已婚/离异女性更有可能有一个共同申请人。工作和电话位于冗余差异解释器的顶部，因此*基于 VIFs，我将放弃电话字段。*

右上方的相关性现在看起来清晰多了。唯一担心的可能是 n_credits 与 credit_his 的关联度为 40%,而 property 与 housing 的关联度为 36%*。基于相关性，我会放弃 n_credits 和 property。*

总之，我删除了*持续时间、n_credits、属性和电话字段，并添加了对数字特征计算*的对数转换。我最初的 21 个字段现在变成了 64 个全数字字段。主要是因为增加了数字变换:均值、中值、最大值、最小值和对数变换。由于使用了信息宁滨而不是热编码，我的类别字段数保持不变。

接下来，我将通过[弹性网套索变量选择](https://en.wikipedia.org/wiki/Elastic_net_regularization)运行我所有的转换变量。(注意:如果有人问我谁是我心目中的英雄，我的第一反应会是[特雷弗·哈斯蒂](https://web.stanford.edu/~hastie/)！).通常使用 [k 倍交叉验证](https://machinelearningmastery.com/k-fold-cross-validation/)，k=10 倍是最佳实践。与每折叠 N/k 的样本数相比，更多折叠的代价是平均运行次数更多(k=10)。由于我的数据很小，我选择了 k=4 折叠交叉验证，因此每折叠有 199 个样本。我以下面的模型结束。

![](img/0658697528c32b43fedd2bf9fe72d7f3.png)

Top left: Confusion matrix. Middle left: ROC curves. Top right: Standardized coefficients. Bottom: Precision, Recall curves vs Threshold.

在左中上方，ROC 曲线看起来相当一致。在底部上方，训练和有效的精确召回曲线似乎在阈值= 0.9 附近相交，这显示了令人放心的一致性。在右上角，最大的系数是“支票账户”。我可以查看信息宁滨，以升序排列 A11(负余额)、A12 ( < 200DM), A13 (> = 200DM)、A14(无支票账户)来预测获得贷款。我们还注意到金额和期限与获得贷款负相关。奇怪的是，“工作技能水平”似乎对预测“好”或“坏”的贷款结果没有太大影响，对于“非技术-非居民”对“非技术-居民”对“管理/官员”。也许这一切都有道理，我可能会想与领域专家检查这一点。

目前，我坚持使用**这个逻辑回归模型，它在 3.6 秒内训练，在 0.5 秒内运行测试数据，训练 ROC_AUC = 0.81，测试 ROC_AUC = 0.83，成本为 11 * 5 + 34 = 89 DM。** *我们比基本型号节省了 50%的成本！！并且使用更少的变量，训练和验证之间的差异更小。*这种模式显然是一种改进，如果我们参加比赛，我们会在 Kaggle 排行榜上，因为 top score 的测试 AUC 约为 78%。很好。

在特征工程之后，接下来的步骤通常是:
1)算法比较
2)超参数调整
3)阈值调整

我认为，通过尝试其他算法，进行超参数和阈值调整，我们甚至可以做得比我们自己的 83%更好，但我将在本博客系列的稍后部分演示这一点。

## 步骤 4)创建使用 BigQuery 模型的 ML 管道

回到我如何在现实生活的 ML 流水线中使用 BigQuery。在我目前的公司，用户并不期望他们的信用申请得到立即的回应。因此，我们可以使用批处理进行“实时”评分。下面是我的 BigQuery 批处理管道可能的样子:

1.  以上面的特征工程 Python 脚本为例，在 allData 上运行一次。
    *最终结果:另一个名为“transformedData”的 bigQuery 表。*
2.  像以前一样将转换后的数据分成训练/测试
3.  使用交叉验证训练另一个 GLM 模型，与之前的过程相同，但仅限于*“trainTransformedData”*表(而不是“trainData”)，并且仅选择我的最终模型中的最终特征(而不是 select *)。
    *最终结果:glmModel 而不是 baseModel。*
4.  将 glmModelCoefficients，modelComparison metrics 保存为 Google Sheet(更永久，回想一下 GCP 大查询免费层每 60 天消失一次)。
    *最终结果:多了两个表:glmModelCoeffiecients，modelComparisons(带有 Google Sheets 中的备份)*。
5.  采用与步骤 1)中相同的要素工程 Python 脚本，将其转换为批处理作业，每隔几个小时仅对尚未转换的新数据运行一次。
    *最终结果:另一个名为“testTransformedData”的 BigQuery 表，标记为 model_run=false。*
6.  进行实时预测-将另一个批处理作业添加到在上述步骤 1)后每 1.5 小时运行一次的批处理作业编制器中。
    *最终结果:所有传入的新数据每 2 小时运行一次，批量实时获得推理分类。*

步骤 6)要运行 BigQueryML，只需在 Java 或 Python 脚本中添加一条 SQL 语句:

```
# SQL to run inference on new data
SELECT
  *
FROM
  ML.EVALUATE(MODEL `cbergman.germanCreditData.glmModel`, (
SELECT   'chk_acct_woe', 'credit_his_woe', 'purpose_woe', 'saving_acct_woe', 'present_emp_woe', 'sex_woe', 'other_debtor_woe', 'other_install_woe', 'housing_woe', 'job_woe', 'foreign_woe','amount_logmean', 'installment_rate_logmedian', 'present_resid_logmedian', 'age_logmedian',  'n_people_max'
FROM `cbergman.germanCreditData.testTransformedData`
WHERE model_run = false);

# Tag inferenced data so it won't be re-inferenced
UPDATE `cbergman.germanCreditData.testTransformedData`
SET model_run = true;
```

## 摘要

我们使用 Google 云平台 BigQuery ML 工具进行了自始至终的 ML 建模。我们使用原始 Python、Scikit-learn、pandas、matlab、seaborn 来进行特征工程。把它变成一个脚本来进行特性工程，作为一个额外的数据工程步骤来创建转换后的 BigQuery 表。实现了一个逻辑回归模型，其测试 ROC_AUC = 0.83，成本为 11 * 5 + 34 = 89 DM(这将使我们进入 Kaggle 排行榜前 10 名)。

我的印象是，就像现在的 BigQuery ML 一样，如果没有额外的特性工程工作，你还不能得到一个值得 Kaggle 使用的好模型，但是你可以得到一个快速的基线模型。谷歌显然正在把资源放在它的 BigQuery ML 产品上，上个月有两个新的算法。仅此一点就说明这是一款值得关注的产品。即使是现在，我也要说这是一个值得学习的 AML 工具，并放入你的云 AI/ML 从业者工具箱中。

下一篇帖子，我将演示使用 Tensorflow 2.0 在相同的数据上构建深度神经网络。

# 资源

1.  BigQuery 文档页数:[https://cloud.google.com/bigquery/docs/](https://cloud.google.com/bigquery/docs/)
2.  博客作者谷歌的 BigQuery O'reilly book 作者:[https://towards data science . com/how-to-do-hyperparameter-tuning-of-a-big query-ml-model-29ba 273 a 6563](/how-to-do-hyperparameter-tuning-of-a-bigquery-ml-model-29ba273a6563)
3.  一般反洗钱:【https://www.automl.org/book/ 
4.  我的 github:【https://github.com/christy/MachineLearningTools 
5.  开放信用数据集:[https://archive . ics . UCI . edu/ml/datasets/statlog+% 28 german+credit+data % 29](https://archive.ics.uci.edu/ml/datasets/statlog+%28german+credit+data%29)
6.  本文中提到的“信息宁滨”的 r 包:[https://cran . r-project . org/web/packages/Information/index . html](https://cran.r-project.org/web/packages/Information/index.html)
7.  对“信息宁滨”的进一步解释:[http://u analytics . com/blogs/information-value-and-weight-of-evidence banking-case/](http://ucanalytics.com/blogs/information-value-and-weight-of-evidencebanking-case/)

请随意使用我的截图和代码，但请做一个好公民，如果你想在自己的工作中使用它们，请记得注明出处。

如果你对我的文章或人工智能或人工智能有任何反馈、评论或有趣的见解要分享，请随时在我的 LinkedIn 上联系我。