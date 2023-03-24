# 选择 TensorFlow/Keras、BigQuery ML 和 AutoML 自然语言进行文本分类

> 原文：<https://towardsdatascience.com/choosing-between-tensorflow-keras-bigquery-ml-and-automl-natural-language-for-text-classification-6b1c9fc21013?source=collection_archive---------6----------------------->

## Google 云平台上三种文本分类方法的比较

谷歌云平台为你提供了三种进行机器学习的方式:

*   Keras 使用 TensorFlow 后端来构建定制的深度学习模型，这些模型是在[云 ML 引擎](https://cloud.google.com/ml-engine/)上训练的
*   [BigQuery ML](https://cloud.google.com/bigquery/docs/bigqueryml) 仅使用 SQL 在结构化数据上构建定制的 ML 模型
*   [Auto ML](https://cloud.google.com/natural-language/automl/docs/) 根据你的数据训练最先进的深度学习模型，无需编写任何代码

![](img/40a56fb066050db5f2dcdbc3b97cfea4.png)

根据你的技能，额外的准确性有多重要，以及你愿意在这个问题上投入多少时间/精力，在它们之间进行选择。使用 BigQuery ML 进行快速问题公式化、实验和简单、低成本的机器学习。一旦您使用 BQML 确定了一个可行的 ML 问题，就使用 Auto ML 来获得无代码的、最先进的模型。只有在你有大量数据和足够的时间/精力投入的情况下，才手工推出你自己的定制模型。

Choosing the ML method that is right for you depends on how much time and effort you are willing to put in, what kind of accuracy you need, and what your skillset is.

在这篇文章中，我将比较文本分类问题的三种方法，这样你就能明白我为什么推荐我所推荐的内容。

# 1.CNN +嵌入+退出 Keras

我在别处详细解释了[问题和深度学习解决方案](/how-to-do-text-classification-using-tensorflow-word-embeddings-and-cnn-edae13b3e575)，所以这一节会非常简短。

任务是给定一篇文章的标题，我希望能够识别它是在哪里发表的。训练数据集来自黑客新闻上发布的文章(BigQuery 中有一个公开的数据集)。例如，下面是一些来源于 GitHub 的标题:

![](img/077663d9912e90edb22172ff2def4a6f.png)

Training dataset

[模型代码](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/09_sequence/txtclsmodel/trainer/model.py)创建一个 Keras 模型，该模型使用一个单词嵌入层、卷积层和 dropout:

```
model = models.Sequential()
num_features = min(len(word_index) + 1, TOP_K)
model.add(Embedding(input_dim=num_features,
                    output_dim=embedding_dim,
                    input_length=MAX_SEQUENCE_LENGTH))model.add(Dropout(rate=dropout_rate))
model.add(Conv1D(filters=filters,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              padding='same'))model.add(MaxPooling1D(pool_size=pool_size))
model.add(Conv1D(filters=filters * 2,
                              kernel_size=kernel_size,
                              activation='relu',
                              bias_initializer='random_uniform',
                              padding='same'))
model.add(GlobalAveragePooling1D())
model.add(Dropout(rate=dropout_rate))
model.add(Dense(len(CLASSES), activation='softmax'))# Compile model with learning parameters.
optimizer = tf.keras.optimizers.Adam(lr=learning_rate)
model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['acc'])
estimator = tf.keras.estimator.model_to_estimator(keras_model=model, model_dir=model_dir, config=config)
```

然后在云 ML 引擎上进行训练，如本 [Jupyter 笔记本](https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/09_sequence/text_classification.ipynb)所示:

```
gcloud ml-engine jobs submit training $JOBNAME \
 --region=$REGION \
 --module-name=trainer.task \
 --package-path=${PWD}/txtclsmodel/trainer \
 --job-dir=$OUTDIR \
 --scale-tier=BASIC_GPU \
 --runtime-version=$TFVERSION \
 -- \
 --output_dir=$OUTDIR \
 --train_data_path=gs://${BUCKET}/txtcls/train.tsv \
 --eval_data_path=gs://${BUCKET}/txtcls/eval.tsv \
 --num_epochs=5
```

我花了几天时间开发最初的 TensorFlow 模型，我的同事 vijaykr 花了一天时间对它进行修改以使用 Keras，可能还花了一天时间对它进行培训和故障排除。

我们有 80%的准确率。为了做得更好，我们可能需要更多的数据(92k 的例子不足以获得使用定制深度学习模型的好处)，可能还需要更多的预处理(如删除停用词、词干、使用可重用的嵌入等)。).

# 2.用于文本分类的大查询 ML

当使用 BigQuery ML、卷积神经网络、嵌入等时。是(无论如何还不是)一个选项，所以我下降到使用一个单词袋的线性模型。BigQuery ML 的目的是提供一种快速、方便的方法来在结构化和半结构化数据上构建 ML 模型。

逐字拆分标题，并在标题的前 5 个单词上训练逻辑回归模型(即，线性分类器)(使用更多的单词不会有太大帮助):

```
#standardsql **CREATE OR REPLACE MODEL advdata.txtclass
OPTIONS(model_type='logistic_reg', input_label_cols=['source'])
AS**WITH extracted AS (
SELECT source, REGEXP_REPLACE(LOWER(REGEXP_REPLACE(title, '[^a-zA-Z0-9 $.-]', ' ')), "  ", " ") AS title FROM
  (SELECT
    ARRAY_REVERSE(SPLIT(REGEXP_EXTRACT(url, '.*://(.[^/]+)/'), '.'))[OFFSET(1)] AS source,
    title
  FROM
    `bigquery-public-data.hacker_news.stories`
  WHERE
    REGEXP_CONTAINS(REGEXP_EXTRACT(url, '.*://(.[^/]+)/'), '.com$')
    AND LENGTH(title) > 10
  )
), ds AS (
SELECT ARRAY_CONCAT(SPLIT(title, " "), ['NULL', 'NULL', 'NULL', 'NULL', 'NULL']) AS words, source FROM extracted
WHERE (source = 'github' OR source = 'nytimes' OR source = 'techcrunch')
)**SELECT 
source, 
words[OFFSET(0)] AS word1, 
words[OFFSET(1)] AS word2, 
words[OFFSET(2)] AS word3,
words[OFFSET(3)] AS word4,
words[OFFSET(4)] AS word5
FROM ds**
```

这很快。上面的 SQL 查询是完整的 enchilada。没什么更多的了。模型训练本身只用了几分钟。我得到了 78%的准确率，这与我用定制的 Keras CNN 模型得到的 80%的准确率相比相当不错。

一旦经过训练，使用 BigQuery 进行批量预测就很容易了:

```
SELECT * FROM ML.PREDICT(MODEL advdata.txtclass,(
SELECT 'government' AS word1, 'shutdown' AS word2, 'leaves' AS word3, 'workers' AS word4, 'reeling' AS word5)
)
```

![](img/acec0c310bf111925106798de0178f6e.png)

BigQuery ML identifies the New York Times as the most likely source of an article that starts with the words “Government shutdown leaves workers reeling”.

[使用 BigQuery](/how-to-do-online-prediction-with-bigquery-ml-db2248c0ae5) 的在线预测可以通过将权重导出到 web 应用程序中来完成。

# 3.AutoML

我尝试的第三个选项是无代码选项，尽管如此，它使用了最先进的模型和底层技术。因为这是一个文本分类问题，要使用的自动 ML 方法是自动 ML 自然语言。

## 3a。启动 AutoML 自然语言

第一步是从 GCP web 控制台启动自动 ML 自然语言:

![](img/49f7d0000da3eaad892ba969d34a676c.png)

Launch AutoML Natural Language from the GCP web console

按照提示进行操作，将会创建一个存储桶来保存将用于训练模型的数据集。

## 3b。创建 CSV 文件，并在谷歌云存储上提供

BigQuery ML 要求您了解 SQL，而 AutoML 只要求您以该工具能够理解的格式之一创建数据集。该工具可以理解排列如下的 CSV 文件:

文本，标签

文本本身可以是包含实际文本的文件的 URL(如果您有多行文本，如评论或整个文档，这很有用)，也可以是纯文本项目本身。如果您直接提供文本项字符串，您需要用引号将其括起来。

因此，我们的第一步是以正确的格式从 BigQuery 导出一个 CSV 文件。这是我的疑问:

```
WITH extracted AS (
SELECT STRING_AGG(source,',') as source, title FROM 
  (SELECT DISTINCT source, TRIM(LOWER(REGEXP_REPLACE(title, '[^a-zA-Z0-9 $.-]', ' '))) AS title FROM
    (SELECT
      ARRAY_REVERSE(SPLIT(REGEXP_EXTRACT(url, '.*://(.[^/]+)/'), '.'))[OFFSET(1)] AS source,
      title
    FROM
      `bigquery-public-data.hacker_news.stories`
    WHERE
      REGEXP_CONTAINS(REGEXP_EXTRACT(url, '.*://(.[^/]+)/'), '.com$')
      AND LENGTH(title) > 10
    )
  )
GROUP BY title
)SELECT title, source FROM extracted
WHERE (source = 'github' OR source = 'nytimes' OR source = 'techcrunch')
```

这产生了以下数据集:

![](img/a7c4c90034db16ea7ddafc9b51fba9c0.png)

Dataset for Auto ML

注意，我去掉了标点符号和特殊字符。空白已经被修剪，SELECT distinct 用于丢弃出现在多个类中的重复项和文章(AutoML 会警告您有重复项，并且可以处理多类标签，但是删除它们会更干净)。

我使用 BigQuery UI 将查询结果保存为一个表:

![](img/723c2b584093f1f1414f1cad7371e05c.png)

Save the query results as a table

然后将表格导出到 CSV 文件:

![](img/dbeecbbdbcdf6cf2253bbb92a60ea2c2.png)

Export the table data to the Auto ML bucket

## 3c。创建自动 ML 数据集

下一步是使用 Auto ML UI 从云存储上的 CSV 文件创建数据集:

![](img/b621d5af6e7cd2e5d7244bc88ea0457c.png)

Create a dataset from the CSV file on Cloud Storage

该数据集需要大约 20 分钟来摄取。最后，我们得到一个充满文本项的屏幕:

![](img/1e5ba4ed2c0663fe0f5625d5422362be.png)

The dataset after loading

当前的 [Auto ML 限制是 100k 行](https://cloud.google.com/natural-language/automl/quotas)，所以我们的 92k 数据集肯定会超出一些限制。数据集越小，消化得越快。

为什么我们有一个标签叫“来源”只有例子？CSV 文件有一个标题行(source，title ),它也已经被摄取了！幸运的是，AutoML 允许我们在 GUI 本身中编辑文本项。所以，我删除了多余的标签和相应的文字。

## 3d。火车

培训就像点击一个按钮一样简单。

Auto ML 然后继续尝试各种嵌入和各种架构，并进行超参数调整，以提出解决问题的好办法。

需要 5 个小时。

## 3e。估价

一旦训练好模型，我们就得到一堆评价统计数据:精度、召回率、AUC 曲线等。但是我们也得到实际的混淆矩阵，从中我们可以计算任何我们想要的东西:

![](img/50928b62a757579fdcd322bdfe52f6ff.png)

总体准确率约为 86%，甚至高于我们定制的 Keras CNN 模型。为什么？因为 Auto ML 能够利用基于谷歌语言使用数据集的模型的迁移学习，即包括我们在 Keras 模型中没有的数据。此外，由于所有数据的可用性，模型架构可以更加复杂。

## 3f。预言；预测；预告

已训练的 AutoML 模型已经部署并可用于预测。我们可以向它发送一个请求，并获取文章的预测来源:

![](img/94714730e27143c03581f3835cbb86df.png)

Predictions from Auto ML

请注意，该模型比 BQML 模型更有信心(尽管两者给出了相同的正确答案)，这种信心是由以下事实驱动的:该 Auto ML 模型是在更多数据上训练的，并且是专门为文本分类问题构建的。

我试了一下《今日头条》的另一篇文章标题，模型显示它来自 TechCrunch:

![](img/11b60e461691f9e638174c1be242cbc4.png)

Correctly identifies the title as being from a TechCrunch article.

# 摘要

虽然这篇文章主要是关于文本分类的，但是一般的结论和建议适用于大多数的 ML 问题:

*   使用 BigQuery ML 进行简单、低成本的机器学习和快速实验，看看 ML 在您的数据上是否可行。有时候，使用 BQML 获得的准确性已经足够了，您可以就此打住。
*   一旦您使用 BQML 确定了一个可行的 ML 问题，就使用 Auto ML 来获得无代码的、最先进的模型。例如，文本分类是一个非常专业的领域，具有高维输入。因此，使用定制的解决方案(例如，Auto ML 自然语言)比只使用单词袋的结构化数据方法做得更好。
*   只有在你有大量数据和足够的时间/精力投入的情况下，才手工推出你自己的定制模型。使用 AutoML 作为基准。如果，经过一些合理的努力，你不能击败自动 ML，停止浪费时间。用自动 ML 就行了。

在 GCP 进行机器学习还有其他一些方法。可以在 ML 引擎中做 xgboost 或者 scikit-learn。深度学习 VM 支持 PyTorch。Spark ML 在 Cloud Dataproc 上运行良好。当然，你可以使用谷歌计算引擎或谷歌 Kubernetes 引擎，并安装任何你想要的 ML 框架。但是在这篇文章中，我将集中讨论这三个。

*感谢*[*Greg Mikels*](https://www.linkedin.com/in/greg-mikels-abb3373a/)*改进了我的原始 AutoML 查询，删除了重复和交叉发布的文章。*