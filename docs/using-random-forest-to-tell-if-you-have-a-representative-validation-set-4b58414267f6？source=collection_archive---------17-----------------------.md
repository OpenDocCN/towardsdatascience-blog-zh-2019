# 使用随机森林来判断是否有代表性的验证集

> 原文：<https://towardsdatascience.com/using-random-forest-to-tell-if-you-have-a-representative-validation-set-4b58414267f6?source=collection_archive---------17----------------------->

## 这是一个快速检查，检查您最重要的机器学习任务之一是否设置正确

![](img/9cdf525d871e4203c76f16eed24dc0a0.png)

Photo by [João Silas](https://unsplash.com/@joaosilas?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

当运行预测模型时，无论是在 Kaggle 比赛中还是在现实世界中，您都需要一个代表性的验证集来检查您正在训练的模型是否具有良好的泛化能力，也就是说，该模型可以对它从未见过的数据做出良好的预测。

那么，我所说的“代表”是什么意思呢？嗯，它真正的意思是，你的训练和验证数据集是相似的，即遵循相同的分布或模式。如果不是这样，那么你就用苹果来训练你的模型，然后试着用橙子来预测。结果将是非常糟糕的预测。

您可以进行大量探索性数据分析(EDA ),并检查两个数据集中的每个要素的行为是否相似。但那可能真的很费时间。测试您是否有一个有代表性的或好的验证集的一个简洁而快速的方法是运行一个随机的森林分类器。

在这个 [Kaggle 内核](https://www.kaggle.com/akosciansky/do-train-and-test-set-share-the-same-distribution)中，我正是这样做的。我首先准备了训练和验证数据，然后添加了一个额外的列“train ”,当数据是训练数据时，它的值为 1，当数据是验证数据时，它的值为 0。这是随机森林分类器将要预测的目标。

```
# Create the new target
train_set['train'] = 1
validation_set['train'] = 0# Concatenate the two datasets
train_validation = pd.concat([train_set, validation_set], axis=0)
```

下一步是准备好独立(X)和从属(y)特性，设置随机森林分类器，并运行交叉验证。

我使用的是度量标准 **ROC AUC** ，这是分类任务的常用度量标准。如果指标是 1，那么你的预测是完美的。如果分数是 0.5，那么你和基线一样好，这是如果你总是预测最常见的结果，你会得到的分数。如果分数低于 0.5，那么你做错了。

```
# Import the libraries
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import cross_val_score# Split up the dependent and independent variables
X = train_validation.drop('train', axis=1)
y = train_validation['train']# Set up the model
rfc = RandomForestClassifier(n_estimators=10, random_state=1)# Run cross validation
cv_results = cross_val_score(rfc, X, y, cv=5, scoring='roc_auc')
```

现在，如果训练集和验证集表现相同，您认为 ROC AUC 应该是多少？……没错， **0.5** ！如果分数是 0.5，那么这意味着训练和验证数据是不可区分的，这正是我们想要的。

一旦我们运行了交叉验证，让我们得到分数…和好消息！分数确实是 0.5。这意味着 Kaggle 主机已经为我们建立了一个代表性的验证集。有时情况并非如此，这是一个很好的快速检查方法。然而，在现实生活中，你必须自己想出一个验证集，这有望派上用场，以确保你设置了一个正确的验证集。

```
print(cv_results)
print(np.mean(cv_results))[0.5000814  0.50310124 0.50416737 0.49976049 0.50078978]
0.5015800562639847
```