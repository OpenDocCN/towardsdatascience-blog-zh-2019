# 构建一个机器学习推荐器

> 原文：<https://towardsdatascience.com/build-a-machine-learning-recommender-72be2a8f96ed?source=collection_archive---------8----------------------->

![](img/b1598414301f5c441f72cf1d9e964d57.png)

Background vector created by starline — www.freepik.com

## 用几行 Python 代码构建机器学习推荐器的快速入门指南

在这篇文章中，我将展示如何使用机器学习快速组装推荐器。我将使用 LightFM 推荐库训练一个预测模型，该库可以很好地扩展到 2000 万到 4000 万次交互的中小型数据集，并训练一个关于用户特征和项目特征的模型。然后，我将进一步介绍推荐系统的典型实现，并汇总单个用户的预测，为一个群体提供推荐。

# 关键定义

在深入研究代码之前，先快速回顾一下用来描述推荐系统机制的一些关键定义。

*条目*–这是评级或分数的内容(例如，一本书、一部电影等。)

*标签* —这些是推荐器的*可能的*输出。(例如评级或分数)。我们会用带标签的项目来训练推荐器，也就是已经被用户评分过的项目。

*交互*–这是一个输入和相关输出值的列表，我们希望根据这些值来训练我们的推荐器。(例如，用户和项目参考以及相关评级。)

*用户特征*–这些值代表每个用户的不同特征、行为和属性(如年龄、位置、性别)。

*物品特征*–这些值代表物品的不同特征(如产品类别、颜色、尺寸)

# 我将使用的数据

我将使用 Goodbooks-10k 数据集来演示推荐器，可以从这里下载:http://www2.informatik.uni-freiburg.de/" " ~齐格勒/BX/BX-CSV-Dump.zip

Goodbooks-10K 数据集由三个文件组成:

*BX-用户*–*用户特征*，具体为 user_id、年龄和位置。

*BX-图书*—*条目特征*包括ISBN、Book_Author 等一堆特征。

*BX-图书评级*–用户给出的图书评级*交互*，特别是 user_id、ISBN 和评级。

# 导入库

首先，我们需要从 LightFM 库中导入一些库和一些库。

```
import numpy as np
from lightfm.data import Dataset
from lightfm import LightFM
from lightfm.evaluation import precision_at_k
from lightfm.evaluation import auc_score
from lightfm.cross_validation import random_train_test_split
from scipy.sparse import coo_matrix as sp
```

# 导入数据

我们将导入 GoodBooks-10K 数据，然后创建以下三个函数，以便在需要时获得数据的新副本。

```
#fetch the interactions data
def get_ratings():
return get_data()[0]#fetch the item features
def get_book_features():
return get_data()[1]#fetch the user features
def get_user_features():
return get_data()[2]
```

# 准备数据

为了使用 LightFM 训练模型，我们将提供以下内容:

[1] *交互*作为一个 sparse . COO _ matrix–我们的交互将是 BX-图书-评级. csv 中提供的用户 ID 和 ISBN

[2] *user_features* 作为包含用户特征的可迭代字符串——我们将简单地包括 BX 用户. csv 中提供的年龄

[3] *item_features* 作为包含项目特性的可迭代字符串——我们将简单地包括 BX-Books.csv 中提供的图书作者

LightFM 有一个数据集构造器，它有许多方便的方法来准备好数据输入到模型中。由于我们希望在模型中包含用户和项目特征，因此准备数据将是一个两步过程。

**步骤 1:创建特性映射**

首先，我们在数据集对象上使用 fit()方法来创建 BX-图书-评级. csv 中的映射

```
dataset = Dataset()dataset.fit((x[‘User-ID’] for x in get_ratings()), (x[‘ISBN’] for x in get_ratings()))num_users, num_items = dataset.interactions_shape()
print(‘Num users: {}, num_items {}.’.format(num_users, num_items))
```

我们使用 dataset.fit_partial()方法为 user_features 和 model_features 创建特性映射。

```
dataset.fit_partial(users=(x[‘User-ID’] for x in get_user_features()), items=(x[‘ISBN’] for x in get_book_features()), item_features=(x[‘Book-Author’] for x in get_book_features()), user_features=(x[‘Age’] for x in get_user_features()))
```

**第二步:建立交互、用户特征和物品特征矩阵**

其次，可以在 dataset 对象上调用 build_interactions()方法来构建交互矩阵。

```
(interactions, weights) = dataset.build_interactions(((x[‘User-ID’], x[‘ISBN’]) for x in get_ratings()))
```

我们在数据集对象上调用 build_item_features()和 build_user_features()方法来构建 item_features 和 user_features。这些函数根据 LightFM 的要求返回 sparse.coo_matrix 类型的对象。

```
item_features = dataset.build_item_features(((x[‘ISBN’], [x[‘Book-Author’]]) for x in get_book_features()))user_features = dataset.build_user_features(((x[‘User-ID’], [x[‘Age’]]) for x in get_user_features()))
```

# 指定型号

为了训练推荐器，我们将使用 LightFM 库中提供的加权近似等级成对(WARP)损失函数。WARP 处理(用户，正项，负项)三元组。

LightFM 对 WARP 算法提供了以下解释:

> [1]对于给定的(用户，正项目对)，从所有剩余项目中随机抽取一个负项目。计算两项的预测值；如果负项的预测值超过了正项的预测值加上一个差值，则执行梯度更新，将正项的等级提高，将负项的等级降低。如果没有等级违规，则继续对负项目进行采样，直到发现违规为止。
> 
> [2]如果您在第一次尝试时发现了一个违反负面示例，请进行大梯度更新:这表明在给定模型的当前状态下，许多负面项目的排名高于正面项目，并且模型必须进行大量更新。如果需要大量的采样来找到一个违规的例子，执行一个小的更新:模型可能接近最优，应该以较低的速率更新。

下面是我们如何用翘曲损失函数指定模型:

```
 model = LightFM(loss=’warp’)
```

该模型可以用许多超参数来指定，但是我将在这里跳过这些。值得一提的是，LightFM 还允许贝叶斯个性化排名损失，尽管这通常表现不太好。

# 训练模型

此时，您可能希望拆分训练数据，以评估模型训练的性能。LightFM 有一个方法，random_train_test_split()为我们做了这件事。我们可以打电话。fit()将模型拟合到交互、项目和用户特征集。

```
(train, test) = random_train_test_split(interactions=interactions, test_percentage=0.2)model.fit(train, item_features=item_features, user_features=user_features, epochs=2)
```

我不打算在这里详细介绍模型训练和评估，尽管下面的代码块显示了如何使用我们使用上面的 random_train_test_split()方法创建的训练集和测试集来测量模型训练精度和以 k 间隔报告的 AUC 分数。

```
train_precision = precision_at_k(model, train,item_features=item_features, k=10).mean()test_precision = precision_at_k(model, test, item_features=item_features,k=10).mean()train_auc = auc_score(model, train,item_features=item_features).mean()test_auc = auc_score(model, test,item_features=item_features).mean()print(‘Precision: train %.2f, test %.2f.’ % (train_precision, test_precision))print(‘AUC: train %.2f, test %.2f.’ % (train_auc, test_auc))print(“testing testing testing”)print(‘printing labels’, get_ratings()[‘ISBN’])
```

# 使用模型为一个组进行预测

为了对小组进行预测，我们需要汇总单个用户的评分，以确定某个项目(书)对小组中所有用户的*相对*重要性。这里我们可以使用多种聚合策略，比如最小痛苦和混合方法。为了简单起见，我将简单地计算该组中所有用户的每个项目(书籍)的预测得分的平均值。

首先，我们创建一个函数来遍历给定组中的所有用户，并使用 item_features 和 user_features 调用 model.predict()来预测每个项目的得分。该函数将每个用户的分数堆叠到我们称为 all_scores 的数组中，以便我们执行一些聚合。

```
def sample_recommendation(model, data, user_ids):n_users, n_items = data.shapeall_scores = np.empty(shape=(0,n_items))for user_id in user_ids:scores =. model.predict(user_id,np.arange(n_items),item_features,user_features)all_scores = np.vstack((all_scores, scores))
```

现在让我们通过对每一项取平均来合计分数。然后，让我们对分数进行排序，并按照我们的模型对标签进行评分的相反顺序获取所有标签(ISBN)。

```
item_averages = np.mean(all_scores.astype(np.float), axis=0)labels = np.array([x[‘ISBN’] for x in get_ratings()])

top_items_for_group = labels[np.argsort(-item_averages)]
```

现在，我们可以打印本组的前 5 项。

```
print(“ Top Recommended ISBN for Group:”)for x in top_items_for_group[:5]:print(“ %s” % x)
```

# 取样组建议

最后，我们可以为一个组抽样推荐。值得一提的是，虽然 LightFM 的伸缩性很好，但采样建议不一定需要同步，就像我在这里演示的那样。相反，可以异步计算推荐值，并定期为所有组重新计算推荐值。

```
group = [3,26,451,23,24,25]sample_recommendation(model, interactions, group)
```

这就是如何构建一个机器学习推荐器，用不多的 Python 代码就能做出基于组的推荐。

有关完整的代码，请参见:[https://github . com/jamesdhope/recommender/blob/master/recommender . py](https://github.com/jamesdhope/recommender/blob/master/recommender.py)

如果你喜欢这个，请给我一些掌声。

# 参考

[1] LightFM，https://github.com/lyst/lightfm

[2] LightFM，【https://lyst.github.io/lightfm/docs/examples/dataset.html】T4

[3]Bluedatalab.com，客户项目