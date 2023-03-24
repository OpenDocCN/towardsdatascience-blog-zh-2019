# 创建 SVD 推荐系统的初学者指南

> 原文：<https://towardsdatascience.com/beginners-guide-to-creating-an-svd-recommender-system-1fd7326d1f65?source=collection_archive---------5----------------------->

![](img/ef84f9f467fdfd4695e965bbecb95e01.png)

Photo by [freestocks](https://unsplash.com/@freestocks?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 序言

有没有登录网飞，看到他们建议你看[重力](https://www.imdb.com/title/tt1454468/)如果你昨晚看了[星际](https://www.imdb.com/title/tt0816692/?ref_=nv_sr_srsg_0)？或者在亚马逊上买了东西，看到他们向我们推荐我们可能感兴趣的产品？或者有没有想过在线广告公司是如何根据我们的浏览习惯向我们展示广告的？这一切都归结于一种叫做推荐系统的东西，它可以根据我们和其他人与产品的互动历史来预测我们可能感兴趣的东西。

正如我所承诺的，我们会做一个推荐系统。为了让你不要自我感觉不好，我们也会做一个很酷的。我们将使用 SVD(奇异向量分解)技术进行协同过滤；这比基本的基于内容的推荐系统高出一个档次。

*协同过滤捕捉志同道合的用户的潜在兴趣模式，并使用相似用户的选择和偏好来建议新项目。*

# 要求

所以让我们开始吧。所以我们需要的东西列在下面。如果你正在阅读这篇文章，你很可能知道并且已经拥有了这些。

1.python >= 2.7
2。熊猫> = 0.17
3。numpy
4。scipy

对于不了解的人来说，熊猫、numpy 和 scipy 是 python 包。这些将使我们的生活变得容易。您可以使用 pip 从终端或命令提示符安装它们。如果你不知道怎么做，谷歌一下。例如，下面的命令安装 pandas 包。

```
$ pip install pandas
```

# 资料组

我们肯定需要一个数据集来处理。我们将使用著名的 Movielens 数据集来制作我们的推荐系统。前往 http://grouplens.org/datasets/movielens/下载电影镜头 100k 数据集。

该数据集包含不同用户对不同电影的大约 100，000 个评级。让我们来探索数据集。创建一个新脚本***exploration . py***并添加以下代码块。**注意**:这里我们将使用单独的脚本，但是你也可以使用一个单独的 iPython 笔记本，这要方便得多。

```
import pandas as pd
import numpy as npdata = pd.read_csv('movielens100k.csv')
data['userId'] = data['userId'].astype('str')
data['movieId'] = data['movieId'].astype('str')users = data['userId'].unique() #list of all users
movies = data['movieId'].unique() #list of all moviesprint("Number of users", len(users))
print("Number of movies", len(movies))
print(data.head())
```

这就对了。您将看到数据集中有 718 个用户和 8915 部电影。

```
Number of users 718
Number of movies 8915
+----+----------+-----------+----------+-------------+
|    |   userId |   movieId |   rating |   timestamp |
|----+----------+-----------+----------+-------------|
|  0 |        1 |         1 |        5 |   847117005 |
|  1 |        1 |         2 |        3 |   847642142 |
|  2 |        1 |        10 |        3 |   847641896 |
|  3 |        1 |        32 |        4 |   847642008 |
|  4 |        1 |        34 |        4 |   847641956 |
+----+----------+-----------+----------+-------------+
```

# 训练和测试分割

我们可以在数据集上使用正常的随机训练测试分割。但是既然我们有可用的时间戳，让我们做一些更好的事情。让我们创建一个新的脚本 ***workspace.py*** 来完成我们所有的工作。在开头添加以下代码。

```
import pandas as pd
import numpy as np
import scipydata = pd.read_csv('movielens100k.csv')
data['userId'] = data['userId'].astype('str')
data['movieId'] = data['movieId'].astype('str')users = data['userId'].unique() #list of all users
movies = data['movieId'].unique() #list of all moviestest = pd.DataFrame(columns=data.columns)
train = pd.DataFrame(columns=data.columns)test_ratio = 0.2 #fraction of data to be used as test set.for u in users:
    temp = data[data['userId'] == u]
    n = len(temp)
    test_size = int(test_ratio*n)temp = temp.sort_values('timestamp').reset_index()
temp.drop('index', axis=1, inplace=True)

dummy_test = temp.ix[n-1-test_size :]
dummy_train = temp.ix[: n-2-test_size]

test = pd.concat([test, dummy_test])
train = pd.concat([train, dummy_train])
```

这样做的目的是，根据给出这些评级时的时间戳，我们对数据进行排序，以将最近的评级保持在底部，并从底部开始从每个用户中抽取 20%的评级作为测试集。因此，我们没有随机选择，而是将最近的评分作为测试集。从推荐者的目标是基于类似产品的历史评级来对未来未遇到的产品进行评级的意义上来说，这更合乎逻辑。

# 效用矩阵

当前形式的数据集对我们没有用。为了将数据用于推荐引擎，我们需要将数据集转换成一种叫做效用矩阵的形式。我们在新的脚本中创建一个函数`**create_utility_matrix**`。将其命名为 ***recsys.py*** 。我们将使用这个脚本中的函数来处理我们的训练和测试集。

作为一个参数，我们传递一个字典，该字典存储我们也传递的数据集“数据”的每一列的键值对。从数据集中，我们将看到每个对应字段的列号或列名，键'T12 用户'' T14]的列【T6]*userId*或列 **0** ，列 ***movieId*** 或键 **'** *项'*和列

*效用矩阵只不过是一个 2D 矩阵，其中一个轴属于用户，另一个轴属于项目(在这种情况下是电影)。所以矩阵的 ***(i，j)*** 位置的值将是用户 ***i*** 给电影 ***j*** 的评分。*

*让我们举一个例子来更清楚一点。假设我们有 5 个评级的数据集。*

```
*+----+----------+-----------+----------+
|    |   userId |   movieId |   rating |
|----+----------+-----------+----------+
|  0 |      mark|     movie1|        5 |
|  1 |      lucy|     movie2|        2 |
|  2 |      mark|     movie3|        3 |
|  3 |     shane|     movie2|        1 |
|  4 |      lisa|     movie3|        4 |
+----+----------+-----------+----------+*
```

*如果我们通过下面描述的`**create_utility_matrix**`函数传递这个数据集，它将返回一个这样的效用矩阵，以及 user_index 和 item_index 的辅助字典，如下所示。*

```
*+----+----+----+
| 5  | nan| 3  |   # user_index = {mark: 0, lucy:1, shane:2, lisa:3}
+----+----+----+   # item_index = {movie1:0, movie2: 1, movie3:2}
| nan| 2  | nan|
+----+----+----+
| nan| 1  | nan|   # The nan values are for user-item combinations
+----+----+----+   # where the ratings are unavailable.
| nan| nan| 4  |
+----+----+----+*
```

*现在来看看功能。*

```
*import numpy as np
import pandas as pd
from scipy.linalg import sqrtmdef **create_utility_matrix**(data, formatizer = {'user':0, 'item': 1, 'value': 2}): """
        :param data:      Array-like, 2D, nx3
        :param formatizer:pass the formatizer
        :return:          utility matrix (n x m), n=users, m=items
    """

    itemField = formatizer['item']
    userField = formatizer['user']
    valueField = formatizer['value'] userList = data.ix[:,userField].tolist()
    itemList = data.ix[:,itemField].tolist()
    valueList = data.ix[:,valueField].tolist() users = list(set(data.ix[:,userField]))
    items = list(set(data.ix[:,itemField])) users_index = {users[i]: i for i in range(len(users))} pd_dict = {item: [np.nan for i in range(len(users))] for item in items} for i in range(0,len(data)):
        item = itemList[i]
        user = userList[i]
        value = valueList[i] pd_dict[item][users_index[user]] = value X = pd.DataFrame(pd_dict)
    X.index = users

    itemcols = list(X.columns)
    items_index = {itemcols[i]: i for i in range(len(itemcols))} # users_index gives us a mapping of user_id to index of user
    # items_index provides the same for items return X, users_index, items_index*
```

# *奇异值分解计算*

*SVD 是*奇异向量分解。*它所做的是将一个矩阵分解成对应于每行和每列的特征向量的组成数组。我们再给 ***recsys.py*** 添加一个函数。它将从“ *create_utility_matrix* 和参数“ *k* ”获取输出，该参数是每个用户和电影将被分解成的特征的数量。*

*奇异值分解技术是由 Brandyn Webb 引入推荐系统领域的，在 Netflix 奖挑战赛期间，Brandyn Webb 以 ***【西蒙·芬克】*** 而闻名。这里我们不做 Funk 的 SVD 迭代版本，而是使用 numpy 的 SVD 实现所提供的任何东西。*

```
*def **svd**(train, k):
    utilMat = np.array(train) # the nan or unavailable entries are masked
    mask = np.isnan(utilMat)
    masked_arr = np.ma.masked_array(utilMat, mask)
    item_means = np.mean(masked_arr, axis=0) # nan entries will replaced by the average rating for each item
    utilMat = masked_arr.filled(item_means) x = np.tile(item_means, (utilMat.shape[0],1)) # we remove the per item average from all entries.
    # the above mentioned nan entries will be essentially zero now
    utilMat = utilMat - x # The magic happens here. U and V are user and item features
    U, s, V=np.linalg.svd(utilMat, full_matrices=False)
    s=np.diag(s) # we take only the k most significant features
    s=s[0:k,0:k]
    U=U[:,0:k]
    V=V[0:k,:] s_root=sqrtm(s) Usk=np.dot(U,s_root)
    skV=np.dot(s_root,V)
    UsV = np.dot(Usk, skV) UsV = UsV + x print("svd done")
    return UsV*
```

# *将它们结合在一起*

*回到 ***workspace.py*** 我们将使用上面的函数。我们将找出使用真实评级的测试集的预测评级的均方根误差。除了创建一个函数，我们还将创建一个列表来保存不同数量的特性，这将有助于我们将来进行分析。*

```
*from recsys import svd, create_utility_matrixdef **rmse**(true, pred):
    # this will be used towards the end
    x = true - pred
    return sum([xi*xi for xi in x])/len(x)# to test the performance over a different number of features
no_of_features = [8,10,12,14,17]utilMat, users_index, items_index = create_utility_matrix(train)for f in no_of_features: 
    svdout = svd(utilMat, k=f)
    pred = [] #to store the predicted ratings for _,row in test.iterrows():
        user = row['userId']
        item = row['movieId'] u_index = users_index[user]
        if item in items_index:
            i_index = items_index[item]
            pred_rating = svdout[u_index, i_index]
        else:
            pred_rating = np.mean(svdout[u_index, :])
        pred.append(pred_rating)print(rmse(test['rating'], pred))*
```

*对于 *test_size = 0.2* ，RMSE 分数大约为 0.96*

*对于 100k 的电影来说，这是一个适中的分数。稍微调整一下，你也许能超过 0.945。但这取决于你。*

*如果你喜欢这篇文章，请告诉我！这里有三个链接供你参考:*

1.  *[](https://github.com/mayukh18/reco)**(SVD 的完整代码以及其他著名 RecSys 算法的实现)***
2.  **[***https://papers with code . com/sota/collaborative-filtering-on-movie lens-100k***](https://paperswithcode.com/sota/collaborative-filtering-on-movielens-100k)*(关于 Movielens100k 的最新成果。这些在官方测试集上得到验证)***
3.  **[***https://sifter.org/~simon/journal/20061211.html***](https://sifter.org/~simon/journal/20061211.html)*(西蒙·芬克最著名的博客详述了他的奇异值分解方法)***