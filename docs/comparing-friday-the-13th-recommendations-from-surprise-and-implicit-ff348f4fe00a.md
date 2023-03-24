# 比较星期五第 13 次建议从惊讶和含蓄

> 原文：<https://towardsdatascience.com/comparing-friday-the-13th-recommendations-from-surprise-and-implicit-ff348f4fe00a?source=collection_archive---------29----------------------->

## 帮助你计划下一个电影之夜的功能代码。

在最近的一篇[博客文章](https://medium.com/@jmcneilkeller/item-item-recommendation-with-surprise-4bf365355d96)中，我详细介绍了如何令人惊讶地进行单品推荐。我选择惊喜，部分是因为它是我训练营的一些导师强烈推荐的(没有双关语)，但我完全意识到还有很多其他选择，其中一些我在我的上一篇推荐帖子中详细介绍过。我认为从隐式开始深入研究我已经跳过的其他一些库是值得的。

Implicit 是一个库，旨在为其他一些已建立的推荐库(如 Annoy 和 NMSLib)提供更快的 Python 实现，所以它似乎是一个不错的起点。我将仅仅触及这个库所提供的皮毛，所以请随意查看这里的文档和这里的 GitHub

现在评估 Surprise 和 Implicit 各自推荐的质量有点武断。没有一些用户的反馈，这完全取决于我的主观意见，哪一个做得更好。本着这种精神，我将把重点放在提取建议的代码上。

在我们推荐任何东西之前，我们首先要构建并运行我们的模型。我已经在之前的博客中详细介绍了如何构建和运行惊喜(这里是[再次是](https://medium.com/@jmcneilkeller/item-item-recommendation-with-surprise-4bf365355d96))，所以我将在这里运行隐式。

首先，我们必须导入数据。

```
import implicitdf = pd.read_csv("final_data.csv", index_col=0) 
# This dataframe has the movie titles, which I'll need later.
```

与 Surprise 不同，Implicit 特别需要一个稀疏矩阵。令人惊讶的是，假设您只传递 userid、itemid 和 weight(或 rating)列，那么您可以将数据帧直接传递给模型。但是使用 Implicit，您需要首先转换您的数据。

```
df_sparse = pd.read_csv("item_item_final.csv", index_col=0)
sparse = scipy.sparse.csr_matrix(df_sparse.values)
```

当谈到模型时，Implicit 有很多选项，包括 onyy、NMSLib 和 Faiss 的版本，但出于说明的目的，我将坚持使用良好的 ole 交替最小二乘法。

```
als_model = implicit.als.AlternatingLeastSquares()
als_model.fit(sparse)
```

现在是有趣的部分——我们可以推荐一些电影。先说惊喜。我会选择单品推荐，这很简单，但会给你带来惊喜。

惊喜的好处在于，无论你的 item _ ids 是什么，都可以是字符串。在这种情况下，这非常有用，因为最终我想要吐出的是一个易于阅读的电影名称列表。Surprise 通过将“原始 id”(即电影名称)转换为“内部 id”(数字)来处理这些字符串。这使得你可以相对容易地将你的标题从模型中取出来。

```
# First, select the title you are recommending for:
hd_raw_id = 'Friday the 13th Part VII: The New Blood (1988)'#Transform that raw_id to an inner_id.
hd_inner_id = KNN_final_model.trainset.to_inner_iid(hd_raw_id)# Get nearest neighbors. 
hd_neighbors = KNN_final_model.get_neighbors(hd_inner_id, k=5)# Convert inner ids of the neighbors back into movie names.
hd_raw = [KNN_final_model.trainset.to_raw_iid(hd_inner_id)
                       for hd_inner_id in hd_neighbors]
```

以下是我的建议:

```
['Friday the 13th Part VIII: Jason '
 'Takes Manhattan (1989)',
 'Leatherface: Texas Chainsaw Massacre '
 'III (1990)',
 'Amityville: A New Generation (1993)',
 'Jaws 3-D (1983)',
 'Texas Chainsaw Massacre: The Next '
 'Generation (a.k.a. The Return of the '
 'Texas Chainsaw Massacre) (1994)']
```

不算太差！我们有所有的恐怖电影，所有的续集，它包括另一个 13 号星期五，这是我的数据集的一部分。至少从表面上看，这是有道理的。

现在为含蓄。得到相似的物品，其实在隐性上是极其简单的。您只需调用“相似项”函数。代码如下:

```
sim_items = als_model.similar_items(70)
```

诀窍是返回一些可理解的东西，尤其是对最终用户。隐式返回 n 个最近邻居的行 id。当然，没有人愿意被推荐一些行 id。这意味着，对于实际应用，您将需要一个助手函数。这是我想到的:

```
def get_item_recs(movie):
    """Takes a movie row from the model and makes a five recommendations.""" 
    id_list = []
    sim_items = als_model.similar_items(movie,5)
    for item in sim_items:
        iid = df_sparse.iloc[item[0],[1]]
        id_list.append(int(iid[0]))
    final = []
    for rec in id_list:
        final.append(df.loc[df['movieId']== rec, 'title'].iloc[0])
    return final
```

这是清单:

```
['Friday the 13th Part VIII: Jason Takes Manhattan (1989)',
 'Iron Eagle IV (1995)',
 'Lawnmower Man 2: Beyond Cyberspace (1996)',
 'Troll 2 (1990)',
 'Problem Child 2 (1991)']
```

这份清单比《惊奇》多了一点多样性，但仍然相当不错。同样，另一个 13 号星期五是它的一部分，它都是续集。榜单上还有铁鹰四和问题儿童 2，有意思。

显然，我没有真正调整 ALS 模型的参数，Implicit 还有许多我没有包括在内的特性和功能，但有趣的是拍摄的不同之处。