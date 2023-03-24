# FIFA 19 评分与 PES 2019 评分有显著差异吗？

> 原文：<https://towardsdatascience.com/do-fifa-19-ratings-differ-significantly-from-pes-2019-ratings-5c0a8d5bd55e?source=collection_archive---------30----------------------->

![](img/7d368bee97157354a3bae6e046eb5c0f.png)

## 足球运动员的 FIFA 19 评级与他们的 PES 2019 评级有多相似。

> “GOOOOOAAAAAL。路易斯·苏亚雷斯多么疯狂的进球。”

在国际足联 19 日巴塞罗那足球俱乐部和拜仁慕尼黑之间的激烈比赛的最后一分钟，我打进了一个 30 码的尖叫。比分现在是 3 比 2，我赢了。我欣喜若狂。另一方面，我的对手很失望。

我简直不敢相信苏亚雷斯站在边线附近被三名防守队员挡住时是如何打进那个漂亮的进球的。一个仍然让我困惑的想法是，苏亚雷斯得分是纯粹出于运气，还是因为他有着令人难以置信的高收视率。

而且我在 YouTube 上看了很多 FIFA 19 上打进的世界级进球，去发现什么样的球员打进了最不可思议的进球。当我观看这些进球时，我注意到大多数进球都是由高水平的球员打进的。

然后，YouTube 的推荐算法让我也看了 PES 2019 上类似的进球。我注意到大致相同的一群高知名度的球员也打进了这些精彩的进球。

因此，很容易得出结论，知名度高的球员更有可能在两个最大的足球游戏平台上创造最佳进球。由于这两款游戏的玩家表现有很强的相似性，我很想知道【FIFA 19 和 PES 2019 上玩家的评分是如何密切相关的。

因此，我决定进行一项研究，以找出 FIFA 19 评级和 PES 2019 评级之间是否存在显著差异。

# 获取数据集

我必须获得包含执行项目所需功能的最佳数据集，所以我使用 Kaggle 通过这个[链接](https://www.kaggle.com/karangadiya/fifa19)获得 FIFA 19 数据集，并在这里获得 PES 2019 数据集[。](https://www.kaggle.com/harshkava/pro-evolution-soccer-pes-2019-players-dataset)

使用 IBM Watson Studio 上的 Jupyter 笔记本，数据集被加载并清理，以确保它们包含所有需要的列。

![](img/fc0eb31edb245eb22a7278a2780b1a2f.png)

A snapshot of the first five rows of the PES 2019 transformed dataset.

![](img/503cbdfa1228ce14ca3e7ce1bf6aaf2a.png)

A snapshot of the first five rows of the FIFA 19 transformed dataset.

两个数据集都被重新设计后。运行了多个代码来合并两个数据集，删除重复的行，以降序排列玩家的评级，并重命名列。

以下是用于执行上述过程的一些代码:

要合并两个数据集:

```
df_combo = pd.merge(df_pes2, df_fifa2)

df_combo1 = df_combo
```

要删除重复的行:

```
df_combo2 = df_combo1.drop_duplicates(['Player Name'],keep='first') 
```

要按降序排列合并的数据集并重命名列，请执行以下操作:

```
df_combo3 = df_combo2.sort_values(by = "FIFA Overall Rating", ascending = **False**)
df_combo3.reset_index(inplace = **True**)
df_combo3.drop("index", axis = 1, inplace = **True**)
df_combo3 = df_combo3.rename(columns = {"Overall Rating" : "PES Overall Rating"})
```

显示特定列时合并两个数据集的结果如下所示:

![](img/9b1d6f3676fca7a196f64af092930d3a.png)

The first 20 rows of the merged dataset showing Player’s name, FIFA 19 ratings, PES 2019 ratings and age.

根据合并数据集的前 20 行，与 PES 2019 球员相比，FIFA 19 球员预计将获得更高的评级。

# 探索性分析

对于新形成的数据集，下一步是查看数据集中的重要列如何相互作用。因此，创建了两个正态分布图，显示 PES 2019 评级和 FIFA 19 评级的正态分布情况。然后，构建第三个正态分布图来结合前两个。

![](img/4db8dc751221a2ec66e0b6d2c761d0f0.png)

A normal distribution graph of PES 2019 players’ overall ratings.

![](img/9615784d525495ca68106c8766f8fc0e.png)

A normal distribution graph of FIFA 19 players’ overall ratings.

![](img/7b1e60b3a9b2e27ef163a078e87099d2.png)

A normal distribution graph comparing how both graphs are normally distributed.

根据图表，FIFA 19 的平均评分为 68.19，低于 PES 2019 的平均评分 70.32。然而，FIFA 19 的标准差为 6.84，高于 PES 2019 的标准差 6.13。

标准差的值表明，FIFA 19 球员的评级比 PES 2019 球员的评级更分散。这个结果可以在第三张图中观察到。

# 结果呢

为了找出 PES 2019 评级和 FIFA 19 评级之间是否存在显著差异，进行了一系列任务。首先，创建一个散点图来显示两组之间的相互关系。

![](img/413ad381a04e6a9b6496b7a61fef299e.png)

A scatter plot of PES 2019 ratings vs FIFA 19 ratings.

根据散点图，PES 2019 评级和 FIFA 19 评级之间似乎存在非常积极的关系。然后，计算相关系数和 p 值。下面是用于获得结果的代码和结果本身的快照。

![](img/0861a2ee9d6498c06e4a44cd9cd79dd6.png)

The results of the correlation coefficient and p-value of the FIFA 19 and PES 2019 ratings.

皮尔逊相关系数是 0.8744，因为 0.8744 的值意味着正的强相关，所以是可疑的。p 值约为 0.0。p 值小于 0.05，因此 FIFA 19 评级和 PES 2019 评级之间存在显著差异，即使它们相关性极强。

用于进行这项研究的完整版本代码可以在这里看到**[](https://github.com/MUbarak123-56/DataBEL/blob/master/PES%202019%20RATINGS%20VS%20FIFA%2019%20RATINGS.ipynb)****。******