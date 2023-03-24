# SVD:模型调优哪里出错了

> 原文：<https://towardsdatascience.com/svd-where-model-tuning-goes-wrong-61c269402919?source=collection_archive---------13----------------------->

## 使用 Python 中的 Surprise 库探索 SVD 推荐系统的超参数调整的限制。

![](img/4ec2c0a8fb2b6924449e05e3a866355a.png)

Not this type of tuning :) Credit: T Millot

# **简介**

有许多方法可以建立推荐系统。 ***潜在因素模型*** 是一系列寻求给定评级矩阵低维分解的技术。 ***SVD*** 是一个潜在因素模型的名称，该模型涉及通过 ***随机梯度下降*** 进行参数估计，由 Simon Funk 在电影推荐的背景下流行起来。本文展示了使用 Python 中的 Surprise 库在 Movie-Lens 100k 数据集上构建和测试一个 ***SVD 推荐系统*** 的过程，重点是超参数调优。

# **设置**

**1 —数据集先决条件**

```
**from** surprise **import** Datasetdata = Dataset.load_builtin(‘ml-100k’)
```

Surprise 是一个 scikit 包，用于构建和分析由 Nicolas Hug 维护的推荐系统。阅读它的文档页面，软件包的一个目标是*“减轻数据集处理的痛苦”*。一种方法是通过内置数据集。Movie-Lens 100k 就是其中的一个数据集，只需上面的简短命令就可以调用它。

```
**from** surprise **import** Dataset, Reader
**import** csv

ratingsPath = '[Your directory]/ratings.csv'reader = Reader(line_format='user item rating timestamp', sep=**','**, skip_lines=1)
data = Dataset.load_from_file(ratingsPath, reader=reader)
```

自定义数据集也可以从文件或 pandas dataframe 加载，在这两种情况下，必须在 Surprise 解析数据集之前定义一个 [Reader 对象](https://surprise.readthedocs.io/en/stable/reader.html#surprise.reader.Reader)。根据我的经验，只要数据集每行包含一个等级，Reader 对象就是可靠的，格式如下，其中顺序和分隔符是任意定义的，时间戳是可选的。实际上，大多数数据集没有这种结构，所以可能需要进行一些删减。

```
user:item:rating:timestamp
```

**2 —调整超参数**

```
**from** surprise **import** SVD,NormalPredictor
**from** surprise.model_selection **import** GridSearchCVparam_grid = {**'n_factors'**:[50,100,150],**'n_epochs'**:[20,30],  **'lr_all'**:[0.005,0.01],**'reg_all'**:[0.02,0.1]}gs = GridSearchCV(SVD, param_grid, measures=[**'rmse'**], cv=3)
gs.fit(data)params = gs.best_params[**'rmse'**]svdtuned = SVD(n_factors=params[**'n_factors'**], n_epochs=params[**'n_epochs'**],lr_all=params[**'lr_all'**], reg_all=params[**'reg_all'**)
```

数据集准备好后，第二步是构建 ***SVD*** 算法。Surprise 利用 scikit-learn 的 *GridSearchCV()* 和*randomsedsearchcv()*方法，只用几行代码就返回了一组经过调整的参数，绕过了调整超参数的手动试错过程。

我们使用上面的网格搜索方法。一旦在 *param_grid* 字典中指定了超参数和潜在值的数组， *GridSearchCV()* 计算在 ***k 倍交叉验证数据集*** 上超参数的每个组合的分数，并返回最小化跨倍平均分数的参数集。折叠次数和分数都可以由用户选择(超超参数？！)—我们使用 3 倍和 ***RMSE*** 准确度得分。

和很多机器学习算法一样， ***SVD*** 有很多运动部件。我们选择因子的数量、时期的数量、所有参数的单一学习率和单一正则化常数，但是完整列表可以在[库文档中找到。](https://surprise.readthedocs.io/en/stable/matrix_factorization.html#surprise.prediction_algorithms.matrix_factorization.SVD)

最后一步是使用 Surprise 的 *SVD()* 函数，手动覆盖优化的超参数。

***SVD*** 的有效性源于*线性代数*中一个名为*矩阵分解*的主题，并被修改以说明评级矩阵的稀疏特性。推导留给读者作为练习，但是对于那些方法论新手来说，Funk 的文章[是必读的。既有娱乐性又有见地！](https://sifter.org/~simon/journal/20061211.html)

**3 —评估框架**

```
**from** EvaluatorScript **import** Evaluator **from** surprise **import** Dataset,SVD,NormalPredictor
**from** surprise.model_selection **import** GridSearchCVdata = Dataset.load_builtin(‘ml-100k’)terminator = Evaluator(data)param_grid = {**'n_factors'**: [50,100,150],**'n_epochs'**: [20,30], **'lr_all'**: [0.005,0.01],**'reg_all'**:[0.02,0.1]}gs = GridSearchCV(SVD, param_grid, measures=[**'rmse'**], cv=3)
gs.fit(data)params = gs.best_params[**'rmse'**]svdtuned = SVD(n_factors=params[**'n_factors'**], n_epochs=params[**'n_epochs'**],lr_all=params[**'lr_all'**], reg_all=params[**'reg_all'**])
svd = SVD()
random = NormalPredictor()terminator.addModel(svdtuned,**'SVDtuned'**)
terminator.addModel(svd,**'SVD'**)
terminator.addModel(random,**'Random'**)

terminator.evaluateModel()
```

设置的最后一步是构建一个评估框架，如下图所示。这一步是个人喜好。我喜欢构建一个*评估器*对象，

a)保存数据集及其所有分割(训练/测试、留一交叉验证等)。

b)通过*持有模型对象。addModel()* 方法。

c)通过*评估模型。evaluateModel()* 方法。简而言之这叫*。*和*。test()* 使用一致的数据集分割，对对象方法建模，并根据一组性能指标评估预测。

在上面的代码摘录中，我创建了一个继承自*评估脚本*的*评估器*对象的实例，并将其命名为*终结器。*通过输入数据集初始化*终结符*后，使用 Surprise 构建并添加了三个模型对象。我们已经看到了上面的 ***调整过的 SVD*** 模型，但是剩下的两个模型是一个 ***未调整的 SVD*** 模型和一个 ***随机*** 模型。

我发现这个框架是超级通用的，因为我可以向它抛出大多数模型对象，无论是令人惊讶的模型还是我自己的模型类。他们只需要*。契合()*和*。test()* 方法和一致的预测输出格式。

Evaluator 对象及其所有依赖脚本(包括一个*度量*脚本)位于一个 Github 文件夹中(此处[为](https://github.com/93tilinfinity/svd-recommender))。顺便说一句，我很想听听你是如何构建你的机器学习脚本的——我总是乐于听取关于实现的不同观点！

# 结果

在 python 中 Surprise 库的帮助下，我们拟合了一个 ***调优的 SVD*** 模型，一个 ***未调优的 SVD*** 模型和一个随机模型。而 ***调优的 SVD*** 模型是我们的重点，我们使用 ***未调优的 SVD*** 模型和一组随机生成的推荐进行相对比较。

调谐 SVD 超参数:

```
{'n_factors': 150, 'n_epochs': 75, 'lr_all': 0.01, 'reg_all': 0.1}
```

未调整的 SVD(默认)超参数:

```
{'n_factors': 100, 'n_epochs': 20, 'lr_all': 0.005, 'reg_all': 0.02}
```

准确度结果:

```
 MAE      RMSE     Time (s)
SVDtuned  0.6487   0.8469   199.3  
SVD       0.6709   0.8719   134.4
Random    1.1359   1.4228   109.4
```

调整后的模型在***【MAE】***和 ***RMSE*** 上大大优于随机生成的推荐，也略微优于未调整的模型。太好了！现在，为了了解这种性能如何转化为*实际的电影推荐*，我们在数据集中挑选出一个用户——用户 56。

```
 — — — — — -
User 56 Total Ratings: 46User 56 5 Star Ratings:
Ace Ventura: When Nature Calls (1995)
Seven (a.k.a. Se7en) (1995)
Dumb & Dumber (Dumb and Dumber) (1994)
Ace Ventura: Pet Detective (1994)
Forrest Gump (1994)
Lion King, The (1994)
Jurassic Park (1993)
Silence of the Lambs, The (1991)
 — — — — — -
```

目测用户的 5 星级电影，一些明显的主题出现了——喜剧、犯罪和戏剧。这让我们了解了测试用户的偏好，所以让我们看看每个模型是如何理解这个用户的偏好的。

```
— — — — — -
**SVDtuned** 
User's Top Recommendations:
Belle Epoque (1992)
Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion GekijÃ´-ban: Air/Magokoro wo, kimi ni) (1997) 
Adam's Rib (1949) 
Dad's Army (1971)
Three Billboards Outside Ebbing, Missouri (2017)
Jetee, La (1962) 
Enter the Void (2009)
Seve (2014) 
Cosmos 
Saving Face (2004)
— — — — — -
**SVD**
User's Top Recommendations:
Shining, The (1980)
Guess Who's Coming to Dinner (1967)
Raiders of the Lost Ark (Indiana Jones and the Raiders of the Lost Ark) (1981)
Snatch (2000)
On the Waterfront (1954)
Good Will Hunting (1997)
Matrix, The (1999)
Kiss Kiss Bang Bang (2005)
Hoop Dreams (1994)
Sunset Blvd. (a.k.a. Sunset Boulevard) (1950)
— — — — — -
**Random**
User's Top Recommendations:
Saint, The (1997)
Dish, The (2001)
Bad Day at Black Rock (1955)
Cry_Wolf (a.k.a. Cry Wolf) (2005)
Osmosis Jones (2001)
Neon Genesis Evangelion: The End of Evangelion (Shin seiki Evangelion GekijÃ´-ban: Air/Magokoro wo, kimi ni) (1997)
Sgt. Bilko (1996)
O.J.: Made in America (2016)
National Velvet (1944)
Reckless Kelly (1994)
-----------
```

*警告:调谐出错！*

关于调优模型建议的第一个想法—晦涩难懂。而且不是好的方面。

调谐模型的最佳推荐电影出现在我们用户的档案之外。虽然某种程度的多样性是好的(无论是通过国际电影还是其他方式)，但用户的低熟悉度往往令人不快。就了解用户的偏好而言，我无法区分调优模型和随机模型。事实上，我并不认可任何热门推荐电影……但这可能更能说明我的情况，而不是推荐系统。

然而，未调整模型的最佳推荐很大程度上符合用户 56 的简档。这份名单似乎在识别用户对戏剧的偏好方面做得很好，同时也融合了犯罪和喜剧。我印象深刻地看到推荐电影的发行日期的广泛范围，特别是考虑到用户首选列表中的紧张范围。

作为一种方法，奇异值分解显然有潜力，但输出质量仍然对超参数非常敏感。到目前为止，我们的判断是针对数据集中的单个用户。我们可以使用一些*‘beyond’*指标来分析全球绩效。

```
 HitRate   5-StarHitRate  Diversity
SVDtuned   0.0066      0.0163       0.7269  
SVD        0.0295      0.0569       0.0319
Random     0.0115      0.0081       0.0496
```

在构建默认模型和我们认为的优化模型之间的某个地方，我们在整个数据集上的建议质量下降了。从*点击率的下降可以明显看出这一点——*这是一个迹象，表明该模型正在努力从可用的项目中识别用户偏好。

由于*点击率*是在 ***留一交叉验证的*** 数据集 ***，****5-star HitRate*简单来说就是 5 星评级的留一电影的点击率。一个更精细的点击率指标，如果你愿意，因为我们真的只关心捕捉用户的偏好。虽然这两个 SVD 模型在识别首选电影方面更好，但是额外的 50 个因素、55 个时期、更高的学习率和更高的惩罚项每个都导致了更差的“调整”模型。

*为什么？我们为了在网格搜索过程中调整参数而优化的分数与我们评估模型所依据的分数不一致。*

# 讨论

Surprise library 让我们可以直接进入数据集的优化模型，这可能会节省数小时的手动工作。然而，它确实遇到了一个关键的障碍。遗憾的是，不能提供真正重要的分数使得内置的调整功能对推荐系统毫无用处。正如我们在上面看到的，使用股票准确性指标是不够的。事实上，他们是 ***调 SVD*** 模式的败落者。

模型构建不是一个有效的寻求真相的过程，但它是为你的用户提供价值的阶梯上的重要一步。为了获得最佳、可用的模型，能够快速启动工作模型、有效调整超参数并进行明确分析非常重要。无论应用程序如何，确保整个建模过程符合适当的性能指标是至关重要的。

github:【https://github.com/93tilinfinity/svd-recommender 

欢迎评论和反馈！