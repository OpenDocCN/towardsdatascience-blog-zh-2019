# 特征工程与急流:NFL 数据碗

> 原文：<https://towardsdatascience.com/feature-engineering-with-rapids-nfl-data-bowl-c8584bc266b5?source=collection_archive---------31----------------------->

在开始这篇文章之前，我想减轻你对从 pandas 转换到 RAPIDS cudf 的怀疑， **RAPIDS cudf 使用与 pandas 相同的 API！**

RAPIDS 正在将表格数据集上的传统数据科学工作流迁移到 GPU 上。最近， [George Sief](https://towardsdatascience.com/@george.seif94?source=post_page-----9ddc1716d5f2----------------------) 发表了一篇[关于走向数据科学](/heres-how-you-can-speedup-pandas-with-cudf-and-gpus-9ddc1716d5f2)的文章，表明 RAPIDS cudf 库可以在 5.12 毫秒内计算包含 1 亿行的给定列的平均值，而在 pandas 上则需要 82.2 毫秒。本文将在 [Kaggle NFL 数据碗挑战赛](https://www.kaggle.com/c/nfl-big-data-bowl-2020)的特性工程背景下，进一步探讨 RAPIDS 和 cudf 实现的加速。我们在使用来自 Digital Storm 的 2 个 NVIDIA Titan RTX 24GB GPU 的 [Data Science PC 上，几乎每个设计的功能都实现了至少 10 倍的速度提升。](https://www.digitalstorm.com/nvidia-data-science.asp)

在本文中，我们将从 NFL 数据碗数据集构建新的要素，如身体质量指数、防守方向与距离、余量和紧急程度。我们还将通过将特性列中的每个值除以特性的最大值来标准化特性。在每种情况下，我们都将对 pandas 和 RAPIDS cudf 库进行基准测试！

你可以从本文[中的代码访问 Jupyter 笔记本，这里](https://github.com/CShorten/RAPIDS-ETL/blob/master/Rapids-vs-Pandas-FeatureEngineering.ipynb)我也制作了一个[视频来解释这个](https://www.youtube.com/watch?v=A9lgUwA8RrY&t=9s)如果你想继续下去的话！

![](img/b9f60f708817d1fee192985fc75cb010.png)

首先，我们将 NFL 数据碗数据集的大小扩大 16 倍，以展示 RAPIDS / cudf 在大型数据集上的强大功能。之所以选择 NFL 数据集，是因为它便于理解特征工程的概念。RAPIDS 确实最适合拥有数百万条记录的大型数据集。

```
# Increase the dataset size by stacking it on top of itself
pd_data = pd.concat([pd_data, pd_data], ignore_index=True)
# repeated 3 more times
```

现在我们将把熊猫数据帧转换成 cudf 和 dask-cudf 数据帧！

```
import cudf
import dask_cudf
cudf_data = cudf.from_pandas(pd_data)
dask_data = dask_cudf.read_csv('./cleaned_data.csv')
```

上面的代码突出了 RAPIDS 最好的部分之一。它完全模仿 pandas API，因此数据科学家不必担心迁移到新语法会带来的头痛。在这段代码中，我们用。from_pandas()语法和 dask 文档建议您直接从 csv 文件加载数据，所以我们这样做。

# 列值平均值

根据最近从 [George Sief 开始的关于数据科学的教程](/heres-how-you-can-speedup-pandas-with-cudf-and-gpus-9ddc1716d5f2)，我们观察了计算数据帧中给定列的平均值所花费的时间:

```
pd_data['PlayerHeight'].mean() # 29.1 ms
cudf_data['PlayerHeight'].mean() # 466 µs
dask_data['PlayerHeight'].mean() # 1.46 ms
```

在这种情况下，我们看到 cudf 比 pandas 快了大约 60 倍。Cudf 在这里的表现优于 dask-cudf，但我怀疑这是因为 dask-cudf 对数据集进行了更好的优化，这种数据集一开始就不可能与 pandas 进行比较。

# 防守者进入禁区 vs 距离

特色创意致谢:[https://www . ka ggle . com/CP mpml/initial-wrangling-Voronoi-areas-in-python](https://www.kaggle.com/cpmpml/initial-wrangling-voronoi-areas-in-python)

这个功能将会查看与防守方向相比第一次进攻所需的码数。

```
pd_data['DefendersInTheBox_vs_Distance'] = pd_data['DefendersInTheBox'] / pd_data['Distance'] # 36.9 mscudf_data['DefendersInTheBox_vs_Distance'] = cudf_data['DefendersInTheBox'] / cudf_data['Distance'] # 3.1 ms
```

# 身体质量指数

特色创意致谢:[https://www . ka ggle . com/bgmello/neural-networks-feature-engineering-for-the-win](https://www.kaggle.com/bgmello/neural-networks-feature-engineering-for-the-win)

该功能将根据身高和体重计算跑步者的身体质量指数。这是我们获得最低加速的特性。

```
pd_data['BMI'] = 703 * (pd_data['PlayerWeight']/pd_data['PlayerHeight']**2) # 64.3 mscudf_data['BMI'] = 703 * (cudf_data['PlayerWeight'] / cudf_data['PlayerHeight']**2) # 15.7 ms
```

# 边缘

这个功能将计算得分差异，也许如果球队赢/输了一些金额，他们更有可能成功运行球。

```
pd_data['Margin'] = pd_data['HomeScoreBeforePlay'] - pd_data['VisitorScoreBeforePlay'] # 32 mscudf_data['Margin'] = cudf_data['HomeScoreBeforePlay'] - cudf_data['HomeScoreBeforePlay'] # 3.46 ms
```

# 紧急

此功能将计算季度利润给出的紧急程度。也许如果现在是第四节，比赛接近尾声，球员会把球跑得更远。

```
pd_data['Urgency'] = pd_data['Quarter'] * pd_data['Margin'] # 33.6ms
cudf_data['Urgency'] = pd_data['Quarter'] * pd_data['Margin'] #3.4ms
```

# 最大值标准化

一种惰性规范化技术是将一列中的所有值除以该列中的最大值。这样做是为了在[0，1]之间缩放值，并便于训练机器学习模型。

```
for col in pd_data:
  pd_data[col] /= pd_data[col].max() # 1.28 sfor col in cudf_data:
  cudf_data[col] /= cudf_data[col].max() # 117 ms
```

# 急流城特色工程:总结性思考

本笔记重点介绍了一些例子，在这些例子中，RAPIDS 实现了 ETL 操作的大规模加速，比如特性工程和标准化。这对于电子健康记录、保险数据、生物信息学和其他各种最好用表格数据格式描述的数据领域的应用程序来说是非常令人兴奋的！感谢阅读，请查看下面解释这一点的视频:

[https://www.youtube.com/watch?v=A9lgUwA8RrY&t = 13s](https://www.youtube.com/watch?v=A9lgUwA8RrY&t=13s)