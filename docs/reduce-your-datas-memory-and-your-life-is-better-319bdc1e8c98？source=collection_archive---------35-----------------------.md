# 减少数据的内存，生活会更美好

> 原文：<https://towardsdatascience.com/reduce-your-datas-memory-and-your-life-is-better-319bdc1e8c98?source=collection_archive---------35----------------------->

![](img/15605ff62643e0d1cd71b1d9a24e70bf.png)

Wing Tsun Practitioners Silhouette

上面的剪影是两个咏春练习者在做技术练习。请注意它们是如何相互靠近的。咏春是一种武术风格，从最小的动作中获得相同的冲击力。这种风格真正体现了“少即是多”的说法。

# 酷，但这和数据有什么关系？

它与数据的内存使用情况有关。在小数据集，这可能不是一个问题。但是如果你的数据集在内存中太大，你的计算机的性能将会降低。这意味着您的可视化将需要更长的时间来加载，特征工程需要更长的时间来生成，您的机器学习应用程序将需要更长的时间来处理。

减少你数据的内存，你所有有数据的应用都会更流畅。

![](img/b5d3b18e2825493aab2ea50b14c6b5ee.png)

You don’t want to drag your data around. [*Image by Lloyd Morgan on Flickr*](https://www.flickr.com/photos/lloydm/216187948)

# 那么，我们如何减少数据的内存使用呢？

有两种方法可以让你在熊猫身上做到这一点:

1.删除任何不相关的列。

2.更改列的数据类型。

我会用我收集的纽约州几何试题的数据给你看一个例子。

对于那些感兴趣的人来说，这些数据是从公开提供的 pdf 中收集的。数据然后被转移到 Excel，然后转移到 Postgresql。在那里，我使用 Python 的 [Sqlalchemy](https://www.sqlalchemy.org/) 库从 Postgresql 中提取数据，并将其放入数据框中。

我假设你熟悉熊猫的[信息](https://pandas.pydata.org/pandas-docs/version/0.23/generated/pandas.DataFrame.info.html)方法。应用这个，我们得到以下结果:

```
df.info()

RangeIndex: 358 entries, 0 to 357
Data columns (total 7 columns):
id              358 non-null int64
ClusterTitle    358 non-null object
Cluster         358 non-null object
Regents Date    358 non-null object
Type            358 non-null object
DateFixed       358 non-null object
Qnumber         358 non-null object
dtypes: int64(1), object(6)
memory usage: 19.7+ KB
```

比这更准确。如果您应用参数`memory_usage='deep'`，Pandas 将告诉您数据帧使用了多少内存。毕竟，当我们能得到精确的值时，它们比估计值要好。这样做显示了相同的信息，但是内存使用行是不同的。

```
df.info(memory_usage='deep') RangeIndex: 358 entries, 0 to 357
Data columns (total 7 columns):
id              358 non-null int64
ClusterTitle    358 non-null object
Cluster         358 non-null object
Regents Date    358 non-null object
Type            358 non-null object
DateFixed       358 non-null object
Qnumber         358 non-null object
dtypes: int64(1), object(6)
memory usage: 139.9 KB
```

所以我们的数据帧专门用了 139.9 KB 的内存。
我们需要的另一个方法是带有参数`deep=True`的`memory_usage`方法，以查看每一列使用了多少字节的内存。让我们将它应用到我们的数据集，看看使用了多少内存。

```
df.memory_usage(deep=True)Index              80
id               2864
ClusterTitle    38222
Cluster         22713
Regents Date    14320
Type            21122
DateFixed       22554
Qnumber         21392
dtype: int64
```

# 删除列

我们将删除 id 列，因为它是 Postgresql 表使用的列。您不需要它来进行分析。为此，我们将使用来自熊猫的`drop`方法，使用`axis='columns'`和`inplace=True`。我们的`axis`参数确保只删除该列。`inplace`参数保留原始数据帧，不包含我们删除的列。

```
df.drop(['id'],axis='columns',inplace=True)
```

现在让我们看看内存减少了多少。

```
df.info(memory_usage='deep')

RangeIndex: 358 entries, 0 to 357
Data columns (total 6 columns):
ClusterTitle    358 non-null object
Cluster         358 non-null object
Regents Date    358 non-null object
Type            358 non-null object
DateFixed       358 non-null object
Qnumber         358 non-null object
dtypes: object(6)
memory usage: 137.1 KB
```

因此少了 2.8 KB 是一个很小的改进。但是当然，我们想做得更好。现在让我们看看我们可以对每一列的数据类型做些什么。

# 改变列数据类型

回想一下，我们使用了`memory_usage`方法来查看每一列使用的内存量(以字节为单位)。我们还从`info`方法中看到了每个列的类型。其中大部分是物品。在这些列中，有些是可以被视为标签的字符串数据列。我们可以减少标签数据(比如集群列)内存的一种方法是将它改为一种`category`数据类型。

让我们使用`astype`方法将集群列更改为`category`数据类型。(对于上下文:聚类列显示问题的总体主题的代码)

```
df['Cluster']=df['Cluster'].astype('category')
```

现在来看看该列的内存使用量是如何变化的。

```
df.memory_usage(deep=True)

Index              80
ClusterTitle    38222
Cluster          1885
Regents Date    14320
Type            21122
DateFixed       22554
Qnumber         21392
dtype: int64
```

哇！1885 字节比 22713 字节好多了！这意味着内存使用减少了 91.7%。现在我们将把 Type 和 Qnumber(问题编号范围从 1 到 36)列转换成相同的数据类型。如果你在看之前需要练习，你可以自己试试。

```
df['Type']=df['Type'].astype('category')
df['Qnumber']=df['Qnumber'].astype('category')
```

现在让我们看看这对单个列的内存使用有什么影响。

```
df.memory_usage(deep=True)

Index              80
ClusterTitle    38222
Cluster          1885
Regents Date    14320
Type              556
DateFixed       22554
Qnumber          3789
dtype: int64
```

太神奇了！Qnumber 数据类型内存使用减少了 82.8%。Type 列的内存使用大幅减少了 97.3%。现在，查看整个数据帧的整体内存使用量减少情况。请记住，我们的数据帧使用了 137.1 KB 的内存。

```
df.info(memory_usage='deep')

RangeIndex: 358 entries, 0 to 357
Data columns (total 6 columns):
ClusterTitle    358 non-null object
Cluster         358 non-null category
Regents Date    358 non-null object
Type            358 non-null category
DateFixed       358 non-null object
Qnumber         358 non-null category
dtypes: category(3), object(3)
memory usage: 79.5 KB
```

总体而言，我们的数据帧的内存使用减少了 42%。我们将内存使用量减少了将近一半！不错！

# 最终想法和额外资源

内存使用更多的是一个计算机科学的概念，但是请记住，在您进行分析的时候，您仍然在编写代码。减少内存使用将全面提高计算机生成可视化的性能，在进行特征工程时提高新列的生成，并提高机器学习应用程序的处理能力。同样，您可以通过删除不必要的列并更改列的数据类型来减少数据的内存使用。

![](img/a964a01320de752100d1f9b352039e95.png)

Image from Floriana on Flickr.

这更多的是为了减少数据的内存使用。如果你觉得这有帮助(或没有)，你有进一步的问题，然后留下评论，让我知道。

关于减少内存使用的更深入的解释，请看 Josh Delvin 的博客文章。

感谢阅读！

直到下一次，

约翰·德杰苏斯