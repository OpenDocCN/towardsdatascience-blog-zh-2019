# DBSCAN 算法:Python Scikit 的完整指南和应用-学习

> 原文：<https://towardsdatascience.com/dbscan-algorithm-complete-guide-and-application-with-python-scikit-learn-d690cbae4c5d?source=collection_archive---------2----------------------->

## 聚类空间数据库

![](img/02e093df63d436c93fbc973adafff72c.png)

Density Based Clustering ? (Picture Credit: [Adil Wahid](https://unsplash.com/photos/nmF_6DxByAw))

我打算在这篇文章中讨论的是—

*   DBSCAN 算法步骤，遵循 Martin Ester 等人的原始研究论文[1]
*   直接密度可达点的关键概念，用于划分集群的核心点和边界点。这也有助于我们识别数据中的噪声。
*   使用 python 和 scikit 的 DBSCAN 算法应用示例——通过根据每年的天气数据对加拿大的不同地区进行聚类来学习。学习使用一个奇妙的工具-底图，通过 python 在地图上绘制 2D 数据。所有的代码(用 python 写的)，图片(用 Libre Office 制作的)都可以在 github 中找到(文章末尾给出了链接)。

在处理不同密度、大小和形状的空间聚类时，检测点的聚类可能是具有挑战性的。如果数据包含噪声和异常值，任务会更加复杂。为了处理大型空间数据库，Martin Ester 和他的合著者提出了带噪声的应用程序的[基于密度的空间聚类](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf) (DBSCAN)，这仍然是引用率最高的科学论文之一。3 使用 Ester 等人的算法的主要原因是

> 1.它需要最少的领域知识。
> 
> 2.它可以发现任意形状的星团。
> 
> 3.适用于大型数据库，即样本数量超过几千个。

## 1.DBSCAN 算法中的定义:

为了更详细地理解 DBSCAN，让我们深入了解一下。***DBS can 算法的主要概念是定位被低密度区域彼此隔开的高密度区域。*** 那么，我们如何测量一个地区的密度呢？以下是两个步骤—

*   点 p 处的密度:从点 P *开始半径为 *Eps (ϵ)* 的圆内的点数。*
*   密集区域:对于簇中的每个点，半径为ϵ的圆至少包含最小数量的点( *MinPts* )。

数据库 *D* 中的点 P 的ε邻域被定义为(遵循 Ester 等人的定义)

N (p) = {q ∈ D | dist(p，q) ≤ ϵ} …(1)

根据密集区域的定义，如果|N (p)|≥ MinPts，则一个点可以被分类为 ***核心点*** *。*顾名思义，核心点通常位于集群内部。*一个* ***边界点*** *在其ϵ-neighborhood (N)内的个数少于 MinPts，但它位于另一个核心点的邻域内。* ***噪声*** *是既不是核心点也不是边界点的任意数据点。*见下图更好理解。

![](img/c3e624ba340e7b704ef57368ae6006c7.png)

Core and Border Points in a Database D. Green data point is Noise. (Source: Re-created by Author, Original Reference [2])

这种方法的一个问题是，边界点的ϵ-neighborhood 包含的点的数量明显少于核心点。由于 *MinPts* 是算法中的一个参数，将其设置为较低的值以包括聚类中的边界点会导致消除噪声的问题。这里出现了*密度可达和密度连接点*的概念。

**直接密度可达**:数据点 *a* 是从点 *b* 直接密度可达，如果—

1.  *| N(b)|≥min pts；即 b 是核心点。*
2.  *a ∈ N(b)即 a 在 b 的ε邻域内*

考虑到边界点和核心点，我们可以理解直接密度可达的概念是不对称的，因为即使核心点落在边界点的ε邻域中，边界点也没有足够的 *MinPts，*因此不能满足这两个条件。

**密度可达:**点 *a* 是从 b 点相对于ϵ和 *MinPts* 可达的密度，如果—

![](img/a34e0ff8b7ec2654b6f733f54cb0efda.png)

密度可达本质上是传递的，但是，就像直接密度可达一样，它是不对称的。

**密度连通:**可能存在这样的情况，当两个边界点将属于同一个聚类，但是它们不共享一个特定的核心点，那么我们说它们是密度连通的，如果存在一个公共核心点，从该公共核心点这些边界点是密度可达的。如你所知，密度连接是对称的。Ester 等人论文中的定义如下—

> “点 **a** 是相对于ϵ和 MinPts 连接到点 **b** 的密度，如果有一个点 **c** 使得 **a** 和 **b** 都是从 **c** w.r.t .到ϵ和 MinPts 的可达密度。”

![](img/7517e99131d11cf1f622546c82259c62.png)

Two border points a, b are density connected through the core point c. Source: Created by Author

## 2.DBSCAN 算法的步骤:

有了上面的定义，我们可以按如下步骤进行 DBSCAN 算法—

1.  该算法从没有被访问过的任意点开始，并且从ϵ参数中检索其邻域信息。
2.  如果该点包含ϵ邻域内的 *MinPts* ，则集群形成开始。否则该点被标记为噪声。稍后可以在不同点的ϵ邻域内找到该点，因此可以使其成为聚类的一部分。密度可达和密度连接点的概念在这里很重要。
3.  如果发现一个点是核心点，那么ϵ邻域内的点也是聚类的一部分。因此，在ϵ邻域内找到的所有点都将被添加，如果它们也是核心点的话，还有它们自己的ϵ邻域。
4.  上述过程继续，直到完全找到密度连接的聚类。
5.  该过程从新的点重新开始，该新的点可以是新群的一部分或者被标记为噪声。

从上面的定义和算法步骤，你可以猜出*DBS can 算法*的两个最大缺点。

*   如果数据库中的数据点形成了不同密度的聚类，那么 DBSCAN 无法很好地对数据点进行聚类，因为聚类取决于ϵ和 *MinPts* 参数，不能为所有聚类单独选择它们。
*   如果数据和特征没有被领域专家很好地理解，那么设置ϵ和 *MinPts* 可能会很棘手，可能需要用不同的ϵ和 *MinPts* 值进行几次迭代比较。

一旦基本原理清楚了一点，现在将看到一个使用 Scikit-learn 和 python 的 DBSCAN 算法的例子。

## 3.带 Scikit-Learn 的 DBSCAN 算法示例:

为了查看 DBSCAN 算法的一个实际例子，我使用了加拿大 2014 年的天气数据来对气象站进行聚类。首先让我们加载数据—

![](img/ae003cbf6806923815ea5dea7902eba5.png)

数据帧由 1341 行和 25 列组成，为了理解列名代表什么，让我们看看下面最重要的特性

![](img/a8fe2d1fb4273e47b3bf2df03e2cad22.png)

因为，我想使用不同的温度作为第一次尝试聚类气象站的几个主要特征，首先，让我们删除“平均温度(Tm)”、“最低温度(Tn)”和“最高温度(Tx)”列中包含 NaN 值的行。

![](img/73b7a3f73972aced4b8382a5dc700f54.png)

在删除上述列中包含 NaN 值的行后，我们剩下 1255 个样本。尽管这几乎是 7%的数据损失，但是考虑到我们仍然有超过 1000 个样本，让我们继续进行聚类。

由于我们要进行空间聚类，并在地图投影中查看聚类，以及不同的温度(“Tm”、“Tn”、“Tx”)、“Lat”、“long”也应作为特征。在这里，我使用了[底图工具包](https://matplotlib.org/basemap/users/intro.html)，这是一个用于绘制 2D 数据的库，以便用 Python 可视化地图。如底图文档中所述— *“底图本身不进行任何绘制，但提供了将坐标转换为 25 种不同地图投影之一的工具”。*底图的一个重要属性是-使用参数纬度/经度(以度为单位，如我们的数据框中所示)调用底图类实例，以获取 x/y 地图投影坐标。

让我们使用底图绘制气象站，以便熟悉它。从导入必要的库开始

```
from mpl_toolkits.basemap import Basemap
import matplotlib
from PIL import Image
import matplotlib.pyplot as plt
from pylab import rcParams
%matplotlib inline
rcParams['figure.figsize'] = (14,10)
```

我们现在准备调用底图类—

![](img/afb1dafa531508b288c6b433c812f333.png)

让我简单解释一下代码块。我开始使用[墨卡托投影](https://en.wikipedia.org/wiki/Mercator_projection) ( *投影=‘merc’*)、低分辨率(*分辨率=‘l’*)调用一个底图类实例，地图域的边界由 4 个参数给出 *llcrnrlon、llcrnrlat、urcrnrlon、urcrnrlat、*，其中 *llcrnrlon* 表示所选地图域左下角的经度，依此类推。*绘制海岸线，绘制国家*顾名思义，*绘制遮罩*绘制高分辨率陆海遮罩图像，将陆地和海洋颜色指定为橙色和天蓝色。使用下面的命令将纬度和经度转换为 x/y 地图投影坐标—

```
xs, ys = my_map(np.asarray(weather_df.Long), np.asarray(weather_df.Lat))
```

这些地图投影坐标将用作要素，以便在空间上将数据点与温度一起聚类。首先让我们看看下面的气象站—

![](img/5953fa50cf8c0562721b5cd90fae3715.png)

Weather Stations in Canada, plotted using Basemap. Source: Author

## 3.1.对天气数据进行聚类(温度和坐标作为特征)

对于聚类数据，我遵循了[sci kit-DBS can](https://scikit-learn.org/stable/auto_examples/cluster/plot_dbscan.html#sphx-glr-auto-examples-cluster-plot-dbscan-py)的学习演示中所示的步骤。

![](img/244f187fd2b3457201a48bf4faccee54.png)

选择温度(' Tm '，' Tx '，' Tn ')和坐标的 x/y 映射投影(' xm '，' ym ')作为特征，并将ϵ和 *MinPts* 分别设置为 0.3 和 10，给出 8 个唯一的聚类(噪声标记为-1)。您可以随意更改这些参数来测试集群会受到多大的影响。

让我们使用底图来可视化这些集群—

![](img/09076896328705f543249c81d7c2d693.png)

8 Unique Clusters in Canada Based on Few Selected Features in the Weather Data. ϵ and MinPts set to 0.3 and 10 Respectively. Source: Author

最后，我将降水(“p”)包含在特征中，并重复相同的聚类步骤，将ϵ和 *MinPts* 设置为 0.5 和 10。我们看到了与以前的聚类的一些不同，因此它给了我们一个思路，当我们缺乏领域知识时，即使使用 DBSCAN 也可以对无监督数据进行聚类。

![](img/c34d0b084964986ab94c783190761472.png)

4 Unique Clusters in Canada Based on Selected Features (now included precipitation compared to previous case) in the Weather Data. ϵ and MinPts set to 0.5 and 10 Respectively. Source: Author.

您可以尝试重复这个过程，包括一些更多的功能，或者，改变聚类参数，以获得更好的整体知识。

最后，我们介绍了 DBSCAN 算法的一些基本概念，并测试了该算法对加拿大气象站的聚类。详细代码和所有图片将在我的 [Github](https://github.com/suvoooo/Machine_Learning/tree/master/DBSCAN_Complete) 中提供。可以与 K-均值聚类进行直接比较，以便更好地理解这些算法之间的差异。希望这将有助于您开始使用一种最常用的聚类算法来处理无监督问题。

分享你的想法和主意，保持坚强。干杯！

页（page 的缩写）s:另一个无监督聚类方法高斯混合和详细的[期望最大化算法](/latent-variables-expectation-maximization-algorithm-fb15c4e0f32c)在另一个帖子中讨论。

**参考文献:**

[1] *“一种基于密度的带噪声大型空间数据库聚类发现算法”*；马丁·埃斯特等人 [KDD-96 会议录。](https://www.aaai.org/Papers/KDD/1996/KDD96-037.pdf)

[2]基于密度的聚类方法；高，j。布法罗大学副教授。[演示链接。](https://cse.buffalo.edu/~jing/cse601/fa12/materials/clustering_density.pdf)

[3] [链接到 Github](https://github.com/suvoooo/Machine_Learning/tree/master/DBSCAN_Complete) ！

***如果你对更深入的基础机器学习概念感兴趣，可以考虑加盟 Medium 使用*** [***我的链接***](https://saptashwa.medium.com/membership) ***。你不用额外付钱，但我会得到一点佣金。感谢大家！！***