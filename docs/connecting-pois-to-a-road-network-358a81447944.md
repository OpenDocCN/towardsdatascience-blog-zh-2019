# 将兴趣点连接和插值到道路网络

> 原文：<https://towardsdatascience.com/connecting-pois-to-a-road-network-358a81447944?source=collection_archive---------12----------------------->

## 基于最近边的可伸缩插值

```
This article discusses the process of handling the integration of point geometries and a geospatial network. You may find a demonstration Jupyter Notebook and the script of the function [here](https://github.com/ywnch/toolbox).
```

![](img/bdaecf3a9bb2796bd7bb73f0ed6bfb26.png)

# 简介:使用地理空间网络

当我们使用网络分析时，我们的分析对象(表示为节点或顶点)应该通过一个图连接起来，并定义邻接和边权重。然而，我们经常从不同的来源收集数据，并需要手动集成它们。这个过程实际上比我想象的要复杂一些。

让我们看看这个例子。在分析城市中的兴趣点(poi)的场景中，我们可能从 OpenStreetMap 中检索到一个步行网络，从 Yelp 中查询到一组餐馆。一种常见的方法是将 poi 捕捉到网络上最近的顶点(结点、交汇点、道路弯道)。虽然这是一种非常简单且高效的实施方式，但此方法存在一些问题，包括(1)忽略 POI 和网络之间的边，以及(2)根据附近折点的可用性得出不一致的捕捉结果。

换句话说，尽管基于捕捉结果的 POI 到网络的步行距离可能以某种方式归因于作为访问阻抗的顶点(例如，距离或步行时间)，但我们仍然会丢失一些地理空间信息(例如，POI 的实际位置和访问它的步行路线)。此外，如果附近唯一的路段是一条长且直的路段，且折点很少，则捕捉距离将不是一个现实的距离。虽然通常在 GPS 轨迹恢复中采用的地图匹配技术解决了类似的问题，但它通常返回校准的轨迹本身，而不是修改实际网络。

在参考资料中，您可能会发现社区提出的一些问题，然而，我发现的解决方案并不十分全面，也不是现成的地理处理工具。对于我的用例来说，最好是创建一个 Python 函数，当我想要将一组点合并到网络中时，这个函数会很方便。

为了记录在案，我正在使用由 [UrbanSim](http://www.urbansim.com/home) 开发的 [pandana](https://github.com/UDST/pandana) 软件包来模拟城市无障碍设施。你可以在这篇[文章](/measuring-pedestrian-accessibility-97900f9e4d56)中看到这是如何做到的。然而，至少有两个原因，我想手工制作自己的网络。首先，我希望根据建筑物而不是交叉点来计算我的输出；因此，将建筑物(或我后来想做的任何东西)连接到网络上是必须的。其次，`pandana`在处理兴趣点捕捉时的行为与我们之前讨论的完全相同:

> *poi 连接到 Pandana 网络中的*最近节点*，该网络假定变量位置和最近网络节点位置之间*没有阻抗*。*

# 想法是:按顺序更新节点和边

![](img/82c2c47a5eb58a2266725b6da1914539.png)

The idea is simply locating the nearest projected point of the POI on the network and connect them.

因此，为了解决问题，让我们将这个过程分成几个步骤:

1.  准备两个代表网络的表格(`geopandas.GeoDataFrame`):一个用于节点，一个用于边，下面称为“内部”节点和边。在我的情况下，这是通过 [osmnx](https://osmnx.readthedocs.io/en/stable/index.html) 包下载的。
2.  准备另一个表，其中包含我们希望在步骤 1 中合并到网络中的点几何。它们将被称为“外部节点”。
3.  用所有外部节点更新内部节点表。这应该像表格连接一样简单，因为困难的部分来自边缘。
4.  找到所有外部节点的预计访问点，并将其更新到表中。现在，与捕捉不同，我们需要为每个外部节点生成一个“投影访问点(PAP)”。这些 pap 是从网络到外部点的最近点。在现实中，这将是你离开大楼后“上路”的地方，并以最短的路线径直走向那条路。
5.  必要时更新内部边。当我们生成 pap 时，它们可能会落在边上而不是顶点上(这基本上是常见的捕捉)。为了确保我们的图形正确识别这种连接，我们需要将原始边分成几段。例如，从 A-B 到 A-P-B。在表中，这将意味着用多个新记录替换原始记录，同时继承相同的道路属性(例如，道路名称、车道数量等。)除了长度。
6.  创建和更新到外部节点的外部连接。实际上，这意味着建立从道路到建筑物的最短步行路径。

# 细节:R 树、最近邻和道路分割

尽管这些听起来很简单，但让我们讨论一下第 4 步到第 6 步中的关键问题。请记住，我最重要的目标是处理时间方面的可伸缩性。例如，新加坡有 185，000 栋建筑和 120，000 条步行路段，虽然这个过程可能是一次性的努力，但我们真的不希望它需要几个小时或几天才能完成。我写的函数的第一个天真的概念验证版本仍然在运行(在一天一夜之后),直到我完成了优化版本的制作、调试、测试、可视化和演示……嗯，你知道我的意思。

请注意，`osmnx`包已经修剪了从 OpenStreetMap 中检索到的网络中的冗余节点(参见本页上的[第 3 节),这意味着那些只是道路中的一个弯道而不是连接两条以上边的交叉点的节点将被删除。这很重要，因为它使图形计算和我们的处理更有效，同时节省空间。但是，对于普遍采用的捕捉方法，我们可能希望尽可能多地保留这些节点，因为与稀疏节点相比，它们可能会使捕捉结果更好。在有许多节点的极端情况下，结果将与我们提出的方法相同。](https://geoffboeing.com/2016/11/osmnx-python-street-networks)

![](img/2b6c3970b36199265b4d871388c8e292.png)

The osmnx package removes redundant nodes in the network downloaded from OpenStreetMap. (Figures retrieved from osmnx introduction article.)

# 第四步:这一步可以进一步分解为两部分:找到最近的边，得到投影点。

**a .找到最近的边:**

为了进行投影，然后插值 PAP，我们需要确定我们实际上是在投影哪条边。虽然`shapely.project`和`shapely.intersection`可以应用于多线串，但它不会返回目标线串段，并且由于存在一些不合理的投影，结果不稳定。

因此，我们需要自己定位最近的边缘。更好的方法是将我们的边存储在适当的数据结构中。因为我们存储的是 LineString 而不是 Point，所以我选择使用 R 树而不是 k-d 树。这将每条边存储为一个矩形边界框，使得高效查询变得非常容易。虽然可以使用`rtree.nearest`，但这可能会产生不良结果。我的理解是，被测量的距离实际上是点和“矩形”之间的距离，而不是线串。一种更全面的方法(并且随后对于最近边的变化是可扩展的)是基于 R-tree 查询 k-nearest neighborhood(kNN ),然后计算到盒子内每个线串的实际距离。

**b .获取投影点:**

用最近的边，我们可以很容易地用`line.interpolate(line.project(point))`得到 PAP。

# 第 5 步:这一步也被分解如下:

**a .确定要更新的边和节点:**

由于每条边上可能有多张纸，我们希望一起处理它们，而不是重复处理。我们根据投影到的线(即最近的边)对所有 pap 进行分组。

**b .通过精心处理将边缘分开**

接下来，我们可以用`shapely.ops.split`函数轻松得到所有线段(用 P1 分裂 A—B，P2 返回 A—P1，P1—P2，P2—B)。但是，每当有人认为这是最容易的部分，这往往是错误可能会困扰你。所以我最终遇到了这里讨论的精度问题。简而言之，当我在线串上投影和插入一个点时，返回的 PAP 不**而不是**与线串相交，使其无法用于拆分线。为了解决这个问题，我们需要*将线串与 PAP* 对齐。这导致了一个非常微小的变化，我们无法识别，但确保交集是有效的，因此，启用了分割。

# 步骤 6:这一步只是通过指定“from”和“to”节点来生成新节点，并将它们更新到边表中。

包括这次更新在内的所有更新都只是简单的表操作，这里就不讨论了。请注意，我们使用`GeoDataFrame`来存储 shapefiles，并将 CRS 转换为`epsg=3857`来计算道路长度，单位为米。

# 讨论

您可能已经知道，地理空间网络和抽象图形结构(例如，社交网络)之间的一个最大区别是，前者的边也是具体的对象，并且具有形状。这就是为什么我们必须通过这些边缘导航并操纵它们。在社交网络中，实现这种节点到边的插值是没有意义的。

我相信这个功能肯定是有提升空间的。例如，使用`cython`优化代码可能会获得更好的性能。也可以尝试使用 PostGIS 来执行这些地理处理步骤，我还没有尝试过，因为我目前更喜欢用 Python 来完成。然而，考虑到新加坡数据的当前运行时间为 7 分钟(前面提到的建筑物到行人网络场景)，目前应该是可以接受的。对于数百个 poi，这个过程应该在几秒钟内完成(如果不是一秒的话)。

![](img/102ef30d04d7d2bc9917b2aaa14c5d9e.png)

In the real world, there can be various footways connecting a building (red star) to the road network. Our method only finds the nearest one (cyan node), which is unrealistic in some cases. Finding a way to generate a set of reasonable routes merely based on the network (w/o computer vision) can be challenging.

我也在考虑一个多方向的 kNN 连接扩展。为了说明这个用例，假设理想的公共汽车站在你的后门，但是通过这个方法产生的最近的人行道连接你的房子和前门的道路。这给了你一个不切实际的步行时间和路线计算，当你在你的后门乘公共汽车时。简单地建立 kNN 连接可能不能完全解决这个问题，因为它们可能都是前门的分段。因此，从点的不同方向获取合理的 kNN 边将是所期望的。

![](img/a97361cd4009175e207784f234ab6868.png)

Ideally, we want to connect to one nearest neighbor (road segment) in different directions. The output on the right shows an example of how simply locating the 5 nearest road segments is not sufficient.

如果你有任何改进的想法，或者知道一个更简单的方法来完成这项任务，请留下评论与我们分享！

# 参考

这些是我从这个问题开始的帖子:

*   如何使用 Networkx 从一个点和一个线形状文件制作一个可传递图？
*   [点图层和线图层之间的最近邻？](https://gis.stackexchange.com/questions/396/nearest-neighbor-between-point-layer-and-line-layer/438)

我还浏览了许多其他帖子，以下是其中一些:

*   [在多线串上创建最近点，用于 shortest_path()](https://gis.stackexchange.com/questions/27452/create-closest-point-on-multilinestring-to-use-in-shortest-path)
*   [了解 RTree 空间索引的使用](https://gis.stackexchange.com/questions/120955/understanding-use-of-spatial-indexes-with-rtree)
*   [使用 geopandas 通过最近点分割线](https://gis.stackexchange.com/questions/268518/split-line-by-nearest-points-using-geopandas)
*   [在大数据集上找到离每个点最近的线，可能使用 shapely 和 rtree](https://stackoverflow.com/questions/46170577/find-closest-line-to-each-point-on-big-dataset-possibly-using-shapely-and-rtree)