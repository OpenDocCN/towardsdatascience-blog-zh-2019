# Python 地理数据科学入门—第 3 部分

> 原文：<https://towardsdatascience.com/getting-started-with-geographic-data-science-in-python-part-3-ddf55130840b?source=collection_archive---------13----------------------->

## 案例研究/项目演练

![](img/ee96411c48d8ec3711fbdf860c7fa407.png)

Photo by [Markus Spiske](https://unsplash.com/@markusspiske?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

这是 Python 地理数据科学入门系列文章的第三篇。您将学习如何用 Python 阅读、操作和分析地理数据。第三部分，也就是本文，涵盖了一个相关的真实世界项目，旨在巩固你的学习成果。

本系列的文章可在此处获得:

[](/master-geographic-data-science-with-real-world-projects-exercises-96ac1ad14e63) [## 通过真实世界的项目和练习掌握地理数据科学

### 真实世界项目和练习

真实世界项目& Exercisestowardsdatascience.com](/master-geographic-data-science-with-real-world-projects-exercises-96ac1ad14e63) [](/getting-started-with-geographic-data-science-in-python-part-2-f9e2b1d8abd7) [## Python 地理数据科学入门—第 2 部分

### 教程、真实世界项目和练习

towardsdatascience.com](/getting-started-with-geographic-data-science-in-python-part-2-f9e2b1d8abd7) 

本案例研究的学习目标是:
1 .对真实世界数据集项目
2 应用空间操作。空间连接和管理地理数据。
3。探索性空间数据分析(ESDA)

## 项目设置

在这个项目中，我们将使用两个数据集:一个按年龄分类的人口数据集和来自瑞典统计局的学龄前数据集。因为我们处理的是学龄前儿童，所以我们将关注 0 至 5 岁的儿童。根据欧盟统计局最近的统计数据，瑞典被认为是欧洲第三大婴儿制造国。在这个项目中，我们将分析 5 岁以下人口的地理分布以及幼儿园的分布。

人口数据集以低级别分类格式(投票区域)出现，每个区域有 700 到 2700 人口。另一方面，幼儿园数据集是坐标点格式，包含全国幼儿园的地址。在对两个数据集进行简短的探索性数据分析后，我们将执行预处理和空间连接地理处理任务来合并两个数据集。这是统计每个地区幼儿园数量的路线图。

1.  包含空间操作:使用空间连接来确定哪些幼儿园位于面区域内，或者换句话说，哪些人口区域包含幼儿园点。
2.  按规模分组到每个地区的幼儿园总数。
3.  将按数据框分组与人口数据集合并。

最后，我们将进行探索性空间数据分析(ESDA ),调查不同地区的幼儿园的空间分布。

让我们首先将两个数据集读入 Geopandas。我们首先需要从 dropbox 链接访问数据集，并将其解压缩。

通过查看人口数据集的前几行，值得注意的是该数据集的几何是一个多边形。每个区域都有一个独特的代码(***【Deso】***)和一堆其他属性。 ***Age0_5*** 代表每个地区 0 到 5 岁的儿童数量。总人口存储在 ***Tot_pop*** 列中。

![](img/153b21f5fef0d4223c08ddd06d2eff59.png)

Population dataset

另一方面，幼儿园有存储每个学校坐标的点几何。学校名称以及地址、城市和邮政编码作为列出现在这个数据集中。

![](img/193b00f7a55d2805f3650265eb79a71f.png)

Preschools dataset.

在下一节中，我们将进一步了解这两个数据集的某些方面。

## 探索性数据分析

在开始地理处理任务之前，让我们分析数据集以总结主要特征。在预处理和建模之前探索数据集可以帮助您更好地掌握数据集。我们从总共有 5985 行和 32 列的人口数据集开始。让我们用`.describe()`方法做一个描述性统计的总结。这是人口数据集描述统计的输出。

![](img/657a834bc8c3d5b7fb912596b6a26267.png)

Descriptive statistics of the population dataset

下图显示了人口数据集中 Age0_5 列的分布。

正如你从下面的分布图中看到的，它向右倾斜，右边有一条长尾巴。

![](img/2971ac6ba497e9d2b92eb1973cb2dac1.png)

distribution plot of children between age 0–5

由于我们正在处理地理数据，我们也可以可视化地图。在本例中，我们将构建一个 0 至 5 岁儿童的 Choropleth 图。首先，我们需要通过计算***age 0 _ 5/Tot _ pop***来计算该区域的子密度，从而创建一个新的***age 05 _ density***列。

输出地图通过颜色显示每个区域的儿童数量。

![](img/fc370383229f6e72b6e084d108079a42.png)

Choropleth Map

这张地图清楚地显示了这些地区儿童的分布情况。在北部，儿童密度很低，而在东部、西部和南部，儿童密度很高。

最后，让我们在 choropleth 地图上叠加幼儿园，看看学校的分布。

输出地图通过颜色和学龄前数据集的叠加显示每个区域的儿童数量。

![](img/517e3d06ac3bda118bea852559251a6a.png)

Choropleth map and Preschools

相当乱。我们无法确切知道每个地区有多少所幼儿园。这就是空间连接和地理处理任务派上用场的地方。在下一节中，我们将执行几个预处理任务来获取每个区域内的幼儿园数量。

## 空间连接

统计每个地区的幼儿园数量。该流程包含以下内容:

1.  使用空间连接来确定面区域内的幼儿园。
2.  按规模分组到每个地区的幼儿园总数。
3.  将按数据框分组与人口数据集合并。

以下代码用于执行空间连接并创建一个新的地理数据框，其中包含每个区域中学校的数量。

因此，如果我们现在查看***merged _ population***数据集的前几行，我们会注意到我们有一个额外的列***prefessional _ count***，其中存储了每个地区的学龄前儿童人数。

![](img/eb5a352551e2dcfe927eb4fb974c881d.png)

spatial join — preschool counts

现在我们可以并排检查年龄密度的分布图，以及基于学龄前儿童数量的新分布图。

![](img/f06dd361ae737e6909bcbeaf4e00c984.png)

Left choropleth map of preschool counts. Right Choropleth map of Age 0–5

然而，我们无法对这两种完全不同的特征进行有意义的比较。正如你所看到的，学龄前儿童的数量在 0-10 之间，而年龄密度在 0-24 之间变化。在统计学中，就像我们之前执行的 EDA 一样，我们假设数据集中的观测值是独立的，但是，对于地理数据集，存在很强的空间依赖性。想想瓦尔多·托布勒的地理第一定律:

> “一切事物都与其他事物相关，但近的事物比远的事物更相关。”

因此，在下一节中，我们将更深入地挖掘并执行探索性的空间数据分析。

## 探索性空间数据分析(ESDA)

空间统计和探索性空间数据分析(ESDA)是非常广泛的。在本节中，我们将只查看空间关联的局部指示符(LISA ),以检测该数据集中的空间自相关，并探索邻近位置的特征及其相关性。这可以帮助我们研究和理解空间分布和结构，以及检测这些数据中的空间自相关。

为了更仔细地查看这些空间统计数据，我们首先读取县数据集，将我们的数据分成瑞典人口最多的两个城市(斯德哥尔摩和哥德堡)的子集，然后与我们预处理的人口数据集合并。

我们将使用 [Pysal](https://pysal.readthedocs.io/en/latest/) 和 Splot 库进行空间自相关。这是权重的设置和变换。我们将 prefessional _ count 作为 y 变量，并执行 Queen weights 转换。然后，我们使用权重创建一个空间滞后列(y_lag ),以获得不同多边形基于其地理区域的相似性。

现在我们可以计算 Moran 的 I local 来检测聚类和离群值。

让我们画出两个城市的莫兰 I。

上面代码的输出如下所示的两个散点图。斯德哥尔摩具有负的空间自相关，而哥德堡市具有正的空间自相关。

![](img/3cc2fa26c384a87c8a7a0c6b5d0554f6.png)

Moran’s I Local Scatterplot

这些图还将点划分为四个象限，以将局部空间自相关的类型指定为四种类型:高-高、低-低、高-低、低-高。右上象限显示 HH，左下象限显示 LL，左上象限显示 LH，左下象限显示 HL。用地图可以清楚地看到这一点，所以让我们把它放到地图上。

下面两张地图清楚地显示了这些地区不同的集群。在斯德哥尔摩，HH 空间聚类不是集中在一个地方，而在哥德堡，我们有大量具有 HH 值的相邻区域。

![](img/d1da9f6ca3fff79afcbc12d80b0b8afc.png)

Local Indicator of Spatial Association (LISA)

这描绘了一个比我们开始的 choropleth 图更清晰的画面，可以给你一个不同的空间集群位于何处的清晰指示。

# 结论

该项目演示了使用 Python 执行地理数据的地理处理任务。我们首先执行探索性数据分析(EDA ),然后执行空间连接并创建新数据集。在最后一部分，我们讲述了探索性空间数据分析(ESDA ),以更深入地了解地理数据集及其空间分布。

该代码可从以下 GitHub 存储库中获得:

[](https://github.com/shakasom/GDS) [## 沙卡姆/GDS

### 地理数据科学教程系列。在 GitHub 上创建一个帐户，为 shakasom/GDS 的发展做出贡献。

github.com](https://github.com/shakasom/GDS) 

您也可以直接从以下链接运行 Google Collaboraty Jupyter 笔记本:

[](https://colab.research.google.com/drive/17-LykLsQI930f1W4eoOlBvteJme0tbQk) [## 谷歌联合实验室

### 编辑描述

colab.research.google.com](https://colab.research.google.com/drive/17-LykLsQI930f1W4eoOlBvteJme0tbQk)