# 纽约的自行车

> 原文：<https://towardsdatascience.com/bikes-of-new-york-6e0bbe5507f8?source=collection_archive---------23----------------------->

![](img/db28e57cac7e890b06eab69eba3f2e1b.png)

[*Photograph by Hannah McCaughey/Map by Norman Garbush.*](https://www.outsideonline.com/2332671/purest-form-bike-angel)

有很多关于花旗自行车的故事，以及它如何改变了一些纽约人的出行方式。但是，花旗自行车不仅仅是一种交通工具，它是一个丰富数据的深井，等待着被争论、分割和转换，以找到一两颗关于城市如何运动的智慧珍珠，甚至可能是一些有趣的事实。

作为一名有抱负的数据科学家，使用花旗自行车数据集是一次有趣丰富的经历，其中不乏令人沮丧的插曲，还有*啊哈！*瞬间。

因为数据集适合伟大的可视化，我想我会在不到一周的时间里自学如何使用 [D3.js](https://d3js.org/) 可视化库，并将其用于这个项目——**天真，我知道**。我对精通 D3 的陡峭学习曲线知之甚少，更不用说成为专家了。

这时，关键部分出现了，我决定专注于一些我力所能及的事情。

# 使用大数据集的挑战

花旗自行车的数据集不止几个，所以我决定把重点放在最后一年，从 2018 年 1 月到 2019 年 2 月。仅此一项就给我留下了 19，459，370 个数据点。是的，你没看错。[这是一个爆炸加载，在一个熊猫数据帧上](/why-and-how-to-use-pandas-with-large-data-9594dda2ea4c)——*而不是*。甚至不要让我开始尝试绘制变量之间的任何关系。

我一直有这样的想法，数据越多越好，所以我从来没有真正想过*有太多的数据*是一个挑战。但它真的是。不仅仅是因为加载时间，而且正如我之前提到的，在如此大的数据集上使用可视化库只会给我留下一串错误和超时消息，此外还有 RAM 内存有限的问题。所以为了处理大数据集，[内存管理](https://datascienceplus.com/processing-huge-dataset-with-python/)应该是你的首要任务之一。

幸运的是，经过一点努力和创造性的思考，我能够扭转局面。首先，我开始从数据集中随机抽取样本，然后我有了自己的*啊哈！*时刻:我可以使用 SQL 查询创建我想要可视化[的数据和关系的子集，然后将 SQL 查询结果转换成大多数可视化库可以处理的 CSV 文件。**问题解决了。**](http://sdsawtelle.github.io/blog/output/large-data-files-pandas-sqlite.html)

# 数据

既然我已经和你分享了我的一些痛苦，我也应该分享一些我的结果。

如前所述，从 2018 年 1 月到 2019 年 2 月，共登记了 19，459，370 次旅行。经过一些清理、切片和争论，我的最终工作数据集减少到 17，437，855 次。这些只是订户的旅行，因为我决定放弃临时乘客和一日游顾客。

根据[花旗自行车的月度报告](https://d21xlh2maitm24.cloudfront.net/nyc/March-2019-Citi-Bike-Monthly-Report.pdf?mtime=20190409100608)，花旗自行车的年度会员总数现已达到 150，929 人。让我们来看看他们是谁。

# 谁在骑花旗自行车？

没有太多关于每个用户的信息，但是从数据中，我们可以得到基于总用户数的年龄和性别。这些汇总并没有给出准确的订阅者数量，而是给出了样本的基本分布。

这是一个很好的学习机会，可以利用 [Plotly](https://plot.ly/python/) 制作一些互动的情节。我发现一开始理解[图层次](https://images.plot.ly/plotly-documentation/images/python_cheat_sheet.pdf?_ga=2.165312252.25572619.1555260029-699716662.1547151759)可能有点繁琐。Plotly 的图表令人惊叹，因为尽管它是使用 Python 和 Django 框架构建的，但在前端它使用了 JavaScript 和 Dr.js 库— **所以毕竟，我确实使用了一点 D3 . js**。

订阅者的最高出生年份类别是从 1987 年到 1991 年。**让我再次声明一下，Citi Bike 目前有 150，929 名订户**，为了获得这些订户的分布情况，我对乘客数据使用了聚合函数，如下面的代码片段所示。

Pandas DataFrame Capturing Birth Year SQL Query

Subscribers by Year of Birth

Code for Interactive Plotly Bar Chart

从性别来看，大多数骑手都是男性。

Pandas DataFrame from SQL query to identify Gender distribution

An interactive bar chart showing subscribers by gender. Male (1), Female (2)

# 他们要骑多久？

平均行程持续时间为 13 分钟，这意味着[用户不会长途骑行](http://gothamist.com/2017/03/15/report_citi_bike_ridership_driven_b.php)——记住，我放弃了临时骑手和一次性客户。

我们还可以看看一周中每天的平均出行次数，正如所料，工作日的出行次数略高于周末。展示了周末乘车和工作日通勤之间的区别。

An interactive plot where circle size represents the average trip duration.

最后，我对骑行的长度和一年中的某一天之间的关系感兴趣。

An interactive plot showing the number of trips per day of the year.

该图给出了全年对花旗自行车需求波动的完整图像，当天气较暖时，4 月至 10 月期间的需求较高。这让我对天气和一天旅行次数之间的关系感到好奇，所以我用来自[国家海洋和大气管理局](https://www.ncdc.noaa.gov/cdo-web/datasets#GHCND)的数据创建了一个新的熊猫数据框架，在我的原始数据框架中有每天的天气摘要。

然后我使用 [Scikit Learn](https://scikit-learn.org/stable/) 库运行了一个多元回归算法。

Multiple Regression using Scikit Learn

> 事实证明，每天旅行次数的 62%的差异可以用天气来解释。

# 他们要去哪里？

一些花旗自行车停靠站肯定比其他更受欢迎。为了绘制最受欢迎的地图，包括骑行的开始和结束，我使用了[叶子](https://python-visualization.github.io/folium/modules.html)库。

但在我实际制作一个交互式地图之前，我运行了一个 SQL 来获取骑行量排名靠前的花旗自行车停靠站。我第一次试着把它们都映射出来，但是最终，程序崩溃了。所以我决定接受 100 美元。

A video that captures the functionality of a Folium interactive map

Code to create an interactive map using the Folium library

乘客量排名前五的车站是:

*   潘兴广场北，1，576，381 人次
*   西 21 街和第六大道，1，148，192 次出行
*   东 17 街&百老汇，1，121，953 次出行
*   百老汇和东 22 街，109，7314 次出行
*   百老汇和东 14 街 96，901 次旅行

具体来说，这些是中央车站、麦迪逊广场公园、联合广场公园和熨斗大厦附近的码头站。

# 30657 号自行车呢？

最后，我想表彰去年出行次数最多的自行车，30657 — *你现在想知道你是否曾经骑过它，是吗？*。

骑了 2776 次，总共行驶了 36448 分钟，这辆自行车可能比我更了解纽约。所以，作为临别礼物，我将把自行车 30657 留给你，这样你就可以看到它的运行。

*来源*

[让花旗自行车为纽约工作的天使](https://www.outsideonline.com/2332671/purest-form-bike-angel)

[花旗自行车博客](https://www.citibikenyc.com/blog)

[利用树叶创作互动犯罪地图](https://blog.dominodatalab.com/creating-interactive-crime-maps-with-folium/)

[花旗自行车月报——2019 年 3 月](https://d21xlh2maitm24.cloudfront.net/nyc/March-2019-Citi-Bike-Monthly-Report.pdf?mtime=20190409100608)

[使用 Pandas 和 SQLite 的大型数据文件](http://sdsawtelle.github.io/blog/output/large-data-files-pandas-sqlite.html)

[D3 . js 的搭便车指南](https://medium.com/@enjalot/the-hitchhikers-guide-to-d3-js-a8552174733a)

[动画 Deck.gl 弧线](https://observablehq.com/@yarynam/animated-deck-gl-arcs)

[着色器之书](https://thebookofshaders.com/)