# Ecom 数据系列:什么是 RFM 分割？

> 原文：<https://towardsdatascience.com/ecom-data-cooler-series-what-is-rfm-segmentation-8c4d6c2c0fc0?source=collection_archive---------22----------------------->

## 让电子商务数据科学概念变得简单，一次一个主题。

## 简单、有效的市场细分，提高打开、点击和转化

Ecom Data Talk Episode 2: What is RFM segmentation?

传统上，营销人员根据人口统计进行细分👨👩🐈👶。然而，今天大多数电子商务品牌更喜欢应用数据科学来开发基于购买行为的行为细分。例如，他们在🛒观看/购买什么产品？他们通过什么途径到达您的网站📍？(意识与意图)。此外，他们的 engagement🖱️水平如何(使用/打开/点击/查看)？

细分有什么好处？虽然有很多，但我会在这里列出我最喜欢的三个:

1.  聚焦和个性化营销
2.  降低营销成本
3.  更好的产品开发

根据行为对客户进行分组可以实现情境化营销，而不是电子邮件群发，从而降低营销成本。通过锁定目标客户子集(细分市场！)有了类似的属性，你可能会获得更好的打开率、更高的转化率和广告支出回报率(ROAS)。最重要的是，你可以让未来的产品迎合顾客的口味。

# 什么是 RFM？

RFM 是一种用于分析客户价值的数据建模方法。它代表新近性、频率和货币，这只是描述客户行为的三个指标。最近度衡量的是从客户上次订购到今天的时间(通常以天为单位)。频率衡量客户的订单总数，货币是他们从这些订单中花费的平均金额。

![](img/8edafd74d5af59ae131cb5930f8c03e2.png)

Photo by [Hoster](https://unsplash.com/@hoster?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

RFM 始于直销时代，至今仍是零售和电子商务中最简单、最有效的工具之一。要创建 RFM，您需要转换您的数据并从高到低分配一个分数。下面是一个简单的示例，它有三个级别(高、中、低)—在一列中对客户的订单进行排序，并为没有订单的客户分配 3 分，为有一个订单的客户分配 2 分，为有两个或更多订单的客户或回头客分配 1 分。这给你你的频率度量。然后，你重复这个过程的新近性和货币。最后，你会得到三个价值，每个价值代表 RFM 的一个支柱，描述了每个客户的价值，其中一个是最好的。

# 为什么 RFM 很重要？我为什么要在乎？

RFM 是必不可少的，因为它允许你快速地将你的客户从最好到最差进行排序和安排，并且它在描述客户行为方面非常有效。有了 RFM 分数，你可以创造许多有用的客户群。我们从 MVP 开始，他们是您的最佳客户，在所有三个维度上都有最高分。高消费的新客户在*新近度*和*货币度*上得分高，但在**频率**上得分低，因为他们最近才购买。相反，你有高价值的流失客户，他们已经有一段时间没有购买了。他们在*最近*和*金钱*上得分较高，但在**最近**上得分较低。最后，低价值客户在所有三个维度上都得分很低，不太可能有任何价值。你可以通过抑制这些用户来节省营销费用*和*提高打开率和点击率。

![](img/096eea78081b17348cb7c8c300549d3c.png)

Photo by [Austin Distel](https://unsplash.com/@austindistel?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

你为什么要在乎？我们已经看到，前 5%的顾客平均花费是其他人的 10 倍，占总收入的三分之一。顶级客户往往有更高的平均订单价值，更有可能成为你品牌的忠实粉丝。忠诚的粉丝往往会产生口碑和推荐，这让他们更有价值。

简而言之，你应该花更多的时间让你的顶级客户开心，RFM 可以帮你弄清楚该关注哪里，做什么。

# 为什么 RFM 能为我做什么？

如果你还没有，开始利用 RFM 为你的营销活动创建客户细分，并开始优化。您可以:

*   为你的贵宾铺上红地毯。
*   为最有可能购买的客户设计培育活动。
*   为即将流失的客户创建个性化的优惠和提醒。
*   重新瞄准流失的高价值客户，赢回他们。

![](img/a4d24993b92803c481e2c88aebc67496.png)

Photo by [Alvaro Reyes](https://unsplash.com/@alvaroreyes?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

RFM 有更多的使用案例。您将能够创建更多相关的、情境化的活动，以不同的方式针对不同的客户。细分有助于提高打开次数和点击量，并带来更多的营销收入。如果你还没有这样做，你绝对应该这样做。我们的一位客户发现，多亏了 RFM，她在年度计划中多了 20%的高价值客户。

总之，RFM 代表近期、频率和货币，近期是最重要的。为什么？那是因为网购是非契约性的商业行为，人们可以自由来去。你只能假设顾客是“活着的”,并且对你感兴趣，当他们通过最近的购买告诉你这一点的时候。

找到你的 RFM。开始将你的客户从最好到最差进行分类，并将这些细分纳入你的营销活动中！

不要忘记，数据就是力量，我们希望将电子商务数据的力量还给人们。立即加入我们，与您的数据一起成长。

由[段](https://segments.tresl.co/)制成

在 [LinkedIn](https://www.linkedin.com/company/tresl) 或[脸书](http://facebook.com/groups/ecomdatascience)上找到我们。

*原载于*[*tresl.co*](https://tresl.co/what-is-rfm-segmentation/)

[1]彼得·s·法德尔、布鲁斯·g·s·哈迪·卡洛克·李、RFM 和 CLV: [使用等值曲线进行客户群分析](http://brucehardie.com/papers/rfm_clv_2005-02-16.pdf) (2004)