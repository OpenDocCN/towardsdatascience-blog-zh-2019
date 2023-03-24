# 递归转换 BigQuery JSON API 响应

> 原文：<https://towardsdatascience.com/transforming-bigquery-json-api-responses-recursively-5c462f5b01?source=collection_archive---------31----------------------->

## 从字段/值行嵌套构建键值对

![](img/ab9a33c2890029eec1c55d2c9d969189.png)

La Sagrada Familia, Barcelona, by **Paolo Nicolello.**

跟我一起说:“嵌套的 JSON 很难用！”。我说的对吗？当然可以！既然我们已经说清楚了，我只想说我完全相信 JSON。它是合乎逻辑的，是通用的，大多数语言都用它来创建快速访问的散列映射风格的数据结构。所有人的胜利！

直到你筑巢。

一群相信嵌套 JSON 的好处的人和那些相信扁平 JSON 是他们选择 API 有效负载幸福的圣杯的人。这场无声的战斗如此激烈，以至于许多扁平化技术充斥着黑客的宝库，如 [*递归方式*](/flattening-json-objects-in-python-f5343c794b10) 和 [*非递归方式*](https://github.com/ebendutoit/json_flattener) 等方法。

# 递归地飞越巢穴

看过电影[盗梦空间](https://www.imdb.com/title/tt1375666/)？这是个好东西。时间线中的时间线。当一切恢复时，你会看到事情是如何组合在一起的。以类似的方式，递归在代码中占用的空间很小，但是可以处理巨大的计算(理解为“嵌套”)复杂性。

好了，够了，让我们开始吧！

# BigQuery 的查询 API 返回 JSON

这是 Google BigQuery 在执行查询后的 API 响应示例:

BigQuery’s query API result looks like this

该模式向您展示了数据是如何构造的，并且各行用字段的**“f”和值**的**“v”来表示哪些值适合该模式。**

现在，看起来像这样的 JSON 不是更容易阅读和操作吗？

Transformed BigQuery result into something more manageable

如果你同意，那么你会得到很好的照顾。

# 解决方案

下面是进行这种转换的 ***node.js*** 代码。请随意使用它，根据您的需求进行调整，通常会让您的生活更简单，让您的数据更快乐。该功能的界面是:

```
convertBQToMySQLResults(schema, rows)
```

您可以像这样传递 BigQuery 结果:

```
// apiResponse is the payload you receive from a BigQuery query API // responseconvertBQToMySQLResults(apiResponse.schema.fields, apiResponse.rows)
```

# JsFiddle 演示

下面是一个 JsFiddle 代码演示:

# 概括起来

JSON 的很多转换都存在。递归解决方案不是最容易调试的，但是它们有最简单的代码足迹。用调试器单步调试代码是以“慢动作”方式查看此类算法的首选方式。本文提供了一种将源自 Google BigQuery 的复杂嵌套 JSON 简化为您可以自己操作和使用的东西的方法。试试吧！快乐转换！