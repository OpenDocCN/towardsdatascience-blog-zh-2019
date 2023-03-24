# 利用 PowerBI 为创业分析提供超强的 Excel

> 原文：<https://towardsdatascience.com/supercharged-excel-for-startup-analytics-with-powerbi-46a15c436eba?source=collection_archive---------18----------------------->

## 如何将 Excel 用作数据分析师而不是数据猴子

Excel 似乎是我遇到过的最讨厌的工具。这是一个遗憾，因为如果你忽略这个坏名声，它是你腰带上最好的分析工具之一。

根据我的经验，Excel 有两个用途:高级报告和数据分析。

![](img/66d3b4e0f4f529b5a3782dad6b90f08d.png)

高层次的报告正在把握你的业务脉搏。您可以在文章【insight 的启动报告模板中查看更多详细信息。它不是那种有数百个 KPI(关键绩效指标)和多达一百页、只有少数人阅读、没有人采取行动的每日报告。对于这些，任何商业智能报告都可以。高水平的报告是你想看的那种报告，因为它讲述了一个重要的故事(企业如何运作，为什么会这样，你应该另找工作吗？).即使大多数[网飞的决定是基于分析](https://www.forbes.com/sites/enriquedans/2018/05/27/how-analytics-has-given-netflix-the-edge-over-hollywood/#7a86e7c66b23)，它有助于增加一点工艺。对于这样的报告，可能只是在图表中添加一个标签，以记住某个特定产品由于库存问题正在下降，或者通知下降位于特定区域。数字很好，但有一点解释会有所帮助。

对于高级报告来说，Excel 很好，因为它让您自动完成大部分工作，同时让您添加最后的润色。

数据分析使商业人士能够理解高层报告下发生的事情。它在构建一个你是英雄的故事。销售数字下降，没有人知道为什么，找到罪魁祸首夏洛克。

Excel 给你提供了很棒的数据透视表，到处添加了一些公式，很好的单元格条件格式。您可能更喜欢使用代码(SQL、Python 或其他什么)进行干净的处理，但要突出显示数据中的内容，Excel 是最好的。

# 1-将数据、计算和呈现分开，以保持理智

你的 Excel 表格反映了你的精神状态。可以干净，也可以混乱。不要让它是后者。

![](img/09f542d29a62ad3e968bb123c55219a4.png)

受 webapps 的[模型-视图-控制器](https://en.wikipedia.org/wiki/Model%E2%80%93view%E2%80%93controller)的启发，我把我的电子表格分成三种标签。

![](img/f0d232df92c55a4cd423b46cc0b37bd3.png)

第一类是演示表，它将仪表板和分析表重新组合在一起。仪表板显然是整洁的部分(你在[启动报告文章](https://dataintoresults.com/post/startup-reporting-template-for-insights/)中看到的部分)。我还添加了一个分析表，在那里我放了一个大的动态交叉表，如果仪表板部件中的某些东西需要更仔细的检查，用户可以在那里更深入地挖掘。

第二种是数据层。在这些表中(每个数据源一个表)，有数据并且只有原始数据。没有公式。我给它们贴上数据标签，并总是把它们标为黄色。

最后一种表是计算表。这是我放置业务逻辑(以及所有那些 Excel 公式)的地方。这些被标记为隐藏和灰色，因为没有用户应该去那里(他们大部分时间是隐藏的)。

这种简单的关注点分离会让你内心平静。如果有问题，你知道你应该首先检查相关的数据表，然后，如果错误不存在，相关的隐藏表。你应该有一个简单的数据血统。

在下一步中，我们将看到如何自动更新数据层。

# 2 —数据连接是自动化的关键—停止复制/粘贴

人们讨厌 Excel 的一个原因是，大多数人最终会花费数小时复制/粘贴。事实上，没有一个 Excel 报表需要点击一次才能刷新。

![](img/ded97b1664f088c98a10d1863a8c0f11.png)

让我们假设您有一个 [Postgresql 作为数据集市](https://dataintoresults.com/post/postgresql-for-data-science-pro-and-cons/)，它每天晚上都会转储一些生产数据(可以是一个共享目录，但数据库更有效)。使用[右 ODBC](https://odbc.postgresql.org/) 驱动(Excel 理解的常用中间件)，可以直接从 Excel 中读取数据库。您可以选择一个表，甚至可以编写一个合适的 SQL 查询。这个不错的工具在 Get Data->From Other Sources->From ODBC(From Database 应该是从微软数据库中读取的)。

![](img/a355c913f629b99192c80cf3f7dc9598.png)

只需按下 Load，数据将从数据库中加载。好的方面是它将打开一个查询面板，您可以在其中随意刷新数据(注意:数据面板中的 Refresh All 按钮也可以工作)。

![](img/553348a7ad89422139a1d955e0b92383.png)

因此，我们只需点击几下鼠标，就能看到带有条件格式的数据透视表，在那里，您可以很容易地发现 2 月份存在季节性效应，2013 年 7 月开始会有大幅提升。

![](img/8478ab874eaf9437fc4b167875056407.png)

使用这样的工具，您可以轻松创建一个[高级报告，就像我们为 Brewnation 所做的那样。请记住为每个数据连接创建一张表，一张或多张用于计算，一张用于演示。](https://dataintoresults.com/post/data-analytics-use-case-brewnation/)

说到计算，使用 Excel 公式会很痛苦并且容易出错。好消息，有了 PowerBI 有了更好的方法。

# 3 — PowerBI for Excel 是获得超能力的关键

Excel 的超能力还有一层。被微软巧妙隐藏。我称之为 PowerBI for Excel，但 [PowerBI](https://powerbi.microsoft.com) 现在是一个独立的产品线，采用不同的方法(但核心是相同的)。在过去，它被称为权力中枢和权力查询，但它早已不复存在。在那些日子里，这只是 Excel 很少有人知道一个功能。

![](img/6f53260647881d52425e033c2652bf62.png)

在上一节中，我们建立了 Excel 和数据源之间的链接。让我们将这些数据加载到 PowerBI 引擎中，而不是加载到工作表中。使用“加载到”打开“导入数据”对话框，而不是“加载”按钮。在这里，我们将创建连接并将数据添加到数据模型中。

数据模型是 PowerBI 的存储部分。您不再受限于 Excel 的一百万行。这种列压缩存储也非常擅长于只占用很小的空间。

要显示此数据模型中的内容，请使用数据功能区中的“管理数据模型”按钮。

![](img/62f35be226657b1141966fc63c0bca3b.png)![](img/80a7c102a30d92226d9533fc4c8fb2bd.png)

可以通过添加新的计算列和聚合度量来增强整个模型。事实上，您的大部分计算层都可以放入 PowerBI 引擎。

如你所见，一旦你知道如何使用它，Excel 真的很强大。它是主流的分析工具。

*原载于 2019 年 5 月 10 日*[*【https://dataintoresults.com*](https://dataintoresults.com/post/supercharged-excel-for-startup-analytics-powerbi/)*。*