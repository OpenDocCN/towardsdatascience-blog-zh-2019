# 使用 Python 和 Robinhood 创建一个简单的低买高卖交易机器人

> 原文：<https://towardsdatascience.com/using-python-and-robinhood-to-create-a-simple-buy-low-sell-high-trading-bot-13f94fe93960?source=collection_archive---------6----------------------->

![](img/caadeb98c847dc402d4ee68b314d4c15.png)

Photo by [Ishant Mishra](https://unsplash.com/@ishant_mishra54?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

所以我最近一直在折腾 Robinhood，一直在努力理解股票。我不是金融顾问或其他什么人，但我想创建一个简单的交易机器人，这样我就可以在创建更复杂的代码之前多理解一点。对于那些还没有查看的人，我创建了一篇关于如何使用 Python 连接 Robinhood 数据的文章。

[](/using-python-to-get-robinhood-data-2c95c6e4edc8) [## 使用 Python 获取罗宾汉数据

### 让我们自动化一些股票，可以用来建造一个交易机器人。

towardsdatascience.com](/using-python-to-get-robinhood-data-2c95c6e4edc8) 

我做的第一件事就是把我所有的股票都存到一个数据框里。我不喜欢如何布局的数据帧和我的需要转置它会使我的机器人更简单。然后将索引移动到滚动条列中。我还将某些列切换为 floats，因为它们当前被表示为一个字符串。代码在下面，输出的数据帧也在下面。

所以我创造了低买高卖。我只想玩玩`average_buy_price`中低于 25 美元的股票，并给自己设定每只股票限购 5 只。然后我把它分成两个独立的数据框架，一个是买入，一个是卖出。

对于买入，如果`percent_change`跌破. 50%,那么只有当股票数量为 1 时*才会触发买入。然后，它以市场价格购买 4 只股票。*

对于 sell 来说，基本上是相反的。如果`percent_change`上涨超过 0.50%，数量为 5，触发卖出。这是我的代码，需要一些清理，但我只是测试出来。我将滚动条移动到一个列表中，然后使用`robin_stocks`来执行命令。

我通常只在工作日开市时运行一次。到目前为止，我已经取得了几美分的正回报。它不多，但它可以发展成更好的东西。

将来，一旦我做了更多的研究，我会计划增加更复杂的交易策略。股票对我来说是相当新的，所以请不要把这当成财务建议。我不对你的任何输赢负责。但希望这能激发一些想法。

在这里进入代码。

别忘了在 [LinkedIn](https://www.linkedin.com/in/melvfernandez/) 上联系我。