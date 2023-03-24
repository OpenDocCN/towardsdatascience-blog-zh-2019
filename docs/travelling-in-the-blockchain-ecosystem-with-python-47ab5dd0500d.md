# 使用 Python 在区块链生态系统中旅行

> 原文：<https://towardsdatascience.com/travelling-in-the-blockchain-ecosystem-with-python-47ab5dd0500d?source=collection_archive---------19----------------------->

![](img/30271e1a610c3c41fb8075bfbfb217d9.png)

Aggregate Market Caps of all Coins

全球有超过 2500 个活跃的*区块链*项目，每个项目都有自己独特的统计特征，我们很少看到对整个*加密*市场的顶级分析，因为清理和收集时间序列太耗时了。

在零售方面，我们没有一套清晰的功能来收集、清理和探索定制区块链生态系统投资组合所需的关键数据。接下来的博文将为更深入的量化方法打下坚实的基础，这些方法与使用*数据科学*和*量化策略*的基于投资组合的波动性、聚类、预测和日志回报等相关。我们现在可以通过 [CrytpoCompare](https://www.cryptocompare.com/) 用他们优秀的 API 来聚集和交换特定的交易数据——对散户投资者是免费的。

这篇文章将介绍长时间序列的并行下载，加速比是 8 倍(取决于您机器上的内核数量)，我将探索每个可用符号的其他实用统计数据( *ticker* )。本帖将探讨利用以太坊等协议(即*平台*)的令牌的性质，以及基于 ERC20 等的令牌。我们将会看到，目前区块链市场的市值为 150，404，143，218 美元(1500 亿美元)，仍然只是其他替代市场的一小部分。

首先，您需要在本地机器上下载 Anaconda，并在一个环境中用 Python 3.5+设置一个 conda，然后启动一个 Jupyter 笔记本来运行 chunks 下面的代码。更好的是，如果你还没有尝试过，在 [Google Collab](https://colab.research.google.com) 中免费运行下面的代码。

![](img/7f72cc6f96d8628ede8867208eed6413.png)

接下来，我们将找出每个协议的令牌项目的数量。

![](img/9e07aabb476daa4b12c30972f4c38005.png)

Counts of Tokens per Protocol

*   这里注意，以太坊协议下有 1700+ ERC20 个令牌。WAVES 和 NEO 分别以 53 个和 25 个令牌紧随其后。然后，我们将创建一个树形图来可视化不同协议分组之间的比例。

![](img/cf2ef15c79d2d0ed7d9ff9bbe4f3a7dc.png)

Token Ratios among Platforms with >1 Coin

我们可以清楚地看到，基于以太坊 ERC20 的项目与市场上所有其他区块链协议相比存在很大的不相称性。这些项目帮助推动了 2017 年底至今的 ICO 市场。

当第一次为这篇文章构建代码时，我试图下载所有列出的 3500+资产，我将在这里为那些在连接所有 crytpo 聚合价格方面有困难的人提供代码。然后，我们将继续在一个并行例程中下载所有历史资产。

```
Wall time: 5min 17s!! for 150 series!!
```

现在，我们将绘制显示前 150 个资产的图表，其中颜色编码的条块表示相应项目中的一个或多个令牌。

![](img/5edeb5e6f6b060c24b9faa2e19d9e96e.png)

请注意，左边的高市值项目通常有一个以上来自其根协议的相关令牌。然后，我们将在一个并行化的例程中深入研究所有可用的令牌/协议——这将大大加快我们的处理时间。

创建一个二元图来显示项目从 2014 年到 2018 年的启动时间，按市值排序。

![](img/f66bf89f4780052b4ee10e6472f1965f.png)

在这里，我用 CryptoCompare 上可用的价格数据绘制了所有 2600 种资产。其他 1000 多项资产要么是前 ico 资产，要么目前没有收集每个符号的汇总交易数据。值得注意的是，在 2017 年 5 月和 2017 年 10 月/2018 年有许多新项目上线的平台期。一些市值较高的项目根据其发布日期增长非常快，而其他类似发布日期的项目却没有增长。接下来，我们将看看大市值 ERC20 令牌在此分布中的位置。

![](img/2276509e4d15173beb30f47d253d6ed7.png)

> 这里的图显示了所有 29 个方案的颜色编码。因为 ERC20s 统治着当前的生态系统，我们看到它们在右边的黄色部分。它们的市值较低，主要是从 2017 年底加密市场的峰值释放到市场上的。这种选择的爆炸可能是密码市场如此迅速衰落的一个重要原因，因为如此多的资本开始分散到许多代币产品中。

在 TechCrunch 最近的一篇[文章中，我们看到了“2018 年迄今已有 1000 个项目失败”。然而，截至 2019 年 1 月 30 日，仍有 2633 个活动项目。为了客观地看待问题，](https://techcrunch.com/2018/06/29/thousands-of-cryptocurrency-projects-are-already-dead/)[福布斯报道](https://www.forbes.com/sites/jeffkauflin/2018/10/29/where-did-the-money-go-inside-the-big-crypto-icos-of-2017/#556ae1e5261b)仅在 2017 年就提供了 800 个 ico。

![](img/f7bda2a0bcb4c7b94481ca86ec310119.png)

以下是 2014 年以来 50 个项目及其各自价格的随机样本。这些资产都是高度正相关的。用上面的并行代码证明给自己看。

我们将使用 Python 中的 Bokeh 模块从 StackOverflow 改编的 gist 中提取一些代码。它非常适合使用 D3 探索多个时间序列——具有缩放功能和基于悬停的信息。

![](img/05ecd4d8281d2eb0bdac9e1f1634a075.png)

同样，我们将按市值查看顶级资产。

![](img/ea4dfa20963ff880414d2980ebfabd8f.png)

> 波动性呢？？我们将在以后的文章中探讨这个问题。

现在我们想从 CryptoCompare 并行拉入**循环补给**，并将其与我们原来的 stats 合并，原来的 stats 只包括**最大补给**。前者是目前被广泛接受的衡量数字资产市值的方法。关于更适合区块链估值的未来和预期测量方法的更深入的讨论，请看[经济学](https://blog.nomics.com/essays/crypto-market-cap-review-emerging-alternatives/)的这篇精彩文章。另一个受欢迎的网站 Coinmarketcap.com 用流通供应量来计算市值。

现在合并统计数据，以便按协议探索聚合市场资本

![](img/f12e9ddf79558b03c43a8b526fb0af30.png)![](img/25a2a16f528b56d55865b27ae1d718e4.png)

现在，让我们按每个硬币的市值来绘制市值图。我们看到比特币(BTC)、Ripple (XRP)和以太坊(ETH)仍在市场份额上占据主导地位。之后，我们将展示协议生态系统中所有令牌的市场价值。

![](img/dc74fd5c55fb5e01f9052052ffd46d9a.png)

Market Cap Shares of Aggregate per Coin

接下来，我们显示了基于 marketcap 分组的协议的 marketcap。现在，基于 ERC20 的令牌开始挤占空间。CryptoCompare 似乎有一些令牌在 marketcap 计算中存在异常值。我希望他们尽快修复这些错误，这些错误在下面的代码中被过滤掉了，例如:

> “比特 CNY”、“BTF”、“NPC”、“MTN”、“圆点”、“I0C”、“阿米什”、“WBTC*”、“蜂巢”、“OCC”、“SHND”(它们似乎在市值计算中反映不准确)

![](img/85f3a9e4adca21ad34dd9b748e62d77a.png)

按协议市场份额分组，瑞士联邦理工学院现在仅次于 BTC。

![](img/1626fe4ddc9dbe4de49af09e6f48de27.png)

Source: [https://giphy.com/explore/bravocado](https://giphy.com/explore/bravocado)

感谢您花时间一起探索区块链的生态系统。期待回答您的任何问题。上述代码在一段时间内应该是完全可复制的。如果您对使用上述数据的帖子有任何建议，请分享您的想法。