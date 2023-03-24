# 如何从 IEX 下载所有历史日内 OHCL 数据:用 Python，异步，通过 API &免费。

> 原文：<https://towardsdatascience.com/how-to-download-all-historic-intraday-ohcl-data-from-iex-with-python-asynchronously-via-api-b5b04a31b187?source=collection_archive---------4----------------------->

你正在寻找一种免费获取大量 1 分钟内股价数据的方法吗？回测你的交易算法并在本地训练你的模型？有一个非常好的方法来做到这一点，我将在下面向你展示。

![](img/81ad2ce165d2035eef7dddd822bd2f60.png)

Photo by [grz3s](https://pixabay.com/de/users/grz3s-642680/) from [Pixabay](https://pixabay.com)

> 在这篇文章中，我将告诉你如何从 IEX 交易所下载免费的日内数据。请注意，IEX 是美国的一家证券交易所，每天交易的股票代码超过 8000 个。没听说过？点击了解更多信息[。](https://en.wikipedia.org/wiki/IEX)

重要的是，这篇文章将向你展示如何从 IEX 交易所**有效地、定期地**下载所有> 8K 票单的 OHCL 日内数据**。**

# 为什么从 IEX 获取数据？

你应该从 IEX 获得日内数据有三个原因:

1.  **免费的。IEX 是唯一一个让你定期免费下载大量日内 OHCL(开盘价、最高价、收盘价、最低价)数据的平台。通常，如果您想要获得大型数据集，日内数据是昂贵的。**
2.  **IEX 的空气污染指数很高。**[IEX API](https://iexcloud.io/docs/api/)有很好的文档记录，易于理解，最重要的是它完美地支持异步请求。
3.  你可以在羊驼上进行纸张交易。 [羊驼](https://alpaca.markets/)是我发现的算法交易最好的 API。他们为 IEX 股票代码提供免费、无限制的纸上交易。因此，你可以直接将你用 IEX 数据开发的策略付诸实践。

# 金融数据的替代 API 服务是什么？

让我提一下我测试过的其他三个服务，你可以考虑作为当天 OHCL 数据的替代来源。出于特殊的原因，所有这些都可能是有趣的。

1.  [**世界交易数据**](https://www.worldtradingdata.com/documentation#intraday-market-data) :如果你想从纳斯达克、纽约证券交易所以及非美国交易所收集当天的 OHCL 数据，这是一个非常便宜的选择。示例:通过 16 美元/月的计划，您可以获得 ca 的 2 分钟分辨率数据。50k 股票。然而，这将需要您频繁下载(每隔 2 周，每天 1 个会话，持续 2 周)，并且您的下载将非常慢，因为根据我的经验，World Trading Data 不太支持异步请求。
2.  [**Tiingo**](https://api.tiingo.com/products/iex-api) **:** 如果你想收集 IEX 自大约。2017，Tiingo 是最便宜的选择。它只会花费你大约。总共 10 美元，因为 Tiingo 的 API 调用限制非常宽松。注:IEX API 不允许您访问过去超过 30 个日历日的日内数据。因此，如果您希望快速获得更长时间的历史数据，Tiingo 可能会很有意思。相比之下，IEX API 非常适合定期和完全免费的下载。
3.  [**Alphavantage**](https://www.alphavantage.co/documentation/#intraday)**:**如果您希望定期收集纳斯达克、纽约证券交易所等其他交易所的当天 OHCL 数据，并且不需要大量的股票代码，那么 alpha vantage 可能足以免费获得这些数据。Alphavantage 涵盖了大量的报价机。然而，免费计划的 API 调用限制相当有限(每分钟 5 个 API 调用，每天 500 个调用)。

# 剧本的目的是什么？

下面介绍的脚本是我个人用来从 IEX 收集 1 分钟日内数据的。您可能希望根据自己的目的调整脚本。然而，也许这正是你想要的。我写这个剧本时有以下三个目的:

1.  **收集数据用于后续处理**例如回测和训练机器学习模型。(不是:用数据喂饱你的生活算法)
2.  **将数据存储在本地**例如，存储在您的电脑上，或者甚至存储在您电脑上的云存储文件夹中。(不是:将数据放入数据库)
3.  **使下载变得简单**例如，我希望能够在任何时候用`python download_IEX.py`运行脚本(不是:必须传递任何参数或必须在特定日期进行下载)

# 脚本是如何工作的？

每当您用`python download_IEX.py`执行[脚本](https://github.com/juliuskittler/IEX_historical-prices/blob/master/script/download_IEX.py)时，您就开始了一个新的下载会话(对于过去有数据可用但尚未下载的所有日期)。每个下载会话的工作方式如下:

1.  **初始化日志:**脚本为当前会话启动一个新的日志文件(见`init_logging()`)。例如，如果您在 10 月 26 日进行培训，您将在`script/log/20191026.log`中找到相应的日志文件。实际上，每个日志文件都包含执行脚本时可以在终端中看到的打印输出。
2.  **获取日期:**脚本获取所有必须完成下载的日期(参见`get_dates()`)。为此，它检查`output`目录中的现有文件夹，并记下数据已经下载的日期。然后，所需的日期被计算为过去 30 天或更短时间内且尚未在`output`文件夹中的所有日期。
3.  **进行下载:**然后，脚本为步骤 2 中提取的每个日期进行一个单独的异步下载会话。对于每个日期，准备一个异步下载会话(参见`asyncio_prep()`，例如，从 IEX API 获取所有可用的报价机。然后，使用函数`download_tickers_asynchronous()`执行该会话，该函数异步读取和写入各个报价机的数据(参见`get_csv()`和`write_csv()`)。

# 我如何设置脚本？

要设置脚本，您需要完成以下三个步骤。

1.  **将存储库下载到您的计算机上**

你可以在这里找到 Github 库。转到`Clone or download`，然后点击`Download zip`，下载存储库。然后，将下载的文件`IEX_historical-prices-master.zip`放在计算机上您选择的文件夹中，并解压缩 zip 目录。

现在，您应该有一个包含以下文件的文件夹(使用您选择的名称):

![](img/30059237e8b760aa3044f48d79917e0f.png)

2.**从 IEX 获得免费的 API 密匙**

去 [IEX 云](https://iexcloud.io)网站免费注册。然后在`SECRET`找到`API Tokens`下你的免费 API 令牌(见下面截图中的红色方块)。

![](img/8f8f9b17bf35a59240210b83ff230393.png)

3.**建立你的** `**script/config.json**` **文件**

在您选择的文本编辑器中打开一个新文件，例如`Visual Studio Code`。然后，将以下内容放入新文件的前三行，并用上一步中的`SECRET`令牌替换`YOUR_TOKEN`。

```
{"TOKEN": "YOUR_TOKEN"}
```

现在，点击`Save as`，在`script`目录下保存名为`config.json`的新文件。您的文件夹现在应该看起来像这样:

![](img/4f5876bb348fd745f6267a9ecdee4da3.png)

# 我如何开始下载？

如上所述设置好脚本后，您可以在`script`文件夹中打开一个新的终端，并使用`python download_IEX.py`执行脚本。如果安装了所有需要的软件包(参见`download_IEX.py`开头的导入)，脚本将开始下载 IEX 当天的数据。

在终端中，您将看到每个下载文件的以下内容:
时间戳、股票代码、股票代码的索引(在所有可用的 IEX 股票代码中)以及股票下载的日期:

![](img/e7db42c42be5e72113b0b5c1b019e01d.png)

# 我如何中断下载？

您可以通过中断正在执行的脚本来中断下载。(如果你在终端中执行这个脚本，`control+c`在 Mac 上也可以。)稍后，您可以通过使用`python download_IEX.py`重新启动来恢复脚本，但是您必须在中断脚本时重新开始下载当前正在下载的数据。

重要的是，默认情况下，脚本启动时不会删除任何文件或文件夹。因此，您可能想要手动删除未完成下载的日期的文件夹。这样，当您使用`python download_IEX.py`重新启动时，脚本将为提到的日期重新开始。

# 我应该多久下载一次？

如果您希望使用此脚本定期下载 IEX 数据，您将希望至少每四周运行一次。原因是根据文档，API 允许你下载“30 个连续日历日”的数据(参见 [IEX 文档](https://iexcloud.io/docs/api/#historical-prices))。这意味着，如果您只进行一次下载会话，例如每八周一次，您将有几天缺少数据。

然而，我实际上能够获得超过 30 个连续日历日的数据。可能文件的意思是“30 个连续日历*周*天”。无论如何，每四周执行一次脚本是安全的。

请注意，无论何时运行该脚本，您都不需要设置任何参数，因为它会自动检测哪些日期您已经下载了数据，哪些日期您可以从 IEX 下载数据。因此，您也可以每两周执行一次脚本，或者在下载会话之间有不规则的中断(有时两周，有时更长)。

# 我如何利用下载的文件？

您可以看到该脚本自动创建了一个按年份、ISO 日历周和日期的文件夹结构。对于每个日期，脚本都会执行一个单独的异步下载会话。这意味着每个交易日都有包含单独 csv 文件的文件夹:每个股票代码一个 csv 文件。

实际上，该脚本为您提供了来自 IEX 的原始数据。以后如何处理这些数据取决于您。例如，您可能希望为同一个跑马灯追加所有文件，以便为每个跑马灯符号获得一个单独的文件。

**重要提示:**

*   当某一天的异步下载会话完成时，该天的文件夹会自动压缩以节省空间。因此，您可能希望在处理原始数据之前解压缩文件夹。
*   每个日期文件夹包含三个数据文件夹`DONE`、`ERROR`和`NONE`。您将需要使用文件夹`DONE`中的文件，因为这些文件包含有效股票代码的实际日内数据。文件夹`ERROR`中的文件大部分是来自 IEX 的测试 ticker(见本 [GitHub 问题](https://github.com/iexg/IEX-API/issues/1266))，文件夹`NONE`中的文件对应于当天没有任何交易的 ticker(见本 [GitHub 问题](https://github.com/iexg/IEX-API/issues/931))。

![](img/eb116d339d0b7d886b9e81f7087442c4.png)

# 下载的数据会是什么样子？

在给定日期的`DONE`文件夹中，您会发现每个 ticker 都有一个 csv 文件:

![](img/5256ea3dc77a1ec569d121e8b29915cf.png)

并且这些文件中的每一个都包含相同的 OHCL 信息。以下是苹果公司(AAPL)的一个例子:

![](img/3ea2706a473a32c5367845735f9149c5.png)

# 参考

*   [Github 上的源代码](https://github.com/juliuskittler/IEX_historical-prices/blob/master/README.md)
*   [IEX 云文档](https://iexcloud.io/docs/api/#historical-prices)
*   [IEX 票据交易所的 API](https://alpaca.markets/)

**感谢您的阅读，期待您的反馈！**