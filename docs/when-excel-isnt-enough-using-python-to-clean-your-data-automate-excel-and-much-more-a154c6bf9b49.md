# 当 Excel 不够用时:使用 Python 清理数据、自动化 Excel 等等…

> 原文：<https://towardsdatascience.com/when-excel-isnt-enough-using-python-to-clean-your-data-automate-excel-and-much-more-a154c6bf9b49?source=collection_archive---------0----------------------->

![](img/e81225077900e3f59936ab4b8b7a01d3.png)

[@headwayio](https://unsplash.com/@headwayio)

## 数据分析师如何在电子表格驱动的组织中生存

Excel 是许多公司非常流行的工具，数据分析师和数据科学家经常发现自己将它作为日常数据分析和可视化工具的一部分，但并不总是出于自愿。这当然是我第一份数据分析师工作的经历，在那份工作中，Excel 是每个人工作流程的一部分。

我的团队使用 Excel 的数据工具 Power Query 来聚合和操作 CSV 文件，并连接到我们的数据库。最终，这些数据将显示为数据透视表或仪表板，并与公司的其他部门共享。我拿到的几乎所有报告都是在 Excel 上运行的，我很快就意识到这是一个大问题。

为了向您描述一下，下面是我多次听到我的同事谈论 Excel 的一些事情，我自己最终也开始这么说:

> *“又死机了！!"*

刷新 Excel 报告中的数据是一项日常任务，有时，这是我们唯一可以立即执行的任务。即使我们的电脑有像样的硬件，我们一打开其他程序就知道了(有人说 Chrome 吗？)虽然 Excel 正在刷新，但几乎可以肯定它会崩溃。

> “还在提神……”

不仅我们在刷新 excel 时无法使用其他应用程序，而且我们的一些报告需要 30 分钟甚至几个小时才能完成刷新。是的，Excel 喜欢拿我们的电脑当人质！

> “我们无法加载那么多数据。”

我们对 Excel 最大的不满是无法加载我们需要的那么多数据。公司里的每个人都要求更多，我们根本无法满足。

很明显需要做些什么。我们在这些问题上浪费了太多时间，几乎没有时间进行任何实际的分析或预测。幸运的是，我非常熟练地使用 Python 和它的工具来操作 CSV 文件，所以我和我的团队开始了优化我们报告的长期任务。

由于我们需要继续在 Excel 中报告，并且没有预算购买 BI 工具，我们决定使用 Python 来完成所有繁重的工作，并让 Excel 负责显示数据。因此，在 Python 和 Windows 任务调度程序的帮助下，我们自动化了收集数据、清理数据、保存结果和刷新 Excel 报告的整个过程。

由于每个人的工作流程是不同的，我想使这篇文章尽可能有用，我将保持高水平，并包括一些伟大的教程的链接，以防你想更深入地挖掘。请记住，其中一些提示可能只在 Windows 机器上有效，这是我当时使用的。

[](https://medium.com/@avourakis/membership) [## 通过我的推荐链接加入 Medium-Andres Vourakis

### 阅读安德烈斯·沃拉基斯(以及媒体上成千上万的其他作家)的每一个故事。您的会员费直接支持…

medium.com](https://medium.com/@avourakis/membership) 

## 1.从 FTP 服务器下载数据

使用 Python 中的`ftplib`模块，您可以连接到 FTP 服务器并将文件下载到您的计算机中。这是我几乎每天都使用的模块，因为我们从外部来源接收 CSV 报告。以下是一些示例代码:

要了解更多关于 FTP 服务器和如何使用`ftplib`的信息，请查看本[教程](https://pythonprogramming.net/ftp-transfers-python-ftplib/)。

## 2.运行 SQL 查询

使用 Python 中的`pyodbc`模块，可以轻松访问 ODBC 数据库。在我的例子中，我用它连接到 Netsuite 并使用 SQL 查询提取数据。以下是一些示例代码:

请注意，为了让模块正常工作，您需要安装适当的 ODBC 驱动程序。更多信息请查看本[教程](http://cdn.cdata.com/help/DNB/odbc/pg_usageinpython.htm)。

## 3.清理数据

使用 Python 中的`pandas`模块，您可以非常容易和高效地操作和分析数据。毫无疑问，这是我拥有的最有价值的工具之一。以下是一些示例代码:

这个[教程](https://www.datacamp.com/community/tutorials/pandas-tutorial-dataframe-python)是开始学习`pandas`的好地方。如果您正在处理大型文件，那么您可能也想看看这篇关于使用 pandas 处理大型数据集的文章。它帮助我减少了很多内存使用。

## 4.刷新 Excel

使用 Python 中的`win32com`模块，您可以打开 Excel，加载工作簿，刷新所有数据连接，然后保存结果。这是如何做到的:

我还没有偶然发现任何关于`win32com`模块的好教程，但是这个[堆栈溢出线程](https://stackoverflow.com/questions/40893870/refresh-excel-external-data-with-python)可能是一个很好的起点。

## 5.在规定的时间运行脚本

在 Windows 任务计划程序的帮助下，您可以在规定的时间运行 python 脚本并自动完成工作。你可以这样做:

启动任务调度器，找到位于**动作**窗格下的**创建基本任务**动作。

![](img/53cf80b7d6860c7b01c72eb712139848.png)

点击**创建基本任务**打开一个向导，您可以在其中定义任务的名称、*触发器*(当它运行时)和*动作*(运行什么程序)。下面的屏幕截图显示了**操作**选项卡，在这里您可以指定要运行的 Python 脚本的名称以及脚本的任何参数。

![](img/c3ebe311a4079b5841dd7accf560bcef.png)

关于创建任务的更多细节，请查看本[教程](https://www.esri.com/arcgis-blog/products/product/analytics/scheduling-a-python-script-or-model-to-run-at-a-prescribed-time/)。

通过将 Python 引入等式，我和我的团队能够显著减少我们处理数据所花费的时间。此外，将历史数据纳入我们的分析不再是一项无法完成的任务。这些改进不仅解放了我们，让我们可以更多地进行分析性思考，还可以花更多的时间与其他团队合作。

我希望这篇文章对你有用。如果你有任何问题或想法，我很乐意在评论中阅读:)

另外，如果你希望支持我成为一名作家，可以考虑[注册成为一名媒体会员](https://medium.com/@avourakis/membership)🙏