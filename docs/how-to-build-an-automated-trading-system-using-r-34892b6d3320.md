# 如何使用 R 构建一个自动交易系统

> 原文：<https://towardsdatascience.com/how-to-build-an-automated-trading-system-using-r-34892b6d3320?source=collection_archive---------4----------------------->

![](img/e7244241a4eceb2bb8c228f22bbb1481.png)

Photo by [M. B. M.](https://unsplash.com/@m_b_m?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

对于所有的 R 狂热者来说，我们知道我们可以使用 R 非常高效地构建任何数据产品，自动化交易系统也不例外。无论你是做高频交易，日内交易，摇摆交易，甚至是价值投资，你都可以使用 R 构建一个交易机器人，它可以密切关注市场，并代表你交易股票或其他金融工具。交易机器人的好处显而易见:

1.  交易机器人严格遵循我们预先定义的交易规则。除了人类，机器人在做交易决定时没有任何情绪。
2.  交易机器人不需要休息。交易机器人可以跨多种金融工具观察每秒钟的市场价格运动，并在时机正确时立即执行订单。

几年前，我非常喜欢股票交易，有人建议我订阅一个股票推荐服务，通过电子邮件和短信发出实时买卖警报。由于当时我忙于白天的工作，我无法跟上他们的警报。我发现很多次，价格只是在买入警报出现后飙升，给我留下了很高的进场价格，或者相反，价格只是在卖出警报出现后下跌，给我留下了很低的出场价格。由于我对警报反应迟钝，我决定建立一个简单的自动交易系统，因为我了解了 r 的强大功能。我建立的系统可以执行以下任务:

1.  开市时自动启动系统，收市后关闭系统
2.  定期检查我的电子邮件收件箱，过滤来自服务提供商的电子邮件
3.  解析邮件获得股票代码、买入或卖出指标、目标价等。
4.  准备适当的订单，并通过 API 将其发送给经纪人。

这个系统一点也不复杂，但是它对我的用例来说是有效的。在网上资源的帮助下，花了一两周的时间进行编码和调试后，整个系统最终运行顺利😄。如果您也有使用 R 构建这样一个系统的需求，下面是我的简单逐步指南:

1.  **开立一个支持 API 的经纪账户**:你需要一个支持基于 API 的订单执行的经纪人，这样我们就可以实现交易自动化。这个领域的一个这样的经纪人是[互动经纪人](https://interactivebrokers.com/) (IB)。IB 全面的 API 和有竞争力的佣金结构非常适合我们的需求。
2.  服务托管:我们需要安排我们的 R 脚本在市场时间自动运行。为此，您可以使用台式机或笔记本电脑，但我建议您使用 AWS EC2 等全天候云服务器来托管我们的项目。至于操作系统，我推荐 Ubuntu，而不是默认的 AMI 映像，因为我们需要使用一些软件，这些软件稍后需要图形用户界面。
3.  **软件**:一旦我们启动了服务器，你就可以通过 SSH 连接到服务器并安装以下软件:

*   *R* :你无缘无故就要装 R。你可以在 [CRAN](https://cran.r-project.org/) 下载最新的 R for Linux 系统，并在那里找到安装说明。
*   Rstudio Server :我强烈推荐你使用 [Rstudio Server](https://www.rstudio.com/products/rstudio/download-server/) ，它不仅是一个很棒的 IDE，还能帮助你更有效地管理你的服务器，比如上传和下载服务器中的文件。
*   *R 包* : (1) [IBrokers](https://cran.r-project.org/web/packages/IBrokers/IBrokers.pdf) :这个包是你的脚本和交互经纪人之间的桥梁。它将大多数交互式代理的 API 调用包装到 R 函数中。(2) Gmailr:我们可以使用这个 API 包装器来捕捉 Gmail 中的邮件，这应该是大多数人的邮件选择。(3) [tidyverse](https://www.tidyverse.org/) :这个包是大部分 R 用户需要的几个有用的包的组合。(4)[data . table](https://github.com/Rdatatable/data.table/wiki):data . table 是我最喜欢的包之一。它让数据操作变得更加有趣。
*   Ubuntu 桌面:安贵让运行交互式经纪人交易软件变得更加容易。由于 EC2 的默认 ubuntu 没有附带 GUI 包，我们需要安装一个。关于如何详细安装 ubuntu 的 GUI，可以参考这篇优秀的[教程](https://medium.com/@Arafat./graphical-user-interface-using-vnc-with-amazon-ec2-instances-549d9c0969c5)。
*   *查看远程桌面*:由于 EC2 服务器没有显示器，为了远程查看桌面，我们需要在服务器上安装 VNC 服务。同样的[教程](https://medium.com/@Arafat./graphical-user-interface-using-vnc-with-amazon-ec2-instances-549d9c0969c5)涵盖了这方面的细节。
*   *互动经纪商的交易软件*:您需要安装互动经纪商的交易软件(交易员工作站或 IB 网关)来启用 API 功能。下载安装[离线版](https://www.interactivebrokers.com/en/index.php?f=15875)后，可以通过上一步安装的远程桌面启动程序。
*   *IBcontroller(可选)*:互动券商的交易软件，每天在指定时间自动关机。如果你不想每天重新打开程序，我建议你安装 [IBcontroller](https://github.com/ib-controller/ib-controller) ，在这里你可以设置一个参数让交易软件“永远在线”。

4.**写一个** **R 脚本**:是时候写点 R 代码了。下面是我为我的程序写的一些例子片段。我觉得你们应该根据自己的需求来设计。

*   获取销售警报电子邮件线程 id 的简单 R 代码片段:

*   下订单的简单 R 代码片段:

上面的脚本所做的只是简单地下一个 100 股“ABC”股票的 10 美元限价买入单。让我们来分解一下这些功能:

*   `twsConnect`:该功能建立与交易软件的连接。 *twsSTK* :此功能是您可以输入符号的地方。
*   `*reqIds*`:该函数生成订单 ID。
*   `twsOrder`:该功能指定订单的详细信息。
*   `twsDisconnect`:此功能断开与交易软件的连接。

5.**任务调度器**:你可以使用 crontab 来调度服务器在特定的时间范围内执行你的 R 脚本。我使用的 crontab 时间表是

基本上就是在周一到周五的上午 9:27 执行`trading_robot.R`脚本。至于在市场关闭后退出脚本，我让脚本自己通过如下所示的 while 循环来处理

7.**申报/仪表盘:**在交易软件中随时可以登录远程桌面查看自己的持仓或账户信息，但是需要你 ssh 到服务器，打开客户端 VNC 软件，不太方便。为了提高效率，你可以使用[的闪亮服务器](https://www.rstudio.com/products/shiny/shiny-server/)和[的 flexdashboard](https://rmarkdown.rstudio.com/flexdashboard) 构建一个简单的仪表板，这样它们可以获取后端数据，并在前端网站上更好地显示出来，只需通过一个 URL 就可以访问。

全部完成！我希望这篇文章能帮助你建立你的交易机器人，如果你有任何问题，请告诉我。

***来自《走向数据科学》编辑的提示:*** *虽然我们允许独立作者根据我们的* [*规则和指南*](/questions-96667b06af5) *发表文章，但我们并不认可每个作者的贡献。你不应该在没有寻求专业建议的情况下依赖一个作者的作品。详见我们的* [*读者术语*](/readers-terms-b5d780a700a4) *。*