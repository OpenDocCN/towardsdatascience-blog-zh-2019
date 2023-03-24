# 一个无服务器管道，用于从 Twitter 检索、验证数据，并将数据放入 Azure SQL Server。

> 原文：<https://towardsdatascience.com/a-serverless-pipeline-to-retrieve-validate-and-immerse-the-data-to-azure-sql-server-from-twitter-1a5757b39bc8?source=collection_archive---------24----------------------->

## 学习如何做数据科学就像学习滑雪一样。你必须这么做。

## 项目陈述:

给定一个 twitter ID，获得最少 100 个关注者(修改后保持在 Azure 功能 5-10 分钟超时时间)，每个关注者收集最多 200 条推文。
将元组(twitterID，followerID，tweetID，tweet)存储到 Azure SQL 服务管理的表中。
1)您必须创建并设置一个免费的 Azure 帐户。
2)在 Azure 账户中创建一个数据库和一个表。用 API 创建一个 twitter 账户
4)。给定 twitter ID，收集该 twitter ID 的追随者 ID
4.1)为每个追随者 ID 收集多达 200 条原始推文
— —不包括转发、消息
5)将其存储到 Azure 表中
6)编写一个客户端来查询 Azure 表。
6.1)列出给定 twitter ID 的所有推文
6.2)列出给定 twitter ID 的关注者 ID

## 使用的技术:

1.  计算机编程语言
2.  Twython 库提取 tweeter 数据
3.  Azure 函数
4.  Azure SQL server

## 我学到的东西:

1.  从本地主机以及 Azure 无服务器和 Azure 数据块使用 Azure SQL Server。
2.  Azure 功能。
3.  学会了使用 twython 库来提取推文。

## 进行[项目](https://github.com/ksw25/Extract-Data-From-Tweeter-And-Save-In-Azure-SQL-Using-Azure-ServerLess)时遵循的步骤的简要总结:

1.  创建了一个 [Tweeter 开发者账号](https://developer.twitter.com/en/apply-for-access.html)。
2.  编写了一个 python 脚本来提取给定用户 ID 的追随者 ID。

Code for getting followers ID for a Given User ID

3.编写了一个 python 脚本，获取上一步提取的关注者 ID，并为每个人检索最多 200 条推文。

Code for getting followers Tweet

4.创建了 Azure SQL 数据库。

5.编写了一个 python 脚本来获取步骤 3 的结果，并将其保存到 Azure SQL server。

Code to Save extracted data to Azure SQL server

6.我创建了一个 Azure 函数项目和 func。修改脚本以处理 Azure 函数。

7.出于以下目的，再创建 2 个客户端函数。

*   列出给定 twitter ID 的所有推文

List all tweets for a given twitter ID

*   列出给定 twitter ID 的关注者 ID

List follower ID for a given twitter ID

## 每个 Azure 功能的链接:

(这些是模板，供大家过目。我已经关闭了这些链接的激活，这样它们就不会工作了。很遗憾，我并不富有😢 😭。

任务 1:保存关注者的 ID 和他们各自的推文。(放在 MyFunctionProj 目录的 TweetWork 中)

*   [https://demo.azurewebsites.net/api/Tweetwork](https://demo.azurewebsites.net/api/Tweetwork)
*   比如 https://demo.azurewebsites.net/api/Tweetwork?name=25073877 的

任务 2:列出给定 twitter ID 的所有推文。(放在 MyFunctionProj 目录下的 Client1BigData 中)

*   [https://demo.azurewebsites.net/api/Client1BigData?code = nyhlelxnjbz 08 qbutk 1 jkbaylvdje 9 vaknx 09 cn 1 vrg = =](https://demo.azurewebsites.net/api/Client1BigData?code=NyhLElXnjBz08QButk1jkbaYLVdJE9vAKnX09CN1vrg==)
*   例如 https://demo.azurewebsites.net/api/Client1BigData?的[code = nyhlelxnjbz 08 qbutk 1 jkbaylvdje 9 vak 9 cn 1 vrg = =&name = 979178022367461376](https://demo.azurewebsites.net/api/Client1BigData?code=NyhLElXnjBz08QButk1jkbaYLVdJE9vAK9CN1vrg==&name=979178022367461376)

任务 3:列出给定 twitter ID 的关注者 ID。(放在 MyFunctionProj 目录下的 Client2BigData 中)

*   【https://demo.azurewebsites.net/api/Client1BigData? code = nyhlelxnjbz 08 qbutk 1 jkbaylvdje 9 vak 9 cn 1 vrg = =
*   例如[https://demo.azurewebsites.net/api/client2bigdata?code = 2MO/r/wvk 5 jqfsbq 1 kka 0 hdwf 1 cfdeyzjpenponkvgis 57 waw = =&name = 25073877](https://demo.azurewebsites.net/api/client2bigdata?code=2MO/r/Wvk5JQFsbQ1KKkA0hdWF1OCfdeyZjpENpoNkVGIS57Waw==&name=25073877)

## 面临的挑战:

*   如果你在 Visual studio 中使用 Mac 调试 Azure 函数，这是非常困难的，因为有时 Visual Studio 不会创建一个确切的扩展/帮助文件来进行调试。我个人觉得，对我来说，一点效果都没有。每次我想检查它的时候，我不得不把功能推到在线。但是我现在找到了解决办法。里面有三个文件。vscode 有时会出错。我会提到他们，应该是什么样子。即，

1.  *task.json*

task.json

*2。launch.json*

*launch.json*

*3。settings.json*

*settings.json*

*   与 AWS lambda 相比，Azure Functions 有其局限性。当我开始写它的时候，我以为它会和 AWS lambda 一样，因为两者都是无服务器的，但是实现它非常困难，原因有二。第一，Azure 功能不允许在线代码编辑，这是 AWS 提供的。

## 跟进:

*   如果我是为一家公司做这个，并且有足够的资源，我会选择 Azure function Dedicated App Plan，它的最大时间限制是 30 分钟。

github:[https://github . com/ksw 25/Extract-Data-From-Tweeter-And-Save-In-Azure-SQL-Using-Azure-server less](https://github.com/ksw25/Extract-Data-From-Tweeter-And-Save-In-Azure-SQL-Using-Azure-ServerLess)

## 致谢:

*   这是我在 NYU 坦登工程学院完成的 CS6513 大数据工具和技术的一部分
*   我感谢 IBM Power Systems 学术计划对计算资源的支持。
*   我感谢 MSFT Azure 为学生提供免费的 Azure 访问。

问候，

卡兰普里特·辛格·瓦德瓦

**硕士**在**计算机科学**| 2020 级

纽约大学坦登工程学院

研究生助教—计算机视觉|纽约大学

[karan.wadhwa@nyu.edu](mailto:karan.wadhwa@nyu.edu)|(929)287–9899 |[LinkedIn](https://www.linkedin.com/in/karanpreet-wadhwa-540388175/)|[Github](https://github.com/ksw25)