# 气流和超级查询

> 原文：<https://towardsdatascience.com/airflow-and-superquery-9acc87c0398d?source=collection_archive---------12----------------------->

## 使用 SuperQueryOperator 实时监控您的大查询开销

![](img/f95e8494758ba2e1f889996b30f5f162.png)

Photo by [Soheb Zaidi](https://unsplash.com/photos/W04aGUFOxf0?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

“费用是多少？”这个问题在科技界被问得如此频繁，以至于一家小型初创企业的每个人在被问到这个问题时都会微微颤抖。答案总是:“我们不确定”。

在数据工程领域，调度工作流的最佳工具之一是 [Apache Airflow](https://github.com/apache/airflow) 。这个工具让许多企业摆脱了僵化的 cron 调度，在有向无环图(Dag)的公海上驾驭大数据浪潮。

当然，这意味着大量数据被移入和移出数据库，伴随着这一辉煌的运动，往往会产生不可避免的成本。

一个这样的数据库，你可以称之为超级计算机，叫做[谷歌大查询](https://cloud.google.com/bigquery/)。它是谷歌云产品的旗舰，支持 Pb 级的数据处理。它非常善于让您少担心数据库基础设施的能力，而更多地担心您的分析质量和需要解决的数据流问题。

BigQuery 要考虑的一个关键因素是，任何个人或组织对提高在这个平台上扫描数据的成本有多开放。即使是最精明的数据工程师也会焦虑地告诉你，他们在扫描他们并不真正想要的数据时犯了错误，并使他们的业务每月分析账单超出了预算。

在步骤 [superQuery](http://superquery.io) 中。superQuery 提供的理念是，你不必担心你的成本，因为你有你需要的所有信息和一些保障措施，让你有机会做出明智的决定。

# 了解气流成本

![](img/efaeaa193459b3498fe2c0b00d2efebd.png)

当您的气流 Dag 愉快地搅拌并将数据推送到您选择的处理系统时，大量日志记录在后台发生。气流日志易于访问，易于阅读，并让您很好地了解 DAG 正在做什么。如果日志文件还可以显示查询执行计划的信息，特别是**成本和扫描的总数据**是多少，这不是很好吗？肯定会的！大概是这样的:

```
--------------------------------------------------------------------
Starting attempt 1 of 4
--------------------------------------------------------------------

[2019-03-11 21:12:02,129] {models.py:1593} INFO - Executing <Task(SuperQueryOperator): connect_to_superquery_proxy> on 2019-03-01T00:00:00+00:00
[2019-03-11 21:12:03,836] {superq_operators.py:54} INFO - Executing: #standardSQL
SELECT COUNT(testField) FROM `mydata.PROD.myTable`;
[2019-03-11 21:12:03,844] {logging_mixin.py:95} INFO - [2019-03-11 21:12:03,843] {base_hook.py:83} INFO - Using connection to: id: mysql_default. Host: superproxy.system.io, Port: 3306, Schema: None, Login: XXXXXX, Password: XXXXXXXX, extra: {}
[2019-03-11 21:12:15,172] {superq_operators.py:68} **INFO - ((
'{
   "startTime":1552331525642,
   "endTime":1552331534624,
   "executionTime":"8988",
   "bigQueryTotalBytesProcessed":****26388279066****,
   "bigQueryTotalCost":"0.12",
   "superQueryTotalBytesProcessed":0,
   "superQueryTotalCost":"0.00",
   "saving":0,
   "totalRows":"1",
}**', '', '1', 'true'),)
[2019-03-11 21:12:17,121] {logging_mixin.py:95} INFO - [2019-03-11 21:12:17,119] {jobs.py:2527} INFO - Task exited with return code 0
```

这个日志块告诉您，您在 Airflow 中的 BigQuery 操作符扫描了 24Gb 的数据，花费了您 0.12 美元。简单。您还可以在第三方工具或 bash 脚本中解析日志文件，并创建扫描 BigQuery 数据的 Dag 的开销汇总。

![](img/baaef7eabc1ab5113718e04a4d40db09.png)

# **这一切是如何运作的？**

SuperQuery 使用 MySql 代理来实现普遍连接，并提供一个 Sql 接口来获取信息。

# 下一步需要什么:超级查询运算符

为了获得与上述相同的功能，需要执行以下步骤:

1.  将 superquery 插件添加到 Airflow，以便使用 SuperQueryOperator。
2.  订阅 superQuery 试用版，获取 superQuery MySql 代理的登录信息
3.  使用下面提供的 DAG 测试您与代理的连接。
4.  当您想要使用这个功能时，在您自己的 Dag 中用 SuperQuery 操作符替换您的 BigQuery 操作符。

# 使用超级查询运算符

这是超级查询操作符的界面:

```
TEST_SQL = """#standardSQLSELECT COUNT(*) FROM `mydata.PROD.myTable`;"""SuperQueryOperator( *task_id*="connect_to_superquery_proxy", *sql*=TEST_SQL, *database*="", *explain*=True,  # False if you don't want information *dag*=dag)
```

以下是操作员的代码，您应该将其复制到 Airflow 中的`plugins` 文件夹中:

下面是一些测试 SuperQuery 连接的代码:

# **总之**

这篇文章描述了当您连接到 BigQuery 并从 big query 实现气流任务时，如何获得气流成本的视图。超级查询代理的使用可以扩展到包括各种详细的执行计划信息，并利用系统提供的好处。

快乐的成本监控(和节约)！