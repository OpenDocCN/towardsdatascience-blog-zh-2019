# 通过 Spring Boot 或 Java 使用 Apache Drill，使用 SQL 查询来查询数据

> 原文：<https://towardsdatascience.com/use-apache-drill-with-spring-boot-or-java-to-query-data-using-sql-queries-f6e186a5f31?source=collection_archive---------32----------------------->

> 原文发布于此:[https://blog . contact sunny . com/tech/use-Apache-drill-with-spring-boot-or-Java-to-query-data-using-SQL-queries](https://blog.contactsunny.com/tech/use-apache-drill-with-spring-boot-or-java-to-query-data-using-sql-queries)

在过去的几篇文章中，我们看到了如何将 Apache [Drill 与 MongoDB](https://blog.contactsunny.com/tech/getting-started-with-apache-drill-and-mongodb) 连接起来，以及如何将[与 Kafka](https://blog.contactsunny.com/tech/analyse-kafka-messages-with-sql-queries-using-apache-drill) 连接起来，使用简单的 SQL 查询来查询数据。但是，当您想将它应用到实际的项目中时，您不能整天坐在那里从终端查询数据。你想写一段代码来替你做脏活。但是如何在代码中使用 Apache Drill 呢？今天，我们将看看如何用 Spring Boot 或几乎任何其他 Java 程序来实现这一点。

![](img/1cfbec7fd14daf6a7e5d1fc9e83c74b5.png)

# 依赖关系

对于这个概念验证，我将编写一个简单的 Spring Boot CommandLineRunner 程序。但是您可以使用几乎任何其他 Java 框架或普通的 Java 代码来实现这一点。如果你有一个依赖管理工具，比如 Maven 或者 Gradle，你可以在项目中添加依赖，然后继续编码。否则，可以添加所需的*。jar* 文件到类路径，这样应该就可以了。所有需要的*。jar* 文件随 Apache Drill 的发行版一起提供。

这个项目我们会和 Maven 一起去。在项目的 *pom.xml* 文件的*依赖项*部分中添加以下 Drill 依赖项。

```
<dependency>
	<groupId>org.apache.drill.exec</groupId>
	<artifactId>drill-jdbc-all</artifactId>
	<version>1.16.0</version>
</dependency>
```

这种依赖性在 Maven 中央存储库中是可用的，因此您不需要添加任何额外的存储库。

# 代码

现在让我们开始有趣的部分。首先，我们需要在 Java 代码中加载 Drill JDBC 驱动程序。为此，我们将使用 *Class.forName()* 方法:

```
Class.forName("org.apache.drill.jdbc.Driver");
```

接下来，我们需要使用 DriverManager 和 Drill JDBC 连接字符串创建一个连接。如果您使用 Apache Drill 的独立本地实例，则需要使用以下连接字符串:

```
jdbc:drill:drillbit=localhost
```

如果您将 Drill 与 Zookeeper 一起使用，则必须使用以下连接字符串:

```
*jdbc:drill:zk=local*
```

因为我使用的是本地实例，所以我将创建一个连接来进行如下钻取:

```
Connection connection = DriverManager.getConnection("jdbc:drill:drillbit=localhost");
```

接下来，我们需要创建一个*语句*对象，使用它我们将运行实际的查询。一旦我们有了对象，我们将使用*。executeQuery()* 方法在对象上得到我们的 *ResultSet* :

```
Statement statement = connection.createStatement();
ResultSet resultSet = statement.executeQuery("select * from kafka.`drill` limit 10");
```

我使用了与前面的 MongoDB 和 Kafka 示例相同的数据集和示例。

从上面的例子可以看出，我们从 Kafka 主题 *drill 中获取数据。*让我们看看如何访问数据:

```
while(resultSet.next()){
    logger.info("ID: " + resultSet.getString(1));
    logger.info("First Name: " + resultSet.getString(2));
    logger.info("Last Name: " + resultSet.getString(3));
    logger.info("Email: " + resultSet.getString(4));
    logger.info("Gender: " + resultSet.getString(5));
    logger.info("IP Address: " + resultSet.getString(6));
    logger.info("----------------------------------");
}
```

接下来，让我们查询一个 MongoDB 数据库并从中获取一些数据:

```
resultSet = statement.executeQuery("select * from mongo.drill.sampleData limit 10");
```

我们可以使用上面相同的 *while()* 循环来打印数据。

差不多就是这样。我们现在能够在 Java 程序中使用 Apache Drill 作为简单的 JDBC 源，从 Drill 支持的任何源获取数据。您可以对复杂的 SQL 查询进行同样的尝试，看看 Drill 是如何工作的。

如果你对这个例子中完整的 Spring Boot 项目感兴趣，你可以在[我的 GitHub 简介](https://github.com/contactsunny/ApacheDrillSpringBootPOC)上查看。