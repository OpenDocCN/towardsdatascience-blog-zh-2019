# 在 PostgresSQL 中进行设置

> 原文：<https://towardsdatascience.com/getting-set-up-in-postgressql-e29db4e6d7fc?source=collection_archive---------28----------------------->

## 一起使用 PSQL 和 Python 熊猫

在学习了编写结构化查询语言(SQL)查询的入门教程后，我面临的最大挑战之一是如何实际使用这些查询。教程给了我们一个很好的开始环境，但并不总是给我们自己的工作提供一个容易接近的桥梁。下面是 SQL 的快速浏览，以及使用 PSQL 服务器进行查询的教程。

首先，我们需要理解为什么我们需要数据库。当作为学生学习时，我们经常使用相对较小的数据集，这些数据集可以保存在我们自己的机器上。然而，在实践中，我们需要存储和检索对于单个机器的内存来说太大的数据的能力。对数据库的需求源于当时被认为是大数据的结果。

数据库系统地存储信息，以便于存储、操作和管理。数据库管理系统(DBMS)有四种主要类型:层次型、网络型、关系型和面向对象的关系型 DBMS。SQL 用于处理关系数据库管理系统或服务器。

这些服务器有 SQL Server、MySQL、SQLite 和 PostgreSQL 之类的名称，这可能会使 SQL 一词既指语言又指关系数据库的用法变得混乱。有几个数据库平台使用 SQL，但略有不同，每个平台的语法都略有不同([来源](https://www.upwork.com/hiring/data/sql-vs-mysql-which-relational-database-is-right-for-you/))我们还可以在 SQL 服务器中使用 SQL 语言的一种语法演变，称为 SQL Alchemy。

我们既有开源服务器，也有授权服务器。下面简单介绍几种类型的数据库管理系统:例如[微软 SQL Server vs MySQL](https://www.upwork.com/hiring/data/sql-vs-mysql-which-relational-database-is-right-for-you/)、 [PostgreSQL vs MYSQL](https://blog.panoply.io/postgresql-vs.-mysql) 和 [PostgreSQL vs MongoDB](https://blog.panoply.io/postgresql-vs-mongodb) 。

![](img/26ae591111bef7a49d2659dd000f621c.png)

PSQL 是一个很好的选择，因为它已经发展了 30 多年，符合 ACID——它代表原子性、一致性、隔离性、持久性，并意味着松散的错误保护，它已经成为许多组织的开源关系数据库。它用于构建应用程序并受信任以保护数据完整性([来源](https://www.postgresql.org/about/))。

到开始，我们将使用来自 Kaggle 数据集 [Shopify 应用商店](https://www.kaggle.com/usernam3/shopify-app-store)的 csv 文件。在工作场所，我们可能不会从 csv 数据文件开始，因为原则上，我们试图处理太大而不能保存在一个简单的 csv 文件中的数据。但是，因为我们是在学习，所以我们将使用一个不需要服务器查看，但可以查询的小数据集进行练习。

我们来看看 Shopify 应用市场的定价数据。加拿大电子商务网站 Shopify 与亚马逊争夺卖家。它通过他们的平台“为超过 60 万家企业提供动力，销售额超过 820 亿美元”。在某些情况下， [Shopify 击败亚马逊成为市场上最受欢迎的电子商务平台](https://www.websitebuilderexpert.com/ecommerce-website-builders/comparisons/shopify-vs-amazon/)。

我们可以看到该数据集中有五个表——应用、类别、计划功能、定价计划和评论。我们将首先创建一个实体关系图(ERD)来理解表中数据列之间的关系。这将使我们能够:

> 确定外键和主键
> 
> 确定每一列的数据类型——一个好的经验法则是，只有包含我们打算对其进行数学计算的数据的列才应该具有整数类型。
> 
> 了解表之间的关系类型:一对多、多对一等。

![](img/01361cfff079b315d8b5539ca632c2a2.png)![](img/661f1e1c4aaee08316f65315f8f68000.png)

Source ([https://www.lucidchart.com/pages/ER-diagram-symbols-and-meaning](https://www.lucidchart.com/pages/ER-diagram-symbols-and-meaning))

我设置了主键，因为它们是父记录。外键包含子记录。( [Source](https://stackoverflow.com/questions/1429407/determining-the-primary-and-foreign-table-in-a-relationship) )这可能真的很难计算出我们拥有的表越多。我们知道一个表可以“只有一个主键”，并且“为每个表添加一个主键是一个好的做法”。[“主键不接受空值或重复值。”](http://www.sqltutorial.org/sql-primary-key/)我们可能需要创建一个由两列或更多列组成的主键，这就是所谓的复合主键。这将在中进一步探讨[。](http://www.postgresqltutorial.com/postgresql-primary-key/)

为了从 csv 文件中获取数据，我们将从 bash 的命令行开始。我们可以在命令行中进入 PSQL 服务器并创建一个数据库。然后我们可以使用 **\connect** 导航到我们的数据库。

![](img/e8fbea8c0e76144c751704f959bc956a.png)![](img/3335a282bf22bacc694f44090c7ff1c3.png)

然后，我们在文本编辑器中创建一个文件，并将其保存为 SQL 文件。我选择使用 Jupyter 笔记本的文本编辑器，但是我们也可以使用 Visual Studio 代码。在这种情况下，我将调用我的文件 create_tables.sql。

```
/* Drop the apps table if it exists */
DROP TABLE IF EXISTS apps;/* Create skeleton table that supplies the table name, column names, and column types */
#set up our SQL tablesCREATE TABLE apps(
  url text PRIMARY KEY
, title text
, developer text
, developer_link text
, icon text
, rating text
, reviews_count integer
, description text
, description_raw text
, short_description text
, short_description_raw text
, key_benefits_raw text
, key_benefits text
, pricing_raw text
, pricing text
, pricing_hint text
;/* Copy data from CSV file to the table */
COPY apps 
FROM '/Users/sherzyang/flatiron/mod_3/week_7/my_practice/shopify-app-store/apps.csv' 
WITH (FORMAT csv);
```

现在，我将返回 bash，尝试在数据库中创建我的第一个表。我发现了错误“未加引号的回车”,我了解到这是 csv 文件的错误，而不是来自我们的 end gone 数组的命令。我了解到这是由于\n 出现在我的数据中。这种[可能已经发生](http://utf-8.icebolt.info/)，因为数据是在 windows 程序中下载的。

![](img/bde4593af225d0b56e43af7a21d8161d.png)

我用这个修复文件，很好，它的工作！

![](img/89e3e0736296434c1ad923864c3d40fb.png)

Source ([http://bit.ly/2JsArGY](http://bit.ly/2JsArGY))

通过编辑我的文本代码，并在 PSQL 中用 **\i create_tables.sql** 调用 bash 中的文件，我能够将其他 csv 文件移动到表中。对于其他命令，我使用了这个关于中级 PSQL 的[伟大教程。我对其他表使用了以下代码:](https://www.dataquest.io/blog/sql-intermediate/)

```
CREATE TABLE plan_features(
  pricing_plan_id text 
, app_url text 
, feature text
;CREATE TABLE categories(
  app_url text 
, category text 
;CREATE TABLE reviews(
  app_url text 
, url text 
, author text
, body text
, rating text
, helpful_count text
, posted_at text
;CREATE TABLE pricing_plans(
  id text PRIMARY KEY 
, app_url text 
, title text
, price text
, hint text
;
```

一旦我在 PSQL 的表中有了数据，我就可以移动到我的 Jupyter 笔记本并连接到我的 sqlalchemy 引擎。

```
import psycopg2
import numpy as np 
import pandas as pd
from sqlalchemy import create_engine#connect to the database and make a cursor
conn = psycopg2.connect("dbname=shopifydb_1")
cur = conn.cursor()#loop through our queries list and execute themfor query in queries_list:
    cur.execute(query)
    conn.commit()#set up a sqlalchemy engineengine = create_engine("postgresql:///shopifydb_1", echo=True)
```

现在我可以查询我的表了。我想看到的是评论和它们所属的应用程序一起出现在一个数据框架中。连接这些表的有趣之处在于，我们意识到一个应用程序和该应用程序的许多评论之间存在一对多的连接。通过这个查询，我们还可以练习创建别名，这为我们提供了表的全名的昵称，以及将列类型从一种类型转换为另一种类型的转换。

```
q = """
SELECT 
  r.rating
, r.body
, r.helpful_count
, p_p.price
, a.rating
, a.reviews_count
, a.key_benefits
, a.pricing
, a.url
, a.titleFROM apps AS a
JOIN pricing_plans p_p
    ON cast(a.url as text) = cast(p_p.app_url as text)
    AND a.title = p_p.title

JOIN reviews r
    ON a.url = r.app_url
"""
```

我们可以使用 SQL 查询一个非常大的数据库来分离出我们需要的数据。如果您犯了一个错误并引入了一个错误，只要您没有提交前面的操作，您总是可以使用下面的代码来回滚光标。

```
conn.rollback()
```

现在我们可以运行查询并以熊猫数据帧的形式读取结果。我们希望这样做，以便对我们正在检查的数据切片执行统计测试、转换、可视化等操作。(对于大量数据的计算，我们将需要使用，比方说， [Apache Spark](https://spark.apache.org/examples.html) ，而不是将数据插入到 pandas 数据帧中。)

```
#read our query as a DataFrame
df_join = pd.read_sql_query(q,con=engine)
```

![](img/be1b5e0e302cac6139e4b0cca10afb3d.png)

最后，如果需要，我们可以将查询保存回 csv 文件。

```
#save our selected query back into a CSV file
df_join.to_csv('data.csv')
```

这是如何使用 psql 及其与 sql 和 python 的 pandas 库的关系的快速浏览。希望这是有用的，并帮助您玩我间谍(比喻)与您的数据更容易一点。