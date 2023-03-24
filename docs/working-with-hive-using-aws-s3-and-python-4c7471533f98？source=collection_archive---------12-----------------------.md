# 使用 AWS S3 和 Python 处理 Hive

> 原文：<https://towardsdatascience.com/working-with-hive-using-aws-s3-and-python-4c7471533f98?source=collection_archive---------12----------------------->

## 使用外部数据存储维护配置单元模式和使用 Python 执行配置单元查询的初学者指南

![](img/6fe6838b4d3936364e9009b8c36bea4f.png)

Photo by [Aleksander Vlad](https://unsplash.com/@aleksow?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/computer?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

在本文中，我将分享我维护 Hive 模式的经验。这将对那些愿意涉足大数据技术的大一新生很有用。这将主要描述如何使用 python 连接到 Hive，以及如何使用 AWS S3 作为数据存储。如果你不熟悉 Hive 概念，请浏览维基百科上的文章。

本文的主要目的是提供一个通过 python 连接 Hive 和执行查询的指南。为此，我使用了“*py hive”*库。我将我的连接类创建为“*Hive connection”*，Hive 查询将被传递到函数中。AWS S3 将用作配置单元表的文件存储。

```
**import** pandas **as** pd **from** pyhive **import** hive**class** HiveConnection:
    @staticmethod
    **def** select_query(query_str: str, database:str =HIVE_SCHEMA) -> pd.DataFrame:
        *"""
        Execute a select query which returns a result set* **:param** *query_str: select query to be executed* **:param** *database: Hive Schema* **:return***:
        """*
        conn = hive.Connection(host=HIVE_URL, port=HIVE_PORT, database=database, username=HIVE_USER)

        **try**:
            result = pd.read_sql(query_str, conn)
            **return** result
        **finally**:
            conn.close()

    @staticmethod
    **def** execute_query(query_str: str, database: str=HIVE_SCHEMA):
        *"""
        execute an query which does not return a result set.
        ex: INSERT, CREATE, DROP, ALTER TABLE* **:param** *query_str: Hive query to be executed* **:param** *database: Hive Schema* **:return***:
        """*
        conn = hive.Connection(host=HIVE_URL, port=HIVE_PORT, database=database, username=HIVE_USER)
        cur = conn.cursor()
        *# Make sure to set the staging default to HDFS to avoid some potential S3 related errors* cur.execute(**"SET hive.exec.stagingdir=/tmp/hive/"**)
        cur.execute(**"SET hive.exec.scratchdir=/tmp/hive/"**)
        **try**:
            cur.execute(query_str)
            **return "SUCCESS"
        finally**:
            conn.close()
```

我将查询保存为单独的字符串。这样，您可以在必要时用外部参数格式化查询。配置单元配置(配置单元 URL、配置单元端口、配置单元用户、配置单元模式)为常量。函数" *select_query"* 将用于检索数据，函数" *execute_query* 将用于其他查询。

Hive 提供了一个 shell 交互工具来初始化数据库、表和操作表中的数据。我们可以通过输入命令“*Hive”*进入 Hive 命令行。您也可以在 shell 中执行本文中给出的所有查询。

# 创建新模式

模式是类似于数据库的表的集合。Hive 中允许使用关键字 SCHEMA 和 DATABASE。我们可以选择任何一个。这里我们用模式代替数据库。可以使用“创建模式”来创建模式。要进入模式内部，可以使用关键字“USE”。

```
**CREATE SCHEMA** userdb;
**USE** userdb;
```

# 创建表格

有三种类型的配置单元表。它们是内部的、外部的和暂时的。内部表存储数据库中表的元数据以及表数据。但是外部表将元数据存储在数据库中，而表数据存储在远程位置，如 AWS S3 和 hdfs。删除内部表时，所有表数据都将随元数据一起被删除。删除外部表时，只会删除元数据。而不是表数据。这样，实际数据将得到保护。如果您将新表指向相同的位置，数据将通过新表可见。

Hive 是一个数据仓库，使用 MapReduce 框架。因此，数据检索的速度对于小型查询来说可能不够公平。为了提高性能，可以对配置单元表进行分区。分区技术可以应用于外部表和内部表。像 bucketing 这样的概念也是有的。您可以选择这些技术中的任何一种来提高性能。

将数据从一个地方复制到另一个地方时，临时表非常有用。它充当数据库会话中保存数据的临时位置。会话超时后，所有临时表都将被清除。创建临时表对" *Pyhive"* 库没有用，因为单个会话中不支持多个查询。即使我们创建了一个表，也不能再使用同一个会话来访问表。但是这在 Hive 命令行中是可能的。您可以创建一个临时表，然后在单个会话中从该表中选择数据。

## 内部表格

以下查询将创建一个具有远程数据存储 AWS S3 的内部表。文件格式为 CSV，字段以逗号结尾。“ *s3_location* 指向数据文件所在的 s3 目录。这是用户为查询字符串定义的外部参数。应该在查询格式化的时候传递。

```
CREATE TABLE `user_info` (
`business_unit` INT,
`employee_id` INT,
`email` VARCHAR(250),
`first_name` VARCHAR(250),
`last_name` VARCHAR(250),
`gender` VARCHAR(250),
`birthday` DATE,
`created_date` DATE,
`updated_date` DATE)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' ESCAPED BY '\\'
LOCATION '{s3_location}'
TBLPROPERTIES (
"s3select.format" = "csv",
"skip.header.line.count" = "1"
);
```

如果数据字符串包含逗号，将会破坏表格结构。所以我定义了一个转义符，在创建表之前，所有不必要的逗号都需要放在这个转义符的前面。

下面是一个示例记录。请注意，电子邮件包含一个逗号。

1，1，**安，史密斯@加米尔**。com，Ann，Smith，女，' 1992–07–01 '，' 2019–09–01 '，' 2019–12–31**'**

上面的记录需要这样格式化 **:** 1，1，**ann\\,smith@gamil.com**，安，史密斯，女，“1992–07–01”，“2019–09–01”，“2019–12–31”

## 外部表格

这里，我用“*业务单位”*和“*创建日期”*对“*用户信息”*表进行了分区

```
CREATE EXTERNAL TABLE `user_info` (
 `employee_id` INT,
 `email` VARCHAR(250), 
 `first_name` VARCHAR(250),
 `last_name` VARCHAR(250),
 `gender` VARCHAR(250),
 `birthday` DATE,
 `updated_date` DATE
)  partitioned by(
`business_unit` INT, 
`created_date` DATE,
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' ESCAPED BY '\\'
STORED AS
INPUTFORMAT
'com.amazonaws.emr.s3select.hive.S3SelectableTextInputFormat'
OUTPUTFORMAT
'org.apache.hadoop.hive.ql.io.HiveIgnoreKeyTextOutputFormat'
LOCATION '{s3_location}' 
TBLPROPERTIES (
"s3select.format" = "csv",
"s3select.headerInfo" = "ignore"
);
```

## 工作单元表

创建临时表的查询。

```
CREATE TEMPORARY TABLE `user_info` (
 `business_unit` INT,
 `employee_id` VARCHAR(250),
 `email` VARCHAR(250),
 `first_name` VARCHAR(250),
 `last_name` VARCHAR(250),
 `gender` VARCHAR(250),
 `birthday` DATE,
 `created_date` DATE,
 `updated_date` DATE
) ;
```

## 翻桌

删除表的查询。如果删除外部表，远程文件存储中的数据不会被删除。

```
DROP TABLE IF EXISTS `user_info`;
```

## 插入数据

一旦使用外部文件存储创建了表，远程位置的数据将通过没有分区的表可见。但是当涉及到带有分区的表时，情况就不一样了。这意味着数据不能直接复制到分区表中。我们需要创建一个没有分区的临时表，并通过提供分区值将数据插入到分区表中。以下查询描述了如何向这样的表中插入记录。

```
INSERT INTO TABLE user_static_info PARTITION (business_unit={business_unit}, `created_date`='{execution_date}')
SELECT
   Employee_id,
   email,
   secondary_email,
   first_name,
   last_name,
   orig_gender,
   gender,
   signup_channel ,
   signup_from_fb ,
   birthday,
   signup_date,
   updated_date,
   last_activity_date,
   subscription_status
FROM
   tmp_user_static_info
WHERE business_id={business_unit}
```

因为单个会话中的多个查询不支持“*py hive*”；我必须创建内部表“*tmp _ user _ static _ info”*，它指向没有分区的 S3 数据目录。然后，在将数据插入外部分区表后，删除了该表。

## 检索数据

选择查询用于检索配置单元中的数据。这些非常类似于 SQL 选择查询。它具有以下形式。您可以根据自己的需求构建查询。

```
SELECT [ALL | DISTINCT] select_expr, select_expr, …
FROM table_reference
[WHERE where_condition]
[GROUP BY col_list]
[HAVING having_condition]
[CLUSTER BY col_list | [DISTRIBUTE BY col_list] [SORT BY col_list]]
[LIMIT number];
```

## 更新和删除数据

配置单元不支持直接更新和删除数据。如果您想更改表格中的任何内容；使用 SELECT 查询将必要的数据复制到新表中。然后，通过删除旧表并重命名新表，可以用新表替换旧表。

## 更改表格

在 Hive 中可以更改表格。但这需要在不影响现有数据的情况下非常小心地完成。因为我们不能改变数据。例如，在中间添加一个新字段不会移动数据。如果我们添加一个新字段作为第二个字段，属于第三列的数据仍将出现在第二列，第四个字段的数据出现在第三个字段，依此类推。最后一个字段将不包含任何数据。这是因为更新配置单元表数据的限制。如果我们添加一个新字段作为最后一个字段，将会有一个空字段，我们可以将数据插入到该字段中。

```
ALTER TABLE user_static_info ADD COLUMNS (last_sign_in DATE);
```

如果我们想删除外部数据，我们可以使用以下步骤。

```
ALTER TABLE user_static_info SET TBLPROPERTIES('EXTERNAL'='False');
DROP TABLE user_static_info;
```

# 例子

最后，以下代码显示了如何使用“ *HiveConnection* ”类中的“ *execute_query* ”函数执行查询。

```
from src.database.hive_con import HiveConnection

create_temp_table_query_str = """CREATE TABLE `user_info` (
`business_unit` INT,
`employee_id` INT,
`email` VARCHAR(250),
`first_name` VARCHAR(250),
`last_name` VARCHAR(250),
`gender` VARCHAR(250),
`birthday` DATE,
`created_date` DATE,
`updated_date` DATE
)
ROW FORMAT DELIMITED FIELDS TERMINATED BY ',' ESCAPED BY '\\'
LOCATION '{s3_location}'
TBLPROPERTIES (
"s3select.format" = "csv",
"skip.header.line.count" = "1"
);""".format(
    s3_location="s3://hive-test/data/user_info/"
)

HiveConnection.execute_query(query_str=create_temp_table_query_str, database=userdb)
```