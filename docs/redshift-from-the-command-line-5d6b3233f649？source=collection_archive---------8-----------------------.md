# 从命令行红移

> 原文：<https://towardsdatascience.com/redshift-from-the-command-line-5d6b3233f649?source=collection_archive---------8----------------------->

![](img/624d14a69dd3915e09c0e5d606db08a6.png)

我最近发现自己在 AWS Redshift 控制台中编写和引用保存的查询，并知道一定有更简单的方法来跟踪我的常用 sql 语句(我主要用于定制复制作业或检查日志，因为我们对所有 BI 使用[模式](https://modeanalytics.com/))。

原来有一种更简单的方法，叫做 psql (Postgres 的基于终端的交互工具)！开始使用 psql 非常容易，您会对 AWS 控制台点击它的次数感到高兴。

## 步骤 1:安装

从[安装 postgres 开始。](https://www.postgresql.org/download/)

```
$ brew install postgres
```

## 步骤 2:建立红移连接

接下来，[连接到你的红移星团](https://docs.aws.amazon.com/redshift/latest/mgmt/connecting-from-psql.html)。

由于您将一直使用 psql，我建议在您的`~/.bash_profile`中创建一个别名，这样您就可以轻松地用一个单词建立数据库连接。我用`redshift`作为我的别名。您可以使用您在中定义的凭据。bash_profile:

```
alias redshift='psql "host=$REDSHIFT_HOST user=$REDSHIFT_USER dbname=$REDSHIFT_DB port=$REDSHIFT_PORT sslmode=verify-ca     sslrootcert=<path to your postgres root .crt file> password=$REDSHIFT_PASSWORD"'
```

## 步骤 3:添加您的开发 IP 地址

您需要确保您的集群安全组被设置为接收来自您将在开发中使用的任何 IP 地址的入站流量。如果您的集群在自定义 VPC 中，您可以从命令行使用 [CLI](https://docs.aws.amazon.com/cli/latest/reference/ec2/authorize-security-group-ingress.html) 的`authorize-security-group-ingress`来完成此操作。否则，如果您使用默认 VPC，您可以在控制台中手动将您的 IP 地址添加到安全组的入站规则中

## 第四步:探索你的仓库

现在您已经连接好了，在命令行上键入`redshift`，并尝试这些方便的命令:

**\dt —** 查看您的表
**\df —** 查看您的函数
**\dg —** 列出数据库角色
**\dn —** 列出模式
**\dy —** 列出事件触发器
**\dp —** 显示表、视图和序列的访问权限

## 步骤 5:进行一次查询

```
$ SELECT * FROM your_schema.your_table LIMIT 10;
```

## 步骤 6:运行一个简单的事务

打开您最喜欢的文本编辑器，编写一个简单的事务(一系列 sql 语句，作为一个整体运行，如果其中任何一个失败，什么也不做)，例如:

```
BEGIN; 
INSERT INTO your_table VALUES (1), (2), (3), (4);
INSERT INTO your_table VALUES (5), (6), (7), (8);
INSERT INTO your_table VALUES (9), (10), (11), (12);
COMMIT;
```

保存这个事务，并使用`\i`命令在 psql repl 中运行:

```
dev=# \i your_transaction.sql
```

玩得开心，让我知道你发现了什么！

Siobhán 是在[着陆](http://www.landed.com)的数据工程负责人。Landed 的使命是帮助重要的专业人士在他们服务的社区附近建立金融安全，当他们准备在昂贵的市场购买房屋时，我们会与教育工作者一起投资。听起来很棒？[加入我们](https://landed.com/jobs)！