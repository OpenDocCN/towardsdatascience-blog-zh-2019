# 在基于 Debian 的 Linux 中设置 PostgreSQL

> 原文：<https://towardsdatascience.com/setting-up-postgresql-in-debian-based-linux-e4985b0b766f?source=collection_archive---------26----------------------->

![](img/293c4f36fbc8d384f5ca1b81d4e7d076.png)

Linux Mint, PostgreSQL, Ubuntu

在尝试在我的 Linux 机器上安装 PostgreSQL 服务器来练习 SQL 命令的过程中，我遇到了一些困难，所以我决定为我未来的自己编写这个教程，以防我需要再次安装。在安装了基本系统之后，我还能够从一个小的测试数据库中获得一个转储，以供练习之用…所以我还包含了我用来加载这个数据库文件的命令。

注:这个过程是在 2019 年 11 月左右在 Linux Mint 19 Tara 和 Ubuntu 18.04 上测试的。

下面的许多步骤都是从这篇不错的 Wixel 帖子中获得的:[https://wixelhq . com/blog/how-to-install-PostgreSQL-on-Ubuntu-remote-access](https://wixelhq.com/blog/how-to-install-postgresql-on-ubuntu-remote-access)

**准备和安装**

Ubuntu 和 Mint(以及其他)的默认库有 Postgres 包，所以我们可以使用 apt 来安装。

```
$ sudo apt update$ sudo apt install postgresql postgresql-contrib
```

**角色设置**

postgres 的安装会创建一个用户 postgres，该用户可以访问默认的 Postgres 数据库。通过首先切换到 postgres 用户并运行`psql`来访问它。

```
$ sudo -i -u postgrespostgres@server:~$ psql
```

您现在可以直接访问默认的 postgres 数据库。注意“`\q`”允许您退出 Postgres 提示符。

```
postgres=# \q
```

现在您已经确认了 Postgres 的功能，是时候设置角色了。仍然在 postgres 帐户中，执行以下命令。

```
postgres@server:~$ createuser --interactive
```

这将引导您完成设置新用户的过程。关于其他标志的更多细节可以在`createuser`手册页中找到。

注意:首先创建一个与您的用户帐户同名的角色作为超级用户可能会很有用。

现在运行`psql`(作为与您刚刚创建的角色同名的用户名)将导致一个错误。虽然您已经创建了一个角色，但是每个角色的默认数据库都是与该角色同名的数据库。因此，有必要提供您希望连接的数据库的名称，或者创建一个相应的数据库，这将在下面的部分中讨论。

**访问特定数据库**

假设您的角色(您的用户名)已经被授权访问一个数据库`{dbname}`，您可以简单地从您的命令提示符下运行以下命令。

```
$ psql {dbname}
```

您也可以指定一个角色`{role}`而不是您的用户名，如下所示。

```
$ psql -U {role} {dbname}
```

**创建新数据库**

作为 postgres 用户，执行以下命令。注意:将`{dbname}`替换为所需数据库的名称，通常是上面创建的角色的名称。

```
postgres@server:~$ createdb {dbname}
```

**加载数据库**

如果您得到了一个数据库的转储，这里用`{filename}.sql`表示，下面将在 PostgreSQL 服务器中创建数据库，并将文件数据加载到数据库中。注意:将`{username}`替换为 PostgreSQL 服务器中现有的角色，将`{dbname}`替换为 PostgreSQL 服务器中现有数据库的名称，将`{filename}`替换为。sql 文件。

```
$ createdb -U {username} {dbname}$ psql {dbname} < {filename}.sql 
```

现在你已经准备好`SELECT something FROM your_tables`了。

PSQLing 快乐！