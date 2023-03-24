# 将 MaxMind GeoIP 与 Redshift 结合使用的简单方法

> 原文：<https://towardsdatascience.com/the-easy-way-to-use-maxmind-geoip-with-redshift-65cf979e63b1?source=collection_archive---------5----------------------->

![](img/ee291c39e2d52589c9d9a8212589e4da.png)

Photo by [Westley Ferguson](https://unsplash.com/@westleyaaron_sink?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

它总是从一个无辜的观察开始。*“我们有很多来自波斯顿的流量，”*你的老板说。你自然会抛出一两个猜测，并讨论为什么会这样。直到你的老板扔下炸弹—

> “你能深究一下吗？”

该死的。你直接走进了那个房间。

现在你陷入了困境。你知道谷歌分析有地理位置的流量，但这不会削减它。如果您想按地区报告这些保留率、生命周期值或重复行为，您需要可以用 SQL 查询的东西，即存储在您的数据仓库中的东西。但是你没有那样的东西。你知道在你的日志数据中有用户 IP 地址，你只需要把它们转换成位置。但是红移没有办法做到这一点。

你需要的是[地理定位](https://en.wikipedia.org/wiki/Geolocation_software)使用 IPs，又名 GeoIP。人们通常从 [MaxMind](https://www.maxmind.com) 开始，主要是因为这是“GeoIP”的第一个谷歌搜索结果。我们将一起使用他们的 IP-to-City 数据集来丰富我们的日志数据，并确定我们的用户来自哪个城市和国家。我们将使用 MaxMind 数据，因为它可靠且稳健。而且是免费的。少了一件麻烦你老板的事。

所以我们去下载 MaxMind 的 GeoLite2 城市数据。打开 zip 文件后，我们发现了许多 CSV 文件，其中最重要的是`GeoLite2-City-Blocks-IPv4.csv`。如果我们往里面看，这是我们看到的:

我们马上注意到一个问题——这个数据看起来像一个 IP，但在末尾有一个斜杠和一个额外的数字。这是一个用 [*CIDR*](https://en.wikipedia.org/wiki/Classless_Inter-Domain_Routing) 符号表示的 *IP 网络*，它表示一系列 IP。它由一个 IP、一条斜线和斜线后的一个数字组成，称为*子网掩码*。这就像你可以说“西 23 街 500 号街区”来描述纽约市的一条街道一样。

如果我们有网络`1.2.3.0/24`,那就意味着“每个 IP 都以`1.2.3.`开头，结尾是 0 到 255 之间的任何数字。换句话说，`1.2.3.0`和`1.2.3.255`之间的任何 IP。因此，如果我们观察到一个 IP 为`1.2.3.95`的用户，它将属于网络`1.2.3.0/24`，因此位于`6252001`的`geoname_id`。子网掩码可以是 1 到 32 之间的任意数字，数字越小，网络越宽。

如果这个 MaxMind 表是红移的，我们怎么加入它？红移不包括任何方便的 [*网络地址类型*](https://www.postgresql.org/docs/9.3/datatype-net-types.html) 喜欢现代 Postgres 或者 [*INET 函数*](https://dev.mysql.com/doc/refman/8.0/en/miscellaneous-functions.html#function_inet-aton) 喜欢 MySQL。相反，我们将使用 IPs 背后的[数学知识来自己完成这项工作。](https://en.wikipedia.org/wiki/IPv4#Addressing)

你可以把一个 IP 看作是一个非常大的数字的一种奇特的表现。IP `1.2.3.4`其实只是引擎盖下的`16,909,060`。类似地，IP 网络只是一系列非常大的数字。网络`1.2.3.0/24`是从`16,909,056`开始到`16,909,311`结束的范围。我们将利用这一点。为此，我们需要一种将 IP 和 IP 网络转换为数字的方法。

使用 MaxMind 提供的 [geoip2-csv-converter](https://github.com/maxmind/geoip2-csv-converter) 工具，我们将把每个网络的整数范围表示添加到 csv 中。

*注意，我用的是* `20190108` *版本。MaxMind 每周更新此数据集，因此您的版本可能会有所不同。*

上传我们修改后的 CSV 到 S3，我们可以`COPY`它变成红移。

现在让我们写一个函数把 IPs 转换成真正的大数字。这里有一个用 SQL 写的简单的。我们将根据做同样事情的 Linux 实用程序把它叫做`inet_aton`。“inet”代表“互联网”，“aton”表示“ **A** 地址**到**t**N**号”。Linux 的人喜欢让事情简单明了。

我们需要做的最后一件事是加载 MaxMind CSV，它包含从`geoname_id`到地球上实际位置的查找。为了加快速度，我们将 gzip 它，上传`GeoLite2-City-Locations-en.csv.gz`文件到 S3，`COPY`它到一个表。

红移优化的一些快速指针。对于像这样的小型、常见的连接维度表，我推荐使用`DISTSTYLE ALL`。这将在集群中的每个节点上创建一个表的副本，从而消除了连接过程中的数据传输步骤。为了加快速度，我还将我们的连接列定义为一个`SORTKEY`。

现在，我们拥有了使用 GeoIP 使用位置数据丰富日志所需的一切。几乎一切。

## 红移很难

如果我们将新的 MaxMind GeoIP 表加入我们的日志数据，我们将立即遇到问题。假设我有一些基本的访问日志，并尝试按流量计算前 50 个地区。

如果您运行了这个查询，您将会有一个**糟糕的时间**。你的查询将运行几分钟，你会开始出汗。与此同时，你的红移管理员将会寻找那个破坏她的星团的人。不要成为那样的人。

这个查询有什么问题？快速浏览一下`EXPLAIN`计划，红移执行查询所采取的步骤列表，就知道了一切。

假设你经营一家冰激凌店，有数百万种美味的口味。如果排队的每个顾客都必须品尝每一种口味，然后才能选择一种，那会怎么样？如果我们试图使用我们的`BETWEEN` join (taste-test)直接将我们的日志数据(客户)与 MaxMind 数据(口味)结合起来，就会发生这种情况。这导致了一个 [*嵌套循环连接*](https://docs.aws.amazon.com/redshift/latest/dg/query-performance-improvement-opportunities.html#nested-loop)*这是让数据库哭泣的最快方法之一。*

*为了加快冰淇淋店的速度，我们将把它分成不同的区域——巧克力在这里，香草在那里，薄荷口味放在一个特别的地方。通过这种方式，顾客会前往符合他们偏好的区域。一旦到了那里，与之前相比，他们会品尝少量的味道。*

# *创建优化的 GeoIP 查找表*

*我们将在 Redshift 中构建一个新表，它将取代针对 GeoIP 查找而优化的`maxmind_ipv4_to_geo`。我们将使用 IPs 的前半部分将其组织成不同的部分，并将每个网络放入相应的部分。一些网络足够宽，以至于它们可以进入多个部分。这些部分的作用几乎就像一个数据库索引，允许 Redshift 缩小每个 IP 要检查的网络。使用一点小魔法，我们把我们的表转换成一个快速优化的表。*

*创建了查找表后，我们可以再进行一次分析。在我们的查询中，我们用`maxmind_ipv4_to_geo`替换了`maxmind_ipv4_lookup`，并添加了一个新的连接条件。我们将使用正则表达式`REGEXP_SUBSTR(log.ip_address, '\\d+\.\\d+')`提取每个 IP 的前半部分，并将其匹配到表`mm_geo.first_16_bits`中相应的部分。然后我们使用 IP 和网络的整数表示来检查它属于哪个网络。通过这种优化，我们的查询很快返回，看不到嵌套循环连接！*

*这样，你就可以开始分析了。您可以使用此查找表来连接到任何其他具有 IP 的红移。只要记住总是包括到`first_16_bits`的连接，因为这是方法背后的魔力。*

*这种方法的灵感来自于我读过的一篇关于在网络设备中优化 IP 查找的论文。不幸的是，我再也找不到报纸了。这篇文章是将该解决方案移植到红移并迭代以简化它的结果。*

*感谢 Slack 激发了这篇文章，感谢 Julian Ganguli 编写了一些早期代码，感谢 Nick James 阅读了早期草稿。*