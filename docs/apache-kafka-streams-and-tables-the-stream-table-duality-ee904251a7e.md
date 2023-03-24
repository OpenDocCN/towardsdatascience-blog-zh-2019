# 阿帕奇卡夫卡流和表，流表二元性

> 原文：<https://towardsdatascience.com/apache-kafka-streams-and-tables-the-stream-table-duality-ee904251a7e?source=collection_archive---------13----------------------->

最初发表于[我的个人博客](https://blog.contactsunny.com/tech/apache-kafka-streams-and-tables-the-stream-table-duality)。

在之前的帖子中，我们试图理解阿帕奇的卡夫卡流的[基础。在本帖中，我们将基于这些知识，看看 Kafka 流如何既可以用作流，也可以用作表。](https://blog.contactsunny.com/tech/getting-started-with-apache-kafka-streams)

![](img/379d85e54bc74f53cfabab22ae7ae6f5.png)

当今，流处理在大多数现代应用程序中已经变得非常普遍。您将至少有一个流进入您的系统进行处理。根据您的应用程序，它通常是无状态的。但并非所有应用都是如此。我们将在数据流之间进行某种数据浓缩。

假设您有一个用户活动流进来。理想情况下，您会将一个用户 ID 附加到该流中的每个事实上。但是在管道中，用户 ID 不足以进行处理。也许您需要更多关于用户的信息才能出现在事实中。为此，您将查询数据库，获取用户记录，并将所需的数据添加到事实中。这个丰富的事实将被发送到另一个流进行进一步处理。

可以想象，流与数据库密切相关，至少在大多数实际应用程序中是如此。这也是为什么 Apache 在 Kafka 流中引入了 KTables 的概念。这使得流表二元性成为可能。

# 流表对偶

在阿帕奇卡夫卡中，流和表一起工作。流可以是表，表可以是流。这是卡夫卡作品的一个特性，我们可以用它来获得这种多样性。让我们看看我的意思。

*   **表** —一个表可以被看作是一个流的变更日志的集合。这就是说，一个表在给定的时间点将具有特定事实的最新值。例如，如果我们在一个电子商务应用程序中维护购物车上每个事件的流，那么一个表将具有购物车的最新状态。如果我们回放 changelog，我们应该能够创建一个实际的表。
*   **Stream** —类似地，一个表可以被视为一个流，特定字段的最新值进入其中。它只是在给定时间点流中某个键的最新值的快照。

现在让我们看看 KStream 和 KTable。

# KStream

KStream 也不过如此，一个卡夫卡式的流。这是一个永无止境的数据流。每条数据——一条记录或一个事实——都是键值对的集合。还要注意的是，进入流的每个事实本质上都是不可变的。在将一个事实发送到流中之后，可以更改任何值的唯一方法是在更新值之后发送另一个事实。

# KTable

KTable 只是流的抽象，其中只保存最新的值。例如，假设我们将以下事实推入表中:

```
{
  "name": "Sunny Srinidhi",
  "city": "Mysore",
  "country": "India"
}
```

在这个事实出现一天后，我搬到了一个新的城市，这个变化必须被系统捕捉到。因此，我将另一个事实与以下数据一起发送到流中:

```
{
  "name": "Sunny Srinidhi",
  "city": "Bangalore",
  "country": "India"
}
```

现在，KTable 中以前的事实将被更新以反映新的值，而不是将其视为一条新的信息。

正如您所看到的，KTable 主要作为数据库中的传统表工作。唯一的区别是，KTable 中的每个条目都被认为是一个 UPSERT(插入或更新)。这意味着，如果 KTable 中有旧版本的数据，它将被更新为最新的值。但是如果没有旧版本，事实将被插入到 KTable 中。

这里需要注意的一点是，KTables 对于值 *null* 有特殊的含义。如果您向一个值为 *null* 的 KTable 发送一个键-值对，它将被认为是一个删除指令，这个事实将从 KTable 中删除。你必须确保不会意外地将程序中的任何 *null* 值发送到 KTable 中，否则你可能会丢失已经存储的数据。

# 全球表格

GlobalKTable 是对已经抽象的 KTable 的抽象。当我们处理分布式应用程序时，这很方便。为了更好的理解，我们举个例子。假设我们有一个正在填充 KTable 的应用程序。由于流量激增，我们在集群中部署了更多的应用程序实例，比如 10 个实例。并且每个实例都从底层 Kafka 主题的单独分区中读取数据。

现在，每个实例都有自己的 KTable 副本。这个本地 KTable 将只填充来自分配给该应用程序实例的特定分区的数据。所以没有一个本地 KTables 拥有所有需要的数据。如果您在表上运行连接或聚合，您的结果会有偏差，因为数据不完整。

在这种情况下，如果使用 GlobalKTable，而不是使用 KTable，该表将填充所有分区的数据。所以在你的 GlobalKTable 的所有本地实例中，你有一个更加完整的数据集。除此之外，GlobalKTables 还有一些聚合和连接好处。

在以后的文章中，我们将看到一些代码示例，其中我们使用 Kafka Streams APIs 来实际实现我们刚刚试图理解的所有概念。

> 在 [Twitter](https://twitter.com/contactsunny) 上关注我，了解更多[数据科学](https://blog.contactsunny.com/tag/data-science)、[机器学习](https://blog.contactsunny.com/tag/machine-learning)，以及通用[技术更新](https://blog.contactsunny.com/category/tech)。此外，你可以[关注我的个人博客](https://blog.contactsunny.com/)，因为我在 Medium 之前发布了许多我的教程、操作方法帖子和机器学习的优点。

如果你喜欢我在 Medium 或我的个人博客上的帖子，并希望我继续做这项工作，请考虑在 Patreon 上支持我。