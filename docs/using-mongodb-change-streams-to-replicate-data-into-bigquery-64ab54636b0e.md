# 使用 MongoDB 更改流将数据复制到 BigQuery 中

> 原文：<https://towardsdatascience.com/using-mongodb-change-streams-to-replicate-data-into-bigquery-64ab54636b0e?source=collection_archive---------13----------------------->

## 我们在使用 MongoDB 变更流构建 MongoDB 到 BigQuery 数据管道时所获得的经验和面临的挑战

![](img/56688bea835fbf60509d1c7866873326.png)

Photo by [Quinten de Graaf](https://unsplash.com/@quinten149?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

在进入技术细节之前，最好回顾一下我们为什么决定建立这个管道。我们开发它有两个主要原因:

1.  在一定规模下，查询 MongoDB 进行分析是没有效率的。
2.  我们没有 MongoDB 中的所有数据(例如条带计费信息)。
3.  数据管道即服务供应商在一定规模下非常昂贵。并且，通常不提供复制删除记录的方法，如*软删除*(例如，使用`deleted_at`字段)。

## 复制无模式数据

在使用这个 MongoDB 数据库时，我们注意到的第一件事是一些集合有一个棘手的*模式*。文档内部有嵌套文档，其中一些也是数组。

通常，一个嵌套的文档代表一个一对一的关系，一个数组代表一对多的关系。幸运的是，Big Query 同时支持[重复字段和](https://cloud.google.com/bigquery/docs/nested-repeated)嵌套字段。

根据我们的研究，复制 MongoDB 数据的最常见方式是在集合中使用时间戳字段。该字段通常被命名为`updated_at`，并在每次记录被*插入*或*更新*时被更新。这种方法很容易用批处理方法实现，它只需要查询所需的集合。当将它应用于我们的数据和集合时，我们发现了两个主要问题:

1.  并非所有我们想要复制的集合都有这个字段。**没有** `**updated_at**` **，我们怎么知道哪些记录被更新复制了呢？**
2.  此方法不跟踪已删除的记录。我们只是将它们从原始集合中删除，并且永远不会在我们的大查询表中更新。

幸运的是，MongoDB 在`oplog`中保存了应用于集合的所有更改的日志。从 MongoDB 3.6 开始，您可以使用[变更流](https://docs.mongodb.com/manual/changeStreams/) API 来查询它们。这样，集合中的每个变化(包括`delete`操作)都会提醒我们。

然后，我们的目标是构建一个管道，将 MongoDD Change Streams 返回的所有变更事件记录移动到一个大的查询表中，其中包含每个记录的最新状态。

# 建设管道

![](img/b763612a2f1bc6cea5f267f8b75d6bb3.png)

Photo by [NeONBRAND](https://unsplash.com/@neonbrand?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

我们的第一种方法是在 Big Query 中为我们想要复制的每个集合创建一个变更流表，并从该集合的所有变更流事件中推断出模式。这被证明是相当棘手的。如果在记录中添加了一个新字段，管道应该足够智能，在插入记录之前修改大的查询表。

因为我们希望在大查询中尽快获得数据，所以我们采用了另一种方法。将所有变更流事件作为 JSON blob 转储到 BigQuery 中。然后，我们可以使用像 [dbt](https://www.getdbt.com/) 这样的工具来提取、转换原始的 JSON 数据，并将其转换成合适的 SQL 表。当然，这有一些缺点，但让我们很快就有了端到端的管道。

管道具有以下组件:

1.  Kubernetes ( `[carden](https://github.com/bufferapp/carden)`)中运行的一个服务，它读取每个集合的 MongoDB 变更流，并将其推送到一个简单的大查询表中(追加所有记录)。
2.  一个 dbt cronjob，它使用原始数据[增量读取源表，并将一个查询具体化为一个新表](https://docs.getdbt.com/v0.12/docs/materializations#section-incremental)。此表包含自上次运行以来更改的每行的最新状态。这是 dbt SQL 在生产环境中的一个示例。

有了这两个步骤，我们就有了从 MongoDB 实时流向 Big Query 的数据**。我们还跟踪*删除*，并且我们拥有我们正在复制的集合中发生的所有更改(对于需要一段时间内的更改信息的某种分析很有用)。**

由于我们在启动 MongoDB 更改流爬行服务之前没有任何数据，因此我们丢失了许多记录。为了解决这个问题，我们决定回填创建假的变化事件。我们转储了 MongoDB 集合，并制作了一个简单的脚本，将文档包装成插入。这些记录被发送到同一个 BigQuery 表中。现在，运行同一个 dbt 模型给我们最终的表，其中包含所有回填的记录。

我们发现的主要缺点是，我们需要用 SQL 编写所有的提取。这意味着大量额外的 SQL 代码和一些额外的处理。目前，使用 dbt 并不太难。另一个小问题是 BigQuery [本身不支持提取 JSON](https://stackoverflow.com/questions/52120182/bigquery-json-extract-all-elements-from-an-array) 中编码的数组的所有元素。

# 结论

对我们来说，好处(迭代时间、变更的容易程度、简单的管道)大于坏处。因为我们刚刚开始使用这个管道，所以让一切端到端地工作并快速迭代是非常有用的！让 BigQuery 只附加更改流表作为一个分离来服务我们。在未来，我们计划迁移到 Apache Beam 和 Cloud 数据流，但那是另一篇文章！

希望你对这些见解感兴趣！你可以在推特上找到我，账号是@davidgasquez 。如果你有任何问题，不要犹豫，尽管来找我。