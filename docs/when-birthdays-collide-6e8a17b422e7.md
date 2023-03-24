# 当生日冲突时

> 原文：<https://towardsdatascience.com/when-birthdays-collide-6e8a17b422e7?source=collection_archive---------24----------------------->

![](img/7c2e26bd4659dbedd0c41e6a886f146f.png)

Happy birthday! Photo by [Marina Utrabo](https://www.pexels.com/@marina-utrabo-390305) on [Pexels](https://www.pexels.com/photo/person-lighting-the-sparklers-1729797/)

想象一个挤满了人的房间——准确地说是 23 个人。也许你在一个高中班级，也许你正在参加一个低调的社交活动。这两个人同一天生日的几率有多大？(为了简单起见，假装闰年不存在，所以没有人在 2 月 29 日出生。)

你的第一个假设可能是这些人的生日在一年中均匀分布。或许你可以推断出，平均来说，一年的 12 个月中每个月都有两个人出生(除了一个月)，所以对于每个月来说，这两个人中的任何一个在某一天出生的概率大约是 1/30。因此，这 23 人中有两人在同一天出生的可能性一定非常非常低。对吗？

**错了！**在 23 人的小组中，有 [50%的几率两人同一天生日](http://www.efgh.com/math/birthday.htm)。当人数增加到 80 人时，赔率跃升到惊人的 [99.98%](http://shanky.org/2018/11/08/understanding-the-birthday-paradox/) ！

如果这看起来令人困惑，那只是因为它确实如此。生日悖论感觉非常违反直觉，直到你看到背后的逻辑。我们就这么做吧！

为了更好的理解这个问题，我们先从数学上来分解一下。

对于任何两个随机选择的人来说，他们有 1/365 的机会在同一天出生(假设他们不是在闰年出生)。因此，这两个人在不同的日子出生的概率是 364/365。

为了找出一个组中所有个体都有唯一生日的概率，我们将 364/365 提升到组中有对的**次方。使用 23 人小组的介绍性示例，这意味着我们想要确定我们可以将这 23 个人分成两个小组的方法的数量。**

![](img/400eeb0356301e99b643da6d5e7df1f7.png)

Photo by [Curtis MacNewton](https://unsplash.com/photos/vVIwtmqsIuk?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/calendar?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

从数量为 *n* 的项目中挑选数量为 *k* 的项目的公式如下:**n！/ ( *k* ！* ( *n* — *k* )！)**当我们为 *k* 插上 2，为 *n* 插上 23，我们的结果就是 253。因此，在我们 23 个样本中有 253 个可能的配对。当我们取 364/365 并将其提升到 253 次方时，结果是~0.4995，表明 23 个陌生人没有一个出生在同一天的概率为 49.95%。通过从 100.00%中减去这个结果，我们能够最终获得至少一对*与*拥有相同出生日期的概率:~50.05%。

生日悖论令人着迷的原因有很多。对于程序员来说，理解生日悖论很有用，因为它解释了**哈希冲突**的概念。

在我们深入探讨碰撞之前，我们先来讨论一下**哈希表**(也称为**哈希表**)。哈希表是一种以随机顺序存储项目的数据结构。这些项目存储在**桶**中，其数量由程序员决定。一个**散列函数**(Python 中的一个内置函数)负责将每一项分配给一个特定的桶索引号。

![](img/53b7dad301d539e400a4bf1e3e96a6c9.png)

Look, it’s a bucket. Photo by [Gregory Culmer](https://unsplash.com/photos/COovdd_3gzI?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/bucket?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

为了完成这个重要的任务，hash 函数为您想要存储的内容分配一个随机生成的数字。然后，它将这个随机数除以哈希表中有多少个存储桶。这个除法的余数是桶索引，然后内容被放在相应的桶中。

你可能会认为散列会导致项目在桶中的平均分布，就像你可能会推断 23 个陌生人的生日可能分布在日历年中一样。但是我们从数学上观察到，有大约 50%的可能性两个陌生人同一天过生日。我们还观察到，当这个小组只有区区 80 个人时，这种可能性会跃升至近 100%。

让我们把每个日历日想象成散列表中的一个桶。当其中一个陌生人发现他们的出生日期与一个小组成员的出生日期一致时，这就相当于一个新项目被分配到一个已经包含一个项目的桶中。这个事件被称为**哈希冲突**。你不需要在哈希冲突发生之前拥有大量的条目——毕竟，我们只需要一小群人来解释生日悖论。

![](img/71ce15912fe8e8f22c9b0a66ce72ca3a.png)

Chaining! Photo by [JJ Ying](https://unsplash.com/photos/PDxYfXVlK2M?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/link?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

有两种主要的方法来处理哈希表中的冲突。第一个是**线性探测**，它通过将新项目分配给下一个桶来处理冲突。第二个是**链接**，这需要将新的项目插入到桶中，即使那里已经有一个项目。哈希表通常用**链表**实现，这是一种线性数据结构。这意味着，除非是空的，否则每个桶都包含一个项目链表。因此，当一个条目被添加到一个已经包含某些内容的桶中时，这个新条目只是被添加到链表的末尾。

生日悖论就像一个哈希冲突解决方案。显然，在任何一天都有可能有不止一个人出生在 T21。因此，可以说真实世界的生日哈希表解决了它与**链接**的冲突！

# 参考资料和进一步阅读

*   “[生日悖论](http://www.efgh.com/math/birthday.htm)”，菲利普·j·埃尔德尔斯基
*   ”[概率和生日悖论](https://www.scientificamerican.com/article/bring-science-home-probability-birthday-paradox/)，《科学美国人》
*   ”[理解生日悖论](https://betterexplained.com/articles/understanding-the-birthday-paradox/)，更好解释
*   "[理解生日悖论](http://shanky.org/2018/11/08/understanding-the-birthday-paradox/)"，Shashank Tiwari