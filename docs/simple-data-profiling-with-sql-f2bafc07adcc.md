# 使用 SQL 进行简单的数据分析

> 原文：<https://towardsdatascience.com/simple-data-profiling-with-sql-f2bafc07adcc?source=collection_archive---------11----------------------->

![](img/78654d21af70bfc37ab75e5fb07225ad.png)

What lies beneath : like the water in this peaceful river, your database can conceal any number of problems. (photo: Author)

## 在开始之前，了解数据的真实状态。

许多组织将其数据存储在符合 SQL 的数据库中。至少在生命周期的一部分中，这些数据库是有机增长的，这并不是闻所未闻的，所以表可能包含也可能不包含它们的标签会让您认为它们包含的内容，并且质量可能符合也可能不符合您的期望并对您的应用程序有用。

无论如何，创建数据库的最初目的很少是为数据科学家创建一个方便的、结构良好的环境来使用分析基础。因此，尽管您渴望将数据迁移到 R 或 Python 中进行分析，并有望用它来构建模型，但是您需要弄清楚哪些数据是您所需要的，哪些数据满足数据质量的基本标准，从而使其值得使用。

确实，主要的数据库包都有专门用于数据分析的工具。然而，同样值得注意的是，这些工具学习起来很复杂，有时看起来像是敲碎核桃的大锤。真正需要的是一些能够快速完成工作的简单查询。

在本系列的早期文章中，我们研究了 SQL 中的[分位数，以及 SQL 中缺失值的探索性统计。这一次，我们将着眼于识别独特的价值，并探索它们出现的频率。](/summarising-data-with-sql-3d7d9dea0016)

唯一值本身并不难识别— SQL 有一个内置的关键字 DISTINCT，它将返回的行限制为具有唯一值的行。因此，通过创建查询 SELECT DISTINCT col _ name from Table 并查看返回了多少行来发现任何行的值的数量是很容易的。

简单地知道特定列中唯一值的数量有点用处，但可能不如知道变量是分类的情况下每个值出现的次数有用(如果您事先不知道，大多数值的多次出现可能是这是分类值的强烈信号)。

这也相对简单，因为您可以将 DISTINCT 等关键字放入 COUNT 等聚合的括号中。因此

```
select count(DISTINCT x) from tablegroup by x
```

更有用的是知道唯一变量的比例。如果字段的标签很差，并且您需要仔细检查哪些列可能是唯一 id，这可能会很有帮助，同时也可以避免将表示简单标识符的字段误认为是包含信息的字段。不要忘记，由于 COUNT 返回一个整数，您需要转换结果以获得一个分数:

```
select cast(count(DISTINCT x) as decimal)/count(X) 
from table
```

对于分类变量，理解唯一值的数量是理解整个数据集的总体有用性的重要一步。

罗伯特·德格拉夫的书《管理你的数据科学项目》[](https://www.amazon.com/Managing-Your-Data-Science-Projects/dp/1484249062/ref=pd_rhf_ee_p_img_1?_encoding=UTF8&psc=1&refRID=4X4S14FQEBKHZSDYYMZY)**》已经通过出版社出版。**

*[*在 Twitter 上关注罗伯特*](https://twitter.com/RobertdeGraaf2)*