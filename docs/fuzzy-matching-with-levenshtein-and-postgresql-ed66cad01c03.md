# 基于 Levenshtein 和 PostgreSQL 的模糊匹配

> 原文：<https://towardsdatascience.com/fuzzy-matching-with-levenshtein-and-postgresql-ed66cad01c03?source=collection_archive---------8----------------------->

## 构建容错搜索引擎

用户体验是多种因素综合的结果:UI 设计、响应时间、对用户错误的容忍度等。这篇文章的范围属于第三类。更具体地说，我将解释如何使用 PostgreSQL 和 Levenshtein 的距离算法来处理搜索表单中的用户错误。例如，我们可能在我们的站点中查找一个名为“Markus”的用户，尽管我们可能以“Marcus”的名称搜索他(这两个名称都存在，并且在不同的语言中是等价的)。

让我们从熟悉 Levenshtein 的距离算法开始。

# Levenshtein 距离算法

Levenshtein 的距离(为了简洁起见，我们从现在开始称之为 LD)旨在计算两个字符串之间的不相似性(因此，值越高，它们越不相似)。此度量表示需要应用于其中一个字符串以使其等于另一个字符串的操作数量。这些操作包括:插入字符、删除字符或替换字符。值得一提的是，每个操作都有一个相关的成本，尽管通常将所有操作的成本都设置为 1。

你很少需要实现 LD 算法，因为它已经存在于许多主流编程语言的库中。然而，理解它是如何工作的是值得的。我们将通过一个示例来确定“Marcus”和“Markus”之间的 LD，将所有操作的等价成本设置为 1:

1.  我们创建一个 m×n 维的矩阵 D，其中 m 和 n 代表我们的两个字符串的长度(向前称为 A 和 B)。
2.  我们从上到下从左到右迭代矩阵，计算相应子串之间的 LD。
3.  我们计算增加的成本(ac)，如果 A[i]等于 B[j]则为 0，否则为 1。
4.  我们执行以下操作:

D[i，j] = min(D[i-1，j] + 1，D[i，j-1] + 1，D[i-1，j-1] + ac)

这意味着单元(I，j)具有由下式中的最小值确定的 LD:

*   正上方的单元格加 1(由于插入操作)。
*   左边的单元格加 1(由于插入操作)。
*   左上方的单元格加上 ac(由于替换操作)。

以下矩阵代表了这一过程的最终结果:

![](img/421cb5c54b22605c2098556713ac71fa.png)

LD matrix of “Marcus” VS “Markus”

上面的矩阵不仅提供了完整字符串之间的 LD，还提供了子字符串任意组合之间的 LD。完整字符串之间的 LD 可以在位置(m，n)找到。在这种情况下，我们可以看到“马库斯”和“马库斯”的 LD 为 1(红色)，这是由“k”替换“c”造成的。

# 在 PostgreSQL 中使用 Levenshtein 距离

现在你已经了解了算法，是时候进入实用部分了。使用 PostgreSQL 应用 LD 算法非常简单，这都要归功于 [fuzzystrmatch](https://www.postgresql.org/docs/current/fuzzystrmatch.html) 扩展。这个扩展为模糊字符串匹配提供了不同的算法。可用的选项有 LD 算法和一组语音函数。请注意，这些语音函数(Soundex、Metaphone 和 Double Metaphone)对于非英语字符串可能无法发挥最佳性能。因此，我认为 LD 是国际申请中最可靠的选择。

不过，让我们继续将这个扩展添加到我们的 PostgreSQL 数据库中:

```
CREATE EXTENSION fuzzystrmatch;
```

此时，计算 LD 就像运行以下查询一样简单:

```
SELECT levenshtein(str1, str2);
```

例如:

```
SELECT levenshtein('Marcus', 'Markus');
```

因此，如果您有一个用户表，并且想要查找与用户输入具有相似名称的所有用户(例如，设置最大 LD 为 1)，您的查询可能如下所示:

```
SELECT name FROM user WHERE levenshtein('Marcus', name) <= 1;
```

在这种情况下，“Marcus”代表用户输入。在这个查询中，我们将搜索名为 Marcus 或者名称的 LD 为 1 的用户(例如，“Markus”)。

请注意，fuzzystrmatch 提供了另一个有趣的函数，在这种情况下会更有效一些:levenshtein_less_equal。该函数允许您定义一个最大距离阈值，超过该阈值后 PostgreSQL 将停止计算。换句话说，如果我们将最大阈值设置为 1，并且我们的字符串对的 ld 为 6，则 fuzzystrmatch 将在达到 LD 为 2 后停止计算，因为无论如何，我们对这样的字符串对的 LD 不感兴趣，从而节省一些计算资源(并产生 LD 为 2)。最后，还值得注意的是，这两个函数都有一个版本，允许您定义每个操作(插入、删除和替换)的成本。