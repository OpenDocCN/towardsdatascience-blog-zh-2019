# 简化你生活的九个 Postgres 概念🐘

> 原文：<https://towardsdatascience.com/nine-postgres-concepts-to-simplify-your-life-a0aa8c9bd87d?source=collection_archive---------15----------------------->

## 我不喜欢无意义的介绍，所以我们直接开始吧。

![](img/620ee876827d2f5d1663da549da4527c.png)

“If we have data, let’s look at data. If all we have are opinions, let’s go with mine.”

## 1.从文本语料库中生成 n 元语法

与使用另一种脚本语言相比，查询文本语料库并立即获得 n 元语法要简单得多。结果数据可以被存储用于多种目的，包括但不限于:关键短语识别、自动完成提示、自动纠正提示。这个集合还可以使用上下文 word2vec 模型创建单词和短语向量。

A way of generating n-grams in Postgres

这里，<ordered record="" set="" of="" words="">是按单词出现的顺序排列的单词记录集(直接获得或通过一些操作获得)。看起来是这样的:</ordered>

Example Record Set of ordered words

## 2.高效地从表中删除重复项

I know not why I use so many GIFs

考虑到多个约束，下面的查询非常适合从表中删除重复项。这里，考虑列 var1 和列 var2 相同的所有行，保留一个副本，其余的被删除。

Efficiently delete duplicates rows from a table with a set of specific constraints.

## 3.忽略冲突的快速更新

You think throwing an error in my face will stop me from updating you, you scum of a table?

通常，更新表中的行值会导致与现有行的冲突。一些流氓条目阻止简单直接的更新查询运行到最后。为了解决这个问题，一个简单的程序可以逐行更新表格，忽略与约束冲突的条目，这很有意思。您可以选择记录这些行，以便在异常处理中进一步处理。

Handle update conflicts smarter.

另一种处理方法是移除冲突约束，正常更新表，处理冲突行(例如移除重复项)并重新添加约束。这不是推荐的方法。

## 4.复制带有现有约束的表

复制表时，约束有时是必不可少的，可以包括如下内容:

```
CREATE TABLE *table_copy* (LIKE *_table* INCLUDING ALL);INSERT INTO *table_copy* SELECT * from *_table*;
```

## 5.处理行锁

很容易出现这样的情况:多个进程持有对公共关系的锁，从而导致死锁情况。过时的查询或断开的连接是过时的，可以这样查询:

Get a list of queries and their details that hold locks over relations in a dB.

可以使用以下查询终止阻塞和/或被阻塞的语句:

```
SELECT pg_terminate_backend(*pid*)
```

## 6.查询 JSON 和 JSONB 列

很容易忘记 JSON(/B)及其数组可以像在脚本语言中一样被查询。将下面的 *jsonb* 结构视为关系中的一个单元格:

以下所有查询都是可能的:

## 7.高效的文本搜索和构建自动完成/自动更正模块！

Correct those typos and complete those sentences by using the power of tri-gram search.

我建议你在开始搜索任务之前阅读这篇博客文章:

*   [**发现 Postgres 指数背后的计算机科学**](http://patshaughnessy.net/2014/11/11/discovering-the-computer-science-behind-postgres-indexes)

在另一篇[文章](/implementing-auto-complete-with-postgres-and-python-e03d34824079)中，我描述了提取名词块的基本步骤，将它们存储在数据库中，查询它们并对结果进行排序。本文根据包含的名词块短语搜索数据点。这个可以扩展到全文搜索！

## 8.分组依据，每组多行

像 grouping 这样的聚合运算符为每个分组元素返回一个分组行。为了获得每个组的多行，下面的查询创造了奇迹。

## 9.有价值的信息

我发现以下三个概念在日常查询中有巨大的价值。

## A.无效的

```
SELECT null **IS** null; 
--trueSELECT null **IS DISTINCT FROM** null;
--falseSELECT null **=** null; 
--nullSELECT null and null; 
--nullSELECT null **isnull**; 
--trueSELECT null notnull; 
--falseSELECT true or null; 
--trueSELECT true and null; 
--nullSELECT null is **unknown**; 
--true
```

## B.对称的

```
SELECT 2 BETWEEN 1 and 5; 
--trueSELECT 2 BETWEEN 5 and 1; 
--falseSELECT 2 BETWEEN SYMMETRIC 5 and 1; 
--true
```

## C.PgAdmin 具有查询的自动补全功能

鲜为人知的功能。从文档中:

> *要使用自动完成功能，请键入您的查询；当您希望查询编辑器建议查询中可能出现的下一个对象名称或命令时，请按下****Control+空格键*** *组合键。例如，键入“*SELECT * FROM*”(不带引号，但有一个尾随空格)，然后按 Control+Space 组合键从自动完成选项的弹出菜单中进行选择。*

## D.睡眠

```
SELECT pg_sleep(5);
--sleeps for 5 seconds indeed!
```

See you in the next one!

## 注意事项:

我喜欢阅读的一个很棒的博客可以在这里找到。