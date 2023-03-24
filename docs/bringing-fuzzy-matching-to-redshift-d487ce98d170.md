# 如何在亚马逊红移中模糊匹配数据集

> 原文：<https://towardsdatascience.com/bringing-fuzzy-matching-to-redshift-d487ce98d170?source=collection_archive---------11----------------------->

## 使用 Python UDF 实现 Amazon 红移中的模糊匹配连接

![](img/e5eba577efe78def7f11face8eaf25cb.png)

Fuzzy Merging — Photo by [Markus Spiske](https://unsplash.com/@markusspiske?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

如果幸运的话，当在数据仓库中处理多个数据集时，会有某种类型的连接列可用于您想要放在一起的表。

当这种情况发生时，生活是美好的。然而，现代*大数据*解决方案正在开辟一个用例，将来自不同来源的各种数据整合在一起。

虽然我们可以[轻松地将这些数据存储在一个地方](https://medium.com/@lewisdgavin/how-to-architect-the-perfect-data-warehouse-b3af2e01342e)，但是将它们连接起来进行分析并不总是那么简单，因为这些数据集通常不是由同一个源系统生成的，所以要连接的干净的 ID 列并不总是可用的。即使为您提供相同信息的列也不总是以相同的格式提供，如果是用户捕获的，您永远无法保证一致性。

基于相似性将数据集连接在一起的一种方法是模糊匹配，特别是当您知道每个数据集中有基于文本的字段几乎相似时，例如用户输入的公司名称或产品名称。

这篇文章不会详细讨论模糊匹配的所有细节，但是会向你展示如何在 Redshift 中使用 Python 实现。

# 模糊匹配

在最简单的层面上，模糊匹配看起来产生两个事物有多相似的相似性分数。我将着重比较字符串来解释这个概念。

作为人类，我们很容易发现打字错误，或者在概念上理解两个相似的东西是相同的。模糊匹配算法试图帮助计算机做到这一点。两个字符串之间的匹配不是布尔真或假，即完全相同或不相同，模糊匹配给出的是接近度分数。

以“布鲁克林大桥”和“布鲁克林大桥”为例。即使在第二个字符串中有拼写错误，人类也很容易发现这是同一个东西。

一种模糊匹配算法，例如给出相似性百分比分数的 [Levenshtein distance](https://en.wikipedia.org/wiki/Levenshtein_distance) ，可能会将这两个字符串评分为至少 90%相似。我们可以使用它来设置我们希望“相似”的阈值，即任何两个模糊分数超过 80%的字符串都是匹配的。

# Python 实现

这些天你可以找到很多 Python 包，所以我不打算重新发明轮子。一个体面的使用 Levenshtein 实现模糊匹配的 python 包是 [**fuzzy wuzzy**](https://github.com/seatgeek/fuzzywuzzy) **。**

按照文档中说明的安装过程，您最终会得到一堆用于比较字符串的函数。

我在这个例子中使用的是一个简单的比率函数，它接受两个字符串并给出一个接近比率。这里有一个示例实现，展示了前面的布鲁克林大桥示例。

```
>from fuzzywuzzy import fuzz
>fuzz.ratio(“brooklyn bridge”, “brooklin bridge”)> 93
```

正如所料，这返回了一个相当高的分数，因为两者非常相似。

# 红移 UDF

用户定义的函数允许您使用 SQL 或 Python 向 Redshift 添加可重复的代码块。python 支持将允许我们采用上一节中的实现并添加到 Redshift 中，这样我们就可以像调用任何其他原生 SQL 函数一样简单地调用它。

首先，我们需要添加 fuzzywuzzy 库到红移。有一些[完整的文档](https://docs.aws.amazon.com/redshift/latest/dg/udf-python-language-support.html#udf-importing-custom-python-library-modules)，但是我将在下面概述基本步骤。

1.  从 github 下载 [fuzzywuzzy](https://github.com/seatgeek/fuzzywuzzy) 回购
2.  在回购中复制一份 fuzzywuzzy 文件夹，并将其压缩。
3.  将此压缩文件夹复制到 S3 桶中
4.  在 Redshift 中运行以下命令来导入 fuzzywuzzy 库

```
CREATE LIBRARY fuzzywuzzy LANGUAGE plpythonu FROM 's3://<bucket_name>/fuzzywuzzy.zip' CREDENTIALS 'aws_access_key_id=<access key id>;aws_secret_access_key=<secret key>'
```

完成后，我们现在可以继续使用红移中的这个库来创建函数。

```
CREATE FUNCTION fuzzy_test (string_a TEXT,string_b TEXT) RETURNS FLOAT IMMUTABLE
AS
$$
  FROM fuzzywuzzy import fuzz 
  RETURN fuzz.ratio (string_a,string_b) 
$$ LANGUAGE plpythonu;
```

我们现在可以测试它，并检查我们看到的结果是否与我们在本地看到的结果相同。

```
SELECT fuzzy_test('brooklyn bridge', 'brooklin bridge');> 93
```

# 包裹

就这么简单。这是一个很好的特性，由于 Python 中可用的库的范围，Python UDF 给了你很大的灵活性。

由于红移星团的力量，这意味着大规模的模糊匹配是可能的，这可能永远不会在笔记本电脑上完成。然而…

如果您打算将它用于连接，需要考虑的一件事是，它显然会比通常慢，因为在连接上不会发生太多优化。因此，如果您要匹配大型字符串数据集，那么请做好等待的准备:)