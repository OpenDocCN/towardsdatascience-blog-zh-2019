# 使用 SQL 查询从 Pandas 数据框架中提取数据的 1 个快速技巧

> 原文：<https://towardsdatascience.com/a-quick-tip-for-pulling-data-from-a-pandas-dataframe-using-sql-queries-bbe3b39d17d?source=collection_archive---------2----------------------->

![](img/6ecfaa1aac9a5168d91c7e537d75bae6.png)

Photo by [Bruce Hong](https://unsplash.com/@hongqi?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/funny-panda?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

> 想获得灵感？快来加入我的 [**超级行情快讯**](https://www.superquotes.co/?utm_source=mediumtech&utm_medium=web&utm_campaign=sharing) 。😎

熊猫图书馆对于任何使用 Python 分析数据的人来说都是非常棒的。

它非常容易使用，正确应用时速度很快，并且功能灵活。有了 Pandas，许多通常需要更多工作的功能，例如检索一些关于数据的基本统计数据，只需调用一个函数就可以完成！

尽管如此，有时我们更喜欢使用一种工具而不是另一种。如果你习惯于在 Excel、Tableau 或 SQL 上浏览数据，那么切换到 Pandas 仍然是一个跳跃。

如果你有 SQL 的背景，有一个很棒的 Python 库可以帮助你平稳过渡: [**Pandasql**](https://github.com/yhat/pandasql) 。

Pandasql 允许您编写 sql 查询来查询 pandas 数据框架中的数据。这允许您避开在熊猫中必须学习大量 Python 的正常要求。相反，您可以简单地在函数调用中编写常规的 SQL 查询，并在 Pandas dataframe 上运行它来检索您的数据！

# 用 Pandasql 查询熊猫数据帧

## 安装

我们可以通过一个快速的 pip 来安装 Pandasql:

```
pip install pandasql
```

## 加载数据

让我们从实际的数据集开始。我们将使用`seaborn`库加载 iris flowers 数据集:

```
*import* pandasql
*import* seaborn *as* sns

data = sns.load_dataset('iris')
```

## 选择

通常，如果我们想检索数据帧中的前 20 项，我们会对熊猫做这样的事情:

```
data.head(20)
```

有了 pandasql，我们可以像在 sql 数据库上运行标准 SQL 查询一样写出它。只需将 pandas 数据帧的名称作为您正在解析的表的名称，数据将被检索:

```
sub_data = pandasql.sqldf("SELECT * FROM data LIMIT 20;", globals())
print(sub_data)
```

我们可以用 SQL 中的 WHERE 进行的常规过滤操作也是适用的。让我们首先使用 pandas 提取所有大于 5 的数据:

```
sub_data = data[data["petal_length"] > 5.0]
```

为了在 SQL 中实现这一点，某些行只需添加 WHERE 调用来实现相同的过滤:

```
sub_data = pandasql.sqldf("SELECT * FROM data WHERE petal_length > 5.0;", globals())
```

当然，我们也可以总是只选择我们想要的列:

```
sub_data = pandasql.sqldf("SELECT petal_width, petal_length FROM data WHERE petal_length > 5.0;", globals())
```

这就是如何使用 SQL 查询从 pandas 数据帧中检索数据。

# 喜欢学习？

在 twitter 上关注我，我会在这里发布所有最新最棒的人工智能、技术和科学！也在 LinkedIn[上与我联系](https://www.linkedin.com/in/georgeseif/)！