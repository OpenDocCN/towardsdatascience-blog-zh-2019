# 将熊猫数据导出到 Elasticsearch

> 原文：<https://towardsdatascience.com/exporting-pandas-data-to-elasticsearch-724aa4dd8f62?source=collection_archive---------8----------------------->

![](img/b38ba3bfae0a4f5fb7885edbfb81d243.png)

Photo by [CHUTTERSNAP](https://unsplash.com/@chuttersnap?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

如何将您的数据帧行发送到 elasticsearch 数据库。

# 介绍

因此，您已经完成了数据的下载和分析，并准备好将其转移到 Elasticsearch 数据库中。本文介绍了如何准备数据并将其发送到 elasticsearch 端点。

## 要求

```
Python 3.6.5
numpy==1.15.0
pandas==0.23.4
elasticsearch==6.3.1import numpy as np
import pandas as pdfrom elasticsearch import Elasticsearch
from elasticsearch import helpers
es = Elasticsearch(http_compress=True)
```

# 清理您的数据

Pandas dataframes 很乐意在您的数据中保存 NaN 值，但是对 null 值的容忍不是 elasticsearch 的特性。仔细想想，这是有道理的；您要求索引器不索引任何内容。您可以通过一些简单的函数运行数据来避免这种情况。

## 空白日期

如果您的数据框架有空白日期，您需要将其转换为 elasticsearch 接受的值。elasticsearch 中的日期可以是格式化的日期字符串(例如“6–9–2016”)、自 Unix 纪元以来的毫秒数或自 Unix Epoc 以来的秒数( [elastic docs](https://www.elastic.co/guide/en/elasticsearch/reference/current/date.html) )。使用毫秒的空日期，因为 Unix 纪元是 1970 年 1 月 1 日。如果你有包括 70 年代早期的历史日期，你可能要考虑一些其他的。

这是一个可以和 dataframe 一起使用的简单函数。apply()清理日期列。

```
from datetime import datetimedef safe_date(date_value):
    return (
        pd.to_datetime(date_value) if not pd.isna(date_value)
            else  datetime(1970,1,1,0,0)
    )df['ImportantDate'] = df['ImportantDate'].apply(safe_date)
```

## 避免其他空白值

任何包含空值的字段都是有问题的，因为空日期。字符串值比日期更容易，但是您需要提供一个值。下面的代码使用 df.apply()函数将空格替换为安全字符串。

```
def safe_value(field_val):
    return field_val if not pd.isna(field_val) else "Other"df['Hold'] = df['PossiblyBlankField'].apply(safe_value)
```

## 创建文档

一旦您确信您的数据已经准备好发送到 Elasticsearch，就该将行转换为文档了。Panda dataframes 有一个方便的“iterrows”函数可以直接实现这一点。它返回行的索引和包含行值的对象。这是一个“pandas.core.series.Series”对象，但其行为类似于传统的 python 字典。这个代码片段演示了如何将 iter 响应分配给变量。

```
df_iter = df.iterrows()index, document = next(df_iter)
```

Elasticsearch 需要 python 格式的数据，使用。Series 对象的 to_dict()方法。但是，您可以选择要发送到数据库的数据，并使用简单的过滤函数。注意下面的函数返回一个字典理解。

```
use_these_keys = ['id', 'FirstName', 'LastName', 'ImportantDate']def filterKeys(document):
    return {key: document[key] for key in use_these_keys }
```

## 发电机

我们终于准备好使用 python 客户机和助手向 Elasticsearch 发送数据了。helper.bulk api 需要一个 Elasticsearch 客户端实例和一个生成器。如果你不熟悉发电机，去了解他们的记忆尊重的好处。如果你没有时间做这个，只需要理解神奇的是`yield`，当 bulk.helpers 函数请求数据时，a 会给出数据。下面是遍历数据框架并将其发送到 Elasticsearch 的代码。

```
from elasticsearch import Elasticsearch
from elasticsearch import helpers
es_client = Elasticsearch(http_compress=True)def doc_generator(df):
    df_iter = df.iterrows()
    for index, document in df_iter:
        yield {
                "_index": 'your_index',
                "_type": "_doc",
                "_id" : f"{document['id']}",
                "_source": filterKeys(document),
            }
    raise StopIterationhelpers.bulk(es_client, doc_generator(your_dataframe))
```

## 分解生成的字典

doc_generator 的工作只是提供一个带有特定值的字典。这里有一些关于这里发生的事情的细节的评论。

```
"_index": 'your_index',
```

这是您在 Elasticsearch 中的索引名称。如果没有索引，可以在这里使用任何有效的索引名。Elasticsearch 会尽最大努力自动索引你的文档。但是，提前创建索引是避免拒绝文档和优化索引过程的好主意。

```
"_type": "_doc",
```

请注意:`_type`已被 Elasticsearch 弃用。版本 6.3.1 仍然支持命名类型，但这是开始转换为' _doc '的好时机。

```
“_id” : f”{document[‘id’]}”,
```

`_id` 是 Elasticsearch 的唯一 id。不要把它和你文档中自己的“id”字段混淆。这可能是添加 itterows()中的`index`变量的好地方，通过类似`f”{document['id']+index}".`的东西使文档更加独特

```
"_source": filterKeys(document),
```

_source 是这个练习的核心:要保存的文档。使用`document.to_dict()`或任何其他有效的 python 字典都可以。

```
raise StopIteration
```

出于礼貌，我加入了这一行。bulk.helpers 函数将处理生成器的突然终止。但是提高 StopIteration 可以省去这个麻烦。

# 结论

注意一些细节，把你的熊猫数据转移到一个弹性搜索数据库是没有戏剧性的。只要记住空值是 elasticsearch 的一个问题。剩下的就是创建一个生成器，将您的行处理成 python 字典。