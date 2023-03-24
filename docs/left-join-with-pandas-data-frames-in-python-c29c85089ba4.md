# Python 中熊猫数据框的左连接

> 原文：<https://towardsdatascience.com/left-join-with-pandas-data-frames-in-python-c29c85089ba4?source=collection_archive---------0----------------------->

关于如何在左连接的结果中正确标记空值来源的教程。

![](img/9cca7e3fa1107f721cbc2f88d9e2c61c.png)

StackOverflow 文章 [Pandas Merging 101](https://stackoverflow.com/questions/53645882/pandas-merging-101) 详细介绍了合并 Pandas 数据帧。然而，我给数据科学课后测试评分的经验让我相信左连接对许多人来说仍然是一个挑战。在本文中，我将展示如何正确处理 Pandas 左连接中的右表(数据框)包含空值的情况。

让我们考虑一个场景，其中我们有一个表`transactions`包含一些用户执行的事务，还有一个表`users`包含一些用户属性，例如他们喜欢的颜色。我们希望用用户的属性来注释事务。以下是数据框:

```
import numpy as np
import pandas as pdnp.random.seed(0)
# transactions
left_df = pd.DataFrame({'transaction_id': ['A', 'B', 'C', 'D'], 
                       'user_id': ['Peter', 'John', 'John', 'Anna'],
                       'value': np.random.randn(4),
                      })# users
right_df = pd.DataFrame({'user_id': ['Paul', 'Mary', 'John',
                                     'Anna'],
                        'favorite_color': ['blue', 'blue', 'red', 
                                           np.NaN],
                       })
```

请注意，彼得不在`users`表中，安娜也没有最喜欢的颜色。

```
>>> left_df
  transaction_id user_id     value
0              A   Peter  1.867558
1              B    John -0.977278
2              C    John  0.950088
3              D    Anna -0.151357>>> right_df
  user_id favorite_color
0    Paul           blue
1    Mary           blue
2    John            red
3    Anna            NaN
```

使用用户 id 上的左连接将用户喜欢的颜色添加到事务表中似乎很简单:

```
>>> left_df.merge(right_df, on='user_id', how='left')
  transaction_id user_id     value favorite_color
0              A   Peter  1.867558            NaN
1              B    John -0.977278            red
2              C    John  0.950088            red
3              D    Anna -0.151357            NaN
```

我们看到彼得和安娜在`favorite_color`列中有`NaN`。然而，丢失的值有两个不同的原因:Peter 的记录在`users`表中没有匹配，而 Anna 没有最喜欢的颜色的值。在某些情况下，这种细微的差别很重要。例如，它对于在初始勘探期间理解数据和提高数据质量至关重要。

这里有两个简单的方法来跟踪左连接结果中缺少值的原因。第一个由`merge`函数通过`indicator`参数直接提供。当设置为`True`时，结果数据帧有一个附加列`_merge`:

```
>>> left_df.merge(right_df, on='user_id', how='left', indicator=True)
  transaction_id user_id     value favorite_color     _merge
0              A   Peter  1.867558            NaN  left_only
1              B    John -0.977278            red       both
2              C    John  0.950088            red       both
3              D    Anna -0.151357            NaN       both
```

第二种方法与它在 SQL 世界中的实现方式有关，它在右边的表中显式地添加了一个表示`user_id`的列。我们注意到，如果两个表中的连接列具有不同的名称，那么这两个列都会出现在结果数据框中，因此我们在合并之前重命名了`users`表中的`user_id`列。

```
>>> left_df.merge(right_df.rename({'user_id': 'user_id_r'}, axis=1),
               left_on='user_id', right_on='user_id_r', how='left')
  transaction_id user_id     value user_id_r favorite_color
0              A   Peter  1.867558       NaN            NaN
1              B    John -0.977278      John            red
2              C    John  0.950088      John            red
3              D    Anna -0.151357      Anna            NaN
```

一个等效的 SQL 查询是

```
**select**
    t.transaction_id
    , t.user_id
    , t.value
    , u.user_id as user_id_r
    , u.favorite_color
**from**
    transactions t
    **left join**
    users u
    **on** t.user_id = u.user_id
;
```

总之，添加一个额外的列来指示 Pandas left join 中是否有匹配，这允许我们随后根据用户是否已知但没有最喜欢的颜色或者用户是否从`users`表中缺失来不同地处理最喜欢的颜色的缺失值。

这篇文章最初出现在 [Life Around Data](http://www.lifearounddata.com/left-join-with-pandas-data-frames-in-python/) 博客上。*照片由 Unsplash 上的* [*伊洛娜·弗罗利希*](https://unsplash.com/@julilona) *拍摄。*