# 让你的工作更有效率的 10 个技巧

> 原文：<https://towardsdatascience.com/10-python-pandas-tricks-that-make-your-work-more-efficient-2e8e483808ba?source=collection_archive---------0----------------------->

## 有些命令您可能已经知道，但可能不知道它们可以这样使用

![](img/055d47294ac89aec8311021f9e61dbda.png)

Photo from [https://unsplash.com/](https://unsplash.com/)

Pandas 是一个广泛用于结构化数据的 Python 包。有很多很好的教程，但在这里我还是想介绍一些读者以前可能不知道的很酷的技巧，我相信它们是有用的。

# 阅读 _csv

每个人都知道这个命令。但是您试图读取的数据很大，尝试添加这个参数: **nrows = 5** 以便在实际加载整个表之前只读取表的一小部分。那么你可以通过选择错误的分隔符来避免这个错误(它可能不总是用逗号分隔)。

(或者，您可以使用 linux 中的' head '命令检查任何文本文件中的前 5 行: **head -n 5 data.txt** (感谢 **Ilya Levinson** 在这里指出了一个拼写错误))

然后，可以通过使用`df.columns.tolist()`提取所有列来提取列列表，然后添加 **usecols = ['c1 '，' c2 '，…]【T10]参数来加载需要的列。此外，如果您知道一些特定列的数据类型，您可以添加参数 **dtype = {'c1': str，' c2': int，…}** ，这样它会加载得更快。这个参数的另一个优点是，如果您有一个既包含字符串又包含数字的列，那么将它的类型声明为 string 是一个很好的做法，这样当您试图使用这个列作为键来合并表时就不会出现错误。**

# select_dtypes

如果数据预处理必须在 Python 中完成，那么这个命令会节省您一些时间。读入表后，每列的默认数据类型可以是 bool、int64、float64、object、category、timedelta64 或 datetime64。您可以首先通过以下方式检查分布

`df.dtypes.value_counts()`

要了解数据帧的所有可能的数据类型，那么

`df.select_dtypes(include=['float64', 'int64'])`

选择只有数字特征的子数据帧。

# 复制

如果您还没有听说过，这是一个重要的命令。如果您执行以下命令:

```
import pandas as pd
df1 = pd.DataFrame({ 'a':[0,0,0], 'b': [1,1,1]})
df2 = df1
df2['a'] = df2['a'] + 1
df1.head()
```

你会发现 df1 变了。这是因为 df2 = df1 不是制作 df1 的副本并将其分配给 df2，而是建立一个指向 df1 的指针。所以 df2 的任何变化都会导致 df1 的变化。要解决这个问题，您可以

```
df2 = df1.copy()
```

或者

```
from copy import deepcopy
df2 = deepcopy(df1)
```

# 地图

这是一个很酷的命令，可以进行简单的数据转换。首先定义一个字典，其中“键”是旧值，“值”是新值。

```
level_map = {1: 'high', 2: 'medium', 3: 'low'}
df['c_level'] = df['c'].map(level_map)
```

举几个例子:真，假到 1，0(用于建模)；定义级别；用户定义的词汇编码。

# 申请还是不申请？

如果我们想创建一个新列，并以其他几列作为输入，apply 函数有时会非常有用。

```
def rule(x, y):
    if x == 'high' and y > 10:
         return 1
    else:
         return 0df = pd.DataFrame({ 'c1':[ 'high' ,'high', 'low', 'low'], 'c2': [0, 23, 17, 4]})
df['new'] = df.apply(lambda x: rule(x['c1'], x['c2']), axis =  1)
df.head()
```

在上面的代码中，我们定义了一个具有两个输入变量的函数，并使用 apply 函数将其应用于列“c1”和“c2”。

但是“应用”的问题是它有时太慢了。比方说，如果你想计算两列“c1”和“c2”的最大值，你当然可以这样做

`df['maximum'] = df.apply(lambda x: max(x['c1'], x['c2']), axis = 1)`

但是您会发现它比这个命令慢得多:

`df['maximum'] = df[['c1','c2']].max(axis =1)`

**要点**:如果你可以用其他内置函数完成同样的工作，就不要使用 apply(它们通常更快)。例如，如果要将列' c '舍入为整数，请执行 **round(df['c']，0)** 或 **df['c']。round(0)** 而不是使用应用函数:`df.apply(lambda x: round(x['c'], 0), axis = 1)`。

# 值计数

这是一个检查值分布的命令。例如，如果您想检查“c”列中每个单独值的可能值和频率，您可以

`df['c'].value_counts()`

这里有一些有用的技巧/论据:
A. **normalize = True** :如果你想检查频率而不是计数。dropna = False :如果你也想在统计数据中包含丢失的值。
C. `df['c'].value_counts().reset_index()`:如果您想将统计表转换成熊猫数据帧并对其进行操作
D. `df['c'].value_counts().reset_index().sort_values(by='index')`:在‘c’列中显示按不同值排序的统计数据，而不是计数。

(更新 2019.4.18 —针对上面的 d，**郝洋**指出了一个更简单的办法，没有。reset_index(): `df['c'].value_counts().sort_index()`

# 缺失值的数量

构建模型时，您可能希望排除缺少太多值的行/缺少所有值的行。你可以用。isnull()和。sum()计算指定列中缺失值的数量。

```
import pandas as pd
import numpy as npdf = pd.DataFrame({ 'id': [1,2,3], 'c1':[0,0,np.nan], 'c2': [np.nan,1,1]})
df = df[['id', 'c1', 'c2']]
df['num_nulls'] = df[['c1', 'c2']].isnull().sum(axis=1)
df.head()
```

# 选择具有特定 id 的行

在 SQL 中，我们可以使用 SELECT * FROM … WHERE ID in ('A001 '，' C022 '，…)来获得具有特定 ID 的记录。如果你想对熊猫做同样的事情，你可以做

```
df_filter = df['ID'].isin(['A001','C022',...])
df[df_filter]
```

# 百分位组

您有一个数字列，并且想要将该列中的值分组，比如前 5%归入第 1 组，5–20%归入第 2 组，20%-50%归入第 3 组，后 50%归入第 4 组。当然，你可以用熊猫来做。切，但我想在这里提供另一种选择:

```
import numpy as np
cut_points = [np.percentile(df['c'], i) for i in [50, 80, 95]]
df['group'] = 1
for i in range(3):
    df['group'] = df['group'] + (df['c'] < cut_points[i])
# or <= cut_points[i]
```

运行速度更快(不使用应用函数)。

# to_csv

同样，这是每个人都会使用的命令。这里我想指出两个窍门。第一个是

`print(df[:5].to_csv())`

您可以使用这个命令准确地打印出将要写入文件的前五行。

另一个技巧是处理混合在一起的整数和缺失值。如果一个列同时包含缺失值和整数，数据类型仍然是 float 而不是 int。导出表格时，可以添加 **float_format='%.0f'** 将所有浮点数四舍五入为整数。如果您只想要所有列的整数输出，就使用这个技巧——您将去掉所有烦人的“. 0”