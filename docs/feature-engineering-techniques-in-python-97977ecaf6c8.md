# Python 中的特征工程技术

> 原文：<https://towardsdatascience.com/feature-engineering-techniques-in-python-97977ecaf6c8?source=collection_archive---------19----------------------->

特征工程是每个机器学习项目中至关重要的一部分。在这篇文章中，我们将围绕一些技术来处理这项任务。请不要犹豫提出新的想法，我会尽可能保持这篇文章的更新。

![](img/20fccf8442b96bf6ef27d6174986c770.png)

## **合并训练和测试**

当执行特征工程时，为了有一个通用的模型，如果你有两个文件，只需将它们合并(训练和测试)就可以了。

```
df = pd.concat([train[col],test[col]],axis=0)
#The label column will be set as NULL for test rows# FEATURE ENGINEERING HEREtrain[col] = df[:len(train)]
test[col] = df[len(train):]
```

## 记忆减少

有时，列的类型编码不是最佳选择，例如，用 int32 编码只包含 0 到 10 的值的列。最流行的函数之一是使用一个函数，通过将列的类型转换为尽可能最好的类型来减少内存的使用。

## 移除异常值

移除异常值的常用方法是使用 Z 值。

如果您希望删除至少有一列包含异常值(用 Z 得分定义)的每一行，您可以使用以下代码:

```
from scipy import stats
df[(np.abs(stats.zscore(df)) < 3).all(axis=1)]
```

## 南诡计

一些基于树的算法可以处理 NAN 值，但他会在 NAN 和非 NAN 值之间有一个步骤，这有时可能是无意义的。一个常见的技巧是用低于所考虑的列中最低值的值填充所有 nan 值(例如-9999)。

```
df[col].fillna(-9999, inplace=True)
```

## 分类特征

您可以使用标注编码来处理分类要素，将其作为数字来处理。您也可以决定将它们视为类别。我建议两者都尝试一下，并通过这一行代码(在标签编码之后)保持交叉验证。

```
df[col] = df[col].astype('category')
```

## 组合/拆分

有时字符串变量在一个变量中包含多个信息。比如`FRANCE_Paris`。您将需要使用正则表达式或使用拆分方法来拆分它，例如:

```
new **=** df["localisation"].str.split("_", n **=** 1, expand **=** True)
df['country'] = new[0]
df['city']=new[1]
```

否则，两个(字符串或数字)列可以合并为一列。例如，包含法国某个部门(75 代表巴黎)和地区代码(001)的列可以变成邮政编码:75001)

```
df['zipcode'] = df['departement_code'].astype(str)
                +'_'
                +df['disctrict_code'].astype(str)
```

## 线性组合

特征工程的一个共同特点是应用简单的数学运算来创造新的特征。例如，如果我们有一个矩形的宽度和高度，我们可以计算面积。

```
df['area'] = df['width'] * df['height']
```

## 计数栏

创建列从流行的`value_count`方法创建列对于基于树的算法来说是一种强大的技术，用来定义一个值是稀有的还是常见的。

```
counts = df[col].value_counts.to_dict()df[col+'_counts'] = df[col].map(counts)
```

## 处理日期

为了分析事件，处理日期和解析日期的每个元素是至关重要的。

首先，我们需要转换日期列(通常被认为是熊猫的字符串列)。其中最重要的一个领域是知道如何使用`format`参数。强烈推荐将[这个站点](http://strftime.org/)保存为书签！:)

例如，如果我们要用下面的格式转换一个日期列:`30 Sep 2019`我们将使用这段代码:

```
df['date'] =  pd.to_datetime(df[col], format='%d %b %Y')
```

一旦您的列被转换为`datetime`，我们可能需要提取新闻列中的日期部分:

```
df['year'] =  df['date'].year
df['month'] = df['date'].month
df['day'] = df['date'].day
```

## 聚合/组统计

为了继续检测稀有和常见的值，这对于机器学习预测非常重要，我们可以决定基于静态方法来检测一个值在子组中是稀有还是常见。例如，我们想通过计算每个子类的平均值来了解哪个智能手机品牌用户的通话时间最长。

```
temp = df.groupby('smartphone_brand')['call_duration']
       .agg(['mean'])
       .rename({'mean':'call_duration_mean'},axis=1)df = pd.merge(df,temp,on='smartphone_brand',how=’left’)
```

利用这种方法，ML 算法将能够辨别哪个呼叫具有与智能手机品牌相关的 call_duration 的非公共值。

## 标准化/规范化

规范化有时非常有用。

为了实现列自身的规范化:

```
df[col] = ( df[col]-df[col].mean() ) / df[col].std()
```

或者可以根据另一列对一列进行规范化。例如，如果您创建一个组统计数据(如上所述),表明每周`call_duration`的平均值。然后，您可以通过以下方式消除时间依赖性

```
df[‘call_duration_remove_time’] = df[‘call_duration’] — df[‘call_duration_week_mean’] 
```

新的变量`call_duration_remove`不再随着时间的推移而增加，因为我们已经针对时间的影响对其进行了标准化。

## Ultime 特色工程技巧

> 每一篇专栏文章都为预处理和模型训练增加了时间计算。我强烈建议测试一个新特性，看看这些特性如何改进(或不改进…)您的评估指标。 ***如果不是这样，你应该删除创建/修改的特征。***