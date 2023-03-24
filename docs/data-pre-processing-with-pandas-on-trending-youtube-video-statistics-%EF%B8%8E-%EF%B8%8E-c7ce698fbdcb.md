# 〠 ❤︎ ✔︎趋势 YouTuBe 视频统计中熊猫的数据预处理

> 原文：<https://towardsdatascience.com/data-pre-processing-with-pandas-on-trending-youtube-video-statistics-%EF%B8%8E-%EF%B8%8E-c7ce698fbdcb?source=collection_archive---------22----------------------->

本文的目的是提供一个标准化的数据预处理解决方案，可以应用于任何类型的数据集。您将学习如何将数据从最初的原始形式转换为另一种格式，以便为探索性分析和机器学习模型准备数据。

# 数据概述

*   这个数据集是来自美国的热门 YouTube 视频的每日记录。
*   数据包括视频标题、频道标题、发布时间、标签、视图、喜欢和不喜欢、描述和评论计数等。
*   数据集的形状是 16580 行* 16 列。

# 数据预处理路线图

![](img/6719c8a27c8a1beadc85f2c08a955f15.png)

# **数据预处理源代码**

## 数据集的基本观点

```
# A quick look at the dataset. Return 5 random rows.
df.sample(5)
```

![](img/a38f88fe10fc0ecbb3a7106273b47902.png)

```
# Return data types
df.dtypes
```

![](img/26c8cc8b15a67c18622fc6c4de1de23f.png)

```
# Dimensions of dataset
df.shape
```

![](img/37231dc963a3a728b1e03423ef18b52d.png)

```
# Statistical summary
df.describe()
```

![](img/ca8939d75b15a21608e96651f63c1335.png)

```
df_summary = df.describe(include="all")
df_summary
```

![](img/07fe669fc1d40019e9378f614a44d576.png)

*   “top”是最常出现的项目
*   “频率”是热门项目显示的次数
*   “NaN”表示无法对该类数据进行计算

```
# Unique value: value counts of a specific column
df['category_id'].value_counts()
```

![](img/122a0d1cbc8f0504e5d24040e95d9a06.png)

上图显示了“category_id”列中的唯一值。值“24”的计数是 3911。

```
# Look at a specific field
df.iloc[23,5]
```

![](img/8ef9c74d58b85fa2c7d2f9193e488532.png)

```
# Look at selected columns
columnWeCareAbout=['title','views','likes','dislikes','comment_count']
df[columnWeCareAbout].sample(5)
```

![](img/c632bd3f50234eb8885b7044c83d091f.png)

## 识别和处理丢失的数据

```
# Use heatmap to check missing data
sns.heatmap(df_summary.isnull(), yticklabels=False, cbar=False, cmap='viridis')
```

![](img/8b4584c0d39f6162864f68666e8bd820.png)

```
# See counts of missing value
for c in df_summary.columns:
    print(c,np.sum(df_summary[c].isnull()))
```

![](img/73e09a7a468649b98838809fe13aacde.png)

```
# Replace missing data
df_summary['views'].fillna(df_summary['views'].mean(), inplace=True)
```

以上代码用平均值填充缺失的数据。你可以在实际案例中考虑插值、中值或其他方法。

```
# Drop a column most value are missing
df_summary.drop(['thumbnail_link'], axis=1, inplace=True)sns.heatmap(df_summary.isnull(), yticklabels=False, cbar=False, cmap='viridis')
```

![](img/3ede342e434daef83ec1fb22b552a1a1.png)

No missing value detected

## 数据格式编排

```
# Change data type if needed
df['left_publish_time'] = pd.to_datetime(df['left_publish_time'], format='%Y-%m-%dT%H:%M:%S')
```

![](img/21886bcf4fbf9d8c3be464f5bc8b4cdb.png)![](img/5f785dec6f62d0909fde1cb3a14b6f27.png)

Change type from object to datetime

```
# Unit conversion# conversion factor
conv_fac = 0.621371# calculate miles
miles = kilometers * conv_fac
```

## 数据标准化

```
# Number in different range which influence the result differently
df[['views','likes','dislikes','comment_count']].head()
```

![](img/dfa6fe35230e22dc4a44e8d7864c3f51.png)

```
# Simple feature scaling
df['views'] = df['views'] / df['views'].max()
df[['views','likes','dislikes','comment_count']].head()
```

![](img/ee42df6765b6069bbea012a137869bdf.png)

New values in column “views”

```
# Min-max
df['likes'] = (df['likes'] - df['likes'].min()) / (df['likes'].max() - df['likes'].min())
df[['views','likes','dislikes','comment_count']].head()
```

![](img/acbf6b22e0ca2b49aab99973f0da4d73.png)

Normalized value in column “likes”

```
# Z-score
df['dislikes'] = (df['dislikes'] - df['dislikes'].mean()) / df['dislikes'].std()
df['comment_count'] = (df['comment_count'] - df['comment_count'].mean()) / df['comment_count'].std()df[['views','likes','dislikes','comment_count']].head()
```

![](img/51e21c92a99f5761be281b3abf01928f.png)

## 扔掉

*   将值分组到箱中
*   将数值转换成分类变量
*   “喜欢”是数字，我们希望将其转换为“低”、“中”、“高”，以更好地表示视频的受欢迎程度

```
binwidth = int(max(df['likes'])-min(df['likes']))/3
binwidth
```

![](img/4640aa75f1aa3f4208088285c467495e.png)

```
bins = range(min(df['likes']), max(df['likes']),binwidth)
group_names = ['Low','Medium','High']
df['likes-binned'] = pd.cut(df['likes'], bins, labels=group_names)
df['likes-binned']
```

![](img/a819dd4fac776fca09e8281837cb4c0e.png)![](img/e5696bb8ccc47766d51c853cb4d9dc9e.png)

Visualizing binned data

## 一键编码

*   为每个独特的类别添加虚拟变量
*   在每个类别中分配 0 或 1
*   将分类变量转换为数值

```
df['category_id'].sample(5)
```

![](img/e8219658fff68f224d4817415443e3fa.png)

```
category = pd.get_dummies(df['category_id'], drop_first=True)
category.head()
```

![](img/feb9863d686e29c12a3185f0d27e6bac.png)

```
# Add dummy values into data frame
df = pd.concat([df, category], axis=1)
df.sample(5)
```

![](img/c899fce4f7cb4e6cd29f01083dd2a938.png)

## 应用 IF 条件

```
# Add a column base on the conditions# df.loc[df.column_name condition, 'new column name'] = 'value if condition is met'df.loc[ df['likes'] > 1000 | df['views'] > 10000, 'popularity'] = 'Yes'  
df.loc[ df['likes'] <= 1000 & df['views'] <= 10000, 'popularity'] = 'No'
```

恭喜你。您完成了一篇冗长的文章，现在您知道了一个标准化的数据预处理解决方案，可以应用于任何类型的数据集。你的数据科学家难题又多了一块！源数据可以在这个[链接](https://www.kaggle.com/datasnaek/youtube-new)找到。

测验:为什么我们需要数据标准化？

下一步:探索性数据分析(EDA ),熊猫对 YouTuBe 视频统计数据进行趋势分析