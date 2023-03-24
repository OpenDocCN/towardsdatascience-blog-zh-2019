# 数据工程-熊猫 101

> 原文：<https://towardsdatascience.com/data-engineering-pandas-101-d8bf5d68750c?source=collection_archive---------15----------------------->

Pandas 是数据分析和工程的一个很好的工具。我已经使用它大约三年了——在那之前，它是 Python 库的大杂烩，有点令人生厌。

> 熊猫= Python 数据分析库

![](img/3e9ff2e0adcb222bc62b9e3acaff497b.png)

那么，熊猫是什么，它为什么伟大？

*   你可以很容易地从本地文件或连接到数据库远程加载数据。
*   你可以用**探索**的方式处理数据，很容易地总结和查看数据的统计数据；
*   你可以**清理**、**操纵**(切片、切块)**重塑**并组合数据，更轻松地沿着链条往下处理；
*   您可以更有效地处理**大型数据集**——减少使用它们时使用的内存，非初学者的深入文章[在此](https://www.dataquest.io/blog/pandas-big-data/)。
*   您可以访问**数据帧**——一种类似表格的数据结构(我们在 R 语言中已经有很长时间了)；
*   您可以对结果数据集进行基本的**绘图**——相当不错；

熊猫官方网站在这里:

[](https://pandas.pydata.org/) [## Python 数据分析库- pandas: Python 数据分析库

### pandas 是一个开源的、BSD 许可的库，提供高性能、易于使用的数据结构和数据…

pandas.pydata.org](https://pandas.pydata.org/) 

安装熊猫非常简单:

```
mkdir panda-playground
cd panda-playground
pip install pandas
```

我们还可以访问大量样本数据集

## 数据集

您可以通过 URL 或本地文件以多种不同的方式在 Pandas 中加载数据集:

```
iris = pd.read_csv('https://raw.githubusercontent.com/mwaskom/seaborn-data/master/iris.csv')
```

来自 MySQL 等数据库，如下所示:

```
cn = MySQLdb.connect(host='localhost', port=3306,user='un', passwd='pass', db='zoo')
pd.read_sql('select * from animals;', con=cn)
```

我们将在以下示例中使用的数据集与视频游戏销售相关，可在此处找到:

```
[https://github.com/jph98/pandas-playground/blob/master/vgsales-12-4-2019.csv](https://github.com/jph98/pandas-playground/blob/master/vgsales-12-4-2019.csv)
```

注意:为了创建下面的表格，我最后从 Pandas 获取表格输出，并将其转换为 CSV 格式，格式为:

```
import pandas as pd
import numpy as nadf = pd.read_csv('vgsales-12-4-2019.csv')
overview = df.describe()
print(overview.to_csv())
```

然后作为 Github gist 上传，再嵌入到这里。非常好用。

## 汇总数据

开箱后，我们可以查看一些数据来了解我们正在处理的内容，前 n 名的结果集:

```
df.head()
```

Output of calling the head() function on the dataframe

注意:当你这样做的时候，你会注意到列会根据 Pandas 中的设置被截断，为了避免这个限制，你可以如下设置 display.max_columns:

```
pd.set_option('display.max_columns', None)
```

或者获取有用的数据摘要，以了解我们在做什么:

```
 df.describe()
```

Summary statistics output from the describe() function

您还可以使用以下方法从数据集中获取一组随机的样本行:

```
df.sample(n=100)
```

或者看看数据框的一些特定属性:

```
# Get the number of rows
len(df.index)# Get column names, get number of columns
df.columns 
len(df.columns)
```

好了，我们来看提取一些数据。注意:当您使用 Pandas 提取数据时，它将包含一个初始列号作为列以及所选数据。

## 子集数据(观察值和变量)

您可以通过以下方式提取一个**特定游戏**或**游戏范围**:

```
df.loc[0]
df.loc[0:3]
```

抓取第一个和第二个视频游戏的**名称**和**流派**栏

```
df.loc[0:1,['Name']]
```

或者实际上提取整个特定列，在本例中，游戏**流派，**但只有唯一值:

```
df['Genre'].unique()
```

![](img/950225a9b70baf6fc6f70248da8e6bd1.png)

## 使用表达式过滤数据

我们还可以在提取中包含表达式，这样我们就可以更加优雅和复杂地查询数据。

让我们显示由 vgchartz.com 计算的所有视频游戏评分。在这种情况下，我对所有得分在 9.2 或以上的精彩游戏感兴趣:

```
So, no surprises there, Super Mario Galaxy, Grand Theft Auto, Zelda, Metal Gear Solid and the truly excellent World of Goo (well done Kyle Gabler) all feature here in the results.
```

现在，我们可以使用以下内容按降序排列这些内容:

```
df[df.Vgchartzscore > 9.2].sort_values(['Vgchartzscore'])
```

并将结果数量限制为前五个游戏:

```
df[df.Vgchartzscore > 9.2].sort_values(['Vgchartzscore']).head(5)
```

注意，你可以修改我们加载到内存中的数据帧。然而，这个**并不**修改底层的 CSV 文件。如果我们想要将数据保存/持久保存到文件中，我们必须使用以下命令显式地将数据帧写入内存:

```
df.to_csv('outputfile.csv')
```

这将保存整个数据框。要将它限制到子集化的数据，只需用 to_csv 再次链接:

```
df[df.Vgchartzscore > 9.2].sort_values(['Vgchartzscore']).head(5).to_csv('outputfile.csv)
```

## 处理缺失数据

我们在数据工程中遇到的一件常见的事情是需要通过清理来修改传入的数据——毕竟我们不想接收坏数据，并希望在某种程度上使事情正常化。

**删除空行或空列** 删除空行(指定轴=0)或空列(指定轴=1)，如下所示:

```
df.dropna(axis=**0**, how='any', thresh=None, subset=None, inplace=False)
```

您可以通过以下方式删除特定列:

```
df.drop(['Rank'], axis=1)
```

**缺少值**
上述数据集中的某些特定于行的列具有 NaN(非数字)值。要获得至少有一列为空的**行**的计数，我们可以使用:

```
df.isnull().sum().sum()
```

或按列细分，包括:

```
df.isnull().sum()
```

我们有几个选择:

*   我们可以排除包含空列的行
*   我们可以通过标准化到已知值来清理行

例如，对于 NaN 的 Vgchartzscore，我们可以使用以下公式将这些值归一化为 0.0:

```
df['Vgchartzscore'].fillna(0, inplace=True)
```

这应该会留下大约 799 行非 0.0 值。

![](img/d7f09c1f4a8ca71950620a7536af7a28.png)

## 分组数据

有很多方法可以重塑 Pandas 中的数据。我们来看看 groupby 函数。我们可以将此与按流派分类的视频游戏数据集一起使用，以了解符合该类别的视频游戏的数量:

```
df.groupby('Genre').first()
```

我们可以通过以下方式获得每个组别的游戏数量:

```
df.groupby('Genre').size()
```

然后，我们还可以应用 sum()或其他支持的函数，如下所示:

```
df.groupby('Genre').first().sum()
```

好了，接下来我们要按平台(PS4、XBox 等)对这一类型进行分类..)来理解游戏，所以:

```
df.groupby(['Genre', 'Platform']).first()
```

注意:如果您想遍历每个组，要做一些后处理，您可以执行以下操作:

```
by_genre = df.groupby(['Genre', 'Platform'])
for name, group in by_genre:
    print(name)
```

最后，我们可以利用聚合函数对我们分组的数据进行一些计算。在下文中，我们将查看各类型平台中 Global_Sales 的平均值和总和:

```
g = df.groupby(['Genre', 'Platform'])
res = g['Global_Sales'].agg(["mean", "sum"])
```

## 结论

因此，Pandas 为您提供了丰富的功能来帮助您进行数据分析和操作。你不会想在 SQL 或原始 Python 中这样做，主要是因为它更复杂，你必须编写更多的代码，或者你受到所提供的功能的限制。聚集和分组是我选择熊猫的两个关键场景。

您可以在下面的 Github 示例库中找到本文提到的代码的 Python 3 示例:

[](https://github.com/jph98/pandas-playground/) [## jph 98/熊猫游乐场

### 我的媒体文章的熊猫乐园存储库- jph98/pandas-playground

github.com](https://github.com/jph98/pandas-playground/) 

这是关于数据工程的一系列帖子中的第一篇，我将在后续的帖子中涉及以下内容(写好后从这里链接):

*   用 matplotlib 实现熊猫数据框的可视化
*   熊猫使用自动气象站的元素进行大型数据处理——SQS 进行协调，然后构建管道进行处理等…

不要脸的塞: )

*   我通过 hwintegral 运行关于 Python 和 Pandas 的教练会议；
*   我提供数据团队指导、数据架构审查和转换(团队和技术)。

如果您的公司需要帮助，请通过以下方式联系:

[](https://www.hwintegral.com/services) [## 服务-硬件集成

### 尽职调查和技术审计在 HW Integral，我们为风险投资和私募股权公司的投资提供尽职调查服务…

www.hwintegral.com](https://www.hwintegral.com/services)