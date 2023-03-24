# 如何用熊猫分析数值型数据？

> 原文：<https://towardsdatascience.com/how-to-use-pandas-to-analyze-numeric-data-9739e0809b02?source=collection_archive---------10----------------------->

## 这比你想象的要容易得多。

Python 已经成为数据科学领域一种流行的编程语言。Pandas 是数据科学中许多受欢迎的库之一，它提供了许多帮助我们转换、分析和解释数据的强大功能。我确信已经有太多的教程和资料教你如何使用熊猫。然而，在这篇文章中，我不仅仅是教你如何使用熊猫。相反，我想用一个例子来演示如何在熊猫的帮助下正确流畅地解读你的数据。

熊猫大战熊猫

![](img/2a7c0c81065bb1c19cabb7594cc9b104.png)

Photo by [Ilona Froehlich](https://unsplash.com/@julilona?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/pandas?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

![](img/d5979bb09a0f65199bb6f983a0640a77.png)

# 在我们开始之前:

1.  **准备一个 python 环境**
2.  **从 Kaggle 下载数据集**
3.  **进口熊猫**

1.  **准备一个 python 环境**

既然我们使用 Python 来分析数据，当然首先我们需要安装 Python。对于初学者，我强烈建议您安装 Anaconda，因为 Anaconda 已经预装了一系列对数据科学非常有用的库。你不需要关心编译器或 IDE。

这里是免费下载 Anaconda 发行版的链接

[去下载](https://www.anaconda.com/distribution/)

Jupyter Notebook 是一个很好的数据分析工具，因为你可以一部分一部分地看到结果，这样你就可以知道结果是否是你预测的，你也可以立即知道你是否犯了任何错误。

![](img/fe78aa0530efbfaf60dcc071a227c94c.png)

Jupyter Notebook in Anaconda Distribution

启动 Jupyter Notebook 后，会弹出一个网页，你会看到这样的页面。您可以选择并创建存储文件的路径。在右边，有一个“新建”按钮。单击此处创建文件夹或 Python 文件。

![](img/c1fc39aba4f850f9634e8b0bab39d0f9.png)

Jupyter 笔记本提供的有用功能和快捷方式太多了。我在这里不做详细描述和解释。另外，我希望你已经掌握了一些编写 Python 代码的基础知识，因为这不是对 Python 的介绍。不管怎样，如果你有任何问题，请随意评论，我会解决你的问题。

2.从 Kaggle 下载数据集

本文将使用的数据集来自 Kaggle。该数据集包括 1985 年至 2016 年不同国家的自杀率及其社会经济信息。

[下载链接](https://www.kaggle.com/russellyates88/suicide-rates-overview-1985-to-2016)

3.进口熊猫

最后准备，进口熊猫

通常我们会给每个库一个缩写。你可以跳过这一步，但每次当你想使用任何功能时，你必须输入完整的名称。所以为什么不给它一个简短的缩写。

虽然这篇文章关注的是使用熊猫，但是我们仍然需要 numpy 的帮助，所以我们也将导入 numpy。

# 我们开始吧

![](img/8039b02f3bcdd67db788dd06258e2f9d.png)

第一步是导入我们从 Kaggle 下载的数据集。虽然您可以从任何路径导入数据，但我的建议是将所有数据和代码放在一起，这样您就不需要键入确切的路径，而只需键入文件名。下载的文件名是“master.csv”。所以我们知道数据集是 csv 文件类型。其他常见的文件类型包括 excel 文件和 json。

熊猫我们用的第一个熊猫函数是 read_csv。称为数据的数据帧是通过以下方式创建的:

```
data= pd.read_csv('master.csv')
```

我们可以使用它将 csv 文件导入到 python 中，并将其存储为数据帧。Dataframe 就像一个 excel 表格。

通常情况下，pandas 会自动解释数据集并识别所有必要的参数，以便正确导入数据集。但是，如果您发现输出不像您期望的那样，您可以手动更改参数。以下是使用 read_csv 时可以修改的一些常见参数:

1.  sep:用于分隔变量间值的分隔符。默认使用逗号'，'。您可以更改 sep，如下所示

```
data = pd.read_csv('master.csv', sep = ';')
```

2.header:通常 csv 文件的第一行显示所有变量的名称。但是，如果在 csv 文件中没有这样的名称行，最好告诉 pandas 没有标题

```
header = None
```

现在您已经导入了数据集。如前所述，您可以在运行代码后直接看到结果。让我们来看看数据集

```
data
```

![](img/cae2b9747353c51eb1a9949e20fd63d6.png)

我希望你会得到和我一样的结果。现在当你向下滚动时，你可以看到一个数字，但不是所有的记录。如果您不想显示这么多记录，您可以通过键入来显示一些顶部或底部的记录

```
data.head(5) # showing top 5 records from the dataframe# ordata.tail(5) # showing bottom 5 records from the dataframe
```

在进行任何分析之前，了解数据集是必要的，因为首先这将使您有一个简要的了解，其次这将防止您在分析时产生误解。

导入后您可能会问的第一个问题是数据集中有多少条记录。幸运的是，您可以通过一行代码得到答案。

```
data.shape
```

![](img/6fd0df4db752a30167b326dc253c7a31.png)

第一个数字(index 0，python 从零开始)显示了行数，第二个数字(index 1)显示了列数。因此数据集有 27，820 行和 12 列。

你可能会问的第二个问题是这 12 列是什么。有很多方法可以显示列名。但是我最喜欢的是用

```
data.info()
```

![](img/174bd351907a1df1e12850b090a16128.png)

info()不仅为您提供数据帧中的所有列名，还为您提供存储在该数据帧中的数据类型。此外，您可以知道数据帧中是否有丢失的值。如上所示，number 列显示了该变量中有多少个非空计数。除人类发展指数外，这一年的所有指数都是 27，820。因此，至少我们知道导入的数据集中有缺失值。

现在你想知道这个数据帧里有什么数据。例如，在那些几乎 28k 的记录中有多少个国家？你想得到所有国家的列表。您可以通过以下方式获得每列的值

```
data['country']
```

![](img/5b0c221395870ac1fcae453b51ca05b2.png)

但是这将返回没有重复数据删除的所有值。如果你想得到一个独一无二的国家，我最喜欢的方法是使用 set 函数。Set 函数将只返回一个没有重复的唯一值列表。

![](img/c9804a6e463c1172f8d3b161b4175b61.png)

总共有多少个国家？

![](img/c62bb6a4c541e78e9bffd9b8290c8fa8.png)

总共有 101 个国家。

然后你想知道每年自杀的人数。在 Excel 中，您可以通过透视表来执行此操作。熊猫也有类似的功能，叫做 pivot_table。假设你现在使用 excel 数据透视表来计算自杀人数。您将把 year 拖到行标签上，并将 sudience _ no 拖到 Values 上

![](img/c3ba8940dd89b87cac7a0dbe9ffd6074.png)

如果您熟悉 excel 数据透视表，那么请记住与 pandas pivot_table 相同的格式。

```
data.pivot_table(index='year',values='suicides_no',aggfunc='sum')
```

![](img/ad6650042b9eb12925ff9b3c284b295d.png)

哦，你也想分解成性？

```
data.pivot_table(index='year',columns='sex',values='suicides_no',aggfunc='max')
```

![](img/ff062dfd47416f079c4ef75a72b8c7a5.png)

这里的“index”表示要放在行标签中的变量；“columns”表示列；“值”表示数据透视表中的数字。就像 excel 一样，你可以计算总和、平均值、最大值或最小值。

```
data.pivot_table(index='year',columns='sex',values='suicides_no',aggfunc=['sum','mean','max','min'])
```

![](img/3bf12445fd79369dde762dee1ecadbfd.png)

您还可以对行或列执行多重索引。

```
data.pivot_table(index=['country','year'],columns='sex',values='suicides_no',aggfunc=['sum'])
```

![](img/a07ef383656498252ca79a33adb3fd77.png)

通过使用数据透视表，我们可以根据不同的变量了解数据集的一般分类。您还可以发现数据集中的任何异常，这可能有助于您进一步的分析。就像下面这样:

```
data.pivot_table(index=['year'],columns='sex',values='suicides_no',aggfunc=['sum'])
```

![](img/76db30f03854202f13bd6bce1bc88d4e.png)

2016 年发生了什么？为什么会突然下降？

现在你想检查数据集，看看 2016 年会发生什么。在 pandas 中，您需要指定要返回的行和列。这可以通过提供索引号或名称或标准来实现。现在的标准是 2016 年。您可以通过以下方式获得结果

```
data[data['year']==2016]
```

如何解读这行代码？

有两部分。内部分“数据['年份']==2016”是需要满足的条件。这里表示数据中的年份列必须是 2016 年。小心有两个等号，代表比较。类似地，你可以有一个更大、更小或不相等的条件

```
data['year'] >= 2016 # larger than or equal to 
data['year']!=2016 # not equal to
```

条件多，没问题。但是，如果你想满足全部或其中一个条件，请确保你明确说明。

```
# year must be larger than 2010 and smaller than 2015
(data['year'] > 2010) & (data['year']<2015)# year can be larger than 2010 or smaller than 2000
(data['year'] > 2010) | (data['year']<2000)
```

外部部分“data[…]”表示返回内部部分中满足条件的所有行。回到示例，我们希望返回年份等于 2016 年的所有行。

![](img/bfec0cc7df5a5565fce11b0328e5a171.png)

似乎没问题。但如果算上 2016 年和 2015 年的国家数量，问题就出现了。

现在，我们想从数据集中获取列 country，其年份必须是 2016 年。最简单的方法是用

```
data[data['year']==2016]['country']
```

![](img/5b2dccc75ba0a80e679783f906b0df15.png)

然后再次使用 set 和 len 函数计算 2016 年的国家数量。

```
len(set(data[data['year']==2016]['country']))
```

在 2015 年做同样的事情。我们来了

![](img/82016cc1e8b0ecc2b6d99027d305e32f.png)

2016 年的数据显然不完整，因此我们不应将 2016 年与其他年份进行比较。所以我们最好删除 2016 年的所有记录。

如上所述，我们可以应用一个条件来过滤 2016 年的记录，如下所示:

```
data2= data[data['year']!=2016]
```

现在，我们创建另一个名为“data2”的数据帧，用于进一步分析。

找到哪些变量与数据集相关至关重要，因为这可以帮助您发现哪些变量会导致结果的变化。例如，在数据集中有一些分类变量:

国家、年份、性别和世代

选一个简单的，比如性。你可以比较一下每种性别的总自杀人数。如果差异很大，那么性别是导致自杀差异的一个因素

当然，我们可以使用 pivot_table 函数来实现。但是我在这里介绍另一个功能，groupby。如果你有使用 SQL 的经验，我相信你对 group by 很熟悉。

```
data2.groupby('sex').agg({'suicides_no':'sum'})
```

就 SQL 而言，上述语句相当于:

```
select sex, sum(suicides_no) from data2 group by sex
```

Python 将首先根据“性别”列中的值对所有记录进行分组。之后 python 会计算每组‘性’的自杀总数。

![](img/f315627514d5f352fdecb6f84f654ea4.png)

男性自杀率远高于女性自杀率。这时，我们可以相当肯定，性别是影响自杀人数的一个因素。

groupby 的高明之处在于，您可以再次将数据集分成更多级别。请记住，当您包括多个级别时，您需要使用方括号[ ]向熊猫提供一个列表。

```
data2.groupby(['sex','generation']).agg({'suicides_no':'sum'})
```

![](img/8ffc45e68458108951ed7d2c45d2cc94.png)

此外，就像 SQL 一样，您可以在 groupby 函数中执行多个聚合。

```
data2.groupby(['sex']).agg({'suicides_no':['sum','max']})
```

![](img/97da14811443f9d04c62af40b8ee2808.png)

或者

```
data2.groupby(['sex']).agg({'suicides_no':'sum', 'population':'mean'})
```

![](img/c5a20ee074c6f0445f1d9ad6aa3a9fbe.png)

现在我们知道性是一个因素。那年龄呢？

```
data2.groupby(['age']).agg({'suicides_no':'sum'})
```

![](img/f072c570185e53fd1f97e785b8729ca4.png)

所以自杀最流行的年龄范围是 35-54 岁。对吗？

不完全是。不像性是一半一半，每个年龄组的人口都不均匀。这是一种常见误解，因为不同的类别有不同的人口规模或基数。

因此，为了更好地进行比较，我们应该更好地比较每个总人口的自杀人数。为了在熊猫中做到这一点，首先创建另一个数据框架来存储每个年龄层的自杀人数和人口。然后计算一个除法，得到每个年龄段的平均自杀人数。

```
# create a dataframe called data_age for age groupby 
data_age = data2.groupby([‘age’]).agg(
{‘suicides_no’:’sum’,’population’:’sum’})# calculate average of suicide number for each age group 
data_age[‘suicides_no’]/data_age[‘population’]
```

代码的第一部分与前面类似。第二部分是执行除法。python 的好处是可以将计算作为一整列来执行，而不是像 excel 那样逐个单元格地执行。所以我们把自杀人数一栏和人口一栏分开。

![](img/bbe111d5fcf74fa2249ccc6c95a10e0d.png)

现在结果不同了。随着年龄的增长，自杀的情况变得更糟。

这时，你可能会说排名不明确，你想根据平均自杀人数来排列结果。那么是时候引入 sort_values 函数了。

```
(data_age['suicides_no']/data_age['population']).sort_values()
```

因为这里只有一列，所以我们不需要指定按哪一列排序。但是，如果数据帧中有多列，则必须指定哪一列。此外，默认顺序是升序，您可以通过在括号内添加 ascending = False 来更改它。就像下面这样:

```
(data_age['suicides_no']/data_age['population']).sort_values(ascending=False)
```

![](img/a5e6fad5194cff10af4c122d94089ed0.png)

现在你可以很容易地看到随着年龄增长的趋势。

我们已经研究了性别和年龄的影响。现在我们转到乡下。我们可以找到哪个国家更流行自杀。

第一步是创建一个表格，包括各国自杀人数和人口。这可以通过使用 groupby 函数来完成。

```
data_country = data2.groupby(['country']).agg({'suicides_no':'sum','population':'sum'})
```

![](img/31175217e2325970e550f34707ca8379.png)

在这里，我们不关心是否所有国家都有相同数量的记录，因为我们正在计算一段时间内的平均数。

然后我们用自杀人数除以人口来计算平均自杀人数。

```
data_country['average_suicide'] = data_country['suicides_no']/data_country['population']
```

这里我们创建了一个名为“average_suicide”的新列，它将存储除法结果。稍后使用 sort_values 函数以降序获得结果。我们可以使用 head 函数只显示前 N 个国家，而不是显示所有国家

```
data_country.sort_values(by='average_suicide',ascending=False).head(10)
```

![](img/76e0bdb652f59b3adc4cf014ae9e7288.png)

从结果中，我们可以看到大多数国家在东欧。那么你可以说地理位置也是影响自杀的一个指标。

今天到此为止。我希望你不仅能学会如何使用 Python，还能学会如何处理数据集，从而得出更好、更准确的结论。

欢迎评论，这样我就知道如何改进和写一个更好的博客。如果你喜欢，就鼓掌，分享给和你一样同样有需要的人。下次见。