# MachineX:使用 NumPy 和 Pandas 进行数据清理

> 原文：<https://towardsdatascience.com/machinex-data-cleaning-in-python-ad6fa5ca109e?source=collection_archive---------11----------------------->

![](img/c905026c37637c843ffcd65a0546ec21.png)

在这篇博客中，我们将学习如何用 NumPy 和 Pandas 进行**数据清理。**

大多数数据科学家仅将 20%的时间用于实际数据分析，80%的时间用于查找、清理和重组海量数据，这是一种低效的数据策略。

数据科学家最初被雇佣的原因是开发算法和建立机器学习模型，而这些通常是他们最喜欢的工作部分。然而，在当今的大多数公司中，数据科学家 80%的宝贵时间都花在了查找、清理和重新组织海量数据上。

如果你刚刚进入这个领域或者计划在这个领域发展你的职业生涯，能够处理杂乱的数据是很重要的，无论这意味着缺少值、不一致的格式、畸形的记录还是无意义的离群值。

在本教程中，我们将使用 python 的 NumPy 和 Pandas 库来清理数据，并看看我们可以在多少方面使用它们。

**数据集**

我们正在使用一些常见的数据集来探索我们的知识，每个数据集都符合我们正在使用的清洗技术，因此您也可以下载自己的数据集并按照说明进行操作。

以下是我们将使用的不同数据集，您可以通过以下链接下载这些数据集，也可以从 [githubRepo](https://github.com/shubham769/Python-data-cleaning) 直接下载:

*   BL-Flickr-Images-Book.csv —包含大英图书馆书籍信息的 csv 文件
*   [university_towns.txt](https://github.com/shubham769/Python-data-cleaning/blob/master/olympics.csv) —包含美国各州大学城名称的文本文件
*   [olympics.csv](https://github.com/shubham769/Python-data-cleaning/blob/master/university_towns.txt) —总结所有国家参加夏季和冬季奥运会的 csv 文件

**注意**:我正在使用 Jupyter 笔记本，推荐使用。

让我们导入所需的模块并开始吧！

```
>>> import pandas as pd
>>> import numpy as np
```

现在，让我们在这两个模块的帮助下开始清理数据。我们导入这两个模块，并指定 pd 和 np 作为对象来使用它们。

**删除数据帧中不需要的列**

为了从数据帧中删除列，Pandas 使用了“ [drop](https://pandas.pydata.org/pandas-docs/version/0.21/generated/pandas.DataFrame.drop.html) ”函数。Pandas 提供了一种使用`[drop()](https://pandas.pydata.org/pandas-docs/stable/generated/pandas.DataFrame.drop.html)`函数从`DataFrame`中删除不需要的列或行的简便方法。让我们看一个简单的例子，我们从一个`DataFrame`中删除了一些列。让我们从 CSV 文件“BL-Flickr-Images-Book.csv”中创建一个`DataFrame`。在下面的例子中，我们传递了一个到`pd.read_csv`的相对路径，这意味着所有的数据集都在我们当前工作目录中名为`Datasets`的文件夹中:

```
>>> dataF = pd.read_csv('Datasets/BL-Flickr-Images-Book.csv')
>>> dataF.head()Identifier             Edition Statement      Place of Publication  \
0         206                           NaN                    London
1         216                           NaN  London; Virtue & Yorston
2         218                           NaN                    London
3         472                           NaN                    London
4         480  A new edition, revised, etc.                    LondonDate of Publication              Publisher  \
0         1879 [1878]       S. Tinsley & Co.
1                1868           Virtue & Co.
2                1869  Bradbury, Evans & Co.
3                1851          James Darling
4                1857   Wertheim & MacintoshTitle     Author  \
0                  Walter Forbes. [A novel.] By A. A      A. A.
1  All for Greed. [A novel. The dedication signed...  A., A. A.
2  Love the Avenger. By the author of “All for Gr...  A., A. A.
3  Welsh Sketches, chiefly ecclesiastical, to the...  A., E. S.
4  [The World in which I live, and my place in it...  A., E. S.Contributors  Corporate Author  \
0                               FORBES, Walter.               NaN
1  BLAZE DE BURY, Marie Pauline Rose - Baroness               NaN
2  BLAZE DE BURY, Marie Pauline Rose - Baroness               NaN
3                   Appleyard, Ernest Silvanus.               NaN
4                           BROOME, John Henry.               NaNCorporate Contributors Former owner  Engraver Issuance type  \
0                     NaN          NaN       NaN   monographic
1                     NaN          NaN       NaN   monographic
2                     NaN          NaN       NaN   monographic
3                     NaN          NaN       NaN   monographic
4                     NaN          NaN       NaN   monographicFlickr URL  \
0  http://www.flickr.com/photos/britishlibrary/ta...
1  http://www.flickr.com/photos/britishlibrary/ta...
2  http://www.flickr.com/photos/britishlibrary/ta...
3  http://www.flickr.com/photos/britishlibrary/ta...
4  [http://www.flickr.com/photos/britishlibrary/ta...](http://www.flickr.com/photos/britishlibrary/ta...)Shelfmarks
0    British Library HMNTS 12641.b.30.
1    British Library HMNTS 12626.cc.2.
2    British Library HMNTS 12625.dd.1.
3    British Library HMNTS 10369.bbb.15.
4    British Library HMNTS 9007.d.28.
```

当我们使用该方法查看前五个条目时，`head()`我们可以看到一些列提供了对图书馆有帮助的辅助信息，但并没有很好地描述书籍本身:`Edition StatementCorporate Author`、`Corporate Contributors`、`Former owner`、`Engraver`、`Issuance type`和`Shelfmarks`。

## 删除列

我们可以通过以下方式删除这些列:

```
>>> data_to_drop = ['Edition Statement',
...            'Corporate Author',
...            'Corporate Contributors',
...            'Former owner',
...            'Engraver',
...            'Contributors',
...            'Issuance type',
...            'Shelfmarks']>>> dataF.drop(data_to_drop, inplace=True, axis=1)
```

上面，我们定义了一个列表，其中包含了我们想要删除的所有列的名称。接下来，我们调用对象上的`drop()`函数，将`inplace`参数作为`True`传入，将`axis`参数作为`1`传入。这告诉 Pandas 我们希望直接在我们的对象中进行更改，并且它应该在对象的列中寻找要删除的值。

当我们再次检查`DataFrame`时，我们会看到不需要的列已被删除:

```
>>> dataF.head()
   Identifier      Place of Publication Date of Publication  \
0         206                    London         1879 [1878]
1         216  London; Virtue & Yorston                1868
2         218                    London                1869
3         472                    London                1851
4         480                    London                1857Publisher                                              Title  \
0       S. Tinsley & Co.                  Walter Forbes. [A novel.] By A. A
1           Virtue & Co.  All for Greed. [A novel. The dedication signed...
2  Bradbury, Evans & Co.  Love the Avenger. By the author of “All for Gr...
3          James Darling  Welsh Sketches, chiefly ecclesiastical, to the...
4   Wertheim & Macintosh  [The World in which I live, and my place in it...Author                                         Flickr URL
0      A. A.  http://www.flickr.com/photos/britishlibrary/ta...
1  A., A. A.  http://www.flickr.com/photos/britishlibrary/ta...
2  A., A. A.  http://www.flickr.com/photos/britishlibrary/ta...
3  A., E. S.  http://www.flickr.com/photos/britishlibrary/ta...
4  A., E. S.  [http://www.flickr.com/photos/britishlibrary/ta...](http://www.flickr.com/photos/britishlibrary/ta...)
```

# 更改数据帧的索引

Pandas `Index`扩展了 NumPy 数组的功能，允许更多的切片和标记。在许多情况下，使用数据的唯一值标识字段作为索引是很有帮助的。

例如，在上一节使用的数据集中，可以预计当图书管理员搜索记录时，他们可能会输入一本书的唯一标识符(值在`Identifier`列中):

```
>>> dataF['Identifier'].is_unique
True
```

让我们使用`set_index`用这个列替换现有的索引:

```
>>> dataF = dataF.set_index('Identifier')
>>> dataF.head()
                Place of Publication Date of Publication  \
206                           London         1879 [1878]
216         London; Virtue & Yorston                1868
218                           London                1869
472                           London                1851
480                           London                1857Publisher  \
206              S. Tinsley & Co.
216                  Virtue & Co.
218         Bradbury, Evans & Co.
472                 James Darling
480          Wertheim & MacintoshTitle     Author  \
206                         Walter Forbes. [A novel.] By A. A      A. A.
216         All for Greed. [A novel. The dedication signed...  A., A. A.
218         Love the Avenger. By the author of “All for Gr...  A., A. A.
472         Welsh Sketches, chiefly ecclesiastical, to the...  A., E. S.
480         [The World in which I live, and my place in it...  A., E. S.Flickr URL
206         http://www.flickr.com/photos/britishlibrary/ta...
216         http://www.flickr.com/photos/britishlibrary/ta...
218         http://www.flickr.com/photos/britishlibrary/ta...
472         http://www.flickr.com/photos/britishlibrary/ta...
480         [http://www.flickr.com/photos/britishlibrary/ta...](http://www.flickr.com/photos/britishlibrary/ta...)
```

每个记录都可以用`loc[]`访问，它允许我们做*基于标签的索引*，这是一个行或记录的标签，不考虑它的位置:

```
>>> dataF.loc[206]
Place of Publication                                               London
Date of Publication                                           1879 [1878]
Publisher                                                S. Tinsley & Co.
Title                                   Walter Forbes. [A novel.] By A. A
Author                                                              A. A.
Flickr URL              http://www.flickr.com/photos/britishlibrary/ta...
Name: 206, dtype: object
```

换句话说，206 是索引的第一个标签。要通过*位置*访问它，我们可以使用`iloc[0]`，它执行基于位置的索引。

# 整理数据中的字段

到目前为止，我们已经删除了不必要的列，并将`DataFrame`的索引改为更合理的内容。在这一节中，我们将清理特定的列，并将它们转换为统一的格式，以便更好地理解数据集并增强一致性。特别是，我们将清洁`Date of Publication`和`Place of Publication`。

经检查，目前所有的数据类型都是`object` [dtype](http://pandas.pydata.org/pandas-docs/stable/basics.html#dtypes) ，这大致类似于原生 Python 中的`str`。

它封装了任何不能作为数字或分类数据的字段。这是有意义的，因为我们处理的数据最初是一堆杂乱的字符串:

```
>>> dataF.get_dtype_counts() 
object    6
```

强制使用数字值有意义的一个字段是出版日期，这样我们可以在以后进行计算:

```
>>> dataF.loc[1905:, 'Date of Publication'].head(10) 
Identifier 
1905           1888 
1929    1839, 38-54 
2836        [1897?] 
2854           1865 
2956        1860-63 
2957           1873 
3017           1866 
3131           1899 
4598           1814 
4884           1820 
Name: Date of Publication, dtype: object
```

一本书只能有一个出版日期。因此，我们需要做到以下几点:

*   删除方括号中的多余日期:1879 [1878]
*   将日期范围转换为它们的“开始日期”，如果有的话:1860–63；1839, 38–54
*   完全去掉我们不确定的日期，用 NumPy 的`NaN`:【1897？]
*   将字符串`nan`转换为 NumPy 的`NaN`值

综合这些模式，我们实际上可以利用一个正则表达式来提取出版年份:

```
regex = r'^(\d{4})'
```

上面的正则表达式是查找一个字符串开头的任意四位数字。上面是一个*原始字符串*(意味着反斜杠不再是转义字符)，这是正则表达式的标准做法。

`\d`代表任意数字，`{4}`重复此规则四次。`^`字符匹配一个字符串的开头，圆括号表示一个捕获组，这向 Pandas 发出信号，表明我们想要提取正则表达式的这一部分。(我们希望`^`避免`[`从管柱开始的情况。)

让我们看看在数据集上运行这个正则表达式会发生什么:

```
>>> extr = dataF['Date of Publication'].str.extract(r'^(\d{4})', expand=False) 
>>> extr.head() 
Identifier 
206    1879 
216    1868 
218    1869 
472    1851 
480    1857 
Name: Date of Publication, dtype: object
```

不熟悉 regex？你可以在 regex101.com[查看上面](https://regex101.com/r/3AJ1Pv/1)的表达式，并在 Python 正则表达式 [HOWTOMakeExpressions](https://docs.python.org/3.6/howto/regex.html) 阅读更多内容。

从技术上讲，这个列仍然有`object` dtype，但是我们可以很容易地用`pd.to_numeric`得到它的数字版本:

```
>>> dataF['Date of Publication'] = pd.to_numeric(extr) 
>>> dataF['Date of Publication'].dtype dtype('float64')
```

这导致大约十分之一的值丢失，对于现在能够对剩余的有效值进行计算来说，这是一个很小的代价:

```
>>> dataF['Date of Publication'].isnull().sum() / len(dataF) 0.11717147339205986
```

太好了！搞定了。！！！

# 使用 applymap 函数清理整个数据集

在某些情况下，将定制函数应用于数据帧的每个单元格或元素会很有帮助。Pandas `.applymap()`方法类似于内置的`map()`函数，只是将一个函数应用于`DataFrame`中的所有元素。

我们将从“university_towns.txt”文件中创建一个`DataFrame`:

```
>>> head Datasets/univerisity_towns.txt
Alabama[edit]
Auburn (Auburn University)[1]
Florence (University of North Alabama)
Jacksonville (Jacksonville State University)[2]
Livingston (University of West Alabama)[2]
Montevallo (University of Montevallo)[2]
Troy (Troy University)[2]
Tuscaloosa (University of Alabama, Stillman College, Shelton State)[3][4]
Tuskegee (Tuskegee University)[5]
Alaska[edit]
```

我们看到，我们有周期性的州名，后跟该州的大学城:`StateA TownA1 TownA2 StateB TownB1 TownB2...`。如果我们观察状态名在文件中的书写方式，我们会发现所有的状态名中都有“[edit]”子字符串。

我们可以通过创建一个 `*(state, city)*` *元组*的*列表并将该列表包装在`DataFrame`中来利用这种模式*

```
>>> university_towns = []
>>> with open('Datasets/university_towns.txt') as file:
...     for line in file:
...         if '[edit]' in line:
...             # Remember this `state` until the next is found
...             state = line
...         else:
...             # Otherwise, we have a city; keep `state` as last-seen
...             university_towns.append((state, line))>>> university_towns[:5]
[('Alabama[edit]\n', 'Auburn (Auburn University)[1]\n'),
 ('Alabama[edit]\n', 'Florence (University of North Alabama)\n'),
 ('Alabama[edit]\n', 'Jacksonville (Jacksonville State University)[2]\n'),
 ('Alabama[edit]\n', 'Livingston (University of West Alabama)[2]\n'),
 ('Alabama[edit]\n', 'Montevallo (University of Montevallo)[2]\n')]
```

我们可以将这个列表包装在一个 DataFrame 中，并将列设置为“State”和“RegionName”。Pandas 将获取列表中的每个元素，并将`State`设置为左边的值，将`RegionName`设置为右边的值。

```
>>> towns_dataF = pd.DataFrame(university_towns,
...                         columns=['State', 'RegionName'])>>> towns_dataF.head()
 State                                         RegionName
0  Alabama[edit]\n                    Auburn (Auburn University)[1]\n
1  Alabama[edit]\n           Florence (University of North Alabama)\n
2  Alabama[edit]\n  Jacksonville (Jacksonville State University)[2]\n
3  Alabama[edit]\n       Livingston (University of West Alabama)[2]\n
4  Alabama[edit]\n         Montevallo (University of Montevallo)[2]\n
```

# `applymap()`

虽然我们可以在上面的 for 循环中清理这些字符串，但 Pandas 让它变得很容易。我们只需要州名和镇名，其他的都可以去掉。虽然我们可以在这里再次使用 Pandas 的`.str()`方法，但是我们也可以使用`applymap()`将 Python callable 映射到 DataFrame 的每个元素。

我们一直在使用术语*元素*，但是它到底是什么意思呢？考虑以下“玩具”数据帧:

```
0           1
0    Mock     Dataset
1  Python     Pandas
2    Real     Python
3   NumPy     Clean
```

在这个例子中，每个单元格(' Mock '，' Dataset '，' Python '，' Pandas '等)。)是一个元素。因此，`applymap()`将独立地对其中的每一个应用一个函数。让我们来定义这个函数:

```
>>> def get_citystate(item): 
...     if ' (' in item: 
...         return item[:item.find(' (')] 
...     elif '[' in item: 
...         return item[:item.find('[')] 
...     else: 
...         return item
```

Pandas 的`.applymap()`只有一个参数，它是应该应用于每个元素的函数(可调用的):

```
>>> towns_dataF =  towns_dataF.applymap(get_citystate)
```

首先，我们定义一个 Python 函数，它将来自`DataFrame`的一个元素作为它的参数。在函数内部，执行检查以确定元素中是否有`(`或`[`。

函数根据检查相应地返回值。最后，在我们的对象上调用`applymap()`函数。现在数据框架更加整洁了:

```
>>> towns_dataF.head()      
State    RegionName 
0  Alabama        Auburn 
1  Alabama      Florence 
2  Alabama  Jacksonville 
3  Alabama    Livingston 
4  Alabama    Montevallo
```

方法`applymap()`从 DataFrame 中取出每个元素，将其传递给函数，原始值被返回值替换。就这么简单！

快乐学习！！！！！！！！！

请查看下面的链接，找到对您的 Python 数据科学之旅有所帮助的其他资源:

*   熊猫[文档](https://pandas.pydata.org/pandas-docs/stable/index.html)
*   NumPy [文档](https://docs.scipy.org/doc/numpy/reference/)
*   熊猫的创造者韦斯·麦金尼[用于数据分析的 Python](https://realpython.com/asins/1491957662/)
*   [数据科学培训师兼顾问 Ted Petrou 撰写的熊猫食谱](https://realpython.com/asins/B06W2LXLQK/)

**参考资料:**
[realPython.com](https://realpython.com/python-data-cleaning-numpy-pandas/#tidying-up-fields-in-the-data)
[data quest . io](https://www.dataquest.io/blog/data-cleaning-with-python/)

**注**:原发布于->[https://blog.knoldus.com/machine-x-data-cleaning-in-python/](https://blog.knoldus.com/machine-x-data-cleaning-in-python/)