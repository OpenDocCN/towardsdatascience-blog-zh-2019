# 使用 python 探索数据。

> 原文：<https://towardsdatascience.com/exploring-the-data-using-python-47c4bc7b8fa2?source=collection_archive---------7----------------------->

## 在本教程中，我们将使用探索性数据分析方法来总结和分析 cars 数据集的主要特征。

![](img/924f93ad5515fe0c3d3146f2abf176cd.png)

Image Credits: [Statistical Agency](https://www.statistika.co/index.php/sr/ourservices/our-experienced-team-provide/241-exploratory-data-analysis-eda)

让我们了解如何使用 python 探索数据，然后在下一个教程中基于这些数据构建一个机器学习模型。本教程的完整代码可以在我的 [GitHub 库](https://github.com/Tanu-N-Prabhu/Python/blob/master/Exploratory%20Data%20Analysis/Exploratory_data_Analysis_2.ipynb)中找到。

# 1)选择一个数据集

我从 Kaggle 选择了一个名为“汽车”的数据集，这个数据集的作者是 Lilit Janughazyan [1]。从童年开始，我就对汽车感兴趣和着迷。我还记得我曾经有一本书，里面贴着不同汽车的图片和说明书。我更了解最新的汽车及其规格。我更像是一张规格表，记着几乎所有关于汽车的信息，向人们解释市场上不同的汽车。当我年轻的时候，我的梦想是根据汽车的规格来预测它的价格。在这种兴趣的帮助下，我想在这次作业中选择一个基于汽车的数据集。我想实现我的梦想，创建一个模型，该模型将输入汽车的规格，如马力、气缸或发动机大小，然后该模型将根据这些规格预测汽车的价格。数据集可以在这里找到:[汽车数据集](https://www.kaggle.com/ljanjughazyan/cars1)

我选择数据集而不是其他数据集的主要原因是，在 Kaggle 中投票最多的类别下有近 110 个关于汽车的数据集(投票最多意味着 Kaggle 上可用的最好和最著名的数据集集合)，几乎所有这些数据集都缺少一个或另一个特征。例如，数据集“汽车数据集”[2]具有汽车的大多数特征，但其中没有价格特征，根据我的兴趣，这是最重要的特征。因此，我花了很多时间筛选出许多数据集，然后我总结出“汽车”数据集，因为这个数据集几乎具有汽车的每一个重要特征，如马力、建议零售价、发票、气缸、发动机尺寸等等。因为这些良好的特征，这是我选择这个数据集而不是 Kaggle 中其他数据集的主要原因。

这个数据集直接以 CSV(逗号分隔值)格式存储在 Kaggle 上。我不需要执行任何操作就可以将数据转换成格式。由于数据已经是 CSV 格式，导入数据集只需很少的工作，我所要做的只是下载、读取 CSV 数据并将其存储在 pandas 数据框中，为此我必须导入 pandas 库。

# 2)获取数据

为了获取数据集或将数据集加载到笔记本中，我所做的只是一个微不足道的步骤。在笔记本左侧的 Google Colab 中，您会发现一个“>”(大于号)。点击它，你会发现一个有三个选项的标签，你可以从中选择文件。然后，您可以在上传选项的帮助下轻松上传数据集。无需安装到 google drive 或使用任何特定的库，只需上传数据集，您的工作就完成了。这就是我如何把数据集放进笔记本的

# 3)擦洗和格式化

**将数据格式化成数据帧**

因为数据集已经是 CSV 格式。我所要做的只是将数据格式化成熊猫数据框。这是通过导入 pandas 库使用名为(read_csv)的 pandas 数据框方法完成的。read_csv 数据帧方法通过将文件名作为参数传递来使用。然后通过执行这个，它将 CSV 文件转换成一个组织整齐的 pandas 数据帧格式。

```
**# Importing the required libraries**import pandas as pd 
import numpy as np
import seaborn as sns #visualisation
import matplotlib.pyplot as plt #visualisation
%matplotlib inline 
sns.set(color_codes=True)**# Loading the CSV file into a pandas dataframe.**df = pd.read_csv(“CARS.csv”)
df.head(5)
```

![](img/e5f684c2b656227af511ed463f777153.png)

**确定实例和特征的数量。**

这个数据集有 428 个实例和 15 个特征，也称为行和列。此处的实例代表不同的汽车品牌，如宝马、奔驰、奥迪和 35 个其他品牌，特征代表品牌、型号、类型、产地、驱动系统、建议零售价、发票、发动机尺寸、气缸、马力、MPG-城市、MPG-公路、重量、轴距和汽车长度。

**去除不相关的特征。**

我将从该数据集中移除一些要素，如传动系、模型、发票、类型和来源。因为这些特征无助于价格的预测。截至目前，我将删除传动系，传动系将不支持预测汽车价格，因为该数据集中的大多数汽车是前轮驱动(52.8%)，其余是后轮和全轮驱动。

类似地，型号、类型和产地是不相关的，在这种情况下也不需要，重要的是品牌而不是汽车的型号，当谈到汽车的类型时，大多数汽车都是轿车类型，我保留了汽车的重量和长度特征，在这种情况下，我可以很容易地确定它是 SUV、轿车还是卡车。我还将删除汽车的发票功能，因为我有 MSRP 作为价格，我不需要发票，因为有任何一种类型的汽车价格更有意义，它可以防止导致模糊的结果(因为 MSRP 和发票都非常密切相关，你不能预测给定发票的 MSRP)。最后，汽车的起源与预测率无关，所以我不得不删除它，大部分汽车来自欧洲。

```
**# Removing irrelevant features**df = df.drop([‘Model’,’DriveTrain’,’Invoice’, ‘Origin’, ‘Type’], axis=1)
df.head(5)
```

![](img/ef85987301cf6587d987abec0cc0ecc3.png)

# 4)探索性数据分析

**使用 info()识别数据类型**

为了识别数据类型，我使用了 info 方法。info 方法打印数据框中数据的摘要及其数据类型。这里有 428 个条目(0–427 行)。移除不相关列之后的数据帧包括 10 列。在这里，品牌、MSRP 是对象类型，而发动机尺寸和气缸是浮动类型，马力、MPG_City、MPG_Highway、重量、轴距和长度是整数类型。因此，数据帧中存在 2 种对象类型、2 种浮点类型和 6 种整数类型的数据。

```
**# To identify the type of data**df.info()**<class ‘pandas.core.frame.DataFrame’>
RangeIndex: 428 entries, 0 to 427 
Data columns (total 10 columns): 
Make 428 non-null object 
MSRP 428 non-null object 
EngineSize 428 non-null float64 
Cylinders 426 non-null float64 
Horsepower 428 non-null int64 
MPG_City 428 non-null int64 
MPG_Highway 428 non-null int64
Weight 428 non-null int64 
Wheelbase 428 non-null int64 
Length 428 non-null int64 
dtypes: float64(2), int64(6), object(2) 
memory usage: 33.5+ KB**
```

**寻找数据帧的尺寸**

为了获得数据框的行数和列数，我使用了 shape 方法。shape 方法获取数据框的行数和列数。这里有 428 行和 10 列。因此 shape 方法返回(428，10)。为了找到数据帧的尺寸，我使用了 ndim (dimension)方法。此方法打印数据框的尺寸。这里，整个数据帧是二维的(行和列)。

```
**# Getting the number of instances and features**df.shape**(428, 10)**# Getting the dimensions of the data frame
df.ndim**2**
```

**查找重复数据。**

这是在数据集上执行的一件方便的事情，因为数据集中可能经常有重复或冗余的数据，为了消除这种情况，我使用了 MSRP 作为参考，这样就不会有超过两个相同的汽车 MSRP 价格，这表明很少有数据是冗余的，因为汽车的价格永远不会非常准确地匹配。因此，在删除重复数据之前，有 428 行，删除之后有 410 行，这意味着有 18 个重复数据。

```
df = df.drop_duplicates(subset=’MSRP’, keep=’first’)
df.count()**Make 410 
MSRP 410 
EngineSize 410 
Cylinders 408 
Horsepower 410 
MPG_City 410 
MPG_Highway 410 
Weight 410 
Wheelbase 410 
Length 410 
dtype: int64**
```

**查找缺失值或空值。**

很多时候，数据集中可能会有很多缺失值。有几种方法可以处理这种情况，我们可以删除这些值，或者用该列的平均值填充这些值。这里，2 个条目在圆柱体特征中有 N/A。这可以通过使用 is_null()方法找到，该方法返回数据框中的空值或缺失值。因此，我没有删除这两个条目，而是用圆柱体列的平均值填充这些值，它们的值都是 6.0。我在查看数据集的第一行和最后几行时发现了这一点。我认为与其删除这是一个好方法，因为每一项数据都是至关重要的。我发现在 Cylinders 特性中有两个存储为 NaN(不是数字)的值。所以我用提到他们索引的切片技术把他们打印出来。

```
**# Finding the null values**print(df.isnull().sum())**Make 0 
MSRP 0 
EngineSize 0 
Cylinders 2 
Horsepower 0 
MPG_City 0 
MPG_Highway 0 
Weight 0 
Wheelbase 0 
Length 0 
dtype: int64****# Printing the null value rows**df[240:242]
```

![](img/27dee8893c934ea46928cc05f491406d.png)

```
**# Filling the rows with the mean of the column**val = df[‘Cylinders’].mean()
df[‘Cylinders’][247] = round(val)val = df[‘Cylinders’].mean()
df[‘Cylinders’][248]= round(val)
```

**将对象值转换为整数类型。**

在查看数据时，MSRP 被存储为对象类型。这是一个严重的问题，因为不可能在图形上绘制这些值，因为在绘制图形期间，所有值必须是整数数据类型是一个基本要求。作者以不同的格式存储了 MSRP(36，000 美元),所以我必须删除格式，然后将它们转换为整数。

```
**# Removing the formatting**df[‘MSRP’] = [x.replace(‘$’, ‘’) for x in df[‘MSRP’]] 
df[‘MSRP’] = [x.replace(‘,’, ‘’) for x in df[‘MSRP’]]df[‘MSRP’]=pd.to_numeric(df[‘MSRP’],errors=’coerce’)
```

**检测异常值**

离群点是不同于其他点的点或点集。有时它们会很高或很低。检测并移除异常值通常是个好主意。因为离群值是导致模型不太精确的主要原因之一。因此，移除它们是个好主意。我将执行 IQR 评分技术来检测和删除异常值。通常，使用箱线图可以看到异常值。下图是 MSRP 的方框图。在图中，你可以发现一些点在方框之外，它们不是别的，就是异常值。我在参考资料部分[3]提到了上述数据科学文章中的异常值技术。

```
sns.boxplot(x=df[‘MSRP’])
```

![](img/f90deb855f7d51a76ca1bc892decb715.png)

```
Q1 = df.quantile(0.25)
Q3 = df.quantile(0.75)
IQR = Q3 — Q1
print(IQR)**MSRP           19086.50 
EngineSize         1.55 
Cylinders          2.00 
Horsepower        85.00 
MPG_City           4.00 
MPG_Highway        5.00 
Weight           872.25 
Wheelbase          9.00 
Length            16.00 
dtype: float6**df = df[~((df < (Q1–1.5 * IQR)) |(df > (Q3 + 1.5 * IQR))).any(axis=1)]
```

使用该技术后，如下图所示，MSRP 盒图不包含异常点，这是一个很大的改进。以前有超过 15 个点的异常值，现在我已经删除了这些异常值。

```
sns.boxplot(x=df[‘MSRP’])
```

![](img/e901f6ae27650bc61d5a3119dfd35b49.png)

**执行 5 个数字汇总(最小值、下四分位数、中值、上四分位数、最大值**

下一步是对数字数据执行 5 个数字的汇总。如前所述，在这种情况下，数字数据是 MSRP、发动机尺寸、马力、气缸、马力、MPG_City、MPG_Highway、重量、轴距和长度。五位数总和包括最小值、下四分位数、中值、上四分位数和最大值，所有这些值都可以通过使用 describe 方法获得。

```
df.describe()
```

![](img/90c46e9f0049cf6a46f7a1774bb44dbf.png)

**把不同的特征彼此对立起来。**

**热图**

热图是一种图表，当我们需要找到因变量时，它是必不可少的。使用热图可以找到特征之间相关性的最佳方法之一。如下所示，价格特征(MSRP)与马力的相关性高达 83%,这一点非常重要，因为变量之间的关系越密切，模型就越精确。这就是如何使用热图找到特征之间的相关性。在热图的帮助下，我可以在构建模型时使用这些相关功能。

```
**# Plotting a heat map**plt.figure(figsize=(10,5))
c= df.corr()
sns.heatmap(c,cmap=”BrBG”,annot=True)
```

![](img/8d9980c76c903e7b90f7a6f67eca0138.png)

**两个相关变量之间的散点图**

我知道功能，尤其是建议零售价和马力更相关。因为我有两个相关的变量，所以我用散点图来显示它们的关系。这里绘制了马力和 MSRP 之间的散点图，如下所示。根据下面给出的图表，我们可以在建模过程中轻松绘制趋势线。我可以很容易地在图中看到一条最合适的线。我没有包括 MSRP 和发动机尺寸或气缸之间的散点图，原因是这些数据与 MSRP 的相关性比 MSRP 和马力(83%)的相关性要小。因为如上所述，MSRP 和发动机尺寸之间的相关性为 54 %, MSRP 和气缸之间的相关性为 64%,所以没有理由绘制这些特征。

```
**# Plotting a scatter plot**fig, ax = plt.subplots(figsize=(5,5))
ax.scatter(df[‘Horsepower’], df[‘MSRP’])
plt.title(‘Scatter plot between MSRP and Horsepower’)
ax.set_xlabel(‘Horsepower’)
ax.set_ylabel(‘MSRP’)
plt.show()
```

![](img/3fc9aac7d475c8a4f461a1f6bc0eb733.png)

# 5)报告初步调查结果

我认为汽车的建议零售价(价格)和马力特性有很大的关系。我将在作业 4 中对此进行更多的探索。现在我知道我的问题陈述是“给定汽车的规格，预测汽车的价格(MSRP)”。主要想法是预测汽车的(建议零售价)价格。现在我知道我必须预测一个值，所以我应该使用回归算法，因为我有两个相关的特征(独立和从属特征)。但有许多类型的回归算法，如线性回归，随机森林回归，套索和岭回归等等。所以我可能会使用其中的一种算法，并在下一篇教程中实现一个机器学习模型来预测价格。因此，这项任务主要涉及探索性的数据分析，我准备好了数据，现在可以建立模型了。

# 参考

[1]janjughazhyan，L. (2017 年)。汽车数据。[在线]Kaggle.com。可在:[https://www.kaggle.com/ljanjughazyan/cars1](https://www.kaggle.com/ljanjughazyan/cars1)【2019 年 8 月 15 日访问】。

[2] Srinivasan，R. (2017 年)。汽车数据集。[在线]Kaggle.com。可在:【https://www.kaggle.com/toramky/automobile-dataset 【2019 年 8 月 16 日进入】。

[3]n .夏尔马(2018 年)。检测和去除异常值的方法。【在线】中等。可在:[https://towards data science . com/ways-to-detect-and-remove-the-outliers-404d 16608 DBA](/ways-to-detect-and-remove-the-outliers-404d16608dba)【2019 年 8 月 15 日获取】。