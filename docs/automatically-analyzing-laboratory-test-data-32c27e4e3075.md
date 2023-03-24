# 自动分析实验室测试数据

> 原文：<https://towardsdatascience.com/automatically-analyzing-laboratory-test-data-32c27e4e3075?source=collection_archive---------30----------------------->

## 教程:自动分析实验室数据以创建性能图

## 如何编写为您执行数据分析的 Python 程序

![](img/d2ff5553d8b1351fc7471eea03c6c624.png)

科学家发现自己拥有大型数据集是很常见的。有时，它以单个文件中数千兆字节数据的形式出现。有时是数百个文件，每个文件包含少量数据。无论哪种方式，都很难管理。很难理解。你的电脑很难处理。您需要一种方法来简化这一过程，使数据集更易于管理，并帮助您跟踪一切。

这就是本教程的内容。我们正在编写 Python 脚本，它将自动为您分析所有数据，并以有意义、直观的文件名存储数据。同时使用来自实际研究的例子，这样你就知道你正在发展的技能是实用和有用的。

本教程的第一篇文章介绍了本教程的概念。如果“热泵热水器”、“性能系数(COP)”和“性能图”这些术语对你来说毫无意义，[你可能想读一下](/tutorial-automatically-creating-a-performance-map-of-a-heat-pump-water-heater-7035c7f208b0)。

第二篇文章介绍了[的配套数据集](https://peter-grant.my-online.store/HPWH_Performance_Map_Tutorial_Data_Set/p6635995_20036443.aspx),[将数据集分割成多个文件](/splitting-data-sets-cac104a05386)，并使用用户友好的名称。

伴随数据集是教程过程中有价值的一部分，因为它允许您跟随。您可以编写与我将要展示的完全相同的代码，运行代码，查看结果，并与我展示的结果进行比较。这样你就能确保你做得对。

在本教程的第二部分结束时，我们现在有三个数据文件，每个文件都包含指定环境温度下的测试结果。在这些文件中，我们可以看到热泵的耗电量、储水箱中的水温以及热水器周围的空气温度。

下一步是处理这些结果。我们需要编写一些代码来自动理解数据，计算热泵的 COP，并绘制数据，以便我们可以直观地理解它。

事不宜迟，我们开始吧。与所有 Python 编程一样，我们需要首先导入包。

# 我需要导入什么包？

这里有几个对这一步非常重要的包。它们是:

*   glob 是一个列表创建包。它读取存储在文件夹中的文件，并创建一个包含所有这些文件的列表。因为 Python 非常擅长遍历列表，所以我们可以使用 glob 来创建所有数据文件的列表，并让 Python 一次分析一个。
*   熊猫:需要我说明这个包裹的重要性吗？pandas 是 Python 中数据分析工具的黄金标准。
*   **os** : os 让我们可以访问操作系统命令。这听起来很吓人，因为使用它你可能会把你的电脑搞砸。不用担心；我们将只使用它来检查和添加文件夹。
*   numpy 是一个伟大的数字包，它支持强大的数据分析工具。在这个特定的例子中，我们将使用 numpy 来创建数据回归。
*   bokeh 是 Python 中的一个交互式绘图工具。它使您能够编写在分析数据时自动生成图的代码，并为用户提供与它们交互的选项。进一步了解散景的一个很好的来源是[用散景进行实际数据可视化](https://www.amazon.com/gp/product/B07DWG4T95/ref=as_li_tl?ie=UTF8&camp=1789&creative=9325&creativeASIN=B07DWG4T95&linkCode=as2&tag=1000xfaster-20&linkId=c503c2259e94061c5d4af3c7408e2223)。

在本教程的这一部分，将不会在脚本中使用散景；然而，这个相同的脚本将是未来部分的基础。现在导入散景是值得的，这样你就不用担心它了。

对于这个项目，我们需要导入整个 glob，pandas，os 和 numpy，而只从 bokeh 导入某些功能。这些都可以使用以下代码调用:

```
import glob
import pandas as pd
import os
import numpy as np
from bokeh.plotting import figure, save, gridplot, output_file
```

注意，pandas 被导入并分配给 pd，numpy 被分配给 np。这意味着我们现在可以通过写“pd”和“np”来引用这些包，而不是熊猫和 numpy。

既然我们所有的包都被导入了，下一步就是创建我们的 glob list 和 for 循环来依次遍历所有的文件。准备好后，您就可以编写代码来分析我们的每个数据文件了。

# 我如何遍历我所有的文件？

第一步是配置 glob。为此，您需要为 glob 提供一个路径和一个文件类型(文件类型是可选的，但是我们的所有数据都在。csv 格式，所以我们希望指定它以避免包含无关文件)。然后当你调用 glob 时，它会在指定的文件夹中创建一个该类型的所有文件的列表。这可以使用以下代码行来完成:

```
Path = r'C:\Users\YourName\Documents\AutomatingDataAnalysis\Files_IndividualTests'Filenames = glob.glob(Path + '/*.csv')
```

Path 变量告诉 glob 在指定的文件夹中查找相关的文件。请注意，上面指定的文件夹与您在本教程第一部分中保存文件的文件夹相同。

第二行调用 glob 包中的 glob 函数来创建所有指定文件的列表。请注意，代码引用了您指定的 Path 变量来说明 glob 应该在哪个文件夹中搜索文件。接下来是声明 glob 应该包含所有。csv 文件，没有别的。

这段代码完成后，glob 将创建一个包含所有。文件夹中的 csv 文件。它将具有[1.csv，2。csv，3.csv，…，n.csv】。对于那些已经下载了[配套数据集](https://peter-grant.my-online.store/HPWH_Performance_Map_Tutorial_Data_Set/p6635995_20036443.aspx)并正在跟随教程的人来说，它将是 PerformanceMap_HPWH_55.csv、PerformanceMap_HPWH_70.csv 和 PerformanceMap_HPWH_95.csv 的*完整路径*

现在您已经有了一个文件列表，下一步是创建一个 for 循环来遍历这些文件。然后，您需要打开这些文件，以便可以对每个文件执行数据分析。为此，您需要以下代码:

```
for Filename in Filenames:
 Data = pd.read_csv(Filename)
```

这段代码自动遍历文件名列表中的每个条目。请注意，这种编写方式导致 Filename 保存列表中每个条目的实际文件名。例如，第一次通过 For 循环时，文件名将包含 PerformanceMap_HPWH_55.csv 的完整路径。

第二行代码使用 pandas 将文件读入内存，并保存到数据中供以后分析。

既然文件已经定位并按顺序打开，下一步就是添加分析每个文件的代码。这将是我们的下一步。

# 如何编写代码来自动分析每个文件？

除了 Python 编程知识之外，这还需要大量关于正在研究的设备的知识。因为我假设你不是 HPWHs 的读者，我会确保写出相关信息。

## 过滤数据以仅包含重要部分

对于此过程，我们只关心 HPWH 中的热泵处于活动状态时的数据。这些设备中的热泵通常消耗 400–600 W，具体取决于环境温度和水温。同时，他们有消耗一些电力的机载电子设备。为了将数据过滤到我们所关心的部分，我们需要移除功耗小于 300 W 的所有数据，该功耗被选择为显著高于板载电子设备的功率消耗，但低于热泵的最小消耗。我们可以使用下面的代码行来实现这一点:

```
Data = Data[Data['P_Elec (W)'] > 300]
```

这一行重置了我们的数据帧，使**仅包含**器件功耗超过 300 W 的数据。但这确实影响了数据帧的索引，因此我们需要重置它，以保持数据帧干净。为此，我们可以使用以下代码:

```
Data = Data.reset_index()
    del Data['index']
```

## 识别测量之间的时间变化

现在，这个数据集中的时间戳数据不容易处理。幸运的是，我们从与实验室测试伙伴的合作中了解到，测量每 10 秒进行一次。因此，我们可以使用以下代码在数据框中创建一个新列，说明测试已经进行了多长时间:

```
Data[‘Time Since Test Start (min)’] = Data.index * 10./60.
```

## 计算储存在水中的能量的变化

影响 HPWHs COP 的一个关键参数是储水箱中的水温。水混合得不够好，无法保持一个恒定的温度。通常情况下，水箱底部是冷水，顶部是热水。为了这个练习，计算水箱的平均温度就足够了。由于我们的实验室测试人员好心地通知我们，他使用了 8 个均匀分布在水箱中的温度测量值，我们可以使用以下公式计算平均水温:

```
Data['Average Tank Temperature (deg F)'] = (1./8.) * (Data['T1 (deg F)'] + Data['T2 (deg F)'] + Data['T3 (deg F)'] + Data['T4 (deg F)'] + Data['T5 (deg F)'] + Data['T6 (deg F)'] + Data['T7 (deg F)'] + Data['T8 (deg F)'])
```

现在，我们真正关心的是水箱的平均温度在不同的测量时间有多大的变化。这样，我们可以确定储存在水箱中的能量的变化，从而确定热泵增加到水中的能量。我们可以使用下面两行代码来实现这一点:

```
Data['Previous Average Tank Temperature (deg F)'] = Data['Average Tank Temperature (deg F)'].shift(periods = 1)
Data.loc[0, 'Previous Average Tank Temperature (deg F)'] = 72.0
```

第一行使用。pandas 数据框的 shift 命令，用于在包含“平均油箱温度(华氏度)”数据的数据框中创建一个新列，但在数据框中下移一行。这会在第一行(索引 0)中创建一个空单元格，这会在执行计算时导致错误。第二行代码通过使用。用 72.0 填充此单元格。我们能做到这一点是因为我们友好的实验室测试人员告诉我们，每次测试都是在华氏 72.0 度时开始的。

现在我们可以计算每两个时间标记之间储存在水中的能量的变化。为此，我们需要知道一些常数和方程:

*   首先，用能量=质量*比热*(最终温度-初始温度)来确定水中能量的变化
*   第二，我们知道 HPWH 的储存罐容纳 80 加仑(再次感谢我们友好的实验室测试人员的交流)，
*   第三，水的密度是 8.3176 磅/加仑，并且
*   第四，水的比热是 0.998 Btu/lb-F。

我们可以将所有这些放在一起，用下面这条线计算储存能量的变化:

```
Data['Change in Stored Energy (Btu)'] =  (80 * 8.3176) * (0.998) * (Data['Average Tank Temperature (deg F)'] - Data['Previous Average Tank Temperature (deg F)'])
```

## 计算 COP

分析每个数据集的下一步是计算作为水箱水温函数的 COP。本教程的目标是将 COP 确定为水温和环境温度的函数，这将有助于理解在每个指定的环境温度下 COP 作为水温的函数。它指向正确的方向。

为了计算热泵的 COP，我们需要进行一些单位转换。耗电量目前以 W 表示，而添加到水中的能量目前以 Btu/时间步长表示。为了进行单位转换，我们使用 1 W = 3.412142 Btu/hr 的比率，然后将 Btu/hr 转换为 Btu/s，并乘以每个时间戳的 10 秒。这给出了代码:

```
Data['P_Elec (Btu/10s)'] = Data['P_Elec (W)'] * (3.412142/60/60) * 10
```

根据定义，COP 是添加到水中的热量除以消耗的电量。因此，可以通过下式计算:

```
Data['COP (-)'] = Data['Change in Stored Energy (Btu)'] / Data['P_Elec (Btu/10s)']
```

## 生成回归

现在我们有了一个表格，显示了三个指定 COP 中每一个 COP 与水温的函数关系。但是我们可以做得更好。如果有一个函数可以用来计算 COP 不是很好吗？只需输入水温，并据此识别 COP？

Numpy 提供了使这变得容易的工具。我们可以使用数字函数“polyfit”来确定将 COP 描述为水温函数的折射率系数。这是一个灵活的函数，允许您通过指定函数末尾的顺序来控制曲线的形状。因为热泵的 COP 作为温度的函数是抛物线，所以我们需要对这个例子进行二阶回归。因此，系数可以用下面一行来标识:

```
Coefficients = np.polyfit(Data[‘Average Tank Temperature (deg F)’], Data[‘COP (-)’], 2)
```

numpy“poly 1d”函数可用于使用这些系数创建回归。这通过以下方式完成:

```
Regression = np.poly1d(Coefficients)
```

现在，您可以使用这种回归方法确定热泵在特定水温下的 COP。请记住，仅针对特定的气温生成回归，因此仅使用正确气温的回归来估计 COP。创建二维性能图是本教程的最终目标，但我们还没有达到。

特定水温下的 COP 可以通过调用函数并使用水温作为输入来识别。例如，如果您想在水温为 72°F 时查找 COP，您可以输入:

```
COP_72 = Regression(72.0)
```

# 我如何保存这些结果？

可以使用我们一直使用的相同技术保存数据，如[自动存储来自分析数据集的结果](/automatically-storing-results-from-analyzed-data-sets-ed918d04bc13)中所述。我们需要 1)确保分析结果的文件夹可用，2)创建一个新的文件名，清楚地说明文件包含的内容，3)保存数据。

在这种情况下，我们希望将数据保存到一个名为“已分析”的新文件中。它应该在同一个数据文件夹中，并用来显示分析的结果。我们可以用下面的代码做到这一点:

```
Folder = Path + '\Analyzed' 
if not os.path.exists(Folder):
    os.makedirs(Folder)
```

第一行创建新文件夹的路径。它将' \Analyzed '添加到当前存在的路径中，声明它正在当前文件夹中查找一个名为“Analyzed”的文件夹。第二行确定该文件夹是否已经存在。如果没有，第三行创建它。

之后，我们需要为数据集和系数设置文件名。这可以通过将我们已经拥有的与字符串的一个子部分相结合来实现。我们可以使用字符串索引来标识我们想要保留的文件名部分。例如，第一个文件的文件名部分“PerformanceMap_HPWH_50”非常清楚地说明了文件包含的内容。因为我们知道文件名的最后四个字符是。我们可以通过使用索引[-26:-4]来分离字符串的这一部分。换句话说，我们希望字符串中的字符从“倒数第 26”到“倒数第 4”，不包括倒数第 4。

接下来，我们可以定制一点文件名。也就是说，我们可以声明我们希望数据文件名声明它包含分析数据，我们希望系数文件名声明它包含系数。我们可以用下面几行写下这两个文件的文件名:

```
Filename_Test = Folder + '\\' + Filename[-26:-4] + '_Analyzed.csv'
Filename_Coefficients = Folder + '\Coefficients_' +  Filename[-6:]
```

然后我们简单地保存文件。分析后的数据可以和熊猫一起储存。to_csv 函数，系数可以用 numpy 保存。tofile 函数如下:

```
Data.to_csv(Filename_Test, index = False)
Coefficients.tofile(Filename_Coefficients, sep = ‘,’)
```

请注意，保存数据集的行 index = False。这意味着保存表时不会保存数据框的索引。还要注意 numpy。tofile 函数要求您指定一个分隔符。在这种情况下，我们使用一个逗号，用代码 **sep = '、** 指定。

# 我怎么知道它工作正常？

在流程的这个阶段，有大量的事情可能会出错。也许实验室测试人员在进行实验时犯了一些错误。可能是乐器坏了。也许代码中有一个错别字，或者一个不正确的单位转换。

必须确保在这个过程中不出现这些问题或任何其他问题。因此，该过程的下一步是检查数据集的错误。我们将在教程的下一阶段讨论这个问题。首先，我将讨论如何手动检查数据错误。这样，你将对我们正在检查的潜在错误以及如何识别它们有一个明确的理解。然后我将讨论如何在这个脚本中添加代码来自动检查这些错误，并在出错时向您发出警告。

# 教程目录

这是一系列文章的一部分，教你自动分析实验室数据和绘制热泵热水器性能图所需的所有技巧。本系列的其他文章可以通过以下链接找到:

[简介](/tutorial-automatically-creating-a-performance-map-of-a-heat-pump-water-heater-7035c7f208b0)

[分割数据集](/splitting-data-sets-cac104a05386)

[检查分析实验室数据的错误](/checking-analyzed-laboratory-data-for-errors-4bd63bcc554d)

[如何编写检查数据质量的脚本](https://medium.com/zero-equals-false/how-to-write-scripts-that-check-data-quality-for-you-d8762dab34ca)

[如何在 Python 中自动生成回归](https://medium.com/zero-equals-false/how-to-perform-multivariate-multidimensional-regression-in-python-df986c35b377)