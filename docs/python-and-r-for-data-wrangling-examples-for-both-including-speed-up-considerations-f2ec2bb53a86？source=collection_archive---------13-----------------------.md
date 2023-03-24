# Python 和 R 的数据争论:并排比较 Pandas 和 Tidyverse 代码，并学习加速技巧。

> 原文：<https://towardsdatascience.com/python-and-r-for-data-wrangling-examples-for-both-including-speed-up-considerations-f2ec2bb53a86?source=collection_archive---------13----------------------->

## 成为双语数据科学家。学习加速代码技巧。用可互操作的 Python 和 R cells 编写双语笔记本。

![](img/b1e970c5ed72032119c799850ac8dffa.png)

© Artur/AdobeStock

几年前，你会专门用这两种语言中的一种来编写数据分析程序:Python 或 r。这两种语言都提供了从数据探索到建模的强大功能，并且拥有非常忠实的粉丝。然而，事情已经发生了变化。如今，有一些库，如 *reticulate* 和 *PypeR、*允许你分别将 Python 代码合并到 *R Markdown* 和 *Jupyter* 笔记本中。了解两种语言的基本功能，如数据争论，可以扩展您的编程视野，允许您与使用任一种语言的人一起工作，并创建双语笔记本，充分利用每种语言的优势。在本文中，我们将分别使用 *pandas* 和 *tidyverse* 库来讨论 Python 和 R 中的数据争论，以及加速代码的技巧。**读完这篇文章后，你会发现 Python 和 R，在许多表达式上非常相似，至少在数据争论方面是如此**。所以，你只要额外付出一点努力，就能掌握两种语言的数据角力，成为“超级 **R** -Pythonista”！

> 做一个超级——T21——巨蟒(而不仅仅是一个巨蟒！)

# 双语 R 降价文件

他的文章附有一个 R Markdown 文件，你可以在 github 上找到。 在这个文件中，数据角力操作实现了两次:在 Python 和 R 单元格中，彼此相邻。这可以通过导入 *reticulate* 库来实现。Python 和 R 单元目前独立工作；在我的下一篇文章中，我将展示 Python 和 R 单元之间的参数传递以及跨语言通信。

*R Markdown* 文件也可以作为 Python 和 R 数据争论操作的两个独立的备忘单。您也可以[下载 *R Markdown* 文件的编织版本](https://drive.google.com/file/d/1k6ASkLWCQoDCWTvHTuLYz-Vo1eCcerAB/view?usp=sharing)。两种语言的数据争论是在相似的结构上进行的:R 数据帧和 Python 数据帧。具体实现的操作有:

A.**创建/读取数据帧。**

B.**总结。**

C.**使用索引、名称、逻辑条件、正则表达式选择**行、列、元素。将介绍 Python 和 R 中的 *filter()* 函数。

D.**删除-添加**行、列。我们将讨论 R 中的 *mutate()* 函数和 Python 中的 *map* 。

E.**对行/列应用函数**，包括 Python 中的 l *ambda* 函数。

F.**加速代码**。我们将讨论一些技术，比如并行化和用于代码加速的函数编译。

## 数据争论操作

在前面提到的 *R Markdown 文件*中，Python 代码被括在下面的括号中，这些括号定义了一个 Python 单元格:

` ` `{python}

```

虽然 R 代码包含在这些括号中:

` ` `{r}

```

这些括号由 *RStudio* 自动生成，点击*插入*选项卡并选择您想要的语言。

**编程提示**:在执行 *R Markdown* 中的任何 Python 单元之前，执行导入 *reticulate* 库的 shell。注意，在同一个单元格中，我们指示 *RStudio* 使用哪个 Python。另外，使用命令 *py_config()在 *RStudio* 控制台中配置 Python 也是一个好主意。*

值得注意的是，在这两种语言中，有多种方式来执行操作，其中许多在前面提到的 *R Markdown* 文件中有所介绍。由于篇幅所限，在本文中，我们将只介绍其中的一部分。让我们开始吧:

## 从 CSV 文件中读取数据帧

正如我们在下面的要点中看到的，在两种语言中，我们都调用一个读取 csv 文件的函数。在 Python 中，这个函数是通过 *pandas* 库调用的。 [heart.csv](https://www.kaggle.com/sonumj/heart-disease-dataset-from-uci) 文件来自 Kaggle 和 UCI 库。

在下面的 Python 代码中值得注意的是使用了一个 *lambda* 函数(一个匿名内联函数)来排除读取 *drop_col* 列表中定义的列。

**编程提示:**在 R 中定义 *read.csv()* 函数的编码是一个好主意，如下所示，以确保正确读取所有列名。当我不使用编码时，第一列的名称被读错了。我不需要用 Python 定义编码。

## 从头开始创建数据框架

正如下面的所示，在这两种语言中，一个*数据帧*可以从一个低阶结构中创建，分别用于 Python 和 R 的*矩阵*和*数组*。

在 R 部分中，*存款帧* *数据帧*包含 3 个人的银行存款(单位:千)。使用 *rnorm()* 函数生成银行存款，该函数生成 6 个随机数，平均值为 12，标准差为 3。

**编程技巧:**正如你在下面的 R 代码中注意到的，我们导入了 *tidyverse* 库，这是一个包含很多包的库，比如 *dplyr* 。我们在 R 代码段中需要的函数只在 *dplyr* 库中定义。那么为什么要进口 *tidyverse，*而不是我们刚需的 *dplyr 呢？一个原因是未来的可扩展性(我们可能需要使用其他包中的函数)。*另一个原因是当我们导入*库(dplyr)* 时，得到一个警告，它的重要函数 *filter()* 被另一个库屏蔽了。如果我们导入*库(tidyverse)* ，就没有这个问题了。

## 摘要

H 在这里，我们将检查两种类型的汇总:(a)汇总关于*数据帧*的基本信息的函数。(b)允许对数据切片进行分组和定制洞察的功能。

**描述数据帧一般信息的函数**

在以下要点中，在 R 部分， *head()* 函数显示了*数据帧*的前几行(观察值)。 *glimpse()* 函数显示观察值的数量、变量(列)以及后者的类型、名称和值。与 *glimpse()* 功能类似的信息由 *str()* 功能显示。一个更有用的函数是来自 *prettyR* 包的 *describe()* 函数，它显示每个变量的基本统计数据和有效案例数。变量统计信息也由 *summary()* 函数显示。与 R 类似，在 Python 部分中， *head()* 函数显示第一行，而 *describe()* 函数显示每一列的基本统计信息，比如平均值、最小值和最大值。

## 分组和汇总组内的信息

R 和 Python 都有基于一个变量或一组变量进行分组的函数。如以下要点所示，在 R 中，我们可以使用以 R 为基数的函数 *tapply()* ，分别计算男性和女性的平均胆固醇。或者，我们可以使用函数 *group_by()* ，后跟汇总函数，如 *count()* 、 *mean()* 或*summary _ all()*。类似地，如下图所示，在 Python 中，组总结是由函数 *groupby()* 执行的，后面是总结函数，如 *count()。*

分组的另一个例子，在 R:

Python 中分组汇总的另一个示例:

**基于索引的行、列、元素选择**

这里需要注意的一点是，R 中的索引从 1 开始，而 Python 中的索引从 0 开始。在 Python 中，我们使用 *iloc* 索引器来访问整数位置。

## 使用名称、正则表达式、逻辑条件进行列/行选择

这里有一些需要注意的亮点:

*   在 R 中使用 *% > %* 管道操作符，它允许你编写简洁的功能代码。
*   使用 R 中的 *select()* 函数选择列，使用列名、正则表达式和/或逻辑表达式。例如，在第 7、8 行中，select() 语句用于选择包含字母 m 的所有列，它们的 *max()* 小于 12。
*   函数 *filter()* ，这是在 r 中进行选择的另一种方法。下面的函数 *filter_all()* 允许我们选择满足标准的所有行。
*   Python 提供了类似的 *filter()* 函数供选择。axis 参数指定操作是应用于列还是行。

**行/列删除和添加**

B elow，值得注意的有以下几点:

*   在 R 中，对列名使用减号操作符来执行列删除。我们也可以使用字符串匹配函数，比如 *starts_with()* ，如下图所示。其他类似的函数有
    *ends_with()，matches()，contains()。*
*   在 R 中，只需使用列名就可以添加列(下面的列 *Brandon* )。另一种流行的方法是来自 *dplyr* 库中的 *mutate()* 函数。这里，我们使用 *mutate()* 添加一个列 *sumdep* ，其中这个新列是两个现有列的总和。请注意与 t *ransmute()* 函数的区别，它只添加新列(不保留旧列)。如下图所示，使用 *mutate()* 的一个特别有趣的方法是将它与 *group by* 结合起来，在组内进行计算。
*   在 Python 部分，值得注意的是添加了一个使用逻辑的列，该逻辑是通过 *map()* 函数实现的。

## 对列/行应用函数

BPython 和 R 都提供了 *apply()* 函数，允许我们根据一个额外的参数在行或列上实现一个函数。例如下面，在 R 中，如果中间的参数是 2，则函数应用于列，而如果是 1，则应用于行。在 Python 中， *apply()* 函数的决定参数是 axis (0，应用于列，1，应用于行)。下面另一个有趣的花絮是使用一个 *lambda* 函数将所有元素乘以 5。

# 加速考虑

在这里，我们将简要讨论三种加速 R 中代码的方法: *data.table* 、并行化和函数编译。对于 Python，将呈现的加速选项是 *modin* 。

![](img/389652a06dda820acccf66261e87b35a.png)

©Syda Productions/AdobeStock

## r:数据表

*数据表*是*数据帧*【2】的扩展，其优点有两方面:

(a)对于大文件，加载数据比*数据帧*快得多。下面是一个例子。我读取一个大文件(66.9MB)，使用函数 *read.csv()，*返回一个*数据帧*，以及 *fread()，*返回一个*数据表*。差异非常显著(例如，对于 data.table，用户+系统时间大约快了 **30 倍！**)。我使用的数据文件的引用是 references 部分中的[3]。

(b)简洁、紧凑的过滤表达式，其执行速度也相当快。使用*数据表*进行过滤以及与 *dplyr* 过滤进行比较的示例如下所示。在这个要点中，我们使用库 *dplyr* 的 *filter()* 计算胆固醇> 280 的男性的平均年龄。我们也用*数据表*计算它，这是用在一行简洁的代码中完成的。

## r:并行

在 R 中，并行化是使用*并行*库[4]实现的，这允许我们的 R 程序利用我们计算机中的所有内核。我们可以使用函数 *detectCores()* 来找出我们有多少个内核，然后我们可以使用函数 *makeCluster()* 来创建一个包含这些内核的集群。这显示在下面的要点中。然后，我们可以使用 *apply()* 系列函数的并行版本来执行各种操作。例如，在下面的要点中， *sapply()* 函数及其并行版本 *parSapply()* 用于对大量数字求平方。我们可以看到, *parSapply()* 的用户+系统时间是 7.95+0.66=8.61，而 plain apply()的用户+系统时间是 11.80+0.03=11.83。

## r:函数编译

另一种提高 R 语言运算速度的方法是编译函数。这是通过*编译器*库及其 *cmpfun()* 函数【5】实现的。下面的要点显示了这一点，以及并行化、编译在时间消耗方面的比较。我们看到，当我们结合并行化和编译时，实现了最快的用户+系统时间(7.73)。就其本身而言，只实现*的*并行化比只实现*的*函数编译(8.87)产生更快的用户+系统时间(8.61)。

结束我们对超速 R 的讨论，值得一提的是库*gpuR*【6】*。*使用 *ViennaCL* 库， *gpuR* 允许在任何 GPU 上并行计算，这与之前的 R 包在后端依赖 *NVIDIA CUDA* 形成对比。

## Python:摩丁

如[7]中所述，modin 是一个*数据框架*库，具有与 *pandas* 完全相同的 API，并允许显著加速工作流(在 8 核机器上快 4 倍)。要使用它，我们只需要修改一行代码。而不是:

*进口熊猫作为 pd* 我们将使用:

*导入 modin.pandas 作为 pd*

一句忠告。modin 是建立在 Ray 之上的，所以要确保你有正确版本的 Ray。

你可以在我的 github 库中找到一个额外的赠品是[另一个双语 R Markdown 文件，](https://github.com/theomitsa/bilingual-R-Markdown/blob/master/arrayoperations)它实现了数组创建(1-D 和 2-D)和使用数组的数学函数(点积、特征值等)。)都有 Python 和 r .你也可以[下载它的针织版](https://drive.google.com/file/d/1ID1RbER17OueZufjoKXZyO7ZzHDCYKB2/view?usp=sharing)。

感谢阅读！

# 参考

1.  Pandey，p .，**从 R vs. Python，到 R 和 Python，**https://towardsdatascience . com/From-R-vs-Python-to-R-and-Python-aa 25 db 33 ce 17
2.  **数据表简介**，[https://cran . r-project . org/web/packages/data . table/vignettes/datatable-intro . html](https://cran.r-project.org/web/packages/data.table/vignettes/datatable-intro.html)
3.  MITx 和 HarvardX，2014 年，“HMX PC 13 _ DI _ v2 _ 5–14–14 . CSV”， ***HarvardX-MITx 个人课程 2013 学年 De-Identified 数据集，v*** *版本 2.0* ，[https://doi.org/10.7910/DVN/26147/OCLJIV](https://doi.org/10.7910/DVN/26147/OCLJIV)，哈佛数据世界，V10
4.  特雷德韦、**并行运行 R 代码**、[https://www.r-bloggers.com/running-r-code-in-parallel/](https://www.r-bloggers.com/running-r-code-in-parallel/)
5.  罗斯，n .**快一点！更高！更强！-忙人加速 R 码指南，**[https://www . R-bloggers . com/faster-higher-stonger-A-Guide-to-Speeding-Up-R-Code-for-Busy-People/](https://www.r-bloggers.com/faster-higher-stonger-a-guide-to-speeding-up-r-code-for-busy-people/)
6.  **包 gpuR** ，[https://cran.r-project.org/web/packages/gpuR/gpuR.pdf](https://cran.r-project.org/web/packages/gpuR/gpuR.pdf)
7.  Pandey，p .，**用 Modin 获得更快的熊猫，甚至在你的笔记本电脑上**，[https://towards data science . com/Get-faster-pandas-with-Modin-even-on-your-laptops-b 527 a2 eeda 74](/get-faster-pandas-with-modin-even-on-your-laptops-b527a2eeda74)