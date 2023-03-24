# 数据缩放对机器学习算法的影响

> 原文：<https://towardsdatascience.com/the-influence-of-data-scaling-on-machine-learning-algorithms-fbee9181d497?source=collection_archive---------22----------------------->

![](img/06f0c60b3d3c0b9caec91615b4cb939c.png)

缩放是数据预处理的行为。

数据再处理包括在数据被用于进一步的步骤之前，对数据进行向上或向下的转换和缩放。属性常常不能用相同的标准、尺度或度量来表达，以至于它们的统计数据会产生失真的数据建模结果。例如，K-Means 聚类算法不是尺度不变的；它通过欧几里得距离计算两点之间的距离。重新回忆一下欧几里德距离的概念—它是一维空间中两点之间的非负差值。因此，如果其中一个属性的值范围很大，计算出的距离将会受到该属性的影响(即较小值的属性贡献很小)。例如，如果其中一个属性以厘米为单位进行测量，然后决定将测量值转换为毫米(即，将这些值乘以 10)，则生成的欧几里德距离会受到显著影响。要让属性将大约*成比例地*添加到最终计算的距离，该属性的范围应该被规范化。

归一化有多种含义，最简单的情况是，它指的是将某个属性标度转换到某个版本，该版本在与另一个属性进行比较时消除了总体统计数据的影响。

试图使用像主成分回归(PCR)这样的分析技术要求所有的属性都在同一尺度上。属性可能具有会影响 PCR 模型的高方差。缩放属性的另一个原因是为了计算效率；在梯度下降的情况下，函数收敛得相当快，而不是根本没有归一化。

有几种归一化方法，其中常见的有 **Z 值**和**最小最大值**。

一些统计学习技术(即线性回归)在缩放属性没有效果的情况下可能受益于另一种预处理技术，如将名义值属性编码为一些固定的数值。例如，任意给一个性别属性赋予值“1”表示女性，赋予值“0”表示男性。这样做的动机是允许将属性合并到回归模型中。确保在某处记录代码的含义。

**选择最佳预处理技术——Z 值还是最小最大值？**

简单的答案是两者兼而有之，这取决于应用。每种方法都有其实际用途。观察值的 Z 值定义为高于或低于平均值的标准偏差数，换句话说，它计算方差(即距离)。如前所述，聚类数据建模技术需要标准化，因为它需要计算欧几里德距离。Z 得分非常适合，并且对于根据特定的距离度量来比较属性之间的相似性至关重要。这同样适用于主成分回归(PCR)；在其中，我们感兴趣的是使方差最大化的成分。另一方面，我们有将数据属性转换到固定范围的最小-最大技术；通常在 0 到 1 之间。Min-max 法取*的函数形式 y =(x-min(x))/(max(x)-min(x))*，其中 x 为向量。例如，在图像处理和神经网络算法(NNA)中，因为像 NNA 的[0，255]这样的大整数输入会中断或减慢学习过程。最小-最大归一化将 8 位 RGB 颜色空间中图像的像素亮度值范围[0，255]更改为 0–1 之间的范围，以便于计算。

**直观地学习数据预处理**

也许在数据集上应用规范化方法可以揭示它发生了什么；我们可以将数据点转换可视化，以便更直观地解释它。因此，让我们从加载来自 [UCI 机器学习数据库](https://archive.ics.uci.edu/ml/machine-learning-databases/wine/wine.data)的数据集开始。这是一个葡萄酒数据集，其特征是第一列中标识为(1，2，3)的三类葡萄酒。这些数据来自一项分析，该分析确定了三种葡萄酒中 13 种成分的含量。

```
df <- read.csv(“wine.data”, header=F)wine <- df[1:3]colnames(wine) <- c(‘type’,’alcohol’,’malic acid’)wine$type <- as.factor(wine$type)
```

使用 read.csv 将葡萄酒数据读取为没有标题的 CSV 文件。葡萄酒类型也通过 as.factor()转换成一个因子。这些步骤不是标准化所必需的，但却是良好的常规做法。

我们选择了三个属性，包括葡萄酒类别，以及标注为酒精和苹果酸的两个成分，它们以不同的尺度进行测量。前者用百分比/体积来衡量，而后者用克/升来衡量。如果我们要在聚类算法中使用这两个属性，我们很清楚需要一种标准化(缩放)的方法。我们将首先对葡萄酒数据集应用 Z 得分归一化，然后应用最小-最大方法。

```
var(wine[,-1])
```

![](img/4281decc374ff81d19149497429a9188.png)

```
std.wine <- as.data.frame(scale(wine[,-1])) #normalize using the Z-score methodvar(std.wine) *#display the variance after the Z-score application*
```

![](img/e451e45f238836a3497f3af096faaa3c.png)

```
mean(std.wine[,1]) *#display the mean of the first attribute*
```

![](img/2e253fa70355d48c68195785b1c991ee.png)

```
mean(std.wine[,2]) *#display the mean of the second attribute*
```

![](img/1c3063cd77987e5baf888573a3172322.png)

我们可以看到酒精和苹果酸是标准化的，方差为 1 和 0。

*注意，平均数被提升到-16 的幂，-17 (e-16，e-17)分别表示接近于零的数。*

接下来，我们创建 min-max 函数，将数据点转换为 0 到 1 之间的值。

```
min_max_wine <- as.data.frame(sapply(wine[,-1], function(x) { return((x- min(x,na.rm = F)) / (max(x,na.rm = F)-min(x,na.rm = F)))}))
```

绘制所有三种不同等级的葡萄酒数据点，如下所示:

```
plot(std.wine$alcohol,std.wine$`malic acid`,col = “dark red”,xlim=c(-5,20), ylim = c(-2,7),xlab=’Alcohol’,ylab=’Malic Acid’, grid(lwd=1, nx=10, ny=10))par(new=T)plot(min_max_wine$alcohol,min_max_wine$`malic acid`,col=”dark blue”,xlim=c(-5,20),ylim=c(-2,7),xlab=’’, ylab=’’,axes = F)par(new=T)plot(wine$alcohol,wine$`malic acid`,col=”dark green”, xlim=c(-5,20),ylim = c(-2,7),xlab=’’, ylab=’’,axes = F)legend(-6,7.5, c(“std.wine”,”min_max_wine “,”input scale”), cex=0.75, bty=”n”, fill = c(“dark red”,”dark blue”,”dark green”))
```

![](img/559a7c7d6d0137bf85910139229300a9.png)

*Three datasets; std.wine (red), min_max_wine (blue), and the original dataset (green) points*

如你所见，有三个数据点集；在绿色集合中，测量值为原始体积百分比，而标准化属性为红色，其中数据以平均值 0 和方差 1 为中心，标准化最小-最大属性范围为 0-1。

这三个集合的形状可能看起来不同，但是，如果您使用新的比例放大每个集合，您会注意到，无论整体形状大小如何，这些点仍然精确地位于彼此相对的相同位置。这些标准化方法通过缩放保持了数据的完整性。