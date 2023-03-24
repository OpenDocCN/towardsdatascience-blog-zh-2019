# 机器学习中的距离函数:用简单语言编写的初级读本，动作点少。

> 原文：<https://towardsdatascience.com/distance-functions-in-machine-learning-a-primer-in-simple-language-with-few-action-points-f5e328759b24?source=collection_archive---------10----------------------->

![](img/a59d976062ac249c0abe4e317469ad80.png)

[https://www.machinelearningplus.com/statistics/mahalanobis-distance/](https://www.machinelearningplus.com/statistics/mahalanobis-distance/)

大多数数据科学问题的关键是定义给定观测值之间的距离或相似性函数。相似性度量将用于度量在给定的 n 维空间中观察值有多近或多远。对于给定的问题，有许多距离函数可供使用，我不想在这里详细描述所有的距离函数，而只是关于这些函数的简短信息，这些信息是我在数据智能和分析的课程中有机会看到的整体观点(由 IITH 的[sob Han Babu](https://www.iith.ac.in/~sobhan/)博士提供)。本指南面向入门级数据科学学生。

**欧几里德距离**

这等于两点之间的直线距离或最短距离或位移(..假设是二维的，但也可以是多维的)。这是成对的距离，也是测量两点间距离的默认度量。它对许多任务都很有用，但在数据具有以下特征的情况下要谨慎使用

a)极端或更多异常值。由于该距离是 L2 范数，当数据点之间存在异常值时，异常值的影响被放大。

b)相关性。这个距离没有考虑到观察中的变量可以相互关联(..为什么？)

c)如果数据集中的变量数量很多(比如> 7)，余弦距离优于欧几里德距离。为什么？

d)当然，根据 L2 范数的定义，变量是彼此相减的，这使得所有的变量必须是实值。如果欧几里德距离必须应用于分类数据，那么数据首先必须被编码成实数值。然而，这不是衡量分类变量之间距离的最佳方式。（..为什么？)

**马哈拉诺比的距离:**

该距离是相关性调整后的距离(..(欧几里得)在一对给定的数据点之间。要了解为什么需要去相关，请访问此 [*页面*](https://www.machinelearningplus.com/statistics/mahalanobis-distance/) 查看示例。给出的示例说明了使用欧几里德距离对变量相关的点进行聚类时的问题。

a)如果数据是数字的、相关的且变量数量适中，则最好使用马哈拉诺比距离(..<10)

**余弦距离:**

顾名思义，它与三角余弦函数有些关系。角的余弦由邻边的长度除以斜边给出。当你考虑两点之间的距离时，想象从原点出发的两个向量，那么向量之间的角度的[余弦由点积除以它们的长度给出。这实际上会减轻长短向量的影响，具有异常值的数据点之间的余弦距离不会像欧几里德距离那样被放大。](http://mathworld.wolfram.com/DotProduct.html)

a)如果有大量变量，最好使用余弦距离(..比方说> 8)。

**匹配和雅克卡系数:**

匹配系数和 Jaccard 系数在推导过程中非常接近，用于衡量分类变量何时出现在数据中。

让我们以只有两个级别的三个变量为例:

X1 : M，N，Y，N

X2: F，Y，Y，N

这些情况可以编码如下(..任意考虑 M，Y 为 1，N，F 为 0):

1,0,1,0

0,1,1,0

匹配系数:(1 和 0 中的相同匹配(…两者/所有类))/(所有类)= 2/4

Jaccard 系数:(1 中的相同匹配(…仅正/利息类别))/所有类别= 1/4

a)使用 Jaccards coef。在匹配 1 比匹配 0 具有更强的相似直觉的情况下

b)如果数据既是 Jaccard 又是匹配 coef，则使用 Kendall Tau 距离。将数据视为名义类。

**混合数据**

当数据中既有分类变量又有数值变量时，通常采用以下方法。

a)使用分类的一键编码或其他编码方法将所有内容转换成数字，并使用欧几里德、马氏、余弦或其他(..曼哈顿、相关性等)来测量数据点的距离/相似性。

b)将数据划分为两组变量，即数值变量和分类变量，并使用适当的距离度量分别计算两个分区之间的距离。然后组合成单个测量值，

(W1 * P1 + W2 * P2) / (W1+W2)。

**更多参考资料:**

[https://www . research gate . net/publication/327832223 _ Distance-based _ clustering _ of _ mixed _ data](https://www.researchgate.net/publication/327832223_Distance-based_clustering_of_mixed_data)

[](/importance-of-distance-metrics-in-machine-learning-modelling-e51395ffe60d) [## 距离度量在机器学习建模中的重要性

### 许多机器学习算法——有监督的或无监督的，使用距离度量来了解输入数据…

towardsdatascience.com](/importance-of-distance-metrics-in-machine-learning-modelling-e51395ffe60d) [](/how-to-measure-distances-in-machine-learning-13a396aa34ce) [## 机器学习中如何测量距离

### 这完全取决于你的观点

towardsdatascience.com](/how-to-measure-distances-in-machine-learning-13a396aa34ce)