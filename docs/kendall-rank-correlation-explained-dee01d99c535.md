# 肯德尔等级相关解释。

> 原文：<https://towardsdatascience.com/kendall-rank-correlation-explained-dee01d99c535?source=collection_archive---------3----------------------->

> *肯德尔秩相关(也称为肯德尔的 tau-b)怎么样？这是什么？我如何开始？我什么时候用肯德尔的 tau-b？嘿，教我你所知道的关于肯德尔等级相关的一切。”——****一颗好奇的心。***

## 什么是相关性？

相关性是一种双变量分析，衡量两个变量之间的关联强度和关系方向。就关系的强度而言，相关系数的值在+1 和-1 之间变化。值为 1 表示两个变量之间的完美关联程度。随着相关系数值趋向于 0，两个变量之间的关系将变弱。关系的方向由系数的符号表示；加号表示正相关，减号表示负相关。

通常，在统计学中，我们衡量四种类型的相关性:

*   [皮尔森相关性](https://medium.com/@joseph.magiya/pearson-coefficient-of-correlation-explained-369991d93404)(参数)
*   肯德尔等级相关(非参数)
*   Spearman 相关(非参数)
*   点双列相关。

## 肯德尔等级相关

也就是俗称的“肯德尔的τ系数”。Kendall 的 Tau 系数和 Spearman 的等级相关系数基于数据的等级评估统计关联。当您正在处理的数据未能通过一个或多个假设测试时，肯德尔等级相关(非参数)是[皮尔森相关(参数)](https://medium.com/@joseph.magiya/pearson-coefficient-of-correlation-explained-369991d93404)的替代方法。当您的样本量很小并且有许多并列的等级时，这也是 Spearman 相关(非参数)的最佳替代方法。

肯德尔等级相关用于测试按数量排列时数据排序的相似性。其他类型的相关系数使用观察值作为相关的基础，Kendall 的相关系数使用成对的观察值，并基于成对观察值之间的一致和不一致的模式来确定关联的强度。

*   **一致:**以相同的方式排序(一致性)，即它们相对于每个变量的顺序相同。例如，如果(x1 < x2)和(y1 < x2)或(x1 > x2)和(y1 > x2)，则一对观察值 X 和 Y 被认为是一致的
*   **不一致:**排序不同(不一致)，即数值排列方向相反。例如，如果(x1 < x2)和(y1 > x2)或(x1 > x2)和(y1 < x2)，一对观察值 X 和 Y 被认为是不一致的

肯德尔的τ相关系数通常比斯皮尔曼的ρ相关系数的值小。计算是基于和谐和不和谐的配对。对错误不敏感。样本量越小，p 值越准确。

## 肯德尔排名相关性答案的问题。

1.  学生的考试成绩(A，B，C…)和花在学习上的时间之间的相关性分类(<2 hours, 2–4 hours, 5–7 hours…)
2.  Customer satisfaction (e.g. Very Satisfied, Somewhat Satisfied, Neutral…) and delivery time (< 30 Minutes, 30 minutes — 1 Hour, 1–2 Hours etc)
3.  *我把第三个问题留给了你和宇宙…玩得开心！*

## 假设

在开始使用肯德尔等级相关之前，您需要检查您的数据是否满足假设。这将确保您可以实际使用有效的结果，而不仅仅是显示器上的数字。

1.  这些变量是在**顺序**或**连续**标度上测量的。顺序量表通常是衡量非数字概念的标准，如满意度、幸福度和不适度。例如非常满意、有些满意、中性、有些不满意、非常不满意。连续秤本质上是区间*(即温度如 30 度)*或者比例变量*(如体重、身高)。*
2.  **理想的**如果您的数据似乎遵循**单调关系。**简单来说，随着一个变量的值增加，另一个变量也增加，并且随着一个变量的值增加，另一个变量减少。原因如下:肯德尔的等级相关性衡量两个变量之间存在的关联的强度和方向(确定是否存在单调关系)。知道了这一点，测试单调关系的存在是有意义的。但是，就像我说的，这是**可取的**。

![](img/b0bd4fa7afed201caeb6d53977aabe8a.png)

Monotonic vs Non-Monotonic Relationship

就这样！又短又甜！你可以走了！

您还可以查看:

*   [肯德尔等级相关性— Python](https://medium.com/@joseph.magiya/kendall-rank-correlation-python-19524cb0e9a0)
*   [皮尔逊相关系数解释](https://medium.com/@joseph.magiya/pearson-coefficient-of-correlation-explained-369991d93404)
*   [皮尔逊相关系数- python](https://medium.com/@joseph.magiya/pearson-coefficient-of-correlation-using-pandas-ca68ce678c04)
*   [我的个人资料了解更多关于数据的信息](https://medium.com/@joseph.magiya)