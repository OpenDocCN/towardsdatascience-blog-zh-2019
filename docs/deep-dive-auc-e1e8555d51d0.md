# 深潜 AUC

> 原文：<https://towardsdatascience.com/deep-dive-auc-e1e8555d51d0?source=collection_archive---------18----------------------->

![](img/3300da834b76b9234ab80169f4512ff9.png)

Jheison Huerta photograph of the via lactea reflecting in the salt desert

## 将机器学习指标转化为现实世界

在本帖中，我们将解释如何将统计数据和机器学习指标转化为对业务团队的更接近的解释。让我们从解释 AUC 的值开始，然后回来解释如何计算它，这将为我们提供一个更好的背景，并帮助您理解它的优势。

[上一篇文章](https://medium.com/@marcos.silva0/confusion-matrix-deep-dive-8a028b005a97?sk=61f17ae466b8d1d484547328bb72b3e7)(深入困惑矩阵)

# 如何解读 AUC

AUC 被计算为范围从 0 到 1 的面积，但是对 **AUC 值的解释是一种可能性**。如果我们从预测中选取任意两个观察值，它们将会以正确的方式**排序**的可能性。也就是说，AUC 为。90(其面积为 90%)的解释是，如果对任何预测值进行两次观察，它们被正确排序的概率是 AUC 本身，90%。这就解释了为什么最小 AUC 是. 50，因为如果你的模型是完全随机的，两个观测值被正确排序的概率是 50%，即随机。正如最大 AUC 为 1。

![](img/5c7a8da103345818317f082535f2e390.png)

Source of gif: [https://www.spectrumnews.org/opinion/viewpoint/quest-autism-biomarkers-faces-steep-statistical-challenges/](https://www.spectrumnews.org/opinion/viewpoint/quest-autism-biomarkers-faces-steep-statistical-challenges/)

# 但毕竟这是什么这样的 AUC。

ROC 曲线的每一端由选定的阈值形成。下面的例子有助于我们理解。想象一下，我们从 20%的削减开始，注意假阴性率和假阳性率，我们在不同的阈值上移动以形成曲线。实际上，当我们要求 sklearn 绘制 ROC 曲线时，它通过如下观察进行观察:

![](img/52712ace17f091a96dbefbeb3c9acf1e.png)

1.  计算类为真的几率。P (y = 1)
2.  从最高概率到最低概率排序。
3.  它从左角开始，如果最可能的观察是正确的，它就向上，如果是错误的，它就向右。
4.  对所有的观察重复这个过程后，我们就有了 ROC 曲线。
5.  我们将 AUC 计算为 ROC 曲线下的面积。

AUC 只不过是在 ROC 曲线下形成的面积。但是它带来了非常有趣的解释。

![](img/cc966c08b6204dde47de3cf816825dea.png)

Source of gif: [https://www.spectrumnews.org/opinion/viewpoint/quest-autism-biomarkers-faces-steep-statistical-challenges/](https://www.spectrumnews.org/opinion/viewpoint/quest-autism-biomarkers-faces-steep-statistical-challenges/)

# AUC 曲线给出了类别分离程度的概念

一个好的模型是一个可以很好地分离两个类别的模型，所以我们的模型在两个类别之间的交集越少，它就越好，因此 AUC 就越大。

![](img/080cfa822e7f9b38a96649f207935f7c.png)

# AUC 对患病率不敏感

我们在混淆矩阵中遇到的一个非常恼人的问题是，普遍性(即类别之间的比率)极大地影响了这些指标，但 AUC 不受此影响，在下面的 gif 中，我们有一个真实的例子。AUC 曲线在右边的绿色部分，我们注意到它没有移动到任何患病率水平。

![](img/65bb3136da751b867892b515717d9c3d.png)

Source of gif: [https://www.spectrumnews.org/opinion/viewpoint/quest-autism-biomarkers-faces-steep-statistical-challenges/](https://www.spectrumnews.org/opinion/viewpoint/quest-autism-biomarkers-faces-steep-statistical-challenges/)

# 解释曲线的形状

当我们有两条不同的 AUC 曲线时，我们必须在每种情况下决定哪一条最适合我们。在下图的情况下，很明显模型 3 是最好的，模型 1 是最差的。这个决定不仅来自于 AUC 值更高，而且曲线形状显示模型 3 对于任何选择的阈值都更好。请注意，在曲线的开始(最容易预测的地方)，三条曲线几乎相同，随着我们向更困难的预测移动，模型越向右开始区分。

![](img/2ffd40f9d046f7390fe03b0e4191679b.png)

现在一个更难的问题，如果曲线有相同的面积，但形状不同，如下图所示？这两种型号哪个更好？在这种情况下，曲线 A 对于疾病检测等问题会更好，因为它比曲线 b 更敏感

![](img/3d37db1d271a2512d09389dae1691b8f.png)

一个有趣的特征是，如果这两条曲线具有相同的 AUC 为 80，并且我们组装了这两个模型，则很可能最终模型将优于两个初始模型，因为它们可以结合两个模型的优点并获得两条曲线的最佳效果。

# 将 AUC 与人类专家进行比较

![](img/005688c44104ce81eed0ef70f24dc4be.png)

一个经过训练的机器学习模型有一个完整的 ROC 曲线，但在实践中，我们只能使用曲线上的单个点，并选择与该点相关的比率。在一个比较检测皮肤癌的模型(蓝色曲线)和多个医生(红点)的例子旁边，注意每个医生是一个单点。有些医生更敏感，有些更准确，但他们仍然选择单个切口，所以他们用点来表示，因为我们的模型有所有点的曲线。

以前的帖子

[](https://medium.com/@marcos.silva0/confusion-matrix-deep-dive-8a028b005a97) [## 困惑矩阵——深度探究

### 将机器学习指标转化为现实世界

medium.com](https://medium.com/@marcos.silva0/confusion-matrix-deep-dive-8a028b005a97) 

参见:

*   [数据科学](https://medium.com/swlh/data-science-and-the-data-scientist-db200aac4ea0)的范围是什么[；](https://medium.com/beacon-insight/ci%C3%AAncia-de-dados-e-o-cientista-de-dados-72634fcc1a4c)
*   [为数据科学家推荐书籍、课程和电影。](/how-to-become-a-data-scientist-2a02ed565336)
*   [解释机器学习；](/the-ultimate-guide-using-game-theory-to-interpret-machine-learning-c384cbb6929)
*   [统计学简史](http://a%20brief%20history%20of%20statistics/)；