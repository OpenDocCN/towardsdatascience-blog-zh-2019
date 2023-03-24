# 从 R 到 Python —线性回归诊断图

> 原文：<https://towardsdatascience.com/going-from-r-to-python-linear-regression-diagnostic-plots-144d1c4aa5a?source=collection_archive---------12----------------------->

![](img/dc7fa096b5026957501715ee50a4b3cf.png)

Symbols taken from wikipedia.

作为一个已经过渡到 Python 的长期 R 用户，我最怀念 R 的一件事是轻松地为线性回归生成诊断图。有一些关于如何在 Python 中进行线性回归分析的很好的资源([参见这里的例子](/linear-regression-on-boston-housing-dataset-f409b7e4a155))，但是我还没有找到一个关于生成诊断图的直观资源，我从 r。

我决定为在 r 中使用 plot(lm) 命令时出现的四个图构建一些包装函数。这些图是:

1.  残差与拟合值
2.  正常 Q-Q 图
3.  标准化残差与拟合值
4.  标准化残差与杠杆

第一步是进行回归。因为这篇文章是关于 R 怀旧的，所以我决定使用一个经典的 R 数据集:mtcars。它相对较小，具有有趣的动态，并且具有明确的连续数据。通过安装 pydataset [模块](https://github.com/iamaziz/PyDataset)，可以非常容易地将这个数据集导入 python。虽然该模块自 2016 年以来一直没有更新，但 mtcars 数据集来自 1974 年，因此我们想要的数据是可用的。以下是我将在本文中使用的模块:

现在，让我们导入 mtcars 数据帧并查看它。

![](img/f17a40d3c3d949dd09c3445b16e7649d.png)

可以通过以下命令访问该数据集的文档:

```
data('mtcars', show_doc = True)
```

我特别感兴趣的是使用气缸数量(cyl)和汽车重量(wt)预测每加仑英里数(mpg)。我假设 mpg 会随着 cyl 和 wt 下降。statsmodels 公式 API 使用与 R *lm* 函数相同的公式接口。请注意，在 python 中，您首先需要创建一个模型，然后拟合该模型，而不是在 r 中创建和拟合模型的一步过程。

重要的是，statsmodels 公式 API 自动将截距包含到回归中。原始 statsmodels 接口不会这样做，因此请相应地调整您的代码。

![](img/7301afd6ea0a67ec342a11aad011b5a6.png)

该模型解释的差异量相当高(R^2 = 0.83)，并且 cyl 和 wt 都是负的和显著的，支持我最初的假设。在这个 OLS 输出中有很多东西要解开，但是在这篇文章中，我不会探究所有的输出，也不会讨论统计意义/非意义。这篇文章是关于构建异常值检测和假设检验图，这在 base R 中很常见。

这些数据符合 OLS 模型的假设吗？让我们开始策划吧！

首先，让我们检查残差中是否存在与拟合值相关的结构。这个情节相对容易创作。这里的计划是从拟合模型中提取残差和拟合值，计算通过这些点的 lowess 平滑线，然后绘制。注释是残差的最大绝对值的前三个指数。

![](img/c9df6b4813cf958502d2f9528735d8c4.png)

在这种情况下，残差中可能有轻微的非线性结构，可能值得测试其他模型。菲亚特 128、丰田卡罗拉和丰田科罗纳可能是数据集中的离群值，但值得进一步探索。

残差服从正态分布吗？为了测试这一点，我们需要第二个图，一个分位数-分位数(Q-Q)图，其理论分位数由正态分布创建。Statsmodels 有一个 qqplot [函数](http://www.statsmodels.org/dev/generated/statsmodels.graphics.gofplots.qqplot.html)，但是很难注释和定制成一个基本的 R 样式图形。不用担心，构建 Q-Q 图相对简单。

我们可以从 stats.probplot()函数中提取理论分位数。我在这里使用了内部学生化残差，因为第三和第四个图形需要它们，但是如果您愿意，也可以使用原始残差。然后，我们将绘制学生化残差与理论分位数的关系图，并添加一条 1:1 的直线进行直观比较。注释是学生化残差中绝对值最大的三个点。

![](img/c2ba3a2d95a0202d67d65a8c05ad09a0.png)

在这里，我们可以看到残差通常都遵循 1:1 线，表明它们可能来自正态分布。虽然难以阅读(就像在 base R 中一样，ah the memories ),但菲亚特 128、丰田卡罗拉和克莱斯勒 Imperial 在学生化残差中表现突出，并且似乎也偏离了理论分位数线。

现在我们可以用标度-位置图来检验同质假设。基数 R 绘制了“标准化残差”与拟合值的平方根。“标准化残差”有点模糊，所以在网上搜索后，发现“标准化残差”实际上是[内部学生化残差](https://stats.stackexchange.com/questions/52522/standardized-residuals-in-rs-lm-output)。如上所述，从拟合模型中提取内部学生化残差非常简单。之后，我们将得到它们绝对值的平方根，然后绘制变换后的残差与拟合值的关系图。如果图中的散布在拟合值的整个范围内是一致的，那么我们可以有把握地假设数据符合同质假设。我已经注释了平方根变换学生化残差的三个最大值。

![](img/95108af3bc70374b6017a28782b7f656.png)

在这种情况下，洛斯平滑器似乎有上升趋势。这可能是异方差的表现。如果我去掉克莱斯勒帝国点，这种异方差可能会更严重，所以这个假设在我们的模型中可能会被违反。

最后，我构建了残差与杠杆图。响应变量也是内部学生化残差。这里的 x 轴是杠杆，通过 OLS 帽矩阵的对角线确定。这里棘手的部分是添加厨师距离的线条(见[这里](https://emredjan.github.io/blog/2017/07/11/emulating-r-plots-in-python/)关于如何在 seaborn 构建这些情节)。注释是具有最大绝对值的前三个学生化残差。

![](img/632176408930fa0683caeccceab13cfe.png)

该图中有一些证据表明，克莱斯勒帝国对该车型有着非同寻常的巨大影响。这是有意义的，因为它是拟合值的可能范围的最小边缘处的异常值。

如果您对自己创建这些诊断图感兴趣，我创建了一个简单的模块(OLSplots.py)来复制上面显示的图。我在我的 github [库](https://github.com/j-sadowski/FromRtoPython.git)上托管了这个模块和 Jupyter 笔记本来进行这个分析。

一些准则:

1.  如果您对回归使用测试训练分割，则应该在测试模型之前，对已训练的回归模型运行诊断图。
2.  当前的模块不能处理在 sklearn 中完成的回归，但是它们应该相对容易在以后的阶段被合并。
3.  OLSplots.py 模块目前没有内置的错误消息，所以您需要自己进行调试。也就是说，如果你将一个拟合的 OLS 模型从 statsmodel 传递给任何函数，它都应该工作正常。
4.  文本注释是用于构建 OLS 模型的熊猫数据框架的索引值，因此它们也适用于数值。

感谢阅读。暂时就这样吧！

进一步阅读

1.  [https://towards data science . com/linear-regression-on-Boston-housing-dataset-f 409 b 7 E4 a 155](/linear-regression-on-boston-housing-dataset-f409b7e4a155)
2.  [https://github.com/iamaziz/PyDataset](https://github.com/iamaziz/PyDataset)
3.  [https://emredjan . github . io/blog/2017/07/11/emiling-r-plots-in-python/](https://emredjan.github.io/blog/2017/07/11/emulating-r-plots-in-python/)
4.  [https://data.library.virginia.edu/diagnostic-plots/](https://data.library.virginia.edu/diagnostic-plots/)
5.  [https://stats . stack exchange . com/questions/52522/standardized-residuals-in-RS-lm-output](https://stats.stackexchange.com/questions/52522/standardized-residuals-in-rs-lm-output)