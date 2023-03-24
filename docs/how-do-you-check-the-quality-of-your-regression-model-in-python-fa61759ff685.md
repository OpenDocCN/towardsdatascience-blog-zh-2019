# 如何在 Python 中检查你的回归模型的质量？

> 原文：<https://towardsdatascience.com/how-do-you-check-the-quality-of-your-regression-model-in-python-fa61759ff685?source=collection_archive---------2----------------------->

## 线性回归在统计学习领域根深蒂固，因此必须检查模型的“拟合优度”。本文向您展示了在 Python 生态系统中完成这项任务的基本步骤。

![](img/8bd53684cbf23b3a578af6654c5566e5.png)

# 为什么它很重要(为什么你可能会错过它)

尽管对最新深度神经网络架构的复杂性和 [xgboost 在 Kaggle 竞赛](http://blog.kaggle.com/tag/xgboost/)上的惊人能力有很多谈论和吹毛求疵，但对于行业的很大一部分来说，使用[数据驱动分析](https://www.oreilly.com/library/view/creating-a-data-driven/9781491916902/ch01.html)和机器学习(ML)技术，[回归仍然是他们日常使用的首选](https://www.surveygizmo.com/resources/blog/regression-analysis/)。

参见 2018-19 年的 KDnuggets 民意调查结果(作者[马修·梅奥](https://medium.com/u/a0bc63d95eb0?source=post_page-----fa61759ff685--------------------------------))。

[](https://www.kdnuggets.com/2019/04/top-data-science-machine-learning-methods-2018-2019.html) [## 2018 年、2019 年使用的顶级数据科学和机器学习方法

### 在最新的 KDnuggets 民意调查中，读者被问到:你对哪些数据科学/机器学习方法和算法…

www.kdnuggets.com](https://www.kdnuggets.com/2019/04/top-data-science-machine-learning-methods-2018-2019.html) 

回归技术有多种形式——线性、非线性、泊松、基于树的——但核心思想几乎保持一致，可以应用于金融、医疗保健、服务业、制造业、农业等领域的各种预测分析问题。

线性回归是一项基本技术，[它深深植根于久经考验的统计学习和推理理论](/linear-regression-using-python-b136c91bf0a2)，并为现代数据科学管道中使用的所有基于回归的算法提供支持。

然而，**线性回归模型的成功还取决于一些基本假设**关于它试图建模的基础数据的性质。要简单直观地理解这些假设，

[](https://www.jmp.com/en_us/statistics-knowledge-portal/what-is-regression/simple-linear-regression-assumptions.html) [## 回归模型假设

### 当我们使用线性回归来模拟反应和预测因子之间的关系时，我们做了一些假设…

www.jmp.com](https://www.jmp.com/en_us/statistics-knowledge-portal/what-is-regression/simple-linear-regression-assumptions.html) 

因此，通过验证这些假设是否“合理地”得到满足来检查您的线性回归模型的质量是极其重要的(通常使用可视分析方法来检查假设，这取决于解释)。

> 问题在于，检查模型质量通常是数据科学任务流中优先级较低的一个方面，其他优先级占主导地位，如预测、缩放、部署和模型调整。

这个论断听起来是不是太大胆了？有一个简单的测试。

在一个行业标准的基于 Python 的数据科学堆栈中，你有多少次使用 **Pandas、NumPy** 、 **Scikit-learn** ，甚至 **PostgreSQL** 进行数据获取、争论、可视化，并最终构建和调整你的 ML 模型？我想，很多次了吧？

现在，你已经多少次使用 [**statsmodels** 库](http://www.statsmodels.org/devel/index.html)通过运行[拟合优度测试](https://www.statisticshowto.datasciencecentral.com/goodness-of-fit-test/)来检查模型了？

在基于 **Python 的数据科学学习课程**中，这样做是很常见的，

![](img/d2f125944611fa370b8ed9075215fc59.png)

“是不是少了点什么”这个问题的答案是肯定的！

![](img/ad89dbacbeb6c4b56911417e6f82883a.png)

通常，有很多关于[正则化](/regularization-in-machine-learning-76441ddcf99a)、[偏差-方差权衡](http://scott.fortmann-roe.com/docs/BiasVariance.html)或可扩展性(学习和复杂性曲线)图的讨论。但是，围绕以下情节和列表有足够的讨论吗？

*   残差与预测变量图
*   拟合与残差图
*   归一化残差的直方图
*   归一化残差的 Q-Q 图
*   残差的夏皮罗-维尔克正态性检验
*   残差的库克距离图
*   预测特征的方差膨胀因子(VIF)

很明显，对于机器学习管道的这一部分，你必须戴上统计学家的帽子，而不仅仅是数据挖掘专家。

![](img/faf65902d626766464ef733988fe90a6.png)

# Scikit-learn 的问题

可以有把握地假设，大多数由统计学家转变为数据科学家的 T21 定期对他们的回归模型进行拟合优度测试。

但是，对于数据驱动的建模，许多年轻的数据科学家和分析师严重依赖于像 **Scikit-learn** 这样的以 ML 为中心的包，尽管这些包是一个令人敬畏的库，并且实际上是机器学习和预测任务的[银弹](https://medium.com/analytics-vidhya/scikit-learn-a-silver-bullet-for-basic-machine-learning-13c7d8b248ee)，但是它们不支持基于标准统计测试的简单快速的模型质量评估。

> 因此，除了使用像 Scikit-learn 这样的以 ML 为中心的库之外，良好的数据科学管道还必须包括一些标准化的代码集，以使用统计测试来评估模型的质量。

在本文中，我们展示了这样一个多元线性回归问题的标准评估集。我们将使用`statsmodels`库进行回归建模和统计测试。

![](img/d2140b2e440e8f8400c0dd8f9293fe12.png)

# 线性回归假设和关键视觉测试的简要概述

## 假设

线性回归模型需要测试的四个关键假设是:

*   **线性**:因变量的期望值是每个独立变量的线性函数，保持其他变量不变(注意，这并不限制您使用独立变量的非线性变换，即您仍然可以建模 *f(x) = ax + bx + c* ，使用 *x* 和 *x* 作为预测变量。
*   **独立性**:误差(拟合模型的残差)相互独立。
*   **同方差** **(恒定方差)**:误差的方差相对于预测变量或响应是恒定的。
*   **正态**:误差由正态分布产生(均值和方差未知，可从数据中估计)。注意，与上面的三个不同，这不是执行线性回归的必要条件。然而，如果不满足这一假设，就不能很容易地计算出所谓的“置信度”或“预测”区间，因为无法使用与高斯分布相对应的众所周知的解析表达式。

对于多元线性回归，从统计推断的角度判断**多重共线性**也很关键。这种假设假设预测变量之间的线性相关性最小或没有线性相关性。

**异常值**也可能是一个影响模型质量的问题，因为它对估计的模型参数具有不成比例的影响。

这是一个视觉回顾，

![](img/70001a090395eb1ce8152e2e18635119.png)

## 有哪些情节可以查？

因此，误差项非常重要。

> 但是有一个坏消息。不管我们有多少数据，我们永远无法知道真正的错误。我们只能对产生数据的分布进行估计和推断。

因此，**真实误差的代理是残差**，残差就是观测值和拟合值之差。

底线-我们需要绘制残差图，检查它们的随机性质、方差和分布，以评估模型质量。**这是线性模型拟合优度估计所需的视觉分析**。

除此之外，多重共线性可以通过相关矩阵和热图进行检查，数据中的异常值(残差)可以通过所谓的**库克距离图**进行检查。

# 回归模型质量评估示例

这个例子[的完整代码报告可以在作者的 Github](https://github.com/tirthajyoti/Machine-Learning-with-Python/blob/master/Regression/Regression_Diagnostics.ipynb) 中找到。

我们正在使用来自 UCI ML 门户网站的[混凝土抗压强度预测](https://archive.ics.uci.edu/ml/datasets/Concrete+Compressive+Strength)问题。混凝土抗压强度是龄期和成分的高度复杂的函数。我们能从这些参数的测量值预测强度吗？

## 检查线性的变量散点图

我们可以简单地检查散点图，以便直观地检查线性假设。

![](img/fdc55e07f45ee85f26fb2e4ec53eba75.png)

## 用于检查多重共线性的成对散点图和相关热图

我们可以使用 **seaborn** 库中的 [**pairplot** 函数来绘制所有组合的成对散点图。](https://seaborn.pydata.org/generated/seaborn.pairplot.html)

![](img/18d3588172b781720eecd7377d78b123.png)

此外，如果数据加载到 Pandas 中，我们可以很容易地计算关联矩阵，并将其传递到 statsmodels 的[特殊绘图功能，以热图的形式显示关联。](https://www.statsmodels.org/stable/generated/statsmodels.graphics.correlation.plot_corr.html#statsmodels.graphics.correlation.plot_corr)

![](img/c748bd01331c9665397f39db812661a9.png)

## 使用 statsmodel.ols()函数进行模型拟合

主模型拟合是使用 statsmodels 完成的。OLS 方法。这是一个令人惊叹的线性模型拟合实用程序，感觉非常像 R 中强大的“lm”函数。最棒的是，它接受 R 风格的公式来构建完整或部分模型(即涉及所有或部分预测变量)。

你可能会问，在大数据时代，为什么要创建一个局部模型而不把所有数据都放进去呢？这是因为数据中可能存在混杂或隐藏的偏差，只能通过 [**控制某些因素**](https://stats.stackexchange.com/questions/78816/how-do-you-control-for-a-factor-variable) 来解决。

在任何情况下，通过该模型拟合的模型摘要已经提供了关于该模型的丰富的统计信息，例如对应于所有预测变量的 t 统计量和 p 值、R 平方和调整的 R 平方、AIC 和 BIC 等。

![](img/05adc87563adbe1ee2edb142214d9f7c.png)

## 残差与预测变量图

接下来，我们可以绘制残差与每个预测变量的关系图，以寻找独立性假设。**如果残差围绕零个 x 轴均匀随机分布，并且不形成特定的集群**，则假设成立。在这个特殊的问题中，我们观察到一些集群。

![](img/ed1895f7b0091339be18728c3be5c21e.png)

## 拟合与残差图，用于检查同质性

当我们绘制拟合响应值(根据模型)与残差的关系时，我们清楚地看到残差的**方差随着响应变量幅度**的增加而增加。因此，该问题不考虑同质性，可能需要某种变量变换来提高模型质量。

![](img/bc3fd4ab8d8757d7553f0f9bb6edb72b.png)

## 归一化残差的直方图和 Q-Q 图

为了检查数据生成过程的正态性假设，我们可以简单地绘制归一化残差的直方图和 Q-Q 图。

![](img/e61bbc0109ea1499ed3320a797e6e2cd.png)

此外，我们可以对残差进行夏皮罗-维尔克检验来检查正态性。

## 使用库克距离图的异常值检测

库克距离本质上衡量的是删除一个给定观察的效果。需要仔细检查 Cook 距离较大的点是否为潜在的异常值。我们可以使用来自 statsmodels 的特殊[异常值影响类来绘制厨师的距离。](http://www.statsmodels.org/devel/generated/statsmodels.stats.outliers_influence.OLSInfluence.summary_frame.html)

![](img/558b94338d3508d9b72013236b47b554.png)

## 方差影响因素

此数据集的 OLS 模型摘要显示了多重共线性的警告。但是如何检查是哪些因素造成的呢？

我们可以计算每个预测变量的[方差影响因子](https://en.wikipedia.org/wiki/Variance_inflation_factor)。它是有多个项的模型的方差除以只有一个项的模型的方差的比率。同样，我们利用了 statsmodels 中的[特殊异常值影响类](https://www.statsmodels.org/stable/generated/statsmodels.stats.outliers_influence.variance_inflation_factor.html)。

![](img/7a031bda6a19b20514d28c0ef35547b2.png)

## 其他残留诊断

Statsmodels 有各种各样的其他诊断测试来检查模型质量。你可以看看这几页。

*   [剩余诊断测试](https://www.statsmodels.org/stable/stats.html#module-statsmodels.stats.stattools)
*   [拟合优度测试](https://www.statsmodels.org/stable/stats.html#goodness-of-fit-tests-and-measures)

# 总结和思考

在本文中，我们介绍了如何在线性回归中添加**用于模型质量评估的基本可视化分析**——各种残差图、正态性测试和多重共线性检查。

人们甚至可以考虑创建一个简单的函数套件，能够接受 scikit-learn 类型的估计器并生成这些图，以便数据科学家快速检查模型质量。

目前，尽管 scikit-learn 没有用于模型质量评估的详细统计测试或绘图功能，但 [Yellowbrick](https://www.scikit-yb.org/en/latest/) 是一个有前途的 Python 库，它可以在 scikit-learn 对象上添加直观的可视化功能。我们希望在不久的将来，统计测试可以直接添加到 scikit-learn ML 估计量中。

*喜欢这篇文章吗？成为* [***中等成员***](https://medium.com/@tirthajyoti/membership) *继续* ***无限制学习*** *。如果你使用下面的链接，我会收到你的一部分会员费，* ***而不增加你的额外费用*** *。*

[](https://medium.com/@tirthajyoti/membership) [## 通过我的推荐链接加入 Medium—Tirthajyoti Sarkar

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

medium.com](https://medium.com/@tirthajyoti/membership)