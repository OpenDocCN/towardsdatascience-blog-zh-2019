# 数据科学的理论基础—我应该关注还是仅仅关注动手技能？

> 原文：<https://towardsdatascience.com/theoretical-foundations-of-data-science-should-i-care-or-simply-focus-on-hands-on-skills-c53fb0caba66?source=collection_archive---------4----------------------->

## 数据科学和机器学习所需的基本数学和理论技能

![](img/bc671ee232ab5512dc8149af7a3bd010.png)

数据科学是一个非常注重实践的领域。数据科学需要扎实的数学和编程基础。作为一名数据科学家，您必须了解数据科学的理论和数学基础，以便能够构建具有真实应用程序的可靠模型。

在数据科学和机器学习中，数学技能和编程技能一样重要。有很多好的软件包可以用来构建预测模型。一些最常见的描述性和预测性分析包包括

*   Ggplot2
*   Matplotlib
*   海生的
*   Sci-kit 学习包
*   插入符号包
*   张量流
*   PyTouch 包
*   Keras 包

重要的是，在使用这些包之前，你要掌握数据科学的基础知识，这样你就不会把这些包简单地当作黑盒工具来使用。

理解机器学习模型功能的一种方法是让你理解每个模型背后的理论和数学基础。作为一名数据科学家，您构建可应用于现实世界问题的可靠高效模型的能力取决于您的数学技能有多好。

本文将讨论一些对数据科学实践至关重要的理论和数学基础。

# (一)统计和概率

统计和概率用于特征的可视化、数据预处理、特征转换、数据插补、降维、特征工程、模型评估等。以下是您需要熟悉的主题:

1.  平均
2.  中位数
3.  方式
4.  标准偏差/方差
5.  相关系数和协方差矩阵
6.  概率分布(二项式、泊松、正态)
7.  p 值
8.  Baye 定理(精确度、召回率、阳性预测值、阴性预测值、混淆矩阵、ROC 曲线)
9.  中心极限定理
10.  R2 分数
11.  均方误差
12.  A/B 测试
13.  蒙特 卡罗模拟

例如，**表示**、**中值、**和**模式**用于显示给定数据集的汇总统计数据。它们也用于数据插补(平均数插补、中位数插补和众数插补)。

**相关系数**和**协方差矩阵**用于研究数据集中各种特征之间的关系，也可用于特征选择和降维。

**概率分布**用于特征缩放，例如特征的标准化和规范化。概率分布和蒙特卡罗模拟也用于模拟数据。例如，如果样本数据根据具有已知平均值和标准偏差的正态分布分布，则可以使用正态分布的随机数生成器来生成群体数据集。

**贝叶斯定理**用于模型测试和评估，以及计算精度得分。

**中心极限定理(CLT)** 是统计学和数据科学中最重要的定理之一。CLT 认为，使用具有大量观测数据的样本数据集进行建模是有利的，因为样本越大，越接近总体。从这里了解更多关于 CLT 的知识: [**利用蒙特卡罗模拟证明中心极限定理**](/proof-of-central-limit-theorem-using-monte-carlo-simulation-34925a7bc64a) 。

**R2 分数**和 **MSE** 用于模型评估。下面是一篇用 R_2 分数和 MSE 进行模型评估的文章:

[**从零开始构建机器学习推荐模型**](https://medium.com/towards-artificial-intelligence/machine-learning-model-for-recommending-the-crew-size-for-cruise-ship-buyers-6dd478ad9900) 。

# ㈡多变量微积分

大多数机器学习模型是用具有几个特征或预测器的数据集构建的。因此，熟悉多变量微积分对于建立机器学习模型极其重要。以下是您需要熟悉的主题:

1.  多元函数
2.  导数和梯度
3.  阶跃函数、Sigmoid 函数、Logit 函数、ReLU(校正线性单位)函数
4.  价值函数
5.  功能绘图
6.  函数的最小值和最大值

关于如何在机器学习过程中使用**多变量微积分**的示例，请参见以下示例:

[**建立你的第一个机器学习模型:线性回归估计器**](/building-your-first-machine-learning-model-linear-regression-estimator-ba86450c4d24)

[**采用最小二乘法的基本感知器模型**](https://medium.com/towards-artificial-intelligence/basic-perceptron-model-using-least-squares-method-17900e0d1eff)

# ㈢线性代数

线性代数是机器学习中最重要的数学技能。数据集被表示为矩阵。线性代数用于数据预处理、数据转换、降维以及模型评估。

以下是您需要熟悉的主题:

1.  向量
2.  向量的范数
3.  矩阵
4.  矩阵的转置
5.  矩阵的逆矩阵
6.  矩阵的行列式
7.  点积
8.  本征值
9.  特征向量

例如，**协方差矩阵**是一个非常有用的矩阵，可以显示特征之间的相关性。使用协方差矩阵，可以选择将哪些特征用作预测变量。下面举例说明协方差矩阵如何用于特征选择和降维: [**利用协方差矩阵进行特征选择和降维**](https://medium.com/towards-artificial-intelligence/feature-selection-and-dimensionality-reduction-using-covariance-matrix-plot-b4c7498abd07) 。

其他高级的特征选择和降维方法有**主成分分析** (PCA)，以及**线性判别分析** (LDA)。要理解 PCA 和 LDA 是如何工作的，你需要理解线性代数的主题，比如转置一个矩阵；矩阵的逆矩阵；矩阵的行列式；点积；特征值；和特征向量。以下是 LDA 和 PCA 的一些实现:

[**机器学习:通过主成分分析进行降维**](https://medium.com/towards-artificial-intelligence/machine-learning-dimensionality-reduction-via-principal-component-analysis-1bdc77462831)

[**机器学习:通过线性判别分析进行降维**](https://medium.com/towards-artificial-intelligence/machine-learning-dimensionality-reduction-via-linear-discriminant-analysis-cc96b49d2757)

# ㈣优化方法

大多数机器学习算法通过最小化目标函数来执行预测建模，从而学习为了获得预测标签而必须应用于测试数据的权重。以下是您需要熟悉的主题:

1.  成本函数/目标函数
2.  似然函数
3.  误差函数
4.  梯度下降算法及其变体(如随机梯度下降算法)

数据科学和机器学习中如何使用优化方法的例子可以在这里找到: [**机器学习:使用梯度下降的 Python 线性回归估计器**](https://medium.com/towards-artificial-intelligence/machine-leaning-python-linear-regression-estimator-using-gradient-descent-b0b2c496e463) 。

总之，我们已经讨论了数据科学和机器学习中需要的基本数学和理论技能。有几门免费的在线课程会教你在数据科学中需要的必要的数学技能。作为一名数据科学家，一定要记住，数据科学的理论基础对于构建高效可靠的模型至关重要。

# 参考

1.  [在具有高度相关特征的数据集上训练机器学习模型](https://medium.com/towards-artificial-intelligence/training-a-machine-learning-model-on-a-dataset-with-highly-correlated-features-debddf5b2e34)。
2.  [使用协方差矩阵图进行特征选择和降维](https://medium.com/towards-artificial-intelligence/feature-selection-and-dimensionality-reduction-using-covariance-matrix-plot-b4c7498abd07)。
3.  拉什卡、塞巴斯蒂安和瓦希德·米尔贾利利**。** *Python 机器学习，第二版*。帕克特出版社，2017 年。
4.  Benjamin O. Tayo，*预测船只船员规模的机器学习模型*，[https://github . com/bot 13956/ML _ Model _ for _ Predicting _ Ships _ Crew _ Size](https://github.com/bot13956/ML_Model_for_Predicting_Ships_Crew_Size)。