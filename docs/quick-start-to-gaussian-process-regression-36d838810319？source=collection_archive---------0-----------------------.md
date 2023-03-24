# 高斯过程回归快速入门

> 原文：<https://towardsdatascience.com/quick-start-to-gaussian-process-regression-36d838810319?source=collection_archive---------0----------------------->

## 理解高斯过程回归(GPR)和使用 scikit-learn 的 GPR 包的快速指南

高斯过程回归(GPR)是一种非参数的贝叶斯回归方法，正在机器学习领域掀起波澜。GPR 有几个好处，可以很好地处理小数据集，并且能够提供预测的不确定性测量。

![](img/cbefb0e16d51274ed562dd7eafb29a9f.png)

## 背景

与许多流行的监督机器学习算法不同，这些算法学习函数中每个参数的精确值，贝叶斯方法推断所有可能值的概率分布。让我们假设一个线性函数: *y=wx+ϵ* 。贝叶斯方法的工作原理是通过指定参数的*、 *p(w)、*、 *w* ，并使用贝叶斯规则基于证据(*即*观察数据)重新定位概率:*

*![](img/90e1b2e230a2fa4b75fe53582c67b4b3.png)*

*Calculating posterior distribution with Bayes’ Rule [1]*

*更新后的分布 *p(w|y，X)* ，称为 ***后验分布*** ，因此结合了来自先验分布和数据集的信息。为了得到在看不见的感兴趣点的预测， *x** ， ***预测分布*** 可以通过用它们的计算后验分布加权所有可能的预测来计算[1]:*

*![](img/ed2dd66f2c1adaea94f132db861c5075.png)*

*Calculating predictive distribution, f* is prediction label, x* is test observation [1]*

*为了便于积分，通常假设先验和似然是高斯型的。使用该假设并求解预测分布，我们得到高斯分布，从中我们可以使用其均值获得点预测，并使用其方差获得不确定性量化。*

****高斯过程回归*** 是非参数的(*即*不受函数形式的限制)，因此 GPR 不是计算特定函数参数的概率分布，而是计算所有符合数据的容许函数的概率分布。然而，与上面类似，我们指定一个先验(在函数空间上)，使用训练数据计算后验，并计算我们感兴趣点的预测后验分布。*

*有几个用于高效实现高斯过程回归的库(*例如* scikit-learn、Gpytorch、GPy)，但为了简单起见，本指南将使用 scikit-learn 的高斯过程包[2]。*

```
***import** sklearn.gaussian_process **as** gp*
```

*在 GPR 中，我们首先假设在 之前有一个 ***高斯过程，这可以使用均值函数*【m(x)】*和协方差函数 *k(x，x’)*来指定:****

*![](img/70bc624cb6a4000f4e36b2fc66c6cfae.png)*

*Labels drawn from Gaussian process with mean function, m, and covariance function, k [1]*

*更具体地说，高斯过程类似于无限维多元高斯分布，其中数据集的任何标注集合都是联合高斯分布。在这个 GP 先验中，我们可以通过选择均值和协方差函数来结合关于函数空间的先验知识。通过将标签分布和噪声分布相加，我们还可以容易地将独立、同分布的*(I . I . d)**【ϵ∨*n*(0，σ)*合并到标签中。***

## ***资料组***

***数据集由观察值、 *X、*及其标签、 *y、s* 组成，分为“训练”和“测试”子集:***

```
**# X_tr <-- training observations [# points, # features]
# y_tr <-- training labels [# points]
# X_te <-- test observations [# points, # features]
# y_te <-- test labels [# points]**
```

**根据高斯过程先验，训练点和测试点的集合是联合多元高斯分布的，因此我们可以这样写它们的分布[1]:**

**![](img/ac8d2f31c559f18c1d37de6d804fec81.png)**

**GP prior rewritten: multivariate distribution of training and testing points**

**这里，K 是协方差核矩阵，其中它的项对应于在观测值处评估的协方差函数。以这种方式编写，我们可以利用训练子集来执行模型选择。**

## **型号选择**

**在 ***模型选择*** 期间，选择并调整 GP 先验中的均值函数和协方差核函数的形式。均值函数通常是常数，或者为零，或者是训练数据集的均值。协方差核函数有很多选择:它可以有很多形式，只要它遵循一个核的性质(*即*半正定和对称)。一些常见的核函数包括常数、线性、平方指数和 Matern 核，以及多个核的组合。**

**一种流行的核是常数核与径向基函数(RBF)核的组合，其针对函数的平滑性进行编码(*即*空间中输入的相似性对应于输出的相似性):**

**![](img/260b5ea5b9004f3e8d2d7e04a9586b38.png)**

**Constant times RBF kernel function**

**这个内核有两个超参数:信号方差σ和长度尺度 *l* 。在 scikit-learn 中，我们可以从各种内核中进行选择，并指定它们的超参数的初始值和界限。**

```
**kernel = gp.kernels.ConstantKernel(1.0, (1e-1, 1e3)) * gp.kernels.RBF(10.0, (1e-3, 1e3))**
```

**在指定了内核函数之后，我们现在可以在 scikit-learn 中为 GP 模型指定其他选择。例如，alpha 是标签上 *i.i.d.* 噪声的方差，normalize_y 是指常数均值函数-如果为假，则为零；如果为真，则训练数据均值。**

```
**model = gp.GaussianProcessRegressor(kernel=kernel, n_restarts_optimizer=10, alpha=0.1, normalize_y=True)**
```

**调整协方差核函数的超参数的一种流行方法是最大化训练数据的对数边际似然。基于梯度的优化器通常用于提高效率；如果上面未指定，默认优化器是' *fmin_l_bfgs_b'* 。因为对数边际可能性不一定是凸的，所以使用具有不同初始化的优化器的多次重启(*n _ restructs _ optimizer)。***

```
**model.fit(X_tr, y_tr)
params = model.kernel_.get_params()**
```

**如果需要，可以通过调用 *model.kernel_ 来获得内核函数的调整后的超参数。get_params()* 。**

## **推理**

**为了计算预测的后验分布，数据和试验观察被排除在后验分布之外。同样，因为我们选择了高斯过程先验，计算预测分布是易处理的，并导致可以完全由均值和协方差描述的正态分布[1]:**

**![](img/cae551be154b9814086bdae4baeb6bfd.png)**

**Predictive posterior distribution for GPR [1]**

**预测是均值 f_bar*，方差可以从协方差矩阵σ*的对角线获得。请注意，计算平均值和方差需要对 K 矩阵求逆，这与训练点数的立方成比例。通过 sci-kit learn 的 GPR 预测功能，推理很容易实现。**

```
**y_pred, std = model.predict(X_te, return_std=True)**
```

**![](img/2ba2ec82cadd05978d485fd824d3a016.png)**

**注意，返回的是标准差，但是如果 *return_cov=True，则可以返回整个协方差矩阵。*然后可以计算 95%的置信区间:高斯分布的标准偏差的 1.96 倍。**

**为了测量回归模型对测试观测值的性能，我们可以计算预测值的均方误差(MSE)。**

```
**MSE = ((y_pred-y_te)**2).mean()**
```

****参考文献:****

**[1] Rasmussen，C. E .，& Williams，C. K. I .，[机器学习的高斯过程](http://www.gaussianprocess.org/gpml/chapters/RW2.pdf) (2016)，麻省理工学院出版社**

**[2]f . Pedregosa，g . Varoquaux，a . gram fort，Michel，v . Thirion，b . Grisel，o .等人。艾尔。，[sci kit-learn:python 中的机器学习](https://scikit-learn.org/0.17/modules/gaussian_process.html) (2011)，《机器学习研究杂志》**