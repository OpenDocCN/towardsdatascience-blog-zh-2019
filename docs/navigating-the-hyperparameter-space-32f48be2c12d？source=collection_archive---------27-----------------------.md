# 导航超参数空间

> 原文：<https://towardsdatascience.com/navigating-the-hyperparameter-space-32f48be2c12d?source=collection_archive---------27----------------------->

开始应用机器学习有时会令人困惑。有许多术语需要学习，其中许多术语的使用并不一致，尤其是从一个研究领域到另一个领域——其中一个术语可能意味着两个不同领域中的两种不同事物。

今天我们来讲一个这样的术语:**模型超参数**。

# 参数 vs .超参数:有什么区别？

让我们从最基本的区别开始，即参数和超参数之间的区别。

**模型参数**是模型内部的属性，并且是在训练阶段学习到的，并且是模型进行预测所需要的。

另一方面， [**模型超参数**不能在训练期间学习，而是预先设定](https://machinelearningmastery.com/difference-between-a-parameter-and-a-hyperparameter/)。它们的值可以使用经验法则或试错法来设置；但事实上，我们无法知道特定问题的最佳模型超参数— **，因此我们调整超参数以发现具有最佳性能的模型参数**。

> 在某种程度上，超参数是我们可以用来调整模型的旋钮。

# 超参数调谐

也叫**超参数优化**，是 [*为一个特定的学习算法*](https://en.wikipedia.org/wiki/Hyperparameter_optimization) 寻找性能最好的超参数集合的问题。

这是一个重要的步骤，因为**使用正确的超参数将会发现模型的参数，从而产生最巧妙的预测**；这也是我们最终想要的。

![](img/6c412631ef2552bb57ff9c6c2ec192d4.png)

这种性质的优化问题有三个基本组成部分:(1)一个**目标函数**，这是我们想要最大化或最小化的模型的主要目的；(2)控制目标函数的一组变量**；(3)一组**约束**，允许变量取某些值，同时排除其他值。**

> 因此，优化问题是找到最大化或最小化目标函数同时满足约束集的一组变量值。

解决优化问题有不同的方法，也有许多实现这些方法的开源软件选项。在本文中，我们将探索**网格搜索、随机搜索和贝叶斯优化**。

# 网格搜索

也被称为**参数扫描** *，*它被认为是最简单的超参数优化算法。它包括彻底搜索手动指定的一组参数，这意味着为指定子集的所有可能组合训练模型。

如果模型可以快速训练，这种方法可能是一个很好的选择，否则，模型将需要很长时间来训练。这就是为什么[不认为使用**网格搜索**来调整神经网络](http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html)的超参数是最佳实践。

一个流行的实现是 [*Scikit-Learn*](https://scikit-learn.org/stable/modules/grid_search.html#grid-search) 的 [*GridSearchCV*](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html#sklearn.model_selection.GridSearchCV) 。其他实现还有 [*Talos*](https://github.com/autonomio/talos) ，一个包括[](/hyperparameter-optimization-with-keras-b82e6364ca53)*用于 [*Keras*](https://keras.io/) 和 [*H2O*](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/welcome.html) 的库，一个为不同的机器学习算法提供[](http://docs.h2o.ai/h2o/latest-stable/h2o-docs/grid-search.html)**网格搜索实现的平台。***

# ***随机搜索***

*****随机搜索**背后的思想与**网格搜索**非常相似，除了它不是穷尽搜索手动指定参数集的所有可能组合，而是[选择组合的随机子集来尝试](http://neupy.com/2016/12/17/hyperparameter_optimization_for_neural_networks.html)。***

> ***这种方法大大减少了运行超参数优化的时间，但是有一个警告:没有[保证会找到最优的超参数集](/hyperparameter-tuning-c5619e7e6624)。***

***一些流行的实现有 [*Scikit-Learn*](https://scikit-learn.org/stable/index.html) 的[*RandomizedSearchCV*](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html#sklearn.model_selection.RandomizedSearchCV)[*hyperpt*](https://github.com/hyperopt/hyperopt)和 [*Talos*](https://github.com/autonomio/talos) 。***

# **贝叶斯优化**

**我们已经确定**随机搜索**和**网格搜索**建立在相似的想法上，它们的另一个共同点是**不使用以前的结果来通知每个超参数子集的评估**，并且反过来**、**它们**花费时间评估不到最优选项**。**

**相比之下，**贝叶斯优化**会跟踪过去的评估结果，使其成为一种自适应的优化方法。寻找满足目标函数的变量的值是一种 [*强有力的策略，该目标函数评估*](https://arxiv.org/pdf/1012.2599.pdf) 是昂贵的。在某种程度上，当我们需要**最小化我们在试图 [**找到全局最优**](https://arxiv.org/pdf/1012.2599.pdf) 时所采取的步骤**的数量时，贝叶斯技术是最有用的。**

> **[贝叶斯优化结合了关于目标函数的先验信念(***【f】***)，并使用从***【f】***中提取的样本更新先验，以获得更好地逼近的后验概率(***f)***](http://krasserm.github.io/2018/03/21/bayesian-optimization/)***——***马丁·克拉瑟(2018)贝叶斯优化。**

**为了实现这一点，**贝叶斯优化**利用了另外两个概念:一个**代理模型**和一个**获取函数**。第一个是指用于 [**逼近目标函数**](https://arxiv.org/pdf/1012.2599.pdf) 的概率模型，第二个是用于 [**确定采样域内的新位置**](https://arxiv.org/pdf/1012.2599.pdf) ，在该位置最有可能对当前最佳评估结果进行改进。这是**贝叶斯优化**模型效率背后的两个关键因素。**

****替代模型**有几种不同的选择，最常见的有 [**高斯过程、随机森林回归和树 Parzen 估计器**](https://app.sigopt.com/static/pdf/SigOpt_Bayesian_Optimization_Primer.pdf) 。至于采集函数，最常见的有[](https://arxiv.org/pdf/1012.2599.pdf)****[**最大概率改善，置信上限**](http://krasserm.github.io/2018/03/21/bayesian-optimization/) 。******

******一些流行的实现有[*Scikit-Optimize*](https://github.com/scikit-optimize/scikit-optimize)*[*贝叶斯优化*](https://scikit-optimize.github.io/notebooks/bayesian-optimization.html)[*SMAC*](https://github.com/automl/SMAC3)[*兰香*](https://github.com/JasperSnoek/spearmint)[*MOE*](https://github.com/Yelp/MOE)[*hyperpt*](https://github.com/hyperopt/hyperopt)。*******

# *****结论*****

*****超参数优化对于机器学习任务来说是一件大事。这是提高模型性能的重要一步，但不是实现这一目标的唯一策略。*****

*****在我们探索的所有方法中，不同的从业者和专家会根据他们的经验提倡不同的方法，但是他们中的许多人也会同意实际上[取决于数据](https://stats.stackexchange.com/questions/302891/hyper-parameters-tuning-random-search-vs-bayesian-optimization)、问题和其他项目考虑因素。最重要的一点是，这是寻找最佳表演模特的**关键步骤。*******