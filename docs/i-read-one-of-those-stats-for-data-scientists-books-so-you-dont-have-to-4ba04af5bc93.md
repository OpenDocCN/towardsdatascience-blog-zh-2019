# 我读了一本“数据科学家统计”的书，这样你就不用看了

> 原文：<https://towardsdatascience.com/i-read-one-of-those-stats-for-data-scientists-books-so-you-dont-have-to-4ba04af5bc93?source=collection_archive---------10----------------------->

![](img/a7b2fef1e173a7c053282c58d87799f4.png)

我目前是一家大型科技公司的数据科学家，每天在工作中运用我的专业知识，自然地发展我的技能。然而，我的一个怪癖是我喜欢边学习边做事。老实说，这是我最初进入数据科学的原因，但那是另一篇博客文章的主题。

在过去的几个月里，我在周末、飞机上，甚至在工作的时候，都在漫不经心地阅读数据科学家实用统计数据。**原因何在？我坚信我永远不会停止学习。**有时，重新引入一个概念、工具或方法这一简单的行为可以引发新的创造性方法来解决你自己可能没有遇到过的问题。

这篇博客旨在分享一些“灵光乍现的时刻”,并整理我在阅读这本书时的想法——这样你就不必这么做了。我还在适当的地方链接了进一步的阅读材料。

## **但在我透露所有秘密之前，让我快速告诉你为什么我认为你应该读这本书——以及其他数据科学书籍。**

在我漫不经心地读这本书的时候(还是非常漫不经心地读)，我在空白处记下了适用于正在讨论的概念的当前、过去和未来的项目。有多少次，我碰巧拿起书来阅读下一章，却发现一个伟大的想法来克服我目前工作中的一个障碍，这是非常可笑的。我不知道这是不是数据科学之神在对我微笑，但它确实改变了我的一些项目。

现在，另一个重要的免责声明—我并不是说书中讨论的任何概念，或者我将在下面强调的任何概念，都是突破性的数据科学技术，而不是我不熟悉的技术。事实上，书中几乎没有什么是开创性的“新消息”。相反，我是说，在日益拥挤的数据科学技术中，信息技术的出现是新思想的完美载体。

好了，序言够了；以下是我从阅读这本书中获得的一些重要收获。

# 探索性数据分析

*   [平均绝对偏差](https://en.wikipedia.org/wiki/Average_absolute_deviation)是测量标准偏差的替代方法。平均绝对偏差让我们了解平均值的平均变化。一个有用的应用是帮助估计抽样误差率。假设我们知道真实值是 X，那么误差(相对于 X 的平均绝对偏差)是如何随着样本的减少而增加的。
*   百分位数对于总结尾部特别有用。不要使用直方图，尝试绘制百分位值。还有其他创造性的方法来可视化分布。
*   [Winsorization](http://www.johnmyleswhite.com/notebook/2015/12/03/some-observations-on-winsorization-and-trimming/) 可以使汇总统计数据和可视化更清晰、更健壮。
*   通常情况下，你通过图形来测量[偏斜度和峰度](https://www.itl.nist.gov/div898/handbook/eda/section3/eda35b.htm)，而不是一个特定的指标。
*   对于探索性分析和可视化来说，将数值数据转换成分类数据有很多好处，可以降低数据的复杂性(和大小)。宁滨是这样做的一种方式。
*   [Spearman 的 rho](https://en.wikipedia.org/wiki/Spearman%27s_rank_correlation_coefficient) 或 [Kendall 的 tau](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient) 是基于秩的相关方法；它们对异常值更具鲁棒性，并且适用于较小的数据集。
*   当散点图有太多值时:绘制前采样，调整 alpha(透明度)，使用六边形显示六边形点的密度或轮廓(看起来像地形图)。
*   [列联表](https://en.wikipedia.org/wiki/Contingency_table)用于比较两个分类变量，最理想的是同时表示关系的计数和百分比。
*   调节和使用[网格图形](http://lattice.r-forge.r-project.org/Vignettes/src/lattice-intro/lattice-intro.pdf)是可视化多元关系的好方法。

# 数据和抽样分布

*   [抽样考虑](/sampling-techniques-a4e34111d808?gi=63001c0e891d)包括有无替换、群体代表性以及总体分布情况。
*   重要的是要记住向平均值的回归:极端的观察结果往往会跟随着更中心的观察结果。因此，精确的估计通常最好用置信区间来代替。
*   [Bootstrapping](https://en.wikipedia.org/wiki/Bootstrapping_(statistics)) 是从样本分析中获得置信区间的有效方法。相关的，bagging 是在模型上应用 bootstrapping 的另一种方式，这样您就可以看到模型参数有多少可变性。
*   当推理样本统计的分布时，正态分布很有帮助。
*   [QQ-Plot](https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot) 拟合是一种直观检查数据是否正态分布或“标准化”的方法。要标准化数据，或者将其与正态分布进行比较，可以减去平均值，然后除以标准差。
*   了解分布是很有用的，这样你就可以识别什么时候结果是偶然产生的。

# 统计实验和显著性检验

*   多重比较和错误发现率对于运行测试的数据科学家来说非常重要。 [Bonferroni 调整](https://en.wikipedia.org/wiki/Bonferroni_correction)降低 alpha 到 alpha /观察次数。
*   [ANOVA](https://en.wikipedia.org/wiki/Analysis_of_variance) 帮助您比较组间测试统计的差异。它帮助你衡量组间的差异是否是由于偶然。
*   [箱线图](https://en.wikipedia.org/wiki/Box_plot)是可视化指标和多个分类组之间关系的绝佳方式。
*   [卡方检验](https://en.wikipedia.org/wiki/Chi-squared_test)测量整个样本的分布变化程度。然后 p 值是根据概率计算的。
*   [费希尔精确检验](https://en.wikipedia.org/wiki/Fisher%27s_exact_test)是卡方检验的进一步版本，当样本数量非常少时可以使用。它依靠重采样来创建所有可能的重新排列(置换)，并计算观察到的结果有多大可能。
*   卡方检验和费希尔精确检验都可以用来计算所需的样本量，以获得具有统计学意义的结果。
*   [在运行测试之前，应计算功效和样本量](http://powerandsamplesize.com/)，以确定这些值之间的关系。例如，保持功率不变，最小尺寸效应越小，需要的样本尺寸就越大。

# 回归和预测

*   相关性量化了关联的强度，回归量化了关系的性质。
*   预测后，[残差](https://en.wikipedia.org/wiki/Errors_and_residuals)测量拟合(预测)值与实际真值的距离。
*   [R 的平方](https://en.wikipedia.org/wiki/Coefficient_of_determination)范围从 0 到 1，测量模型中数据的变化比例。
*   [特征重要性](https://christophm.github.io/interpretable-ml-book/feature-importance.html)和特征显著性是指建模的不同方面。特征重要性是指根据系数的大小对特征进行排序。要素显著性是指要素按 p 值大小排序。
*   [交叉验证](https://en.wikipedia.org/wiki/Cross-validation_(statistics))是一种通过预测和比较测试样本上的结果来确认在训练样本上训练的模型的有效性的方法。
*   选择模型时，最好使用[奥卡姆剃刀](https://en.wikipedia.org/wiki/Occam%27s_razor)的原则——在其他条件相同的情况下，应优先使用简单的模型，而不是复杂的模型。
*   “修剪”回归模型最常见的方法是逐步回归、向前和向后选择。
*   岭回归和套索回归通过降低系数而不是丢弃系数来惩罚模型。
*   另一种获得精确系数的方法是上下加权一些训练数据。
*   在数据范围之外推断回归模型是不好的做法。
*   如果您使用回归进行解释或分析特征重要性，请使用自举系数。如果使用回归进行预测，请使用引导预测。
*   [热编码](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f)是一种将分类数据转换成虚拟/布尔数据的方法。
*   针对预测排序或有序数据的回归称为有序回归。
*   具有相关预测会使解释回归系数的符号和值变得困难(并且会增加标准误差)。[多重共线性](https://en.wikipedia.org/wiki/Multicollinearity)在您创建虚拟变量而忘记创建 n-1 个变量时是一个特别的问题。
*   回归的另一个问题是混淆变量，即您遗漏了回归方程中未包含的重要变量。
*   异常值可以通过标准化残差(真实值离预测值有多远)来确定。
*   [部分残差图](https://en.wikipedia.org/wiki/Partial_residual_plot)也有助于理解预测因子和结果之间的关系。这里，您在 X 轴上绘制了一个给定的特征，在 Y 轴上绘制了部分残差。部分残差=残差+估计回归系数*特征值。
*   [异方差](http://www.statsmakemecry.com/smmctheblog/confusing-stats-terms-explained-heteroscedasticity-heteroske.html)是在预测值的范围内缺少恒定的残差方差。
*   当查看变量散点图时，查看平滑线来显示关系会有所帮助。这些选项包括黄土、超级光滑和内核光滑。

# 分类

*   如果你试图对两个以上的类别进行分类，最好是把它们分解成多个二元分类问题。尤其是当一个病例比其他病例更常见时。
*   分类技术的一个有趣的应用是[线性鉴别分析](https://en.wikipedia.org/wiki/Linear_discriminant_analysis) (LDA)，其目标是减少变量的数量，同时最大化类别的分离。
*   通常，我们将数据分为训练数据和测试数据，并根据测试数据评估我们的模型的准确性。您可以对 p(真实类别的预测概率)使用各种截止阈值，并评估精确度和召回率的权衡。这方面的工具包括[混淆矩阵](https://en.wikipedia.org/wiki/Confusion_matrix)和 [ROC 曲线](https://en.wikipedia.org/wiki/Receiver_operating_characteristic)。
*   在评估模型的有效性时，Lift 尤其重要。Lift 考虑的是与盲选或随机选择相比，性能要好得多。这在不平衡分类问题中特别有用。
*   当数据集不平衡时，一些策略是对最普遍的类进行欠采样，或者对少数类进行过采样。您也可以使用数据生成方法。

# 统计机器学习

统计机器学习的一个例子是 [K 最近邻](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm) (KNN)。

*   在 KNN，实际上有各种各样的距离度量和思考距离的方式，包括:欧几里德距离、曼哈顿距离和马哈拉诺比斯距离。
*   当依赖于计算距离函数的统计方法时，更重要的是将变量标准化，这样就不会意外地对某些要素赋予更大的权重。一种方法是 z 分数。
*   KNN 提出的一个问题是选择哪个国王。考虑这个问题的一个方法是考虑数据的结构。高度结构化的数据允许较低的 K，而结构化程度越低或信噪比越低，K 就应该越高。
*   KNN 也可以应用于特征工程。

[树模型](/decision-trees-and-random-forests-df0c3123f991)是对涉及类似决策树结构的分类和回归技术的一般描述。

*   一个重要的考虑因素是，你愿意让你的树有多深，或者有多少裂缝。分裂越多，过度拟合的风险就越高。人们可以通过硬编码来限制树的数量(通过超参数调整),也可以对树的生长进行惩罚。
*   随机森林和装袋比决策树更强大，但更难解释。
*   随机森林的一个有用输出是变量重要性的度量，它根据预测因子对模型精度的贡献对预测因子进行排序。
*   增强比打包本质上的集成方法，加上将训练集中在错误分类的数据点上的方法更复杂。

[超参数](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))需要在拟合模型之前设置；作为培训过程的一部分，它们没有得到优化。超参数调整可以通过直觉或者像[网格搜索](/grid-search-for-model-tuning-3319b259367e)和随机搜索这样的方法来完成。

正则化是一种避免过拟合的方法。正则化修改了成本函数，以降低模型的复杂性。岭回归是一种增加惩罚的方法。

# 无监督学习

*   无监督学习的用途是:聚类，以确定有意义的数据组，减少数据的维数，探索性的理解。
*   聚类对于“冷启动问题”来说是一个特别重要的工具，它允许您识别和标记要研究的子群体，在营销中尤其有用。
*   [主成分分析](https://en.wikipedia.org/wiki/Principal_component_analysis) (PCA)涉及协变预测变量的组合，是线性判别分析(LDA)的无监督版本。
*   [K-Means 聚类](https://en.wikipedia.org/wiki/K-means_clustering)是进行无监督学习最常用的技术之一。
*   由于我们再次使用距离函数，变量的规范化(即标准化)是必不可少的。
*   解释集群的大小和方式有助于您理解底层数据中的模式。例如，不平衡的聚类可能是由遥远的异常值或与其他数据非常不同的数据组产生的。
*   [肘法](https://en.wikipedia.org/wiki/Elbow_method_(clustering))是选择聚类数的有效方法。
*   [层次聚类](https://en.wikipedia.org/wiki/Hierarchical_clustering)是 K-Means 的替代方法，对发现异常值更敏感。当聚类可能包含其他聚类时，它们也更容易解释。