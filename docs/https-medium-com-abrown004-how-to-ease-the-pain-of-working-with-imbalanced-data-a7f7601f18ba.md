# 如何减轻处理不平衡数据的痛苦

> 原文：<https://towardsdatascience.com/https-medium-com-abrown004-how-to-ease-the-pain-of-working-with-imbalanced-data-a7f7601f18ba?source=collection_archive---------27----------------------->

## *使用不平衡数据集创建模型的方法和资源概述*

您最终收集并清理了数据，甚至完成了一些探索性数据分析(EDA)。所有的努力终于有了回报——是时候开始玩模型了！然而，你很快意识到 99%的二进制标签属于多数类，而只有 1%属于少数类。使用准确性作为主要评估指标，您的模型将所有预测归类为多数类，并且仍然 *99%准确*。你做了一点谷歌搜索来找到栈溢出的正确搜索词，然后当你读到处理不平衡数据的痛苦时，你深深地叹了口气…

![](img/cb241f9c2a27ff36837c758d116bcb04.png)

# 什么是不平衡数据？

欺诈检测通常用作不平衡数据集的一个例子。也许每 1000 笔交易中只有 1 笔是欺诈性的，这意味着只有 0.1%的标签属于少数类别。疾病数据也可能是不平衡的。在环境健康方面，在 10^6，患癌症风险升高的典型临界值是 1，即百万分之一。在我的项目中，我处理的是诉讼数据，与整个数据集相比，被起诉的公司很少。总之，不平衡数据就像它听起来的那样:少数类的标签很少，这使得精度度量对于评估模型结果有些无用。

# 我应该做些什么不同？

## 使用不同的指标

另一个关键区别是使用传统准确性指标的替代指标。因为我主要对识别*真阳性*和避免*假阴性*感兴趣，而不是最大化准确性，所以我使用了曲线下面积接收算子特征(AUC ROC)度量。[这篇 TDS 文章](/understanding-auc-roc-curve-68b2303cc9c5)对 AUC ROC 指标有很好的解释。总之，AUC ROC = 0.5 实质上意味着该模型相当于随机猜测，AUC ROC = 1 意味着该模型可以完美地区分少数类和多数类。

我还依靠混淆矩阵来最大化*真阳性*和*假阳性*的数量，并最小化*假阴性*的数量。由于我在处理诉讼数据，我决定将*假阳性*结果视为未来被起诉的可能性很高的迹象。因此，*假阳性*结果对我也很重要。我喜欢这个 Kaggle 内核中的代码来可视化混乱矩阵。

![](img/6d488342bd4a190d58884513ff2dc77f.png)

Summary of True Positives and False Negatives using the Confusion Matrix

我还计算了更传统的精确度和召回率指标(下面的公式可供快速参考)来比较不同的学习算法，我还将不太传统的遗漏率(假阳性率)添加到指标列表中进行比较。

> 精度= TP / (TP + FP)
> 
> 召回率(真实阳性率)= TP / (TP + FN)
> 
> 漏检率(假阳性率)= FN / (FN +TP)

## 方法学

处理不平衡数据有两种主要的方法:(1)在算法层面和(2)在数据层面。我将在下面的小节中总结每种方法。

# 算法相关方法

## 成本敏感学习

在算法级别考虑不平衡数据集的解决方案需要理解支持成本敏感学习的算法。在我的案例中(也可能用于疾病和欺诈检测)，识别*真阳性*是该模型的主要目标，即使以选择一些*假阳性*为代价。这就是成本敏感学习派上用场的地方。[成本敏感学习](https://cling.csd.uwo.ca/papers/cost_sensitive.pdf)考虑不同类型的误分类(*假阳性* & *假阴性*)。

## 逻辑回归

经典的逻辑回归算法是不平衡数据集的稳健模型。逻辑回归算法包括一个[损失函数](https://hackernoon.com/introduction-to-machine-learning-algorithms-logistic-regression-cbdd82d81a36)，用于计算错误分类的成本。使用 SciKit-Learn，可以使用惩罚权重来操作损失函数，该权重包括“L1”或“L2”正则化，具体取决于所使用的求解器。

## 支持向量机

在 SciKit-Learn 中，[支持向量分类器](https://scikit-learn.org/stable/modules/generated/sklearn.svm.SVC.html)包括一个“class_weight”参数，可用于为少数类赋予更多权重。“‘平衡’模式”使用 y 值来自动调整输入数据中与类别频率成反比的权重。"[本文](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.303.4068&rep=rep1&type=pdf)进一步详述了开发成本敏感型 SVM 的细节。

## 朴素贝叶斯

SciKit-Learn 包括一个[补码朴素贝叶斯](https://scikit-learn.org/stable/modules/naive_bayes.html)算法，这是一个对成本敏感的分类器，它“使用每个类的补码的统计数据来计算模型的权重。”优化模型权重是处理不平衡数据集的有效方法。

## 集成方法—增强

Boosting 算法对于不平衡数据集来说是[理想的](https://medium.com/urbint-engineering/using-smoteboost-and-rusboost-to-deal-with-class-imbalance-c18f8bf5b805)“因为在每次连续迭代中，少数类被赋予更高的权重。”例如，“ [AdaBoost 通过在每次迭代期间调整误分类数据的权重，迭代地构建弱学习器的集成。](https://medium.com/urbint-engineering/using-smoteboost-and-rusboost-to-deal-with-class-imbalance-c18f8bf5b805)

# 数据相关方法

## 重新取样

在数据级别解决类不平衡问题通常涉及操纵现有数据，以强制用于训练算法的数据集达到平衡。这种方法称为重采样，典型的[重采样技术](https://pypi.org/project/imbalanced-learn/)包括:

*   对少数民族阶层过度取样，
*   欠采样多数类，
*   结合过采样和欠采样，或者
*   创建系综平衡集。

## 对少数民族阶层进行过度采样

过采样包括通过创建合成数据来平衡数据集，以增加少数类中的结果数量。一种常见的过采样方法称为合成少数过采样技术(SMOTE)，它使用 k 近邻来创建合成数据。

更进一步，SMOTEBoost 结合了过采样和提升。 [SMOTEBoost](https://medium.com/urbint-engineering/using-smoteboost-and-rusboost-to-deal-with-class-imbalance-c18f8bf5b805) 是“一种基于 SMOTE 算法的过采样方法，在每次增强迭代中注入 SMOTE 方法。”

## 对多数类欠采样

欠采样涉及减少包含在多数类中的数据，以平衡训练数据集。注意这个*减少了*数据集的大小。欠采样的一种常见技术是随机欠采样(RUS)，它随机选择多数类的子集。[这篇中篇文章](https://medium.com/anomaly-detection-with-python-and-r/sampling-techniques-for-extremely-imbalanced-data-part-i-under-sampling-a8dbc3d8d6d8)详细介绍了各种欠采样方法。

RUSBoost 将欠采样与增强相结合。与 SMOTEBoost 类似，“ [RUSBoost](https://medium.com/urbint-engineering/using-smoteboost-and-rusboost-to-deal-with-class-imbalance-c18f8bf5b805) 通过在每次增强迭代中执行随机欠采样(RUS)而不是 SMOTE 来实现相同的目标。”

## 整体方法——装袋

[Bagging](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.BaggingClassifier.html) 是数据级集成技术的一个例子。[打包涉及](https://machinelearningmastery.com/ensemble-machine-learning-algorithms-python-scikit-learn/)“从训练数据集的不同子样本中构建多个模型(通常是同一类型)。”在诸如随机森林决策树算法的算法中，装袋可以减少方差并防止过度拟合。

# 结论

对于我的数据集，最好的方法是结合使用少数类的整体提升和过采样方法(SMOTEBoost ),并使用逻辑回归算法实现 0.81 的 AUC ROC 值。有了花在特征工程上的额外时间，我可以进一步提高 AUC ROC 值。