# 机器学习项目 17 —比较分类算法

> 原文：<https://towardsdatascience.com/machine-learning-project-17-compare-classification-algorithms-87cb50e1cb60?source=collection_archive---------13----------------------->

在过去的 7 个项目中，我们使用不同的分类算法实现了同一个项目，即“[逻辑回归](https://medium.com/@omairaasim/machine-learning-project-10-predict-which-customers-bought-an-iphone-ea7b153db676)”、“ [KNN](https://medium.com/@omairaasim/machine-learning-project-11-whose-my-neighbor-k-nearest-neighbor-3e9184ce5f89) ”、“ [SVM](https://medium.com/@omairaasim/machine-learning-project-12-using-support-vector-classification-8f940c25101a) ”、“[核 SVM](https://medium.com/@omairaasim/machine-learning-project-13-using-kernel-support-vector-machine-9ca23bf39ac0) ”、“[朴素贝叶斯](https://medium.com/@omairaasim/machine-learning-project-14-naive-bayes-classifier-step-by-step-a1f4a5e5f834)”、“[决策树](https://medium.com/@omairaasim/machine-learning-project-15-decision-tree-classifier-step-by-step-aaaae0c2a111)、[随机森林](https://medium.com/@omairaasim/machine-learning-project-16-random-forest-classifier-414bb558d2c2)。

我为每种算法分别写一篇文章的原因是为了理解每种算法背后的直觉。

**# 100 daysofml code # 100 projects inml**

在真实的场景中，当我们遇到一个问题时，我们无法预测哪种算法会表现得最好。显然从问题中，我们可以判断出是需要应用回归还是分类算法。但是很难事先知道应用哪种回归或分类算法。只有通过反复试验和检查性能指标，我们才能缩小范围并选择某些算法。

今天，我将向您展示如何比较不同的分类算法，并挑选出最佳的算法。而不是用一个算法实现整个项目，然后发现性能不好，我们会先检查一堆算法的性能，然后再决定用哪一个来实现项目。

让我们开始吧。

# 项目目标

我们将使用在[项目 10](https://medium.com/@omairaasim/machine-learning-project-10-predict-which-customers-bought-an-iphone-ea7b153db676) 中使用的相同数据集。我们的目标是评估几种分类算法，并根据准确性选择最佳算法。

示例行如下所示。完整的数据集可以在访问[。](https://github.com/omairaasim/machine_learning/tree/master/project_17_compare_classification_algorithms)

# 步骤 1:加载数据集

我们将自变量“性别”、“工资”和“年龄”赋给 x。因变量“购买的 iphone”捕捉用户是否购买了该手机。我们会把这个赋值给 y。

# 步骤 2:将性别转换为数字

我们有一个必须转换成数字的分类变量“性别”。我们将使用 LabelEncoder 类将性别转换为数字。

# 步骤 3:特征缩放

除了决策树和随机森林分类器，其他分类器要求我们对数据进行缩放。所以让我们现在就开始吧。

# 步骤 4:比较分类算法

这是所有有趣的事情发生的地方:)

我将比较 6 种分类算法——我在以前的项目中介绍过的算法。也可以随意添加和测试其他内容。

*   逻辑回归
*   KNN
*   内核 SVM
*   朴素贝叶斯
*   决策图表
*   随机森林

我们将使用 10 倍交叉验证来评估每个算法，我们将找到平均精度和标准偏差精度。

首先，我们将创建一个列表，并添加我们想要评估的不同分类器的对象。然后，我们遍历列表，使用 cross_val_score 方法来获得准确性。

```
Here is the output:Logistic Regression: Mean Accuracy = 82.75% — SD Accuracy = 11.37%
K Nearest Neighbor: Mean Accuracy = 90.50% — SD Accuracy = 7.73%
Kernel SVM: Mean Accuracy = 90.75% — SD Accuracy = 9.15%
Naive Bayes: Mean Accuracy = 85.25% — SD Accuracy = 10.34%
Decision Tree: Mean Accuracy = 84.50% — SD Accuracy = 8.50%
Random Forest: Mean Accuracy = 88.75% — SD Accuracy = 8.46%
```

# 结论

从结果中，我们可以看到，对于这个特定的数据集， **KNN** 和**内核 SVM** 表现得比其他人更好。所以我们可以把这两个人列入这个项目的候选名单。这与我们分别实现这些算法得出的结论完全相同。

希望你玩得开心。你可以在这里找到《T4》的全部源代码。