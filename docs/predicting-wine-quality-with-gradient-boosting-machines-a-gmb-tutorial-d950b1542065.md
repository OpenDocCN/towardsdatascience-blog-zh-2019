# 用梯度提升机预测葡萄酒质量

> 原文：<https://towardsdatascience.com/predicting-wine-quality-with-gradient-boosting-machines-a-gmb-tutorial-d950b1542065?source=collection_archive---------15----------------------->

## 培训和部署 GBM 模型的必要条件

![](img/0efa696d142066d32fb01ff26fb84369.png)

Photo by [Vincenzo Landino](https://unsplash.com/@vincenzolandino?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 介绍

本文给出了一个训练梯度增强机器(GBM)模型的教程，该模型仅使用葡萄酒的价格和书面描述来预测葡萄酒的质量。在许多真实的数据科学项目中，通过部署其他人可以用来获得实时预测的实时模型，将项目带入下一阶段是很重要的。在本文中，我使用 AWS SageMaker 创建一个活动端点来演示模型训练和部署过程。

![](img/09b0df7cb46dfeee132e2fd706a60c15.png)

我想从 Winc 购买一瓶“夏日之水”玫瑰酒，这是一家在线葡萄酒订阅服务公司。因为我不能在购买前试用，所以我想在决定购买前先了解一下它的质量。在这篇文章的最后，我们将看到使用经过训练的 GBM 模型对这款葡萄酒的质量预测。

让我们跳进数据和建模！

# 数据

该模型使用梯度推进树回归和来自 [Kaggle](https://www.kaggle.com/zynicide/wine-reviews) 的真实数据来预测一瓶葡萄酒的质量点(y)。葡萄酒的“分数”分为 0-100 分，由《葡萄酒鉴赏家》分类如下:

*   95–100 经典:一款好酒
*   90–94 杰出:一款具有卓越品质和风格的葡萄酒
*   85–89 非常好:具有特殊品质的葡萄酒
*   80–84 好:坚实、制作精良的葡萄酒
*   75–79 平庸:可能有小瑕疵的可饮用的葡萄酒
*   50–74 不推荐

## 特征工程

在这个模型中使用的特征(X)是一瓶酒的价格和从对来自描述的非结构化文本数据执行潜在语义分析(LSA)中获得的潜在特征。

使用以下步骤设计潜在文本特征:

*   使用清理文档字符串的典型方法来清理和处理文本数据，例如删除标点符号和将文本转换为小写字母。
*   使用 LSA 对葡萄酒描述进行特征工程(使用 TF-IDF 和 SVD 进行矢量化，然后将文本的主体压缩成 25 个潜在特征)。

![](img/455e6e005dc6785fb4da59cf9a664263.png)

**Figure 1:** Example of wine description, price, and points data.

# 模型

使用 python 中的 xgboost 拟合梯度推进回归树模型，并使用平均绝对误差(MAE)进行评估

## 那么什么是梯度增压机呢？

一般来说，梯度增强机器是弱学习者集合到一个模型中以创建强学习者。梯度推进机器可用于回归或分类任务。它们通常应用于基于树的模型，但是理论上也可以应用于任何类型的弱学习者。

![](img/5543c7c5eec9a80355027da40a51c0a5.png)

**Figure 2:** This visualization of one of the weak learner trees in the XGBoost model illustrates how the tree splits on the price and latent description of the wine. We can see that the price is very influential for predicting the wine quality points! This weak learner also found something meaningful in one of the latent description topics from the LSA.

训练数据用于适应每个弱学习者。Boosting 和 Bagging 都可以用来将这些弱学习者集成到一个模型中。Bagging 并行构建所有弱学习者。Boosting 采用了一种更系统的方法，并通过将权重应用于前一个弱学习者错误预测的观察值，顺序构建弱学习者，每个弱学习者都试图更好地解释上一个弱学习者错过的模式。

在随机梯度增强中，训练数据的样本用于拟合每个弱学习者。

## Adaboosting

AdaBoosting 是最简单有效的二值分类 Boosting 算法。它顺序地用一次分裂来拟合决策树。这些小而弱的学习树被称为“决策树桩”。训练观察中的每个观察接收基于分类误差的权重，并且使用训练数据上的更新的权重来训练下一个决策树桩。还基于分类器的总误分类率为每个树桩分配一个权重。然后，该模型使用每个树桩上的权重来整合预测。具有大量错误分类的树桩接收较低的权重，导致它们的预测在集合预测中贡献较少。

## 梯度推进

梯度提升顺序地使弱学习者适应损失函数的梯度(导数),试图解释先前弱学习者错过的模式。当每一个学习者都适合时，使用加法模型来集成弱学习者。新的弱学习器的输出被加到先前弱学习器的输出上，以调整预测。这导致了一个递归方程，每个弱学习者都试图解释一个前一个学习者没有发现的模式。

第一个弱学习器被初始化为常数，例如平均值。

![](img/6571592ed394d23d348a9538a7652482.png)

然后，函数 h(x)被拟合到残差。残差是损失函数的梯度。

![](img/54930b2c917ba911c197346fdc6c0611.png)

其中 h(x)是适合损失函数梯度的弱学习器。Gamma 代表学习速率或步长。

![](img/2e7b87c057992dd0c7a63653e1b00203.png)

最终的模型对每个特征都有许多项，每一项都将预测推向不同的方向。

![](img/954245552c634990a18c3d1f395f970d.png)

因为弱学习器适合于预测损失函数的梯度，所以可以选择所使用的任何可微分损失函数，从而允许该方法应用于分类和回归问题。

# 用 Python 训练模型

首先，从导入所需的 python 库开始。

现在，加载葡萄酒数据并查看它的样本。

![](img/18185ca17c0ff4afb7ffc09121fb1dc1.png)

通过删除标点符号、数字并将所有字符转换为小写来预处理文本描述。

现在描述已经清理完毕，TF-IDF 用于向量化单词，SVD 用于将矩阵压缩成 5 个潜在向量。这种从文本数据中压缩特征的方法被称为潜在语义分析。为了简单起见，选择了 5 个潜在特征，但是在实践中，可以使用肘图来选择潜在特征的正确数量。

![](img/294c75b373c991e4a6818a590689cb45.png)

执行测试/训练分割，为 xgboost 格式化数据，并训练模型！

对测试数据进行预测，并使用平均绝对误差评估模型。平均而言，预测的质量点相差 1.84 个点。还不错！

![](img/efc1e5a124a2bd77136316ccf8939a23.png)

使用 xgboost 中的要素重要性图查看对模型影响最大的要素。看起来价格是预测葡萄酒质量时最重要的特征。

![](img/d396eab90503d43162907af1e9944e28.png)

**Figure 3:** Feature importance.

# 使用 AWS SageMaker 进行部署和推理

在[这个笔记本](https://github.com/statisticianinstilettos/wine_quality_predictions/blob/master/deploy_wine_model.ipynb)中，我使用 SageMaker 的 estimator 来训练模型，并将其作为一个实时端点。估计器启动一个训练实例，并使用`[train_wine_gbt.py](https://github.com/statisticianinstilettos/wine_quality_predictions/blob/master/train_wine_gbt.py)`脚本中的代码来训练模型，将模型保存到 s3，并定义端点的输入和输出。可以使用 SageMaker 的许多内置模型进行训练和部署，但我想为实时预测指定我自己的特征转换和输出，这可以使用像`train_wine_gbt.py`这样的 python 脚本来完成。

现在这个模型已经被训练和部署了，我可以用它来预测任何一瓶酒的质量！使用来自 Winc 网站的“夏日之水”的文本描述和价格，模型预测这款酒为 87，属于“非常好”的类别！

![](img/8ebb9830eb0ed0dba056240acca17c4c.png)

如果您想重用这些代码，请点击这里查看项目[的 GitHub repo，在那里我有 jupyter 笔记本用于培训和部署。](https://github.com/statisticianinstilettos/wine_quality_predictions)

干杯！