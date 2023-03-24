# 地震破坏模拟的人工智能

> 原文：<https://towardsdatascience.com/ai-for-earthquake-damage-modelling-7cefae22e7e1?source=collection_archive---------33----------------------->

## 人工智能和预测性分析如何帮助更快地从地震中恢复的案例研究

![](img/2fd0b8c902c924fee222e0deaeb8e1f2.png)

Image credit : Sky News

2015 年 4 月，尼泊尔发生了 7.8 兆瓦或 8.1 毫秒的大地震，最大麦卡利烈度为八度(严重)。根据尼泊尔开放数据门户网站，它影响了 **3，677，173** 个人和 **762，106** 财产。灾后，尼泊尔花了数年时间收集和评估损失，这是世界上最大的损失评估数据之一。像地震这样的大规模灾难之后，恢复通常分为两个阶段

1.  收集人口、建筑和法律数据
2.  领域专家使用这种大规模嘈杂数据进行的损害评估

> 基于建筑物的位置和结构，我们的目标是预测 2015 年尼泊尔廓尔喀地震对建筑物造成的破坏程度。

# 数据看起来怎么样？

在本案例研究中，我们使用了结构、所有权和损坏数据来准备训练和测试数据集。原始数据来自尼泊尔的[开放数据门户。](https://eq2015.npc.gov.np/#/download)如果您想使用我的前置数据，您可以从下面的链接中获得(结束注释部分)。现在，让我们更仔细地看看清理后的数据

![](img/00bba4b89843a68f2a64665259542b71.png)

shape of the cleaned training and test data

![](img/e20c36c6ff84cb9cccb40b0d6fc49b18.png)

数据不平衡，60%的“高”损害等级，22%的“低”损害等级和 18%的“中等”损害等级。为了处理不平衡的数据，需要手动采样。从最初清理的 700，000 个数据中，已经对每个类的 100，000 个数据进行了采样，并且已经为训练准备了 300，000 个数据点的训练集。分层抽样已用于准备最终的训练、测试和验证数据集。极少数数据点包含缺失值(< 30),so we ignored these data points.

# Is Age a factor?

Our final data set is of 41 dimension.Our independent variables are either numerical,categorical or binary.We have analysed the numerical and categorical variable in order to gain insights over data.For example Let’s take a snapshot of how buildings developed over last 10 years were affected

![](img/77b85a4a9002beb86c3137044ab8838c.png)

The ‘Age Factor’

Interestingly there are some properties where age is more than 950 years ! Are these outliers? As per Wikipedia there are few properties in Nepal which are actually that old. As per our data the number is **2521** )。

![](img/dafe064722a5bfe13c16a51d2c3dac3a.png)

damage grade of properties aged over 950 years

# **绩效指标**

我们预测破坏程度从 1 到 3(低、中、高)。损坏程度是一个顺序变量，意味着排序很重要。这可以看作是一个*分类*或*有序回归*问题。(有序回归有时被描述为介于分类和回归之间的问题。)

为了衡量我们算法的性能，我们使用了 **F1 分数**，它平衡了分类器的[精度和召回率](https://en.wikipedia.org/wiki/Precision_and_recall)。传统上，F1 分数用于评估二元分类器的性能，但由于我们有三个可能的标签，所以我们使用了一个称为**微平均 F1 分数**的变体。

![](img/3e9a7ba2aa0bf0bbb2f75732943e0387.png)

|TP| is True Positive, |FP| is False Positive, |FN| is False Negative, and |k| represents each class in |1,2,3|

# **车型及性能:**

在预处理和数据准备之后，我们从一个随机模型作为基线开始。尝试了各种机器学习模型，如逻辑回归、nystrome 逼近的线性 SVM(用于内核技巧)、随机森林、Light GBM 等。我们从一个非常基本的逻辑回归模型开始，复杂性逐渐增加。

为了从各种模型中获得最佳效果，必要时使用了 GridsearchCV 和简单的交叉验证技术。在实践中，经过调整的逻辑回归、SVM 和随机森林模型产生了在 0.65 到 0.69 范围内的微观平均 f1 分数。为了得到更好的分数，多数投票分类器和轻量级 GBM 模型被开发出来。让我们看看如何为多类分类问题定义一个定制的评估标准，以应用轻量级 GBM

![](img/ed8483180c55be71c1268071a74d3755.png)

通过对 lightGBM 和多数投票分类器进行适当的超参数调整，我们能够分别获得 0.78 和 0.74 的 f1 分数。我们也尝试了各种深度学习架构(MLP、LSTM、1D CNN)，但与经过调整的机器学习模型相比，性能很差。

以下是所获得结果的对比视图

![](img/c0bc843c4578b6c5e7544a5e8c861344.png)

# **现实世界影响**

自动化评估可以帮助两种类型的最终用户

1.  **政府机构**:政府机构无需人工干预就能更接近、更快速地了解地震造成的损失，从而促进损失恢复。
2.  **保险公司**:大规模灾难发生后，保险公司的理赔系统被大量新的索赔淹没。索赔处理人员查看所有损坏数据并确定损坏严重程度变得更加困难。将索赔系统与基于人工智能的损害评估服务相结合，将有助于索赔处理人员查看单个指标(损害等级)并决定损害的严重程度。这可以加快索赔处理的速度。

# **结束注释**

您可以在[我的 GitHub 资源库](https://github.com/arpan65/Earthquake-Damage-Modelling)中找到这个案例研究的所有必要文件、代码和数据集。

## 引用:

1.  [https://eq2015.npc.gov.np/#/](https://eq2015.npc.gov.np/#/)
2.  [https://arxiv.org/abs/1606.07781](https://arxiv.org/abs/1606.07781)
3.  [https://www.npc.gov.np/en](https://www.npc.gov.np/en)
4.  https://en.wikipedia.org/wiki/April_2015_Nepal_earthquake