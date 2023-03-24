# 使用分类算法预测 Kickstarter 上的活动成功

> 原文：<https://towardsdatascience.com/https-medium-com-natashaborders-predicting-campaign-outcome-on-kickstarter-17e71b5e7ac9?source=collection_archive---------13----------------------->

## 型号选择、功能重要性讨论和 Flask 应用程序试用。

![](img/dd7222444e495e9e5d28052b5b3ee783.png)

作为我探索创意产业的数据科学探索的一部分，使用分类算法来尝试和预测 Kickstarter 活动的结果似乎是一个完美的项目想法。当我写这篇文章的时候，Kickstarter 已经帮助 [182，897](https://www.kickstarter.com/help/stats) 个项目获得成功，他们的使命是帮助将创意项目带入生活。随着[向 Kickstarter 项目承诺了 4，284，585，270 美元](https://www.kickstarter.com/help/stats)，这对于新的和有经验的创作者来说都是一个强大的平台。

![](img/887db8520fafea3ac0ed99fb2e60e411.png)

An example of a Kickstarter campaign page.

Kickstarter 活动采用全有或全无的融资模式。如果一个活动失败了，没有钱转手，这就把失败的更沉重的负担放在了活动创造者身上，他们可能已经把自己的时间和金钱投入到了活动中。

![](img/e39c8462016bc8934591a746fd858b50.png)

Photo by [Kobu Agency](https://unsplash.com/@kobuagency?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

出于我的分析目的，我专注于开发一个整体健壮的预测算法，同时确保创作者得到照顾。为了实现这一点，我选择将重点放在 [AUC ( **曲线下面积** ) ROC ( **接收器工作特性**)曲线](/understanding-auc-roc-curve-68b2303cc9c5)上作为主要评估指标，同时注意我的分析中“成功”类别的精度分数，以确保我们不会预测到太多的成功结果是失败(从而最大限度地降低假阳性率)。

这项分析的数据来自 Web Robots [网站](https://webrobots.io/kickstarter-datasets/)，该网站持续汇编 Kickstarter 数据，我检查了 2018 年 4 月至 2019 年 3 月期间全年成功和失败的活动。

![](img/d74e30ae0f1fd2829c09b94d87c87bab.png)

An example of a wildly successful Kickstarter campaign.

所使用的信息类型是活动启动时可用的数据，例如:

*   活动的描述(用作文字长度)
*   活动持续时间(天数)
*   无论该活动是否得到 Kickstarter 的支持
*   以美元为单位的目标
*   活动的地点(无论是否在美国)
*   活动的类别

在我的数据集中，略多于一半的活动是成功的(54%的成功，46%的失败)，所以这些类别相当平衡。

作为对几种分类算法的初步评估结果，并基于曲线下面积(AUC)作为我的主要评估指标，XGBoost 和 Logistic 回归表现最佳。

![](img/ba8067af0bbfe05335ea75999c83a459.png)

The Receiving Operating Characteristic (ROC) Curves for the models used in the analysis.

仔细检查营销活动成功类别的精确度分数，以确保我们没有预测到太多会导致失败的成功，总体表现最佳的是 XGBoost，精确度分数为 0.71。

![](img/375144f408e017298149758c70b4e9cc.png)

Results for the XGBoost model performance on the test dataset.

在对测试数据运行 XGBoost 之后，它表现一致，这意味着数据中没有过度拟合或欠拟合。success 类的精度分数略有增加，从 0.71 增加到 0.72。

在为这一分析选定最佳模型后，我回顾了哪些特性对成功的营销活动最为重要。

使用名为 [SHAP 值](https://github.com/slundberg/shap)的指标，我们可以检查哪些属性对营销活动的成功更重要。

![](img/06e4965c436ce22d0402d8dd6e18b961.png)

Feature importance for XGBoost Kickstarter model.

颜色表示特征的大小，方向表示对模型结果的正面或负面影响。污点外观来自于聚集在这些区域的许多数据点。例如，有一个小目标会对许多活动产生积极影响，而有一个大目标会对较小比例的活动产生不利影响。得到 Kickstarter 的支持对那些拥有它的活动来说有很大的不同，但没有得到它也没那么痛。

这里的类别与艺术进行了比较，因此我们可以得出结论，设计和游戏比艺术获得了更高的吸引力，而新闻和手工艺似乎对该活动的成功几率产生了负面影响。

![](img/408b54c2b710b8425caf3a62e3c8a95a.png)

My app demo page — take it for a spin!

为了演示这种预测算法的工作，我使用逻辑回归模型开发了一个 Flask 应用程序，这是第二高性能的算法，被证明比 XGBoost 更容易、更快部署。

应用程序本身可以在[这里](https://predicting-kickstarter-success.herokuapp.com/)找到——请随意玩你最喜欢的 Kickstarter 活动，并让我知道结果如何！

总之，建议创作者采取以下措施来增加活动成功的机会:

*   设定一个符合活动范围的小目标(美元)
*   不要将活动延长超过 30 天
*   仔细考虑类别
*   如果可能的话，获得 Kickstarter 的支持

展望未来，我想开发一个 XGBoost Heroku 应用程序，增加功能和视觉吸引力，了解成功的活动筹集了多少资金，设置延伸目标是否有益，并深入研究活动实现的持续挑战以及如何改进。

这个项目的所有资料都可以在我的 [GitHub](https://github.com/natashaborders) 页面找到。

> ***娜塔莎·鲍德斯，MBA***
> 
> ***链接于:@natashaborders***