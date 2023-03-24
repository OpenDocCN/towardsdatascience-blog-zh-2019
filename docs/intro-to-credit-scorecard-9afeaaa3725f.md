# 信用记分卡简介

> 原文：<https://towardsdatascience.com/intro-to-credit-scorecard-9afeaaa3725f?source=collection_archive---------1----------------------->

## 如何建立一个简单的信用记分卡的分步指南

![](img/7df7089ef8978b9e809417947cec2af8.png)

Source: NerdWallet

# 背景

随着金融科技初创企业的崛起，过去 5 年里出现了许多新的消费信贷机构，与传统银行展开竞争。它们通常瞄准银行认为规模太小的利基市场，或在金融危机期间因亏损而不得不削减放贷的市场。

这些新的纯在线消费贷款机构的主要竞争优势之一是技术元素，没有每个大型银行都有的传统 it 系统的拖累，他们建立了更快、无纸化、更用户友好的移动界面系统，并在承保流程中利用除传统金融数据之外的新数据源。

例如，商业贷款机构 [iwoca](https://www.iwoca.co.uk/faq/?utm_source=Google+AdWords&utm_medium=PPC&utm_campaign=1905711499&ad_group=72226008204&feed_item_id=1181854575&target_id=kwd-42576349472&click_location=9045953&interest_location=9045953&match_type=e&network=g&device=c&device_model=&creative=380202581659&keyword=iwoca&placement=&experiment_id=&ad_position=1t1&gclid=Cj0KCQiAiZPvBRDZARIsAORkq7eKIo1yRpQc2zFVszLaax_sg545E4X6XcPAZ7AM-b5urOOWn2vlr28aAqF2EALw_wcB) 使用关联公司账户、增值税申报表甚至 ebay 或亚马逊上的销售交易信息来确定新贷款。消费者贷款银行 [lendable](https://www.lendable.co.uk/frequently-asked-questions) 引以为豪的是在几分钟内给出个性化的贷款报价，而不是传统银行需要的几天或几周。

随着快速和自动决策的承诺，他们回复到拥有自动和快速的信用风险模型来评估风险。在这篇文章和后面的文章中，我打算介绍消费金融信用风险领域最常用的机器学习模型。

# 什么是信用记分卡

![](img/2ebc636ac6e7d6df83fb017bc2b7ed06.png)

Photo by [The New York Public Library](https://unsplash.com/@nypl?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

我们大多数人都熟悉**信用评分**的概念，这是一个代表个人信誉的数值。像银行这样的所有信贷机构都有复杂的信贷模型，这些模型使用应用程序中包含的信息，如工资、信贷承诺和过去的贷款业绩，来确定应用程序或现有客户的信用评分。该模型输出一个分数，该分数表示如果贷方向某人提供贷款或信用卡，该贷方按时还款的可能性有多大。

信用记分卡就是这样一种信用模型，它是最常见的信用模型之一，因为它对客户来说相对容易解释，并且它已经存在了几十年，因此开发过程是标准的，并且被广泛理解。

但是，值得注意的是，分数的范围可能因机构而异，拒绝分数较低的申请的截止点可能因贷款人而异，甚至可能因同一贷款人但不同产品而异。

# 建立信用记分卡

T 目标变量通常采用二进制形式，根据数据的不同，对于履约客户，它可以是 0，而对于违约客户或逾期付款超过 90 天的客户，它可以是 1。在本文的其余部分，我们将把“坏客户”称为某种违约的客户，而把其他客户称为“好客户”。

## 步骤 1:数据探索和清理

这是所有模型拟合中的一个基本步骤，但由于它不是专门针对构建信用记分卡模型的，所以我们将跳过这一部分。不要忘记将数据集分为训练数据集和测试数据集。

## 步骤 2:数据转换—证据权重法

然后，我们需要转换所有的独立变量(如年龄、收入等。)使用证据权重法。该方法基于每个分组级别的好申请人与坏申请人的比例，测量分组的“强度”以区分好的和坏的风险，并试图找到自变量和目标变量之间的单调关系。

![](img/f4f9eed6cfb85cc670db15f05e9c907b.png)

**连续变量的变换步骤:**

1.  将数据分成多个箱，通常大约 10 个，最多 20 个
2.  计算好事件的百分比和坏事件的百分比
3.  通过取自然对数来计算 WOE
4.  用计算出的 WOE 值替换原始数据

*如果自变量是分类变量，那么跳过上面的 1，并遵循剩下的步骤。*

***用 Python 举例:***

将您的数据放入箱中，并对每个箱的坏计数和好计数进行分组，以便您的数据看起来类似于下面的方框。对于每个 bin 组，可以使用下面的代码计算 WoE。负值表示特定分组中不合格申请者的比例高于合格申请者。

![](img/49fe0ae7d1efb7339830b789db7f0dc5.png)

在转换结束时，如果开始时有 20 个独立变量，那么下一步将有 20 个 WOE_variablename 列。

> **使用 WoE 转换的好处:**

*   它有助于与逻辑回归中使用的对数优势建立严格的线性关系
*   它可以处理丢失的值，因为这些值可以合并在一起
*   异常值或极值可以被处理，因为它们也被分箱，并且馈入模型拟合的值是经过 WoE 变换的值，而不是原始极值
*   它还处理分类值，因此不需要虚拟变量

## 步骤 3:使用信息值进行特征选择

信息值(IV)来源于信息论，它度量自变量的预测能力，这在特征选择中是有用的。执行特征选择以确定是否有必要在模型中包含所有特征是一种良好的做法，大多数情况下，我们会希望消除弱的特征，因为更简单的模型通常是首选的。

> 根据 Siddiqi (2006)，按照惯例，信用评分中 IV 统计值可以解释如下

![](img/4e5f031cdcce1b9c4c49430563ce49c8.png)

***用 Python 举例:***

![](img/cfaca7e880a74d9d0b47217fb987ad9e.png)

继续前面的示例，这里我们计算出“年龄”的 IV 约为 0.15，这意味着年龄具有“中等预测能力”，因此我们将保留用于模型拟合。IV 值小于 0.02 的变量应被删除。

## 步骤 4:模型拟合和解释结果

现在，我们使用新转换的训练数据集的权重来拟合逻辑回归模型。

当将模型缩放到记分卡中时，我们将需要模型拟合的逻辑回归系数以及转换后的 WoE 值。我们还需要将模型中的分数从对数优势单位转换为分数系统。

对于每个自变量 Xi，其对应的得分是:

> Score_i= (βi × WoE_i + α/n) ×因子+偏移量/n

其中:
βi —变量 Xi 的逻辑回归系数
α —逻辑回归截距
WoE —变量 Xi 的证据值权重
n —模型中自变量 Xi 的个数
因子，偏移量—称为标度参数，其中

*   因子= pdo/ln(2)
*   偏移量=目标分数-(因子× ln(目标赔率))

![](img/f001e9ba6fec732e17e86f321bcd80f7.png)

对于上面的例子，我们选择 600 的**目标分数来表示好客户对坏客户的**赔率为 50 比 1**，20 的**增加意味着双倍的**赔率。注意比例的选择不影响记分卡的预测力度。**

最终总得分是基于自变量输入值的所有得分的总和。然后，贷方将根据模型化的总分和截止点(根据其他信用违约模型设定)评估收到的申请。

> 总分=**σScore _ I**

![](img/da8dbbc197ef2961e94010bd206e27fa.png)

An example of scorecard implementation, source { Credit Scoring — Scorecard Development Process by Sandy Liu, link at end of of this post}

# 来源

1.  [证据权重和信息价值说明](https://www.listendata.com/2015/03/weight-of-evidence-woe-and-information.html)
2.  [逻辑回归中用 WoE 代替变量](https://stats.stackexchange.com/questions/189568/replacing-variables-by-woe-weight-of-evidence-in-logistic-regression)
3.  [信用评分—记分卡开发流程](https://medium.com/@yanhuiliu104/credit-scoring-scorecard-development-process-8554c3492b2b)
4.  [通过机器学习进行信用评分](https://medium.com/henry-jia/how-to-score-your-credit-1c08dd73e2ed)