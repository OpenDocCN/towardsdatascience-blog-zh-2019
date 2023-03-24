# 别让他们走了！

> 原文：<https://towardsdatascience.com/dont-let-them-go-818c03d7f09e?source=collection_archive---------12----------------------->

## ***利用数据科学(pySpark)检测客户流失。***

# 介绍

如今，许多公司将他们的商业模式从一次性收费转变为按月或按年收费。客户有权随时取消订阅，或者在某些情况下降级到免费订阅模式。另一方面，公司希望将客户保持在付费水平。

[](https://www.chargebee.com/blog/learn-recurring-subscription-billing-one-time-license-fee/) [## 订阅计费与一次性许可费:了解它们的比较——charge bee 的 SaaS 派遣

### 您是否不确定 SaaS 业务将采用何种计费模式？您应该收取一次性费用还是应该…

www.chargebee.com](https://www.chargebee.com/blog/learn-recurring-subscription-billing-one-time-license-fee/) 

通常，离开的顾客会有[一些迹象](https://conversionxl.com/blog/customer-churn/)表明他们即将离开。这些标志因服务而异；例如，对于一家电话公司，离开的客户通常会更频繁地致电支持部门，他们会提交一些投诉，或者他们很少使用服务。这些都是公司即将失去这个客户的一些指标！在另一种类型的服务中，比如在线照片编辑服务，不满意的迹象包括频繁访问某些页面和主题，如降级常见问题、联系我们页面，以及不寻常的访问帮助页面的频率。

[](https://conversionxl.com/blog/customer-churn/) [## 客户流失的主要指标，以及如何应对

### 你最近为你的 SaaS 公司赢得了大量的客户。表面上看，这似乎很棒。更多…

conversionxl.com](https://conversionxl.com/blog/customer-churn/) 

这些公司的目标是在客户的流失决定发生之前发现它，因此，在维护他们的成本远低于获得新客户的成本之前，联系他/她以给予一些折扣或其他奖励来保持订阅。

客户的流失被称为“客户流失”，或者“客户流失”，而留住客户并避免他们离开被称为“客户保留”。

[](https://www.ngdata.com/what-is-customer-churn/) [## 什么是客户流失？定义和如何减少它— NGDATA

### 什么是客户流失？定义和如何减少客户流失—客户流失的定义简单地说，客户流失…

www.ngdata.com](https://www.ngdata.com/what-is-customer-churn/) [](https://dataschool.com/red-flag-customer-churn/) [## 客户流失的危险信号 Chartio 的数据学校

### 客户流失率是现有客户停止与你做生意的比率。它可以通过…来计算

dataschool.com](https://dataschool.com/red-flag-customer-churn/) 

# 使用机器学习来检测客户流失。

我们有一个名为' *Sparkify* 的虚拟公司的例子，它提供付费和免费收听服务，客户可以在两种服务之间切换，他们可以随时取消订阅。

![](img/d248723832b27cb34175bb2fbb299669.png)

[Image by author]

给定的客户数据集非常大(12GB)，因此用于分析和机器学习的标准工具在这里没有用，因为它不适合计算机的内存(即使数据可以适合 32GB 计算机的内存，分析和计算需要的数量也远远超过这个数量，并且分析会使系统崩溃)。执行此类分析的安全方法是使用[大数据工具](https://www.guru99.com/big-data-tools.html)，如 [Apache Spark](https://spark.apache.org/) ，这是最快的大数据工具之一。

[](/8-open-source-big-data-tools-to-use-in-2018-e35cab47ca1d) [## 2018 年将使用的 8 种开源大数据工具

### 如今，大数据分析是任何业务工作流程的重要组成部分。为了充分利用它，我们建议使用…

towardsdatascience.com](/8-open-source-big-data-tools-to-use-in-2018-e35cab47ca1d) 

# 准备数据集

给定的数据集包含 18 个字段，这些字段包括`usedId`、`sessionID`、订阅`level`、访问过的`page` (用户做出的动作，如降级、升级、收听下一个文件、邀请朋友……)、`time`戳、用户的`gender`、`location`和`name`，以及一些文件的信息，如`author`、`artist`和`length`。

> 对于本教程，我们只处理了原始数据集的 128MB 切片。

***准备步骤***

*   一些记录包含空的`userId`或空的`sessionID`，这些是注销的用户，所以我们应该首先删除它们。
*   数据集缺少指示用户是否搅动的指标，我们应该为此添加一个`churn`字段。

***数据探索(探索性数据分析，EDA)***

首先，我们来看看流失用户的数量。

![](img/17834543d8f01e7299eedf5f051a1fb9.png)

We see a huge number of cancelation, about 25% of the users canceled! [Image by author]

然后，按性别或按订阅探索流失用户。

![](img/47eb2781527fcc98ad215695f628a17b.png)

[Image by author]

免费用户略多于付费用户，而被取消的免费用户与被取消的付费用户比例几乎相同！看来付费/免费状态并不影响帐户的终止。另一方面，活跃的男性比女性多，取消的男性也比女性多。似乎男性比女性更倾向于取消约会；即*性别似乎会影响流失决策*。

我们来看看流失用户什么时候最活跃

![](img/01af4aef3e8b7df59b5a5f1c8026b6a6.png)

[Image by author]

流失用户在月初最为活跃。因此，大多数取消发生在月底，这是合理的，以避免续费。

用户的操作系统会影响他们的活动吗？

![](img/c2411b9db5e601ff5992ac6a48d6b896.png)

[Image by author]

这里我们发现最快乐的用户是 iPad 用户(没有取消)，然后是 iPad 用户。
大多数倾向于流失的用户是那些使用 Windows 8.x、Windows XP、Linux 的用户。这可能会引发一个关于客户使用的软件的问题，它是否像 iPad 和 iPhone 的软件一样好用，让客户满意？

我们还探索了许多其他功能，在下面的 [GitHub 页面](https://github.com/drnesr/Sparkify/blob/master/Sparkify.ipynb)中有详细介绍。

[](https://github.com/drnesr/Sparkify/blob/master/Sparkify.ipynb) [## drnesr/Sparkify

### DSND 顶石项目。在 GitHub 上创建一个帐户，为 drnesr/Sparkify 开发做出贡献。

github.com](https://github.com/drnesr/Sparkify/blob/master/Sparkify.ipynb) 

**创建或提取可能有影响的特征(特征工程)**

在探索了数据集并了解了要包含的特征或需要从现有数据中提取的特征后，我们得出了以下一组可能影响 特征的*。*

1.  类别特征(具有离散特定值的特征)。这包括性别、订阅级别和操作系统。
2.  数字特征(连续值)。这包括每个会话的平均文件长度、会话的持续时间、会话的计数、总订阅天数，以及像竖起大拇指、竖起大拇指、邀请朋友、每个会话收听的文件这样的动作的频率。

机器学习算法只处理数值，因此，我们应该将类别特征转换为数字，例如，给一个性别一个 1。，另一个 0。如果特性包含无序值，那么我们应该通过 [one-hot-encoding](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f) 方法对它们进行编码，这涉及到为每个值创建一个单独的列，用 1 或 0 来表示它是否适用。

通过完成这个操作，我们有了一个新的用于分析的数据集，其形式为`userID` > > `feature01`、`feature02`、……

# 建模

我们测试了 5 个机器学习模型进行分类，看看哪一个产生的准确率最高。使用的模型有逻辑回归模型、决策树分类器模型、梯度提升树(GBTs)模型、随机森林模型、多层感知器分类器模型。

作为对数据的最后调整，我们已经对所有输入特征进行了标准化，并将它们组合成一个向量。然后，我们将数据集分成 80%用于训练模型，20%用于测试。

**逻辑回归模型**

![](img/87dd1d45c10f9042f4c20d228fc5e24d.png)

[Image by author]

正如我们在上表中看到的，逻辑回归模型的准确性相对较好，训练数据集和测试数据集的准确率分别为 82%和 75%。其他指标，如精确度、召回率和 F 值都比精确度值略低。这显示了该模型检测流失客户的良好性能。附图显示了每个特征的权重及其方向效应；比如`submit_upgrade`功能就是客户开心的指标，不会很快退订。快乐用户的其他重要功能是他们听的歌曲数量(`NextSong` 功能)。

另一方面，`mean_session_hours`和访问`Help`、`save_settings`和`Home`页面的频率是不满意的客户的指标，这些客户很快就会流失。

**决策树分类器模型**

![](img/1bd249d81da3537a6f3b8e2238a34c86.png)

[Image by author]

这个模型和所有被测试的分类模型都有一个' [***特征重要性***](https://datawhatnow.com/feature-importance/) '输出，这表明这个特征对结果的影响更大，不管它有正面还是负面的影响。特征重要性表明最具影响力的特征是`days_total_subscription`，其表明订阅长度对流失可能性的影响。第二个特征是`thumbs_down`、`Roll_advert`的数量，以及其他显示的特征。

[](https://medium.com/bigdatarepublic/feature-importance-whats-in-a-name-79532e59eea3) [## 功能重要性—名称中包含什么？

### 大数据共和国的斯文·斯金格

medium.com](https://medium.com/bigdatarepublic/feature-importance-whats-in-a-name-79532e59eea3) 

这个模型看起来非常严格，因为它忽略了 31/37 特征的影响，而只关注 6 个特征。然而，尽管如此，在训练和测试数据集上，准确性和其他性能指标都非常高。

**梯度增强树(GBTs)模型**

![](img/df8ca2059c2f3886b79057bf3b3ab395.png)

[Image by author]

与前两个模型相比，这个模型在训练数据集上具有更高的准确性和性能度量*，但是在测试数据集上的结果更差，这意味着模型[过拟合](https://en.wikipedia.org/wiki/Overfitting)数据。功能的重要性表明，最重要的功能是`NextSOng`访问量(播放的歌曲数量)，这似乎是客户满意度的指标，以及`Thumbs_UP`指标。`Error`页面是这里的第二个 runnerup，这似乎表明用户几乎厌倦了错误，很快就会离开。*

[](https://elitedatascience.com/overfitting-in-machine-learning) [## 机器学习中的过度拟合:什么是过度拟合以及如何防止过度拟合

### 你知道有一个错误吗......成千上万的数据科学初学者在不知不觉中犯的错误？还有这个…

elitedatascience.com](https://elitedatascience.com/overfitting-in-machine-learning) [](https://medium.com/greyatom/what-is-underfitting-and-overfitting-in-machine-learning-and-how-to-deal-with-it-6803a989c76) [## 机器学习中什么是欠拟合和过拟合，如何处理。

### 每当处理一个数据集来预测或分类一个问题时，我们倾向于通过实现一个设计…

medium.com](https://medium.com/greyatom/what-is-underfitting-and-overfitting-in-machine-learning-and-how-to-deal-with-it-6803a989c76) 

**随机森林模型**

![](img/bf535536dc4a887f2c11c1dda2b992c9.png)

[Image by author]

这个模型和以前的 GBT 一样，有明显的过拟合，训练精度很高，测试精度很低。随机森林模型在特征的重要性方面与*决策树分类器*一致，因为两者都显示最重要的指标是`days_total_subscription`和`Thumbs_Down`，而它在某种程度上与 GBT 在包括所有重要特征方面一致。(注意，重要性低于 3%的所有特征都被收集在`MINOR`类别中。)

# 结论

机器学习建模成功预测了最有可能以退订告终的客户活动。尽管所有模型的结果都很好，但*决策树分类器模型*似乎是这里最好的。但是，其他模型需要使用不同的设置进行重新调整，以减少过度拟合。