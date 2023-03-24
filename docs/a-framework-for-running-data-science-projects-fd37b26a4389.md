# 运行数据科学项目的框架

> 原文：<https://towardsdatascience.com/a-framework-for-running-data-science-projects-fd37b26a4389?source=collection_archive---------25----------------------->

![](img/38b8000677659390751ae65e2cedfb11.png)

Seattle photo by [Luca Micheli](https://unsplash.com/@lucamicheli?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) & Boston photo by [Osman Rana](https://unsplash.com/@osmanrana?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/boston?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

# 开始一个数据科学项目

对于从数据领域起步的人来说，从零开始建立一个项目似乎有点令人生畏。这可能很有诱惑力，只是把你所知道的一切都投入进去，看看什么会有效，但事实上(和其他事情一样)，耐心最终会有回报的。这就是为什么数据科学家开发了各种旨在帮助管理数据项目和提供结构的框架。这些框架中的大多数都遵循类似的步骤，这些步骤将在本文中概述。

为了给出一个真实的例子，我将参考我一直在做的一个项目，比较西雅图和波士顿的 Airbnb 数据，试图提取有用的商业见解。遵循本文中的步骤将会给你一个开始的地方，并有希望让你专注于底层数据。

# 了解你的目标(主题理解)

在执行任何任务之前，重要的是要知道分析的目标是什么，或者你要解决什么问题。这似乎是显而易见的，但是你会惊讶于很多次有人会要求你做一些没有明确目标的事情。首先，你应该问为什么要执行任务，找出利益相关者的目标，理解关键绩效指标(KPI)。在整个项目中记住这些事情是很重要的，这样你就不会忽略什么是重要的。

![](img/9888f7e07be73e5af8268ec13db3cc94.png)

Airbnb conference photo by [Teemu Paananen](https://unsplash.com/@xteemu?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/airbnb?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

在我的案例中，使用 Airbnb 数据的目的是根据各种因素(例如位置、便利设施或大小)向用户推荐一个挂牌价格。这个商业案例很简单；价格更好的用户不会低估他们的财产而导致收入损失，也不会高估他们的预订损失。为了让我在项目过程中保持专注，我想出了一些简短的商业目标，以便在实现最终目标的过程中回答:

*   Airbnb 两个市场的价格会随季节变化吗？
*   Airbnb 的哪些房源变量会对价格产生最大影响？
*   我能根据市场、设施和季节性预测价格吗？

将一项任务分解成业务问题、小目标或里程碑有助于保持专注，确保实现核心业务目标。

# 数据理解

现在是时候尝试一下，看看你有什么可用的数据。一个很好的起点是考虑你可能需要哪些数据点来实现你的目标，以及这个目标如何与你当前的数据保持一致。在 Airbnb 项目中，我需要我的因变量:我想要预测的价格。为了创建我的模型，我还需要我的独立变量，例如评论分数、位置、客人数量和设施。

![](img/3a98154d726a2f64ea8c6fe4d256af11.png)

An example of a Pandas DataFrame during the Data Understanding step

接下来，检查数据的质量是很好的。我这样做是通过寻找 NaN 的，简单的错误，并检查以确保所有的数据集可以合并。这最后一点听起来微不足道，但如果我的因变量和自变量有不同的标识符，数据将是无用的。最后，在这一点上创建最初的可视化将有助于对数据的整体理解。

![](img/d84258dfe26f5ab13175085404d9898d.png)

Initial analysis | Distribution of Airbnb prices

上面的直方图显示了 Airbnb 各个市场的价格差异。与波士顿相比，西雅图的平均价格较低，差价较小。这表明项目后期可能需要考虑地区差异。它显示，波士顿的平均房价将远远高于西雅图的房价。

在数据中翻箱倒柜之后，你可能会发现几个结果中的一个:你已经有了执行分析所需的一切，需要更多的数据，或者项目不可行，因为所需的数据不可用。

# 数据准备

现在我们已经有了所有需要的数据，是时候将它处理成所有分析和建模所需的格式了。这通常是流程中最耗时的部分(60–80%的时间),并且随着数据需求的发展，可能会与任何分析/建模交织在一起。

![](img/0dd9de8b1a621f0ccf701e4cc2978d53.png)

Python photo by [Chris Ried](https://unsplash.com/@cdr6934?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/data?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

首先要做的是对数据进行一些清理。这可能包括处理 NaN、格式化日期、合并文件、剥离字符串、编辑价格等等。在 Airbnb 项目的这一点上，我为清理和合并我的数据做了一些基本的功能。我强烈建议这样做，因为您可以将它们添加到一个包中，并在整个项目中多次使用它们。

![](img/f1d2ec873074699f22381e58eed5a4a5.png)

Example from a basic cleaning function used on the Airbnb dataset

下一部分将取决于项目最终想要达到的目标。在进行更多清理之前，我做了一些基本分析，以了解更多关于 Airbnb 的数据。对于许多项目，将需要一些更高级的处理，尤其是执行机器学习(ML)任务。例如，对分类变量进行编码，对基于文本的数据运行 NLP(自然语言处理),或者执行 PCA(主成分分析)来降低维数。任何建模之前的最后一步是在训练和测试数据集之间分割数据，因为这允许您保留一些数据用于模型调整和评估。

# 系统模型化

这是大多数博客文章、研究论文和面向客户的演示会关注的部分。然而，如果你幸运的话，它最多会占用你 10-20%的时间。在这一部分，我们创建机器学习模型，有洞察力的可视化或执行投影。

在我的项目中，我做了一些分析，试图解决我的第一个商业问题:Airbnb 两个市场的价格是否随季节变化？

![](img/3c62cdc4ef0cf77e5c14d34abf2dd50a.png)

Seasonal trends | Seattle & Boston

季节性趋势变得相当令人惊讶，因为这两个市场全年的变化程度相似。它们在年初都有一个季节性低点，在 8 月份左右有一个高点。这种季节性变化可用于调整最终的 ML 模型，以获得全年价格的更好预测。

在上面的分析之后，我继续讨论 Airbnb 数据集的核心建模问题。为此，我进行了回归分析，根据因变量来预测上市价格。使用的回归模型是一种被称为随机森林的集合方法(这种方法的细节可以在[这里](/an-implementation-and-explanation-of-the-random-forest-in-python-77bf308a9b76)找到)。模型训练如下所示:

![](img/835c81c93b311d5d13231205934d5584.png)

Parameterisation of the Random Forests ML model

当优化任何形式的建模时，确保用可靠的性能指标来验证测试是很重要的。这些将根据业务案例而有所不同。在我的回归分析中，我使用了 MAE(平均绝对误差),因为它不会像 RMSE(均方根误差)等其他方法那样对较大的误差进行加权。这对于预测 Airbnb 价格很有用，因为我们更喜欢一个更通用的模型，即使一些离群值被预测错了。

![](img/0f20a6bf2a7e559daebd470f4c5c2639.png)

Performance metrics for model parameterisation showing RMSE and MAE

模型制作和优化后，在部署任何东西之前，都需要检查、测试和评估。

# 估价

评估您的模型/分析是该过程中最重要的步骤之一。这是为了评估模型或见解在多大程度上可以推广到新的数据集。该过程的第一步是使用建模前搁置的测试数据，以确保模型在参数化过程中不会过度拟合。

我的模型评估的另一个重要部分是查看每个变量如何影响模型预测。这被称为功能重要性，有助于回答我的第二个商业问题:Airbnb 的哪些房源变量将对价格产生最大影响？

![](img/27e3f4efe9c7c129c0972b3c80a87348.png)

Feature importance of the Random Forests model

从上面的图来看，影响价格的最重要的特征是(如你所料)房产大小指标，如卧室、浴室和客人。驱动该模型的令人惊讶的特征是，该物业是否有急救箱或一氧化碳探测器；这些可能是价格较低的房产正在提升其舒适度的一个指标。执行特征重要性的另一个关键要点是，一些变量对模型有负面影响，因此应该在建模前移除。

评估建模效果的最后一步是将该过程应用于另一个市场，并观察结果的对比。对我来说，这有助于回答整个商业问题:我能根据市场、变量和季节性预测价格吗？

![](img/400cd5f3a3be2d7f7d85f731ef2c3387.png)

Actual Airbnb prices against the predicted prices

上面的结果显示了 Airbnb 的实际价格和预测价格。这条线代表完美的预测，从这条线到每个点的距离就是预测的误差。总的来说，模型很好地预测了总体趋势，除了异常值属性。这些异常值可能预订率低，或者有其他原因导致价格与趋势不符(例如 Airbnb 的质量)。总的来说，该模型可以很好地预测价格，为用户提供建议，从而实现核心业务目标。

通过查看您的模型验证指标和 KPI，您可以最终决定您是否成功实现了您的业务目标。

# 成果和部署

最后，是时候将你创造的一切付诸实践了。这可能是为外部流程部署一个模型(想想网飞推荐引擎),或者根据分析得出的见解调整内部流程。对于 Airbnb 项目，这将需要根据用户提供的便利设施在托管页面上执行价格建议。

![](img/48442686829d2096af2d824c3fef67f8.png)

Just sit back and relax! Photo by [Ali Yahya](https://unsplash.com/@ayahya09?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/success?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

如果你认为项目到此结束，那你就大错特错了，在现实世界中，一旦你部署了一个项目，就会有无穷无尽的改进、更新和错误修复要处理。就结论而言，你能做的最好的事情就是希望你在所有的迭代之间得到一个新的项目来集中你的注意力！

# 完成流程

总之，在任何项目中要经历的主要步骤是:

1.  主题理解(上下文和业务案例)
2.  数据理解(收集和初步分析)
3.  数据处理(准备、清理、擦洗和争论)
4.  建模(建模、分析、探索和验证)
5.  评估(解释、优化和评估)
6.  部署(实施、改进和迭代)

上面的框架很大程度上遵循了 CRISP-DM 流程，并添加了来自其他框架的额外注释。这在很大程度上被视为执行数据科学项目的行业标准，应该为运行任何数据项目提供坚实的基础。

## 传统数据科学框架

这篇文章从 CRISP-DM 之外的一些数据框架中获得了灵感。这里有几个例子，如果你想谷歌一下或者进一步研究的话:

*   CRISP-DM(数据挖掘的跨行业标准流程)
*   KDD(数据库中的知识发现)
*   OSMEN(获取、擦洗、探索、建模、解释)
*   数据科学生命周期

## 关于我

目前，我是 ASO Co(一家应用营销公司)的数据科学家。有关该项目的任何进一步信息或查看我在分析中的表现，请在下面找到我的各种社交平台的链接:

GitHub:[https://github.com/WarwickR92/Airbnb-Market-Analysis](https://github.com/WarwickR92/Airbnb-Market-Analysis)

领英:[https://www.linkedin.com/in/warwick-rommelrath-a7b4575a/](https://www.linkedin.com/in/warwick-rommelrath-a7b4575a/)

这个项目中的所有数据都是通过 Kaggle 上的 Airbnb 提供的，并与我的 Udacity 数据科学家 Nanodegree 一起使用。