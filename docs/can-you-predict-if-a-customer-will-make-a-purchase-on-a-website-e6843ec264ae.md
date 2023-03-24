# 你能预测顾客是否会在网站上购物吗？

> 原文：<https://towardsdatascience.com/can-you-predict-if-a-customer-will-make-a-purchase-on-a-website-e6843ec264ae?source=collection_archive---------6----------------------->

## 使用 XGB 分类器和随机森林分类器根据客户的行为来预测他们是否会从某个网站进行购买。

![](img/fdb628c7ce93f7216887cd11cd1f9cf9.png)

[Source](https://www.pexels.com/photo/working-macbook-computer-keyboard-34577/)

# 概观

本文讨论了我们为一个电子商务网站提供的客户行为数据的分析。我们从讨论我们这个项目的动机开始。然后我们将讨论数据集及其特性。然后我们最初的数据探索和特征工程。最后，我们将描述我们用来预测网站访问者是否会购买的模型，这种模型的结果，以及我们从这些模型中收集的见解。由[马丁·贝克](https://www.linkedin.com/in/martbeck/)、[叶姬·李](https://www.linkedin.com/in/yeggilee/)、[坦维·莫迪](https://www.linkedin.com/in/tanvi-modi/)、[萨莎·奥佩拉](https://www.linkedin.com/in/alexandra-opela/)、[杰克森·罗斯](https://www.linkedin.com/in/jackson-r-ross/)、[马特·兹洛特尼克](https://www.linkedin.com/in/matthew-zlotnik/)合著，作为我们商业数据科学最终项目的成果。

# 动机

在美国，大约 9%的零售总额来自电子商务网站。事实上，像亚马逊这样的公司已经创建了零售帝国，成为一个如此巨大的电子商务网站。随着电子商务在当今经济中越来越流行，对于该行业的企业来说，了解影响网站访问者购物的因素并能够将注意力放在潜在客户身上是非常重要的。我们认为，如果能够预测网站访问者的购买行为，这将是一件有趣的事情，因为这可能会有许多影响，例如电子商务网站能够更好地定向广告或找出可能导致销售增加的因素。

# 数据特征及初步探索

该分析中使用的数据是加州大学欧文分校的机器学习知识库中提供的在线购物者购买意向数据集。该数据集的主要目的是预测该特定商店网站访问者的购买意图。数据集中的变量可以分为三类:与用户登陆的页面相关的数据、Google Analytics 指标和用户访问数据。如果你很好奇，想知道更多关于数据的细节和特性，你可以点击[查看](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset#)。

否则，让我们直接开始探索吧。在最初研究数据时，我们注意到的第一件事是标签的不平衡。在这个数据集中，85%的网站访问者最终没有进行购买，这是因为大多数人只是浏览橱窗，通常并不想购买。

![](img/48a6bdae0e874c568108b8377ed741b2.png)

Distribution of visitors that didn’t make a purchase and those that did

我们也很想知道是否有某个月份比较突出，或者某个月份的顾客购买率比其他月份高，比如在冬季假期期间，人们可能会因为打折或送礼而进行更多的网上购物。然而，令人惊讶的是，随着夏季流量的下降，大多数网站流量发生在 5 月，而大多数购买发生在 11 月，其购买量几乎是 5 月的两倍。

![](img/12f37e7d38cdc75e0f794cb71223dc90.png)

Distribution of purchasing and non-purchasing site visitors over months

我们注意到的最后一件事是，客户倾向于从如下所示的相同类型的地方访问网站。不幸的是，提供数据集的网站没有更详细地描述每种流量类型，我们只能假设它们与直接流量形式或间接流量形式相关，如来自另一个页面的引用或谷歌搜索。

![](img/d05f477e42315cbeb468b3810728cd01.png)

# 数据清理

为了清理数据，我们首先将分类变量和布尔变量(访问者类型、月份、周末、收入)更改为一次性编码变量和虚拟变量。接下来，我们将初始数据集中作为字符串列出的所有数字都改为整数。接下来，我们将数据分为 X 和 Y 数据框，并探索 X 数据中的相关性，以检查共线性。我们注意到与页面相关的数据都是高度相关的(左图)，所以我们用在每种类型的页面上花费的平均时间(右图)替换了这些变量，并消除了共线性问题。随着数据的清理和共线性的消除，我们可以放心地进入项目的分析部分了。

![](img/61d9cb61374954cefddc8ee3f0ec94dd.png)

Correlation chart of original data (left), correlation chart of data with collinearity removed (right)

# 建模和特征选择

在我们的分析中，我们最终决定使用三种不同的模型:SGD 分类器、随机森林分类器和 XGB 分类器。他们都能够为我们提供足够的准确性分数和 AUC 分数。

## 线性分类

我们尝试的第一个模型是简单的线性分类模型。为此，我们利用了 sklearn 的 SGD 分类器。估计器使用带有随机梯度下降(SGD)学习的线性模型。在每个样本处，损失的梯度被估计，并且模型基于学习速率被更新。

这个带有默认参数的简单模型实际上给了我们非常好的结果。我们设法用这个模型获得了 87.7%的准确度分数和 0.76 的 AUC 分数。

![](img/38c8e182e74dc7bac95404607ce45fba.png)

Linear classifier model code snippet and accuracy/auc score

## 随机森林分类器

我们运行的下一个模型是随机森林分类器。这确实需要一些参数调整来优化其性能。起初，我们在数据集的所有特征上运行模型，并且能够实现 88.7%的更高准确度，但是 AUC 分数下降到 0.728。经过大量调整后，我们意识到特征选择可能是进一步提高精度的最佳方法。

![](img/e9a15776ffc479aec0114afb21cf3af9.png)

Random forest classifier code snippet and accuracy/auc score

## 特征选择

已经有了这么好的准确性分数和 AUC 分数，我们决定尝试特征选择，看看是否有可能进一步增加我们模型的分数。通过使用 sklearn feature_selection 的 chi2 来完成特征选择。它能够为我们提供一个我们的特征列表和它们的重要性分数。最后，我们选择删除 chi2 中得分为 3 或更低的特性。这将最终删除 5 个功能，如 Jul、OperatingSystems、TrafficType 等。

![](img/981d6eb52781a59b60399e2e8fabbd87.png)

Code snippet of chi2 feature selection and each features given importance score

## 随机森林分类器(特征选择)

在去掉最不重要的特征后，我们重新运行随机森林模型，看看它是否能产生更好的结果。特征选择 RF 模型比我们在每个特征上运行的模型产生更好的结果。它将准确性分数提高到 89.9%，将 AUC 分数提高到 0.771

![](img/413f40c0581fc2e0aa89ba403a02079c.png)

Random forest classifier code snippet with feature selection and accuracy/auc score

## XGBoost 的分类器(特征选择)

我们在数据集上运行的最终模型是 XGB 分类器，其特征选择已经在之前的 chi2 中使用过。这最终成为我们的模型，AUC 得分最高，为 0.773，准确率高达 89.3%。然而，即使 XGB 分类器是 AUC 得分的最佳模型，我们运行的每个模型的得分都在彼此相对较小的范围内，这表明选择任何模型都已被证明是足够的。

![](img/2d252f6539a861a32354ede56719ce89.png)

XGBoost classifier code snippet and accuracy/auc score

# 结论和未来工作

据估计，大约有 19.2 亿人在网上购物。这大约占世界人口的 25%，而且这个数字还在快速增长，预计到 2021 年将达到令人印象深刻的 21.4 亿人。显然，能够从在线流量预测电子商务销售额对任何公司都是有益的。从我们的结果来看，公司应该专注于提高页面之间的移动性，以鼓励用户浏览不同的产品，因为页面价值是决定是否购买的最重要的特征之一。此外，诸如 5 月和 11 月之类的特定月份具有更高的购买频率，这意味着电子商务公司应该利用这些月份并提供额外的销售和交易来鼓励产品销售。虽然我们的数据及其在更大范围内的应用存在局限性，但我们的分析表明，基于我们能够从数据集中提取的特征，可以在一定的信心水平内预测网站访问者的购买行为。

为了进一步改进这个项目，我们认为能够更好地塑造网站访问者的背景的数据将是重要的。可以使用的可能数据点是用户购买历史、第三方数据、愿望清单等。这将使我们能够更好地分析潜在客户的重要特征和特点。最后，添加类别权重可能是有用的，这将对错误分类代表不足的类别增加更大的惩罚，以便减少假阳性或假阴性。这将减少不平衡数据的偏差，而不会对代表性不足的样本进行“过度训练”。

如果您想查看更多信息:

[包含代码和数据集的该项目的 Github 库](https://github.com/MartinBeckUT/BDSFinalProject)

# 参考

UCI 的机器学习知识库数据集:UCI 机器学习知识库。(2018).网上购物者购买意向数据集。2019 年 12 月 10 日检索，来自[https://archive . ics . UCI . edu/ml/datasets/Online+购物者+购买+意向+数据集](https://archive.ics.uci.edu/ml/datasets/Online+Shoppers+Purchasing+Intention+Dataset)

电子商务总销售额信息:萨卡尔，o .，波拉特，o .，卡特西奥卢，m .，&卡斯特罗，Y. (2018)。基于多层感知器和 LSTM 递归神经网络的在线购物者购买意向实时预测。神经计算和应用(31)。https://doi.org/10.1007/s00521-018-3523-0

结论电商统计:法学，t . j .(2019 . 11 . 20)。19 个强大的电子商务统计数据将指导您在 2019 年的战略。检索自[https://www . ober lo . com/blog/ecommerce-statistics-guide-your-strategy。](https://www.oberlo.com/blog/ecommerce-statistics-guide-your-strategy.)