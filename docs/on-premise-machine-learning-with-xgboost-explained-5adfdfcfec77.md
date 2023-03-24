# 解释了带有 XGBoost 的内部机器学习

> 原文：<https://towardsdatascience.com/on-premise-machine-learning-with-xgboost-explained-5adfdfcfec77?source=collection_archive---------10----------------------->

## 使用 Docker 容器在您自己的环境中运行机器学习模型的分步指南

![](img/1f0eeb1a029fcbe525dcfdc59eb296d6.png)

Source: Pixabay

可以在云上运行机器学习(ML)模型(亚马逊 SageMaker，谷歌云机器学习等。).我相信了解如何在你自己的环境中运行机器学习也很重要。没有这些知识，ML 技能集就不完整。这有多种原因。并非所有人都在使用云，您必须提供本地解决方案。如果不亲自动手配置环境，您将会错过学习更多 ML 知识的大好机会。

内部 ML 模型培训不仅与环境安装和设置相关。当你在云中训练 ML 模型时，你会使用供应商 API (Amazon SageMaker，Google 等)。)，这个 API 通常有助于更快地解决问题，但是它对您隐藏了一些有趣的东西——这将有助于更好地理解 ML 过程。在这篇文章中，我将一步一步地介绍 ML 模型，它可以不使用云 API 进行训练，而是直接使用来自开源库的 API。

让我们开始吧。首先，您需要启动内部 ML——Docker 映像(虽然您可以在没有 Docker 的情况下配置 ML 环境，但为了更好的维护和更简单的设置，我建议您使用 Docker)。

同去官方 *Jupyter 笔记本数据科学堆栈* [图片](https://hub.docker.com/r/jupyter/datascience-notebook)。用 *docker run* 命令创建一个容器(检查图像文件中所有可用的参数)。我建议注意你用 *-v* 参数映射工作目录的地方。该参数的第一部分指向 OS 上的文件夹，第二部分在 *:* 之后指向 Docker 容器中的文件夹(通常是/home/jovyan/work)。

[XGBoost](https://xgboost.ai/) 安装在 Jupyter 笔记本容器中。

您必须使用此命令进入 Docker 容器提示符*Docker exec-it container name bash，*才能运行以下命令:

*康达安装-y gcc*

*pip 安装 xgboost*

安装了 XGBoost 之后，我们可以继续学习 ML 模型——任何 ML 实现的核心部分。我使用 Jupyter 笔记本来建立和训练 ML 模型，这就是为什么我选择 Jupyter 的 Docker 图像。Jupyter notebook 提供了一种结构化的方法来实现 Python 代码，开发人员可以单独重新运行每个 notebook 部分，这提供了很大的灵活性，特别是在编码和调试 Python 代码时——不需要一直重新运行整个 Python 代码。首先，我们从进口开始。我建议在笔记本的开头保留所有导入(是的，您可以在笔记本的任何部分进行导入)。这种方式提高了代码的可读性——始终清楚正在使用什么导入:

第一步，用熊猫库读取训练数据。从我的 [GitHub](https://github.com/abaranovskis-redsamurai/automation-repo) repo 下载本例中使用的培训数据(*invoice _ data _ Prog _ processed . CSV*)。在我之前的帖子中阅读更多关于数据结构的内容— [机器学习—日期特征转换解释](https://medium.com/@andrejusb/machine-learning-date-feature-transformation-explained-4feb774c9dbe)。数据包含有关发票支付的信息，它指示发票是否按时支付以及是否延迟支付—延迟了多长时间。如果发票按时支付或延迟很小，则决策列被赋值为 0。

将数据从文件加载到 Pandas 数据框后，我们应该检查数据结构——决策列值是如何分布的:

XGBoost 处理数值(连续)数据。分类特征必须转换成数字表示。Pandas 库提供了 *get_dummies* 函数，帮助将分类数据编码成一个(0，1)数组。这里我们翻译分类特征 customer_id:

编码后—数据结构包含 44 列。

在运行模型训练之前，了解特征如何与决策特征相关联是很有用的。在我们的例子中，正如所料，最相关/最有影响力的特性是日期和总数。这是一个好的迹象，意味着 ML 模型应该被适当地训练:

接下来，我们需要识别 X/Y 对。y 是决策要素，它是数据集中的第一列。所有其他列用于标识决策功能。这意味着我们需要将数据分成 X/Y，如下所示:

这里，我们将数据分成训练/测试数据集。使用 *train_test_split* 函数 sklearn 库。数据集很小，因此使用其中较大的一部分进行训练— 90%。数据集使用分层选项构建，以确保决策特征在训练和测试数据集中得到很好的体现。函数 *train_test_split* 方便地将 X/Y 数据返回到单独的变量中:

这是关键时刻。用 XGBoost 运行 ML 模型训练步骤。 *%%time* 打印训练花费的时间。XGBoost 支持分类和回归，这里我们使用分类和 XGBClassifier。参数取决于数据集，对于不同的数据集，您需要调整它们。根据我的发现，其中包含的参数是需要注意的(阅读 XGBoost 文档中关于每个参数的更多信息)。

我们不是简单地运行模型训练，而是使用训练自我评估和早期停止的 XGBoost 特性来避免过度拟合。除了训练数据，还将测试数据传递给 ML 模型构建函数— *model.fit* 。该功能分配有 10 轮提前停止。如果 10 轮没有改善，训练就会停止，选择最优的模型。使用*对数损失*度量评估培训质量。使用 *verbose=True* 标志运行训练，以打印每个训练迭代的详细输出:

基于模型训练的输出，您可以看到最佳迭代是 Nr。71.

为了评估训练精度，我们执行*模型，预测*函数，并通过 X 测试数据帧。该函数为 X 集合的每一行返回一个预测数组。然后，我们将预测数组中的每一行与实际决策特征值进行匹配。精确度是这样计算的:

我们用测试数据执行了 model.predict。但是如何用新数据执行 model.predict 呢？下面是一个例子，它为 model.predict 提供了从静态数据构建的 Pandas 数据框架。付款延迟一天(发票开具后 4 天付款，而预期付款为 3 天)，但由于金额少于 80 英镑，此类付款延迟被认为没有风险。XGBoost model.predict 返回决策，但通常调用 model.predict_proba 可能更有用，它返回决策的概率:

一旦模型定型，保存它是一个好的做法。在我的下一篇文章中，我将解释如何从外部通过 Flask REST 接口访问训练好的模型，并使用 Node.js 和 JavaScript 向 Web app 公开 ML 功能。可以使用 *pickle* 库保存模型:

最后，我们根据 logloss 和分类错误的输出绘制训练结果。这有助于理解被选为最佳的训练迭代实际上是否是一个好的选择。基于该图，我们可以看到迭代 71 在训练和测试误差方面是最优的。这意味着 XGBoost 决定查看这个迭代是好的:

一个 XGBoost 提前停止和结果绘制的解决方案受到了这篇博客文章的启发— [使用 Python 中的 XGBoost 提前停止来避免过拟合](https://machinelearningmastery.com/avoid-overfitting-by-early-stopping-with-xgboost-in-python/)。

这篇文章完整的 Jupyter 笔记本可以从我的 [GitHub](https://github.com/abaranovskis-redsamurai/automation-repo/blob/master/invoice-risk-model-local.ipynb) repo 下载。训练数据可以从[这里](https://github.com/abaranovskis-redsamurai/automation-repo/blob/master/invoice_data_prog_processed.csv)下载。