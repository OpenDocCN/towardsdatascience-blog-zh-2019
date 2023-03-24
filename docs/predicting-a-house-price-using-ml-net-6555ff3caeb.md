# 用最大似然法预测房价。网

> 原文：<https://towardsdatascience.com/predicting-a-house-price-using-ml-net-6555ff3caeb?source=collection_archive---------25----------------------->

![](img/81801ee73baed76858480784af9883b7.png)

Photo by [todd kent](https://unsplash.com/@churchoftodd?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

ML。Net 是一个开源的跨平台机器学习框架。NET 开发人员。Python(例程是用 C++编写的)通常用于开发许多 ML 库，例如 TensorFlow，当您需要在上紧密集成 ML 组件时，这会增加额外的步骤和障碍。Net 平台。ML.Net 提供了一套很棒的工具，让你可以使用。你可以找到更多关于 ML 的信息。网[这里](https://dotnet.microsoft.com/apps/machinelearning-ai/ml-dotnet)

针对 ML 1.4.0 更新

ML。NET 允许你开发一系列的 ML 系统

*   预测/回归
*   问题分类
*   预测性维护
*   图像分类
*   情感分析
*   推荐系统
*   集群系统

ML。NET 为机器学习提供了一个开发者友好的 API，支持典型的 ML 工作流:

*   加载不同类型的数据(测试、IEnumerable、二进制、拼花、文件集)
*   [转换数据](https://docs.microsoft.com/en-us/dotnet/machine-learning/resources/transforms)特征选择、归一化、改变模式、类别编码、处理缺失数据
*   为你的问题选择一个合适的算法(线性，提升树，SVM，K-Means)。
*   培训(培训和评估模型)
*   评估模型
*   部署和运行(使用训练好的模型)

使用 ML.NET 框架的一个显著优势是，它允许用户快速试验不同的学习算法，更改特征集、训练和测试数据集的大小，以获得针对其问题的最佳结果。实验避免了一个常见的问题，即团队花费大量时间收集不必要的数据，并产生性能不佳的模型。

# 学习管道

ML。NET 将用于数据准备和模型定型的转换合并到一个管道中，然后将这些转换应用于定型数据和用于在模型中进行预测的输入数据。

讨论 ML 的时候。NET 中，重要的是要认识到使用:

*   转换器—这些转换器转换和处理数据，并产生数据作为输出。
*   评估者——获取数据并提供一个转换器或模型，例如在训练时
*   预测-这使用单行或一组要素并预测结果。

![](img/1fd72e12ee8216e812c79c110fdc6bc6.png)

我们将在简单的回归样本中看到这些是如何发挥作用的。

# 房价样本

了解功能如何适应数据准备的典型工作流，使用测试数据集和使用模型来训练模型和评估适合度。我研究了如何实现一个简单的回归应用程序，在给定大约 800 套房屋销售的一组简单特性的情况下，预测房屋的销售价格。在示例中，我们将看一看多元线性回归的监督学习问题。监督学习任务用于从一组特征中预测标签的值。在这种情况下，我们希望使用一个或多个特征来预测房屋的销售价格(标签)。

重点是让一个小样本启动并运行，然后可以用它来试验特性和训练算法的选择。你可以在这里找到这篇文章的代码

我们将使用一组销售数据来训练该模型，以便在给定一组超过 800 套房屋销售的特征的情况下预测房屋的销售价格。虽然样本数据具有各种各样的特征，但是开发有用系统的一个关键方面是理解所用特征的选择会影响模型。

# 开始之前

你可以找到一个 [10 分钟教程](https://dotnet.microsoft.com/learn/machinelearning-ai/ml-dotnet-get-started-tutorial/intro)，它将带你完成安装和先决条件——或者只使用以下步骤。

您需要安装 Visual Studio 16.6 或更高版本。安装了 NET Core 你就可以到达那里[这里](https://visualstudio.microsoft.com/downloads/?utm_medium=microsoft&utm_source=docs.microsoft.com&utm_campaign=button+cta&utm_content=download+vs2017)

开始建设。你只需要下载并安装[就可以了。NET SDK](https://download.visualstudio.microsoft.com/download/pr/48d03227-4429-4daa-ab6a-78840bc61ea5/b815b04bd689bf15a9d95668624a77fb/dotnet-sdk-2.2.104-win-gs-x64.exe) 。

通过在命令提示符下运行以下命令来安装 ML.Net 软件包

```
dotnet add package Microsoft.ML --version 1.4.0
```

# 数据类

我们的第一项工作是定义一个数据类，在加载我们的。房屋数据的 csv 文件。需要注意的重要部分是[LoadColumn()]属性；这些允许我们将字段与输入中的不同列相关联。它为我们提供了一种简单的方法来适应我们可以处理的数据集的变化。当用户想要预测房子的价格时，他们使用数据类来给出特征。请注意，在定型模型时，我们不需要使用类中的所有字段。

![](img/659e03bd57605817ae7b57efee1361f3.png)

# 培训和保存模型

CreateHousePriceModelUsingPipeline(…)方法在创建用于预测房价的模型时做了大部分有趣的工作。

代码片段显示了您可以如何:

*   从. csv 文件中读取数据
*   选择训练算法
*   选择训练时要使用的功能
*   处理字符串特征
*   创建一个管道来处理数据和训练模型

在这些类型的问题中，您通常需要归一化入站数据，考虑特征房间和卧室，房间的值范围通常大于卧室，我们归一化它们以对拟合具有相同的影响，并加快(取决于教练)拟合时间。我们使用的训练器会自动对特性进行规范化，尽管如果你自己需要的话，框架会提供工具来支持规范化。

类似地，根据所用特征的数量(如果模型过度拟合)，我们将正则化应用于训练器——这基本上保留了所有特征，但为每个特征参数添加了权重以减少影响。在示例中，培训师将处理规范化；或者，您可以在创建培训师时进行规范化调整。

![](img/32e2bbc0561802c8626604124f3fa356.png)

# 估价

训练后，我们需要使用测试数据评估我们的模型，这将表明预测结果和实际结果之间的误差大小。减少误差将是对相对小的数据集进行迭代过程的一部分，以确定最佳的特征组合。ML 支持不同的方法。NET 我们使用[交叉验证](https://en.wikipedia.org/wiki/Cross-validation_(statistics))来估计从一次运行到另一次运行的模型质量的方差，这也消除了提取单独的测试集进行评估的需要。我们显示质量度量来评估和获得模型的准确性度量

![](img/3def733132b77d8fa4b3ce985987cb17.png)

我们将看看 R 平方指标:

*   [R 平方](http://statisticsbyjim.com/regression/interpret-r-squared-regression/) —提供拟合优度的度量，因为线性回归模型将在 0 - > 1 之间，越接近 1 越好。

您可以在这里找到可用的[指标的描述](https://docs.microsoft.com/en-us/nimbusml/concepts/metrics)

“按原样”运行代码会产生 0.77 的 R 平方，我们希望对此进行改进。第一步是开始试验特性选择，也许从减少特性列表开始*在*之后，我们有了更清晰的理解，然后考虑获取更广泛的数据。

# 保存模型以备后用

存储用于预测房价的模型很简单

![](img/b5ddab521b718aafd2d191913a647dfb.png)

# 加载和预测房屋销售价格

一旦您调整了功能并评估了不同的培训，您就可以使用该模型来预测销售价格。我认为这是 ML.NET 框架的闪光点，因为我们可以在。Net 来支持使用该模型的不同方式。

![](img/61d98542715a5378919e88eb55e88850.png)

对于这种类型的 ML 应用程序，典型的用途是创建一个简单的 REST 服务，在部署到 Windows Azure 的 docker 容器中运行。一个用 javascript 编写的 web 应用程序使用该服务，让人们快速了解房子应该卖多少钱。

使用。Net Core 我们可以在不同的硬件平台上运行后端，Visual Studio 2019、2017 使健壮服务的创建、部署和管理变得快速而简单。

# 在 GitHub 中找到代码:

[GitHub](https://github.com/junwin/MLNetRegression)

*原载于 2019 年 2 月 21 日*[*junwin . github . io*](https://junwin.github.io/posts/2019-02-21-MLNetHousePriceRegression.html)*。*