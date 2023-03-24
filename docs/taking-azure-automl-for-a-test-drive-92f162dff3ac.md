# 试用 Azure AutoML

> 原文：<https://towardsdatascience.com/taking-azure-automl-for-a-test-drive-92f162dff3ac?source=collection_archive---------29----------------------->

![](img/c631f49d6a43bd57ee215eb5d8320fe5.png)

[微软](https://medium.com/u/940e606ec51a?source=post_page-----92f162dff3ac--------------------------------)最近在预览版中引入了 Azure AutoML。我喜欢自动化机器学习的概念，因为这可能是发生在非数据科学家身上的最好的事情。它把数据科学带到大众中，一次一个自动化实验。

# 何时使用自动化 ML

Automated ML 使机器学习模型开发过程民主化，并使其用户(无论他们的数据科学专业知识如何)能够识别任何问题的端到端机器学习管道。

各行业的数据科学家、分析师和开发人员可以使用自动化 ML 来:

*   无需丰富的编程知识即可实现机器学习解决方案
*   节省时间和资源
*   利用数据科学最佳实践
*   提供敏捷的问题解决方案

在下面的例子中，我使用的是 Kaggle telcos 流失数据集，这里是。

第一步是创建机器学习工作空间。

![](img/686391d3c2a22716d0e476891851eef3.png)

Creating a machine learning workspace (note I have hidden some details)

单击“create ”(创建)后，您将看到下面的屏幕:

![](img/94bc6a99c9f785ac6feaf896f2539abd.png)

Once the machine learning workspace is created you will see the above screen

您也可以查看详细信息，单击部署详细信息，您将看到以下详细信息:

![](img/ae4bd5da5f8ff3e1c2bf5d0b677a84e9.png)

The various resources that get spun up as part of your machine learning workspace

完成后，点击“转到资源”

![](img/3e3165a619d4fc0ecc3ac1523359e624.png)

Once you click on Go to Resource you will see the above.

点击“创建新的自动化机器学习模型(预览)”

您将看到以下屏幕:

![](img/ea599a229e89424ea7939b708fcc3d2c.png)

在下一步中，单击“创建实验”

![](img/d540dd74bb93591119fd116ad5924363.png)

Give your experiment a name

创建新计算机

![](img/731f115b109ecfa669b3cebf9b19eeb0.png)

A standard DS12 V2 machine

单击附加设置，并将最小节点设置为 1。

![](img/c8b9da8c3278c1f003a5357ed72774b2.png)

change the minimum number of nodes to 1

![](img/34599615c4b1d9182290403e4098d8dd.png)

This can take a couple of minutes

现在选择计算机并上传电信流失文件。

![](img/5f400b381678259306022fbce0bf5a0c.png)

选择文件并点击上传按钮。

![](img/3c7ec3ea774cf380bae7e24ce59af226.png)

您将看到显示的列。

我们还有数据分析选项，可以帮助您了解更多关于数据的信息。

![](img/6395245454e9dec2c236cfb1123f0f64.png)

查看配置文件，我不希望模型在 CustomerID 上训练

![](img/3346e9d70783656722cb76dcc5423b9e.png)

Notice I have ignored the column CustomerID

我们试图解决的三种机器学习问题，我们有三种选择。

![](img/249d91db3ffc18bbd29ef0a2a7bcc4ea.png)

我将使用分类，因为我试图预测客户流失，因此目标列是客户流失。

![](img/6e9fcb4597e262f76c07febef649e1e3.png)

(可选)高级设置:可用于更好地控制培训作业的附加设置。

# 自动(标准)预处理

在每一个自动化机器学习实验中，你的数据都会被自动缩放或归一化，以帮助算法表现良好。在模型训练期间，将对每个模型应用以下缩放或标准化技术之一。

![](img/f63149e4c2879c8547f90b00abd7d4f0.png)

下一步是我们点击加速器并点击开始！

![](img/cf263bfb869f0eadd859090a6fb12aee.png)

一段时间后，这些值会得到更新:

![](img/e74ab09ff99c07448eab30f71f96ba96.png)

当它仍在运行时，您可以查看它在自动化和调优过程中运行的各种算法。：

![](img/af81650cb88c2a98d1cc7a406ddc83e1.png)

您还可以下载模型，并对其进行调整以适应您的用例。

一旦运行完成。

![](img/4bb7df882a88b8519ec7121cc22b522f.png)

它会自动选择最佳模型供您下载和部署。

![](img/5f6299d76c6bcf819be1723207195a55.png)

然后，您可以下载并部署该模型！

我将在另一篇文章中介绍下载和部署。

AutoML 对于人工智能社区来说是一个令人兴奋的创新，并且确实是科学上另一个突破的机会。

注意:这里表达的观点是我个人的，不代表我的雇主的观点。