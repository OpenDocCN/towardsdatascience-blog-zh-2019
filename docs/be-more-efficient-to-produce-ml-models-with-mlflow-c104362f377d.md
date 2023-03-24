# 使用 mlflow 生成 ML 模型更有效

> 原文：<https://towardsdatascience.com/be-more-efficient-to-produce-ml-models-with-mlflow-c104362f377d?source=collection_archive---------13----------------------->

你好，在这篇文章中，我将在去年推出的一款名为 mlflow 的工具上做一个实验，以帮助数据科学家更好地管理他们的机器学习模型。

![](img/309893870826b23bc4185a87abb8f877.png)

本文的想法不是为我将要构建机器学习模型的用例构建完美的模型，而是更深入地研究 mlflow 的功能，并了解如何将其集成到 ML 管道中，以便为数据科学家/机器学习工程师的日常工作带来效率。

# mlflow kezako？！

[mlflow](https://mlflow.org/) 是由 [databricks](https://databricks.com/) 开发的 python 包，被定义为机器学习生命周期的开源平台。围绕 mlflow()有三个支柱。

![](img/9da1692df5023ccc9537a40a7abf6076.png)

他们的[文档](https://www.mlflow.org/docs/latest/index.html)真的很棒，他们有很好的[教程](https://www.mlflow.org/docs/latest/tutorial.html)来解释 mlflow 的组件。对于这篇文章，我将把我的测试集中在 mlflow 的跟踪和模型部分，因为我将诚实地告诉你，我没有看到项目部分的要点(看起来像一个 conda 导出和一个以特定顺序运行 python 脚本的配置文件),但我确信它可以在 ml 管道的再生方面帮助一些人。

现在让我们看一下我想用来测试 mlflow 的案例。

# 用例的描述

为了测试 mlflow，我将使用我在 2017 年完成 Udacity ML 工程师纳米学位时使用的相同用例:

> ***建立法国用电量预测系统***

你可以在我的 [Github 库](https://github.com/jeanmidevacc/udacity_mlen)的这个[文件夹](https://github.com/jeanmidevacc/udacity_mlen/tree/master/capstone)中找到我此刻制作的关于纳米度的所有资源。

我不打算在数据分析中输入太多细节，你可以在存储库中的报告中找到这些数据，但基本上法国的电力消耗是季节性的。

![](img/b7f8a2da1f2dfbc771809417c582300d.png)

由于家庭大部分时间都在使用电暖，耗电量很大程度上取决于室外温度。

![](img/1cbf768276df86fe6d3b5e3df787e626.png)

关于这一分析更重要的是，为了在 2019 年重新运行它，我回到 RTE 的 opendata 网站，我惊喜地看到[网站](https://opendata.reseaux-energies.fr/pages/accueil/)通过在平台上添加更多数据而发展(他们与其他能源管理公司建立了联系)，所以现在有了能源消耗数据和一些额外的信息，例如[地区天气](https://opendata.reseaux-energies.fr/explore/?refine.theme=M%C3%A9t%C3%A9orologie&sort=modified)。

对于这个项目，我将使用以下功能来训练一个模型:

*   当天的信息，如星期几、月份、星期几，以及是否在法国度假[假期](https://pypi.org/project/holidays/)
*   关于法国每日室外温度(最低、平均和最高)的信息，一个是每个地区室外温度的全球平均值(avg ),另一个是基于每个地区人数的加权平均值

您可以在这个[存储库](https://github.com/jeanmidevacc/mlflow-energyforecast)中找到处理来自 open 的原始数据的[笔记本](https://github.com/jeanmidevacc/mlflow-energyforecast/blob/master/1.build_dataset.ipynb)

对于建模部分，正如我所说，这个想法不是建立一个超级模型来预测 99.999%的能源消耗，而是更多地了解如何将 mlflow 集成到我一年前为我的纳米学位建立的管道中。

我将测试以下型号:

*   来自 scikit learn 的 [KNN 回归器](https://scikit-learn.org/stable/modules/generated/sklearn.neighbors.KNeighborsRegressor.html)(具有各种参数和特征)
*   来自 scikit learn 的 [MLP 回归器](https://scikit-learn.org/stable/modules/generated/sklearn.neural_network.MLPRegressor.html)(具有各种参数和特征)
*   一个手工制作的分段线性回归叫做 [PTG](http://www.ibpsa.org/proceedings/BS2015/p2854.pdf)

该想法将使用 2016 年至 2019 年的数据(不包括)对 2019 年的数据进行训练和测试算法。

# 机器学习管道的描述

在下图中，我设想了一个简单的 ml 流管道来回答这种用例。

![](img/a111494bc856c3e2fd3e5bf355c986f7.png)

这个 ML 管道有 4 个主要步骤:

*   第一个步骤是收集所有数据，构建用于制作预测器的特征
*   在测试和构建阶段，将使用训练数据构建具有正确参数的正确模型，并在测试集上进行测试
*   服务部分，当你有正确的模型时，你需要服务它，用它来做预测
*   当您有新数据时，预测部分将对这些数据进行处理，以对可用于使用所提供的模型进行预测的要素进行转换

**那么 mlflow 在这条管道中的位置是什么？**

对我来说，这个库非常适合第 2、3 步，第 4 步也有一点点，能够涵盖所有这些范围真的很棒。

现在让我们更详细地了解一下 mlflow 的用法。

# 测试和构建模型(物流跟踪)

正如我们之前所说的，我们需要找到合适的模型来解决这个预测问题。为了初始化管道，我们需要定义一个实验，即**电力消费预测**，它也有一个实验(本例中为 1)。

模型(模型或参数)的每个测试将被定义为 mlflow 中的一次运行(并由 runid 标记)，并存储在将出现在当前文件夹中的 mlruns 文件夹中。

为了组织我的代码，我保留了 mlflow 教程中的相同结构。这是我用来训练 KNN 回归器的一段代码的例子。

使用这种方法，每个模型都由模型的种类和用于构建它的特性来标记。对于评估指标，我的评估重点是:

*   精度指标，如 [RMSE](https://en.wikipedia.org/wiki/Root-mean-square_deviation) 、 [R 平方](https://en.wikipedia.org/wiki/Coefficient_of_determination)或[平均绝对误差](https://en.wikipedia.org/wiki/Mean_absolute_error)
*   执行 model.fit 和 model.predict 的时间，因为选择一个好的模型需要的不仅仅是准确性。

我将这种方法应用于所有模型，所有数据都存储在 mlruns 文件夹中。

要使用 mlflow UI，只需在您的工作目录中执行以下命令。

您应该通过页面 localhost:5000 访问 UI，并选择正确的实验。有一个实验的主页面截图。

![](img/bbd801160c13d262b9c598f9eb5f8a35.png)

在这个页面上，有为每个模型计算的所有指标，有可能已经关联的所有标签，还有一些关于用户和记录的模型位置的额外信息。在该页面中，我们可以通过单击运行日期来访问每次运行。

![](img/f12fe6d8cf1756004cc0141aa4e7ef2e.png)

在运行页面中有:

*   您可以在“参数”部分找到模型中应用的参数
*   运行期间计算的指标
*   与模型相关联的标签

UI 的另一个重要部分是工件，其中有包含模型信息的文件夹的组织。

从主页面，您可以选择所有可以在它们之间进行比较的模型。比较面板包含两个面板，一个面板带有一个表，将模型与所有分解的指标并排放置。

![](img/ce99a264b303087182041c58608dc99e.png)

还有另一个面板可以用[和](https://plot.ly/)进行可视化，并比较它们之间的模型。我快速制作了这个可视化面板的动画。

![](img/4be5091251f2605c09d744f0306db538.png)

我发现这最后一个功能非常有限，因为不同类别的模型之间的比较(如 KNN 与 MLP)似乎更适合家庭模型的比较。

但是这个 UI 并不是进行日志分析的唯一方法，所有的日志都可以通过 python 中的命令收集到一个数据帧中。

```
mlflow.search_runs(experiment_ids="1")
```

这样，您就可以在 python 环境中进行更深入的分析，例如，根据 RMSE 分数获得最佳模型。有一个从 Python 中的日志进行的简单分析。

![](img/82c50f314cf75cb74ad4d41d453459cb.png)

最佳模型有一个特定的 runid，可以在执行部署后使用。

# 模型服务(mlflow 模型)

使用 mlflow，您可以使用以下命令快速部署本地模型:

```
mlflow serve 
-m path_to_the_model_stored_with_the_logfuction_of_mlflow 
-p 1234
```

只需要用-m 将 mlflow serve 命令引导到模型的文件夹中，并分配一个新端口(默认端口与 mlflow UI 的端口相同，这可能会很烦人)

要执行预测调用，您需要向此 API 的/invocation 端点发送一个 POST 请求，并在此参数上包含一个已经在 json 中使用 orient split 进行了转换的数据帧(使用转换代码将更容易理解👇)

```
toscore = df_testing.to_json(orient = "split")
```

之后，您可以通过 Postman 或 Python 的 POST 请求轻松调用 API

```
import requests 
import json 
endpoint = "http://localhost:1234/invocations" 
headers = {"Content-type": "application/json; format=pandas-split"} response = requests.post(endpoint, json = json.loads(toscore) , headers=headers)
```

但是现在最大的问题是**如何在网上部署它？**

mlflow 做得很好，你有内置函数来快速地在[微软 AzureML](https://mlflow.org/docs/latest/models.html#azureml-deployment) 或 [AWS Sagemaker](https://mlflow.org/docs/latest/models.html#sagemaker-deployment) 上部署模型。因为我更喜欢 AWS，所以我将把部署重点放在 AWS Sagemaker 上。

部署分两个阶段:

```
mlflow sagemaker build-and-push-container
```

当然，该命令必须在安装了 [Docker](https://www.docker.com/) 的机器上运行，并且 AWS 用户拥有在 AWS 上部署东西的正确权限，例如，我的管理员访问权限可能不是好事，但 YOLO。

```
mlflow sagemaker deploy
```

但即使在 Windows 或 Linux 上，它对我也不起作用。所以我尝试了这篇博文中的另一种方法。在这个要点中有一个代码总结。

部署非常快(大约 10 分钟)，然后你可以用这段代码在 AWS 上调用 API deploy(来自 [databricks 博客](https://docs.databricks.com/_static/notebooks/mlflow/mlflow-quick-start-deployment-aws.html)

因此，可以调用该模型来预测该端点的能耗(这种方法由 AWS 用户处理所有的身份验证，所以我想这是非常安全的)。

我真的很喜欢在 mlfow 上进行实验，机器学习的版本控制通常是一个大话题，但在我目前的公司育碧(如果你想加入育碧大家庭，也许会有一份工作给你[在这里](https://www.ubisoft.com/en-us/careers/search.aspx))它开始成为一个真正的大问题。

在我看来， **mlflow 真的很棒**做线下实验找到合适的模型，快速做出原型。

我仍然对一些非常具体的问题有些担忧:

*   在 ml 模型中，有算法，但数据也非常重要，所以标签功能可以用于提供所用数据的信息，但在我看来，这可能还不够
*   模型类需要从熊猫的数据框架中做出预测，这很好，但可能也有一点限制
*   缺乏在特性/数据上自动记录模型的方法，这些特性/数据用于帮助调用 API 背后的模型部署
*   管理 Tensorflow 的模型看起来超级复杂(pytorch 看起来更容易)，但最后一点可能是因为我对这些框架不太熟悉(在 Tensorflow 与 Pytorch 的战争中没有立场😀 ).

但是说实话，你做了一个非常棒的工具来帮助数据科学家进行实验

*原载于 2019 年 11 月 13 日*[*the-odd-dataguy.com*](http://the-odd-dataguy.com)*。*