# 使用 Python3 介绍 AWS Lambda、Layers 和 boto3

> 原文：<https://towardsdatascience.com/introduction-to-amazon-lambda-layers-and-boto3-using-python3-39bd390add17?source=collection_archive---------1----------------------->

## 面向数据科学家的无服务器方法

![](img/85f6344b05f7bb29f67554d62ee2cc0d.png)

Photo by [Daniel Eledut](https://unsplash.com/@pixtolero2?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

Amazon Lambda 可能是当今最著名的无服务器服务，提供低成本，几乎不需要云基础设施治理。它提供了一个相对简单明了的平台，用于在不同的语言上实现功能，比如 Python、Node.js、Java、C#等等。

亚马逊 Lambda 可以通过 AWS 控制台或者 [AWS 命令行界面](https://aws.amazon.com/cli/?nc1=f_ls)进行测试。Lambda 的一个主要问题是，一旦函数和触发器变得更复杂，设置起来就变得很棘手。本文的目标是向您展示一个易于理解的教程，让您用外部库配置第一个 Amazon Lambda 函数，并做一些比打印“Hello world！”更有用的事情。

我们将使用 Python3、boto3 和 Lambda 层中加载的其他一些库来帮助我们实现我们的目标，将 CSV 文件加载为 Pandas 数据帧，进行一些数据辩论，并将报告文件中的指标和图表保存在 S3 存储桶中。虽然使用 AWS 控制台来配置您的服务不是在云上工作的最佳实践方法，但我们将展示使用控制台的每个步骤，因为这对初学者来说更便于理解 Amazon Lambda 的基本结构。我相信在阅读完这篇教程之后，你会对将部分本地数据分析管道迁移到 Amazon Lambda 有一个很好的想法。

## 设置我们的环境

在我们开始使用 Amazon Lambda 之前，我们应该首先设置我们的工作环境。我们首先使用 [conda](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html) (也可以使用 [pipenv](https://pypi.org/project/pipenv/) )(2)为项目(1)和环境 Python 3.7 创建一个文件夹。接下来，我们创建两个文件夹，一个保存 Lambda 函数的 python 脚本，另一个构建 Lambda 层(3)。我们将在本文后面更好地解释 Lambda 层的组成。最后，我们可以创建文件夹结构来构建 Lambda 层，这样它就可以被 Amazon Lambda (4)识别。我们创建的文件夹结构将帮助您更好地理解 Amazon Lambda 背后的概念，并组织您的函数和库。

```
# 1) Create project folder
**mkdir medium-lambda-tutorial**# Change directory
**cd medium-lambda-tutorial/**# 2) Create environment using conda **conda create --name lambda-tutorial python=3.7
conda activate lambda-tutorial**# 3) Create one folder for the layers and another for the 
# lambda_function itself
**mkdir lambda_function lambda_layers**# 4) Create the folder structure to build your lambda layer
**mkdir -p lambda_layers/python/lib/python3.7/site-packages
tree** .
├── lambda_function
└── lambda_layers
    └── python
        └── lib
            └── python3.7
                └── site-packages
```

## 亚马逊 Lambda 基本结构

在尝试实现我的第一个 Lambda 函数时，我遇到的一个主要问题是试图理解 AWS 调用脚本和加载库所使用的文件结构。如果您按照默认选项“从头开始创作”(图 1)来创建 Lambda 函数，那么您最终会得到一个文件夹，其中包含您的函数名称和名为 **lambda_function.py** 的 Python 脚本。

![](img/0d499d4066659ece25c515895a1a1628.png)

Figure 1\. Creating a Lambda function using Author from scratch mode.

**lambda_function.py** 文件结构非常简单，代码如下:

```
import jsondef lambda_handler(event, context):
    # TODO implement
    return {
        'statusCode': 200,
        'body': json.dumps('Hello from Lambda!')
    }
```

这 8 行代码是理解 Amazon Lambda 的关键，所以我们将逐行解释它。

1.  `import json`:你可以导入 Python 模块在你的函数上使用，AWS 为你提供了一个已经在 Amazon Lambda 上构建的可用 Python 库的[列表，比如`json` 等等。当你需要不可用的库时，问题就出现了(我们稍后将使用](https://gist.github.com/gene1wood/4a052f39490fae00e0c3) [Lambda 层](https://docs.aws.amazon.com/lambda/latest/dg/configuration-layers.html)来解决这个问题)。
2.  `def lambda_handler(event, context):`这是[主函数](https://docs.aws.amazon.com/lambda/latest/dg/python-programming-model-handler-types.html)当你运行服务时，你的 Amazon Lambda 会调用这个函数。它有两个参数`event`和`context`。第一个用于传递可以在函数本身上使用的数据(稍后将详细介绍)，第二个用于提供运行时和元数据信息。
3.  这里就是奇迹发生的地方！您可以使用`lambda_handler`函数的主体来实现任何您想要的 Python 代码。
4.  `return`函数的这一部分将返回一个默认字典，其中`statusCode`等于 200，而`body`则返回一个“Hello from Lambda”。您可以在以后将这个返回更改为任何适合您需要的 Python 对象。

在运行我们的第一个测试之前，有必要解释一下与 Amazon Lambda 相关的一个关键话题: [**触发器**](https://docs.aws.amazon.com/lambda/latest/dg/invoking-lambda-functions.html) 。触发器基本上是调用 Lambda 函数的方式。有很多方法可以使用事件来设置触发器，比如将文件添加到 S3 桶，更改 DynamoDB 表上的值，或者通过 Amazon API Gateway 使用 HTTP 请求。你可以很好地集成你的 Lambda 函数，以供各种各样的 AWS 服务调用，这可能是 Lambda 提供的优势之一。我们可以这样做来与你的 Python 代码集成的一个方法是使用 [boto3](https://boto3.amazonaws.com/v1/documentation/api/latest/index.html) 来调用你的 Lambda 函数，这是我们将在本教程后面使用的方法。

正如您所看到的，AWS 提供的模板结构非常简单，您可以通过配置一个测试事件并运行它来测试它(图 2)。

![](img/c85085ce5b9bfc8f7c6a717ee81b6cbd.png)

Figure 2\. Running your first Amazon Lambda test.

由于我们没有对 Lambda 函数的代码做任何修改，测试运行了这个过程，我们收到了一个绿色的警告，描述了成功的事件(图 3)。

![](img/5fb0f8b89991c20dfb9d7f0cef81b01f.png)

Figure 3\. A successful call of our Lambda Function.

图 3 展示了 Lambda 调用结果的布局。在上面部分，您可以看到返回语句中包含的字典。下面是**摘要**部分，我们可以看到一些与 Lambda 函数相关的重要指标，如请求 ID、函数持续时间、计费持续时间以及配置和使用的内存量。我们不会深入讨论 [Amazon Lambda 定价](https://aws.amazon.com/lambda/pricing/?nc1=f_ls)，但重要的是要知道它的收费依据是:

*   功能运行的持续时间(四舍五入到最接近的 100 毫秒)
*   使用的内存/CPU 数量
*   请求的数量(调用函数的次数)
*   进出 Lambda 的数据传输量

总的来说，测试和使用它真的很便宜，所以在小工作负载下使用 Amazon Lambda 时，您可能不会有计费问题。

与定价和性能相关的另一个重要细节是 CPU 和内存的可用性。您选择运行您的函数的内存量，并且“ [Lambda 与配置的内存量成比例地线性分配 CPU 能力](https://docs.aws.amazon.com/lambda/latest/dg/resource-model.html)”。

在图 3 的底部，您可以看到**日志输出**会话，在这里您可以检查 Lambda 函数打印的所有执行行。在 Amazon Lambda 上实现的一个很棒的特性是它与 [Amazon CloudWatch](https://aws.amazon.com/cloudwatch/?nc1=f_ls) 集成在一起，在那里你可以找到你的 Lambda 函数生成的所有日志。关于监控执行和日志的更多细节，请参考 [Casey Dunham great Lambda 文章](https://stackify.com/aws-lambda-with-python-a-complete-getting-started-guide/)。

我们已经介绍了 Amazon Lambda 的基本功能，因此在接下来的几节课中，我们将增加任务的复杂性，向您展示一个真实世界的使用，提供一些关于如何在日常基础上运行无服务器服务的见解。

## 增加层次，拓展可能性

使用 Python 的一个好处是可以获得大量的库，帮助您实现快速解决方案，而不必从头开始编写所有的类和函数。如前所述，Amazon Lambda 提供了一个 Python 库[列表](https://gist.github.com/gene1wood/4a052f39490fae00e0c3)，您可以将其导入到您的函数中。当您不得不使用不可用的库时，问题就出现了。一种方法是将这个库安装在你的`lambda_function.py`文件所在的文件夹中，压缩这些文件，然后上传到你的 Amazon Lambda 控制台。在本地安装库并在每次创建新的 Lambda 函数时上传库，这个过程可能是一项费力且不方便的任务。为了让您的生活更轻松，亚马逊为我们提供了将我们的库作为 Lambda 层上传的可能性，它由一个文件结构组成，您可以在其中存储库，将其独立加载到 Amazon Lambda，并在需要时在您的代码中使用它们。一旦你创建了一个 Lambda 层，它可以被任何其他新的 Lambda 函数使用。

回到我们组织工作环境的第一个会话，我们将使用在`lambda_layer`文件夹中创建的文件夹结构在本地安装一个 Python 库 Pandas。

```
# Our current folder structure
.
├── lambda_function
└── lambda_layers
    └── python
        └── lib
            └── python3.7
                └── site-packages# 1) Pip install Pandas and Matplotlib locally
**pip install pandas -t lambda_layers/python/lib/python3.7/site-packages/.**# 2) Zip the lambda_layers folder
**cd lambda_layers**
**zip -r pandas_lambda_layer.zip ***
```

通过使用带参数`-t`的`pip`,我们可以指定要在本地文件夹(1)上安装库的位置。接下来，我们只需要压缩包含库(2)的文件夹，我们就有了一个准备作为层部署的文件。保持我们在开始时创建的文件夹的结构(python/lib/python 3.7/site-packages/)是很重要的，这样 Amazon Layer 就可以识别您的压缩包中包含的库。点击 AWS Lambda 控制台左侧面板上的“层”选项，然后点击“创建层”按钮创建一个新层。然后我们可以指定名称、描述和兼容的运行时(在我们的例子中是 Python 3.7)。最后，我们上传我们的压缩文件夹并创建层(图 4)。

![](img/506880828e09d3582a920c93e611a127.png)

Figure 4\. Creating a Lambda Layer.

这花了不到一分钟的时间，我们的 Amazon 层已经准备好用于我们的代码了。回到 Lambda 函数的控制台，我们可以通过点击图层图标，然后点击“添加图层”(图 5)来指定要使用的图层。

![](img/ccd15867b9234c6f7c5d9414a3d64870.png)

Figure 5\. Adding a layer to a Lambda function.

接下来，我们选择我们刚刚创建的层及其各自的版本(图 6)。从图 6 中可以看出，AWS 提供了一个 Lambda 层，其中包含随时可以使用的 Scipy 和 Numpy，因此如果您只需要这两个库中的一个，就不需要创建新的层。

![](img/dd8456a22f9fb3e8179a4e0a1930d08d.png)

Figure 6\. Selecting our new Pandas Layer.

选择我们的熊猫层后，我们需要做的就是把它导入你的 Lambda 代码，因为它是一个已安装的库。

## 最后，我们开始编码吧！

现在我们已经准备好了环境和熊猫层，我们可以开始编写代码了。如前所述，我们的目标是创建一个 Python3 本地脚本(1)，它可以使用定义的参数(2)调用 Lambda 函数，在位于 S3 (3)上的 CSV 上使用 Pandas 执行简单的数据分析，并将结果保存回同一个桶(4)(图 7)。

![](img/b3b41a3031f279d2ed00ede4f0742ae2.png)

要让 Amazon Lambda 访问我们的 S3 存储桶，我们只需在控制台上的会话 [**执行角色**](https://docs.aws.amazon.com/lambda/latest/dg/lambda-intro-execution-role.html) 中添加一个角色。尽管 AWS 为您提供了一些角色模板，但我的建议是在 IAM 控制台上创建一个新的角色，以准确地指定 Lambda 函数所需的权限(图 8 的左侧面板)。

![](img/f570c6170ccd2d049f05ab9202505b9b.png)

Figure 8\. Defining the role and basic settings for Lambda function.

我们还将可用内存量从 128MB 更改为 1024MB，并将超时时间从 3 秒更改为 5 分钟(图 8 中的右图)，以避免内存不足和超时错误。[亚马逊 Lambda 限制](https://docs.aws.amazon.com/lambda/latest/dg/limits.html)RAM 内存总量为 3GB，超时为 15 分钟。因此，如果您需要执行高度密集的任务，您可能会发现问题。一种解决方案是将多个 Lambdas 链接到其他 AWS 服务，以执行分析管道的步骤。我们的想法不是对 Amazon Lambda 进行详尽的介绍，所以如果你想了解更多，请查看这篇来自艾毅的文章。

在展示代码之前，描述我们将在小项目中使用的数据集是很重要的。我选择了来自 Kaggle 的 [Fifa19 球员数据集，这是一个 CSV 文件，描述了游戏中出现的球员的所有技能(表 1)。它有 18.207 行和 88 列，你可以从每个球员那里得到国籍，俱乐部，薪水，技能水平和更多的信息。我们下载了 CSV 文件，并上传到我们的 S3 桶(重命名为](https://www.kaggle.com/karangadiya/fifa19) [fifa19_kaggle.csv](https://medium-lambda-tutorial-bucket.s3.amazonaws.com/fifa19_kaggle.csv) )。

Table 1\. The fifa19_kaggle.csv 20 first rows.

所以现在我们可以专注于我们的代码了！

正如我们在上面的脚本中看到的，前 5 行只是导入库。除了熊猫，其他所有的库都可以使用，而不必使用层。

接下来，我们有一个名为`write_dataframe_to_csv_on_s3` (第 8 到 22 行)的附属函数，用于将熊猫数据帧保存到一个特定的 S3 桶中。我们将使用它来保存在分析过程中创建的输出数据帧。

我们代码中的另一个函数是 main `lambda_handler,`,当我们调用 Lambda 时，这个函数将被调用。我们可以看到，`lambda_handler`上的前 5 个赋值(第 28 到 32 行)是传递给`event`对象的变量。

从第 35 行到第 41 行，我们使用 boto3 下载 s 3 存储桶上的 CSV 文件，并将其作为熊猫数据帧加载。

接下来，在第 44 行，我们使用 Dataframe 上的 group by 方法来聚合`GROUP`列，并获得`COLUMN`变量的平均值。

最后，我们使用函数`write_dataframe_to_csv_on_s3`在指定的 S3 桶上保存`df_groupby`，并返回一个以 statusCode 和 body 为键的字典。

正如之前在 Amazon Lambda 基本结构会话中所描述的，事件参数是一个对象，它携带了可用于`lambda_handler`函数的变量，我们可以在配置测试事件时定义这些变量(图 9)。

![](img/a51a50644d5537604cde72b96ede8318.png)

Figure 9\. Defining the variables on the test event.

如果我们运行这个测试，使用与测试 JSON 的 5 个键相关的正确值，我们的 Lambda 函数应该处理来自 S3 的 CSV 文件，并将结果 CSV 写回到桶中。

尽管在测试事件中使用硬编码的变量可以展示我们的 Lambda 代码的概念，但这不是调用函数的实际方法。为了解决这个问题，我们将创建一个 Python 脚本(`invoke_lambda.py`)来使用 boto3 调用我们的 Lambda 函数。

我们将只使用三个库: **boto3** 、 **json** 和 **sys** 。从第 5 行到第 10 行，当通过命令行运行脚本时，我们使用`sys.argv`来访问参数。

```
**python3 invoke_lambda.py <bucket> <csv_file> <output_file> <groupby_column> <avg_column> <aws_credentials>**
```

我们提供给`invoke_lambda.py`的最后一个参数(aws_credentials)是一个 JSON 文件，其中包含我们访问 aws 服务的凭证。您可以使用 [awscli](https://aws.amazon.com/cli/?nc1=h_ls) 配置您的凭证，或者使用 [IAM](https://docs.aws.amazon.com/IAM/latest/UserGuide/id_credentials_access-keys.html) 生成密钥。

在我们的主函数`invoke_lambda`中，我们使用 [boto3 客户端](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html)来定义对 Amazon Lambda 的访问(第 38 行)。下一个名为`payload`的对象是一个字典，包含了我们想要在 Lambda 函数中使用的所有变量。这些是可以使用`event.get('variable').`访问的 Lambda 变量

最后，我们简单地用目标 Lambda 函数名、调用类型和携带变量的有效负载来调用`client.invoke()`(第 54 行)。[调用类型有三种](https://boto3.amazonaws.com/v1/documentation/api/1.9.42/reference/services/lambda.html#Lambda.Client.invoke) : **请求响应**(默认)**、**到**同步调用函数。保持连接打开，直到函数返回响应或超时"；**事件**，异步调用 Lambda 或者当您需要验证用户信息时 **DryRun** 。对于我们的主要目的，我们将使用默认的 RequestResponse 选项来调用我们的 Lambda，因为它等待 Lambda 流程返回响应。由于我们在 Lambda 函数上定义了一个 try/except 结构，如果流程运行时没有错误，它将返回一个状态代码 200，并显示消息“Success！”，否则它将返回状态代码 400 和消息“错误，错误的请求！”。**

**当使用正确的参数运行时，我们的本地脚本`invoke_lambda.py`需要几秒钟才能返回响应。如果响应是肯定的，状态代码为 200，那么您可以检查您的 S3 存储桶来搜索由 Lambda 函数生成的报告文件(表 2)。当我们使用“俱乐部”和“总体”两列进行分组以获得平均值时，我们显示了平均玩家总体技能水平最高的 20 个俱乐部。**

**Table 2\. The first 20 rows of the output CSV file generated using our Lambda function.**

## **最终考虑**

**希望这个快速介绍(没那么快！)帮助你更好地理解这种无服务器服务的具体细节。它可以帮助您在数据科学项目中尝试不同的方法。有关使用 AWS 的无服务器架构的更多信息，请查看 Eduardo Romero 的这篇精彩文章。**

**如果你觉得你需要更深入地了解 AWS Lambda，我最近发表了一篇文章，描述了 Lambda 背后的基础设施和它的一些其他功能。**

## **非常感谢你阅读我的文章！**

*   **你可以在我的[个人资料页面](https://medium.com/@gabrielsgoncalves) **找到我的其他文章🔬****
*   **如果你喜欢它并且**想成为中级会员**，你可以使用我的 [**推荐链接**](https://medium.com/@gabrielsgoncalves/membership) 来支持我👍**

## **更多资源**

**[](https://medium.com/@gergoszerovay/why-you-need-python-environments-and-how-to-manage-them-with-conda-protostar-space-cf823c510f5d) [## 为什么需要 Python 环境以及如何使用 Conda-protostar . space 管理它们

### 我不应该只安装最新的 Python 版本吗？

medium.com](https://medium.com/@gergoszerovay/why-you-need-python-environments-and-how-to-manage-them-with-conda-protostar-space-cf823c510f5d) [](https://stackify.com/aws-lambda-with-python-a-complete-getting-started-guide/) [## 使用 Python 的 AWS Lambda:完全入门指南

### 在这篇文章中，我们将了解什么是亚马逊网络服务(AWS) Lambda，以及为什么它可能是一个好主意，用于您的…

stackify.com](https://stackify.com/aws-lambda-with-python-a-complete-getting-started-guide/) [](https://medium.com/hackernoon/build-a-serverless-data-pipeline-with-aws-s3-lamba-and-dynamodb-5ecb8c3ed23e) [## 用 AWS S3 兰巴和 DynamoDB 构建无服务器数据管道

### AWS Lambda plus Layers 是管理数据管道和实现无服务器管理的最佳解决方案之一

medium.com](https://medium.com/hackernoon/build-a-serverless-data-pipeline-with-aws-s3-lamba-and-dynamodb-5ecb8c3ed23e) [](https://medium.com/@eduardoromero/serverless-architectural-patterns-261d8743020) [## 无服务器架构模式

### 构建在 AWS 无服务器堆栈之上的云架构模式。

medium.com](https://medium.com/@eduardoromero/serverless-architectural-patterns-261d8743020) [](/diving-deeper-into-aws-lambda-a52b22866767) [## 深入探究 AWS Lambda

### 了解 Lambda 的基本基础设施，如何运行可执行文件和解析 CloudWatch 日志

towardsdatascience.com](/diving-deeper-into-aws-lambda-a52b22866767)**