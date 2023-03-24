# 用 Docker 在 AWS 云中训练神经网络的实例

> 原文：<https://towardsdatascience.com/a-practical-example-of-training-a-neural-network-in-the-aws-cloud-with-docker-b44a8c061e6e?source=collection_archive---------23----------------------->

## 如何在 AWS GPU 实例上的 Docker 内在 CIFAR-10 上的 InceptionV3 模型之上训练浅层神经网络

![](img/60a145efd22740a25fc545826c9580e4.png)

我的上一篇文章[数据科学过程中 Docker 的用例示例](https://jenslaufer.com/data/science/use-cases-of-docker-in-the-data-science-process.html)是关于数据科学中的 Docker。这次我想用一个实际的例子来弄脏我的手。

在这个案例研究中，我想向您展示如何在 AWS 上 Docker 容器内的 CIFAR-10 图像上的深度 InceptionV3 模型之上训练一个浅层神经网络。我正在为这个项目使用 Python，Tensorflow 和 Keras 的标准技术栈。这个项目的[源代码可以在 Github 上获得。](https://github.com/jenslaufer/neural-network-training-with-docker)

您将从本案例研究中学到什么:

*   从 docker-machine 命令行设置 AWS 上支持 GPU 的云实例
*   在您的 docker 文件中使用 tensorflow docker 图像
*   使用 docker-compose 设置用于训练神经网络的多容器 Docker 应用程序
*   将 MongoDB 设置为持久性容器，用于模型的训练元数据和文件存储
*   用 MongoDB 实现简单的数据插入和查询
*   一些简单的 docker、docker-compose 和 docker-machine 命令
*   卷积神经网络的迁移学习

让我们定义这个小项目的要求:

*   培训必须在 AWS 中的一个支持 GPU 的实例上进行
*   灵活地将整个培训管道移植到谷歌云或微软 Azure
*   使用 nvidia-docker 在云实例上激活完整的 GPU 能力
*   MongoDB 上模型元数据的持久性以实现模型的可复制性。
*   多容器应用程序(训练容器+ MongoDB)的 docker-compose 用法
*   使用 docker-machine 管理云实例，并通过 docker-compose 从本地命令行开始培训

让我们潜入更深的地方。

# 1.先决条件

1.  安装 [Docker](https://docs.docker.com/install/) 连同 [Docker Machine](https://docs.docker.com/machine/) 和 [Docker 组成](https://docs.docker.com/compose/)(工具安装在 Mac 和 Windows 上的标准 Docker 安装中)
2.  在 [AWS](https://aws.amazon.com) 上创建账户
3.  安装和设置 [AWS 命令行客户端](https://github.com/aws/aws-cli)

# 2.AWS 实例作为 Docker 运行时环境

要在 AWS 上训练神经网络，首先需要在那里设置一个实例。你可以从 [AWS Web 控制台](https://console.aws.amazon.com/ec2/v2/home?region=us-east-1#Home:)或者从命令行使用 [AWS 命令行客户端](https://github.com/aws/aws-cli)来完成这项工作。

我用 docker-machine 命令向您展示了第三种方便的方法。该命令包装了不同云和本地提供商的驱动程序。通过这种方式，你可以获得谷歌云计算、微软 Azure 和亚马逊 AWS 的独特界面，这使得在平台上设置实例变得容易。请记住，一旦建立了实例，您就可以出于其他目的重用它。

我正在用 Ubuntu 18.04 Linux(ami-0891 F5 DC c59 fc 5285)创建一个 AWS 实例，它已经安装了 [CUDA 10.1](https://developer.nvidia.com/cuda-zone) 和 [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 。这些组件是为培训启用 GPU 所必需的。AMI 的基础是一个标准的 AWS Ubuntu 18.04 Linux 实例(ami-0a313d6098716f372)，我用这些组件对其进行了扩展。我把这张照片分享给公众，让生活变得更容易。

我使用 p2.xlarge 实例类型，这是 AWS 上最便宜的 GPU 实例。p2.xlarge 实例类型为您配备了 Tesla K80 的 GPU 能力。

```
docker-machine create --driver amazonec2\
 --amazonec2-instance-type p2.xlarge\ 
 --amazonec2-ami ami-0891f5dcc59fc5285\ 
 --amazonec2-vpc-id <YOUR VPC-ID>\
 cifar10-deep-learning
```

你需要一个 VPC 身份证来设置。您可以使用 AWS 命令来获得它:

您也可以从 [AWS Web 控制台](https://console.aws.amazon.com/vpc/home?region=us-east-1#vpcs:sort=VpcId)获取 VPC ID

有关更多信息，请查看带 AWS 文档的[对接机。](https://docs.docker.com/machine/drivers/aws/)

***警告:p2.xlarge 每小时收费 0.90 美元。请不要忘记在完成培训课程后停止实例***

# 3.培训脚本

您希望使用不同的训练参数来训练神经网络，以找到最佳设置。训练之后，您在测试集上测试模型质量。这是一个分类问题。为了简单起见，我建议使用准确性度量。最后，您将训练日志、模型权重和架构持久化，以备将来使用。这样一切都是可复制和可追踪的。

你可以通过用你的浅层网络替换基本模型的顶层来进行迁移学习，然后你冻结基本模型的权重并在整个网络上进行训练。

在这个案例研究中，我的做法有所不同。我正在移除基本模型的顶层，然后将图像输入到基本模型中，并在 MongoDB 中持久化结果特性。预测比训练需要更少的计算能力，一旦提取出瓶颈特征，我可以重用它们。你在瓶颈特征上训练浅层网络。

培训脚本的输入和输出要求:

**输入**

Docker 容器是从 MongoDB 集合参数化而来的，其中包含一个训练会话的所有参数。

*   损失函数
*   乐观者
*   批量
*   时代数
*   用于训练的所有样本的子集百分比(用于使用较少图像测试管道)

![](img/4732edca27c9b23085dbccff47c0186c.png)

**输出**

*   模型架构文件
*   模型权重文件
*   培训课程日志
*   测试集上的模型准确性

![](img/7bf3aa3ea6e6241b7599a6a44eaf769a.png)

我将整个培训管道放在一个脚本 [src/cnn/cifar10.py](https://github.com/jenslaufer/neural-network-training-with-docker/blob/c045323c372bb46535f563c456117a8befa4b05f/src/cnn/cifar10.py) 中，它由整个培训管道的一个类组成:

1.  将 CIFAR-10 映像下载到容器文件系统。
2.  加载带有 imagenet 权重的基础模型(InceptionV3)并移除顶层
3.  训练图像和测试图像的瓶颈特征提取:保持 MongoDB 中的特性以备将来使用。
4.  浅层神经网络的创建和编译；在 MongoDB 中持久化模型架构
5.  浅层模型的训练；MongoDB 中模型权重和训练日志的持久化
6.  测试集上的模型测试；在 MongoDB 中保持准确性度量

# 4.集装箱化

## a.)Dockerfile

训练神经网络所需的一切我都放入了 Dockerfile，它定义了训练的运行时环境。

![](img/e13b86f9b097fc9f84ee1210b3643eed.png)

*第 1 行:基础图像的定义。设置和配置继承自该映像。用的是官方的 tensorflow 图，有 python3 和 GPU 支持。*

*第 3 行:本地目录 src 中的所有内容，像训练脚本和入口点，都被复制到 Docker 映像中。*

*第 5 行:容器在 src 目录下启动*

*第 7 行:python 安装需求*

*第 9 行:将 src 目录添加到 PYTHONPATH 中，告诉 python 在这个目录中寻找模块*

*第 11 行:图像入口点的定义。这个入口点脚本在容器启动时执行。这个脚本启动了我们的 python 训练脚本。*

入口点 shell 脚本非常简单明了:它不带参数启动 python 模块。该模块在启动时从 MongoDB 获取训练参数。

## b.)Docker 容器构建

首先，我需要建立一个码头工人的形象。你可以跳过这一步，因为我在 Docker Hub 上分享了现成的 [Docker 映像。第一次引用图像时，会自动下载该图像。](https://hub.docker.com/r/jenslaufer/neural-network-training-with-docker/tags)

## c.)多容器应用程序

我的设置中有两个 docker 容器:一个用于训练的 Docker 容器和一个用于保存元数据和作为文件服务器的 MongoDB。

在这个场景中使用 docker-compose。您在 docker-compose.yml 中定义了构成应用程序的容器

![](img/5311af3ae89145d0568855f658605bf4.png)

*第 4–5 行:使用带有标签 0.1.0-GPU 的 jenslaufer/neural-network-training-with-docker 图像的训练容器的定义。该图像自动从公共 Docker Hub 库下载*

*第 7 行:tensorflow 的运行时环境*

*第 9 行:训练容器需要 trainingdb 容器来执行。在代码中，您使用 mongodb://trainingdb 作为 Mongo URI*

*第 11–12 行:MongoDB 数据库的定义。来自 Docker Hub 的官方 mongo 镜像用于版本 3.6.12*

*第 14–15 行:内部端口 27017 在外部端口 27018 可用*

*第 16 行:Mongo 守护进程启动*

您可以看到，使用 docker compose 设置多应用程序非常简单——只需几行代码就可以设置一个数据库，而无需复杂的安装例程。

# 5.训练神经网络

您需要执行这个命令来确保 docker 命令是针对我们的 AWS 实例的:

```
docker-machine env cifar10-deep-learning
```

之后，您可以列出您的计算机

```
NAME                  ACTIVE DRIVER    STATE   
cifar10-deep-learning *      amazonec2 Running
```

确保您看到活动环境的星号。这是所有 docker 命令执行的环境。请记住，您是在本地 shell 中执行命令的。很方便。

现在，您可以第一次启动容器。

Docker 将所有图像下载到 AWS 实例。MongoDB 启动并一直运行，直到您停止容器。用对接器训练的神经网络执行训练模块。该模块从 MongoDB 获取训练会话，MongoDB 在第一次启动时是空的。集装箱在完成培训课程后停止。

让我们添加培训课程参数。

为此，您登录到 MongoDB 容器(本地 shell 中的所有内容):

你打开 mongo 客户端。然后使用 use 命令选择 DB“training”。然后，您可以添加一个只有 5%图像的训练会话，一个批量大小为 50 和 20 个时期的 rmsprop 优化器。如果一切顺利，这是一个快速测试。

![](img/f9a054d117992ba7ec26c3e09846974c.png)

您离开 MongoDB 并重启容器:

现在的问题是你看不到发生了什么。您可以使用 docker log 命令获取 docker 容器的日志。

```
docker logs -f neural-network-training-with-docker
```

现在，您可以通过这种方式在本地机器上跟踪远程 docker 容器上的培训会话。

# 6.模型评估

您可以使用 MongoDB 快速比较不同培训课程的结果，因为我在测试集上保存了所有参数和准确性度量。数据库的优点是可以对其执行查询，这比将结果保存在 CSV 或 JSON 中要好得多。

让我们列出精确度最高的三个模型。

![](img/5f79bbae865a921333d907d13d67b691.png)

您还可以在数据库中查询特定培训课程的模型文件。您可以看到，您拥有模型架构和权重的 hdf5 文件。还有一个包含培训历史的 JSON 文件，您可以使用它来分析培训本身。它可用于可视化培训过程。

您可以从 MongoDB 自动加载最佳模型，并将其发布到 Flask、Spring Boot 或 Tensorflow 应用程序中。

![](img/4e9406b891cbf982b0a9a16bdcfa0ebc.png)

您可以使用 mongofiles 命令将文件下载到本地文件系统:

![](img/60ebe79ca0af57b8055fdf0d5ae7f250.png)

# 结论

在本案例研究中，您使用 docker-machine 从命令行在 AWS 上设置了一个支持 GPU 的云实例。目标是用额外的计算能力更快地训练神经网络。该实例可重复用于其他培训容器。

在下一步中，您实现了一个脚本，其中包含了在具有迁移学习的 InceptionV3 模型之上训练一个浅层全连接神经网络所需的所有步骤。该脚本使用 MongoDB 实例作为持久层来存储训练元数据和模型文件。您创建了一个 Dockerfile，其中包含在 GPU 云实例上使用 Tensorflow 训练网络所需的基础设施。然后，您在 docker-compose 文件中定义了一个多容器设置，包括 training 容器和 MongoDB 容器。

你在 Docker 的 AWS 云上训练了神经网络。您从本地命令行开始培训。您从本地命令行登录到 MongoDB 来添加培训会话，并在之后了解培训会话。

改进流程的后续步骤:

*   浅层神经网络的结构是硬编码的。最好也从持久层加载它。
*   使用除 InceptionV3 之外的其他基本模型的一般方法
*   您使用了准确性度量来测试模型的质量。最好是找到一种更通用的方法来持久化更多的度量。
*   您使用了带有默认参数的优化器。一个改进是一般地加载优化器特定的参数。
*   目标可能是从容器中移除 python 脚本，并在 docker-compose 文件中的环境变量的帮助下，将它作为 Python 模块从入口点脚本的存储库中加载。

*您需要关于数据科学设置的建议吗？*

请让我知道。给我发一封 [*邮件*](mailto:jenslaufer@jenslaufer.com)

*原载于 2019 年 4 月 23 日*[*【https://jenslaufer.com】*](https://jenslaufer.com/data/science/practical-example-of-deep-learning-in-docker.html)*。*