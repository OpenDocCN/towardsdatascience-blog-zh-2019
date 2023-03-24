# 在 Microsoft Azure 中创建用于流式物联网数据的无服务器解决方案—第一部分

> 原文：<https://towardsdatascience.com/creating-a-serverless-solution-for-streaming-iot-data-in-microsoft-azure-part-i-5056f2b23de0?source=collection_archive---------37----------------------->

![](img/ebbd23299a39df6d8e7941416b2175e1.png)

Photo by [Markus Spiske](https://unsplash.com/@markusspiske?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

毫无疑问:物联网设备的使用(以及它们产生的数据)正在疯狂增长。根据 Gartner 的一项研究，到 2020 年，将有 58 亿台物联网设备在线，比 2018 年的 39 亿台增长 46%。到 2025 年，物联网设备[产生的数据总量预计](https://www.statista.com/statistics/1017863/worldwide-iot-connected-devices-data-size/)将达到惊人的 79.4 吉字节，这几乎是 2018 年数据量的六倍。

奇妙的是，像微软 Azure 这样的平台使得接收、存储和分析这些数据变得非常容易。在本文中，我将向您介绍如何从物联网温度传感设备采集流式遥测数据，并将其存储在数据湖中，同时对其进行一些基本的异常检测。本文的所有代码都可以在[这个 GitHub 库](https://github.com/yardbirdsax/azure-iot-solution)中找到。

该解决方案的组件包括:

*   [Azure 物联网中心](https://azure.microsoft.com/en-us/services/iot-hub/)，用于接收物联网设备发送的数据，并将原始数据捕获到数据湖。
*   [Azure Stream Analytics](https://azure.microsoft.com/en-us/services/stream-analytics/) ，用于处理发送到物联网中心的数据并检测异常。
*   [Azure Event Hub](https://azure.microsoft.com/en-us/services/event-hubs/) ，用于在检测到异常时从流分析接收消息，以便进行下游处理。
*   [Azure Data Lake Storage](https://azure.microsoft.com/en-us/services/storage/data-lake-storage/) ，用于存储发送到物联网中心的原始数据，以供以后可能的分析。

# 先决条件

如果您想跟进，您必须运行位于链接的 GitHub 存储库的`terraform`文件夹中的 Terraform 部署。在存储库的自述文件中有这样做的说明。这是可选的，因为在整篇文章中我会有大量的屏幕截图，但无论如何都要挖掘并构建自己的实验室！

# 解决方案的详细组件

## Azure 物联网中心

Azure 物联网中心提供了一种简单的 PaaS 产品，允许物联网设备向消息传递端点发送数据。它们很容易扩展，您可以从每月零成本开始(免费层每天提供多达 8，000 条消息，非常适合概念验证)。当部署实时安装时，你应该总是从基本层或标准层开始，而不是免费层，因为根据[微软](https://docs.microsoft.com/en-us/azure/iot-hub/iot-hub-scaling)的说法，你不能将免费层集线器升级到其他层。在做出决定之前，请仔细查看提供的功能列表和消息吞吐量数字，当然，您可以在以后扩展解决方案。

为了允许设备连接到物联网集线器，您必须提供设备身份。这可以通过 Azure CLI 界面轻松完成，如下所示:

Commands to create a new IoT Device in IoT Hubs

这将显示在各种可用的客户端库中使用的连接字符串(其中有很多)。

![](img/af81d19d085019f797ce64262a4cb39f.png)

您还必须创建**共享访问策略**，以允许下游应用程序(如 Azure 流分析或 Azure 函数)访问消费设备消息。在本例中，Terraform 模板已经创建了一个名为“streamanalytics”的模板，它被授予了**服务连接**权限，允许它从 hub 读取消息。有关可能授予的所有权限的列表，请参见[本](https://docs.microsoft.com/en-us/azure/iot-hub/iot-hub-devguide-security#iot-hub-permissions)。

> ***提示:永远不要使用内置的*** `***iothubowner***` ***策略，因为那样会授予 hub 基本上无限制的权限。此外，为每个消费应用程序创建一个策略，以便您可以轻松地轮换访问键，而不必在大量地方替换它们。***

![](img/4bce9edd445ffc9b40e76dff762797a8.png)

在使用消息时，您希望为每个读取消息的应用程序配置不同的**使用者组**。这允许每一个独立操作，例如跟踪它们在消息流中的位置。在我们的例子中，部署创建了一个`streamanalytics`消费者组，因为它将读取消息。

![](img/ce5f3e1766352eca5ec4b49ac24451c3.png)

最后，还有**路由**的概念，它决定消息被路由到哪里！这允许你向多个端点发送消息，比如 Azure Event Hub 兼容的端点、服务总线队列和 Azure 存储。后者对于捕获原始数据并将其存储到数据湖中非常有用，这就是我们在这里配置它的原因。

![](img/ee7660d490c24326915f364345b51b5b.png)

配置存储端点时，您可以根据文件夹结构配置文件的创建方式。您还可以配置文件写入的频率，以及在将文件写入存储帐户之前必须通过的批处理量。

![](img/f04b811c632755f2cacdc58c2ef47df1.png)

> ***提示:为了获得数据的全功能以后使用，请确保您配置的存储帐户为***[***Azure Data Lake Storage gen 2***](https://docs.microsoft.com/en-us/azure/storage/blobs/data-lake-storage-namespace)***启用。此外，确保文件的路径和名称是可排序的；例如，您可以确保路径以年开始，然后是月，然后是日，等等。***

## Azure 数据湖存储

Azure 数据湖存储提供了一种健壮且经济高效的方式来在现有的 Azure Blob 存储平台上存储大量数据。它与众多分析平台完全兼容，如 Hadoop、Azure Databricks 和 Azure SQL 数据仓库。虽然在此概念验证中未启用，但您可以通过[将网络访问](https://docs.microsoft.com/en-us/azure/storage/common/storage-security-guide?toc=%2fazure%2fstorage%2fblobs%2ftoc.json#network-security)限制为仅授权的公共 IP 或 Azure VNET 范围，以及仅 Azure 可信服务(在这种情况下是必需的，因为我们使用了物联网中心)来轻松保护对存储帐户的访问。

## Azure 流分析

Azure Stream Analytics 使用简单易学的 SQL 语法，可以轻松处理流式物联网数据。在我们的例子中，我们使用内置的异常检测功能来检测意外的温度变化。

流分析作业基于[流单元](https://docs.microsoft.com/en-us/azure/stream-analytics/stream-analytics-streaming-unit-consumption) (SUs)进行扩展，流单元定义了分配给特定作业的计算资源(CPU 和内存)的数量。

流分析部署有三个主要部分:输入、输出和处理数据的查询。让我们更详细地看一下它们。

## 投入

输入定义了流分析从哪里获取数据进行处理。在我们的例子中，我们将输入定义为 Azure IoT Hub，它是作为我们部署的一部分创建的。

![](img/359dbc9db95eb30b94472a754561fe88.png)

因为该资源是通过自动化部署创建的，所以已经输入了用于连接到物联网中心的所有信息。如前所述，您应该始终专门为流分析连接创建特定的共享访问策略，而不是使用内置的高权限策略(或任何其他预先存在的策略)。您还希望确保创建一个特定的消费者群体。通过点击“测试”按钮，我们可以确认一切工作正常。

![](img/32e32652428de0bcb6cb5cf403c6b977.png)

Azure 流分析也支持从 Azure Blob 存储和 Azure 事件中心接受数据。有关不同类型的流输入的更多细节，请参见[这份微软文档](https://docs.microsoft.com/en-us/azure/stream-analytics/stream-analytics-define-inputs)。

还值得注意的是，流分析接受一种不同的输入，称为**参考数据**。这用于加载静态数据集，这些数据集可用于连接和丰富流数据。例如，您可以使用一个包含不同传感器设备的详细信息的文件，作为在流的输出中包含有用信息的手段。有关使用参考数据流的更多详细信息，请参见[本微软文档](https://docs.microsoft.com/en-us/azure/stream-analytics/stream-analytics-use-reference-data)。

## 输出

Azure Stream Analytics 支持其产生的数据的大量输出，包括 blob 存储、Azure SQL 数据库、事件中心和许多其他内容。有关可能输出的完整列表，请参见本文档。

在我们的例子中，我们为 Azure Event Hubs 配置了一个输出，它被用作检测到的异常的目的地。与输入一样，明智的做法是使用特定的授权策略来允许流分析作业连接到事件中心。

![](img/549ef27c7c2f14aff868b6382d3c2ed8.png)

请注意，有一个属性我们必须手动设置，因为 Terraform 提供者当前(截至 2019 年 11 月)不支持它，这就是**分区键列**。正确地设置这一点很重要，这样可以确保相关事件组总是在同一个分区中结束，这样就可以按顺序处理它们。在我们的例子中，一个合理的选择是`device`列，因为这样我们可以确保来自一个特定设备的所有事件都在同一个分区中结束。

## 询问

查询是定义流分析所做工作的核心。在这里，我们可以使用简单的 SQL 方言来聚合、转换和丰富通过流分析作业传输的数据。对语言和所有可能性的完整解释将会是一篇独立的文章，所以现在我们将简单地描述当前作业中的查询。

该查询由三部分组成。第一种使用内置的`[AnomalyDetection_ChangePoint](https://docs.microsoft.com/en-us/stream-analytics-query/anomalydetection-changepoint-azure-stream-analytics)`功能来检测温度数据中的异常，通过识别的设备进行分区，并限于最近 30 分钟的数据。我们还在数据中设置时间戳列，以便流分析知道如何对数据进行排序。第二部分从异常检测函数的输出中检索两个值，这两个值告诉我们数据是否被认为是异常的，如果是，算法对数据的意外程度如何。最后，我们查询一些数据被检测为异常的字段，并将结果输出到异常作业输出(之前定义为事件中心)。

这样一来，我们就可以回顾我们概念验证的最后一个组件，即 Azure Event Hubs。

# Azure 活动中心

[Azure Event Hubs](https://azure.microsoft.com/en-us/services/event-hubs/) 是一个可扩展的 PaaS 产品，为应用程序提供发布/订阅消息平台。在我们的例子中，我们使用它作为目的地来接收检测到的异常事件，以便由一个或多个下游应用程序进行处理。

活动中心由以下组件组成。

## 活动中心

事件中心是一个逻辑存储桶，或者用消息传递体系结构中常用的一个短语来说，是一个主题，相关的事件在其上发布。在事件中心内，您可以定义一些属性，如分区数量(提示，选择比您认为需要的更多的分区，因为如果不重新创建事件中心，它们将无法更改)、消费者组(类似于物联网中心，用于定义处理数据的应用程序，以确保同步应用程序在流中的位置)、捕获(便于将事件发送到存储帐户)和消息保留(消息应在中心保留多长时间；考虑您可能需要重新处理多长时间，最多 7 天)。

在我们的例子中，我们定义了一个名为`anomalies`的事件中心，它将从我们的流分析作业接收输出。

为什么要使用事件中心而不是更简单的队列产品，比如 Azure Queues？通过使用发布/订阅消息总线，而不是队列，我们考虑到了多个消息消费者的可能性；例如，我们可以使用一个消费者生成松弛警报，但是也可以将结果发布到仪表板的某种数据存储中。正如后面所讨论的，我想在这方面准备一些东西，但是我已经超过了我自己为这篇文章规定的时间限制，所以将推迟到后面的文章。

## 事件中心名称空间

事件中心名称空间是相关事件中心的逻辑集合，是我们为名称空间内的所有事件中心资源定义分配的处理能力的地方。与存储帐户一样，事件中心命名空间名称必须是全局唯一的。

在我们浏览完各种资源之后，让我们通过系统实际运行一些数据，看看结果！

# 实时数据流

首先，我们必须模拟一个物联网设备向 Azure 物联网中心发送数据。为此，我构建了一个 Docker 容器，您可以使用下面的命令运行它。您将需要我们在上一节中检索的 IoT Hub 连接字符串。请确保在单独的命令提示符下运行该命令，因为它会一直控制命令行，直到您退出容器。

接下来，我们需要启动流分析作业。这可以通过门户来完成。

![](img/cd32f80d9f12ab01e257fcf1616d1568.png)

您可以选择“现在”作为开始时间，这样作业将从当前日期和时间开始消耗数据。

从`Query`窗格中，您可以单击`Test Query Results`按钮来查看查询的示例输出。

![](img/8297b76e97cf49b59fc517c886a3db6c.png)

> ***注意:因为只显示异常情况，您可能实际上看不到任何结果。在我们有意引入异常数据的情况下，您可以在下一个操作后重复该步骤。***

现在，为了显示异常到达事件中心，我们将引入一些显著不同的数据，从而触发异常检测算法。

首先，退出容器进程，然后指定一些附加选项重新运行它，如下所示。

为了显示输出正在到达事件中心，我们可以浏览到事件中心名称空间资源的`Metrics`面板，并查看`Incoming Messages`度量。确保您选择 30 分钟作为您的时间窗口，以获得最佳视图，您可能需要点击几次刷新按钮。

![](img/28384f0e7485f3abc8543ca8eddb76a8.png)![](img/57d5d2fc01fdef22b4eb97d969c9ca88.png)![](img/03eb6eaa811a395e873fd25954d0f26d.png)

# 摘要和待办事项

在本文中，我们介绍了使用 Azure 平台从物联网设备接收和使用消息的参考设置，包括物联网中心、Azure 流分析和 Azure 事件中心。在接下来的一篇文章中，我将介绍如何使用 Azure 函数从这个解决方案中消费和生成 Slack 警报。敬请期待！