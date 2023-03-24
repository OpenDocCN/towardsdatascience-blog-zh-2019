# 我如何在 AWS 上构建一个(微小的)实时远程信息处理应用程序

> 原文：<https://towardsdatascience.com/how-i-built-a-tiny-real-time-telematics-application-on-aws-a9956f5afe65?source=collection_archive---------12----------------------->

*这个故事最初发表在我的博客*[*@ chollinger.com*](https://chollinger.com/blog/2019/08/how-i-built-a-tiny-real-time-telematics-application-on-aws/)

![](img/1e4f116b1f1075cd9904f459775ee5e3.png)

在 2017 年，[我写了关于](https://chollinger.com/blog/2017/03/tiny-telematics-with-spark-and-zeppelin/)如何构建一个基本的、开源的、Hadoop 驱动的远程信息处理应用程序(使用 Spark、Hive、HDFS 和 Zeppelin ),它可以在驾驶时跟踪你的运动，向你显示你的驾驶技术如何，或者你超速的频率——所有这些都不需要依赖第三方供应商代表你处理和使用这些数据。

这一次，我们将通过使用 Amazon 的 AWS、一个实际的 GPS 接收器和一台 GNU/Linux 机器，重新修补这个应用程序，并将其转换成一个更现代的“无服务器”实时应用程序。

*【1】见结论*

# 2019 年的微型远程信息处理

我最近写了一篇关于 [Hadoop 与公共云的文章。](https://chollinger.com/blog/2019/07/a-look-at-apache-hadoop-in-2019/)结论的一部分是，公共云提供商(如 AWS、GCP 和 Azure)可以以牺牲对技术堆栈的自主权为代价提供巨大的好处。

在本文中，我们将看到这一新发展的好处如何帮助我们重新设计一个 2 年之久的项目。

![](img/85806c1c498c34ed250c701e9e2f9ff0.png)

# 体系结构

# 目标

我们的目标很简单:

1.  应该捕获和收集在汽车中进行的每一次旅行；我们应该能够看到我们去了哪里，我们什么时候去的，我们走了什么路线，我们走得有多快
2.  可视化应该显示我们的路线和速度
3.  简单的问题，比如“我今天的最高速度是多少？”应该有可能
4.  运行成本应该合理

# 指导原则

我们的指导原则应该如下:

1.  应该实时接收和处理数据；如果您正在移动并且正在收集数据，那么输出应该最迟在几分钟内可用；如果您没有互联网连接，数据应该被缓存并稍后发送[1]
2.  我们不想为基础设施和服务器管理而烦恼；一切都应该在完全受管理的环境中运行(“无服务器”)
3.  架构和代码应该简单明了；我们希望它能在几个小时内准备好

# 超出范围

最后但同样重要的是，我们还忽略了一些事情:

1.  使用的设备将是笔记本电脑；类似的设置也适用于 Android、RaspberryPI 或任何 SOC 设备，只要它有 Linux 内核
2.  互联网连接将通过手机热点提供；将不使用单独的 SIM 卡来提供本地连接
3.  电力传输将通过 12V 或电池完成
4.  某些“企业”组件— LDAP 集成、VPC、长规则集、ACL 等。—超出范围；我们假设这些都存在于企业云中
5.  认证将被简化；不使用 Oauth/SSAO 流——我们将使用设备的 MAC ID 作为唯一 ID(即使它实际上不是)
6.  我们坚持查询 S3 的数据；更具可扩展性的解决方案，如 DynamoDB，将不在范围之内

# 架构图

这就引出了以下 AWS 架构:

![](img/d68008cec760a16b061bcc7c3a4aec6b.png)

通过这些步骤-

1.  移动客户端通过使用 **gpsd** Linux 守护进程实时收集数据
2.  AWS **物联网 Greengrass 核心**库通过直接在设备上运行 Lambda 函数*来模拟本地 AWS 环境。**物联网 Greengrass** 为我们管理部署、认证、网络和各种其他事情，这使得我们的数据收集代码非常简单。一个本地函数将处理数据*
3.  **Kinesis Firehose** 将获取数据，使用 **Lambda** 运行一些基本的转换和验证，并将其存储到 AWS S3
4.  Amazon**Athena**+**quick sight**将用于分析和可视化数据。QuickSight 的主要原因是它能够可视化地理空间数据，而不需要外部工具或数据库，如 nomist

*[1]更快的处理很容易通过更多的钱来实现，例如通过 Kinesis 轮询间隔(见下文)——因此，我们定义了一个“接近实时”的目标，因为有人——我——必须为此付费😉*

# 步骤 1:获取实时 GPS 数据

在最初的文章中，我们使用了 [SensorLog](https://apps.apple.com/us/app/sensorlog/id388014573) ，一个很棒的 iOS 小应用程序，来获取 iPhone 的所有传感器数据，并将其存储到 CSV 文件中，然后在**批处理**加载场景中进行处理。这是一个简单的解决方案，但是投资 15 美元，你就可以得到一个真正的 GPS 接收器，它几乎可以在任何 GNU/Linux 设备上运行，比如 RaspberryPI 或笔记本电脑。

# 设置 gpsd

所以，这一次我们将依赖于 [**gpsd**](https://gpsd.gitlab.io/gpsd/gpsd.html) ，这是 Linux 内核的 GPS 接收器接口守护程序。使用这个和一个便宜的 GPS 加密狗，我们可以直接从 USB TTY 获得实时 GPS 数据。我们还将能够使用 Python 来解析这些数据。

我们将使用这个 GPS 接收器: [Diymall Vk-172](https://smile.amazon.com/gp/product/B00NWEEWW8/ref=ppx_yo_dt_b_asin_title_o00_s00?ie=UTF8&psc=1) ，对于硬件，我使用我的 System76 Gazelle 笔记本电脑，运行 Pop！_OS 19.04 x86_64 和 5 . 0 . 0–21 内核。还有其他选择。

![](img/05c9bdd8b05ce3595fd249630b55d690.png)

这种设置很简单:

本质上，我们配置 gpsd 守护进程从正确的 TTY 中读取并在屏幕上显示它。上面的脚本只是一个指南—您的 TTY 界面可能会有所不同。

用测试这个

```
christian @ pop-os ➜ ~ gpsmon
```

你应该看到数据进来了。

只需确保您靠近窗户或室外，以便获得连接，否则您可能会看到超时:

![](img/beb359dff2d70a42c4027aabc883f0be.png)

```
TypeDescriptionDBUS_TYPE_DOUBLETime (seconds since Unix epoch)DBUS_TYPE_INT32modeDBUS_TYPE_DOUBLETime uncertainty (seconds).DBUS_TYPE_DOUBLELatitude in degrees.DBUS_TYPE_DOUBLELongitude in degrees.DBUS_TYPE_DOUBLEHorizontal uncertainty in meters, 95% confidence.DBUS_TYPE_DOUBLEAltitude in meters.DBUS_TYPE_DOUBLEAltitude uncertainty in meters, 95% confidence.DBUS_TYPE_DOUBLECourse in degrees from true north.DBUS_TYPE_DOUBLECourse uncertainty in meters, 95% confidence.DBUS_TYPE_DOUBLESpeed, meters per second.DBUS_TYPE_DOUBLESpeed uncertainty in meters per second,  95% confidence.DBUS_TYPE_DOUBLEClimb, meters per second.DBUS_TYPE_DOUBLEClimb uncertainty in meters per second,  95% confidence.DBUS_TYPE_STRINGDevice name
```

为了简单起见，我们将重点关注纬度、经度、高度、速度和时间。

# 第二步:AWS 物联网核心和绿草

AWS 物联网核心将为你的设备部署一个**λ**功能。这个函数将在本地运行，收集 GPS 数据，并通过 MQTT 将其发送回 AWS。它还将在互联网连接不可用的情况下处理缓存。

# 局部λ函数

首先，我们必须编写一个函数来实现这一点:

该功能使用 gps 和 greengrass 模块以预定义的批次收集数据。

在这一点上，我们还定义了我们的缺省值，在常见的情况下，某个属性——如纬度、经度或速度——不能被读取。稍后我们将使用一些 ETL/过滤器。

# AWS 物联网集团

接下来，创建一个 **AWS 物联网核心组**(详情请参见 [AWS 文档](https://docs.aws.amazon.com/greengrass/latest/developerguide/gg-config.html))。

![](img/f6b662cbfd426641f563b0bd43c48cb6.png)![](img/bdfd1f1b5c8961ca93a2450867d25c8a.png)

创建组后，下载证书和密钥文件，并确保获得适合您的特定架构的正确客户端数据:

![](img/92a3f2265dcdf9865f289214cb73c09a.png)

# 部署绿草客户端

然后，我们可以将 **Greengrass 客户端**部署到我们的设备上。默认配置假设有一个专用的根文件夹，但是我们将在用户的主目录中运行它:

如果您将它部署到一个专用的设备上(守护进程将持续运行，例如在 Raspberry Pi 上)，我建议坚持使用缺省的/greengrass。

# 将该功能部署到 AWS

接下来，我们需要将我们的 **Lambda** **函数**部署到 AWS。由于我们使用自定义 pip 依赖项，请参见 **deploy_venv.sh** 脚本，该脚本使用 Python 虚拟环境来打包依赖项:

在 AWS 控制台上，您现在可以上传代码:

![](img/7c1e9379c9f5bac01e267b6e4d7442c1.png)

创建一个**别名**和一个**版本**是很重要的，因为稍后在配置物联网管道时会参考到这一点:

![](img/122301b8d97e4b65ac049ad07639e90d.png)

# 在 AWS 物联网核心上配置 Lambda

接下来，回到我们之前创建的 AWS IoT 核心组，添加一个 Lambda 函数。

![](img/1ff9f9079eee0d32cae49da94bd3ec9a.png)![](img/9ff8094ce909535881aae1d9b8ca3f5d.png)

请记住:由于我们将无法运行容器(因为我们需要通过 TTY 与 USB GPS 设备对话)，请确保配置正确:

![](img/b8c5db2230a7711a92f935455e57270d.png)

另一个值得一提的是自定义用户 ID。客户端运行在某个用户名下，我强烈建议为它设置一个服务帐户。

完成后，点击 deploy，Lambda 函数将被部署到您的客户端。

![](img/9111e9230a7f979be2b6060e7eda9c45.png)

# 在本地测试该功能

最后，在部署之后，确保用户正在运行容器并检查本地日志:

![](img/b525969b803e6edb3a635debccda1a51.png)![](img/1ccbd6a18243331874e1d831085ca1ea.png)

(这是在我的办公室运行的，因此仅显示纬度/经度为 0/0)

太好了！现在我们的 Lambda 函数在本地运行，并每秒钟将我们的位置发送给 AWS。干净利落。

# 第三步:Kinesis 消防软管和 ETL

接下来，我们将数据发送到 Kinesis Firehose，它将运行一个 Lambda 函数并将数据存储到 S3，以便我们以后可以查询它。

通过这样做(而不是直接触发 Lambda ),我们将数据打包到可管理的包中，这样我们就不必为每一条记录调用 Lambda 函数(并为此付费)。我们也不需要处理逻辑来组织 S3 桶上的键。

# 创建 ETL Lambda 函数

首先，我们将再次创建一个 Lambda 函数。这一次，这个功能将**在 AWS** 上运行，而不是在设备上。我们称之为**远程信息处理-etl** 。

该函数只是过滤无效记录(纬度/经度对为 0 的记录，这是我们前面定义的默认值),并将速度和高度的“nan”字符串更改为整数-999，我们将其定义为错误代码。

该函数的输出是 base64 编码数据以及一个“ **Ok** ”状态和原始的 **recordID** 。

我们还需要确保每行有一个**JSON****并且没有数组**，这是 json.dumps(data)的默认设置。这是 Athena 使用的 JSON Hive 解析器的一个限制。请原谅代码中的恶意代码。
自然，这里可以进行更复杂的处理。

完成后，将该功能部署到 AWS。

# 测试功能

完成后，我们可以用类似于下面的测试记录对此进行测试:

```
{ "invocationId": "85f5da9d-e841-4ea7-8503-434dbb7d1eeb", "deliveryStreamArn": "arn:aws:firehose:us-east-1:301732185910:deliverystream/telematics-target", "region": "us-east-1", "records": [ { "recordId": "49598251732893957663814002186639229698740907093727903746000000", "approximateArrivalTimestamp": 1564954575255, "data": "WyJ7XCJsYXRcIjogMC4wLCBcImxvbmdcIjogMC4wLCBcImFsdGl0dWRlXCI6IFwibmFuXCIsIFwidGltZXN0YW1wXCI6IFwiMjAxOS0wOC0wNFQyMTozNjoxMC4wMDBaXCIsIFwic3BlZWRcIjogXCJuYW5cIn0iLCAie1wibGF0XCI6IDAuMCwgXCJsb25nXCI6IDAuMCwgXCJhbHRpdHVkZVwiOiBcIm5hblwiLCBcInRpbWVzdGFtcFwiOiBcIjIwMTktMDgtMDRUMjE6MzY6MTAuMDAwWlwiLCBcInNwZWVkXCI6IFwibmFuXCJ9IiwgIntcImxhdFwiOiAwLjAsIFwibG9uZ1wiOiAwLjAsIFwiYWx0aXR1ZGVcIjogXCJuYW5cIiwgXCJ0aW1lc3RhbXBcIjogXCIyMDE5LTA4LTA0VDIxOjM2OjExLjAwMFpcIiwgXCJzcGVlZFwiOiBcIm5hblwifSIsICJ7XCJsYXRcIjogMC4wLCBcImxvbmdcIjogMC4wLCBcImFsdGl0dWRlXCI6IFwibmFuXCIsIFwidGltZXN0YW1wXCI6IFwiMjAxOS0wOC0wNFQyMTozNjoxMS4wMDBaXCIsIFwic3BlZWRcIjogXCJuYW5cIn0iLCAie1wibGF0XCI6IDAuMCwgXCJsb25nXCI6IDAuMCwgXCJhbHRpdHVkZVwiOiBcIm5hblwiLCBcInRpbWVzdGFtcFwiOiBcIjIwMTktMDgtMDRUMjE6MzY6MTIuMDAwWlwiLCBcInNwZWVkXCI6IFwibmFuXCJ9IiwgIntcImxhdFwiOiAwLjAsIFwibG9uZ1wiOiAwLjAsIFwiYWx0aXR1ZGVcIjogXCJuYW5cIiwgXCJ0aW1lc3RhbXBcIjogXCIyMDE5LTA4LTA0VDIxOjM2OjEyLjAwMFpcIiwgXCJzcGVlZFwiOiBcIm5hblwifSIsICJ7XCJsYXRcIjogMC4wLCBcImxvbmdcIjogMC4wLCBcImFsdGl0dWRlXCI6IFwibmFuXCIsIFwidGltZXN0YW1wXCI6IFwiMjAxOS0wOC0wNFQyMTozNjoxMy4wMDBaXCIsIFwic3BlZWRcIjogXCJuYW5cIn0iLCAie1wibGF0XCI6IDAuMCwgXCJsb25nXCI6IDAuMCwgXCJhbHRpdHVkZVwiOiBcIm5hblwiLCBcInRpbWVzdGFtcFwiOiBcIjIwMTktMDgtMDRUMjE6MzY6MTMuMDAwWlwiLCBcInNwZWVkXCI6IFwibmFuXCJ9IiwgIntcImxhdFwiOiAwLjAsIFwibG9uZ1wiOiAwLjAsIFwiYWx0aXR1ZGVcIjogXCJuYW5cIiwgXCJ0aW1lc3RhbXBcIjogXCIyMDE5LTA4LTA0VDIxOjM2OjE0LjAwMFpcIiwgXCJzcGVlZFwiOiBcIm5hblwifSIsICJ7XCJsYXRcIjogMC4wLCBcImxvbmdcIjogMC4wLCBcImFsdGl0dWRlXCI6IFwibmFuXCIsIFwidGltZXN0YW1wXCI6IFwiMjAxOS0wOC0wNFQyMTozNjoxNC4wMDBaXCIsIFwic3BlZWRcIjogXCJuYW5cIn0iLCAie1wibGF0XCI6IDAuMCwgXCJsb25nXCI6IDAuMCwgXCJhbHRpdHVkZVwiOiBcIm5hblwiLCBcInRpbWVzdGFtcFwiOiBcIjIwMTktMDgtMDRUMjE6MzY6MTUuMDAwWlwiLCBcInNwZWVkXCI6IFwibmFuXCJ9Il0=" }, ...
```

![](img/52e439475e80f519c272e377668585e2.png)

# AWS Kinesis 消防软管

一旦我们的函数开始工作，我们希望确保来自设备的所有传入数据都自动调用该函数，运行 ETL，并将数据存储到 AWS S3。

我们可以这样配置消防水带流:

![](img/56f22ba8dad236fbc20f2070b2541d4d.png)![](img/d1143fa4d2fc802c2e8c93fc189b85ef.png)

在第二步中，我们告诉流使用 telematics-etl Lambda 函数:

![](img/a79ab34dbca53d71a64cf0bc046b7ba6.png)

也是 S3 的目标。

![](img/876d9d98d6e9b41c809808217ed991af.png)

以下设置定义了将数据推送到 S3 的阈值和延迟；此时，可以应用调优来使管道运行得更快或更频繁。

# 连接物联网核心和 Kinesis 消防软管

为了让它自动触发，我们只需要一个物联网核心操作，将我们的队列数据发送到 Firehose:

![](img/774747f97d093bbfb63808b95c9494f8.png)

# 步骤 3.5:端到端测试

此时，建议通过简单地启动**greengrasd**服务并沿途检查输出来端到端地测试整个管道。

![](img/fe171fe76c0be15a455d95b4f35f1dff.png)

一旦服务启动，我们可以确保该功能正在运行:

![](img/b79c30709237f7c4645bf289877be824.png)

在[物联网控制台](https://us-east-1.console.aws.amazon.com/iot/home?region=us-east-1#/test)上，我们可以跟踪所有 MQTT 消息:

![](img/90032942b4247aef7295be5c84add60c.png)

一旦我们在这里看到数据，它们应该会出现在 Kinesis Firehose 中:

![](img/7f8399f3bebe64cdf05617888a97ba74.png)

接下来，检查 CloudWatch 日志中的 telematics-etl Lambda 函数，最后是 S3 的数据。

# 关于收集真实数据的一个注记

可以想象，使用笔记本电脑收集数据可能会很棘手——除非你碰巧是一名警察，大多数商用汽车(和交通法😉)不占路上用终端。

虽然依靠一个无头的盒子当然是可能的(并且对于日常使用来说更现实)，但是我建议至少用一个有屏幕的东西来运行一组数据收集，这样你就可以验证 GPS 数据的准确性。

![](img/25902ec0cb0cc1ee2635737bbefa091e.png)

# 步骤 4:分析和可视化数据

一旦我们收集了一些数据，我们就可以前往 AWS Athena，将一个 SQL 接口附加到我们在 S3 上的 JSON 文件。
**Athena** 使用的是 **Apache Hive** 方言，但确实提供了几个助手，让我们的生活更轻松。我们将首先创建一个数据库，并将一个表映射到我们的 S3 输出:

![](img/071995c91fd0f298b53f1aa95678a865.png)

我们现在可以查询数据:

![](img/fb087913d0719f4c230cf0cf6920a6e7.png)

并查看我们的行程输出。

您可能已经注意到，我们跳过了一个更复杂的、基于 SQL 的 ETL 步骤，它会自动对旅行进行分组，或者至少以一种有意义的方式组织数据。为了简化流程，我们跳过了这一步——但它肯定属于需要改进的“待办事项”列表。

# 示例查询

> *“我们应该能够看到我们去了哪里，我们什么时候去的，我们走了什么路线，我们走得有多快”*

正如我们的目标所指出的，我们想知道一些事情。例如，我们在 2019-08-05 旅行中的最高速度是多少？

简单—我们将速度(单位:米/秒)乘以 2.237，得到英里/小时，选择该速度的最大值，并按天分组:

这给了我们 58.7 英里每小时，这似乎是正确的洲际旅行。

![](img/749315b76d318a7f4e217b3c01ff3939.png)

# 肉眼观察

查询很好。但是视觉效果呢？

如概述中所强调的，我们使用 **QuickSight** 来可视化数据。QuickSight 是该用例的一个简单选择，因为它提供了开箱即用的地理空间可视化，其行为类似于 Tableau 和其他企业可视化工具包。请记住，在具有 d3.js 的 Elastic Beanstalk 上的自定义仪表板可以以更快的数据刷新速率提供相同的值—quick sight standard 需要手动刷新，而 QuickSight Enterprise 可以每小时自动刷新一次数据。

虽然这不符合“实时”的目的，但它有助于开箱即用的简单、基本的分析。在路上刷新数据会产生大约 1 分钟的延迟。

![](img/1364b42867c9070943439eacc3da6bad.png)

设置很简单——在 AWS 控制台上注册 QuickSight，添加 Athena 作为数据集，然后拖放您想要的字段。

![](img/49783335946ef8dab7a43f14900cf15e.png)![](img/3dbccbfbc7f90d86bc161b4e6edda94f.png)

编辑数据集时，可以为地理空间分析定义纬度和经度两个字段:

![](img/128b10cca91354063fb011b10eb85207.png)

只需将正确的字段拖动到一些分析中，我们就会得到一个漂亮的小地图，显示一次旅行:

![](img/2c3ca9846aece81f66ec08119b87a813.png)

很多时候，你甚至不需要 SQL。如果我们想按分钟显示我们的平均速度，我们可以使用带有自定义格式(HH:mm)的时间戳值，并将默认的 sum(mph)更改为 average(mph)来构建一个图表，如下所示:

![](img/3c765911784fe3c8cb895ed456694211.png)![](img/a623c4c4fa354fc352ed5f0908a3541c.png)

使用更多定制的 SQL 来做更好的事情也是微不足道的。例如，在数据集上查看“高速”场景可以这样进行:

![](img/5b7c143c162a4afbc8a28ce7bf0479df.png)

然后添加到数据集:

![](img/d5955bb14edb4d8f3b774d17e5669255.png)

突然间，你几乎可以看到亚特兰大东部那条路上的所有红绿灯。

请记住，QuickSight 是一个相当简单的工具，与其他“大型”BI 工具甚至是 Jupyter 笔记本电脑的功能无法相比。但是，从本文的精神来看，它易于使用和快速设置。

# 结论

与 2 年前的“微型远程信息处理”项目相比，这一管道更简单，以接近实时的方式运行，可以更容易地扩展，并且不需要基础设施设置。整个项目可以在几个小时内完成。

当然，我们已经跳过了几个步骤——例如，一个更深入的 ETL 模块，它可以准备一个更干净的数据集或一个更可扩展的长期存储架构，如 DynamoDB。

对“无服务器”体系结构的关注使我们能够快速启动并使用我们需要的资源—没有时间花费在体系结构管理上。

然而，闪光的不都是金子。虽然我们确实取得了快速的进展，手头也有了可行的解决方案(当然，带着笔记本到处跑可能只是一种“概念验证”状态😉)，我们放弃了很多组件的自主权。这不完全是“供应商锁定”——代码很容易移植，但不会在另一个系统或云提供商上运行。

**物联网核心 Greengrass** 处理客户端部署、证书、代码执行、容器化和消息队列。

Kinesis Firehose 接管了 Spark Streaming、Kafka 或 Flink 等成熟流媒体框架的角色；它通过 Lambda 处理代码执行、传输、缩放、ETL 资源，并沉入存储阶段。

**Athena** 在一点点上弥合了差距——通过依赖 Hive 方言和开源 SerDe 框架，表定义和 SQL 可以很容易地移植到本地 Hive 实例。

**Lambda** 可以用类似的术语来看待——它只是 Python 加上了一些额外的库。关掉这些并使用例如 Kafka 队列将是微不足道的。

因此，**结论**——这又一次是一个完全没有意义的项目，尽管很有趣。它展示了即使是 AWS 的一个小子集也可以是多么强大，它是多么(相对)容易设置，如何将现实世界的硬件与“云”结合使用，以及如何将旧的想法转化为更时髦的——我更喜欢这个词，而不是“现代”——基础设施和架构。

*所有开发都是在 PopOS 下完成的！19.04 在 2019 System76 Gazelle 笔记本电脑* *上使用 12 个英特尔 i7–9750h v cores @ 2.6 GHz 和 16GB RAM 的内核 5.0.0 上，完整源代码可在*[*GitHub*](https://github.com/otter-in-a-suit/TinyTelematics)*上获得。*

*原载于 2019 年 8 月 7 日 https://chollinger.com**T21*[。](https://chollinger.com/blog/2019/08/how-i-built-a-tiny-real-time-telematics-application-on-aws/)