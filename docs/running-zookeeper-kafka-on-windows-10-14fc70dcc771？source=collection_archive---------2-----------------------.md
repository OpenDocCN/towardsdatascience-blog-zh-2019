# 在 Windows 10 上运行 Apache Kafka

> 原文：<https://towardsdatascience.com/running-zookeeper-kafka-on-windows-10-14fc70dcc771?source=collection_archive---------2----------------------->

![](img/14f44feeaa03e4672d5668309b949451.png)

> 卡夫卡的成长呈爆炸式增长。超过三分之一的财富 500 强公司使用卡夫卡。这些公司包括顶级旅游公司、银行、十大保险公司中的八家、十大电信公司中的九家，等等。LinkedIn、微软和网飞每天用 Kafka 处理四个逗号的消息(10 亿条)。
> 
> **Kafka 用于实时数据流，收集大数据，或进行实时分析(或两者兼有)**。Kafka 与内存中的微服务一起使用，以提供耐用性，并可用于向 [CEP](https://dzone.com/articles/complex-event-processing-for-stream-analytics) (复杂事件流系统)和 IoT/IFTTT 风格的自动化系统提供事件。— DZone

网上有很多关于这个话题的文章，但是很多都是断章取义的，或者只是从不能在 Windows 上运行的文章中复制粘贴而来的。虽然这是一个简单的安装有一些问题。

这篇文章将帮助你远离陷阱，在 Windows 10 平台上培养 Kafka。

Kafka 依赖 Zookeeper 作为其分布式消息传递核心。所以 zookeeper 服务器需要先启动并运行，这样 Kafka 服务器才能连接到它。

在你下载 Zookeeper 和 Kafka 之前，确保你的系统已经安装了 7-zip。(处理 tar 和 gz 文件的伟大工具。)

在本文中，首先，我们将确保我们有一些必要的工具。然后我们将安装、配置和运行 Zookeeper。之后，我们将安装、配置和运行 Kafka。

所以，让我们开始吧。

# 安装工具

《动物园管理员》和《卡夫卡》是用 Java 写的，所以你需要 JDK。许多网站会告诉你只安装 JRE。不用麻烦了。只需安装 JDK。这是 JRE 附带的。然后设置 HOME 和 PATH 变量。但首先，7 比 0。

## 安装 7-zip

你可以在这里找到 7-zip—

[](https://www.7-zip.org/download.html) [## [计] 下载

### 英文中文更简单。中国传统。世界语法语德语印度尼西亚语日语葡萄牙语巴西西班牙语泰语…

www.7-zip.org](https://www.7-zip.org/download.html) 

当您安装 7-zip 时，请确保将其添加到您的右键菜单快捷方式中。这会让你的生活更轻松。

## 安装 JDK

它过去是免费的，现在仍然是免费的，只是现在你需要在 Oracle 创建一个帐户才能下载它。这是链接-

[](https://www.oracle.com/technetwork/java/javase/downloads/index.html) [## Java SE 下载

### Java SE 13.0.1 是 Java SE 平台的最新版本了解更多信息寻找 Oracle OpenJDK 版本？甲骨文…

www.oracle.com](https://www.oracle.com/technetwork/java/javase/downloads/index.html) 

JDK 将首先安装自己，然后它会问你要安装 JRE 的位置。

默认安装位置是“C:\Program Files\…”

通常把 Java 放在那里是没问题的，但是最好不要放在你的 Kafka 系统中。卡夫卡是为 Linux 设计的。我不会用包含空格的 Windows 目录名来推它。

我把我的 JDK 和 JRE 放在一个没有空格的路径中。我的安装位置是 C:\Java

## 设置路径

我们需要为所有的安装设置路径。因此，我们将在本文中多次讨论这个问题。我们将设置两种路径变量——用户变量和系统变量。在用户变量中，我们将添加安装位置的路径，在系统变量中，我们将设置 bin 目录的路径。

在 Windows 搜索栏中键入“Environment”。它会出现一个选项来编辑控制面板中的系统变量。点击它，你就会进入系统属性。点击右下角写着环境变量…的按钮。

## 用户变量

在顶部框中的用户变量下，单击 New 并添加 JAVA_HOME 和 JRE_HOME。对于 JAVA_HOME 的值，单击浏览目录按钮并导航到安装 JDK 的位置。我用的是 C:\Java\jdk1.8.0_221\

对于 JRE_HOME 的值，单击 Browse Directories 按钮并导航到安装 JRE 的位置。对我来说，那就是 C:\Java\jre1.8.0_221\

## 系统变量

在系统变量框中，双击路径并在末尾添加以下两个条目

%JAVA_HOME%\bin

%JRE_HOME%\bin

## 试验

如果设置正确，可以在任何目录下打开命令提示符。(一种快速的方法是使用 windows 文件资源管理器转到任何目录，然后在文件资源管理器的地址栏中键入“cmd”(不带引号)。它会在那个位置打开一个提示。)

导航到一个不同于 Java 安装的目录，这样您就可以真正知道您的路径变量是否在工作。

java 类型-版本

你应该看看这个-

```
C:\>java -version
java version “1.8.0_221”
Java(TM) SE Runtime Environment (build 1.8.0_221-b11)
Java HotSpot(TM) 64-Bit Server VM (build 25.221-b11, mixed mode)
```

如果您得到类似于 *java 不被识别为内部或外部命令，*那么您的路径设置不正确。为了确保这不是安装本身的问题，您可以转到 JDK 的 bin 目录并键入上面的命令。如果它在那里工作，但在其他地方不工作，这意味着系统找不到它。

对于在命令提示符下键入的每个命令，您的计算机都会在路径变量列表中查找匹配项。这就是为什么像这样的错误通常是路径问题。

# 安装动物园管理员

下载 Zookeeper 二进制文件。这里有一个可以下载的镜像

确保下载名称中包含 bin 的文件，而不是另一个文件。如果你下载了非 bin 文件，那么当你试图启动服务器时，你会得到一个错误。

```
[http://apache-mirror.8birdsvideo.com/zookeeper/stable/](http://apache-mirror.8birdsvideo.com/zookeeper/stable/)
```

右键单击该文件，并使用 7-zip 在同一位置提取它。这将提取 tar 文件，但它仍然不是真正的文件，所以您将该文件提取到哪里并不重要。

对于下一步，位置很重要:

右键单击 tar 文件，将其解压缩到名称中没有空格的目录。我把我的目录放在 C:\Apache\
中，所以我得到了一个类似这样的目录。所以

```
C:\Apache\apache-zookeeper-3.5.6-bin
```

截至本文撰写之时，zookeeper 的稳定版是 3.5.6。你的可能不一样。

请注意名称后面的-bin。如果你没有看到这个，那么你已经下载并提取了错误的文件。回到镜子前。这很重要，否则当你启动 Zookeeper 服务器时，你会得到一个类似这样的错误

```
Error: Could not find or load main class org.apache.zookeeper.server.quorum.QuorumPeerMain
```

## 配置 Zookeeper

所有的配置都发生在一个文件中——配置文件，它在 conf 目录中。

转到 zookeeper 安装的 conf 目录。对我来说，它在

```
C:\Apache\apache-zookeeper-3.5.6-bin\conf
```

将 zoo_sample.cfg 文件重命名为 zoo.cfg

用文本编辑器打开它。

在这个文件中，您将看到一个名为 dataDir 的条目，其值为/tmp。它基本上告诉你该做什么

```
# the directory where the snapshot is stored. 
# do not use /tmp for storage, /tmp here is just
# example sakes.
```

大多数网上的文章都告诉你用类似 dataDir=:\zookeeper-3.5.6\data 这样的东西来替换这一行

如果你这样做，你会遇到这样的错误-

```
ERROR— Unable to access datadir, exiting abnormally
Unable to create data directory :zookeeper-3.5.6data\version-2
Unable to access datadir, exiting abnormally
```

为了避免这个错误，请将您的日志指向比 bin 目录高一级的路径，如下所示

```
dataDir=../logs
```

(它可以是更高一级，也可以在同一目录中。您也可以键入从根目录 C:\\)开始的绝对 windows 路径

这将在你的 zookeeper 安装目录中创建一个日志目录，当你运行服务器时，它将存储快照。

配置完成。让我们设置路径变量，这样系统可以从任何地方找到它。

## 设置路径

像以前一样，开始在 Windows 搜索栏中键入 Environment。它会出现一个选项来编辑控制面板中的系统变量。点击它，你就会进入系统属性。点击右下角写着环境变量…的按钮。

## 用户变量

在顶框的用户变量下，点击 New 并添加 ZOOKEEPER_HOME。对于该值，单击浏览目录按钮并导航到 Zookeeper 的安装位置。对我来说，那就是 C:\ Apache \ Apache-zookeeper-3 . 5 . 6-bin

## 系统变量

在系统变量框中，双击路径并在末尾添加以下内容

%ZOOKEEPER_HOME%\bin

## 启动 Zookeeper 服务器

在 zookeeper bin 目录下打开命令提示符，输入

zkserver

它会开始发出一大堆信息。下面是一些有趣的例子。(为了视觉上的清晰，我已经清理了多余的词语)。它们看起来像这样

2019–11–19 11:17:17986[myid:]—信息-> *获取信息*

2019–11–19 11:17:17986[myid:]—WARN->*获取警告*

> myid 为空，因为我的 dataDir 中没有 myid 文件。Zookeeper 通过 id 跟踪集群中的每台机器。要给机器分配一个 id，只需输入一个包含一个数字的文件名 myid(没有任何扩展名)。我在单一服务器模式下运行 Zookeeper 进行开发，因此没有必要设置 id。但是，如果我创建一个编号为 5 的文件(可以是任意的，但是如果集群中有多台机器，则需要是唯一的)，那么命令行应该是这样的
> 
> 2019–11–19 12:05:21400**[myid:5]**—INFO[main:FileSnap @ 83]—读取快照..\ logs \版本 2\snapshot.a6

```
Server environment:os.name=Windows 10
Server environment:os.arch=amd64
Server environment:os.version=10.0
Server environment:user.name=Bivas Biswas
Server environment:user.home=C:\Users\Bivas Biswas
Server environment:user.dir=C:\Apache\apache-zookeeper-3.5.6-bin\bin
Server environment:os.memory.free=946MB
Server environment:os.memory.max=14491MB
Server environment:os.memory.total=977MB
minSessionTimeout set to 4000
maxSessionTimeout set to 40000
Created server with tickTime 2000 minSessionTimeout 4000 maxSessionTimeout 40000 datadir ..\logs\version-2 snapdir ..\logs\version-2
Logging initialized [@5029ms](http://twitter.com/5029ms) to org.eclipse.jetty.util.log.Slf4jL
```

最后一行中对 log4j 的引用是对 zookeeper 使用的日志基础设施的引用。您还会注意到，它正在将快照记录到我们之前在配置文件中指定的日志目录中。

```
Snapshotting: 0x0 to C:Apachezookeeper-3.5.6-binlogs\version-2\snapshot.0
```

经过几秒钟的数据喷涌，它应该来到这些黄金线

```
 Started AdminServer on address 0.0.0.0, port 8080 and command URL /commands
Using org.apache.zookeeper.server.NIOServerCnxnFactory as server connection factory
Configuring NIO connection handler with 10s sessionless connection timeout, 3 selector thread(s), 40 worker threads, and 64 kB direct buffers.
binding to port 0.0.0.0/0.0.0.0:2181
```

现在 Zookeeper 服务器运行在 localhost:2181 上。端口 8080 上的 AdminServer 是新增的。我们可以使用浏览器上的端口来监控 zookeeper。

但是，你不能去运行 zookeeper 的 2181 端口。Zookeeper 是 Kafka 用来作为 Kafka 集群的核心内核的。如果您在浏览器上导航到该端口，这将向它发送一些它不期望的 TCP 流量，您将使服务器崩溃。这是你将得到的—

```
 Exception causing close of session 0x0: Len error 1195725856
```

所以，就这样了。你的 Zookeeper 可以在 Windows 10 上运行，而不需要使用 docker composer 或 Linux VM。

# 下一个——卡夫卡

卡夫卡是一个信息经纪人。它可以让你创建你认为像聊天室的话题。您在该主题上发布一条消息，订阅该主题的人将会收到该消息。接受者被称为消费者。信息张贴者被称为生产者。

Kafka 还有另外两种能力。一个是流处理 API，它基本上接收这些消息，并在接收者收到之前将它们转换为不同的值。这在实时数据流中实时发生。

另一个是连接器 API，让 Kafka 连接到数据库或存储系统来存储数据。然后，这些数据可用于 Hadoop、Map Reduce 等集群的进一步处理。除了向消费者实时传递消息之外，这种情况也会发生。

如今，卡夫卡是一个一体化的解决方案。以前，你需要一个像 Apache Storm 这样的流处理框架来转换流，但是有了 Kafka 的原生流 API，我们就不像以前那样需要 Storm 了。这取决于您的使用案例和对您有意义的拓扑，但最好有选择。

## 安装卡夫卡

从这里下载卡夫卡-

```
http://kafka.apache.org/downloads.html
```

获取二进制下载。在这一部分，你可能会看到多个标有 Scala x.xxx 的版本。如果你使用 Scala 作为客户端，那么就获取与你的 Scala 版本相匹配的版本。我使用 NodeJS 作为我的客户端，所以我得到哪个并不重要。在撰写本文时，Apache 推荐[Kafka _ 2.12–2 . 3 . 1 . tgz](https://www.apache.org/dyn/closer.cgi?path=/kafka/2.3.1/kafka_2.12-2.3.1.tgz)所以这是我得到的版本。

使用 7-zip 将 tgz 解压缩到一个 tar 文件中。然后使用 7-zip 将 tar 文件解压到一个路径中没有空格的位置。我用 C:\Apache，所以解压后，我的卡夫卡住在这里——

```
C:\Apache\kafka_2.12–2.3.1
```

这篇文章越来越长，所以我打算把它分成两部分。接下来，我们将了解如何设置和配置 Kafka 服务器。

## 配置 Kafka

我们不会为 Kafka 设置任何环境变量。卡夫卡是寻找动物园管理员和 JDK 的人。甚至生产者和消费者都生活在卡夫卡生态系统中。它们不是在你的电脑上寻找卡夫卡的独立应用程序。简而言之，没有环境变量可以乱来。

但是，需要设置配置文件。

转到您的 Kafka 配置目录。对我来说是在

```
C:\Apache\kafka_2.12–2.3.1\config
```

我们可以从一个示例 server.properties 文件开始。

对于一个经纪人，我们只需要建立这一个文件。如果我们需要多个代理，那么为每个代理复制一次这个文件。例如，如果您需要两个消息代理，那么您最终会得到 server.b1.properties 和 server.b2.properties。

在每个文件中，您将更改以下内容—

*   经纪人 id

```
# The id of the broker. This must be set to a unique integer for each broker.broker.id=0
```

如果您只使用 1 个代理，则将其保留为 0。没什么好改变的。如果您有另一个代理，那么更改其他文件中的 id，使它们是唯一的。

```
# The id of the broker. This must be set to a unique integer for each broker.broker.id=1
```

*   更改日志目录。我坚持我的绝对路线。您可以在这里使用任何 Windows 风格的路径

```
# A comma separated list of directories under which to store log fileslog.dirs=C:\Apache\kafka_2.12–2.3.1\logs
```

*   复制因素就像硬盘上的 RAID。它将数据从一个代理复制到另一个代理，以实现冗余和容错。对于开发，我将把它保持在 1

浏览该文件中的字段。您会注意到超时值、分区值和默认 Zookeeper 端口号，如果出现问题，这些都将在以后的调试中派上用场。

默认情况下，Apache Kafka 将在端口 **9092** 上运行，Apache Zookeeper 将在端口 **2181 上运行。**

至此，我们完成了卡夫卡的配置。让我们启动服务器。

## 运行卡夫卡

确保 Zookeeper 服务器仍在运行。

导航到 Kafka 安装目录中的 bin 目录。在那里你会看到一个 windows 目录，进入那里。这是所有令人敬畏的 windows 实用程序存储的地方。对我来说，就是这里-

打开一个新的终端窗口

```
C:\Apache\kafka_2.12–2.3.1\bin\windows
```

(如果您忘记进入 windows 目录，而只是从 bin 目录启动，下面的命令将只是在 Visual Studio 代码中打开 shell 文件，而不是运行批处理文件)

```
kafka-server-start.bat C:\Apache\kafka_2.12–2.3.1\config\server.properties
```

你会看到这样的结果-

```
Client environment:java.compiler=<NA> (org.apache.zookeeper.ZooKeeper)
Windows 10 (org.apache.zookeeper.ZooKeeper)
Client environment:os.arch=amd64 (org.apache.zookeeper.ZooKeeper)
Client environment:os.version=10.0 (org.apache.zookeeper.ZooKeeper)
Client environment:user.name=Bivas Biswas 
Client environment:user.home=C:\Users\Bivas Biswas Client environment:user.dir=C:\Apache\kafka_2.12–2.3.1\bin\windows (org.apache.zookeeper.ZooKeeper)
Initiating client connection, connectString=localhost:2181 sessionTimeout=6000 watcher=kafka.zookeeper.ZooKeeperClient$ZooKeeperClientWatcher$@57c758ac (org.apache.zookeeper.ZooKeeper)
[ZooKeeperClient Kafka server] Waiting until connected. (kafka.zookeeper.ZooKeeperClient)
```

此时，如果服务器等待 Zookeeper 响应超时，请转到运行 Zookeeper 的命令终端，然后按 enter 键。有时候如果《动物园管理员》空闲一段时间，我见过卡夫卡超时。

如果一切顺利，您将看到来自组协调器的一些元数据转储和偏移消息，看起来像这样，带有闪烁的等待光标

```
[GroupCoordinator 0]: Preparing to rebalance group console-consumer-83701 in state PreparingRebalance with old generation 1 (__consumer_offsets-10) (reason: removing member consumer-1–9bf4ef2d-97af-4e59–964e-5bb57b457289 on heartbeat expiration) (kafka.coordinator.group.GroupCoordinator)[GroupCoordinator 0]: Group console-consumer-83701 with generation 2 is now empty (__consumer_offsets-10) (kafka.coordinator.group.GroupCoordinator)
```

你会在动物园管理员终端看到一些活动。它可能会拍摄新的快照并启动新的日志文件。此时，您的卡夫卡已经启动并运行。

# 包扎

动物园管理员需要记住的几件事——

我们设置为日志目录的 dataDir 将很快被快照填满。

在我的测试中，用一个主题运行 Kafka 不到 15 分钟，产生了两个 65MB 的快照文件。这些快照是事务性日志文件，每次检测到节点发生变化时都会写入这些文件。

当 Zookeeper 将几个日志文件合并成一个更大的文件时，它会创建副本，但不会清理旧文件。所以自己清理这个目录。您可以使用 bin 目录中的 zkTxnLogToolkit 来配置日志保留策略。

如果您在 AWS 上部署 EC2 实例，并且使用 t2.micro 的免费层，服务器不会启动。

这是因为 zookeeper 和 Kafka 的默认堆大小约为 1 GB，而 t2.micro 上的内存为 1GB，因此它会抱怨内存空间不足。

为了避免这个错误，请在具有 4GB 内存的 t3.medium 实例上运行 Kafka，而不是在配置文件中减少堆大小。

没有什么比跟随一篇文章更令人沮丧的了，这篇文章的步骤并不像它所说的那样有效。这就是我开始写这篇文章的原因，当时我浏览了大量的破碎的文章并修复了错误。

如果您在这些步骤中遇到任何错误，请告诉我。如果你有困难，请在下面留言，我会尽快回复你。

快乐流媒体！