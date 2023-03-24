# tensor flow on edge——用树莓派打造“智能”安全摄像头

> 原文：<https://towardsdatascience.com/tensorflow-on-edge-or-building-a-smart-security-camera-with-a-raspberry-pi-5bb2fc039b0f?source=collection_archive---------9----------------------->

## 构建一个“智能”、Raspberry Pi 支持的边缘计算摄像头设置，运行 Tensorflow 对象检测模型来检测入侵者

![](img/333c6975f980d6b35307eaa9eb54b657.png)

我的户外相机被光线、风、汽车或除了人以外的任何东西触发的时间太长了。过于谨慎的安全摄像头可能是一个特征，但也是一个令人讨厌的特征。

我需要一个解决这个问题的方法，但不要太过火。简单，优雅，但有效的东西。

乡亲们，来见见我亲切地称之为“ [**”的稻草人——Cam**](https://github.com/otter-in-a-suit/scarecrow)。一个 **Raspberry Pi** 供电的摄像头，它通过 **Tensorflow 对象检测**来检测人们，并在他们踏上我的门廊的第二秒钟用格外响亮刺耳的音乐(或警告声)来迎接他们。

通过 Tensorflow、Raspberry Pi、摄像头、扬声器和 Wifi 进行实时物体检测，确保夜晚宁静。有点吧。

# 我们在做什么？

更严重的是，使用实时视频流和机器学习进行对象检测是一个实际的真实世界用例。虽然大多数用例可能令人毛骨悚然，但我确实在检测物体或人方面找到了一些价值*，最初发表于 2019 年 12 月 9 日*[*https://chollinger.com*](https://chollinger.com/blog/?p=424&preview=true&_thumbnail_id=429)*。*在私人安全摄像头上。

如果家中的安全摄像头网络能够检测到真实、潜在的威胁——人类和松鼠——并做出相应的反应(例如，通过发送警报)，这将极大地提高这些设备的实用性，这些设备主要依赖于运动检测或连续视频记录——这两者要么非常容易出错，要么最多是被动反应(向你展示事后发生的事情)。

# 关于安全摄像头的话题

然而，大多数消费级视频监控系统非常糟糕、简单明了。它们要么需要昂贵的硬连线，要么依赖于每 10 分钟设置一次的 janky 红外和运动检测，因为汽车会驶过房屋，依赖于第三方公司的 grace 和软件更新，通常是订阅模式，通常无法通过 API 访问。

我的未命名的户外相机设置通常由风中飘扬的旧荣耀引发，而不是由人引发。

# 解决方案

这是我们要做的:

我们将使用带有[摄像头模块](https://www.raspberrypi.org/products/camera-module-v2/)的 [Raspberry Pi 4](https://www.raspberrypi.org/products/raspberry-pi-4-model-b/) 来检测视频。这可以持续运行，不需要依靠运动传感器来启动。

为了检测物体，我们将使用谷歌的[张量流物体检测 API](https://github.com/tensorflow/models/tree/master/research/object_detection) 。这个库使我们能够使用开箱即用的对象检测(稍后将详细介绍)，而不需要手动训练和调整模型，也不需要[云部署](https://chollinger.com/blog/2019/07/a-look-at-apache-hadoop-in-2019/)。

为了与摄像机对话，我们将依赖于 OpenCV。

现在，给你一个问题:我的旧 RasPi 运行的是 32 位版本的 Raspbian。Tensorflow 与 32 位操作系统不兼容(当然，可能会有[替代方案](https://www.balena.io/blog/balena-releases-first-fully-functional-64-bit-os-for-the-raspberry-pi-4/))。此外，虽然新的覆盆子是一个强大的小机器，但它远不能与现代计算机相比——特别是在 3 和更早的版本上。

为了减轻这种情况，我们将通过 Pi 上的网络将视频流式传输到一台更强大的机器上——一台[家庭服务器](https://chollinger.com/blog/2019/04/building-a-home-server/)、NAS、计算机、一台旧笔记本电脑——并在那里处理信息。

这是一个叫做**边缘计算**的概念。根据这个概念，我们本质上使用功能较弱、较小的机器来实现低延迟通信，方法是在物理上靠近边缘节点的机器上执行繁重的工作，在本例中，运行 Tensorflow 对象检测。通过这样做，我们避免了互联网上的往返，以及不得不为 AWS 或 GCP 上的云计算付费。

为了实现这一点，我们将使用 [VidGear](https://github.com/abhiTronix/vidgear) ，具体来说是 [NetGear API](https://github.com/abhiTronix/vidgear/wiki/NetGear#netgear-api) ，这是一个为通过网络传输视频而设计的 API，使用 ZeroMQ。只是要警惕一个 [bug，](https://github.com/abhiTronix/vidgear/issues/45)要求你使用开发分支。

一旦我们检测到流中有人，我们就可以使用 ZeroMQ 向树莓发送信号，播放一些非常大声、令人讨厌的**音频**来吓跑人们。

![](img/c63ece3a70bd9db342e5e87a3f97ea8b.png)

# 设置开发环境

虽然可能不是最有趣的部分，但首先，我们需要一个开发环境。为此，我在 Linux 上使用了 Python 3.7 的 Jupyter 笔记本。

我们将基于以下教程开展工作:[https://github . com/tensor flow/models/blob/master/research/object _ detection/object _ detection _ tutorial . ipynb](https://github.com/tensorflow/models/blob/master/research/object_detection/object_detection_tutorial.ipynb)

旁注:由于我使用的是 Tensorflow 2.0 测试版，我不得不将 tf.gfile 的 API 修改为~/中的 tf.io.gfile。local/lib/python 3.7/site-packages/object _ detection/utils/label _ map _ util . py

克隆笔记本，按照说明设置虚拟环境，并安装所有依赖项。

# 局部测试张量流

一旦完成，我们很可能想要用测试数据循环本地视频，而不必实际连接现场摄像机。

这里我们真正需要做的是改变“while True”循环，以绑定到 OpenCV 会话。VideoCapture 方便地接受整数(对于网络摄像头)或视频文件的路径。

![](img/58e452ac9b89632c143188d6aada4573.png)

Some parts of the image were manually blurred for privacy — also, those are invited visitors 😊

一旦运行笔记本，您应该会看到检测模型的覆盖图。太好了，模型起作用了，我们可以稍后进行微调。

# 网络，网络！

如前所述，我们不会依赖我们的 Rasberry Pi 来运行检测—因此我们将建立一些网络通信。我本来想自己写这篇文章(我目前正在学习围棋，所以这将是一个很好的用例)，但是如果有一个库，为什么还要这么麻烦呢？[1]

你需要什么:

下面是一篇关于我如何设置 Pi 的示例文章:

接下来，我们将不得不考虑 VidGear 的开发分支，因为在撰写本文时，主服务器上还没有某个 bugfix。

现在，我们可以在本地测试流式视频:

您应该会看到视频回放。您会注意到多处理库的一些用途——我们使用它来确保 while 循环在定义的超时后终止，以用于开发目的。

*[1]玩笑暂且不提:SQLite 的创始人 Richard Hipp 在 Changelog 播客上的* [*采访中给出了很多很好的理由来解释为什么这可能是个坏主意——但是为了简单起见，我们将坚持使用库*](https://changelog.com/podcast/201)

# 把 2 和 2 放在一起

接下来，让我们将它部署到两台独立的机器上——只要执行 Tensorflow 的服务器运行 64 位版本的 Linux(Linux 内核),我们就可以开始了。在家里，我有一个本地的 gitlab 和 jenkins 实例为我做这件事。但是实际上，任何部署选项都是可行的——scp 可以成为您的朋友。

我们需要通过指定 IP 地址和端口对代码做一些小的调整:

```
client = NetGear(address = '192.168.1.xxx, port = '5454', protocol = 'tcp', pattern = 0, receive_mode = True, logging = True) server = NetGear(address='192.168.1.xxx, port='5454', protocol='tcp',pattern=0, receive_mode=False, logging=True)
```

现在，我们可以在服务器上启动 sender.py，在客户机上启动 receiver.py，这样我们就可以开始了:网络视频。整洁！

# 积分张量流

张量流的集成在这一点上是微不足道的:因为我们已经建立了如何做以下事情:

*   使用 OpenCV 循环播放本地视频
*   使用 Tensorflow 对象检测在本地视频上运行实际模型
*   将视频从一台机器传输到另一台机器

最后一步可以简单地通过导入我们的 tensorflow_detector 模块来实现，将接收到的图像转换为一个 numpy 数组(因为这是 API 所期望的)，并调用“run _ inference _ for _ single _ image(2)”。很简单！

# 对着摄像机说话

现在，我们可以调整我们的服务器代码，实际使用 Tensorflow 来检测来自不同机器的图像。但是我们不想要固定的图像——我们需要一个实时流。

幸运的是，这非常简单，因为我们已经在 Pi 的设置过程中启用了相机模块:将相机的带状电缆连接到 Pi，启动它，然后运行:

raspistill -o ~/image.jpg

这个应该拍张照。

![](img/8d44c09493edbaf3417309cd110d6380.png)

为了用 Python 做同样的事情，OpenCV 的 API 同样简单:在 PI 端，通过将视频文件的来源改为摄像机的 ID(0，因为我们只有一个), OpenCV 将获取摄像机流并将其发送到我们的服务器。

一旦我们运行这个，我们应该会看到这样的内容:

![](img/d43ca35ae4e5e23fe30888ee5d3a129c.png)

# 触发摄像头上的音频

现在，我们能够将视频从树莓上传到服务器，并检测到一个人。现在缺少的环节是使用 GStreamer 的 [playsound](https://github.com/TaylorSMarks/playsound) 库来播放音频。其他替代方法是使用 **ossubprocess** 调用或**pygame**——可以设置一个配置选项来改变这种行为。

(您可以使用“alsamixer”在 Pi 上设置音量)。

为此，我们将再次使用 **ZMQ** ，在 Pi 上启动一个监听器线程，等待来自服务器的输入，设置预定义的枚举值。一旦客户端收到这些信息，它就会触发一个音频信号并播放声音。

所有这些都阻塞了各自的线程，也就是说，只要音频播放，听众就不会关心新消息，服务器也无法发送新消息。

为了避免重复同样的信息，我们可以使用一个简单的 sempahore 结构。

# 测试一切

现在一切都正常了，有趣的部分来了:设置它。

首先，让我们把摄像机放在某个地方:

![](img/eabc894c231d0b1ac80d072f8ba5bf95.png)

并启动服务器和客户端，看看当我们走进房间时会发生什么！

正如你所看到的，它需要一秒钟才能检测到我——部分原因是摄像机角度、我们选择的模型以及一些网络延迟。在现实生活中，一个人需要走进门，这不是一个问题。

> “警告，你是…”的信息来自那个小喇叭！

# 后续步骤

虽然这已经工作得相当好了——它可以传输视频，检测人，并发送信号发送音频——但这里肯定还有工作要做。

然而，为了这个项目的目的，我们忽略了几件事:我们正在使用的模型仅仅是一个[预先训练的](https://github.com/tensorflow/models/blob/master/research/object_detection/g3doc/detection_model_zoo.md)模型，它既没有针对我们的特定用例进行调整，也不一定为我们正在使用的机器提供最佳平衡。

此外，代码中的一些内容可能需要一些改进——网络延迟还不错，但可能需要一些工作，音频被连续快速触发，并且使用我们当前的 ZeroMQ 模式，我们只能支持一个客户端。

虽然使用纸板盒几乎是完美的定义，但定制的 3D 打印覆盆子盒和稍好的冷却可能是一个好主意。

# 结论

无论如何，我对这个小项目很满意，因为它说明了构建自己的安全摄像头、将其连接到网络、运行对象检测并触发外部因素(在本例中为音频，但也可能是电话通知或其他东西)是多么简单，而无需使用任何公共云产品，从而节省住宅互联网的延迟和成本。

*所有的开发都是在 PopOS 的领导下完成的！2019 System76 Gazelle 笔记本电脑上的 19.04 内核 5.5.1，12 个英特尔 i7–9750h v cores @ 2.6 GHz 和 16GB RAM*、*以及 Raspian 10 上的 Raspberry Pi 4 4GB。*

*完整源代码可在*[*GitHub*](https://github.com/otter-in-a-suit/scarecrow)*上获得。*

*原载于 2019 年 12 月 9 日*[*https://chollinger.com*](https://chollinger.com/blog/?p=424&preview=true&_thumbnail_id=429)T22。