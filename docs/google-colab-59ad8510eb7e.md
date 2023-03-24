# 在 Google 联合实验室上运行 Python 脚本的快速指南

> 原文：<https://towardsdatascience.com/google-colab-59ad8510eb7e?source=collection_archive---------8----------------------->

## 今天就开始用免费的 GPU 训练你的神经网络吧

![](img/31c6a6a462f963030c954639db6f0724.png)

Photo by [Christopher Gower](https://unsplash.com/@cgower?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

如果你正在寻找一种交互式的方式来运行你的 Python 脚本，比方说你想和几个朋友一起开始一个机器学习项目，不要再犹豫了——Google Colab 是最适合你的解决方案。你可以在线工作，把你的代码保存在本地的谷歌硬盘上，它允许你

*   使用免费的 GPU(和 TPU)运行您的脚本！)
*   利用预装的 Python 库和 Jupyter 笔记本特性
*   在任何你想去的地方工作，它就在云上
*   与同事共享代码和协作

简言之，

> Google Colab = Jupyter 笔记本+免费 GPU

与大多数(如果不是全部)替代方案相比，它的界面可以说是更干净的。我想出了一些代码片段，供你掌握 Google Colab。当您需要一些现成的代码来解决 Colab 上的常见问题时，我希望这成为一篇必不可少的文章。

## 目录

*   [基础知识](https://medium.com/p/59ad8510eb7e#a166)
*   [文件](https://medium.com/p/59ad8510eb7e#4424)
*   [库](https://medium.com/p/59ad8510eb7e#75b9)
*   [机器学习](https://medium.com/p/59ad8510eb7e#cfc5)
*   [备注](https://medium.com/p/59ad8510eb7e#2cb4)

*原载于我的博客*[*edenau . github . io*](https://edenau.github.io)*。*

# 基础

## 启用 GPU/TPU 加速

进入**‘运行时’>‘更改运行时类型’>‘硬件加速器’**，选择‘GPU’或‘TPU’。您可以通过以下方式检查 GPU 是否已启用

```
import tensorflow as tf
device_name = tf.test.gpu_device_name()
if device_name != '/device:GPU:0':
  raise SystemError('GPU device not found')
```

如果 GPU 未启用，将会引发错误。请注意，您最多只能连续运行 12 小时的会话，并且环境不会跨会话持续存在。

## 运行单元

***SHIFT + ENTER***

## 执行 Bash 命令

只需在前面加一个`!`，例如:

```
!ls '/content/gdrive/My Drive/Colab Notebooks/'
```

让我们检查他们正在使用的操作系统、处理器和 RAM 的信息:

```
!cat /proc/version
!cat /proc/cpuinfo
!cat /proc/meminfo
```

Linux，不出意外。

# 文件

## 访问 Google Drive 上的文件

使用以下脚本:

```
from google.colab import drive
drive.mount('/content/gdrive')
```

然后，您将被要求登录您的谷歌帐户，并复制一个授权码。 ***点击链接，复制代码，粘贴代码。***

```
Go to this URL in a browser: [https://accounts.google.com/signin/oauth/.](https://accounts.google.com/signin/oauth/identifier?client_id=947318989803-6bn6qk8qdgf4n4g3pfee6491hc0brc4i.apps.googleusercontent.com&as=kh_zwwnAjZxRR9c036hjvA&nosignup=1&approval_state=!ChRkUmNrRGNkbHRZc3YwMjVPZU9ydRIfQTNkNU9vUnBieTBSY0t0V0xtY192ZHJvOHJ0aWh4WQ%E2%88%99APNbktkAAAAAXEiHtWgCVch04PQ_PvopPgfrZKdEK4lI&oauthgdpr=1&oauthriskyscope=1&xsrfsig=ChkAeAh8T59BFeu89K-KwgpUkeRD7fzyWfJYEg5hcHByb3ZhbF9zdGF0ZRILZGVzdGluYXRpb24SBXNvYWN1Eg9vYXV0aHJpc2t5c2NvcGU&flowName=GeneralOAuthFlow)..
Enter your authorization code:
 ·········· 
Mounted at /content/gdrive
```

## 上传文件

你可以简单地将文件手动上传到你的 Google Drive，并使用上面的代码访问它们。或者，您可以使用以下代码:

```
from google.colab import files
uploaded = files.upload()
```

## 运行可执行文件

将可执行文件复制到`/usr/local/bin`，给自己执行的权限。

```
!cp /content/gdrive/My\ Drive/Colab\ Notebooks/<FILE> /usr/local/bin
!chmod 755 /usr/local/bin/<FILE>
```

# 图书馆

## 安装库

在 bash 命令中使用`pip`:

```
!pip install <PACKAGE_NAME>
```

或者`conda`:

```
!wget -c [https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh](https://repo.continuum.io/archive/Anaconda3-5.1.0-Linux-x86_64.sh)
!chmod +x Anaconda3-5.1.0-Linux-x86_64.sh
!bash ./Anaconda3-5.1.0-Linux-x86_64.sh -b -f -p /usr/local
!conda install -q -y --prefix /usr/local -c conda-forge <PACKAGE_NAME>import sys
sys.path.append('/usr/local/lib/python3.6/site-packages/')
```

# 机器学习

## 张量板

使用`ngrok`:

```
# Run Tensorboard in the background
LOGDIR = '/tmp/log'
get_ipython().system_raw(
    'tensorboard --logdir {} --host 0.0.0.0 --port 6006 &'
    .format(LOGDIR)
)# Use ngrok to tunnel traffic to localhost
! wget [https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip](https://bin.equinox.io/c/4VmDzA7iaHb/ngrok-stable-linux-amd64.zip)
! unzip ngrok-stable-linux-amd64.zip
get_ipython().system_raw('./ngrok http 6006 &')# Retrieve public url
! curl -s [http://localhost:4040/api/tunnels](http://localhost:4040/api/tunnels) | python3 -c \
    "import sys, json; print(json.load(sys.stdin)['tunnels'][0]['public_url'])"
```

你会得到一个类似这样的超链接:

```
[https://d5b842b9.ngrok.io](https://d5b842b9.ngrok.io/)
```

您可以通过链接访问您的 Tensorboard！

# 评论

我的许多项目都是使用 Google Colab 开发的。查看以下文章了解更多信息。

[](/5-python-features-i-wish-i-had-known-earlier-bc16e4a13bf4) [## 我希望我能早点知道的 5 个 Python 特性

### 超越 lambda、map 和 filter 的 Python 技巧

towardsdatascience.com](/5-python-features-i-wish-i-had-known-earlier-bc16e4a13bf4) [](/visualizing-bike-mobility-in-london-using-interactive-maps-for-absolute-beginners-3b9f55ccb59) [## 使用交互式地图和动画可视化伦敦的自行车移动性

### 探索 Python 中的数据可视化工具

towardsdatascience.com](/visualizing-bike-mobility-in-london-using-interactive-maps-for-absolute-beginners-3b9f55ccb59) [](/handling-netcdf-files-using-xarray-for-absolute-beginners-111a8ab4463f) [## 绝对初学者使用 XArray 处理 NetCDF 文件

### 探索 Python 中与气候相关的数据操作工具

towardsdatascience.com](/handling-netcdf-files-using-xarray-for-absolute-beginners-111a8ab4463f) 

感谢您的阅读！

*原载于我的博客*[*edenau . github . io*](https://edenau.github.io)*。*