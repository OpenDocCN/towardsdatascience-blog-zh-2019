# 如何在 Oracle Cloud 上用 Ubuntu 18 设置 GPU VM

> 原文：<https://towardsdatascience.com/how-to-set-up-a-gpu-vm-with-ubuntu-18-on-oracle-cloud-a484967f1bdf?source=collection_archive---------26----------------------->

# 介绍

如果你想开发一个深度学习模型，并希望获得成功的可能性很大，你需要两样东西:

*   大量的数据，用于模型的训练
*   GPU 形式的足够的计算能力

深度学习模型通常基于深度(多层)神经网络。通常用于模型训练的算法是基于某种形式的“[反向传播](https://en.wikipedia.org/wiki/Backpropagation)算法，它需要许多“张量”运算。对于这种操作，GPU 比 CPU 更有效，因为它可以以一定程度的并行性执行，即使使用现代 CPU 也无法实现。举例来说，英伟达 P100 拥有 3584 个内核，能够处理 5.3 万亿次浮点运算。

因此，与使用 CPU 所需的时间相比，如果您采用现代 GPU，您将能够将训练模型所需的时间减少至少一个数量级。

但是 GPU 还是很贵。另外，在这个领域，发展势头强劲，你的 GPU 可能会很快老化。

因此，选择**采用基于云的环境**，你只需为使用付费，这是现在最好的选择。仅举一个例子，在 Oracle Cloud 上，您可以使用一个带有一个 NVidia P100 (16 GB)、12 个 OCPU 和 72 GB Ram (VM)的 VM。GPU2.1 形状)约 30 $/天。如果需要，您可以使用更多的 GPU 来获得图形。

话虽如此，为 **TensorFlow** 正确设置环境并不是一件最简单的事情，你有可能无法充分利用你的 GPU。我在网上做了一些研究，发现文档并不完美。为此，我决定写这篇关于**如何为 TensorFlow 和 GPU** 设置 Ubuntu 18 环境的笔记。

# 环境。

如前所述，我正专注于在 Oracle 云基础架构(OCI)中设置虚拟机，我希望使用以下组件:

*   Ubuntu 18.04
*   Anaconda Python 发行版
*   张量流
*   Jupyter 笔记本

稍微复杂一点的是操作系统库、用于 GPU 的 Nvidia 驱动程序、CUDA 工具包版本和 TensorFlow 版本之间的正确对齐。如果所有这些都没有正确对齐，您可能会面临 TensorFlow 程序将正确运行的环境，但不使用 GPU，并且执行时间要长得多。不完全是你想要的。

我在这里记录了到目前为止我发现的最简单的一系列步骤，老实说，我是从个人角度开始写这篇笔记的。然后我决定，也许这是值得分享的。

# 虚拟机。

从 OCI 控制台，我为创建虚拟机选择的设置如下:

*   外形:VM。GPU2.1 (1 个 GPU，12 个 OCPU)
*   操作系统:Ubuntu 18.04
*   公共 IP:是
*   启动卷:100 GB 的磁盘空间

我还添加了一个公钥，以便能够使用 ssh 远程连接。

虚拟机的创建速度很快。很好。

# 虚拟机操作系统设置。

首先要做的是更新可用包的列表:

```
sudo apt update
```

那么，既然我们要使用 [**Jupyter 笔记本**](https://jupyter.org/) ，我们就需要在板载防火墙中打开端口 8888。推荐的方法如下:

```
sudo iptables -I INPUT -p tcp -s 0.0.0.0/0 --dport 8888 -j ACCEPT
sudo service netfilter-persistent save
```

不要使用 ufw。您可能会取消将虚拟机连接到存储所需的一些设置。

在此之后，我们需要添加一个入站网络安全规则，以便能够从浏览器连接到端口 8888:

```
Log into OCI console.Under **Core Infrastructure** go to **Networking** and then **Virtual Cloud Networks**.Select the right cloud network.Under **Resources** click on **Security Lists** and then the security list that you’re interested in.Under **Resources**, click **Ingress Rules** and then **Add Ingress Rule**.Enter 0.0.0.0/0 for the Source CIDR, TCP for the IP Protocol, and 8888 for the destination port.At the bottom of the page, click **Add Ingress Rules**.
```

此时，我们需要安装正确的 Nvidia 驱动程序。这是一个关键点，我已经浪费了一些时间。对于 Ubuntu 18，我发现的最简单的方法如下:

```
sudo apt install ubuntu-drivers-commonsudo ubuntu-drivers autoinstall
```

重新启动后(强制)，您可以使用以下命令检查驱动程序是否正确安装(输出被报告)。从命令中，您还可以获得驱动程序的版本。

```
**nvidia-smi**Mon Sep 30 13:34:03 2019+ — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — -+| NVIDIA-SMI 430.26 Driver Version: 430.26 CUDA Version: 10.2 || — — — — — — — — — — — — — — — -+ — — — — — — — — — — — + — — — — — — — — — — — +| GPU Name Persistence-M| Bus-Id Disp.A | Volatile Uncorr. ECC || Fan Temp Perf Pwr:Usage/Cap| Memory-Usage | GPU-Util Compute M. ||===============================+======================+======================|| 0 Tesla P100-SXM2… Off | 00000000:00:04.0 Off | 0 || N/A 37C P0 24W / 300W | 0MiB / 16280MiB | 0% Default |+ — — — — — — — — — — — — — — — -+ — — — — — — — — — — — + — — — — — — — — — — — ++ — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — -+| Processes: GPU Memory || GPU PID Type Process name Usage ||=============================================================================|| No running processes found |
```

“nvidia-smi”命令在未来也将有助于检查 GPU 在训练过程中的使用。

# 软件安装。

下一步是安装 Anaconda Python 发行版。

在撰写本文时，“七月版”是最新的版本，但是您应该查看 Anaconda 站点。

```
wget [https://repo.continuum.io/archive/Anaconda3-2019.07-Linux-x86_64.sh](https://repo.continuum.io/archive/Anaconda3-2019.07-Linux-x86_64.sh)bash Anaconda3–2019.07-Linux-x86_64.sh -becho -e ‘export PATH=”$HOME/anaconda3/bin:$PATH”’ >> $HOME/.bashrcsource ~/.bashrc
```

使用以下命令对 Anaconda 发行版进行最后一次更新是值得的:

```
conda update -n base -c defaults condaconda init bash
source ~/.bashrc
```

至此，我得到了 4.7 版本。最后两个命令允许使用“conda 激活”命令。

接下来，我们需要创建一个新的“conda”虚拟环境，我们称之为“gpu”

```
conda create --name gpu
conda activate gpu
```

之后，我们可以在创建的“GPU”env 中安装 Python:

```
conda install python
```

下一个重要步骤是 **TensorFlow** 的安装。这里要安装支持**GPU**的版本。重要的是**使用 conda 而不是 pip** 进行安装，因为这样可以确保正确安装和满足所有依赖关系。

```
conda install tensorflow-gpu
```

接下来，Jupyter 安装:

```
conda install nb_condajupyter notebook --generate-config
```

之后，您需要在 Jupyter 的配置中写入以下几行

```
In /home/ubuntu/.jupyter/jupyter_notebook_config.py file, add the following linesc = get_config()
c.NotebookApp.ip = '0.0.0.0'
c.NotebookApp.open_browser = False
c.NotebookApp.port = 8888
```

这样，您就可以从 VM 外部(任何 IP 地址)连接到 Jupyter。

然后，因为您只希望通过 SSL 进行连接，所以可以使用 OpenSSL 生成自签名证书:

```
openssl req -x509 -nodes -days 365 -newkey rsa:2048 -keyout jupyter-key.key -out jupyter-cert.pem
```

完成所有这些步骤后，您可以使用以下命令启动 Jupyter

```
jupyter notebook — certfile=jupyter-cert.pem — keyfile=jupyter-key.key
```

(最终，您可以添加 nohup)。

浏览器会给你一个安全警告，但你知道你在做什么，可以马上继续。

# 测试你的 GPU。

现在，终于到了检查一切是否正常的时候了。在开始训练复杂的深度学习模型之前，我们希望确定 TensorFlow 将使用 GPU。因此，我们使用 TensorFlow 进行测试。

在笔记本电池类型中:

```
import tensorflow as tffrom time import strftime, localtime, timewith tf.Session() as sess:devices = sess.list_devices()devices
```

在输出中，您应该在可用设备列表中看到 GPU。

然后作为最后的检查(我在 StackOverflow 上找到的)

```
import tensorflow as tfif tf.test.gpu_device_name():print(‘Default GPU Device: {}’.format(tf.test.gpu_device_name()))else:print(“Please install GPU version of TF”)
```

您希望看到第一个“print”语句的输出。

作为最终检查:

*   看看 Jupyter 生产的日志；
*   运行你的模型，同时执行“nvidia-smi”命令；您应该会看到大于 0 的“易变 GPU 利用率”

# 结论。

如果你想在深度学习领域认真实验，你需要 GPU。有几种方法可以用 GPU 建立 Linux 环境:在这篇文章中，我记录了如何在 Oracle OCI 上用最新的 LTS Ubuntu 版本建立一个环境。

此外，我还记录了如何确保 TensorFlow 实际使用 GPU。