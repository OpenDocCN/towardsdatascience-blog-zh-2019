# 如何通过 SSH 在深度学习平台上远程使用 Jupyter

> 原文：<https://towardsdatascience.com/how-to-use-jupyter-on-your-deep-learning-rig-remotely-with-ssh-cefd78fbfe2c?source=collection_archive---------26----------------------->

![](img/db094f2ec573904261067593a68a0c37.png)

当处理多台机器时，将你的文件和硬件分开可能是一件好事，也可能是一件可怕的坏事。我们都遇到过这样的情况，我们需要将工作转移到 USB 闪存驱动器上，以便在旅行时将它们放在笔记本电脑上。最重要的是，你的笔记本电脑也有臭名昭著的

> 集成显卡

没有我们每天都要访问的文件系统会对我们的性能产生负面影响。对于我们这些喜欢轻薄笔记本电脑、宁愿在沙发上而不是电脑椅上工作的人来说，对远程台式机的渴望可能会更强烈。

![](img/e5a6c0d770201c24929fb64cc7c17ff3.png)

我决定用 RedHat 来运行我的 SSH 服务器。RedHat 是非常成熟的，并且是最常运行的服务器发行版之一。

老实说，如果你是 Linux 新手，发行版并不重要，因为 Linux 在整体上是相当一致的。

我不打算回顾安装过程或用户创建过程，因为它们都很简单“你希望它是什么”，并键入答案。

现在我们已经安装了操作系统，我们需要做什么呢？对于我来说，我有两个 GPU，和一个不支持内核驱动程序的 Wifi 适配器，所以第一步是驱动程序。我们可以获得并安装带有驱动程序的闪存驱动器，并使用 build-essential 或 DKMS 来安装它们:

```
make
sudo make install
sudo modprobe __
```

或者，在将文件复制到/usr/src 之后

```
sudo dkms add -m ___ --version 1.0
sudo dkms build -m ___ --version 1.0
sudo dkms install -m ___ --version 1.0
modprobe ___
```

现在，我们的 wifi 驱动程序已经就绪，我们必须实际连接到我们的 wifi，第一步是打开我们的网络连接。

```
sudo ifconfig wlan0 up
```

然后扫描可用的 wifi SSIDs:

```
sudo iwlist scan | more
```

并添加我们的 WPA 密钥:

```
wpa_passphrase
```

然后我们将这些信息放到一个文本文件中。会议)

```
wpa_passphrase (SSID)> /etc/wpa_supplicant/wpa_supplicant.conf
```

然后我们将 CD 放入该目录，并连接！

```
sudo wpa_supplicant -B -D driver -i interface -c /etc/wpa_supplicant/wpa_supplicant.conf && sudo dhclient
```

恭喜，我们正式有 wifi 了！

现在我们可以做我们最喜欢的两件事，更新我们的包和库。需要注意的是，包管理器当然取决于你选择的发行版。对于 RedHat 可以是 dnf 也可以是 yum，Debian(或者 Ubuntu)会用 apt，Arch 会用 Pacman，openSuse 会用 man。所以如果你没有选择使用 RedHat，就用你各自的包管理器替换我的 dnf。

```
sudo dnf update && sudo dnf upgrade
```

在按下 y 键并输入至少一次之后，你现在必须得到你的新的最好的朋友:SSH。虽然现在有很多 SSH 工具，但是我们将会使用 SSH 的 FOSS 版本。

```
sudo dnf install openssh-server
```

现在让我们用主机名、端口等快速配置我们的网络。

```
$ nano ~/.ssh/config
```

![](img/2ee9011c02f6ed8c67fd8696c4e0b672.png)

现在，我们只需为我们的服务器启用 systemctl:

```
sudo systemctl enable ssh
sudo systemctl start ssh
```

如果你使用的是 Ubuntu，你需要运行以下命令为你的防火墙打开一个端口(端口 22 ):

```
sudo ufw allow ssh
sudo ufw enable
sudo ufw status
```

现在 SSH 应该正在运行，您的服务器应该已经启动并正在运行。如果你使用 CUDA 或 OpenCL，现在是 wget 或 dnf 安装一些图形驱动程序的好时机。

在将要连接到 SSH 服务器的客户机上，您还需要安装 openssh 客户机:

```
sudo dnf install openssh-client
```

这样一来，我们可以连接到我们的无头服务器:

```
ssh emmett@192.168.0.13
```

如果你不知道 IPv4 地址(最后带点的部分)，WiFi 和互联网的情况会有所不同:

> 无线局域网（wireless fidelity 的缩写）

```
ipconfig getifaddr en1
```

> 以太网

```
ipconfig getifaddr en0
```

祝贺您，您现在已经连接到了您的 ssh 服务器，您可以像使用任何其他终端一样使用它。如果你碰巧正在使用 Nautilus，你也可以把服务器添加到你的文件管理器中，方法是转到文件，然后在左边点击其他位置，在底部有一个包含 ssh、ftp 等的组合框。你可以利用的关系。

![](img/fcc041ce9b943a85ed033849cb0b90b3.png)

> 那我们怎么和 Jupyter 一起用呢？

当然，首先我们必须安装 Jupyter，所以回到你的服务器，安装 pip for python3:

```
sudo apt-get install python3-pip
```

然后用 pip 安装 Jupyter Lab:

```
sudo pip3 install jupyterlab
```

要启动笔记本服务器，只需运行

```
jupyter notebook
```

要访问服务器，只需转到您的客户机，打开您选择的 web 浏览器，并输入您的 SSH 构建的 IP。现在你可以拔掉除了机器电源线以外的所有东西。

# 下一步

下一步是扔掉你的鼠标和键盘，因为你已经正式

> 没头了