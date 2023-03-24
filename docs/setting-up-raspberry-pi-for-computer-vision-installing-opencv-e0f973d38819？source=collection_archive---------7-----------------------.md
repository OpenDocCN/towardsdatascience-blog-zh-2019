# 为计算机视觉设置 Raspberry Pi(安装 OpenCV)

> 原文：<https://towardsdatascience.com/setting-up-raspberry-pi-for-computer-vision-installing-opencv-e0f973d38819?source=collection_archive---------7----------------------->

![](img/852d993c2b1a9778d928a89c49c167a9.png)

Photo by [Harrison Broadbent](https://unsplash.com/@hbtography?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# **设置 Rasbian Noobs**

Noobs 是 Raspbian 的一个易于安装的版本。 [**Raspbian**](https://www.raspberrypi.org/downloads/raspbian/) 是基金会官方支持的操作系统。Raspbian 预装了大量用于教育、编程和一般用途的软件。它有 Python，Scratch，Sonic Pi，Java 等等。

注意:Rasbian 是一个基于 Linux 的操作系统。

注:这个过程可能看起来漫长而无聊，但这是一次性的事情。您也可以复制带引号的命令。

1.  从[https://www.raspberrypi.org/downloads/noobs/](https://www.raspberrypi.org/downloads/noobs/)下载 zip 文件
2.  确保解压缩文件夹，并将所有内容复制到 SD 卡(建议大于 4GB)。
3.  通过将一个名为“SSH”的文件(不带任何扩展名)放入 SD 卡的引导分区(引导文件夹内)，启用 [SSH](https://en.wikipedia.org/wiki/Secure_Shell) 。
4.  使用手机充电线(非 C 型)给 raspberry pi 通电，并通过以太网将其连接到笔记本电脑。

> 第一次，我建议使用外部显示器、键盘和鼠标进行初始设置。通过***ifconfig****记下 pi 的 IP，然后继续通过 wifi/以太网 essentials。*
> 
> 然而，如果你没有初始外设，请浏览此视频:[如何在没有显示器或键盘的情况下设置树莓派](https://www.youtube.com/watch?v=toWBmUsWD6M)

# **获取连接到以太网的 Raspberry Pi 的 IP(Ubuntu)**

1.  启动 Raspberry Pi，并将以太网连接到 Pi 和笔记本电脑。
2.  在你的笔记本上。转到编辑连接设置。
3.  导航到 ipv4 选项。选择方法:“共享给其他计算机”。
4.  连接到有线连接。然后打开命令提示符并键入命令

> nano/var/lib/misc/dnsmasq . leases "。

你将会从那得到覆盆子 pi Ip。

5.然后打开命令提示符并键入:ssh pi@

# **在 Pi 上设置 OpenCV 3**

> ***宋承宪不会？*** 如果您在网络上看到您的 Pi，但无法对其进行 ssh，您可能需要启用 SSH。这可以通过 Raspberry Pi 桌面首选项菜单(你需要一根 HDMI 线和一个键盘/鼠标)或从 Pi 命令行运行 **sudo 服务 ssh** start 来轻松完成。

第一步是将文件系统扩展到包括 micro-SD 卡上的所有可用空间:

> 步骤 1: sudo raspi-config
> 
> 步骤 2:接着选择“扩展文件系统”:
> 
> 第三步:一旦出现提示，你要选择第一个选项，***“A1。展开文件系统*** ， ***回车*** *然后* ***结束***
> 
> 步骤 4:到终端键入“sudo reboot”

(可选)第二种方法是删除所有不必要的文件以清理空间

> sudo apt-get 净化钨-发动机
> 
> sudo apt-get 清除图书馆*
> 
> sudo apt-get clean
> 
> sudo apt-get 自动删除

现在，在步骤 3 中，我们将安装所有需要的依赖项

> sudo apt-get 更新&& sudo apt-get 升级
> 
> sudo apt-get 安装内部版本-基本 cmake pkg-config
> 
> sudo apt-get install libjpeg-dev libtiff 5-dev lib jasper-dev libpng 12-dev
> 
> sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-dev
> 
> sudo apt-get 安装程序
> 
> sudo apt-get 安装 libgtk2.0-dev libgtk-3-dev
> 
> sudo apt-get install lib atlas-base-dev gfortran

为 python 3 和 python 2 安装 OpenCV

> 安装 python2.7-dev python3-dev

现在，对于第四步，我们将下载 opencv 源代码，然后为 pi 构建它。

> CD ~ & & wget-O opencv.zip[https://github.com/Itseez/opencv/archive/3.3.0.zip](https://github.com/Itseez/opencv/archive/3.3.0.zip)&&解压 opencv . zip

(可选)安装 opencv-contrib 库

> wget-O opencv_contrib.zip[https://github.com/Itseez/opencv_contrib/archive/3.3.0.zip](https://github.com/Itseez/opencv_contrib/archive/3.3.0.zip)&&解压 opencv _ contrib . zip

现在，在第 5 步中，我们将为 python 和

> wget[https://bootstrap.pypa.io/get-pip.py](https://bootstrap.pypa.io/get-pip.py)&&须藤 python get-pip.py & &须藤 python3 get-pip.py

大多数情况下，你通过这个博客在单板计算机上设置你的深度学习模块，如 Raspberry Pi。因此，我们将公开 opencv 安装，以便在所有虚拟环境中访问。如果您想使用 envwrapper 或类似工具构建自己的虚拟环境，请激活并继续。

安装 [**numpy**](https://www.numpy.org/)

> pip 安装数量

现在，我们将编译并安装 opencv。这个过程需要时间，取决于 pi 模型。在我的 B+ plus 模型中，大约花了 1.5 小时。所以去喝杯咖啡吧！！

> cd ~/opencv-3.3.0/ && mkdir 内部版本&& cd 内部版本
> 
> CMAKE-D CMAKE _ BUILD _ TYPE = RELEASE \-D CMAKE _ INSTALL _ PREFIX =/usr/local \-D INSTALL _ PYTHON _ EXAMPLES = ON \-D OPENCV _ EXTRA _ MODULES _ PATH = ~/OPENCV _ contrib-3 . 3 . 0/MODULES \-D BUILD _ EXAMPLES = ON..

*如果您使用的是 python3，请确保您的 python 解释器和库设置在 python 3 或 python 3.5 文件夹中(或 python2 的 python2.7 文件夹中)*

在开始编译过程之前，您应该**增加您的交换空间大小**。这使得 OpenCV 能够用树莓 PI 的所有四个内核进行**编译，而不会因为内存问题而导致编译挂起。键入终端**

> sudo nano /etc/dphys-swapfile

然后将 CONF _ 交换大小编辑为

> CONF _ 交换大小=1024

现在，我们将重新启动交换空间。

> sudo/etc/init . d/d phys-交换文件停止
> 
> sudo/etc/init . d/d phys-swap file start

现在，我们跳到安装 opencv 的最后阶段。我们现在将编译 opencv。

> 品牌-j4

现在，你需要做的就是在你的 Raspberry Pi 3 上安装 OpenCV 3

> sudo make 安装
> 
> sudo ldconfig

**Bug 修复:**老实说不知道为什么，也许是 CMake 脚本中的 Bug，但是在为 Python 3+编译 OpenCV 3 绑定时，输出。所以文件被命名为 cv2 . cpython-35m-arm-Linux-gnueabihf . so(或其变体),而不是简单的 cv2.so(就像在 Python 2.7 绑定中一样)。同样，我不确定为什么会发生这种情况，但这很容易解决。我们需要做的就是重命名文件:

对于 python3

> CD/usr/local/lib/python 3.5/site-packages/
> 
> sudo mv cv2 . cpython-35m-arm-Linux-gnueabihf . so cv2 . so
> 
> cd ~/。virtualenvs/cv/lib/python 3.5/site-packages/
> 
> ln-s/usr/local/lib/python 3.5/site-packages/cv2 . so cv2 . so

**通过打开 python bash 和**来测试安装

> 导入 cv2

现在，我们将通过删除安装文件来释放空间，并将 pi 恢复到它原来的交换空间。

> RM-RF opencv-3 . 3 . 0 opencv _ contrib-3 . 3 . 0
> 
> sudo nano /etc/dphys-swapfile

然后将 CONF _ 交换大小编辑为

> CONF _ 交换大小=1024

现在，我们将重新启动交换空间。

> sudo/etc/init . d/d phys-交换文件停止
> 
> sudo/etc/init . d/d phys-swap file start

# 恭喜你，你现在完成了。享受计算机视觉！！