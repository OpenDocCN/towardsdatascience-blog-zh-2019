# 在 PiZero W 中安装 OpenCV

> 原文：<https://towardsdatascience.com/installing-opencv-in-pizero-w-8e46bd42a3d3?source=collection_archive---------4----------------------->

我们大多数人在 Pi 中安装 openCV 时都遇到过问题。在这篇博客中，我将一步一步地告诉你如何在你的 pi-zero W 中安装 opencv。

虽然 Pi Zero 对于高级视频处理来说不够快，但它仍然是一个很好的工具，可以用来学习计算机视觉和 OpenCV 的基础知识。我建议你晚上继续关注这个博客，因为仅仅编译 OpenCV 就需要 12 个多小时。

让我们开始…

# **要求:**

1> pi 零 W 硬件

2> SD 卡(最低 16 GB)

3 >任何安装在你的 pi zero 上的 Raspbian 操作系统(我用过 Raspbian Buster)

假设您有给定的需求，让我们开始设置。

# **第一步:文件系统扩展:**

很少再需要扩展文件系统，因为 NOOBs 和 Raspbian 的独立安装都会在第一次启动时自动扩展文件系统。但是，为了确保系统已扩展，请运行以下命令:

```
pi@raspberrypi:~ $ raspi-config --expand-rootfs
```

# **第二步:增加交换空间**

**交换空间**是硬盘驱动器(HDD)的一部分，用于虚拟内存。有一个**交换文件**允许你的计算机操作系统假装你有比实际更多的内存。这将增加 openCV 的编译过程。否则你会以内存耗尽的错误结束。

要增加交换大小，请使用以下命令打开 pi zero 的交换文件:

```
pi@raspberrypi:~ $ sudo nano /etc/dphys-swapfile
```

转到交换大小，并将其从 100 更改为 2048。如下图所示。

```
.
.
.
# where we want the swapfile to be, this is the default
#CONF_SWAPFILE=/var/swap# set size to absolute value, leaving empty (default) then uses computed value
#   you most likely don't want this, unless you have an special disk situation
#CONF_SWAPSIZE=100
CONF_SWAPSIZE=2048
.
.
.
```

重启你的系统。

```
pi@raspberrypi:~ $ sudo reboot
```

# **步骤 3:安装依赖关系:**

首先让我们更新和升级现有的软件包:

```
pi@raspberrypi:~ $ sudo apt-get update
pi@raspberrypi:~ $ sudo apt-get upgrade
```

如果您使用 Raspbian Buster，请运行以下命令:

```
pi@raspberrypi:~ $sudo apt update
pi@raspberrypi:~ $sudo apt upgrade
```

出现提示时，输入“Y”。

安装开发人员工具:

```
pi@raspberrypi:~ $ sudo apt-get install build-essential cmake pkg-config
```

安装 IO 包:

```
pi@raspberrypi:~ $ sudo apt-get install libjpeg-dev libtiff5-dev libjasper-dev libpng12-dev
```

以及一些视频 I/O 包(尽管您不太可能使用 Raspberry Pi Zero 进行大量视频处理):

```
pi@raspberrypi:~ $ sudo apt-get install libavcodec-dev libavformat-dev libswscale-dev libv4l-devpi@raspberrypi:~ $ sudo apt-get install libxvidcore-dev libx264-dev
```

我们需要为 OpenCV 的 GUI 界面安装 GTK 开发库:

```
pi@raspberrypi:~ $ sudo apt-get install libgtk2.0-dev
```

OpenCV 利用的常规优化包:

```
pi@raspberrypi:~ $sudo apt-get install libatlas-base-dev gfortran
```

# 步骤 4:获取 OpenCV 源代码

所以现在我们有了所有的依赖关系。让我们从 opencv github 中拉出 OpenCV 的发布。

请注意，我已经发布了 3.4.1 版本。您也可以尝试其他版本。只需替换发布版本号。

```
pi@raspberrypi:~ $cd ~pi@raspberrypi:~ $ wget -O opencv.zip [https://github.com/Itseez/opencv/archive/3.4.1.zip](https://github.com/Itseez/opencv/archive/3.0.0.zip)pi@raspberrypi:~ $unzip opencv.zip
```

让我们一起抓取 opencv Contrib，因为 SIFT 和 SURF 已经从 opencv 的默认安装中移除:

```
pi@raspberrypi:~ $ wget -O opencv_contrib.zip [https://github.com/Itseez/opencv_contrib/archive/3.4.1.zip](https://github.com/Itseez/opencv_contrib/archive/3.0.0.zip)pi@raspberrypi:~ $ unzip opencv_contrib.zip
```

一旦两个存储库都被下载并在您的系统中展开，最好将它们删除以释放一些空间。

```
pi@raspberrypi:~ $ rm opencv.zip opencv_contrib.zip
```

# 步骤 5:设置 Python

如果您的系统中已经安装了 python2.7，您可以跳过这一步，否则请安装 Python 2.7 头文件，以便我们可以编译 OpenCV + Python 绑定:

```
pi@raspberrypi:~ $ sudo apt-get install python2.7-dev
```

之后安装 pip，一个 python 包管理器。

```
pi@raspberrypi:~ $ wget [https://bootstrap.pypa.io/get-pip.py](https://bootstrap.pypa.io/get-pip.py)pi@raspberrypi:~ $ sudo python get-pip.py
```

构建 Python + OpenCV 绑定的唯一要求是安装了 [NumPy](http://www.numpy.org/) ，所以使用 pip 安装 NumPy:

```
pi@raspberrypi:~ $ pip install numpy
```

# 步骤 6:为 Raspberry Pi Zero 编译并安装 OpenCV

现在我们准备编译和安装 OpenCV。

构建文件:

```
pi@raspberrypi:~ $ cd ~/opencv-3.4.1/pi@raspberrypi:~ $ mkdir buildpi@raspberrypi:~ $ cd buildpi@raspberrypi:~ $ cmake -D CMAKE_BUILD_TYPE=RELEASE \ -D CMAKE_INSTALL_PREFIX=/usr/local \ -D INSTALL_C_EXAMPLES=ON \ -D INSTALL_PYTHON_EXAMPLES=ON \ -D OPENCV_EXTRA_MODULES_PATH=~/opencv_contrib-3.0.0/modules \ -D BUILD_EXAMPLES=ON ..
```

现在是编译的时候了。请记住，自行编译可能需要 9 个多小时，有时您可能会认为系统已经冻结。但是不要失去耐心，让系统工作，直到你看不到任何错误。

```
pi@raspberrypi:~$ make
```

假设您的文件编译成功，没有任何错误，现在将使用以下命令安装 opencv。

```
pi@raspberrypi:~$ sudo make installpi@raspberrypi:~$ sudo ldconfig
```

# 步骤 7:完成安装

检查 cv2.so 文件的路径:

查找/-name“cv2 . so”

```
pi@raspberrypi:~$ /usr/local/python/cv2/python-2.7/cv2.so
```

现在您所要做的就是将 cv2.so 文件(绑定文件)符号链接到 python lib 的站点包中。

```
pi@raspberrypi:~$ cd /usr/local/lib/python2.7/site-packages
```

运行以下命令:ln-s[cv2 . so 文件的路径] cv2.so。在我的例子中，如下所示:

```
pi@raspberrypi:~$/usr/local/lib/python2.7/site-packages $ ln -s /usr/local/python/cv2/python-2.7/cv2.so cv2.so
```

# 步骤 8:验证您的 OpenCV 安装

现在是时候验证我们的 openCV 安装了。

启动 Python shell 并导入 OpenCV 绑定:

```
pi@raspberrypi:/usr/local/lib/python2.7/site-packages$ pythonimport cv2>>> cv2.__version__3.4.1’
```

现在你已经在你的系统上安装了一个新的 openCV。现在导出 python 可以从任何地方访问它的路径。

```
pi@raspberrypi:~$ export PYTHONPATH=/usr/local/lib/python2.7/site-packages:$PYTHONPATH
```

现在，您可以删除 opencv-3.4.1 和 opencv_contrib-3.4.1 目录，释放文件系统上的大量空间:

但是在运行这个命令之前要小心！在清空这些目录之前，确保 OpenCV 已经正确安装在你的系统上，否则你将不得不重新开始(漫长的，12 个多小时)编译*！*

如果你遇到任何错误，请在评论区告诉我。

如果您想要 python3 的 OpenCV，那么您可以简单地使用 sym-link 将绑定文件指向您的 python3 站点包并导出路径