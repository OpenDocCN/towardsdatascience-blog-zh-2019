# Python 和 WSL

> 原文：<https://towardsdatascience.com/python-and-the-wsl-597fbe05659f?source=collection_archive---------6----------------------->

## 在 Windows 和 Linux 上获得更好的 Python 体验。

![](img/64399c3ffccac5cf051ef23683a0c4d0.png)

Photo by [Miti](https://unsplash.com/@gigantfotos?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

我最近写了一篇简短的介绍性文章，介绍优步的 H3 地理空间索引算法。为了说明 H3 背后的思想，我使用了公开可用的 Python API。和大多数 Python 代码库一样，这是以包的形式提供的，包是 Python 代码重用的通用语言。您可以使用专门的工具来安装和管理这些软件包。对于常规的 Python 发行版，这个工具是 PIP，通常工作得很好，尤其是如果你是使用 Windows 机器的*而不是*。

在过去 12 个月的大部分时间里，我一直在使用高端 MacBook Pro，安装和管理 PIP 软件包已经变得轻而易举。这种体验与我之前使用 Windows 的体验形成了鲜明对比。然后，失败率很高，特别是带有需要在包安装期间编译的 C 或 C++代码的包。对于这些软件包，几乎总是有正确的解决方案:[克里斯托夫·高尔克的网页](https://www.lfd.uci.edu/~gohlke/pythonlibs/)。在这里，您可以找到各种预打包的" *wheel* "文件，它们在安装过程中不需要使用编译器。别人已经替你做了家务。您下载“ *wheel* ”文件，并使用您忠实的 PIP 安装它。上面的樱桃是您可以选择的选项，从 Python 版本、处理器类型，甚至编译选项。无处不在的 NumPy 包就是这种情况，它有普通版本和 MKL 版本。我从未能在我的 Mac 电脑上安装支持 MKL 的 NumPy，所以 Windows 用户在这里能得到更好的服务。问题在于像 H3 这样的特色套餐。除非他们出名，否则你很可能在 Christoph 的页面上找不到他们。如果你在那里看不到它们，很可能你在任何地方也找不到它们。

这就是我的经历，在发表了 H3 的文章后，我决定在我的侏罗纪 Windows 10 盒子上试试。我按照指示使用 PIP 安装了 H3 软件包，但是问题马上就出现了。在我的特殊例子中，CMake 工具的明显缺失使得*在安装了*之后仍然存在。我知道我在做什么，所以我立即放弃了。解决这些问题通常需要安装一个模糊版本的 Microsoft C++编译器，确保它出现在 PATH 变量中，等等。我钦佩那些有技能和耐心配置他们的 Windows 机器来干净地编译这些包的人。不幸的是，我不属于那个群体。

当我正在考虑放弃这个的时候，我记得读过一些关于在 Windows 上运行 [Tensorflow](https://www.tensorflow.org/) 的文章。这篇博客帖子来自斯科特·汉瑟曼，标题是“ [*在 Windows*](https://www.hanselman.com/blog/PlayingWithTensorFlowOnWindows.aspx) 上玩张量流。”这段文字可以追溯到 2016 年，当时 Tensorflow 甚至无法在 Windows PC 上编译。作者是如何设法绕过它的？他用了 WSL。

# 输入 WSL

WSL 代表“ [*Windows 子系统 for Linux*](https://en.wikipedia.org/wiki/Windows_Subsystem_for_Linux) *，*”，它允许您在 Windows 系统*中运行 Linux 系统，而无需*虚拟机。当斯科特写下他著名的博客文章时，它还处于测试阶段。现在 WSL 功能齐全，并且已经有了工具支持。在我们进入所有这些好东西之前，让我解释一下我是如何让 H3 代码运行的。

首先，您必须[启用 WSL](https://docs.microsoft.com/en-us/windows/wsl/install-win10) ，然后从微软商店安装一个 Linux 发行版。我选择了 Ubuntu 18，但是你可以在读到这篇文章的时候选择任何可用的版本。请理解，你很可能会得到一个操作系统的裸机版本，所以你应该准备安装很多辅助工具，比如 PIP 本身。Ubuntu 18 已经安装了 Python 3.6.7，所以这似乎是正确的选择。

使用命令行工具从 GitHub 下载代码轻而易举。完成后，我试图创建一个虚拟环境。不幸的是，VENV 包不可用，所以我必须安装它。Ubuntu 帮助了我，告诉我应该发出什么命令，我又回到了我的路上。除了意外缺少“轮子”包之外，设置虚拟环境太容易了。在等待老 CPU 咀嚼安装程序列表后，我准备好了。我启动了 Jupyter Lab 命令，得到了一个 URL 作为回答。将 URL 复制到 Windows 浏览器，你瞧！我的浏览器上有一个 Jupyter 笔记本！

WSL 1 与 Windows 共享相同的基础结构。它通过在 Windows 之上创建一个 Linux 翻译层来发挥其魔力。Linux 应用程序使用操作系统的资源，就好像它们运行在真实的 Linux 机器上一样，不管是真实的还是虚拟的。IP 地址是相同的，这就是为什么您可以在 WSL 上启动 web 服务器，并在 Windows 上使用它，而不需要任何网络攻击。

# 刀架

到目前为止，我很想说你正在体验两个世界的精华。但是有一个问题。你能用 WSL 使用你最喜欢的 IDE 吗？我寻求这个问题的答案，并感到惊喜。

Visual Studio Code 最近[发布了一个插件](https://code.visualstudio.com/docs/remote/wsl)，它允许你在 WSL 盒子上编辑、调试和运行 Python 代码。用户界面仍然在 Windows 上，但是 Python 代码和解释器在 WSL 上。在撰写本文时，该插件仍处于测试阶段，但可以像宣传的那样工作。

[PyCharm](https://www.jetbrains.com/help/pycharm/using-wsl-as-a-remote-interpreter.html) 也支持基于 WSL 的开发，尽管只是在付费版本上。

不幸的是，没有 CUDA 支持，尽管有计划在未来的版本中加入。对于深度学习项目来说，这将是一个很好的补充。

# 结论

这是治疗 Windows 疾病的灵丹妙药吗？虽然肯定不是一个完整的解决方案，但微软正朝着正确的方向前进。WSL 的某些方面必须改进，比如 WSL 不运行时对文件系统的远程访问，以及对深度学习人群的 CUDA 支持。无论如何，我想说这为在 Windows 平台上获得更好的数据科学体验打开了一扇门。这应该值得你去做。

# 笔记

[1][WSL 的第二版](https://devblogs.microsoft.com/commandline/announcing-wsl-2/)将使用一个精简的轻量级虚拟机。

对 WSL 如何工作的描述是我对版本 1 的理解。如果你想更详细地了解这个漂亮软件的工作原理，请参考更权威的资料。

[](https://www.linkedin.com/in/joao-paulo-figueira/) [## joo Paulo Figueira-数据科学家- tb.lx by 戴姆勒卡车和公共汽车| LinkedIn

### 查看 joo Paulo Figueira 在全球最大的职业社区 LinkedIn 上的个人资料。圣保罗列出了 1 份工作…

www.linkedin.com](https://www.linkedin.com/in/joao-paulo-figueira/)