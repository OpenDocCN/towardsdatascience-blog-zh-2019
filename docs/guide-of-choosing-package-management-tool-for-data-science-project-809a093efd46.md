# 数据科学项目包管理工具选择指南

> 原文：<https://towardsdatascience.com/guide-of-choosing-package-management-tool-for-data-science-project-809a093efd46?source=collection_archive---------11----------------------->

## 从 pipenv，conda，anaconda project，docker 等选择合适的工具。

![](img/5e3e648d4260f49af70c765cd16c8d9a.png)

Photo by [Clem Onojeghuo](https://unsplash.com/@clemono2?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/choose?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

如果您曾经从事过数据科学项目，您一定问过这个问题。

***如何管理包、依赖、环境？***

谷歌了一段时间，应该会看到一些关键词，**康达，巨蟒，迷你康达，巨蟒-项目，Pipenv，木星，木星实验室，Docker …**

这个列表可以更长。这样的调查很无聊，可能会占用你一整天的时间。为了节省您的时间，我写这篇文章作为一个总结，帮助您为您的项目选择最好的工具。

# 工具的快速总结

Python 世界中有许多工具可以进行不同种类的包管理，但是没有一种工具是万能的。我们应该根据自己的需要选择合适的。

*   [pip](https://pypi.org/project/pip/) 可以管理 Python 包。pip 包是一个源包。安装后还得编译，经常因为系统 OS 原因编译失败。此外，pip 不会检查与已安装软件包的冲突。
*   [pyenv](https://github.com/pyenv/pyenv) 可以管理 Python 版本
*   [Virtualenv](https://virtualenv.pypa.io/en/latest/) 和 [venv](https://docs.python.org/3/library/venv.html) 可以创建不同的虚拟 Python 环境。您可以为特定项目选择不同的环境。
*   [pipenv](https://github.com/pypa/pipenv) 将 pip、pyenv 和 virtualenv 组合在一起。但是仍然不能解决 pip 的编译问题。
*   [康达](https://conda.io/en/latest/)是一个环境管理体系。它支持二进制包，这意味着我们不需要在安装后编译源代码。
*   [Anaconda](https://www.anaconda.com) 是由 conda 管理的 Python 科学计算发行版，包含 Conda、numpy、scipy、ipython notebook 等百个包，
*   Conda Forge 是另一个比 Anaconda 更通用的 Python 发行版。但是这里有个坑。康达锻炉和蟒蛇并不完全兼容。如果您有一个同时使用 Anaconda 和 Conda Forge 包的项目，这可能会导致冲突。
*   [Anaconda 项目](https://link.zhihu.com/?target=http%3A//anaconda-project.readthedocs.io/)可以在一个项目中创建多个虚拟环境，同时管理 conda 依赖和 pip 依赖，但是缺少 CUDA 等系统包和一些命令行工具。
*   像 [Apt](https://www.wikiwand.com/en/APT_(Package_Manager)) 这样的系统包管理工具可以安装系统包，但是受到操作系统发行版的限制。比如 Ubuntu 14.04 不能安装 CUDA 9。
*   [Docker](https://www.docker.com) 可以在容器中安装一个操作系统发行版，方便切换操作系统版本。但是容器和主机必须共享相同的操作系统内核，所以没有办法直接在 macOS 上运行 Linux 发行版的 docker 容器。
*   [nvidia-docker](https://github.com/NVIDIA/nvidia-docker) 可以将 GPU 设备文件和驱动挂载到 docker 容器，但只支持 Linux。

# 一些建议

1.  如果您临时编写了几行项目中没有的代码，请使用 Anaconda 环境。
2.  如果您需要创建多个数据挖掘或科学计算项目，那么使用 Anaconda 项目来隔离这些项目，而不是使用 pipenv。
3.  如果你需要使用 Python 为网站创建多个项目，那么用 pipenv 而不是 conda 来隔离这些项目。
4.  对于 Anaconda 项目管理的那些项目，如果需要安装纯 Python 库，先用 pip 包。如果是需要额外编译的库，先用 conda 包。
5.  如果需要隔离系统环境，使用 Docker 的 Linux 版本在容器中安装系统依赖项。
6.  conda 和 Linux 发行版都有二进制包，我们更喜欢使用 conda。因为 Linux 发行版的发布周期很慢，而且版本很旧。

> ***查看我的其他帖子*** [***中***](https://medium.com/@bramblexu) ***同*** [***一个分类查看***](https://bramblexu.com/posts/eb7bd472/) ***！
> GitHub:***[***bramble Xu***](https://github.com/BrambleXu) ***LinkedIn:***[***徐亮***](https://www.linkedin.com/in/xu-liang-99356891/) ***博客:***[***bramble Xu***](https://bramblexu.com)

# 参考

*   【pyenv，virtualenv，anaconda 有什么区别？
*   [https://docs . conda . io/projects/conda-build/en/latest/resources/variants . html](https://docs.conda.io/projects/conda-build/en/latest/resources/variants.html)
*   [用 Anaconda 项目包装你的回购协议](https://blog.daftcode.pl/how-to-wrap-your-repo-with-anaconda-project-c7ee2259ec42)