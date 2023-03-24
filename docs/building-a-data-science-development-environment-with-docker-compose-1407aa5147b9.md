# 用 Docker Compose 构建数据科学开发环境

> 原文：<https://towardsdatascience.com/building-a-data-science-development-environment-with-docker-compose-1407aa5147b9?source=collection_archive---------12----------------------->

## 学习 Docker、Docker Compose 和 Cookiecutter 用于项目管理

当你有了一个伟大的新想法时，你最不想做的事情就是把数据放在一边，腾出一个干净的工作空间。我们每个人都沉迷于在 IDE 中做“一件快速的事情”,而没有考虑到依赖管理和可再现性的最佳实践。如果我们对自己诚实，我们知道什么是“最佳实践”吗？

![](img/1c01103c4163401d5086f0e30da34b83.png)

[Classic.](https://xkcd.com/1987/)

## 不管教程怎么说，虚拟环境并不容易。

有时候，我花在正确设置开发环境上的时间比处理数据和模型的时间还要多。

Anaconda 经常引发问题，尤其是 R 和 NLP 库(但这值得单独发布)。我也用过[家酿](https://brew.sh)、康达、pip、 [venv](https://docs.python.org/3/library/venv.html) 、pyenv、 [virtualenv](https://virtualenv.pypa.io/en/latest/) 和 virtualenvwrapper、 [pyenv-virtualenv](https://github.com/pyenv/pyenv-virtualenv) 和 pyenv-virtualenvwrapper 等工具管理包和 Python 版本。(我会再写一篇文章，详细介绍现有的各种工具，以及如何将它们应用到实际的工作流程中。)

如果您希望它们正常工作，它们要求您密切关注操作系统如何与软件、用户权限、路径变量和 shell 配置文件进行交互。

尽管有一个原始的设置，我仍然遇到了包冲突、$PATH 问题和其他各种神秘的功能故障。令人欣慰的是，我学到了很多关于 UNIX 的知识。

## Docker 让您在一个隔离的环境中开发代码并与数据交互。

Docker 将项目的所有代码和依赖项打包在一个可执行的容器中，该容器可以保存和重用。一个[容器映像](https://www.codementor.io/jquacinella/docker-and-docker-compose-for-local-development-and-small-deployments-ph4p434gb)是一个独立的软件，它包含了重建原始环境所需的一切:代码、运行时、系统工具、系统库和设置。

你可以通过 Dockerhub 分享图片；如果你写好了 docker 文件，你的项目每次都会以同样的方式在任何机器上运行。它仍然不是万无一失的，但由于它是自包含的，不良交互的机会更少，如果你搞砸了，你可以杀死容器并重启它。

Docker Compose 让你可以运行多容器映像(就像一个数据库服务器、一个 IDE 和一个 web 应用程序一起工作)。虽然可能有更好地优化容器化本身的工具，但 Docker 可以轻松快速地扩展，可以并行化应用程序，并广泛用于开发//部署和[持续集成](https://www.youtube.com/watch?v=xSv_m3KhUO8)工作流。毫不奇怪，我见过的每一个工作中的数据科学家都认为 Docker 是不可或缺的。

为了学习如何使用 Docker 和 Docker Compose(并为自己做一些有用的东西)，我制作了一个 [cookiecutter](https://cookiecutter.readthedocs.io/en/latest/) ，用于在 Python 中为自然语言处理项目创建虚拟环境。(如果你愿意，你可以[自己提取 repo](https://github.com/lorarjohns/cookiecutter_compose)并使用它来创建 Python 脚本或者使用 pandas、numpy、seaborn 和各种 NLP 库的 Jupyter 笔记本。)本质上，它是一个可定制、可编辑的项目模板，在您的计算机上创建一个目录，为 python 包编写 dockerfile、docker-compose 文件和 requirements.txt，并创建一个 Makefile，让您可以快速构建新的 Docker 映像并与之交互。(注意:这仍是一项正在进行的工作。)

在自学 Docker 创建 Python 包的过程中，您肯定会学到一些东西。这里有一个例子:

## 教训一:不一定要相信教程。

总是仔细检查你遇到的任何教程或资源。几乎所有时间，它们都是过时的。在线发布的快速迭代和很少甚至没有信息验证意味着在你开始实现任何代码之前，你必须从头到尾阅读。

几乎所有的代码都不适合你。有些码头文件是不好的做法；其他的语法已经过时或者无法运行；注意那些构建在 Python 或 Ubuntu 过时版本上的应用程序的讽刺之处，它们实际上应该使用不同的构建版本。(而“[最新](https://www.freecodecamp.org/news/an-introduction-to-docker-tags-9b5395636c2a/)”标签的意思并不是你想的那样。)除非 docker-compose.yml 的语法精确，并且 docker 文件的组成是经过深思熟虑的，否则您无法成功地构建图像，或者获得您想要的结果。几乎可以肯定，你自己写比从要点或教程中复制粘贴要好。

仅仅因为它在互联网上并不意味着它是有效的。写文件。设置测试并测试您的代码。

## 第二课:你的项目需要的时间比你想象的要长。

获取高质量的信息需要时间。Docker 有很好的文档，但是它并不总是有很好的索引或者是最新的。(我在一个 GitHub 问题中的 docker-compose 文件中找到了一个 bug 的答案，其中一个 docker 开发人员回答说:“哎呀——是的，我们改变了语法，忘记更新文档了。”)

这些工具需要付出比简单的“Hello World”教程或博客文章更多的努力才能获得。一旦你越过了玩具的例子，想要*使用*你正在写的代码去做真正的工作，它很快就变得多面化了。为了我的项目，我必须非常熟悉计算机网络，Linux/Ubuntu，bash 脚本， [GNU make](https://www.gnu.org/software/make/) ，阅读和编辑其他人的源代码，git 和 GitHub，以及 cookiecutter 包。添加 Docker 和 Docker Compose、Python 以及所有您希望包含在您的环境中的映像(例如 Redis、Flask、Django、Nvidia ),您会看到许多相互关联的部分。可能会让人不知所措。

迷失在一个复杂的话题上是令人沮丧的。很难保持简单，但是不要[试图一次画出整个猫头鹰](https://knowyourmeme.com/photos/572078-how-to-draw-an-owl)。得到一些最小的工作。我让 cookiecutter 开始工作，然后是 Makefile，然后是 Python 的 Docker 映像，然后是 docker-compose 文件，然后是所有的东西。

它仍然远非完美。我有弱点要修正，有更多的测试要写，有更好的结构要实现…等等。尽管如此，我通过自己构建这个项目的胆量学到了很多，并且我将通过一点一点地改进它来学到更多。

## 第三课:你不能“只做数据科学”

任何在学术界以外找工作的人都知道这一点，但对于那些没有找到工作的人来说，值得注意的是:如果你想找一份“数据”工作，你的技能需要证明不仅仅是“纯”数据科学。职称与职业甚至职责并没有清晰的对应关系，而且市场发展太快，使其成为一个安全的赌注。附有“科学家”或“工程师”的“数据科学”或“机器学习”的职位描述通常不仅仅描述善于发现、获取、清理、可视化和分析数据的候选人；他们还希望有人了解前沿的机器学习和经典统计模型、分布式计算、NoSQL 数据库、算法、高等数学、软件开发和测试，以及大量的领域知识。

你可能宁愿建立随机的森林和网络搜集酷数据，但如果你是一名求职者，花一些时间学习一些不那么性感的行业工具是值得的。了解您的同事正在使用什么，并能够谈论其利弊、优势和缺陷(更不用说实施这些工具了)，会让您成为更好的合作者，从而提高工作效率，因为很少有数据专业人员会完全孤立地工作。

## 试一试，一步一个脚印。

[下载 Docker](https://docs.docker.com/docker-for-mac/install/) 。拉一个简单的图片，习惯 Docker 的基本工作流程。(如果你不介意它有多初级，你可以拉我的回购——有一个自述，而且会是一个教程，当然你不能相信，TBA 很快。)使用 [repo2docker](https://github.com/jupyter/repo2docker) 将你最喜欢的 GitHub repo 转换成一个容器，并运行它。拆开 Docker 文件，看看它是如何将基本代码转换成 Docker 映像的。

学习 Docker，让它为你工作。如果你想理解它，就去做一个你感兴趣的项目，不管是因为它的技术复杂性还是因为它的主题。对网络上的不良信息感到沮丧。构建、测试、中断、重复代码的时间比你认为需要的时间要长。然后，制作一个实用的应用程序——不像一个价值百万美元的软件，但基本上能完成工作。反思你所学的一切；相当满意。然后，在新容器的 ide 内的内置 seaborn 数据集上兜风之后，您可以考虑添加更多的功能并收紧您的测试。但也许不是今晚。