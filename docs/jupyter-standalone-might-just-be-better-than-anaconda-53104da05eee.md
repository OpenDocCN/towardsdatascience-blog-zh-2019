# Jupyter Standalone 可能比 Anaconda 更好

> 原文：<https://towardsdatascience.com/jupyter-standalone-might-just-be-better-than-anaconda-53104da05eee?source=collection_archive---------5----------------------->

![](img/4d314e83cecf07568feec914d215c043.png)

如果您已经在数据科学领域呆了很长时间，那么您可能非常了解 Anaconda navigator 和 Jupyter notebook。当数据科学家需要在虚拟内核上逐个单元地执行时，它们都是很好的工具。但是有没有可能在不掉落 Jupyter 的情况下掉落 Anaconda Navigator？

# 康达的优势

Anaconda 的根源是环境虚拟化。当然，与独立的 Jupyter 相比，这可以看作是使用 Anaconda 的一个巨大优势。此外，Anaconda 附带了“conda”包管理器，它不像常规的 Python 包索引那样具有扩展性。

由于这些特性，对于一个没有经验的人来说，Anaconda Navigator 的启动学习曲线不像设置 docker 映像和虚拟环境来运行 Jupyter 那样激烈。然而，Conda 的一个显著缺点是缺乏常规包装索引。因此，只有通过一个相当具有挑战性的漏洞，才可能安装尚未发布到 Conda 的传统 Python 包。对于使用大量 API 和各种包的人来说，这当然是个问题。

考虑到这一点，Anaconda 对于数据科学家来说无疑是一个很好的工具，因为像 VSCode、Spark managers 等扩展应用程序都可以很容易地实现到 navigator 中，以便在 conda 终端中工作，所以很容易理解为什么这是 Windows 开发人员的普遍选择。有趣的是，我的很多 Windows 朋友都把他们的康达 REPL 当成终端来使用，所以它的价值对他们来说是显而易见的。

# Jupyter 通过 SH 的优势

但是随着 Anaconda 在 Windows 上的易用性，在我们已经有了一个包管理器和一个终端来推送命令的操作系统上将会出现什么情况呢？嗯，对我个人来说，我更喜欢 Jupyter SH，希望我可以证明我自己对这种方法的支持和反对。

尽管我们失去了 Conda 环境，但我们保留了从终端启动时使用的任何虚拟环境，所以如果我

```
$ -  source env/bin/activate
```

然后运行 Jupyter 笔记本，

```
$ (env) - jupyter notebook
```

我将保留我的虚拟 env pip 环境。显而易见，这是在 Jupyter 中管理依赖关系生态系统的好方法。

我喜欢使用 Jupyter 而不是 Anaconda 的另一个原因是文件夹选择，任何时候一个笔记本被深埋在你电脑的文件夹迷宫中，在 Jupyter 中打开它都是非常困难的。Jupyter 的文件界面从任何意义上来说都不是完美无缺的，尽管它确实完成了任务。使用带有 SH 的独立 Jupyter，我可以在我电脑的任何位置打开一个终端，输入“Jupyter Notebook”，就可以直接导航到我想要篡改的文件。

我也是一个 Docker 的忠实用户，在 Docker 上使用 Conda 可能是一件很麻烦的事情……然而，用 Docker 在普通 Jupyter SH 上建立一个完全虚拟的操作系统是非常容易的。

# 最后的想法

尽管这些观点肯定没有得到普遍认同，但我还是独立使用 Jupyter，因为我喜欢它的多功能性。我喜欢 Anaconda，除了不需要它之外，我并不特别愿意安装它。只要你不通过软件包管理器来安装 Jupyter(它会破坏一切),那么你如何安装 Jupyter 并不重要。)

考虑到这一点，我很好奇有多少人在独立版中使用 Jupyter，而不是在 Anaconda 中使用它，他们对这种体验有什么想法。它们都是非常棒的工具，并且说明了有时候有两种完全不同的方法来做完全相同的事情。