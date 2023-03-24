# 一项经常被忽视的数据科学技能

> 原文：<https://towardsdatascience.com/an-often-overlooked-data-science-skill-ea4254e19c6b?source=collection_archive---------8----------------------->

![](img/5f7e22a7a25482b6ef08411e34f61442.png)

Source: Pexels

你刚刚开始了你作为数据科学家的第一份工作，并且你很兴奋开始使用你的随机森林技能来实际改变。您已经做好了启动 Jupyter 笔记本的所有准备，但却意识到您首先需要“SSH”到另一台机器来运行您的模型。该公司利用云计算来大规模执行机器学习。你肯定听说过 AWS、Google Cloud Compute 和 Microsoft Azure，但是几乎没有在远程机器上运行模型的经验。

这种情况可能会让一些人感到惊讶，但我一直都在看。学校项目倾向于关注可以在笔记本电脑上合理运行的问题，甚至来自较小数据集领域的经验丰富的数据科学家也可以在他或她的笔记本电脑上运行模型。

然而，随着模型变得越来越渴求数据，数据科学家在远程机器(云或本地)上工作时越来越有可能需要舒适和高效。以下是我如何开始发展这些技能。

![](img/d2c9653cddd87c73d89161a91da74876.png)

Source: Pexels

# 了解终端的基本知识

无论您使用的是 OSX、Linux 还是 Windows，您现在都可以访问基于 Linux/Unix 的终端(如果您使用的是 Windows，请参见[此处](https://itsfoss.com/install-bash-on-windows/))。在 Mac 上，按 command+空格键打开搜索栏，键入“终端”,然后按 enter。你现在应该感觉你是在矩阵里——一个真正的编码者。终端将允许你直接向计算机输入命令，而不需要任何图形支持。当使用远程机器时，这是非常有价值的。您将使用终端连接到机器，向它发送命令，并浏览您的文件。

有一本很棒的书叫做《命令行中的数据科学》,我强烈推荐阅读[入门](https://www.datascienceatthecommandline.com/chapter-2-getting-started.html)章节。仅这一章就能让你轻松地直接从终端操作你的机器。

如果你真的喜欢冒险，花点时间用 [zsh](http://zpalexander.com/switching-to-zsh/) 和 [iterm2](https://iterm2.com/) 修改你的终端。

# 发现 Vim

Vim 是一个文本编辑器，您可以直接从终端使用，它预装在大多数 Linux 系统上，或者非常容易安装。

Vim 有一个不错的学习曲线，但是即使只是学习基础知识，也可以让您快速地从终端对文件进行更改。当从远程机器上工作时，这可以节省您大量的时间。

要开始，请查看此[互动指南](https://scotch.io/tutorials/getting-started-with-vim-an-interactive-guide)。

你也可以学习[emacs](http://www.gnu.org/software/emacs/tour/)——如果你讨厌你的[小手指](https://hackernoon.com/escape-from-finger-strain-hell-ex-church-of-emacs-member-tells-all-d95425ad958f)。

# SSH 和 SCP

SSH 是一个可以从终端运行的命令，用于连接到远程机器。SCP 是一个命令，允许您将数据从本地机器(如您的笔记本电脑)复制到远程机器。作为数据科学家，在机器之间移动数据的简单方法是非常有价值的。

有大量优秀的文章向您介绍这些命令。下面是我喜欢的一个[教程](https://linuxacademy.com/blog/linux/ssh-and-scp-howto-tips-tricks/)。

# 饭桶

我发现越来越多的数据科学家在开始他们的第一份工作时，至少掌握了一些 Git 知识，这令人惊讶。如果你还没怎么用过 Git，现在停下来，创建一个 [GitHub](http://github.com/) 账户。

当使用远程机器时，代码的源代码控制变得更加重要，因为运行代码的位置每天都在变化(尤其是在云上)，Git 将允许您轻松地跟踪、克隆和合并您正在使用的任何机器上的变化。

这个 GitHub [指南](https://guides.github.com/activities/hello-world/)是一个很好的起点。如果你对该指南感到满意，那么你已经掌握了相当好的基础知识。根据我的经验，Git 的基础知识可以让你走得更远。

# 屏幕

屏幕是一个简单但非常有用的工具。一旦您了解了 SSH 并连接到您的第一台远程机器，您可能会痛苦地认识到，如果您的连接中断，在那台机器上运行的任何东西都将死亡。这对于长时间运行的数据科学工作来说并不理想。

答案是 Screen(注意:还有其他类似的更高级的工具，比如 tmux，但是我发现 screen 对初学者来说是最简单的)

Screen 允许您在一个进程中运行您的命令，即使断开连接，该进程也不会终止。

这很容易上手，你可以在这里学习如何[。](https://linuxize.com/post/how-to-use-linux-screen/)

![](img/294141cc4337a6b8ffeb9ebb65e5b750.png)

Source: Pexels

# 把所有的放在一起

至此，我们已经讨论了在远程机器上工作的基本工具。经验是最好的老师，所以我建议最重要的是使用这些工具。为此，我会在谷歌云计算上开一个免费账户。我推荐谷歌，因为它的免费账户将允许你访问 GPU 计算，如果你喜欢深度学习，这真的很有用。

这里有一篇很棒的文章可以带你完成这个过程:

[](https://medium.com/@jamsawamsa/running-a-google-cloud-gpu-for-fast-ai-for-free-5f89c707bae6) [## 免费为 fast.ai 设置 Google Cloud 实例 GPU

### 编辑*本指南是为 fastai 版本 1 编写的，该版本在当前日期和时间(2018 年 1 月)正处于…

medium.com](https://medium.com/@jamsawamsa/running-a-google-cloud-gpu-for-fast-ai-for-free-5f89c707bae6) 

走完所有这些步骤需要时间，你肯定会遇到困惑。但是，如果你坚持不懈，你将会创建一个免费使用的，带有 GPU 的远程机器。然后，您可以练习使用 SSH 从终端访问它，使用 Vim 编辑文件，使用 SCP 将数据传输到您的机器，使用 g it 移动和跟踪代码，以及使用 screen 在断开连接时不会丢失您的进程。

熟悉这种类型的数据科学工作流将让你为大数据和大计算时代做好准备，并让你在开始第一份工作时变得非常高效。

本文可在[这里](https://learningwithdata.com/posts/tylerfolkman/an-often-overlooked-data-science-skill-ea4254e19c6b/)找到。