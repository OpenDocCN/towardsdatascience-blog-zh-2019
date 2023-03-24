# 你应该知道的 10 个 Git 命令

> 原文：<https://towardsdatascience.com/10-git-commands-you-should-know-df54bea1595c?source=collection_archive---------1----------------------->

## 使用 Git 节省时间的技巧

![](img/c0683d62f3d62eee227a0e3bae0ec272.png)

在本文中，我们将讨论作为开发人员、数据科学家或产品经理应该知道的各种 Git 命令。我们将看看如何用 Git 检查、移除和整理。我们还将介绍使用 Bash 别名和 Git 编辑器配置来避免 Vim 和节省时间的方法。

如果您对基本的 Git 命令不太熟悉，在阅读本文之前，可以看看我以前的一篇关于 Git 工作流的文章。

[](/learn-enough-git-to-be-useful-281561eef959) [## 学会足够有用的东西

### GitHub 项目的 4 个基本工作流程

towardsdatascience.com](/learn-enough-git-to-be-useful-281561eef959) 

这里有 10 个需要了解的命令和它们的一些常见标志。每个命令都链接到该命令的 Atlassian 位存储桶指南。

# 检查东西

让我们先来看看检查变更。

![](img/17ccbc03fe3bf66b1e513709ee5d8a53.png)

*   `[**git diff**](https://www.atlassian.com/git/tutorials/saving-changes/git-diff)` —在本地查看所有文件更改。可以附加文件名以仅显示一个文件的更改。📂
*   `[**git log**](https://www.atlassian.com/git/tutorials/git-log)` —查看所有提交历史记录。也可用于带有`git log -p my_file`的文件。输入`q`退出。📓
*   `[**git blame my_file**](https://www.atlassian.com/git/tutorials/inspecting-a-repository/git-blame)` —在*我的文件*中查看谁在何时更改了什么。👉
*   `[**git reflog**](https://www.atlassian.com/git/tutorials/rewriting-history/git-reflog)` —显示本地存储库头的更改日志。有利于寻找丢失的作品。🔎

用 Git 检查东西并不令人困惑。相比之下，Git 为删除和撤销提交和文件更改提供了大量选项。

# 撤销事情

`git reset`、`git checkout`和`git revert`用于撤销对您的存储库所做的更改。这些命令可能很难保持正确。

`git reset`和`git checkout`可以用于提交和单个文件。`git revert`仅用于提交级别。

如果您只是处理您自己的本地提交，而这些提交还没有合并到协作远程工作中，那么您可以使用这些命令中的任何一个。

如果您正在协同工作，并且需要中和远程分支中的提交，那么`git revert`就是您的工具。

![](img/d6c742ab972f840dd4bfed8479e3b256.png)

这些命令中的每一个都有不同的选项。以下是常见的用法:

*   `[**git reset --hard HEAD**](https://www.atlassian.com/git/tutorials/resetting-checking-out-and-reverting)` —放弃自最近一次提交以来已暂存和未暂存的更改。

指定一个不同的提交而不是`HEAD`来放弃自该提交以来的更改。`--hard`指定丢弃暂存和未暂存的更改。

确保您没有放弃来自您的合作者所依赖的远程分支的提交！

*   `[**git checkout my_commit**](https://www.atlassian.com/git/tutorials/undoing-changes)` —丢弃自 *my_commit 以来未暂存的更改。*

`HEAD`通常用于`my_commit`来放弃自最近一次提交以来对本地工作目录的更改。

`checkout`最适合用于本地撤销。它不会弄乱您的合作者所依赖的远程分支的提交历史！

如果使用分支而不是提交来使用`checkout`，那么`HEAD`会切换到指定的分支，并且工作目录会更新以与之匹配。这是`checkout`命令更常见的用法。

*   `[**git revert my_commit**](https://www.atlassian.com/git/tutorials/undoing-changes/git-revert)`—撤销 *my_commit* 中更改的效果。`revert` 在撤销更改时进行新的提交。

`revert`对于协作项目是安全的，因为它不会覆盖其他用户分支可能依赖的历史。

![](img/529137fcf15afd41c3c31f98640de0de.png)

`revert` is safe

有时，您只想删除本地目录中未被跟踪的文件。例如，您可能运行了一些代码，这些代码创建了许多不同类型的文件，而您不希望这些文件出现在您的 repo 中。哎呀。😏你可以在一瞬间清理它们！

*   `[**git clean -n**](https://www.atlassian.com/git/tutorials/undoing-changes/git-clean)`—删除本地工作目录中未跟踪的文件。

`-n`标志用于不删除任何内容的试运行。

使用`-f`标志实际删除文件。

使用- `d`标志删除未跟踪的目录。

默认情况下，*不跟踪文件。gitignore* 不会被删除，但是这种行为可以改变。

![](img/863d795b15fb8d9820c346d82e109464.png)

现在您已经知道了在 Git 中撤销事情的工具，让我们再看两个命令来保持事情有序。

# 整理东西

*   `[**git commit --amend**](https://www.atlassian.com/git/tutorials/rewriting-history#git-commit--amend)` —将您的暂存更改添加到最近的提交中。

如果没有暂存任何内容，此命令只允许您编辑最近的提交消息。仅当提交尚未集成到远程主分支时，才使用此命令！⚠️

*   `[**git push my_remote --tags**](https://www.atlassian.com/git/tutorials/syncing/git-push)`—将所有本地标签发送到远程 repo。有利于版本变更。

如果你正在使用 Python 并对你构建的包进行修改， [bump2version](https://pypi.org/project/bump2version/) 会自动为你创建标签。一旦你推送了你的标签，你就可以在你的新闻稿中使用它们了。[这是我制作你的第一个 OSS Python 包的指南](/build-your-first-open-source-python-project-53471c9942a7?source=friends_link&sk=576540dbd90cf2ee72a3a0e0bfa72ffb)。跟随 [me](https://medium.com/@jeffhale) 确保你不会错过版本控制的部分！

# 救命，我卡在 Vim 里出不来了！

使用 Git，您可能偶尔会发现自己被扔进了 Vim 编辑器会话。例如，假设您试图在没有提交消息的情况下提交—除非您更改了 Git 的默认文本编辑器，否则 Vim 会自动打开。如果你不知道 Vim，这种情况很糟糕——参见[上的 4000+up 投票以及如何摆脱这种情况的答案](https://stackoverflow.com/a/11828573/4590385)。😲

![](img/8a1a52d850730887aeff10ef1fef40ea.png)

Freedom!

下面是您用保存的文件逃离 Vim 的四步计划:

1.  按下`i`进入插入模式。
2.  在第一行键入您的提交消息。
3.  按下退出键— `esc`。
4.  输入`:x`。别忘了冒号。

瞧，你自由了！😄

# 更改默认编辑器

为了完全避免 Vim，您可以在 Git 中更改您的默认编辑器。[以下是带有常用编辑器命令的文档](https://www.atlassian.com/git/tutorials/setting-up-a-repository/git-config)。下面是将默认值更改为 Atom 的命令:

`git config --global core.editor "atom --wait"`

如果您喜欢 Visual Studio 代码:

`git config --global core.editor "code --wait"`

假设您已经安装了文本编辑器，现在您可以在其中解决 Git 问题了。耶！👍

我应该注意到 VSCode 有很多很棒的 Git 功能。详情见此处的文档[。](https://code.visualstudio.com/docs/editor/versioncontrol)

# 为 Git 命令创建快捷方式

![](img/765a735ed0c8e794145af22f299fab85.png)

如果您在 macOS 上为您的终端使用 Bash，您可以通过向您的*添加以下别名来创建 Git 命令的快捷方式。bash_profile* 。

```
alias gs='git status '
alias ga='git add '
alias gaa='git add -A '
alias gb='git branch '
alias gc='git commit '
alias gcm='git commit -m '
alias go='git checkout '
```

您可以调整以上内容，为您喜欢的任何 Git 命令创建快捷方式。

如果你没有一个. *bash_profile* ，你可以用下面的代码在 macOS 上创建一个:

`touch ~/.bash_profile`

然后打开它:

`open ~/.bash_profile`

查看更多关于*的信息。bash_profile* 这里[这里](https://stackoverflow.com/a/30462883/4590385)。

现在，当你在终端中输入`gs`时，它和输入`git status`是一样的。请注意，您可以在您的终端中输入快捷方式后的其他标志。

您也可以创建 Git 别名，但是这需要您在快捷命令前键入`git`。谁需要这些额外的按键？😉

# 包装

在本文中，您已经看到了一些关键的 Git 命令，并配置了您的环境以节省时间。现在你有了 Git 和 GitHub 的基础。准备好下一步了吗？

*   查看[这篇 Bitbucket Git 教程](https://www.atlassian.com/git/tutorials/learn-git-with-bitbucket-cloud)以深入了解。
*   探索这个[交互式指南](https://learngitbranching.js.org/)来 Git 分支。分支可能会令人困惑，所以绝对值得一看。🔎
*   去玩，去学习，去给别人解释不同之处。

我希望这篇 Git 和 GitHub 介绍对你有用。如果你有，请分享到你最喜欢的社交媒体渠道，这样其他人也可以找到它。👏

我写了如何有效地使用 Python、Docker 和其他编程和数据科学工具。如果你对此感兴趣，请关注我，在这里阅读更多。

[![](img/ba32af1aa267917812a85c401d1f7d29.png)](https://dataawesome.com)

去干掉他们！😁

![](img/c28be700d2f2feeb9ae553dd7f7bf151.png)