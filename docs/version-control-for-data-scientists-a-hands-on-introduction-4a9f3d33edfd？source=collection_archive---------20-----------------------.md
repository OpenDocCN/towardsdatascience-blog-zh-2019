# 数据科学家的版本控制:实践介绍

> 原文：<https://towardsdatascience.com/version-control-for-data-scientists-a-hands-on-introduction-4a9f3d33edfd?source=collection_archive---------20----------------------->

从历史上看，许多数据科学家不使用像版本控制系统这样的“软件开发”工具。如今，随着他们的代码变得越来越复杂，数据科学家越来越多地受到他们的软件工程合作伙伴的影响，学习如何熟练地使用 Git 这样的版本控制系统变得越来越重要。

在这个简短的动手介绍中，您将学到足够的 Git，以便当您获得一份数据科学家的工作时，能够跟踪您的变化，并与您的同事分享。

![](img/b215b9a82d62ab673ae9419d92ccc980.png)

# 什么是版本控制？

版本控制系统允许你跟踪你对你的工作所做的改变。这有点像谷歌文档中的“跟踪修改”，但不同的是，你可以保存一组文件的修改，而不仅仅是单个文件内的修改。大多数版本控制系统也支持分支的思想，允许不同的人对相同的底层文件进行不同的更改，然后在以后将他们的工作合并在一起。

# 数据科学家如何使用版本控制？

作为一名数据科学家，即使你只处理一个文件(比如一个 Jupyter 笔记本)，你也可以使用版本控制系统来跟踪你的变更(通常会这样)。它允许您定期保存您的工作，这使得您可以轻松地将笔记本恢复到早期版本。

随着项目变得越来越复杂，版本控制变得更加有价值。用一个 Jupyter 笔记本开始一个项目是很常见的。随着时间的推移，笔记本变得充满了清理导入数据的小功能，以至于很难专注于笔记本的重要部分。

解决这个问题的一个好方法是将这些函数分解成单独的 Python 文件，您可以使用一行代码调用这些文件。通过这种方式，任何希望了解您的项目的人都可以从 Jupyter 笔记本中获得一个高层次的视图，然后如果他们想了解您的数据清理脚本的细微差别，他们可以随时深入研究您的支持 Python 文件。这种策略也使得编写[自动化单元测试](http://pytest.org/en/latest/)来确认您的数据清理脚本将您期望的转换应用于各种类型的输入变得更加容易。

一旦您的项目有多个需要保持同步的文件，像 Git 这样的版本控制系统就特别有用，因为它允许您对多个文件进行一系列更改，然后将它们一起“提交”,这样您就可以轻松地将所有文件恢复到提交后的状态。

# 安装 Git

如果你还没有安装 Git，请点击[这里](https://git-scm.com/downloads)，按照你的操作系统的安装程序进行操作。

# Git 101

让我们从定义几个关键概念开始，这些概念在我们讨论 Git 时会有所帮助:

*   **一个库**——这是 Git 对一个项目的名称。它包括项目中的所有文件，以及它们如何随时间变化的所有信息。如果您有一个存储库的完整副本(通常称为“repo”)，您可以查看项目的当前状态，但也可以查看项目以前所处的任何状态。
*   **提交** —在 Git 中，历史由一系列保存在变更日志中的提交组成。每当您对项目进行一组有意义的更改时，您都应该提交它们，这样您就可以在将来将项目恢复到那个状态。
*   **暂存区** —这就像一个用于版本控制的购物篮。在这里，您可以加载您希望在下一次提交时使用的更改集，因此，如果您编辑了三个文件，但希望对其中两个文件进行一次提交，对第三个文件进行另一次提交，您只需使用命令“暂存”前两个文件，然后使用适当的消息提交它们，然后分别添加和提交最后一个文件。

# Git 入门

让我们用 Git 做一些实践。如果你用的是 windows，打开“Git Bash”程序，如果你用的是 Mac 或 Linux，只需打开一个终端窗口。*重要的是不要在 windows 机器上只打开 Powershell 或默认终端——它不会正常工作。*

转到您的主目录中的某个目录(这样您就有写权限)。让我们确保您还没有在一个属于 Git 存储库的目录中(不太可能，但是会发生):

```
> git status
fatal: not a git repository (or any of the parent directories): .git 
```

很好。我们向 Git 询问我们所在的存储库的状态，它让我们知道我们不在 Git repo 中。这很好——在另一个中创建一个 Git repo 会让您和 Git 都感到困惑！

现在让我们创建一个新的 Git repo 和目录:

```
> git init my_first_repo 
Initialized empty git repository in /Users/peterbell/Dropbox/code/my_first_repo/.git/
```

完美。所以它在我所在的目录下创建了一个存储库。让我们使用 Unix 的“更改目录(cd)”命令去那里:

```
**>** cd my_first_repo
my_first_repo git:(master) 
```

好了，我的终端通过显示`git:(master)`消息来告诉我什么时候我在 Git repo 中。让我们看看这个项目的状态:

```
> git status
On branch master
No commits yet
nothing to commit (create/copy files and use "git add" to track)
```

酷毙了。如果您看到稍微不同的消息，不要担心——它们因操作系统和 Git 版本而异，但底线是 Git 告诉我们还没有任何提交，我们在“主”分支(主分支)上，这里没有任何文件要保存到版本控制中。

# 初始配置

让我们检查一下您是否有 Git 的基本配置，这样当您保存文件时，它就知道您的姓名和电子邮件地址。

```
> git config --global user.name
Peter Bell
```

使用上面的命令，我们可以访问您计算机上 Git 的配置设置。`*--global*`标志意味着我们正在查看配置设置，这些设置将应用于您在这台机器上作为这个用户登录的所有项目。不常见的`*--system*`标志用于访问机器上所有用户共享的设置。最后，`*--local*`标志访问特定项目的设置——所以只有当您运行命令时，它才在 Git repo 中工作。

当您向 git config 传递一个不带值的键(在本例中是键)时，它返回现有的值。如果您还传递了一个值，它将设置该值。根据您的设置，您可能会看到您的名字，什么也没看到，一条 Git 没有正确设置的消息，甚至一条找不到文件的错误消息。如果你看到除了你的名字之外的任何东西，像这样设置你的名字:

```
> git config --global user.name ‘Your Name’ 
```

然后运行:

```
> git config --global user.name
Your Name
```

现在你应该看到你的名字了！让我们为您的电子邮件地址做同样的事情:

```
> git config --global user.email
this@that.com
```

如果它没有您想要的值，请将其设置为某个值。不需要引号:

```
> git config --global user.email this@that.com 
> git config --global user.email 
this@that.com
```

还有很多其他的设置，但是至少 Git 现在知道用什么名字和电子邮件地址来保存你的提交了！

# 添加一些文件

创建测试文件最简单的方法是使用 Unix 命令“touch”如果文件存在，它只会更新时间戳。如果没有，它将创建一个空白文件，然后我们可以添加到版本控制中。

所以让我们创建三个文件。它们不会有任何内容，但我们会给它们起一些名字，在进行真正的数据科学项目时可能会用到。

```
> touch index.ipynb 
> touch import.py 
> touch clean.py 
> git status 
On branch master 
No commits yet 
Untracked files: (use "git add <file>..." to include in what will be committed) 
      clean.py 
      import.py 
      index.ipynb 
nothing added to commit but untracked files present (use "git add" to track) 
```

好的，我们还是在主分支上。我们还没有提交(保存到 Git 中的永久历史中)，这三个文件是“未被跟踪的”——在我们添加它们之前，Git 并没有真正关注它们。

现在假设我们想对 Jupyter 笔记本文件(index.ipynb)进行一次初始提交，然后对导入和清理脚本进行另一次提交。

```
> git add index.ipynb
> git status
On branch master 
No commits yet
Changes to be committed 
  (use "git rm --cached <file>..." to unstage)
      new file:   index.ipynb
Untracked files:
  (use "git add <file>..." to include in what will be committed)
      clean.py
      import.py
```

这告诉我们，当我们现在提交时，index.ipynb 文件将被保存。让我们这样做:

```
> git commit -m 'Add Jupyter Notebook file' 
[master (root-commit) 998db10] Add Jupyter Notebook file 
1 file changed, 0 insertions(+), 0 deletions(-) 
create mode 100644 index.ipynb
```

好吧，这是怎么回事？首先，我告诉 Git 提交—将这组更改保存到历史中。我向它传递了`-m`标志，以传递提交消息。我在`-m`后面加上了我希望与这个提交相关联的消息，用单引号或双引号括起来。提交消息的目的是让将来的任何人都更容易理解我做了什么更改以及为什么做这些更改。

重要的是要知道，每次提交都需要两件事——一条提交消息和至少一个添加、修改、重命名或删除的文件。根据您的操作系统和 Git 的版本，如果您没有传递提交消息，它要么会为您创建一个默认消息，要么会将您扔进您与 Git 一起使用的任何文本编辑器(注意，它可能有点像 [vi](https://en.wikipedia.org/wiki/Vi) )来添加提交消息。

而回应是什么意思？嗯，它告诉我们，我们仍然在主项目上，我们刚刚对这个项目进行了根(非常第一次)提交。它为我们提供了十六进制 SHA-1 哈希的前 7 个字符，这是 Git 存储库中每次提交的唯一标识符，它共享了我的提交消息以及有多少文件被更改。在本例中，我们添加了 1 个文件，但是没有添加或删除任何内容行，因为文件是空的。它还显示了提交中的文件(index.ipynb ),并显示“创建模式 100644 ”,您几乎可以忽略它。

酷毙了。我们现在的 Git 状态如何？

```
> git status
On branch master
Untracked files:
(use "git add <file>..." to include in what will be committed)
	clean.py
	import.py
nothing added to commit but untracked files present (use "git add" to track)
```

完美。所以它看到我们还有两个未被跟踪的文件。让我们将它们添加到临时区域:

```
> git add . 
```

在 Git 中，有很多方法可以将文件添加到“暂存区”。您可以一次给它们命名一个(`git add clean.py import.py`)。您可以使用 [fileglob](https://github.com/begin/globbing) 模式(`git add *.py`)匹配一组文件，或者您可以只添加 repo 中的所有文件(`git add .`)。

无论采用哪种方法，都会将另外两个新文件添加到临时区域。

```
> git status
On branch master
Changes to be committed:
  (use "git reset HEAD <file>..." to unstage)
	new file:   clean.py
	new file:   import.py
```

所以我们要做的就是承诺他们:

```
> git commit -m ‘Add import and cleaning scripts’
[master 625e7a1] Add import and cleaning scripts
 2 files changed, 0 insertions(+), 0 deletions(-)
 create mode 100644 clean.py
 create mode 100644 import.py
```

很好——它在 master 上进行了一次新的提交(对我来说是`625e7a1`—对你来说会有所不同，因为它部分基于这次和以前提交中使用的用户名和电子邮件),并添加了两个新文件(但没有文本行，因为在这个简单的教程示例中，它们都是空白文件)。

恭喜你！您刚刚创建了一个新的 Git repo，并暂存和添加了一些文件！

# 集结地是怎么回事？

现在，你可能会问一个非常合理的问题*“为什么我们必须运行两个单独的命令* `*git add*` *然后* `*git commit*` *来保存我们的工作？”*

首先，这不是你必须一直做的事情。作为一名数据科学家，您将大部分时间花在修改文件上——通常是您的 Jupyter 笔记本和一些支持 Python 的文件。当你修改文件时，Git 给出了一个快捷方式`git commit -am “your message here*"*`,它将在一行中添加修改的文件并提交它们，所以大多数时候你只需要键入一个命令。

但是 staging area 的真正强大之处在于，每当您做出多个更改，然后想要返回并将它们分类到单独的提交中时。

您可能会问*“为什么要有一堆不同的提交呢？”*这是一个特别常见的问题，来自那些使用过 subversion 之类的旧版本控制系统的人，在 subversion 中提交是一个较慢的过程，他们通常只是写一整天的代码，然后保存他们的更改，并显示一条类似于“我在周一做的事情”的信息！

创建有意义的提交消息的原因很重要，每种类型的更改都有一个提交(“更新的可视化”、“添加了一个分类数据的热编码”等)，这样当您或您的团队回到您的日志时，就很容易理解您是如何到达这里的，并找到甚至恢复(撤销)任何有问题的东西。这和你不把你的变量命名为“a”、“b”和“c”是一个道理——计算机不会介意，但是下次你拿起代码并试图弄清楚它到底是什么的时候，它不会让你的生活变得更容易！

# 后续步骤

关于 Git 还有很多要学的。我们还没有介绍分支、从远程服务器推和拉、撤销更改、更高级的配置设置，或者如何检查以前的提交，但是一旦您理解了 staging area 的基本原理，您将会领先于许多已经使用 Git 一段时间的人。在接下来的几周里，请关注本系列的更多文章！

*最初发表于*[](https://flatironschool.com/blog/version-control-for-data-scientists)**。**