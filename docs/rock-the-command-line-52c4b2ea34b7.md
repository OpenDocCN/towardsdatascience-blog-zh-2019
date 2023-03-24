# 摇摆命令行

> 原文：<https://towardsdatascience.com/rock-the-command-line-52c4b2ea34b7?source=collection_archive---------2----------------------->

## 节省您时间的 21 个 Bash 命令

本文是对 Bash 命令的简单介绍。知道如何在命令行中移动是节省时间的一项基本技能。我们将讨论 21 个最常见的命令及其关键标志。我们还将介绍 Bash 快捷方式，为您节省大量的击键次数。⌨️

![](img/3cf20f0e2b7b4755d02aebd92ffbfa70.png)

A shell

## 条款

术语 *Unix* 、 *Linux* 、 *Bash* 、 *shell* 、*命令行、终端、*和 *shell 脚本*是不同的东西，但是它们共享我们将讨论的命令。这里有一个差异和相似之处的快速分析。

[*Unix*](https://en.wikipedia.org/wiki/Unix) 是贝尔实验室在 20 世纪 70 年代开发的一种流行的计算机操作系统。它不是开源的。

[*Linux*](https://en.wikipedia.org/wiki/Linux) 是最流行的 Unix 衍生品。它运行着世界上大量类似计算机的设备。

你的 [*终端*](https://en.wikipedia.org/wiki/Terminal_emulator) 是让你进入操作系统的模拟器程序。您可以同时打开多个终端窗口。

[*shell*](https://en.wikipedia.org/wiki/Shell_(computing)) 语言用于向底层操作系统发送命令。

[*Bash*](https://www.gnu.org/software/bash/) 代表*伯恩再次脱壳。*它是最常见的与操作系统通信的 shell 语言。这也是 macOS 的默认外壳。要了解更多关于支持 Linux 和 Bash 的开源基金会的信息，请查看本文。

[*命令行界面*](https://en.wikipedia.org/wiki/Command-line_interface) *(CLI)* 是指基于键盘输入的界面，在这里输入命令。通常，它用于实时信息和文件操作。将 CLI 与通常使用鼠标的图形用户界面(GUI)进行对比。CLI 通常被称为命令行。

[*脚本*](https://en.wikipedia.org/wiki/Scripting_language) 指的是包含一系列 shell 命令的小程序。脚本被写入文件，可以重用。您可以在脚本中创建变量、条件语句、循环、函数等等。

唷。明白了吗？我们下面讨论的命令适用于上面所有斜体的术语。我会交替使用 *Bash* 、 *shell* 和*命令行*。

另请注意:我交替使用*目录*和*文件夹*。这两个术语的意思是一样的。

标准输入/输出[流](https://en.wikipedia.org/wiki/Standard_streams)是标准输入( *stdin* )、标准输出( *stdout* )和标准误差( *stderror* ) *。他们会突然冒出一大堆。当我使用术语 *print 时，*我指的是打印到 stdout，而不是打印机。*

最后，将下面前缀为 *my* _ *whatever* 的命令替换为 your whatever。😄

事不宜迟，下面是我们将在本文中讨论的命令列表。

# 21 大 Bash 命令

## 获取信息

`**man**`:打印一个命令的手册(帮助)
`**pwd**`:打印工作目录
`**ls**`:列出目录内容
`**ps**`:查看正在运行的进程

## 操作

`**cd**`:更改工作目录
`**touch**`:创建文件
`**mkdir**`:创建目录
`**cp**`:复制
`**mv**`:移动或重命名
`**ln**`:链接

## 重定向和管道

`**<**`:重定向标准输入
`**>**`:重定向标准输出
`**|**`:将一个命令的内容传送到下一个命令

## 阅读

`**head**` : 读取文件的开头
`**tail**` : 读取文件的结尾
`**cat**`:读取一个文件或者连接文件

## 结束

`**rm**`:删除
`**kill**` **:** 结束一个流程

## 搜索

`**grep**`:搜索
`**ag**`:搜索

## 档案馆

`**tar**`:将多个文件合并成一个文件

让我们开始吧！

# 前 21 条命令解释

首先让我们看看以 [*stdout*](https://en.wikipedia.org/wiki/Standard_streams) 形式返回信息的命令，这意味着标准输出。一般情况下， *stdout* 会写到你的终端。

## 获取信息

`**man command_name**`:打印命令手册。就像帮助一样。

`**pwd**`:打印当前工作目录的文件路径。您经常需要知道自己在文件系统中的位置。

`**ls**`:列出目录内容。另一个超级常用命令。

`ls -a`:也用`-a`列出隐藏文件。

`ls -l`:使用`-l`查看文件的更多信息。

注意标志可以这样组合:`ls -al`。

`**ps**`:查看正在运行的流程。

`ps -e` **:** 用`-e`打印所有正在运行的进程，而不仅仅是与当前用户 shell 关联的进程。这通常是你想要的。

# 操作

`**cd my_directory**`:将工作目录更改为*我的 _ 目录*。使用 *my_directory* 的相对路径`../`在目录树中上移一级。

![](img/a517a52877e14192e6a3406e7321aff4.png)

CD

`**touch my_file**`:在指定的路径位置创建 *my_file* 。

`**mkdir my_directory**`:在指定的路径位置创建 *my_directory* 。

`**mv my_file target_directory**`:将*我的文件*移动到*目标目录*。

`mv`也可用于重命名文件或文件夹，如下所示:

`mv my_old_file_name.jpg my_new_file_name.jpg`

`**cp my_source_file target_directory**`:将*源文件*复制一份，放入*目标目录*。

`**ln -s my_source_file my_target_file**`:用符号链接将*我的目标文件*链接到*我的源文件*。

当 *my_source_file 的*内容更新时， *my_target_file 的*内容会自动更新。如果我的目标文件的内容被更新，那么我的源文件的内容也被更新。酷毙了。

如果 *my_source_file* 被删除，那么*my _ target _ file’*的内容被删除，但是文件继续存在。只是一个空文件。

如果 *my_target_file* 被删除， *my_source_file* 继续存在，其内容保持不变。

`-s`标志也允许你链接目录。

* * 2019 年 4 月 12 日更新，纠正并澄清 *ln -s* 和 *mv* 行为。感谢杰森·沃尔沃克。**

现在让我们看看输出重定向和管道是如何工作的。

# 重定向和管道

`**my_command < my_file**`:将 stdin 重定向到 *my_file* 。当 my_command 需要用户输入来执行某项操作时非常有用。

`**my_text > my_file**`:将 stdout 重定向到 *my_file* 。创建 *my_file* 如果它不存在。覆盖 *my_file* 如果它确实存在。

例如`ls > my_folder_contents.txt`创建一个文本文件，列出你的工作目录的内容。

使用 double `>>`将 stdout 附加到 *my_file* 而不是覆盖它。

现在我们来看看管道命令。

![](img/ea04476943b33f38aa44d8bf3c8ed073.png)

Pipe the result of one command to the other

`**first_command | second_command**`:管道符|用于将一个命令的结果发送给另一个命令。管道左侧命令的 stdout 被传递给管道右侧命令的 stdin。

“一切都是管道”是 Unix 中的一句口头禅——因此几乎任何有效的命令都可以通过管道传输。

用管道链接命令会创建一个管道。多个管道可以像这样连接在一起:

`first_command | second_command | third_command`

![](img/f0f7ce2bebd3f84454cc24cb13b22eb4.png)

Pipeline

请注意，管道并行执行所有命令。这种行为偶尔会导致意想不到的结果。点击阅读更多[。](https://stackoverflow.com/a/9834118/4590385)

说到读取，我们来看看从命令行怎么做。

# 阅读

`**head my_file**`:读取 my_file 的前几行。也可以读取其他 stdin。

`**tail my_file**` : 读我的 _ 文件的最后几行。也可以读取其他 stdin。

![](img/b6c95eb018ecf35424e6312a082d618c.png)

Head at the front, tail at the back.

如果您是使用 pandas 的数据科学家，那么最后两个命令应该听起来很熟悉。如果不是，*头*尾*尾*都是映射得比较好的隐喻，应该不会太难记。

让我们看看另一种读取文件的方法。

`cat`根据传递的文件数量，打印一个文件或连接多个文件。

![](img/df548d5f4d5afe9dff461ffccfc7f45f.png)

cat

`**cat my_one_file.txt**`:一个文件，`cat`将内容打印到 stdout。

当您给 cat 命令两个或更多文件时，它的行为会有所不同。

`cat my_file1.txt my_file2.txt`:对于两个或更多的文件，`cat` con *cat* 将文件的内容合并在一起，并将输出打印到 stdout。

如果您想将连接的文件保存为一个新文件，使用`>`写操作符，如下所示:

`cat my_file1.txt my_file2.txt > my_new_file.txt`

现在我们来看看移除和结束事物。

# 结束

`**rm my_file**`:从你的文件系统中删除 *my_file* 。

`rm -r my_folder`:删除 *my_folder* 以及 *my_folder* 中的所有文件和子文件夹。`-r`用于递归。

如果不想每次删除都有确认提示，请添加`-f`。

`**kill 012345**` **:** 通过给它时间关闭来优雅地结束指定的运行进程。

`**kill -9 012345**` **:** 立即强制结束指定的运行过程。`-s SIGKILL` 与`-9`意思相同。

# 搜索

接下来的几个命令——`grep`、`ag`和`ack`——用于搜索。Grep 是古老的、值得信赖的兄弟——可靠，但是速度较慢，用户友好性稍差。

![](img/fcbf2ac762d5a27fe722d96afd24c1b5.png)

Get a grep!

`**grep my_regex my_file**`:在 *my_file* 中搜索 *my_term* 。为每个匹配返回文件的整行。 *my_term* 默认为正则表达式。

`grep -i my_regex my_file` : `-i`使搜索不区分大小写。

`grep -v my_regex my_file`:返回所有不包含 *my_term* 的行。`-v`返回逆，像很多语言中的*不是*。

`grep -c my_regex my_file` : 用`-c`返回找到匹配的次数。

`grep -R my_regex my_folder`:用`-R`递归搜索文件夹和所有子文件夹中的所有文件。

现在让我们转向*Ag*——*grep*更年轻、更快、更帅的兄弟姐妹。

![](img/6310ccee1c40affae6cc5607708aa835.png)

Get it?

如果您运行以下命令并发现您的机器上没有 *ag* ，请参见此处的安装说明[。在装有自制软件的 Mac 上运行`brew install the_silver_searcher`。(更新于 2019 年 8 月)。](https://github.com/ggreer/the_silver_searcher)

`**ag my_regex my_file**`:返回行号和任何匹配的行。

`ag -i my_regex my_file` : `-i`为不区分大小写。

*Ag* 自动读取你的*。gitignore* 文件并排除任何匹配文件或文件夹的结果。相当酷！

`ag my_regex my_file–skip-vcs-ignores`:用`–skip-vcs-ignores`覆盖自动版本控制系统文件读取。

也可以做一个*。忽略*文件，从*标签*中排除文件路径。

第三个同胞是 *Ack* 。Ag 和 Ack*几乎是同卵双胞胎——他们 99%可以互换。Ag* 更快，所以我会坚持使用它。

# 档案馆

现在让我们看看如何制作 tarball 档案。

`**tar my_source_directory**`:将源目录下的多个文件合并成一个 tarball 文件。此命令对于分发其他人将下载的文件很有用。

![](img/9dbe62c1c1b7171853e95794c983e447.png)

tar

一个 tarball 有*。tar* 文件扩展名，代表磁带存档。磁带告诉你这个命令有多老了！

`tar -cf my_file.tar my_source_directory`:用 *my_source_directory* 的内容创建一个名为 *my_file.tar* 的 tarball 文件。`-c`用于创建，`-f`用于文件。

用`-xf`提取一个 tar 文件。`-x`用于提取，`-f`用于归档。

`tar -xf my_file.tar`将 *my_file.tar* 中的文件展开到当前工作目录。

现在让我们来看看拉链和解压*。tar* 文件。

`tar -cfz my_file.tar.gz my_source_directory`使用 gzip 压缩文件。`-c`表示创建，`-f`表示文件，`-z`表示压缩。Gzip 为文件的消费者节省了空间和下载时间。

通过在我们之前看到的提取命令中添加`-z`标志来解压缩`.tar.gz`文件。

`tar -xfz my_file.tar.gz`。`-x`为提取，`-f`为文件，`-z`为压缩。

tar 有很多其他的标志可以使用。

# Bash 别名

创建 Bash 别名来保存您在终端中的击键。然后你可以做类似于键入`bu`而不是`python setup.py sdist bdist_wheel`的事情。

只需在您的`~/.bash_profile`中添加以下内容:

`alias bu="python setup.py sdist bdist_wheel"`

如果你没有一个`~/.bash_profile`文件，你可以从命令行用`touch`命令创建一个。

然后重启你的终端，用两次击键[构建你的 Python 包](/build-your-first-open-source-python-project-53471c9942a7?source=friends_link&sk=576540dbd90cf2ee72a3a0e0bfa72ffb)。没有比输入 2 个字母而不是 44 个字母更好的了。😃

添加您喜欢的任何其他别名，并观察您的生产力增长。🌴

让我们回顾一下我们已经讲过的内容。

# 回顾:21 个 Bash 命令

## 获取信息

`**man**`:打印命令
的手册(帮助)`**pwd**`:打印工作目录
`**ls**`:列出目录内容
`**ps**`:查看正在运行的进程

## 操作

`**cd**`:更改工作目录
`**touch**`:创建文件
`**mkdir**`:创建目录
`**cp**`:复制
`**mv**`:移动或重命名
`**ln**`:链接

## 重定向和管道

`**<**`:重定向标准输入
`**>**`:重定向标准输出
`**|**`:将一个命令的内容管道化到下一个命令

## 阅读

`**head**` : 读取文件的开头
`**tail**` : 读取文件的结尾
`**cat**`:读取文件或连接文件

## 结束

`**rm**`:删除
`**kill**` **:** 结束一个流程

## 搜索

`**grep**`:搜索
`**ag**`:搜索

## 档案馆

`**tar**`:将多个文件合并成一个文件

# 包装

在本文中，您已经看到了 21 个最常见的 shell 命令。如果你还有一个你认为应该上榜的，请在 Twitter @discdiver 上告诉我。

您还看到了如何创建 Bash 别名来节省时间。

如果您想深入了解，这里有一些资源:

*   *征服命令行*是马克·贝茨的一本很棒的免费电子书。

[](http://conqueringthecommandline.com/book/basics) [## 征服命令行

### 学习掌握和征服最有价值和最有用的命令行工具，用于基于 Unix 和 Linux 的系统。在这个…

conqueringthecommandline.com](http://conqueringthecommandline.com/book/basics) 

*   来自 gnu.org 的官方 Bash 文档在这里。
*   `Sed`和`Awk`听起来像是两兄弟，但它们实际上是 Bash 中常见的文本处理实用程序。在这里了解更多关于他们的[。](https://www.tldp.org/LDP/abs/html/sedawk.html)
*   `cURL` —读作“curl”——用于通过 url 和测试服务器传输数据。点击了解更多[。](https://ec.haxx.se/curl-does.html)
*   如果你想学习如何把这些命令和其他 Bash 代码放到脚本中，[这里有](https://guide.bash.academy/commands/)一个很好的指南。
*   这里有一个关于 Bash 脚本的大备忘单。

像任何语言一样，学习 Bash 需要练习。用它来提高你的工作效率，并享受把它教给别人的乐趣。😄

我写了如何使用编程和数据科学工具，如 Docker、Git 和 Python。如果你对此感兴趣，在这里阅读更多[和关注我。👏](https://medium.com/@jeffhale)

[![](img/ba32af1aa267917812a85c401d1f7d29.png)](https://dataawesome.com)

炮轰快乐！

![](img/8416c2dc932facf02b2f094981836a90.png)