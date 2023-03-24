# 你应该知道的 10 种 Python 文件系统方法

> 原文：<https://towardsdatascience.com/10-python-file-system-methods-you-should-know-799f90ef13c2?source=collection_archive---------6----------------------->

## 用 os 和 shutil 操作文件和文件夹

您可以编写 Python 程序来与文件系统交互，做一些很酷的事情。如何做到这一点并不总是非常清楚。

本文是当前和有抱负的开发人员和数据科学家的指南。我们将重点介绍 10 个基本的 *os* 和 *shutil* 命令，这样您就可以编写脚本来自动化与文件系统的交互。

![](img/67e6e4518b875c9357cdf3f807a0c916.png)

Like a file system

文件系统有点像房子。假设你正在进行大扫除，你需要把笔记本从一个房间搬到另一个房间。

![](img/31c8ffa05925ac5de2ecc4291d6251fb.png)

Directories are like boxes. They hold things.

这些盒子就像目录。他们拿着东西。在这种情况下，笔记本。

![](img/9b754934bc27dc9019c82cfefb79927b.png)

笔记本就像文件一样。你可以给他们读和写。你可以把它们放在你的目录箱里。

明白吗？

在本指南中，我们将看看来自 *os* 和 *shutil* 模块的方法。 [*os*](https://docs.python.org/3/library/os.html) 模块是用于与操作系统交互的主要 Python 模块。 [*shutil*](https://docs.python.org/3/library/shutil.html#module-shutil) 模块也包含高级文件操作。出于某种原因，你用 *os* 创建目录，但是用 *shutil* 移动和复制它们。去想想。😏

## 更新:pathlib 讨论于 2019 年 2 月 16 日添加

在 Python 3.4 中， [*pathlib*](https://docs.python.org/3/library/pathlib.html) 模块被添加到标准库中，以改进文件路径的工作，从 3.6 开始，它与标准库的其余部分配合得很好。与我们将在下面讨论的方法相比， *pathlib* 方法为解析文件路径提供了一些好处——即 *pathlib* 将路径视为对象而不是字符串。虽然 pathlib 很方便，但是它没有我们将要探索的所有底层功能。此外，在未来几年的代码中，你无疑会看到下面的 *os* 和 *shutil* 方法。所以熟悉他们绝对是个好主意。

我打算在以后的文章中讨论 *pathlib* ，所以[跟着我](https://medium.com/@jeffhale)确保你不会错过。现在要了解更多关于 *pathlib* 模块的信息，请看这篇[文章](https://realpython.com/python-pathlib/)和这篇[文章](https://pbpython.com/pathlib-intro.html)。

在我们深入研究之前，还需要知道其他一些事情:

*   本指南是为 Python 3 设计的。2020 年 1 月 1 日之后将不再支持 Python 2。
*   您需要将 *os* 和 *shutil* 导入到您的文件中才能使用这些命令。
*   我的示例代码可以在 [GitHub](https://github.com/discdiver/os-examples) 上找到。
*   用你自己的论点代替下面引号中的论点。

现在我们已经了解了上下文，让我们开始吧！这里列出了你应该知道的 10 个命令。

# 10 种文件系统方法

下面的列表遵循这种模式:

**方法—描述—等效 macOS Shell 命令**

## 获取信息

*   `**os.getcwd()**` —以字符串形式获取当前工作目录路径— `pwd`
*   `**os.listdir()**` —以字符串列表的形式获取当前工作目录的内容— `ls`
*   `**os.walk("starting_directory_path")**` —返回当前目录和所有子目录中的目录和文件的名称和路径信息的生成器—没有完全等效的简短 CLI，但`ls -R`提供子目录名称和子目录中文件的名称

## 改变事物

*   `**os.chdir("/absolute/or/relative/path")**` —改变当前工作目录— `cd`
*   `**os.path.join()**`—创建路径供以后使用—没有等效的简短 CLI
*   `**os.makedirs("dir1/dir2")**` —制作目录— `mkdir -p`
*   `**shutil.copy2("source_file_path", "destination_directory_path")**` —复制文件或目录— `cp`
*   `**shutil.move("source_file_path*"*, "destination_directory_path")**` —移动文件或目录— `mv`
*   `**os.remove("my_file_path")**` —删除文件— `rm`
*   `**shutil.rmtree("my_directory_path")**` —删除一个目录以及其中的所有文件和目录— `rm -rf`

大家讨论一下。

# 获取信息

`**os.getcwd()**`

`os.getcwd()`以字符串形式返回当前工作目录。😄

`**os.listdir()**`

`os.listdir()`以字符串列表的形式返回当前工作目录的内容。😄

`**os.walk("my_start_directory")**`

`os.walk()`创建一个生成器，可以返回当前目录和子目录的信息。它遍历指定起始目录中的目录。

`os.walk()`为其遍历的每个目录返回以下项目:

1.  字符串形式的当前目录路径
2.  字符串列表形式的当前目录中的子目录名称
3.  字符串列表形式的当前目录中的文件名

它对每个目录都这样做！

使用带有 *for* 循环的`os.walk()`来遍历目录及其子目录的内容通常很有用。例如，下面的代码将打印当前工作目录的目录和子目录中的所有文件。

```
import oscwd = os.getcwd()for dir_path, dir_names, file_names in os.walk(cwd):
    for f in file_names:
        print(f)
```

这就是我们获取信息的方式，现在让我们看看改变工作目录或移动、复制或删除文件系统的命令。

![](img/6283cad0201e0ac8f16df45274466c6e.png)

# 改变事物

`**os.chdir("/absolute/or/relative/path")**`

此方法将当前工作目录更改为提供的绝对或相对路径。

如果您的代码随后对文件系统进行了其他更改，那么最好处理使用 try-except 方法时引发的任何异常。否则，您可能会删除不想删除的目录或文件。😢

`**os.path.join()**`

`[os.path](https://docs.python.org/3/library/os.path.html#module-os.path)`模块有许多常用的路径名操作方法。您可以使用它来查找有关目录名和部分目录名的信息。该模块还有检查文件或目录是否存在的方法。

`os.path.join()`通过将多个字符串连接成一个漂亮的文件路径，创建一个可以在几乎任何操作系统上工作的路径。😄

以下是来自[文档](https://docs.python.org/3/library/os.path.html#module-os.path)的描述:

> 智能地连接一个或多个路径组件。返回值是*路径*和**路径*的任何成员的串联，每个非空部分后面都有一个目录分隔符(`*os.sep*`)，最后一个除外…

基本上，如果您在 Unix 或 macOS 系统上，`os.path.join()`会在您提供的每个字符串之间添加一个正斜杠(“/”)来创建一个路径。如果操作系统需要一个“\”来代替，那么 *join* 知道使用反斜杠。

`os.path.join()`也[向其他开发者提供了清晰的信息](https://stackoverflow.com/a/13944874/4590385)你正在创造一条道路。一定要用它来代替手动的字符串连接，以免看起来像个菜鸟。😉

`**os.makedirs("dir1/dir2")**`

`os.makedirs()`制作目录。`mkdir()`方法也创建目录，但是它不创建中间目录。所以我建议你用`os.makedirs()`。

`**shutil.copy2("source_file", "destination")**`

在 Python 中有很多复制文件和目录的方法。`shutil.copy2()`是一个很好的选择，因为它试图尽可能多地保留源文件的元数据。更多讨论见[这篇文章](https://stackabuse.com/how-to-copy-a-file-in-python/)。

![](img/e9afa54ded667af6edb4c68a05487f1d.png)

Move things

`**shutil.move("source_file*"*, "destination")**`

使用`shutil.move()`改变文件的位置。它使用`copy2`作为引擎盖下的默认。

`**os.remove("my_file_path")**`

有时你需要删除一个文件。`os.remove()`是你的工具。

`**shutil.rmtree("my_directory_path")**`

`shutil.rmtree()`删除目录以及其中的所有文件和目录。

![](img/a24d313849c90dabbc465113fb66ec1e.png)

Remove things

小心删除东西的函数！您可能希望使用 *print()* 打印将要删除的内容。然后在您的 remove 函数中运行 substitute for*print()*当您确定它不会删除错误的文件时。向 Al Sweigart 致敬，他在[中提出了用 Python](https://automatetheboringstuff.com/chapter9/) 自动完成枯燥工作的想法。

这是完整的名单。

# 10 个文件系统方法概述

下面的列表遵循这种模式:**方法—描述—等效的 macOS Shell 命令**

## 获取信息

*   `**os.getcwd()**` —以字符串形式获取当前工作目录路径— `pwd`
*   `**os.listdir()**` —以字符串列表的形式获取当前工作目录的内容— `ls`
*   `**os.walk("starting_directory_path")**` —返回包含当前目录和所有子目录中的目录和文件的名称和路径信息的生成器—没有完全等效的简短 CLI，但`ls -R`提供子目录名称和子目录中文件的名称

## 改变事物

*   `**os.chdir("/absolute/or/relative/path")**` —改变当前工作目录— `cd`
*   `**os.path.join()**`—创建路径供以后使用—没有等效的简短 CLI
*   `**os.makedirs("dir1/dir2")**` —制作目录— `mkdir-p`
*   `**shutil.copy2("source_file_path", "destination_directory_path")**` —复制文件或目录— `cp`
*   `**shutil.move("source_file_path*"*, "destination_directory_path")**` —移动文件或目录— `mv`
*   `**os.remove("my_file_path")**` —删除文件— `rm`
*   `**shutil.rmtree("my_directory_path")**` —删除一个目录以及其中的所有文件和目录— `rm -rf`

# 包装

现在您已经看到了在 Python 中与文件系统交互的基础。在您的 IPython 解释器中尝试这些命令以获得快速反馈。然后向其他人解释它们来巩固你的知识。你会比在你的房子里搬动一箱箱笔记本时更少疼痛。🏠但是锻炼会很好，所以现在你可以去健身房了。🏋️‍♀️😉

如果你想学习用 Python 读写文件，请查看 [*打开*函数](https://docs.python.org/3/library/functions.html#open)。记得像这样使用上下文管理器: `with open(‘myfile’) as file:`。😄

我希望这篇 Python 文件系统操作的介绍对您有用。如果你有，请分享到你最喜欢的社交媒体渠道，这样其他人也可以找到它。

我写关于 Python、Docker、数据科学等等的文章。如果你对此感兴趣，请阅读更多[此处](https://medium.com/@jeffhale)并关注我的 Medium。

[![](img/ba32af1aa267917812a85c401d1f7d29.png)](https://dataawesome.com)

感谢阅读！👏