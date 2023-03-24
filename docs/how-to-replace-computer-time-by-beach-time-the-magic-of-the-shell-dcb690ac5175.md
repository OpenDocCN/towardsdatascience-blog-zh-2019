# 如何用海滩时间取代电脑时间——贝壳的魔力

> 原文：<https://towardsdatascience.com/how-to-replace-computer-time-by-beach-time-the-magic-of-the-shell-dcb690ac5175?source=collection_archive---------34----------------------->

![](img/dcba495a819128a9fe51fc77239db5dc.png)

Beach in San Sebastian — Photo by [Athena Lam on Unsplash](https://unsplash.com/@thecupandtheroad?source=post_page---------------------------)

我编程越多，我就越懒。我只是不明白，为什么我应该做一些电脑自己能做得更好、更快、更可靠的事情。在我接近懒惰工作者的过程中，我发现 shell 脚本是一个很好的朋友和助手。虽然在过去，您可以经常使用 Windows(如果您愿意的话)，但在越来越多的计算在云中执行，服务器大多基于 Linux 的时代，在我看来，每个数据科学家和数据工程师都应该至少对 shell 脚本有一个基本的了解。(奖金:*看到我最喜欢的时间，当 shell 脚本实际上使我高兴的时候，在底部的附录。提示:它确实包含了大量的海滩时间。因为我相信，在内心深处，我们都想把时间花在做其他事情上，而不是手动移动文件，所以我想与您分享我对 shell 的个人基本知识的介绍。希望它能让你的生活变得轻松一点(如果你愿意的话，还能让你在海滩上多待些时间:)。

# 基础:移动和基本的文件操作

# 四处走动

如果你登录了，你可能想知道的第一件事是你在哪里(提示:你可能在你的主目录中)。您可以通过将当前工作目录打印到屏幕上来找到答案:

```
# print working directory
pwd
```

接下来，您应该通过键入以下命令列出它的内容

```
# list contents of current directory
ls
```

许多 bash 命令允许修饰符，即所谓的标志。它们主要由一个字母组成，在命令后面加上一个“-”。您可以通过一个接一个地写入多个标志来组合它们。有许多可能的标志。这里有一些例子

```
# include hidden files
ls -a# include hidden files and print more details 
ls -la# list only directory (without content)
ls -1d directoryname
```

要了解有关命令的更多信息，请使用手册(手册页):

```
# print manual for mycommand
man mycommand# for example:
man ls
```

为了四处移动，您使用`cd`(改变目录)命令:

```
# change to directory called mydirectory inside current directory
cd mydirectory# change to directory above
cd ..# move to directory which is also inside the directory above (basically a "parallel" directory)
cd ../mydirectory# change into previous work directory
cd -
```

# 高级移动

您可以使用`pushd` / `popd`在堆栈中添加/删除目录。一旦添加到堆栈中，您就可以在堆栈中的目录之间跳转。请注意，在构建您的堆栈时，您需要添加两次最终目录，因为最终位置总是会被覆盖(听起来比实际复杂，只要尝试一下，您就会明白我的意思)。

```
# add mydirectory to stack
pushd mydirectory# show directories in stack
dirs -v # delete top repository from stack
popd # change to directory numbered n (eg 2) in the stack
cd ~2
```

# 与文件和文件夹的基本交互

您可以通过以下方式创建一个简单的文本文件

```
# create a text file called mynewtextfile.txt
touch mynewtextfile.txt
```

文件由以下人员复制、移动或删除:

```
# copy file
cp oldfilename newfilename# move/rename file
mv oldfilename newfilename# delete file
rm oldfilename
```

为了创建(制作)新目录:

```
mkdir mynewdirectory
```

目录像文件一样被复制、移动和删除。然而，复制和删除需要`-r`(递归)标志:

```
# copy directory
cp -r folder_old folder_new# delete directory
rm –r folder_to_remove# rename directory (does not require -r flag)mv old_folder new_folder
```

# 与文件交互和将命令链接在一起——稍微不太基本

# 与文本文件交互

既然我们知道了如何移动文件，我们还想用它们做一些有用的事情。

有四个主要选项可以访问文本文件的内容。我建议尝试一下，看看它们能做什么，它们的行为有何不同。

```
# prints whole file to screen
cat mytextfile# prints file to screen one screenful at a time
more mytextfile# prints file to screen, allowing for backwards movement and returns to previous screen view after finishing
# note: less does not require the whole text to be read and therefore will start faster on large text files then more or text-editors
less mytextfile# use a text editor (for example nano or here vi)
vi mytextfile
```

关于编辑的选择:我个人是 Vim 的忠实粉丝。然而，我承认一开始它确实有一个陡峭的学习曲线。如果你想从对初学者更友好的东西开始，你可以看看 nano。然而，为了将来，请记住 VIM，一旦您熟悉自己的工作方式，文本处理的速度将会惊人。

还可以返回文档的前 n 行或后 n 行

```
# show first 10 rows of a document
head -10 mytextfile# show last 10 rows of a document
tail -10 mytextfile
```

使用`grep`在文档中查找文本

```
# look for the string python in mytextfile
grep python mytextfile# search case-insensitive
grep -i python mytextfile# return line-numbers with the results
grep -n python mytextfile# search for a filename ("mybadfilename" in the example) (case insensitive) in all files with the ending *.py and return the occurences together with the line number
grep -in mybadfilename *.py
```

在上一个例子中，我们已经看到了一个占位符的例子。*.py 表示所有以. py 结尾的文件。

# 重定向输出

一些命令打印到屏幕上。为了将输出重定向到一个文件，我们可以使用`>`和`>>`。`>>`将输出附加到现有文件，或者如果文件尚不存在，则创建一个新文件。相反，`>`总是创建一个新文件。如果同名文件已经存在，它将覆盖该文件。以下是如何将`grep -in mybadfilename *.py`命令的输出重定向到文件的示例:

```
# creates new file; if file exists, overwrites it
mycommand > mytextfile
# example:
grep -in mybadfilename *.py > myoutputfile# appends output to file; if myoutputfile does not exist yet, creates it
mycommand >> mytextfile
# exammple:
grep -in mybadfilename *.py >> myoutputfile
```

如果除了将输出重定向到文件，我们**还**想要将输出打印到屏幕上，我们可以使用`| tee`。注意，完整的命令需要出现在`|`之前。

```
# print output to screen plus re-direct it to file
mycommand | tee myoutputfile# example:
grep -in mybadfilename *.py | tee myoutputfile
```

在前面的示例中，我们已经看到了管道(|)命令的用法。它是如何工作的？`|`将输出重定向到通常“从右边”获取输入的函数，因此输入应该在函数调用之后。一个例子:如前所述，`grep`需要语法`grep sth filename`。但是，您可能有一个返回输出的程序，并且想要在这个输出中寻找一些东西。这就是`|`发挥作用的地方。例如，`ps aux`显示系统上运行的所有进程。您可能希望搜索包含特定字符串的进程，例如 launch_。你应该这样做:

```
# grep for the string launch_ in the output of ps aux
ps aux | grep launch_
```

# 变量和脚本

# 变量

Bash 是一种脚本语言，不是类型化的。使用`=`符号定义和分配变量。变量名、`=`符号和值之间不能有任何空格。您可以使用`$`后跟变量名来访问变量的内容。您可以使用`echo`打印到屏幕上。

```
# define string variable
my_string_variable="this_is_a_string"# define numeric variable
my_numeric_variable=3# print variable to screen
# (will print this_is_a_string to the screen)
echo $my_string_variable
```

变量通常用于定义路径和文件名。当变量在文本中被重新求解时，需要在变量名周围加上`{}`。例如，考虑刚刚创建的变量 my_string_variable。假设你想打印' this_is_a_string_1 '。为了打印变量 my_string_variable 的内容，后跟 _1，请在变量名两边使用{}:

```
# incorrect (bash will think that the variable is called "my_string_variable_1"):
echo $my_string_variable_1# instead use:
echo ${my_string_variable}_1
```

在第二个例子中，bash 解析对 this_is_a_string 的引用，然后将 _1 追加到结果字符串中。

# 环

Bash 使用`for ... do ... done`语法进行循环。该示例显示了如何使用循环将文件 myfilename1 和 myfilename2 重命名为 myfilename1.bac 和 myfilename2.bac。请注意，列表元素之间没有逗号分隔。

```
rename files by appending a .bac to every filename
# no comma between list elements!
for myfilename in myfilename1 myfilename2
do
  mv $filename ${filename}.bac;
done
```

为了遍历整数列表，首先使用序列生成器生成一个列表:

```
for i in $(seq 1 3)
do
 echo $i
done
```

注意:`$()`打开一个子 shell，在这里解析()的内容。然后将结果返回到外壳。在上面的例子中，`seq 1 3`产生的序列 1 2 3 被传递回外壳，然后在外壳中循环。例如，这种行为可用于循环包含特定模式的文件:

```
for myfile in $(ls *somepattern*)
do
  cp myfile myfile.bac
done
```

# 编写和执行(非常)基本的脚本

要创建脚本，创建一个包含 bash 语法的文本文件，使其可执行并运行它。让我们看一个非常基本的(公认非常无用的)例子。创建包含以下内容的文件:

```
#!/bin/bash# print myfilename.txt
echo "Hello World!"exit 0
```

并将其保存为 print_hello_world.sh。注意文件的第一行，它告诉 shell 使用哪个解释器。您可以通过为所有者添加执行权限来使其可执行，并由。/scriptname:

```
# add execution rights for file myfirstbashscript.sh for the owner of the file
chmod u+x print_hello_world.sh# run 
./print_hello_world.sh
```

如果不是硬编码“Hello World！”，您希望用户将待问候的传递给脚本，您可以将它作为变量传递给脚本。让我们用以下内容创建一个新文件 print_hello_user.sh:

```
#!/bin/bash# print "Hello " + user-input 
echo "Hello " $1exit 0
```

如果我们给它执行权，像这样执行

```
./print_hello_user.sh "Universe"
```

它会将“Hello Universe”打印到屏幕上。为什么？“Universe”作为文件名作为名为 1 的变量传递给脚本后的第一个输入变量，然后通过 print 语句中的$1 命令引用它。

# 最后的提示和技巧

*   尽可能使用 tab 补全:要自动补全，请按“Tab”键。如果有多个选项，按“Tab”两次以显示所有选项。
*   `ESC + .`将从上一行带回最后一个令牌。例子:`cp file_a file_b`；然后在下一行`ESC + .`会产生 file_b。
*   括号结束:你可以使用`{}`来缩短你的代码。例如，如果你想重命名一个文件，你可以输入`mv myfilename{,.bac}`。这作为`mv myfilename myfilename.bac`执行。对于交互式工作非常有用(虽然我不会在脚本中使用它)。
*   `tail -f myfilename` : `tail filename`在执行点产生尾部。但是，您可能希望能够在编写输出脚本时跟踪它们。`tail -f`照常开始`tail`，但是当新的行出现在输出文件的末尾时，继续追加。
*   `watch -n somenumber command`每隔几秒执行一次命令。例如，`watch -n 2 ls`每 2 秒运行一次 ls。观看文件传输非常棒。

# 结论

在这篇文章中，我们看了使用 shell 的基本介绍。我们已经看到了如何在 shell 环境中定位自己，如何四处移动以及一些与文件的基本交互。最后，我们已经创建并运行了我们的第一个脚本，并查看了一些我最喜欢的技巧。虽然这应该会给您一个好的开始，但这只是对 shell 脚本这个怪异而奇妙的世界的一个小小的介绍。如果你想了解更多，这里的是一个很好的、广泛的脚本备忘单，可能会对你有进一步的帮助。有关该主题的完整内容，请查看 Mendel Cooper 的[对 shell 脚本艺术的深入探索](http://tldp.org/LDP/abs/html/index.html?source=post_page---------------------------)。和往常一样， [StackOverflow](https://stackoverflow.com/?source=post_page---------------------------) 也提供了大量的建议和帮助:)玩得开心！

# *附录:Shell 脚本如何让我花更多时间在沙滩上

我在圣塞巴斯蒂安攻读博士学位，圣塞巴斯蒂安是巴斯克地区的首府，也是著名的“La Concha”海滩的所在地。我的论文非常注重计算，需要协调许多不同的技术。我仍然天真地记得设置我的计算机来自动生成大量用于计算的输入文件，将计算提交给超级计算中心，等待它们完成，从输出中提取相关数据，可视化结果，创建一个完整的网页层次结构，并将所有这些推送到网络服务器，这样来自世界各地的多人就可以协作查看结果。只需按一下按钮，它就能完全自动地完成所有这些工作，而且做得非常可靠，从未出错。而我呢？我正在海滩享受午餐:)

*原载于*[*https://walken ho . github . io*](https://walkenho.github.io/introduction-to-bash/?source=post_page---------------------------)*。*