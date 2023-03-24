# Python 虚拟环境指南

> 原文：<https://towardsdatascience.com/virtual-environments-104c62d48c54?source=collection_archive---------1----------------------->

## 它们是什么，如何使用它们，以及它们实际上是如何工作的。

![](img/a17ca7c8fb9f2d44d428e9f321cb311b.png)

The Gate of Dependency Hell: “Abandon all hope, ye who enter here.” [Illustration](https://commons.wikimedia.org/wiki/File:Gustave_Doré_-_Dante_Alighieri_-_Inferno_-_Plate_8_(Canto_III_-_Abandon_all_hope_ye_who_enter_here).jpg) by [Gustave Doré](https://en.wikipedia.org/wiki/Gustave_Doré).

Python 的虚拟环境让生活变得更加轻松。*轻松很多*。

☄在本指南中，我们将涵盖虚拟环境的基础知识以及如何使用它们。然后，我们将深入了解虚拟环境实际上是如何工作的。

**⚠️注意**:在本指南中，我们将在 macOS Mojave 上使用最新版本的 Python 3.7.x。

## **目录**

[为什么要使用虚拟环境？](https://medium.com/p/104c62d48c54#ee81)
[什么是 Virtualenv？！](https://medium.com/p/104c62d48c54#e923)
[使用虚拟环境](https://medium.com/p/104c62d48c54#8025)
[管理环境](https://medium.com/p/104c62d48c54#15be)
[虚拟环境如何工作](https://medium.com/p/104c62d48c54#1839)
[延伸阅读](https://medium.com/p/104c62d48c54#9762)

# 为什么要使用虚拟环境？

虚拟环境为大量潜在问题提供了一个简单的解决方案。特别是，它们可以帮助您:

*   通过允许你为不同的项目使用不同版本的包来解决依赖问题。例如，您可以对项目 X 使用包 A v2.7，对项目 y 使用包 A v1.3。
*   通过在一个需求文件中捕获所有的包依赖，使你的项目**自包含**和**可重复**。
*   在您没有管理员权限的主机上安装软件包。
*   通过消除在系统范围内安装包的需要来保持你的全局目录整洁，你可能只需要一个项目。

听起来很方便，不是吗？当你开始构建更复杂的项目并与其他人合作时，你会发现虚拟环境是多么重要。如果你像我一样是一名数据科学家，你也会想熟悉他们的多语言表亲 [Conda environments](https://medium.com/@msarmi9/a-guide-to-conda-environments-bc6180fc533) 。

但是首先要做的是。

# 什么是虚拟？！

到底是什么*虚拟环境？*

虚拟环境是用于**依赖管理**和**项目** **隔离**的 Python 工具。它们允许 Python **站点包**(第三方库)被本地安装在特定项目的一个隔离的目录中，而不是被全局安装(即作为系统范围 Python 的一部分)。

太好了。这听起来不错，但是什么是虚拟环境呢？嗯，虚拟环境就是一个包含三个重要组件的目录:

*   安装第三方库的`site-packages/`文件夹。
*   [符号链接](https://en.wikipedia.org/wiki/Symbolic_link)到系统上安装的 Python 可执行文件。
*   [脚本](https://en.wikipedia.org/wiki/Shell_script)确保执行的 Python 代码使用安装在给定虚拟环境中的 Python 解释器和站点包。

最后一点是所有的 s***下降的地方。稍后我们将更深入地了解一下，但是现在让我们看看我们实际上是如何使用虚拟环境的。

![](img/2629a039ecd97f44c8744f928942486b.png)

Virgil appeases Cerberus — Canto VI. [Illustration](https://commons.wikimedia.org/wiki/File:Inferno_Canto_6_lines_24-26.jpg) by Gustave Doré.

# 使用虚拟环境

## 创造环境

假设我们想要为一个名为`test-project/`的项目创建一个虚拟环境，它有如下的目录树。

```
test-project/
├── data        
├── deliver           # Final analysis, code, & presentations
├── develop           # Notebooks for exploratory analysis
├── src               # Scripts & local project modules
└── tests
```

我们需要做的就是执行`[venv](https://docs.python.org/3/library/venv.html)`模块，它是 Python 标准库的一部分。

```
% cd test-project/
% python3 -m venv venv/       # Creates an environment called venv/
```

**⚠️注:**您可以根据您的环境用不同的名称替换“venv/”。

瞧啊。一个虚拟的环境诞生了。现在我们的项目看起来像这样:

```
test-project/
├── data        
├── deliver      
├── develop      
├── src      
├── tests    
└── venv                 # There it is!
```

♻️提醒:虚拟环境本身就是一个目录。

剩下唯一要做的就是通过运行我们前面提到的脚本来“激活”我们的环境。

```
% source venv/bin/activate             
(venv) %                               # Fancy new command prompt
```

我们现在在一个活动的虚拟环境中(由以活动环境的名称为前缀的命令提示符指示)。

在这一点上，我们将像往常一样工作在我们的项目上，安全地知道我们的项目与我们系统的其余部分是完全隔离的。在我们的环境内部，我们无法访问系统范围的站点包，并且我们安装的任何包在我们的环境外部都是不可访问的。

当我们完成我们的项目时，我们可以用

```
(venv) % deactivate
%                                    # Old familiar command prompt
```

## 安装软件包

默认情况下，只有`pip`和`setuptools`安装在新环境中。

```
(venv) % pip list                    # Inside an active environmentPackage    Version
---------- -------
pip        19.1.1
setuptools 40.8.0
```

如果我们想安装一个特定版本的第三方库，比如说`numpy`的 v1.15.3，我们可以照常使用`pip`即可。

```
(venv) % pip install numpy==1.15.3
(venv) % pip listPackage    Version
---------- -------
numpy      1.15.3
pip        19.1.1
setuptools 40.8.0
```

现在我们可以在脚本或活动 Python shell 中导入`numpy`。例如，假设我们的项目包含一个脚本`tests/imports-test.py`，其代码如下。

```
#!/usr/bin/env python3          

import numpy as np
```

当我们直接从命令行运行这个脚本时，我们会得到:

```
(venv) % tests/imports-test.py           
(venv) %                                 # Look, Ma, no errors!
```

成功。我们的剧本顺利地引进了🥳.的`numpy`

![](img/c00913ea258fb9f05972cb8ad5cebdb2.png)

Dante and Virgil cross the river Styx — Canto VIII. [Illustration](https://commons.wikimedia.org/wiki/File:Inferno_Canto_8_verses_27-29.jpg) by Gustave Doré.

# 管理环境

## 需求文件

使我们的工作可以被其他人复制的最简单的方法是在我们项目的**根目录**(顶层目录)中包含一个需求文件。为此，我们将运行`pip freeze`，它列出已安装的第三方包及其版本号，

```
(venv) % pip freeze
numpy==1.15.3 
```

并将输出写入一个文件，我们称之为`requirements.txt`。

```
(venv) % pip freeze > requirements.txt
```

每当我们更新一个包或者安装一个新的包时，我们可以使用这个命令来重写我们的需求文件。

现在，我们共享项目的任何人都可以通过使用我们的`requirements.txt`文件复制我们的环境，在他们的系统上运行我们的项目。

## 复制环境

等等——我们到底该怎么做？

想象一下，我们的队友 Sara 从我们团队的 GitHub 库中取出了我们的 T8。在她的系统上，项目的目录树如下所示:

```
test-project/
├── data        
├── deliver      
├── develop      
├── requirements.txt
├── src    
└── tests
```

注意到任何轻微的不寻常的事情吗？是的，没错。没有`venv/`文件夹。我们已经把它从我们团队的 GitHub 库中排除了，因为包含它[会引起麻烦](https://stackoverflow.com/questions/6590688/is-it-bad-to-have-my-virtualenv-directory-inside-my-git-repository)。

这就是为什么拥有一个`requirements.txt`文件对于复制你的项目代码来说是*必要的*。

为了在她的机器上运行我们的`test-project/`, Sara 需要做的就是在项目的根目录中创建一个虚拟环境

```
Sara% cd test-project/
Sara% python3 -m venv venv/
```

并使用咒语`pip install -r requirements.txt`在一个活动的虚拟环境中安装项目的依赖项。

```
Sara% source venv/bin/activate
(venv) Sara% pip install -r requirements.txtCollecting numpy==1.15.3 (from -r i (line 1))
Installing collected packages: numpy
Successfully installed numpy-1.15.3                  # Woohoo! 🎊
```

现在，Sara 系统上的项目环境与我们系统上的环境完全相同。很整洁，是吧？

## 解决纷争

可悲的是，事情并不总是按照计划进行。最终你会遇到问题。也许你已经错误地更新了一个特定的站点包，现在发现自己处于[依赖地狱](https://en.wikipedia.org/wiki/Dependency_hell)的第九层，无法运行你项目的一行代码。话说回来，也许也没那么糟糕。也许你只发现自己在[第七关](https://en.wikipedia.org/wiki/Inferno_(Dante)#Seventh_Circle_(Violence))。

无论你发现自己处于什么水平，逃离火焰并再次看到阳光的最简单方法是**重新创建**你项目的虚拟环境。

```
% rm -r venv/                           # Nukes the old environment
% python3 -m venv venv/                 # Makes a blank new one
% pip install -r requirements.txt       # Re-installs dependencies
```

就是这样。多亏了你的`requirements.txt`档案，你才得以重操旧业。总是在项目中包含需求文件的另一个原因。

![](img/683abaf3b57836d1501a5dd672cec041.png)

Dante speaks with the traitors in the ice — Canto XXXII. [Illustration](https://commons.wikimedia.org/wiki/File:Gustave_Dore_Inferno32.jpg) by Gustave Doré.

# 虚拟环境是如何工作的

你想知道更多关于虚拟环境的事情，是吗？比如活动环境*如何知道*如何使用正确的 Python 解释器以及如何找到正确的第三方库。

## echo $PATH

这一切都归结于[路径](https://en.wikipedia.org/wiki/PATH_(variable))的值，它告诉您的 shell 使用哪个 Python 实例以及在哪里寻找站点包。在您的 base shell 中，路径看起来或多或少会像这样。

```
% echo $PATH
/usr/local/bin:/usr/bin:/usr/sbin:/bin:/sbin
```

当您调用 Python 解释器或运行`.py`脚本时，您的 shell 会按照的顺序搜索路径**中列出的目录，直到遇到 Python 实例。要查看哪个 Python 实例路径先找到，运行`which python3`。**

```
% which python3
/usr/local/bin/python3                 # Your output may differ
```

也很容易看出这个 Python 实例在哪里寻找带有`[site](https://docs.python.org/3/library/site.html#site.getsitepackages)`模块的站点包，该模块是 Python 标准库的一部分。

```
% python3                           # Activates a Python shell
>>> import site                      
>>> site.getsitepackages()          # Points to site-packages folder['/usr/local/Cellar/python/3.7.3/Frameworks/Python.framework/Versions/3.7/lib/python3.7/site-packages']
```

运行脚本`venv/bin/activate`会修改 PATH，这样我们的 shell 会在搜索系统的全局二进制文件之前搜索项目的本地二进制文件*。*

```
% cd ~/test-project/
% source venv/bin/activate
(venv) % echo $PATH~/test-project/venv/bin:/usr/local/bin:/usr/bin:/usr/sbin:/bin:/sbin
```

现在我们的外壳知道使用我们项目的本地 Python 实例

```
(venv) % which python3
~/test-project/venv/bin/python3
```

以及在哪里可以找到我们项目的本地站点包。

```
(venv) % python3
>>> import site
>>> site.getsitepackages()['~/test-project/venv/lib/python3.7/site-packages']    # Ka-ching 🤑
```

## 理智检查

还记得我们之前的`tests/imports-test.py`剧本吗？它看起来像这样。

```
#!/usr/bin/env python3          

import numpy as np
```

我们能够毫无问题地从我们的活动环境中运行这个脚本，因为我们环境的 Python 实例能够访问我们项目的本地站点包。

如果我们在项目的虚拟环境之外运行来自*的相同脚本会发生什么？*

```
% tests/imports-test.py                # Look, no active environmentTraceback (most recent call last):
  File "tests/imports-test.py", line 3, in <module>
    import numpy as npModuleNotFoundError: No module named 'numpy' 
```

是的，我们得到了一个错误— **正如我们应该得到的**。如果我们不这样做，这将意味着我们能够从项目外部访问项目的本地站点包，破坏了拥有虚拟环境的整个目的。我们得到一个错误的事实证明我们的项目**完全** **与我们系统的其余部分**隔离。

## 环境的目录树

有一件事可以帮助我在头脑中组织所有这些信息，那就是对环境的目录树有一个清晰的了解。

```
test-project/venv/               # Our environment's root directory
├── bin
│   ├── activate                           # Scripts to activate
│   ├── activate.csh                       # our project's
│   ├── activate.fish                      # virtual environment.
│   ├── easy_install
│   ├── easy_install-3.7
│   ├── pip
│   ├── pip3
│   ├── pip3.7
│   ├── python -> /usr/local/bin/python    # Symlinks to system-wide
│   └── python3 -> python3.7               # Python instances.
├── include
├── lib
│   └── python3.7
│       └── site-packages              # Stores local site packages
└── pyvenv.cfg
```

![](img/991d47d066c70151198ec323a77b4ef7.png)

Dante and Virgil return to the mortal realm — Canto XXXIV. [Illustration](https://commons.wikimedia.org/wiki/File:Inferno_Canto_34_verses_127-129.jpg) by Gustave Doré.

# 进一步阅读

如果你的好奇心还没有得到满足，你还想了解更多关于虚拟环境的知识，我强烈推荐 Real Python 的关于虚拟环境的极好的[入门](https://realpython.com/python-virtual-environments-a-primer/)。如果你发现自己沉迷于古斯塔夫·多雷的出色插图，我强烈推荐你阅读但丁的 [*地狱*](https://en.wikipedia.org/wiki/Inferno_(Dante)) *。*

除此之外，我们差不多做到了。如果你想了解我最新的数据科学帖子，请随时在 twitter 上关注我。

干杯，祝阅读愉快。