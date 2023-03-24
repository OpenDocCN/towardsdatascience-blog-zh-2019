# 了解 Python 虚拟环境

> 原文：<https://towardsdatascience.com/understanding-python-virtual-environments-7ccf690320d7?source=collection_archive---------10----------------------->

## 使用 VR 的 Python 虚拟环境介绍

![](img/cfab0f735188ba7efa211b72c5b9ccc5.png)

A man is using three screens at once. Photo by [Maxim Dužij](https://unsplash.com/photos/qAjJk-un3BI?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/python-programming?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

不，看这篇文章不需要 VR 眼镜。只要一袋注意力和兴奋就够了。

如果您是数据科学和 python 领域的新手，虚拟环境可能看起来是一个非常复杂的想法——但事实恰恰相反。它简单易懂，使用起来更简单！如果你以前体验过虚拟现实(VR)，你也会有一个良好的开端(例如，玩虚拟现实游戏或出于任何目的尝试虚拟现实眼镜)。这篇文章涵盖了从什么是环境到设置和使用虚拟环境的所有内容！

**什么是环境？**

人类的环境可以指他们的周围环境——他们居住的地方。对于像 python 这样的编程语言来说，环境的概念是类似的。对于安装在计算机上的 python，计算机就是它的环境。它通常被称为“本地”环境，因为语言也可以在服务器上运行(服务器只是在数据中心运行的计算机，可以通过互联网访问)。

**什么是虚拟环境？**

你体验过虚拟现实或者看到有人体验过吗？如果没有，这里有一个例子:

![](img/6580d763ddf5b93cfebe58914bab0124.png)

Photo by [Hammer & Tusk](https://unsplash.com/photos/3kB63Vz7xVg?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/virtual?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

回想一下人类对环境的定义——他们的周围环境。对于上面照片中的人来说，他们的周围就是他通过镜头看到的东西。对他来说，现实的概念(在一定程度上)已经从他用肉眼看到的变成了他现在感知到的。

python 的虚拟环境是一个类似的想法:你在一个环境(你的计算机)中给它一个单独的“环境”,并允许它在那里运行。

然而，人类的虚拟现实体验和 python 的虚拟现实体验之间有一些关键的区别。首先，你可以为 python 创建多个虚拟环境(你可以认为 python 有无限多的脸，而人类只有一张脸，所以 python 可以戴任意多的 VR 眼镜)。其次，虽然人类仍然可以触摸和感受“真实”环境中的物体，但 python 不一定能做到这一点。当你戴上 python 的 VR 眼镜时，它可能会也可能不会访问“真实”的环境(取决于你的设置，对于这篇文章来说有点太高级了)。因此，在虚拟环境中，安装在您的计算机上的所有库和包(也称为真实环境)都是不可访问的，反之亦然。

**我为什么要关心？**

好吧，这听起来很酷，但是虚拟环境的目的是什么？为什么要把酷炫的 VR 眼镜交给枯燥复杂的编程语言？

**方便。稳定。安心。**不是为了编程语言，而是为了作为程序员的你。注意不同的项目可能需要不同版本的库，甚至可能需要不同的语言？这就是虚拟环境有用的地方。这个想法是为每个项目创建一个单独的虚拟环境。

因为一种语言的资源(库、包等)对真实环境和其他虚拟环境是隐藏的，所以版本之间没有干扰和冲突。例如，假设您在项目 A 的虚拟环境中更新了库 X。因为您在项目 A 的虚拟环境中更新了库 X，所以您可以确保这不会导致库 X 在任何其他环境中更新。因此，您的代码及其在其他地方的依赖项不受影响。

**好的，我如何制作一个虚拟环境？**

对于这个故事，我将介绍 [virtualenv](https://docs.python.org/3/library/venv.html) 库，但是有许多不同的(也许更好的)方法来创建 python 虚拟环境。我还将使用 Mac，这是一个基于 UNIX 的系统。因此，如果您的计算机使用不同的系统，步骤可能会有所不同。

您的系统上可能没有安装 virtualenv。您可以使用以下命令进行检查:

```
virtualenv --version
```

如果您安装了 virtualenv,“终端”将打印您系统上安装的 virtual env 版本(如 16.4.3)。否则，它会给你一个“命令未找到”的错误。如果是这种情况，请输入以下命令:

```
pip install virtualenv
```

PS:如果你没有 [pip](https://pypi.org/project/pip/) ，可以为你的系统下载。默认情况下，它包含在 Python 3.4 或更高版本的二进制安装程序中。

接下来，导航到您想要创建虚拟环境的目录。您可以只为环境创建另一个目录，并使用以下命令切换到该目录:

```
mkdir Environments && cd Environments
```

这将创建一个名为“环境”的新文件夹，并在终端中打开该目录。您现在已经准备好为 python 创建 VR 体验了。要创建环境，请使用以下命令

```
virtualenv project_name 
```

上面的命令在当前工作目录中创建了一个名为“project_name”的环境。将“项目名称”替换为您的项目名称。完成上述步骤后，您的终端窗口应该看起来像这样:

![](img/1da7be00c598cf5c9977382020d94d2a.png)

Terminal screen after creating the virtual environment.

但是，虚拟环境尚未激活。每当您想要进入环境或激活它(也就是戴上眼镜)时，您需要运行以下命令:

```
source project_name/bin/activate
```

Tada！你现在在你的虚拟环境中。你如何确认你确实在一个虚拟环境中？您的终端将在所有行中以 project_name 为前缀，如下所示:

![](img/f6239def89578f06d843820af3a1a551.png)

Terminal screen after activating python virtual environment

就是这样！您已经成功学习、创建并激活了一个 python 虚拟环境。现在，您可以在这个虚拟环境中处理您的项目。您可以像在普通终端上一样安装库，只是这些库将安装在虚拟环境中，而不是“真实”环境中。您可以使用以下命令退出虚拟环境，以“注销”虚拟环境:

```
deactivate
```

**进一步学习的资源:**

你还没有完全发现虚拟环境的世界。您可能想探索许多有趣的事情:

这个故事比较了不同的 python 虚拟环境工具(回想一下，我只展示了 virtualenv):

[](/comparing-python-virtual-environment-tools-9a6543643a44) [## 比较 Python 虚拟环境工具

### 设置 Python 虚拟环境的各种工具

towardsdatascience.com](/comparing-python-virtual-environment-tools-9a6543643a44) 

这是一篇非常适合虚拟环境专家的文章:

[](https://medium.freecodecamp.org/manage-multiple-python-versions-and-virtual-environments-venv-pyenv-pyvenv-a29fb00c296f) [## 如何管理多个 Python 版本和虚拟环境🐍

### 2019 年 1 月补充:如果你在升级到 macOS Mojave 后回到这个博客，请查看这个 github 问题…

medium.freecodecamp.org](https://medium.freecodecamp.org/manage-multiple-python-versions-and-virtual-environments-venv-pyenv-pyvenv-a29fb00c296f) 

以下文章讨论了与虚拟环境相关的最新发展:

 [## 再见虚拟环境？

### 如果您是 Python 开发人员，您可能听说过虚拟环境—“一个自包含的目录树…

medium.com](https://medium.com/@grassfedcode/goodbye-virtual-environments-b9f8115bc2b6) 

我希望这篇文章能帮助您理解什么是虚拟环境，并让您开始使用它们。如果您有任何问题，请不要犹豫回应这个故事或联系我。祝 Python 虚拟现实愉快！