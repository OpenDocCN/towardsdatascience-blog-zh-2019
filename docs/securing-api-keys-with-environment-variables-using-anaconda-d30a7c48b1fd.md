# 使用 Anaconda 保护带有环境变量的 API 密钥

> 原文：<https://towardsdatascience.com/securing-api-keys-with-environment-variables-using-anaconda-d30a7c48b1fd?source=collection_archive---------10----------------------->

## 如何在公共论坛分享项目时保证私人信息的安全？

![](img/ed308cc421a02d54d62b5961bce55ca2.png)

如果您正在阅读本文，我假设您已经熟悉 API、令牌、私钥和公钥。您的私钥和公钥作为一种身份验证的形式，通过开发人员的 API 来识别您的身份并跟踪您的活动。这意味着，如果有人得到了你的密钥，他们就可以冒充你访问 API 中的数据，这可能会产生一些可怕的后果。

大多数崭露头角的数据科学家面临的挑战是，他们希望在公共论坛上推广他们的项目工作。如果这些项目中的任何一个使用 API，个人将不得不把他们的密钥加载到他们的笔记本中。换句话说，公开笔记本意味着危及 API 密钥的安全。

这个问题有几种常见的解决方案:第一种是用类似于`******`或`YOUR_PRIVATE_KEY_HERE`的东西来替换这个键。这在一定程度上是可行的，但是这种方法容易出现人为错误。对于单个项目的多次提交，每次都必须删除并重新添加令牌。当您按下 commit 并忘记清除密钥时，它就被破坏了。这不是会不会发生的问题，而是什么时候发生的问题。另一种选择是将 API 密钥保存到一个单独的 JSON 文件中，读取私钥，并在推送到 GitHub 时排除该文件。虽然这稍微好一点，但这里的问题是，其他想要探索您的工作的人必须对分叉的存储库进行编辑才能运行代码。此外，这并不能完全消除人为错误的可能性。

如果有一种方法可以将密钥和令牌保存到我们的系统文件中，以便在需要时调用，这不是很好吗？如果其他人可以使用相同的代码从他们自己的系统中调用他们自己的密钥和令牌，那不是更好吗？事实证明，有一个办法:环境变量！

# 什么是环境变量？

环境变量顾名思义:保存在本地环境中的变量，可以在任何项目中调用。因为它们不是存储库本身的一部分，所以没有必要限制推送到 GitHub 的信息。在下面的例子中，一个私有 API 密匙已经被保存为一个名为`PRIVATE_API_KEY`的环境变量。稍后我们将看看如何创建变量本身。

```
import osapi_key = os.environ.get("PRIVATE_API_KEY")
```

正如我们所见，这看起来非常干净简单。每当我们需要使用密钥时，我们所要做的就是调用`api_key`变量。更有经验的数据科学家在审查这段代码时应该会立即意识到需要使用他们自己的私钥，并将它们保存在他们自己的环境中。一旦完成，程序应该可以正常运行，不会出现任何问题。

这里需要注意的是，审查者需要使用相同的变量名保存 API 键。如果这个人已经使用不同的变量名将相关的键保存到了他们的环境中，他们可能会选择修改代码而不是他们的环境。如果 API 开发人员在他们的文档中为这些键提供推荐的命名约定，将会使这个过程更加优雅。但那是另一天的话题。

# 创建环境变量

在这篇文章中，我们不打算讨论跨项目管理环境的最佳实践。相反，我假设您已经在合适的环境中工作，并且知道如何使用 Anaconda 在它们之间切换。如果你不熟悉这个，我推荐你看看这个视频，作为这篇文章的基础。

创建这些变量的总体过程如下:

1.  导航到存储包和详细信息的环境路径。
2.  创建文件和目录，用于在调用时激活变量，并在使用完变量后停用变量。
3.  将变量添加到相关文件中。

我将向您展示如何在 macOS、Linux 和 Windows 中处理所有这三个步骤。所有的指令都直接改编自 Anaconda 文档[。](https://docs.conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#windows)

## macOS 和 Linux

以下命令可用于使用终端处理步骤 1 和 2:

```
cd $CONDA_PREFIX
mkdir -p ./etc/conda/activate.d
mkdir -p ./etc/conda/deactivate.d
touch ./etc/conda/activate.d/env_vars.sh
touch ./etc/conda/deactivate.d/env_vars.sh
```

因此，我们现在有两个名为`env_vars.sh`的文件，它们分别存储在不同的目录中。我们需要在您选择的文本编辑器中打开这些文件来创建变量。对于`activate.d`目录中的文件，我们将添加以下内容:

```
*#!/bin/sh*

export PRIVATE_API_KEY='YOUR_PRIVATE_KEY_HERE'
```

对于 deactivate.d 目录中的文件，我们添加以下内容:

```
*#!/bin/sh*

unset MY_KEY
```

第一行`#!/bin/sh`让操作系统知道这些文件是 shell 脚本，以确保`export`和`unset`命令被正确执行。

一旦这些文件被保存并关闭，在终端输入`conda activate analytics`将`YOUR_PRIATE_KEY_HERE`保存到环境变量`PRIVATE_API_KEY`。使用完这些变量后，可以使用`conda deactivate`将它们从环境中删除。

## Windows 操作系统

使用命令 shell，可以使用以下命令来处理步骤 1 和 2:

```
cd %CONDA_PREFIX%
mkdir .\etc\conda\activate.d
mkdir .\etc\conda\deactivate.d
type NUL > .\etc\conda\activate.d\env_vars.bat
type NUL > .\etc\conda\deactivate.d\env_vars.bat
```

因此，我们现在有两个名为`env_vars.bat`的文件，它们分别存储在不同的目录中。我们需要在您选择的文本编辑器中打开这些文件来创建变量。对于`activate.d`目录中的文件，我们将添加以下内容:

```
set PRIVATE_API_KEY='YOUR_PRIVATE_KEY_HERE'
```

对于 deactivate.d 目录中的文件，我们添加以下内容:

```
unset PRIVATE_API_KEY=
```

一旦保存并关闭这些文件，在终端中输入`conda activate analytics`将`YOUR_PRIATE_KEY_HERE`保存到环境变量`PRIVATE_API_KEY`。使用完这些变量后，可以使用`conda deactivate`将它们从环境中删除。

# 结论

正如您所看到的，虽然将 API 键保存为环境变量需要几个步骤，但这本身并不困难或复杂。一旦就位，这些变量就可以快速而容易地访问，并且当您使用完它们时，只需一个命令就可以停用它们。这比每次你想对 GitHub 进行更新或提交新的提交时添加和删除键要有效得多。在向外界展示您的项目时，它会为您提供更多安全保障。