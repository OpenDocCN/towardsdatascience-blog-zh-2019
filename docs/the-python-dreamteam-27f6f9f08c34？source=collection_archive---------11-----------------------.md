# Python 包 Dreamteam

> 原文：<https://towardsdatascience.com/the-python-dreamteam-27f6f9f08c34?source=collection_archive---------11----------------------->

作为一名数据科学家，我几乎完全用 Python 编码。我也很容易被配置的东西吓到。我真的不知道什么是`PATH`。我不知道我的笔记本电脑上的`/bin`目录里有什么。这些似乎都是你必须熟悉的，以免当你试图改变任何东西时 Python 在你的系统上崩溃。经过几年的努力，我偶然发现了 pipenv/pyenv 组合，它似乎以一种对我来说最有意义的方式，在很大程度上解决了我的 Python 设置难题。

直到最近，我还在用自制的安装 Python 3 和`venv`作为我的 Python 依赖辩论者。大多数情况下，`venv`对我来说非常有用。从 3.3 开始，它就包含在 Python 中了，所以它感觉像是 Python 生态系统中的一等公民。存储一个充满虚拟环境的目录，并在我想要创建或激活虚拟环境的任何时候输入完整的路径，这感觉很奇怪，但似乎工作得很好。在用 Python 3.7.0 安装 TensorFlow 遇到问题后，我决定寻找一个替代方案。

![](img/1c01103c4163401d5086f0e30da34b83.png)

[https://xkcd.com/1987/](https://xkcd.com/1987/)

在某些情况下，我的项目工作流程是从克隆一个标准的研究存储库开始的，使用一个叫做 Cookiecutter 的工具。这个标准回购有一个方便的默认`requirements.txt`(与 Jupyter，TensorFlow 等)，以及一个一致的目录结构，帮助我们保持我们的研究项目漂亮整洁。为了能够在几个月/几年后重新访问一个项目，记录所有依赖项的特定版本以允许项目的环境被容易地重新创建是很关键的。这也让其他人可以让你的项目在他们的系统上工作！在之前的`venv`中，我写了一个 Makefile，它将特定的版本写到一个`requirements_versions.txt`中。这并不理想，因为它不会记录特定的 Python 版本，有时您会忘记运行 Make 命令。

根据我的工作流程和以前使用`venv`的经验，我有一些关键需求:

1.无缝记录特定的 Python/包版本。
2。很好地处理多个 Python 版本。
3。尽可能把东西放在项目目录中。

# pyenv

pyenv 是一个非常好的管理多个 Python 版本的工具。您可以轻松地设置您的全局 Python 版本，使用特定版本启动 shell，或者为特定项目设置版本。

在 MacOS 上，安装相对简单:

1.`xcode-select — install`
2。`brew install openssl readline sqlite3 xz zlib`
3。`brew update`
4。`brew install pyenv`
5。将`eval “$(pyenv init -)”`添加到您的 shell 配置文件中。
6。`exec “$SHELL”`

现在，您可以通过 pyenv 的简单命令轻松安装和使用不同的 Python 版本。其中包括`pyenv install`安装特定版本、`pyenv global`设置全局版本、`pyenv local`设置特定目录版本。您还可以使用环境变量`PYENV_VERSION`来设置特定会话的版本。

# pipenv

在我看来，pipenv 是 python 最好的包管理器。它会自动为您的项目创建和管理虚拟环境。它还与 pyenv 合作，根据需要安装和使用 python 版本，这正在改变生活。

在 MacOS 上，安装非常简单:

1.`brew install pipenv`

因为它是独立安装的，所以您也不会看到任何奇怪的，

`You are using pip version 9.0.1, however version 18.0 is available.`

似乎永远不会消失。

pipenv 使用`Pipfile`而不是`requirements.txt`工作。当您第一次在一个项目目录中运行`pipenv install`(您可以像使用`pip install`一样使用它)时，它会在那个目录中创建一个`Pipfile`。您甚至可以使用`pipenv install -r requirements.txt`从`requirements.txt`安装。当您安装、删除或更新软件包时，该文件将自动更新。它还记录了你的 python 版本！激活环境就像从项目目录中运行`pipenv shell`一样简单。不要再试图回忆你把你的环境放在哪里或者你把它叫做什么！

# 结论

忽视适当的包版本管理是非常容易的，尤其是作为一名数据科学家。然而，不适当的版本管理导致的问题可能会越积越多。从不能让合作者运行您的代码，到几个月后不能运行您自己的代码，您很容易浪费大量时间来修复依赖性问题。做对也可能是一件令人沮丧的事情。对我来说，pipenv/pyenv 组合是一种享受。它自动化了适量的工作，而不牺牲一致性。