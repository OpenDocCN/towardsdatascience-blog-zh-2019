# 如何为数据科学或其他任何事物建立一个令人敬畏的 Python 环境

> 原文：<https://towardsdatascience.com/how-to-setup-an-awesome-python-environment-for-data-science-or-anything-else-35d358cc95d5?source=collection_archive---------6----------------------->

![](img/807c7d2eb3719055051d7ee4bab813c4.png)

# 介绍

用 Python 编码很棒，而且随着每个新版本的发布，越来越棒！对我来说，这主要是由于大量免费可用的库、其可读性以及最近引入的类型注释。然而，尤其是我们作为数据科学家，往往会产生大而乱的 Jupyter 笔记本或 python 文件，难以理解。除此之外，当一个项目依赖于同一个库的不同版本时，我们经常会遇到版本冲突。修复这个问题需要太多时间，并且经常会导致其他项目出现问题。必须有一种方法来避免这种情况，并方便我们的生活。在本文中，我将介绍我使用的工具和我通常遵循的技术。希望这能帮助未来的我记住所有这些，更重要的是，也能帮助你学到一些有用的东西。事不宜迟，让我们把手弄脏吧。

# Python 环境

## 翻译

让我们从使用 python 时最重要的事情开始:*解释器*。当然，你可以简单地下载并安装你最喜欢的 python 版本，然后把所有东西都放进去。但是，如果您的程序需要不同的 python 版本，或者程序依赖于同一第三方模块的不同版本，并且您希望在这些程序之间无缝切换，该怎么办呢？

> [Pyenv](https://github.com/pyenv/pyenv) 会帮你做到的！

Pyenv 是一组三个工具，我在这里介绍其中的两个，它们是 *pyenv* (用于安装 python)和 *pyenv-virtualenv* (用于配置你的全局工具)。您可以通过以下方式安装它们

```
curl [https://pyenv.run](https://pyenv.run) | bash
```

之后，给你的加上以下几行。bashrc (同为。zshrc)在您的终端中使用 pyenv

```
export PATH="~/.pyenv/bin:$PATH"
eval "$(pyenv init -)"
eval "$(pyenv virtualenv-init -)"
```

最后，重启你的终端。如果你是 mac 用户，当然可以用 brew 安装 pyenv。

现在，您可以使用 pyenv 安装几乎任何 python 解释器，包括 pypy 和 anaconda。注意，pyenv 在您的机器上本地构建 python。构建 python 需要几个库。在我的 Ubuntu 机器上，我必须安装以下几个软件，以免遇到问题

```
sudo apt-get install build-essential libsqlite3-dev sqlite3 bzip2 libbz2-dev zlib1g-dev libssl-dev openssl libgdbm-dev libgdbm-compat-dev liblzma-dev libreadline-dev libncursesw5-dev libffi-dev uuid-dev
```

现在，要安装 python 解释器，只需

```
pyenv install VERSION_YOU_WOULD_LIKE_TO_INSTALL
```

您可以通过 pyenv 列出所有可用的版本

```
pyenv install --list
```

具体来说，让我们安装 python 3.7.5，并将其作为默认的全局解释器

```
pyenv install 3.7.5
pyenv global 3.7.5
```

当您键入 Python-version 时，应该会返回 Python 3.7.5

## 依赖性管理

在 python 中管理项目依赖关系可能会变得混乱或手动。有很多工具可以帮你做到这一点。

> 我用的最多的是[诗词](https://poetry.eustace.io/)。

除了许多其他的东西，诗歌可以帮助你轻松地

*   管理项目的依赖关系，
*   通过虚拟环境分离您的项目，
*   轻松构建应用程序和库。

作者建议安装诗歌的方式是

```
curl -sSL [https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py](https://raw.githubusercontent.com/sdispater/poetry/master/get-poetry.py) | python
```

另一种方式，我将向您展示的是使用 pip 和 pyenv-virtualenv。你可能会问:为什么不只是皮普？因为这将在你的全球环境中安装诗歌及其依赖项，而你可能不需要也不想要它们。所以，这里是必要的命令

```
**# Create a virtual environment called tools that is based on 3.7.5**
pyenv virtualenv 3.7.5 tools 
**# Install poetry into the tools virtual env**
pyenv activate tools
pip install poetry 
**# Check installed poetry version**
poetry --version
**# Leave the virtual env** 
pyenv deactivate 
**# This does not work yet** 
poetry --version
**# Add your tools virtual env to the globally available ones**
pyenv global 3.7.5 tools
**# Now this works and you can start using poetry**
poetry --version
```

在我们开始使用 poem 创建我们的第一个项目之前，我建议对它进行配置，这样它可以在项目目录内的. venv 文件夹*中创建您项目的虚拟环境。当您使用像 VS Code 或 PyCharm 这样的 ide 时，这非常方便，因为它们会立即识别出这一点并选择正确的解释器*

```
# That seems to be peotry prior to 1.0.0
poetry config settings.virtualenvs.in-project true# That is poetry since 1.0.0
poetry config virtualenvs.in-project true
```

请注意，您只需设置此配置一次，因为它是全局设置并保持不变的。

最后，我们已经准备好了使用诗歌创建我们的第一个项目，太棒了！我把这个项目叫做 *dsexample* ，一个我知道的愚蠢的名字，但是我不想浪费更多的时间去寻找一个更好的名字。为了向您展示如何使用诗歌，我添加了特定版本的 pandas 和带有所有额外要求的 fastapi

```
**# Initialze a new project**
poetry new dsexample 
cd dsexample
**# Add modules and create virtual environment.**
poetry add pandas=0.25 fastapi --extras all
**# As an example of how you could add a git module**
poetry add tf2-utils --git [git@github.com](mailto:git@github.com):Shawe82/tf2-utils.git
```

如果你想看一个使用我在这里给你的所有东西的真实项目，请去我的 [Github Repo](https://github.com/shawe82) 。

## 一致的格式和可读性

现在，我们已经创建了我们的项目，我们将开始在它上面添加越来越多的代码。理想情况下，您的代码库格式一致，以确保可读性和可理解性。这可能会成为一个非常乏味的过程，尤其是如果你不是唯一一个在代码库上工作的人。

> [*黑色*](https://black.readthedocs.io/en/stable/) 前来救援！

Black 是 python 的一个工具，可以让你专注于必要的内容。它通过自动化将您从手工代码格式化中解放出来。因为它太棒了，让我们把它添加到我们的 dsexample 项目中，并格式化所有文件

```
**# We add black as a development dependency with --dev as we don't
# need it when it comes to production**
poetry add --dev black=19.3b0
**# Assume we are inside the current toplevel dsexample folder**
poetry run black .
```

很好，现在所有的文件看起来都很好。

## 类型正确性

从 Python 3.5 开始，如果我说错了请纠正我，类型注释是标准库的一部分。通过类型注释，您的代码变得更容易理解、维护，并且不容易出错。为什么不容易出错？因为您可以静态地检查变量和函数的类型是否与预期的匹配。当然，这必须是自动化的

> 又来了 [mypy](https://mypy.readthedocs.io) ！

Mypy 是 python 代码的静态类型检查器，可以在错误出现之前发现它们。使用诗歌向您的项目添加 mypy 和类型检查就像添加 black 一样简单

```
**# We add mypy as a development dependency with --dev as we don't
# need it when it comes to production**
poetry add --dev mypy
**# Assume we are inside the current toplevel dsexample folder**
poetry run mypy .
```

运行 mypy 可能会产生很多错误。当然，你可以把它配置成只警告你感兴趣的东西。你可以在你的项目中添加一个 mypy.ini 文件，并让你和未来的我参考[文档](https://mypy.readthedocs.io/en/latest/config_file.html)以获得更多细节。

## 自动化自动化

有了 black 和 mypy，我们可以避免手动格式化代码或遇到可避免的错误。但是，我们仍然需要手动执行这些工具。那不也应该是自动化的吗？是啊！

> [预提交](https://pre-commit.com/)就是你所需要的一切。

Pre-commit 是一个在您将代码提交到存储库之前执行检查的工具(我想当然地认为您的代码在 git 版本控制之下)。当这些检查失败时，您的提交将被拒绝。这样，您的存储库将永远看不到 mall 格式的代码，或者没有类型检查的代码，或者其他任何依赖于您将要包含的检查的代码。所以让我们安装预提交。

您可以使用 poem 将它直接安装到您的项目中，或者安装到您的本地机器上。我更喜欢后者，因为预提交只在本地使用，而不在 CI/CD 服务器上使用。相反，black 和 mypy 应该运行在 CI/CD 服务器上，因此，将它们添加到项目的 dev 依赖项中是有意义的。下面是我建议如何利用现有的工具虚拟环境来安装它

```
**# Install pre-commit** **into the tools virtual env**
pyenv activate tools
pip install pre-commit 
**# Leave the virtual env** 
pyenv deactivate
**# As we have already added the tool venv, it will work directly**
pre-commit --version
```

要使用它，你首先需要添加一个名为*的配置文件。将 pre-commit-config.yaml* 提交到项目的顶层文件夹。在这个文件中，您配置了所有应该运行的钩子。使用 mypy 和 black，这个文件看起来像

```
repos:
-   repo: [https://github.com/ambv/black](https://github.com/ambv/black)
    rev: 19.3b0
    hooks:
    - id: black
      language_version: python3.7
-   repo: [https://github.com/pre-commit/mirrors-mypy](https://github.com/pre-commit/mirrors-mypy)
    rev: v0.740
    hooks:
    - id: mypy
```

最后，您必须告诉预提交通过执行来设置挂钩

```
**# I assume your are in the top level folder**
pre-commit install
```

现在，钩子将在每次提交时运行。黑色挂钩不仅会检查格式问题，还会相应地格式化文件。每当您添加一个新的钩子时，建议您在所有文件上手动运行预提交，因为它只涉及自上次提交以来已经更改的文件

```
pre-commit run --all-files
```

就这样，我们实现了自动化！

# 包裹

有很多工具可以帮助你专注于重要的事情。在这里，我只给你介绍了几个，当然，还有其他更多。希望你已经学到了新的东西，可以专注于内容。感谢您的关注，如果有任何问题、意见或建议，请随时联系我。