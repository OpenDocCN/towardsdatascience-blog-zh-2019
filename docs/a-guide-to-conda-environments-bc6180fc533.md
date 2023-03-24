# 康达环境权威指南

> 原文：<https://towardsdatascience.com/a-guide-to-conda-environments-bc6180fc533?source=collection_archive---------0----------------------->

## 如何使用 conda for Python & R 管理环境

![](img/5935245b849abbc8ffca4586e48de60a.png)

Conda’s natural environment. [Illustration](https://www.flickr.com/photos/terzocchio/8322976511) by [Johann Wenzel Peter](https://en.wikipedia.org/wiki/Johann_Wenzel_Peter).

Conda 环境就像是 [Python 的虚拟环境](https://medium.com/@msarmi9/virtual-environments-104c62d48c54)的表亲。两者都有助于管理依赖性和隔离项目，它们以相似的方式工作，有一个关键的区别:conda 环境是**语言不可知的**。也就是说，它们支持 Python 以外的语言。

☄️在本指南中，我们将介绍使用 Python 的`conda`创建和管理环境的基础知识

**⚠️注:**在本指南中，我们将在 macOS Mojave 上使用最新版本的 Conda v4.6.x、Python v3.7.y 和 R v3.5.z。

## 目录

[Conda 与 Pip 和 Venv —有什么区别？](https://medium.com/p/bc6180fc533#7d04)
[使用康达环境](https://medium.com/p/bc6180fc533#e814)
[安装包](https://medium.com/p/bc6180fc533#5193)
[管理环境](https://medium.com/p/bc6180fc533#266b)
[带 R 的环境](https://medium.com/p/bc6180fc533#c010)
[进一步阅读](https://medium.com/p/bc6180fc533#39a9)

# 康达 vs .皮普 vs. Venv —有什么区别？

在我们开始之前，你们中的一些人可能想知道`conda`、`pip`和`venv`之间的区别。

很高兴你问了。我们不能说得比这更好了:`[pip](https://pip.pypa.io/en/stable/)`是 Python 的一个包管理器*。*T5 是 Python 的环境管理器*。`conda`既是包又是环境管理器，并且**是语言不可知的**。*

鉴于`venv`只为 Python 开发创建隔离环境，`conda`可以为任何语言创建隔离环境(理论上)。

而`pip`只安装来自 [PyPI](https://pypi.org) 的 Python 包，`conda`两者都可以

*   从像 [Anaconda Repository](https://repo.anaconda.com) 和 [Anaconda Cloud](https://anaconda.org) 这样的库安装软件包(用任何语言编写)。
*   在活动的 Conda 环境中使用`pip`从 PyPI 安装软件包。

多酷啊。

👉🏽如果想要一个比较这三者的图表，请点击[这里](https://conda.io/projects/conda/en/latest/commands.html#conda-vs-pip-vs-virtualenv-commands)(不要忘记向右滚动！).

![](img/48abc7cafe10f73418c3d85d3238e348.png)

[Morning Mist](https://commons.wikimedia.org/wiki/File:Cole_Thomas_Morning_Mist_Rising_Plymouth_New_Hampshire_(A_View_in_the_United_States_of_American_in_Autunm_1830.jpg) by [Thomas Cole](https://en.wikipedia.org/wiki/Thomas_Cole).

# 使用 Conda 环境

## 创造环境

要使用`conda`为 Python 开发创建一个环境，运行:

```
% conda create --name conda-env python                  # Or use -n
```

💥**重要提示:**用您的环境名称替换“conda-env”。从现在开始，我们将始终使用“conda-env”来命名我们的环境。

这个环境将使用与您当前 shell 的 Python 解释器相同的 Python 版本。要指定不同版本的 Python，请使用:

```
% conda create -n conda-env python=3.7
```

你也可以在创建环境时安装额外的包，比如说`numpy`和`requests`。

```
% conda create -n conda-env numpy requests
```

⚠️ **注意:**因为`conda`确保安装包时满足依赖关系，Python 将与`numpy`和`requests`一起安装😁。

您还可以指定想要安装的软件包版本。

```
% conda create -n conda-env python=3.7 numpy=1.16.1 requests=2.19.1
```

⚠️ **注意:** [建议](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-pkgs.html#installing-packages)同时安装你想包含在一个环境中的所有软件包，以帮助避免依赖冲突。

最后，您可以通过调用来激活您的环境:

```
% conda activate conda-env           
(conda-env) %                          # Fancy new command prompt
```

并使用以下命令将其禁用:

```
% conda deactivate
%                                      # Old familiar command prompt
```

## 环境生活的地方

当您使用 Python 的`venv`模块创建一个环境时，您需要通过指定它的路径来说明它位于何处。

```
% python3 -m venv /path/to/new/environment
```

另一方面，用`conda`创建的环境默认位于 Conda 目录的`envs/`文件夹中，其路径如下所示:

```
% /Users/user-name/miniconda3/envs          # Or .../anaconda3/envs
```

我更喜欢`venv`采用的方法，原因有二。

1️⃣通过将环境包含为子目录，可以很容易地判断一个项目是否利用了一个隔离的环境。

```
my-project/
├── conda-env                    # Project uses an isolated env ✅
├── data                             
├── src                  
└── tests
```

2️⃣它允许你对所有的环境使用相同的名字(我使用“conda-env”)，这意味着你可以用相同的命令激活每个环境。

```
% cd my-project/
% conda activate conda-env
```

💸**好处:**这允许你给激活命令起别名，并把它放在你的`.bashrc`文件中，让生活简单一点。

**⚠️注:**如果你将所有的环境都保存在 Conda 的`env/`文件夹中，你将不得不给每个环境取一个不同的名字，这可能会很痛苦😞。

那么，*你如何把环境放到你的康达的`env/`文件夹之外呢？通过在创建环境时使用`--prefix`标志而不是`--name`。*

```
% conda create --prefix /path/to/conda-env             # Or use -p
```

**⚠️注:**这使得一个名为“conda-env”的环境出现在指定的路径中。

就这么简单。然而，将环境放在默认的`env/`文件夹之外有两个缺点。

1️⃣ `conda`无法再用`--name`旗找到你的环境。相反，您需要沿着环境的完整路径传递`--prefix`标志。例如，在安装包时，我们将在下一节中讨论。

2️⃣您的命令提示符不再以活动环境的名称为前缀，而是以其完整路径为前缀。

```
(/path/to/conda-env) %
```

你可以想象，这很快就会变得一团糟。比如说，像这样的东西。

```
(/Users/user-name/data-science/project-name/conda-env) %        # 😨
```

幸运的是，有一个简单的解决方法。你只需要修改你的`.condarc`文件中的`env_prompt`设置，你只需要简单的一笔就可以完成。

```
% conda config --set env_prompt '({name}) '
```

**⚠️注:**如果你已经有一个`.condarc`文件，这将编辑你的文件，如果你没有，则创建一个。关于修改您的`.condarc`文件的更多信息，请参见[文档](https://conda.io/projects/conda/en/latest/user-guide/configuration/index.html)。

现在，您的命令提示符将只显示活动环境的名称。

```
% conda activate /path/to/conda-env
(conda-env) %                                        # Woohoo! 🎉
```

最后，您可以查看所有现有环境的列表。

```
% conda env list# conda environments:
#
                         /path/to/conda-env
base                  *  /Users/username/miniconda3
r-env                    /Users/username/miniconda3/envs/r-env
```

**⚠️注:**`*`指向当前活动环境。有点烦人的是，即使没有环境活动，它也会指向“base”🤷🏽‍♂️.

![](img/b74cf8a298718998bd47a01f0564842d.png)

An [American Lake Scene](https://commons.wikimedia.org/wiki/File:Cole_Thomas_American_Lake_Scene_1844.jpg) by Thomas Cole.

# 安装软件包

用`conda`安装包有两种方式。

活跃环境中的 1️⃣。

2️⃣从您的默认外壳。

后者要求您使用与创建环境时相同的标志(`--name`或`--prefix`)指向您想要安装软件包的环境。

无论您使用哪种标志，前者都同样有效。

💥**重要提示:**我们*强烈*建议坚持前一种方法，因为它消除了无意中在系统范围内安装软件包的危险。

**♻️提醒:**本指南中的所有环境均命名为“conda-env”。您可以用您的环境名替换“conda-env”。

## 从巨蟒库

默认情况下，`conda`从 [Anaconda 存储库](https://repo.anaconda.com)安装软件包。一旦您创建了一个环境，您可以通过两种方式安装额外的软件包。

活跃环境中的 1️⃣。

```
(conda-env) % conda install pandas=0.24.1                   # 🐼
```

2️⃣从您的默认外壳。

```
% conda install -n conda-env pandas=0.24.1      # Or -p /path/to/env
```

同样，您可以通过两种方式更新环境中的软件包。

活跃环境中的 1️⃣。

```
(conda-env) % conda update pandas
```

2️⃣从您的默认外壳。

```
% conda update -n conda-env pandas             # Or -p /path/to/env
```

您还可以用两种方式列出给定环境中安装的软件包——是的，您猜对了。

活跃环境中的 1️⃣。

```
(conda-env) % conda list
```

2️⃣从您的默认外壳。

```
% conda list -n conda-env                      # Or -p /path/to/env
```

## 来自其他 Conda 存储库

如果在默认的 [Anaconda 仓库](https://repo.anaconda.com)中找不到一个包，你可以试着在 [Anaconda Cloud](https://anaconda.org) 上搜索它，它托管了由第三方仓库如 [conda-Forge](https://conda-forge.org) 提供的 Conda 包。

要从 Anaconda Cloud 安装一个包，您需要使用`--channel`标志来指定您想要安装的存储库。例如，如果你想安装康达-福吉的`opencv`，你可以运行:

```
(conda-env) % conda install --channel conda-forge opencv     # Or -c
```

幸运的是，`conda`跟踪软件包是从哪里安装的。

```
(conda-env) % conda list# packages in environment at /path/to/conda-env:
#
# Name                Version          Build             Channelnumpy                 1.16.1           py37h926163e_0
opencv                4.1.0            py37h0cb0d9f_3    conda-forge
pandas                0.24.2           py37h0a44026_0
```

`numpy`和`pandas`的空白通道条目代表`default_channels`，默认情况下它被设置为 Anaconda 存储库。

**⚠️注:**为了简洁起见，我们只展示了上面的一些软件包。

您也可以永久添加一个频道作为包源。

```
% conda config --append channels conda-forge
```

这将修改您的`.condarc`文件，如下所示:

```
env_prompt: '({name}) '      # Modifies active environment prompt
channels:                    # Lists package sources to install from
- defaults                   # Default Anaconda Repository
- conda-forge
```

🚨**注意:**你的渠道顺序*事关*。如果一个软件包可以从多个渠道获得，`conda`将从您的`.condarc`文件中列出的*最高*渠道安装它。有关管理渠道的更多信息，请参见[文档](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-channels.html)。

## 来自 PyPI

如果 Anaconda 存储库或 Anaconda Cloud 中没有可用的包，您可以尝试用`pip`安装它，默认情况下，`conda`会在任何用 Python 创建的环境中安装它。

例如，要安装带有`pip`的请求，您可以运行:

```
(conda-env) % pip install requests
```

请注意，`conda`正确地将 PyPI 列为`requests`的通道，从而很容易识别出安装了`pip`的包。

```
(conda-env) % conda list# packages in environment at /path/to/conda-env:
#
# Name                Version          Build             Channelnumpy                 1.16.1           py37h926163e_0
opencv                4.1.0            py37h0cb0d9f_3    conda-forge
pandas                0.24.2           py37h0a44026_0
requests              2.21.0                   pypi_0    pypi
```

🚨**注意:**由于`pip`软件包不具备`conda`软件包的所有特性，强烈建议尽可能安装带有`conda`的软件包。有关`conda`与`pip`封装的更多信息，请点击[此处](https://www.anaconda.com/understanding-conda-and-pip/)。

![](img/dc29b28b506bdd9f713a0021c3de91b8.png)

[Moonlight](https://commons.wikimedia.org/wiki/File:Cole_Thomas_Moonlight_1833-34.jpg) by Thomas Cole.

# 管理环境

## 环境文件

使您的工作可以被其他人复制的最简单的方法是在您的项目根目录中包含一个文件，该文件列出了您的项目环境中安装的所有包及其版本号。

Conda 将这些[环境文件称为](https://conda.io/projects/conda/en/latest/user-guide/tasks/manage-environments.html#sharing-an-environment)。它们是 Python 虚拟环境需求文件的精确模拟。

像其他任何事情一样，您可以用两种方法创建环境文件。

活跃环境中的 1️⃣。

```
(conda-env) % conda env export --file environment.yml       # Or -f
```

2️⃣从您的默认外壳。

```
% conda env export -n conda-env -f /path/to/environment.yml
```

您的`environment.yml`文件看起来会像这样:

```
name: null                          # Our env was made with --prefix
channels:
  - conda-forge                     # We added a third party channel
  - defaults
dependencies:
  - numpy=1.16.3=py37h926163e_0
  - opencv=3.4.2=py37h6fd60c2_1
  - pandas=0.24.2=py37h0a44026_0
  - pip=19.1.1=py37_0
  - pip:                            # Packages installed from PyPI
    - requests==2.21.0
prefix: /Users/user-name/data-science/project-name/conda-env
```

**⚠️注:**为了简洁起见，我们只展示了上面的一些软件包。

## 复制环境

给定一个`environment.yml`文件，您可以轻松地重新创建一个环境。

```
% conda env create -n conda-env -f /path/to/environment.yml
```

💸**附加功能:**您还可以使用以下功能将`environment.yml`文件中列出的软件包添加到现有环境中:

```
% conda env update -n conda-env -f /path/to/environment.yml
```

![](img/16f0164318841e4b9ea6a1ce95624443.png)

[View in the White Mountains](https://commons.wikimedia.org/wiki/File:Cole_Thomas_View_in_the_White_Mountains_1827.jpg) by Thomas Cole.

# R 环境

要在一个环境中使用 R，您需要做的就是安装`r-base`包。

```
(conda-env) % conda install r-base
```

当然，您可以在第一次创建环境时这样做。

```
% conda create -n r-env r-base
```

**⚠️注意:**用您的环境名替换“r-env”。

conda 的 R 包可以从 Anaconda Cloud 的 [R 通道](https://anaconda.org/r)获得，默认情况下包含在 Conda 的`[default_channels](https://docs.conda.io/projects/conda/en/latest/user-guide/configuration/use-condarc.html#default-channels-default-channels)`列表中，所以在安装 R 包时不需要指定 R 通道，比如说`tidyverse`。

```
% conda activate r-env
(r-env) % conda install r-tidyverse 
```

**⚠️注:**所有来自 r 通道的包裹都带有前缀“`r-`”。

如果你愿意，你可以安装`r-essentials`包，它包括 80 多个最流行的科学 R 包，像`tidyverse`和`shiny`。

```
(r-env) % conda install r-essentials
```

最后，如果你想安装 Conda 没有提供的 R 包，你需要从 [CRAN](https://cran.r-project.org) 构建这个包，你可以在这里找到[的说明。](https://docs.conda.io/projects/conda-build/en/latest/user-guide/tutorials/build-r-pkgs.html)

# 进一步阅读

如果你偶然发现自己想知道 Conda 环境到底是如何工作的，看看这篇关于 Python 的虚拟环境如何工作的简介。Conda 环境以完全相同的方式工作。

除此之外，我们差不多做到了。如果你想了解我最新的数据科学帖子，欢迎在推特上关注我。

干杯，祝阅读愉快。

# 2019 年 8 月更新:Conda 修订版

你真的每天都能学到新东西。今天早上，我的朋友 [Kumar Shishir](https://medium.com/u/9b9b5e57fb76?source=post_page-----bc6180fc533--------------------------------) 告诉我另一个**非常有用的** `conda`特性:conda 修订版。

我简直不敢相信自己的耳朵。我怎么能在完全和完全不知道这样一个辉煌的特征的情况下憔悴了这么久？

修订版会随着时间的推移跟踪您的环境的变化，允许您轻松地移除包及其所有依赖关系。

例如，假设我们创建了一个新的`conda-env`并安装了`numpy`，然后安装了`pandas`。我们的修订历史如下所示:

```
(conda-env) % conda list --revisions 2019-08-30 16:04:14  (rev 0)               # Created our env+pip-19.2.2
+python-3.7.42019-08-30 16:04:30  (rev 1)               # Installed numpy+numpy-1.16.4
+numpy-base-1.16.42019-08-30 16:04:39  (rev 2)               # Installed pandas+pandas-0.25.1
+python-dateutil-2.8.0
+pytz-2019.2
```

想象一下，我们不再想在我们的环境中拥有`pandas`,因为它(不知何故)与我们早期的依赖不兼容，或者因为我们不再需要它。

修订版允许我们将环境回滚到以前的版本:

```
(conda-env) % conda install --revision 1
(conda-env) % conda list --revisions         # (Showing latest only) 2019-08-30 16:08:05  (rev 3)                 # Uninstalled pandas -pandas-0.25.1
-python-dateutil-2.8.0
-pytz-2019.2
```

每个包装上的`—`标志告诉我们，我们已经成功地将从我们的环境中移除。现在，我们准备回到一些数据科学😎。

# 更新 02/2020:清除你的 Tarballs！

随着您构建更多的项目，每个项目都有自己的环境，您将开始从已安装的包中快速积累`tarballs`。

要删除它们并释放一些磁盘空间，请运行:

```
% conda clean --all                     # no active env needed
```