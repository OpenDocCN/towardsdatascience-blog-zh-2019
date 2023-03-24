# 使用 R 进行文本分析应该知道的 5 个包

> 原文：<https://towardsdatascience.com/r-packages-for-text-analysis-ad8d86684adb?source=collection_archive---------2----------------------->

## 科学家应该知道的用于文本分析的 R 数据中最有用的包的完整概述

![](img/46728e05e163820425912289aedddd15.png)

Photo by [Patrick Tomasso](https://unsplash.com/@impatrickt?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/learn?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

# 1.包罗万象:Quanteda

```
install.packages("quanteda")
library(quanteda)
```

**Quanteda** 是定量文本分析的首选包。由 Kenneth Benoit 和其他贡献者开发的这个包是任何进行文本分析的数据科学家的必备工具。

为什么？因为这个包可以让你做很多事情。这包括从自然语言处理的基础——词汇多样性、文本预处理、构建语料库、令牌对象、文档特征矩阵——到更高级的统计分析，如 wordscores 或 wordfish、文档分类(如朴素贝叶斯)和主题建模。

该软件包的一个有用的教程是由渡边康平和斯特凡·缪勒开发的([链接](https://tutorials.quanteda.io))。

# 2.变形金刚:Text2vec

```
install.packages("text2vec")
library(text2vec)
```

**如果你正在基于文本数据构建机器学习算法，Text2vec** 是一个极其有用的包。这个软件包允许你从文档中构造一个文档术语矩阵(dtm)或者术语同现矩阵(tcm)。因此，您可以通过创建从单词或 n 元语法到向量空间的映射来对文本进行矢量化。在此基础上，您可以让模型适合 dtm 或 tcm。这包括主题建模(LDA，LSA)，单词嵌入(GloVe)，搭配，相似性搜索等等。

这个包的灵感来自 Gensim，一个著名的用于自然语言处理的 python 库。你可以在这里找到[这个包的有用教程。](http://text2vec.org/index.html)

# **3。适配器:Tidytext**

```
install.packages("tidytext")
library(tidytext)
```

**Tidytext** 是数据争论和可视化的基本包。它的一个好处是可以很好地与 R 中的其他 tidy 工具协同工作，比如 dplyr 或 tidyr。事实上，它就是为此目的而建造的。识别清理数据总是需要大量的工作，并且这些方法中的许多不容易适用于文本，Silge & Robinson (2016)开发了 tidytext，以使文本挖掘任务更容易、更有效，并与已经广泛使用的工具保持一致。

因此，这个包提供了允许你将文本转换成整齐格式的命令。分析和可视化的可能性是多种多样的:从情感分析到 tf-idf 统计、n-grams 或主题建模。这个包在输出的可视化方面特别突出。

你可以在这里找到[包的有用教程。](https://www.tidytextmining.com)

# 4.匹配者:Stringr

```
install.packages("stringr")
library(stringr)
```

作为一名数据科学家，你几乎已经和字符串打交道了。它们在许多数据清理和准备任务中发挥着重要作用。作为包生态系统 tidyverse(也包括 ggplot 和 dplyr)的一部分，stringr 包提供了一组内聚的函数，允许您轻松地处理字符串。

对于文本分析来说， **stringr** 是一个特别方便的处理正则表达式的包，因为它提供了一些有用的模式匹配函数。其他功能包括字符操作(在字符向量中操作字符串中的单个字符)和空白工具(添加、删除、操作空白)。

CRAN-R 项目有一个关于这个包的有用教程([链接](https://cran.r-project.org/web/packages/stringr/vignettes/stringr.html))。

# 5.炫耀:Spacyr

```
install.packages("spacyr")
library(spacyr)
spacy_install()
spacy_initialize()
```

你们大多数人可能知道 Python 中的 spaCy 包。嗯， **spacyr** 在 R 中提供了一个方便包装器，使得以简单的格式访问 spaCy 的强大功能变得容易。事实上，如果你仔细想想，这是一个相当不可思议的包，它允许 R 利用 Python 的能力。为了访问这些 Python 功能，spacyr 通过在 R 会话中初始化来打开一个连接。

该软件包对于更高级的自然语言处理模型至关重要，例如为深度学习准备文本，以及其他有用的功能，如语音标记、标记化、解析等。此外，它还可以很好地与 quanteda 和 tidytext 包结合使用。

你可以在这里找到一个有用的教程。

*我定期撰写关于数据科学和自然语言处理的文章。关注我的*[*Twitter*](https://twitter.com/celine_vdr)*或*[*Medium*](https://medium.com/@celine.vdr)*查看更多类似的文章或简单地更新下一篇文章。* ***感谢阅读！***