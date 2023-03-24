# Python Pro 提示:想在 Python 中使用 R/Java/C 或者任何语言？

> 原文：<https://towardsdatascience.com/python-pro-tip-want-to-use-r-java-c-or-any-language-in-python-d304be3a0559?source=collection_archive---------20----------------------->

## 蟒蛇短裤

## Python 提供了一种基本而简单的方法来处理这样的需求，我们必须在多种语言之间来回切换

![](img/ee88b28569984b07e79945d35ce09b23.png)

[**巨蟒**](https://amzn.to/2XPSiiG) **很棒。真的很棒。**

但是随着时间的推移，这个领域变得/将变得与语言无关。许多伟大的工作正在用许多其他语言完成。

虽然我仍然把 Python 作为主要语言，但如果它能完成工作，我会毫不犹豫地转向另一种语言。

事实上，每一种语言都以这样一种方式进化，它在某些领域建立了自己的据点。例如， ***有些人可能会发现使用 R 进行回归更容易，或者在 R 中使用 ggplot 进行绘图(尽管我真诚地感到 Python 已经在*** [***可视化***](/3-awesome-visualization-techniques-for-every-dataset-9737eecacbe8) ***部门取得了长足的进步。*)**

有时是因为某个特定的库是用 Java/C 编写的，而有人还没有把它移植到 Python 上。

但是有没有更好的方法来处理这种不断的麻烦呢？

我喜欢 Python 是因为我现在很好的理解了它。与用 R 或 Java 或 Scala 相比，用 Python 做这么多事情对我来说很容易。

***如果我只想使用 R 中的线性回归包，为什么我必须在 R 中编写我的数据准备步骤？***

或者如果我只想使用 Stacknet 包，为什么我必须学习用 Java 创建图表？

现在 Python 和 R 有了很多包装器。如何在 Python 中使用 R 或者如何在 R 中使用 Python？`rpy2`和`reticulate`

这些套餐都不错，可能会解决一些问题。但是他们没有解决一般性的问题。每次我想从一种语言切换到另一种语言时，我都需要学习一个全新的包/库。完全不可伸缩。

在这一系列名为 [**Python Shorts**](https://towardsdatascience.com/tagged/python-shorts) 的帖子中，我将解释 Python 提供的一些简单构造，一些必要的技巧和我在数据科学工作中经常遇到的一些用例。

这篇文章是关于利用另一种语言的一个特殊的包/库，同时不离开用我们的主要语言编码的舒适。

# 问题陈述

我将从一个问题陈述开始解释这一点。假设我不得不使用 R 创建一个图表，但是我想用 Python 准备我的数据。

这是任何数据科学家都可能面临的普遍问题。用一种语言做一些事情，然后转到另一种语言做一些其他的事情。

***不离开 Jupyter 笔记本就能做到吗？还是我的 Python 脚本？***

# 解决方案

![](img/564466e364bb6f4b3aba2cc5c00fbf6c.png)

以下是我如何实现这一点。对一些人来说，这可能看起来很黑客，但我喜欢黑客。

```
import pandas as pd
data=pd.read_csv("data.csv")
data = preprocess(data)data.to_csv("data.csv",index=None)
os.system("Rscript create_visualization.R")
```

*`***os.system***`***命令为我提供了一种使用 Python*** 访问我的 shell 的方法。外壳是你可以随意使用的有力的[工具](https://becominghuman.ai/shell-basics-every-data-scientist-should-know-376df75dd09c)。您几乎可以在 shell 上运行任何语言。*

*将在 python 中运行的相应的`Rscript`看起来类似于:*

```
*data<-read.table("data.csv")
ggplot(...)
ggsave("plot.png")*
```

*然后，我可以加载 png 文件，并使用类似 markdown hack 的工具在我的 Jupyter 笔记本上显示它。*

```
*![alt text](plot.png "Title")*
```

*对于不想离开 R 的舒适环境的 R 用户来说，R 还有一个类似于`os.system`的`system`命令，可以用来在 R 中运行 Python 代码*

# *结论*

*[***中的`***os.system***`***Python******](https://amzn.to/2XPSiiG)***通过让我们从 Python 中调用 shell 命令，为我们提供了一种在 Python 中做每一件事的方法。****

*我已经在我的很多项目中使用了它，在这些项目中，我使用 Mutt 发送电子邮件。或者运行一些 Java 程序或者瞎搞。*

*这看起来像是一种不成熟的方法，但是它很有效，而且足够一般化，以至于当你想用任何其他语言做一些事情并将其与 Python 集成时，你不必学习一个新的库。*

*如果你想了解更多关于 Python 3 的知识，我想从密歇根大学调出一门关于学习[中级 Python](https://bit.ly/2XshreA) 的优秀课程。一定要去看看。*

*将来我也会写更多初学者友好的帖子。让我知道你对这个系列的看法。在[](https://medium.com/@rahul_agarwal)**关注我或者订阅我的 [**博客**](https://mlwhiz.com/) 了解他们。一如既往，我欢迎反馈和建设性的批评，可以通过 Twitter [@mlwhiz](https://twitter.com/MLWhiz) 联系。***