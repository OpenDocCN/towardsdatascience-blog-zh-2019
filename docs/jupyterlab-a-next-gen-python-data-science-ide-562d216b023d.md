# JupyterLab——新一代 Python 数据科学集成开发环境

> 原文：<https://towardsdatascience.com/jupyterlab-a-next-gen-python-data-science-ide-562d216b023d?source=collection_archive---------9----------------------->

## 一篇将 Python 数据科学家推向 JupyterLab 的文章。

在使用 Python 进行数据科学项目时，您可能会问自己，哪种 IDE 最能满足您在数据探索、清理、准备、建模、评估和部署方面的需求。您可能还在 Google 上做了一些研究，浏览了标题为“顶级 Python 数据科学 IDE”的各种页面，并开始绝望地意识到，没有一个提到的产品为所有必要的实现步骤结合了无缝的外观和感觉。最终，你回到了你所熟知的，但是独立的工具集。好消息是，有一个非常有前途的项目正等待发布 1.0 版本，以解决我们日常的数据科学需求。我指的是 **JupyterLab。**

![](img/42df567d5e855ed0c8ef813059a8463d.png)

A processed image of Jupiter from Juno’s ninth close flyby provided at [13]. NASA / SWRI / MSSS / GERALD EICHSTÄDT / SEÁN DORAN © CC NC SA

## 为什么是 JupyterLab？

近年来，Python 笔记本作为一种以交互和布局良好的方式显示代码和结果的工具受到了很多关注。这无疑有助于降低开始编程的门槛，并有助于教育，因为输入及其处理后的输出立即显示在浏览器中，这是许多用户非常熟悉的。尽管 Python 笔记本很受欢迎，但需要做的编码越多，经典的 Python IDE 或文本编辑器就越方便。**如果有一种工具能取二者之长，从而将两个世界结合起来，那该多好？** JupyterLab 正在通过使用户能够以灵活、集成和可扩展的方式处理文档和活动，如 Jupyter 笔记本、文本编辑器、终端和自定义组件，从而实现这一目标[1]。

Jupyter notebooks 的“Jupyter web 界面的演变”到 JupyterLab 是基于 2015 年进行的用户体验调查，强调了以下三个成功因素[2]:

1.  用户喜欢笔记本体验。
2.  用户想要组合和混合不同的 Jupyter 构建模块。
3.  用户需要轻松协作的能力。

根据我使用 Jupyter 笔记本的经验，这些因素并不奇怪。

**优点:**

Jupyter 笔记本电脑在可视化功能方面尤为强大。例如，像 Google Facets 这样的工具已经被开发用于 Jupyter 笔记本[3]。

**+** 与图形的交互非常方便，例如只需使用`%matplotlib notebook`或 ipywidgets [4]。

**+** 通过将一个单元格从 code 改为 Markdown，可以为一段代码添加一个漂亮简洁的文档。

Jupyter 笔记本是一个非常好的数据讲述和演示工具，因为它可以显示文档和代码输出。

**缺点:**

**-** 缺乏内置的变量检查器是有经验的标准 IDE 用户在 Jupyter 笔记本中首先缺少的东西之一。我想强调的是，有一个非常有用的社区贡献的非官方扩展使用了笔记本的元数据[8]。

**-** Jupyter 笔记本在开发代码时没有提供方便的文件浏览器视图[5]。因此，读写文件变得很笨拙。

**-** 为了与操作系统的终端交互或使用作为插件添加的终端视图，用户需要在终端命令前加上一个感叹号`!`。

**-** 打开和浏览文件很麻烦，因为你需要先加载文件，然后选择合适的方式以编程方式显示它。这比在 IDE 中双击打开例如 jpg 文件需要更多的努力。

**-** 测试和模块化在 Jupyter 笔记本中很难处理。

**-** 缺少与版本控制系统的无缝集成，尽管 nbdime 之类的插件使笔记本的区分和合并变得更加容易，这是一个有趣的进步[7]。

**-** 缺乏方便的可视化调试和剖析功能，尽管有像 pixedebugger[10]这样非常有前途的开发。

我想强调的是，这并不是一份详尽的利弊清单。“缺点”一节中列出的陈述表明所提及的功能根本无法实现。它也被列在缺点下面，以防它在 Jupyter 笔记本中不直观。

让我们看看当前可用的 JupyterLab 版本(0.35.6)的详细信息，看看从 Jupyter 笔记本迁移到 JupyterLab 时会涉及哪些内容。

## Python 和 Jupyter 笔记本文件共享一个内核

JupyterLab 让您可以开发复杂的 python 代码以及 Jupyter 笔记本，并轻松地将它们连接到同一个内核。我认为这是解决弊端的一个关键特征。

在下面的动画中，您将看到如何在 JupyterLab 中连接多个 Python 文件和笔记本。

![](img/74ada53c40f6ba16f397f59f1ed24b14.png)

Creation of two Python files and one Jupyter notebook in JupyterLab. Consecutively, you see the selection of one common kernel for each of the files. At the end you can observe that all three files have access to the same kernel as they are using the the variables `a` and `b` interactively.

现在来看看下面的动画，它展示了将数据加载到数据框架、单独开发模型，同时利用 Jupyter 笔记本的强大功能以无缝方式测试和可视化模型的简单性。除了拥有一个通用的变量检查器和文件浏览器之外，所有这些都是可能的。这里可以看到一个简单的手动函数逼近任务。

![](img/910437a7a1b6d9179b652d1aea4595d8.png)

Exploration of the csv file and loading it into a dataframe in a kernel which is shared among the open files. The dataframe is visible in the variable inspector. First the given `x` and `y` vectors are plotted in blue. Afterwards, the function approximator plotted in orange is iteratively improved by manually adjusting the function fun in the file model.py. The approximator covers fully the given data input at the end. Therefore, only an orange line is visible anymore.

这有效地分离了提取、建模和可视化，而不必为了共享数据帧而读写文件。这为您的日常工作节省了大量时间，因为它降低了文件加载中的错误风险，并且因为在项目的早期阶段设置 EDA 和试验要快得多。此外，它有助于减少代码行的数量，以防您像我一样向数据管道中添加同样多的`asserts`。

如果您在项目的相同上下文中需要一个真正快速的终端，那么您只需打开 launchpad 并创建一个新的终端视图。如果要检查模型或算法所需的资源，这尤其有用，如下图所示。

![](img/56044eeb06f44f969e201cc815a97dd0.png)

JupyterLab- Ian Rose (UC Berkeley), Chris Colbert (Project Jupyter) at 14:30 shows how to open a terminal within JupyterLab [9].

用 JupyterLab 打开数据文件也很简单。它以一种很好的方式呈现，例如以 csv 文件的表格形式呈现，并且利用了延迟加载，因此使它很快，并且它支持巨大的文件大小。下一个动画展示了如何从 csv 文件中打开虹膜数据集。

![](img/21a07541f0807dd9c4cf196525879b41.png)

JupyterLab- Ian Rose (UC Berkeley), Chris Colbert (Project Jupyter) at 19:15 shows the IRIS data set in a csv file being opened with a simple click [9].

你也可以通过点击打开图像文件，这在处理计算机视觉任务时非常方便。在下面的动画中，你可以看到 Jupyterlab 如何在一个单独的最后使用的面板中渲染哈勃望远镜的图像。

![](img/5e7761dc5f62cc5b03beae83c26c939b.png)

JupyterLab- Ian Rose (UC Berkeley), Chris Colbert (Project Jupyter) at 17:58, shows an image being rendered in by clicking on it in the built in file explorer [9].

此外，您可以通过 JupyterLab 的 Git 扩展导航和使用 Git，如下所示。

![](img/b7d0e4093689fd3244cb09166d514986.png)

[Parul Pandey](https://medium.com/u/7053de462a28?source=post_page-----562d216b023d--------------------------------)’s gif showing the navigation in the Git extension provided in [6].

在撰写本文时，JupyterLab 中还没有可视化调试和分析功能。目前计划在未来发布[11]。因此，开发将在 1.0 版本发布后最早开始。尽管有这样的计划，在 Jupyterlab [12]中仍有工作要做，以使 PixieDebugger 适用于笔记本电脑。

## 结论

JupyterLab 为 Jupyter 笔记本电脑添加了一个完整的 IDE，这无疑是 Jupyter 笔记本电脑的一个强大发展。它可以很好地集成到数据科学家的日常工作中，因此也可以被视为下一代工具。数据提取、转换、建模可视化和测试的分离已经非常容易。

考虑到这一点，我希望看到 1.0 版本很快推出。如果你对 JupyterLab 项目感到兴奋，并想自己尝试一下，只需按照 [Parul Pandey](https://medium.com/u/7053de462a28?source=post_page-----562d216b023d--------------------------------) 的文章中的说明:

[](/jupyter-lab-evolution-of-the-jupyter-notebook-5297cacde6b) [## 朱庇特实验室:朱庇特笔记本的进化

### 为了给更好的东西让路，所有美好的事情都会结束。

towardsdatascience.com](/jupyter-lab-evolution-of-the-jupyter-notebook-5297cacde6b) 

Jupyter 项目，JupyterLab 概述(2018)，[https://JupyterLab . readthedocs . io/en/stable/getting _ started/Overview . html](https://jupyterlab.readthedocs.io/en/stable/getting_started/overview.html)

[2] N. Tache，JupyterLab:Jupyter web 界面的演变(2017)，[https://www . oreilly . com/ideas/JupyterLab-The-evolution-of-The-Jupyter-web-interface](https://www.oreilly.com/ideas/jupyterlab-the-evolution-of-the-jupyter-web-interface)

[3] J. Wexler，Facets:机器学习训练数据的开源可视化工具(2017)，[https://ai . Google blog . com/2017/07/Facets-Open-Source-Visualization-Tool . html](https://ai.googleblog.com/2017/07/facets-open-source-visualization-tool.html)

[4] 5agado，朱庇特笔记本中的交互可视化(2017)，[朱庇特笔记本中的交互可视化](/interactive-visualizations-in-jupyter-notebook-3be02ab2b8cd)

[5] I. Rose 和 G. Nestor，Jupyter lab:Jupyter 笔记本的进化(2018)，[https://www.youtube.com/watch?v=NSiPeoDpwuI&feature = youtu . be&t = 254](https://www.youtube.com/watch?v=NSiPeoDpwuI&feature=youtu.be&t=254)

[6] P. Pandey，Jupyter Lab:Jupyter 笔记本的进化(2019)，[https://towardsdatascience . com/Jupyter-Lab-Evolution-of-the-Jupyter-Notebook-5297 cacde 6b](/jupyter-lab-evolution-of-the-jupyter-notebook-5297cacde6b)

[7] Project Jupyter，Jupyter Notebook Diff and Merge tools(2019)，[https://github.com/jupyter/nbdime](https://github.com/jupyter/nbdime)

[8] Jupyter Contrib Team，Variable Inspector (2019)，[https://Jupyter-Contrib-nb extensions . readthe docs . io/en/latest/nb extensions/varInspector/readme . html](https://jupyter-contrib-nbextensions.readthedocs.io/en/latest/nbextensions/varInspector/README.html)

[9] I. Rose 和 C. Colbert，Jupyter 项目的 JupyterLab 下一代用户界面(2018 年)，[https://www.youtube.com/watch?v=t5q19rz_FNw&feature = youtu . be](https://www.youtube.com/watch?v=t5q19rz_FNw&feature=youtu.be)

[10] D. Taieb，你一直想要的 Jupyter 笔记本的可视化 Python 调试器(2018)，[https://medium . com/codait/The-Visual-Python-Debugger-for-Jupyter-Notebooks-youve-Always-Wanted-761713 babc 62](https://medium.com/codait/the-visual-python-debugger-for-jupyter-notebooks-youve-always-wanted-761713babc62)

[11]Jupyter 项目，JupyterLab (2019)，[https://github . com/Jupyter/roadmap/blob/master/JupyterLab . MD](https://github.com/jupyter/roadmap/blob/master/jupyterlab.md)

[12]Jupyter 项目，JupyterLab (2017 年)，[https://github.com/jupyterlab/jupyterlab/issues/3049](https://github.com/jupyterlab/jupyterlab/issues/3049)

[13] M. Bartels，美国宇航局发布了来自朱诺任务(2017 年)的令人难以置信的木星新图像宝藏，[https://www . Newsweek . com/NASA-Releases-Treasure-Trove-Incredible-New-Images-Jupiter-Its-Juno-Mission-705210](https://www.newsweek.com/nasa-releases-treasure-trove-incredible-new-images-jupiter-its-juno-mission-705210)

本文旨在通过结合作者的实践经验和深入的文献研究，提供 JupyterLab 可能是 Python 数据科学家的首选 ide 的理由。它不应作为安装指南，也不应作为功能的列表和比较。