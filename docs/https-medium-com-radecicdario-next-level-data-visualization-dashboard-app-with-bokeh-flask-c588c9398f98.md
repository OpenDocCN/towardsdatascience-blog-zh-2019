# 下一级数据可视化—带有散景和烧瓶的仪表板应用程序

> 原文：<https://towardsdatascience.com/https-medium-com-radecicdario-next-level-data-visualization-dashboard-app-with-bokeh-flask-c588c9398f98?source=collection_archive---------3----------------------->

视觉化很重要。也许对作为数据分析师/科学家的你来说不是，你可能对查看表格数据并从中得出结论感到自信。但通常情况下，你需要向你的老板和/或大学教授(如果你还在上大学的话)展示你的作品。不像你，他们可能不太熟悉数据，如果你向他们抛出一堆数字，他们会感到困惑。

为了避免这种情况，一种方法是通过 ***Matplotlib*** 和/或 ***Seaborn*** 生成图表，并将您的上级图表直接显示在 ***Jupyter 笔记本*** 中或作为保存的 PNG 文件。虽然这绝不是展示你成绩的一个坏方法，但它肯定不会让你的上司目瞪口呆，因此，你将无法在人群中脱颖而出。

值得庆幸的是，有一个非常酷的库叫做 ***Bokeh*** ，这篇文章将介绍创建一个简单的仪表板应用程序的过程，这个程序很可能会让你的上级感到惊讶。不仅如此，您的上级将能够手动过滤或深入查看数据，而不仅仅是查看图表。

![](img/881df8f3100fbe3d9d3e1be28a7a2a2a.png)

这里要注意一点，**这不是一个关于如何使用散景或 Flask** 的教程。这篇文章只讲述了如何通过组合这些库来实现数据可视化。在开始之前，我假设您至少对以下技术有所了解:

*   *HTML*
*   *CSS*
*   *烧瓶(蟒蛇)*
*   *散景(我用的是 1.2.0 版本)*

并且能够流利地:

*   *Python(循环和条件)*
*   *熊猫/Numpy 栈*

如果你的 Python 不是很好，你仍然可以跟上，但是我不能保证你会理解所有的东西。在整篇文章中，我不会解释任何的 ***HTML*** 和 ***CSS*** 片段，并将主要关注 ***散景*** 和 ***烧瓶*** 。

**我将使用著名的** [**泰坦尼克号数据集**](https://raw.githubusercontent.com/datasciencedojo/datasets/master/titanic.csv) **。如果过程中卡了，可以在我的**[**GitHub Repo**](https://github.com/dradecic/Python-Bokeh-Flask--Dashboard-App)**获取代码。**

这是你要做的。请注意下拉菜单以及值是如何变化的。重新应用时，图表会根据相关的乘客等级重新绘制(不过，我为糟糕的设计技巧道歉)。

![](img/1a2f6350ff527ff5cbaa641b81cbb84e.png)

Final Dashboard

好了，事不宜迟，我们开始吧！

# 第一部分。创建目录和文件

首先，在你的电脑上创建一个空的根目录，并随意命名。在其中，创建以下文件和文件夹:

![](img/32d5f503acca9d9e5f13c27b16054565.png)

其中 ***下一级可视化*** 是我的根目录名。然后你可以在根目录下打开你喜欢的代码编辑器，我的是 [Visual Studio Code](https://code.visualstudio.com/) 。

# 第二部分。HTML 和 CSS

现在，让我们来关注一下 ***HTML*** 和 ***CSS*** 文件。我只是把要点放在这里，你可以随意复制。这些是为了把所有东西放在它应该在的地方，并适当地设计它的样式。如果你懂一些网页设计的话，可以看到我用的是 ***Bootstrap*** ，所以设计部分所需的工作量是最小的。

[https://gist.github.com/dradecic/a024729623eefddd02a3625b36598b7a](https://gist.github.com/dradecic/a024729623eefddd02a3625b36598b7a)

[https://gist.github.com/dradecic/4c75e01a2c5077409f7e394538358042](https://gist.github.com/dradecic/4c75e01a2c5077409f7e394538358042)

[https://gist.github.com/dradecic/04988be18320968a814a17f1a4c3ffa9](https://gist.github.com/dradecic/04988be18320968a814a17f1a4c3ffa9)

[https://gist.github.com/dradecic/c06b8139fe018aae6067f1170833e9d0](https://gist.github.com/dradecic/c06b8139fe018aae6067f1170833e9d0)

# 第三部分。Python 设置

现在你需要做的是一些基本的内务处理，导入模块，声明几个常量和帮助函数，最后设置基本的 ***Flask*** 应用程序。

我先从进口说起，有不少。你将需要 ***Numpy*** 和 ***熊猫*** 当然还有很多来自 ***散景*** 的东西。您还需要导入 ***Flask*** ，这样图表就可以呈现到已经定义好的 ***HTML*** 模板中。在这里，您还将加载到数据集中，并在其中创建新的属性，该属性被称为“*头衔*，并将表示此人的头衔——就像**先生**。，**错过**。等。请记住，如果您没有按照我的指示创建目录和文件结构，您可能需要修改 CSV 文件的位置。

[https://gist.github.com/dradecic/0cf3ac6cea2ca11b3c7831550b268b26](https://gist.github.com/dradecic/0cf3ac6cea2ca11b3c7831550b268b26)

现在您需要定义一些常量属性——这不是必须的，但是可以让您以后更容易地更改。我所说的常量属性指的是图表上使用的字体、标题大小、刻度大小、填充等等。
在这里你还将声明两个辅助函数，第一个是***【palette _ generator()***，它将帮助你对图表上的条形进行动态着色。它基本上会返回一个足够长的颜色列表。
下一个辅助函数调用 ***plot_styler()*** 。它将使用先前定义的常量属性，并为图表设置各种属性，如字体大小等。之所以定义它，是因为否则您需要手动设置每个单独地块的样式。

[https://gist.github.com/dradecic/380da5fb3ce3a979e33e7e8ce7e6e14b](https://gist.github.com/dradecic/380da5fb3ce3a979e33e7e8ce7e6e14b)

最后，您需要为 ***烧瓶*** 应用程序定义一个基本结构，如下所示:

[https://gist.github.com/dradecic/c63cc8211a19a9087e438509ef8d2f02](https://gist.github.com/dradecic/c63cc8211a19a9087e438509ef8d2f02)

这将使您能够从终端/CMD 运行 app.py，并在 *localhost:5000* 上查看应用程序，但我不建议现在就运行应用程序。

Python 设置现在已经完成，您可以继续下一部分了。

# 第四部分。定义图表生成函数

这个仪表板应用程序将由 3 个图表组成。是的，我知道这不是很多。但是你可以自己创建额外的图表，这三个图表将帮助你了解它。

如果您想知道将以下函数的代码放在哪里，请将其放在 ***chart()*** 函数之后。

第一个将基于“*幸存*”列的值生成一个条形图——因为只有两个选项，我不会将图表放大，并将它放在一些虚拟文本的右侧(参见**index.html**)。该函数将根据乘客所处的等级对数据集进行子集划分，并输出这些表示的条形图。我还添加了**悬停工具**来使图表具有交互性。这里你将使用早先定义的***plot _ styler()***函数来节省一些时间和空间。
本身功能代码不难理解，如果你对 ***熊猫*** 和 ***散景*** :

[https://gist.github.com/dradecic/9ac65a6aea88070ca77ead136b5bb9ae](https://gist.github.com/dradecic/9ac65a6aea88070ca77ead136b5bb9ae)

下一个函数将为乘客标题生成一个图表。记得我们之前做了一个新的属性' *Title* '，包含了像**先生**这样的头衔。，**错过**。等。现在，我们的目标是可视化特定乘客等级中有多少具有特定头衔的乘客。
***class _ titles _ bar _ chart()***大部分与前一个相似，因为两者都显示条形图，所以确实不需要额外解释。

[https://gist.github.com/dradecic/788278cc89f043a69f3e94b363924467](https://gist.github.com/dradecic/788278cc89f043a69f3e94b363924467)

最后一个函数， ***age_hist()*** 将呈现给定乘客类别的“*年龄*列的直方图。 ***散景*** 中的直方图比 ***Matplotlib*** 中的直方图要难一些。首先，您需要用 ***Numpy*** 创建一个直方图，它将返回直方图本身，以及边缘的位置。所有这些都将被传递到 **ColumnDataSource** ，边缘被进一步分为左边缘和右边缘(这对绘图交互性很有用——您可以获得有关 bin 宽度的信息)。
从那以后，该过程与 ***散景*** 中的任何其他图表类型非常相似。

[https://gist.github.com/dradecic/7aceddf4176fec2bfd1011326e25ef15](https://gist.github.com/dradecic/7aceddf4176fec2bfd1011326e25ef15)

图表生成函数现在已经创建好了，现在是时候将所有内容显示到屏幕上了。

# 第五部分。把这一切联系在一起

好了，这是你期待已久的部分。我知道要写很多代码，但是我希望即使你没有丰富的经验，也能理解。

在当前状态下，屏幕上不会呈现任何内容。这是因为你已经声明了图表绘制函数，但你没有从任何地方调用它们。现在我要你声明一个名为 ***redraw()*** 的新函数，它将产生 3 个变量，每个变量将保存 3 个图表绘制函数的返回值。然后你只需归还它们:

[https://gist.github.com/dradecic/04c0e24e9eb5116091f6849c34d229ee](https://gist.github.com/dradecic/04c0e24e9eb5116091f6849c34d229ee)

完成后，剩下唯一要做的就是编辑主要路线。由于仪表板在一个表单中有一个下拉菜单，所以 route 需要有提供给它的 **GET** 和 **POST** 方法。然后，您可以获取下拉列表的值。
作为边缘情况检查，如果值不知何故为 0 或不存在，您将需要为乘客类 **1** 调用 *redraw()* 函数。
如果该值存在，再次调用 ***redraw()*** ，但是这里传入取来的值。

现在，您有 3 个变量代表当前乘客等级的图表。从这里你需要使用散景的**组件**模块创建**脚本**和 **div** 。现在您将返回 HTML 文件名，即**index.html**，以及那 3 个脚本和 3 个 div:

如果你看一下**index.html**你会看到 div 和脚本是如何放置的。基本上就是这样。整个代码都写好了，除了启动应用程序之外，没有什么要做的了。

# 第六部分。启动应用程序和最后的话

如果你以前从未编写过 Flask 应用程序，你不必担心。发射的过程非常简单。

导航到您的应用程序的根目录，并打开终端/CMD 窗口。从那里，简单地写:

***python app . py***

其中 ***app.py*** 代表 Python 文件的名称。如果你的名字与众不同，请更改它。现在，该应用程序正在运行，您可以在以下网址看到它:

***http://***[***http://localhost:5000/***](http://localhost:5000/)

基本就是这样。现在，您有了一个创建更复杂的仪表板应用程序(当然设计得更好)的起始模板。我希望你已经学到了一些东西，这将有助于你脱颖而出，与你所处的环境无关。

直到下一次…

*喜欢这篇文章吗？成为* [*中等会员*](https://medium.com/@radecicdario/membership) *继续无限制学习。如果你使用下面的链接，我会收到你的一部分会员费，不需要你额外付费。*

[](https://medium.com/@radecicdario/membership) [## 通过我的推荐链接加入 Medium-Dario rade ci

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

medium.com](https://medium.com/@radecicdario/membership)