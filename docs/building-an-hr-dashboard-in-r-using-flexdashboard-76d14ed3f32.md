# 使用 Flexdashboard 在 R 中构建 HR 仪表板

> 原文：<https://towardsdatascience.com/building-an-hr-dashboard-in-r-using-flexdashboard-76d14ed3f32?source=collection_archive---------0----------------------->

![](img/a85746c7896118e5ddef76f556844261.png)![](img/af2b02616636009dc2e6b9d6470bbc39.png)![](img/2bde45ef9941c98f271f8c2d4d3f0885.png)

The HR Dashboard

假设你是一名人力资源经理，你的老板要求你为首席执行官准备一份关于本年度公司员工招聘和流失的演示文稿。哦，你很兴奋！你已经在这个项目上工作了一段时间，并且已经整理好了所有的数据。你迫不及待地想用你的洞察力让 CEO 眼花缭乱。但是等一下，她非常忙，而你只有五分钟的时间。你想给她看的东西时间太少了！

你是做什么的？什么工具对你最有帮助？你会带着 25 张幻灯片参加 5 分钟的会议吗？您是否会将多个 Excel 工作簿投射给一个擅长分析简明数据的人？

请记住:

> 枪战时不要带刀

在这种情况下，我的朋友，你唯一需要的枪就是仪表板。当您希望可视化大量复杂数据，并向用户提供与特定目标或业务流程 [](https://en.wikipedia.org/wiki/Dashboard_(business)) 相关的关键绩效指标的一览视图时，仪表板是完美的工具。

r 给了我们漂亮的软件包“flexdashboard ”,它为我们提供了一个易于创建的动态仪表板。我们将在一个 [R markdown 文件](https://rmarkdown.rstudio.com/articles_intro.html)中写出代码，这是一种在 R 中创建动态文件的文件格式

以下是我们将要介绍的内容:

1.  仪表板结构概述
2.  创建一个 R 降价文件
3.  使用 flexdashboard 的仪表板布局
4.  使用 dplyr 进行数据操作
5.  使用 plotly 合并情节

## 仪表板结构概述

假设您的公司在三个地区拥有子公司:

*   美洲
*   APAC(亚太地区)
*   欧洲

因此，我们有三个区域文件(Excel 工作簿)，其中包含相关的数据字段，如姓名、国家、电子邮件地址等。区域团队为每个新员工和离职(自然减员)输入数据。我们需要一个仪表板，它能有效地为执行管理层可视化这些信息。这包括月度趋势、跨国家和内部职能部门的划分等。

您可以在这里访问包含模拟数据的文件:[https://github . com/sagarkulkarny/HR-Movement-Dashboard-using-flex Dashboard-](https://github.com/selectsagar/HR-Movement-Dashboard-using-Flexdashboard-)

这个想法是读取单个文件，合并它们，然后操纵数据得到图。所以让我们开始吧！

## 创建一个 R 降价文件

打开 RStudio 并创建一个新的 R 脚本文件。使用`install.packages(“flexdashboard”)`安装 flexdashboard 库

软件包安装完成后，通过选择“文件”->“新建文件”->“R markdown”创建一个新的 R Markdown 文件，如下所示:

![](img/0ccab0a6766ac7fcd5b476e0b5194ba1.png)

Fig. 2.1: Create a new R markdown file

系统会提示您输入文件名。现在，如果你想为任何目的创建一个降价文件，你可以点击 OK，你就可以开始了。

![](img/c312c6b76a16e324cbeb4dd6ed12cf2a.png)

Fig. 2.2: Creating a new R markdown file

然而，Flex Dashboard 也为我们的仪表板提供了一个模板。如果您选择“来自模板”，然后选择“Flex Dashboard”，您将获得一个新的带有占位符的降价文件，如下所示:

![](img/b4e8a452110c8972af2470898614a0c2.png)

Fig. 2.3: Selecting flex dashboard template

![](img/b77a5a224622751cf06a2ab85e4a911b.png)

Fig. 2.4 flex dashboard template

您可以看到该文件没有标题，因此我们必须将它保存在我们的系统中。将它保存在与我们的区域文件相同的文件夹中是有意义的，但是您可以将它存储在您喜欢的任何地方。保存文件后，我们可以试着运行它，看看模板占位符有什么作用。

通常，为了运行 R 脚本，我们只需选择代码并在控制台中运行它。然而，R Markdown 文件在控制台之外运行*，因为它们根据文件输出 HTML 内容或 PDF 或 word 文件。要查看这个输出，我们需要通过点击文件顶部附近的编织图标或按下`Ctrl+Shift+K`来"*编织"*文件。*

![](img/8a020f8477cae9fd9a283c83164128e1.png)

Fig 2.5: Basic layout using the template place holders

恭喜你！我们刚刚创建了第一个仪表板布局！它什么都没有的事实应该无关紧要！

## 使用 flexdashboard 的仪表板布局

在这一步中，我们将决定仪表板的布局，并通过在 R markdown 文件中编码来实现它。

我们先概念化一下布局。我们可以有一个单页的仪表板，但这将意味着在一个小空间里塞满了大量的信息。Flexdashboard 为我们提供了一个选项，通过使用`===`标题或仅使用`#`来划分代码中的部分，让[多页](https://rmarkdown.rstudio.com/flexdashboard/using.html#multiple_pages)出现在我们的仪表板上。让我们在模板中尝试一下。让我们在三个位置添加多个页眉:第一个`Column {}`前的“第一页”，第二个`Column{}`前的“第二页”，最后的“第三页”。要命名页面，您只需在`===`标题上方的行中写下名称，如下所示:

编织代码现在在仪表板上显示三页，第一页包含图表 A，第二页包含图表 B 和 C，第三页是空白的。你能相信给仪表板添加“多页”功能如此简单吗？简单是 R Markdown 和 flex dashboard 的旗舰工具之一，使用 R Markdown 意味着您可以简单地完成许多令人惊叹的事情。

在仪表板上有一个主页来显示新员工和自然减员的要点，并为新员工和自然减员提供单独的页面是有意义的，如果我们想深入了解一下，可以去那里。让我们相应地重命名我们的页面，即第 1 页为“Dash”，第 2 页为“New Hires”，第 3 页为“Attrition”。

现在，在一个页面中，布局方向基于行或列(或行和列)。可以使用`---`标题或`##`添加新的行或列。在这个项目中，我将使用行布局。为此，我们将对代码进行两处修改:

1.  将块封装在顶部的`---`中的 YAML 标题中的方向从列改为行
2.  将列/行标题中的所有列更改为行

The YAML header

Column changes to Row

您可以在这里查看可能的选项:[https://rmarkdown . r studio . com/flex dashboard/using . html # layout](https://rmarkdown.rstudio.com/flexdashboard/using.html#layout)

布局的下一步是放置实际的图表。我们可以为地块创建空间，并使用`###`标题后跟空间名称来命名它们。这些空间被称为盒子，你可以在上面的代码中看到名为图表 A、图表 B 和图表 c 的例子。

[总结](https://rmarkdown.rstudio.com/lesson-12.html)布局标题:

1.  多页页眉或一级页眉:`===`或`#`
2.  行列表头或二级表头:`---`或`##`
3.  箱式表头或三级表头:`###`

## 数据操作

现在我们已经有了布局，在开始在仪表板上绘制图表之前，我们需要将数据整理好。到目前为止，我们所有的代码都是 HTML 布局。但是，我们在哪里编写负责所有幕后工作的实际 R 脚本代码呢？

在 R markdown 中，这样的代码需要由分隔符````r{}` 和`````封装，也称为代码块分隔符。R markdown 的工作方式是将代码块末尾的结果嵌入到 HTML dashboard 输出中。

在您的新文件中，您可以看到有四个这样的代码块。第一个代码块在已经有`library(flexdashboard)`的`{r setup}`下。我们将使用这个块来编写代码，以读取和合并我们的区域文件。我们将使用其他块为情节编写代码。

让我们从调用所有必需的包并读取我们各自的数据文件开始:

Packages and reading the data files

上面的代码还读取一个文件“country coord ”,该文件包含数据中国家的经度和纬度数据。在绘制世界地图的时候会很有用。

`rbind()`函数合并来自三个文件的数据，并将它们一个贴在另一个之上，以创建一个全局数据集。建议去过一遍资料，熟悉一下结构。注意数据的类型(连续的、离散的)、列的值(例如，进入/退出在列“运动类型”中被捕获，我们将广泛使用该列来过滤数据)等。

现在我们已经有了必要的数据，我们可以开始用可视化填充我们的仪表板。我们第一页“Dash”上的第一个组件是一个值框。

一个[值框](https://rmarkdown.rstudio.com/flexdashboard/using.html#value_boxes)就像一个标签，显示一个简单的值和一个图标。这些是我们仪表板中的价值箱:

![](img/7f4add2d64a74d10a1843fe5cdd338a6.png)

Fig. 3.1 Value box

我们将在第一个代码块中编写这些代码，如下所示:

让我们一行一行地回顾一下。

*   您会注意到 Dash 页面标题的标签旁边有一个小小的地球图标。那是由`{data-icon=“fa-global”}`获得的。“fa”代表[字体牛逼](https://fontawesome.com/icons?d=gallery&m=free)这是一个免费图标的网站。您还可以查看所有受 Flex Dashboard 支持的免费选项。
*   注意不同的输出使用了不同的标题(页面、列和代码块)。
*   使用 dplyr 的过滤函数和管道运算符计算新雇员数
*   `ValueBox()`函数接受数值、图标、标题和颜色参数。注意页眉中类似于地球图标的图标的`“fa-user-plus”`值。
*   “净变化”值框的逻辑略有不同。由于净变化(新聘人数-离职人数)可能是正数，也可能是负数，我想相应地改变图标，即向上箭头表示正净值，向下箭头表示负净值。这是通过使用 r 中的`if-else`循环完成的

## 使用 plotly 合并情节

在值框之后是我们的两个图:趋势图和世界地图图。

![](img/4d6bbea96b11036a59e81d47bdc68635.png)

Fig 3.2: Movement by month and by region plots

让我们看看下面的代码:

对于“按月移动”图，我们首先按“月”和“移动类型”对数据集进行分组，然后进行总结。这将创建下表:

![](img/1cbb3b0f6fbd64bd9f8eaf6203624c3a.png)

Fig. 3.3: Tibble for “Movement by Region” plot

我们先试着在脑海中构建剧情。我们将月作为 x 轴，计数作为 y 轴，两条线显示趋势，一条代表进场，一条代表出场。一旦我们框定了这个顺序，使用 plotly(或 plot_ly()编写代码块)创建一个情节就非常简单了。

我们需要记住这一点:

*   它试图建立在 ggplot2 所使用的图形语法之上。因此，每个后续函数都将前一个函数作为第一个参数。因此，我们需要为 plotly 函数使用管道操作符`%>%`来代替 ggplot2 中使用的`+`
*   因此，创建上述情节的最简单方法是:

我们包含了`hoverinfo` 和`text` 参数，当鼠标指针悬停在图上时，该图显示计数。

此外，记住通过调用 plot 来结束代码块，在本例中是`p1` ,否则仪表板输出上的绘图空间将保持空白。

我们“Dash”页面上的最后一个图是世界地图图。这个绘图函数将地理坐标(经度和纬度)作为 x 和 y 坐标。因此，我通过执行左连接将“country coord”excel 工作簿中的两列添加到全局数据集。对此的解释包含在代码中。

在 plotly 中指定世界地图的方法是将`locationmode` 参数设置为`“world”`。然后我们将 long 和 lat 值分别作为`x`和`y` 坐标传递。圆的`size` 设置为`count` 的值，而`color` 设置为`Entry` 或`Exit`。

不知不觉中，您已经完成了我们仪表板的整个“Dash”页面！

![](img/9aedeb840d2dd5dfb89b613c2acc1b73.png)

Fig. 3.3: The “Dash” page

与 ggplot2 相比，使用 plotly 的一个好处是前者为我们提供了图形的动态表示，我们可以在其中切换标签，图形相应地做出响应。例如，点击“按月移动”图图例中的“退出”，取消选择它，只显示“进入”趋势。

![](img/e59ddcd38d8dbbd13750a5c0ddf826da.png)

Fig. 3.4: Dynamic nature of plotly plots (you can see that “Exit” has grayed out in the legend)

我填充了接下来的两页，在这两页中，我加入了几个甜甜圈图，作为对通常的值框和条形图的补充。逻辑非常简单，与上面解释的相似。

![](img/1b2073748915d2ae5b205bf4664806a7.png)

Fig. 3.5: New Hire page

![](img/4e3aff0cca68fd899496dd7b9321459c.png)

Fig. 3.6: Attrition page

## 结论

祝贺您创建了第一个仪表板！我希望这是以后更多的开始！

如果您有兴趣进一步研究这个问题，您可以在此处访问整个`rmd` 文件和支持工作簿:[https://github . com/sagarkulkarny/HR-Movement-Dashboard-using-flex Dashboard-](https://github.com/selectsagar/HR-Movement-Dashboard-using-Flexdashboard-)

Flex Dashboard 包含了很多我在这篇文章中没有涉及到的组件，比如标尺、导航条、故事板等等。它是在 r 中创建仪表板的更易于使用和直观的包之一。

我希望我至少能够对所涉及的概念进行一些澄清。不言而喻，我们欢迎并感谢所有的建议和反馈。

## 参考

*   [https://rmarkdown.rstudio.com/flexdashboard/using.html](https://rmarkdown.rstudio.com/flexdashboard/using.html)
*   https://rmarkdown.rstudio.com/lesson-1.html
*   【https://plotly-r.com/ 

附:这是我的第一篇帖子，我感谢大卫·罗宾逊[方差解释](http://varianceexplained.org/r/start-blog/)、[马纳利·辛德](https://medium.com/u/4d1f3e34f42e?source=post_page-----76d14ed3f32--------------------------------)、[德里克·姆维蒂](https://medium.com/u/4b814c3bfc04?source=post_page-----76d14ed3f32--------------------------------)、[朱利安·塔格尔](https://medium.com/u/22b532482f5b?source=post_page-----76d14ed3f32--------------------------------)和[卡特诺里亚](https://medium.com/u/a87f051c374a?source=post_page-----76d14ed3f32--------------------------------)他们令人振奋的文章。如果你喜欢我的帖子，你也会喜欢他们的！