# 帮助读者重构 Python 网页抓取。

> 原文：<https://towardsdatascience.com/helping-a-reader-with-python-web-scraping-refactored-c14d609134a3?source=collection_archive---------33----------------------->

![](img/aee50693bd7ba0b86dbb2b2bf9716e72.png)

来自印度的微生物学家 Bhargava Reddy Morampalli 从我的旧博客上读到了我的第一篇关于网络抓取的文章。如果你没有机会看到这篇文章，你可以在这里阅读。

[](/python-web-scraping-refactored-5834dda39a65) [## Python 网络抓取重构

### 我在旧博客上的第一篇文章是关于一个网络抓取的例子。网络抓取是运用 API 之外的一种手段…

towardsdatascience.com](/python-web-scraping-refactored-5834dda39a65) 

他在阅读了我的帖子后联系了我，请求我帮助他实现基于我在这个网站上的文章的网络抓取。在这里，我将像我们在上面的文章中所做的那样重构 Bhargava 的例子。我认为这是有用的，因为不是所有的网站都有相同的结构。

![](img/81a691971d588facfa406665a9dd6daf.png)

Made with imgflip.com

* *注意:由于网站改变了布局(或者员工可能会改变)，您得到的结果可能会有所不同。最终结果截至 2019 年 8 月 21 日。对于那些阅读我的这篇文章的原始版本的人来说，站点结构已经改变，所以这里的代码会略有不同。**

## 我们要废弃的网站。

该网站也是一个列出学术机构员工的页面。请务必看一看它，看看网站是如何构建的。在这里，Bhargava 想要从每个教授那里删除以下信息。

1.  名字
2.  指定
3.  电子邮件
4.  资格
5.  出版物的数量

和上一篇文章一样，我将遵循同样的步骤，这样你就可以加强对这个过程的理解(如果你需要的话)。

## 导入我们的图书馆和网站。

正如我们之前所做的，我们将导入相同的库，并让我们的`web_page`被请求拉入。

## 研究 HTML 并做初步的废弃。

接下来，我们将调查 HTML，看看我们的教授藏在哪里。同样，如果您右键单击站点并选择“查看页面源代码”来查看 HTML 代码。然后，我们将使用 ctrl-f 搜索我们的第一位教授 Pramod Kumar T M .博士。

![](img/d950feec812c3ffb4b86bd0151ab9675.png)

Found our first Professor…deep in HTML tags…

所以我们找到了他。但问题是我们如何得到他的名字和其他信息。如果您进一步探究，您会发现教授们被包含在`tab-pane active in fade` div 类标签中。因此，为了从废弃中获得我们的结果，我们将首先使用 div 类提取所有标签。

我们还将检查结果的数量，以确保我们得到正确的数量。你应该得到 17 个结果。

## 让我们看看从哪里获得我们的数据！

我们将使用第一个结果来研究从哪里获得我们的信息。然后我们将应用列表理解来创建我们的`pandas`数据框架。如果你检查结果[0](我们的第一个教授的个人资料)，我们会得到一吨的 HTML。浏览时，您应该会发现以下内容:

```
<div class="col-sm-9">
<h2>Dr Pramod Kumar T M</h2>
<p>
<strong>Designation:</strong> Principal<br/>
<strong>Email Id:</strong> 
pramodkumar@jssuni.edu.in<br/>
<strong>Qualification:</strong> 
B. Pharma, M.Pharma, Ph.D.<br/>
<strong>No of Publication:</strong> 121
</p>
</div>
```

同样，我们希望获得每位教授的姓名、头衔、电子邮件、资格和发表的论文数量。再次尝试自己确定哪些标签包含了我们需要的所有信息。然后继续读下去，看看你是否正确。

![](img/06f982d10d383347fd5d6a1bce5c29f2.png)

Image from [BDDroppings](https://bulldogsdroppings.com/2017/11/23/an-essential-tremor-journey-decisions-decisions/).

*名称*:在`<h2>`标签之间。

*剩下的四个项目*:这有点棘手。你可能会认为我们的一些信息包含在`<strong>`标签中。但是如果你试着用`find`函数运行你的代码，并把“强”作为参数，你会发现这不是一个成功的方法。具体来说，它将返回第一对具有字符串“Designation:”的强标签，这不是我们想要的……另外，`<br>`标签对我们没有帮助，因为它们没有包含任何信息。

再看一下这一步中的 HTML 代码。我重新安排了它，这样你就可以看到有一个标签正确地包含了我们想要的所有信息。你看到了吗？

![](img/da961ff0c9718461631d4e2cd4f90762.png)

Image made at imgflip.com

那是`<p>`的标记。让我们转到测试代码，看看如何从`<p>`标签中检索信息。

## 用第一份侧写测试我们的报废。

用第一个条目创建我们的`test_result`，上面是检索该条目所有信息的代码。

注意，为了检索最后四条信息，我们使用了 BeautifulSoup4 函数`.find(‘p’)`和`.contents`。

如果您运行`test_result.find(‘p’).contents`，您将看到以下内容:

```
[<strong>Designation:</strong>,
 ' Principal',
 <br/>,
 <strong>Email Id:</strong>,
 ' pramodkumar@jssuni.edu.in',
 <br/>,
 <strong>Qualification:</strong>,
 ' B. Pharma, M.Pharma, Ph.D.',
 <br/>,
 <strong>No of Publication:</strong>,
 ' 121']
```

我们得到的回报是我们的`<p>`标签之间的所有字符串和标签作为上面列表的元素！这样，我们就可以将每个所需项目的索引附加到我们的代码行中以获得它们。具体来说:

*名称*:索引 1

*电子邮件*:索引 4

*资格*:指标 7

*出版数量*:指数 10(或者指数 1，取决于你想怎么做)

当您提取信息时，您可能会注意到在返回的字符串的开头有额外的空格(您也可以在上面的列表中看到)。这种额外的间距会对您将来的数据准备产生负面影响。在数据框中搜索字符串数据时，Python 对此非常敏感。额外的间距会导致您丢失一些您需要的项目，从而产生错误的总数或平均值。从而使你的分析不准确。为了处理这个问题，我们附加了`strip`方法来删除字符串开头(和结尾，你知道，只是以防万一)的多余空格。

正如 Bhargava 希望我说的，我们希望对所有条目都这样做，不管它们的总数是多少。因此，现在我们将初始化一个`pandas`数据帧，并应用列表理解来生成列。

## 提取数据并创建数据框。

正如我们上次所做的，我们将为每一列数据创建列表理解。对于出版物的数量，我们必须创建一个特殊的函数，因为有些教授没有任何出版物。它们的默认值为 0。最后，我们将把收集的数据导出到一个 CSV 文件中。

## 了解我们的数据概况。

我们可以单独创建图表来分析我们拥有的新数据。但是最近有了一种新的更快的方法来获得我们数据的概观。那是一个叫做[熊猫简介](https://github.com/pandas-profiling/pandas-profiling)的新包装。查看文档，快速开始使用这个非常方便的工具。

## 最后的想法

这就结束了我们的第二轮网络搜集。对于扩展，我们可以查看这所大学的其他系，并删除他们的信息。菜单上列出的与我们的网页格式相同。我们可以用同样的方式导入所有的网页(当然是使用函数)，然后创建另一个函数来整理脚本。还有额外的信息，如教授的工作历史和他们的教育，这也可能是有趣的探索，看看是否有重叠的背景。这也将为 NLP 实践提供机会。

如果您想了解更多关于[请求](http://docs.python-requests.org/en/master/)和 [beautifulsoup4](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) 库的信息，请点击它们名称上的链接查看它们的文档。

## 感谢

我很高兴 Bhargava 读了我的第一篇博客文章，并安慰地向我寻求帮助，帮助他尝试网络抓取。他确实寻求帮助的事实表明，如果你不怕问，你可以找到你需要的信息，而且我们生活在世界的不同一边。

犯错误和不知道事情是生活的一部分。正是从提问和经验(有时是网上搜索)中学习让我们成长。在我自己抓取了网页之后，我让 Bhargava 查看完整的代码或者提供提示(通过 LinkedIn)。他选择了后者，这样他可以获得经验和成长。

![](img/12055e19ff5518f47763f145034bacfd.png)

Image from mindgasms.theblogpress.com

花点时间感谢 Bhargava，无论是在下面的评论中，在他的 LinkedIn 上，还是在 T2 的 Twitter 上。

直到下一次，

约翰·德杰苏斯