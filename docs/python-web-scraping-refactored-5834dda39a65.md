# Python 网络抓取重构

> 原文：<https://towardsdatascience.com/python-web-scraping-refactored-5834dda39a65?source=collection_archive---------14----------------------->

我在旧博客上的第一篇文章是关于一个网络抓取的例子。网络抓取是使用 API 之外的一种方式，例如 [tweepy](http://www.tweepy.org/) 来为你的分析或你的机器学习模型获取信息。真实数据并不总是在一个方便的 CSV 文件中。你需要自己去拿。

我学到的例子来自凯文·马卡姆在 Youtube 上的[网络抓取演示视频(Twitter 标签:](https://www.youtube.com/watch?v=r_xb0vF1uMc&list=PL5-da3qGB5IDbOi0g5WFh1YPDNzXw4LNL) [@justmarkham](https://twitter.com/justmarkham?lang=en) )。他的例子使用了一个循环来收集所有的信息。但是 Python 有一个比循环更有效、可读性更强的工具。那些是[列表理解](https://docs.python.org/3/tutorial/datastructures.html)。如果你不确定这些是什么，看看下面的帖子了解更多。一旦你完成了，请继续阅读，学习如何通过从这所[布朗克斯高中](https://www.newvisions.org/ams2/pages/our-staff2)的职员页面上搜集姓名、职位和电子邮件地址来有效地进行网络搜集。

 [## Python 基础-列表理解

### 学习 python 几周后，我开始真正掌握了窍门。我喜欢有条件的心流，而且…

medium.com](https://medium.com/@erikgreenj/python-basics-list-comprehensions-30ef0df40fea) 

**注。根据您阅读本文的时间，最终结果可能会有所不同，因为网页上的数据可能已经发生了变化。最终结果截至 2019 年 4 月 3 日***

# 酷！我们如何开始？

让我们从导入和获取网页开始。

我们导入`requests`来获得带有`requests.get()`的 web_page。`pandas`稍后将用于清理我们的报废数据。`bs4`库中的`BeautifulSoup`将用于解析我们的糊状 HTML，以帮助我们获得我们想要的信息。我们将在下面这样做。

![](img/3e44a8760807963cc3b55dbde385a1a7.png)

Photo by [Elli O.](https://unsplash.com/@oelli?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

接下来，我们将研究来自`web_page`的 HTML 代码，以找到我们需要的代码段。要查看网页的 HTML 代码，请转到该网页，右键单击并选择“查看页面源代码”。然后，您可以按 ctrl-f 找到一个职员的名字，以查看嵌入了他们的名字和信息的 HTML 代码。

如果您稍微滚动一下代码，您应该会注意到代码行中包含了一些信息，例如:

`<title>……</title>`

或者

`<p>…..</p>`

这些在 HTML 代码中被称为标签。在这些标签之间隐藏着我们想要获取的信息。

由于我们看到所需的信息在`<div>`标签和`class=’matrix-content’`标签之间，我们可以假设所有教师的信息都在该类的每个标签中。这就是为什么我们使用标签和类作为 soup 的`find_all`属性的参数。

我们需要从第一个教师简介出现的索引开始，因为我们只收集教师信息。第一个出场的老师是“布罗根先生”。你可以使用 ctrl-f 在 HTML 代码中搜索他的名字。如果数的话(当然是从 0 开始)，布罗根先生的指数是 29。这就是为什么我们从指数 29 开始重新定义结果。对结果长度的检查和对被移除的员工的心算证实了我们可以进入下一步了！

# 现在获取所有教师数据！对吗？

![](img/f661e1488fe16f5d66ef82c34c820806.png)

Photo by [Florian Olivo](https://unsplash.com/@rxspawn?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

我们会的。在此之前，我们应该先看看如何从一个老师那里获得我们想要的信息。然后我们将把它推广到我们的理解列表中。让我们来看一个教师简介的 HTML 代码。我们将再次检查布洛根先生的信息:

```
<div class="matrix-content">
        <h5>Mr. Brogan</h5>                   
        <div class="matrix-copy"><p>
    Special Education: Geometry, Particular Topics of Geometry</p>
<p>
    <em>rbrogan31@charter.newvisions.org</em></p>
</div>
                                            </div>
```

同样，我们需要确定包含教师姓名、职位和电子邮件的标签。花点时间试着自己回答这个问题，然后继续读下去，看看你是否正确。这将为研究您需要在 python 抓取代码中指示 HTML 的哪些部分提供良好的实践。还记得我之前给你们看的例子吗？

*教师姓名标签:*姓名在标记为`<h5>`的标签之间。

*位置标签*:位置位于类别标签`<div class=”matrix-copy”>`后的`<p>`标签之间。

*邮件标签*:邮件在标签`<p>`和`<em>`之间。由于`<em>`标签直接封装了电子邮件，这就是我们将在抓取代码中指出的标签。

太好了！现在我们已经找到了需要指示的标签，让我们为我们的第一位老师编写代码，以确定我们将如何遍历老师条目来获取所有数据！

首先，我们将我们的第一位老师定义为`test_result`。

*教师姓名:*通过在`<h5>`标签上使用`find`方法，我们得到了带有我们教师姓名的代码行。但这并不能给我们没有标签的名字。我们不希望代码中有标签。因此，为了提取姓名文本，我们将把`.text`添加到`find`方法中，以获得标签的文本属性。

*Position(s)* :我们将使用与名称相同的`find`方法，但是这次我们的参数将是标签`<p>`。这样做可以得到我们的位置，但同样我们不希望附加标签。再次使用`.text`返回以下内容…..

```
'\n\tSpecial Education: Geometry, Particular Topics of Geometry'
```

这给了我们比我们想要的更多。具体来说，我们在开始时得到了新行(`\n`)和制表符(`\t`)的字符串代码。由于我们的信息是在一个字符串中，我们可以用我们的代码行删除不需要的部分，从字符串的任何地方删除这些字符。

电子邮件:获取这些信息要简单得多。再次使用带有标签`<em>`的`find`方法作为我们的参数。使用`.get_text()`方法有助于我们做到这一点，因为一些电子邮件嵌入在多个`<em>`标签中。

# 现在我们得到了我们想要的所有数据！

没错！所以我们开门见山吧。

首先，我们初始化一个数据框对象。然后，我们使用列表理解结合来自`test_result`的代码来获取所有教师的姓名和职位。我们还利用这些列表理解来创建数据帧`df`的前两列。

当我第一次运行电子邮件收集的代码时，我遇到了一个属性错误。这就是变量管理器方便的地方。检查网页或 HTML 代码会发现“Veninga 女士”在`<em>` 标签中没有电子邮件地址。它位于第二组`<p>`标签之间。因为页面很小，你可以这样做，但是对于较大的信息集合，你最好在列表理解产生错误的地方打印。

为了解决这个问题，我们将尝试创建一个`get_email`函数，除了用`<p>`上的`find_all`方法设置第二组`<p>`标签内的所有电子邮件，然后使用索引来获得我们想要的`<p>`标签。我们还将删除多余的文本，就像获取职位信息一样。

与此同时，其他人的电子邮件将照常被删除。再次运行代码使我们能够成功地获得所有条目。可以通过检查记录列表的长度来证明这一点(它应该返回 66)。您可以使用`df.shape[0]`来检查您的数据中的行数(66 个教师的 66 行)。

# 真快！我们去分析这些数据吧！

我们可以…但是我们会发现我们收集的数据有错误。您可以检查的一件事是，如果您有重复的条目，并删除它们。有些教师可能会教授多个科目(如数学和英语)，因此他们的名字会出现多次。

通过对`df.duplicated`的所有布尔值求和，我们得到值 11。所以我们有 11 个老师的名字出现了不止一次。然后，我们使用`df.drop_duplicates`保留教师姓名的第一个条目，并丢弃其余条目。最后，我们将数据帧导出到一个 CSV 文件中，用于将来的分析。

# 最后的想法

网络抓取给你一种神奇的感觉，因为一旦你找到你需要的标签，你就可以从任何网站获取信息。我希望这个演练对那些考虑学习如何用 Python web scrap 的人有所帮助。

当然，可以做一些特征工程来帮助分析。我不打算包括这些细节，因为我想把重点放在网页抓取方面。

选项包括:

1.创建一个性别列，方法是在该时期按标题拆分姓名，然后使用 pandas 将标题映射到适当的性别。

2.将职位分开(因为大多数教师似乎教授不止一种类型的班级)。

对于熊猫练习，你可以尝试自己做上面的。

![](img/ba94754382a80f3249bc14c895a97855.png)

Photo by [Thao Le Hoang](https://unsplash.com/@h4x0r3?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

然后，我们可以转向 Tableau 或`matplotlib`进行可视化和统计，以回答与教师人数和其他特许公立布朗克斯学校相比，这些数据的问题。

直到下一次，

约翰·德杰苏斯