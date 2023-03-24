# 用 Python 和 BeautifulSoup 刮汉萨德

> 原文：<https://towardsdatascience.com/scraping-hansard-with-python-and-beautifulsoup-f2887f0bc937?source=collection_archive---------31----------------------->

## 使用网络搜集脚本自动收集议会官方报告中的数据

![](img/ae52749d605d7b39baeaad71c9aeb370.png)

Photo by [Deniz Fuchidzhiev](https://unsplash.com/@dfuchidzhiev?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

请注意:我是以个人身份写这篇文章的。表达的任何观点都不是我雇主的观点。

我最近有一个问题:众议院议长约翰·伯科主持了多少次首相问答时间(PMQs)？

通常我会通过搜索议会资料来回答这样的问题，这种搜索服务包含议员的贡献、议会会议记录、议会文件以及下议院和上议院图书馆的简报。

这项服务可在[https://search-material . parliament . uk](https://search-material.parliament.uk.)获得

对于这个特定的问题，我无法从搜索议会材料中得到这个问题的直接答案，原因可以填充几个中等职位。相反，我决定通过网上搜集英国议会议事录来寻找答案。

我从一个粗略的计划开始，我认为我可以找到 John Bercow 主持了多少次 PMQs 会议:

1.  在[https://hansard . parliament . uk](https://hansard.parliament.uk)上查找约翰·伯科任职期间发生的所有 pmq 记录
2.  对于这些 PMQs 记录中的每一条，浏览页面并找到一位演讲者所做的贡献。
3.  对于演讲者所做的每项贡献，提取该人的具体姓名、他们做出贡献的日期以及当天 PMQ 记录的 URL。
4.  计算返回的唯一 PMQs 记录的数量，并从该数量中减去主持会议的发言人不是 John Bercow(即副发言人)的记录。

在我进入代码之前，我应该指出，在开始这个项目之前，我知道一些事情，它们在我的道路上帮助了我:

*   我知道英国议会议事录中所有的 PMQs 记录都有一个标准的名字:他们都被称为“首相”。
*   我知道演讲者所做的贡献总是被列为演讲者，而不是成员的名字。
*   我知道当‘partial = True’放在 URL 的末尾时，Hansard HTML 更容易阅读。

查询议会信息通常需要知道数据是如何格式化的，以及信息是如何存储的。我发现 Twitter 是一个询问有关议会数据问题的好地方，那里有很多知识渊博的人很乐意提供帮助。

对，对代码。

## 需要的包

这些是我用过的包

```
import csv
from bs4 import BeautifulSoup
import pandas as pd
import requests
```

*   csv 允许您操作和创建 csv 文件。
*   BeautifulSoup 是网络抓取库。
*   熊猫将被用来创建一个数据框架，把我们的结果放到一个表中。
*   Requests 用于发送 HTTP 请求；查找网页并返回其内容。

## 找到所有的 PMQ 记录

首先，我需要为英国首相收集所有的英国议会议事录辩论记录。

```
hansardurls = []
for i in range(1,20):
    url = '[https://hansard.parliament.uk/search/Debates?endDate=2019-10-28&house=Commons&searchTerm=%22Prime+Minister%22&startDate=2009-06-23&page={}&partial=true'.format(i)](https://hansard.parliament.uk/search/Debates?endDate=2019-10-28&house=Commons&searchTerm=%22Prime+Minister%22&startDate=2009-06-23&page={}&partial=true'.format(i))
    rall = requests.get(url)
    r = rall.content
    soup = BeautifulSoup(r,"lxml")
    titles = soup.find_all('a',class_="no-underline")
    for t in titles:
        if t['title'].lower() == "prime minister [house of commons]":
            hurl = '[https://hansard.parliament.uk'+t['href'](https://hansard.parliament.uk/'+t['href')]
            hansardurls.append(hurl)
print(len(hansardurls))
```

我打开 https://hansard . parliament . uk，点击查找辩论，然后搜索约翰·伯科担任议长期间的“首相”。这给了我一组长达 19 页的搜索结果。

![](img/1766d801baddf2428be8789e94fe5c18.png)

Results set from Hansard

代码首先创建一个名为‘hansardurls’的空列表，我将 PMQ 记录的 URL 放在那里。

然后代码在 19 页的搜索结果中循环。每个循环执行以下操作:

*   请求 URL 并返回所请求页面的内容
*   使用 BeautifulSoup 处理 HTML
*   在 HTML 中搜索以找到该页面上每个辩论的链接，并给它们分配变量“title”。我进入结果页面，右键单击一个辩论标题，然后选择“Inspect ”,找到了这些链接。这会将您带到控制台，向您显示该页面元素的 HTML。
*   查找仅包含“首相”的头衔。有一些辩论题目提到了首相，但不是 pmq，所以这些被排除了
*   有些标题是大写的，所以所有的标题都用小写。降低()以捕捉所有相关的辩论标题
*   “if”语句是说，如果辩论的标题是“首相”，则为该辩论创建一个完整的 URL，并将其添加到名为“hansardurls”的列表中。title 元素中的链接前面没有“hansard.parliament.uk ”,这需要首先添加以使它们成为可用的链接。
*   最后一行打印出找到并添加到列表中的 PMQ URL 的数量: **327** 。

## **导出 PMQ 网址**

我想在一个单独的 CSV PMQ 的网址，原因我将进入稍后。

```
with open(‘hansardurls.csv’, ‘w’, newline=’’) as myfile:
 wr = csv.writer(myfile, quoting=csv.QUOTE_ALL)
 wr.writerow(hansardurls)
```

这段代码创建了一个名为“hansardurls.csv”的文件，并将列表“hansardurls”中的每个对象写入该 csv 中的单独一行。

## 查找每次 PMQ 会议的演讲者贡献

现在我有了约翰·伯科担任议长期间所有 PMQ 会议的链接，我想看看每个 pmq，看看是否有一位议长做出了贡献，如果有，是约翰·伯科还是他的副手。

```
Speakercontrib = []
Speakingtime = []
urloftime = []
```

首先，我创建了 3 个单独的列表:一个保存在提问时间发言的人的名字，一个保存成员发言的日期，一个保存他们发言的 pmq 的 URL。

```
for h in hansardurls:
    rall = requests.get(h)
    r = rall.content
    soup = BeautifulSoup(r,"lxml")
    time = soup.find('div',class_="col-xs-12 debate-date").text
    contributors = soup.find_all('h2',class_="memberLink") 
```

然后，我编写了一个 for 循环来请求每个 PMQs URL，返回该页面的 HTML，然后使用 BeautifulSoup 来查找页面上对应于发言成员姓名的元素。我通过打开一个 PMQs 页面，右键单击一个成员的名字，找到这个名字是什么类型的元素以及这个元素的类，找到了这个问题。

![](img/eaccbf1536067052a883f0b7e9990a04.png)

The screen you see after clicking inspect on a page element.

```
for c in contributors:
        link = c.find('a')
        try:
            member = link.text
        except: 
            print(c)
        if "Speaker" in member:
            Speakercontrib.append(member)
            Speakingtime.append(time)
            urloftime.append(h)
```

然后，我对 PMQs 页面上每个成员的名字运行了一个 for 循环。该循环执行以下操作:

*   在成员的名称页面元素中查找链接。包含成员姓名文本和 ID 的实际链接嵌套在 h2 标记中，如上面的屏幕截图所示。
*   尝试找到包含成员姓名的链接文本。我在这里使用了一个 try 语句，因为在《英国议会议事录》中有一些投稿没有附上姓名。这些通常是当几个成员同时站起来或叫出来。如果没有 try 语句，这个脚本在遇到“Hon. Members rose”投稿时就会停止。该语句打印出成员名称，让我知道这是不是“Member rose”的贡献，或者是否有其他错误发生。

![](img/d5a4c737311006e66d0c242374141a6e.png)

An example of the phantom contribution, “Members rose — “

*   运行一个 if 语句，如果成员的名字包含“Speaker ”,将名字、投稿日期和页面的 URL 添加到各自的列表中。

## 创建表格

所有数据都是从英国议会议事录中收集来的，但这些数据是在 3 个不同的列表中。我使用 Pandas 创建了一个数据框架，并将 3 个列表添加到数据框架中，以创建一个可用的结果表。

```
speakersdf = pd.DataFrame(
    {'Date': Speakingtime,
     'Speaker': Speakercontrib,
     'url': urloftime
    })
```

这段代码创建了一个名为“speakersdf”的数据帧，并将三个列表作为列添加进来。

然后，我将数据帧导出为 CSV 文件:

```
speakersdf.to_csv('speakerspmqdf.csv')
```

我浏览了一下电子表格，发现约翰·伯科在 319 个下午问中讲过话，副议长林赛·霍伊尔在 1 个下午问中讲过话。

但是等等…有 327 个 PMQs 网址…*少了 7 个 PMQs】。*

这就是我在单独的文档中导出 PMQs URLs 的原因。我从 speakersdf 表中取出 URL，并将其与 hansardurls 列表中的 URL 进行比较(我在 Excel 中完成了此操作)，找到了 7 个缺失的 PMQs URLs。

当我进入这 7 个网页时，我发现演讲者在这 7 次 PMQ 会议中没有做出贡献，所以他们没有被脚本选中。在这种情况下，我会手动查看议事录，找出在 PMQs 开始前是哪位发言者在主持会议。

最终结果:
**约翰·伯科在任内主持了 326 场 PMQ 会议(至 2019 年 10 月 28 日)***

* *这取决于我从英国议会议事录中收集的数据是否完整和正确，以及我收集这些数据的工作流程是否准确。请不要把它当作真理。*

## 如何改进这段代码？

这段代码并不完美。

我想改进的代码部分:

*   引入一个函数，该函数在查找演讲者贡献时自动对照收集的 URL 列表检查 hansardurls 列表，以找到丢失的 URL。
*   我想用演讲者的贡献创建数据框架，而不需要先创建和填充 3 个单独的列表。

如果你认为这段代码可以在其他方面改进，请告诉我。我仍然在学习，非常感谢任何建议或指导。