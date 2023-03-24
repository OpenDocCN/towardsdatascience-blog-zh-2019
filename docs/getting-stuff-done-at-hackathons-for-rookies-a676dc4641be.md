# 为新手在黑客马拉松上完成任务

> 原文：<https://towardsdatascience.com/getting-stuff-done-at-hackathons-for-rookies-a676dc4641be?source=collection_archive---------26----------------------->

![](img/670570d3205db03a650a626333902f52.png)

我非常喜欢我的第一次黑客马拉松([你可以从以前的帖子](https://medium.com/@robblatt/learning-the-value-of-scope-during-a-game-jam-558e0da23ba0)中了解我关于 scope 的经历)。通过 BetaNYC，有机会参加所有能力的移动黑客马拉松，这是 2019 年更大的全国公民黑客日的一部分。

我是“可靠访问地铁”团队的成员，与[运输中心](https://transitcenter.org/)和[纽约平等访问联合会](https://twitter.com/EqualAccessNy2)合作，我们的提示是这样的:

> 我们**想要探究电梯和自动扶梯故障的原因，并以阐明模式和洞察解决方案的方式呈现数据。**

如果你要去参加你的第一次黑客马拉松，请记住一些来自一个二年级黑客马拉松的事情，我会在我做的工作之间做这些事情。

# 评估你的团队

立即开始编码很有诱惑力，但是如果你马上投入进去，你将永远不会有作为一个团队取得成功所必需的专注。大家都是技术吗？每个人都用同一种语言写作吗？你的背景是什么？大家对什么感兴趣？

# 了解手头的问题

对我们来说，这个问题比一个下午的工作所能解决的要大得多。我们的团队中有两位非技术主题专家，来自 TransitCenter 的 Colin 和来自 United for Equal Access NY 的 Dustin，他们帮助引导了围绕我们的提示的对话。

MTA 在其季度交通和公共汽车委员会会议上发布 PDF。每份报告都有数百页长，但我们关注的是利用率低于 85%的电梯和自动扶梯的报告。埋在[这份 491 页的 2019 年 Q2 报告](http://web.mta.info/mta/news/books/pdf/190923_1000_Transit.pdf)是第 380-384 页的表格。

每一排被分配到一部电梯或自动扶梯，每一部电梯或自动扶梯都在地铁站，其可用性为季度的%,以及注释。评论是报告中最有问题的部分。以下是哈德逊站 ES622 号自动扶梯的评论示例:

> 自动扶梯于 2019 年 4 月 15 日至 2019 年 4 月 24 日停止运行，以维修和调整 combstop 和碰撞安全装置。由于电线松动，控制器也进行了维修。洒水系统失灵了，引发了洪水。水被抽出来了，喷水系统也修好了；自动扶梯经过测试后重新投入使用。由于安全检查和相关维修工作，自动扶梯于 2019 年 5 月 17 日至 2019 年 5 月 23 日停止运行。调整了 combstop 安全装置，更换并调整了左侧扶手链；自动扶梯经过测试后重新投入使用。

在此期间，有两次停机，每次停机都有多种原因导致停机，并且需要采取多种措施来修复。该信息在其他任何地方都不可用，所以我们的目标是从 PDF 中提取表格。

# 接受任务

使这些信息可用的第一步是从 PDF 中提取，从那里我们可以开始从评论中分离出每个事件。

有人被指派去确定电梯和自动扶梯故障的所有类型的原因以及如何修理它们，有人被指派去尝试构建脚本来解释不同类型的问题和修理。许多人接受任务，寻找不同的方法将 PDF 表格转换成可操作的数据库。

# 低下头，做些工作

你已经得到了你的任务，你知道它将如何发挥更大的作用，现在是时候做一些工作了。在我的例子中，我要从 pdf 创建 csv 文件。

![](img/68f52b9ff67ca9f15c51fcb65480424d.png)

Here’s the first page of the PDF I needed to convert

为了让一切快速运行，我使用了 [PyPDF2](https://pypi.org/project/PyPDF2/) 来读取 PDF。我最初打算使用更复杂的东西来读取文件，但是当我开始编码时，我只有不到四个小时的工作时间来完成一些功能。

PyPDF2 提取了文本并揭示了文件格式的一些问题。这里有一个例子:

```
‘ES235’,
 ’34 St-Herald Sq ‘,
 ‘BDFM84.34%’,
 ‘The escalator was out of service from 12/4/18 to 12/11/18 due to worn out handrail and countershaft chains ‘,
 ‘as well as defective brake assemblies. The countershaft assembly and chain were replaced and adjusted. ‘,
 ‘The right handrail chain was adjusted. The main brakes were replaced and adjusted as well as a controller ‘,
 ‘’,
 ‘relay; the escalator was tested and returned to service. The escalator was out multiple times due to the ‘,
 ‘’,
 ‘activation of various safety devices. Those safety devices were tested and adjusted as needed.’,
```

换行符是硬编码的，地铁线和百分比在同一条线上，评论中有一堆额外的换行符，没有解释原因。很乱，但这是我们能做的。

# 休息一下，吃点零食

记住你和其他人一起参加活动。参加社交活动，如果有食物，你应该去吃一顿饭，并和房间里的其他人联系。这些都是认识人的好方法。

# 问问周围

如果你独自做项目的一部分，你可能会和一群做类似工作的人坐在一起，在这次黑客马拉松中，有几个人朝着相同的目标工作。我碰巧坐在我熨斗学校的同学 [Jen McKaig](https://github.com/jenmckaig) 旁边，能够谈论我们遇到的一些问题非常有帮助。

让我们来看看我的函数在哪里:

```
def convert_transit_pdf_to_csv(pdf, start, end):

    '''
    Input is a PDF file in the local directory, the first page
    of the PDF and the last page of the PDF. The function adjusts
    the first page to account for zero-based numbering. It will
    output a csv file back into the directory.

    There are a few issues with the code as-written. If an escalator
    or elevator is avialable 0.00% of the time, it will add an 
    additional digit or character to the availability column. There 
    is one other issue I've encountered where the subway lines
    aren't formatted properly.

    The comments will occaisonaly cut off and a fix for that is the 
    first priority once this code is re-visited.
    '''page_range = list(range(start - 1,end))pdfFileObj = open(pdf, 'rb')pdfReader = PyPDF2.PdfFileReader(pdfFileObj)lines = []
    availability = []
    units = []
    stations = []
    conditions = []condition = ''for page in page_range:
        pageObj = pdfReader.getPage(page) 
        current_page = pageObj.extractText().split('\n')**# the last two lines of each page are the page identifiers, so it's the current page without the last two lines**
for i in range(len(current_page[:-2])):

**# removes some titles that would otherwise be caught
**            if not re.match(string = current_page[i], pattern = '.* THAN 85% AVAILABILITY'):
                if len(current_page[i]) > 1:# this is less than ideal and occasionally cuts off the last line
# of a comment if it's under 40 characters. This was about as quick
# and dirty as it comes.
                    if len(current_page[i]) > 40:
                        condition += current_page[i]**# this would be [-6:] if all availabilities were above 10%,
# but some are available 0.00% of the time** if re.match(string = current_page[i][-5:], pattern = '\d\.\d{2}\%'):
                        availability.append(current_page[i][-6:])
                        lines.append(current_page[i][:-6])**# identifies the elevator or escalator unit**
                    if re.match(string = current_page[i], pattern = 'E[LS]\d{3}'):
                        units.append(current_page[i])
                        stations.append(current_page[i + 1])
                        if len(condition) > 1:
                            conditions.append(condition)
                            condition = ''**# specifically looks for the end of the page and ends the 'condition'**
                if i == len(current_page[:-2]) - 1:
                    conditions.append(condition)
                    condition = ''
    df_stations = pd.DataFrame(
        {'units': units,
         'stations': stations,
         'lines' : lines,
         'availability' : availability,
         'condition': conditions
        })df_stations.to_csv(pdf + ' converted.csv')
```

你可以在我的 [Github 知识库上看到关于 beta NYC Mobility for All Abilities hackathon](https://github.com/robblatt/BetaNYC-Hackathon-2019-09-21)的最新更新。