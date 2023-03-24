# 斯托克霍姆公寓价格的数据科学方法

> 原文：<https://towardsdatascience.com/a-data-science-approach-to-stockholms-apartment-prices-part-1-dcee0212596d?source=collection_archive---------20----------------------->

## [SE 公寓项目](https://medium.com/tag/se-apartment-project)

## 用 Python 中的 Web 抓取获取和清理数据

![](img/146f9e1e01f7b199d9d0688bf0187fcf.png)

# **项目背景**

被认为是世界上价格最高的房地产市场之一，我认为斯德哥尔摩的公寓市场可以作为一个有趣的案例研究，有很多有趣的数据。将大部分数据转化为易于理解的有用见解，并建立一个预测未来公寓房源销售价格的模型。

我还决定确保用一个数据科学家处理这个问题的方式来处理这个问题，处理一些现实情况下会出现的头痛和挑战。这意味着我想挑战自己，将可用的数据转换成可以实际使用的干净数据。

*注意:通常这个角色包括提出一个对业务有益的具体目标。但是在这种情况下，我选择了一种非常通用的方法，我还不知道我可能在寻找什么样的模式。*

> “数据科学家是能够**获取、筛选、探索、建模、**和**解释数据、**融合**黑客技术、统计学和机器学习**的人。数据科学家不仅擅长处理数据，[还将数据本身视为一流产品](https://www.dezyre.com/article/data-scientist-skills-must-haves/134)。— [*希拉里·梅森*](https://hilarymason.com/about/) *，快进实验室创始人。*

在这里阅读这个项目的下一部分:
[***用机器学习把数据变成洞察和预测器。***](https://medium.com/p/13e51772b528/)

所有源代码都在这里:
[**https://github . com/gustaf VH/Apartment-ML-Predictor-Stockholm _-with-web scraper-and-Data-Insights**](https://github.com/gustafvh/Apartment-ML-Predictor-Stockholm_-with-WebScraper-and-Data-Insights)

# **项目目标**

1.**收集在线可用的真实生活数据**，并能够**收集、清理和过滤这些数据**成一个高质量和高数量的数据集。**这次没有干净的 Kaggle-datasets 的捷径。**

2.根据收集到的数据，**制作一个机器学习模型**，该模型基于相关的 f **特征，如大小、房间数量和位置**，以*可行的准确度*对其销售价格做出准确的猜测，从而实际上被认为是有用的。

3.**将我收集的所有数据转化为洞察力**，希望能教会我**一些关于斯德哥尔摩房地产市场的令人惊讶或有用的东西**。

# **使用的特殊工具**

*   **Python** :整个项目都是用 Python 编写的，因为它对这个领域的库有很大的支持
*   [**Selenium**](https://selenium.dev/)**(Python):**流行的在 Chrome 中执行网页抓取程序的端到端测试软件。
*   [**熊猫**](https://pandas.pydata.org/)**&**[**Numpy**](https://numpy.org/)**:**Python 库为易于使用的数据结构和数据分析工具。
*   [**Yelp API**](# https://www.yelp.com/developers/documentation/v3/business_search)**:**API 接收附近的兴趣点以及基于地址的坐标。
*   [**Seaborn**](https://seaborn.pydata.org/)**&**[**Matplotlib**](https://matplotlib.org/)**:**Python 的数据可视化库旨在直观地展示我们的结果。

# 获取良好和丰富的数据。

为了开始这个项目，我需要好的数据。从这个意义上说，好的数据意味着它主要满足两个要求，质量和数量。就质量而言，重要的是我们的数据具有不同的属性(特征),这些属性与我想要了解和预测的内容相关。如果我想预测一套公寓的销售价格，好的特征将是，例如，公寓的*大小*，它的*位置*，*房间数量*，*何时售出*等等。

我可以在 Hemnet 上找到所有这些数据，这是一个瑞典的公寓列表网站，所以我决定使用这些数据。

***关于网络抓取非常重要:*** *就我而言，我绝不会将此作为商业产品或用于任何金钱利益，但始终尊重其他人或公司的数据，并确保您总是仔细检查适用于网络抓取的内容，以免侵犯版权。*

![](img/2caac6f6ae1902006f18b2042e3e1bf5.png)

Demo of how my Selenium Web Scraper works by automatically controlling an instance of Chrome.

简而言之，Web Scraper 的工作原理是打开 Chrome 浏览器窗口，导航到 Hemnets Stockholm 列表，获取该页面上的所有公寓列表，然后导航到下一页，重复这个过程。

我遇到并幸运解决的一些问题:

*   Hemnet 将每个搜索查询的列表限制为 50 页。我以此为契机，在公寓大小之间获得了*均匀分布的数据。我将我的搜集工作分成 20 个查询部分，每个部分对应于一个特定的公寓大小，然后我告诉搜集者为我搜集。为了前任。一部分是面积在 20 至 25 平方米之间的所有公寓。这导致我得到:
    **每页 50 个列表* 50 页* 20 段= 50，0 00 个列表** 现在我也有了比以前更好的数据传播，结果非常好！*

```
segments = [20, 25, 30, 35, 40, 45, 50, 55, 60, 65, 70, 75, 80, 85, 90, 95, 100, 105, 110, 115, 250]*for* i in range(0, (len(segments)-1)):
driver = initCrawler(segments[i], segments[i+1])apDataNew =           getRowsFromHnet(driver, 50)
apData = apData.append(apDataNew, *ignore_index*=True)
```

*   **谷歌广告横幅。**这些都很棘手，因为它们的类是动态的，当我试图点击“下一页”按钮时会导致 Scraper 误点击。不幸的是，这导致 Scraper 在我的例程中多次获得相同的页面，因为它有时没有正确地点击下一页按钮。正因为如此，在删除重复的条目后，我最终得到了 15 000 个条目，而不是 50 000 个。然而，我没有深究这个问题，因为幸运的是，15 000 仍然是足够继续下去的数据。
*   **iframe 中的隐私策略弹出窗口。**每次都弹出来，因为我们处于匿名模式，需要点击。原来它是在一个 iframe 中实现的，这使事情变得复杂，但是用 Seleniums 窗口管理工具解决了。

在 Yelp 的帮助下，为我们的数据添加特征。
在 Yelps API 的帮助下，我们可以得到很多基于某个位置的细节信息。例如，通过发送地址，我们可以获得该位置的坐标(纬度和经度)。这非常有用，因为我们现在可以将基于字符串的地址，如“Hamngatan 1”*转换成更可量化的东西*，如两个数字。这将使位置成为一个可用的功能，以后创造的可能性，几何比较公寓的位置。

此外，我们还接收兴趣点功能，以添加到我们的公寓数据集。这被定义为 Yelps 认为值得列出的地方的数量，这些地方在公寓的 2 公里以内。你可能会问，这有什么关系？它之所以相关，是因为它可以表明你的公寓位于市中心的位置，这可能会反映在销售价格中。因此，添加这个特性可能有助于我们以后在模型中的准确性。

```
response = yelp_api.search_query(*location*=adress, *radius*=2000, *limit*=1)*return* (latitude, longitude, pointsOfInterestsNearby)
```

# 清理、擦洗和过滤数据

这是我认为我学到的将问题应用到现实世界的最多的部分，因为数据从来没有像你希望的那样被很好地格式化和干净，这次也不例外。首先，我开始清理数据并重新格式化:

*   **改变日期顺序并制作数值。**我使用 python 字典结构将月份转换成数值。

```
*# 6 december 2019 --> 20191206* df.Date = [apartment.split(' ')[2] + (apartment.split(' ')[1]) + (apartment.split(' ')[0]).replace('december', '12')
```

*   **将除地址和经纪人之外的所有特征都变成数字(浮点数)。**
*   **削减销售价格。**从“Slutpris 6 250 000 kr”转换为“6 250 000”。
*   **删除错误获取的坐标。Yelp API 确实在一些情况下认为一个地址在瑞典完全不同的地方，并返回这些坐标。这可能会扭曲数据，因此如果一个公寓的坐标明显超出斯德哥尔摩的范围，整行都会被删除。**

```
df = df.loc[17.6 < df['Longitude'] < 18.3]
df = df.loc[59.2 < df['Latitude'] < 59.45]
```

*   **添加新的组合功能 PricePerKvm。由于奖金与公寓的面积密切相关，直接比较不同面积公寓的价格并不能显示价格有多合理。为了直接比较奖品，我创建了一个名为(PricePerKvm)的新功能，即价格除以平方米。**

# **请击鼓:我们的最终数据集**

经过大量的清理、擦洗和过滤，我们最终得到了一个包含 14 171 套公寓房源的数据集，这些房源具有 9 种不同的特征。

![](img/f49169064beb9ea4330a71a030618d83.png)

[Sample of 5 out of 14 171 entries]

在这一部分中，我了解到数据科学项目中的大部分时间将包括收集和清理数据，因为从时间上来说，**这一部分几乎占了整个项目时间的 80%。**现实生活中的数据很少按照你想要的方式进行清理、过滤或结构化。它也不总是统一的，所以对于下一个项目*我想接受将不同类型的数据集结合成一个的挑战。*

当谈到这个数据集时，它应该是我们下一部分的足够数据— **建立一个预测性的机器学习模型，并将数据转化为见解！**

在这里阅读这个项目的下一个故事:
[***用机器学习把数据变成洞察和预测器。***](https://medium.com/@gustaf.halvardsson/a-data-science-approach-to-stockholms-apartment-prices-part-2-13e51772b528)

所有源代码均可在此获得:
[**https://github . com/gustaf VH/Apartment-ML-Predictor-Stockholm _-with-web scraper-and-Data-Insights**](https://github.com/gustafvh/Apartment-ML-Predictor-Stockholm_-with-WebScraper-and-Data-Insights)