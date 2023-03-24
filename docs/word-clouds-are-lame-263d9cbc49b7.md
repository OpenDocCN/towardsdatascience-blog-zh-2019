# 词云很蹩脚

> 原文：<https://towardsdatascience.com/word-clouds-are-lame-263d9cbc49b7?source=collection_archive---------9----------------------->

探索词云作为数据可视化的局限性。

![](img/08649c2cd92577a6fc14d7445aa57ed5.png)

Author: Shelby Temple; Made with Tableau

词云最近已经成为数据可视化的主要内容。它们在分析文本时特别受欢迎。根据 Google Trends 的数据，流行程度的上升似乎始于 2009 年左右，搜索关键词的兴趣目前仅在条形图之下。

![](img/baad84c7247933b2a01ee223b774cd69.png)

3 Month Moving Average; Source: Google Trends, Author: Shelby Temple, Made with Tableau

与条形图不同，单词云作为数据可视化有明显的局限性。显然，我不是唯一有这种想法的人。数据 viz 目录([【datavizcatalogue.com](https://datavizcatalogue.com/))**提到词云对于分析准确性不是很大。 [Daniel McNichol](https://towardsdatascience.com/@dnlmc) ，在《走向数据科学》上发表的一篇文章中，称文字云为文本数据的饼状图。**

**我的主要问题是，这种可视化通常是无趣的，并提供很少的洞察力。你通常会得到显而易见的单词和普通单词的混合。**

**你是说,“哈利”这个词在《哈利·波特》中出现了很多次。令人震惊！**

**![](img/ca368424ca7f2143dad2afed12f7cac2.png)**

**Source: [Word Clouds and Wizardry, Tableau Public](https://public.tableau.com/profile/dave.andrade#!/vizhome/WordCloudsandWizardry/HarryPotterWordCloudDashboard); Author: [Dave_Andrade](https://public.tableau.com/profile/dave.andrade#!/)**

**当我突然意识到云这个词是蹩脚的，我开始问自己更多的问题:**

*   **为什么每当有新的时髦 Twitter 话题或文本需要“分析”时，人们就会觉得有必要在整个互联网上发布单词云？**
*   **云这个词到底有什么局限性？**
*   **有没有一个合适的时间来使用单词云？**

# ****文字云背后的故事****

**我在一本名为[文本可视化介绍](https://www.springer.com/la/book/9789462391857)的书中找到了单词云的起源故事，所以功劳归于曹楠和崔薇薇。**

**在这本书里，他们将单词云归因于社会心理学家斯坦利·米尔格拉姆在 1976 年创造的可视化。他让人们说出巴黎的地标。然后，他画了一张地图，上面用地标的名字作为文本，如果地标得到更多的响应，字体就会变大。**

**然而，这并不完全是云这个词流行的原因。为此，我们应该感谢照片分享网站 Flickr。大约在 2006 年，他们实现了单词 cloud visual，作为一种通过流行标签浏览网站的方式。他们称他们的单词云实现为标签云。UX 社区接受了这个设计，它开始出现在互联网上。因此，你可能会遇到人们仍然称之为词云，标签云。**

**Flicker 开心地为在 UX 社区掀起“云”这个词的热潮道歉。**

**![](img/466315a8f6955d07adb70e365f56b0ad.png)**

**3 Month Moving Average; Source: Google Trends; Author: Shelby Temple; Made with Tableau**

**那么，2010 年前后发生了什么，让词云趋势上升到类似于条形图的搜索兴趣水平？**

**我不知道这要归功于哪个人或哪个形象化的形象——但看起来，当前“云”这个词的流行与挖掘文本以获得洞察力的流行有关。**

**一个主要的应用是编辑一堆评论并挖掘它们的洞察力。例如，像亚马逊这样的网站可以获取产品的评论和评级(通常是 5/5 颗星)，并找出产品的优势和劣势。也许当过滤低评价时，“不适合”这个短语经常出现。**

**对于一家有很多分支机构的公司来说，常见的投诉是可以解决的。如果一家塔可钟餐厅的管理经常遭到差评……也许那里的管理很糟糕？**

**不幸的是，word clouds 很少回答这些问题或解决这些应用程序，但人们仍然在制造它们！**

**抛开商业应用，任何对 R 或 Python 有基本了解的人都可以拉推文，做一个词云。它们现在更容易制造的事实显然有助于它们的流行。**

**总而言之:**

1.  **1976 年，斯坦利·米尔格拉姆(Stanley Milgram)制作了第一张 word 云状可视化地图，作为巴黎的地标**
2.  **Flickr 在 2006 年开创了网络词汇云野火——他们表示抱歉**
3.  **大约在 2010 年，数据科学和文本挖掘爱好者拿起了单词 cloud torch，并使它们比以往任何时候都更受欢迎！**

# **探索“云”这个词的局限性**

****低-信息:****

**你上一次基于一个词云做决定或做重要外卖是什么时候？对我来说，答案是永远不会，我想大多数人都在同一艘船上。单词云提供了底层信息，通常只是一个单词出现的频率。如果没有额外的上下文，频繁出现的单词通常没有多大意义。谁说的，为什么说，什么时候说的，对谁说的？**

**我见过用颜色来表示第二层信息的用法。也许词的大小就是频率，那么颜色就是词的类别。我对此百感交集，因为这与人们习惯于在 word clouds 中看到的内容相悖。在某种程度上，它剥夺了使用词云的唯一优势之一——人们非常熟悉它们作为可视化词频的方法。**

****句子结构方面的语境:****

**单词云通常一次只查看一个单词。这是有缺陷的，因为它会产生误导性的最终产品。如果你的公司推特上有一堆推文说，“不酷！”云这个词会把“不”和“酷”分开。这可能会误导人们认为他们的公司 Twitter 很酷，而实际上它并不酷。**

****排名:****

**单词云甚至不是它们想要做的最好的可视化。给你看看最常用/最流行的词。当然，有时你能说出哪个是最流行的词，但是第二、第三和第十个最流行的词呢？没那么容易。在实现这一点上，排序条形图的作用要大得多。**

***排名第十的最受欢迎的单词是什么？***

**![](img/b571951f8f4dd747140da3e6ccc370b0.png)**

**Author: Shelby Temple; Made with Tableau**

***现在怎么样了？***

**![](img/d2af55c2896fd7883cead95eefb0f499.png)**

**Author: Shelby Temple; Made with Tableau**

****其他限制:****

**其他一些问题包括单词云强调长单词多于短单词。单词的位置和顺序可能会令人迷惑，因为它们通常只是随机出现在不同的地方。此外，通常不清楚常见/无聊的词是否被过滤掉了，例如-**the**、 **as** 、 **is** 和**或**。最后，如果你使用流行度/频率以外的尺度，人们可能会措手不及。**

# **为什么词云还在流行？**

**如前所述，文本挖掘只是数据科学浪潮中的另一名冲浪者。随着数据科学、大数据和人工智能越来越受欢迎，自然语言处理(NLP)和文本挖掘也将呈上升趋势。**

**奇怪的是，我现在开始把单词 cloud 和 NLP 的 [hello world](https://en.wikipedia.org/wiki/%22Hello,_World!%22_program) 联系起来。就像大多数程序员的第一个程序一样，他们只是简单地打印“你好，世界！”—从 Twitter 中挖掘推文并将其可视化为词云几乎已经成为文本挖掘的入门任务。一个微妙的区别是，文字云看起来比说“你好，世界”的文字有趣得多第二个区别是，将非结构化文本数据处理成单词云要复杂一些。由于这一点，我认为人们更倾向于分享他们的第一个词云。它是数据科学中的一种奇怪的弯曲。**

> ***自豪地在互联网上发布首个单词云***
> 
> **其他人:“是的，我记得我的第一个单词云。”**

# **最后的想法**

**有没有一个很好的场景来创建单词云？**

**我也这么认为即使在指出其局限性之后，我认为单词云还是有一些优点的。**

1.  **它们通常色彩鲜艳，看起来很漂亮**
2.  **就像饼图一样，几乎不需要任何描述就能让人理解——这是一种人们熟悉的模式**
3.  **这是一个很好的文本挖掘任务介绍**
4.  **没有很多其他的可视化工具容易制作，并且专门用于非结构化文本分析**

**但是，我们如何解决词云被过度使用的问题呢？我认为可视化和 NLP 专家有很大的机会为文本挖掘提出新的包和可视化技术。我认为，如果有人构建了一个新的 Python 或 R 包，可以轻松地消化文本结构，并以令人兴奋的方式可视化它们，人们显然会利用它。**

**在那之前，每当有新的令人兴奋的 Twitter 话题时，我们都需要做好准备。**

**![](img/638cf2ffcf4869f5f534a4c3512ed3fd.png)**

**Author: Unknown**

**需要灵感？这里有一些我见过的更好的文本可视化。注意没有单词云:**

**[通过抄本认识的朋友——洛娜·伊登](https://public.tableau.com/profile/lorna.eden#!/vizhome/FRIENDSTHROUGHTHETRANSCRIPTSIRONVIZ/FrontPage)**

**[对甲壳虫乐队的分析——亚当·麦肯](https://public.tableau.com/profile/adam.e.mccann#!/vizhome/BeatlesAnalysis/BeatlesAnalysis)**

**[哈利·波特的咒语——斯凯勒·约翰逊](https://public.tableau.com/en-us/s/gallery/spells-harry-potter)**

**[圣诞电影的原话——扎克·盖修](https://public.tableau.com/profile/zak.geis7550#!/vizhome/TheWordsofChristmasMovies/TheWordsofChristmasMovies)**

**[神圣文本中使用的词语——肯·弗莱拉格](https://public.tableau.com/en-us/s/gallery/word-usage-sacred-texts)**

**[南方公园第一季文字分析——罗迪·扎科维奇](https://public.tableau.com/profile/rody.zakovich#!/vizhome/SouthParkSeasonOneWordsAnalysis/TheWordsofSouthParkSeason1)**