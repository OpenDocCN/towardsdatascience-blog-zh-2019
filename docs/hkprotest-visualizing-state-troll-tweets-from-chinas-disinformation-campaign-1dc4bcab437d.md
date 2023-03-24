# #抗议:可视化来自中国造谣运动的国家巨魔推文

> 原文：<https://towardsdatascience.com/hkprotest-visualizing-state-troll-tweets-from-chinas-disinformation-campaign-1dc4bcab437d?source=collection_archive---------17----------------------->

## 使用各种可视化工具，深入探究北京在社交媒体上反对香港抗议活动背后的言论。

![](img/2073b1ba4fce19d8768c7782fd5f0358.png)

A Scattertext plot of key words in tweets and retweets by Chinese state trolls targeting the protest movement in Hong Kong.

8 月下旬，随着香港持续数月的反政府街头抗议导致紧张局势加剧，Twitter、脸书和谷歌宣布，它们发现并处理了中国政府巨魔在其平台上发起的造谣活动。

这些活动旨在“抗议运动和他们对政治变革的呼吁”， [Twitter 在 8 月 19 日](https://blog.twitter.com/en_us/topics/company/2019/information_operations_directed_at_Hong_Kong.html)发布了 360 万条来自该公司所谓的“国家支持的协调行动”的国家巨魔推文。[脸书](https://newsroom.fb.com/news/2019/08/removing-cib-china/)和[谷歌](https://www.blog.google/outreach-initiatives/public-policy/maintaining-integrity-our-platforms/)发布了新闻声明，但尚未发布与中国政府造谣活动相关的数据集。

这是我研究 Twitter 发布的巨魔推文的第二部分，继上个月的快速探索之后。在这里，我将尝试使用一系列自然语言处理和可视化工具，筛选出中国巨魔推文中的关键词和修辞。

# **关键假设和回购**

首先:这是最新笔记本项目的[回购](https://github.com/chuachinhon/twitter_hk_trolls_cch)。CSV 文件太大，无法上传到 Github 上。直接从推特[下载。](https://blog.twitter.com/en_us/topics/company/2019/information_operations_directed_at_Hong_Kong.html)

这个数据集有超过 360 万行和 59 种语言的推文。这里非常嘈杂——充斥着体育和色情相关的推文，以及政府官员和逃亡的中国亿万富翁郭文贵之间的口水战。

现已被暂停的国家巨魔账户也在用多种语言发布推文。例如，一个将账户语言设置为英语的巨魔可以同时用英语和中文，或者更多语言发微博。许多账户也已经休眠了很长时间。

为了使这个项目更易于管理，我采取了以下步骤:

*   **只关注英文和中文推文**(因为它们主要针对香港人)，但确保我从有中文语言设置的账户中捕捉英文推文，反之亦然。
*   **将 2017 年设定为分析的起点**，因为许多人认为俄罗斯 2016 年在美国的假情报活动激发了最近国家假情报工作的重大战术变化。当然，中国巨魔早在 2017 年就已经在 Twitter 上活跃了。
*   使用明显的关键词，如“香港”和“警察”，作为主动过滤的定位点。这清楚地将选择偏差引入了分析和可视化。但是考虑到数据集中的噪声量，我觉得这是一个可以接受的折衷方案。

# **信号和噪声:关键发现**

## **#1。非常低的信噪比**

如果没有积极的过滤，很难从数据集中发现更多信息。这适用于英文和中文推文(以及转发)。以下面的图表为例，它显示了在对郭的推文进行轻微过滤后，前 50 个最常见术语的频率分布:

![](img/d50c619d2d357aed600f18978d068da1.png)

前 50 名名单中几乎没有任何一个词可以暗示这些推文是关于什么的。中文巨魔推文同样嘈杂，充斥着涉及犯罪故事、美中关系和日光浴的垃圾内容:

![](img/95992c9c69f0facbb2c552bcc4c659ad.png)

## **2。极少数“主角”演员/巨魔**

In its press release, Twitter highlighted the activities of 2 accounts — Dream News(@ctcc507) and HK 時政直擊(@HKpoliticalnew) — without explicitly saying why it chose to “showcase” these two out of the 890 unique userids in the dataset.

通过挖掘数据集，很快就发现这两个账户是一小撮超级活跃的巨魔，推动了北京对抗议运动的言论。这些“领头的巨魔”活跃地用英文和中文发推文，并经常被不太活跃的账户转发。不清楚@ctcc507 和@HKpoliticalnew 是否直接控制了其他的巨魔账号。

这两个主要客户的推文示例:

*   @ ctcc5072019–07–02 12:31:00+08:00 发微博:‘立法会是香港人的。以势力为代表的那些别有用心的人躲在幕后围攻立法会，这是对大多数香港人的严重侮辱。
*   @ HKpoliticalnew2019–06–14 09:50:00+08:00 发微博:‘很明显，先是有特工在现场指挥和扰乱香港，然后是美国政府出面指责和制裁香港。这些代理人通过散布有关香港的负面信息向中国和香港政府施加压力。#香港[https://t.co/Q6dGQUUHoG'](https://t.co/Q6dGQUUHoG')
*   @ctcc507，2019–06–22 21:51:00+0800 发微博:‘别有用心的人企图通过‘颜色革命’在香港兴风作浪，煽动不明真相的学生团体和香港市民，围攻警察总部，意图破坏香港稳定。https://t.co/awxlQnMF4A'[，](https://t.co/awxlQnMF4A')

这些中国国有巨魔账户的显示名称看起来像新闻媒体会采用的名称，这并非巧合。这一增选新闻媒体身份的举动直接出自[俄罗斯 2016 年剧本](https://cdn2.hubspot.net/hubfs/4326998/ira-report-rebrand_FinalJ14.pdf)。

其他一些稍微活跃一点的中文账号包括:@charikamci、@KondratevFortu、@jdhdnchsdh、@shaunta58sh、@Resilceale、@ qujianming1、@vezerullasav158 和@ardansuweb。但是没有一个接近@ctcc507 和@HKpoliticalnew 在巨魔网络中的影响力水平。

## **3。每个角落都有“外国代理人”**

巨魔推文的内容在很大程度上是可以预见的，从呼吁支持陷入困境的香港警方到谴责抗议者的暴力行为。

但是在“带头”的国家巨魔中，一个特别占主导地位的线索是[阴谋论，即美国是香港](https://www.nytimes.com/2019/08/08/world/asia/hong-kong-black-hand.html)抗议活动背后的“黑手”，这些抗议活动吸引了数百万人走上街头。

我稍后将详细介绍散点图，但它有一个很好的功能，您可以在用于图表的语料库中搜索关键词，并显示该词出现在哪个 tweet/retweet 中。以下是你在[搜索“不可告人”](https://www.dropbox.com/sh/jmb1oy0kak18cwy/AABfHXYoA_P8d6Tw-scNpDVia?dl=0)时出现的内容:

![](img/db572c2fd35f6a94102a7af8d457d8e3.png)

And if you search for [“color revolution”](https://en.wikipedia.org/wiki/Colour_revolution), or **顏色革命**:

![](img/616777bd05ba1ababf2dcc112ab37d82.png)

我已经上传了[交互式散点图](https://github.com/chuachinhon/twitter_hk_trolls_cch/tree/master/scattertext_charts)给那些想在不使用或重新运行 Jupyter 笔记本的情况下进行实验的人。

## **4。中文推文中更具攻击性、偏执的语气**

在用中文发微博时,“领头”的“巨魔”似乎采取了更尖锐、更偏执的语气。在接下来的两条推文中，账户@HKpoliticalnew 实质上指责美国驻香港领事馆与“香港的叛徒”合作，破坏这座城市的稳定。

此外，巨魔还声称,“许多”领事馆工作人员实际上是中央情报局的特工，是中央情报局下令发生流血冲突的:

*   @HKpoliticalnew; tweeting at 2019–07–01 05:12:00+08:00: ‘美國香港領使館千人員工，不小是 CIA 特工連同香港漢奸在港推動顏色革命 Over 1,000 US Consulate staffs, many with CIA launched Color Revolution to destabilize HK 香港警察「阿 sir 我撐你」 支持者逼爆添馬。\n\n 多名藝人藍衣撐警 譚詠麟：家和萬事興。\n\n#香港 #撐警行動 [https://t.co/alO0lhkMc6']](https://t.co/alO0lhkMc6'%5D)
*   @HKpoliticalnew, tweeting at 2019–06–22 22:16:00+08:00: 美國喺香港推動顏色革命 總指揮嚟自香港美國領事館 下令一定要制造流血事件 美國要見血讓其媒稱…

完整的内容分析超出了本分析的范围。但是看到一份关于中国造谣运动的完整学术研究将会很有趣。

在接下来的几节中，我将更深入地研究英文和中文推文，以及用于可视化关键术语的工具。

# **可视化中国国家巨魔推文**与分散文本

将非结构化的文本可视化是困难的。虽然频率令牌图和树形图有助于快速洞察混乱的文本，但从视觉角度来看，它们并不特别令人兴奋，当您想要检查某些关键词在原始文本中是如何使用的时，它们也没有用。

![](img/4a3612cac5b96f4206cf55cf89797b5a.png)

上面的树形图，突出了前 2 个巨魔账户最常用的 50 个关键英语单词，给出了一个非常粗略的用法。但是当你想直接指出一个你感兴趣的单词时，它是不精确的，也不是特别有用。

进入 [**散射文本**](https://github.com/JasonKessler/scattertext) ，它将自己描述为“在中小型语料库中区分术语”的“性感、互动”工具。它有一些怪癖，比如你想分析的文本对二进制类别的硬性要求。但在其他方面，它开箱即用，包括中文文本，详细的教程将使您能够尝试各种各样的可视化。

在这种情况下，互动功能特别有用，允许我在图表中搜索关键词，并查看哪些推文和转发使用了该特定单词或短语。

![](img/4f406c7a9d5e4b384c21edfb8f9617dd.png)

我已经在这里[上传了交互文件](https://github.com/chuachinhon/twitter_hk_trolls_cch/tree/master/scattertext_charts)，这样你就不必为了尝试散点图而运行这个笔记本了。以下是解读图表的方法:

图表中的单词根据它们之间的联系被涂上颜色。蓝色的与原创推文联系更紧密，而红色的与转发推文联系更紧密。每个点对应一个提到的单词或短语。

**定位:**
-更靠近图顶部的单词代表“原始”推文中最频繁使用的单词。

-圆点越靠右，该词或短语在转发中使用的次数越多(例如:立法会)。

-在 tweets 和 rewteets 中频繁出现的词，如“警察”和“中国”，出现在右上角。

-在推文或转发中不常使用的词出现在左下角。

**重点区域:**
——左上角:这些词在推文中出现频率很高，但在转发中并不常见。我们仍然可以在这里看到大量的噪音，就像“sterling”和“osullivan”这样的词出现在最顶端。

-右下角:同样，经常出现在转发中而不是推文中的词出现在右下角。在这里，我们可以看到一些术语，这些术语指出了哪些是网络钓鱼者转发的热门账号——那些由中国官方媒体拥有的账号。

您可以按照自己的意愿过滤数据集，并绘制符合您需求的自定义散点图。在下面的图表中，我绘制了@ctcc507 和@HKpoliticalnew 的推文和转发:

![](img/83c34f2c208721d89a4f94a4eadd8138.png)

散点图也适用于中文文本，如下两个例子所示:

![](img/a01d7d247054859c561a6695bde06df4.png)

The chart above plots a subset filtered for tweets containing the characters “香港”(Hong Kong). The chart below is a plot for tweets containing “顏色革命”(Color Revolution), “外國勢力”(Foreign Forces) and “美國”(United States):

![](img/50fffeeca22cbe0aceaab8369da114b0.png)

生成散点图的两个笔记本分别是这里的和这里的。我对笔记本中的图表有一个稍微详细的解释，供那些希望深入了解这个话题的人参考。

我还在这个[笔记本](https://github.com/chuachinhon/twitter_hk_trolls_cch/blob/master/notebooks/2.2_NLP_Chinese_cch.ipynb)中对中国巨魔的推文进行了更详细的“去层化”尝试。例如，看到推文和转发中使用的关键词的差异是很有趣的:

![](img/33d28f564e3ca04b639e744d413b6b81.png)![](img/6c604a251eb73954270f6fd70e2703b2.png)

# **最终想法**

该数据集将明显受益于认真的网络分析，以找出巨魔帐户和关键字之间的联系。我没有这方面的经验，很想看看这方面的专家有什么建议。

我假设 Twitter 有一些很棒的内部工具来找出这些巨魔账户之间的网络联系，以便能够大规模关闭此类操作。

这给我们带来了一个主要问题，Twitter 和其他社交媒体巨头如何发现他们平台上的这种国家造谣活动。到目前为止，这些科技公司都没有暗示他们用来清除这些“巨魔”的方法或标准。他们的执法行动是否过于宽松或过于全面？我们现在只能猜测。

最后，鉴于香港正在发生的事件，这种“桌面分析”似乎微不足道。但希望当尘埃落定时，它会增加我们对这个复杂问题的理解。

如果您发现任何错误，请 ping 我@

推特: [@chinhon](https://twitter.com/chinhon)

领英:【www.linkedin.com/in/chuachinhon 

我早期的项目是关于推特上的[造谣活动](/using-data-science-to-uncover-state-backed-trolls-on-twitter-dc04dc749d69):[https://github.com/chuachinhon/twitter_state_trolls_cch](https://github.com/chuachinhon/twitter_state_trolls_cch)