# 最好的日本动漫工作室是哪家？

> 原文：<https://towardsdatascience.com/which-is-the-best-japanese-anime-studio-f44fa642a03e?source=collection_archive---------30----------------------->

## *从不同角度看动画工作室*

由[艾里刘](https://medium.com/@airy3104)和[巴维什贝拉拉](https://medium.com/@bhaveshb619)

![](img/0ee1e019acb1adc7b047062cd02b91ed.png)

Photo by [Bruce Tang](https://unsplash.com/@brucetml?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

如今，动画已经变得越来越主流，它的市场继续在全世界扩张。2017 年，动漫市场创下了 198 亿美元的销售新纪录，这在很大程度上归功于海外需求(Jozuka，2019)。这个快速发展的行业引发了人们研究它的兴趣，我们也是其中的一部分。根据我们的在线调查，一些动漫迷出于对动漫的热情和对行业的好奇，自己进行了分析。他们的作品包括创建动漫排行榜、分析偏好动漫流派的变化趋势、动漫受众的人口统计学特征(Bilgin，2019；拉菲克，2019；过度溶解，2019)。除此之外，日本动画协会一直在发布关于日本动画产业的年度报告，这已经成为关于动漫产业的在线分析的主要来源之一(AJA，2019)。

这些作品虽然很有见地，但没有一部是专门看动漫工作室的。互联网上已经有一些动漫工作室的排名，而它们只是基于主观的网络投票(Teffen，2017；Lindwasser，2019)。因此，通过对动漫工作室的客观分析，启发我们填补空白。**更具体地说，我们将评估:1)哪个动漫工作室是最成功的工作室；2)成功工作室的特征是什么？**

## 数据源

我们能够获得 Kaggle.com 动漫的详尽数据集(MyAnimeList 数据集，2018)。这个数据集几乎有 2Gb 大，包含 3 个子数据集:“动画列表”、“用户列表”、“用户动画列表”。还有这些数据集的过滤数据和清理数据版本。总体而言，该数据集从 302，675 个独立用户和 14，478 部独立动画中获取数据。

我们主要利用了 AnimeList 数据集，该数据集包含 31 个列，如动画名称、动画 ID、工作室、流派、评级、喜爱(某个动画被添加到用户喜爱列表中的次数)、流行度(有多少人观看了该动画)等。

## **数据准备**

由于原始数据集是基于动画的，并且一部动画可以由存储在一个单元中的多个工作室制作，所以我们的第一步是将不同的工作室分成多个列。之后，我们使用 pivot 选项将多个工作室分成单独的行。我们还使用相同的技术拆分包含多种风格的列。

然后为了全面分析动漫工作室，我们创建了一个涵盖不同方面的公式:**整体工作室评分=人气评分+忠实观众评分+动漫质量评分+动漫数量评分**。对于每个动漫工作室来说，分数越高意味着排名越高。再进一步阐述四个标准: **1)** **人气评分** *(该工作室制作的所有动漫的受众总和除以该工作室制作的动漫数量)*看该工作室动漫的平均受欢迎程度； **2)忠实观众评分** *(动漫被加入自己喜爱列表的总次数除以工作室出品动漫的观众总数)*看工作室的动漫能有多成功地将一个普通观众转化为忠实粉丝； **3)质量评分** *(工作室制作的所有动漫的平均评分)*查看工作室作品的平均质量；4) **数量评分** *(工作室制作的动漫总数)*。在把四个分数加起来之前，我们还把它们标准化了，使它们在同一个尺度上。

# 那么，谁是赢家？

在计算了所有工作室的总分后，我们得到了我们的获胜者:**东映动画**。综合得分 441.3，比第二工作室高 68。

![](img/b7edaae566dcdff78a1559d3ea6a59b9.png)

Source: [https://www.anime-planet.com/anime/studios/toei-animation](https://www.anime-planet.com/anime/studios/toei-animation)

我们还强调了将在以下段落中详细分析的前 20 个工作室。

![](img/ec1977684f7ce6d011331b809fe67a3b.png)![](img/472403df14a10cf8c9f801c985ab1880.png)

Figure 1\. Top 20 studios scores

# **工作室如何根据我们设定的不同标准开展工作**

我们感兴趣的不仅仅是每个工作室的总体分数，还有他们个人的分数。因此，我们创建了四个散点图，同时显示总体得分和其他标准。每个点代表一个工作室(见图 2)。我们还应用了颜色来显示工作室的总体得分，因此更容易识别模式:一个点越绿，它在我们的排名中就越高(它获得的总体得分越高)，而越红，它在排名中就越低。

> **1。目前的动漫产业由大型工作室主导**

可见顶级工作室和“正常多数”之间的差距是巨大的。很少有工作室是绿色的，大多数是橙色或红色的，这意味着它们在我们的排名中得分很低。这一结果反映了当前的动漫产业结构，即由大型工作室主导，而由于预算限制，小工作室很难生存(Margolis，2019)。

![](img/fed2eaf698038ff3b5c7d2bfaaf1ef35.png)

Figure 2\. Scatterplots of studio scores

> **2。热门工作室有不同的策略。他们有的针对大众市场，有的针对小众受众群体。**

至于顶级工作室，它们都获得了不错的质量分数(见图 3)。然而，这些工作室在受欢迎程度、每部制作的动漫拥有多少忠实观众以及制作的动漫数量方面的表现却大相径庭。

![](img/857d9468debbd0b70c48d0dde14df8fc.png)

Figure 3\. Highlighted top studios in the scatterplots

比如 top one 工作室:东映动画获得了很高的数量分，这意味着它制作了大量的动漫(见图 4)。因此，东映动画的整体人气也很高。再者，忠实观众评分也是有竞争力的。这意味着在看了东映动画的作品后，很多观众把它加入了他们的最爱。尽管如此，平均人气得分还是比较低的，这说明东映动画制作的动漫并不是都受到普通大众的欢迎。然而，京都动画的情况完全不同。京都动画在我们的排名中排名第 8。类似东映动画，有着高质量的评分，甚至比东映动画还高一点点。虽然京都动画制作的动画比东映动画少得多，因此一般来说它在动画观众中并不那么受欢迎。尽管如此，它的平均受欢迎程度要高得多，这反映出即使京都动画也不是那么“多产”，他们制作的每部动画都是高质量的，其中大多数都会受到普通大众的欢迎。

![](img/685e9cab73e1fb0bb2d8b4706276bd78.png)![](img/119c8b0d5d97d2ac7cee5dc620d6e231.png)

Figure 4\. Toei animation (left) and Kyoto Animation(right) highlighted in the scatterplots

# **工作室如何相互合作**

该数据集还显示了不同工作室之间在制作动漫方面的大量合作。如图 5 所示，我们已经在一个图中可视化了所有的工作室，并用网络图突出显示了它们之间的联系。**节点的大小**表示该工作室与其他工作室合作的总次数。规模越大，协作数量越高。疯人院在制作动画时更喜欢合作。而**节点的颜色**表示工作室在我们计算的排名中的排名。红色阴影越深，工作室在我们的排名列表中的排名越好。此外，连接两个工作室的线的粗细取决于这两个工作室合作制作的独特动画的数量。更粗的线意味着两个工作室之间更高的协作实例。

![](img/c9d95b350f56d2add19895b19b25f58f.png)

Figure 5\. Network graph of studio collaboration pattern

> **并非所有顶级工作室都是积极的合作者，他们有时也会与小工作室合作。**

出乎意料的是，尽管排名靠前的顶级工作室往往是主要合作者，但他们在其他工作室中的受欢迎程度并不总是与他们的总体分数成正比。此外，顶级工作室不一定只与顶级工作室合作。例如，尽管东映动画是排名第一的工作室，但与疯人院、Production I.G .和 Sunrise 等工作室相比，合作频率较低(见图 6)。东映动画和它的合作者之间的关系也很弱，因为界限相对较窄，这意味着他们没有合作很多动画。

![](img/38f1320e6fe66dc362ab5334327a4224.png)

Figure 6\. Toei animation highlighted in the network graph

# 所有电影公司中最受欢迎的类型是什么:动作片、冒险片和奇幻片

由于我们的数据集中存在各种各样的动漫类型，我们有兴趣找出最受欢迎的类型。我们通过创建一个包含我们数据集中所有动漫类型的单词云来做到这一点(见图 7)。这样做之后，我们能够确定“喜剧”是最受欢迎的类型，紧随其后的是“动作”、“冒险”和“幻想”类型。除此之外，“浪漫”、“科幻”、“戏剧”、“生活片段”和“儿童”也是主要类型。

![](img/0fe5e64e84561f4aab32fc1ec9f02681.png)

Figure 7\. Word Cloud of popular anime genres

# **前 20 大工作室分析**

接下来，我们决定只关注排名前 20 的电影公司，看看他们是否有任何使他们成功的相似策略，或者他们是否有自己独特的策略。

> **1。前 20 大电影公司中的前 3 大类型是喜剧、动作片和冒险片。**

如图 8 所示，我们决定制作一个柱形图来显示排名前 20 的电影公司的类型构成，以了解他们的策略。为了使可视化更具可读性，每个工作室只显示前 10 个流派，而前 10 个流派被分组在“其他”下。工作室也根据他们的排名进行了分类。

可以看出，大多数顶级工作室的流派构成非常多样化。他们不再局限于几种动漫类型。这里的流行类型与所有动漫工作室之一一致，其中“喜剧”排在第一位，其次是“动作”和“冒险”。虽然“魔法”这一类型在顶级工作室中比在所有动画工作室中更受欢迎。

![](img/2520ceec4f48fc29afa1e8e756df0cbd.png)

Figure 8\. Stacked bar chart of genre composition for the top 20 studios

> **2。排名前 20 的工作室大多制作 PG-13 动画。**

另一个引起我们注意的数据字段是“Ratings”列。为了找到我们排名前 20 的工作室的大多数动画所属的最高评级类别，我们创建了一个树形图(见图 9)。这样做有助于我们确定“PG-13”分级类别是最受欢迎的分级类别，占前 20 大工作室制作的所有动画的 50%以上。

![](img/e1fbf41014196dedbc2f6719f1773bc2.png)

Figure 9\. Treemap of rating composition for the top 20 studios

> **3。排名前 20 的工作室一半以上的动画制作不到 50 集。**

在为排名前 20 的工作室工作时，我们惊讶地发现这些顶级工作室为他们的每部动画制作的剧集数量。为了确定这是一个反复出现的趋势还是只是一个异常现象，我们创建了一个方框图，其中来自这 20 家顶级工作室的所有动画都绘制在一个图表上，并根据每个工作室的剧集数量进行排列(见图 10)。

这种形象化突出表明，工作室首先喜欢为一部动画制作少于 50 集的剧集，然后如果这 50 集为该剧制造了炒作，就制作更多的剧集。这可以解释为 20 个顶级工作室中有 19 个为超过 50%的动画制作了不到 50 集。此外，还有一些非常受欢迎的动漫，如龙珠 Z、火影忍者和漂白剂，与同一工作室制作的动漫相比，其剧集数量非常高。

![](img/4edc6f527b8cb1f72b93565968caf871.png)

Figure 10\. A box plot of anime episodes by the top 20 studios

# **局限性和未来工作**

虽然我们能够从我们的可视化中挖掘出一些有趣的见解，但我们仍然希望承认我们在这个项目中面临的某些限制。

1.  由于我们的数据集是从 MyAnimeList 中提取的，因此它仅限于该网站的用户，其中大多数用户来自美国、英国、加拿大、俄罗斯和巴西。然而，日本和中国等其他国家也拥有庞大的动漫消费群体。虽然这些观众倾向于使用不同的网站来观看或评论动画，因此不包括在数据集中。因此，如果我们可以用其他不同的数据集来补充该数据集，我们将能够获得更全面的分析。
2.  在我们的公式中可以考虑更多的事实。例如，我们还可以比较工作室的收入，这将为我们分析工作室提供商业视角。在这个项目中，我们试图收集这样的数据，但由于时间和资源的限制，我们无法找到完整的数据。

# **链接到可视化演示**

请随意通过这些链接与我们的可视化文件进行交互。由于 Gephi 的性质，我们无法在线分享网络图。如果你感兴趣，请随时发消息，并要求该文件。尽情享受吧！

 [## 散点图

### scores of fallanimestudios

public.tableau.com](https://public.tableau.com/views/Scoresofanimestudios/Dashboard1?:display_count=y&publish=yes&:origin=viz_share_link)  [## Wordcloud

### 万物有灵

public.tableau.com](https://public.tableau.com/views/Wordcloudofanimegenres/Dashboard1?:display_count=y&publish=yes&:origin=viz_share_link)  [## 堆叠条形图

### Top20StudiosGenreComposition

public.tableau.com](https://public.tableau.com/views/Top10genrescompositionoftop20studios/Dashboard1?:display_count=y&publish=yes&:origin=viz_share_link)  [## 树形图

### top 20 studiostratingcomposition

public.tableau.com](https://public.tableau.com/views/AnimeratingcompositionoftheTop20studios/Dashboard1?:display_count=y&publish=yes&:origin=viz_share_link)  [## 箱线图

### Top20StudiosEpisodes

public.tableau.com](https://public.tableau.com/views/HowmanyepisodesdoesTop20studiosprefertoproduceperanime/Dashboard2?:retry=yes&:display_count=y&:origin=viz_share_link) 

# **参考文献**

AJA (2019)。动漫行业数据。检索于 2019 年 12 月 10 日，来自[https://aja.gr.jp/english/japan-anime-data](https://aja.gr.jp/english/japan-anime-data)

f .比尔金(2019)。动漫的故事。检索于 2019 年 12 月 10 日，来自[https://www.kaggle.com/fatihbilgin/story-of-anime](https://www.kaggle.com/fatihbilgin/story-of-anime)

Jozuka，E. (2019 年 7 月 29 日)。动漫如何塑造日本的全球身份？2019 年 12 月 10 日检索，来自[https://www . CNN . com/style/article/Japan-anime-global-identity-hnk-intl/index . html](https://www.cnn.com/style/article/japan-anime-global-identity-hnk-intl/index.html)

Lindwasser，A. (2019)。有史以来最伟大的 15 个动画工作室，排名。2019 年 12 月 10 日检索，来自[https://www . ranker . com/list/best-anime-studios-of-all-time/Anna-lindwasser](https://www.ranker.com/list/best-anime-studios-of-all-time/anna-lindwasser)

Margolis，E. (2019)。日本动漫产业的阴暗面。检索于 2019 年 12 月 10 日，来自[https://www . vox . com/culture/2019/7/2/20677237/anime-industry-Japan-artists-pay-labor-abuse-neon-genesis-evangelion-网飞](https://www.vox.com/culture/2019/7/2/20677237/anime-industry-japan-artists-pay-labor-abuse-neon-genesis-evangelion-netflix)

MyAnimeList 数据集。(2018).检索于 2019 年 12 月 10 日，来自

[https://www.kaggle.com/azathoth42/myanimelist](https://www.kaggle.com/azathoth42/myanimelist)。

过度溶解。(2019).动漫:好的，坏的和受欢迎的。检索于 2019 年 12 月 10 日，来自[https://airelevant.netlify.com/post/popular_anime/](https://airelevant.netlify.com/post/popular_anime/)

拉菲克·h .(2019 年)。分析 r .中的动漫数据检索于 2019 年 12 月 10 日，来自[https://towards data science . com/analyzing-Anime-data-in-r-8d2c 2730 de 8c](/analyzing-anime-data-in-r-8d2c2730de8c)

泰芬。(2017 年 8 月 27 日)。日本粉丝最喜爱的 10 个动漫工作室。检索于 2019 年 12 月 10 日，来自[https://goboiano . com/the-10-most-loved-anime-studios-ranking-by-Japanese-fans/](https://goboiano.com/the-10-most-loved-anime-studios-ranked-by-japanese-fans/)