# 健身数据科学:50 岁是新的 30 岁——第一部分

> 原文：<https://towardsdatascience.com/data-science-for-fitness-50-is-the-new-30-part-i-b5ffb543b555?source=collection_archive---------14----------------------->

下面的文章将试图解释一个对我来说非常有趣的经历，与算法音乐作曲 algos(我在 2013-2014 年开发的无神经网络)一起，是我从事的最有价值的项目之一:健身的数据科学。

在这一系列数据科学的实际应用中(围绕 MNIST、Reddit、Yelp 数据集等的教程你还不累吗？)我正在起草，我打算讲迷你故事:我是怎么到那里的，为什么，什么时候，等等。触及与题目相关的主题。

这不是直接的代码审查，虽然有些会被涵盖。这些小故事将更多地围绕一个主题讲述数据策略、数据科学及其实际应用，而不仅仅是代码。

回到健身，我在 2008 年 9 月开始对这个话题敏感。当时我在金融危机的“归零地”雷曼兄弟(Lehman Brothers)工作，在那里，我积累了大量围绕奇异数据集开发和推出结构化金融交易的专业知识。雷曼兄弟倒闭后，我对着镜子认真审视了自己很久。我看到一位情绪低落、身体不适(严格来说是肥胖)的华尔街高管，他曾专攻备受指责的金融领域。但我也看到了我内心的数据科学家。

定量分析师本质上是数据科学家，有很多时间序列分析和金融背景。定量分析师是解决问题的人，那时我遇到了大问题:由于肥胖和工作压力，我的健康状况逐渐恶化。

![](img/0da3ea039d0d912256b0d4ab87fbffe8.png)

Fitness contrast: mid 30s, vs late 40s

镜子里的 quant/数据科学家回头看着我说“你可以用数据解决你的健身问题！”

让我现在停下来告诉你，如果你有兴趣学习快速减肥的方法，那么你不妨转到另一篇文章，因为这个故事不适合你。或者如果你想学习数据科学或编程，那么这篇文章也不适合你。

另一方面，如果你正在寻找独特的知识来帮助你实现健身目标，以新的方式看待事物，如果你喜欢数据科学，并且你正在考虑改善你的健身，那么这篇文章可能会为你提供正确的动力，也许会让你朝着正确的方向发展自己的研究和健身计划。

回到故事:我是一个胖胖的量化分析师，正在寻找一种合理的减肥方法，并且符合我量化分析师的心态。一个优秀的定量分析师知道，问对问题会让你离答案只差一半。因此，在 2008 年，我致力于研究这个课题，开发一个科学和公正的分析，并使用我自己的数据和发现。

作为一名优秀的量化/数据科学家，我对数据有点痴迷。在我与[“数据科学手册”](https://www.amazon.com/Data-Science-Handbook-Insights-Scientists/dp/0692434879/)的采访中，我甚至讲述了我在 1992 年如何编写一个专家系统的故事(基本上，一个[层次分析法](https://en.wikipedia.org/wiki/Analytic_hierarchy_process)，一种基于数学和心理学的组织和分析复杂决策的结构化技术)，以决定我是否应该向我的妻子求婚…

无论如何，从 1985 年开始，我就一直手动收集某些类型的数据，用于我所从事的任何分析。从那以后，我开始收集和存储数据，首先是以书面形式，然后是 5 1/4 英寸的磁盘、3 1/2 英寸的软盘、磁带、zip 磁盘、CD、DVD、硬盘，你能想到的都有。1990 年，我得到了我的第一个电子邮件和 Unix(Linux 的鼻祖)账户，那一年我开始在我的离线数据收集中添加一些基本类型的在线数据收集，因为我可以访问早期的互联网。

但首先我必须调整某些工具，或者开发自己的工具，因为 Mosaic(第一个浏览器)还没有发明出来，我必须使用 Gopher、IRC 和 FTP 来收集数据。

![](img/99631af58e57a0488445ae6061276f9b.png)

Above is a sample I put together of the many devices and many types of data I have used to capture fitness data over the years

关于健康，我从 1987 年到 2000 年收集的数据非常少:每周使用 2-3 台设备收集几个数据点。然而，自 2000 年以来，我的数据变得更加密集，每天都有数百个数据点，并且使用 30 多种不同的设备和服务。

上面的图表显示了这些年来我用来获取自己身体参数的一些设备和服务。

我收集了各种各样的数据集，加上从医学杂志上发表的论文、与 40 岁以上身体健康的人的交谈中收集的知识，以及从 bodybuilding.com 等网站收集的数据。并结合自己的经历:

1.  *我能使用数据科学的方法来帮助我减肥吗？*
2.  *我能否利用数据科学和定量分析的元素，在相对较短的时间内找到最适合我的方法？*
3.  *我可以编写代码来整合来自我正在使用和已经使用的多个硬件、软件和基于网络的系统的数据吗？*
4.  *我能否制定自定义指标、测试假设，并针对偏离我的健身目标的情况开发近乎实时的警报？*
5.  *我可以使用分类模型进行分析吗？我可以使用回归模型吗？*
6.  *我能否从数据中获得可操作的情报？*

答案是肯定的。

![](img/aef69d4a1e36482081c1755cad220c90.png)

In the Facebook post above, privately share only among close friends, you can see my peak shape in January 2012 and my fitness evolution since 2009\. A very important factor in my program was the incorporation of domain expertise, in particularly from people over 40s (good combination of theoretical and practical knowledge), and in good shape. Thanks again to Gregg Avedon and Steven Herman, whose knowledge complemented and enhanced my own knowledge, and provided inspiration.

以下是多年来获得的一些参数。我打算以后在我的 [GitHub](https://github.com/lmsanch) 或 [Bitbucket](https://bitbucket.org/lmsanch/) 账户中分享我的数据摘录和一些代码。如果你想了解最新信息，请在[sourcer](https://sourcerer.io/lmsanch)跟我来，因为我 99%的代码都在私人回购中，sourcer 以一种非常好的格式整合了我所有的编码活动。

![](img/33bd3e2235321501e67a67943e4fad98.png)

Somewhere in the Caribbean, in 2010

![](img/45d7100163992f2dd3a677e4fdeb9f40.png)

Some of my programming language/library specific expertise, percentile rank against all other users in GitHub, GitLab, BitBucket, etc. and the top areas of concentration of my code

```
*A little review about Sourcerer: It still has a few bugs, but the work these guys are doing is* ***GREAT*** *and very useful, specially for people like me, whose code is mostly in private repos and with very little contribution to open source projects (Wall Street vs Silicon Valley. Anybody can relate?).* *Using machine learning, Sourcerer analyzes your code and ranks your coding skills (commits, lines of code, code frequency, style, etc.), against ALL other users in GitHub, BitBucket,  GitLab, etc. and summarizes your expertise by technology, programming languages, etc.* *None of your proprietary code in private repos is shared with Sourcerer, simply, just analyzed.* *To the left is a sample of my Sourcerer profile. If you are a coder with public and private repos in many places, you should definitely check it out.*
```

回到这篇文章，下面粗体的参数将是我在未来的“健康数据科学:50 岁是新的 30 岁”系列中试图解释的。他们是我从胖到适合的转变中影响最大的，也是我开发的机种中解释价值最高的。它们是:

*   **每日消耗的总热量**
*   **消耗的卡路里分解(来自蛋白质、碳水化合物、脂肪的卡路里)**
*   **每次锻炼消耗的热量:**举重、单板滑雪、跑步、骑自行车、打高尔夫、其他
*   **肌肉量**
*   **脂肪量(体脂%)**
*   **内脏脂肪**
*   **VO2 最大值**
*   **运动后过量耗氧量(EPOC)**
*   **恢复时间**
*   **培训效果**
*   T-3 总测试*
*   T-4 总测验*
*   平均体温*
*   总脂肪
*   饱和脂肪
*   纤维
*   胆固醇
*   总重量
*   身体年龄
*   血糖水平
*   体重指数

我的系统的结果是杰出的。这是一个总结:

![](img/b180d61bc0c32d0d450b44b2bb541048.png)

Approximate peak to trough key measures: From 250 lbs total top weight in 2008 to 190 lbs lowest weight in 2009–2011

比 2009-2011 年身体脂肪减少 72 磅更显著的是 8 年内增加了 30 磅肌肉，如果你在 40 多岁或 50 多岁时不使用合成代谢类固醇，这在代谢上非常困难。见鬼，即使在你 20 多岁或 30 多岁的时候，也很难获得那么多。

![](img/6d76ae983471d76553a43fce699ed4bb.png)

Another great help came from [Gregg Avedon](https://en.wikipedia.org/wiki/Gregg_Avedon) (18 times on the cover of Men’s Health magazine), from Florida, fitness model and writer for said magazine. Gregg looks great in his early 50’s, and his “health food recipes” had a lot of the key nutrition ratios I later found out worked well for me. We kept some interesting communication via Facebook etc. for tips, progress reports. etc.

我说这些结果是杰出的，因为我们需要考虑到，在 30 岁以上，每个人都患有与年龄相关的[肌肉减少症，](https://en.wikipedia.org/wiki/Sarcopenia)每十年造成 3%到 5%的肌肉质量损失(在老年时加速，50 岁后每年肌肉质量损失 0.5%到 1%)。即使你很活跃，你仍然会有一些肌肉流失。肌肉的任何损失都很重要，因为它会降低力量和灵活性。所以，在某种程度上，逆转肌肉流失是一种让你看起来和感觉更年轻的方法。

除了饮食和锻炼的好处，我还会在短时间内经历大规模的肌肉增长(请查看上面我的 1/2012 图片)，使用一种有争议的“技术”，称为 ***“糖原超级补偿”。***

![](img/f21da734cabfec8b41c073a5c9bbf9d6.png)

Another friend and mentor, from the over 50 crowd, Steven Herman. Steven is from NYC, a former Madison Avenue executive now fitness model & instructor, as a pastime. Looking great is his mid 50s.

Gylcogen super compensation 不是我正常健身程序的一部分，而是一个“辅助实验”，试图量化肌肉生长的某些参数。随着时间的推移，这是不可持续的，所以我不推荐给任何人，但你可以在这里阅读更多关于它的内容。

那么，我是如何克服重重困难取得这些成就的呢？定量分析/数据科学和学科。但这一切都始于数据收集、数据整合和标准化。

**开始—数据整合**

下面的图表显示了这些年来我的总体重波动，我测试了许多减肥系统(Nutrisystem，Atkins 等。)，结果喜忧参半。

![](img/8b498706b98c57048f1d3d36d4dae673.png)

阴影区域表示所获取和分析的知识的不同特征。健身数据和知识是从 1985 年到 2008 年期间获得的(超过 23 年的随机试错和没有适当计划的训练)。从 2008 年开始，我分析了自己的数据，并纳入了健身和营养领域人士的专业知识，为一个非常具体的双重目标函数进行了优化:最大减脂和最大增肌(这很棘手，因为新陈代谢不可能同时完成这两项)。

为了使用我捕获并分布在不同介质上的所有不同数据类型，我必须将它们整合到一个数据库中。我选择 MongoDB，因为:

*   **这是一个无模式的数据库，所以我的代码定义了我的模式，对我来说这对于数据科学来说是完美的。**
*   因为它以 BSON(二进制 JSON)的形式存储数据，所以它有助于存储非常丰富的粒度数据，同时能够保存数组和其他文档。例如，我可以看到锻炼中消耗的总热量，但在更精细的检查中，我可以看到我的一个设备捕捉到的一秒一秒的热量消耗(即 FitBit 实时心率数据，计算卡路里等。)都在同一个 mongodb“文档”里。

![](img/c77cbdb94e417542b952cabaea4f22b2.png)

“Volatility” higher than certain threshold for some key fitness and nutrition parameters do not let you to get “six pack abs” no matter how hard you work out.

*   MongoDB 支持动态查询。
*   很容易扩展。
*   不需要复杂的连接。
*   与任何关系数据库相比，性能调优都很容易。
*   由于其使用内部内存进行存储的特性，可以更快地访问数据。
*   也支持按正则表达式搜索。
*   MongoDB 遵循其较新版本的常规发布周期，如 Docker 等。

简而言之，我发现处理时间序列和不同时间序列中的不同特性是最好的数据库。(再举一个例子，我所有的算法音乐数据库最初都存储在 HDF 文件中，现在迁移到 mongodb，还有我处理过的人寿和财产保险的保险数据)。

下面是一个代表这个项目不同阶段的图表，以及我使用的工具:

![](img/136b3941e888e3d7960049e3e2f6c828.png)

在接下来的文章中，我将解释我的系统的一些重要特性，并进一步展开，没有特定的顺序，在一些关键的发现中:

![](img/3704c3590250da037585970de3affb37.png)

The cumulative work load and type of work load for a given unit of time (frequency and intensity) is a VERY important aspect in an optimized fitness program.

1.  营养:食物的种类，进餐的时间，食物的数量。
2.  锻炼的类型和强度:与以下方面的关系:a)改善健康和肌肉质量，b)过度训练和减少肌肉质量，以及 c)保持肌肉质量。因素的顺序很重要。时间序列很重要。
3.  心理方面:对自己的遵从，量化，激励。

在上面的图表中，我发现大多数没有取得任何进展或进展很小的人(大部分人)都在模式“B”和“C”下工作，他们不知道自己的最佳值是多少，因为他们没有收集数据，如果他们收集了数据，他们不知道如何分析数据并将其放在自己的目标环境中。

截至 2019 年 2 月，我目前更多地处于“C”型情况，因为我很难保持前一阵子对自己健身的严格遵守。尽管如此，我还是设法保持了大部分已获得的肌肉质量，并可以在 8-12 周内再次转换到“A”型，从我目前的基线恢复体型。

由于这个项目涵盖了数据科学和健身领域的许多方面，我希望听到您的反馈，告诉我下一篇文章的重点是什么:

1.  数据收集？
2.  数据可视化？
3.  机器学习模型和验证？
4.  其他？

在此之前，我希望听到您的评论/反馈。

如果你喜欢这篇文章，你可能想看看我其他关于数据科学和金融、健身、音乐等的文章。你可以关注我的私人回复[这里、](https://sourcerer.io/lmsanch) twitter 帖子[这里](https://twitter.com/lmsanch)中的活动，或者你可以问下面的 Qs 或者在 SGX 分析的[给我发邮件](mailto:luis.m.sanchez@sgxanalytics.ai)。

如果你对我应用于算法音乐作曲的时间序列分析和模拟很好奇，可以在 [Apple Music](https://itunes.apple.com/us/album/techno-vivaldi/id993458190?i=993458193&app=music&ign-mpt=uo%3D4) 、 [Spotify](https://open.spotify.com/user/1220283645/playlist/1uY0yeJeL7ktn0dlpClWz8?si=lbbjHChRReSDs5bDCt-QPg) 、[和 SoundCloud](https://soundcloud.com/luis-m-sanchez-1/sets/ai-music) 听听我的 AI 生成音乐。

干杯

```
***Parts of this story were originally published in my personal blog a few years back. All the pictures and charts are mine and/or have permission to post.***
```

***注来自《走向数据科学》的编辑:*** *虽然我们允许独立作者根据我们的* [*规则和指导方针*](/questions-96667b06af5) *发表文章，但我们不认可每个作者的贡献。你不应该在没有寻求专业建议的情况下依赖一个作者的作品。详见我们的* [*读者术语*](/readers-terms-b5d780a700a4) *。*