# 着眼于人工智能的案例研究

> 原文：<https://towardsdatascience.com/looking-at-ai-focused-case-studies-139e0bb98ff5?source=collection_archive---------10----------------------->

## 除了讨论它们将如何在美国和欧盟发展之外

这篇文章是为 Darakhshan Mir 博士在巴克内尔大学的计算机和社会课程写的。我们讨论技术中的问题，并用伦理框架来分析它们。 [Taehwan Kim](https://medium.com/u/cf7b96fac557?source=post_page-----139e0bb98ff5--------------------------------) 和我一起合作这个项目。

![](img/9b82096fba37509f2f4d80fad48874b4.png)

Photo by [Franck V.](https://unsplash.com/@franckinjapan?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

在我的[上一篇帖子](/impact-of-automation-on-the-economy-33ef17352f5b)中，我讨论了人工智能(AI)对就业市场的影响，并强调了在监管人工智能方面的远见的必要性。我认为，我们的社会无法应对日益依赖人工智能所带来的道德后果。许多国家的技术领导层都认识到了这个问题，在过去的几年里，他们都提出了以有效的方式促进人工智能发展的战略。[国家人工智能战略概述](https://medium.com/politics-ai/an-overview-of-national-ai-strategies-2a70ec6edfd)简要讨论了自 2017 年初以来提出的不同人工智能政策。

![](img/e9fc346880acbe71b0befd5172418cfd.png)

**Figure 1:** No two policies are the same. They focus on different aspects of AI: the role of the public sector, investment in research, and ethics. | [Tim Dutton](https://medium.com/politics-ai/an-overview-of-national-ai-strategies-2a70ec6edfd)

在这篇文章中，我主要关注美国和欧盟提出的政策之间的差异。此后，我讨论了三种(半)假设情景，以及它们如何在人工智能发展有巨大反差的两个地区上演。为了进行分析，我广泛使用了[数据伦理介绍](https://www.scu.edu/media/ethics-center/technology-ethics/IntroToDataEthics.pdf)中描述的伦理框架。

# 美国和欧盟人工智能政策比较

![](img/702f27d3ef60f0d07e05ead77f90a450.png)

The US focuses on continued innovation with limited regulations from the government. | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Flag-map_of_the_United_States.svg)

2016 年 10 月，白宫发布了首个应对人工智能带来的社会挑战的战略[3]。这份报告强调了公开 R&D 的重要性，以及让任何推动人工智能研究的人承担责任的重要性。这表明，应该允许这一领域的创新蓬勃发展，公共部门应该对私营部门实施最低限度的监管。人们的理解是，以自由市场为导向的人工智能发展方式几乎不需要政府干预。

最近的一份报告[4]发表于特朗普政府执政一年后，重点关注保持美国在该领域的领导地位，消除创新障碍，以便公司雇佣本地人，而不是搬到海外。

![](img/363dcf7136a64309d1c00bc5cc084a32.png)

The EU promotes greater regulation while still being competitive in the global AI ecosystem. | [Wikimedia Commons](https://commons.wikimedia.org/wiki/File:Flag_map_of_the_European_Union.png)

另一方面，欧盟公布的报告将 AI 视为“智能自主机器人”的组成部分[5]。人工智能被认为是其他技术系统自动化的推动者[6]。该报告建议在自主、人类尊严和不歧视原则的基础上建立更严格的道德和法律框架。事实上，这是在 2018 年欧盟委员会通过关于人工智能的[通信时进一步发展的。原始报告还讨论了公共部门在确保上述道德框架得到贯彻而不带有行业偏见方面的主要责任。](https://ec.europa.eu/digital-single-market/en/news/communication-artificial-intelligence-europe)

## 案例研究一

> 众所周知，一家总部位于硅谷的公司在人工智能研究方面投入了大量资金。最近，他们提出了一种最先进的生成对抗网络(GAN)，能够生成超现实的人脸。该程序可以生成特定性别、种族和年龄的人脸。

GANs 的应用越来越多，通过其使用传播错误信息的风险也越来越大。在这个虚假内容如此盛行的时代，GANs 对已经在进行的打击虚假信息的努力构成了重大挑战。[这个 X 是不存在的](https://thisxdoesnotexist.com)是一个汇编流行甘榜的网站。他们中的大多数都是无害的，然而，[这个人并不存在](https://thispersondoesnotexist.com)引发了关键的伦理问题。

![](img/909b6134b9c776e43b856f17f84aeb8e.png)

**Figure 2:** [StyleGAN](https://github.com/NVlabs/stylegan) developed by NVIDIA. All these faces are “fake.” Or are they?

该项目可能带来的好处和危害风险是什么？

正如我之前所写的，美国对研发的关注相对更大。这样的项目可能会获得进一步的资助，希望该程序生成的人脸可以取代其他面部识别算法所需的训练数据。与此同时，虚假身份的风险可能会增加，人们会伪装成与他们的面部特征非常匹配的人。

然而，在欧盟，由于更严格的法规，类似的项目可能永远不会普及。然后，其他机器学习算法只能在“真实”人脸上训练，这反过来可能会导致隐私问题。在这种情况下，数据从业者遵循合乎道德的数据存储实践非常重要。已经有无数的例子，人们的照片在未经他们同意的情况下被下载，然后用于人工智能实验。

## 案例研究二

> 西雅图华盛顿一家初创企业 X，想投身电商行业。最近，X 从华盛顿大学招募了许多有才华的计算机科学毕业生，并建立了一个新颖的配对算法来匹配消费者和产品。当 X 对他们的产品进行匿名调查时，他们发现大多数参与者更喜欢他们的产品，而不是目前主导电子商务行业的 Y 公司的产品。然而，由于缺乏数据和消费者，X 很难蓬勃发展。

我的上一篇文章讨论了大数据集作为人工智能初创公司进入壁垒的概念。如 Furman 等人[7]所述，依赖于持续互联的用户网络的企业，无论是直接的(如脸书)还是间接的(如亚马逊)，都比同行业的进入者具有优势。

***两个经济体中的利益相关者受到怎样的影响？这里最相关的道德问题是什么？***

最大的利益相关者是竞争者和他们各自的客户。尽管美国和欧盟都有反垄断法，但自本世纪初以来，这两个地区在反垄断法规方面有所松动[8]。由于大量的游说，美国的公司享有“反垄断执法的宽大”，而欧盟的公司通常更具竞争力[9]。

![](img/d4268c53192d82ebf777a6e0a7ffc973.png)

**Figure 3:** FANG + Microsoft dominate the US tech industry as well as investments in AI. [10] | [Image Source](http://theconversation.com/big-tech-isnt-one-big-monopoly-its-5-companies-all-in-different-businesses-92791)

因此，美国不太可能采取允许竞争者之间共享数据的数据可移植性政策。这将阻碍任何由垄断(搜索引擎)或寡头垄断(拼车)主导的行业中新来者的增长。特别是在我们的案例研究中，初创企业可能很难保持相关性并吸引更多客户，因为现任者知道更大客户群的偏好。这也有可能导致针对买不起产品的目标人群的歧视性营销行为。

在欧盟，数据共享将伴随着无数的数据存储和隐私问题。重要的是，公司要制定缓解策略，以防未经授权的第三方访问客户数据。

## 案例研究三

> 一位拥有欧盟和美国双重国籍的 CEO 经常往返于工作地点(美国)和住所(欧盟)。最近，一些关于她个人生活的谣言在互联网上浮出水面，这严重影响了她的公司的声誉。她希望找到一种方法来消除谣言，但她受到她居住的两个地方的政策的限制。

![](img/b6011cb691bce7d96f841acc1660f384.png)

**Figure 4:** According to Bertram et al. [11], Google had received 2.4 million requests to delist links in the first three years of Right to be Forgotten. | [Image Source](https://en.softonic.com/articles/how-to-delete-google-search-results)

2014 年 5 月，欧洲法院引入了“被遗忘权”，让人们对自己的个人数据有更多的控制权。它允许欧洲人请求一个搜索引擎，比如谷歌，从它的搜索结果中删除特定的链接。搜索引擎没有义务接受每一个除名请求。它衡量一个人的隐私权是否比公众对相关搜索结果的兴趣更重要。

有趣的是，被遗忘的权利在欧洲以外并不适用，因此即使对欧洲人来说，美国的搜索结果也保持不变。

***对社会的透明和自治有什么危害吗？***

这两个概念是密切相关的，经常是携手并进的。Vallor 等人[2]将透明度定义为能够看到一个给定的社会系统或机构如何工作的能力。另一方面，自主是掌握自己生活的能力。

被遗忘权在欧盟的采用促进了更大的自主意识，因为如果数据*合理地*不准确、不相关或过多，个人可以删除他们的个人数据。然而，这是很棘手的，因为搜索引擎本身，而不是一个独立的第三方，负责决定是否删除所请求的链接。

搜索引擎公司已经拥有很大的权力，让他们决定退市请求可能会导致“个人、社会和商业”的损失[2]。我们不禁要问，在引入权利之后，建立一个公正的机构来处理这些请求是否是合乎逻辑的下一步。

***有没有下游影响？***

在美国，被遗忘权在很大程度上是不允许的，因为它可能会限制公民的言论自由。总体而言，这可能会导致一个更加透明的社会，尽管代价是对搜索结果中出现的个人数据的自主权减少。这种情况的一个极端副作用是，公司可能会丢失大量“相关”数据，这些数据原本可以用来训练他们的人工智能算法。

# 结论

正如我们在上面的三个案例研究中看到的，美国和欧盟关注人工智能政策的不同方面。除了促进行业的公共 R&D，美国还提倡自由市场的自由理念。欧盟有更严格的法规，但希望保持其在人工智能领域的竞争优势。与美国不同，欧盟不想将监管责任推给私人部门。

![](img/36eabb43000642d300bdd5fa59e6920f.png)

[Image Source](https://medium.com/syncedreview/nvidia-open-sources-hyper-realistic-face-generator-stylegan-f346e1a73826)

Taehwan 和我曾认为，对人工智能的开发和使用制定更多的规定会减少问题。然而，案例研究讲述了一个不同的故事。每种策略都提出了自己的一系列伦理问题。例如，投资 GANs 可能意味着在数据隐私问题和虚假身份之间做出选择。在数据驱动市场的情况下，监管机构需要衡量是否要实施数据可移植性做法，以打破垄断。

## —前进的道路

上述(极端的)情况是可以避免的，只要有一个从各种各样的利益相关者那里获取信息的计划。这两个地区提出的策略是全面的，但是正如 Cath 等人[6]所认为的，回答下面的问题将是形成*完美政策*的关键。

> 二十一世纪成熟的信息社会的人类工程是什么？

这篇论文提供了一种双管齐下的方法来清晰地理解我们对人工智能社会的愿景。首先，作者建议成立一个独立的委员会，在包括政府和企业在内的不同利益相关者之间进行调解。第二，他们要求把人的尊严放在所有决定的中心。从本质上说，这样做是为了确保在决策中考虑到最脆弱的利益攸关方的利益。

毫无疑问，要实现这个雄心勃勃的计划，需要大规模的协作努力。然而，我们可以从类似的承诺中得到启发，例如最近的《一般数据保护条例》(GDPR) [13]。欧盟和美国仍然需要各自的人工智能政策，但至少讨论它们的优先事项可以帮助它们达成一些共识。

## 参考

[1]蒂姆·达顿。“国家人工智能战略概述。”*政治+AI*(2018 年 6 月)。

[2]瓦勒，香农和雷瓦克，威廉。《数据伦理导论》

[3]总统办公厅国家科学技术委员会技术委员会。"[为人工智能的未来做准备](https://obamawhitehouse.archives.gov/sites/default/files/whitehouse_files/microsites/ostp/NSTC/preparing_for_the_future_of_ai.pdf)"(2016 年 10 月)。

[4]科学和技术政策办公室。"[2018 年美国工业人工智能白宫峰会摘要](https://www.whitehouse.gov/wp-content/uploads/2018/05/Summary-Report-of-White-House-AI-Summit.pdf)"(2018 年 5 月)。

[5]内部政策总局。"[欧洲机器人民法规则](http://www.europarl.europa.eu/RegData/etudes/STUD/2016/571379/IPOL_STU(2016)571379_EN.pdf)"(2016).

[6] Cath，Corinne 等人，“人工智能和‘好社会’:美国、欧盟和英国的方法。”科学与工程伦理(2018)。

[7]弗曼、杰森和罗伯特·西曼。“人工智能与经济。”创新政策与经济(2019)。

[8]古铁雷斯，赫尔曼和菲利庞，托马斯。"欧盟市场如何变得比美国市场更具竞争力:一项关于制度变迁的研究."美国国家经济研究局(2018 年 6 月)。

[9]古铁雷斯，赫尔曼和菲利庞，托马斯。"[你在美国支付的费用比在欧洲多](https://www.washingtonpost.com/news/theworldpost/wp/2018/08/13/middle-class/?noredirect=on&utm_term=.29d3023e1ce1)"《华盛顿邮报》(2018 年 8 月)。

[10]布欣，雅克等人。艾尔。"人工智能:下一个数字前沿？" [MGI 报告](https://www.mckinsey.com/business-functions/mckinsey-analytics/our-insights/how-artificial-intelligence-can-deliver-real-value-to-companies)，麦肯锡全球研究院(2017 年 6 月)。

[11]伯特伦、西奥等，“被遗忘的权利的三年”Elie Bursztein 的网站(2018)。[链接](https://drive.google.com/file/d/1H4MKNwf5MgeztG7OnJRnl3ym3gIT3HUK/view)。

[12]下议院科学技术委员会。“机器人和人工智能。”(2017).

13 欧洲联盟。《一般数据保护条例》(2018 年 5 月)。[链接](https://eugdpr.org/)。

如果你喜欢这个，请看看我的其他[媒体文章](https://medium.com/@ymittal)和我的个人博客[。请在下面评论我该如何改进。](http://yashmittal.me/blog)