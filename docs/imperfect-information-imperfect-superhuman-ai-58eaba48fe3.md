# 不完美的信息，不完美的超人人工智能

> 原文：<https://towardsdatascience.com/imperfect-information-imperfect-superhuman-ai-58eaba48fe3?source=collection_archive---------29----------------------->

![](img/dfdca7ba2949ba21f7735c560a1cfcd3.png)

[1]

应该允许 AI 参与我们的赌博吗？卡耐基梅隆大学的脸书人工智能研究人员刚刚创造了一种可以在扑克中击败人的人工智能。这是一项巨大的成就。在扑克游戏中，没有最佳答案，也没有要寻找的一系列赢棋。相反，最大化回报和最小化风险是最好的赌博。但是，我们能负责任地运用训练有素的代理人在不完全信息的情况下取得成功吗？这些类型的超人人工智能会在哪些方面下错赌注，我们如何围绕它们设计系统以造福社会？

![](img/e21e6853464641746a4c5bda62e440c0.png)

[2]

我们来看一个思维实验。医疗诊断和治疗是机器学习研究中最发达和最多产的领域之一。诊断在很大程度上是一个分类问题。你有大量来自患者的输入数据，比如症状数据、环境数据等。机器学习算法用于在大量数据中寻找模式，以诊断患者。通常算法发现的模式是如此错综复杂，以至于专业人士并不总是能理解它们。诊断是一个经典的机器学习应用，伴随着它的伦理问题。为了实验起见，我们假设医生非常敏锐或者一个 ML 算法准确地给出了正确的诊断。

然而，推荐治疗并不是一个标准的分类问题。这是一个不完全信息的游戏。你必须根据个人情况和诊断结果来选择最佳的治疗方案。算法的工作是观察治疗、诊断和个人的各种成功机会，并推荐最佳的治疗方案来挽救他们的生命。他们可能没有时间接受更多的治疗，所以每一个建议都必须是高质量的。这个问题是脸书和 CMU 的研究可能被应用的地方。毕竟，如果每个人都得到正确的诊断，并给予尽可能最好的治疗，那难道不是一个值得建设的世界吗？

![](img/1e675c4d620f72c2c5c196a3792a911c.png)

[3]

可悲的是，这可能是不可能的。如果我们让一个人工智能代理来推荐治疗，我们可能会发现和医生一样的盲点。推荐治疗方法的代理人可能不得不考虑各种疗法的成功率，作为要分析的许多数据特征的一部分。成功率可能会被 p-hacked，或者不诚实地操纵统计数据，使其高于实际水平。例如，[立普妥，一种宣称有 36%成功率而实际只有 1%成功率的药物。](https://www.crossfit.com/health/the-cardinal-sins-of-skewed-research-part-2-racking)人工智能代理推荐治疗方法并查看成功率等指标可能会下错赌注。

医生也可能成为错误成功率的牺牲品。这个弱点就是问题所在。医生和超人 AI 都有同样的缺陷:腐败或有偏见的数据。但是医生可以为他们的诊断提供理由。尽你所能去寻找，但是在 50，000 个矩阵中找出以不同的顺序方式相乘的确切模式并正确识别其原因是不现实的。也许你可以制造一个人工智能来理解并把人工智能的见解翻译成人类的理解，但是你会再次遇到同样的问题。

即使我们能够在信息不完善的领域超越人类，我们仍然依赖于手头问题的准确和无偏见的数据。赌一赌这是否可能是人工智能未来的一个关键问题。

![](img/3ffe43bfa05d6efa55fb5cd270222a2d.png)

[4]

图片来源:[【1】](https://proxy.duckduckgo.com/iu/?u=https%3A%2F%2Fwww.luxor.com%2Fcontent%2Fdam%2FMGM%2Fluxor%2Fcasino%2Ftable-games%2Fluxor-casino-table-games-cards-chips.tiff&f=1)[【2】](https://proxy.duckduckgo.com/iu/?u=https%3A%2F%2Fgovernmentciomedia.com%2Fsites%2Fdefault%2Ffiles%2Fstyles%2Ffeatured_article_image%2Fpublic%2F2018-05%2Frobot%2520ai%2520medicine%2520health.jpg%3Fitok%3De5PYXMcY&f=1)[【3】](https://proxy.duckduckgo.com/iu/?u=http%3A%2F%2F2s7gjr373w3x22jf92z99mgm5w-wpengine.netdna-ssl.com%2Fwp-content%2Fuploads%2F2015%2F11%2Fshutterstock_error_data_alexskopje.jpg&f=1)[【4】](https://proxy.duckduckgo.com/iu/?u=https%3A%2F%2Fupload.wikimedia.org%2Fwikipedia%2Fcommons%2Fthumb%2Fe%2Fe6%2FHazy_Crazy_Sunrise.jpg%2F1200px-Hazy_Crazy_Sunrise.jpg&f=1)