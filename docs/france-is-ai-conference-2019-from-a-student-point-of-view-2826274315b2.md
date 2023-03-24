# 法国 2019 年人工智能大会:从学生的角度看

> 原文：<https://towardsdatascience.com/france-is-ai-conference-2019-from-a-student-point-of-view-2826274315b2?source=collection_archive---------39----------------------->

![](img/65296f30a5c93e93ed39d694ea059135.png)

The amazing team of organizers!

我知道距离我的上一篇文章已经过去了很长时间，发生了很多事情，我有其他的优先事项，我的一些其他文章仍然需要结束，但这次不是，所以我们带着一些新的东西再来一次！

今天，我将讲述我上周参加的一个非常有趣的活动，我将尝试综合我所理解的内容，并给你一些我在研究后发现的有用链接，如果你想深入了解的话。

长话短说:去年 6 月作为一名计算机系统工程师毕业，现在我在巴黎攻读计算机视觉硕士，在大城市的一个优势是你可以找到的活动和会议的数量和质量，我很幸运地发现了“[法国是 AI](https://franceisai.com/) ”。

法国是人工智能，是 2016 年发起的一项倡议，旨在支持和促进法国人工智能生态系统。活动在巴黎市中心的创业园区[站 F](https://stationf.co/) 举行。

![](img/45303ee731eabe4dbdc83903a6629dd4.png)

STATION F, Paris (France)

老实说，我想去的主要原因是为了见见伟大的 [Yoshua Benjio](https://mila.quebec/yoshua-bengio/) ，但不幸的是，他因腿骨折而无法参加活动，然而，其余发言者的质量正如我所期待的那样好，在大多数情况下，我必须实时学习许多新单词才能赶上。

我将尝试从我的角度对我所看到的给予反馈，并给你一些链接，以防你想看得更深。你可以在这里找到会议的日程[。正如你所看到的，这是非常激烈的，所以我将只谈论我可以跟随的人，我想提一下，我喜欢他们使用的细节和技术词汇的水平，这在其他活动中通常不会出现。](https://francedigitale.tilkee.io/v/71cb1328dd)

我记得的第一个演讲是“关于多智能体学习的评估和培训”，由 Google DeepMind 的一位研究科学家做的，在那里他[展示了](https://photos.app.goo.gl/GCWdVKTX2d39U1E87)他们的最新成果: [α-Rank](https://www.nature.com/articles/s41598-019-45619-9.pdf) ，这是一种在大规模多智能体交互中对智能体进行评估和排序的方法。演讲者鼓励寻找更古老的方法，并举了一个强化学习的例子，这是一种古老的方法，但在过去几年里引起了科学界的极大兴趣。然后，他给出了使用纳什均衡的 3 个缺点(详情请查阅论文)，以及为什么他们使用马尔可夫-康利链。

下一位演讲者是 NAVER LABS 的研究员 Naila Murray，她谈到了“在虚拟世界中学习视觉模型”,她展示了他们在过去几年中一直在做的一系列[工作](https://europe.naverlabs.com/Research/Computer-Vision/Learning-Visual-Representations/),这里我要打断一下，我不知道这意味着什么，但根据我的理解:使用合成数据训练模型时存在过度拟合的问题，他们找到了一些使用合成视频生成真实世界数据的解决方案，如果我错了，请纠正我。

然后我们与谷歌大脑研究员 Nicolas Papernot 讨论了对抗性攻击(和著名的“熊猫”照片)，以及基于 DNN 的解决方案中的[隐私保护](https://arxiv.org/abs/1906.02303)，以及在使用[私人训练数据](https://arxiv.org/pdf/1610.05755.pdf)时如何使用有效的方法。

![](img/3482cfdbd891a741c6850a338b295e4c.png)

If you reverse-engineer a model, you can easily trick it to have this kind of results

然后由 [Cretio](https://www.criteo.com/) 做了一个关于他们如何大规模使用广告推荐系统的演示，并使用了一个 [SVD](https://en.wikipedia.org/wiki/Singular_value_decomposition) 变换近似(随机化 SVD)和非常大的矩阵。[在这里](https://photos.app.goo.gl/vNn8Ca8bJGE38RJ37)你可以看到他们系统的简要描述。

![](img/346f8db8d6ffa3c9914bfe12ddaa4532.png)

Criteo recommender systems presentation

下一个发言人伤害了我的感情😂我的意思是我去年的毕业设计是关于特征选择的，听她的演讲让我觉得:

![](img/d32ce78b9b4d44b29d42649f8f117861.png)

更严重的是，他们正在处理一个“[高维小样本](https://photos.app.goo.gl/MVjx34XM8FNqGkFJ9)”的问题，当你有这么多特征，但没有足够的数据来训练一个稳健的模型时，这就是基因组学的主要困难之一。她提出了三个建议:1)使用结构化的先验知识 2)对每对特征和表型之间的关联进行统计测试 3)使用[核](https://storage.googleapis.com/pub-tools-public-publication-data/pdf/158ffda8848df14ff80c5902891bc73b2cac890c.pdf)。

我喜欢下一位演讲者，让·庞塞，Inria 的研究主管，不仅仅是因为这完全直接关系到我的研究领域，还因为他用来说服你这些[挑战的重要性的](https://photos.app.goo.gl/Q4d6dnkn8eYPiYyq5)[论据](https://photos.app.goo.gl/pzZkoNEmyKxswNiL7)和方法。

更多关于对抗性攻击的演讲是由巴黎多菲纳大学教授兼 MILES 的负责人 Jamal Atif 提出的。我以前没有这方面的知识，我听不懂他在说什么，但他介绍了他的团队开发的许多方法，这些方法似乎非常有效。

然后，就在午餐前，我们听取了来自纽约一家专注于 NLP 的初创公司[humping Face](https://medium.com/u/b1574f0c6c5e?source=post_page-----2826274315b2--------------------------------)的 Thomas Wolf 的演讲，他在演讲中介绍了[迁移学习](https://photos.app.goo.gl/6mABPpxFKpNf735h7)，其[工具](https://photos.app.goo.gl/qECjtHgDC1x3dqrA9)，以及[趋势](https://photos.app.goo.gl/JAQh5Yqj7QxCuGfR9)。我认为这是一家非常有前途的初创公司，尤其是在开发了他们的版本[BERT](https://github.com/google-research/bert)distil BERT 之后。

休息之后，我们听取了夏羽·马利特关于“深度神经网络的数学解释”的演讲，以及他如何在学习的重要性上[输给了](https://photos.app.goo.gl/FPt7zdRT7WFXMPsF6)(他试图证明你不需要有一个大的数据集来获得好的结果)。

在那之后，Google 的工程主管 Noe Lutz——Brain Applied Zurich 向我们介绍了 TensorFlow 2.0，我不得不说他们所做的改进是惊人的，尤其是因为我有很多东西要赶上(我猜从 1.7 开始就没有使用 tf 了)。

“使用机器学习管理城市移动性”是 Dominique Barth 做的一个非常有趣的演讲，因为它完全是关于使用强化学习( [1](https://photos.app.goo.gl/x6hzjiLEhq5Bt6Jf9) 、 [2](https://photos.app.goo.gl/d4Gh8Vy7md8Bxf1B6) 、 [3](https://photos.app.goo.gl/GFpaoGkoCvqFSSX29) 、 [4](https://photos.app.goo.gl/2F4gYRJwR3BCajGN6) 、 [5](https://photos.app.goo.gl/Ybasv6x2xh3nrmnE8) 和 [6](https://photos.app.goo.gl/uJja7oWVkUJ5uFZ57) )来解决这个问题，并且在一定程度上理解演讲者所谈论的内容总是一件令人愉快的事情。

接下来的[演讲](https://photos.app.goo.gl/HNH7RaPF1VgggYJ7A)由脸书人工智能研究所的研究科学家 Natalia Neverova 进行，她指出视觉人类理解的下一个里程碑是获得[图像中每个像素的语义含义](https://photos.app.goo.gl/vEdeb6ZhyANZ6Rqk7)以及它们到物体特定部分的映射，以及从 2D 重建 3D 姿势和形状的问题。跟踪最先进的问题是很重要的，因为即使你只是一名学生或来自行业，你也可以为解决这些问题做出贡献，因为你有自己的观点，可能连像 FAIR 这样的大实验室都没有考虑到。

下一位演讲者是微软法国公司的 CTO & CSO Bernard Ourghanlian，他再次谈到了人工智能中的隐私、对抗性攻击和伦理，以及[全同态加密](https://photos.app.goo.gl/sad8VVZfDi6Lq4GX6)方法如何解决许多问题，并解释了其背后的[动机](https://photos.app.goo.gl/ezTH6dqREbngESFa7)，还说这是一种[有前途的](https://photos.app.goo.gl/K4W1U5zREC3emFzp8)方法，即使它可能会产生潜在的性能成本。他还展示了用于此目的的微软编译器 CHET。

在这个休息时间，我有机会与谷歌苏黎世机器学习研究负责人 Olivier Bousquet 面对面交谈，我问他他们正在研究的热门话题，不出所料，强化学习再次出现在桌面上，这使得讨论更加有趣，因为我可以保持对话活跃(不幸的是，我不太擅长与女士们交谈😥).

说到 Olivier，他是下一个演讲者，他的演讲是关于“解决许多任务的模型”。他首先展示了人工智能在具体任务方面有多好，如 [ImageNet 分类](https://photos.app.goo.gl/UA5KJbVSjU1d8PtT6)，以及它们如何超越了人类现在的误差容忍水平(最高 1%误差< = > 99%+准确率)。但是他说我们仍然[远没有解决](https://photos.app.goo.gl/tqqchbjvBLrX4xkd8)视觉或语言的实际问题，为什么我们不考虑一个[模型来解决许多任务](https://photos.app.goo.gl/j17jFzTBUctJrG2j8)。怎么做呢？他认为有一个[秘方](https://photos.app.goo.gl/n74uZq9ndDJHX4mt6)可能行得通(提示:是自我监督和[转移学习](https://photos.app.goo.gl/2Nw1VECPj9MKgnYd6))他举了 [BERT](https://photos.app.goo.gl/hNiQMMNo3NTAu5HRA) 的例子(不是芝麻街的那个)。最后，他谈到了一种[方法](https://photos.app.goo.gl/JfFoqMNrwxxPcn4x8)他们正在研究制作多任务模型:通过使用多个基本模型。

下一个演讲是关于“将纵向形状数据集聚类成独立或分支轨迹的混合物”，由我们大学的教授 Stephanie Allassonniere 主讲😎。他们的[贡献](https://photos.app.goo.gl/XhLeJiYgX6AJdTSb6)是从一组患者中重组疾病的进化，以提供整个群体的进化。怎么会？我不会比她解释得更好( [1](https://photos.app.goo.gl/osFGNGWp7fSKoAbh9) 、 [2](https://photos.app.goo.gl/WFqGkHbbZMmMFXaY7) 、 [3](https://photos.app.goo.gl/zpQvxJ5SN1vC2Yf27) 、 [4](https://photos.app.goo.gl/K4SisdwmWg5kLmNL7) 、 [5](https://photos.app.goo.gl/CADRDMkJ3jGRF9cp7) )，但总的来说，我所理解的主要思想是根据特征子集来推断疾病的演变。

![](img/7ad224f974e95fd577f34ca361ceedd5.png)

Test, learn then Scale, I really like this “motto”

下一位演讲者是史蒂夫·贾勒特，他介绍了人工智能在法国跨国电信公司 Orange Group 的应用，他是人工智能、数据工程和人工智能战略的负责人。我真的很喜欢他将今天的人工智能与 1995 年的手机进行比较，以及 Orange 对非洲国家的愿景，这些国家不一定能接入互联网。

下一个演讲是由来自 Criteo 的研究员 Vianny Perchet 所做的“竞争与合作的人工智能”。这让我看到了数据的一个非常关键的用例:你如何确保你用来训练你的模型的数据的完整性，尤其是在竞争中？你怎样才能找到一个双赢的局面或者一个“纳什均衡”？

贾丝汀·卡塞尔的演讲与我们一整天所看到的有所不同，她谈到了“社交人工智能”，以及这些系统的目标如何不是达到人类的水平，而是，特别是在与孩子们互动时，[不要愚弄他们](https://photos.app.goo.gl/Ajijm43RY1BYpkXE9)，并给他们能力说这不是一个真正的人类，但即使在这种情况下，[利用它的帮助](https://photos.app.goo.gl/DptLk16mrjjxJpa1A)。

我错过了下一位演讲者(来自 INRIA/ENS 的 Francis Bach)的大部分演讲，他谈到了“机器学习的分布式优化”，但你可以从他的[结论](https://photos.app.goo.gl/5TgSFb4G7nvoBNn8A)中看到他，给出了他们得到的一些结果以及关于如何优化分布式 ML 算法的其他观点。

来自 CentraleSupelec 的 Maria Vakalopoulou 向我们介绍了他们最近在医疗保健以及人工智能和医学成像方面的工作，以及她对该领域未来的看法。

现在，大家都在等待的演讲开始了:2018 年图灵奖得主(严和辛顿)因其对人工智能进步的卓越贡献而获奖，Yoshua Bengio，正如我们之前提到的那样[不能来](https://photos.app.goo.gl/3ZPXpkxgyTPndNsZ7)，但仍然通过视频电话发表了他的演讲。他提到了很多方面，其中之一是目前的系统 1 和系统 2 认知类别，最后，他谈到了他将如何定义达到“人类水平的人工智能”的下一步，并缩小两个系统之间的差距。他在两周前做了一个类似的演讲，内容完全相同，所以你可以在这里查看。

然后，我们有一个有趣的小组，讨论如何在欧洲进行人工智能监管，以及各国如何合作，在当地监管的基础上增加一个共同的、更强有力的监管。

Nicolas 和 Emmanuel 透露，就人工智能投资而言，法国显然是欧洲第一国家，主要是因为英国因英国退出欧盟困境失去了一些兴趣，但欧洲仍有很长的路要走，才能赶上美国。

最后，我知道这是一篇很长的文章(你可以想象我们当时的日子)，我要感谢主办方:[法兰西数码](https://medium.com/u/f741a713edcb?source=post_page-----2826274315b2--------------------------------)，感谢他们邀请我参加这次会议。我来到法国才一个月，我不知道其他会议/活动如何，但我喜欢大多数演讲中的技术含量。

如果你在那里，你有什么要补充的，或者你想纠正什么，请留下评论，我接受所有的反馈，因为我还在学习。

你正在阅读 Amine 的一篇文章，他是一名年轻的计算机视觉学生，他是这样进行机器学习的:

![](img/197b36dd0014fabc88e5813f28a0e99c.png)

Credits: [instagram.com/neuralnetmemes](https://www.instagram.com/neuralnetmemes/)