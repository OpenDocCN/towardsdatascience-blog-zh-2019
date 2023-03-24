# Deepfakes:一个未知和未知的法律景观

> 原文：<https://towardsdatascience.com/deepfakes-an-unknown-and-uncharted-legal-landscape-faec3b092eaf?source=collection_archive---------23----------------------->

![](img/15e21d2f9770ed4e4fd685024c3a9cc6.png)

[https://m.wsj.net/video/20181012/1012-deepfakes-finalcut/1012-deepfakes-finalcut_960x540.jpg](https://m.wsj.net/video/20181012/1012-deepfakes-finalcut/1012-deepfakes-finalcut_960x540.jpg)

2017 年标志着“深度造假”的到来，随之而来的是另一个由技术引发的法律模糊的新领域。术语“ [deepfake](https://en.wikipedia.org/wiki/Deepfake) ”指的是使用深度学习篡改的图像、视频或音频片段，以描述并未真正发生的事情。起初，这听起来可能很普通，也没什么坏处——Adobe 的 Photoshop 已经有几十年的历史了，像詹姆斯·卡梅隆的*阿凡达*这样的电影完全依赖于计算机生成图像(CGI)技术。然而，让 deepfakes 与众不同的是它们的[范围、规模和复杂程度](https://muse.jhu.edu/article/715916/pdf)。虽然先进的数字更改技术已经存在多年，但它们在获取方面受到成本和技能的限制。逼真地修改视频和图像曾经需要高性能和昂贵的软件以及广泛的培训，而创建一个深度假几乎任何人都可以用电脑。这种便利带来了新的和前所未有的法律和道德挑战。

***什么是 Deepfake？***

这些令人信服的伪造视频的第一次出现是在 2017 年 12 月的[，当时 reddit 上的一名匿名用户以“Deepfakes”的笔名将名人的面孔叠加到色情视频上。Deepfakes 在这种背景下蓬勃发展，刺激了诸如](https://theweek.com/articles/777592/rise-deepfakes) [DeepNude](https://www.theverge.com/2019/6/27/18761496/deepnude-shuts-down-deepfake-nude-ai-app-women) 这样的应用程序的建立，这些应用程序提供免费和付费的服务来改变女性的图像，使其看起来完全裸体。其他应用如 [FakeApp](https://www.theverge.com/2018/2/11/16992986/fakeapp-deepfakes-ai-face-swapping) 将这种体验推广到更广泛的背景，如欺凌、勒索、假新闻和政治破坏。与这些视频的创作同样令人震惊的是它们的泛滥——在它们短暂的历史中，deepfakes 在主流媒体上掀起了波澜。尤其是当唐纳德·特朗普总统转发众议院议长南希·佩洛西(D-CA)的视频[时，显然是含糊不清的话，或者当一个巨魔发布马克·扎克伯格](https://www.theverge.com/2019/5/24/18637771/nancy-pelosi-congress-deepfake-video-facebook-twitter-youtube)发表威胁性演讲的 [deepfake。](https://www.cnet.com/news/deepfake-video-of-facebook-ceo-mark-zuckerberg-posted-on-instagram/)

***deep fakes 是如何工作的？***

在分析 deepfakes 的法律和社会后果之前，理解它们是如何工作的以及它们为什么如此有效是很重要的。Deepfakes 需要一种特定类型的人工智能(AI)和机器学习，称为[深度学习](https://whatis.techtarget.com/definition/deepfake)，它利用人工神经网络变得更聪明，更善于检测和适应数据中的模式。Deepfakes 依赖于一种特殊形式的深度学习，称为生成对抗网络(GANS)，由蒙特利尔大学的[研究人员在 2014 年推出。GANS 将自己与其他深度学习算法区分开来，因为它们利用了两种相互竞争的神经网络架构。这种竞争发生在数据训练过程中，其中一种算法生成数据(生成器)，另一种算法同时对生成的数据进行鉴别或分类(鉴别器)。神经网络在这种情况下是有效的，因为它们可以自己识别和学习模式，最终模仿和执行。正如凯伦·郝在](https://arxiv.org/abs/1406.2661)[麻省理工学院技术评论](https://www.technologyreview.com/s/612501/inside-the-world-of-ai-that-forges-beautiful-art-and-terrifying-deepfakes/)中所引用的那样，“(甘)的过程‘模仿了一个伪造者和一个艺术侦探之间的反复较量，他们不断试图智取对方。’“虽然这种算法听起来可能相当先进，但它的可用性和可访问性被开源平台如谷歌的 [TensorFlow](https://www.tensorflow.org/) 或通过 GitHub(deep nude 在被关闭前存放其代码的地方)提供的公共存储库放大了。这项技术正变得越来越易于使用，因此也越来越危险，尤其是当它的情报迅速积累，变得越来越有说服力的时候。

![](img/bf70b0c06dbf0560ae9bc901c65cbde4.png)

[https://www.edureka.co/blog/wp-content/uploads/2018/03/AI-vs-ML-vs-Deep-Learning.png](https://www.edureka.co/blog/wp-content/uploads/2018/03/AI-vs-ML-vs-Deep-Learning.png)

***deep fakes 的含义:***

deepfake 视频的后果可能是显而易见的:参与者的公开羞辱，错误信息，以及新的和脆弱的真理概念化。一个更难回答的问题是，如何保护个人，进而保护美国社会，免受 deepfakes 的严重影响。某些州已经采取了措施，例如弗吉尼亚州更新了其[报复色情法](https://techcrunch.com/2019/07/01/deepfake-revenge-porn-is-now-illegal-in-virginia/)以包括 deepfakes，或者得克萨斯州的 [deepfake 法](https://www.theverge.com/2019/7/1/20677800/virginia-revenge-porn-deepfakes-nonconsensual-photos-videos-ban-goes-into-effect)将于 2019 年 9 月 1 日生效，以规范选举操纵。虽然事后追究个人责任可以对其他人起到威慑作用，但一旦这些视频被发布到网上，往往已经造成了严重的损害。因此，在社交媒体和互联网病毒式传播的时代，保护第一修正案权利和最大限度地减少诽谤传播之间的平衡变得不那么明显，特别是随着社交媒体平台成为政治和社会话语的论坛。

除了更新或制定法律，还值得考虑将审查的负担转移到社交媒体平台上，让这些视频获得牵引力和传播。例如，如果脸书在南希·佩洛西的深层假货被认定为假货时将其移除，许多伤害本可以避免。虽然该公司提醒用户视频被篡改，但他们选择不删除视频，以保护宪法第一修正案的权利。然而，在上述扎克伯格 deepfake 发布之后，在 2019 年阿斯彭创意节上，[马克·扎克伯格承认](https://www.theatlantic.com/technology/archive/2019/06/zuckerberg-very-good-case-deepfakes-are-completely-different-from-misinformation/592681/)脸书可能会在未来对他们的 deepfake 政策进行修改。但是到目前为止，[1996 年《通信行为规范法》第 230 条](https://www.minclaw.com/legal-resource-center/what-is-section-230-of-the-communication-decency-act-cda/)保护脸书或任何其他社交媒体平台免于为其用户发布的内容承担任何责任。这实际上给了这些平台更多的自由裁量权来决定宪法权利的范围。随着 deepfakes 变得更加普遍和有害，政府必须找到一种方法来监管大型科技公司，并调和它们已经成为民主平台的事实。

另一个需要记住的法律问题是法庭上的深度假证。多年来，数字图像和视频镜头已经为沉默证人理论下的可验证和可靠的证据创造了条件。然而，如果 deepfakes 持续生产，这些图像的可验证性将变得更加模糊和难以证明，从而根据美国联邦证据规则的[规则 901(b) (9)加强对所有数字证据的审查。对数字证据全面丧失信心的后果将是](https://www.law.cornell.edu/rules/fre/rule_901)[增加对目击者证词的依赖](https://journals.sagepub.com/doi/abs/10.1177/1365712718807226)，从而让法院在已经有偏见的系统中做出更有偏见的裁决。无论在何种背景下，法律体系都必须预测和调整 deepfakes 崛起带来的巨大变化，特别是因为它们的传播和影响难以遏制。

***打击深度假货:***

除了考虑社交媒体网站的潜在监管变化和预测法律困境，重要的是要考虑替代方法来规避 deepfakes 造成的潜在损害。罗伯特·切斯尼和丹妮尔·香橼提出了一种解决方案，即“[认证不在场证明服务](https://www.foreignaffairs.com/articles/world/2018-12-11/deepfakes-and-new-disinformation-war)”。理论上，这些服务将创建跟踪个人的数字生活日志，以便他们最终可以推翻 deepfakes 提出的说法。这种解决方案很有意思，因为它损害了个人隐私，可能会遭到错误信息的拒绝。也许这表明了社会对隐私的冷漠，或者我们对假货的恐惧程度——或者两者兼而有之。无论如何，经过认证的不在场证明看起来像是一种绝望的防御措施，揭示了个人为了保护自己愿意牺牲多少。

Deepfakes 对美国社会构成严重威胁。虽然政治极化和假新闻已经玷污了事实，因此，我们的民主基础，我们必须制定战略，以防止信任和真理的进一步崩溃。因此，预测法律问题并相应调整法律，对于确保利用创新技术改善社会和抑制恶意工具的传播至关重要。