# 数据科学促进我们的智力发展

> 原文：<https://towardsdatascience.com/data-science-for-our-mental-development-3fccbd29eff2?source=collection_archive---------16----------------------->

![](img/f7752a38c3362f58f925ab4f0e11fe81.png)

情感是人类社会的基本要素。仔细想想，一切值得分析的东西都是受人类行为影响的。心怀不满的员工对网络攻击影响很大，他们可能会忽视尽职调查，或者参与[内部人员滥用](https://www.ca.com/content/dam/ca/us/files/ebook/insider-threat-report.pdf)。股票市场取决于经济气候的影响，而经济气候本身又取决于大众的总体行为。在交流领域，众所周知,[我们说的话只占信息的 7%,而其余 93%都编码在面部表情和其他非语言暗示中。心理学和行为经济学的整个领域都致力于这一领域。也就是说，有效测量和分析情绪的能力将使我们能够以显著的方式改善社会。例如，加州大学旧金山分校的心理学教授](http://www.nonverbalgroup.com/2011/08/how-much-of-communication-is-really-nonverbal)[保罗·艾克曼](https://psychology.berkeley.edu/people/paul-ekman)在他的书 [*中描述了说谎:在市场、政治和婚姻中欺骗的线索*](https://www.amazon.com/Telling-Lies-Marketplace-Politics-Marriage-ebook/dp/B00ECXIF6C) 中，当病人对这种意图撒谎时，阅读面部表情可以帮助心理学家发现潜在自杀企图的迹象。听起来像是面部识别模型的工作？神经映射呢？我们能有效地从神经冲动中映射出情绪状态吗？提高认知能力呢？甚至情商和有效沟通？使用我们可用的大量非结构化数据，世界上有许多问题需要解决。

话虽如此，就像每个数据科学问题一样，我们需要深入研究情感建模的核心挑战:

*   如何框定问题？我们的示范课应该是什么样的？我们在优化什么？
*   我们应该收集哪些数据？我们在寻找什么相关性？我们应该在哪些方面比其他方面投入更多？
*   获取这些数据有什么问题吗？关于获得情感的社会和文化观点是什么？需要维护哪些隐私法规？数据安全呢？

有关有效设计人工智能产品的更多信息，请阅读我的 [*UX 数据科学家和人工智能产品设计指南*](/ux-design-guide-for-data-scientists-and-ai-products-465d32d939b0) 。在这篇博客中，我旨在概括人工智能如何在未来帮助我们开发智力，并讨论一些当今的解决方案。

# 卫生保健

病人对医生撒谎并不少见。尴尬和与医生面对面的时间太少加剧了男性和女性的信任障碍。一个名为[的数字健康平台 ZocDoc](https://www.zocdoc.com/about/news/new-zocdoc-study-reveals-women-are-more-likely-than-men-to-lie-to-doctors/) 显示，近一半(46%)的美国人避免告诉医生健康问题，因为他们感到尴尬或害怕被评判。大约三分之一的人说他们隐瞒了细节，因为他们找不到合适的机会或在预约中没有足够的时间(27%)，或者因为医生没有问任何问题或具体是否有什么困扰他们(32%)。这种情况的一个主要影响是在自杀领域。根据世界卫生组织(世卫组织)的数据，每年有多达 80 万人死于自杀，其中 60%的人面临严重的抑郁症。尽管抑郁症使患者处于从事自杀行为的更高风险中，但自杀型抑郁症患者和普通抑郁症患者之间的差异并不容易察觉。

[Deena Zaidi](https://hackernoon.com/@deenazaidi) 在她的博客中描述， [*机器学习使用面部表情来区分抑郁症和自杀行为*](https://hackernoon.com/machine-learning-uses-facial-expressions-to-distinguish-between-depression-and-suicidal-behavior-c9e1cd095026) *，*一个对风险因素进行了深入评估的自杀专家将如何预测病人未来的自杀想法和行为，其准确率与一个对病人毫不知情的人相同。这与基于抛硬币做出决定没有什么不同。虽然使用监督学习模型读取面部表情仍在开发中，但该领域已经显示出许多前景。

![](img/0b6a47ab1e7d0c4d379af7e31e508676.png)

Duchenne (top) vs non-Duchenne (bottom) smiles analyzed from SVM results help detect suicidal risks (Source: [Investigating Facial Behavior Indicators of Suicidal Ideation by Laksana et al](https://www.extremetech.com/wp-content/uploads/2017/06/fg2017-submitted.pdf).)

与南加州大学、卡内基梅隆大学和辛辛那提儿童医院医学中心的科学家合作撰写的一份报告调查了非语言面部行为以检测自杀风险，并声称发现了区分抑郁症患者和自杀患者的模式。使用 SVM，他们发现面部行为描述符，如涉及眼轮匝肌收缩的微笑百分比(杜兴微笑)在自杀和非自杀组之间有统计学意义。

# 认知能力

认知能力是一种基于大脑的技能，我们需要它来完成从最简单到最复杂的任何任务。它们更多地与我们如何学习、记忆、解决问题和集中注意力的机制有关，而不是与任何实际知识有关。人们热衷于提高认知。谁不想更好地记住名字和面孔，更快地理解困难的抽象概念，更好地“发现联系”呢？

![](img/52641c9033dfb4f1df23a7540525219d.png)

[Elevate](https://itunes.apple.com/us/app/elevate-brain-training/id875063456?mt=8) app on the Apple Store

目前，有一些应用正在帮助我们训练认知能力。一个这样的例子是 [Elevate](https://itunes.apple.com/us/app/elevate-brain-training/id875063456?mt=8) ，它由大脑游戏组成，用户可以在正确的难度水平上玩，以提高心算、阅读和批判性思维能力。[最佳认知功能](http://www.nickbostrom.com/posthuman.pdf)的价值如此明显，以至于没有必要详细阐述这一点。为了理解我们周围世界的更深层次的含义，我们不断地拓展我们五种感官的界限。例如，在图像识别领域，人工智能已经能够[“看”得比我们更好](https://www.entrepreneur.com/article/283990)，通过观察远在 RGB 光谱之外的变量，这反过来帮助我们超越我们自己的视觉限制。然而，当我们可以虚拟化时，为什么要把自己限制在 2D 屏幕上来可视化 3D 物体呢？

![](img/3621657370036d226cfcbca1255a9c5e.png)

[Nanome.AI](https://nanome.ai/) develops Augmented Reality for analyzing abstract molecular structures

增强现实让我们感觉自己仿佛被传送到了另一个世界。计算材料学和生物学是我思考这个问题时瞬间想到的领域。作为一名过去的计算材料科学家，我知道可视化复杂的分子结构对许多研究人员来说是一个挑战。[纳米。AI](https://nanome.ai/) 帮助在增强现实中可视化这些复杂的结构。更进一步，已经有很多初创公司在解剖学领域使用[增强现实技术来培训外科医生](https://www.nanalyze.com/2018/09/startups-augmented-reality-surgery/)。

![](img/76245d5754c31eca2b60b0bc8a1a82ae.png)

[Parallel Coordinate](https://datavizcatalogue.com/methods/parallel_coordinates.html) plot visualizing 7-dimensional space

为了让我们更好地体验我们周围的世界，数据可视化和降维算法的新习惯用法一直在不断产生。例如，我们有[平行坐标](https://datavizcatalogue.com/methods/parallel_coordinates.html)，它允许我们通过高维空间进行可视化和过滤，而 [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) 在将复杂空间分解为 2D 或 3D 空间后，很受欢迎。

# 情商

情商是一种意识到、控制和表达自己情绪的能力，也是一种明智而富有同情心地处理人际关系的能力。所有人都会经历情绪，但只有少数人能够在情绪出现时准确地识别出来。这可能是缺乏自我意识，或者仅仅是我们有限的情感词汇。很多时候，我们甚至不知道自己想要什么。我们努力以一种特定的方式与周围的人联系，或者消费一种特定的产品，只是为了能感受到一种无法描述的独特情感。我们感受到的不仅仅是快乐、悲伤、愤怒、焦虑或恐惧。我们的情绪是上述所有因素的复杂组合。理解我们自己以及我们周围人的情绪的能力对于情绪健康和维持积极的人际关系至关重要。

![](img/e11f87eb0a916e996ac91758b67313cf.png)

Distributed patterns of brain activity predict the experience of discrete emotions detected using fMRI scans on top and sensitivity range at the bottom (Source: [Decoding Spontaneous Emotional States in the Human Brain by Kragal et al.](https://www.researchgate.net/publication/308122235_Decoding_Spontaneous_Emotional_States_in_the_Human_Brain))

随着[神经映射](https://www.researchgate.net/publication/308122235_Decoding_Spontaneous_Emotional_States_in_the_Human_Brain)的创新，我们将更好地理解我们作为人类是谁，以及我们可以达到的无数种情绪状态。有监督的学习已经提供了一些普通情感的途径。通过对脑电波进行无监督学习，我们复杂的情绪模式可能会得到更好的理解。例如，一个简单的异常值检测算法也许可以揭示新的情绪模式或值得注意的重要情绪应激源。这样的研究有可能揭示提高我们情商的创新方法。

![](img/3313ab084dad8c77cdca053349c37c86.png)

Demonstration of how reading microexpression accurately can help negotiate during business transactions (Source: [TED Talk: How Body Language and Micro Expressions Predict Success — Patryk & Kasia Wezowski](https://www.youtube.com/watch?v=CWry8xRTwpo))

甚至有助于防止自杀的监督图像识别模型也可以让个人读出他们交谈对象的情绪。例如，[一个关于微表情的 TED 演讲](https://www.youtube.com/watch?v=CWry8xRTwpo)展示了如何在商业谈判中估算出理想的价格点，只需保持对特定面部表情的注意。同一个 TED 演讲中提到，善于解读微表情的销售人员比不会的销售人员多卖 20%。因此，这 20%的优势可能可以通过购买能够显示说话者情绪的眼镜来实现。

情商既包括理解我们自己的情绪，也包括对周围的人更加敏感。随着对我们情绪状态的深入研究，我们也许可以解开我们从未体验过的新情绪。在合适的人手里，人工智能可以作为我们的延伸，帮助我们与生活中我们重视的人建立有意义的联系。

# 经验和想象力

想象力是一种能力或行为，可以形成新的想法、图像或对感官无法感知的外部事物的概念。人工智能对我们的经验和想象力的影响将来自更好的认知能力和情商的聚合。简而言之，获得更丰富的认知能力和情商模式将让我们体验当今普通人难以想象的想法。

牛津大学哲学教授[尼克·博斯特罗姆](https://nickbostrom.com/)，在他的论文 [*为什么我长大后想成为后人类*](https://nickbostrom.com/posthuman.pdf) 中，描述了获得新的体验模式将如何增强我们的经验和想象力。让我们假设我们今天的体验模式是在空间 x 中表示的。10 年后，让我们假设体验模式是在空间 Y 中表示的。空间 Y 将比空间 x 大得多。Y 的这个未来空间可能会接触到除了我们传统的快乐、悲伤和疯狂之外的新类型的情感。这个新的 Y 空间甚至可以让我们更准确地理解反映我们想要表达的抽象思想。我们每个人都有可能以文森特·梵高最疯狂的想象也无法理解的方式来看待这个世界！

这个新的 Y 空间实际上可以打开一个新的可能性世界，它超出了我们目前的想象。未来的人们将以比我们今天更丰富的程度来思考、感受和体验这个世界。

# 沟通

当你缺乏情商时，很难理解你是如何给别人留下印象的。你觉得被误解了，因为你没有用人们能理解的方式传达你的信息。即使经过实践，情商高的人也知道，他们不会完美地传达每一个想法。人工智能有潜力通过增强自我表达来改变这一点。

![](img/9b47698fc850967ff6f6300c0e859658.png)

Google glasses translate German into English (Source: [Google buys Word Lens maker to boost Translate](https://www.cnet.com/news/google-buys-word-lens-maker-to-boost-translate/))

10 年前，我们的大部分交流仅限于电话和电子邮件。今天，我们可以参加视频会议，增强现实和社交媒体上的各种应用程序。随着我们认知能力和情商的提高，我们可以通过分辨率更高、抽象程度更低的习语来表达自己。[谷歌眼镜可以即时翻译外文文本](https://www.cnet.com/news/google-buys-word-lens-maker-to-boost-translate/)。在前面的章节中，我已经提到了使用谷歌眼镜阅读微表情的可能性。然而，为什么我们的交流仅限于我们“看得见”的东西呢？

![](img/26e40bbca99698749bb237222427f023.png)

Controlling drones using sending electrical impulses to headgear based sensor (Source: [Mind-controlled drone race: U. of Florida holds unique UAV competition](https://www.rt.com/usa/340654-drone-brain-race-mind/))

佛罗里达大学的学生只用意念就实现了对无人机的控制。我们甚至可以使用振动游戏机，利用我们的触觉使马里奥赛车游戏更加逼真。今天的增强现实只会限制我们的视觉和听觉。在未来，增强现实可能真的允许我们去闻、尝和触摸我们的虚拟环境。除了访问我们的 5 种感官，我们对某些情况的情绪反应可能会通过人工智能的力量进行微调和优化。这可能意味着分享我们在*灵异活动*中的主角的恐惧，感受*艾玛*的心碎，或者对*口袋妖怪*的冒险感到兴奋。

# 一些潜在的担忧

虽然我已经列举了能够使用人工智能阅读情感并帮助我们了解自己和周围人的所有积极因素，但我们不能忘记潜在的挑战:

*   **数据安全**:根据[世界隐私论坛](https://www.csoonline.com/article/2882052/data-breach/health-data-breaches-could-be-expensive-and-deadly.html)的数据，被盗医疗凭证的街头价值约为 50 美元，相比之下，仅被盗信用卡信息或社会安全号码的街头价值就为 1 美元。同样，心理健康信息是敏感的个性化数据，可能会被黑客利用。就像黑客试图窃取我们的信用卡和医疗保险信息一样，获取情感数据可能会在黑市上获得丰厚回报。
*   **政府数据法规**:对于任何高度敏感的个性化数据，不同国家有不同的法规需要遵守。在美国，与医疗保健相关的数据将需要遵守 [HIPAA](https://www.hhs.gov/hipaa/for-professionals/security/laws-regulations/index.html) 法规，而与金融应用相关的数据将需要 [PCI](https://www.pcisecuritystandards.org/) 。如果我们看得更远，欧盟有 [GDPR](https://www.insideprivacy.com/international/china/china-issues-new-personal-information-protection-standard/) ，中国有 [SAC](https://www.insideprivacy.com/international/china/china-issues-new-personal-information-protection-standard/) 。
*   伦理界限:和任何新技术一样，社会可能不喜欢自己的情感被获取。让我们面对事实。我们可能可以接受医生检查我们的情绪数据来改善我们的健康，但不能接受保险公司试图向我们收取更高的保费。按照同样的思路，我们已经有了[操纵大众心理](https://www.theguardian.com/uk-news/2018/mar/26/cambridge-analytica-trump-campaign-us-election-laws)操纵美国选举的嫌疑。然而，伦理规范在很大程度上依赖于特定社会中被认为是“正常”的东西。不能接受的东西，以后也不一定能接受。虽然数据科学在这一领域的某些方面的应用可能会让公众感到不舒服，但其他应用，如防止信贷欺诈和反洗钱的欺诈分析，以及亚马逊和网飞等推荐系统的营销分析，都是完全可以接受的。当引入一个新想法时，社会的接受程度将在很大程度上取决于所收集的数据被用来解决的问题的类型。关于开发人工智能产品的更多细节，请查看我的博客 [*UX 数据科学家和人工智能产品设计指南*](/ux-design-guide-for-data-scientists-and-ai-products-465d32d939b0) *。*