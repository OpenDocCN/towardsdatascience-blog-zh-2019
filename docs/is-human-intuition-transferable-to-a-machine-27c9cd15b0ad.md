# 人类的直觉能被注入机器的直觉吗？

> 原文：<https://towardsdatascience.com/is-human-intuition-transferable-to-a-machine-27c9cd15b0ad?source=collection_archive---------42----------------------->

![](img/58e4453a2edc47b660fbfed39c27af10.png)

Source of Image: [https://www.if4it.com/core-domain-knowledge-critical-foundation-successful-design-thinking/](https://www.if4it.com/core-domain-knowledge-critical-foundation-successful-design-thinking/)

我们经常看到不同的机器和深度学习模型努力模仿人类水平的表现。例如，我们还不能创建能够检测对象、理解它们之间的关系并总结图像中的事件的视觉模型，这些模型可以与任何人执行这些任务的能力相媲美。你有没有想过为什么？

即使给我们的模型提供了海量的数据，用不同的[超参数](/intuitive-hyperparameter-optimization-grid-search-random-search-and-bayesian-search-2102dbfaf5b)排列，我们仍然无法达到人类水平的性能。我们很清楚，这个模型没有捕捉到我们的大脑几乎在潜意识中能够捕捉到的细微差别。这是因为包含了人类随时间积累的先前多面知识。这被称为“人类知识”。即使在用我们的机器和深度学习模型尝试了不同的策略之后，我们仍然觉得好像缺少了什么。我们的模型中缺少人类知识。我们人类之所以是我们，是因为我们在生活中所面对的。我们之所以是我们，也许是因为我们的童年、我们的错误、我们的经历以及我们看待生活的方式塑造了我们的生活。

你认为我们人工制造的计算机或任何模型能够像我们一样执行任务吗？你认为机器有希望达到我们的理解水平吗？。答案既不是肯定的，也不是否定的。让我解释一下为什么。

这不是一个响亮的“不”，因为已经有太多的人试图将上下文的概念注入我们的机器和深度学习模型，以便让机器在任何场景中学习什么、为什么和谁[8]。这不是一个简单的肯定，因为这仍然是一项正在进行的工作。现在，你可能会有这样一个问题，你如何让一台机器在任何给定的情况下掌握上下文的概念。

答案并不简单。有许多方法可以将上下文的概念引入到模型中。其中最著名的一次是多模态知识图的使用。知识图是试图模仿真实世界关系并保持其语义完整的表示。例如，通过使用<subject predicate="" object="">三联体，像“男人骑自行车”这样的句子被表示为。</subject>

> **知识图表:**现实世界关系的基于图表的知识表示。
> 
> “知识图(I)主要描述组织在图中的真实世界实体及其相互关系，(ii)定义模式中实体的可能类别和关系，(iii)允许任意实体彼此之间潜在的相互关系，以及(iv)覆盖各种主题领域。[1]"

![](img/ddcf0ef7bf774a0491cbd2d49596cbf7.png)

Source of Image:[https://medium.com/@sderymail/challenges-of-knowledge-graph-part-1-d9ffe9e35214](https://medium.com/@sderymail/challenges-of-knowledge-graph-part-1-d9ffe9e35214)

已经有许多知识图是通过尽职调查构建的，并且由于其大规模的跨学科适用性而非常著名。一些最著名的是 DBPedia，麻省理工学院媒体实验室的 ConceptNet，谷歌的知识图谱等。一些知识图也是多语言的，这为知识图增加了多方面的知识。

![](img/4c97d23e1a4c7eabcb83e546e6f98dca.png)

Different Knowledge Graph Examples. Source of Image: [https://www.csee.umbc.edu/courses/graduate/691/fall18/07/](https://www.csee.umbc.edu/courses/graduate/691/fall18/07/)

> “知识图可以被设想为与特定领域或组织相关的各种事物的网络。它们不限于抽象的概念和关系，还可以包含文档和数据集等事物的实例。[2]"

知识图的一个局限是它们不是动态的[9]。至少就我所知，我还没有遇到过动态的知识图，当它获得新信息时会自动更新。所有知识图都是静态的，需要使用人工干预来更新。在大多数情况下，这是非常棘手的，因为人们需要小心我们在图表中添加了什么样的信息。特别是在当今世界，虚假数据和错误信息是一个巨大的问题，验证真实性和技术健全性是极其重要的。因此，人类干预成为必要，知识图保持静态，无法获得动态性质。

> 知识图表是静态的，需要人工干预来检查添加到其中的数据的真实性！

然而，知识激发的学习不一定只是知识图表。如此多的研究人员提出了如此多令人兴奋的新想法，以将上下文信息注入到我们的机器和深度学习模型中。

让我们考虑像图像修补这样的计算机视觉任务。(如果你不知道什么是图像修复，那么请看看我的文章“ [*不同的计算机视觉任务*](https://medium.com/@ananya.banerjee.rr/different-computer-vision-tasks-b3b49bbae891) ”来更好地理解它)。对于像图像修补或图像重建这样的任务，一些人提出了将图像作为一个整体来考虑并试图抓住图像中缺少的东西的想法。而其他人则专注于通过将重要权重与先验相关联，仅基于丢失区域附近的周围像素的上下文来预测图像的丢失区域的详细上下文[4]。很少有人尝试创建一个上下文编码器，该编码器具有类似编码器-解码器的框架，其中编码器捕获图像的上下文并将其投影到潜在特征表示中，解码器使用这些表示来产生缺失的图像上下文[5]。一些人采用了多模态方法，该方法考虑了图像注释的语言表示和语言属性以及图像的视觉表示和视觉属性，并提出了结合来自图像和文本的知识来理解上下文的联合模型[6]。

在其他计算机视觉任务(如视觉关系检测)中，很少有人尝试使用语言先验知识来添加信息。我遇到的另一个想法是将从概念网(知识图)获得的知识结合到神经图像字幕模型中，称为概念网增强神经图像字幕(CNET-NIC)[7]，以帮助增加模型的信息。

这些只是人们如何想出新方法来弥合人类直觉和机器理解之间的差距的几个例子。

我希望这篇文章对你有所帮助。

下次见！

感谢您的阅读！

**资源:**

[1]保罗海姆。知识图精化:方法和评估方法综述。语义网期刊，(预印本):1–20，2016。

[2] A .布鲁莫尔。从本体论的分类法到知识图，2014 年 7 月。[https://blog . semantic web . at/2014/07/15/from-taxonomies-over-ontologies to-knowledge-graphs](https://blog.semanticweb.at/2014/07/15/from-taxonomies-over-ontologiesto-knowledge-graphs)【2016 年 8 月】。

[3]卢，c .，克里希纳，r .，伯恩斯坦，m .，，L. (2016 年 10 月).基于语言先验的视觉关系检测。在*欧洲计算机视觉会议*(第 852–869 页)。斯普林格，查姆。

[4]Yeh，R. A .，Chen，c .，Lim，t .，Schwing，A. G .，Hasegawa-Johnson，m .，& Do，M. N. (2017 年)。基于深度生成模型的语义图像修复。在*IEEE 计算机视觉和模式识别会议论文集*(第 5485–5493 页)。https://wiki.dbpedia.org/

[5]Pathak，d .，Krahenbuhl，p .，Donahue，j .，Darrell，t .，& Efros，A. A. (2016 年)。上下文编码器:通过修补进行特征学习。在*IEEE 计算机视觉和模式识别会议论文集*(第 2536–2544 页)。

[6]n .加西亚，b .雷诺斯特和 y .中岛(2019 年)。通过绘画中的多模态检索理解艺术。 *arXiv 预印本 arXiv:1904.10615* 。

[7]周，杨，孙，杨，&霍纳瓦尔，V. (2019 年 1 月)。利用知识图改进图像字幕。在 *2019 年 IEEE 计算机视觉应用冬季会议(WACV)* (第 283–293 页)。IEEE。

[8]Sheth，a .，Perera，s .，Wijeratne，s .，& Thirunarayan，K. (2017 年 8 月)。知识将推动机器理解内容:从当前的例子中推断。在*网络智能国际会议论文集*(第 1-9 页)中。ACM。

[9]帕德希，s .，拉利斯塞纳，s .，&谢思，A. P. (2018 年)。创建实时动态知识图。

***更多资源:***

概念网:[http://conceptnet.io/](http://conceptnet.io/)

内尔:永无止境的语言学习:【http://rtw.ml.cmu.edu/rtw/index.php?】T2

知识图定义:[https://www . research gate . net/profile/Wolfram _ Woess/publication/323316736 _ forward _ a _ Definition _ of _ Knowledge _ Graphs/links/5 A8 d 6 e 8 f 0 f 7 e 9 b 27 C5 B4 B1 c 3/forward-a-Definition-of-Knowledge-Graphs . pdf](https://www.researchgate.net/profile/Wolfram_Woess/publication/323316736_Towards_a_Definition_of_Knowledge_Graphs/links/5a8d6e8f0f7e9b27c5b4b1c3/Towards-a-Definition-of-Knowledge-Graphs.pdf)

了解更多关于知识图的信息:[https://medium . com/@ sderymail/challenges-of-Knowledge-graph-part-1-d 9 FFE 9 e 35214](https://medium.com/@sderymail/challenges-of-knowledge-graph-part-1-d9ffe9e35214)