# 应用人工智能:从概念到 ML 组件

> 原文：<https://towardsdatascience.com/applied-ai-going-from-concept-to-ml-components-7ae9c5d823d3?source=collection_archive---------17----------------------->

![](img/782579722fdafe395aa6267d14c9fad0.png)

敞开你的心扉，接受将机器学习应用于现实世界的不同方式。作者 Abraham Kang 特别感谢 Kunal Patel 和 Jae Duk Seo 为本文提供意见和建议。

![](img/dedaa5c58bfee8a169f8ee30e9fcd46e.png)

Photo by [Franck V.](https://unsplash.com/@franckinjapan?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 行动纲要

# 候选问题

许多人对使用人工智能自动化组织内的冗余流程感兴趣。让我们从一个具体的问题开始，我注意到的是，当不好的事情发生时，律师通常会从客户那里收集事实。这些事实构成了个人可以起诉的[诉因](https://en.wikipedia.org/wiki/Cause_of_action)(疏忽、殴打、攻击、故意施加精神痛苦)的基础。一旦根据法律依据和事实确定了诉讼原因，就要写好诉状并提交给法院，以便开始法律诉讼。诉状是一份法律文件，列出了导致对另一方采取行动的法律依据的事实。手动创建此文档可能非常耗时，并且类似的事实会导致类似的行动原因。例如，如果有人打了另一个人，通常会有一个“电池”。如果有人不小心伤害了其他人，或者有人在商店里滑倒，可能会受到疏忽的起诉。基于这个问题，我们有一个客户想使用人工智能来学习如何根据描述所发生事情的事实段落写一份投诉。

# 理解问题

试图让人工智能/人工智能阅读事实，并找出一种方法让人工智能/人工智能撰写完整的投诉可能会超出模型的能力范围，可能需要数年才能解决。然而，如果你花时间去理解和思考潜在的问题，你可以找到现有的技术(稍加修改)来解决这个难题的不同部分。例如，当你看一份诉状时，它以对双方及其立场(原告对被告)以及代表他们的律师的描述开始。可能有集体诉讼部分、管辖权证明(法院是否有权管辖各方)、各方描述、地点证明(我们是否在适当的法院地点)、诉讼原因列表和事实描述。当你看这些部分的时候，你必须考虑构建各个部分的数据来自哪里。在某些情况下，你不会有答案，但如果你仔细观察，你会发现投诉的不同部分之间的模式和相关性。这将允许你考虑你对神经网络的输入和候选输出。

# 获取神经网络的输入

我们本身没有任何数据，但可能有一种方法可以从所有现有的投诉中解析出事实，并将它们用作我们神经网络的输入。提交给法院的每一份投诉都成为公共信息，因此会有大量的数据。这种解决方案将要求律师写下他们的事实，就像他们直接将它们插入到投诉中一样，但这对于能够让机器学习提供生成的投诉来说是一个小小的不便。提出完整的投诉可能很困难。所以我们把问题分解一下。

# 分解问题

从逻辑上讲，如何将文档的生成分解成更小的部分？你需要看一个例子:[https://www . heise . de/downloads/18/1/8/9/1/3/4/6/NP-v-Standard-Innovation-complaint . pdf](https://www.heise.de/downloads/18/1/8/9/1/3/4/6/NP-v-Standard-Innovation-Complaint.pdf)。为了让它变得有趣，我挑选了一家成人玩具制造商，这样可能会激发你的好奇心。基本上，我们希望最终从律师提供的事实中生成一份诉状(pdf 以上)。因此，如果你看看这份文件和其他投诉，你会发现类似的结构模式。

所以，你认为什么是分解事物的最好方法…在你有时间思考之前，不要向下滚动。

….真的好好想想…..

好吧，如果你说使用模板按部分分解，那么这可能是最好的方法。

当你分解投诉时，投诉中会列出行动的原因。每个诉因(违反联邦窃听法案、伊利诺伊州窃听法规、侵扰隔离、不当得利、欺诈和欺骗性商业行为法案等)。)有基于事实的支持规则和理由。所以现在有两个问题。如何从事实文本中得出行动的原因，如何在每个行动原因下生成支持文本？

# 寻找行动的原因

当我们看案件的事实时，我们需要找到我们可以起诉的所有诉讼原因(违反的法律)。从文本中寻找行动的原因没有直接的解决方法，所以我们必须从根本上思考。

你认为我们可以用什么样的现有技术来观察文本并推断文本的意思或描述。如果你说的是多标签文本分类或者多标签情感分析，那么你就领先了([https://paperswithcode.com/task/text-classification](https://paperswithcode.com/task/text-classification)，[https://papers with code . com/task/perspective-analysis)。](https://paperswithcode.com/task/sentiment-analysis).)分析文本以确定其相关的行动原因的过程类似于对文本进行分类或寻找相关文本的情感。还有一些相关的问题，例如，随着法律的出台，诉讼原因需要更新。可能有另一种方法来创建事实的嵌入，然后基于三元组(https://arxiv.org/pdf/1503.03832.pdf)或四元组损失(https://arxiv.org/pdf/1704.01719.pdf)将诉因与事实联系起来，以在嵌入空间中将共享相似词的诉因推到一起，并将不相关的诉因推得更远。然后，使用聚类技术，找出与决定性词语嵌入相近的诉因，这些词语嵌入用于支持与诉状的个别诉因部分中的词语相关的论点。

# 在单个诉讼原因的支持论据部分生成文本

既然您已经知道如何从文本中获得高层次的诉因，那么您如何为每个单独的诉因部分(违反联邦窃听法案、伊利诺伊州窃听法规、侵扰隔离、不当得利、欺诈和欺骗性商业行为法案等)生成支持性的论据文本。)?

这一个不那么直截了当。想一想什么样的神经网络架构可以生成文本(不要向下滚动，直到你有了一些想法)…

….打开你的心扉…使用原力…

文本生成算法([https://paperswithcode.com/task/data-to-text-generation](https://paperswithcode.com/task/data-to-text-generation)，[https://paperswithcode.com/area/nlp/text-generation](https://paperswithcode.com/area/nlp/text-generation))可能是一种选择，但即使是最好的算法也会经常产生乱码。更好的选择可能是使用类似神经网络的架构参与翻译([https://paperswithcode.com/task/machine-translation](https://paperswithcode.com/task/machine-translation)，[https://papers with code . com/task/unsupervised-machine-translation](https://paperswithcode.com/task/unsupervised-machine-translation)，[https://papers with code . com/paper/unsupervised-clinical-language-translation](https://paperswithcode.com/paper/unsupervised-clinical-language-translation))。此外，为每个行动原因建立一个单独的“翻译”神经网络可能是一个好主意，它可以帮助每个神经网络专注于识别用于为每个行动原因生成支持论据的关键事实。

# 打扫

通过语法检查器/修正器([https://papers with code . com/task/grammatic-error-correction](https://paperswithcode.com/task/grammatical-error-correction))运行每个诉讼原因的支持参数文本的候选文本可能是个好主意。这样，任何明显的混乱都会得到解决。

# 结论

我希望你学会了如何更广泛地应用机器学习解决方案。如果你遇到困难，请告诉我，因为我绝对有兴趣听到人们试图用机器学习来解决的问题。