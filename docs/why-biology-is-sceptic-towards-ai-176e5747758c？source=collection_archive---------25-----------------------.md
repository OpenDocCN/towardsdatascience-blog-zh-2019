# 为什么生物学对人工智能持怀疑态度

> 原文：<https://towardsdatascience.com/why-biology-is-sceptic-towards-ai-176e5747758c?source=collection_archive---------25----------------------->

## [生命科学的深度学习](https://towardsdatascience.com/tagged/dl-for-life-sciences)

## 为什么精准医疗不是

![](img/766648c9ce440d5146f8875a4de587a1.png)

[Image source](https://unsplash.com/photos/qI7USKbZY_A)

这是生命科学的 [**深度学习**](https://towardsdatascience.com/tagged/dl-for-life-sciences) 专栏的第六篇文章，我在这里演示了深度学习如何用于[古代 DNA](/deep-learning-on-ancient-dna-df042dc3c73d) 、[单细胞生物学](/deep-learning-for-single-cell-biology-935d45064438)、[生物医学组学集成](/deep-learning-for-data-integration-46d51601f781)、[临床诊断学](/deep-learning-for-clinical-diagnostics-ca7bc254e5ac)和[显微成像](/deep-learning-on-microscopy-imaging-865b521ec47c)。今天我将解释这一系列帖子的动机，并分享为什么**生物学和医学** **在概念上** **对 AI** 有不同的观点 **，这是我多年来与具有生物学和医学背景的人一起工作所学到的。**

# 超越全基因组关联研究

我来机器学习并不是因为它很酷，每个人都在谈论它，而是在寻找一种常用统计学的替代方法时发现了它。当来自所有基因的[遗传变异](https://en.wikipedia.org/wiki/Genetic_variation)被扫描与疾病的相关性时，我正在通过**基于 p 值的排名**进行统计分析，搜索与疾病相关的基因**全基因组关联研究**，就像下面二型糖尿病(T2D)的[曼哈顿图](https://en.wikipedia.org/wiki/Manhattan_plot)。

![](img/8592a424960a2a61e9956ed44f9894de.png)

Manhattan plot for Type 2 Diabetes (T2D) from [DIAGRAM, Nature Genetics 42 (7), 579–589 (2010)](https://www.nature.com/articles/ng.609)

如果你不熟悉曼哈顿图，一个点是一个遗传变异体，x 轴是该遗传变异体在人类基因组上的位置，y 轴是该遗传变异体与疾病的相关性/关联性的 **-log10(p 值)**。曼哈顿图上的峰值表明与该疾病有最强关联的基因片段。通过这种方式，**发现了许多与人类疾病相关的基因**，希望这些知识能够用于 [**临床诊断**](https://en.wikipedia.org/wiki/Medical_diagnosis) 对疾病的预测和预防。然而，人们很快意识到，在关联研究中发现的**基因不能预测常见疾病**，如[糖尿病](https://en.wikipedia.org/wiki/Diabetes)、[心血管](https://en.wikipedia.org/wiki/Cardiovascular_disease)、[精神分裂症](https://en.wikipedia.org/wiki/Schizophrenia)等。我在这里详细讨论了。我觉得失败是由于基因的 p 值排序，并开始编写我自己的算法，该算法优化了**预测能力** **而不是关联的 p 值。**后来我意识到我所编码的东西被称为 [**随机森林**](https://en.wikipedia.org/wiki/Random_forest) 在研究中被称为机器学习。这就是我对这个迷人领域感兴趣的原因。

# 生物怀疑论

然而，我对机器/深度学习/人工智能的热情并没有被我的大多数具有生物学背景的同事分享。当我主张向 Google Deep Mind 学习，试图找到我们的修辞问题的答案，如 [**【缺失的遗传性问题】、**](https://www.nature.com/news/2008/081105/pdf/456018a.pdf) 时，我感觉到消极的沉默突然拥抱了观众中的每个人，我很快被学术教授包围，他们解释说机器学习是**一种炒作**、**一个黑盒**和**“我们不需要预测** **，但希望** **理解生物机制”。**

![](img/cc8acc24b260926600880bec7e8dc8a3.png)

From [B.Maher, Nature, volume 456, 2008](https://www.nature.com/news/2008/081105/pdf/456018a.pdf)

他们中的许多人想知道为什么他们应该关心深度学习在[国际象棋](https://en.wikipedia.org/wiki/Deep_Blue_(chess_computer))、[围棋](https://en.wikipedia.org/wiki/AlphaGo)和[星际争霸](https://en.wikipedia.org/wiki/DeepMind#AlphaStar)中的进展，这是在谈论深度学习时通常强调的。事实上，对于一个典型的生物学家来说，很难看出星际争霸与活细胞有什么关系。人工智能努力构建的一般智能和生物学之间肯定有联系，但在文献中很少解释。当与生物学家交谈时，我通常会尝试给出更多令人赞赏的例子，说明深度学习解决了长期存在的蛋白质折叠问题。

![](img/cbb2df455d8d8accff530a48e5199c0e.png)

[Source](https://www.sciencemag.org/news/2018/12/google-s-deepmind-aces-protein-folding)

# 为什么生物学不喜欢预测

现在让我们简单讨论一下上面提到的关于深度学习的**刻板印象**。嗯，关于炒作我同意，但如果它提高了我们的生活质量，这是一个坏的炒作吗？具有讽刺意味的是，即使是为[显微镜图像分析](/deep-learning-on-microscopy-imaging-865b521ec47c)进行深度学习的生物学家也往往不喜欢和不信任深度学习，但他们不得不这样做，因为 a)这是一种宣传，b)与手动分割相比，它通过自动特征提取提供了更高的准确性。

![](img/e99f897ab7f2257debe4cf1c0fa91782.png)

This is how I feel talking Deep Learning to biologists

关于黑盒，任何机器学习都给你按照对你感兴趣的生物过程的重要性排列的特征，也就是说，你得到了这个过程的关键人物。如果你知道你的生物过程的驱动因素，你应该对这个过程有相当多的了解，不是吗？所以我一直不明白他们为什么叫它黑匣子。

![](img/ffa184032037c4926fe9af38f42a3b80.png)

现在我们来到生物学家和自然科学家心态的基石点。**预测**。由于这个词在这个社区名声不好，我尽量不在和生物学家交流时使用这个词。人们通常认为，在进行预测时，人工智能人员过于关注他们模型的准确性，而忽视了他们所研究过程的潜在机制。这有一些根据，但通常是一个令人难以置信的刻板印象。

> 不了解潜在的机制，怎么能做出好的预测呢？

如果一个模型可以预测，应该总有办法回滚它，看看模型里面是什么，以及**为什么它会预测**。或者，全基因组关联研究提供了一系列具有生物学意义的特征/基因，但它们不能预测疾病，这意味着它们不能用于临床，所以它们的选择方式有**问题**。如果**基因不能用于临床**，那么基于生物学意义选择基因又有什么意义呢？关心准确性，分析师间接关心潜在的生物机制。

在我看来，这种怀疑和保守主义减缓了生命科学中诸如[](https://en.wikipedia.org/wiki/Population_genetics)****[**古 DNA**](https://en.wikipedia.org/wiki/Ancient_DNA)**[**进化生物学**](https://en.wikipedia.org/wiki/Evolutionary_biology) **这些机器/深度学习/AI 基本缺席的领域的进展**。我的预测是，强化学习可能是进化生物学计算分析的未来。******

****Reinforcement Learning bipedal organism walking****

# ****为什么医学是开放的****

****相比之下，和有医学背景的同事聊起来，感觉就完全不一样了。他们对 AI 非常热情和好奇，即使他们不完全遵循我对神经网络的推理。这种开放的态度部分是因为机器/深度学习/AI 已经存在，广泛用于[](https://en.wikipedia.org/wiki/Precision_medicine)**(其中**早期疾病预测**是主要目标之一)，并在**癌症诊断的放射成像**方面取得进展。Deep Mind 最近发布了一种令人印象深刻的临床应用算法，可以在 48 小时内**早期预测急性肾损伤**。******

****![](img/c86241457c7f57841626ede7f4587603.png)****

****From [Tomasev et al., *Nature*, 572, 116–119 (2019)](https://www.nature.com/articles/s41586-019-1390-1)****

****医学界对人工智能产生浓厚兴趣的另一个原因是，他们迫切希望利用一切可用的手段来帮助他们的病人。如果人工智能可以早期预测疾病，这是最重要的，因为**我们没有时间**深入研究潜在的生物机制。事实上，几次去诊所和**看到痛苦中的人**让我大开眼界，我当然更清楚地理解了为什么我要做研究。****

# ****摘要****

****在本文中，我们了解到**机器/深度学习/AI 与传统的基于 p 值的统计方法**相比，更具有** **临床适用性** **框架**，因为它特别优化了对临床至关重要的** **预测能力**。尽管预测在生物界名声不佳，但当试图达到更高的准确性时，它不一定会牺牲生物学理解的深度。****

**像往常一样，让我在评论中知道你最喜欢的生命科学领域，你想在深度学习框架内解决这个问题。在媒体[尼古拉·奥斯科尔科夫](https://medium.com/u/8570b484f56c?source=post_page-----176e5747758c--------------------------------)关注我，在 Twitter @尼古拉·奥斯科尔科夫关注我，在 [Linkedin](http://linkedin.com/in/nikolay-oskolkov-abb321186) 关注我。我计划写下一篇关于**如何用深度学习检测尼安德特人渗入的区域**的帖子，敬请关注。**