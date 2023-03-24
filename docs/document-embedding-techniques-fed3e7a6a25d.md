# 文档嵌入技术

> 原文：<https://towardsdatascience.com/document-embedding-techniques-fed3e7a6a25d?source=collection_archive---------1----------------------->

## 关于这一主题的著名文献综述

单词嵌入——将单词映射到数字向量空间——近年来已被证明是自然语言处理(NLP)任务的一种非常重要的方法，使各种依赖向量表示作为输入的机器学习模型能够享受更丰富的文本输入表示。这些表示保留了单词的更多语义和句法信息，从而提高了几乎所有可以想象的 NLP 任务的性能。

这种新颖的想法本身及其巨大的影响促使研究人员考虑如何为更大的文本单元(从句子到书籍)提供更丰富的矢量表示这一好处。这一努力导致了一系列产生这些映射的新方法，对该问题提出了各种创新的解决方案，并取得了一些显著的突破。

这篇文章是我在介绍自己这个话题的时候写的(作为[大熊猫](https://www.bigpanda.io/)项目的一部分，在那里[我咨询了几年](http://www.shaypalachy.com/) ❤️🐼)，旨在展示从业者产生文档嵌入的不同方式。

> **注意:**我在这里使用单词 ***文档*** 来指代 ***任何单词序列*** ，从句子和段落到社交媒体帖子，一直到文章、书籍和更复杂结构的文本文档(例如表格)。

![](img/f75a712a1591da184f30046801962128.png)

Figure 1: A common example of embedding documents into a wall

在这篇文章中，我不仅会谈到直接扩展单词嵌入技术的方法(例如，以 *doc2vec* 扩展 *word2vec* 的方式)，还会谈到其他值得注意的技术，这些技术有时会产生其他输出，例如在ℝⁿ.中从文档到矢量的映射

只要有可能，我还会尽量提供链接和参考资料，既指向原始论文，也指向被评审方法的代码实现。

> **注:**这个题目和学习结构化文本表示的问题有些关联，但不等价(例如【刘& Lapata，2018】)。

## 目录

1.  [文档嵌入的应用](#0b0b)
2.  [突出方法和趋势](#a2f6)
    - [文档嵌入方法](#5616)-
    -[趋势和挑战](#1a94)
3.  [经典技巧](#92f5)
    - [词汇袋](#fea6)-
    -[潜在狄利克雷分配(LDA)](#e586)
4.  [无监督文档嵌入技术](#1409)
    - [n-gram 嵌入](#3e17)
    - [平均词嵌入](#ecd3)
    -[sent 2 vec](#e3d4)
    -[段落向量(*doc 2 vec*)](#2180)
    -[doc 2 vec](#0242)-[Skip-think 向量](#b22a)
    -[fast sent](#e6e8)

5.  [监督文档嵌入技术](#d7eb)
    - [从标记数据中学习文档嵌入](#1160)
    - [特定任务监督文档嵌入](#d742)
    --[GPT](#0c75)
    ---[深度语义相似度模型(DSSM)](#c222)
    - [联合学习句子表示](#885e)
    -[通用句子编码器](#2bd3)
    -[根森](#608f)
6.  [如何选择使用哪种技术](#da9f)
7.  [最后的话](#fb63)
8.  [参考文献](#3f52)

# 文档嵌入的应用

将文档映射到信息向量表示的能力具有广泛的应用。以下只是部分列表。

[ [Le & Mikolov，2014](https://cs.stanford.edu/~quocle/paragraph_vector.pdf) ]在几个文本分类和情感分析任务上展示了他们的*段落向量*方法的能力，而[ [戴等人，2015](https://arxiv.org/pdf/1507.07998.pdf) ]在文档相似性任务和[ [Lau & Baldwin，2016](https://arxiv.org/pdf/1607.05368.pdf) ]在一个论坛问题复制任务和[语义文本相似性(STS) SemEval 共享任务](http://ixa2.si.ehu.es/stswiki/index.php/Main_Page)上对其进行了基准测试。

【 [Kiros 等人，2015](https://arxiv.org/abs/1506.06726) 】已经展示了他们的*跳过思维*向量用于语义相关度、释义检测、图像句子排序、问题类型分类和四个情感和主观性数据集。[Broere，2017]使用它们来预测词性标签和依存关系。

【[陈等，2018](https://arxiv.org/pdf/1810.09302.pdf) 】展示了 *BioSentVec* ，他们在生物医学文本上训练的句子嵌入集，在句子对相似度任务上表现良好([官方 Python 实现](https://github.com/ncbi-nlp/BioSentVec))。

最后，[深度语义相似性模型被各种作者](https://www.microsoft.com/en-us/research/project/dssm/)用于信息检索和网页搜索排名、广告选择/相关性、上下文实体搜索和兴趣任务、问题回答、知识推理、图像字幕和机器翻译任务。

# 突出的方法和趋势

我最后写这一部分，已经花了很多时间思考如何组织这篇文章，如何将下面几节中涉及的各种技术归类到突出的方法中，以及在检查该领域中不同的作品如何相互关联以及它们彼此遵循的方式时会出现什么趋势。

但是，请注意，虽然文档嵌入的问题由来已久，但许多目前有影响力的解决方案还很年轻，而且这个领域最近(大约在 2014 年)在当代基于编码器-解码器的单词嵌入技术取得成功之后又出现了复兴，所以这仍然处于早期阶段。说了这么多，我希望这一部分能把下面的部分放到一个更广阔的背景中，并以一种有意义的方式框定它们。

## 文档嵌入方法

绘制该领域的一种可能方法是以下四种主要方法:

1.  **总结词语向量**
    这是*的经典做法。*单词袋*对一个热门单词向量就是这样做的，你可以对它应用的各种加权方案都是这种总结单词向量的方法的变体。然而，这种方法在与最先进的单词表示一起使用时也是有效的(通常通过求平均值而不是求和)，尤其是当单词嵌入考虑到这种用途而被优化时，并且可以抵抗这里介绍的任何更性感的方法。*
2.  ***主题建模**
    虽然这通常不是 LDA 和 PLSI 等主题建模技术的主要应用，但它们最近生成了一个文档嵌入空间*来建模和解释语料库中的单词分布，其中维度可以被视为隐藏在数据中的潜在语义结构，因此在我们的上下文中很有用。在这篇文章中，我并没有真正涉及这种方法(除了对 LDA 的简单介绍)，因为我认为 LDA 很好地代表了这种方法，而且这种方法也广为人知。**
3.  ***编码器-解码器模型**
    这是场景中最新的无监督添加，具有 *doc2vec* 和 *skip-thought 之类的功能。*虽然这种方法自 21 世纪初就已经存在，名为*神经概率语言模型*，但最近随着它在单词嵌入生成方面的成功应用，它获得了新的生命，当前的研究集中在如何将其应用扩展到文档嵌入。这种方法比其他方法从不断增加的大量未标记语料库中获益更多。*
4.  ***监督表示学习**
    这种方法的生命力归功于神经网络模型的伟大崛起(或复兴)，以及它们使用各种非线性多层运算符学习输入数据的丰富表示的能力，[这可以近似大范围的映射](https://en.wikipedia.org/wiki/Universal_approximation_theorem)。通过简单地将旧的单词袋输入到神经网络学习中，以解决一些与监督文本相关的问题，您可以获得一个模型，其中隐藏层包含输入文本的丰富表示，这正是我们所追求的。*

*有几个无监督的方法不适合上述任何一组(特别是，*快速思考*和*单词移动器的距离*浮现在脑海中)，但我认为大多数技术都属于这四大类别之一。*

> ***注:**虽然人们很容易指出经典的单词袋技术缺乏订单信息，但这实际上是一种规律，而不是例外。这里回顾的大多数新方法获得的主要信息是将分布假设扩展到更大的文本单元。基于神经网络的序列模型是例外。*

## *趋势和挑战*

*当从整体上考察文档嵌入技术的研究和应用时，出现了几个大的趋势，以及人们可能会发现的几个挑战。*

1.  ***编码器-解码器优化:**研究的一个显著部分集中在优化确切的架构(例如 NN/CNN/RNN)和一些组件/超参数(例如 n-gram、投影函数、称重等)。)的无监督编码器-解码器方法来学习文档嵌入。虽然这种微调的部分目标是提高各种任务的成功指标，但在更大的语料库或更短的时间内训练模型的能力也是一个目标。*
2.  ***学习目标设计:**无监督(或自我监督)表示学习的关键在于设计一个学习目标，它利用数据中可自由获得的标签，以某种方式生成对下游任务有用的表示。对我来说，这是最令人兴奋的趋势，我认为这是对 NLP 任务影响最大的趋势，可能等同于单词嵌入技术。目前，我认为只有*思维敏捷的*和*字移动器的距离*可以作为编码器-解码器方法的替代方案。这一趋势的另一个吸引人的方面是，这里的创新可能也适用于单词嵌入的问题。*
3.  ***基准测试:**一般来说，作为机器学习研究领域趋势的一部分，文档嵌入，也许是因为它是一个年轻的子领域，很好地证明了对大范围和大量任务的技术基准测试的研究的日益关注(参见 [the GLUE leaderboard](https://gluebenchmark.com/leaderboard) )。然而，几乎每一篇关于这一主题的论文都宣称其结果与目前的 SOTA 技术相当或更好，这还没有导致一个明确的领导者出现在这群人的前面。*
4.  ***开源:**同样，作为更广泛趋势的一部分，易于使用的代码实现技术(通常也是实验)的火热发布实现了可重复性，并推动了学术界以外更广泛的数据科学社区的参与和对真实世界问题的使用。*
5.  ***跨任务适用性:**虽然不是所有的无监督技术都以相同水平的全面性作为基准，但这可能更多是监督嵌入学习的情况。无论如何，依赖于文本数据中不同类型的信息的各种各样极其不同的 NLP 任务，使得这个问题成为一个突出的问题。从几个任务中联合学习嵌入是一种有趣的方式，其中有监督的方法可能解决这个挑战。*
6.  ***标记语料库:**非常大的标记语料库的有限可用性也是未来监督方法的一个问题。这可能代表了未来几年无监督方法在监督表示学习方面的真正优势。*

> ***注意:**如果你觉得这部分有点断章取义，我建议你在看完这篇文章中的大部分技巧后再来看一看。*

# *经典技术*

*本节简要介绍了两种已建立的文档嵌入技术:*单词袋*和*潜在狄利克雷分配*。[随意跳过](#1409)。*

## *词汇袋*

*该方法在[Harris，1954]中提出，将文本表示为其单词的包( [multiset](https://en.wikipedia.org/wiki/Multiset) )(丢失语法和排序信息)。这是通过决定将形成映射所支持的词汇表的一组 *n* 个单词，并给词汇表中的每个单词分配一个唯一的索引来完成的。然后，每个文档由一个长度为 *n* 的向量表示，其中第 *i* 个条目包含单词 *i* 在文档中出现的次数。*

*![](img/0a9c048a3f81bc69d86c204672e15d1c.png)*

*Figure 2: A bag-of-words representation of an example sentence*

*比如那句“狗咬狗的世界，宝贝！”(在清除标点符号之后)可以由 550 长度的向量 *v* 表示(假设选择了 550 个单词的词汇)，除了以下条目之外，该向量在任何地方都是零:*

*   *V₇₆=1，作为词汇表的第 76 个单词是*世界*。*
*   *V₂₀₀=2，作为词汇表的第 200 个单词是*狗*。*
*   *V₃₂₂=1，作为词汇表中的第 332 个单词是吃。*
*   *单词 *baby* 没有被选择包含在词汇表中，因此它在没有向量条目时导致值 1。*

*尽管它非常简单，除了单词出现频率之外的所有信息都丢失了，并且表示大小有快速增长以支持丰富词汇的趋势，但这种技术几十年来几乎只在大量的 NLP 任务中使用，并取得了巨大的成功。尽管近年来文本的矢量表示取得了重大进展，但这种方法的常见细微变化(见下文)今天仍在使用，而且并不总是作为第一个被迅速超越的基线。*

***n 字袋** 为了重新获得由字袋方法丢失的一些词序信息，短词序列(长度为二、三等)的频率。)可用于(附加地或替代地)构建单词向量。自然，单词袋是这种方法的一个特例，因为 *n=1* 。*

*对于那句“狗咬狗的世界，宝贝！”，单词对是“狗食”、“吃狗”、“狗世界”和“世界宝宝”(有时也有“<start>狗”和“宝宝<end>”)，词汇由输入语料库中所有连续的单词对组成(或用其增强)。</end></start>*

*![](img/7148a1e555236408b86ba071a3a2dbc1.png)*

*Figure 3: 2-grams representation of the sentence “The movie is amazing”*

*这种方法的一个主要缺点是词汇大小对唯一单词数量的非线性依赖，这对大型语料库来说可能非常大。过滤技术通常用于减少词汇量。*

***tf-idf 加权** 词袋上下文中最后一个值得一提的相关技术是 [*词频——逆文档频率*](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) ，通常表示为 *tf-idf* 。该方法用每个单词的*逆文档频率* (IDF)对上述单词(或 n-gram)频率向量重新加权。一个单词的 IDF 就是语料库中文档数量的对数除以该单词出现的文档数量。*

*![](img/1450074a857266439b5920960ac41755.png)*

*简而言之，TF 术语随着该词出现得越来越频繁而增长，而 IDF 术语随着该词出现得越来越少而增长。这是为了调整频率分数，因为一般来说，有些词出现的频率更高(或更低)。参见[ [Salton & Buckley，1988](http://pmcnamee.net/744/papers/SaltonBuckley.pdf) ]对术语加权方法的全面概述。*

## *潜在狄利克雷分配*

*LDA 是一个生成统计模型，它允许用未观察到的组来解释观察结果集，从而解释为什么数据的某些部分是相似的。例如，如果观察是收集到文档中的单词，那么它假设每个文档是少量主题的混合物，并且每个单词的出现都归因于文档的一个主题。*

*为了将这一点与词袋联系起来，前一种方法可以被认为是一种过于简单的概率模型，将文档作为词的分布。单词袋向量则代表我们对每个文档中的非标准化单词分布的最佳近似；但是这里的文档是基本的概率单位，每一个都是其唯一分布的单个样本。*

*因此，问题的关键是通过增加一个潜在的(隐藏的)K 主题的中间层，从这个简单的文档概率模型转变为一个更复杂的模型。*

*![](img/410942f4488713e2425a31b88f8abbdd.png)*

*Figure 4: The probabilistic model shift from bag-of-words to LDA*

*主题现在的特征是单词上的分布，而文档是主题上的分布。文档的概率模型对应于文档的生成模型；为了生成一组长度为 *{Nᵢ}* 的 *M* 个文档，假设预定数量的 *K* 个主题，其中 *Dir()* 表示一个[狄利克雷分布](https://en.wikipedia.org/wiki/Dirichlet_distribution):*

1.  *对于每个题目 *v* ，取样一个单词分布φᵥ~ *Dir(β)* 。*
2.  *对于每个文档 *i* ，抽取一个主题分布(或混合)θᵢ~ *Dir(* α *)* 。*
3.  *生成长度为 *Nᵢ* 的文件 *i* ，每个字 *j* :
    3.1。样题 zᵢⱼ ~ *Multinomial(θᵢ)* 作词 *j* 。3.2。样字*j*~*multinomial(zᵢⱼ)*。*

*给定这个模型和一个文档集，问题就变成了推理，在推理过程中可以找到上述各种分布的近似值。其中有θᵢ，每个文档的主题分布 *i* ，维度矢量 *K* 。*

*因此，在推断模型的过程中，推断出维度为 *K* 的向量空间，该向量空间以某种方式捕获了我们语料库中的主题或主题以及它们在其中的文档之间共享的方式。当然，这可以作为这些文档的嵌入空间，并且——取决于对 *K* 的选择——它可以比基于词汇表的文档小得多。*

*事实上，虽然 LDA 的主要使用情况是无监督的主题/社区发现，但是其他情况包括使用所得的潜在主题空间作为文档语料库的嵌入空间。还要注意，其他主题建模技术——例如[非负矩阵分解(NMF)](https://en.wikipedia.org/wiki/Non-negative_matrix_factorization) 和[概率潜在语义索引(PLSI)](https://en.wikipedia.org/wiki/Probabilistic_latent_semantic_analysis)——可以以类似的方式用于学习文档嵌入空间。*

> ***注意:**从业者对于概率主题模型的一个主要问题是它们的稳定性。因为训练主题模型需要概率分布的采样，所以可以预期相同语料库的模型随着随机数生成器的种子变化而不同。这个问题由于主题模型对相对小的语料库变化的敏感性而变得复杂。*

# *无监督文档嵌入技术*

*本节中介绍的许多方法都是受著名的单词嵌入技术的启发，其中最主要的是 *word2vec* ，它们有时甚至是这些方法的直接推广。这些单词嵌入技术有时也被称为*神经概率语言模型*；这些不是相同的术语，因为概率语言模型是*单词序列的概率分布*，但是由于这种方法是作为一种学习语言模型的方法在[ [Bengio，2003](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) 中介绍的，所以它们是紧密相关的。*

*因此，对单词嵌入技术的基本理解对于理解本节至关重要。如果你不熟悉这个主题，Chris McCormick 写得很好的关于 *word2vec* 的两部分教程[是一个很好的起点(](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)[第二部分](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)，Joshua Bengio 教授写的关于神经网络语言模型的 Scholarpedia 文章[也是如此(参见](http://www.scholarpedia.org/article/Neural_net_language_models) [Hunter Heidenreich 的帖子，以获得关于单词嵌入的更一般和简明的概述](/introduction-to-word-embeddings-4cf857b12edc)，以及 [Alex Minnar 的然而，为了深刻理解细节，我强烈建议您阅读[](http://alexminnaar.com/2015/05/18/word2vec-tutorial-continuousbow.html) [Bengio，2003 年](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf) ]、[ [Mikolov 等人，2013 年 a](https://arxiv.org/pdf/1301.3781.pdf) ]和[ [Pennington 等人，2014 年](https://www.aclweb.org/anthology/D14-1162) ]关于该主题的开创性论文，这些论文在许多方面塑造了这一子领域。*

*即使假设你熟悉这个模型的一个重要假设，我仍然希望注意到这个模型所做的一个重要假设，这个假设可能被这里回顾的每一个模型所继承:*分布假设*。下面是来自[维基百科](https://en.wikipedia.org/wiki/Distributional_semantics#Distributional_hypothesis)的简要描述:*

> *语言学中的**分布假设**来源于语言使用的[语义理论](https://en.wikipedia.org/w/index.php?title=Semantic_theory&action=edit&redlink=1)，即在相同语境中使用和出现的词语倾向于表达相似的意思。“一个词由它所结交的朋友来表征”的基本思想是由[弗斯](https://en.wikipedia.org/wiki/J._R._Firth)推广开来的。分布假设是统计语义学的基础。*

*事实上，很容易看出 *word2vec* 以及其他用于学习单词表示的自我监督方法严重依赖于这一假设；毕竟，该模型的关键在于，在学习从单词本身预测单词的上下文(或反之亦然)时学习的单词表示表示捕获深层语义和句法概念和现象的向量空间。意义，从一个词的上下文中学习可以教我们关于它的意义和它的句法作用。*

*在本节中，涵盖自我监督的文档表示学习，您将看到所有这些方法都保持了对单词的假设，并以某种方式将其扩展到更大的文本单元。*

## *n 元文法嵌入*

*[ [Mikolov 等人，2013b](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf) ]扩展了 *word2vec* 的 skip-gram 模型，通过使用数据驱动的方法识别大量短短语(作者专注于两个和三个单词的短语)，然后在 *word2vec* 模型的训练期间将这些短语视为单独的标记，来处理短短语。自然，这不太适合学习更长的短语——因为当短语长度增加时，词汇量会爆炸——而且*肯定不会推广到看不见的短语*以及跟随它的方法。*

*Moshe Hazoom 写了一篇关于这种方法的精彩实用评论，他的雇主将其用于一个专注于金融领域的搜索引擎。*

## *平均单词嵌入*

*从有意义的单词嵌入中构造文档嵌入有一种非常直观的方法:给定一个文档，对该文档的单词对应的所有向量执行一些向量运算，以在相同的嵌入空间中将它们概括为单个向量；两种常见的汇总运算符是 average 和 sum。*

*在此基础上，您也许已经可以想象扩展 *word2vec* 及其相关的编码器-解码器架构来学习*如何将单词向量组合到文档嵌入中会很有趣；这个方法之后的方法属于这一类。**

*第二种可能性是使用固定的(不可学习的)算子来进行向量汇总，例如平均，并使用旨在产生丰富的文档嵌入的学习目标来学习在前一层中的单词嵌入；一个常见的例子是使用一个句子来预测上下文句子。因此，这里的主要优点是单词嵌入被优化用于平均到文档表示中。*

*![](img/38be19b6bd707ded016c2162c47bd858.png)*

*Figure 5: Siamese CBOW network architecture from [[Kenter et al, 2016](https://arxiv.org/pdf/1606.04640.pdf)]*

*[ [肯特等人，2016](https://arxiv.org/pdf/1606.04640.pdf) ]正是这样做的，使用一个简单的神经网络对单词向量进行平均，通过预测给定的句子表示来学习单词嵌入。他们将结果与平均的 *word2vec* 向量和 *skip-thoughts* 向量进行比较(参见下面相应的小节)。【 [Hill 等人，2016](https://www.aclweb.org/anthology/N16-1162) 】比较了过多的方法，包括训练 CBOW 和 skip-gram 单词嵌入，同时优化句子表示(这里使用单词向量的元素相加)。[ [Sinoara 等人，2019](http://wwwusers.di.uniroma1.it/~navigli/pubs/KBS_Sinoaraetal_2019.pdf) ]还提出了一种将单词嵌入向量和其他知识源(如词义向量)直接组合到它们的质心中来表示文档。*

*最后，[ [Arora 等人，2016](https://pdfs.semanticscholar.org/3fc9/7768dc0b36449ec377d6a4cad8827908d5b4.pdf) ]进一步表明，当增加两个小的变化时，这种方法是一种简单但难以击败的基线:(1)使用平滑的逆频率加权方案，以及(2)从词向量中移除常见话语成分；这个成分是使用 PCA 发现的，并且它被用作最频繁话语的校正术语，推测与句法有关。作者提供了一个 [Python 实现](https://github.com/peter3125/sentence2vec)。*

> ***注意:**当观察基于注意力的机器翻译模型时，也许可以找到正确平均的单词“嵌入”的力量的另一个证明。单向解码器 RNN 获得先前翻译的单词作为输入，加上不仅要翻译的当前单词的“嵌入”(即来自编码器 RNN 的双向激活)，而且要翻译其周围单词的“嵌入”；这些以加权的方式被平均成一个上下文向量。它教导了这种加权平均能够保持来自编码器网络激活的复杂的组成和顺序相关的信息(回想一下，这些不像在我们的情况中那样是孤立的嵌入；每一个都注入了前面/后面单词的上下文)。*

## *Sent2Vec*

*在[ [Pagliardini 等人，2017](https://aclweb.org/anthology/N18-1049) ]和[ [Gupta 等人，2019](https://www.aclweb.org/anthology/N19-1098) ](包括[一个官方的基于 C++的 Python 实现](https://github.com/epfml/sent2vec))中介绍了这种技术，这种技术在很大程度上是上述两种方法的结合: *word2vec* 的经典 CBOW 模型都进行了扩展，以包括单词 n-grams *和*，它们适用于优化单词(和 n-grams)嵌入，以便对它们进行平均*

*![](img/9d0e07f3dcc26054a753642cbeba7012.png)*

*Figure 6: sent2vec can be thought of as an unsupervised version of fastText*

*此外，去除了输入二次采样的过程，而是将整个句子视为上下文。这意味着 **(a)** 放弃使用频繁的单词二次采样——以便不阻止 n-grams 特征的生成——以及**(b)***word 2 vec*使用的动态上下文窗口被取消:整个句子被视为上下文窗口，而不是在 1 和当前句子的长度之间均匀地采样每个二次采样单词的上下文窗口大小。*

*另一种思考 *sent2vec* 的方式是作为 *fastText* 的无监督版本(见图 6)，其中整个句子是上下文，可能的类别标签都是词汇单词。巧合的是，[ [Agibetov 等人，2018](https://link.springer.com/article/10.1186/s12859-018-2496-4) ]针对生物医学句子分类的任务，比较了使用 *sent2vec* 向量作为特征的多层感知器与 *fastText* 的性能。*

## *段落向量(doc2vec)*

*有时被称为 *doc2vec* ，这种方法在[ [Le & Mikolov，2014](https://cs.stanford.edu/~quocle/paragraph_vector.pdf) 中提出，可能是第一次尝试将 *word2vec* 推广到处理单词序列。作者介绍了 p *图向量*模型的两种变体:*分布式内存*和*分布式单词包。**

***段落向量:分布式内存(PV-DM)** PV-DM 模型通过添加内存向量来增强标准编码器-解码器模型，旨在从输入中捕捉段落的主题或上下文。这里的训练任务和*连续包话*挺像的；一个单词可以从它的上下文中预测出来。在这种情况下，上下文单词是前面的单词，而不是周围的单词，就像段落一样。*

*![](img/c18e24f52bde361622b62d974154ac8c.png)*

*Figure 7: The Distributed Memory model of Paragraph Vectors (PV-DM)*

*为了实现这一点，每个段落都被映射到一个唯一的向量，由矩阵中的一列表示(用 *D* 表示)，词汇表中的每个单词也是如此。上下文是固定长度的，从段落上的滑动窗口中采样。段落向量在从同一段落生成的所有上下文之间共享，但不在段落之间共享。自然地，单词嵌入是全局的，并且可以使用预先训练的单词嵌入(参见下面的*实现和增强*)。*

*正如在 *word2vec* 中，向量必须以某种方式汇总成一个单一的向量；但是与 word2vec 不同，作者在他们的实验中使用了串联。请注意，这保留了订单信息。类似于 *word2vec* ，一个简单的 softmax 分类器(在这种情况下，实际上是分层的 softmax)被用于这个概括的向量表示，以预测任务输出。训练以标准方式进行，使用随机梯度下降并通过反向传播获得梯度。*

*注意，只有训练语料库中的段落具有来自 *D* 的与其相关联的列向量。在预测时，需要执行推理步骤来计算新段落的段落向量:文档向量是随机初始化的。然后，重复地从新文档中选择一个随机单词，并使用梯度下降来调整输入到隐藏层的权重，使得对于所选择的单词，softmax 概率最大化，而隐藏到 softmax 输出的权重是固定的。这导致新文档的表示为训练语料库文档向量(即 *D* 的列)的混合，自然地驻留在文档嵌入空间中。*

***段落向量:分布式单词包(PV-DBOW)**
*段落向量*的第二种变体，顾名思义，或许是 *word2vec* 的 *skip-gram* 架构的并行；分类任务是仅使用段落向量来预测单个上下文单词。在随机梯度下降的每次迭代中，采样一个文本窗口，然后从该窗口中采样单个随机单词，形成下面的分类任务。*

*![](img/75cac6a1563ee4f2e1ee7e94a7e5d901.png)*

*Figure 8: The Distributed Bag of Words model of Paragraph Vectors (PV-DBOW)*

*训练在其他方面是相似的，除了单词向量不与段落向量一起被联合学习的事实。这使得 PV-DBOW 变体的内存和运行时性能都要好得多。*

> ***注:**在 [its Gensim 实现](https://radimrehurek.com/gensim/models/doc2vec.html)中，PV-DBOW 默认使用随机初始化的字嵌入；如果 dbow_words 设置为 1，则在运行 dbow 之前，会运行一步 skip-gram 来更新字嵌入。[ [Lau & Baldwin，2016](https://arxiv.org/pdf/1607.05368.pdf) ]认为，尽管 dbow 在理论上可以处理随机单词嵌入，但这严重降低了他们所研究的任务的性能。*
> 
> *一个直观的解释可以追溯到模型的目标函数，即最大化文档嵌入与其组成单词嵌入之间的点积:如果单词嵌入是随机分布的，则更难以将文档嵌入优化为接近其更关键的内容单词。*

***应用、实现和增强** 【Le&miko lov，2014】演示了*段落向量*在几个文本分类和情感分析任务中的使用，而【Dai et al，2015】在文档相似性任务的背景下对其进行了检查，并且【Lau & Baldwin，2016】将其与论坛问题复制任务和 [*语义文本相似性(STS) SemEval*](http://ixa2.si.ehu.es/stswiki/index.php/Main_Page) 共享任务进行了基准测试。后面的两篇论文对该方法进行了扩展评估(前者侧重于 PV-DBOW 变体)，将其与其他几种方法进行了比较，并给出了实用建议(后面的[包括代码](https://github.com/jhlau/doc2vec))。*

*这个方法有[一个 Python 实现，作为 gensim 包](https://radimrehurek.com/gensim/models/doc2vec.html)的一部分，还有[一个 PyTorch 实现](https://github.com/inejc/paragraph-vectors)。同样，【 [Lau & Baldwin，2016](https://arxiv.org/pdf/1607.05368.pdf) 】也[提供了他们考试用的代码](https://github.com/jhlau/doc2vec)。*

*最后，提出了对该方法的各种改进。例如，【[李等，2016](https://arxiv.org/abs/1512.08183) 】将该方法扩展为也包含 n-gram 特征，而【thong tan&phiconstrakul，2019】建议在计算嵌入投影时使用余弦相似度而不是点积(还提供了[一个 Java 实现](https://github.com/tanthongtan/dv-cosine))。*

## *Doc2VecC*

*【[陈，2017](https://arxiv.org/pdf/1707.02377.pdf) 】提出了一种有趣的方法，该方法受段落向量方法(PV-DM)的分布式记忆模型和平均单词嵌入的方法的启发来表示文档。*

*![](img/a298587d467594a05dd16da23ae50b04.png)*

*Figure 9: The architecture of the Doc2VecC model*

*与*段落向量*，*doc 2 vecc*(*文档
向量穿越讹误*的首字母缩写)由输入层、投影层和输出层组成，预测目标词(上例中为“仪式”)。相邻单词的嵌入(例如“开始”、“for”、“The”)提供局部上下文，而整个文档的向量表示(以灰色显示)用作全局上下文。与直接学习每个文档的唯一向量的*段落向量*相反，*doc 2 vec*将每个文档表示为从文档中随机采样的单词的嵌入的平均值(例如，在位置 *p* 的“性能”，在位置 *q* 的“称赞”，以及在位置 *r* 的“巴西”)。*

*此外，作者选择通过随机删除单词的重要部分来破坏原始文档，通过平均化剩余单词的嵌入来表示文档。这种损坏机制允许在训练期间加速，因为它显著地减少了在反向传播中要更新的参数的数量。作者还展示了它如何引入一种特殊形式的正则化，他们认为这导致了观察到的性能改善，基准测试是情感分析任务、文档分类任务和语义相关度任务，而不是过多的最先进的文档嵌入技术。*

*在[一个公共的 Github 库](https://github.com/mchen24/iclr2017)中可以找到一个基于 C 的方法和代码的开源实现，用于重现论文中的实验。*

*[ [希尔等人，2016 年](https://www.aclweb.org/anthology/N16-1162) ]还将破坏或添加噪声到文档嵌入学习过程以产生更鲁棒的嵌入空间的一般思想应用于*跳过思维*模型(见以下小节)，以创建他们的顺序去噪自动编码器(SDAE)模型。*

## *跳跃思维向量*

*在[ [Kiros et al，2015](https://arxiv.org/abs/1506.06726) 中提出，这是另一个早期推广 *word2vec* 的尝试，并与[一起发布了一个官方的纯 Python 实现](https://github.com/ryankiros/skip-thoughts)(最近还宣称实现了 [PyTorch](https://github.com/sanyam5/skip-thoughts) 和 [TensorFlow](https://github.com/tensorflow/models/tree/master/research/skip_thoughts) )。*

*然而，这以另一种直观的方式扩展了*word 2 vec*——特别是 *skip-gram* 架构:基本单元现在是句子，一个编码的句子用于预测它周围的句子。使用在上述任务上训练的编码器-解码器模型来学习矢量表示；作者使用具有 GRU 激活的 RNN 编码器和具有条件 GRU 的 RNN 解码器。两个不同的解码器被训练用于前一个和下一个句子。*

*![](img/49645ec1dd20eba0d96f20fad5eec1b9.png)*

*Figure 10: The skip-thoughts model. Given a tuple of contiguous sentences, the sentence sᵢ is encoded and tries to reconstruct the previous sentence sᵢ₋₁ and the next sentence sᵢ₊₁*

***skip-thought 中的词汇扩展**
*skip-thought*编码器使用单词嵌入层，将输入句子中的每个单词转换为其对应的单词嵌入，有效地将输入句子转换为一系列单词嵌入。这个嵌入层也由两个解码器共享。*

*![](img/438bb854c7cd5d05ad79a050d613c621.png)*

*Figure 11: In the skip-thoughts model, sentence sᵢ is encoded by the encoder; the two decoders condition on the hidden representation of the encoder’s output hᵢ to predict sᵢ₋₁ and sᵢ₊₁ [from [Ammar Zaher’s post](https://sourcediving.com/building-recipe-skill-representations-using-skip-thought-vectors-8a6e4c38ae6c)]*

*然而，作者只使用了 20，000 个单词的小词汇量，因此在各种任务中使用时可能会遇到许多看不见的单词。为了克服这一点，通过为参数化该映射的矩阵 *W* 求解非正则化的 *L2* 线性回归损失，学习从在大得多的词汇表(例如 *word2vec* )上训练的单词嵌入空间到*跳过思想*模型的单词嵌入空间的映射。*

***应用、增强和进一步阅读** 作者演示了将*跳过思维*向量用于语义相关度、释义检测、图像句子排序、问题类型分类和四个情感和主观性数据集。【 [Broere，2017](http://arno.uvt.nl/show.cgi?fid=146003) 】通过对 *skip-thought* 句子表征进行训练逻辑回归来预测词性标签和依存关系，从而进一步研究其句法属性。*

*[ [Tang et al，2017a](https://arxiv.org/abs/1706.03146) ]提出了一种针对 *skip-thought* 的邻域方法，丢弃排序信息并使用单个解码器预测上一句和下一句。【【唐等，2017b 】扩展该检查，提出对模型的三个增强，他们声称使用更快和更轻的模型来提供可比的性能: **(1)** 仅学习解码下一句话， **(2)** 在编码器和解码器之间添加 *avg+max* 连接层(作为允许非线性非参数特征工程的方式)，以及 **(3)** 执行良好的字嵌入初始化。最后，【 [Gan et al，2016](https://arxiv.org/pdf/1611.07897.pdf) 】在广泛的应用中，使用基于 CNN 的分层编码器而不是仅基于的编码器来应用相同的方法。*

*在[ [Lee & Park，2018](https://openreview.net/pdf?id=H1a37GWCZ) ]中提出的另一种变体，通过基于文档结构为每个目标句子选择整个文档中有影响的句子来学习句子嵌入，从而使用元数据或文本样式来识别句子的依存结构。此外，【 [Hill 等人，2016](https://www.aclweb.org/anthology/N16-1162) 】提出了*顺序去噪自动编码器(SDAE)* 模型，这是 *skip-thought* 的变体，其中输入数据根据一些噪声函数被破坏，模型被训练以从被破坏的数据中恢复原始数据。*

*对于关于 *skip-thought* 模型的进一步非学术阅读， [Sanyam Agarwa 在他的博客](http://sanyam5.github.io/my-thoughts-on-skip-thoughts/)上给出了该方法的一个非常详细的概述，而 [Ammar Zaher 展示了其用于构建烹饪食谱的嵌入空间](https://sourcediving.com/building-recipe-skill-representations-using-skip-thought-vectors-8a6e4c38ae6c)。*

## *快速发送*

*【 [Hill 等人，2016](https://www.aclweb.org/anthology/N16-1162) 】在*skip-thinks*模型的基础上提出了一个明显更简单的变体； *FastSent* 是一个简单的加法(对数双线性)句子模型，旨在利用相同的信号，但计算开销低得多。给定一些上下文句子的 BOW 表示，该模型简单地预测相邻的句子(也表示为 BOW)。更正式地说， *FastSent* 学习模型词汇表中每个单词 *w* 的源 uᵂ和目标 vᵂ嵌入。对于连续句的训练示例 Sᵢ₋₁,Sᵢ,Sᵢ₊₁，Sᵢ被表示为其源嵌入的总和**=*[*∑*](https://en.wikipedia.org/wiki/%E2%88%91)t34】uᵂ超过 *w∈Sᵢ* 。这个例子的成本是简单的[*∑*](https://en.wikipedia.org/wiki/%E2%88%91)𝜙(***sᵢ****，* vᵂ)除以*w∑*sᵢ₋₁∪sᵢ₊₁，其中𝜙是 softmax 函数。这篇论文附有官方的 Python 实现。**

## **思维敏捷的向量**

**[ [Logeswaran & Lee，2018](https://arxiv.org/pdf/1803.02893.pdf) ]将文档嵌入任务——预测句子出现的上下文的问题——重新表述为监督分类问题(见图 12b)，而不是之前方法的预测任务(见图 12a)。**

**![](img/c94774f5aa3b15ecca6c39c538308ddf.png)**

**Figure 12: The Quick-Thought problem formulation (b) contrasted with the Skip-Thought approach (a)**

**要点是使用当前句子的含义来预测相邻句子的含义，其中含义由从编码函数计算的句子的嵌入来表示；注意这里学习了两个编码器:输入句子的 *f* 和候选句子的 *g* 。给定一个输入句子，由编码器(本例中为 RNNs)进行编码，但模型不是生成目标句子，而是从一组候选句子中选择正确的目标句子；候选集是从有效的上下文句子(基本事实)和许多其他非上下文句子中构建的。最后，所构建的训练目标最大化了为训练数据中的每个句子识别正确上下文句子的概率。将以前的句子预测公式视为从所有可能的句子中选择一个句子，这种新方法可以被视为对预测问题的一种判别近似。**

**作者评估了他们在各种文本分类、释义识别和语义相关任务上的方法，并提供了官方 Python 实现[。](https://github.com/lajanugen/S2V)**

## **文字移动器嵌入(WME)**

**一个最近的方法，来自 IBM 的研究，是*字移动器嵌入*()，在 [Wu et al，2018b](https://arxiv.org/pdf/1811.01713v1.pdf) 中提出。[提供了一个官方的基于 C、Python 包装的实现](https://github.com/IBM/WordMoversEmbeddings)。**

**【[库什纳等人，2015](http://proceedings.mlr.press/v37/kusnerb15.pdf) 】提出 W *移动者距离*(WMD)；这将两个文本文档之间的相异度度量为一个文档的嵌入单词需要在嵌入空间中“行进”**到达另一个文档的嵌入单词的最小距离量(见图 13a)。此外，[ [Wu et al，2018a](https://arxiv.org/pdf/1802.04956.pdf) ]提出了 D2KE(到核和嵌入的距离)，一种从给定的距离函数推导正定核的通用方法。****

**![](img/6ebcb6f2d2649c00253edf1a4fc71289.png)**

**Figure 13: Contrasting WMD with WME. (a) WMD measures the distance between two documents *x* and y, while (b) WME approximates a kernel derived from WMD with a set of random documents 𝜔.**

**WME 基于三个组件来学习不同长度文本的连续矢量表示:**

1.  **以无监督的方式学习高质量单词嵌入的能力(例如，使用 *word2vec* )。**
2.  **使用 W*order Mover 的距离* (WMD)基于所述嵌入为文档构建距离度量的能力。**
3.  **使用 D2KE 从给定的距离函数导出正定核的能力。**

**使用这三个组件，应用以下方法:**

1.  **使用 D2KE，通过由*字移动器到来自给定分布的随机文档𝜔的距离* (WMD)给出的无限维特征映射，构造一个正定*字移动器的核* (WMK)。由于使用了 WMD，特征图考虑了由预训练单词嵌入给出的语义空间中的文档之间的单个单词的对齐(参见图 13b)。**
2.  **基于该核，通过核的随机特征近似来导出文档嵌入，其内积近似精确的核计算。**

**这个框架是可扩展的，因为它的两个构建模块，word2vec 和 WMD，可以被其他技术代替，例如 GloVe(用于单词嵌入)或 S-WMD(用于将单词嵌入空间转换成文档距离度量)。**

**作者在 9 个真实世界文本分类任务和 22 个文本相似性任务上评估了 WME，并证明它始终匹配，有时甚至超过其他最先进的技术。**

## **句子-BERT (SBERT)**

**NLP 中的 2018 年以 transformers 的崛起为标志(见图 14)，最先进的神经语言模型受[ [Vaswani et al 2017](https://arxiv.org/pdf/1706.03762.pdf) ]中提出的 transformer 模型的启发-一种序列模型，它无需卷积和递归，而是使用注意力将序列信息纳入序列表示。这个蓬勃发展的家庭包括伯特(及其延伸)，GPT (1 和 2)和 XL 口味的变形金刚。**

**![](img/f14103b972d348a691620ac2f94da03a.png)**

**Figure 14: Rise of the transformers**

**这些模型生成输入标记(通常是子词单元)的上下文嵌入，每个标记都注入了其邻域的信息，但并不旨在为输入序列生成丰富的嵌入空间。BERT 甚至有一个特殊的[CLS]令牌，其输出嵌入用于分类任务，但对于其他任务来说仍然是输入序列的不良嵌入。[ [雷默斯&古雷维奇，2019](https://arxiv.org/pdf/1908.10084.pdf)**

***Sentence-BERT* ，在【 [Reimers & Gurevych，2019](https://arxiv.org/pdf/1908.10084.pdf) 中提出，并伴随有[一个 Python 实现](https://github.com/UKPLab/sentence-transformers)，旨在通过使用连体和三联体网络结构来调整 BERT 架构，以导出语义上有意义的句子嵌入，可以使用余弦相似度进行比较(见图 15)。**

**![](img/f0fab68297892b3b569926d7a16d9359.png)**

**Figure 15: The SBERT architecture in training on a classification objective (left) and inference (right)**

# **监督文档嵌入技术**

**前一节中介绍的非监督方法允许我们从大型未标记的语料库中学习有用的表示。这种方法不是自然语言处理所独有的，它通过设计学习目标来关注学习表示，这些学习目标利用数据中可自由获得的标签。因此，这些方法的强度和稳健性不仅在很大程度上取决于学习框架，还取决于人工设计的学习目标要求或带来在各种下游任务中证明有用的有意义特征或知识的学习的程度。例如，我们希望单词和文档嵌入空间能够很好地捕获语义和句法信息。**

**学习有意义的数据表示(在我们的例子中是单词序列)的对比方法是利用显式标签(几乎总是由人类注释者以某种方式生成)。这里，与各种任务的相关性取决于明确的任务和标签与最终应用的接近程度，同样，该任务对可概括的特征和知识的学习有多好。**

**我们将看到监督方法的范围从那些直接利用特定的标记任务来学习表征，到那些重组任务或从中提取新的标记任务来引出更好的表征。**

## **从标记数据中学习文档嵌入**

**已经有各种尝试使用标记的或结构化的数据来学习句子表示。具体来说，[ [Cho 等人，2014a](https://www.aclweb.org/anthology/D14-1179) ]和[ [Sutskever 等人，2014](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf) ]可能是首次尝试应用编码器-解码器方法来显式学习带有标记数据的句子/短语嵌入；第一个使用了用于统计机器翻译的平行短语语料库 *Europarl* ，第二个使用了来自 WMT 14 数据集的英语到法语的翻译任务。另一个值得注意的尝试在[ [Wieting 等人，2015](https://arxiv.org/pdf/1511.08198.pdf) 和[Wieting & Gimpel，2017]中提出，其中单词嵌入及其到文档嵌入的映射被联合学习，以最小化释义对之间的余弦相似性(来自[PPDB 数据集](http://paraphrase.org/#/))。[ [Hill 等人，2015](https://arxiv.org/pdf/1504.00548.pdf) ]训练神经语言模型，将字典定义映射到由那些定义定义的单词的预训练单词嵌入。最后，【 [Conneau et al，2017](https://www.aclweb.org/anthology/D17-1070.pdf) 】在斯坦福自然语言推理任务上训练了各种架构的 NN 编码器(见图 16)。**

**![](img/2a68b7d7a86372b7e4263595bc3addd5.png)**

**Figure 16: Generic NLI training scheme**

****文档相似性的上下文嵌入** 上述方法的一个具体情况是由文档相似性驱动的。[ [Das 等人，2016](https://www.aclweb.org/anthology/P16-1036) ]展示了通过社区问答的暹罗网络学习到的最大化两个文档之间相似性的文档嵌入(参见图 17)**

**![](img/fb79d6b62992c7559e04e6f63ba541fe.png)**

**Figure 17: The SCQA network consists of repeating convolution, max pooling and ReLU layers and a fully connected layer. Weights W1 to W5 are shared between the sub-networks.**

**同样，[ [Nicosia & Moschitti，2017](https://www.aclweb.org/anthology/K17-1027) ]在学习二进制文本相似性的同时，使用暹罗网络产生单词表示，将同一类别中的示例视为相似。(参见图 18)**

**![](img/27195cbeb716de4d971ac33bfdf1bc92.png)**

**Figure 18: The architecture of the siamese network from [[Nicosia & Moschitti, 2017](https://www.aclweb.org/anthology/K17-1027)]. Word embeddings of each sentence are consumed by a stack of 3 Bidirectional GRUs. Both network branches share parameter weights.**

****跨语言降秩岭回归(Cr5)** 【Josifoski 等人，2019】介绍了一种将任何语言编写的文档嵌入到单一的、与语言无关的向量空间的方法。这是通过训练基于岭回归的分类器来完成的，该分类器使用语言特定的词袋特征来预测给定文档所涉及的概念。当将学习的权重矩阵限制为低秩时，作者表明它可以被分解以获得从语言特定的词袋到语言无关的嵌入的期望映射。[提供了官方的 Python 实现](https://github.com/epfl-dlab/Cr5)。**

## **特定任务的监督文档嵌入**

**产生文档嵌入的普通监督方法使用各种神经网络体系结构，学习将单词向量映射到文档向量的合成算子；这些被传递给受监督的任务，并依赖于类标签，以便通过合成权重反向传播(参见图 19)。**

**因此，网络的几乎所有隐藏层可以被认为产生输入文档的矢量嵌入，直到该层的网络前缀是从单词矢量到嵌入空间的学习映射。在[ [Wieting et al，2015](https://arxiv.org/pdf/1511.08198.pdf) ]中可以找到基于单词向量和监督学习任务的学习句子向量的不同方法的严格检查。**

**![](img/39ca441072bac1fb581bcd261e41af88.png)**

**Figure 19: Neural networks implicitly learn to map word embedding sequences to document embeddings**

**请注意，虽然所使用的单词嵌入可以是预先生成的，并且是任务不可知的(至少在一定程度上)，但是从单词嵌入到文档嵌入的映射是任务特定的。虽然这些对于相关的任务可能是有用的，但这种方法肯定不如无监督的方法健壮和通用，至少在理论上是这样。[ [Kiros 等人，2015 年](https://arxiv.org/pdf/1506.06726.pdf) ]**

**值得注意的应用包括使用 RNNs 的情感分类[Socher 等人，2013]，使用 CNN 的各种文本分类任务[Kalchbrenner 等人，2014] [Kim，2014]，以及使用递归卷积神经网络的机器翻译和文本分类任务[Cho 等人，2014a，2014 b][赵等人，2015]。**

****GPT**
[拉德福德等人，2018](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf)[提出了*生成式预训练* (GPT)方法](https://openai.com/blog/language-unsupervised/) ( [伴随一个 Python 实现](https://github.com/openai/finetune-transformer-lm))，结合无监督和有监督的表示学习，使用[瓦斯瓦尼等人 2017](https://arxiv.org/pdf/1706.03762.pdf) 提出的 transformer 模型在无标签语料库上学习一个无监督的语言模型，然后使用有监督的数据分别对其用于每个任务进行微调。[他们后来在[](https://openai.com/blog/better-language-models/) [拉德福德等人，2019](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) ]中展示了 GPT-2 ，专注于支持他们工作的无监督学习部分，再次[发布了官方的 Python 实现](https://github.com/openai/gpt-2)。**

****深度语义相似度模型(DSSM)** T22【微软研究项目，DSSM 是一种深度神经网络建模技术，用于在连续的语义空间中表示文本串，并对两个文本串之间的语义相似度进行建模(见图 20)。**

**![](img/172adfd9e9efa3cb9aaeabbd2f09a2ba.png)**

**Figure 20: The architecture of a DSSM neural network**

**在其他应用中，DSSM 用于开发潜在的语义模型，该模型将不同类型的实体(例如，查询和文档)投影到公共的低维语义空间中，用于各种机器学习任务，例如排序和分类。例如，【[黄等，2013](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf) 】使用 it 将查询和文档投影到一个公共的低维空间，在该空间中，文档与给定查询的相关性被计算为它们之间的距离。**

**实现包括 [TensorFlow](https://kishorepv.github.io/DSSM/) 、 [Keras](https://github.com/airalcorn2/Deep-Semantic-Similarity-Model) 和[两种 PyTorch](https://github.com/nishnik/Deep-Semantic-Similarity-Model-PyTorch) [变体](https://github.com/moinnadeem/CDSSM)。**

## **共同学习句子表征**

**[ [Ahmad 等人，2018](https://arxiv.org/pdf/1810.00681v1.pdf) ]提出，从多个文本分类任务中联合学习句子表示，并将它们与预先训练的单词级和句子级编码器相结合，会产生对迁移学习有用的健壮句子表示**

**![](img/f21390d5549cb71405214bb8c2075bd8.png)**

**Figure 21: Jointly learning sentence embeddings using auxiliary tasks**

**[ [于&姜，2016](https://www.semanticscholar.org/paper/Learning-Sentence-Embeddings-with-Auxiliary-Tasks-Yu-Jiang/2d38f7aab07d4435b2110602db4138ef20da4cc0) ]类似地表明，使用两个辅助任务来帮助诱导句子嵌入对于跨领域的情感分类来说应该是很好的，与情感分类器本身一起共同学习这个句子嵌入(图 21)。**

****通用句子编码器** 在 [Cer et al，2018a](https://arxiv.org/pdf/1803.11175.pdf) 和 [Cer et al，2018b](https://www.aclweb.org/anthology/D18-2029/) 中提出，并伴随[一个 TensorFlow 实现](https://tfhub.dev/google/universal-sentence-encoder/2)，该方法实际上包括了两种可能的句子表征学习模型:*变换器*模型和*深度平均网络(DAN)* 模型(见图 22)。两者都被设计成允许多任务学习，支持的任务包括(1)类似于无监督学习的任务；(2)用于包含解析的会话数据的会话输入-响应任务；以及(3)用于对监督数据进行训练的分类任务(参见前面的小节)。作者专注于迁移学习任务的实验，并将他们的模型与简单的 CNN 和 DAN 基线进行比较。该方法后来[扩展到解决多语言设置](https://ai.googleblog.com/2019/07/multilingual-universal-sentence-encoder.html)。**

***transformer* 模型直接基于 [Vaswani et al 2017](https://arxiv.org/pdf/1706.03762.pdf) 中提出的 transformer 模型，这是第一个完全基于注意力的序列转导模型，用多头自注意力取代了编码器-解码器架构中最常用的递归层(见图 22a)。**

**该模型使用转换器架构的编码子图来构建句子嵌入。编码器使用注意力来计算句子中单词的上下文感知表示，同时考虑其他单词的顺序和身份。上下文感知单词表示被平均在一起以获得句子级嵌入。**

**![](img/4497daff0eaac42d2f8cafe648ac8381.png)**

**Figure 22: The two models of the Universal Sentence Encoder: (a) Transformer and (b) DAN**

**相反，在丹模型中，在[ [Iyyer et al，2015](https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf) ]中，单词和双字母组合的输入嵌入首先被平均在一起，然后通过前馈深度神经网络(DNN)产生句子嵌入(见图 22b)。**

****GenSen** 与通用句子编码器非常相似，GenSen 方法,【 [Subramanian 等人，2018](https://arxiv.org/pdf/1804.00079.pdf) 】与[官方 Python 实现](https://github.com/Maluuba/gensen)一起提出，结合多个监督和非监督学习任务来训练基于 RNN w/ GRU 的编码器-解码器模型，从该模型中提取嵌入。四个支持的任务是:(1) *跳过思维*向量，(2)神经机器翻译，(3)选区分析，以及(4)自然语言推理(三向分类问题；给定一个前提和一个假设句子，目标是将它们的关系分为蕴涵、矛盾或中性。[官方 Python 实现发布](https://github.com/Maluuba/gensen)。**

# **如何选择使用哪种技术**

**我在这里没有简单的答案，但这里有一些可能的要点:**

1.  ****平均单词向量是一个很好的基线**，所以一个好主意是通过集中精力生成非常好的单词向量，并且首先简单地平均它们，来开始你对好的文档嵌入的探索。毫无疑问，文档嵌入的大部分功能来自于构建它们所基于的词向量，我认为可以肯定地说，在继续前进之前，该层中有大量的信息需要优化。你可以尝试不同的预训练单词嵌入，探索哪些源域和哪些方法(例如 word2vec vs GloVe vs BERT vs ELMo)可以更好地捕捉你需要的信息类型。然后，通过尝试不同的摘要操作符或其他技巧(像[ [阿罗拉等人，2016](https://pdfs.semanticscholar.org/3fc9/7768dc0b36449ec377d6a4cad8827908d5b4.pdf) ]中的那些)来稍微扩展这一点可能就足够了。**
2.  ****绩效可能是一个关键的考虑因素**，尤其是在方法中没有明确的领导者的情况下。在这种情况下，无论是平均字向量的[，还是像](#ecd3) [*sent2vec*](#e3d4) 和 [*FastSent*](#e6e8) 这样的精益方法，都是不错的选择。相比之下，使用 *doc2vec* 时，每个句子所需的实时向量表示推理可能会被证明在给定应用约束的情况下代价很高。 [SentEval，在[](https://github.com/facebookresearch/SentEval) [Conneau & Kiela，2018](https://arxiv.org/pdf/1803.05449.pdf) ]中提出的一个句子表征的评估工具包，就是在这种背景下值得一提的工具。**
3.  **考虑学习目标对你的任务的有效性。上面提到的不同的自我监督技术以不同的方式扩展了分布假设，跳跃思维和快速思维分别根据句子和段落在文档中的距离建立了它们之间的强关系。这也许适用于书籍、文章和社交媒体帖子，但可能不适用于其他文本序列，尤其是结构化的文本，因此可能会将您的文档投影到一个对它们不适用的嵌入空间。同样，WME 所依赖的单词对齐方法可能并不适用于每种情况。**
4.  ****开源实现非常丰富**，因此针对您的任务对不同的方法进行基准测试可能是可行的。**
5.  **没有明确的任务负责人。论文经常针对分类、转述和语义相关度任务测试不同的方法。然而，当考虑关于该主题的所有文献时，特别是考虑 2018 年的两个最新基准测试的结果时，上述结论就会出现，第一个由[ [Logeswaran & Lee，2018](https://arxiv.org/pdf/1803.02893.pdf) 在介绍他们的*快速思考*方法时完成，第二个由[ [Wu 等人，2018b](https://arxiv.org/pdf/1811.01713v1.pdf) 作为他们关于*字移动器嵌入*的论文的一部分完成。**

# **最后的话**

**就是这样！一如既往，我确信我写的帖子并不完整，所以请随时对以上概述提出更正和补充建议，要么在此评论，要么[直接联系我](http://www.shaypalachy.com/)。:)**

**我还要感谢 Adam Bali 和 Ori Cohen，他们提供了非常有价值的反馈。去看看他们的帖子！**

**最后，我发现值得一提的是，代码为的*论文有[致力于文档嵌入](https://paperswithcode.com/task/document-embedding)的任务，脸书研究公司开源了 [SentEval，这是一个在[](https://github.com/facebookresearch/SentEval) [Conneau & Kiela，2018](https://arxiv.org/pdf/1803.05449.pdf) ]中展示的句子表示的评估工具包。***

**现在坐好，让参考文献淹没你。📕 📗 📘 📙**

# **参考**

**Agibetov，a .，Blagec，k .，Xu，h .，和 Samwald，M. (2018 年)。[用于生物医学句子分类的快速可扩展神经嵌入模型](https://link.springer.com/article/10.1186/s12859-018-2496-4)。 *BMC 生物信息学*， *19* (1)，541。**

**艾哈迈德，吴伟国，白，徐，彭，张国威(2018)。[学习用于文本分类的健壮的、可转移的句子表示](https://arxiv.org/pdf/1810.00681v1.pdf)。 *arXiv 预印本 arXiv:1810.00681* 。**

**阿罗拉，s .，梁，y .，，马，T. (2016)。一个简单但难以攻克的句子嵌入基线。[ [非官方实施](https://github.com/peter3125/sentence2vec)**

**Bengio，r . Ducharme，Vincent，p .和 Jauvin，C. (2003 年)。[一个神经概率语言模型](http://www.jmlr.org/papers/volume3/bengio03a/bengio03a.pdf)。*机器学习研究杂志*，*3*(2 月)，1137–1155。**

**B.Broere，(2017)。[跳过思维向量的句法属性](http://arno.uvt.nl/show.cgi?fid=146003)。*蒂尔堡大学硕士论文*。**

**Cer，d .，Yang，y .，Kong，S. Y .，Hua，n .，Limtiaco，n .，John，R. S .，和 Sung，Y. H. (2018 年)。[通用语句编码器](https://arxiv.org/pdf/1803.11175.pdf)。 *arXiv 预印本 arXiv:1803.11175* 。**

**Cer，d .，Yang，y .，Kong，S. Y .，Hua，n .，Limtiaco，n .，John，R. S .，和 Strope，B. (2018 年 11 月)。[英语通用句子编码器](https://www.aclweb.org/anthology/D18-2029/)。在*2018 自然语言处理经验方法会议记录:系统演示*(第 169-174 页)。**

**陈，男(2017)。[通过讹误对文档进行有效的矢量表示](https://arxiv.org/pdf/1707.02377.pdf)。 *arXiv 预印本 arXiv:1707.02377* 。**

**陈，秦，彭，杨，陆，钟(2018)。 [BioSentVec:为生物医学文本创建句子嵌入](https://arxiv.org/pdf/1810.09302.pdf)。arXiv 预印本 arXiv:1810.09302。**

**Cho，k .，Van merrinboer，b .，Gulcehre，c .，Bahdanau，d .，Bougares，f .，Schwenk，h .，和 Bengio，Y. (2014 年)。[使用用于统计机器翻译的 RNN 编码器-解码器学习短语表示](https://arxiv.org/abs/1406.1078)。 *arXiv 预印本 arXiv:1406.1078* 。**

**Cho，k .，Van merrinboer，b .，Bahdanau，d .，和 Bengio，Y. (2014 年)。[关于神经机器翻译的性质:编码器-解码器方法](https://arxiv.org/abs/1409.1259)。 *arXiv 预印本 arXiv:1409.1259* 。**

**Conneau，a .，Kiela，d .，Schwenk，h .，Barrault，l .，和 Bordes，A. (2017 年)。[从自然语言推理数据中监督学习通用句子表示](https://www.aclweb.org/anthology/D17-1070.pdf)。 *arXiv 预印本 arXiv:1705.02364* 。**

**Conneau，a .，& Kiela，D. (2018 年)。 [Senteval:通用语句表示评估工具包](https://arxiv.org/pdf/1803.05449.pdf)。 *arXiv 预印本 arXiv:1803.05449* 。**

**戴，A. M .，Olah，c .，& Le，Q. V. (2015 年)。[用段落向量嵌入文档](https://arxiv.org/pdf/1507.07998.pdf)。 *arXiv 预印本 arXiv:1507.07998* 。**

**Das，a .，Yenala，h .，Chinnakotla，m .，& Shrivastava，M. (2016 年 8 月)。[我们站在一起:相似问题检索的连体网络](https://www.aclweb.org/anthology/P16-1036)。《计算语言学协会第 54 届年会论文集》(第 1 卷:长篇论文)(第 378–387 页)。**

**甘，钟，蒲，杨，，李，何，林(2016)。使用卷积神经网络的句子表示的无监督学习。 *arXiv 预印本 arXiv:1611.07897* 。**

**甘，钟，蒲，杨，，李，何，林(2016)。[使用卷积神经网络学习通用语句表示](https://arxiv.org/pdf/1611.07897.pdf)。arXiv 预印本 arXiv:1611.07897 。**

**Gupta，p .，Pagliardini，m .，和 Jaggi，M. (2019)。[通过解开上下文 n 元语法信息实现更好的单词嵌入](https://www.aclweb.org/anthology/N19-1098)。 *arXiv 预印本 arXiv:1904.05033* 。**

**哈里斯，Z. S. (1954)。分配结构。Word，10(2–3)，146–162。**

**f .希尔、k .乔、a .科尔霍宁和 y .本吉奥(2015 年)。[通过嵌入字典学习理解短语](https://arxiv.org/pdf/1504.00548.pdf)。*计算语言学协会汇刊*， *4* ，17–30。**

**Hill，f .，Cho，k .，和 Korhonen，A. (2016 年)。[从未标记数据中学习句子的分布式表示](https://www.aclweb.org/anthology/N16-1162)。 *arXiv 预印本 arXiv:1602.03483* 。**

**黄，张平，何，高，邓，阿塞罗，海克，(2013 年 10 月)。[使用点击链接数据学习用于 web 搜索的深度结构化语义模型](https://www.microsoft.com/en-us/research/wp-content/uploads/2016/02/cikm2013_DSSM_fullversion.pdf)。在*第 22 届 ACM 国际信息会议论文集&知识管理*(第 2333–2338 页)。ACM。**

**Iyyer、v . Manjunatha、j . Boyd-Graber 和 h . daum III(2015 年)。[深度无序组合匹敌文本分类的句法方法](https://people.cs.umass.edu/~miyyer/pubs/2015_acl_dan.pdf)。在*计算语言学协会第 53 届年会暨第 7 届自然语言处理国际联席会议论文集(第 1 卷:长篇论文)*(第 1 卷，第 1681-1691 页)。**

**Josifoski，m .，Paskov，I. S .，Paskov，H. S .，Jaggi，m .，& West，R. (2019，1 月)。[作为降秩岭回归的跨语言文档嵌入](https://dl.acm.org/citation.cfm?id=3291023)。第十二届 ACM 网络搜索和数据挖掘国际会议论文集(第 744–752 页)。ACM。**

**Kalchbrenner，e . Grefenstette 和 Blunsom，P. (2014 年)。用于句子建模的卷积神经网络。 *arXiv 预印本 arXiv:1404.2188* 。**

**茨韦塔纳·肯特、博里索夫和米·德·里基(2016 年)。 [Siamese cbow:优化句子表达的单词嵌入](https://arxiv.org/pdf/1606.04640.pdf)。 *arXiv 预印本 arXiv:1606.04640* 。**

**金，尹。"用于句子分类的卷积神经网络." *arXiv 预印本 arXiv:1408.5882* (2014)。**

**Kiros，r .，Zhu，y .，Salakhutdinov，R. R .，Zemel，r .，Urtasun，r .，Torralba，a .，& Fidler，S. (2015 年)。[跳过思维向量](https://arxiv.org/abs/1506.06726)。在*神经信息处理系统的进展*(第 3294–3302 页)。**

**m .库斯纳、孙、n .科尔金和 k .温伯格(2015 年 6 月)。[从单词嵌入到文档距离](http://proceedings.mlr.press/v37/kusnerb15.pdf)。在*机器学习国际会议*(第 957–966 页)。**

**刘，J. H .，，鲍德温，T. (2016)。[对 doc2vec 的实证评估，以及对文档嵌入生成的实际见解](https://arxiv.org/pdf/1607.05368.pdf)。 *arXiv 预印本 arXiv:1607.05368* 。[ [代码](https://github.com/jhlau/doc2vec) ]**

**Le，q .，& Mikolov，T. (2014 年 1 月)。[句子和文档的分布式表示](https://cs.stanford.edu/~quocle/paragraph_vector.pdf)。在*机器学习国际会议*(第 1188–1196 页)。**

**李，t .，&帕克，Y. (2018)。[使用基于文档结构的上下文的无监督句子嵌入](https://openreview.net/forum?id=H1a37GWCZ)。**

**Logeswaran，l .，& Lee，H. (2018 年)。[学习句子表征的高效框架](https://arxiv.org/pdf/1803.02893.pdf)。arXiv 预印本 arXiv:1803.02893。**

**李，b，刘，t，杜，x，张，d，赵，z(2015)。[通过预测 n-grams 学习文档嵌入，用于长电影评论的情感分类](https://arxiv.org/abs/1512.08183)。 *arXiv 预印本 arXiv:1512.08183* 。**

**刘，杨，&拉帕塔，M. (2018)。学习结构化文本表示。计算语言学协会汇刊，6，63–75。**

**Mikolov，Chen，k .，Corrado，g .，& Dean，J. (2013 年)。[向量空间中单词表示的有效估计](https://arxiv.org/pdf/1301.3781.pdf)。arXiv 预印本 arXiv:1301.3781。**

**Mikolov，Sutskever，I .，Chen，k .，Corrado，G. S .，& Dean，J. (2013 年)。[单词和短语的分布式表示及其组合性](https://papers.nips.cc/paper/5021-distributed-representations-of-words-and-phrases-and-their-compositionality.pdf)。在*神经信息处理系统的进展*(第 3111–3119 页)。**

**尼科西亚，m .，& Moschitti，A. (2017 年 8 月)。[使用分类信息学习结构语义相似性的上下文嵌入](https://www.aclweb.org/anthology/K17-1027)。在*第 21 届计算自然语言学习会议论文集(CoNLL 2017)* (第 260–270 页)。**

**Pagliardini，m .，Gupta，p .，和 Jaggi，M. (2017 年)。[使用合成 n-gram 特征的句子嵌入的无监督学习](https://aclweb.org/anthology/N18-1049)。 *arXiv 预印本 arXiv:1703.02507* 。**

**Pennington、r . Socher 和 c . Manning(2014 年 10 月)。 [Glove:单词表示的全局向量](https://www.aclweb.org/anthology/D14-1162)。在*2014 年自然语言处理(EMNLP)经验方法会议记录*(第 1532-1543 页)。**

**a .拉德福德、k .纳拉辛汉、t .萨利曼斯和苏茨基弗(2018 年)。 [*用无监督学习提高语言理解*](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) 。技术报告，OpenAI。**

**a .、吴、j .、Child、r .、Luan、d .、Amodei、d .、& Sutskever，I. (2019)。[语言模型是无人监督的多任务学习者](https://d4mucfpksywv.cloudfront.net/better-language-models/language_models_are_unsupervised_multitask_learners.pdf)。 *OpenAI 博客*， *1* (8)。**

**Reimers 和 Gurevych，I. (2019)。[句子伯特:使用连体伯特网络的句子嵌入](https://arxiv.org/pdf/1908.10084.pdf)。 *arXiv 预印本 arXiv:1908.10084* 。**

**鲁道夫，m .，鲁伊斯，f .，阿塞，s .，，布莱，D. (2017)。分组数据的结构化嵌入模型。在*神经信息处理系统的进展*(第 251–261 页)。**

**g .索尔顿和 c .巴克利(1988)。[自动文本检索中的术语加权方法](http://pmcnamee.net/744/papers/SaltonBuckley.pdf)。*信息处理&管理*， *24* (5)，513–523。**

**Sinoara，R. A .、Camacho-Collados，j .、Rossi，R. G .、Navigli，r .、& Rezende，S. O. (2019)。[用于文本分类的知识增强文档嵌入](https://www.sciencedirect.com/science/article/pii/S0950705118305124)。*基于知识的系统*， *163* ，955–971。**

**Socher，r .、Perelygin，a .、Wu，j .、Chuang，j .、Manning，C. D .、ng，a .、Potts，C. (2013 年 10 月)。[情感树库语义合成的递归深度模型](https://nlp.stanford.edu/~socherr/EMNLP2013_RNTN.pdf)。在*2013 年自然语言处理经验方法会议记录*(第 1631-1642 页)。**

**Subramanian，s .，Trischler，a .，Bengio，y .，& Pal，C. J. (2018)。[通过大规模多任务学习来学习通用分布式句子表示](https://arxiv.org/pdf/1804.00079.pdf)。 *arXiv 预印本 arXiv:1804.00079* 。**

**Sutskever，I .，Vinyals，o .，& Le，Q. V. (2014 年)。[用神经网络进行序列对序列学习](https://papers.nips.cc/paper/5346-sequence-to-sequence-learning-with-neural-networks.pdf)。在*神经信息处理系统的进展*(第 3104–3112 页)。**

**唐，苏，金，洪，方，王，张，&德萨，V. R. (2017)。[反思跳跃思维:一种基于邻域的方法](https://arxiv.org/abs/1706.03146)。 *arXiv 预印本 arXiv:1706.03146* 。**

**唐，苏，金，洪，方，王，张，&德萨，V. R. (2017)。[修剪和改进跳跃思维向量](https://www.groundai.com/project/trimming-and-improving-skip-thought-vectors/1)。 *arXiv 预印本 arXiv:1706.03148* 。**

**Thongtan 和 t . phiquethrakul(2019 年 7 月)。基于余弦相似度训练的文档嵌入的情感分类。《计算语言学协会第 57 届会议记录:学生研究研讨会》(第 407-414 页)。**

**Vaswani，a .、Shazeer，n .、Parmar，n .、Uszkoreit，j .、Jones，l .、Gomez，A. N .、… & Polosukhin，I. (2017)。[注意力是你所需要的全部](https://arxiv.org/pdf/1706.03762.pdf)。*神经信息处理系统进展*(第 5998-6008 页)。**

**j .维廷、m .班萨尔、k .金佩尔和 k .利维斯库(2015 年)。[走向普遍的倒装句嵌入](https://arxiv.org/pdf/1511.08198.pdf)。arXiv 预印本 arXiv:1511.08198。**

**j .维廷和 k .金佩尔(2017 年)。重新考察用于反对比句子嵌入的循环网络。 *arXiv 预印本 arXiv:1705.00364* 。**

**吴，严，刘怡红，徐，李，李，李(2018)。 [D2ke:从距离到内核和嵌入](https://arxiv.org/pdf/1802.04956.pdf)。arXiv 预印本 arXiv:1802.04956 。**

**Wu，l .，Yen，即，Xu，k .，Xu，f .，Balakrishnan，a .，Chen，P. Y，... & Witbrock，M. J. (2018)。 [Word Mover 的嵌入:从 Word2Vec 到文档嵌入](https://arxiv.org/pdf/1811.01713v1.pdf)。 *arXiv 预印本 arXiv:1811.01713* 。**

**俞军、蒋军(2016 年 11 月)。[跨领域情感分类辅助任务学习句子嵌入](https://www.semanticscholar.org/paper/Learning-Sentence-Embeddings-with-Auxiliary-Tasks-Yu-Jiang/2d38f7aab07d4435b2110602db4138ef20da4cc0)。在*2016 年自然语言处理经验方法会议论文集*(第 236–246 页)。**

**张，杨，陆，张(2019)。用子词信息和网格改进生物医学词嵌入。*科学数据*， *6* (1)，52。**

**赵，洪，陆，张，等(2015 年 6 月)。自适应层次句子模型。在*第二十四届国际人工智能联合大会*。**