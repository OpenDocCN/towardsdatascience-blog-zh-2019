# 在非 NLP 上下文中嵌入 Word2Vec 详细信息

> 原文：<https://towardsdatascience.com/embeddings-with-word2vec-in-non-nlp-contexts-details-e879b093d34d?source=collection_archive---------12----------------------->

![](img/c56b9640791e9742338654153f4c8c3e.png)

Photo by [Charles Gao](https://unsplash.com/@charlesgs?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

## 探索使用基于 Word2Vec 的模型在业务上下文中创建项目嵌入的细节。

本文档要求熟悉 wor2vec[1，2，3]类模型和深入学习的文献。

在本文档中，我们将探讨在非 NLP 业务上下文中使用 Word2Vec 类模型创建嵌入向量的细节。我们将使用 Instacart Dataset[4，31]创建产品嵌入，评估模型体系结构/参数，并将生成的嵌入用于项目/购物篮相似性/互补性预测。

示例代码[31]可从 Gensim 获得。

# 为什么是 Word2Vec？

正是计算效率、可扩展性和“带负采样的 SkipGram”(SGNS)体系结构[2]使它变得特别。当您还将 SGNS 可视化为双编码器模型[5]时，正是输入层的易扩展性使其更加特殊。

对 SGNS 体系结构和参数有扎实的了解是:

*   评估业务上下文中的嵌入创建[6]
*   在进一步的下游模型/任务中使用这些嵌入[12，13]
*   评估其他嵌入模型(Node2Vec、Graph 神经网络、Transformers、FastText、Bert…)。

# **为什么要用 Word2Vec 创建嵌入？**

大家可能已经知道，嵌入可以作为具有不同输入和输出的复杂多层监督模型的副产品来生成。这不是最佳的嵌入生成模型。原因何在？

您希望嵌入能够真正反映数据集中存在的目标/业务功能(例如，协作过滤)。要实现这一点，您可能希望模型中的嵌入参数仅参与模型输出的计算，而不涉及其他层或参数。在 SGNS 模型中，唯一的参数是嵌入参数，而模型输出只是 Target(In)和 Context(Out)嵌入参数的点积。所以你的目标函数仅仅用这些嵌入参数来表示。这是将目标函数语义注入到嵌入向量参数中的最佳方式；因此，您可以按原样有效地使用/评估嵌入，或者将它们用作其他下游模型的输入。

相反，在具有附加输入参数或嵌入层之上的层的模型中，目标函数与嵌入参数和模型中的其他参数一起表示。因此，在这种情况下，仅评估嵌入参数(例如，嵌入向量空间中的相似性)将是目标函数的次优表示。

从技术上来说，并加以总结；在网络中，您不希望将具有可学习参数的附加层放在嵌入层之上(或附近)。在嵌入层的上方应该有相似性函数(例如，点积> sigmoid)。见下文。

![](img/d350dae5d80fea2961f8d017ab19e144.png)

Embedding Generation Model Architectures

作为补充说明；与上述架构相比，对于一些业务案例，您可能希望您的嵌入反映多个任务(例如，点击概率和喜欢概率)。如果任务是相关的，你可以通过创建多任务学习模型的嵌入来实现更好的概括[22，23，24]。选择取决于您的架构设计。

# **非 NLP 设置中的 word 2 vec**

Word2Vec 和 Doc2Vec 的非 NLP 改编的标志性论文有；分别是 Prod2Vec[7]和 Meta-Prod2Vec[8]。您可以嵌入任何对象，只要您可以为该对象定义相应的上下文/环境；无论是顺序的(例如搜索日志、产品交互序列[7])还是图表[9，10，27](例如社交网络、分子)。正确定义你的上下文的能力[11]是你的模型最关键的部分，其次是你的其他 SGNS 参数。

# **根据订单生成产品嵌入(Instacart 数据集)**

Instacart 数据集[4]包含来自 200，000 多个 Instacart 用户的超过 300 万份杂货订单，其中每个订单包含一系列购买的产品。我们将为每个产品创建嵌入向量。我们将不使用任何产品/用户侧信息，并且仅将数据集中固有的协同过滤语义嵌入到嵌入向量中。尽管这些生成的产品嵌入可以单独用于对产品相似性/互补性和当前/下一篮子/项目推荐进行建模；理想情况下，为了获得更高的模型精度，您应该考虑具有附加输入功能的更高级的模型。与检索/排名架构一样，您也可以简单地考虑并使用这些嵌入作为检索阶段模型(向量相似性搜索),并进一步使用选择的嵌入作为具有附加输入特征的更复杂的排名模型的输入。(检索/排名模型[12])。

![](img/6153c780d90cfef6c80eee6a7ad88e7c.png)

# 对偶嵌入空间

SGNS 为每个对象学习两个不同的嵌入向量；入和出向量(也称为目标和上下文向量)。为什么每个对象需要两个不同的向量？许多人在下游任务[26]中通过平均输入和输出向量来使用单个向量，或者丢弃输出向量而只使用输入向量。还可以通过在训练时在模型中使用共享嵌入参数层来学习单个嵌入向量(具有共享参数的暹罗网络[25])。

那么为什么要为每个对象创建两个单独的向量呢？让我们检查技术和逻辑推理。

**技术:**让我们把心态从 NLP 转移到 Instacart 数据集；“词”变成了“产品”，“句”变成了“订单筐”。产品的上下文是当前订单篮中的其他产品。对于产品“一袋香蕉”，考虑我们对目标(入)和上下文(出)使用相同向量的情况。“一袋香蕉”在语义上不会出现在其自身的上下文中(上下文是订单篮)。通过对“一袋香蕉”使用相同的向量 v；分配一个低概率的 p(“一袋香蕉”|“一袋香蕉”)是不可能的，因为给 v v 分配一个低值是不可能的。

**逻辑:**使用双重输入输出向量使我们能够评估产品在目标或上下文环境中的概率。因此，我们可以计算乘积相似性(输入向量中的余弦相似性)或乘积互补性(输入和输出向量之间的余弦相似性)。最终，这种“双重嵌入空间”架构形成了用于生产的更高级的相似性/互补性预测模型的基础[16，17，18，19，20]。

![](img/27bbed906e2fcd0c1e94d3c2b4177aa6.png)

Word2Vec Model in Tensorflow(Also refered as Dual Encoder Model, Siamese Networks or Dual Tower Networks)

# 模型参数

让我们评估 SGNS 参数；

**窗口大小:**设置窗口大小取决于任务。在 Airbnb 案例[11]中，列表嵌入是通过用户的列表点击会话序列生成的，在会话中被连续点击的列表可能比在会话中第一个和最后一个被点击的列表在语义上更相关。因此，设置一个小的窗口大小(3-6)可能适合于缩小序列中的相关性窗口。数据越多，可以使用的窗口越小。

然而，在 Instacart 数据集的情况下，订单购物篮中的产品与购物篮中的所有其他产品相关，因为我们的目标函数是“购物篮级别”内的相似性/互补性预测。因此，我们的窗口大小是每个订单的篮子大小计数。作为一个额外的理论说明，如果您的数据集足够大，并且如果您在每个时期的订单篮中打乱产品的顺序，您可以使用较小的窗口大小；并且可以实现与使用更大的窗口尺寸相同的结果。

**数据集生成:**目标上下文 **(** In-Out)数据对是使用窗口大小参数从数据集构建的。对于每个目标，您可以为以下目标向数据集添加额外的数据对:

*   添加目标元数据以实现更好的泛化(Meta-Prod2Vec)[8]。例如，目标产品类别
*   将其他对象嵌入到相同的嵌入空间中，例如品牌[8],例如目标品牌
*   添加额外的目标-上下文对以影响或添加额外的关联到嵌入向量[11]

**纪元:**纪元的数量对结果没有边际影响，您可以通过离线收敛评估轻松决定。但是，请注意，原始 Word2Vec 代码[36]和 Gensim 等库不使用小型批处理(没有小型批处理，模型参数会随数据集中的每个数据更新)，因此与使用小型批处理的模型相比，增加历元数不会产生相同的效果。

**候选采样:**候选采样算法实现了高效的任务学习架构，而无需计算整个标签类的完整 softmax，29]。由于 SGNS 使用负抽样方法[2]，抽样分布生成和相关抽样参数在建立成功的 SGNS 模型中起着至关重要的作用。那么，如何设置负采样架构呢？

*   一般采样-使用采样分布参数从相同的输入数据集中提取负样本(详见下文)。
*   特定于上下文的采样—使用目标上下文选择阴性样本。在 Instacart 案例中，对于特定产品，您可以从相同的产品类别/通道中选择阴性样品。这种“硬否定”技术使模型能够更快更好地收敛。然而，你需要在这方面投入资源，因为你需要能够为每个目标选择否定。可以在小批量训练期间检索负分布，或者可以预先生成静态负分布数据集。这种选择取决于您的培训硬件、分布式培训架构和成本。

**负采样噪声分布参数(** α **):** 使用频率平滑参数(α)从您的分布中采样负样本，其中项目的频率被提升到α的幂。使用α，您可以调整选择流行或稀有物品作为否定的概率。

*   α=1 是均匀分布-使用数据集中的原始项目频率。
*   0
*   α=0 is unigram distribution — item frequency is 1 in dataset.
*   α<0 — low-frequency items are weighted up.

**阴性样本数(k):** 对于我们抽样分布中的每个目标，选择 k 个阴性样本。在下一节中，我们将评估 k 和α之间的相关性。

# **评估-订单篮中的下一个产品预测**

我们将使用 Instacart 数据集，通过预测当前订单篮中的下一个商品来评估模型参数(k，α)。

样本代码[31]。(为清楚起见，使用 Gensim。)

在训练数据集上训练我们的模型之后，使用测试集，我们将在客户的订单篮子中隐藏一个随机产品，并通过使用篮子中的其他产品来预测隐藏的产品。我们将通过平均篮子中产品的目标嵌入来计算“篮子向量”。然后，利用计算出的“篮子向量”，我们将在上下文(外部)向量空间中搜索最近的项目，并将最近的项目作为推荐呈现。这个推荐基本上就是“这里有推荐给你的产品，以你已经放在篮子里的东西计算”。下面是采用不同 k 和α的命中率@10 分析。

![](img/cf7cd7552d719e664a92fb334dac4cbe.png)

Hitrate@10

![](img/b207731ea6b197c5fec90fb62873aba1.png)

我们看到高α (α=0.75，α=1)时精度低。直觉存在；在高α的情况下，流行的高频项目/用户支配分布，并降低模型的泛化能力。

随着α的降低，我们预测更多“不太频繁”的产品，这导致更好的模型得分，在α=0 时达到最大值。那么你会为这个模型选择哪个α呢？我会选择α=-0.5，因为尽管它的得分低于α=0，但我会认为它在在线评估中的得分会更高，假设它会为客户提供更多样化的推荐(意外发现，新奇)。

**α和 k 的相关性:**

**与高α(α>0)**；增加 k，**会降低**精度。直觉是；在高α的情况下，你推荐的是不能反映真实分布的流行/高频项目。如果你在这种情况下增加 k，你会选择更多的流行/高频项目作为负项，这导致进一步过度拟合不正确的分布。端点是；如果想推荐高α的热门单品，没必要进一步增加 k。

**用低α(α<0)；**增加 k，**增加**精度。直觉是；α值较低时，你会从尾部推荐更多不同的商品(不常见的商品)。在这种情况下，如果增加 k，就会从尾部选择更多的负样本，使模型能够看到更多不同的项目，从而很好地适应多样化的分布。端点是；如果你想推荐不同的低α的项目(新奇的，意外的),你可以进一步增加 k 直到它开始过量。

注意:不要忘记使用余弦距离进行评估(标准化向量的点积)。如果使用欧氏距离，可能会过分强调数据集中的频繁项；并且还可能由于您嵌入的初始化值而错误地表示不常见的项目。

# **进一步分析**

到目前为止，我们已经分析了 Word2Vec。它的优势在于简单性和可伸缩性，但这也是它的弱点。使用 Word2Vec，不容易合并项目属性或项目-项目关系属性(图形结构)。这些额外的项目/交互特征对于更好的模型概括/准确性以及缓解冷启动问题是必需的。

是的，您可以使用 Meta-Prod2Vec [8]添加项目功能，使用图随机游走模型([9，10])也可以添加节点功能。但问题是，所有这些添加的‘特征’都是 id，而不是矢量；也就是说，它们是存储在后端的特性的 id。对于最先进的模型，您可能希望使用矢量作为项目特征(例如，编码的项目图像矢量)。此外，添加的项目特征没有加权，因此，每个项目属性对项目和模型输出的影响不会作为参数学习。

Word2Vec 的另一个缺点是它是直推式的；这意味着当一个新项目被添加到可用项目列表中时，我们必须重新训练整个模型(或者继续训练模型)。

使用相对较新的归纳图神经模型[34，35]，我们可以添加项目/关系特征作为可学习的向量，并且还可以获得训练模型看不到的新项目的嵌入。

图形神经网络研究是未来几年令人兴奋的领域。

# **结论**

本文档的主要要点；

*   非自然语言处理环境下基于 Word2Vec 的模型分析。
*   模型参数分析，主要是α和 k 之间的相关性，用于业务上下文的相似性/互补性预测。

未来的文档将继续探索更高级的嵌入模型和下游深度模型。

[1]向量空间中单词表示的有效估计(【https://arxiv.org/abs/1301.3781】T2

[2]词和短语的分布式表征及其组合性(【https://arxiv.org/abs/1310.4546】

[3] word2vec 解释:推导 Mikolov 等人的负采样单词嵌入法([https://arxiv.org/abs/1402.3722](https://arxiv.org/abs/1402.3722))

[4]2017 年 Instacart 在线杂货购物数据集”([https://www.instacart.com/datasets/grocery-shopping-2017](https://www.instacart.com/datasets/grocery-shopping-2017))

[5]万物嵌入:神经网络时代的搜索([https://youtu.be/JGHVJXP9NHw](https://youtu.be/JGHVJXP9NHw))

[6]Embeddings @ Twitter([https://blog . Twitter . com/engineering/en _ us/topics/insights/2018/embeddingsattwitter . html](https://blog.twitter.com/engineering/en_us/topics/insights/2018/embeddingsattwitter.html))

[7]收件箱中的电子商务:大规模的产品推荐(【https://arxiv.org/abs/1606.07154】T2

[8] Meta-Prod2Vec —使用边信息进行推荐的产品嵌入(【https://arxiv.org/abs/1607.07326】T4)

[9]深度行走:在线学习社交表征([https://arxiv.org/abs/1403.6652](https://arxiv.org/abs/1403.6652))

[10] node2vec:网络的可扩展特征学习([https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf](https://cs.stanford.edu/~jure/pubs/node2vec-kdd16.pdf))

[11]在 Airbnb 使用搜索排名嵌入的实时个性化([https://www . KDD . org/KDD 2018/accepted-papers/view/Real-time-personal ization-using-embedding-for-Search-Ranking-at-Airbnb](https://www.kdd.org/kdd2018/accepted-papers/view/real-time-personalization-using-embeddings-for-search-ranking-at-airbnb))

[12]深度神经网络为 YouTube 推荐([https://ai.google/research/pubs/pub45530](https://ai.google/research/pubs/pub45530))

[13]推荐接下来看什么视频:多任务排名系统([https://dl.acm.org/citation.cfm?id=3346997](https://dl.acm.org/citation.cfm?id=3346997))

[14] Word2vec 适用于建议:超参数问题([https://arxiv.org/abs/1804.04212](https://arxiv.org/abs/1804.04212))

[15]用于信息检索的神经模型([https://youtu.be/g1Pgo5yTIKg](https://youtu.be/g1Pgo5yTIKg)

[16]从购物篮和浏览会话中推断互补产品([https://arxiv.org/pdf/1809.09621.pdf](https://arxiv.org/pdf/1809.09621.pdf)

[17]使用四元组网络的互补相似性学习([https://arxiv.org/abs/1908.09928](https://arxiv.org/abs/1908.09928))

[18]推断可替代和互补产品的网络([https://arxiv.org/abs/1506.08839](https://arxiv.org/abs/1506.08839))

[19]用于互补产品推荐的上下文感知双表征学习([https://arxiv.org/pdf/1904.12574.pdf](https://arxiv.org/pdf/1904.12574.pdf))

[20]指数家族嵌入(【https://arxiv.org/abs/1608.00778】T2

[21]大规模网络中的袖珍结构嵌入(【https://youtu.be/B-WFdubGkIo】T4)

[22]在 Pinterest 学习视觉搜索的统一嵌入([https://arxiv.org/abs/1908.01707](https://arxiv.org/abs/1908.01707))

[23]深度神经网络中多任务学习的概述([https://ruder.io/multi-task/](https://ruder.io/multi-task/))

[24]利用多门混合专家对多任务学习中的任务关系进行建模([https://www . KDD . org/KDD 2018/accepted-papers/view/Modeling-Task-Relationships-in-Multi-Task-Learning-with-Multi-gate-Mixture-](https://www.kdd.org/kdd2018/accepted-papers/view/modeling-task-relationships-in-multi-task-learning-with-multi-gate-mixture-))

[25] DeepFace:缩小与人脸验证中人类水平性能的差距( [DeepFace:缩小与人脸验证中人类水平性能的差距](https://www.researchgate.net/publication/263564119_DeepFace_Closing_the_Gap_to_Human-Level_Performance_in_Face_Verification))

[26] GloVe:单词表示的全局向量([https://nlp.stanford.edu/projects/glove/](https://nlp.stanford.edu/projects/glove/))

[27]图卷积网络:综合评述([https://rdcu.be/bW6sK](https://rdcu.be/bW6sK))

[28]关于单词嵌入—第 2 部分:逼近软最大值([https://ruder.io/word-embeddings-softmax/](https://ruder.io/word-embeddings-softmax/))

[29]候选抽样-张量流([https://www.tensorflow.org/extras/candidate_sampling.pdf](https://www.tensorflow.org/extras/candidate_sampling.pdf)

[30]使用负采样优化 Skip-Gram 的计算效率([https://aegis 4048 . github . io/Optimize _ Computational _ Efficiency _ of _ Skip-Gram _ with _ Negative _ Sampling](https://aegis4048.github.io/optimize_computational_efficiency_of_skip-gram_with_negative_sampling))

[31][https://github . com/boraturant/word 2 vec _ insta cart _ Similarity _ complementary](https://github.com/boraturant/Word2Vec_Instacart_Similarity_Complementarity)

[32] CS224W:带图的机器学习(【http://web.stanford.edu/class/cs224w/】T2

[33]图的节点嵌入算法(【https://youtu.be/7JELX6DiUxQ】T4)

[34]大型图上的归纳表示学习([https://arxiv.org/abs/1706.02216](https://arxiv.org/abs/1706.02216))

[35]关系归纳偏差、深度学习和图形网络([https://deep mind . com/research/open-source/graph-nets-library](https://deepmind.com/research/open-source/graph-nets-library))

[https://code.google.com/archive/p/word2vec/](https://code.google.com/archive/p/word2vec/)