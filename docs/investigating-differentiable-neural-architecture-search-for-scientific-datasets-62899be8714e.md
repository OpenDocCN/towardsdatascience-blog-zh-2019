# 研究用于科学数据集的可区分神经架构搜索

> 原文：<https://towardsdatascience.com/investigating-differentiable-neural-architecture-search-for-scientific-datasets-62899be8714e?source=collection_archive---------28----------------------->

## **哈佛数据科学顶点项目，2019 年秋季**

![](img/b2075395a784efe8d36a1c55c74d0001.png)

*In partnership with* [*Google AI*](https://ai.google/)*.*

**团队成员:**迪伦·兰德尔、朱利安·拉斯里、迈克尔·伊曼纽尔、庄嘉玮

*免责声明:本博客中表达的观点和意见仅属于作者个人，并不代表哈佛或谷歌。*

# 对高效神经结构搜索(NAS)的需求

深度学习把我们从特征工程中解放出来，却产生了一个“[架构工程](https://www.reddit.com/r/MachineLearning/comments/4nwn2e/in_deep_learning_architecture_engineering_is_the/)的新问题。已经发明了许多神经网络体系结构，但是体系结构的设计感觉上更像是一门艺术而不是科学。人们对通过[神经架构搜索(NAS)](https://www.fast.ai/2018/07/16/auto-ml2/) 来自动化这种耗时的设计过程非常感兴趣，正如数量迅速增长的研究论文所示:

![](img/cab14eb65d56b410b37da5fafa22e280.png)

Figure from [Best Practices for Scientific Research on Neural Architecture Search](https://arxiv.org/abs/1909.02453).

典型的 NAS 工作流包括(1)在预定义的搜索空间内提出候选体系结构，(2)评估所提出的体系结构的性能，以及(3)根据搜索策略提出下一个候选体系结构。

![](img/f5f0fef7ff288825e94011f4ee3843ac.png)

Figure from [Neural Architecture Search: A Survey](https://arxiv.org/abs/1808.05377).

有许多搜索策略(在[调查](https://arxiv.org/abs/1808.05377)中回顾)，包括强化学习、贝叶斯优化、进化算法、基于梯度的优化，甚至随机搜索。一些策略可能非常昂贵，消耗大量能源，并导致数百万美元的云计算账单:

![](img/a8bf90beb56756d41fb88ca555f9d796.png)

Figure from [Energy and Policy Considerations for Deep Learning in NLP](https://arxiv.org/abs/1906.02243); NAS cost is based on evolutionary architecture search on Transformer.

在这个项目中，我们研究了一种高效的基于梯度的搜索方法，称为 [DARTS(可区分架构搜索)](https://arxiv.org/abs/1806.09055)，最近[在 ICLR 2019](https://openreview.net/forum?id=S1eYHoC5FX) 发表。据显示，DARTS 比以前的方法如 [NASNet](https://arxiv.org/abs/1707.07012) 和 [AmoebaNet](https://arxiv.org/abs/1802.01548) 需要的 GPU 时间少 100 倍，并且与来自谷歌大脑的 [ENAS](https://arxiv.org/abs/1802.03268) 方法具有竞争力。我们将把 DARTS 与随机搜索(实际上相当不错，见下表)和最先进的手工设计的架构(如 [ResNet](https://arxiv.org/abs/1512.03385) )进行比较。

![](img/18691094b073bdda36ed9b641e8ae614.png)

Figure from the [DARTS paper](https://arxiv.org/abs/1806.09055).

# 科学数据集

大多数 NAS 研究，包括最初的 DARTS 论文，都使用标准图像数据集报告了实验结果，如 [CIFAR](https://www.cs.toronto.edu/~kriz/cifar.html) 和 [ImageNet](http://www.image-net.org/) 。然而，我们认为深度学习显示出科学研究的前景，包括[生物学](https://www.nature.com/articles/d41586-018-02174-z)、[医学](https://www.nature.com/collections/gcgiejchfa)、[化学](https://arxiv.org/abs/1701.04503)和[各种物理科学](https://dl4physicalsciences.github.io/)。在这个项目中，我们想看看 DARTS 是否对科学数据集有用，以及神经结构如何在这些领域之间转移。

我们使用来自材料科学、天文学和医学成像的三个数据集，具体来说:

*   **石墨烯 Kirigami:** 切割石墨烯优化应力/应变。来自《物理评论快报》上的论文[使用机器学习加速搜索和设计可拉伸石墨烯 kiri gami](https://journals.aps.org/prl/abstract/10.1103/PhysRevLett.123.069901)。
*   **星系动物园:**从望远镜图像中分类星系形态。来自[银河动物园 2](https://arxiv.org/abs/1308.3496) 和[卡格尔银河动物园挑战赛](https://www.kaggle.com/c/galaxy-zoo-the-galaxy-challenge/overview)。
*   **胸透:**从胸透预测疾病。来自论文 [ChestX-ray8](https://arxiv.org/abs/1705.02315) 和 [Kaggle NIH 胸部 x 光数据集](https://www.kaggle.com/nih-chest-xrays/data)。

![](img/92b4083710e56701ff3648045ac35202.png)

Example data from the three datasets (left to right): Graphene Kirigami, Galaxy Zoo, and Chest X-Ray.

我们还以[](http://yann.lecun.com/exdb/mnist/)**(一个众所周知的用于分类手写数字的图像数据集)作为非科学基线进行了实验。**

# **可区分神经结构搜索(DARTS)**

**有 3 个关键的想法使飞镖有效:(1)只搜索小“细胞”，(2)重量共享，和(3)持续放松。每个想法都解决了传统搜索方法中的一个问题，如下图所示。**

**![](img/ddffe54ae507c0f42e9bf1b73c4ba114.png)**

**Issues with brute-force NAS and solutions proposed by DARTS.**

# **DARTS 搜索小“单元”以减少搜索空间**

**为了避免搜索所有“任意架构”的巨大空间，DARTS 利用了一个重要的观察结果，即流行的 ConvNet 架构通常包含重复的块，按顺序堆叠。例如，ResNet 是残差块的序列:**

**![](img/8fecb024d68db809d1a169ae2228d5e2.png)**

**Figures from [here](http://torch.ch/blog/2016/02/04/resnets.html) and [here](https://www.researchgate.net/figure/The-representation-of-model-architecture-image-for-ResNet-152-VGG-19-and-two-layered_fig2_322621180).**

**遵循类似的思路，DARTS 只搜索最优的“块”或“单元”，而不是整个模型。一个单元通常由大约 5-10 个操作组成，用有向无环图(DAG)表示。所有单元按顺序堆叠以形成整个模型。**

**![](img/efaaa7541f324e48f2ef552801a60cbe.png)**

**Example of a reduction cell stacked to create a full model.**

**DARTS 中定义了两种类型的单元:(1)保持输出空间维度与输入空间维度相同的“正常单元”，以及(2)将输出空间维度减半同时使滤波器/通道数量加倍的“缩减单元”。所有正常细胞共享相同的架构(操作和连接),但具有独立的待训练权重——所有 reductions 细胞也是如此。**

# **DARTS 使用“连续松弛”来实现建筑参数的梯度下降**

**下图总结了查找最佳单元的步骤:**

**![](img/5e0fa5ef2fc5510fbaa2ed65923d4f28.png)**

**Schematic of the DARTS continuous relaxation and discretization methods.**

**“找到最佳单元”等同于“在 DAG 的边上找到操作的最佳位置”。DARTS 不是独立评估各种布局(每种布局都需要从头开始训练)，而是叠加所有候选操作(例如，Conv 3x3、最大池、身份等)。)上，因此它们的权重可以在单个过程中一起训练。edge (i，j)处的实际操作是所有候选操作 o(x)的平均值，用α加权:**

**![](img/b9906576f5915b7057a3eb418e6746c4.png)**

**Each edge is a linear combination of pre-defined operations weighted by the softmax output of the architecture parameters.**

**有了一定的架构权重α的选择，相应的架构原则上可以被训练收敛，导致最优的模型权重 *w* *(α)和最终的*验证* *损失* *L* ( *w* *(α)，α)。 *L* w.r.t .到α的梯度给出了架构参数梯度下降的方向。**

**通过每次训练 *w* 收敛来计算真实损耗 *L* 代价太高。因此，DARTS 只训练 *w* 一个步骤来获得代理损失:**

**![](img/c599c787ffc405f8ee750494152f5d4a.png)**

**DARTS utilizes a one-step approximation to avoid training weights to convergence at each architecture update step.**

**其中最佳模型权重 *w* *(α)通过一步训练来近似。**

**α和 *w* 的训练交替进行；**

**![](img/bcdf2f03011bae3dc9ae4ead50402a2c.png)**

**Bi-level optimization procedure for training regular model weights and architecture parameters.**

# **实验结果**

**下面，我们总结了用 DARTS、随机搜索和 ResNet 得到的结果。我们报告了有和没有架构离散化(我们分别称之为“离散”和“连续”)的 DARTS 的结果。**

**![](img/93da081471f813202c619ed91ea3cc98.png)**

**Results of DARTS (continuous & discrete), Random Search, and ResNet on each dataset.**

# **MNIST**

**如前所述，我们首先用 MNIST 做实验。这立即让我们发现了超参数在训练中确保稳定性的重要作用。下面我们展示了适当调整学习率(常规权重和架构权重)的效果。**

**![](img/441bd27eccd94668a0003c0baa94b4a8.png)**

**Learning curves of DARTS on the MNIST dataset demonstrate the important of appropriate hyperparameters.**

**在 MNIST，雷斯内特比飞镖表现更好。这并不奇怪，因为手工设计的架构通常专门针对 MNIST(以及 ImageNet、CIFAR 等)等数据集进行了优化。).考虑到这一点，我们继续前进到主要的科学数据集。**

# **石墨烯 Kirigami**

**石墨烯 Kirigami 的任务是从石墨烯片的切割构型预测拉伸性能。这些数据来自于(计算成本很高的)数值模拟。**

**![](img/4de42098894d2d2c01a0f4c3ef169c4d.png)**

**Overview of the Graphene Kirigami task.**

**我们发现，连续(非离散)飞镖和 ResNet 在这项任务中表现得差不多。但是，我们也发现，这个任务可能过于简单，无法评价飞镖。下面我们展示了 DARTS、ResNet 和“微小”ResNet 的结果。请注意，即使这个“微小”的模型也达到了大约 0.92 的最佳 R。**

**![](img/b9964b5eb0045ce989b261d5d08c0614.png)**

**Table showing that even tiny models can perform well on the Graphene Kirigami task. *Time to run 30 epochs on a single GPU.**

**这让我们得出结论，石墨烯 Kirigami 任务对飞镖来说太简单了，没有用。我们通过查看 a)学习到的架构权重的分布，以及 b)随机搜索架构的性能分布，进一步研究了这一概念，如下所示:**

**![](img/ad2957e4b9a8c55159d67a6b1b8255b4.png)**

**Graphene Kirigami architecture weights (left) and random search performance (right). Our conclusion is that simple problems admit many performant architectures, and that DARTS induces low sparsity in architecture weight space as a result.**

**特别值得注意的是，DARTS 学习的大部分架构权重几乎没有从它们的初始值 1/8 改变(在这个实验中我们有 8 个操作)。我们还看到，除了几个主要的异常值之外，来自随机搜索的架构都表现得非常相似。这使我们得出结论，石墨烯 Kirigami 问题允许许多高性能的架构，因此 DARTS 同样不会学习稀疏架构。**

# **银河动物园**

**受到石墨烯任务中发现的明显缺乏难度的启发，我们寻求一个更复杂的科学兴趣问题。我们选定了银河动物园挑战赛。与典型的分类问题不同，由于目标标签是人类贴标机的平均预测，因此该任务基于均方根误差(RMSE)度量来评分。任务是将星系图像回归到人类标签员对 37 种不同属性的平均预测。**

**![](img/0b95bd04484a1eb3e2adcf5565c8b9c1.png)**

**Summary of the Galaxy Zoo “decision tree”. It is a logical map over the various observed galaxy morphologies.**

**令人兴奋的是，我们发现飞镖(连续)表现最好。如果我们检查架构权重和随机搜索性能(如下)，我们会看到 DARTS 学习了比石墨烯任务更稀疏的细胞。此外，随机架构性能的可变性(注意对数标度)非常大。这表明，对于星系动物园问题，确实有一些架构比其他架构好得多。看来飞镖在学习细胞中引起了一些稀疏，反映了这一点。**

**![](img/0ce8180a2951c3ad2bec46d6ab52f33d.png)**

**Galaxy Zoo architecture weights (left) and random search performance (right). Large variance in architectures → sparse cell learned by DARTS.**

**虽然这些结果表明 DARTS 能够学习优于 ResNet 的单元，但是在离散化架构之后，性能严重下降。这突出了一些评审员对原始 DARTS 论文的[评论中提出的观点，特别是:](https://openreview.net/forum?id=S1eYHoC5FX&noteId=r1ekErZ53Q)**

> **从某种意义上说，如何从一次性模型转移到单一模型的最后一步是这项工作最有趣的方面，但也是留下最多问题的一个方面:为什么这样做？是否存在这样的情况，我们通过将解舍入到最接近的离散值而任意地损失惨重，或者性能损失是有限的？从放松到离散选择的其他方式是如何工作的？**

**我们的发现表明，离散化步骤是启发式的，我们表明，它可以在实践中失败。**

# **胸部 x 光**

**为了看看我们在《银河动物园》上的结果是否可以应用到另一个困难的科学数据集上，我们研究了胸部 x 光成像数据集。任务是对来自胸部 X 射线图像的各种疾病(多种疾病可能存在或不存在)进行多标签(二元)分类。**

**![](img/3f3165b9b2f28686acd71664703e0030.png)**

**Example chest X-rays (left) and disease proportions in the training set (right).**

**在这里，我们发现连续飞镖适度优于 ResNet。检查架构权重和随机搜索性能(如下所示)，我们看到一个类似于银河动物园的故事。从随机搜索性能图来看，似乎有些架构的性能比其他架构好得多(再次注意对数标度)。我们看到 DARTS 似乎已经学会了一些体系结构权重的稀疏性，这反映了所有体系结构空间中的一些体系结构比其他体系结构更适合这个任务的概念。**

**![](img/384e638621265e02fd23b950acb2f8f0.png)**

**Chest X-Ray architecture weights (left) and random search performance (right) distributions. Again, large variance in architectures → sparse cell learned by DARTS.**

**我们再次注意到，离散的飞镖模型表现明显不如连续的。**

# **结论**

**总之，我们认为飞镖是一个有用的工具，但对于简单的任务来说，它是多余的。我们表明 ResNet 和 random search 的性能相当好，但是 DARTS 的性能稍好一些。我们注意到 DARTS 引入了许多额外的超参数，必须仔细调整以确保稳定的学习。最重要的是，我们表明 DARTS 的“离散化步骤”在实践中可能会失败，需要更好的技术来确保健壮性。**

**我们对任何考虑飞镖的人的最后建议是:如果性能的小幅度提高对你很重要(例如准确度提高 1%)，飞镖可能是有帮助的，值得你花时间去研究。**

# **未来的工作**

**我们对未来工作的建议一般来自结论。具体来说，我们建议:**

*   **为超参数自动调整和/或设置更好的默认值**
*   **通过以下方式修正离散化试探:a)用 sparsemax 代替 softmax 来鼓励稀疏性，或者对架构权重进行 *Lp* 正则化(用*p*1 ),或者 b)在训练期间动态修剪架构以移除组件，消除重新训练离散化模型的需要。**

**作为最后一点的替代，可以简单地取消离散化步骤(正如我们在这里报告连续结果时所做的)。但是，这将不再是真正的“架构搜索”，而可能被视为一种全新的架构类型(这并没有什么错，但也许有些人对坚持架构搜索的目标感兴趣)。**

# **承认**

**我们要感谢 Pavlos Protopapas(哈佛)、Javier Zazo(哈佛)和 Dogus Cubuk(谷歌)在整个工作过程中给予的支持和指导。**

**有关哈佛数据科学峰会的更多信息，请访问:【capstone.iacs.seas.harvard.edu **

# **代码可用性**

**我们处理各种科学数据集的定制 DARTS 代码可以在:[https://github.com/capstone2019-neuralsearch/darts](https://github.com/capstone2019-neuralsearch/darts)找到**

**运行代码的完整说明可以在这里找到:[https://github . com/capstone 2019-neural search/AC 297 r _ 2019 _ NAS](https://github.com/capstone2019-neuralsearch/AC297r_2019_NAS)**