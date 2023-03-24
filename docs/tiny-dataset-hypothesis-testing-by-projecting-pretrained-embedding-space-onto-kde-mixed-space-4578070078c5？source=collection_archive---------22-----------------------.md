# 将预训练嵌入空间投影到 KDE 混合空间的小数据集假设检验

> 原文：<https://towardsdatascience.com/tiny-dataset-hypothesis-testing-by-projecting-pretrained-embedding-space-onto-kde-mixed-space-4578070078c5?source=collection_archive---------22----------------------->

## 一种主要用于通过面向领域的信息转换来帮助快速原型化、引导主题建模、假设检验、概念验证的方法。

![](img/53e92ad7172fd4b4237dda91e58e0dba.png)

文本分类任务通常需要高样本数和精细的语义可变性来进行可靠的建模。在许多情况下，手头的数据在样本计数、类别的过度偏斜和低可变性(即词汇多样性和语义)方面都是不够的。在这篇文章中，我将介绍一种新颖的方法来克服这些常见的障碍。这种方法的目的主要是帮助快速原型制作、引导主题建模、假设检验、概念验证(POC)，甚至是创建最小可行产品(MVP)。

该方法由以下步骤组成:

1.  加载我们的小数据集或主题词汇表(用于主题建模用例)
2.  选择最合适的预训练嵌入
3.  创建集群
4.  最后，使用核密度估计(KDE)创建新的嵌入

**第一步:加载我们的数据**

我们从一个非常小的数据集开始。我们用的是[施莱歇尔的寓言](https://www.google.com/url?sa=t&rct=j&q=&esrc=s&source=web&cd=1&cad=rja&uact=8&ved=2ahUKEwjfxP6z4rbhAhWMblAKHdcOCyAQFjAAegQIBBAB&url=https%3A%2F%2Fen.wikipedia.org%2Fwiki%2FSchleicher%2527s_fable&usg=AOvVaw3BWgDfgrlXUpN5Z2fqrWn_)，其中的每一句话都是一个文档样本。

**第二步:选择最佳嵌入空间**

单词在本质上看起来是绝对的，但通过 Word2Vec & GloVe 等嵌入方法，它们现在可以被视为有限语义、密集表示的空间中的点。这种伪欧几里得空间表示可以极大地帮助简化语义任务。由于许多领域的数据短缺，通常的做法是从预先训练好的易于使用的嵌入模型开始。

作为一般规则，我们的域应该被完全表示。因此，所选择的嵌入应该包含尽可能多的单词，这些单词将是我们的数据的(主题建模)词汇。为了防止词汇之外(OOV)的单词，所选择的模型应该包含非常大量的记号。我通常选择最低的适应维度空间，因为更高的维度空间可以在来自嵌入空间的单词和我们的领域之间有更大的距离。换句话说，这可能导致聚类边界从我们的域向由预先训练的嵌入所表示的原始域倾斜。最后，我试图选择一个尽可能接近我的用例的嵌入空间，如图 1 所示。

![](img/9cedf955d136b08c4e5d36881dad18bd.png)

Figure.1 — A t-SNE projection of our dataset overlaid on top of the chosen embedding space, sampled for visibility.

我用 Gensim 来训练一个 Word2Vec 模型。我通常在任何数据集上训练一个模型。然而，如下所示，最好是在一个大的外部数据集上训练，最好是与您自己的数据集混合。在这个实现中，空间维度是手动设置的，这给了我们关于聚类边界的优势。

一旦选择了嵌入空间，我们就根据我们的词汇表解析这个空间。以下假设有助于我们完成这项任务:

1.  该空间是伪语义的，即，它是自动选择的，并且不受嵌入空间的真实上下文语义( [Ker{}](https://en.wikipedia.org/wiki/Kernel_(linear_algebra)) )的指导。它确保源数据在语义上尽可能均匀地分布单词距离，这有助于很好地定义聚类边界。
2.  源数据应具有足够低的属性域偏差，以允许多个属性域基于所确定的距离。如前所述，这种假设似乎是一厢情愿的想法。
3.  单词之间的差异由单个半径定义，即，在空间中没有方向依赖性。

下面的代码从预先训练的嵌入空间列表中选择最佳的编码空间，该列表由斯坦福大学提供，可从[这里](https://nlp.stanford.edu/projects/glove/)获得。请注意，以下过程使用标准的文本预处理方法，如文本清理、标点符号和停用词删除，然后是词干处理&词汇化。如下面的代码所示，可以使用 Gensim 包在您选择的任何数据集上创建其他嵌入文件，例如:

**第三步:聚类**

根据前面的假设，在步骤 2 中，我们如何选择正确的聚类算法、聚类的数量以及每个质心在空间中的位置？这些问题的答案在很大程度上依赖于领域。然而，如果您不确定如何添加您的领域的指导性约束或增强，我建议一种通用的、剥离的方法。一般来说，聚类的数量应该通过观察数据集来设置，因为任何转换的语义可分性应该与域本身中的未来任务相关。最小聚类数应该由您在未来任务中预见的最低类数来确定。例如，如果在不久的将来，您看到您的数据或域中的文本分类任务不超过 10 个类，那么最小分类计数应该设置为 10。然而，如果这个数字高于嵌入空间的维数，那么下限应该更大，并且在这一点上是未定义的。在任何情况下，它都不应该超过数据集的词汇或主题数，记住在这个用例中它是非常低的。

像聚类边界不确定性、每个聚类的 P 值分析、自适应阈值和有条件的聚类合并和分割等问题超出了本文的范围。

我们假设嵌入空间中的相邻单词在语义上足够接近，可以加入到某个语义簇中。为了定义聚类，我们需要确定一个距离度量。对于这个任务，让我们看看占据嵌入空间的令牌，并找出最接近的两个。设这两者之间的余弦距离为 Ro，则定义簇字邻接的最小距离为 R = Ro / 2 - ε，此时簇计数最大。换句话说，进行简单的实例到实例距离聚类来对单词进行分组。在主题建模的情况下，Ro 将是来自不同主题的最近单词之间的最小距离。

下面的代码使用选择的手套嵌入空间，使用 K=2 的最近邻对其进行聚类，并使用余弦相似度来确定最小距离。

前一种方法确保聚类将包含来自数据集的至少一个单词，同时记住在嵌入空间中总是存在未分配的单词。

将未分配的点(词)聚集成生成的簇的直接方法是[标签传播/标签传播](https://en.wikipedia.org/wiki/Label_Propagation_Algorithm)，如图 2 & 3 所示。

然而，由于较高的运行时间复杂性(代码#5)，您可能希望使用更快且不太精确的方法，如线性 SVM(代码#6)。由于运行时复杂性问题，下面的代码比较了这两种方法。这一步是一个“蛮力”聚集，在未来的探索中，当我们的数据集预计会更丰富时，可能会产生不太理想的结果。

![](img/1f58a0ebfa0b3af4495142db6c2a8e09.png)

Figure.2 — A t-SNE projection after label-spreading of our dataset and a selection of samples from our chosen embedding space. please note that this is purely for illustrative purposes, as the real 2D display of the labeling would be similar to Figure 3.

![](img/7216c3832404f1dbabc4d377e1daab07.png)

Figure 3: A t-SNE projection after label-spreading, using a sample of tokens from our embedding space, colors represent the different labels. Please note that this is in a higher space compared to Figure 2 without dimensionality reduction.

让我们来关注一下为什么核心聚集之后是样本聚集是有意义的**。**嗯，我们希望将嵌入限制在我们的数据锚/主题(单词)上。这需要语义上的接近。一旦实现了这一点，假设最外围的单词在未来的样本中不太可能出现。让我再强调一次——当我们从**获得非常少的**数据开始，并且想要制作一个概念验证或基本产品(MVP)时，就会出现这种用例。

**步骤 4:使用 KDE 创建新的嵌入空间**

现在，所有的单词都被分配到一个聚类中，我们需要一个更有信息的表示，这将有助于未来的未知样本。由于嵌入空间是由语义接近度定义的，我们可以通过空间中该位置处每个聚类的概率密度函数(PDF)的密度来编码每个样本。

换句话说，在一个簇中位于密集区域而在另一个簇中密度较低的单词，将使用通过使用新的**密度嵌入**所投射的信息来展示这种行为。请记住，嵌入维数实际上是聚类计数，并且嵌入的顺序是相对于初始化该嵌入时提出的聚类来保持的。使用我们的小数据集，得到的投影可以在图 4 中看到。最后，下面的代码使用 KDE 创建一个新的嵌入。

![](img/c21c846591805f92ae75280599ebdfbd.png)

Figure.4 — A t-SNE projection of the final density encoding map, which is a mixture model. label colors may have changed but they correspond to the label clusters as seen in Figure 2.

感谢 [**Ori Cohen**](https://medium.com/@cohenori) 和**[**Adam Bali**](https://medium.com/@adambali1)的宝贵批判、校对、编辑和评论。**

**Natanel Davidovits
奇异问题解决者。数学建模、优化、计算机视觉、NLP/NLU &数据科学方面的专家，拥有十年的行业研究经验。**