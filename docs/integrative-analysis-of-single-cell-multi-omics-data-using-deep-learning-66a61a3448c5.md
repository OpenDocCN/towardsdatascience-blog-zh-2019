# 使用深度学习对单细胞多组学数据进行综合分析(带视频教程)

> 原文：<https://towardsdatascience.com/integrative-analysis-of-single-cell-multi-omics-data-using-deep-learning-66a61a3448c5?source=collection_archive---------16----------------------->

视频教程#1:

Tutorial on Saturn Cloud

视频教程#2:

单细胞 RNA 测序(scRNA-seq)提供了一种全面、公正的方法，利用下一代测序技术，以单细胞分辨率分析包括 T 细胞在内的免疫细胞。最近，通过测序( [CITE-seq](https://cite-seq.com/) )对转录组和表位进行细胞索引等令人兴奋的技术已经被开发出来，通过联合测量多个分子模式(如来自同一细胞的蛋白质组和转录组)来扩展 scRNA-seq。通过利用与寡核苷酸结合的抗体，CITE-seq 同时产生基于测序的表面蛋白表达和基因表达的读数。

由于基因和蛋白质表达传达了关于细胞的独特和互补的信息，CITE-seq 提供了一个独特的机会，将转录组和蛋白质组数据结合起来，以比单独使用其中一种高得多的分辨率破译单个细胞的生物学。这需要能够有效整合来自两种模态的单细胞数据的计算方法。在这篇文章中，我们将使用无监督的深度学习，更具体地说，autoencoder，对 CITE-seq 数据进行综合分析。

# 数据

我们将使用 2017 年由 [Stoeckius M *等人*](https://www.nature.com/articles/nmeth.4380) 发表的首个 CITE-seq 数据集。作者测量了大约 8000 个脐带血单核细胞(CBMCs)的单细胞转录组以及 13 种蛋白质的表达。有两个 CSV 文件——一个用于基因表达，另一个用于蛋白质表达，可以从[这里](https://www.dropbox.com/sh/sm8vqmmv1d6cmst/AAC6aQoPtlReMSKmITnkxNiQa?dl=0)下载。 [Seurat](https://satijalab.org/seurat/) 是一个 R 包，设计用于单细胞基因组数据的质量控制、分析和探索。在将数据输入到我们的自动编码器之前，我们将首先对数据进行预处理，并使用 Seurat 软件包基于基因表达进行细胞聚类和注释。代码可在[本](https://github.com/naity/citeseq_autoencoder/blob/master/clustering.ipynb)随附笔记本中找到。

预处理后，基因和蛋白质表达数据被连接在一起，其中每一列是基因或蛋白质，而每一行是细胞(每个细胞都有一个唯一的条形码)。该数据集包含总共 7895 个细胞的 2000 个基因和 10 种蛋白质的表达水平(3 种由于富集差而被去除)。

Concatenated gene and protein expression data

# 自动编码器

Autoencoder 是一种无监督的深度学习模型或神经网络，由三个主要组件组成:编码器、瓶颈和解码器，如下图所示。编码器压缩输入，瓶颈层存储输入的压缩表示。相反，解码器试图根据压缩数据重建输入。

![](img/52221506ec9a77a07f8456470bfd995c.png)

Image source: [https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd](https://medium.com/@curiousily/credit-card-fraud-detection-using-autoencoders-in-keras-tensorflow-for-hackers-part-vii-20e0c85301bd)

瓶颈层的尺寸通常大大低于输入的尺寸。因此，编码器将尝试了解尽可能多的关于输入的有意义的信息，同时忽略噪声，以便解码器可以更好地重建输入。Autoencoder 可以用作降维算法，存储在瓶颈层中的输入的低维表示可以用于数据可视化和其他目的。此外，由于其灵活的神经网络架构，它提供了无限的方法来将基因和蛋白质表达数据整合到 autoencoder 中，正如我们将在下面看到的。

# 履行

由于基因和蛋白质数据具有显著不同的维度，我们将首先使用两个不同的编码器分别对它们进行编码，然后将输出连接起来，这些输出将通过另一个编码器来生成瓶颈层。随后，解码器将尝试基于瓶颈层重建输入。总体神经网络架构如下所示:

![](img/316787a5650f555eebdc7e5c7731056c.png)

Autoencoder architecture

使用 Pytorch 和 fastai 库，自动编码器的实现非常简单。在这个例子中，压缩的基因和蛋白质数据的维数分别是 120 和 8，瓶颈层由 64 个神经元组成。完整的代码可以在[这个](https://github.com/naity/citeseq_autoencoder/blob/master/autoencoder_gene_only.ipynb)(仅限基因)和[这个](https://github.com/naity/citeseq_autoencoder/blob/master/autoencoder_gene_protein.ipynb)(基因和蛋白质)的随机笔记本中找到。

Autoencoder implementation

# 结果

除了根据基因和蛋白质表达数据训练的模型之外，我们还仅使用基因表达数据训练对照模型。通过这种方式，我们将能够辨别对 CITE-seq 数据进行整合分析的任何优势。在训练模型后，我们提取存储在瓶颈层的原始输入的 64 维压缩表示，随后在由 [t-SNE](https://en.wikipedia.org/wiki/T-distributed_stochastic_neighbor_embedding) 生成的二维地图上可视化。

如下图所示，转录组和蛋白质组数据的综合分析产生了比单独使用基因表达数据更好的结果。例如，当仅使用基因表达数据时，CD8 T 细胞(红色箭头)和 CD4 T 细胞簇一起形成了一个大的“岛”，并且不能被分开。相比之下，当将蛋白质表达数据与基因表达数据结合时，CD8 T 细胞(红色箭头)形成与 CD4 T 细胞群完全分离的独特群。因此，组合分析在分析 CITE-seq 数据时比使用单一分子模式的数据更有效。

![](img/fcb703619800032d6291f830828cdb5c.png)

Visualization based on gene expression data only. The red arrow points at CD8 T cells.

![](img/d497395af505a8c26a8c662fb1f9a089.png)

Visualization based on both gene and protein expression data only. The red arrow points at CD8 T cells.

# 总结和未来方向

在这里，我们建立了一个基于自动编码器的深度学习模型，用于单细胞 CITE-seq 数据的降维和可视化。我们证明转录组和蛋白质组数据的综合分析在区分各种免疫细胞类型方面获得了更高的分辨率。

一个限制是在预处理步骤中仅使用基因表达数据将细胞分配给不同的免疫细胞类型。开发一个深度学习模型将是非常有趣的，该模型还可以使用来自多个分子模态的数据来执行细胞聚类和表征。

# 源代码

数据:
[https://www . Dropbox . com/sh/opxrude 3s 994 lk 8/aaaiwrzfviksxkpyomlwqhea？dl=0](https://www.dropbox.com/sh/sm8vqmmv1d6cmst/AAC6aQoPtlReMSKmITnkxNiQa?dl=0)

朱庇特笔记本:
【https://github.com/naity/citeseq_autoencoder】T4

我是一名具有生物信息学和编程技能的免疫学家。我对数据分析、机器学习和深度学习感兴趣。

网址: [www.ytian.me](http://www.ytian.me)
博客:[https://medium.com/@yuan_tian](https://medium.com/@yuan_tian)
领英:[https://www.linkedin.com/in/ytianimmune/](https://www.linkedin.com/in/ytianimmune/)
推特:[https://twitter.com/_ytian_](https://twitter.com/_ytian_)