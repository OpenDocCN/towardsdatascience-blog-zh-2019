# 相对与绝对:用模拟理解成分数据

> 原文：<https://towardsdatascience.com/relative-vs-absolute-understanding-compositional-data-with-simulations-fdc15e0c781e?source=collection_archive---------17----------------------->

我在工作中经常被要求做的一件事是比较不同样本的 ngs 数据。这些可能是复制，也可能来自完全不同的环境或处理步骤。我们都直觉地理解这些数据总是受到测序深度的限制，因此需要仔细地得出结论。直到我看到 [**@** WarrenMcG](https://twitter.com/WarrenMcG) 的[这篇论文](https://www.biorxiv.org/content/10.1101/564955v1)的预印本，我才缺乏分析这些数据的词汇或技术。这开始了几天对成分数据的深入挖掘，我生成了一些合成数据，以便更好地理解其中的细微差别。在这篇文章中，我试图展示我所学到的东西，并希望有助于传播将组成数据视为常规无约束数据的危险。

# 什么是成分数据

成分数据是指数字构成整体比例的任何数据。例如，你将一枚硬币抛 10 次，得到的正面和反面的数量就是成分数据。这些数据的主要特征是它们是“封闭的”或受限的，如果我告诉你我们通过投掷双面硬币得到了 6 个正面，你就知道有 4 个反面，而我没有给你提供任何进一步的信息。从概率的角度来说，这意味着数据中的所有成分并不是相互独立的。因此，组成数据不存在于欧几里得空间中，而是存在于一个称为单形的特殊约束空间中。对于那里的工程师来说，每次绘制多相平衡图时，你可能会遇到单纯形图，就像这里使用 [R 库三元](https://cran.r-project.org/web/packages/Ternary/vignettes/Using-Ternary.html)绘制的水的多相平衡图一样:

![](img/2bbf174164992fc9fc84bfa689b54975.png)

本质上，一个特定的固定量的水可以在不同的条件下存在于不同的相中，但是蒸汽、液态水和冰的总量应该总是等于我们开始时的固定量。成分数据存在于比我们想象的更多的地方。一些例子包括投票数据，食品标签上的营养信息，你呼吸的空气中不同气体的比例，甚至你屏幕上像素的 RGB 值。

# 将成分数据视为无约束数据的问题

根据 [Aitchison](https://www.jstor.org/stable/2345821?seq=1#page_scan_tab_contents) ，一般来说，任何成分数据分析方法都应满足 4 个不同的标准。传统的分析通常不能满足所有这些要求，因此会导致错误的结果。我将用模拟数据来说服我自己，希望你能理解传统分析方法的这些缺陷。我使用[absimseq](https://github.com/warrenmcg/absSimSeq)生成一些虚拟的 [RNA-seq](https://en.wikipedia.org/wiki/RNA-Seq) 之类的数据。简而言之，我有 4 个来自控制条件的样本和 4 个来自处理条件的样本。每个样本由 100 个天然基因和 92 个刺突(对照)基因组成，样本中所有基因的计数总计约 20K 个读数。与对照条件相比，模拟处理条件具有 20%的差异表达的天然基因(用于描述细胞/组织内基因拷贝数变化的奇特生物学术语)。差异表达的确切大小与我们在此试图证明的无关。既然我们已经建立了数据，让我们继续讨论传统分析方法在处理成分数据时通常无法考虑的 4 个标准。

你可以在这里找到我使用的所有数据和代码:[https://github.com/kimoyerr/CODA.git](https://github.com/kimoyerr/CODA.git)

## 比例不变性

组成数据的一个共同属性是它们可以用不同大小的向量来表示。空气由 78%的氮气、21%的氧气和 1%的剩余体积组成。在向量符号中，这可以写成[78，21，1]。就体积的百万分率(ppmv)而言，这也可以指定为[780840、209460 和 9780]。尽管这些符号非常不同，但它们代表了相同的物理样本。然而，传统的分析方法，如欧几里德距离和层次聚类，会被这种表示上的差异所迷惑。让我们用模拟的 RNA-Seq 数据来证明这一点。在 R 中，执行以下操作:

```
load('sim_counts_matrix_100.rda')
orig.dist <- dist(t(counts_matrix))orig.dendro <- as.dendrogram(hclust(d = dist(t(counts_matrix))))

# Create dendro
dendro.plot <- ggdendrogram(data = orig.dendro, rotate = TRUE)# Preview the plot
print(dendro.plot)
```

![](img/d854fb9cd61fb802fae46601ae3de3f3.png)

Clustering of original data

正如所料，我们看到四个对照样品(样品 1 至 4)聚集在一起，四个处理样品(样品 5 至 8)聚集在一起。

现在让我们缩放一些不同的样本，如下所示。例如，如果您对一些样品进行了比平时更高深度的测序，但仍想一起分析所有样品，就会发生这种情况。

```
# Scale the samples differently
scaled.counts_matrix = counts_matrix %*% diag(c(1,1,5,5,1,1,5,5))
scaled.dist <- dist(t(scaled.counts_matrix))scaled.dendro <- as.dendrogram(hclust(d = dist(t(scaled.counts_matrix))))
# Create dendro
scaled.dendro.plot <- ggdendrogram(data = scaled.dendro, rotate = TRUE)# Preview the plot
print(scaled.dendro.plot)
```

![](img/c0c4067bc975209811a7545a23689e8a.png)

Clustering of scaled data

新的树状图现在显示了聚集在一起的缩放样本(样本 3、4、7 和 8 ),这是不正确的，因为唯一改变的是这些样本的读取总数，但它们的单个组件相对于彼此是相同的。

## 扰动不变性

这与我们的分析结论不应依赖于用来表示成分数据的单位的观察结果有关。回到空气成分的例子，我们可以混合使用百分比和 ppmv 值来表示数据，比如:[78%、21%和 9780ppmv]。这仍然代表着相同的物理样本，我们应该能够将它与另一个类似空气的样本进行比较，在这个新的坐标系和旧的坐标系中，并得出相同的结论。让我们再次使用模拟数据来计算新的扰动坐标空间中样本之间的距离:

```
#Multiply each row (gene) with a scalar value
perturbed.counts_matrix = counts_matrix * c(seq(1,192,1))
colnames(perturbed.counts_matrix) = colnames(counts_matrix)
perturbed.dist <- dist(t(perturbed.counts_matrix))perturbed.dendro <- as.dendrogram(hclust(d = dist(t(perturbed.counts_matrix))))
# Create dendro
perturbed.dendro.plot <- ggdendrogram(data = perturbed.dendro, rotate = TRUE)# Preview the plot
print(perturbed.dendro.plot)
```

![](img/dfdc5d7a8e0dc75fc98949fe6be29c82.png)

Clustering of perturbed data

从上面的图可以清楚地看出，与比例不变性问题相比，这是一个小得多的问题。来自对照和处理条件的样本聚集在一起，但是样本之间的距离不同。这可能会导致错误的结论，具体取决于所提的问题。在大多数 RNA-seq 或 NGS 数据中，在不同条件下比较样品时，扰动方差不是大问题。

## 亚位置相干性

这一特性确保了无论包含哪些数据成分，分析都会得出相同的结论。例如，在我们的空气示例中，我们可以删除代表 1%体积的组件。现在样本在 2 维空间([78.5%，21.5%])中表示，而不是在原始的 3 维空间([78%，21%，1%])。在这个低维空间中的任何分析都不应不同于在原始三维空间中的比较。从下一代测序(NGS)数据的角度来看，这是一个重要的属性，在这种情况下，我们并不总是能够保证始终找到数据中的所有成分。这可能有很多原因，有时是有意的，有时是无意的，不可避免的。

为了在传统分析方法中模拟这种特性(或缺乏这种特性)，我从由 100 个基因+ 92 个对照组成的原始数据集生成了由前 50 个基因+ 92 个对照组成的第二个数据集。较小数据集中的 50 个基因具有与原始数据集中相同的绝对(无约束)值。然后，通过使所有读取的总和为大约 20K(如在原始数据集中一样),这些不受约束的数据被封闭(受约束)。

计算两个数据集中 50 个常见基因之间的相关系数，然后进行比较:

```
load('sim_counts_matrix_100.rda')
counts.all <- counts_matrix
 # Load the sub-compositional data made up of only the first 50 genes (features) + 92 controls from the original data of 100 genes (features) + 92 controls
load('sim_counts_matrix_50.rda')
counts.sub.comp <- counts_matrix# Get the correlation between the 50 common genes
cor.all <- as.vector(cor(t(counts.all[1:50,])))
cor.sub.comp <- as.vector(cor(t(counts.sub.comp[1:50,])))
tmp <- as.data.frame(cbind(cor.all,cor.sub.comp))
names(tmp) <- c('correlation_all', 'correlation_sub_comp')
tmp$abs.diff <- as.factor(ifelse(abs(tmp$correlation_all - tmp$correlation_sub_comp)>0.5,1,0))# Plot
ggplot(tmp,aes(correlation_all,correlation_sub_comp, color=abs.diff)) + geom_point(size=2) + th + scale_colour_manual(values = c("1" = "Red", "0" = "Blue")) + theme(legend.position = "none")
```

![](img/007e57362ba7dd25e8af63234faa665a.png)

Differences in the correlation between genes (features) depending on what other genes are present in the data

在上面的图中，我们可以看到两个数据集中的大多数相关系数是相似的，有些系数有显著差异(以红色显示)。根据所提问题的不同，这些差异可能会导致结论的显著差异。这可能是基因网络分析中的一个主要问题。

## 置换不变性

这一特性意味着不管原始数据的顺序如何，分析方法都应该得出相同的结论。我能想到的大多数分析方法一般都遵循这个特性。请让我知道，如果你知道任何违反这一点的分析方法，我会在帖子中包括它们。

# 如何正确分析成分数据

现在，我们确信传统的分析方法并不总是满足分析成分数据的所有 4 个重要属性，我们能做什么？三十多年来，Aitchison 做了一些令人难以置信的工作，想出了更好的技术。根据我处理 NGS 数据的经验，我感觉他的建议在很大程度上被当成了耳旁风。希望这篇博文能够说服其他人在分析成分数据时更加谨慎。在下一篇文章中，我将展示一些分析成分数据的方法，再次强调 NGS 数据和应用。

![](img/97520dd08772aaeceab1ed84b25a55d6.png)

[https://quotecites.com/quote/Hermann_Hesse_378](https://quotecites.com/quote/Hermann_Hesse_378)