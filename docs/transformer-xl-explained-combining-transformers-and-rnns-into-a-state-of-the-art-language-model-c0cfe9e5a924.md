# Transformer-XL 解释:将 Transformer 和 RNNs 结合成一个最先进的语言模型

> 原文：<https://towardsdatascience.com/transformer-xl-explained-combining-transformers-and-rnns-into-a-state-of-the-art-language-model-c0cfe9e5a924?source=collection_archive---------4----------------------->

## “Transformer-XL:固定长度上下文之外的注意力语言模型”摘要

语言建模已经成为一种重要的 NLP 技术，这要归功于它能够应用于各种 NLP 任务，例如机器翻译和主题分类。今天，有两种主要的语言建模架构——递归神经网络(RNNs)和转换器。前者逐个处理输入的单词或字符，以了解它们之间的关系，而后者接收一段标记，并使用注意机制立即了解它们之间的依赖关系。

尽管这两种体系结构都取得了令人印象深刻的成就，但它们的主要限制是捕获长期依赖，例如，使用文档开头的重要单词来预测后续部分中的单词。谷歌和卡耐基梅隆大学的一篇新论文“Transformer-XL :固定长度上下文之外的专注语言模型”结合了这两种方法。新模型在输入数据的每个片段上使用转换器的注意力模块，并使用递归机制来学习连续片段之间的依赖性。

Transformer-XL 在多种语言建模数据集上实现了最先进的(SOTA)结果，如 enwik8(单词级)和 text8(字符级)，同时在推理过程中明显快于之前的 SOTA Transformer 架构(300x-1800x)。

# **背景——变形金刚**

一种流行的语言建模方法是递归神经网络(RNNs)，因为它们可以很好地捕捉单词之间的依赖关系，特别是在使用像 [LSTM](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) 这样的模块时。然而，由于[消失梯度](https://en.wikipedia.org/wiki/Vanishing_gradient_problem)，rnn 往往很慢，并且它们学习长期依赖性的能力仍然有限。

[2017 年发明的变形金刚](https://arxiv.org/abs/1706.03762)，引入了一种新方法——注意力模块。注意力模块不是一个接一个地处理令牌，而是接收一段令牌，并使用三个学习到的权重矩阵(查询、关键字和值)一次性学习所有令牌之间的依赖关系，这三个权重矩阵形成了注意力头。Transformer 网络由多个层组成，每个层都有几个注意力头(和附加层)，用于学习令牌之间的不同关系。

与许多 NLP 模型一样，输入标记首先被嵌入到向量中。由于注意力模块中的并发处理，该模型还需要添加关于标记顺序的信息，这一步骤称为位置编码，有助于网络了解它们的位置。一般来说，这个步骤是用正弦函数完成的，该函数根据令牌的位置生成一个矢量，而不需要任何学习参数。

![](img/6c9cb1235e643d59c5e22ab0de47c0f0.png)

An example of a single Attention Head on a single token (E1). Its output is calculated using its Query vector, and the Key and Value vectors of all tokens (In the chart we show only one additional token E2) — The Query and the Key define the weight of each token, and the output is the weighted sum of all Value vectors.

注意:关于变形金刚的深入评论可以在 Jay Alammar 伟大的[博客文章](http://jalammar.github.io/illustrated-transformer/)中找到。

虽然最初的转换器用于机器翻译(带有编码器-解码器机制)， [Al-Rfou 等人](https://arxiv.org/abs/1808.04444)提出了一种语言建模的架构。它的目标是根据其先前的字符预测一个段中的字符，例如，它使用 *X1* … *Xn-1* 预测字符 *Xn* ，而右边的下一个字符被屏蔽(见下图)。这种 64 层的变压器模型限于 512 个字符的相对较短的输入，因此它将输入分割成段，并分别从每个段中学习。为了在评估中处理较长的输入，它通过在每一步中将输入移动一位来一次预测一个字符。

![](img/2e8c1de69fae17002991673cc9740845.png)

Training and Evaluation of the vanilla Transformer language model. Source: [Transformer-XL](https://arxiv.org/abs/1706.03762)

在流行的基准测试(enwik8 和 text8)中，该模型优于 RNN 模型，但是，它仍然有两个缺点:

1.  有限的上下文相关性-字符之间的最大相关性距离受限于输入的长度。例如，模型不能“使用”几个句子前出现的一个单词。
2.  上下文分段—对于长度超过 512 个字符的文本，该长度的每个片段都是从头开始单独训练的。因此，对于每个片段的第一个标记以及片段之间，根本不存在上下文(依赖性)。这导致低效的训练，并可能影响模型性能。

# **Transformer-XL 的工作原理**

Transformer-XL 在很大程度上依赖于 vanilla Transformer (Al-Rfou 等人)，但引入了两种创新技术——**递归机制**和**相对位置编码**——来克服 vanilla 的缺点。相对于普通转换器的另一个优点是，它可以用于单词级和字符级语言建模。

## 重现机制

重现机制的目标是通过使用来自先前分段的信息来实现长期依赖性。与普通版本类似，Transformer-XL 处理第一段标记，但保留隐藏层的输出。当处理下面的段时，每个隐藏层接收两个输入:

1.  该段的前一个隐藏层的输出，如普通版本(下图中的灰色箭头)。
2.  前一段(绿色箭头)中的前一个隐藏层的输出，允许模型创建长期依赖关系。

从技术上讲，这两个输入被连接起来，然后用于计算当前段(的当前层的当前头)的键和值矩阵。这种添加为网络提供了关于每个令牌的权重(重要性)的更多信息，但是它没有改变值矩阵。

![](img/f5b7881adaec11546570f888afff71db.png)

Training and Evaluation of the Transformer-XL language model. Source: [Transformer-XL](https://arxiv.org/abs/1706.03762)

通过以相同的方式(在 GPU 存储器的限制下)使用来自几个先前段的信息，甚至仅在评估期间，可以扩展该概念以包含更长的依赖性。

递归机制的另一个优点是其评估速度——在每一步中，它可以前进整个段(而不是像普通版本中那样前进一个令牌),并使用先前段的数据来预测当前段令牌。

相对位置编码

递归机制也带来了新的挑战—原始的位置编码分别处理每个段，因此，来自不同段的令牌具有相同的位置编码。例如，第一段和第二段的第一个标记将具有相同的编码，尽管它们的位置和重要性不同(来自第一段的标记可能较低)。这种混淆可能会错误地影响网络。

相反，本文提出了一种新的位置编码，它是每个注意模块的一部分，而不是仅在第一层之前对位置进行编码，并且基于标记之间的相对距离而不是它们的绝对位置。从技术上来说，它扩展了注意力头部得分(Qi **⋅** Kj)的简单相乘，以包括四个部分:

1.  内容权重-原始分数，当然没有添加原始位置编码。
2.  相对于当前内容的位置偏差(Qi)。它使用类似的正弦函数来接收记号之间的距离(例如 *i-j* ，而不是当前记号的绝对位置。
3.  学习的全局内容偏差-该模型添加了调整其他令牌内容(Kj)的重要性的学习向量。
4.  学习的全局偏差-另一个学习的向量，其仅基于记号之间的距离来调整重要性(例如，最后的先前单词可能比先前段落中的单词更重要)。

# **结果**

作者比较了该模型在单词级和字符级数据集上的性能，并将其与其他著名模型(RNNs 和 Transformers)进行了比较。Transformer-XL 在几个不同的数据集基准上取得了最先进的(SOTA)结果:

1.  在大型词级数据集 [WikiText-103](https://www.salesforce.com/products/einstein/ai-research/the-wikitext-dependency-language-modeling-dataset/) 上，18 层 Transformer-XL (257M 参数)达到了 18.3 的[困惑度](https://en.wikipedia.org/wiki/Perplexity)，而之前的 SOTA[Baevski&Auli](https://arxiv.org/abs/1809.10853)达到了 20.5。
2.  在字符级数据集 [enwik8](http://mattmahoney.net/dc/textdata.html) 上，12 层 Transformer-XL 达到了每字符 1.06 位(bpc)，与之前由 [Al-Rfou 等人](https://arxiv.org/abs/1808.04444)使用六倍多参数的 SOTA 结果相似。24 层 Transformer-XL 实现了 0.99 bpc 的新 SOTA。
3.  有趣的是，该模型还在只有短期依存关系的数据集上实现了 SOTA 结果-[10 亿个单词](http://www.statmt.org/lm-benchmark/)只有单个句子-以及在小数据集- [Penn Treebank](https://catalog.ldc.upenn.edu/LDC99T42) 只有 100 万个令牌。这表明该模型在这些情况下也可能是有效的。

在下面的图表中可以看到递归机制和相对位置编码的好处。它比较了不同上下文长度(注意头中使用的先前标记的数量)的无重现或新编码的困惑分数。完整的 Transformer-XL 明显优于其他产品，并且能够利用更长期的依赖性。此外，它还能够捕获比 RNN 更长的依赖关系(长 80%)。

![](img/71aff6dcf4bd77fb5d138d3388a93028.png)

Transformer-XL ablation study. Source: [Transformer-XL](https://arxiv.org/abs/1706.03762)

最后，如前所述，该模型在推理过程中也比普通的转换器快得多，尤其是对于较长的上下文。例如，对于长度为 800 个字符的上下文，速度提高了 363 倍，对于长度为 3，800 个字符的上下文，速度提高了 1，874 倍。

# **实施细节**

该模型是开源的，并且在 TensorFlow 和 PyTorch 中均[实现](https://github.com/kimiyoung/transformer-xl/)(包括预训练的模型)。没有指定每个数据集的训练持续时间。

# **结论**

Transformer-XL 展示了几种不同数据集(大/小、字符/单词等)上语言建模的最新结果。它结合了深度学习的两个突出概念——重现和注意力——允许该模型学习长期依赖关系，并可能对需要这种能力的深度学习的其他领域有效，如音频分析(例如每秒 16k 样本的语音数据)。

该模型尚未在情感分析或问题回答等 NLP 任务上进行测试，与其他基于 Transformer 的模型(如 [BERT](https://www.lyrn.ai/2018/11/07/explained-bert-state-of-the-art-language-model-for-nlp/) )相比，这种强语言模型的优势是什么仍然是一个开放的问题。

*关注深度学习最新研究，订阅我的简讯* [*LyrnAI*](https://www.lyrn.ai)