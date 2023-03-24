# 字节对编码——现代自然语言处理的黑马

> 原文：<https://towardsdatascience.com/byte-pair-encoding-the-dark-horse-of-modern-nlp-eb36c7df4f10?source=collection_archive---------3----------------------->

## 1994 年首次推出的一种简单的数据压缩算法为当今几乎所有先进的 NLP 模型(包括 BERT)提供了增压。

# 背景

在 NLP 领域，过去几年是一段激动人心的时光。从稀疏的基于频率的词向量到密集的语义词表示预训练模型的演变，如 [Word2vec](https://en.wikipedia.org/wiki/Word2vec) 和 [GloVe](https://nlp.stanford.edu/projects/glove/) 为学习词义奠定了基础。多年来，它们作为可靠的嵌入层初始化，在缺乏大量特定任务数据的情况下训练模型。由于在维基百科上预先训练的单词嵌入模型要么受到词汇量的限制，要么受到单词出现频率的限制，像`athazagoraphobia`这样的罕见单词将永远不会被捕获，从而导致在文本中出现未知的`<unk>`标记。

# 处理生僻字

抛开字符级嵌入不谈，爱丁堡大学的研究人员通过使用字节对编码(BPE)在神经机器翻译中应用[子词单元](https://www.aclweb.org/anthology/P16-1162.pdf)，在解决生僻字问题上取得了第一次真正的突破。今天，受 BPE 启发的子词标记化方案已经成为大多数高级模型的规范，包括非常流行的上下文语言模型家族，如 [BERT](https://github.com/google-research/bert) 、 [GPT-2](https://openai.com/blog/better-language-models/) 、[罗伯塔](https://arxiv.org/abs/1907.11692)等。有些人把伯特称为新时代的开端，然而，我认为 BPE 是这场竞赛中的一匹黑马，因为它在现代 NLP 模型的成功中没有得到应有的关注。在本文中，我计划更详细地介绍字节对编码是如何实现的，以及它为什么工作！

# 字节对编码的起源

像许多其他受传统科学启发的深度学习应用一样，字节对编码(BPE)子词标记化也深深植根于一种简单的无损数据压缩算法中。Philip Gage 在 1994 年 2 月版的 C Users Journal 的文章“[A New Algorithm for Data Compression](https://www.drdobbs.com/a-new-algorithm-for-data-compression/184402829)”中首次介绍了 BPE，这是一种数据压缩技术，它通过用一个不在数据中出现的字节替换常见的连续字节对来工作。

![](img/6d242feee5f5d39842472fd8d9ada3c0.png)

# 将 BPE 重新用于子词标记化

为了执行子词标记化，BPE 在其实现中被稍微修改，使得频繁出现的子词对被合并在一起，而不是被另一个字节替换以实现压缩。这将基本上导致罕见的单词`athazagoraphobia`被分成更频繁的子单词，例如`['▁ath', 'az', 'agor', 'aphobia'].`

**第 0 步。**初始化词汇。

**第一步。**将语料库中的每个单词表示为字符的组合以及特殊的单词结束标记`</w>`。

**第二步。**迭代统计词汇表所有记号中的字符对。

**第三步。**合并最频繁出现的对，将新字符 n-gram 添加到词汇表中。

**第四步。**重复步骤 3，直到完成期望数量的合并操作或达到期望的词汇大小(这是一个超参数)。

# 是什么让 BPE 成为秘制酱？

BPE 带来了字符和单词级混合表示之间的完美平衡，这使得它能够管理大型语料库。这种行为还支持使用适当的子词标记对词汇表中的任何罕见单词进行编码，而不会引入任何“未知”标记。这尤其适用于像德语这样的外语，在德语中，许多复合词的存在会使学习丰富的词汇变得困难。有了这种记号化算法，现在每个单词都可以克服他们对被遗忘的恐惧(athazagoraphobia)。

# 参考

1.  菲利普·盖奇，*一种新的数据压缩算法*。[《多布斯博士杂志》](http://www.drdobbs.com/article/print?articleId=184402829&dept_url=/)
2.  森里奇(r .)、哈多(b .)、伯奇(a .)(2015 年)。具有子词单元的生僻字的神经机器翻译。 *arXiv 预印本 arXiv:1508.07909* 。
3.  Devlin，j .，Chang，M. W .，Lee，k .，& Toutanova，K. (2018 年)。Bert:用于语言理解的深度双向转换器的预训练。 *arXiv 预印本 arXiv:1810.04805* 。
4.  刘，y .，奥特，m .，戈亚尔，n .，杜，j .，乔希，m .，陈，d，…和斯托扬诺夫，V. (2019)。Roberta:稳健优化的 bert 预训练方法。 *arXiv 预印本 arXiv:1907.11692* 。