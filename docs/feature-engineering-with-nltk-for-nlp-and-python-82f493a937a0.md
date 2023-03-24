# 面向 NLP 和 Python 的 NLTK 特征工程

> 原文：<https://towardsdatascience.com/feature-engineering-with-nltk-for-nlp-and-python-82f493a937a0?source=collection_archive---------22----------------------->

![](img/5d2cf678c3a51d94661c77c0010c9e32.png)

Herman Melville (1860), ‘*Journal Up the Straits,’* [*Library of Congress*](https://www.loc.gov/rr/print/list/235_pom.html)

上周，我复习了自然语言处理(NLP)的自然语言工具包(NLTK)的一些基本功能[。通过将这些基本函数应用到赫尔曼·梅尔维尔的《莫比·迪克》中，我继续了我的 NLP 之旅。文本文档由 Gutenberg 项目提供，该网站上的几本书可以通过 python NLTK 包获得。](/nltk-applications-for-nlp-and-python-dc8c5381668a)

我在之前的博客中详细描述了清理过程，我不得不清理两部电视剧的各种脚本。这一过程将根据手头的任务而变化。今天我只是用 NLTK 方法探索数据的文本。一般流程是:

1.  加载语料库或文档的文本
2.  对原始文本进行标记
3.  删除停用词和标点符号
4.  应用词干或词汇化

古腾堡项目的文本莫比迪克的解释已经相当干净了，也就是说，只有标点符号需要从文本中删除，所有的单词都需要小写。我还删除了序言和序言，因为它不是梅尔维尔的原始故事的一部分。

我在文件中保留的唯一标点符号是撇号。NLTK 允许您定制 regex 模式，将字符视为一个标记。例如，任何带撇号的单词都被视为一个标记，*即* d'ye 是一个标记，而不是两个单独的标记。应用正则表达式模式的代码是:`nltk.regexp_tokenize(raw_text, pattern)`，其中`raw_text`是代表文档的字符串，`pattern`是代表您希望应用的正则表达式模式的字符串。

# 探索文本

一个*单词包*是一个单词在文本中出现的次数。这个计数可以是文档范围的、语料库范围的或语料库范围的。

![](img/b2698b24a3ccd79e289c47d44b4ce8f3.png)

A visualization of the text data hierarchy

NLTK 提供了一个简单的方法，可以创建一个单词包，而不必手动编写代码来遍历一个标记列表。首先创建一个`FreqDist`对象，然后应用令牌列表。最后，您可以使用`.most_common()`方法查看最常见的令牌。

要查看删除停用词后文本中唯一单词的总数，只需查看频率分布列表的长度:`len(moby_dick_freqdist)`。每个令牌的计数都是名义上的，如果您只研究一个文档，这是很好的。如果你有几个文档(一个语料库),那么标准化计数对于比较不同长度的文档是必要的。这可以通过对频率分布列表中每个元组的值求和，然后将每个计数除以该总数来实现。

How to make a normalized frequency distribution object with NLTK

# 二元模型、n 元模型和 PMI 得分

每个标记(在上面的例子中，每个唯一的单词)代表文档中的一个维度。在停用词被移除后和目标变量被添加前，*莫比迪克*有 16939 个维度。为了减少文档的维数，我们可以将两个或多个单词组合起来，如果它们作为一个标记而不是两个单独的标记来传达大量信息。如果我们选择一对词，这将被称为**二元组**。

让我们检查句子“我爱热狗。”有三对词，(“我”、“爱”)、(“爱”、“热”)、(“热”、“狗”)。前两个单词对是随机的，在一起不表达任何重要的意思。然而， *hot* 和 *dogs* 这两个词合在一起传达了一种食物的名称。如果我们将这最后一对视为一个记号，那么句子的维度就减少了一个。这个过程被称为创建二元模型。一个 **ngram** 与一个**二元模型**不同，因为一个 ngram 可以将 *n* 个单词或字符作为一个令牌来处理。

NLTK 提供了一个二元模型方法。导入 NLTK 后，您可以将 bigram 对象`nltk.collocations.BigramAssocMeasures()`存储为一个变量。然后，您可以利用 NLTK 的 collector 和 scorer 方法来查看相关联的二元模型及其规范化的频率分数。

How to make bigrams with NLTK

`score_ngrams()`方法返回二元模型列表及其相关的归一化频率分布。

![](img/ee122113cc36ced75444bf9cf82c9b80.png)

The top five bigrams for Moby Dick

不是所有单词对都能传达大量信息。NLTK 提供了[逐点交互信息](http://www.nltk.org/howto/collocations.html) (PMI)计分器对象，该对象分配一个统计度量来比较每个二元模型。该方法还允许您筛选出出现次数少于最小次数的令牌对。在调用 bigram 方法后，您可以应用一个频率过滤器`.apply_freq_filter(5)`，其中 5 是 bigram 必须出现的最小次数。最后，使用 PMI 参数`score_ngrams(bigram_measures.pmi)`调用 scorer 方法。

![](img/f04853d51fccc6e22172d4e45cb22643.png)

The top five bigrams by PMI score for Moby Dick

# 结论

NLTK 有许多强大的方法，允许我们用几行代码评估文本数据。二元模型、n 元模型和 PMI 分数允许我们降低语料库的维度，这在我们继续进行更复杂的任务时节省了我们的计算能量。一旦文档被清理，就可以很容易地应用 NLTK 方法。