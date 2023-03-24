# [NLP]基础:测量文本的语言复杂性

> 原文：<https://towardsdatascience.com/linguistic-complexity-measures-for-text-nlp-e4bf664bd660?source=collection_archive---------6----------------------->

![](img/34056d20dfe54b19f60c64543f78306f.png)

Photo by [Nong Vang](https://unsplash.com/@californong?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/reading-a-book?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

确定文本的语言复杂性是您在自然语言处理中学习的第一个基本步骤。在本文中，我将带您了解什么是语言复杂性，以及如何度量它。

# 预处理步骤

首先，你需要对你的语料库进行标记化。换句话说，您需要将文本语料库中的句子分解成单独的单词(标记)。此外，你还应该删除标点符号，符号，数字，并将所有单词转换为小写。在下面的代码行中，我向您展示了如何使用 *quanteda* 包来实现这一点。

```
# Creating a corpus
mycorpus <- corpus(data, text_field = "text")# Tokenisation
tok <- tokens(mycorpus, what = "word",
                  remove_punct = TRUE,
                  remove_symbols = TRUE,
                  remove_numbers = TRUE,
                  remove_url = TRUE,
                  remove_hyphens = FALSE,
                  verbose = TRUE, 
                  include_docvars = TRUE)
tok <- tokens_tolower(tok)
```

然后，我在语料库的预处理中增加了一个步骤:删除停用词。这允许你在测量文本的词汇复杂度时得到更有意义的结果。为什么？因为停用词的词汇丰富性很小，只用于在句子中绑定单词。因此，为了不在分析中产生任何噪声，最好将它们移除。但是，请记住，这样做会大大减少字数。

```
tok <- tokens_select(tok, stopwords("english"), selection = "remove", padding = FALSE)
```

请注意，一些词汇复杂性的度量方法可能在您的语料库对象上工作得很好(正如您将在下面看到的)，因此您不需要经历预处理步骤。

# 类型与标记

在自然语言处理中，您会经常听到“类型”或“标记”这样的词，理解它们所指的是很重要的。

**令牌**指的是你语料库中的字数。它指任何种类的词，甚至是停用词。下面的代码行向你展示了如何使用你的标记化对象找出你的语料库中的标记数量。

```
ntoken(tok)
```

**类型**指的是在你的语料库中找到的独特单词的数量。换句话说，虽然标记计算所有单词而不管它们是否重复，但是类型只显示唯一单词的频率。从逻辑上讲，类型的数量应该低于令牌的数量。

```
ntype(tok)
```

# 复杂性度量——它们是什么？

在语言学中，复杂性是一个文本的特征，但在实践中有多种度量，因此有多种隐含的定义。在自然语言处理中，这些度量对于描述性统计是有用的。我将向你展示评估文本复杂性的两种最流行的方法:文本的易读性(**文本可读性**)和丰富性(**文本丰富性**)。

# 词汇可读性

可读性测量试图量化文本的阅读难度。通常采用的基准是儿童书籍——被归类为“简单”类。使用 *quanteda* 包，一个有用的方法是从下面代码中显示的文本可读性的基本函数开始。

```
readability <- textstat_readability(mycorpus, c("meanSentenceLength","meanWordSyllables", "Flesch.Kincaid", "Flesch"), remove_hyphens = TRUE,
  min_sentence_length = 1, max_sentence_length = 10000,
  intermediate = FALSE)head(readability)
```

注意，这个函数允许您使用语料库对象，而不是标记化对象。在你的论点中，你必须指定你希望使用哪种词汇可读性的度量(点击[这里](https://rdrr.io/cran/quanteda/man/textstat_readability.html)找出所有你可以使用的)。可以包括非常简单直白的比如**“meansentencelongth”**计算平均句子长度、**“meanwodsyllable”**计算平均单词音节。

否则，您可以选择统计上更加稳健和复杂的度量。比较流行的是**弗莱施-金凯德**可读性评分或者**弗莱施的阅读容易度评分**。这两种测试使用相同的核心指标(单词长度和句子长度)，但它们有不同的权重因子。这两个测试的结果大致成反比:在 Flesch 阅读难易程度测试中得分高的文本在 Flesch-Kincaid 测试中的得分应该较低。

弗莱施的阅读容易度测试中的高分表明材料更容易阅读；数字越小，段落越难阅读。作为一个基准:高分应该容易被一个平均 11 岁的孩子理解，而低分最好被大学毕业生理解。

# 词汇丰富度

与文本可读性类似， *quanteda* 包也有一个使用词汇多样性度量来评估文本丰富程度的功能。

最受欢迎的措施是**类型令牌配额(TTR)** 。这一措施背后的基本思想是，如果文本更复杂，作者使用更多样的词汇，因此有更多的类型(独特的单词)。这种逻辑在 TTR 的公式中是显而易见的，它计算类型的数量除以标记的数量。因此，TTR 越高，词汇复杂度越高。

```
dfm(tok) %>% 
  textstat_lexdiv(measure = "TTR")
```

虽然 TTR 是一个有用的方法，但是你应该记住它可能会受到文本长度的影响。文本越长，引入小说词汇的可能性越小。因此，更长的文本可能更倾向于等式的标记侧:添加更多的单词(标记)，但是越来越少的单词(类型)被表示。因此，如果你有一个大的语料库要分析，你可能想使用额外的词汇丰富度来验证你的 TTR 的结果。

你可以使用的另一个词汇丰富度的量度是 **Hapax 丰富度**，定义为只出现一次的单词数除以总单词数。要计算这个值，只需在文档特征矩阵上使用一个逻辑运算，为每个出现一次的术语返回一个逻辑值，然后对这些行求和得到一个计数。最后但同样重要的是，计算它占总字数(ntokens)的比例，以便在整个语料库中进行更好的解释。

```
ungd_corpus_dfm <- dfm(ungd_corpus)
rowSums(ungd_corpus_dfm == 1) %>% head()
hapax_proportion <- rowSums(ungd_corpus_dfm == 1)/ntoken(ungd_corpus_dfm)
hapax_proportion
```

最后，值得注意的是，复杂性度量在语言学和自然语言处理领域得到了广泛的讨论和辩论。许多学者试图对这些方法进行相互比较，或者回顾现有的方法以开发新的更好的方法。不用说，这是一个不断发展的领域。

*我经常写关于数据科学和自然语言处理的文章。在* [*Twitter*](https://twitter.com/celine_vdr) *或*[*Medium*](https://medium.com/@celine.vdr)*上关注我，查看更多类似的文章或简单地更新下一篇文章。* ***感谢阅读！***