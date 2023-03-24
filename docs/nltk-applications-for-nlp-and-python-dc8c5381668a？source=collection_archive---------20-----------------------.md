# NLP 和 Python 的 NLTK 应用程序

> 原文：<https://towardsdatascience.com/nltk-applications-for-nlp-and-python-dc8c5381668a?source=collection_archive---------20----------------------->

![](img/9c06c1265fa9afa5095208471f745932.png)

The Natural Language toolkit’s (NLTK) development started in 2001 by Steven Bird and Edward Loper of the University of Pennsylvania

上周我完成了一个[朴素贝叶斯分类器](https://medium.com/analytics-vidhya/curb-your-enthusiasm-or-seinfeld-2e430abf866c)，它决定了*宋飞*和*抑制你的热情*剧本之间的区别。我开始利用自然语言工具包( [NLTK](https://www.nltk.org/) )的一些功能，但是还有一些额外的部分我本来可以使用，这样可以简化我的代码并节省我的时间。我将在这里介绍 NLTK 的一些预处理和分析功能。

## 清洁

自然语言处理(NLP)最流畅的部分是清理原始文本数据。这个过程根据文本的内容和项目的总体目标而有所不同。文本分类项目和文本预测项目将具有完全不同的预处理步骤，因为文本的顺序在分类中并不像在预测中那样重要。上周我的项目的目标是根据它们所属的节目对抄本进行分类，所以我不需要维护每个文档中文本的顺序。

我最初的清洁功能很简单。首先，我用 regex 删除了 stage 注释，然后用一个自定义列表删除了文本中的一些标点符号，使所有文本都变成小写，用空格分割文本，删除字符朗读指示和停用词，最后返回文本。我没有借助 NLTK 的力量就做到了这一点。这个功能大部分是成功的，结果各不相同，因为文字记录是由不同的爱好者用不同的风格偏好写的。该函数最值得注意的问题是，它只删除指定的停用词，这意味着语料库的维数非常大。由于包含停用词而导致的高维度对于深度学习任务来说是站不住脚的，这意味着该功能将是不够的。

NLTK 库包含每种支持的语言的停用词列表(*在撰写本文时，有 22 种支持的语言有停用词列表，您可以通过访问 nltk.org 或在您的本地机器*上检查 `*user/nltk_data/corpora/stopwords*` *来检查它们)。您必须指定要使用哪种语言的停用词列表，否则 NLTK 将在您的分析中包括所有语言的所有停用词。在 python 中，你可以通过输入`from nltk.corpus import stopwords`来导入停用词表，通过输入`import string`来导入标点符号表。您可以将这些列表保存为自定义变量，以提取项目中不需要的字符。下面是更新的清理功能，它利用 NLTK 并指定只使用英语。*

An updated text cleaning function that uses NLTK. Note that “English” is a parameter for the ‘stopwords’ method. Also, you can add or remove any stop word or punctuation mark included in the NLTK library, which makes customization easier.

完成清理步骤后，您可以继续进行词干化和词汇化。

## 分析

NLTK 还包括几个分析函数，允许您轻松地从语料库中计算重要的统计数据。例如，NLTK 有自己的令牌频率计数方法。`FreqDist`创建两个元素元组的列表，其中每个元组代表一个单词及其频率计数。您可以通过调用`.most_common()`方法来进一步访问最常见的元素。如果给定一个数字参数，该方法将按照降序返回文档或语料库中最常见的标记的数量。如果您要比较不同大小的多个文档，那么您可以对频率计数进行标准化，以便进行更合适的比较。

# 结论

NLTK 包含用于文本预处理和语料库分析的有用工具。您不需要为每个 NLP 项目创建自己的停用词列表或频率函数。NLTK 为您节省了时间，这样您就可以专注于 NLP 任务，而不是重写函数。