# 面向情感分析的文本预处理技术

> 原文：<https://towardsdatascience.com/updated-text-preprocessing-techniques-for-sentiment-analysis-549af7fe412a?source=collection_archive---------17----------------------->

## 让我们讨论一些技术的缺点以及如何改进它们

![](img/aa7a35b928f504673a921388000ab855.png)

Photo by [Charles 🇵🇭](https://unsplash.com/@charlesdeluvio?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/feedback?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

人们使用了许多文本预处理技术，但是有多少是真正有用的呢？我从事文本数据工作已经快 6 个月了，我觉得当你开发一个会被很多人使用的产品时会遇到很多挑战。

这里有一些你随处可见的技巧-

1.  移除数字、标点符号、表情符号等。
2.  删除停用词
3.  词汇化或词干化

移除数字后会发生什么？

数字在情感分析中起着重要的作用。怎么会？考虑你提供情绪分析服务的食品交付应用程序，通过文本反馈。现在，如果在文本预处理过程中你删除了所有的数字，那么你将如何区分两个反馈，即“我会给食物打 5 分，满分 5 分”和“我会给食物打 1 分，满分 5 分”。这只是一个例子。数字在很多情况下都发挥了作用。

**移除表情符号会发生什么？**

表情符号可以帮助你正确评估写评论的人的情绪。怎么会？如果有人写反馈“食物是😋另一个人写道，“食物是😖".在表情符号的帮助下，这个人想说的话清晰可见。还有更多的案例。

**当你移除停用字词时会发生什么？**

我在以前的文章中讨论过这个问题。如果你想深入了解，那么你一定要阅读这篇文章。

[](https://medium.com/zykrrtech/why-you-should-avoid-removing-stopwords-aa7a353d2a52) [## 为什么应该避免删除停用词

### 删除停用词真的能提高模型性能吗？

medium.com](https://medium.com/zykrrtech/why-you-should-avoid-removing-stopwords-aa7a353d2a52) 

既然我们在这里，让我们快速讨论当停用字词被移除时会发生什么问题。删除停用词最常见的方法是借助 NLTK 库。NLTK 的停用词表也包含像 not、not、don 等这样的词，出于显而易见的原因，这些词不应该被删除，但这并不意味着你不应该删除任何停用词。

如果你正在使用像 TF-IDF，Count Vectorizer，BOW 这样的方法，那么就有必要删除停用词(不是全部),否则它们会产生很多噪音。使用预训练嵌入的深度学习方法是目前处理所有这些的最佳方法。深度学习方法的好处是它们不需要任何类型的预处理。如果你对深度学习方法更感兴趣，那么你可以看看我的文章

[](https://medium.com/analytics-vidhya/going-beyond-traditional-sentiment-analysis-technique-b9c91b313c07) [## 超越传统的情感分析技术

### 传统方法失败在哪里？

medium.com](https://medium.com/analytics-vidhya/going-beyond-traditional-sentiment-analysis-technique-b9c91b313c07) 

**当你变元或变干时会发生什么？**

Lemmatize 和 stem 都生成单词的词根形式，除了 stem 可能生成字典中不存在的单词。变元化比词干化使用得更广泛，对于本文，我们也考虑变元化。

当您使用 TF-IDF、CountVectorizer 或 BOW 模型时，引理化起着重要的作用。为什么？因为每个单词都会被还原成它的基本形式。例如——“狗”将被简化为“狗”。由于这些方法创建了一个稀疏矩阵，我们需要将所有单词转换成它们的基本形式，以减少唯一单词的数量。问题是，很多时候你会因为把一个单词转换成它的基本形式而失去它的上下文。

具有单词嵌入的深度学习情感分析遵循不同的方法。它将创建一个 n 维向量，而不是稀疏矩阵。所以，在 n 维向量中，像“狗”和“狗”这样的词会彼此更接近。因此，变元化的需要变得不必要。除此之外，矢量嵌入还有很多优点。

到目前为止，我们已经讨论了许多问题。其中一些仍在研究范围内，但我们可以尝试解决一些。

1.  让我们从停用词问题开始。解决这个问题的最好办法是不要删除每一个在某种程度上有用的单词。然而，最好的方法是使用深度学习方法，像 Glove 一样预先训练嵌入，不需要删除停用词。
2.  让我们谈谈第二个问题，即表情符号。我认为可行的解决方案是创建一个 python 字典，用一些单词替换所有出现的表情符号。这方面的一个例子是替换“😊“用‘快乐’或者也许别的什么。虽然这种方法看起来很幼稚，实际上也是如此，但它仍然比删除表情符号要好。
3.  数字呢？深度学习是解决方案。

**结论**

深度学习方法真的好吗？

是的，他们更强大。DL 方法在预测时考虑词序。如果你把一个句子中的单词混在一起，然后应用 TF-IDF、CountVectorizer 或 BOW，那么得到相同结果的可能性就太高了。

最近，像 BERT，Robert 等预先训练的模型证明了 NLP 任务可以通过深度学习方法更好地完成。这些方法的优点是，它们的工作方式就像迁移学习一样，与从头开始训练它们相比，你实际上需要的例子要少得多。

如果你有兴趣了解更多关于情绪分析的工作原理，你可以在这里阅读—[https://monkeylearn.com/sentiment-analysis/](https://monkeylearn.com/sentiment-analysis/)。