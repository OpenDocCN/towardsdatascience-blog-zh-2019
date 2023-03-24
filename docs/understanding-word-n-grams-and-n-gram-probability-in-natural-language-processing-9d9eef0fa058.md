# 理解自然语言处理中的单词 N 元语法和 N 元语法概率

> 原文：<https://towardsdatascience.com/understanding-word-n-grams-and-n-gram-probability-in-natural-language-processing-9d9eef0fa058?source=collection_archive---------1----------------------->

## [快速文本系列](https://towardsdatascience.com/tagged/the-fasttext-series)

## 这并没有听起来那么难。

![](img/761a5f636d5daa90027e96e11c0679b3.png)

> 最初发表于[我的博客](https://blog.contactsunny.com/data-science/understanding-word-n-grams-and-n-gram-probability-in-natural-language-processing)。

N-gram 大概是整个机器学习领域最容易理解的概念了吧，我猜。一个 N-gram 意味着 N 个单词的序列。举例来说，“中型博客”是一个 2-gram(二元模型)，“中型博客文章”是一个 4-gram，而“写在介质上”是一个 3-gram(三元模型)。嗯，那不是很有趣或令人兴奋。没错，但是我们还是要看看 n-gram 使用的概率，这很有趣。

# 为什么是 N-gram 呢？

在我们继续讲概率的东西之前，让我们先回答这个问题。为什么我们需要学习 n-gram 和相关的概率？嗯，在自然语言处理中，简称 NLP，n-grams 有多种用途。一些例子包括自动完成句子(如我们最近在 Gmail 中看到的)，自动拼写检查(是的，我们也可以这样做)，在一定程度上，我们可以检查给定句子的语法。我们将在后面的文章中看到一些例子，当我们谈论给 n 元文法分配概率的时候。

# n 元概率

让我们以一个句子完成系统为例。这个系统建议在给定的句子中接下来可以使用的单词。假设我给系统一句话“非常感谢你的”,并期望系统预测下一个单词是什么。现在你我都知道，下一个词是“救命”的概率非常大。但是系统怎么知道呢？

这里需要注意的一点是，与任何其他人工智能或机器学习模型一样，我们需要用庞大的数据语料库来训练模型。一旦我们做到了这一点，系统或 NLP 模型将对某个单词在某个单词之后出现的“概率”有一个非常好的了解。因此，希望我们已经用大量数据训练了我们的模型，我们将假设模型给了我们正确的答案。

我在这里讲了一点概率，但现在让我们在此基础上继续。当我们建立预测句子中单词的 NLP 模型时，单词在单词序列中出现的概率才是最重要的。我们如何衡量呢？假设我们正在使用一个二元模型，我们有以下句子作为训练语料:

1.  非常感谢你的帮助。
2.  我非常感谢你的帮助。
3.  对不起，你知道现在几点了吗？
4.  我真的很抱歉没有邀请你。
5.  我真的很喜欢你的手表。

让我们假设在用这些数据训练了我们的模型之后，我想写下句子“我真的很喜欢你的花园。”因为这是一个二元模型，该模型将学习每两个单词的出现，以确定某个单词出现在某个单词之后的概率。例如，从上面例子中的第二句、第四句和第五句，我们知道在单词“really”之后，我们可以看到单词“appreciate”、“sorry”或单词“like”出现。所以模型会计算这些序列中每一个的概率。

假设我们正在计算单词“w1”出现在单词“w2”之后的概率，则公式如下:

```
*count(w2 w1) / count(w2)*
```

其是单词在所需序列中出现的次数，除以该单词在预期单词在语料库中出现之前的次数。

从我们的例句中，让我们来计算“like”这个词出现在“really”这个词之后的概率:

```
count(really like) / count(really)
= 1 / 3
= 0.33
```

同样，对于另外两种可能性:

```
count(really appreciate) / count(really)
= 1 / 3
= 0.33count(really sorry) / count(really)
= 1 / 3
= 0.33
```

因此，当我键入短语“我真的”，并期望模型建议下一个单词时，它只会在三次中得到一次正确答案，因为正确答案的概率只有 1/3。

作为另一个例子，如果我对模型的输入句子是“谢谢你的邀请”，并且我期望模型建议下一个单词，它将会给我单词“你”，因为例句 4。这是模型知道的唯一例子。你可以想象，如果我们给模型一个更大的语料库(或更大的数据集)来训练，预测将会改善很多。同样，我们在这里只使用二元模型。我们可以使用三元模型甚至四元模型来提高模型对概率的理解。

使用这些 n 元语法和某些单词在某些序列中出现的概率可以改进自动完成系统的预测。同样，我们使用 can NLP 和 n-grams 来训练基于语音的个人助理机器人。例如，使用 3-gram 或 trigram 训练模型，机器人将能够理解句子之间的差异，如“温度是多少？”和“设定温度”

我希望这是一个足够清晰的解释，可以理解自然语言处理中 n 元语法这个非常简单的概念。我们将使用这种 n 元语法的知识，并使用它来[优化我们的机器学习模型](https://blog.contactsunny.com/data-science/optimising-a-fasttext-model-for-better-accuracy)用于文本分类，我们在早些时候的[介绍 fastText 库](https://blog.contactsunny.com/data-science/an-intro-to-text-classification-with-facebooks-fasttext-natural-language-processing)帖子中构建了该模型。

> 在 [Twitter](https://twitter.com/contactsunny) 上关注我，了解更多[数据科学](https://blog.contactsunny.com/tag/data-science)、[机器学习](https://blog.contactsunny.com/tag/machine-learning)，以及通用[技术更新](https://blog.contactsunny.com/category/tech)。还有，你可以[关注我的个人博客](https://blog.contactsunny.com/)。

如果你喜欢我在 Medium 或我的个人博客上的帖子，并希望我继续做这项工作，请考虑[在 Patreon](https://www.patreon.com/bePatron?u=28955887) 上支持我。