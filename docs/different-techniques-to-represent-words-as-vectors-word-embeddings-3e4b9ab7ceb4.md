# 将单词表示为向量的不同技术(单词嵌入)

> 原文：<https://towardsdatascience.com/different-techniques-to-represent-words-as-vectors-word-embeddings-3e4b9ab7ceb4?source=collection_archive---------4----------------------->

## 从计数矢量器到 Word2Vec

![](img/2227c940131a50f555b18d01c59e718b.png)

Photo by [Romain Vignes](https://unsplash.com/@rvignes?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

目前，我正在做一个 Twitter 情感分析项目。在阅读如何向我的神经网络输入文本时，我发现我必须将每条推文的文本转换成指定长度的向量。这将允许神经网络对推文进行训练，并正确地学习情感分类。

因此，我简要地分析了将文本转换成向量的各种方法——通常称为单词嵌入。

> **单词嵌入**是自然语言处理(NLP)中一套语言建模和特征学习技术的统称，其中词汇表中的单词或短语被映射到实数向量。— [维基百科](https://en.wikipedia.org/wiki/Word_embedding)

在本文中，我将探索以下单词嵌入技术:

1.  计数矢量器
2.  TF-IDF 矢量器
3.  哈希矢量器
4.  Word2Vec

# 示例文本数据

我正在创造 4 个句子，我们将在上面应用这些技术并理解它们是如何工作的。对于每种技术，我将只使用小写单词。

# 计数矢量器

![](img/16e28dcfb94d282d534f373d6468876f.png)

Photo by [Steve Johnson](https://unsplash.com/@steve_j?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

将文本转换为矢量的最基本方法是通过计数矢量器。

**步骤 1:** 在完整的文本数据中识别唯一的单词。在我们的例子中，列表如下(17 个单词):

```
['ended', 'everyone', 'field', 'football', 'game', 'he', 'in', 'is', 'it', 'playing', 'raining', 'running', 'started', 'the', 'towards', 'was', 'while']
```

步骤 2: 对于每个句子，我们将创建一个长度与上面(17)相同的零数组

第三步:一次看一个句子，我们将阅读第一个单词，找出它在句子中的总出现次数。一旦我们知道了这个词在句子中出现的次数，我们就可以确定这个词在上面列表中的位置，并在那个位置用这个计数替换同一个零。对所有的单词和句子重复这一过程

## 例子

就拿第一句话来说吧， ***他在打野战。*** 其向量为`[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]`。

第一个词是`He`。它在句子中的总数是 1。还有，在上面的单词列表中，它的位置是从开始算起的第 6 位(都是小写)。我将更新它的向量，现在它将是:

```
[0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

考虑第二个词，也就是`is`，向量变成:

```
[0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

类似地，我也将更新其余的单词，第一句话的向量表示将是:

```
[0, 0, 1, 0, 0, 1, 1, 1, 0, 1, 0, 0, 0, 1, 0, 0, 0]
```

同样的情况也会在其他句子中重复出现。

## 密码

`sklearn`提供了 CountVectorizer()方法来创建这些单词嵌入。导入包后，我们只需要在完整的句子列表上应用`fit_transform()`,就可以得到每个句子的向量数组。

上述要点中的输出显示了每个句子的向量表示。

# TF-IDF 矢量器

![](img/f83c95e80a636d3fc034184dafb2d1c6.png)

Photo by [Devin Avery](https://unsplash.com/@officialdavery?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

虽然计数矢量器将每个句子转换成它自己的矢量，但它不考虑单词在整个句子列表中的重要性。例如，`He`出现在两个句子中，它不能提供区分这两个句子的有用信息。因此，它在句子的整个向量中应该具有较低的权重。这就是 TF-IDF 矢量器的用武之地。

TF-IDF 由两部分组成:

1.  **TF(词频)** —定义为一个词在给定句子中出现的次数。
2.  **IDF(逆文档频率)** —它被定义为总文档数除以单词出现的文档数的以 e 为底的对数。

**步骤 1:** 识别完整文本数据中的唯一单词。在我们的例子中，列表如下(17 个单词):

```
['ended', 'everyone', 'field', 'football', 'game', 'he', 'in', 'is', 'it', 'playing', 'raining', 'running', 'started', 'the', 'towards', 'was', 'while']
```

**步骤 2:** 对于每个句子，我们将创建一个长度与上面(17)相同的零数组

**步骤 3:** 对于每个句子中的每个单词，我们将计算 TF-IDF 值，并更新该句子的向量中的相应值

## 例子

我们将首先为所有句子组合中的所有 17 个唯一单词定义一个零数组。

```
[0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

我就拿第一句的`he`这个词， ***他在打野战*** 为它申请 TF-IDF。然后，该值将在句子的数组中更新，并对所有单词重复。

```
Total documents (N): 4
Documents in which the word appears (n): 2
Number of times the word appears in the first sentence: 1
Number of words in the first sentence: 6Term Frequency(TF) = 1Inverse Document Frequency(IDF) = log(N/n)
                                = log(4/2)
                                = log(2)TF-IDF value = 1 * log(2)
             = 0.69314718
```

更新的向量:

```
[0, 0, 0, 0, 0, 0.69314718, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
```

对于所有其他单词，情况也是如此。但是，有些库可能使用不同的方法来计算该值。例如，`sklearn`，计算逆文档频率为:

```
IDF = (log(N/n)) + 1
```

因此，TF-IDF 值为:

```
TF-IDF value = 1 * (log(4/2) + 1)
             = 1 * (log(2) + 1)
             = 1.69314718
```

重复时，该过程将第一句的向量表示为:

```
[0, 0, 1.69314718, 0, 0, 1.69314718, 1.69314718, 1.69314718, 0, 1.69314718, 0, 0, 0, 1, 0, 0, 0]
```

## 密码

`sklearn`提供了计算 TF-IDF 值的方法`TfidfVectorizer`。但是，它对其应用了`l2`规范化，我会忽略使用标志值`None`并保持`smooth_idf`标志为假，因此上面的方法被它用于 IDF 计算。

上述要点中的输出显示了每个句子的向量表示。

# 哈希矢量器

![](img/0d32e9625dab497a209a10f0e51458e1.png)

Photo by [Nick Hillier](https://unsplash.com/@nhillier?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

这个矢量器非常有用，因为它允许我们将任何单词转换成它的散列，并且不需要生成任何词汇。

**第一步:**定义为每个句子创建的向量的大小

**步骤 2:** 对句子应用哈希算法(比如 MurmurHash)

**步骤 3:** 对所有句子重复步骤 2

## 密码

由于这个过程只是一个散列函数的应用，我们可以简单地看一下代码。我将使用来自`sklearn`的`HashingVectorizer`方法。将规范化设置为“无”会将其删除。鉴于以上讨论的两种矢量化技术，我们在每个矢量中都有 17 列，我也将这里的要素数量设置为 17。

这将生成必要的散列值向量。

# Word2Vec

![](img/7e511f1f6fc2c91fdb62bb10d6403e90.png)

Photo by [Mahesh Ranaweera](https://unsplash.com/@mahesh_ranaweera?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

这些是一组神经网络模型，其目的是在向量空间中表示单词。这些模型在理解语境和词与词之间的关系方面是高效的。相似的单词在向量空间中被放置得很近，而不相似的单词被放置得很远。

描述单词是如此令人惊奇，它甚至能够识别关键关系，例如:

```
King - Man + Woman = Queen
```

它能够解释男人对于国王，女人对于王后的意义。通过这些模型可以确定各自的关系。

该课程有两种模式:

1.  **CBOW(连续单词包):**神经网络查看周围的单词(比如左边 2 个和右边 2 个)，并预测中间的单词
2.  **跳跃图:**神经网络接受一个单词，然后尝试预测周围的单词

神经网络有一个输入层、一个隐藏层和一个输出层，用于对数据进行训练和构建向量。由于这是神经网络如何工作的基本功能，我将跳过一步一步的过程。

## 密码

为了实现`word2vec`模型，我将使用`gensim`库，它提供了模型中的许多特性，比如找出奇怪的一个，最相似的单词等等。然而，它没有小写/标记句子，所以我也这样做了。然后将标记化的句子传递给模型。我已经将 vector 的`size`设置为 2，`window`设置为 3，这定义了要查看的距离，`sg` = 0 使用 CBOW 模型。

我用`most_similar`的方法找到所有和`football`这个词相似的词，然后打印出最相似的。对于不同的培训，我们会得到不同的结果，但在我尝试的最后一个案例中，我得到了最相似的单词`game`。这里的数据集只有 4 个句子。如果我们同样增加，神经网络将能够更好地找到关系。

# 结论

我们找到了。我们已经了解了单词嵌入的 4 种方式，以及如何使用代码来实现这一点。如果你有任何想法，想法和建议，请分享并告诉我。感谢阅读！