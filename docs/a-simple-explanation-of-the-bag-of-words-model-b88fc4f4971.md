# 单词袋模型的简单解释

> 原文：<https://towardsdatascience.com/a-simple-explanation-of-the-bag-of-words-model-b88fc4f4971?source=collection_archive---------5----------------------->

## 快速、简单地介绍单词袋模型以及如何用 Python 实现它。

![](img/46728e05e163820425912289aedddd15.png)

Photo by [Patrick Tomasso](https://unsplash.com/@impatrickt?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/words?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

**单词袋** (BOW)模型是一种通过计算每个单词出现的次数将任意文本转换成**固定长度向量**的表示方法。这个过程通常被称为**矢量化**。

我们用一个例子来理解这个。假设我们想要向量化以下内容:

*   猫坐下了
*   猫坐在帽子里
*   *戴帽子的猫*

我们将每一个都称为文本**文档**。

# 第一步:确定词汇

我们首先定义我们的**词汇表**，它是在我们的文档集中找到的所有单词的集合。在上述三个文档中唯一找到的单词是:`the`、`cat`、`sat`、`in`、`the`、`hat`和`with`。

# 第二步:计数

为了对我们的文档进行矢量化，我们所要做的就是**计算每个单词出现的次数**:

![](img/9c8086228edeff56cde7779ce67fed36.png)

现在每个文档都有长度为 6 的向量了！

*   *猫坐着* : `[1, 1, 1, 0, 0, 0]`
*   猫坐在帽子里
*   *戴帽子的猫* : `[2, 1, 0, 0, 1, 1]`

请注意，当我们使用 BOW 时，我们丢失了上下文信息，例如，单词在文档中出现的位置。这就像一个字面上的**单词包**:它只告诉你*哪些*单词出现在文档中，而不是*它们出现在*的什么地方。

# 在 Python 中实现 BOW

既然您已经知道了 BOW 是什么，我猜您可能需要实现它。下面是我的首选方法，它使用了 [Keras 的 Tokenizer 类](https://keras.io/preprocessing/text/):

运行该代码会给我们带来:

```
Vocabulary: ['the', 'cat', 'sat', 'hat', 'in', 'with']
[[0\. 1\. 1\. 1\. 0\. 0\. 0.]
 [0\. 2\. 1\. 1\. 1\. 1\. 0.]
 [0\. 2\. 1\. 0\. 1\. 0\. 1.]]
```

注意，这里的向量长度是 7，而不是 6，因为在开头有额外的`0`元素。这是一个无关紧要的细节——Keras 储量指数`0`并且从来不把它分配给任何单词。

# BOW 有什么用？

尽管是一个相对基本的模型，BOW 经常被用于自然语言处理任务，如文本分类。它的优势在于它的简单性:它的计算成本很低，有时当定位或上下文信息不相关时，越简单越好。

我已经写了一篇使用 BOW 进行亵渎检测的博文——如果你对 BOW 的实际应用感到好奇，就来看看吧！

*最初发表于*[T5【https://victorzhou.com】](https://victorzhou.com/blog/bag-of-words/)*。*