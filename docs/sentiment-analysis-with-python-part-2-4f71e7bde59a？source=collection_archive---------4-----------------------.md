# 用 Python 进行情感分析(第 2 部分)

> 原文：<https://towardsdatascience.com/sentiment-analysis-with-python-part-2-4f71e7bde59a?source=collection_archive---------4----------------------->

## 改进电影评论情感分类器

![](img/3a004f95500b88d2a4854dbe69d141d2.png)

Photo by [izayah ramos](https://unsplash.com/photos/cR05i3WgRlQ?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/hollywood?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

在这个系列的第一部分，我们建立了一个准系统的电影评论情感分类器。下一篇文章的目标是概述几种可以用来增强 NLP 模型的技术。也就是说，我们不会在任何特定的话题上做太深入的探讨。

如果你还没有，你可以在这里阅读我的第一篇文章:

[](/sentiment-analysis-with-python-part-1-5ce197074184) [## 用 Python 进行情感分析(第 1 部分)

### IMDb 电影评论分类

towardsdatascience.com](/sentiment-analysis-with-python-part-1-5ce197074184) 

本系列中使用的所有代码以及补充材料都可以在这个 [GitHub 资源库](https://github.com/aaronkub/machine-learning-examples/tree/master/imdb-sentiment-analysis)中找到。

# 文本处理

对于我们的第一次迭代，我们做了非常基本的文本处理，如删除标点符号和 HTML 标签，并使一切都小写。我们可以通过删除停用词和规范文本来进一步清理。

为了进行这些转换，我们将使用自然语言工具包 (NLTK)中的库。这是一个非常流行的 Python 的 NLP 库。

## 删除停用词

停用词是非常常见的词，如“如果”、“但是”、“我们”、“他”、“她”和“他们”。我们通常可以在不改变文本语义的情况下删除这些单词，并且经常(但不总是)这样做可以提高模型的性能。当我们开始使用更长的单词序列作为模型特征时，删除这些停用词变得更加有用(见下面的 n-grams)。

**在**之前

```
"bromwell high is a cartoon comedy it ran at the same time as some other programs about school life such as teachers my years in the teaching profession lead me to believe that bromwell high’s satire is much closer to reality than is teachers the scramble to survive financially the insightful students who can see right through their pathetic teachers’ pomp the pettiness of the whole situation all remind me of the schools i knew and their students when i saw the episode in which a student repeatedly tried to burn down the school i immediately recalled at high a classic line inspector i’m here to sack one of your teachers student welcome to bromwell high i expect that many adults of my age think that bromwell high is far fetched what a pity that it isn’t"
```

**在**之后

```
"bromwell high cartoon comedy ran time programs school life teachers years teaching profession lead believe bromwell high's satire much closer reality teachers scramble survive financially insightful students see right pathetic teachers' pomp pettiness whole situation remind schools knew students saw episode student repeatedly tried burn school immediately recalled high classic line inspector i'm sack one teachers student welcome bromwell high expect many adults age think bromwell high far fetched pity"
```

**注意:**在实践中，删除停用词的一个更简单的方法是在 scikit-learn 的任何“矢量器”类中使用`stop_words`参数。如果你想使用 NLTK 的停用词完整列表，你可以做`stop_words='english’`。在实践中，我发现使用 NLTK 的列表实际上会降低我的性能，因为它太昂贵了，所以我通常提供我自己的单词列表。例如，`stop_words=['in','of','at','a','the']`。

## 正常化

文本预处理的一个常见的下一步是通过将一个给定单词的所有不同形式转换成一个来使你的语料库中的单词规范化。存在的两种方法是*词干*和*词汇化*。

**词干**

词干被认为是更原始/更强力的规范化方法(尽管这不一定意味着它的性能会更差)。有几种算法，但总的来说，它们都使用基本规则来切断单词的结尾。

NLTK 有几个词干算法实现。我们将在这里使用波特词干分析器，但是你可以在这里通过例子探索所有的选项: [NLTK 词干分析器](http://www.nltk.org/howto/stem.html)

**词汇化**

词汇化的工作原理是识别给定单词的词性，然后应用更复杂的规则将该单词转换为其真正的词根。

## 结果

**没有归一化**

```
"this is not the typical mel brooks film it was much less slapstick than most of his movies and actually had a plot that was followable leslie ann warren made the movie she is such a fantastic under rated actress there were some moments that could have been fleshed out a bit more and some scenes that could probably have been cut to make the room to do so but all in all this is worth the price to rent and see it the acting was good overall brooks himself did a good job without his characteristic speaking to directly to the audience again warren was the best actor in the movie but fume and sailor both played their parts well"
```

**词干**

```
"thi is not the typic mel brook film it wa much less slapstick than most of hi movi and actual had a plot that wa follow lesli ann warren made the movi she is such a fantast under rate actress there were some moment that could have been flesh out a bit more and some scene that could probabl have been cut to make the room to do so but all in all thi is worth the price to rent and see it the act wa good overal brook himself did a good job without hi characterist speak to directli to the audienc again warren wa the best actor in the movi but fume and sailor both play their part well"
```

**术语化**

```
"this is not the typical mel brook film it wa much le slapstick than most of his movie and actually had a plot that wa followable leslie ann warren made the movie she is such a fantastic under rated actress there were some moment that could have been fleshed out a bit more and some scene that could probably have been cut to make the room to do so but all in all this is worth the price to rent and see it the acting wa good overall brook himself did a good job without his characteristic speaking to directly to the audience again warren wa the best actor in the movie but fume and sailor both played their part well"
```

# n-grams

上次我们在模型中只使用了单个单词的特征，我们称之为 1-grams 或 unigrams。我们还可以通过添加两个或三个单词序列(二元模型或三元模型)来为我们的模型增加更多的预测能力。例如，如果一条评论有三个单词序列“不喜欢电影”，我们只会用单字母模型单独考虑这些单词，可能不会捕捉到这实际上是一种负面的*情绪，因为单词“喜欢”本身将与正面评论高度相关。*

scikit-learn 库使这变得非常容易。只需对任何“矢量器”类使用`ngram_range`参数。

接近 90%了！因此，除了单个单词之外，简单地考虑两个单词的序列将我们的准确率提高了 1.6 个百分点以上。

**注意**:对于你的模型，n 的大小在技术上没有限制，但是有几件事需要考虑。第一，增加克数不一定会给你更好的表现。其次，随着 n 的增加，矩阵的大小呈指数增长，因此，如果您有一个由大型文档组成的大型语料库，您的模型可能需要很长时间来训练。

# 陈述

在第 1 部分中，我们将每个评论表示为一个二元向量(1 和 0 ),语料库中的每个唯一单词都有一个槽/列，其中 1 表示给定的单词在评论中。

虽然这种简单的方法可以很好地工作，但我们有办法将更多的信息编码到向量中。

## 字数

我们可以包括给定单词出现的次数，而不是简单地记录一个单词是否出现在评论中。这可以给我们的情感分类器更多的预测能力。例如，如果一个电影评论者在评论中多次说“惊人”或“糟糕”,那么该评论很可能分别是正面的或负面的。

## TF-IDF

另一种表示语料库中每个文档的常见方法是对每个单词使用 [*tf-idf 统计量*](https://en.wikipedia.org/wiki/Tf%E2%80%93idf) *(* 词频-逆文档频率 *)* ，这是一种加权因子，我们可以使用它来代替二进制或字数表示。

有几种方法可以进行 tf-idf 转换，但简单地说，tf-idf 的目标是表示给定单词在文档(在我们的例子中是电影评论)中出现的次数相对于该单词在语料库中出现的文档数——其中出现在许多文档中的单词的值接近于 0，出现在较少文档中的单词的值接近于 1。

**注意:**现在我们已经讨论了 n 元语法，当我提到“单词”时，我实际上是指任何 n 元语法(单词序列)，如果模型使用大于 1 的 n。

# 算法

到目前为止，我们选择将每个评论表示为一个非常稀疏的向量(很多零！)中的每个唯一的 n 元语法都有一个槽(减去出现太频繁或不够频繁的 n 元语法)。*线性分类器*通常在以这种方式表示的数据上比其他算法执行得更好。

## 支持向量机(SVM)

回想一下，线性分类器往往在非常稀疏的数据集上工作得很好(就像我们拥有的这个)。另一种可以在短训练时间内产生很好结果的算法是具有线性核的支持向量机。

下面是一个 n 元语法范围从 1 到 2 的例子:

关于支持向量机有很多很好的解释，它们比我做得好得多。如果你有兴趣了解更多，这是一个很好的教程:

[](https://blog.statsbot.co/support-vector-machines-tutorial-c1618e635e93) [## 支持向量机教程

### 从例子中学习支持向量机

blog.statsbot.co](https://blog.statsbot.co/support-vector-machines-tutorial-c1618e635e93) 

# 最终模型

这篇文章的目标是给你一个工具箱，当你试图为你的项目找到正确的模型+数据转换时，你可以试着把它们混合在一起。我发现，删除一小组停用词以及从 1 到 3 的 n 元语法范围和线性支持向量分类器给了我最好的结果。

我们突破了 90%大关！

# 摘要

我们已经讨论了几种可以提高 NLP 模型准确性的文本转换方法。这些技术的哪种组合会产生最好的结果取决于任务、数据表示和您选择的算法。尝试多种不同的组合来看看什么是有效的，这总是一个好主意。

我非常确信，通过这篇文章中概述的东西的不同组合，可以获得更高的准确性。我将把这个留给更有抱负的读者。:)请评论你的结果和方法！

# 下次

本系列的下一部分将探索构建情感分类器的深度学习方法。