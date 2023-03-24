# ELMo:上下文语言嵌入

> 原文：<https://towardsdatascience.com/elmo-contextual-language-embedding-335de2268604?source=collection_archive---------4----------------------->

## 使用 ELMo 的深度语境化语言表示创建语义搜索引擎，以及为什么在 NLP 中语境就是一切

![](img/d369de8a331364a42191d45fb6cdefb6.png)

Semantic sentence similarity using the state-of-the-art ELMo natural language model

本文将探索自然语言建模的最新进展；深层语境化单词嵌入。重点在于如何使用最先进的 ELMo 模型来检查给定文档中的句子相似性，并创建一个简单的语义搜索引擎。完整的代码可以在 Colab 笔记本[这里](https://colab.research.google.com/drive/13f6dKakC-0yO6_DxqSqo0Kl41KMHT8A1)查看。

## 语境在自然语言处理中的重要性

众所周知，语言是复杂的。上下文可以完全改变一个句子中单个单词的意思。例如:

> 他踢了**桶一脚。**
> 
> 我还没有划掉我的**清单上的所有项目。**
> 
> 水桶装满了水。

在这些句子中，虽然单词 bucket 总是相同的，但它的意思却非常不同。

![](img/d97d014ea80b9acedf1e263fdb164420.png)

Words can have different meanings depending on context

虽然我们可以很容易地破译语言中的这些复杂性，但创建一个能够理解给定环境文本中单词含义的不同细微差别的模型是困难的。

正是由于这个原因，传统的单词嵌入(word2vec、GloVe、fastText)有所欠缺。每个单词只有一种表达方式，因此它们不能捕捉每个单词的意思是如何根据周围的环境而变化的。

## 介绍 ELMo 深层语境化的词汇表征

输入 ELMo。它由 AllenNLP 在 2018 年开发，超越了传统的嵌入技术。它使用深度的双向 LSTM 模型来创建单词表示。

ELMo 不是一个单词及其对应向量的字典，而是在单词使用的上下文中分析单词。它也是基于字符的，允许模型形成词汇表外单词的表示。

因此，这意味着 ELMo 的使用方式与 word2vec 或 fastText 完全不同。ELMo 没有在字典中“查找”单词及其相应的向量，而是通过将文本传递给深度学习模型来动态创建向量。

## 一个可行的例子，ELMo 的实际使用不到 5 分钟

我们开始吧！我将在这里添加主要的代码片段，但是如果您想要查看完整的代码集(或者确实想要通过点击笔记本中的每个单元格来获得奇怪的满足感)，请在这里查看相应的 [Colab 输出](https://colab.research.google.com/drive/13f6dKakC-0yO6_DxqSqo0Kl41KMHT8A1)。

根据我最近的几篇帖子，我们将使用的数据是基于现代奴隶制的回报。这些都是公司的强制性声明，以传达他们如何在内部和供应链中解决现代奴隶制问题。在这篇文章中，我们将深入探究 ASOS 的回归(一家英国在线时装零售商)。

如果你有兴趣看到在这个数据集上进行的 NLP 实验的迷你系列中的其他帖子，我在本文的结尾提供了这些的链接。

**1。获取文本数据，清理并标记**

令人惊讶的是，使用 Python 字符串函数和 spaCy 做到这一点是多么简单。这里我们通过以下方式进行一些基本的文本清理:

a)删除换行符、制表符、多余的空格以及神秘的“xa0”字符；

b)使用空格将文本分成句子。sents 的迭代器。

ELMo 既可以接收句子字符串列表，也可以接收列表(句子和单词)列表。这里我们选择了前者。我们知道 ELMo 是基于字符的，因此对单词进行标记不会对性能产生任何影响。

```
nlp = spacy.load('en_core_web_md')#text represents our raw text documenttext = text.lower().replace('\n', ' ').replace('\t', ' ').replace('\xa0',' ') #get rid of problem chars
text = ' '.join(text.split()) #a quick way of removing excess whitespace
doc = nlp(text)sentences = []
for i in doc.sents:
  if len(i)>1:
    sentences.append(i.string.strip()) #tokenize into sentences
```

**2。使用 TensorFlow Hub 获取 ELMo 模型:**

如果您还没有接触过 TensorFlow Hub，它可以为 TensorFlow 提供大量预先训练好的模型，从而节省大量时间。对我们来说幸运的是，其中一个模特是埃尔莫。我们只需要两行代码就可以加载一个完全训练好的模型。多么令人满意…

```
url = "[https://tfhub.dev/google/elmo/2](https://tfhub.dev/google/elmo/2)"
embed = hub.Module(url)
```

为了在 anger 中使用这个模型，我们只需要多几行代码来将它指向我们的文本文档并创建句子向量:

```
# This tells the model to run through the 'sentences' list and return the default output (1024 dimension sentence vectors).
embeddings = embed(
    sentences,
    signature="default",
    as_dict=True)["default"]#Start a session and run ELMo to return the embeddings in variable x
with tf.Session() as sess:
  sess.run(tf.global_variables_initializer())
  sess.run(tf.tables_initializer())
  x = sess.run(embeddings)
```

**3。使用可视化来检测输出**

令人惊讶的是，观想作为一种更好地理解数据的方式经常被忽视。图片说明千言万语，我们将创建一个一千字的图表*来证明这一点(实际上是 8511 个字)。*

这里，我们将使用主成分分析和 t-SNE 将 ELMo 输出的 1024 个维度减少到 2 个，以便我们可以查看模型的输出。如果你想了解更多，我在文章的最后提供了关于如何实现这一点的进一步阅读。

```
from sklearn.decomposition import PCApca = PCA(n_components=50) #reduce down to 50 dim
y = pca.fit_transform(x)from sklearn.manifold import TSNEy = TSNE(n_components=2).fit_transform(y) # further reduce to 2 dim using t-SNE
```

使用神奇的 Plotly 库，我们可以在任何时候创建一个美丽的，互动的情节。下面的代码显示了如何呈现我们的降维结果，并将其连接到句子文本。根据句子的长度，还添加了颜色。当我们使用 Colab 时，最后一行代码下载 HTML 文件。这可以在下面找到:

[](https://drive.google.com/open?id=17gseqOhQl9c1iPTfzxGcCfB6TOTvSU_i) [## 句子编码

### 交互式句子嵌入

drive.google.com](https://drive.google.com/open?id=17gseqOhQl9c1iPTfzxGcCfB6TOTvSU_i) 

创建它的代码如下:

```
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import download_plotlyjs, init_notebook_mode, plot, iplotinit_notebook_mode(connected=True)data = [
    go.Scatter(
        x=[i[0] for i in y],
        y=[i[1] for i in y],
        mode='markers',
        text=[i for i in sentences],
    marker=dict(
        size=16,
        color = [len(i) for i in sentences], #set color equal to a variable
        opacity= 0.8,
        colorscale='Viridis',
        showscale=False
    )
    )
]
layout = go.Layout()
layout = dict(
              yaxis = dict(zeroline = False),
              xaxis = dict(zeroline = False)
             )
fig = go.Figure(data=data, layout=layout)
file = plot(fig, filename='Sentence encode.html')from google.colab import files
files.download('Sentence encode.html')
```

探索这种可视化，我们可以看到 ELMo 在根据语义相似性对句子进行分组方面做了出色的工作。事实上，该模型的有效性令人难以置信:

![](img/d369de8a331364a42191d45fb6cdefb6.png)

Download the HTML for yourself (link above) to see ELMo in action

**4。创建语义搜索引擎:**

现在我们确信我们的语言模型运行良好，让我们将它应用到语义搜索引擎中。这个想法是，这将允许我们通过搜索查询的语义接近度而不是关键字来搜索文本。

这实际上很容易实现:

*   首先，我们获取一个搜索查询，并对其运行 ELMo
*   然后，我们使用余弦相似度将它与文本文档中的向量进行比较；
*   然后，我们可以从文档中返回与搜索查询最接近的“n”个匹配项。

Google Colab 有一些很棒的特性来创建表单输入，非常适合这个用例。例如，创建一个输入就像在变量后添加 ***#@param*** 一样简单。下面显示了字符串输入的情况:

```
search_string = "example text" #@param {type:"string"}
```

除了使用 Colab 表单输入，我还使用了“IPython.display.HTML”来美化输出文本，并使用一些基本的字符串匹配来突出显示搜索查询和结果之间的常见单词。

让我们来测试一下。让我们看看 ASOS 在他们现代奴隶制的回归中，在道德准则方面做了些什么:

![](img/b0946115c309899603fb3e131509b085.png)

A fully interactive semantic search engine in just a few minutes!

这太神奇了！匹配超越了关键词，搜索引擎清楚地知道‘伦理’和道德是密切相关的。我们发现符合诚信准则以及道德标准和政策的内容。两者都与我们的搜索查询相关，但不直接基于关键字链接。

我希望你喜欢这篇文章。如果您有任何问题或建议，请留下您的意见。

## *延伸阅读:*

以下是我在 NLP 和探索公司现代奴隶制回归的迷你系列中的其他帖子:

[](/clean-your-data-with-unsupervised-machine-learning-8491af733595) [## 使用无监督的机器学习清理您的数据

### 清理数据不一定是痛苦的！这篇文章是一个如何使用无监督机器学习的快速例子…

towardsdatascience.com](/clean-your-data-with-unsupervised-machine-learning-8491af733595) [](/supercharging-word-vectors-be80ee5513d) [## 增压词向量

### 一个在你的 NLP 项目中提升快速文本和其他单词向量的简单技术

towardsdatascience.com](/supercharging-word-vectors-be80ee5513d) 

为了找到更多关于降维过程的信息，我推荐下面的帖子:

[](https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b) [## 在 Python 中使用 PCA 和 t-SNE 可视化高维数据集

### 任何与数据相关的挑战的第一步都是从探索数据本身开始。这可以通过查看…

medium.com](https://medium.com/@luckylwk/visualising-high-dimensional-datasets-using-pca-and-t-sne-in-python-8ef87e7915b) 

最后，要想了解更多关于最新语言模型的信息，下面的内容值得一读:

[http://jalammar.github.io/illustrated-bert/](http://jalammar.github.io/illustrated-bert/)