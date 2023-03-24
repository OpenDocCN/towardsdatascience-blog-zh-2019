# 使用 TF IDF 通过关键词提取形成描述性章节摘要。

> 原文：<https://towardsdatascience.com/using-tf-idf-to-form-descriptive-chapter-summaries-via-keyword-extraction-4e6fd857d190?source=collection_archive---------11----------------------->

![](img/6f8ad3b61f972006778dcd9d1c93718c.png)

Source: [https://pixabay.com/photos/library-books-education-literature-869061/](https://pixabay.com/photos/library-books-education-literature-869061/)

TF IDF 是一种自然语言处理技术，用于提取一组文档或章节中的重要关键字。首字母缩写代表“*术语频率-逆文档频率”*并描述了该算法如何工作。

## 数据集

作为我们的数据集，我们将采用玛丽·雪莱的《弗兰肯斯坦》的脚本(由古腾堡项目[提供)，并基于 TFIDF 算法的输出生成每章中事件的感觉。为了做到这一点，我们首先要理解为什么 TFIDF 算法如此成功:](http://www.gutenberg.org/files/84/84-0.txt)

## TFIDF 算法

该算法分两部分工作。首先，我们计算我们的“**项频率**”。这将根据每个单词在文档中出现的次数对其进行排序(加权)——我们重复的次数越多，它越重要的可能性就越大。

然后，这个术语通过除以整个组中的单词数而被标准化。这是因为出现在一小段中的单词，例如标题，比成千上万行中的同一个单词具有更大的权重。

接下来，我们有'**逆文档频率**'部分。这很重要，因为它根据单词在特定文本片段中的个性对单词进行排序。在这里，我们将在文本的单个部分*中频繁使用的单词与在各处*中频繁使用的单词分开。这意味着只有当前文档/章节的本地标识词被标记为重要。我们使用以下公式进行计算:

```
*1 + log_exp ( number_documents / (document_frequency + 1))*
```

这两个术语结合起来提供了每个单词相对于其所在部分的重要性的权重。

## 预处理

与所有数据科学任务一样，我们需要为算法中使用的数据做好准备。将所需文本读入 python 后，我们可以用空格替换所有句点，并用 regex 模块(re)删除所有非单词字符:

```
text = text.replace('.',' ')
text = re.sub(r'\s+',' ',re.sub(r'[^\w \s]','',text) ).lower()
```

接下来，我们根据单词' *chapter* ' *(+一个数字)*分割数据集，尽管这可以是 LaTeX 中的`\section{.*}`或您选择的任何其他分隔符。

```
corpus = re.split('chapter \d+',text)
```

## 将数据输入算法

最简单的方法是使用 python 中的 [scikit-learn](https://scikit-learn.org/stable/modules/generated/sklearn.feature_extraction.text.TfidfVectorizer.html) 包。在这里，我们可以直接输入我们的数据，并获得它来计算每个组的单词排名:

```
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizervectorizer = TfidfVectorizer()
vectors = vectorizer.fit_transform(corpus)names = vectorizer.get_feature_names()
data = vectors.todense().tolist()# Create a dataframe with the results
df = pd.DataFrame(data, columns=names)
```

## 过滤掉停用词

在自然语言处理中，停用词是一组对最终输出没有什么意义的词。这些通常是常用词，如`I,a,all, any,and,the,them,it,dont,has`等。使用自然处理库`nltk`,我们可以过滤掉包含这些的所有列。

```
from nltk.corpus import stopwords
nltk.download('stopwords')
st = set(stopwords.words('english'))#remove all columns containing a stop word from the resultant dataframe. df = df[filter(lambda x: x not in list(st) , df.columns)]
```

## 打印出每章排名前 N 位的单词

这是通过选择每一行/数据集来完成的——由于我们之前的选择，这些代表不同的章节，并在打印之前选择 N 个排名最高的列。

```
N = 10;for i in df.iterrows():
    print(i[1].sort_values(ascending=False)[:N])
```

## 解释结果

现在，这就是我们看到我们的算法在预测关键词方面有多好的地方。因为有 25 个章节，我们在下面只展示了一些随机选择的章节。在该选择中，我们将章节提要与由 TFIDF 算法选择的排名最高的关键字进行比较，并决定它在描述现有事件方面的表现。

**第一章**弗兰肯斯坦设定了场景，描述了他的母亲如何发现了被一个意大利家庭遗弃的伊丽莎白，并收养了她。

```
mother      0.084748
beaufort    0.079967
child       0.068566
father      0.062509
orphan      0.054138
daughter    0.053630
poverty     0.049939
italy       0.047980
infant      0.043612
abode       0.043612
Name: 1, dtype: float64
```

**第 3 章**维克多(弗兰肯斯坦)离开日内瓦去上大学。在这里，自然哲学教授 Krempe 说服他，他学习炼金术的时间是一种浪费。结果，他参加了沃尔德曼教授举办的化学讲座，教授说服他从事科学研究。

```
science       0.074137
natural       0.071721
professor     0.065336
philosophy    0.059502
modern        0.054772
krempe        0.049489
waldman       0.049489
lecture       0.043557
chemistry     0.043557
principal     0.036860
Name: 3, dtype: float64
```

第 4 章维克托斯对他在理解生命科学方面的工作表现出极大的热情。以至于他对创造生命的迷恋成为他唯一的追求，这在很大程度上导致他忽视了生活中的朋友，并秘密进行令人讨厌的实验。

```
pursuit       0.053314
study         0.050588
life          0.046087
one           0.040652
corruption    0.039524
would         0.036956
science       0.036134
eagerness     0.035029
natural       0.034180
secret        0.034180
Name: 4, dtype: float64
```

这个怪物一直在学习一个小茅屋里的居民的语言和历史。费利克斯最近在这里遇到了萨菲。萨菲的母亲是一名信奉基督教的阿拉伯人，在嫁给萨菲的父亲之前曾被土耳其人奴役。

```
felix        0.164026
safie        0.136081
turk         0.112066
paris        0.087163
leghorn      0.084299
daughter     0.083509
deliverer    0.070249
lacey        0.045272
merchant     0.045272
christian    0.042149
Name: 14, dtype: float64
```

维克多和亨利穿越了英格兰和苏格兰，但是维克多开始迫不及待地想要开始他的工作，摆脱他和怪物的束缚。

```
oxford        0.060545
city          0.058641
edinburgh     0.048436
lakes         0.042927
scotland      0.042927
might         0.038131
visited       0.037911
matlock       0.036327
cumberland    0.036327
clerval       0.034506
Name: 19, dtype: float64
```

**第二十三章**伊丽莎白被怪物杀死。在未能说服裁判官让这个生物负责后，维克多·沃斯让他的生命去摧毁他的创造。

```
magistrate    0.048762
room          0.047945
exert         0.039379
pistol        0.039379
arms          0.036325
punishment    0.034900
rushed        0.034054
might         0.033068
elizabeth     0.032818
wandered      0.032088
Name: 23, dtype: float64
```

维克多失去了所有的家人，他追踪怪物来到北方的冰雪之地。在临终前，维克多讲述了他的故事，并恳求沃尔顿在他死后继续他的追求。

```
yet             0.051022
ice             0.048866
vengeance       0.037918
shall           0.033370
still           0.031682
die             0.030744
frankenstein    0.030744
would           0.027350
death           0.026679
feel            0.026679
Name: 24, dtype: float64
```

## 结论

看起来，至少对于这本玛丽·谢利的小说来说，*术语频率-逆文档频率*算法很容易使用，并作为提取每章描述性关键词的可靠方法。所以为什么不亲自尝试一下，看看你会有什么发现呢？

## 进一步的工作

这种和平的延续:将情感和情绪分析应用于《弗兰肯斯坦》,可以使用下面的链接找到。

[](/using-sentiment-analysis-to-explore-emotions-within-text-ae48e3e93999) [## 使用情感分析探索文本中的情感

### 应用两种自然语言处理技术来比较玛丽·雪莱的《弗兰肯斯坦》中的情感和 TF-IDF 关键词分析…

towardsdatascience.com](/using-sentiment-analysis-to-explore-emotions-within-text-ae48e3e93999)