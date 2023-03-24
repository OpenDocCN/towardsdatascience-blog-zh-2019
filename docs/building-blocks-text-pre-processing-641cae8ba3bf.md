# 构建块:文本预处理

> 原文：<https://towardsdatascience.com/building-blocks-text-pre-processing-641cae8ba3bf?source=collection_archive---------7----------------------->

在我们系列的上一篇文章中，我们介绍了自然语言处理的概念，你可以[在这里](https://medium.com/@shashank.kapadia/introduction-to-natural-language-processing-nlp-2a8fae09ed03)阅读，现在你大概也想自己尝试一下吧？太好了！事不宜迟，让我们深入研究统计自然语言处理的构建模块。

在本文中，我们将介绍关键概念、Python 中的实际实现以及应用时要记住的挑战。完整的代码可以在 GitHub 的 [Jupyter 笔记本上获得。](https://github.com/kapadias/mediumposts/blob/master/nlp/Building%20Blocks%20Text%20Pre-Processing.ipynb)

![](img/c266c8a7481f180cb461b0210b57cd1d.png)

# **文字规范化**

规范化文本意味着在将文本转换为更高级建模的特征之前，将其转换为更方便的标准形式。把这一步想象成把人类可读的语言转换成机器可读的形式。

标准化文本的标准框架包括:

1.  标记化
2.  停止单词删除
3.  形态标准化
4.  配置

> *数据预处理由许多步骤组成，任何数量的步骤可能适用于也可能不适用于给定的任务。更一般地说，在本文中，我们将讨论一些预先确定的文本主体，并执行一些基本的转换分析，这些分析可用于执行更进一步的、更有意义的自然语言处理[1]*

# **标记化**

给定一个字符序列和一个已定义的文档单元(文本的简介)，*标记化*的任务是将它分割成小块，称为 ***标记*** ，可能同时丢弃某些字符/单词，如标点符号【2】。通常，有两种类型的标记化:

1.  **单词标记化:**用于通过唯一的*空格字符分隔单词。*根据应用程序的不同，单词标记化也可以标记多单词表达式，如 *New York* 。这往往与一个叫做*命名实体识别*的过程紧密相关。在本教程的后面，我们将看看*搭配(短语)建模*，这有助于解决这个挑战的一部分
2.  句子分词和单词分词一样，是文本处理中至关重要的一步。这通常是基于标点符号，如“.”, "?", "!"因为它们倾向于标记句子的边界

**挑战:**

*   缩写的使用可以促使标记器检测没有边界的句子边界。
*   数字、特殊字符、连字符和大写。在“不要”、“我愿意”、“约翰的”这些表达中，我们有一个、两个还是三个代币？

**实现示例:**

```
from nltk.tokenize import sent_tokenize, word_tokenize#Sentence Tokenization
print ('Following is the list of sentences tokenized from the sample review\n')sample_text = """The first time I ate here I honestly was not that impressed. I decided to wait a bit and give it another chance. 
I have recently eaten there a couple of times and although I am not convinced that the pricing is particularly on point the two mushroom and 
swiss burgers I had were honestly very good. The shakes were also tasty. Although Mad Mikes is still my favorite burger around, 
you can do a heck of a lot worse than Smashburger if you get a craving"""tokenize_sentence = sent_tokenize(sample_text)print (tokenize_sentence)
print ('---------------------------------------------------------\n')
print ('Following is the list of words tokenized from the sample review sentence\n')
tokenize_words = word_tokenize(tokenize_sentence[1])
print (tokenize_words)
```

**输出:**

```
Following is the list of sentences tokenized from the sample review

['The first time I ate here I honestly was not that impressed.', 'I decided to wait a bit and give it another chance.', 'I have recently eaten there a couple of times and although I am not convinced that the pricing is particularly on point the two mushroom and \nswiss burgers I had were honestly very good.', 'The shakes were also tasty.', 'Although Mad Mikes is still my favorite burger around, \nyou can do a heck of a lot worse than Smashburger if you get a craving']
---------------------------------------------------------

Following is the list of words tokenized from the sample review sentence

['I', 'decided', 'to', 'wait', 'a', 'bit', 'and', 'give', 'it', 'another', 'chance', '.']
```

# 停止单词删除

通常，作为停用词移除过程的一部分，有几个普遍存在的词被完全从词汇表中排除，这些词看起来对帮助分析的目的没有什么价值，但是增加了特征集的维度。通常有两个原因促使这种移除。

1.  **不相关:**只允许对有内容的单词进行分析。停用词，也称为空词，因为它们通常没有太多意义，会在分析/建模过程中引入噪声
2.  **维数:**去除停用词还可以显著减少文档中的标记，从而降低特征维数

**挑战:**

在停用字词删除过程之前将所有字符转换为小写字母会在文本中引入歧义，有时会完全改变文本的含义。例如，“美国公民”将被视为“美国公民”，或者“it 科学家”将被视为“IT 科学家”。由于*us*和*it*通常都被认为是停用词，这将导致不准确的结果。因此，通过*词性*标记步骤，通过识别“us”和“IT”不是上述示例中的代词，可以改进关于停用词处理的策略。

**实施例:**

```
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize# define the language for stopwords removal
stopwords = set(stopwords.words("english"))
print ("""{0} stop words""".format(len(stopwords)))tokenize_words = word_tokenize(sample_text)
filtered_sample_text = [w for w in tokenize_words if not w in stopwords]print ('\nOriginal Text:')
print ('------------------\n')
print (sample_text)
print ('\n Filtered Text:')
print ('------------------\n')
print (' '.join(str(token) for token in filtered_sample_text))
```

**输出:**

```
179 stop words

Original Text:
------------------

The first time I ate here I honestly was not that impressed. I decided to wait a bit and give it another chance. 
I have recently eaten there a couple of times and although I am not convinced that the pricing is particularly on point the two mushroom and 
swiss burgers I had were honestly very good. The shakes were also tasty. Although Mad Mikes is still my favorite burger around, 
you can do a heck of a lot worse than Smashburger if you get a craving

 Filtered Text:
------------------

The first time I ate I honestly impressed . I decided wait bit give another chance . I recently eaten couple times although I convinced pricing particularly point two mushroom swiss burgers I honestly good . The shakes also tasty . Although Mad Mikes still favorite burger around , heck lot worse Smashburger get craving
```

# 形态标准化

一般来说，形态学是研究单词是如何由更小的有意义的单位组成的，*词素。*比如*狗*由两个语素组成: ***狗*** 和 ***s***

两种常用的文本规范化技术是:

1.  **词干化:**该过程旨在识别单词的词干，并用它来代替单词本身。提取英语词干最流行的算法是波特算法，这种算法已经被反复证明是非常有效的。整个算法太长太复杂，无法在这里呈现[3]，但是你可以在这里找到细节
2.  **词汇化:**这个过程指的是使用词汇和词形分析来正确地做事情，通常旨在仅移除*屈折*词尾，并返回单词的基本或词典形式，这被称为 ***词汇*** 。

> *如果遇到令牌* **saw** *，词干可能只返回* **s** *，而词汇化将根据令牌是用作动词还是名词【4】*来尝试返回 **see** *或* **saw**

***实施例:***

```
*from nltk.stem import PorterStemmer
from nltk.stem import WordNetLemmatizer
from nltk.tokenize import word_tokenizeps = PorterStemmer()
lemmatizer = WordNetLemmatizer()tokenize_words = word_tokenize(sample_text)stemmed_sample_text = []
for token in tokenize_words:
    stemmed_sample_text.append(ps.stem(token))lemma_sample_text = []
for token in tokenize_words:
    lemma_sample_text.append(lemmatizer.lemmatize(token))

print ('\nOriginal Text:')
print ('------------------\n')
print (sample_text)print ('\nFiltered Text: Stemming')
print ('------------------\n')
print (' '.join(str(token) for token in stemmed_sample_text))print ('\nFiltered Text: Lemmatization')
print ('--------------------------------\n')
print (' '.join(str(token) for token in lemma_sample_text))*
```

***输出:***

```
*Original Text:
------------------

The first time I ate here I ***honestly*** was not that impressed. I decided to wait a bit and give it ***another*** ***chance***. I have recently eaten there a ***couple*** of times and although I am not convinced that the pricing is particularly on point the two mushroom and swiss burgers I had were honestly very good. The shakes were also tasty. Although Mad Mikes is still my favorite burger around, you can do a heck of a lot worse than ***Smashburger*** if you get a ***craving***.

Filtered Text: Stemming:
------------------

the first time I ate here I ***honestli*** wa not that impress . I decid to wait a bit and give it ***anoth*** ***chanc*** . I have recent eaten there a ***coupl*** of time and although I am not convinc that the price is particularli on point the two mushroom and swiss burger I had were honestli veri good . the shake were also tasti . although mad mike is still my favorit burger around , you can do a heck of a lot wors than ***smashburg*** if you get a ***crave*** .Filtered Text: Lemmatization
--------------------------------

The first time I ate here I ***honestly*** wa not that impressed . I decided to wait a bit and give it ***another*** ***chance*** . I have recently eaten there a ***couple*** of time and although I am not convinced that the pricing is particularly on point the two mushroom and swiss burger I had were honestly very good . The shake were also tasty . Although Mad Mikes is still my favorite burger around , you can do a heck of a lot worse than ***Smashburger*** if you get a ***craving*** .*
```

***挑战:***

*通常，完整的形态学分析至多产生非常有限的分析益处。从相关性和维度缩减的角度来看，两种形式的标准化都不能提高语言信息的总体性能，至少在以下情况下是这样:*

***实施例:***

```
*from nltk.stem import PorterStemmer
words = ["operate", "operating", "operates", "operation", "operative", "operatives", "operational"]ps = PorterStemmer()for token in words:
    print (ps.stem(token))*
```

***输出:***

```
*oper
oper
oper
oper
oper
oper
oper*
```

> *作为可能出错的一个例子，请注意，波特词干分析器将以下所有单词的词干转换为 ***oper****

*然而，由于*以各种形式操作*是一个常见的动词，我们预计会失去相当的精确性[4]:*

*   *运营和研究*
*   *操作和系统*
*   *手术和牙科*

*对于这种情况，使用词汇归类器并不能完全解决问题，因为在特定的搭配中使用了特定的屈折形式。从术语规范化中获得更好的价值更多地取决于单词使用的语用问题，而不是语言形态学的形式问题[4]*

*对于所有用于生成上述结果的代码，点击[此处](https://github.com/kapadias/mediumposts/blob/master/nlp/Building%20Blocks%20Text%20Pre-Processing.ipynb)。*

*就是这样！现在你知道梯度下降，线性回归和逻辑回归。"*

# *即将到来..*

*在下一篇文章中，我们将详细讨论搭配(短语)建模的概念，并一起演练它的实现。敬请关注，继续学习！*

# ***参考文献:***

*[1]预处理文本数据的一般方法— KDnuggets。[https://www . kdnugges . com/2017/12/general-approach-预处理-text-data.html](https://www.kdnuggets.com/2017/12/general-approach-preprocessing-text-data.html)*

*[2]记号化—斯坦福 NLP 组。[https://NLP . Stanford . edu/IR-book/html/html edition/token ization-1 . html](https://nlp.stanford.edu/IR-book/html/htmledition/tokenization-1.html)*

*[3]牛疾病中的文本挖掘—ijcaonline.org。[https://www.ijcaonline.org/volume6/number10/pxc3871454.pdf](https://www.ijcaonline.org/volume6/number10/pxc3871454.pdf)*

*[4]词干化和词汇化—斯坦福大学 NLP 组。[https://NLP . Stanford . edu/IR-book/html/html edition/stemming-and-lemma tization-1 . html](https://nlp.stanford.edu/IR-book/html/htmledition/stemming-and-lemmatization-1.html)*

**如果你有任何反馈，请在本文中发表评论，在* [*LinkedIn*](https://www.linkedin.com/in/shashankkapadia/) *上给我发消息，或者给我发邮件(shmkapadia[at]gmail.com)**