# 文字游戏:矢量化、标记和情感分析

> 原文：<https://towardsdatascience.com/a-game-of-words-vectorization-tagging-and-sentiment-analysis-c78ff9a07e42?source=collection_archive---------9----------------------->

## 用自然语言处理分析《权力的游戏》第一册中的单词

![](img/b548549701471f0748be471123eb6085.png)

Image from [simisi1](https://pixabay.com/users/simisi1-5920903/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=4180794) from [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=4180794)

完全披露:我没有看过《权力的游戏》,但我希望通过分析文本来了解更多。如果你想了解更多关于基本文本处理的背景知识，你可以阅读我的[其他文章](https://medium.com/@madelinemccombe/text-processing-is-coming-c13a0e2ee15c)。所有 5 本书的文本都可以在 Kaggle 上找到[。在本文中，我将使用清理后的文本来解释以下概念:](https://www.kaggle.com/khulasasndh/game-of-thrones-books#005ssb.txt)

*   矢量化:单词袋、TF-IDF 和 Skip-Thought 矢量
*   矢量化后
*   词性标注
*   命名实体识别(NER)
*   组块和叮当声
*   情感分析
*   其他 NLP 包

# **矢量化**

目前，我有一个词条列表，但是我如何以一种机器能够理解的方式来组织它们呢？我将探索几个**向量化方法**，它们将单词列表转化为可用于不同机器学习方法的数字数组。这样做时，确保从文本中删除停用词和其他不必要的词是很重要的，这样创建的数组/向量系统将只有重要的维度可以建模。

一种方法叫做**单词包**，它定义了文本中包含的唯一单词的字典，然后找到文本中每个单词的计数。例如，如果我要从 *A Game of Thrones* 中收集一个独特的单词列表，然后按章节将整个列表拆分成单词，我将得到一个每行有一章、每个单词的计数跨列的数组。这种方法的缺点是它不能保持单词的顺序([坠落，死亡]和[坠落，爱]有非常不同的意思)，而且它不能捕捉单词的任何实际意思。此外，如果我们将文本按章节划分，包含更多单词的章节会无意中被赋予更大的权重，因为它们在整个行中有很高的计数。然而，这仍然是查看术语分布的好方法，并且如果您想要查看某个特定单词出现了多少次，这是很有用的。这是一个在《权力的游戏》文本上的实现，按章节划分:

```
from sklearn.feature_extraction.text import CountVectorizerbow = CountVectorizer()
BOW = bow.fit_transform(page_lemm)
bagOFwords = pd.DataFrame(BOW.toarray())
bagOFwords.columns = bow.get_feature_names()
```

在这个例子中，`page_lemm`是长度为 572(页数)的列表，每个元素是该页上的一串单词。`CountVectorizer()`函数自动对所有字符串中的单词进行标记和计数。在使用上面的代码之前，我做了一些后台的停用词和词条去除，你可以在[我的 github](https://github.com/madelinemccombe/LaunchDS/blob/master/TextAnalysis_GoT2.ipynb) 上看到。这段代码创建了一个 dataframe，其中每一行对应于书中的一章，每一列对应于文本中的一个单词。框架的主体包含每章每个单词的计数。

另一种解决单词袋问题的方法叫做 **TF-IDF，**或者术语频率-逆文档频率。TF-IDF 与前面的方法类似，只是每行每列中的值是根据文档中的术语数量和单词的相对稀有程度来调整的。**词频**等于一个词在文档中出现的次数除以文档中的总字数。**逆文档频率**计算语料库中所有文档中的稀有单词的权重，其中稀有单词具有高 IDF 分数，并且出现在语料库中所有文档中的单词具有接近零的 IDF。这使得那些可能有很多意思的词，即使很少，在最后的分析中仍然有影响力。这么想吧——是知道‘手’这个词在一本书的所有章节中都有使用会更好，还是知道‘死亡’只在其中 10 章出现会更有影响？TF 和 IDF 相乘得到最终的 TF-IDF 分数。在中可以找到这个过程的一步一步。Python 中的 scikit-learn 包(`sklearn`)有一个函数`TfidfVectorizer()`，它将为您计算 TF-IDF 值，如下所示:

```
from sklearn.feature_extraction.text import TfidfVectorizervectorizer = TfidfVectorizer()
got_tfidf = vectorizer.fit_transform(page_lemm)
tfidf = pd.DataFrame(got_tfidf.toarray())
tfidf.columns = vectorizer.get_feature_names()
```

如您所见，这两种方法的代码非常相似，采用相同的输入，但在 dataframe 中给出不同的内容。计算 TF-IDF 分数，而不是每个单词的计数。这里是根据平均单词数的前 10 个单词和根据平均 TF-IDF 分数的前 10 个单词的比较。有一些重叠，但是 TF-IDF 给出的字符名称的平均分比单词袋高。我强调了这两种方法之间不重叠的地方。

![](img/a1129f305ae02c69c958429353c1b381.png)

**跳跃思维** **矢量**是另一种矢量化方法，它使用神经网络中的迁移学习来预测句子的环境。**迁移学习**的概念是，机器可以将它从一项任务中“学到的”知识应用到另一项任务中。这是几乎所有机器学习技术背后的思想，因为我们试图让机器以比人类学习更快、更可量化的速度学习。特别是对于文本处理，其想法是一种算法建立一个神经网络，从数千本不同的书籍中学习，并找出句子结构，主题和一般模式。然后，这种算法可以应用于尚未阅读的书籍，它可以预测或模拟文本中的情感或主题。你可以[在这里](https://monkeylearn.com/blog/beginners-guide-text-vectorization/)阅读更多关于这种方法以及它与 TF-IDF 的比较。

另一种严重依赖神经网络的矢量化方法是 **word2vec** ，它计算两个单词之间的余弦相似度，并在空间中绘制单词，以便将相似的单词分组在一起。你可以在这里阅读这个方法的一个简洁的实现。

# **矢量化后**

现在你有了一个数字数组，接下来呢？使用单词包，您可以执行逻辑回归或其他分类算法来显示数组中哪些文档(行)最相似。当试图查看两篇文章在主题上是否相关时，这是很有帮助的。skip think Vectors 和 Word2Vec 都是基于文本内的含义来聚类单词，这是一种叫做**单词嵌入**的方法。这项技术很重要，因为它保留了单词之间的关系。尤其是在处理评论文本数据(任何带有数字评级的文本评论)时，这些技术可以产生关于消费者的感受和想法的有价值的见解。由于*权力的游戏*没有默认的分类值，我没有办法验证模型，我将在下面解释分析文本的替代方法。

# **位置标记**

**词性标注** (POS)是使用上下文线索将词性分配给列表中的每个单词。这很有用，因为同一个词有不同的词性，可能有两种完全不同的意思。例如，如果你有两个句子['一架飞机可以飞'和'房间里有一只苍蝇']，正确定义'飞'和'飞'是很重要的，以便确定这两个句子是如何相关的(也就是根本没有)。按词性标注单词可以让你进行组块和分块，这将在后面解释。一个重要的注意事项是，词性标注应该在标记化之后*和移除任何单词之前*立即进行，以便保留句子结构，并且更明显地知道单词属于哪种词性。一种方法是使用`nltk.pos_tag()`:

```
import nltkdocument = ' '.join(got1[8:10])
def preprocess(sent):
  sent = nltk.word_tokenize(sent)
  sent = **nltk.pos_tag**(sent)
  return sentsent = preprocess(document)
print(document)
print(sent)
```

*【他说“死了就是死了”。“我们和死人没关系。”，’“他们死了吗？”罗伊斯轻声问道。“我们有什么证据？”]*

*[，…(' ` '，' ` ')，('我们'，' PRP ')，('有'，' VBP ')，('无'，' DT ')，('商'，' NN '，('有'，' IN ')，(' the '，' DT '，('死'，' JJ '，('.', '.')、(“' '”、“' '”、…]*

这是上面创建的一个片段，你可以看到形容词被表示为“JJ”，名词被表示为“NN”，等等。该信息将在以后分块时使用。

# **命名实体识别**

有时，进一步定义特殊单词的词性会很有帮助，尤其是在尝试处理关于时事的文章时。除了名词之外，“伦敦”、“巴黎”、“莫斯科”和“悉尼”都是有特定含义的地点。同样的道理也适用于人名、组织名、时间名、金钱名、百分比名和日期名等等。这个过程在文本分析中很重要，因为它是理解文本块的一种方式。通常，要对文本应用 NER，必须先执行标记化和词性标注。nltk 包有两个内置的 NER 方法，这两个方法都在本文[中有很好的解释。](https://medium.com/explore-artificial-intelligence/introduction-to-named-entity-recognition-eda8c97c2db1)

执行 NER 并能够可视化和排序结果的另一个有用方法是通过 spaCy 包。一个很好的演练可以在找到[。我使用这种方法研究了 get 文本，得到了一些有趣的结果:](/named-entity-recognition-with-nltk-and-spacy-8c4a7d88e7da)

```
import spacy
from collections import Counter
import en_core_web_sm
nlp = en_core_web_sm.load()
from pprint import pprintdoc = nlp(document3)
pprint([(X.text, X.label_) for X in doc.ents])
```

*('乔治·r·r·马丁'，'人')，
('威玛·罗伊斯爵士'，'人')，
('五十'，'红衣主教')，
('威尔'，'人')，
('罗伊斯'，'人')，
('八日'，'日期')，
('九日'，'红衣主教')，
('威玛·罗伊斯'，'人'，* ***('加雷斯'，'人')，
('加雷斯'，'组织')，【组织】。***

在上面的代码中，`document3`是一个字符串中的*一个权力的游戏*的全文。这个包可以有效地发现和分类所有类型的实体。在 Gared 的一些例子上有点混乱(在某一点上它把他归类为 PERSON，另一个归类为 ORG，后来又归类为 WORK_OF_ART)。然而，总的来说，这比单纯的词性标注更能洞察文本的内容。下面是每种实体类型的匹配数和找到的顶级实体。不出所料，文中有很多名字。

```
labels = [x.label_ for x in doc.ents]
items = [x.text for x in doc.ents]print(Counter(labels))
print(Counter(items).most_common(5))
```

*计数器({ '红衣主教':340，'日期':169，' FAC': 34，' GPE': 195，'法律':2，' LOC': 24，'金钱':1，' NORP': 32，'序数':88，' ORG': 386，* ***'人':2307*** *，'产品':35，'数量':23，'时间':86，' WORK_OF_ART': 77})*

*(****乔恩****’、259)、(* ***奈德****’、247)、(* ***艾莉亚****’、145)、(* ***罗伯特****’、132)、(*

# **分块和分块**

组块和分块是从文本中提取有意义短语的两种方法。它们结合了词性标注和正则表达式来生成与所请求的短语结构相匹配的文本片段。组块的一个实现是找到提供不同名词描述的短语，称为**名词短语组块**。名词短语块的形式通常由决定/所有格、形容词、可能的动词和名词组成。如果你发现你的组块中有你不想要的部分，或者你宁愿在特定的位置拆分文本，一个简单的方法就是通过 **chinking** 。这定义了在分块时应该移除或分割的小块(称为缝隙)。我不打算在这篇文章中探索 chinking，但可以在这里找到一个教程。

使用 NLTK 进行特定类型分块的最简单方法是使用`nltk.RegexpParser(r‘<><><>’)`。这允许您指定您的名词短语公式，并且非常容易解释。每个< >引用一个单词的词性进行匹配，正常的正则表达式语法适用于每个< >。这非常类似于`nltk.Text().findall(r’<><><>’)`的概念，但是只是用了 POS 而不是实际的单词。在创建要解析的正则表达式字符串时需要注意的几件事是，词性缩写(NN =名词，JJ =形容词，PRP =介词，等等。)可能因软件包而异，有时最好从更具体的开始，然后扩大搜索范围。如果你现在非常迷茫，可以在这里找到这个概念的[。此外，在此之前温习一下句子结构和词性可能是个好主意，这样你就能完全理解组块分析的结果。下面是一个应用于掘地工具的示例:](https://medium.com/@gianpaul.r/tokenization-and-parts-of-speech-pos-tagging-in-pythons-nltk-library-2d30f70af13b)

```
document2 = ' '.join(got1[100:300])
big_sent = preprocess(document2) # POS tagging wordspattern = 'NP: {<DT>?<JJ>*<NN.?>+}'
cp = nltk.RegexpParser(pattern)
cs = cp.parse(big_sent)
print(cs)
```

*(……，(* ***NP 暮光/NNP*** *)深化/VBD。/.(* ***NP The/DT 万里无云/NN sky/NN*** *)转身/VBD (* ***NP a/DT 深/JJ 紫/NN*** *)，/，(****NP The/DT color/NN****)of/IN(****NP an/DT 老/JJ 青***

这是一个非常类似于 NER 的想法，因为你可以将 NN 或 NNP(名词或专有名词)组合在一起以找到物体的全名。此外，匹配的模式可以是词类的任意组合，这在查找特定种类的短语时非常有用。但是，如果词性标注不正确，您将无法找到您要查找的短语类型。我在这里只寻找名词短语，但是在我的 [github 代码](https://github.com/madelinemccombe/LaunchDS/blob/master/TextAnalysis_GoT2.ipynb)中包含了更多类型的组块。

# **情绪分析**

情感分析是计算机如何将目前为止所涉及的一切结合起来，并想出一种方法来传达一篇文章的整体主旨。它将句子、段落或文本的另一个子集中的单词与字典中的单词列表进行比较，并根据句子中单个单词的分类来计算情感得分。这主要用于分析评论、文章或其他观点，但我今天将把它应用于 GOT。我主要感兴趣的是看这本书的整体基调是积极的还是消极的，以及这种基调在各章之间的变化。进行情感分析有两种方法:你可以在以前分类的文本上训练和测试一个模型，然后用它来预测同一类型的新文本是正面还是负面，或者你可以简单地使用内置于函数中的现有词典来分析和报告正面或负面得分。下面是后者的一个例子或者是*《权力的游戏*第一页的一些句子:

```
from nltk.sentiment.vader import SentimentIntensityAnalyzer
nltk.download('vader_lexicon')sid = **SentimentIntensityAnalyzer**()
for sentence in sentences:
  print(sentence)
  ss = sid.polarity_scores(sentence)
  for k in sorted(ss):
    print('{0}: {1}, '.format(k, ss[k]), end='')
  print()
```

*……“死人让你害怕吗？”
compound:****-0.7717****，neg: 0.691，neu: 0.309，pos: 0.0，
威玛·罗伊斯爵士带着刚刚的一丝微笑问道。
复合:****0.3612****，neg: 0.0，neu: 0.783，pos: 0.217，
Gared 未上钩。
复合:****0.0****，neg: 0.0，neu: 1.0，pos: 0.0，…*

因为这是在分析一本书的文本，而不是评论的文本，所以很多句子将会有一个中性的复合得分(0)。然而，这完全符合我的目的，因为我只是在寻找这本书的语言随着时间推移的一般趋势。但是当提到死亡的时候，一个负分被应用，这仍然是一件好事。

**TextBlob** 是另一个有用的可以进行情感分析的包。一旦你把你的文本转换成一个 TextBlob 对象(`textblob.textBlob()`)，它就具有对纯文本进行标记化、词汇化、标签化的功能，并生成一个 WordNet，它可以量化单词之间的相似性。这个包有很多不同的文本对象，允许非常酷的转换，[在这里解释](https://textblob.readthedocs.io/en/dev/quickstart.html#sentiment-analysis)。甚至有一个`correct()`功能会尝试纠正拼写错误。在本文中，我不打算深入讨论其中的大部分，因为我正在尝试分析一本书，它通常应该具有正确的拼写和语法，然而，当处理特别混乱的文本数据时，这些工具中的许多将是有用的。下面是 TextBlob 在*权力的游戏*第一页的情绪分析版本:

```
from textblob import TextBlob
def detect_polarity(text):
    return TextBlob(text).sentiment
for sentence in sentences:
  print(sentence)
  print(detect_polarity(sentence))
```

*“死人让你害怕吗？”
情绪(polarity =****-0.2****，主观性=0.4)
威玛·罗伊斯爵士带着刚才那一丝微笑问道。
情绪(极性=****0.3****，主观性=0.1)
加雷斯没有上钩。
情绪(极性=****0.0****，主观性=0.0)*

nltk 和 textblob 的情感评分之间有相似性，但是 nltk 版本有更多的可变性，因为它是一个复合评分。或者，textblob 情感具有主观性得分，这有助于判断句子可以被分类的准确程度。下面是每种方法的页面情感分布。总的来说，Textblob 给出了更高的情感评级，而 nltk 与分数的差异更大。

![](img/fc1a71db269be219aa7fc46e6ad22691.png)

如果你试图从社交媒体文本或表情符号中收集情绪，VADER 情绪分析是一个专门为这项任务设计的工具。它内置了俚语(lol，omg，nah，meh 等。)甚至能看懂表情符号。如何使用它的一个很好的演练可以在这里找到。此外，如果 Python 不是您进行文本分析的首选语言，那么在不同的语言/软件中还有其他方法来进行情感分析，这里的[将对此进行解释](https://medium.com/@datamonsters/sentiment-analysis-tools-overview-part-2-7f3a75c262a3)。

# **其他 NLP 包**

在本文中，我只解释了`nltk`、`textblob`、`vaderSentiment`、`spacy`和`sklearn`包的功能，但是根据您试图完成的任务，它们有许多优点和缺点。其他一些可能更适合你的任务是多语种和 Genism。 [**Polyglot**](https://pypi.org/project/polyglot/) 以具有分析大量语言的能力而闻名(根据任务支持 16–196 种)。 [**Genism**](https://pypi.org/project/gensim/) 主要用于对文本的无监督学习任务，并且需要用不同的包进行任何预处理。你可以在这里找到所有这些信息的图表。

# **结论**

我从撰写本文中学到的一个关键点是，完成一项任务至少有三种方法，而确定最佳选择取决于您使用的数据类型。有时候你会优先考虑计算时间，而其他时候你会需要一个可以很好地进行无监督学习的包。文本处理是一门迷人的科学，我迫不及待地想看看它在未来几年里将我们引向何方。在本文中，我介绍了矢量化以及它如何确定文本之间的相似性，标记允许将意义附加到单词上，以及情感分析，它可以大致判断文本的积极或消极程度。我从《权力的游戏》中收集了很多见解，比如有很多死亡，先生是一个常见的头衔，拼写为 Ser，龙的例子没有我被引导相信的那么多。不过，我现在可能被说服去看书了！我希望你喜欢这篇文章！

我的代码副本，有更多的例子和解释，可以在 github 上找到[！请随意获取和使用代码。](https://github.com/madelinemccombe/LaunchDS/blob/master/TextAnalysis_GoT2.ipynb)

我的另一篇文章《文本预处理来了》可以[在这里找到](/text-processing-is-coming-c13a0e2ee15c)！