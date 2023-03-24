# 用 Doc2Vec 实现多类文本分类

> 原文：<https://towardsdatascience.com/implementing-multi-class-text-classification-with-doc2vec-df7c3812824d?source=collection_archive---------9----------------------->

![](img/8ec03ba38a32452bae40bf7c63f88d3e.png)

Image by [Gerd Altmann](https://pixabay.com/users/geralt-9301/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=1668918) from [Pixabay](https://pixabay.com/?utm_source=link-attribution&amp;utm_medium=referral&amp;utm_campaign=image&amp;utm_content=1668918) ([Image Link](https://pixabay.com/illustrations/film-negative-photographs-slides-1668918/))

# 介绍

在本文中，您将学习如何在使用 Doc2Vec 表示文档时将文本文档分类到不同的类别。我们将通过一个简单易懂的例子来了解这一点，这个例子使用 Doc2vec 作为特征表示，使用逻辑回归作为分类算法，通过流派对电影情节进行分类。 [**电影数据集**](https://github.com/RaRe-Technologies/movie-plots-by-genre) 包含简短的电影情节描述，它们的标签代表类型。数据集中有六个**流派**:

1.  科幻小说
2.  行动
3.  喜剧
4.  幻想
5.  动画
6.  浪漫性

# 什么是 Doc2Vec，我们为什么需要它？

那么，为什么要选择 **doc2vec** 表示法而不是使用广为人知的 bag-of-words( **BOW** )方法呢？对于复杂的文本分类算法， **BOW** 不适合，因为它缺乏捕捉文本中单词的语义和句法顺序的能力。因此，将它们用作机器学习算法的特征输入不会产生显著的性能。另一方面，Doc2Vec 能够检测单词之间的关系，并理解文本的语义。Doc2Vec 是一种无监督算法，它为段落/文档/文本学习固定长度的特征向量。为了理解 **doc2vec** 的基本工作，需要理解 **word2vec** 如何工作，因为它使用相同的逻辑，除了文档特定向量是添加的特征向量。关于这个的更多细节，你可以阅读这个[博客](https://medium.com/scaleabout/a-gentle-introduction-to-doc2vec-db3e8c0cce5e)。现在我们知道了为什么使用它，以及 doc2vec 将如何在这个程序中使用，我们可以进入下一个阶段，实际实现分类器。

让我们开始构建一个电影情节分类器吧！！

# **实施**

这涉及到以下三个主要部分:

1.  加载和准备文本数据
2.  使用 **doc2vec** 模型获取特征向量
3.  训练分类器

以下是输入 csv 文件的前三行:

```
,movieId,plot,tag0,1,"A little boy named Andy loves to be in his room, playing with his toys, especially his doll named ""Woody"". But, what do the toys do when Andy is not with them, they come to life. Woody believes that he has life (as a toy) good. However, he must worry about Andy's family moving, and what Woody does not know is about Andy's birthday party. Woody does not realize that Andy's mother gave him an action figure known as Buzz Lightyear, who does not believe that he is a toy, and quickly becomes Andy's new favorite toy. Woody, who is now consumed with jealousy, tries to get rid of Buzz. Then, both Woody and Buzz are now lost. They must find a way to get back to Andy before he moves without them, but they will have to pass through a ruthless toy killer, Sid Phillips.",animation1,2,"When two kids find and play a magical board game, they release a man trapped for decades in it and a host of dangers that can only be stopped by finishing the game.",fantasy
```

导入所需的库:

我们使用多处理技术来利用所有内核，以便通过 Doc2Vec 进行更快的训练。tqdm 包用于在训练时显示进度条。我们对 Doc2Vec 使用 gensim 包。出于分类目的，使用来自 scikit-learn 的逻辑回归。NLTK 包用于标记化任务。

```
from gensim.test.utils import common_texts
from gensim.models.doc2vec import Doc2Vec, TaggedDocument
from sklearn.metrics import accuracy_score, f1_score
from sklearn.model_selection import train_test_splitfrom sklearn.linear_model import LogisticRegression
from sklearn import utils
import csv
from tqdm import tqdm
import multiprocessingimport nltk
from nltk.corpus import stopwords
```

## 1.阅读和准备文本数据

以下代码用于从 csv 读取数据，并用于标记化功能，在创建培训和测试文档作为 **doc2vec** 模型的输入时，将使用这些代码。数据有 2448 行，我们选择前 2000 行用于训练，其余的用于测试。

```
tqdm.pandas(desc="progress-bar")# Function for tokenizingdef tokenize_text(text):
    tokens = []
    for sent in nltk.sent_tokenize(text):
        for word in nltk.word_tokenize(sent):
            if len(word) < 2:
                continue
            tokens.append(word.lower())
    return tokens# Initializing the variablestrain_documents = []
test_documents = []
i = 0# Associating the tags(labels) with numberstags_index = {'sci-fi': 1 , 'action': 2, 'comedy': 3, 'fantasy': 4, 'animation': 5, 'romance': 6}#Reading the fileFILEPATH = 'data/tagged_plots_movielens.csv'
with open(FILEPATH, 'r') as csvfile:
with open('data/tagged_plots_movielens.csv', 'r') as csvfile:
    moviereader = csv.reader(csvfile, delimiter=',', quotechar='"')
    for row in moviereader:
        if i == 0:
            i += 1
            continue
        i += 1
        if i <= 2000: train_documents.append(          TaggedDocument(words=tokenize_text(row[2]), tags=[tags_index.get(row[3], 8)] )) else:
            test_documents.append( TaggedDocument(words=tokenize_text(row[2]),
 tags=[tags_index.get(row[3], 8)]))print(train_documents[0])
```

第一行的 training_document 输出是 TaggedDocument 对象。这将标记显示为 TaggedDocument 的第一个参数，labelID 显示为第二个参数(5: Animation)。

```
TaggedDocument(['little', 'boy', 'named', 'andy', 'loves', 'to', 'be', 'in', 'his', 'room', 'playing', 'with', 'his', 'toys', 'especially', 'his', 'doll', 'named', '``', 'woody', "''", 'but', 'what', 'do', 'the', 'toys', 'do', 'when', 'andy', 'is', 'not', 'with', 'them', 'they', 'come', 'to', 'life', 'woody', 'believes', 'that', 'he', 'has', 'life', 'as', 'toy', 'good', 'however', 'he', 'must', 'worry', 'about', 'andy', "'s", 'family', 'moving', 'and', 'what', 'woody', 'does', 'not', 'know', 'is', 'about', 'andy', "'s", 'birthday', 'party', 'woody', 'does', 'not', 'realize', 'that', 'andy', "'s", 'mother', 'gave', 'him', 'an', 'action', 'figure', 'known', 'as', 'buzz', 'lightyear', 'who', 'does', 'not', 'believe', 'that', 'he', 'is', 'toy', 'and', 'quickly', 'becomes', 'andy', "'s", 'new', 'favorite', 'toy', 'woody', 'who', 'is', 'now', 'consumed', 'with', 'jealousy', 'tries', 'to', 'get', 'rid', 'of', 'buzz', 'then', 'both', 'woody', 'and', 'buzz', 'are', 'now', 'lost', 'they', 'must', 'find', 'way', 'to', 'get', 'back', 'to', 'andy', 'before', 'he', 'moves', 'without', 'them', 'but', 'they', 'will', 'have', 'to', 'pass', 'through', 'ruthless', 'toy', 'killer', 'sid', 'phillips'], [5])
```

## 2.从 doc2vec 模型中获取特征向量

接下来，我们初始化 [gensim doc2vec 模型](https://radimrehurek.com/gensim/models/doc2vec.html)并训练 30 个时期。这个过程非常简单。Doc2Vec 体系结构也有两个类似 word2vec 的算法，它们是这两个算法的对应算法，即“连续单词包”(CBOW)和“Skip-Gram”(SG)。doc2vec 中的一种算法称为段落向量分布单词袋(PV-DBOW ),它类似于 word2vec 中的 SG 模型，只是增加了额外的段落 id 向量。这里训练神经网络来预测给定段落中周围单词的向量和基于段落中给定单词的段落 id 向量。第二种算法是段落向量(PV-DM)，类似于词向量中的 CBOW。

**doc2vec** 模型的几个重要参数包括:

`dm` ({0，1}，可选)1: PV-DM，0: PV-DBOW

`vector_size`特征向量的维数(我们选择 300)

`workers`这是我们分配了核心数的线程数

其余参数详情可在[这里](https://radimrehurek.com/gensim/models/doc2vec.html)找到。初始化之后，我们使用`train_documents`构建词汇表

```
cores = multiprocessing.cpu_count()

model_dbow = Doc2Vec(dm=1, vector_size=300, negative=5, hs=0, min_count=2, sample = 0, workers=cores, alpha=0.025, min_alpha=0.001)
model_dbow.build_vocab([x for x in tqdm(train_documents)])train_documents  = utils.shuffle(train_documents)
model_dbow.train(train_documents,total_examples=len(train_documents), epochs=30)def vector_for_learning(model, input_docs):
    sents = input_docs
    targets, feature_vectors = zip(*[(doc.tags[0], model.infer_vector(doc.words, steps=20)) for doc in sents])
    return targets, feature_vectorsmodel_dbow.save('./movieModel.d2v')
```

## 3.训练分类器

最后，使用上述特征向量构建函数训练逻辑回归分类器。这里，我们在训练时使用为`train_documents`生成的特征向量，并在预测阶段使用`test_documents`的特征向量。

```
y_train, X_train = vector_for_learning(model_dbow, train_documents)
y_test, X_test = vector_for_learning(model_dbow, test_documents)

logreg = LogisticRegression(n_jobs=1, C=1e5)
logreg.fit(X_train, y_train)
y_pred = logreg.predict(X_test)print('Testing accuracy for movie plots%s' % accuracy_score(y_test, y_pred))
print('Testing F1 score for movie plots: {}'.format(f1_score(y_test, y_pred, average='weighted')))
```

`dm=1`时的输出如下:

```
Testing accuracy 0.42316258351893093
Testing F1 score: 0.41259684559985876
```

这种准确性只能通过很短的文本的少量记录来获得，因此，这可以通过添加更好的功能来提高，如 n-grams，使用停用词来消除噪声。

# **进一步发展**

在此基础上，您可以更容易地使用其他数据集进行实验，并且更改 doc2vec 的算法就像更改`dm`参数一样简单。希望这有助于您使用 Doc2Vec 开始您的第一个文本分类项目！

作者:[迪皮卡·巴德](https://www.linkedin.com/in/dipika-baad-154a2858/)