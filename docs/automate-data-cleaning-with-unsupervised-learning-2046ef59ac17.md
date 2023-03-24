# 通过无监督学习自动清理数据

> 原文：<https://towardsdatascience.com/automate-data-cleaning-with-unsupervised-learning-2046ef59ac17?source=collection_archive---------16----------------------->

## 为您的 NLP 项目清理文本从未如此有趣和容易！

![](img/ef5ec588697e57712ca159c7a9d0b3ae.png)

Photo by [Karim MANJRA](https://unsplash.com/@karim_manjra?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

我喜欢处理文本数据。至于计算机视觉，现在 NLP 中有很多现成的资源和开源项目，我们可以直接下载或使用。其中一些很酷，允许我们加快速度，把我们的项目带到另一个水平。

最重要的是，我们不能忘记，所有这些工具都不是魔法。他们中的一些人宣称有很高的表现，但如果我们不允许他们做到最好，他们就什么都不是。这是一个严酷的事实，尤其是在 NLP 领域，我们经常面临数据中很难去除的高水平噪声(比处理图像更困难)。

在这篇文章中，我提出了我的解决方案来提高我所掌握的文本数据的质量。我开发了一个工作流程，旨在以无人监管的方式自动清理数据。我说*‘自动’*是因为如果我们不得不一直手动检查数据以理解模型输出的内容，那么遵循无监督的方法是没有用的。我们需要确定性，不想浪费时间。

# 数据集

在这里，我的目标是“清洗”研究论文。在这种情况下，清理意味着保留文本的相关部分，丢弃不包含有价值信息的部分。在机器学习论文的子集中，我认为文本句子是相关的部分，而包含大多数代数和符号的部分是无用的。直接访问这些片段可以简化诸如主题建模、摘要、聚类等任务。

我在 Kaggle 上找到了 NIPS 2015 论文’[数据集。我直接下载 *Paper.csv.* 这个数据集包含了 NIPS 2015 (](https://www.kaggle.com/benhamner/nips-2015-papers#Papers.csv) [神经信息处理系统](https://nips.cc/))的 403 篇论文；世界顶级机器学习会议之一，主题涵盖从深度学习和计算机视觉到认知科学和强化学习)。

![](img/81180fec0dfe018477cc59d8f71f332a.png)

Topic Modelling on NIPS papers by [Kaggle](https://www.kaggle.com/benhamner/nips-2015-papers#Papers.csv)

# 工作流程

*   选择一篇论文，应用基本的预处理将它分成句子/段落
*   创建句子的加权嵌入
*   嵌入空间中句子的无监督模型化
*   噪声文本的检测和表示

这种能力是通过使用预训练的单词嵌入模型及其相对较深的词汇表来提供的。

## 预处理

我计算了一个标准的预处理，从感兴趣的单个文件中删除数字，而不是字母数字字符和停用词。这里的要点是通过句子/段落的创建来提供的。有一些软件包提供这种功能；我试过了，但是他们并没有说服我。因此，当我处理来自整个文档的文本语料库时，我自己编写了更好地满足我需要的简单函数。

```
from nltk.corpus import stopwords
from gensim.parsing.preprocessing import *stop_words = set(stopwords.words('english'))def create_sentence(text):
    output = []
    for s in re.split('\.\n|\.\r\n|\. ', text):
        output.extend(re.split('\n\n|\r\n\r\n', s))
    return outputdef clean_sentence(sentence):
    sentence = strip_numeric(strip_non_alphanum(sentence))
    sentence = sentence.lower().split()
    return [w for w in sentence if w not in stop_words]
```

## 句子嵌入

一个好的句子的数字表示是整个程序成功的基础。我们希望有意义的句子在嵌入空间中彼此相似且更近，同时我们希望有噪声的句子远离有意义的句子，并且它们之间也相似。为了实现这一关键步骤，我们使用了强大的预训练单词嵌入，我选择了 GloVe。

我们创建句子嵌入作为单个单词嵌入的加权平均值。通过计算由句子组成的论文语料库上的 TfIdf 来给出权重。在组合期间，对于特定句子中的每个单词，我们选择相对手套嵌入(如果存在)并将其乘以 TfIdf 单词权重。每个句子中加权单词嵌入的总和被归一化，除以句子中 TfIdf 单词权重的总和。

```
emb_dim = 300
model = KeyedVectors.load_word2vec_format('GoogleNews-vectors-negative300.bin', binary=True)tfidf = TfidfVectorizer()
tfidf.fit(list(map(' '.join, cleaned_corpus)))
```

为了让这个过程变得简单和轻松，我选择了 Gensim 来加载和操作手套。Gensim 需要*。手套的 bin* 格式(此处有[和其他文件格式](https://www.kaggle.com/pkugoodspeed/nlpword2vecembeddingspretrained))；随意使用你喜欢的阅读方法。

```
def calculate_emb(clean_text, original_text):

    tfidf_feat = tfidf.get_feature_names()
    tfidf_emb = tfidf.transform(list(map(' '.join, clean_text)))
    final_sent_emb = []
    final_sent = []

    for row, sent in enumerate(tqdm(clean_text)):
        sent_vec = np.zeros(emb_dim)
        weight_sum = 0
        for word in sent:
            try:
                vec = model[word]
                tfidf_score = tfidf_emb[row, tfidf_feat.index(word)]
                sent_vec += (vec * tfidf_score)
                weight_sum += tfidf_score
            except:
                passif sum(sent_vec)!=0:
        sent_vec /= weight_sum
        final_sent.append(original_text[row])
        final_sent_emb.append(sent_vec)

    final_sent_emb = np.asarray(final_sent_emb)
    final_sent = np.asarray(final_sent)    

    return final_sent_emb, final_sent
```

用这种方法计算句子嵌入，我们已经自由地得出了第一个重要的结果。手套词汇中不存在的单词(3000000 个唯一单词)不被考虑在内(*如果不在手套中它就不存在*)，即它们是非常不常见的术语，我们可以将它们视为噪音。现在想象一个全是不常见符号的句子，它以全零的嵌入结束，我们可以立即排除它。

## 无监督模型化和表示

如上所述，为语料库中的每个句子计算句子嵌入的创建。我们以一个维数为(N，300)的数组结束，其中 N 是句子的数量，300 是手套嵌入维数。为了便于管理，我们用主成分分析法对它们进行降维，并在降维后的空间上应用无监督异常检测算法。我们的异常点显然位于远离密度中心的地方。

![](img/3e70dd48f6a6095ba77e142a7e88a4f5.png)

2D representation of sentence embeddings

这些异常点(句子)的检测对于隔离林来说是一项完美的任务。这个集合模型在子群中操作空间的划分:将一个样本从其他样本中分离出来所需的分裂次数越少，这个观察结果成为异常的概率就越高。

```
pca = PCA(n_components=2)
X = pca.fit_transform(final_sent_emb)IF = IsolationForest(contamination=0.3, behaviour='new')
IF.fit(X)plt.figure(figsize=(8,6))
plt.scatter(X.T[0], X.T[1], c=IF.predict(X), cmap='viridis')
```

![](img/7fb3717234a0df80857045419d5e8a06.png)

Isolation Forest at work on sentence embeddings

我们的工作已经完成，我们只需要检查结果！黄色部分是有价值的句子和嘈杂的文字部分。为了达到这个结果，我们只需要设置“污染参数”,它告诉我们丢弃信息的严重程度。下面我附上了我刚刚分析的论文中的一个例子。第一个片段显示了有价值的句子，其中包含有用的信息。

![](img/c543579e81cf7f441b8f7a4a5d407457.png)

Good sentences

第二个片段是一个句子的集合，其中可能包含很多噪音。

![](img/dfe498ee25e8168b6c430630449bcde8.png)

Bad sentences

# 摘要

在这篇文章中，我开发了一个自动清理文本数据的解决方案。给定一个文本语料库，我们能够选择最有价值的句子，扔掉嘈杂的文本部分。这种解决方案的目标是计算成本不高；我们使用了预先训练的单词嵌入，并拟合了一个简单的异常句子检测算法。整个工作流程可以扩展到文本语料库的收集，始终遵循无监督的方法。

[**查看我的 GITHUB 回购**](https://github.com/cerlymarco/MEDIUM_NoteBook)

保持联系: [Linkedin](https://www.linkedin.com/in/marco-cerliani-b0bba714b/)