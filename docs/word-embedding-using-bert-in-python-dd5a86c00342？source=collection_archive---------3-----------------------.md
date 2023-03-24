# Python 中使用 BERT 的单词嵌入

> 原文：<https://towardsdatascience.com/word-embedding-using-bert-in-python-dd5a86c00342?source=collection_archive---------3----------------------->

## 使用深度学习 NLP 模型(BERT)将单词嵌入向量，只需几行 Python 代码

![](img/4398fb46521d0a1974df8296e47bbab6.png)

# 单词嵌入:它们是什么？

在 NLP 的世界中，以向量形式或单词嵌入来表示单词或句子打开了各种潜在应用的大门。这种将单词编码成向量的功能是 NLP 任务的强大工具，例如计算单词之间的语义相似度，利用它可以构建语义搜索引擎。例如，这里有一个单词嵌入的应用程序，通过它 [Google 可以更好地理解使用 BERT 的搜索查询。可以说，它是最强大的语言模型之一，在机器学习社区中变得非常流行。](https://www.blog.google/products/search/search-language-understanding-bert/)

使用大的句子语料库对 BERT(来自变压器的双向编码器表示)模型进行预训练。简而言之，训练是通过屏蔽句子中的几个单词(根据论文作者的说法，大约 15%的单词)并让模型预测被屏蔽的单词来完成的。随着模型训练预测，它学习产生单词的强大内部表示，如单词嵌入。今天，我们将看到如何轻松地建立和运行 BERT 模型，并将单词编码到单词嵌入中。

# BERT 单词嵌入模型设置

使用 Pytorch 和 Tensorflow 运行 BERT 模型有一套可用的选项。但是，为了让您非常容易地接触到 BERT 模型，我们将使用 Python 库来帮助我们快速设置它！

B *ert-as-a-service* 是一个 Python 库，它使我们能够在本地机器上部署预先训练好的 BERT 模型并运行推理。它可以用于服务于任何已发布的模型类型，甚至是针对特定下游任务进行微调的模型。此外，它需要后端的 Tensorflow 来处理预训练的模型。因此，我们将继续在控制台中安装 Tensorflow 1.15。

```
pip3 install tensorflow-gpu==1.15
```

接下来，我们将安装 bert 即服务客户端和服务器。同样，这个库不支持 Python 2。因此，请确保您拥有 Python 3.5 或更高版本。

```
pip3 install -U bert-serving-server bert-serving-client
```

BERT 服务器将模型部署在本地机器上，客户端可以订阅它。此外，用户可以在同一台机器上安装这两个服务器，或者在一台机器上部署服务器，然后从另一台机器上进行订阅。安装完成后，下载您选择的 BERT 模型。你可以在这里找到[所有车型的列表](https://github.com/google-research/bert#pre-trained-models)。

# 部署模型

现在初始设置已经完成，让我们用下面的命令启动模型服务。

```
bert-serving-start -model_dir /path_to_the_model/ -num_worker=1
```

例如，如果模型的名称是 uncased_L-24_H-1024_A-16，并且它位于“/model”目录中，则该命令如下所示

```
bert-serving-start -model_dir /model/uncased_L-24_H-1024_A-16/ -num_worker=1
```

“num_workers”参数用于初始化服务器可以处理的并发请求的数量。然而，只需使用 num_workers=1，因为我们只是在用单个客户端来处理我们的模型。如果您正在部署多个客户端进行订阅，请相应地选择“num_workers”参数。

# 通过 BERT 客户端订阅

我们可以运行一个 Python 脚本，从中使用 BERT 服务将我们的单词编码成单词嵌入。鉴于此，我们只需导入 BERT-client 库并创建一个 client 类的实例。一旦我们这样做了，我们就可以输入我们想要编码的单词或句子的列表。

```
from bert_serving.client import BertClient()client = BertClient()vectors = client.encode([“dog”],[“cat”],[“man”])
```

我们应该将想要编码的单词输入 Python 列表。上面，我输入了三个列表，每个列表都有一个单词。因此,“vectors”对象的形状为(3，embedding_size)。通常，嵌入大小是 BERT 模型编码的单词向量的长度。事实上，它将任意长度的单词编码成一个恒定长度的向量。但是这可能在不同的 BERT 模型之间有所不同。

# 计算单词之间的相似度

好的，目前为止一切顺利！如何处理只有一些数字的向量？它们不仅仅是数字。我前面说过，这些向量代表了单词在 1024 维超空间中的编码位置(这个模型的 1024 uncased _ L-24 _ H-1024 _ A-16)。此外，用某种相似度函数来比较不同单词的向量，将有助于确定它们的相关程度。

余弦相似性就是这样一个函数，它给出 0.0 到 1.0 之间的相似性得分。假设 1.0 表示单词意思相同(100%匹配)，0 表示它们完全不同。下面是单词嵌入之间余弦相似性的 scikit-learn 实现。

```
**from** sklearn.metrics.pairwise **import** cosine_similaritycos_lib = cosine_similarity(vectors[1,:],vectors[2,:]) #similarity between #cat and dog
```

# 伯特完成单词嵌入！

你也可以输入一个完整的句子而不是单个单词，服务器会处理好的。有多种方式可以将单词嵌入组合起来，以形成像连接这样的句子嵌入。

查看其他关于[物体检测](https://hackerstreak.com/yolo-made-simple-interpreting-the-you-only-look-once-paper/)、[真伪验证](https://hackerstreak.com/siamese-neural-network-for-signature-verification/)和[更多](https://hackerstreak.com/posts/)的文章！

*原载于*[*https://hackerstreak.com*](https://hackerstreak.com/)