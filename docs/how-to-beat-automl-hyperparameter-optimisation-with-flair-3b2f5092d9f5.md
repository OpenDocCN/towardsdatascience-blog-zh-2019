# 如何用天赋打败谷歌的自动超参数优化

> 原文：<https://towardsdatascience.com/how-to-beat-automl-hyperparameter-optimisation-with-flair-3b2f5092d9f5?source=collection_archive---------11----------------------->

## 使用 Flair 进行文本分类的超参数优化

![](img/acb33386efd4b125dbbf663bea685705.png)

这是我们之前关于[最先进的文本分类](/text-classification-with-state-of-the-art-nlp-library-flair-b541d7add21f)的帖子的后续。我们解释了如何使用 Flair Python NLP 库进行超参数优化，以在文本分类中获得优于 Google AutoML 自然语言的最佳结果。

# 什么是超参数优化，为什么我们不能简单地手动完成？

超参数优化(或调整)是为机器学习算法选择一组最佳参数的过程。数据预处理器、优化器和 ML 算法都接收一组指导其行为的参数。为了实现最佳性能，需要对它们进行调整，以适应所用数据集的统计属性、要素类型和大小。深度学习中最典型的超参数包括学习速率、深度神经网络中的隐藏层数、批量大小、退出率…

在自然语言处理中，我们还会遇到许多其他与预处理和文本嵌入相关的超参数，如嵌入类型、嵌入维数、RNN 层数等

通常，如果我们足够幸运，问题足够简单，只需要一个或两个具有一些离散值的超参数(例如 k-means 中的 k)，我们可以简单地尝试所有可能的选项。但是随着参数数量的增加，试错法变得困难。

> 我们的搜索空间随着调整的参数数量呈指数增长。

假设离散选项，这意味着如果我们有 8 个参数，其中每个参数有 10 个离散选项，我们最终得到超参数的 10⁸可能组合。假设训练一个模型通常需要相当多的时间和资源，这使得手工挑选参数不可行。

有许多超参数优化技术，如网格搜索、随机搜索、贝叶斯优化、梯度方法以及最终的 TPE。[树形结构的 Parzen 估计器](https://papers.nips.cc/paper/4443-algorithms-for-hyper-parameter-optimization.pdf) (TPE)是我们在 Flair 的包装器 Hyperopt 中使用的方法，Hyperopt 是一个流行的 Python 超参数优化库。

# 使用 Flair 进行超参数调谐

Flair 提供了一个简单的 API 来调整您的文本分类器参数。然而，我们需要告诉它需要调整哪些类型的超参数，以及应该为它们考虑哪些值。
运行优化器并不比训练分类器本身更难，但它需要更多的时间和资源，因为它本质上要执行大量的训练。因此，建议在 GPU 加速的硬件上运行。

我们将对在 Kaggle 垃圾短信收集数据集上训练的文本分类器模型进行超参数优化，以区分垃圾短信和非垃圾短信。

## 做好准备

要准备数据集，请参考[最新文本分类](/text-classification-with-state-of-the-art-nlp-library-flair-b541d7add21f)的“预处理—构建数据集”部分，在此我们获得`train.csv`、`test.csv`和`dev.csv`。确保数据集存储在与运行 Flair 的脚本相同的目录中。

您可以通过运行以下命令来检查您是否有可用于培训的 GPU:

```
import torch
torch.cuda.is_available()
```

它返回一个 boolean 值，指示 CUDA 是否可用于 PyTorch(在它上面写有 Flair)。

## 调整参数

超参数优化的第一步很可能包括**定义搜索空间。**这意味着定义我们想要调整的所有超参数，以及优化器应该只考虑它们的一组离散值还是在一个有界的连续空间中搜索。

对于离散参数，使用:

```
search_space.add(Parameter.PARAMNAME, hp.choice, options=[1, 2, ..])
```

对于均匀连续的参数，使用:

```
search_space.add(Parameter.PARAMNAME, hp.uniform, low=0.0, high=0.5)
```

所有可能参数的列表可在[这里](https://github.com/zalandoresearch/flair/blob/master/flair/hyperparameter/parameter.py)看到。

接下来，您需要指定一些参数，这些参数涉及我们想要使用的文本分类器的类型，以及要运行多少个`training_runs`和`epochs`。

```
param_selector = TextClassifierParamSelector(
    corpus=corpus, 
    multi_label=False, 
    base_path='resources/results', 
    document_embedding_type='lstm',
    max_epochs=10, 
    training_runs=1,
    optimization_value=OptimizationValue.DEV_SCORE
)
```

注意`DEV_SCORE`被设置为我们的优化值。这是非常重要的，因为我们不想基于测试集优化我们的超参数，因为这会导致过度拟合。

最后，我们运行`param_selector.optimize(search_space, max_evals=100)`，它将执行优化器的 100 次评估，并将结果保存到`resources/results/param_selection.txt`

运行整个过程的完整源代码如下:

```
from flair.hyperparameter.param_selection import TextClassifierParamSelector, OptimizationValue
from hyperopt import hp
from flair.hyperparameter.param_selection import SearchSpace, Parameter
from flair.embeddings import WordEmbeddings, FlairEmbeddings
from flair.data_fetcher import NLPTaskDataFetcher
from pathlib import Pathcorpus = NLPTaskDataFetcher.load_classification_corpus(Path('./'), test_file='test.csv', dev_file='dev.csv', train_file='train.csv')word_embeddings = [[WordEmbeddings('glove'), FlairEmbeddings('news-forward'), FlairEmbeddings('news-backward')]]search_space = SearchSpace()
search_space.add(Parameter.EMBEDDINGS, hp.choice, options=word_embeddings)
search_space.add(Parameter.HIDDEN_SIZE, hp.choice, options=[32, 64, 128, 256, 512])
search_space.add(Parameter.RNN_LAYERS, hp.choice, options=[1, 2])
search_space.add(Parameter.DROPOUT, hp.uniform, low=0.0, high=0.5)
search_space.add(Parameter.LEARNING_RATE, hp.choice, options=[0.05, 0.1, 0.15, 0.2])
search_space.add(Parameter.MINI_BATCH_SIZE, hp.choice, options=[16, 32, 64])param_selector = TextClassifierParamSelector(
    corpus=corpus, 
    multi_label=False, 
    base_path='resources/results', 
    document_embedding_type='lstm',
    max_epochs=10, 
    training_runs=1,
    optimization_value=OptimizationValue.DEV_SCORE
)param_selector.optimize(search_space, max_evals=100)
```

我们的搜索空间包括学习率、文档嵌入隐藏大小、文档嵌入 RNN 层数、丢失值和批量大小。注意，尽管只使用了一种类型的单词嵌入(一堆新闻向前、新闻向后和手套),我们仍然必须将它传递给搜索空间，因为它是一个必需的参数。

# 结果

优化器在 GPU 上运行了大约 6 个小时，执行了 100 次评估。最终结果写入`resources/results/param_selection.txt`。

最后几行显示最佳参数组合，如下所示:

```
--------evaluation run 97
dropout: 0.19686569599930906
embeddings: ./glove.gensim, ./english-forward-v0.2rc.pt, lm-news-english-backward-v0.2rc.pt
hidden_size: 256
learning_rate: 0.05
mini_batch_size: 32
rnn_layers: 2
score: 0.009033333333333374
variance: 8.888888888888905e-07
test_score: 0.9923
...
----------best parameter combination
dropout: 0.19686569599930906
embeddings: 0 
hidden_size: 3
learning_rate: 0 <- *this means 0th option*
mini_batch_size: 1
rnn_layers: 1
```

根据进一步评估确认的调优结果的`test_score`，我们获得了测试 f1 分数 **0.9923 (99.23%)** ！

这意味着我们以微弱优势超过了谷歌的 AutoML。

![](img/83bb2e99ecebc010db54b37cc61be311.png)

Results obtained on Google AutoML Natural Language

*提示:如果* `*precision*` *=* `*recall*` *那么*`*f-score*`*=*`*precision*`*=*`*recall*`

**这是否意味着我可以按照这个指南一直达到最先进的效果？**

简短回答:不。该指南应该让您很好地了解如何使用 Flair 的超参数优化器，并且不是 NLP 文本分类框架的全面比较。使用所描述的方法肯定会产生与其他最先进的框架相当的结果，但是它们会根据数据集、使用的预处理方法和定义的超参数搜索空间而变化。

**注意**当选择最佳参数组合时，Flair 会考虑所获得结果的损失和变化。因此，损失最低且 f1-得分最高的车型不一定会被选为最佳车型。

## 那么我现在如何使用参数来训练一个实际的模型呢？

要在实际模型上使用最佳性能参数，您需要从`param_selection.txt`中读取最佳参数，并手动将它们一个接一个地复制到将训练我们的模型[的代码中，就像我们在第 1 部分](https://medium.com/@tadejmagajna/text-classification-with-state-of-the-art-nlp-library-flair-b541d7add21f)中所做的那样。

虽然我们对这个库非常满意，但是如果能够以一种更加代码友好的格式提供最佳参数，或者更好的是，在优化过程中可以选择简单地导出最佳模型，那就更好了。