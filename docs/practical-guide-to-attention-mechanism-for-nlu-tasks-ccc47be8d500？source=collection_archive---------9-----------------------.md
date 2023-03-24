# NLU 任务注意机制实用指南

> 原文：<https://towardsdatascience.com/practical-guide-to-attention-mechanism-for-nlu-tasks-ccc47be8d500?source=collection_archive---------9----------------------->

## 测试动手策略以解决注意力问题，从而改进序列到序列模型

聊天机器人、虚拟助手、增强分析系统通常会接收用户的查询，例如“给我找一部史蒂文·斯皮尔伯格的动作片”。系统应该正确地检测意图“寻找电影”，同时用值“动作”填充槽“类型”，用值“史蒂文·斯皮尔伯格”填充槽“导演”。这是一个自然语言理解(NLU)任务，称为**意图分类&槽填充**。通常使用基于递归神经网络(RNN)的方法，以及利用具有序列到序列模型的编码器-解码器架构，可以获得最先进的性能。在本文中，我们展示了通过添加注意机制来进一步提高性能的实践策略。

![](img/00f3ab3acc7695acc09351f95616800a.png)

[Why kids just need your time and attention](https://mensline.org.au/changingforgood/blog/why-kids-just-need-your-time-and-attention/)

2014 年，在 Sutskever 通过发现序列到序列模型而彻底改变了深度学习之后，正是 2015 年注意力机制的发明最终完成了这个想法，并打开了我们每天都享受的惊人的机器翻译的大门。注意力很重要，因为当与[神经单词嵌入](/representing-text-in-natural-language-processing-1eead30e57d8)结合使用时，它已经被证明可以在机器翻译和其他自然语言处理任务中产生最先进的结果，并且是 BERT、GPT-2 等突破性算法的一个组成部分，这些算法正在创造 NLP 准确性的新纪录。因此，注意力是我们迄今为止最大努力的一部分，以创造真正的机器自然语言理解。如果成功的话，它将对社会和几乎每一种商业形式产生巨大的影响。

在本系列的[第一篇文章](/natural-language-understanding-with-sequence-to-sequence-models-e87d41ad258b)中，我们介绍了机器翻译任务来激励序列到序列模型。我们还提供了一个实用的实现来解决槽填充任务。在本文中，我们将通过添加注意机制来改进我们的解决方案，在航班请求查询的自然语言理解方面实现最先进的准确性。我们将把注意力机制引入并编码到我们的序列对序列模型中。我们强烈建议在阅读本文之前，先阅读下面的第一篇文章。

[](/natural-language-understanding-with-sequence-to-sequence-models-e87d41ad258b) [## 基于序列对序列模型的自然语言理解

### 如何预测客户询问背后的意图？Seq2Seq 型号说明。在 ATIS 数据集上演示的槽填充…

towardsdatascience.com](/natural-language-understanding-with-sequence-to-sequence-models-e87d41ad258b) 

我们之前的序列到序列模型对于长句子有严重的问题。例如，该模型既不能理解“下午 4 点以后”的航班时间，也不能理解下面用户查询中的航空公司名称“continental”。

```
>>> input_query = "list all flights from oakland to dallas after 4 pm with continental"
>>> print(predict_slots(input_query, encoder_model, inf_model))O O O O B-fromloc.city_name O B-toloc.city_name O B-depart_date.month_name B-depart_date.day_number I-depart_date.day_number O
```

为什么该模型在短查询上做得很好，但在长查询上却产生无意义的预测？上下文向量被证明是序列到序列模型的瓶颈，它使得处理长句非常具有挑战性。直到 [Bahdanau 等人，2014](https://arxiv.org/abs/1409.0473) 和 [Luong 等人，2015](https://arxiv.org/abs/1508.04025) 提出解决方案。作者介绍了“注意力”，并展示了它如何通过根据需要关注输入序列的相关部分来提高机器翻译系统的质量。

但在深入研究注意力技术之前，我们先用一个童话来激发注意力机制。

# 需要关注的湖泊

> 巴福是个年轻人，他相信自己是宇宙中最美丽的人。每天他都会穿过附近的森林，到达附近的湖边。然后，他会俯身在湖面上，花很多时间欣赏自己的美丽，因为他的脸会倒映在水面上。一天，在做这件事的时候，他掉进了水里，淹死了。当湖水开始抱怨并泪流满面时，周围森林里的树木问湖水:“你为什么哭？”。湖水回答说:“我哭是因为巴福死了”。树回答说:“你不应该哭！我们，这些树，才是应该感到极大痛苦的人，因为每天，当巴福经过时，我们都希望，他能停下来片刻，敢看我们一眼，这样我们也能有机会欣赏他的美丽。你一直有他在你的海岸！”树继续说:“巴福从未考虑过我们。现在，他死了，我们永远不会赞美他的美丽。”湖水回答:“啊哈！巴福长得好看吗？我不知道。我哭是因为，每次他看着我的时候，我只有一次机会从他眼中看到自己的美丽。”

你在问，为什么我要给你讲这样一个故事，虽然这篇文章讲的是机器学习。嗯，你是对的。但是，让我们考虑一下动机。**在我们的故事中，所有三个演员，巴福、湖和树都需要关注**。注意力是你所需要的吗？社交网络真的全是关于注意力吗？那么自然语言理解吗？让我们深入主题。

# 注意机制的背景

和社交网络一样的是关注度；理解序列数据也可以是关于注意力的。我们在[上一篇文章](/natural-language-understanding-with-sequence-to-sequence-models-e87d41ad258b)中构建的序列到序列模型过于强调单词之间的接近，但它也过于关注上游上下文而不是下游上下文。传统单词嵌入的另一个限制，如我们之前的实现中所使用的，是假设单词的含义在句子中相对稳定。通常情况不是这样。另一方面，注意两个用户查询，将它们转换成一个矩阵，其中一个查询的单词构成列，另一个查询的单词构成行。因此，**注意力学习语境关系**。这在自然语言理解中非常有用，因为它还允许调查单个查询的某些部分如何与其他部分相关，这被称为**自我关注**。如果我们能够建立语义依赖图的有向弧，说明一个拥挤的句子之间的关系，那么我们就能够理解句子的意思。

![](img/cb443cf33bc454814ef08cb773b31e3f.png)

([source](https://skymind.ai/wiki/attention-mechanism-memory-network))

注意力模型对经典的序列对序列模型的改进如下。编码器不是传递编码阶段的最后一个隐藏状态，而是将所有隐藏状态传递给解码器。如果你需要回忆隐藏状态是如何工作的，请参考我们下面的文章，其中详细解释了循环网络。

[](/sentiment-analysis-a-benchmark-903279cab44a) [## 情感分析:一个基准

### 递归神经网络解释。使用 FCNNs、CNN、RNNs 和嵌入对客户评论进行分类。

towardsdatascience.com](/sentiment-analysis-a-benchmark-903279cab44a) [](/lstm-based-african-language-classification-e4f644c0f29e) [## 基于 LSTM 的非洲语言分类

### 厌倦了德法数据集？看看 Yemba，脱颖而出。力学的 LSTM，GRU 解释和应用，与…

towardsdatascience.com](/lstm-based-african-language-classification-e4f644c0f29e) 

每个编码器的隐藏状态与输入句子中的某个单词最相关。解码器给每个隐藏状态一个分数，然后将每个隐藏状态乘以其 softmaxed 分数。通过这种方式，解码器放大分数高的隐藏状态，淹没分数低的隐藏状态。它对每个单词都重复这一过程，因此在每个解码步骤中都很注意。

![](img/6cbca18d2ace62b2f33b014e63f83bce.png)

[source](https://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)

当序列到序列模型注意时，编码器将所有隐藏状态传递给解码器，而不是只传递编码阶段的最后一个隐藏状态。此外，该模型执行评分练习以产生注意向量，从而允许解码器在每个解码步骤中注意输入句子的特定部分。

![](img/cc3252c30e928be19968c305d58a3254.png)

source: Luong et al., 2015

# 注意填槽

提出了基于注意力的学习方法，并在意图分类和槽填充方面实现了最先进的性能([来源](https://doi.org/10.21437/Interspeech.2016-1352))。我们利用官方的 [Tensorflow 2.0 神经机器翻译教程](https://colab.research.google.com/github/tensorflow/docs/blob/master/site/en/r2/tutorials/text/nmt_with_attention.ipynb)，在添加必要的开始和结束标记后，修改代码以处理来自 ATIS 数据集的用户查询作为输入序列，意图槽填充作为目标序列。下面我们定义超参数，以及一个 TensorFlow 数据集迭代器，它将用于批量处理数据以降低内存消耗。

注意，我们需要将编码器创建为 Python 类。

实现 Bahdanau 注意力，描述为下面的加法等式(4)，其中 *h_t* 是编码器输出， *h_s* 是隐藏状态。

![](img/fc13f8b4efa46605fc0320d4bda0a923.png)

Bahdanau 附加注意在下面的类中计算。

现在我们可以如下实现解码器。编码器和解码器都具有嵌入层和具有 1024 个单元的 GRU 层。

稀疏分类交叉熵被用作损失函数。下面我们定义优化器和损失函数以及检查点。

使用教师强制技术、Tensorflow 内置函数的梯度计算和反向传播的定制训练循环允许我们调整嵌入权重、GRU 细胞权重和注意力权重。下面的函数对一批数据执行训练步骤。计算梯度并执行反向传播。使用 64 批次大小和 256 维嵌入。

正如我们在下图中看到的，损耗从初始值 0.51 下降到 18 个周期后的 0.0005。

![](img/89b96af8347e3f0ce0f0664a32cb3dca.png)

推理的执行类似于训练循环，只是我们在这里没有使用教师强制。解码器在每个时间步长的输入是其先前的预测以及隐藏状态和编码器输出。当模型预测结束令牌(EOS)时，我们停止预测。我们存储每个时间步的注意力权重。

下面我们给出了准备查询的代码。我们还实现了一个函数，它将查询语句和词汇变量作为输入，创建一个向量张量，执行编码器和解码器进行推理。该函数还计算注意力权重。还有一个额外的功能可以在热图中绘制注意力权重。

对于一个玩具示例“列出所有飞往波士顿的航班”，槽填充质量是合理的。产生的注意力剧情挺有意思的。它显示了在预测用户查询时，模型关注输入句子的哪些部分。通过查看单词“to”和“Boston ”,去波士顿旅行的意图被正确地标记为目的地城市(B-toloc.city_name)。

```
translate("list all flights to boston")
```

![](img/37c8dc79d4443b71fee2488e229cc5e5.png)

当我们向系统提交一个稍微复杂一点的查询“我想乘坐 continental 下周上午 8 点的航班”时，我们很高兴地看到模型如何识别我们希望乘坐的航空公司“continental”，以及相对时间“下周”。

```
translate('I want to fly next week at 8 am with continental')
```

![](img/136b38531503f5475c3498a754a5763d.png)

如果我们问“请告诉我机场 mco 到丹佛的最便宜的票价”，该模型有趣地使用“最便宜”和“票价”这两个词来识别目的地城市“丹佛”的相对成本(B-cost_relative label)意图。而这一切，都发生在“MCO”这个特定的机场。令人惊讶的是，用户查询中的单词“airport”并没有影响将“MCO”标记为机场的决定。从训练集来看，该模型显然也学习了机场代码。为了更好地概括，我们预计单词“airport”附近的 3 个字符的缩写会影响模型的决策。

```
translate('show me the cheapest fare in the airport mco to denver')
```

![](img/1d1126a3b97117f0893aed744e80c50f.png)

下一个查询很复杂:“我想乘坐西北航空公司的航班从底特律飞往华盛顿，大约上午 9 点出发”。这个模型完全正确！这里的注意机制很神奇。

```
translate('i want to fly from detroit to washington on northwest airlines and leave around 9 am')
```

![](img/c3d7ecd8b8dea486901dff7201cd5538.png)

最后，又一个成功预测的例子:“从克利夫兰到达拉斯的航班票价代码 h 是多少”。

```
translate('what what is fare code h on a flight from cleveland to dallas')
```

![](img/14edcf61307c327918471d571efe4bac.png)

# 结论

在本文中，我们通过加入 Bahdanau 注意机制，改进了我们之前的自然语言理解的序列对序列模型。我们通过正确识别客户查询中我们应该注意的语义上有意义的信息，证明了注意力如何显著提高对长句的预测。

由于在 ATIS 数据集上实际实现了几个关于航班请求的模型，我们证明了注意力是多么重要，这种类型的机制如何在机器翻译和其他自然语言处理任务中产生最先进的结果。

在接下来的文章中，我们将把我们的结果与从 Transformer 嵌入的强大单词结合起来。

[](/bert-for-dummies-step-by-step-tutorial-fb90890ffe03) [## 伯特为假人-一步一步教程

### 变压器 DIY 实用指南。经过实践验证的 PyTorch 代码，用于对 BERT 进行微调的意图分类。

towardsdatascience.com](/bert-for-dummies-step-by-step-tutorial-fb90890ffe03) [](/lost-in-translation-found-by-transformer-46a16bf6418f) [## 伯特解释道。迷失在翻译中。被变形金刚发现。

### 打造下一个聊天机器人？伯特，GPT-2:解决变压器模型的奥秘。

towardsdatascience.com](/lost-in-translation-found-by-transformer-46a16bf6418f)