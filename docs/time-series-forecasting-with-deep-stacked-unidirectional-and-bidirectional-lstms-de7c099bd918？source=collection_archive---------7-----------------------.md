# 基于深度堆叠单向和双向 LSTMs 的时间序列预测

> 原文：<https://towardsdatascience.com/time-series-forecasting-with-deep-stacked-unidirectional-and-bidirectional-lstms-de7c099bd918?source=collection_archive---------7----------------------->

![](img/adf362ebfb328ac71f797885342040c2.png)

这篇文章假设读者对 LSTMs 的工作原理有一个基本的了解。不过，你可以在这里得到 LSTMs [的简要介绍。另外，如果你是时间序列预测的绝对初学者，我推荐你去看看这个](https://www.analyticsvidhya.com/blog/2017/12/fundamentals-of-deep-learning-introduction-to-lstm/)[博客](https://medium.com/open-machine-learning-course/open-machine-learning-course-topic-9-time-series-analysis-in-python-a270cb05e0b3)。
这篇文章的主要目的是展示深度堆叠单向和双向 LSTMs 如何作为基于 Seq-2-Seq 的编码器-解码器模型应用于时间序列数据。我们将首先介绍这个架构，然后看看实现它的代码。

# 让我们深入研究这个模型

等等。！在进入架构的细节之前，让我们理解序列到序列模型的特殊性。因此，顾名思义，序列对序列模型将一系列特征作为输入，并输出一个目标序列作为输入目标序列的延续(它可以预测未来的“n”个时间步)。
它基本上有两个部分，编码器输出输入序列的上下文向量(编码)，然后传递给解码器解码和预测目标。

![](img/00848b0a312a4352a6058055e62f37bb.png)

Lol，这可能有点让人不知所措，但是随着我们进一步深入和可视化架构，你会慢慢理解这些术语的。

# 模型架构

让我们从基本的编码器-解码器架构开始，然后我们可以逐步添加新的功能和层来构建更复杂的架构。
**1。使用单向 LSTMs 作为编码器**

![](img/ff9216b0e2b863c6233661ee0981ad52.png)

这里，LSTM 编码器将时间序列作为输入(每个 LSTM 单元一个时间步长)，并创建输入序列的编码。该编码是由所有编码器 LSTM 单元的隐藏和单元状态组成的向量。编码然后作为初始状态与其他解码器输入一起被传递到 LSTM 解码器，以产生我们的预测(解码器输出)。在模型训练过程中，我们将目标输出序列设置为模型训练的解码器输出。

**2。使用双向 LSTMs 作为编码器**

![](img/0b78d9c9b1c07175888f4ea4545a9698.png)

双向 LSTMs 具有两个递归分量，前向递归分量和后向递归分量。前向组件计算隐藏和单元状态，类似于标准单向 LSTM，而后向组件通过以逆时间顺序(即从时间步长 Tx 到 1 开始)取输入序列来计算它们。使用后向组件的直觉是，我们正在创建一种方式，网络可以看到未来的数据，并相应地学习其权重。这可能有助于网络捕获标准(前向)LSTM 无法捕获的一些相关性。BLSTM 也是大多数 NLP 任务的首选算法，因为它能够很好地捕捉输入序列中的相关性。
在 BLSTMs 中，前向组件的隐藏和单元状态与后向组件的不同。因此，为了获得编码，前向组件的隐藏和单元状态必须分别与后向组件的隐藏和单元状态连接。

**3。使用堆叠单向 LSTMs 作为编码器**

![](img/457c919de7413189df9e9308e25944bd.png)

当这些层堆叠在一起时，编码器和解码器的第一层 LSTM 单元的输出(单元状态)被传递到第二层 LSTM 单元作为输入。似乎具有几个隐藏层的深度 LSTM 架构可以有效地学习复杂的模式，并且可以逐步建立输入序列数据的更高级别表示。
双向 LSTMs 也可以以类似的方式堆叠。第一层的前向和后向分量的输出分别传递给第二层的前向和后向分量。

## 实施细节:D

## **数据准备—**

我们将在标准的“墨尔本每日最低气温”(单变量时间序列)数据集上应用上述模型(从[这里](https://datamarket.com/data/set/2324/daily-minimum-temperatures-in-melbourne-australia-1981-1990#!ds=2324&display=line)下载)。但是，在进入培训场景之前，让我们首先准备数据。

1.  将数据标准化。

**注意:** reshape 仅用于将单变量 1D 数组转换为 2D，如果数据已经是 2D，则不需要调用。

2.初始化参数。

3.生成输入输出序列对。

上述函数返回一批大小为“total_start_points”的输入序列和输出(目标)序列，它们将分别被馈送到编码器和解码器。**注意**返回的序列是形状的 3D 张量(batch_size，input_seq_len，n_in_features ),因为 keras 要求输入为这种格式。

## 模型—

我们定义了一个单一的函数来构建架构，这取决于传递的隐藏维度列表，以及在调用该函数时设置为“真”或“假”的参数“双向”。
**注意**当编码器是双向的时，我们在解码器 LSTM 中有‘hidden _ dim * 2’来容纳级联的编码器状态。

## 模特.飞度:D

默认情况下，Keras 在训练时会打乱数据，因此我们可以(不一定)在“model.fit”函数中设置“shuffle=False ”,因为我们已经在随机生成序列。
**注意**我们输入零作为解码器输入，也可以使用 [**教师强制**](https://machinelearningmastery.com/teacher-forcing-for-recurrent-neural-networks/) (其中一个解码器单元的输出作为输入输入到下一个解码器单元)(此处未涉及)。

## 结果

![](img/1e01ada3cca41f7a330eab2e4934861a.png)

Unidirectional Layer-1 and Layer-2

![](img/393c831dd845eddd47239651a2575689.png)

Bidirectional Layer-1 and Layer-2

哇！所有的模型只被训练了 100 个时期(具有相同的参数),与单向 lstm 相比，双向 lstm 在学习数据中的复杂模式方面表现突出。因此，所描述的模型可以应用于许多其他时间序列预测方案，甚至可以应用于多变量输入情况，在这种情况下，您可以将具有多个特征的数据作为 3D 张量进行传递。

你可以在我的 [GitHub 库](https://github.com/manohar029/TimeSeries-Seq2Seq-deepLSTMs-Keras)中找到这个例子的 Jupyter 笔记本实现。
我希望你喜欢这篇文章，并让你很好地理解了如何使用深度堆叠 LSTMs 进行时间序列预测。非常感谢您的反馈或改进建议。