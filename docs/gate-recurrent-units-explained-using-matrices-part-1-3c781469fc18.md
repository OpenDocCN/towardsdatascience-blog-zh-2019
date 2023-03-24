# 用矩阵解释门控循环单位:第 1 部分

> 原文：<https://towardsdatascience.com/gate-recurrent-units-explained-using-matrices-part-1-3c781469fc18?source=collection_archive---------7----------------------->

由:[闪耀的拉塞尔-普勒里](https://www.linkedin.com/in/sparkle-russell-puleri-ph-d-a6b52643/)和[道林-普勒里](https://www.linkedin.com/in/dorian-puleri-ph-d-25114511/)

很多时候，我们都在使用深度学习框架，这些框架执行构建模型所需的所有操作。然而，首先理解一些基本的矩阵运算是有价值的。在本教程中，我们将带您完成理解 GRU 工作原理所需的简单矩阵运算。

详细的笔记本可以在 https://github.com/sparalic/GRUs-internals-with-matrices[或 https://github.com/DPuleriNY/GRUs-with-matrices](https://github.com/sparalic/GRUs-internals-with-matrices)找到

## 什么是门控循环单元(GRU)？

门控递归单元(如下图所示)是一种递归神经网络，它解决了长期依赖性问题，这可能导致梯度消失更大的香草 RNN 网络体验。GRUs 通过存储前一时间点的“记忆”来帮助通知网络未来的预测，从而解决了这个问题。乍一看，人们可能认为这个图相当复杂，但事实恰恰相反。本教程的目的是揭穿使用线性代数基础的 GRUs 的困难。

![](img/b7e8c655b6bfac14e5f2b13093de52c3.png)

GRUs 的控制方程为:

![](img/35aefce5c2f353247fb723c91d9f36a2.png)

Governing equations of a GRU

其中 z 和 r 分别代表更新门和复位门。而 h_tilde 和 h 分别表示中间存储器和输出。

## **GRUs vs 长期短期记忆(LSTM) RNNs**

GRUs 和流行的[lstm](http://colah.github.io/posts/2015-08-Understanding-LSTMs/)(由 [Chris Olah](http://colah.github.io/posts/2015-08-Understanding-LSTMs/) 很好地解释)之间的主要区别是门的数量和单元状态的维护。与 GRUs 不同，LSTMs 有 3 个门(输入、遗忘、输出),并保持内部存储单元状态，这使其更加灵活，但在存储和时间方面效率较低。然而，由于这两种网络都擅长解决有效跟踪长期相关性所需的消失梯度问题。通常使用以下经验法则在它们之间做出选择。当在这两者之间做出决定时，建议您首先训练 LSTM，因为它有更多的参数，更灵活一些，然后是 GRU，如果两者的性能之间没有相当大的差异，那么使用更简单有效的 GRU

# 方法

为了进一步说明 RNNs 的优雅，我们将带您了解理解 GRU 内部工作所需的线性代数基础知识。为此，我们将使用一小段字母来说明我们认为理所当然的矩阵计算是如何使用预打包的包装函数创建许多常见的 DL 框架的。本教程的目的不是让我们倒退，而是帮助我们更深入地理解 rnn 如何使用线性代数工作。

示例使用以下示例字符串作为输入数据:

` text =数学数学数学数学'

然而，算法本质上是某种数学方程，因此我们的原始文本在提交给 GRU 层之前必须用数字形式表示。这在下面的预处理步骤中完成。

## 数据预处理

第一步，创建一个包含所有唯一字符的字典，将每个字母映射到一个唯一的整数:

字符字典:{'h': 0，' a': 1，' t': 2，' M': 3}

我们的编码输入现在变成:
数学数学= [3，1，2，0，3，1，2，0]

**第一步:创建数据批次** 这一步是通过用户指定我们想要创建多少批次(B)，或者给定我们的词汇表(V)的序列长度(S)来实现的。下图演示了如何创建和编码批处理。

假设我们需要以下参数:
1。批量大小(B) = 2
2。序列长度(S) = 3
3。词汇(V) = 4
4。输出(O) = 4

![](img/80958424ab95e79313709f23ad70a3cd.png)

**那么什么是时间序列呢？**
如果您对 RNN 进行基本搜索，您将会找到下图。这个图像是展开形式中发生的事情的概括视图。然而，x(t-1)、x(t)和 x(t+1)(以红色突出显示)在我们的批次中意味着什么？

![](img/96c55e1c235f9cff78a29413b708f654.png)

Vanilla RNN architecture

在我们的小批量中，时间序列代表每个序列，信息从左到右流动，如下图所示。

![](img/70e3a01d6823c49af49e250c7866028b.png)

Schematic of data flow for each one-hot encoded batch

## 数据集的维度

![](img/0a7b2698c85458f2c8d24487d58c19bc.png)

Batch anatomy

## 步骤 1:用代码演示

整形后，如果你检查 X 的形状，你会发现你得到一个形状的秩为 3 的张量:3 x 3 x 2 x 4。这是什么意思？

![](img/8405b6b07d146b1bff27e24cb039ce30.png)

Dimensions of the dataset

## 会演示什么？

数据现在已经准备好进行建模了。然而，我们想强调本教程的流程。我们将演示为批次 1 中的第一个序列(以红色突出显示)执行的矩阵运算(如下所示)。这个想法是为了理解第一个序列的信息是如何传递给第二个序列的，以此类推。

![](img/7f84f928d64c627df2175dc9cb2f0ff4.png)

Sequence used for walk through (Sequence 1 batch 1)

为此，我们需要首先回忆一下这些批次是如何被输入算法的。

![](img/236f278580d6887276993fad11b6c3d3.png)

Schematic of batch one being ingested into the RNN

更具体地说，我们将遍历序列 1 在 GRU 单元中完成的所有矩阵运算，并将在该过程中计算结果输出 y_(t-1)和 h_t(如下所示):

![](img/a0093161df9a6213432012848ce8f706.png)

First time step of of batch 1

## 步骤 2:定义我们的权重矩阵和偏差向量

在这一步中，我们将带您完成用于计算 z 门的矩阵运算，因为其余三个方程的计算完全相同。为了帮助理解这一点，我们将通过将内部等式分解为三部分来遍历复位门 z 的点积，最后我们将对输出应用 sigmoid 激活函数，以挤压 0 和 1 之间的值:

![](img/2a3560e099a8077ee6b7e0946dfafa66.png)

Reset gate

但首先让我们定义网络参数:

## 什么是隐藏尺寸？

上面定义的隐藏大小是学习参数的数量，或者简单地说，是网络内存。该参数通常由用户根据手头的问题来定义，因为使用更多的单元可能会使训练数据过拟合。在我们的例子中，我们选择了隐藏大小 2，以便更容易说明这一点。这些值通常被初始化为来自正态分布的随机数，它们是可训练的，并且在我们执行反向传播时被更新。

![](img/e0bd35e4b1b5ddff9a2f3c94ffb5b646.png)

Anatomy of the Weight matrix

## 我们体重的大小

我们将使用第一批遍历所有矩阵运算，因为对于所有其他批来说，这是完全相同的过程。然而，在我们开始任何上述矩阵运算之前，让我们讨论一个叫做广播的重要概念。如果我们看 batch 1 (3 x 2 x 4)的形状和 Wz (4 x 2)的形状，首先想到的可能是，我们将如何对这两个形状不同的张量执行元素式矩阵乘法？

答案是我们使用一个叫做“广播”的过程。广播被用来使这两个张量的形状兼容，这样我们可以执行我们的元素矩阵运算。这意味着 Wz 将被广播到一个非矩阵维度，在我们的例子中是我们的序列长度 3。这意味着更新等式 z 中的所有其他项也将被广播。因此，我们的最终等式将是这样的:

![](img/3b9b7501ade0d67d7ce540a83140743c.png)

Equation for z with weight matrices broadcasted

在我们执行实际的矩阵运算之前，让我们想象一下第一批中的序列 1 是什么样子的:

![](img/f88803aae52b10b62802126c03e00571.png)

Illustration of matrix operations and dimensions for the first sequence in batch 1

## 更新门:z

更新门决定了过去的信息对当前状态的有用程度。这里，sigmoid 函数的使用导致更新门值在 0 和 1 之间。因此，该值越接近 1，我们包含的过去的信息就越多，而值越接近 0，则意味着只保留新信息。

**现在让我们开始数学…** 第一项:注意，当这两个矩阵用点积相乘时，我们是每行乘以每列。这里，第一个矩阵(x_t)的每一行(用黄色突出显示)都要乘以第二个矩阵(Wz)的每一列(用蓝色突出显示)。

术语 1:应用于输入的权重

![](img/ba8eb7e3cdf8670f5c5a81a29247ebd3.png)

Dot product of the first term in the update gate equation

术语 2:隐藏权重

![](img/ca5321d5f0d6e48bfcda084e74adee7c.png)

Dot product of the second term in the update gate equation

术语 3:偏差向量

![](img/b2fe0997944fc4adc8fc3fa2b5c53cc9.png)

Bias vector

## 将所有这些放在一起:z_inner

![](img/a5ecc75a8d61608c8596ad663fffffbd.png)

Inner linear equation of the reset gate

然后，使用 sigmoid 激活函数将结果矩阵中的值压缩在 0 和 1 之间:

![](img/bc98126f27a0c7f81da83daf4078f6aa.png)

Sigmoid equation

## 复位门:r

重置门允许模型忽略可能与未来时间步不相关的过去信息。在每一批中，复位门将重新评估先前和新输入的综合性能，并根据新输入的需要进行复位。再次因为 sigmoid 激活函数，更接近 0 的值将意味着我们将继续忽略先前的隐藏状态，并且对于更接近 1 的值，情况相反。

![](img/4a0cb0bc1ff7331e42d42ac63877a7d3.png)

Reset gate

## 中间内存:波形符

中间存储单元或候选隐藏状态将来自先前隐藏状态的信息与输入相结合。因为第一项和第三项所需的矩阵运算与我们在 z 中所做的相同，所以我们将只给出结果。

![](img/d3f56f93d98052249a89d1cb25dbc3e6.png)

Intermediate/candidate hidden state

第二学期:

![](img/ce180b401df40a1d3dc13a3d85855d7d.png)

Second term matrix operations

## 将所有内容放在一起:颚化符

![](img/5dee220f7712c1ff14f0e97a9194b20e.png)

Inner linear equation calculation

然后，使用 tanh 激活函数将结果矩阵中的值压缩在-1 和 1 之间:

![](img/9324959c73339e640d54cd8e3dc43fc9.png)

Tanh activation function

最后:

![](img/0e3f15469c3fa6d12d7944020cfcbff6.png)

Candidate hidden state output

## 在时间步长 t:h(t-1)输出隐藏层

![](img/0baaf09f904134cfb17f38aa7fa54fb0.png)

Hidden state for the first time step

![](img/6cf881b33f059d243929f350a46cca6c.png)

Resulting matrix for hidden state at time step 1

## 批次 1 中的第二个序列(时间步长 x_t)如何从这种隐藏状态中获取信息？

回想一下，h(t-n)首先被初始化为零(在本教程中使用)或随机噪声，以开始训练，之后网络将学习和适应。但是在第一次迭代之后，新的隐藏状态 h_t 现在将被用作我们的新的隐藏状态，并且在时间步长(x_t)对序列 2 重复上述计算。下图演示了这是如何做到的。

![](img/6ced3db962d0bc0b679f93d62f82c996.png)

Illustration of the new hidden state calculated in the above matrix operations

这个新的隐藏状态 h(t-1)将不用于计算批量中第二个时间步的输出(y(t+1))和隐藏状态 h(t)，以此类推。

![](img/02cff108d522b13440ddcbdc16ff4054.png)

Passing of hidden states from sequence1 to sequence 2

下面我们演示如何使用新的隐藏状态 h(t-1)来计算后续的隐藏状态。这通常使用循环来完成。该循环迭代每个给定批次中的所有元素，以计算 h_(t-1)。

## 代码实现:第 1 批输出:h(t1)、h(t)和 h(t+1)

## 第二批的隐藏状态是什么？

如果你是一个视觉的人，它可以被看作是一个系列，在 h(t+1)的输出，然后将被送到下一批，整个过程再次开始。

![](img/f39fc18e87754593688630cd0789453b.png)

Passing of hidden states across batches

## 步骤 3:计算每个时间步的输出预测

为了获得我们对每个时间步长的预测，我们首先必须使用线性层来转换我们的输出。回想一下隐藏状态下的列的维数 h(t+n)本质上是网络尺寸/隐藏尺寸的维数。然而，我们有 4 个唯一的输入，我们希望我们的输出也有 4 个。因此，我们使用所谓的密集层或全连接层将输出转换回所需的维度。然后，根据所需的输出，将这个完全连接的层传递给一个激活函数(对于本教程为 Softmax)。

![](img/d8e39f85fe8e1904b12b2b83ffbda1a7.png)

Fully connected/Linear layer

最后，我们应用 Softmax 激活函数将我们的输出归一化为一个概率分布，其总和为 1。Softmax 函数:

![](img/dcb743c0848f88e07a463a2159b5af58.png)

Softmax equation

根据教科书的不同，您可能会看到不同风格的 softmax，特别是使用 softmax max 技巧，它会减去整个数据集的最大值，以防止大型 y _ linear y/full _ connected 的值爆炸。在我们的情况下，这意味着我们的最大值 0.9021 将首先从 y_linear 中减去，然后应用于 softmax 方程。

让我们分解一下，请注意，我们不能像前面那样对序列进行子集划分，因为求和需要整批中的所有元素。

1.  从完全连接的图层中的所有元素中减去整个数据集的最大值:

![](img/5bde611799cbd417366fbdd1e4173962.png)

Applying the Max trick for Softmax equation

2.求指数矩阵中所有元素的和

![](img/37e5cd23f9d12eae357dcc9b502b5779.png)

Sum of the exponents for each row

![](img/f81d48560851ab46adbabcdf680fa61d.png)

Final Softmax output for the first sequence in batch 1

## 最后，训练我们的网络(仅向前)

这里，我们通过在网络中多次运行每个批次来训练输入批次的网络，这被称为一个时期。这允许网络多次学习序列。随后进行损失计算和反向传播，以最大限度地减少我们的损失。在本节中，我们将一次性实现上面显示的所有代码片段。鉴于输入尺寸较小，我们将仅演示正向传递，因为损失函数和反向传播的计算将在后续教程中详细介绍。

这个函数将向网络输入一系列字母，帮助创建一个初始状态，避免胡乱猜测。如下所示，生成的前几个字符串有点不稳定，但是经过几次处理后，至少接下来的两个字符是正确的。然而，考虑到词汇量很小，这个网络很可能会过度适应。

## 最后的话

本教程的目的是通过演示简单的矩阵运算如何组合成如此强大的算法，提供 GRUs 内部工作的一个演示。

## 接下来:[用矩阵解释的门递归单元:第 2 部分训练和损失函数](https://medium.com/@sparklerussell/gated-recurrent-units-explained-with-matrices-part-2-training-and-loss-function-7e7147b7f2ae)

## 参考资料:

1.  递归神经网络的不合理有效性
2.  使用 Pytorch 的 Udacity 深度学习
3.  面向编码人员的 fastai 深度学习
4.  深度学习——直接的兴奋剂
5.  深度学习书籍
6.  http://colah.github.io/posts/2015-08-Understanding-LSTMs/