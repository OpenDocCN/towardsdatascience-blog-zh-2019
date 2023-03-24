# 用神经网络中的正弦激活函数创造替代真理

> 原文：<https://towardsdatascience.com/creating-alternative-truths-with-sine-activation-function-in-neural-networks-d45aac83ee52?source=collection_archive---------10----------------------->

## 利用正弦激活函数快速训练神经网络

你好！今天我要讲的是在神经网络中使用正弦函数作为激活。我会试着回答“这是什么？”，“它如何改变神经网络的未来？”、“这种方法的缺点是什么？”最后我还会演示一个例子。

# 什么是 sin 激活功能？

您可能已经从成千上万的出版物中听说过，神经网络使用单调函数，将神经网络的输出映射到 0 和 1 之间。这是正确的方法，因为这些激活函数给出类似概率的输出。但是，这种方法的缺点是输出只有一个真和一个错。但是在现实生活中，完全不同的值可能对一个事件给出相同的输出。因此，我们需要能够触及多个价值领域来获得真理，而不是试图接近一个小的价值领域来获得真理。

Josep M.Sopena、Enrique Romero、Rene Alquezar 撰写的论文“具有周期和单调激活函数的神经网络:分类问题的比较研究”也主张我们使用正弦激活的方法。下图展示了多种可能性。

![](img/f1da43f8b62d578070393d7c5c0fe4d7.png)

[Figure from paper](https://pdfs.semanticscholar.org/1820/d7de08bc28ec424feb387fcfa563c240f6a4.pdf)

> [具有非单调激活功能的单元可以将空间划分为两个以上的区域。如果函数是周期的，那么区域的数目是无限的，可以解释为一个波前在变量空间中传播。](https://pdfs.semanticscholar.org/1820/d7de08bc28ec424feb387fcfa563c240f6a4.pdf)

从图中我们可以看到，sigmoid 函数的输出只能是特定值的 1。但是，正弦函数的输出可以无限次为 1。这实际上意味着，如果神经网络应该为 x 和 x+100 个输入提供输出 1，我们可以通过使用 sin(x)和 sin(x+100)来使 model 的函数接近 y(x)=1 和 y(x+100)=1。如果我们不使用 sin 函数，我们的网络必须调整其权重和偏差，以便它们可以将 x 和 x+100 映射到 0 和 1 之间的范围。

# 它如何改变神经网络的未来？

证明了具有单调函数的神经网络给出了令人满意的结果。但是他们真正的问题是训练。它们被训练得如此之慢，是因为网络需要调整的参数数量达到了数百万。如果我们将使用 sin 函数作为激活，网络应该进行的调整次数将会更少。因此，网络的训练时间将显著减少。这可以降低神经网络模型的训练成本。

# 这种方法的缺点是什么？

不确定性和容易过拟合。由于具有正弦激活函数的网络在调整权值时简单而快速，这也造成了过拟合。不要过度拟合网络，我们需要给模型一个小的学习率，这样我们可以防止过度拟合。将网络的输出映射到无限概率空间实际上是增加了不确定性。对一个值的调整可能导致另一个值映射到一个非常不同的概率。

# 履行

正弦激活函数的正向和反向操作肯定有不同的实现。我只是尝试了我脑海中的那个。

```
def error_function_for_sin_single(output,y):
    to_search_best = np.sin(np.linspace(output-1,output+1,10000)*np.pi/2)
    index = np.argmin(np.abs(to_search_best-y))
    to_be = np.linspace(output-1,output+1,10000)[index]
    error = to_be-output
    #print("to be:",to_be,"as is",output,"error",error/10)
    return errordef error_function_for_sin_multiple(self,output,y):
    derror_doutput = []
    for cnt in range(y.shape[0]):
        derror_doutput.append(self.error_function_for_sin_single( output[cnt], y[cnt]))
    derror_doutput = np.array(derror_doutput)
    #print("____________")
    return derror_doutput/2
```

## 代码的解释:

*   假设我们有一个输出数组(output)比如[o1，o2，o3，…]仍然没有被正弦函数激活。我们有“未来”数组(y)
*   首先，我们通过 error _ function _ for _ sin _ multiple 函数中的 for 循环从数组中取出每个元素。
*   然后，用我们从数组中取的元素调用 error_function_for_sin_single 函数。
*   在 error_function_for_sin_single 函数中，我们计算输出值(to_search_max)周围的正弦函数。(在正弦函数中，我将输出乘以 pi/2，因为我希望值 1 为 pi/2，它稍后将映射为值 1，作为正弦函数的输出。)
*   然后我们通过计算 y 和 to_search_best 之间的差来找到最小误差的指标。
*   给出最小误差的索引实际上是输出应该是的值。因此，我们找到这个值和输出之间的差异，以便我们可以反馈到神经网络进行反向传播。
*   找到错误后，我们将它们附加到一个列表中，以便将它们全部交给反向传播。

## 演示如何快速接近目标值

数据集:MNIST 数字数据库

算法:使用正弦基函数的前馈神经网络

层数:输入= 4704(我做基本特征提取，所以不是 784)，L1 = 512，L2: 128，输出:10

学习率:0.0000001

我试图过度训练模型，将图像映射到其标签，以测量最少的纪元。

## 使用正弦激活功能:误差在 13 个周期内降至 0.044。

## 无正弦激活功能:误差在 19 个周期内降至 0.043。

另外，我上面提到的论文对正弦激活函数做了几个实验。其中一个是“螺旋问题”

来自[论文](https://pdfs.semanticscholar.org/1820/d7de08bc28ec424feb387fcfa563c240f6a4.pdf)的结果:

> 具有简单架构的标准 BP 还没有找到这个问题的解决方案(参见[Fahlman and Labiere，1990])。[Lang 和 Witbrock，1988]使用具有复杂架构的标准 BP(2–5–5–5–1，带快捷方式)在 20，000 个时代内解决了该问题。“Supersab”平均需要 3500 个历元，“Quickprop”需要 8000 个，“RPROP”需要 6000 个，而“Cascade Correlation”需要 1700 个。为了解决这个问题，我们构建了一个架构为 2–16–1 的网络，在隐藏层使用正弦作为激活函数，在输出层使用双曲正切。这种架构是迄今为止用来处理这个问题的最简单的架构。结果如表 1 所示。初始权重范围的重要性是显而易见的。对于小范围的非线性参数，学习是不可能的(见表 1 中的前两行

![](img/d3fea33c5b870bc24e2fc85b4dd46c97.png)

The results of experiment made on spiral problem.

我希望你喜欢阅读这篇文章。虽然每个人都在谈论寻找复杂问题解决方案的极其复杂的神经网络，但我认为我们仍然应该检查神经网络的基本算法。因为基本面的变化带来的影响更大。