# 如何用 Python 从头开始构建一个简单的神经网络

> 原文：<https://towardsdatascience.com/how-to-build-a-simple-neural-network-from-scratch-with-python-9f011896d2f3?source=collection_archive---------10----------------------->

## 不使用框架建立神经网络的快速指南。

![](img/e1836a0bd3baa6a745b51361e8d3493c.png)

Photo by [Franck V.](https://unsplash.com/@franckinjapan?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

神经网络每天都变得越来越受欢迎，作为机器学习和人工智能的核心领域，它们将在未来几年在技术、科学和工业中发挥重要作用。这种高受欢迎程度已经产生了许多框架，允许您非常容易地实现神经网络，而无需了解它们背后的完整理论。另一方面，神经网络机制的严格理论解释需要一些高级数学知识。

在这篇文章中，我们将做一些介于。具体来说， ***为了更扎实地理解神经网络，*** 我们将从头开始实际实现一个 NN，不使用任何框架*但为了简单起见我们将省略证明*。*这可能比使用框架更难，但是你会对算法背后的机制有更好的理解。当然，在大型项目中，框架实现是首选，因为它更容易和更快地建立。*

*本教程中使用的工具只是带有 numpy 库(线性代数运算的科学库)的 Python。假设您已经安装了 python 和 pip，您可以通过运行以下命令来安装 numpy:*

```
*pip install numpy*
```

*神经网络实际上是许多变量的函数:它接受输入，进行计算并产生输出。我们喜欢把它想象成不同层中的神经元，一层中的每个神经元都与上一层和下一层中的所有神经元相连。所有的计算都发生在这些神经元内部，并且依赖于将神经元相互连接起来的**权重**。因此，我们所要做的就是学习正确的权重，以获得期望的输出。*

*它们的结构通常非常复杂，包括许多层，甚至超过一百万层(*2020 年 12 月更新:GPT-3 现在使用 175 个参数！)*神经元为了能够处理我们这个时代的大数据集。然而，为了理解大型深度神经网络如何工作，应该从最简单的开始。*

*因此，下面我们将实现一个非常简单的两层网络。为了做到这一点，我们还需要一个非常简单的数据集，因此我们将在示例中使用 XOR 数据集，如下所示。A 和 B 是 NN 的 2 个输入，A***XOR****B*是输出。我们将尝试让我们的 NN **学习权重**,这样无论它接受哪一对 A 和 B 作为输入，它都将返回相应的结果。*

*![](img/4447bcb7016b59501c412f296d409587.png)*

*The XOR truth table*

*所以，我们开始吧！*

*首先，我们需要定义我们的神经网络的结构。因为我们的数据集相对简单，所以只有一个隐藏层的网络就可以了。所以我们会有一个输入层，一个隐藏层和一个输出层。接下来，我们需要一个激活函数。sigmoid 函数是最后一层的好选择，因为它输出 0 到 1 之间的值，而 tanh(双曲正切)在隐藏层中效果更好，但其他所有常用函数也可以(例如 ReLU)。所以我们的神经网络结构看起来会像这样:*

*![](img/6dbe8f07c87e5ec5ea5389ba5f5ed0aa.png)*

*这里，要学习的参数是权重 W1、W2 和偏差 b1、b2。正如你所看到的，W1 和 b1 连接输入层和隐藏层，而 W2，b2 连接隐藏层和输出层。根据基本理论，我们知道激活 A1 和 A2 计算如下:*

```
*A1 = h(W1*X + b1)
A2 = g(W2*A1 + b2)*
```

*其中 g 和 h 是我们选择的两个激活函数(对于 us sigmoid 和 tanh ), W1、W1、b1、b2 通常是矩阵。*

*现在让我们进入实际的代码。代码风格总体上遵循了吴恩达教授在本次[课程](https://www.coursera.org/learn/machine-learning)中提出的指导方针。*

***注意:**你可以在我的知识库[这里](https://gitlab.com/kitsiosk/xor-neural-net)找到完整的工作代码*

*首先，我们将实现我们的 sigmoid 激活函数，定义如下: **g(z) = 1/(1+e^(-z))** 其中 z 通常是一个矩阵。幸运的是 numpy 支持矩阵计算，所以代码相对简单:*

*Sigmoid implementation*

*接下来，我们必须初始化我们的参数。权重矩阵 W1 和 W2 将从正态分布随机初始化，而偏差 b1 和 b2 将被初始化为零。函数 initialize_parameters(n_x，n_h，n_y)将 3 层中每一层的单元数作为输入，并正确初始化参数:*

*Parameters initialization*

*下一步是实现向前传播。函数 forward_prop(X，parameters)将神经网络输入矩阵 X 和参数字典作为输入，并返回 NN A2 的输出以及稍后将在反向传播中使用的缓存字典。*

*Forward Propagation*

*我们现在必须计算损失函数。我们将使用交叉熵损失函数。Calculate_cost(A2，Y)将 NN A2 和基本事实矩阵 Y 的结果作为输入，并返回交叉熵成本:*

*Cost Calculation*

*现在是神经网络算法中最难的部分，反向传播。这里的代码可能看起来有点奇怪和难以理解，但我们不会深入研究它为什么在这里工作的细节。该函数将返回损失函数相对于我们的网络的 4 个参数(W1，W2，b1，b2)的梯度:*

*Backward Propagation*

*很好，现在我们有了损失函数的所有梯度，所以我们可以进行实际的学习了！我们将使用**梯度下降**算法来更新我们的参数，并使我们的模型以作为参数传递的学习率进行学习:*

*Gradient Descent Algorithm*

*到目前为止，我们已经实现了一轮培训所需的所有功能。现在，我们所要做的就是将它们放在一个名为 model()的函数中，并从主程序中调用 model()。*

*Model()函数将特征矩阵 X、标签矩阵 Y、单元数量 n_x、n_h、n_y、我们希望梯度下降算法运行的迭代次数以及梯度下降的学习速率作为输入，并组合上述所有函数以返回我们的模型的训练参数:*

*Model function that combines all the above functions.*

***训练部分**现在结束了。上面的函数将返回我们的神经网络的训练参数。现在我们只需要做出我们的**预测**。函数 predict(X，parameters)将矩阵 X 作为输入，该矩阵 X 具有我们想要为其计算 XOR 函数的 2 个数字和模型的训练参数，并通过使用阈值 0.5 返回期望的结果 y_predict:*

*Prediction function*

*我们最终完成了所有需要的功能。现在让我们进入主程序，声明我们的矩阵 X，Y 和超参数 n_x，n_h，n_y，num_of_iters，learning_rate:*

*Main program: Initialization of variables and hyperparameters*

*设置好以上所有内容后，在上面训练模型就像调用下面这行代码一样简单:*

*Learn the parameters using the model() function*

*最后，让我们预测一对随机的数字，比如说(1，1):*

*Make a prediction with A=1, B=1 using the learned parameters*

*那是真正的代码！让我们看看我们的结果。如果我们用这个命令运行我们的文件，比如说 xor_nn.py*

```
*python xor_nn.py*
```

*我们得到下面的结果，因为 1XOR1=0，所以这个结果确实是正确的。*

*![](img/f3187b0e3a2d98d41363409cc8c15066.png)*

*Result of our NN prediction for A=1 and B=1*

*就是这样！我们只用 Python 从头开始训练了一个神经网络。*

*当然，为了训练具有许多层和隐藏单元的较大网络，您可能需要使用上述算法的一些变体，例如，您可能需要使用批量梯度下降而不是梯度下降，或者使用更多的层，但是简单 NN 的主要思想如上所述。*

*您可以随意使用超参数，尝试不同的神经网络架构。例如，您可以尝试更少的迭代次数，因为成本似乎下降得很快，1000 次迭代可能有点大。记住你可以在我的 GitLab 库[这里](https://gitlab.com/kitsiosk/xor-neural-net)找到完整的工作代码。*

***感谢**的阅读，我很乐意讨论您可能有的任何问题或纠正:)如果您想谈论机器学习或其他任何事情，请在 [LinkedIn](https://www.linkedin.com/in/konstantinoskitsios/) 或我的[网页](http://kitsios.eu/)上找到我。*