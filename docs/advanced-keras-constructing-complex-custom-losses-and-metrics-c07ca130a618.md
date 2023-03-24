# 高级 Keras —构建复杂的定制损失和指标

> 原文：<https://towardsdatascience.com/advanced-keras-constructing-complex-custom-losses-and-metrics-c07ca130a618?source=collection_archive---------1----------------------->

![](img/48eecadb1fdb25854284d75ee79540a1.png)

Photo Credit: Eyal Zakkay

TL；DR——在本教程中，我将介绍一个简单的技巧，它将允许您在 Keras 中构造自定义损失函数，该函数可以接收除`y_true`和`y_pred`之外的参数。

# 背景— Keras 损失和指标

在 Keras 中编译模型时，我们向`compile`函数提供所需的损失和度量。例如:

`model.compile(loss=’mean_squared_error’, optimizer=’sgd’, metrics=‘acc’)`

出于可读性的目的，从现在开始我将集中讨论损失函数。然而，所写的大部分内容也适用于度量标准。

来自 Keras 的[损失文件](https://keras.io/losses/):

> 您可以传递现有损失函数的名称，也可以传递 TensorFlow/Theano 符号函数，该函数为每个数据点返回一个标量，并采用以下两个参数:
> 
> **y_true** :真标签。张量流/Theano 张量。
> 
> **y_pred** :预测。tensor flow/与 y_true 形状相同的 Theano 张量。

因此，如果我们想使用一个常见的损失函数，如 MSE 或分类交叉熵，我们可以通过传递适当的名称轻松做到这一点。

Keras 文档中提供了可用的[损失](https://keras.io/losses/)和[指标](https://keras.io/metrics/)列表。

# 自定义损失函数

当我们需要使用一个可用的损失函数(或度量)时，我们可以构造我们自己的自定义函数并传递给`model.compile`。

例如，构建自定义指标(来自 Keras 的文档):

# 多参数损失/度量函数

您可能已经注意到，损失函数**必须只接受两个参数** : `y_true`和`y_pred`，它们分别是目标张量和模型输出张量。但是如果我们希望我们的损失/度量依赖于这两个之外的其他张量呢？

为此，我们需要使用[函数闭包](https://en.wikipedia.org/wiki/Closure_(computer_programming))。我们将创建一个损失函数(使用我们喜欢的任何参数)**，它返回一个`y_true`和`y_pred`的函数**。

例如，如果我们(出于某种原因)想要创建一个损失函数，将第一层中所有激活的均方值加到 MSE 上:

请注意，我们已经创建了一个返回合法损失函数的函数(不限制参数的数量)，该函数可以访问其封闭函数的参数。

**更具体的例子:**

前面的例子是一个不太有用的用例的玩具例子。那么什么时候我们会想要使用这样的损失函数呢？

假设你正在设计一个可变的自动编码器。您希望您的模型能够从编码的潜在空间重建其输入。然而，你也希望你在潜在空间中的编码是(近似)正态分布的。

而前一个目标可以通过设计一个仅取决于你的输入和期望输出`y_true`和`y_pred`的重建损失来实现。对于后者，你需要设计一个作用于潜在张量的损失项(例如 Kullback Leibler 损失)。为了让你的损失函数接近这个中间张量，我们刚刚学过的技巧可以派上用场。

示例用途:

这个例子是序列到序列变分自动编码器模型的一部分，更多上下文和完整代码请访问[这个报告——Sketch-RNN 算法的一个 Keras 实现](https://github.com/eyalzk/sketch_rnn_keras)。

如前所述，尽管示例是针对损失函数的，但是创建定制度量函数的工作方式是相同的。

*撰写本文时的 Keras 版本:2.2.4*

**参考文献:**

[1] [Keras —损失](https://keras.io/losses/)

[2] [Keras —指标](https://keras.io/metrics/)

[3] [Github 问题—向目标函数传递附加参数](https://github.com/keras-team/keras/issues/2121)