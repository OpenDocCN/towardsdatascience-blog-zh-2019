# 神经网络中超参数调整简单指南

> 原文：<https://towardsdatascience.com/simple-guide-to-hyperparameter-tuning-in-neural-networks-3fe03dad8594?source=collection_archive---------0----------------------->

关于超参数优化的分步 Jupyter 笔记本演练。

![](img/47c5b9d2c878f20267f4c1373fd992ba.png)

Image courtesy of [FT.com](https://www.ft.com/content/0a879bec-48bd-11e8-8c77-ff51caedcde6).

这是我的全连接(香草)神经网络系列的第四篇文章。在本文中，我们将优化一个神经网络并执行超参数调整，以便在 [*Beale 函数*](https://en.wikipedia.org/wiki/Test_functions_for_optimization) 上获得一个高性能的模型——这是许多常用于研究各种优化技术有效性的测试函数之一。这种分析可以在任何函数中重复使用，但是我建议您在另一个常见的测试函数中亲自尝试一下，以测试您的技能。就我个人而言，我发现优化一个神经网络会令人难以置信地沮丧(虽然不像 GAN 那么糟糕，如果你熟悉的话)..)除非你有一个清晰明确的程序可以遵循。我希望你喜欢这篇文章，并发现它很有见地。

您可以访问下面的前几篇文章。第一个为那些不熟悉的人提供了神经网络主题的简单介绍。第二篇文章涵盖了更多的中间主题，如激活函数、神经结构和损失函数。

[](/simple-introduction-to-neural-networks-ac1d7c3d7a2c) [## 神经网络简介

### 神经网络的详细概述，有大量的例子和简单的图像。

towardsdatascience.com](/simple-introduction-to-neural-networks-ac1d7c3d7a2c) [](/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98) [## 神经网络体系结构综合介绍

### 神经架构、激活函数、损失函数、输出单元的详细概述。

towardsdatascience.com](/comprehensive-introduction-to-neural-network-architecture-c08c6d8e5d98) [](/neural-network-optimization-7ca72d4db3e0) [## 神经网络优化

### 涵盖优化器，动量，自适应学习率，批量标准化，等等。

towardsdatascience.com](/neural-network-optimization-7ca72d4db3e0) 

所有相关代码现在都可以在我的 GitHub 存储库中找到:

[](https://github.com/mpstewart1/Neural-Networks) [## GitHub-mpstewart 1/神经网络

### 这个存储库包含与我的全连接神经网络系列相关的 Jupyter 笔记本内容。

github.com](https://github.com/mpstewart1/Neural-Networks) 

# **比厄的功能**

神经网络现在在工业和研究中相当普遍，但令人尴尬的是，其中很大一部分人无法很好地与它们合作，以产生能够超越大多数其他算法的高性能网络。

当应用数学家开发一种新的优化算法时，他们喜欢做的一件事是在测试函数上测试它，这有时被称为*人工景观*。这些人造景观帮助我们找到一种方法，从以下方面比较各种算法的性能:

*   收敛(他们到达答案的速度)
*   精确度(它们与精确答案有多接近)
*   健壮性(它们是对所有功能都表现良好，还是只对一小部分功能表现良好)
*   一般性能(例如计算复杂性)

只要向下滚动维基百科关于优化测试函数的文章，就可以看到一些函数非常糟糕。他们中的许多人被选中，因为他们强调了可能困扰优化算法的具体问题。在本文中，我们将关注一个看起来相对无害的函数，叫做*比厄函数*。

*比厄函数*看起来是这样的:

![](img/f189f0c54e28ea375b52dc53fcd60704.png)

The Beale function.

这个函数看起来并不特别可怕，对吗？这是一个测试函数的原因是，它评估优化算法在具有非常浅的梯度的平坦区域中执行得有多好。在这些情况下，基于梯度的优化程序特别难以达到任何最小值，因为它们不能有效地学习。

本文的其余部分将跟随我的 GitHub 资源库中的 Jupyter 笔记本教程。我们将讨论处理这种人造景观的方法。这种情况类似于神经网络的损耗面。训练神经网络时，目标是通过执行某种形式的优化(通常是随机梯度下降)来找到损失面上的全局最小值。

通过学习如何处理一个困难的优化函数，读者应该更好地准备处理现实生活中实现神经网络的场景。

对于那些不熟悉 Jupyter 笔记本的读者来说，可以在这里阅读更多关于它的内容。

在我们接触任何神经网络之前，我们首先必须定义函数并找到它的最小值(否则，我们怎么知道我们得到了正确的答案？).第一步(在导入任何相关的包之后)是在我们的笔记本中定义 Beale 函数:

然后，我们设置一些函数边界，因为我们对这种情况下的最小值(从我们的图中)以及网格的步长有一个大概的估计。

然后，我们根据这些信息制作一个网格，并准备寻找最小值。

现在我们做一个(可怕的)初步猜测。

然后我们使用`scipy.optimize`函数，看看会弹出什么答案。

这是结果:

![](img/b69d75736e845c1fb69cd2118f412aa4.png)

看起来答案是(3，0.5)，如果你把这些值代入方程，你会发现这是最小值(维基百科页面上也是这么说的)。

在下一部分，我们将开始我们的神经网络。

![](img/6b9e741293750d20c0548e4e3dccb97d.png)

# **神经网络中的优化**

神经网络可以被定义为结合输入并试图猜测输出的框架。如果我们足够幸运地有一些结果，称为“地面真相”，来比较网络产生的输出，我们可以计算出**误差**。因此，网络进行猜测，计算某个误差函数，在尝试最小化该误差的同时再次猜测，并且再次猜测，直到误差不再下降。这就是优化。

在神经网络中，最常用的优化算法是 **GD(梯度下降)**算法。梯度下降中使用的*目标函数*就是我们想要最小化的*损失函数*。

本教程将集中在 Keras 现在，所以我会给一个简短的 Keras 复习。

## 复习者

`Keras`是一个用于深度学习的 Python 库，可以运行在 Theano 和 TensorFlow 之上，这两个强大的 Python 库分别由脸书和谷歌创建和发布，用于快速数值计算。

Keras 的开发是为了尽可能快速、简单地开发深度学习模型，用于研究和实际应用。它运行在 Python 2.7 或 3.5 上，可以在 GPU 和 CPU 上无缝执行。

Keras 建立在模型的理念之上。在其核心，我们有一个称为`Sequential`模型的层序列，它是层的线性堆栈。Keras 还提供了 functional API，这是一种定义复杂模型的方法，例如多输出模型、有向无环图或具有共享层的模型。

我们可以使用顺序模型总结 Keras 中深度学习模型的构建如下:

1.  **定义你的模型**:创建一个`Sequential`模型，添加图层。
2.  **编译你的模型**:指定损失函数和优化器，调用`.compile()`函数。
3.  **适合您的模型**:通过调用`.fit()`函数对模型进行数据训练。
4.  **进行预测**:通过调用`.evaluate()`或`.predict()`等函数，使用模型对新数据进行预测。

您可能会问自己——如何在模型运行时检查它的性能？这是一个很好的问题，答案是通过使用*回调*。

## 回调:在我们的模型训练时偷看一眼

您可以通过使用`callbacks`来查看您的模型的各个阶段发生了什么。回调是在训练过程的给定阶段应用的一组函数。在训练期间，您可以使用回调来获得模型的内部状态和统计数据的视图。您可以将一个回调列表(作为关键字参数回调)传递给顺序类或模型类的`.fit()`方法。回调的相关方法将在训练的每个阶段被调用。

*   你已经熟悉的一个回调函数是`keras.callbacks.History()`。这自动包含在`.fit()`中。
*   另一个非常有用的是`keras.callbacks.ModelCheckpoint`，它保存了模型在训练中某一点的重量。如果您的模型运行了很长一段时间，并且发生了系统故障，那么这可能是有用的。那么，并非一切都失去了。例如，只有当`acc`观察到改进时，才保存模型权重，这是一个很好的做法。
*   `keras.callbacks.EarlyStopping`当监控的数量停止改善时，停止训练。
*   `keras.callbacks.LearningRateScheduler`会改变训练时的学习率。

我们稍后将应用一些回调。有关`callbacks`的完整文档，请参见[https://keras.io/callbacks/](https://keras.io/callbacks/)。

我们必须做的第一件事是导入许多不同的功能，让我们的生活更轻松。

如果您希望您的网络使用随机数工作，但为了使结果可重复，您可以做的另一个步骤是使用*随机种子*。这每次都会产生相同的数字序列，尽管它们仍然是伪随机的(这是比较模型和测试可再现性的好方法)。

## 步骤 1 —确定网络拓扑(并非真正意义上的优化，但非常重要)

我们将使用 MNIST 数据集，它由手写数字(0–9)的灰度图像组成，尺寸为 28x28 像素。每个像素是 8 位，因此其值的范围是从 0 到 255。

获取数据集非常容易，因为`Keras`内置了一个函数。

我们的 *X* 和 *Y* 数据的输出分别是(60000，28，28)和(60000，1)。最好打印一些数据来检查值(以及数据类型，如果需要的话)。

我们可以通过查看每个数字的一个图像来检查训练数据，以确保它们没有从我们的数据中丢失。

![](img/81f8f5f7df5b179f51f5f313582d12ea.png)

最后一项检查是针对训练集和测试集的维度，这可以相对容易地完成:

我们发现我们有 60，000 个训练图像和 10，000 个测试图像。接下来要做的是预处理数据。

## 预处理数据

要运行我们的神经网络，我们需要预处理数据(这些步骤可以互换执行):

*   首先，我们必须使 2D 图像数组成为 1D(展平它们)。我们可以通过使用`numpy.reshape()`或`keras`方法的数组整形来实现这一点:一个叫做`keras.layers.Flatten`的层将图像格式从 2d 数组(28×28 像素)转换为 28 * 28 = 784 像素的 1D 数组。
*   然后，我们需要使用以下变换来归一化像素值(给它们 0 到 1 之间的值):

![](img/b43e953847185d2f4766f9c3e3cfef30.png)

在我们的例子中，最小值是零，最大值是 255，所以公式变成简单的 *𝑥:=𝑥/255.*

我们现在想一次性编码我们的数据。

现在我们终于准备好构建我们的模型了！

## 步骤 2 —调整`learning rate`

最常见的优化算法之一是随机梯度下降(SGD)。SGD 中可以优化的超参数有`learning rate`、`momentum`、`decay`和`nesterov`。

`Learning rate`控制每批结束时的重量，而`momentum`控制前一次更新对当前重量更新的影响程度。`Decay`表示每次更新的学习率衰减，而`nesterov`根据我们是否要应用内斯特罗夫动量取值“真”或“假”。

这些超参数的典型值为 lr=0.01，衰变=1e-6，动量=0.9，nesterov=True。

学习率超参数进入`optimizer`函数，我们将在下面看到。Keras 在`SGD`优化器中有一个默认的学习率调度程序，它在随机梯度下降优化算法中降低学习率。学习率根据以下公式降低:

*lr = lr×1/(1+衰变∫历元)*

![](img/427095f900fc8bbcb7f0c01d8cbd9cbc.png)

Source: [http://cs231n.github.io/neural-networks-3](http://cs231n.github.io/neural-networks-3)

让我们在`Keras`中实现一个学习率适应计划。我们将从 SGD 和 0.1 的学习率值开始。然后，我们将训练 60 个时期的模型，并将衰减参数设置为 0.0016 (0.1/60)。我们还包括动量值 0.8，因为当使用自适应学习率时，它似乎工作得很好。

接下来，我们构建神经网络的架构:

我们现在可以运行模型，看看它的表现如何。这在我的机器上花了大约 20 分钟，可能快或慢，取决于你的机器。

在它完成运行后，我们可以为训练集和测试集绘制准确度和损失函数作为历元的函数，以查看网络的表现如何。

损失函数图如下所示:

![](img/5945513a373f7d118b406d90e9c0326b.png)

Loss as a function of epochs.

这就是准确性:

![](img/07abbccef047b8ea6c8ad08d5ce16f92.png)

我们现在来看看如何应用定制的学习率。

## 使用`LearningRateScheduler`应用自定义学习率变化

编写一个函数，执行指数学习率衰减，如下式所示:

𝑙𝑟=𝑙𝑟₀ × 𝑒^(−𝑘𝑡)

这与前面的类似，所以我将在一个代码块中完成，并描述不同之处。

我们在这里看到，这里唯一改变的是我们定义的`exp_decay`函数的出现及其在`LearningRateScheduler`函数中的使用。请注意，我们这次还选择在模型中添加了一些回调函数。

我们现在可以绘制学习率和损失函数，作为历元数的函数。学习率曲线非常平滑，因为它遵循我们预先定义的指数衰减函数。

![](img/115e00628504c4bc3422b8e3574d3d99.png)

与以前相比，损失函数现在看起来也更平滑了。

![](img/8dd75f83a87e5cfa989bb5953ad417a3.png)

这向您展示了开发一个学习率调度器是提高神经网络性能的一个有用的方法。

## 步骤 3 —选择一个`optimizer`和一个`loss function`

当构建一个模型并使用它进行预测时，例如，为图像分配标签分数(“猫”、“飞机”等)。)，我们想通过定义一个“损失”函数(或目标函数)来衡量自己的成败。优化的目标是有效地计算最小化该损失函数的参数/权重。`keras`提供各种类型的[损失函数](https://github.com/keras-team/keras/blob/master/keras/losses.py)。

有时“损失”函数度量“距离”我们可以用适合问题或数据集的各种方法来定义两个数据点之间的“距离”。使用的距离取决于数据类型和要解决的问题。例如，在自然语言处理(分析文本数据)中，汉明距离更为常见。

**距离**

*   欧几里得的
*   曼哈顿
*   其他的，比如汉明，测量字符串之间的距离。“carolin”和“cathrin”的汉明距离是 3。

**损失函数**

*   MSE(用于回归)
*   分类交叉熵(用于分类)
*   二元交叉熵(用于分类)

## 步骤 4 —决定`batch size`和`number of epochs`

**批量大小**定义了通过网络传播的样本数量。

例如，假设您有 1000 个训练样本，您想要设置一个等于 100 的`batch_size`。该算法从训练数据集中获取前 100 个样本(从第 1 个到第 100 个)并训练网络。接下来，它获取第二个 100 个样本(从第 101 个到第 200 个)并再次训练网络。我们可以继续这样做，直到我们通过网络传播了所有的样本。

使用批量大小的优势< number of all samples:

*   It requires less memory. Since you train the network using fewer samples, the overall training procedure requires less memory. That’s especially important if you cannot fit the whole dataset in your machine’s memory.
*   Typically networks train faster with mini-batches. That’s because we update the weights after each propagation.

Disadvantages of using a batch size < number of all samples:

*   The smaller the batch, the less accurate the estimate of the gradient will be.

The **历元数**是一个超参数，它定义了学习算法在整个训练数据集中工作的次数。

一个时期意味着训练数据集中的每个样本都有机会更新内部模型参数。一个时期由一个或多个批次组成。

对于选择批次大小或周期数没有硬性规定，也不能保证增加周期数会比减少周期数得到更好的结果。

## 步骤 5 —随机重启

这个方法在`keras`中似乎没有实现。这可以通过改变`keras.callbacks.LearningRateScheduler`很容易地做到。我将把这个留给读者做练习，但它本质上涉及到在指定数量的时期后，在有限的次数内重置学习率。

# 使用交叉验证调整超参数

现在，我们将使用 Scikit-Learn 的 GridSearchCV 为我们的超参数尝试几个值并比较结果，而不是手动尝试不同的值。

为了使用`keras`进行交叉验证，我们将使用 Scikit-Learn API 的包装器。它们提供了一种在 Scikit-Learn 工作流程中使用顺序 Keras 模型(仅单输入)的方法。

有两种包装器可用:

`keras.wrappers.scikit_learn.KerasClassifier(build_fn=None, **sk_params)`，它实现了 Scikit-Learn 分类器接口，

`keras.wrappers.scikit_learn.KerasRegressor(build_fn=None, **sk_params)`，实现 Scikit-Learn 回归器接口。

## **尝试不同的权重初始化**

我们将尝试通过交叉验证优化的第一个超参数是不同权重初始化。

我们网格搜索的结果是:

![](img/f3a2fa18efa955f92b35704af93b2302.png)

我们看到，使用 lecun_uniform 初始化或 glorot_uniform 初始化的模型获得了最好的结果，并且我们可以使用我们的网络实现接近 97%的准确度。

## 将您的神经网络模型保存到 JSON

分层数据格式(HDF5)是一种数据存储格式，用于存储大型数据数组，包括神经网络中的权重值。

可以安装 HDF5 Python 模块:`pip install h5py`

Keras 允许您使用 JSON 格式描述和保存任何模型。

## 使用多个超参数进行交叉验证

通常，我们对查看单个参数如何变化不感兴趣，而是查看多个参数变化如何影响我们的结果。我们可以同时对多个参数进行交叉验证，有效地尝试它们的组合。

**注意:神经网络中的交叉验证计算量很大**。实验前先思考！将您正在验证的要素数量相乘，以查看有多少种组合。使用 *k* 折叠交叉验证来评估每个组合( *k* 是我们选择的参数)。

例如，我们可以选择搜索以下各项的不同值:

*   批量
*   时代数
*   初始化模式

选择在字典中指定，并传递给 GridSearchCV。

我们现在将对`batch size`、`number of epochs`和`initializer`的组合进行网格搜索。

结束前的最后一个问题:如果我们必须在 GridSearchCV 中循环的参数和值的数量特别大，我们该怎么办？

这可能是一个特别麻烦的问题——想象一下，有五个参数被选择，而我们为每个参数选择了 10 个潜在值。这种独特组合的数量是 10⁵，这意味着我们必须训练数量多得离谱的网络。这样做是疯狂的，所以通常使用 RandomizedCV 作为替代。

RandomizedCV 允许我们指定所有的潜在参数。然后，对于交叉验证中的每个折叠，它选择一个随机的参数子集用于当前模型。最后，用户可以选择一组最佳参数，并将其用作近似解。

# **最终意见**

感谢您的阅读，希望这篇文章对您有所帮助和启发。我期待着从读者那里听到他们对这个超参数调整指南的应用。本系列的下一篇文章将涵盖全连接神经网络的一些更高级的方面。

## **延伸阅读**

深度学习课程:

*   吴恩达的机器学习课程有一个很好的神经网络介绍部分。
*   Geoffrey Hinton 的课程:[用于机器学习的 Coursera 神经网络(2012 年秋季)](https://www.coursera.org/course/neuralnets)
*   [迈克尔·尼尔森的免费书籍*神经网络和深度学习*](http://neuralnetworksanddeeplearning.com/)
*   约舒阿·本吉奥、伊恩·古德菲勒和亚伦·库维尔写了一本关于深度学习的书
*   [雨果·拉罗歇尔在舍布鲁克大学的课程(视频+幻灯片)](http://info.usherbrooke.ca/hlarochelle/neural_networks/content.html)
*   [斯坦福关于无监督特征学习和深度学习的教程(吴恩达等人)](http://ufldl.stanford.edu/wiki/index.php/Main_Page)
*   [牛津大学 2014-2015 年 ML 课程](https://www.cs.ox.ac.uk/people/nando.defreitas/machinelearning/)
*   [英伟达深度学习课程(2015 年夏季)](https://developer.nvidia.com/deep-learning-courses)
*   [谷歌在 Udacity 上的深度学习课程(2016 年 1 月)](https://www.udacity.com/course/deep-learning--ud730)

面向 NLP:

*   [Stanford CS224d:自然语言处理的深度学习(2015 年春季)，作者 Richard Socher](http://cs224d.stanford.edu/syllabus.html)
*   [NAACL HLT 2013 上的教程:自然语言处理的深度学习(无魔法)(视频+幻灯片)](http://nlp.stanford.edu/courses/NAACL2013/)

以视觉为导向:

*   [用于视觉识别的 CS231n 卷积神经网络](http://cs231n.github.io/)作者 Andrej Karpathy(之前的版本，更短更不完善:[黑客的神经网络指南](http://karpathy.github.io/neuralnets/))。

重要的神经网络文章:

*   [神经网络中的深度学习:概述](https://www.sciencedirect.com/science/article/pii/S0893608014002135)
*   [使用神经网络的持续终身学习:综述——开放存取](https://www.sciencedirect.com/science/article/pii/S0893608019300231)
*   [物理储层计算的最新进展:综述—开放存取](https://www.sciencedirect.com/science/article/pii/S0893608019300784)
*   [脉冲神经网络中的深度学习](https://www.sciencedirect.com/science/article/pii/S0893608018303332)
*   [集成神经网络(ENN):一种无梯度随机方法——开放存取](https://www.sciencedirect.com/science/article/pii/S0893608018303319)
*   [多层前馈网络是通用逼近器](https://www.sciencedirect.com/science/article/pii/0893608089900208)
*   [深度网络与 ReLU 激活函数和线性样条型方法的比较——开放访问](https://www.sciencedirect.com/science/article/pii/S0893608018303277)
*   [脉冲神经元网络:第三代神经网络模型](https://www.sciencedirect.com/science/article/pii/S0893608097000117)
*   [多层前馈网络的逼近能力](https://www.sciencedirect.com/science/article/pii/089360809190009T)
*   [关于梯度下降学习算法中的动量项](https://www.sciencedirect.com/science/article/pii/S0893608098001166)