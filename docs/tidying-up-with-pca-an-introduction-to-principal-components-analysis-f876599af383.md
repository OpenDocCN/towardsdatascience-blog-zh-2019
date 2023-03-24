# 主成分分析整理:主成分分析导论

> 原文：<https://towardsdatascience.com/tidying-up-with-pca-an-introduction-to-principal-components-analysis-f876599af383?source=collection_archive---------4----------------------->

![](img/6b17b6ac9e252ca28037c1711f95131d.png)

[主成分分析(PCA)](https://en.wikipedia.org/wiki/Principal_component_analysis%20) 是一种[降维](https://en.wikipedia.org/wiki/Dimensionality_reduction)的技术，是减少数据集中预测变量数量的过程。

更具体地说，PCA 是一种[无监督](https://en.wikipedia.org/wiki/Unsupervised_learning)类型的[特征提取](https://en.wikipedia.org/wiki/Feature_extraction)，其中原始变量被组合并缩减为它们最重要和最具描述性的成分。

主成分分析的目标是识别数据集中的模式，然后将变量提取到它们最重要的特征，以便在不丢失重要特征的情况下简化数据。主成分分析询问是否一个数据集的所有维度都能引发快乐，然后给用户一个选项来消除那些不快乐的维度。

![](img/c3f14c6c311f74778f1df4aafeeeb234.png)

PCA 是一种非常流行的技术，但是实施它的人通常不太理解它。我写这篇博文的目的是提供一个关于为什么使用 PCA 以及它如何工作的高层次概述。

**维度的诅咒(或者说，为什么要费神降维？)**

[维度的诅咒](https://en.wikipedia.org/wiki/Curse_of_dimensionality)是一组现象，表明*随着维度的增加，数据的可管理性和有效性往往会降低*。在高层次上，维数灾难与这样一个事实有关，即随着维度(变量/特征)被添加到数据集，点(记录/观察)之间的平均和最小距离增加。

![](img/11b75e164a8332fa7caa7f8746125340.png)

*我发现，当我开始思考聚类或 PCA 等主题时，将变量可视化为维度，将观察结果可视化为记录/点会有所帮助。数据集中的每个变量都是一组坐标，用于绘制问题空间中的观察值。*

随着已知点和未知点之间的距离增加，创建好的预测变得更加困难。此外，数据集中的特征可能不会在目标(独立)变量的上下文中增加太多价值或预测能力。这些特征并没有改善模型，而是增加了数据集中的[噪声](https://en.wikipedia.org/wiki/Noise_(signal_processing)%20)，以及模型的整体计算负载。

由于维数灾难，降维通常是分析过程的关键组成部分。特别是在数据具有高维度的应用中，如[计算机视觉](https://en.wikipedia.org/wiki/Computer_vision)或[信号处理](https://en.wikipedia.org/wiki/Signal_processing)。

当收集数据或应用数据集时，并不总是显而易见或容易知道哪些变量是重要的。甚至不能保证你选择或提供的变量是正确的*变量。此外，在大数据时代，数据集中的变量数量可能会失控，甚至变得令人困惑和具有欺骗性。这使得手工选择有意义的变量变得困难(或不可能)。*

别担心，PCA 会查看数据集中连续变量的整体结构，以便从数据集中的噪音中提取有意义的信号。它旨在消除变量中的冗余，同时保留重要信息。

![](img/d6ecf635b4b48b8543962233c103df84.png)

PCA 也爱乱七八糟。

**PCA 如何工作**

PCA 最初来自于线性代数领域。它是一种变换方法，在数据集中创建原始变量的(加权[线性](https://en.wikipedia.org/wiki/Linear_combination))组合，目的是新的组合将尽可能多地捕获数据集中的[方差](https://en.wikipedia.org/wiki/Variance%20)(即点之间的间隔)，同时消除相关性(即冗余)。

PCA 通过使用从原始变量的[协方差矩阵](https://en.wikipedia.org/wiki/Covariance_matrix)计算的特征向量和特征值，将数据集中的原始(以平均值为中心的)观察值(记录)转换为一组新的变量(维度)来创建新的变量。

那是一口。让我们分解一下，从均值开始——以原始变量为中心。

PCA 的第一步是[将所有输入变量的值](https://en.wikipedia.org/wiki/Centering_matrix%20)居中(例如，从值中减去每个变量的平均值)，使每个变量的平均值等于零。[居中是一个重要的](https://stats.stackexchange.com/questions/189822/how-does-centering-make-a-difference-in-pca-for-svd-and-eigen-decomposition)预处理步骤，因为它确保生成的组件只关注数据集中的方差，而不捕捉数据集的整体平均值作为重要变量(维度)。如果没有均值居中，PCA 找到的第一个主成分可能[对应于数据的均值](https://stats.stackexchange.com/questions/22329/how-does-centering-the-data-get-rid-of-the-intercept-in-regression-and-pca%20)，而不是最大方差的方向。

一旦数据被居中(并且[可能被缩放](https://scikit-learn.org/stable/auto_examples/preprocessing/plot_scaling_importance.html)，取决于变量的单位)，数据的协方差矩阵需要被计算。

协方差是一次在两个变量(维度)之间测量的，它描述了变量的值彼此之间的相关程度:例如，随着变量 x 的观察值增加，变量 y 也是如此吗？较大的协方差值(正或负)表示变量之间有很强的线性关系。接近 0 的协方差值表示弱的或不存在的线性关系。

![](img/7ef2995791e790d735c3c34fd890ebee.png)

*这个来自*[*https://stats . stack exchange . com/questions/18058/how-would-you-explain-协方差-to someone-someone-who-understand-only-the-mean*](https://stats.stackexchange.com/questions/18058/how-would-you-explain-covariance-to-someone-who-understands-only-the-mean)*的可视化对于理解协方差是超级有帮助的。*

协方差总是在二维空间中测量。如果处理两个以上的变量，确保获得所有可能的协方差值的最有效方法是将它们放入一个矩阵中(因此称为协方差矩阵)。在协方差矩阵中，对角线是每个变量的方差，对角线上的值是彼此的镜像，因为变量的每个组合在矩阵中包含两次。这是一个正方形的对称矩阵。

![](img/e9bab8dc20e3817d1dfc7bd1c4d1f6a0.png)

在本例中，变量 A 的方差为 0.67，第二个变量的方差为 0.25。两个变量之间的协方差是 0.55，这反映在矩阵的主对角线上。

因为它们是正方形和对称的，协方差矩阵是[可对角化的](https://en.wikipedia.org/wiki/Diagonalizable_matrix)，这意味着可以在矩阵上计算[特征分解](https://en.wikipedia.org/wiki/Eigendecomposition_of_a_matrix)。这是 PCA 找到数据集的特征向量和特征值的地方。

[线性变换](https://en.wikipedia.org/wiki/Linear_map)的[特征向量](https://en.wikipedia.org/wiki/Eigenvalues_and_eigenvectors)是一个(非零)向量，当对其应用相关的线性变换时，该向量改变其自身的[标量](https://en.wikipedia.org/wiki/Scalar_(mathematics))倍数。特征值是与特征向量相关的标量。我发现对理解特征向量和值最有帮助的事情是看一个例子(如果这没有意义，试试看 Khan Acadamy 的这个[矩阵乘法](https://www.khanacademy.org/math/precalculus/precalc-matrices/multiplying-matrices-by-matrices/v/matrix-multiplication-intro)课程)。

![](img/64fb7ebe51b1c78cff242334ae326d31.png)

*有人告诉我，矩阵乘法中包含一个*是不符合惯例的，但为了清楚起见，我还是保留了它。我向读到这篇文章的任何被冒犯的数学家道歉。*
在这个例子中，

![](img/8f9a15e1dfad2432867972d3acd87bee.png)

是特征向量，5 是特征值。

在高层次理解 PCA 的背景下，关于特征向量和特征值，你真正需要知道的是协方差矩阵的特征向量是数据集中主要成分的轴。特征向量定义了由 PCA 计算的主分量的方向。与特征向量相关联的特征值描述了特征向量的大小，或者观察值(点)沿着新的轴分散了多远。

![](img/be07d6aa3273184961013419640789de.png)

第一个特征向量将跨越在数据集中发现的最大方差(点之间的间隔)，并且所有随后的特征向量将垂直于(或者用数学术语来说，[正交](https://en.wikipedia.org/wiki/Orthogonality))在它之前计算的特征向量。这就是我们如何知道每一个主成分都是不相关的。

如果你想了解更多关于特征向量和特征值的知识，有[多个](http://www.math.jhu.edu/~bernstein/math201/EIGEN.pdf) [资源](https://www.khanacademy.org/math/linear-algebra/alternate-bases/eigen-everything/v/linear-algebra-introduction-to-eigenvalues-and-eigenvectors) [散布在](http://math.mit.edu/~gs/linearalgebra/ila0601.pdf) [互联网上](http://setosa.io/ev/eigenvectors-and-eigenvalues/)正是为了这个目的。为了简洁起见，我将避免在博客中试图教线性代数(糟糕的)。

PCA 找到的每个特征向量将从数据集中的原始变量中提取方差的组合。

![](img/2d0d3171493aadce797a106fd8b9daab.png)

*在本图中，主成分 1 说明了变量 A 和 b 的差异。*

特征值很重要，因为它们为新导出的变量(轴)提供了排序标准。主成分(特征向量)按特征值降序排序。具有最高特征值的主成分被“首先挑选”作为主成分，因为它们占数据中最大的方差。

您可以指定返回几乎与原始数据集中的变量一样多的主成分(通常是从[到 n-1](https://stats.stackexchange.com/questions/123318/why-are-there-only-n-1-principal-components-for-n-data-if-the-number-of-dime) ，其中 n 是原始输入变量的数量)，但是大部分方差将在顶部的主成分中考虑。有关选择多少主要组件的指导，请查看此[堆栈溢出讨论](https://stackoverflow.com/questions/12067446/how-many-principal-components-to-take)。或者你可以一直问自己，“自我，有多少维度会激发快乐？”(这是一个笑话，你可能应该只使用一个 [scree 情节。](https://en.wikipedia.org/wiki/Scree_plot))

![](img/452676a612628a8520cf74da2831017d.png)

*Scree 图显示了每个主成分的方差。该屏幕图是为 Alteryx Designer 中的主要组件工具的报告输出而生成的。*

在确定了数据集的主成分后，需要将原始数据集的观测值转换为所选的主成分。

为了转换我们的原始点，我们创建一个投影矩阵。这个投影矩阵就是选择的特征向量串接成的一个矩阵。然后，我们可以将原始观测值和变量的矩阵乘以我们的投影矩阵。这个过程的输出是一个转换后的数据集，投射到我们的新数据空间中——由我们的主成分组成！

![](img/4176e1adf4f39435efa4af26248e3aac.png)

就是这样！我们完成了 PCA。

![](img/70d167416711ee84e04ab89b34a37c45.png)

**假设和限制**

在应用 PCA 之前，有一些事情需要考虑。

[在执行 PCA](https://stats.stackexchange.com/questions/69157/why-do-we-need-to-normalize-data-before-principal-component-analysis-pca%20) [之前对数据](https://stats.stackexchange.com/questions/53/pca-on-correlation-or-covariance)进行标准化可能很重要，特别是当变量有[不同的单位或标度](https://www.stat.ncsu.edu/people/bloomfield/courses/st783/chapter-08-2.pdf)时。您可以在 Designer 工具中选择选项*缩放每个字段，使其具有单位方差。*

PCA 假设数据可以用线性结构来近似，并且数据可以用较少的特征来描述。它假设线性变换能够并且将会捕获数据的最重要的方面。它还假设数据中的高方差意味着信噪比高。

降维确实会导致一些信息的丢失。由于没有保留所有的特征向量，因此会丢失一些信息。然而，如果没有包含的特征向量的特征值很小，你没有丢失太多的信息。

使用 PCA 要考虑的另一个问题是，变量在转换后变得不太容易解释。一个输入变量可能意味着一些特定的东西，如“紫外线照射”，但 PCA 创建的变量是原始数据的大杂烩，不能以一种清晰的方式解释，如“紫外线照射的增加与皮肤癌发病率的增加相关”当你向别人推销你的模型时，更难解释也意味着更难解释。

**优势**

主成分分析是流行的，因为它可以有效地找到一个更少维度的数据集的最佳表示。它在过滤噪声和减少冗余方面是有效的。如果您有一个包含许多连续变量的数据集，并且您不确定如何为您的目标变量选择重要的特征，那么 PCA 可能非常适合您的应用。类似地，PCA 也很受欢迎，用于可视化高维数据集(因为我们这些贫乏的人类很难在超过三维的空间中思考)。

**附加资源**

我最喜欢的教程(包括基础数学的概述)来自奥塔哥大学的 Lindsay I . Smith[关于主成分分析的教程](https://ourarchive.otago.ac.nz/bitstream/handle/10523/7534/OUCS-2002-12.pdf?sequence=1&isAllowed=y)。

这是 UCSD 的 Jon Shlens 关于主成分分析的另一个很棒的[教程](https://www.cs.princeton.edu/picasso/mats/PCA-Tutorial-Intuition_jp.pdf)

[关于 PCA](http://alexhwilliams.info/itsneuronalblog/2016/03/27/pca/) 你所知道和不知道的一切，来自博客它的神经元专注于神经科学中的数学和计算。

[3 个简单步骤中的主成分分析](http://sebastianraschka.com/Articles/2015_pca_in_3_steps.html)有一些很好的说明，并被分解成离散的步骤。

Jeremy Kun 博客中的[主成分分析](https://jeremykun.com/2012/06/28/principal-component-analysis/%C2%A0)是一篇很好的、简洁的文章，其中包含了对到[特征脸](https://en.wikipedia.org/wiki/Eigenface)的[引用。](https://jeremykun.com/2011/07/27/eigenfaces/)

[主成分分析的一站式商店](/a-one-stop-shop-for-principal-component-analysis-5582fb7e0a9c)来自 Matt Brems。

[原](https://community.alteryx.com/t5/Data-Science-Blog/Tidying-up-with-PCA-An-Introduction-to-Principal-Components/ba-p/382557)。经允许重新发布。