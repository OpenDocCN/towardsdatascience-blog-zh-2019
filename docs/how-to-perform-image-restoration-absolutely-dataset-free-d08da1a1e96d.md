# 如何在完全无数据集的情况下执行图像恢复

> 原文：<https://towardsdatascience.com/how-to-perform-image-restoration-absolutely-dataset-free-d08da1a1e96d?source=collection_archive---------5----------------------->

## 无学习神经网络图像恢复

![](img/d0430e72c11254c85d4a951f1c967da8.png)

**深度学习需要大量的数据。**这个短语在考虑对其数据应用深度学习方法的人群中变得流行起来。当没有足够“大”的数据时，人们通常会担心，这主要是因为人们普遍认为深度学习只能使用大量数据。这不是真的。

尽管在某些情况下，你确实需要大量的数据，但是有一些网络可以在一张图片上进行训练。最重要的是，在实践中，即使没有大型数据集，网络本身的结构也可以防止深度网络过度拟合。

这篇文章是关于“[](https://dmitryulyanov.github.io/deep_image_prior)**之前的深度图像”，这是 Dmitry Ulyanov 在 2018 年 CVPR 发表的一篇有趣的论文。本文证明了 CNN 的结构足以解决图像恢复问题。简而言之，这篇论文声称 CNN 包含了自然图像的“知识”。此外，作者将这一主张用于图像恢复任务，如图像去噪、超分辨率、绘画等。**

**在这篇文章中，我将涉及三件事:首先，图像恢复任务和一些用例的概述。其次，概述“深度图像先验”以及它如何用于图像恢复任务。最后，我们将使用[深度图像先验来执行去噪任务——使用神经网络进行图像恢复，但不需要学习 PyTorch 中实现的 GitHub 知识库](https://github.com/erezposner/deep-image-prior)。**

# **图像恢复**

**当我们提到图像恢复问题时，我们基本上是指我们有一个降级的图像，我们希望恢复干净的非降级图像。图像降质的原因有很多，主要是图像在传输、形成和存储过程中会发生降质。**

**图像恢复的任务很多，我们来说三个主要任务:**

****去噪和一般重建****

**图像去噪是指试图恢复被附加噪声或压缩等来源污染的图像。**

**![](img/dbfa2baf5d007dda58bd7990854ecc16.png)**

**Figure 1 — Building! — (Left) Noisy Image, (Right) Denoised Image**

## **超分辨率**

**超分辨率的目标是获取低分辨率图像，并对其进行上采样以创建高分辨率版本。**

**![](img/5d27df6245a6871c379d265e63e2bc12.png)**

**Figure 2 — Fruits! — (Left) Low-Resolution Image, (Right) High-Resolution Image**

****图像内置****

**图像补绘是对图像和视频中丢失或退化的部分进行重建的过程。这种技术通常用于从图像中移除不想要的对象或恢复旧照片的损坏部分。下图显示了画中画结果示例。**

**![](img/48a0edd69f6684ed38d5e8dd60ddb0da.png)**

**Figure 3 — (Left) Damaged Image, (Right) Reconstructed Image**

**当然，还有更多的用例，但是现在让我们试着理解这篇论文的新颖技术。**

# **深度图像先验**

## **1.什么是“先验”？**

**考虑一下，你需要自己执行超分辨率任务。例如，给你一张低分辨率的图像(下图 4 中左边的图像)，一支笔和一张纸，然后要求你解决它。希望这就是你要画的(下图 4 中的右图)。**

**![](img/16fa3bee812d539a5fbc834da18f4278.png)**

**Figure 4 — drawing a face from memory — low resolution to high resolution**

**那么，你是怎么做到的呢？你可能会利用你对世界的了解；什么是脸，脸的结构，即眼睛、鼻子、嘴等的位置。您还可以使用低分辨率图像中的特定信息。因此，我们可以更直观地将先验定义为我们在缺乏信息的情况下的基本信念。例如，在图像的情况下，图像上的先验基本上代表我们认为自然图像应该看起来像什么。**

## **2.已知和明确的前科**

**如果您希望计算机进行图像恢复，例如图像去噪，您可能会收集大量干净和有噪声的图像数据集，并训练一个深度神经网络，将有噪声的图像作为输入，只获得干净的图像作为输出。因此，可以说网络通过数据集学习先验知识。这种方法叫做**事先学习。****

**问题是，这种方法需要大量的噪声和干净的图像对。**

**![](img/942794cdc2d3af936f3306dbbc292bcb.png)**

**Figure 5 — Left —noisy images as an input to our network, Right — clean images as the networks output**

**解决这个任务的另一种方法是执行**显式先验**或**手工先验**，其中我们不需要使用除了我们的图像之外的任何附加数据。**

**我们可以把这个问题看作一个优化问题，以产生期望的干净图像 ***x*** ，其中我们的目标是创建一个图像 ***x**** ，它既接近于噪声图像***【x^***，又像干净图像 ***x*** 一样“自然”或“清晰”。**

**![](img/cafda1492bdf0569e4c8a0e2b998fc63.png)**

**Figure 6 — Image restoration task — Given the observed corrupted image x^ we want to to get the restored image x* that is close to our corrupted image but as clear as possible**

**例如，我们可以使用用于去噪任务的像素值之间的 l2 距离或者其他任务相关的数据项来测量被标注为数据项***【e(x,x^】、*** 的“接近度”。**

**除了数据项，我们假设有一个函数 *R(x)* 可以测量图像的“不自然”或“不清晰”。在这种情况下，我们的优化目标的公式将是最大后验分布，以估计来自经验数据的未观察值:**

**![](img/2e589dd12faf40738a8db890917ee660.png)**

**E(x;x^)is the data term which is negative log of the likelihood and R(x) is the image prior term which is negative log of the prior.**

**数据项将项拉向原始图像，确保图像不会偏离太远。另外右边的项，即 R(x)，将 ***x*** 拉向自然图像的方向，(有希望地)降低噪声。所以我们可以把 *R(x)* 看成一个正则项。如果没有它，优化器将“过度适应”噪声图像。因此，我们定义先验/正则项的方式对于获得好的结果至关重要。**

**不幸的是，我们没有自然图像的精确先验。传统上，我们使用手工制作的特征来表示先验，但是这些总是包含一定程度的任意性。本文的本质是细胞神经网络可以作为图像的先验；换句话说，CNN 以某种方式“知道”自然图像应该是什么样子，不应该是什么样子。**

## **2.定义先验的网络结构**

**因此，呈现最小化图像 ***x*** 上的函数的任务**

**![](img/8cc4ecc3f29f8bab9e23670b415df91a.png)**

**Optimize at image space**

**传统的方法是在图像空间最小化该函数，在该空间进行初始估计，基本上用噪声初始化图像，然后计算该函数相对于*x 的梯度，更新我们的权重并重复，直到收敛。***

***![](img/6accb2bbf68ef6d5cd110197b888ce25.png)***

***Figure 7 — Visualization of the conventional approach optimizing over image space *x****

***但是我们能换一种方式吗？我们可以说，每个图像 ***x*** 都是一个函数的输出，该函数将一个值从不同的空间映射到图像空间。***

***![](img/10cff257551e2a3f6899adbf6a658f56.png)***

***这里，我们有参数空间θ，并且我们有从参数空间θ到图像 ***x*** 的映射，并且不是在图像上优化，而是在θs 上完成优化***

***![](img/ffbc96ff7238d39a25fa479119d2affd.png)***

***Figure 8 — Visualization of the proposed parametric approach optimizing over parameter space θ***

***在图 8 中，我们可以看到，我们从参数空间中的初始值开始，并且我们立即将其映射到图像空间，计算相对于 g(的梯度。)，接着使用梯度下降进行θ更新，并重复直到收敛。***

***那么，我们为什么要这么做呢？在图像空间上优化和在参数空间上优化有什么区别？函数 g(.)可以被视为一个超参数，它可以被调整以突出显示我们想要得到的图像。即“自然”图像。如果我们仔细想想，函数 g(θ)实际上定义了一个先验。因此不是优化两个分量的和。我们现在将只优化数据项。***

***![](img/5225480bd68aebf39b93d4b96c83a783.png)***

***我们可以定义一个网络结构，例如 UNet 或 Resnet，并将θ定义为网络参数。因此，我们将最小化函数表示如下:***

***![](img/3da86ec598faa0b7d6681db95fc9593a.png)***

***其中， *z* 是随机固定输入图像，θ是随机初始化的权重，将使用梯度下降来更新该权重，以获得期望的输出图像。***

***明白了吗？这里的变量是θ！与其他类型的网络不同，在其他类型的网络中，固定权重并改变输入以获得不同的输出，在这里，他们固定输出并改变权重以获得不同的输出。这就是他们如何得到映射函数 g(。)到图像空间。***

## ***3.为什么使用这种参数化？***

***你可能会认为使用这种参数化会导致相同的噪声图像，因为我们基本上是过度拟合原始噪声图像。作者在论文中指出，虽然自然图像的优化更快更容易。***

***![](img/377c956052c174fbd9c135615a0d3eb8.png)***

***Figure 9— Learning curves for the reconstruction task using: a natural image, the same plus i.i.d. noise, the same randomly scrambled, and white noise. Naturally-looking images result in much faster convergence, whereas noise is rejected.***

***每条曲线代表我们优化图像和噪声以及添加噪声的图像时的损失变化。该图显示，与噪声相比，自然图像的损失收敛得更快。这意味着，如果我们在适当的时机中断训练，我们可以获得一个“自然”的图像。这就是为什么这篇论文将 CNN 视为先验:它(不知何故)**偏向于产生自然图像**。这允许我们使用 CNN 解码器作为在某些限制下生成自然图像的方法。***

# ***结果***

***让我们看看一些常见任务的结果。***

## ***JPEG 压缩图像的盲复原***

***深度图像先验可以恢复复杂退化的图像(在这种情况下是 JPEG 压缩)。随着优化过程的进行，深度图像先验允许恢复大部分信号，同时在最终过度拟合输入(在 50K 次迭代时)之前消除光晕和块效应(在 2400 次迭代之后)。***

***![](img/95463de73a6c39b79ba015136cbb3430.png)***

***Figure 10— Deep Image prior output for different iterations***

## ***图像内嵌***

***在下图中，in-painting 用于移除覆盖在图像上的文本。深度图像先验结果导致几乎完美的结果，几乎没有伪影。***

***![](img/8a31abe5f3c7d98b68b268ae0d2af911.png)***

***Figure 11 — in-painting result***

## ***图像去噪***

***深度图像先验成功地恢复了人造和自然模式。***

***![](img/356bae6bd92be8f0f792618cfa5de4f4.png)***

***Figure 12 — Image denoising results***

# ***PyTorch 中深度图像先验的实现***

***既然我们已经看到了深度图像先验背后的概念和数学。让我们实现它，并在 PyTorch 中执行去噪任务。整个项目在[深度图像先验中可用——使用神经网络进行图像恢复，但不需要学习 GitHub 知识库](https://github.com/erezposner/deep-image-prior)。***

***笔记本结构如下:***

## ***预处理***

***前几个单元格与导入库有关，所以要确保正确安装了所有的依赖项。如果在 [GitHub 库](https://github.com/erezposner/deep-image-prior)中列出，你需要安装来执行代码的库列表。此外，这是你选择图像去噪的地方。***

***在这个例子中，我选择了一张使用[杂色噪声生成器](https://github.com/erezposner/Shot-Noise-Generator) GitHub 知识库应用杂色噪声的图片，如下图所示。***

***![](img/87fa820afef7aa0893a14707089bf413.png)***

***Figure 13— U.S Capitol Building — (Left) Clean image, (Right) Noisy image after adding additive noise***

## ***最佳化***

***下面的代码是神奇的地方，随机初始化的图像 *z* 在 closure()函数中反复更新。计算数据项(本例中为 MSE ),更新参数空间θ get。***

***该块将在每次迭代中生成一个图像，这样您就可以跟踪优化过程的进度。当你对结果满意时，停止这个过程并运行下面的代码来创建一个很棒的 gif 来可视化整个过程。***

***请看下面我们实现的去噪任务的结果。右边是嘈杂的图像，左边是整个复原过程。***

***太神奇了！***

***![](img/c5b268fea57415163a63365f9317e6a0.png)***

***Figure 14 — (Left) —Clean image *x** restoration result using Deep Image Prior starting from random initialization up to convergence , (Right) — The Noisy image x^***

# ***结论***

***如果你对源代码感兴趣，可以在我的[深度图像先验中找到——用神经网络进行图像恢复，但不需要学习 GitHub 库](https://github.com/erezposner/deep-image-prior)。***

***一如既往，如果您有任何问题或意见，请随时在下面留下您的反馈，或者您可以随时通过 [LinkedIn](http://www.linkedin.com/in/erezposner) 联系我。***

***在那之前，下一篇文章再见！😄***