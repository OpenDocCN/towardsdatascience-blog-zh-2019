# 5 分钟论文综述:进化随机梯度下降

> 原文：<https://towardsdatascience.com/5-minute-paper-review-evolutionary-stochastic-gradient-descent-7aa1e0a46314?source=collection_archive---------15----------------------->

## 进化算法如何加速深度神经网络的优化

本文提出了一种新的进化版本的深度神经网络随机梯度下降。随机梯度下降(缩写为 SGD)是由 Robins 和 Monro 在他们的论文“随机近似方法”中首次提出的。它本质上是一种迭代优化方法，在迭代 *k* 期间随机抽取一个样本，并使用它来计算其梯度。然后，在给定步长(或学习速率)的情况下，将得到的随机梯度用于更新深度网络的权重。)在原始 SGD 算法中，这些被更新的权重被视为单个参数向量 *θ* 。

![](img/7f79144435b35b93f6f06d6e622cd161.png)

Photo by [timJ](https://unsplash.com/@the_roaming_platypus?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

在他们的介绍中，作者认为 SGD 和进化算法(EAs)之间的互补关系值得研究。与 SGD 不同，EAs 通常是无梯度的。作者假设进化算法在处理复杂优化问题时比遗传算法更有优势，因为它们没有梯度。他们将 EAs 和 SGD 结合起来的提议可能有助于在大型分布式网络上实现优化。

作者提出的新方法使用参数向量 *θ* 的经验风险作为适应度函数，并在给定一组参数 *θs* 的情况下搜索最小适应度值。在他们的方法和实现部分，作者给出了提出的进化随机梯度下降(ESGD)算法背后的数学的详细描述。主要思想是给定一组随机初始化的参数向量，该算法搜索给出最低经验风险的参数向量。最好的后代是通过*m*-精英平均适应度选择的，精英平均适应度实质上是按照升序排列的最好的 *m* 个个体的平均适应度。例如，当 *m* = 3 时，则 *m-* 精英平均适应度给出整个群体中最好的 3 个个体的适应度。

![](img/5daa71550077863f3215f6481ab8f581.png)

Cui, X., Zhang, W., Tüske, Z. and Picheny, M. (2019). *Evolutionary Stochastic Gradient Descent for Optimization of Deep Neural Networks*. [online] arXiv.org. Available at: [https://arxiv.org/abs/1810.06773](https://arxiv.org/abs/1810.06773)

在每一代 ESGD 中，初始种群是通过参数的随机向量化产生的。使用预定的优化器(包括常规 SGD 和自适应矩- ADAM 优化器)和一组固定的超参数，更新每个个体，直到其适应度下降。然后对群体的最终适应值进行排序，并选择前 *m* 个个体加上 *-m* 个附加个体作为父代。下一代的后代是通过这些表现最好的个体的中间重组和随机零均值高斯噪声的添加而产生的。当群体的经验风险最小时，产生最小适应值的参数集被选为模型的真实参数。

使用 ESGD 在包括 BN-50、SWB300、CIFAR10 和 PTB 在内的众所周知的基准数据集上进行了多个实验。这些数据集涵盖了涉及语音识别(BN-50 和 SWB300)、图像识别(CIFAR10)和语言建模(PTB)的任务。选择用于评估的三个深度学习模型是深度神经网络(DNN)、ResNet-20(残差网络)和 LSTM(长短期记忆网络)。使用两个附加基线模型的适应值作为参考，作者表明由他们提出的算法产生的适应值在上述数据集内总是不增加的。

![](img/507fdb48fcf9ff5c67d7d01a5661936e.png)

Cui, X., Zhang, W., Tüske, Z. and Picheny, M. (2019). *Evolutionary Stochastic Gradient Descent for Optimization of Deep Neural Networks*. [online] arXiv.org. Available at: [https://arxiv.org/abs/1810.06773](https://arxiv.org/abs/1810.06773)

在他们的讨论中，作者提出了保持 ESGD 种群多样性的重要性。这是因为如果群体在其适应值方面变得同质，则算法将达到过早收敛，并且将不会产生理想的结果。因此，ESGD 中引入了 *m-* 精英选择策略，作为一种在种群中诱导多样性的措施。除了种群多样性，作者还指出了由 ESGD 产生的互补优化器的重要性。众所周知，Adam 优化器可以比传统的 SGD 更快地达到收敛。然而，在更长的运行中，SGD 往往会赶上并达到比 Adam 更好的最佳点。作者建议，在 ESGD 中，不仅选择最佳参数，而且选择它们的互补优化器，因为它们负责产生更好的适应值。

总之，ESGD 算法提供了一种为深度学习算法选择最佳参数的新方法。该算法的有效性已经在 4 个数据集和 3 个不同的深度学习模型上得到验证。总的来说，SGD 算法可以被看作是一种共同进化，个体一起进化，为下一代产生更健康的后代。

![](img/91654ad6d3ec3d1a4afd1f8735fcfd96.png)

Photo by [Suzanne D. Williams](https://unsplash.com/@scw1217?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

这是一篇写得很好，对我来说特别有趣的研究论文，因为优化是我研究中最重要的方面之一。它实际上在其方法部分之上有一个实现部分。这有助于像我这样的实际头脑更好地理解所提出的数学在实践中是如何实现的。我想这篇论文自从被选为 2018 年 NeuroIPS (NIPS)会议的参赛作品以来，已经经历了严格的审查。因此，这篇文章条理清晰、格式良好，对我来说并不奇怪。然而，对于在 ESGD 群体中用于优化的实际优化器家族，尤其存在模糊性。作者可以通过列出他们算法中使用的所有优化器和超参数来提供更多的清晰度，而不是仅仅列出几个例子，然后继续下一部分。我也很好奇为什么作者选择使用传统的 SGD 和 ADAM 作为优化器的选择，而不是其他优化器，如 RMSProp、Adagrad 和 Adadelta。似乎没有解释为什么特别选择 SGD 和 ADAM 作为群体的优化器。

![](img/9798faf5b1cd9a7d181ca4d66437874b.png)

Cui, X., Zhang, W., Tüske, Z. and Picheny, M. (2019). *Evolutionary Stochastic Gradient Descent for Optimization of Deep Neural Networks*. [online] arXiv.org. Available at: [https://arxiv.org/abs/1810.06773](https://arxiv.org/abs/1810.06773)

实验部分很好地总结了在单基线模型、top-15 基线模型和 ESGD 上运行的多个实验所获得的结果。然而，我发现图 1 中的第四张图非常令人困惑。该图包含 PTB 数据集的 LSTM 网络结果。即使作者解释了 ESGD 模型中困惑的增加，他们也没有解决 top-15 模型中适应值的平坦化。

我最喜欢的文章部分可能是讨论部分。作者就他们算法的重要性提供了许多有意义的讨论。我很高兴看到作者们讨论他们算法的计算效率。他们得出结论，鉴于现代 GPU 的并行计算能力，ESGD 算法应该与常规 SGD 算法花费大约相同的时间来计算端到端。然而，作者没有为 ESGD 提供任何未来的步骤或可能的扩展。我希望看到的一个可能的扩展不仅是网络参数和优化器的共同进化，而且是实际网络架构的共同进化。