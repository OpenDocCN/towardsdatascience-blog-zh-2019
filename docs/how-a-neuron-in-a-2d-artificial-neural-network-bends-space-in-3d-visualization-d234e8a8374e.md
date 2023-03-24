# 2D 人工神经网络中的神经元如何在 3D 可视化中弯曲输出

> 原文：<https://towardsdatascience.com/how-a-neuron-in-a-2d-artificial-neural-network-bends-space-in-3d-visualization-d234e8a8374e?source=collection_archive---------26----------------------->

这是为那些想直观了解人工神经网络中神经元内部工作的人准备的。你只需要一点点高中水平的数学和一点点耐心。我们将构建一个简单的程序来区分苹果和橘子，然后从这里开始深入研究。开始了。

第一步:去水果市场买苹果和橘子。

第二步:我们中的一个书呆子根据两个属性给它们打分——红色和甜蜜。

> 瞧——我们现在有二维空间了。x 代表红色，Y 代表甜味。这就是我所说的 2D 神经元，即有两个输入的神经元。

我们大多数人都会同意苹果通常更甜更红，因此我们最终会得到一个类似下图的图表。

让我们也画一条好的线来分类他们。我们如何提出一条线的规范是另一个故事的主题，但这不会妨碍我们的可视化。

![](img/b99523f2327003d646135b036a6fbc9f.png)

2D graph for apples and oranges on a plane (x-axis: redness, y-axis: sweetness)

# 但是线到底是什么？

在斜率截距格式中，为 ***y = mx+ c*** (其中 m 为斜率，c 为 y 截距)。

一切都好吗？通过一些简单的重新排列，这可以表示为

***w1 * x + w2 * y + b1 = 0***

> 如果我现在用这个方程并在它周围画一些圈，它将开始看起来像一个我们非常熟悉的整体——是的，这是我们的人工神经网络的基本构件，也就是神经元。

![](img/dec786d6fe52643e0a0fb40466a74339.png)

现在，如果我们用一个 *z* 代替等式中的 0，它将把我们的视觉带到一个全新的维度——我是指字面上的意思。

***w1 * x+w2 * y+B1 = z***

或者*z =****w1 * x+w2 * y+B1***

这是一个 3D 平面的方程(不设猜奖)。所以 2D X-Y 轴上的任何一条线都只是平面方程被设置为 0 的特例，也就是说，它是平面与平坦的 X-Y 平面相交的所有点的集合。

![](img/d01614129e08a7fb3ebe2e2320e96296.png)

Neuron intermediate output without any activation

# 激活情况如何？

到目前为止，我故意忽略的部分是神经元的激活，所以我们也把它放进去。我用过一个 ReLU，就是把所有的负值都换成零。所以，g(z) = max{0，z}

> 下面的粉红色部分是神经元的最终输出。如果床单平放在地板上，水果就是橙子。如果床单悬在空中，那就是苹果。此外，越往上，它是苹果的概率就越高。

![](img/15c6d93193bfd5f5bb74a7ad917ae51d.png)

Final neuron activation in 3D with a ReLU activation

![](img/478a5c9d6162f86272828dc453b60ffd.png)

Final neuron activation in 3D with a ReLU activation — different perspective

# 多神经元的这种输出弯曲看起来如何？

我正致力于本系列的下一页，将可视化带到多个神经元——一旦发表，我将添加一个链接。如果你认为我们应该选择一些其他的激活函数(比如 sigmoid)来增加趣味，请给我留言。谢谢你。