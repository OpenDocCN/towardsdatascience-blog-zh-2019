# 可视化深度神经网络的损失情况…..但是我们能相信他们吗？

> 原文：<https://towardsdatascience.com/visualizing-loss-landscape-of-deep-neural-networks-but-can-we-trust-them-3d3ae0cff46e?source=collection_archive---------11----------------------->

## ***我们能相信深度神经网络的损失景观可视化吗？***

![](img/8edbcdce9c49678516fb6be13f0c30cb.png)

Landscape from this [website](https://pixabay.com/illustrations/evening-sun-sunset-backlighting-55067/)

**简介**

最近[开发了一种方法](https://arxiv.org/abs/1712.09913)来可视化深度神经网络的损失情况。我个人认为这是一个巨大的突破，然而，我对创建的可视化的有效性感到有点怀疑。今天，我将研究作者的可视化方法，并介绍一些我认为非常酷的其他方法。

**方法**

创建亏损景观的整个过程非常简单直接。

1.  训练网络
2.  创建随机方向
3.  给固定重量加上不同的扰动量，看看损失值是如何变化的。

唯一需要注意的是这些随机方向是如何产生的。我们来看看作者的方法。

![](img/0b309f9cfeeac8c7b6c335aa84ec801c.png)

他们的方法被称为“过滤器标准化”,非常容易理解。(这里是[链接](https://github.com/tomgoldstein/loss-landscape)到作者的代码)。基本上，对于四维张量如(64，3，3，3)，我们将匹配关于第一维的范数，因此(64，1，1，1)，在权重的范数和随机方向的范数之间。(用一个更简单的术语来说，我们可以将这理解为匹配权重和随机方向之间的比例)。

![](img/d05273a81471b4d14834d3ee6fb1bfff.png)

以上是运行作者代码时的部分结果。现在我们可以利用张量运算来简化整个过程。(我稍后会展示)

**网络**

![](img/3ad6b173f68701bac9a349d700b8d0a5.png)

**绿球** →输入图像(64，64，3)
**蓝色矩形** →卷积+ ReLU 激活
**红色矩形** →软最大输出

对于这篇文章，我在 CIFAR 10 数据集上训练了三个九层完全卷积神经网络(如上所示)。无任何归一化、批量归一化和[局部响应归一化。](https://prateekvjoshi.com/2016/04/05/what-is-local-response-normalization-in-convolutional-neural-networks/)

![](img/db9c4b6fa2287989afe1768d79902738.png)

并且从上面的图中我们可以看到，批量归一化的网络取得了最高的性能。

从现在起我将把每个网络称为如下的
**正常**:没有任何归一化层的网络
**批量规范**:具有批量归一化层的网络
**局部规范**:具有局部响应归一化层的网络

**滤波归一化**

上面的代码片段展示了如何使用张量运算进行过滤器标准化。

![](img/65a9bacc1dada369965f2a894b9e8457.png)![](img/f6dcbe18cfed6c1e729cf7aebdc32d5a.png)![](img/c8a56d1b4e8a9fcff3f43d0b4e827db1.png)

Normal, Batch Norm, Local Norm

![](img/07ed1ac7d222277b92da801ae26e8ce3.png)![](img/c87441dc38c3af48405689899beac751.png)![](img/d91c4c1ea5588e83be3210167061cf8f.png)

Normal, Batch Norm, Local Norm — Log Scale

当我们使用滤波归一化方法来可视化损失景观时，我们可以看到每个景观看起来并没有太大的不同。只有在我们以对数比例显示景观的情况下，我们才能看到，事实上，局部响应归一化的景观要清晰得多。

![](img/c1ae3244bc6949ce8a22cff031609571.png)![](img/252e130e2346346f6aae216b9522762d.png)

当我们将这三幅图按原始比例叠加在一起时，我们可以看到它们看起来是多么相似。

**滤波器正交化**

上述方法只是作者方法的简单修改，我们从简单的高斯分布生成随机方向，但是通过 [QR 分解](https://www.tensorflow.org/api_docs/python/tf/linalg/qr)我们使方向正交化。

![](img/7be765b078eebaf7aade07a87858b5b9.png)![](img/0b963b2c8fe5cbca3fb30eb51d00a00b.png)![](img/00e047266930b9061a5a7ea1db948305.png)

Normal, Batch Norm, Local Norm

![](img/57a033b1f02cf300f96037a3e11e988a.png)![](img/4bb1e6b25c2601622647ae6f1ab4068c.png)![](img/c2e9170220096ef4b8b97acf81c92d6d.png)

Normal, Batch Norm, Local Norm — Log Scale

当我们将不同维度的方向正交化时，我们可以立即看到创建的损失景观是如何彼此不同的。与作者的方法相比，我们可以看到三个网络之间的损耗情况有所不同。

![](img/289b01893f59610c12d2460bba23b19e.png)![](img/ee4b4fd733e875df4ea877d0130d684f.png)

**正交权重投影**

这与滤波器正交化基本相同，唯一的区别在于，对不同维度的收敛权重执行 ZCA 白化，而不是从高斯分布生成。

![](img/d63d1725e4fe4af000bfc87b62e37b74.png)![](img/a5ffd45e9b9d08d04fa9e103db206f9d.png)![](img/90d6d2954394e5ad935e8d0c1f1e99a3.png)

Normal, Batch Norm, Local Norm

![](img/435a64cf7eae5954dc7b25b7f71e07b1.png)![](img/8b0a78407fc4c2fa52edafe19a90a930.png)![](img/9d3abebf8c8140428729e6e545b0b185.png)

Normal, Batch Norm, Local Norm — Log Scale

类似于滤波器正交化，我们可以看到生成的可视化之间的一些差异。

![](img/b82317236484b6882d34a4aff7a3c5d4.png)![](img/63c0da285d40ab9b79bcce5df5ab591e.png)

**重量的主要方向**

最后一种方法是在不同的维度之间，在它们的第一主方向上扰动权重。

![](img/9e2a4a27f0dd997a0d347f63839307e2.png)![](img/aab3d57d59799d4df1f3103f29b9aee9.png)![](img/e570ec55b50669ea18cbd809a0c75f89.png)

Normal, Batch Norm, Local Norm

![](img/a3ef906e368dcf89bdfb8c8fc70db84e.png)![](img/cb8482f24d34600d9f2b4cd0dda9bbe0.png)![](img/6e17ee42251b99f47e209e637574760c.png)

Normal, Batch Norm, Local Norm — Log Scale

我们可以清楚地看到所产生的损失情况之间的差异。

![](img/d82ca61e664d0243098c1405cf34ebe2.png)![](img/0d3ee67b9fc437e8e57186a898f3d6ad.png)

**讨论**

我写这篇文章的唯一原因是为了表明，根据我们使用的方向，创造的损失景观可以发生巨大的变化。因此，我们需要质疑生成的损失景观的有效性，它们是否真正反映了训练网络的特征。

**代码**

![](img/1a246fa37decad2506bf48a7f31667c2.png)

要访问创建可视化效果的代码，请[点击此处。](https://github.com/JaeDukSeo/Daily-Neural-Network-Practice-3/blob/master/Loss%20LanScape/0%20create%20viz.ipynb)
要查看整篇博文的代码，请[点击此处](https://github.com/JaeDukSeo/Daily-Neural-Network-Practice-3/tree/master/Loss%20LanScape)。

**遗言**

我不想做任何大胆的断言，但似乎不同的方向选择会产生不同的视觉效果。问题仍然存在，哪个方向是最‘正确’的？可有正确的，哪一个揭示了真相？此外，我想提一下名为“[尖锐极小值可以推广到深度网络](https://arxiv.org/abs/1703.04933)的论文，该论文表明已经收敛到尖锐极小值的深度神经网络可以很好地推广，并且该理论不适用于具有 ReLU 激活的网络。就像那篇论文如何证明我们的观察可以根据我们的定义而改变一样，我们应该致力于创造反映真理的定义。

还有更多研究要做，我很期待。如果你希望看到更多这样的帖子，请访问我的[网站](https://jaedukseo.me/)。

**参考**

1.  李，h，徐，z，泰勒，g，斯图德，c，&戈尔茨坦，T. (2017)。可视化神经网络的损失景观。arXiv.org。2019 年 5 月 3 日检索，来自[https://arxiv.org/abs/1712.09913](https://arxiv.org/abs/1712.09913)
2.  tomgoldstein/loss-landscape。(2019).GitHub。检索于 2019 年 5 月 3 日，来自[https://github.com/tomgoldstein/loss-landscape](https://github.com/tomgoldstein/loss-landscape)
3.  [https://prateekvjoshi . com/2016/04/05/what-is-local-response-normalization-in-convolutionary-neural-networks/](https://prateekvjoshi.com/2016/04/05/what-is-local-response-normalization-in-convolutional-neural-networks/)