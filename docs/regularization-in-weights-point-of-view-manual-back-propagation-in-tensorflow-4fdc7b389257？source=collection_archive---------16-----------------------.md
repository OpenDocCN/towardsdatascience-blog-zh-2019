# 权重观点中的正则化[张量流中的手动反向传播]

> 原文：<https://towardsdatascience.com/regularization-in-weights-point-of-view-manual-back-propagation-in-tensorflow-4fdc7b389257?source=collection_archive---------16----------------------->

## 不同的正则化如何影响最终的权重值？

![](img/07a61c90eba1cdbe867ec7a3843d8693.png)

GIF from this [website](https://giphy.com/gifs/Reishunger-gnnung-2ZZL7srhu4QtcFNF3L)

我要感谢我的导师布鲁斯博士鼓励我进一步研究这个想法。此外，我想感谢我的实验室成员，在[瑞尔森视觉实验室](https://ryersonvisionlab.github.io/)、[贾森](https://github.com/SenJia)关于正规化的有益讨论。

更多类似的文章，请访问我的网站， [Jaedukseo.me](https://jaedukseo.me/) 。

**简介**

为了防止我们的模型过度拟合，我们可以使用不同种类的正则化技术。其中一种技术是在我们想要优化的损失函数中加入一个正则项。因此，当我们对模型中的权重求导时，我们会对权重添加额外的梯度。

但是不同的正则项如何影响权重呢？更具体地说，它会如何影响价值观的传播？意味着直方图的形状。

**实验设置**

![](img/bbcb078bd05ba19f5b2dad815b9d680e.png)

**蓝色球体** →输入图像数据( [STL10](https://cs.stanford.edu/~acoates/stl10/) 或 [CIFAR10](https://www.cs.toronto.edu/~kriz/cifar.html) )
**黄色矩形** →带 ReLU 激活的卷积层
**红色方块** →交叉熵损失项用于分类

网络本身只由六层组成，并不是一个大网络。然而，我们将使用两个不同的数据集，CIFAR 10 和 STL 10。

此外，我们将在有和没有批量标准化的情况下训练网络 10 集，这是为了观察相同的现象是否会反复发生。(现象是，权重收敛到直方图的特定形状。).最后，下面是我们要比较的所有正则项的列表。

***Z:基线(无正则化)
A:ABS(Theta)
B:Theta
C:sqrt(Theta)
D:—log(1+Theta)
E:—tanh(Theta)
F:—tanh(Theta)
G:—tanh(ABS(Theta))
H:—tanh(ABS(Theta)
I:sin(Theta)
J:ABS(sin***

**结果位移**

![](img/44bf5186b86b6a0d966aac706d886fdc.png)

最左边的 gif 显示了 10 集的重量直方图和粉红色的平均直方图。

![](img/c7e3afacda88ddf89fab1188fb465696.png)

如上所述，从红色开始，它代表使用 Adam optimizer 150 个周期后的最终重量。粉色直方图代表平均重量值的直方图。现在放大可视化中最右边的图像。

![](img/655b08ef4207d99ef2ad52b20edc6b36.png)

每个方框显示了每一层的最终直方图，最终图显示了最大/最小/平均训练(红色)/测试(蓝色)性能。

![](img/af9ba73d1f057bd9fc8023cb898d2f9f.png)

**结果**

> **Z:基线(无调整)**

![](img/be1479f7033866b7e3d3541e361a5b0b.png)![](img/502d014f37321751cb20c7512b716c6e.png)![](img/55a8a40fed56c4dc346a3e92258eee7e.png)

without Batch Normalization (STL)

![](img/24e81375b4021dd0f098ddda3560eb0a.png)![](img/13e56fd06df429a1409475794d0c7e6c.png)![](img/42cd9b8fa684c3cfc5763ca2baa11062.png)

With Batch Normalization (STL)

![](img/0ef3349c04206f8a01525432cc9f2e18.png)![](img/44df714656a77c616b23b9118f928526.png)![](img/bb01c0cdb4b6624bf39cc6016f6968d4.png)

without Batch Normalization (CIFAR)

![](img/3c380dc21cbfe81a3e8f87183a677e74.png)![](img/1d85d5545b5b67f0df2b7a1362d08a90.png)![](img/49094c8bd1710dac943ff4bc432cd560.png)

with Batch Normalization (CIFAR)

> **答:ABS(θ)**

![](img/a238de3992d4455e217caa0e38e2d007.png)![](img/e3b83fd4104b9917aaaf5269a7572bef.png)![](img/9827f702669b608099a75a69fc3f48d4.png)

without Batch Normalization (STL)

![](img/15dc09ae08f6378e4226c399b223fd5b.png)![](img/f65519da9fc49a70c210d0b84b7d476a.png)![](img/f8efb66eb25b8a713147e46b349e85ef.png)

with Batch Normalization (STL)

![](img/6ee3e9b40b9cb2b824ae1f90846d8125.png)![](img/6ee3e9b40b9cb2b824ae1f90846d8125.png)![](img/a1bdbab6642240470bcacdf27b3cf1b7.png)

without Batch Normalization (CIFAR)

![](img/01f3b9a3fe5661325683af311d6e3ec6.png)![](img/a11c052bc24f216b1f6bac4e17cfed14.png)![](img/b1e06ce30d7c1a9a27ef25c9bf25e89f.png)

with Batch Normalization (CIFAR)

> **B:θ**

![](img/89ea9d187fdfdceeebcdbbb92bf5e8e2.png)![](img/2480d5c0497700ee4b1fe6bcf6a17d64.png)![](img/cf288d9b4016cc85318ec03dccaa9e96.png)

without Batch Normalization (STL)

![](img/8e86acaa05a193489f04bebffe3c0fd0.png)![](img/821e6ba8a2a768065af5c820d8e0a70b.png)![](img/48a3958f6fd952186979bbe3bd9cbf3e.png)

with Batch Normalization (STL)

![](img/e71847088ce90a15ec3b5886f5294dbe.png)![](img/5f750520712aca0c6db268de2aca0807.png)![](img/6160f0d4ee4b2ec491943517e96a6f5c.png)

without Batch Normalization (CIFAR)

![](img/216ef8892df9b39ac0d0ecb6431db48a.png)![](img/834e8b75b79a6c1d23134d9487b1ce2b.png)![](img/f05394897c1e1cee790c0ca6d92eb13d.png)

with Batch Normalization (CIFAR)

> **C: sqrt(Theta )**

![](img/9620c400be7181675f7a8c2ea9c68296.png)![](img/ae94fc6755b7183b72db743058c18a08.png)![](img/dc98263e5b09f4affb1630216e4804cf.png)

without Batch Normalization (STL)

![](img/f1ea70ccc027e02d6a637936ae721ec2.png)![](img/3f0eb6874a28e5b962f0914c4d67b7ca.png)![](img/5f7fae78ced02026d559e99c581fb81c.png)

with Batch Normalization (STL)

![](img/6ee3e9b40b9cb2b824ae1f90846d8125.png)![](img/6ee3e9b40b9cb2b824ae1f90846d8125.png)![](img/a24c0951de5ea8b28123537312d08b7e.png)

without Batch Normalization (CIFAR)

![](img/5668988487e0c9b69066690e3daf671a.png)![](img/45c0c1c2b834b4f9e61ad2940267d6ea.png)![](img/f7bf4425e0fdb128baa9a4d7a6ca9431.png)

with Batch Normalization (CIFAR)

> **D:——log(1+Theta)**

![](img/883d135ec43b418420eb80a1ab1a507f.png)![](img/ccc17449a84fa226978475ae95489d58.png)![](img/00cb11a5e77cf25ee5ed857eed5bd45f.png)

without Batch Normalization (STL)

![](img/0bfda7cd66a94fd4dbf62cbe9d351c70.png)![](img/ceed59779e5c6573323c5b56d00db272.png)![](img/87f9bb01ac089c490c95e35cffe91d25.png)

with Batch Normalization (STL)

![](img/99ea2c388906d585863cf9fd3291ba5f.png)![](img/7acb68a456b30899dc3cfe9dd3e16087.png)![](img/b3da09520e8f61e61f1531b07c68ff8b.png)

without Batch Normalization (CIFAR)

![](img/53dee41f8e586617865e1193d83f5010.png)![](img/b6555761f8814fd5a7bd4271d6981430.png)![](img/c0666bc8bbc4cd82bcc479ce7c7a9b59.png)

with Batch Normalization (CIFAR)

> **E:——tanh(Theta)**

![](img/37661e377f09d383b8a7e061105adacf.png)![](img/137e5c3acbe457835bd3d942e379eab2.png)![](img/5f24ec13e9fb3077ff80f6997234e301.png)

without Batch Normalization (STL)

![](img/3a861dbfc4e4286e465f12664cfbf8c6.png)![](img/608e01d8d67dcfd3b692fda1a322ffc7.png)![](img/5e098ed26fa3e1ae43fe40e3d9e5a633.png)

with Batch Normalization (STL)

![](img/900b5bad13c1908470628fcf0b6f3dcb.png)![](img/9b5f5e4f021ba6a0217dc53475067191.png)![](img/019e5f1fcbc0d27112fae48ae789eb97.png)

without Batch Normalization (CIFAR)

![](img/aaa40bcc146085eaa081beb46538f7a1.png)![](img/d02a37d89dcccef8b2f20a11ce62f09a.png)![](img/c455a1b1218c0968bcb85482562af739.png)

with Batch Normalization (CIFAR)

> **F:--tanh(Theta)**

![](img/4a3f65e5d208a2448676cbb086ac057d.png)![](img/2ebdf788bf65b852c60b54fe2b24e75f.png)![](img/b00a08303540ce7017f7aae111be7206.png)

without Batch Normalization (STL)

![](img/44d5bfdf18747b84f4be8472c5120579.png)![](img/7a9b28d437e1b9b311a6d484103e75b4.png)![](img/e025bfa669863409019a4368d27f0722.png)

with Batch Normalization (STL)

![](img/b7e100993f5d7bbb7a00294b961bddb4.png)![](img/92f6cdc1dafa2c03b72414be1716141e.png)![](img/7746d0fff90202e8bc84077d8464a1db.png)

without Batch Normalization (CIFAR)

![](img/98caefe1db6f1c8a06cf6e34edde7d8d.png)![](img/4e77ca9d77146d3c29b08e8efd7fa566.png)![](img/7e4973fa96a92c305543feacc08d3bfb.png)

with Batch Normalization (CIFAR)

> **G:——tanh(ABS(Theta))**

![](img/1be3e8022ef45ed5154704611250ee13.png)![](img/d303be956b4d6158b037f7ae973ad7dc.png)![](img/2c94e3ee5fc27ca64615a22e6483fb02.png)

without Batch Normalization (STL)

![](img/a2aa0c89459bcc29ff8c4b6ede0fd69d.png)![](img/05da6601c87669de0f0c4a84a00d6648.png)![](img/2b1a89bd8620bf762e562d5c316f2efe.png)

with Batch Normalization (STL)

![](img/5bf7045422e13d20a4d9e610fb44f64b.png)![](img/9bf2002939c71cee3c0ea79935905556.png)![](img/696c72a1f72c3d03a71281a7bdf63fb4.png)

without Batch Normalization (CIFAR)

![](img/6b064cf7116e49772fbf354c1bb4d8d8.png)![](img/2208819bdf1a6a2b92bbff78a24ad8e3.png)![](img/e02625ddcc09c9372391f61244c087c1.png)

with Batch Normalization (CIFAR)

> **H: -tanh(abs(Theta) )**

![](img/2c8d89e12822eb2856d0b34176467aef.png)![](img/b2c2956a726a2c3dab63e492ce480d02.png)![](img/2203b1d772674f49704a1330662e268c.png)

without Batch Normalization (STL)

![](img/a8b363fe21ca1033c4e419092c655ca5.png)![](img/a0badc94834fea1ff55c55c50104d059.png)![](img/90a508b98f7047394024c40f9f142f0b.png)

with Batch Normalization (STL)

![](img/c3188f6b92ad2000aa06615173b9d6b2.png)![](img/1799bec3914f50cbc33c7ae1db22b768.png)![](img/00facf1b9810c8c1d6fbfe8d58c37b6f.png)

without Batch Normalization (CIFAR)

![](img/0346316742e67fa7cf5213762549e39f.png)![](img/6af7c85a7f6ed245913de4fdbb3266d8.png)![](img/ec053847e889284ee5d21242531327c0.png)

with Batch Normalization (CIFAR)

> **I:sin(θ)**

![](img/6ee3e9b40b9cb2b824ae1f90846d8125.png)![](img/6ee3e9b40b9cb2b824ae1f90846d8125.png)![](img/fcc7125560af4db48e6ab97dd439fd1f.png)

without Batch Normalization (STL)

![](img/c51b7f8adbe58e87ed4f223a20577150.png)![](img/3a7c2d9389c649ac16f8c22c7757d727.png)![](img/c502b23eda2ecd3dee0a2f30a6277366.png)

with Batch Normalization (STL)

![](img/6ee3e9b40b9cb2b824ae1f90846d8125.png)![](img/6ee3e9b40b9cb2b824ae1f90846d8125.png)![](img/b264ec7b6c29e7cd69431dcd8ac858e4.png)

without Batch Normalization (CIFAR)

![](img/54091bd36601f579d89752cafd0fbfde.png)![](img/d81ae23c18b2aa62811941d2533df26f.png)![](img/ba367e16ec97d02eea39a31aff7d12bf.png)

with Batch Normalization (CIFAR)

> **J: abs(sin(Theta))**

![](img/aa73c2247e99c782333bba9206058035.png)![](img/ffcf29b2b38ade87567eb97a0d928534.png)![](img/097da19a5c446875df8dfd357a0ce89e.png)

without Batch Normalization (STL)

![](img/5e0f4ac76a0641bc65198f59eb7ac123.png)![](img/52ebb1d267d997b9a0c4e352422ee3c1.png)![](img/f195d05b53c2e0c3896268de24beab2a.png)

with Batch Normalization (STL)

![](img/6ee3e9b40b9cb2b824ae1f90846d8125.png)![](img/6ee3e9b40b9cb2b824ae1f90846d8125.png)![](img/e889ba603f4ca05f4283cfc584948a49.png)

without Batch Normalization (CIFAR)

![](img/89be2ebfe60da8bc35d570f5a8917223.png)![](img/cf3883cb2360ef9a6951da3dd8da27d0.png)![](img/e322963602f510d6a134ed2d67269bd8.png)

with Batch Normalization (CIFAR)

> **K:对数(θ)**

![](img/f1e067a5c65aa5d2c067aed89d83ca0d.png)![](img/9b768784bbb19e48fef2f909e464fdf6.png)![](img/9aa6cac6365ed050869c41d29a7fc505.png)

without Batch Normalization (STL)

![](img/a06e5b10d6ca134e197c4de62d85a71d.png)![](img/2bb48a65ccf54949ed9a512765d24a7d.png)![](img/5b20bdce93d1b0ee444d8bb0c0f29f6a.png)

with Batch Normalization (STL)

![](img/6ee3e9b40b9cb2b824ae1f90846d8125.png)![](img/6ee3e9b40b9cb2b824ae1f90846d8125.png)![](img/e889ba603f4ca05f4283cfc584948a49.png)

without Batch Normalization (CIFAR)

![](img/89be2ebfe60da8bc35d570f5a8917223.png)![](img/cf3883cb2360ef9a6951da3dd8da27d0.png)![](img/e322963602f510d6a134ed2d67269bd8.png)

with Batch Normalization (CIFAR)

> **L: Theta * log(Theta )**

![](img/81d0c4f94cce935d5d4e09d4671793a8.png)![](img/67e5df7ab333eab2efeab0df4bbcc9e4.png)![](img/90ee33a7f5888e8e4efe0806d01e4862.png)

without Batch Normalization (STL)

![](img/f4bebbe59b361964d830730b59f9b752.png)![](img/d7d570657de03e22253e1e93bebfac11.png)![](img/84c4beeb33c3512ee4cb56e98399f452.png)

with Batch Normalization (STL)

![](img/ba3c194ef00aeb11c4ea5ea1392283ff.png)![](img/c334393fa8f53c5b116194f590e34e04.png)![](img/f38ac907db6dfcb20104d679de32e963.png)

without Batch Normalization (CIFAR)

![](img/6ee3e9b40b9cb2b824ae1f90846d8125.png)![](img/774f530fdbd24ae1ff9f81c4fc37a095.png)![](img/3a726bb8c0f9fedb6085928425dab403.png)

with Batch Normalization (CIFAR)

**讨论/代码**

虽然没有任何明确的证据或定理，我可以从这个实验中证明。我能够做出一些经验性的观察。

1.  批量标准化(没有任何 alpha 或 beta 参数)很可能导致模型过度拟合。
2.  增加批次归一化，使训练更加稳定。此外，它还使权重收敛于类高斯分布。(我们怀疑这可能和中心极限定理有关。)
3.  小批量导致更严重的过度拟合。
4.  在权重的稀疏性和过度拟合程度之间似乎存在微弱的负相关。(如果权重值稀疏，意味着权重中有许多零值，过度拟合程度越低)。

事后看来，最后的观察非常有意义，如果网络权重本身包含大量零值，并且有一些约束将权重保持在那些区域中，那么网络自然很难过度拟合。

我想提出的一些有趣的联系是关于我们的大脑是如何编码视觉输入的。从多个实验中，我们知道 V1 细胞创建了给定视觉输入的稀疏表示，这与具有稀疏权重集非常不同，然而，在稀疏的名义下，它似乎有一些联系。也许，仅仅是也许，对于任何有效的学习模型来说，可能需要一个好的稀疏的权重集。

![](img/da810b99f7b00cb07af9bf90e22e7f81.png)

要访问 CIFAR 10 的代码请[点击这里](https://github.com/JaeDukSeo/Daily-Neural-Network-Practice-2/blob/master/Class%20Stuff/regularization%20in%20deeper%20gradient%20point%20of%20view/blog/a%20FINAL-CIFAR.ipynb)，要访问 STL 10 的代码请[点击这里](https://github.com/JaeDukSeo/Daily-Neural-Network-Practice-2/blob/master/Class%20Stuff/regularization%20in%20deeper%20gradient%20point%20of%20view/blog/a%20FINAL.ipynb)。

![](img/1b2e735f66f05a577fc71630fd149811.png)

最后，如果您希望访问 CIFAR 10 和 STL 10 的每一层的完整结果，无论是否进行批规格化。

请点击此处访问 CIFAR 10 的[完整结果。
请点击此处访问 STL 10](https://medium.com/@SeoJaeDuk/archived-post-cifar-10-full-results-for-regularization-in-weights-point-of-view-manual-back-5df9405c8271) 的[完整结果。](https://medium.com/@SeoJaeDuk/archived-post-stl-10-full-results-for-regularization-in-weights-point-of-view-manual-back-542329164b5e)

**超级有用的数学推理，关于你完全(不应该)信任的每个正则化子在做什么**

在这一节中，我想简单介绍一下每个正则项对权重的影响。但是，请注意，这不是一个数学定义。直线代表调节项的值，而虚线代表梯度大小。

![](img/420e6b9fe5fd2bb3ee5760df9e34e26b.png)![](img/b80f62ad642f32fdb9547736d28b9c46.png)![](img/a435b784a3635756f735f33f8c1441e3.png)

上面显示了正则化项 abs(x)，x，和 sqrt(x)的样子以及它的导数。需要注意的一点是，当导数的幅度结束时，特别是对于 abs(x)，我们注意到，当 x 位于 0 时，梯度的幅度停止。

因此，该模型将不得不平衡它应该具有多少稀疏权重和它将从没有稀疏权重得到多少惩罚。而具有 x 正则项的网络可以承受多个小数量级的权重。

![](img/70a55aa23341782f3139f8ada3cf59ed.png)![](img/d6f47f45a3d62cf905c8188f95011c11.png)![](img/07de60cc6f5dc93df7c1aa7917d8dce3.png)

对于涉及 log 的正则化项，我们注意到，由于梯度增加接近无穷大，训练会极不稳定。(对于正无穷大和负无穷大)

对于带有双曲正切的正则化项，我们可以看到只有某些区域的梯度值大于零。

![](img/e8d9fede962cc8d03299851adf2b035f.png)![](img/0ed34924629774584c8a7c17e08dc5d6.png)

最后，对于 sin 类型的正则项，我们可以看到，网络试图收敛到某些数字，目标并不真正涉及稀疏性。

**最后的话**

在 ICA 中，目标之一是对给定数据进行稀疏表示。虽然使网络的权重稀疏，可以达到类似的结果，我想知道他们的性能会有什么不同。

具有稀疏性约束可以在增加神经网络的泛化能力方面发挥巨大作用。然而，不同风格的稀疏约束可能是一个更重要的问题来解决。

更多类似的文章，请访问我的网站， [Jaedukseo.me](https://jaedukseo.me/) 。

**参考**

1.  笔记本，o .，&达利，S. (2017)。覆盖 jupyter 笔记本中以前的输出。堆栈溢出。检索于 2019 年 1 月 15 日，来自[https://stack overflow . com/questions/38540395/overwrite-previous-output-in-jupyter-notebook](https://stackoverflow.com/questions/38540395/overwrite-previous-output-in-jupyter-notebook)
2.  n .布鲁斯(2016)。尼尔·布鲁斯。尼尔·布鲁斯。检索于 2019 年 1 月 15 日，来自[http://www.scs.ryerson.ca/~bruce/](http://www.scs.ryerson.ca/~bruce/)
3.  在和 Alpha 之间填充— Matplotlib 3.0.2 文档。(2019).Matplotlib.org。检索于 2019 年 1 月 15 日，来自[https://matplotlib . org/gallery/recipes/fill _ between _ alpha . html](https://matplotlib.org/gallery/recipes/fill_between_alpha.html)
4.  Matplotlib . py plot . plot-Matplotlib 3 . 0 . 2 文档。(2019).Matplotlib.org。检索于 2019 年 1 月 15 日，来自[https://matplotlib . org/API/_ as _ gen/matplotlib . py plot . plot . html](https://matplotlib.org/api/_as_gen/matplotlib.pyplot.plot.html)
5.  plots，h .，Raviv，o .，Castro，s .，Isaac，é。，& Twerdochlib，N. (2010 年)。在 matplotlib 图中隐藏轴文本。堆栈溢出。检索于 2019 年 1 月 17 日，来自[https://stack overflow . com/questions/2176424/hiding-axis-text-in-matplotlib-plots](https://stackoverflow.com/questions/2176424/hiding-axis-text-in-matplotlib-plots)
6.  NumPy . histogram—NumPy 1.15 版手册。(2019).Docs.scipy.org。检索于 2019 年 1 月 17 日，来自[https://docs . scipy . org/doc/numpy-1 . 15 . 0/reference/generated/numpy . histogram . html](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.histogram.html)
7.  Matplotlib . axes . axes . hist 2d-Matplotlib 3 . 0 . 2 文档。(2019).Matplotlib.org。检索于 2019 年 1 月 17 日，来自[https://matplotlib . org/API/_ as _ gen/matplotlib . axes . axes . hist 2d . html](https://matplotlib.org/api/_as_gen/matplotlib.axes.Axes.hist2d.html)
8.  matplotlib——条形图、散点图和直方图——生物学家的实用计算。(2019).People.duke.edu。检索于 2019 年 1 月 17 日，来自[http://people . duke . edu/~ CCC 14/pcfb/numpympl/matplotlibbarplots . html](http://people.duke.edu/~ccc14/pcfb/numpympl/MatplotlibBarPlots.html)
9.  mplot3d 示例代码:bars3d_demo.py — Matplotlib 1.3.1 文档。(2019).Matplotlib.org。检索于 2019 年 1 月 17 日，来自[https://matplotlib . org/1 . 3 . 1/examples/mplot3d/bars 3d _ demo . html](https://matplotlib.org/1.3.1/examples/mplot3d/bars3d_demo.html)
10.  mplot3d 示例代码:poly 3d _ demo . py—Matplotlib 1 . 3 . 0 文档。(2019).Matplotlib.org。检索于 2019 年 1 月 17 日，来自[https://matplotlib . org/1 . 3 . 0/examples/mplot3d/polys3d _ demo . html](https://matplotlib.org/1.3.0/examples/mplot3d/polys3d_demo.html)
11.  直方图？，P. (2017)。绘制一系列直方图？。Mathematica 堆栈交换。检索于 2019 年 1 月 17 日，来自[https://Mathematica . stack exchange . com/questions/158540/plot-a-sequence-of-histograms](https://mathematica.stackexchange.com/questions/158540/plot-a-sequence-of-histograms)
12.  图形？，W. (2013)。“np.histogram”和“plt.hist”有什么区别？为什么这些命令不绘制相同的图形？。堆栈溢出。检索于 2019 年 1 月 17 日，来自[https://stack overflow . com/questions/20531176/what-is-the-difference-between-NP-histogram-and-PLT-hist-why-dont-these-co](https://stackoverflow.com/questions/20531176/what-is-the-difference-between-np-histogram-and-plt-hist-why-dont-these-co)
13.  NumPy . Lin space—NumPy 1.15 版手册。(2019).Docs.scipy.org。检索于 2019 年 1 月 17 日，来自[https://docs . scipy . org/doc/numpy-1 . 15 . 0/reference/generated/numpy . linspace . html](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.linspace.html)
14.  结果，第(2017)页。PLT . hist()vs NP . histogram()-意外结果。堆栈溢出。检索于 2019 年 1 月 17 日，来自[https://stack overflow . com/questions/46656010/PLT-hist-vs-NP-histogram-unexpected-results](https://stackoverflow.com/questions/46656010/plt-hist-vs-np-histogram-unexpected-results)
15.  plot，c .，& Navarro，P. (2012 年)。更改 matplotlib 3D 绘图的轴平面的背景颜色。堆栈溢出。检索于 2019 年 1 月 17 日，来自[https://stack overflow . com/questions/11448972/changing-the-background-color-of-a-matplotlib-3d-plot-planes/12623360](https://stackoverflow.com/questions/11448972/changing-the-background-color-of-the-axes-planes-of-a-matplotlib-3d-plot/12623360)
16.  NumPy . arange—NumPy 1.11 版手册。(2019).Docs.scipy.org。检索于 2019 年 1 月 17 日，来自[https://docs . scipy . org/doc/numpy-1 . 11 . 0/reference/generated/numpy . arange . html](https://docs.scipy.org/doc/numpy-1.11.0/reference/generated/numpy.arange.html)
17.  LLC，I. (2019)。命令行工具:Convert @ ImageMagickImagemagick.org。检索于 2019 年 1 月 17 日，来自 https://imagemagick.org/script/convert.php
18.  使用 Python 的 Matplotlib 制作 3D 绘图动画。(2012).糖分过高。检索于 2019 年 1 月 17 日，来自[https://zulko . WordPress . com/2012/09/29/animate-your-3d-plots-with-python-matplotlib/](https://zulko.wordpress.com/2012/09/29/animate-your-3d-plots-with-pythons-matplotlib/)
19.  鼠，H. (2017)。如何在 python 中旋转 3d 绘图？(或作为动画)使用鼠标旋转三维视图。堆栈溢出。检索于 2019 年 1 月 17 日，来自[https://stack overflow . com/questions/43180357/how-to-rotate-a-3d-plot-in-python-or-as-a-a-animation-rotate-3d-view-using-mou？noredirect=1 & lq=1](https://stackoverflow.com/questions/43180357/how-to-rotate-a-3d-plot-in-python-or-as-a-animation-rotate-3-d-view-using-mou?noredirect=1&lq=1)
20.  贾，S. (2019)。贾森——概述。GitHub。检索于 2019 年 1 月 28 日，来自[https://github.com/SenJia](https://github.com/SenJia)
21.  瑞尔森视觉实验室。(2018).Ryersonvisionlab.github.io 于 2019 年 1 月 28 日检索，来自[https://ryersonvisionlab.github.io/](https://ryersonvisionlab.github.io/)
22.  STL-10 数据集。(2019).Cs.stanford.edu。检索于 2019 年 1 月 28 日，来自[https://cs.stanford.edu/~acoates/stl10/](https://cs.stanford.edu/~acoates/stl10/)
23.  [https://www.cs.toronto.edu/~kriz/cifar.html](https://www.cs.toronto.edu/~kriz/cifar.html)
24.  德斯莫斯图表。(2019).德斯莫斯图形计算器。检索于 2019 年 1 月 29 日，来自[https://www.desmos.com/calculator](https://www.desmos.com/calculator)