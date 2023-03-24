# 学习模拟

> 原文：<https://towardsdatascience.com/learning-to-simulate-c53d8b393a56?source=collection_archive---------9----------------------->

## 学习如何模拟更好的合成数据可以改善深度学习

*在* ***ICLR 2019*** *发表的论文可以在* [*这里*](https://arxiv.org/abs/1810.02513) *。我还有一张* [*幻灯片*](https://docs.google.com/presentation/d/1AEqPi-bJ1q-o1bxosBjcqEgCb2Vs2TcJF6cI4nHZKTg/edit?usp=sharing) *以及一张* [*海报*](https://natanielruiz.github.io/docs/lts_poster.pdf) *详细解释了这项工作。*

![](img/8392cc2a8b133deba76f2f354ae009a1.png)

Photo by [David Clode](https://unsplash.com/@davidclode?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/brain?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

深度神经网络是一项令人惊叹的技术。有了足够多的标记数据，他们可以学习为图像和声音等高维输入产生非常准确的分类器。近年来，机器学习社区已经能够成功解决诸如分类对象、检测图像中的对象和分割图像等问题。

上述声明中的警告是**带有足够的标记数据。对真实现象和真实世界的模拟有时会有所帮助。在[计算机视觉](https://www.di.ens.fr/willow/research/surreal/data/)或[机器人控制应用](https://openai.com/blog/generalizing-from-simulation/)中，有合成数据提高深度学习系统性能的案例。**

模拟可以给我们免费标签的精准场景。但是我们就拿侠盗猎车手 V (GTA)来说吧。[研究人员利用了通过自由漫游 GTA V 世界收集的数据集](https://arxiv.org/abs/1608.02192)，并一直使用该数据集来引导深度学习系统等。许多游戏设计师和地图创作者都致力于创造 GTA V 的错综复杂的世界。他们煞费苦心地设计它，一条街一条街，然后精细地梳理街道，添加行人，汽车，物体等。

![](img/b718c71238e12dc7d24fcd6332e199fe.png)

An example image from GTA V (Grand Theft Auto V)

这个很贵。无论是时间还是金钱。使用**随机模拟场景**我们可能不会做得更好。这意味着重要的边缘情况可能严重欠采样，我们的分类器可能不知道如何正确地检测它们。让我们想象我们正试图训练一个检测**危险场景**的分类器。在现实世界中，我们会很少遇到像下面这样的危险场景，但它们非常重要。如果我们生成大量的随机场景，我们将很少有像下面这样的危险场景。对这些重要案例进行欠采样的数据集可能会产生一个对这些案例失败的分类器。

![](img/4ba7664a2a774b51e33686f9c86d38b4.png)

Example of a dangerous traffic scene. These important cases can be undersampled when randomly sampling synthetic data. Can we do better?

**学习模拟**的想法是，我们可以潜在地学习如何最佳地生成场景，以便深度网络可以学习非常好的表示，或者可以在下游任务中表现良好。

为了测试我们的工作，我们使用虚幻引擎 4 和 Carla 插件[创建了一个*参数化程序交通场景模拟器*。我们的模拟器创建了一条具有不同类型十字路口(X、T 或 L)的可变长度的道路。我们可以在道路上放置建筑物，并在道路上放置 5 种不同类型的汽车。建筑物和汽车的数量由可调参数以及汽车的类型控制。我们还可以在 4 种不同的天气类型之间改变天气，这 4 种天气类型控制照明和雨的效果。主要思想是学习控制不同任务(例如语义分割或对象检测)的这些场景特征的最佳参数。](http://carla.org)

A demo of our procedural scene simulator. We vary the length of the road, the intersections, the amount of cars, the type of cars and the amount of houses. All of these are controlled by a set of parameters.

为了获得传感器数据，我们在我们生成的场景的道路上放置一辆汽车，它可以从生成的场景中捕获 RGB 图像，这些图像自动具有语义分割标签和深度注释(免费！).

An inside view of the generated scenes from our simulator with a fixed set of parameters

但是，模拟算法的学习比这更一般。我们不必将它专门用于交通场景，它可以应用于*任何类型的参数化模拟器*。我们的意思是，对于任何将参数作为输入的模拟器，我们提供了一种搜索最佳参数的方法，使得生成的数据对于深度网络学习下游任务是最佳的。据我们所知，我们的工作是首先进行**模拟优化，以最大限度地提高主要任务**的性能，以及**将其应用于交通场景**。

**继续我们算法的关键点**。传统的机器学习设置如下，其中数据从分布 P(x，y)中采样(x 是数据，y 是标签)。这通常是通过在现实世界中收集数据并手动标记样本来实现的。这个数据集是固定的，我们用它来训练我们的模型。

![](img/421bde21aca34ecffb0886fba4f5f276.png)

Traditional machine learning setup

通过使用模拟器来训练主任务网络，我们可以从模拟器定义的新分布 Q 中生成数据。这个数据集不是固定的，只要计算和时间允许，我们可以生成尽可能多的数据。尽管如此，在该域随机化设置中生成的数据是从 Q 中随机采样的**。获得一个好的模型所需的数据可能很大，并且性能可能不是最佳的。**我们能做得更好吗？****

![](img/83c2da1269e333f7e39250820121bd64.png)

我们引入了学习模拟，它优化了我们在主要任务上选择的度量——通过定义与该度量直接相关的奖励函数 R(通常与度量本身相同)来训练流水线。我们从参数化模拟器 Q(x，y |θ)中采样数据，用它在算法的每次迭代中训练主任务模型。然后，我们定义的奖励 R 用于通知控制参数θ的策略的更新。通过在验证集上测试训练好的网络来获得奖励 R。在我们的例子中，我们使用普通的策略梯度来优化我们的策略。

非正式地说，我们试图找到最佳参数θ，它给出了分布 Q(x，y |θ),使主要任务的精度(或任何度量)最大化。

![](img/23a6a406cbaf4e38e015d644146bd82a.png)

Learning to simulate setup

学习模拟问题的数学公式是一个双层优化问题。试图用基于梯度的方法来解决它对较低层次的问题提出了光滑性和可微性约束。在这种情况下，模拟器也应该是可微的，这通常是不正确的！这就是为什么像香草政策梯度这样的无导数优化方法是有意义的。

![](img/155a1598d304358532770b7fb854bcd0.png)

Mathematical formulation of the bi-level learning to simulate optimization problem

我们在**实例计数**和**语义分割**上演示我们的方法。

我们研究的汽车计数任务很简单。我们要求网络计算场景中每种特定类型的汽车数量。下面是一个右侧带有正确标签的示例场景。

![](img/56ec8c35ec271b00d880db9c06be4cdb.png)

Car counting task example

我们使用**学习来模拟**来解决这个问题，并与仅使用**随机模拟**的情况进行比较。在下图中，关注**红色和灰色**曲线，它们显示了学习模拟(LTS)如何在 250 个时期后获得更高的回报(计数车辆的平均绝对误差更低)。随机抽样的情况会有短暂的改善，但是一旦抽样的随机批次不足以完成任务，性能就会下降。灰色曲线在几次迭代中缓慢上升，但是学习模拟**收敛于蓝色曲线**所示的最佳可能精度(这里我们使用地面真实模拟参数)。

![](img/d25de497cdec111878d0cf893ba289fa.png)

Reward for the car counting task. Note how learning to simulate converges to the best possible reward (on a simulated dataset) shown by the blue curve.

发生了什么事？一个很好的方法是通过可视化我们场景中不同场景和物体的概率。我们绘制了一段时间内的天气概率。我们生成的地面实况验证数据集对某些天气(晴朗的正午和晴朗的日落)进行了过采样，而对其余天气进行了欠采样。这意味着与其他类型的天气相比，有更多清晰的中午和清晰的日落天气的图像。我们可以看到，我们的算法恢复了粗略的比例！

![](img/207d80023c99c10c1783ebe133c94c7f.png)

Weather probabilities (logits) over time

让我们对汽车产卵概率做同样的事情。我们的地面实况数据集对某些类型的汽车(银色日产和绿色甲壳虫)进行了过度采样。学习模拟也反映了训练后的这些比例。本质上，该算法推动模拟器参数生成类似于地面真实数据集的数据集。

![](img/6fb606d809a2aa1634a06d06822807e6.png)

Car probabilities (logits) over time

现在，我们展示一个示例，说明学习模拟如何提高在 [KITTI 交通分割数据集](http://www.cvlibs.net/publications/Geiger2013IJRR.pdf)上进行随机模拟的准确性，该数据集是在现实世界中捕获的数据集**。**

![](img/3d2724a0badb4f684a8a3b5078484416.png)

An example image from the KITTI dataset.

![](img/c91e74330190f4b0d3126ea79a09982f.png)

An example of ground-truth semantic segmentation labels on our simulator. In a simulator, you can get object labels for free — no need for a human annotator

作为我们的基线，我们分别训练主任务模型 600 次，模拟器使用不同的随机参数集为每个模型生成数据。我们监控每个网络的验证 Car IoU 指标，并选择验证奖励最高的网络。然后我们在*看不见的 KITTI 测试设备*上测试它。我们训练学习以模拟 600 次迭代，并获得汽车 IoU(广泛分割度量)的 **0.579，**远高于使用随机参数基线(random params)获得的 **0.480** 。我们还使用另一种无导数优化技术(随机搜索)展示了我们的结果，这种技术在本实验中没有获得好的结果(尽管它在汽车计数中工作得相当好)。最后，我们还通过在 982 个带注释的真实 KITTI 训练图像(KITTI 训练集)上进行训练来展示我们用于分割的 ResNet-50 网络的实际性能，以展示上限。

![](img/b41a2a764b6ac0c4f76c6a0e60798e70.png)

Results for semantic segmenation on the unseen KITTI test set for car semantic segmentation

学习模拟可以被视为元学习算法，其调整模拟器的参数以生成合成数据，使得基于该数据训练的机器学习模型分别在验证和测试集上实现高精度。我们证明它在实际问题中胜过领域随机化，并且相信它是一个非常有前途的研究领域。在不久的将来，看到这方面的扩展和应用会发生什么将是令人兴奋的，我鼓励每个人都来看看模拟和学习模拟如何帮助你的应用或研究。

欢迎所有问题！我的网站在下面。

[](https://natanielruiz.github.io) [## 纳塔尼尔·鲁伊斯

### 研究我探索了计算机视觉的几个课题，包括面部和手势分析，模拟和…

natanielruiz.github.io](https://natanielruiz.github.io)