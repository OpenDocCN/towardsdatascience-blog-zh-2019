# 基于 Keras 的命运大序多任务学习

> 原文：<https://towardsdatascience.com/multi-task-learning-on-fate-grand-order-with-keras-261c5e8d3fb8?source=collection_archive---------10----------------------->

![](img/51d9c20bcf6fdb505cf1a450880bad9e.png)

Gender: Female, Region: Europe, Fighting Style: Melee, Alignment: LE, Main Colors: [‘Black’, ‘Blue’, ‘Purple’]. Model saw a number of other images of this character in training, but all were of a significantly different style to this.

标准的图像分类任务遵循二元分类(猫对狗)或者可能多类分类(猫对狗对人)的趋势，但是许多真实世界的任务，如理解图像或文本，通常需要不止一个有用的标签。出于这种更有用的需要，我认为许多任务在本质上可以被框定为多标签，本质上就是单个样本在输出数组中有几个正确的标签。

在之前的一篇文章中，我使用了一些小 CNN 来为鞋子生成元数据标签。在那篇文章中，我有关于材料、图案、颜色等等的变量。我的第一步是制作一个大型多标签模型，一次性预测所有值。我发现这在那种情况下行不通，因为对于有限的数据集来说，特征向量是多么稀疏。所以我依靠创造一系列更小的多标签模型。在我写完那篇文章后，有人在评论区问我为什么不使用多任务模型来完成那项工作(这是一个非常好的问题！)

据我目前的理解，多任务模型是指单个模型优化解决一系列通常相关的问题。从机械上来说，这是通过将来自模型管道的某个核心部分的输出馈送到一系列输出头来完成的，这些输出头的损耗可以被评分和组合，通常通过相加，然后网络可以根据总损耗来调整其权重。

总的想法是，通过学习相关的任务，网络将共享上下文，并学习更好地执行这些任务，而不是独自承担每个单独的任务。吴恩达在他的[深度学习课程](https://www.coursera.org/lecture/machine-learning-projects/multi-task-learning-l9zia)中从高层次讲述了这一点，围绕建立自动驾驶汽车的多任务模型。

> 这篇文章中的所有图片都是测试图片，在它们的描述中有网络生成的标签。

![](img/39e511607a2602454f317c691d66149a.png)

Gender: Female, Region: Asia, Fighting Style: Melee, Alignment: TN, Main Colors: [‘White’, ‘Blue’, ‘Green’]. This one in interesting because I encoded teal as blue and green in the target array so it did well.

# 构建数据集和数据结构化

![](img/744301c0739afdad974cd366b772a933.png)

Gender: Female, Region: Europe, Fighting Style: Melee, Alignment: LG, Main Colors: [‘Silver’, ‘Gold’, ‘Blue’]. This is more similar to the style of the first image that appeared in the training set. It looks like the first artist photoshopped this basic image to make that wallpaper.

在进入模型的细节之前，第一步是为手头的问题构建一个数据集。在这种情况下，我决定围绕命运大订单游戏和系列建立一个壁纸/艺术的图像数据集(当时这只是一个有趣的想法)。我过去在这个游戏上写过一些帖子，在那里我构建了一些基于神经网络的机器人来玩这个游戏。在这篇文章中，我并不特别关心游戏机制，我只是去建立了一个由 400 张图片组成的数据集，这些图片可能包含 40 个不同的角色，并用 26 个不同的标签标注了不同的主题。这包括角色的性别，他们来自的地区，战斗风格，图像的主要颜色，以及角色排列(合法的好，真正的中立，混乱的邪恶，等等。如果你不知道角色的排列，并想阅读更多，请点击[这里](http://easydamus.com/alignment.html)。我的目标只是建立一系列的分类变量，我认为包括一些更难的变量，如字符对齐，将是对更典型的标签如基于颜色的标签的有趣补充。

给它贴标签相当费时，占用了周五晚上的大部分时间。简化过程的是，由于大多数数据不会根据给定的角色(性别、地区、排列、战斗风格)而改变，我能够在单独的 csv 中为它们填写配置文件，并将它们合并到图像的颜色符号中，这节省了大量时间。最终的结果是一些不同的分类变量，当我把它们全部取出时，产生了 26 个不同的类。

# 模型结构和训练

现在开始建模。从技术上来说，我可以采用与之前类似的方法，但这不会成为一篇有趣的文章。因此，我将数据集分为 5 个“任务”(性别、地区、战斗风格、排列和颜色)，每个任务都将成为模型中的一个头部，在这里进行预测、记录损失、对所有任务求和，然后模型会相应地调整其权重。

为了找出潜在的模型架构，我的一般出发点是看看我是否可以使用预先训练的网络作为特征提取器，然后将输出传递给 5 个不同的任务输出头。在对架构结构进行了一些实验后，我发现使用 VGG19 网络作为固定的特征提取器，并在模型头部之前将其输出特征传递给几个密集的计算层是非常成功的。这是一个很好的发现，因为它大大减少了需要优化的参数数量，因为我能够保持 VGG19 权重不变。

我在下面贴了一份模型代码的副本，并将浏览它的一部分。第 3 行用 imagenet 权重初始化 VGG19 网络，然后我遍历并冻结所有层。在第 9 行，我构建了模型输入，这是一个普通的 VGG 风格的 224x224 像素彩色图像，在第 10 行被输入到 VGG19 模型中。接下来，我创建了一些大小为 256 的密集层。我对这些的想法是，因为我离开了冻结的 VGG19 网络，这些层将做繁重的工作来学习和共享不同任务的上下文。一旦构建了这个密集的计算模块，我就开始将该模块的输出(变量 x)输入到标记为 y1、y2、y3、y4、y5 的每个“头”中。现在，我将每个头部创建为一个由 128 个节点组成的密集层，然后是一个由 64 个节点组成的层。最后，所有这些头的顶部都有一个最终输出层，其中节点的数量就是类的数量。对于性别、地域、战斗风格和排列，这些层都有 softmax 激活，我后来给它们提供了交叉熵损失。对于最终的彩色输出头，它是一个 sigmoid 激活，并提供有一个二进制交叉熵损失函数。

![](img/8d49bcf4ff88794c40bc30e1c2108d5a.png)

Multi-task model structure, function takes in a list of 5 losses, 5 metrics, and a level for dropout to initialize the network.

我在 Nvidia 1080 GPU 上运行了 50 个纪元和几分钟的网络。我也开始学习率相当高，但增加了一个功能，以削减学习率的一半，每 5 个时代，所以接近运行结束时，学习率低得多，权重将被调整得更加缓慢。

# 结果

这个模型结构是在相当小的数据集上训练的，该数据集有 26 个类和 400 个数据点。我认为，由于数据的稀疏性，它的性能对于直接的多标签模型来说很难处理，但是由于上下文确实在不同的头部之间共享，所以该模型能够在所有 5 个不同的任务中表现良好。

不同任务的损失如下所示，它显示了不同的任务对于模型来说具有不同的处理难度。性别和颜色之类的事情对模型来说相对容易，这两项任务的准确率分别为 59%和 63%。这可能是一种近乎随机的猜测，因为女性角色的性别比例失调。最困难的任务是排列，其中有 9 个职业倾向于合法的好，因为许多典型的“英雄”类型的角色被归类为有合法的好类型排列。对于这一类，模型开始时非常接近随机猜测的 18.75%

我加入了字符对齐任务，因为我认为这对模型来说很难学习。我认为这是常见的多任务问题，有某些任务会比其他任务更难。因此，我认为这将是一个很好的基准，来衡量网络如何很好地利用来自不同任务的上下文

```
val_gender_acc: 0.5938
val_region_acc: 0.4688
val_fighting_style_acc: 0.4062
val_alignment_acc: 0.1875
val_color_acc: 0.6328
```

在考虑这些更困难的任务时，我在考虑如何提高这些任务的模型性能。这花了我一点时间，但我意识到我可以使用相同的技术来排除网络故障，并将它们应用到那些困难的任务中。这很酷，因为这意味着即使您将所有这些问题合并到一个模型中，您仍然可以依靠您在标准模型开发中熟悉的故障排除技术。

当我让网络训练 50 个历元时，它在第 26 个历元时达到其最低的验证集损失，因此这就是被保存的模型。所有类别都得到了改进，但我很高兴看到字符对齐的验证准确率从最初的 18.75%上升到了 71.88%。我确实认为，我可以通过专门向该体系结构的这一分支添加额外的节点来提高性能，但我现在将把它作为未来的测试。

```
val_gender_acc: 0.9062
val_region_acc: 0.8438
val_fighting_style_acc: 0.8125
val_alignment_acc: 0.7188
val_color_acc: 0.7812
```

![](img/fce173054f22ec49b0733c20afbd4811.png)

Gender: Male, Region: Middle East, Fighting Style: Ranged, Alignment: LG, Main Colors: [‘Gold’, ‘Blue’, ‘White’]. Model was never exposed to this character in training and does a good job overall. His alignment is not Lawful Good though, he’s pretty evil.

# 结束语

我提到的一件事是，网络能够共享不同任务的背景，在我的研究中，我看到的一个领域是颜色和排列似乎彼此有很高的相关性。对于大多数壁纸/扇面艺术，我训练的具有邪恶排列的人物比那些具有良好排列的人物有更暗的图像。因此，对于本帖中的第一个图像，角色有一个合法的好对齐，但被模型标记为合法的邪恶。

![](img/51d9c20bcf6fdb505cf1a450880bad9e.png)

Gender: Female, Region: Europe, Fighting Style: Melee, Alignment: LE, Main Colors: [‘Black’, ‘Blue’, ‘Purple’].

一个更典型的深色图片和一个邪恶的角色在下面的测试图片中。这个角色有一个混乱邪恶的排列，并且经常有深色的艺术品。因此网络正确地推断出测试图像具有邪恶的特征。

![](img/c80323a897abdd1293e1f9a01141b7d4.png)

Gender: Female, Region: Europe, Alignment: CE, Main Colors: [‘Black’, ‘Purple’, ‘Red’]

当第一个图像中的字符以该字符的更正常的配色方案显示时，模型正确地将它们识别为合法商品。虽然我认为分别在对齐和颜色任务上训练的两个网络可以学习这些功能，但并行学习这些任务意味着网络可以直接从不同的任务中学习。

![](img/d0768c5d73c248c7b8dd447793f95a00.png)

Gender: Male, Region: Europe, Fighting Style: Melee, Alignment: LG, Main Colors: [‘Gold’, ‘Silver’, ‘Blue’]. I like this particular artist wlop and did not give any examples of this style in the training set. With this image the gender is incorrect, but the other characteristics are good.

除了上下文共享的好处之外，多任务网络的结构意味着我可以按照特定任务的难度来调整它，就像我调整传统网络一样。这产生了几个世界中最好的。该网络可以在不同的任务之间共享权重和上下文，同时一次性优化所有任务。这意味着减少网络的重复工作(也减少了我的重复工作，因为我必须运行和维护它们)。然后，我可以像调整标准网络一样调整它们，这意味着如果有困难，我可以用我熟悉的方式解决。在一个高层次上，我使用的策略是增加计算能力(节点和层)直到它过拟合，然后用正则化和丢弃来打击网络，以帮助它做出更一般化的预测，并帮助提高验证集性能。

总的来说，这对我来说是一个很好的练习，因为能够建立这种类型的网络架构是有用的，因为它有助于解决需要为给定的数据点生成多个标签的常见现实问题，并且这种解决方案比一群网络更容易维护。

我目前正在研究多任务学习的 Pytorch 实现，所以当我有时间修改它并可能制作一个新的数据集时，我希望能就此发表一篇文章，因为同一数据集上的两篇文章可能有用，也可能没用…除此之外，结合 RNNs 和 CNN 的多任务模型可能会很好。像对齐这样的类别的一个问题是，我很难仅基于图像来解决它，但我知道不同角色的知识/背景，这非常有帮助。由于额外的基于文本的上下文，添加用于特征提取的 RNN 可能是提高模型性能的好方法。这种文本+图像的方法也是我认为有很多现实应用的方法，因为大多数数据点都有这些类型的数据(猫图片+博客帖子，服装图片+描述，物品图像+评论)。

> 请随意查看 github repo 的这篇文章 ，我在那里用过的两个 jupyter 笔记本都在。