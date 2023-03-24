# Pytorch 闪电 vs PyTorch Ignite vs Fast.ai

> 原文：<https://towardsdatascience.com/pytorch-lightning-vs-pytorch-ignite-vs-fast-ai-61dc7480ad8a?source=collection_archive---------4----------------------->

![](img/aa4f3f75f4cb7ce7845f803e23f3366b.png)

Apparently a lion, bear, and tiger are friends

PyTorch-lightning 是一个最近发布的库，是 PyTorch 的一个类似 Kera 的 ML 库。它将核心培训和验证逻辑留给您，并将其余部分自动化。(顺便说一句，我说的 Keras 是指没有样板，没有过度简化)。

作为《闪电》的核心作者，我几次被问到《闪电》和 fast.ai、 [PyTorch ignite](https://github.com/pytorch/ignite) 的核心区别。

在这里，我将**尝试**对所有三个框架进行客观的比较。这种比较来自于在所有三个框架的教程和文档中客观发现的相似性和差异。

注意:本文捕获了撰写本文时(2019 年 8 月)这些框架中可用的特性。这些框架有更新的版本。

# 动机

![](img/55d8b46d2bf04a95c9bafca0939e07d2.png)

Ummmm

Fast.ai 最初是为了便于教授 [fast.ai 课程](https://www.fast.ai/2018/10/02/fastai-ai/)而创建的。它最近还演变成一个常用方法的实现库，如 GANs、RL 和迁移学习。

[PyTorch Ignite](https://github.com/pytorch/ignite#why-ignite) 和 [Pytorch Lightning](https://github.com/williamFalcon/pytorch-lightning#why-do-i-want-to-use-lightning) 都是为了给研究人员尽可能多的灵活性，要求他们为训练循环和验证循环中发生的事情定义函数。

Lightning 还有另外两个更加雄心勃勃的动机:可复制性和民主化的最佳实践，只有 PyTorch 超级用户才会实施(分布式培训、16 位精度等等)。我将在后面的章节中详细讨论这些动机。

因此，在基本层面上，目标用户是明确的:对于 fast.ai 来说，它是对深度学习感兴趣的人，而另外两个则专注于活跃的人工智能研究人员(即生物学家、神经科学家等)

# 学习曲线

![](img/28473adb9cf70ae579f755590406a810.png)

Framework Overload

Lightning 和 Ignite 都有非常简单的界面，因为大部分工作仍然由用户在纯 PyTorch 中完成。主要工作分别发生在[发动机](https://pytorch.org/ignite/engine.html)和[训练器](https://williamfalcon.github.io/pytorch-lightning/Trainer/)物体内部。

然而，Fast.ai 确实需要在 PyTorch 上学习另一个库。大多数时候，API 并不直接在纯 PyTorch 代码上操作(有些地方是这样的)，但是它需要像 [DataBunches](https://docs.fast.ai/basic_data.html#DataBunch) 、 [DataBlocs](https://docs.fast.ai/data_block.html#The-data-block-API) 等抽象。当做某事的“最佳”方式不明显时，这些 API 非常有用。

然而，对于研究人员来说，重要的是不必学习另一个库，直接控制研究的关键部分，如数据处理，而无需其他抽象操作。

在这种情况下，fast.ai 库具有更高的学习曲线，但如果您不一定知道做某事的“最佳”方法，而只想采用好的方法作为黑盒，那么这是值得的。

# 闪电 vs 点燃

![](img/7c3f15ae04bf9010cf784dc0b285dd3d.png)

More like sharing

从上面可以清楚地看到，考虑到用例和用户的不同，将 fast.ai 与这两个框架进行比较是不公平的(然而，我仍然会将 fast.ai 添加到本文末尾的比较表中)。

Lightning 和 ignite 之间的第一个主要区别是它的操作界面。

在 Lightning 中，有一个标准接口(见 [LightningModule](https://williamfalcon.github.io/pytorch-lightning/LightningModule/RequiredTrainerInterface/) )包含每个模型必须遵循的 9 个必需方法。

这种灵活的形式为培训和验证提供了最大的自由度。这个接口应该被认为是一个 ***系统，*** 不是一个模型。系统可能有多个模型(GANs、seq-2-seq 等),也可能是 as 模型，比如这个简单的 MNIST 例子。

因此，研究人员可以随心所欲地尝试许多疯狂的事情，只需担心这 9 种方法。

Ignite 需要非常相似的设置，但没有每个型号都需要遵循的标准 ***接口。***

注意**运行**功能可能有不同的定义，即:可能有许多不同的事件被添加到训练器中，或者它甚至可能被命名为不同的名称，如 ***main、train 等……***

在一个复杂的系统中，训练可能会以奇怪的方式发生(看看你的 GANs 和 RL)，对于查看这些代码的人来说，发生了什么并不明显。而在 Lightning 中，您会知道查看 training_step 来弄清楚发生了什么。

# 再现性

![](img/e7622cf8b1a8aea68e384e8072982ee5.png)

When you try to reproduce work

正如我提到的，创造闪电还有第二个更远大的动机:**再现性**。

如果你试图阅读某人对一篇论文的实现，很难弄清楚发生了什么。我们只是设计不同的神经网络架构的日子已经一去不复返了。

现代 SOTA 模型实际上是 ***系统*、**，它们采用许多模型或训练技术来实现特定的结果。

如前所述，LightningModule 是一个 ***系统*** ，而不是一个模型。因此，如果你想知道所有疯狂的技巧和超级复杂的训练发生在哪里，你可以看看训练步骤和验证步骤。

如果每个研究项目和论文都使用 LightningModule 模板来实现，那么就很容易发现发生了什么(但可能不容易理解哈哈)。

人工智能社区的这种标准化也将允许生态系统蓬勃发展，该生态系统可以使用 LightningModule 接口来做一些很酷的事情，如自动部署，审计系统的偏差，甚至支持将权重散列到区块链后端，以重建用于可能需要审计的关键预测的模型。

# 现成的功能

![](img/756ec40f34e6d37dd7f6940b79df217b.png)

Ignite 和 Lightning 的另一个主要区别是 Lightning 支持开箱即用的功能。开箱即用意味着您没有**额外的代码。**

为了说明，让我们尝试在同一台机器上的多个 GPU 上训练一个模型

**点燃(** [**演示**](https://github.com/pytorch/ignite/blob/master/examples/mnist/mnist_dist.py) **)**

**闪电(** [**演示**](https://github.com/williamFalcon/pytorch-lightning/blob/master/examples/new_project_templates/single_gpu_node_ddp_template.py) **)**

好吧，两者都不错…但是如果我们想在许多机器上使用多 GPU 呢？让我们在 200 个 GPU 上训练。

**点燃**

…

嗯，对此没有内置的支持…你必须对这个例子进行一些扩展，并添加一个容易提交脚本的方法。然后你必须负责加载/保存，而不是用所有的进程覆盖权重/日志，等等…你明白了吧。

**闪电**

使用 lightning，您只需设置节点数量并提交适当的作业。[这里有一个关于正确配置作业的深入教程](https://medium.com/@_willfalcon/trivial-multi-node-training-with-pytorch-lightning-ff75dfb809bd)。

开箱即用的特性是您 ***不需要做任何事情就能获得的特性。*** 这意味着你现在可能不需要它们中的大部分，但当你需要说…积累渐变，或渐变剪辑，或 16 位训练时，你不会花几天/几周的时间通读教程来让它工作。

只要设置合适的闪电标志，继续你的研究。

Lightning 预建了这些功能，因此用户可以花更多时间进行研究，而不是进行工程设计。这对于非 CS/DS 研究人员特别有用，例如物理学家、生物学家等，他们可能不太熟悉编程部门。

这些特性使 PyTorch 的特性民主化，只有超级用户才可能花时间去实现。

下面的表格比较了所有 3 个框架的功能，并按功能集进行了分组。

*如果我错过了什么重要的东西，请发表评论，我会更新表格！*

# 高性能计算

# 调试工具

# 可用性

# 结束语

![](img/2c545b3a656489c20bd04c9f99281b53.png)

在这里，我们对三个框架进行了多层次的深入比较。每个人都有自己的优点。

如果你刚刚开始学习或者没有掌握所有最新的最佳实践，不需要超高级的培训技巧，并且有时间学习新的库，那么就用 fast.ai 吧。

如果你需要最大的灵活性，选择点燃闪电。

如果你不需要超级先进的功能，并且可以添加你的 tensorboard 支持、累积梯度、分布式训练等，那么就用 Ignite 吧。

如果您需要更多高级功能、分布式培训、最新和最棒的深度学习培训技巧，并且希望看到一个实现在全球范围内标准化的世界，请使用 Lightning。

**编辑**:

2019 年 8 月:编辑以反映 Ignite 团队的反馈。

2021 年 2 月:正在进行研究以验证 Fast.ai v1.0.55 中可用的功能