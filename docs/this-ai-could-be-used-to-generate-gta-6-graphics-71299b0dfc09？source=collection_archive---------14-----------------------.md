# 这个人工智能可以用来生成 GTA 6 图形

> 原文：<https://towardsdatascience.com/this-ai-could-be-used-to-generate-gta-6-graphics-71299b0dfc09?source=collection_archive---------14----------------------->

在视频游戏中设计虚拟世界是一项耗时耗力的任务。像[侠盗猎车手](https://www.youtube.com/watch?v=QkkoHAzjnUs) (GTA)这样的开放世界游戏拥有巨大的虚拟环境，玩家可以自由导航，即使是大型游戏工作室，构建这些视觉效果也需要[长达 4 到 5 年](https://www.quora.com/How-much-time-did-it-take-Rockstar-to-develop-GTA-V)。因此，发布新游戏的周转时间往往相当长。这就是深度学习算法可以帮助减少开发时间的地方，它接管了创意艺术家设计和渲染游戏视觉效果的任务。在这篇文章中，我将回顾两篇最先进的研究论文，它们可以帮助执行在 GTA 这样的游戏中设计和可视化虚拟世界的任务。

# 具有深度学习的图像合成

![](img/8a619b02dacda425466b8d0b675e64af.png)![](img/0479b750a3fd747a08f8a3c87474dd21.png)

**Left:** A semantic label map depicting the objects appearing in this scene. **Right:** A fake image synthesised by a Deep Neural Network from this semantic label map. [[source](https://tcwang0509.github.io/pix2pixHD/)]

我们可以训练一个神经网络来学习我们希望包含在游戏虚拟世界中的各种对象或资产的外观。然后，如果我们给它一个语义标签图(如上所示)来描述这些对象的位置，它可以呈现出逼真的视觉效果。

这项任务被称为使用语义标签的**图像合成**，这个研究领域在过去几年里有了很多新的发展。我今天要介绍的一篇在这项任务中表现出色的研究论文名为 vid2vid。

# 视频到视频合成(vid2vid)

![](img/21a37dc619a6a6989bc598bd0d5f1e2b.png)

麻省理工学院和英伟达的研究人员发表了一篇题为“[视频到视频合成](https://arxiv.org/abs/1808.06601)的论文，该论文能够从高分辨率、时间一致的图像中合成视频。它使用一种特殊的 GAN 架构来确保视频中的合成帧看起来逼真，并且不同帧之间具有视觉一致性。

![](img/4a9f1206d6313f60850236b1d1afd2ce.png)

The GAN architecture used in vid2vid.

vid2vid 网络的**生成器**不仅使用了我们想要转换的当前语义图，还使用了以前帧的少量语义图。此外，它使用来自其先前输出的最终合成帧，并将这些帧组合在一起以计算 F **低图**。这提供了理解两个连续帧之间的差异所需的信息，因此能够合成时间上一致的图像。在鉴别器方面，它使用 I **图像鉴别器**来控制输出质量，除此之外，它还使用**视频鉴别器**来确保合成图像的逐帧序列根据流程图是有意义的。这确保了帧之间几乎没有闪烁。最重要的是，它采用了一种**渐进增长**方法，首先完善较低分辨率的输出，并利用这种知识逐步提升，以产生更高分辨率的输出。看看下图中这个网络的惊人结果。

![](img/8a06cdc211cf4d20c27edd82d50445eb.png)![](img/f3f11119f0c45e5d60d72be3b3141700.png)

Short clip of fake virtual city synthesised by the vid2vid network trained on cityscapes dataset. [[source](https://tcwang0509.github.io/vid2vid/)]

# 基于 GAN 的方法的问题

虽然 vid2vid GAN 网络的视觉质量令人印象深刻，但如果我们想在游戏中实际使用它，还有一个实际问题。你一定注意到了像 GTA 这样的游戏有一个昼夜循环，它改变了虚拟世界的外观。此外，雨和雾等其他天气效应完全改变了这个世界的面貌。这意味着任何试图渲染虚拟世界图形的神经网络也必须能够根据它们的照明和天气效果对不同的视觉风格进行渲染。然而，由于**模式崩溃**的现象，用 GANs 产生不同外观的图像是一个问题。

![](img/8cbbe0dd9dea940add4473219e7cc714.png)

Mode Collapse in GANs resulting in synthesised images unable to produce visually diverse outputs.

## vid2vid 中 GAN 的模式崩溃

想象在某个更高维的坐标空间里有一堆训练图像，我们在上图中简单的用二维表示。这些点中的一些代表白天的样本，一些代表夜晚的图像。现在，当我们开始训练无条件 g an 时，我们首先生成一组随机图像，这些图像将通过生成器推送。现在，训练过程基本上试图将这些假图像推向训练图像，以便它们看起来是真实的。这导致了一个问题，其中一些训练图像可能被遗漏并且永远不会被使用。随着训练的进行，这将导致生成器只产生相同类型的图像。因此，GANs 遭受模式崩溃，并且由这种方法生成的图像在本质上不能是视觉多样的。

这就是我如何看到这篇旨在使用最大似然估计来解决这个问题的研究论文。

# 使用条件 IMLE 的图像合成

![](img/5771bd7e42084334da6ed6c3d0f7ee18.png)

Berkeley 的研究人员发表了论文“[通过条件 IMLE](https://arxiv.org/pdf/1811.12373.pdf) 从语义布局进行多样化图像合成”，旨在通过基于 GAN 的 vid2vid 网络训练过程解决上述问题。它不是专注于提高输出帧的质量，而是专注于能够从完全相同的语义地图中合成不同的图像。这意味着我们可以使用这种方法在任何光照或天气条件下渲染相同的场景，而不像 GANs 那样一个语义标签只能产生一个输出。本文展示了如何使用隐式似然估计或 IMLE 来实现这一点。让我们试着理解为什么在这个特殊的用例中，IMLE 看起来比甘斯做得更好。

![](img/f5f4a790c0b17430df218c114683ff4e.png)

Uncondiitonal case of Implicit Maximum Likelihood Estimation (IMLE) training process.

它首先选取一个训练图像，然后试图将一个随机生成的图像拉近它。请注意，这个过程与 GANs 中的工作方式相反。接下来，它选取另一个训练图像，并向它拉另一个随机图像。重复这个过程，直到我们覆盖了所有的训练图像。这意味着，我们的训练过程现在涵盖了所有白天和夜晚的图像，因此，我们的生成器被强有力地训练以产生不同风格的图像。现在，注意，这是 IMLE 的无条件情况，其中我们从随机噪声图像而不是语义标签图开始，但是训练过程对于两种情况都是相同的。当我们使用语义图时，唯一改变的是输入编码，所以让我们来看看。

![](img/59d47a6db1c16b2b5f45687acd45ba45.png)

Conditional case of IMLE, where input is the semantic label and not a random image like we saw before. A random input noise channel is added to the input encoding which is used to control the visual style of the network’s output.

我们没有使用 **RGB 语义标签**作为输入，而是将地图分解成多个通道。每个通道对应于贴图中的一种对象类型。接下来是这篇论文最重要的部分，我个人认为也是最有趣的部分。它使用一个额外的**噪声输入通道**来控制输出风格的外观。因此，对于该通道中的一个随机噪声图像，输出将遵循固定的输出风格，如白天效果。如果我们改变这个通道到一些其他随机噪声图像，它将遵循另一种风格，像可能是夜间效果。通过**在这两幅随机图像之间插入**，我们实际上可以控制输出图像中的时间。这真的很酷，很迷人！

![](img/afdd2f0ac372f2d1c249399fafdde338.png)

Day-and-night cycle of a virtual world imagined by a Deep Neural Network [[source](https://people.eecs.berkeley.edu/~ke.li/projects/imle/scene_layouts/)]

# 试用这个 AI 来渲染 GTA 5

我试图从游戏 GTA 5 的一个短片中重现这种效果。我用一个图像分割网络获得了游戏的语义标签，然后通过 IMLE 训练过的网络运行它。考虑到是完全相同的**发生器网络能够产生 GTA 的白天和夜晚片段，结果令人着迷！**

![](img/da47ebdef865d7827e2e369366378ca4.png)

你可以在我的 [Youtube 频道](http://youtube.com/c/DeepGamingAI)上观看更多这些视频格式的结果，视频嵌入在下面。

# 结论

在我们今天看到的两篇论文中， **vid2vid** 和 **IMLE** 基于图像合成，我们可以真正看到我们在基于人工智能的图形渲染方面走了多远。在我们开始试验这种新的基于人工智能的图形技术之前，我们只需要清除几个障碍。我预测从今天起大约十年后，像侠盗猎车手这样的游戏将会有某种基于人工智能的资产渲染来帮助减少游戏的开发时间。游戏开发的未来令人激动！

# 参考

1.  [视频到视频合成](https://tcwang0509.github.io/vid2vid/)
2.  [通过条件 IMLE 从语义布局合成多样图像](https://people.eecs.berkeley.edu/~ke.li/projects/imle/scene_layouts/)
3.  [语义切分](https://github.com/NVIDIA/semantic-segmentation)

感谢您的阅读。如果你喜欢这篇文章，你可以关注我在[媒体](https://medium.com/@chintan.t93)、 [GitHub](https://github.com/ChintanTrivedi) 上的更多作品，或者订阅我的 [YouTube 频道](http://youtube.com/c/DeepGamingAI)。