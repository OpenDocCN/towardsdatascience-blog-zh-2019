# FGO·斯泰尔根:这种英雄精神并不存在

> 原文：<https://towardsdatascience.com/fgo-stylegan-this-heroic-spirit-doesnt-exist-23d62fbb680e?source=collection_archive---------12----------------------->

![](img/110f54cb961068468ee1019bb6baacd0.png)

Sample outputs from FGO StyleGAN. Also here is a the link to view it for [free](/fgo-stylegan-this-heroic-spirit-doesnt-exist-23d62fbb680e?source=friends_link&sk=70eb6968f4d9e4acc94a5d3f378d477c)

当我第一次看到 Nvidia 的 StyleGAN 的结果时，我觉得它看起来像一堆黑色的魔法。我在 GANs 领域的经验不如深度学习的其他部分，这种经验的缺乏和我真的缺乏 GPU 火力来培养自己的风格的想法阻碍了我更快地投入进来。对于规模，在 StyleGAN github 上，Nvidia 列出了 GPU 规格，基本上是说在 8 个 GPU 上从头开始训练需要大约 1 周时间，如果你只有一个 GPU，训练时间大约为 40 天。因此，运行我的一个 GPU 设备 40 天，从时间和我的电费来看，听起来很可怕。有了这些限制，我暂时放下了培养一名时尚达人的雄心。

![](img/61b3d17d1c304906a8e22bf2f88ea580.png)

FGO StyleGAN outputs

# 信仰的飞跃:自定义 FGO StyleGAN

虽然我和其他人一样喜欢黑魔法，但我也喜欢理解正在发生的事情，尽可能地揭开事物的神秘面纱，并构建自己的事物版本。

几周前，我的一个队友在 LinkedIn 上给我发了一些视频，视频中时装模特以一种视频风格相互变形，我认为这是 StyleGAN 的一种应用。深入调查后，我发现自从我上次查看以来，在 StyleGAN 周围的社区中已经进行了很多工作。就我个人而言，这些天我在 Pytorch 中做了很多工作，但是我认为当你试图将研究应用到你自己的项目中时，通常最简单的方法是使用研究中使用的任何工具。在这种情况下，虽然有一个 [Pytorch port](https://github.com/rosinality/style-based-gan-pytorch) 看起来功能相当不错，但最好的做法是使用基于 Tensorflow 的代码，这项研究是用 Nvidia 的[开源的代码完成的。](https://github.com/NVlabs/stylegan)

然而，是一个名叫 [Gwern Branwen](https://www.gwern.net/) 的人制作了一个网站[这个外服并不存在](https://www.thiswaifudoesnotexist.net/)。坦率地说，如果我没有看到 Gwern 关于他们如何走过他们的 S [tyleGAN](https://www.gwern.net/Faces) 的帖子，并好心地为一个以 512x512 分辨率训练的基于动漫的 StyleGAN 提供预训练的重量，我就不会真的花时间和资源来训练 StyleGAN。

Gwern 展示了基于动画的 StyleGANs，他们使用自己提供的重量训练或由他人训练。虽然我的项目与那里的项目相似，但社区中有人训练了一个“剑脸”StyleGAN，这篇文章中的 StyleGAN 是一个普通命运大令 StyleGAN。

# GAN 背景简介

生成对抗网络(GAN)是深度学习的一个有趣领域，其中训练过程涉及两个网络:生成器和鉴别器。生成器模型开始自己创建图像，它从随机噪声开始，而鉴别器通过查看训练示例和生成器输出来给出反馈，并预测它们是“真”还是“假”。随着时间的推移，这种反馈有助于生成器创建更真实的图像。

StyleGAN 是对英伟达之前一款名为 ProGAN 的型号的改进。ProGAN 经过训练可以生成 1024x1024 的高质量图像，并通过实施渐进的训练周期来实现这一点，在该周期中，它以低分辨率(4x4)开始训练图像，并通过添加额外的层来随着时间的推移提高分辨率。训练低分辨率图像有助于使训练更快并提高最终图像的质量，因为网络能够学习重要的较低水平特征。然而，ProGAN 控制生成图像的能力有限，这正是 StyleGAN 的用武之地。StyleGAN 基于 ProGAN，但增加了生成器网络，允许控制三种类型的功能。

1.  粗糙:影响姿势、一般发型、脸型等
2.  中:影响更精细的五官、发型、眼睛睁开/闭上等。
3.  精细:影响配色方案(眼睛、头发和皮肤)和微观特征。

这只是对 StyleGAN 的简单描述，更多信息请查看[论文](https://arxiv.org/abs/1812.04948)或其他关于[媒体](/explained-a-style-based-generator-architecture-for-gans-generating-and-tuning-realistic-6cb2be0f431)的文章。

![](img/9121b9f773a48b4cde7497d7c4875e2f.png)

This one shows a few male faces. However a lot of them turn into super evil looking images? Maybe guys are just evil? who knows. This one also shows a number of lower quality generated images probably due to me not removing low resolution images properly when I created the dataset.

# 数据集构建和准备

为了让 StyleGAN 运行起来，最困难的部分是决定我想如何处理这个问题，并获得一个格式正确的数据集。我在前进的道路上犯了一些错误，我也将走过这些错误。根据最初的论文和我看到的许多其他风格，我决定制作一个头像数据集。至于话题，我手头仅有的足够大的数据集与《FGO》有关。

![](img/43bbde4bcbfdc26c0f29dd0589fb6d18.png)![](img/5591647eb928c1b1da0fd6c6189c1b49.png)![](img/258b02a3a6a0b1b637c5a37b7dd095b4.png)![](img/1369c7b883362502e2b25e3b6e69a560.png)![](img/51bf86015fa278c3cfe48d8806c43b9e.png)![](img/98eec56a11269b9a7effba641daa1a09.png)

I used a previously built Tensorflow Object detector to crop out the heads from around 6K FGO wallpaper/fan art images.

我使用的 FGO 数据集由大约 6000 张不同尺寸的壁纸、游戏图片、粉丝艺术等图片组成。我通过一个基于 Tensorflow 的 FGO 调谐头部检测器在这个数据集上提取了大约 8K 个头部。完成后，我用手快速检查并清理了数据集，将最终数据集削减到大约 6K 头。我学到的一个教训是，我应该在过程的这一部分花更多的时间，因为有许多质量较低的图像，还有背景、盔甲、非人物图像的图像留在数据集中，这导致生成的图像中出现奇怪的伪影或只是质量较低的图像。

像其他 Tensorflow 网络一样，StyleGAN 依赖于使用 *tfrecord* 文件，并且可以由 StyleGAN repo 中的 [*dataset_tool.py*](https://github.com/NVlabs/stylegan/blob/master/dataset_tool.py) 文件生成，用于训练。训练 StyleGAN 的一个重要注意事项是，数据集中的图像需要与用于预训练权重的 StyleGAN 具有相同的大小和相同的颜色格式。所以对我来说，使用 1024x1024 的原始网络很困难，因为找到那个尺寸的动漫头像有点困难，我的 GPU 可能无法处理它。这使得 Gwern 训练的 512x512 网络非常有吸引力，因为它更容易找到我可以获得接近该分辨率的头部的图像，并且我的 GPU 可以更容易地处理它。我把所有的图片都转换成 sRGB 图片，并把它们的尺寸调整到 512x512。

为了调整大小，我使用了 Python 的 PIL 库进行处理和重新格式化。一旦完成，我就准备开始漫长而艰苦的过程，根据我的 FGO 用例对 StyleGAN 进行微调。

# 培养

虽然从预训练的重量开始很有帮助，但训练绝不是一个短暂的过程。毕竟，在我的 1080 卡上花费了一周多一点的 GPU 时间(约 180 小时)后，我停止了训练，这相当于约 85 个“滴答”,这是 StyleGAN 用作纪元的 30K 图像周期……因此它生成了 250 万张 FGO 图像。

![](img/0885fd8ac8cb18e1a8ec4ea241d2196c.png)

Images generated during training for 85 ticks. A few boxes were based on low quality of non headshot images so they just never developed well. However it is interesting to see the model trying to generate headgear (to varying success) as well as a few male characters who show up and I think have cool overly dramatic facial features and hair

StyleGAN 的体能训练过程由 Nvidia repo 中的 2 个脚本控制。为此，我主要遵循了 Gwern 在他们的博客中提出的建议。

## 培训/training_loop.py

要调整的代码的主要区域从第 112 行开始。本节设置了许多默认参数，因为 StyleGAN 不接受命令行参数，所以必须在这里设置它们。

![](img/55e5c47a5390035c9357e7fc033dadde.png)

要设置的主要参数之一是 **resume_kimg** 参数。如果它被设置为 0，那么 StyleGAN 将在看似随机的初始化开始训练。因此，我将它设置为 **resume_kimg=7000** 在这个级别，斯泰勒根能够利用预训练的重量，并在一个比其他情况好得多的点开始。

![](img/35c6b53befa9593710926d57f0077050.png)

我遇到了 OOM 问题，并最终追踪到它与第 266 行的度量行有关，所以我最终只是注释掉了它。到目前为止，我还没有遇到其他问题。

## Train.py

有两个地方需要调整，第一个是指定使用哪个数据集的地方。为此，我添加了我准备好的名为“脸”的数据集。

```
desc += '-faces'; dataset = EasyDict(tfrecord_dir='faces',resolution=512);              train.mirror_augment = True
```

![](img/90817fedfbffe80fecf534cda9b78d9f.png)

第二个方面是详细说明培训过程。因为我有 1 个 GPU，所以我将 GPU 的数量设置为 1，并指定每个迷你批次应该有 4 个图像。之后是一些未调整的学习率调度和总数为 99000K 的图像。这实质上设置了模型将训练多长时间。

# 未来的改进

对于这一个，我认为我可以在两个方面改进我的工作流程。一个是硬件，这是一个非常耗时的项目，如果我决定做更多的 GAN 工作，我认为升级到我的主 GPU 装备将是一个很好的生活质量改善。其次是数据集质量。我没有彻底清理我的数据集，并为此付出了代价，我认为这是一个很好的警示故事。

## 五金器具

这个项目是我经历过的时间更密集的项目之一，所以坦率地说，它让我考虑是否值得升级我的计算资源。为此，我可以采取两条主要路线，要么使用更多的云资源，如 AWS，要么物理升级我的 GPU。

虽然 AWS 可能是更容易的方法，但我认为通过我所做的大量培训，我会很快产生足够的 AWS 费用，让我希望我升级了我的 GPU。社区中的人们提到，构建自己的 GPU 平台比使用 AWS 等资源大约便宜[10 倍](https://medium.com/the-mission/why-building-your-own-deep-learning-computer-is-10x-cheaper-than-aws-b1c91b55ce8c)。

对我来说，花数百美元来测试一种风格似乎很可怕。

因此，我目前的主要想法是升级到 1080 ti ~ $ 800–1000，或者如果我真的想全押，一次性成本约$1300 的 2080TI 卡应该会有收益。如果任何人对此有想法，请随时告诉我！

两者都应该有助于减少所需的训练时间，因为它们比我目前的主 1080 GPU 快得多。

![](img/09e94fb5ab3da83517a8fbb903e05524.png)

More selected outputs

## 数据集质量

与数据科学中的所有事情一样，数据集的初始质量和清理工作可以决定一个项目的成败。为了这个项目，我加快了清洁过程，以便快速训练初始风格。我觉得这很好，但也是我的一个错误。如果我花更多的时间来清洁，结果可能会好得多。

在我的辩护中，我第二天要飞去参加一个咏春研讨会，我想早点开始练习咏春，而不是失去我离开时可以得到的 48 小时的训练时间。

一些初步想法，**去除低质量图像**，**去除非头像图像**，**去除重复。**

去除低质量图像的一个简单过程是删除低于某一分辨率的图像或者没有高于某一截止值(比如 300 像素)的边的图像。

去除非头部图像有点复杂，但可以用 CNN 和一个相当简单的标记头部和非头部图像的过程来完成。但是它需要额外的时间来构建。

重复的图像可能对这个过程没有太大帮助，但我也不确定这会有多严重？我为这个项目建立了一个数据集，使用了 google image pulls 来收集一些与 FGO 相关的术语和字符。这个过程会产生大量的副本。例如，如果我搜索类似“FGO 军刀”的东西，这将会给我“军刀类”或通常被称为“军刀”的 FGO 人物。接下来，如果我搜索“for 阿尔托莉亚·潘德拉贡”，他通常被称为“军刀”，我会得到很多重复的图像。

我可以使用感知哈希之类的东西对数据集进行重复数据删除，使用 [ImageHash](https://pypi.org/project/ImageHash/) 库，这是我为工作或 CNN 做的，效果也很好。

到目前为止，我认为最重要的事情是确保图像的质量足够高，StyleGAN 将学会制作好的 512x512 图像。希望对高分辨率数据集进行仔细的手动修剪可以去除大多数非头部图像，复制可能会使 GAN 倾向于制作更多的这些字符，但目前对我来说这似乎不是最糟糕的问题。

![](img/a87cb92ad43809010e53afe16e6c1b0c.png)

One of the things I was happy about is that the GAN learned to generate Jeanne’s head piece that you can see appear for a bit in both of these examples

# 结束语

正如我之前所说，这是我参与过的计算量更大的项目之一，但结果非常有趣，我想再做几个。从最初训练 from 斯泰勒根中学到的经验应该有助于理顺这些过程。

在看这些的时候，我有一个有趣的想法，那就是我是否可以在计算机视觉问题中使用 StyleGANs 进行数据增强。Gwern 指出，其他人已经在小于 5K 样本的训练模型上取得了相当大的成功，而 FGO·斯泰勒根也取得了类似的成功。这对于样本相对较少的领域非常有用，并且可以通过 NVIDIA 添加到 StyleGAN over ProGAN 中的功能控件来控制样本的生成。虽然这种方法的可行性还有待观察，但至少它可能很酷，而且可能会被宣传为 R&D 的工作项目。谁知道呢？

与此同时，我将享受为这个 FGO 风格产生更多的输出样本和图像。我目前有第二个 StyleGAN 训练，我使用动画重量作为人物照片数据集的起点。我不确定结果会如何，但我会在适当的时候分享这些结果。

> 随意查看 Git 回购[这里](https://github.com/sugi-chan/FGO_StyleGAN)

![](img/99ca4b663abc4fd81fdcb80f0b723b14.png)

Both of these showcase some of the issues I had where low resolution or non head images were in the dataset so it rotates through some weird/creepy stages.