# 图像到图像的翻译

> 原文：<https://towardsdatascience.com/image-to-image-translation-69c10c18f6ff?source=collection_archive---------17----------------------->

图像到图像的翻译是一类视觉和图形问题，其目标是学习输入图像和输出图像之间的映射。它的应用范围很广，如收藏风格转移、物体变形、季节转移、照片增强等。

# CycleGAN

![](img/55d711049ad70a29d8b12adde0040758.png)

`Unpaired Image-to-Image Translation Using Cycle-Consistent Adversarial Networks(ICCV 2017)`

[论文/](https://arxiv.org/pdf/1703.10593v6.pdf) [项目/](https://junyanz.github.io/CycleGAN/) [语义学者](https://www.semanticscholar.org/paper/Unpaired-Image-to-Image-Translation-Using-Networks-Zhu-Park/55c22f9c8f76b40793a8473248873f726abd8ce9)

作者提出了一种在缺少成对例子的情况下学习将图像从源域 X 翻译到目标域 Y 的方法。目标是学习映射 G : X → Y，使得来自 G(X)的图像分布与使用对抗损失的分布 Y 不可区分。因为这种映射是高度欠约束的，所以我们将其与逆映射 F : Y → X 耦合，并引入循环一致性损失来强制 F(G(X)) ≈ X(反之亦然)。

![](img/def26fdaac581aa9c2e8c7fbefe8d90d.png)

成对的训练数据(左)由一一对应的训练实例组成。不成对的训练集没有这样的对应关系(图取自论文)

![](img/735d7d3467fe6287178b54f8d18d353c.png)

图取自报纸。

该模型包含两个映射函数 G : X → Y 和 F : Y → X，以及相关的对抗性鉴别器 DY 和 DX。DY 鼓励 G 将 X 转换为与域 Y 不可区分的输出，反之亦然。为了进一步规范映射，他们引入了两个“循环一致性损失”,这两个损失捕捉了这样的直觉:如果我们从一个域转换到另一个域，然后再转换回来，我们应该到达我们开始的地方。

# 斯塔根

![](img/106e39a73f55f261cfdcd46b75d799ea.png)

`Unified Generative Adversarial Networks for Multi-Domain Image-to-Image Translation(CVPR 2018)`

[论文/](https://arxiv.org/pdf/1711.09020v3.pdf) [代码/](https://github.com/yunjey/stargan) [语义学者](https://www.semanticscholar.org/paper/StarGAN%3A-Unified-Generative-Adversarial-Networks-Choi-Choi/4273c24df71ec59c0c1cad95342d01cf53bb7d8d)

现有的图像到图像的转换方法在处理多于两个域时具有有限的可扩展性和鲁棒性，因为对于每对图像域应该独立地建立不同的模型。StarGAN 是一种新颖且可扩展的方法，仅使用一个模型就可以为多个领域执行图像到图像的翻译。

![](img/ab182087c6c7279a014c2b564addd33b.png)

跨域模型和我们提出的模型 StarGAN 之间的比较。(a)为了处理多个域，应该为每对图像域建立跨域模型。(b) StarGAN 能够使用单个生成器学习多个域之间的映射。图中显示了连接多个域的星型拓扑。(图取自论文)

![](img/d9621f403a3e9463d7fcb9f0e38d02b0.png)

StarGAN 概述，由两个模块组成，一个鉴别器 D 和一个生成器 G. (a) D 学习区分真假图像，并将真实图像分类到其对应的域。(b) G 接受图像和目标域标签作为输入，并生成假图像。目标域标签被空间复制并与输入图像连接。g 尝试从给定原始域标签的伪图像重建原始图像。(d) G 试图生成与真实图像无法区分的图像，并被 d 归类为目标域

*原载于*[*amberer . git lab . io*](https://amberer.gitlab.io/papers_in_ai/img2img-translation.html)*。*