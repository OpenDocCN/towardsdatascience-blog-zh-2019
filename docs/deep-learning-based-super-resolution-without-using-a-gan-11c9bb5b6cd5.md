# 基于深度学习的超分辨率，不使用 GAN

> 原文：<https://towardsdatascience.com/deep-learning-based-super-resolution-without-using-a-gan-11c9bb5b6cd5?source=collection_archive---------1----------------------->

本文描述了用于图像改善、图像恢复、修复和超分辨率的技术和训练深度学习模型。这利用了 Fastai 课程中教授的许多技术，并利用了 Fastai 软件库。这种训练模型的方法是基于非常有才华的人工智能研究人员的方法和研究，我已经在信息和技术方面归功于他们。

据我所知，我在训练数据中应用的一些技术在这些学习方法中是独一无二的(截至 2019 年 2 月)，只有少数研究人员将所有这些技术结合在一起使用，他们大多数可能是 Fastai 的研究人员/学生。

# 超分辨率

超分辨率是放大和/或改善图像细节的过程。通常，低分辨率图像被作为输入，并且同一图像被放大到更高的分辨率，这是输出。高分辨率输出中的细节在细节基本未知的地方被填充。

超分辨率本质上是你在电影和连续剧中看到的东西，比如 CSI，有人放大图像，图像质量提高，细节就出现了。

我第一次听说“人工智能超级分辨率”是在去年 2018 年初的优秀 YouTube 2 分钟论文中，该论文以对最新人工智能论文的简短精彩评论为特色(通常长于 2 分钟)。当时这看起来像是魔法，我不明白这怎么可能。绝对符合阿瑟·C·克拉克的名言“任何先进的技术都与魔法无异”。我没有想到，不到一年的时间，我就可以训练自己的超分辨率模型，并撰写相关文章。

这是我正在撰写的一系列文章的一部分，作为我在人工智能和机器学习方面正在进行的学习和研究的一部分。我是一名软件工程师和分析师，我的日常工作是成为一名人工智能研究员和数据科学家。

我写这些部分是为了加强我自己的知识和理解，希望这也能对其他人有所帮助和兴趣。我试图用尽可能简单的英语来讲述大部分内容，希望它对任何熟悉机器学习的人来说都有意义，并有一些更深入的技术细节和相关研究的链接。这些主题和技术很难理解，我花了好几个月的时间进行实验和写作。如果你不同意我所写的，或者认为它是错误的，请联系我，因为这是一个持续的学习过程，我会感谢你的反馈。

下面是一个低分辨率图像的示例，对其进行了超分辨率处理以提高其分辨率:

![](img/03ff9954f4afd399b6f73b4aa2b5c6ed.png)

Left low resolution image. Right super resolution of low resolution image using the model trained here.

基于深度机器学习的超分辨率试图解决的问题是，基于传统算法的放大方法缺乏细节，无法消除缺陷和压缩伪像。对于手动执行这些任务的人来说，这是一个非常缓慢和艰苦的过程。

好处是从从未存在或已丢失的图像中获得更高质量的图像，这在许多领域甚至在医学应用中挽救生命都是有益的。

另一个用例是计算机网络间传输的压缩。想象一下，如果您只需要发送一个 256x256 像素的图像，而实际上需要的是一个 1024x1024 像素的图像。

在下面的一组图像中，有五幅图像:

*   要放大的低分辨率输入图像
*   通过最近邻插值放大的输入图像
*   通过双线性解释放大的输入图像，这是您的互联网浏览器通常需要的
*   通过该模型的预测，输入图像被升级和改进
*   目标影像或地面实况，其被缩小以创建较低分辨率的输入。

目标是将低分辨率图像改进为与目标一样好(或更好)，称为地面实况，在这种情况下，地面实况是我们缩小为低分辨率图像的原始图像。

![](img/9b608e219bb7026a8dd8a88312143940.png)

Comparing the low resolution image, with conventional upscaling, a deep learning model prediction and the target/ground truth

为了实现这一点，数学函数采用缺少细节的低分辨率图像，并在其上产生细节和特征的幻觉。在这样做的过程中，该功能找到了原始摄像机可能从未记录的细节。

这个数学函数被称为模型，放大的图像是模型的预测。

一旦这个模型和它的训练被解释了，在这篇文章的结尾提到了潜在的伦理问题。

# 图像修复和修补

为超分辨率而训练的模型对于修复图像中的缺陷(jpeg 压缩、撕裂、折叠和其他损坏)也应该是有用的，因为模型对某些特征应该看起来像什么有概念，例如材料、毛发甚至眼睛。

图像修复是修饰图像以移除图像中不需要的元素(如铁丝网)的过程。对于训练来说，常见的是剪切图像的一些部分，并训练模型来替换丢失的部分，这是基于应该有什么的先验知识。当由熟练人员手动执行时，图像修复通常是一个非常缓慢的过程。

![](img/2b577347a7e66c2d86642c203ae4ca2b.png)

Left an image with holes punched into it and text overlayed. Middle deep learning based model prediction of repaired image. Right the target or Ground truth without defects.

超分辨率和修复似乎经常被认为是分开的和不同的任务。然而，如果一个数学函数可以被训练来创建图像中没有的额外细节，那么它也应该能够修复图像中的缺陷和缺口。这假设这些缺陷和缺口存在于训练数据中，以便模型学习它们的恢复。

# 超分辨率的 GANs

大多数基于深度学习的超分辨率模型是使用生成对抗网络(GANs)训练的。

GANs 的局限性之一是，它们实际上是一种懒惰的方法，因为它们的损失函数，即 critical，是作为过程的一部分来训练的，而不是专门为此目的而设计的。这可能是许多模型只擅长超分辨率而不擅长图像修复的原因之一。

# 普遍适用

许多深度学习超分辨率方法不能普遍适用于所有类型的图像，几乎都有其弱点。例如，为动物的超分辨率训练的模型可能不适合人脸的超分辨率。

用本文中详细描述的方法训练的模型似乎在包括人类特征在内的各种数据集上表现良好，这表明在任何类别的图像上有效放大的通用模型是可能的。

## X2 超分辨率的例子

以下是在 Div2K 数据集上训练的同一模型的 X2 超分辨率(图像大小加倍)的十个示例，该数据集是各种主题类别的 800 幅高分辨率图像。

例子一来自一个在不同种类的图像上训练的模型。在早期的训练中，我发现改善人类图像的效果最差，而且呈现出更艺术的平滑效果。然而，在通用类别数据集上训练的这个版本的模型已经设法很好地改善了这张图像，仔细观察面部、头发、衣服褶皱和所有背景中添加的细节。

![](img/01e81b36939b5c2abbaafa3f8b873514.png)

Super resolution on an image from the Div2K validation dataset, example 1

例子二来自一个在不同种类的图像上训练的模型。该模型为树木、屋顶和建筑窗户添加了细节。再次令人印象深刻的结果。

![](img/aab50055ca36727658fe735c0b45792e.png)

Super resolution on an image from the Div2K validation dataset, example 2

例子三来自一个在不同类别的图像上训练的模型。在不同数据集上训练模型的过程中，我发现人脸的结果最不令人满意，但是在不同类别的图像上训练的模型已经成功地改善了面部的细节，并查看了添加到头发上的细节，这令人印象深刻。

![](img/6483e3c1677640a0ed1fe38166b22f46.png)

Super resolution on an image from the Div2K validation dataset, example 3

例子四来自一个在不同种类的图像上训练的模型。添加到镐轴、冰、夹克褶皱和头盔上的细节令人印象深刻:

![](img/33ed64a23dec358e258a90b290bc8565.png)

Super resolution on an image from the Div2K validation dataset, example 4

例子五来自一个在不同种类的图像上训练的模型。花卉的改进令人印象深刻，鸟眼、鸟嘴、皮毛和翅膀的细节也是如此:

![](img/c65b16fb6eb0faa352b0bc5f6a3efed5.png)

Super resolution on an image from the Div2K validation dataset, example 5

例子六来自一个在不同种类的图像上训练的模型。这个模型成功地给人的手、食物、地板和所有的物体添加了细节。这真是令人印象深刻:

![](img/5369bfec84f15dea3b2902cdda604777.png)

Super resolution on an image from the Div2K validation dataset, example 6

例子七来自一个在不同种类的图像上训练的模型。该模型将毛发聚焦，并保持背景模糊:

![](img/62cb834ad7f0c460b1848be2b7176c4e.png)

Super resolution on an image from the Div2K validation dataset, example 7

例八来自一个在不同类别的图像上训练的模型。该模型很好地锐化了窗口之间的线条:

![](img/9a1c0110c6f2806b04e804fa19da286a.png)

Super resolution on an image from the Div2K validation dataset, example 8

例子九来自一个在不同种类的图像上训练的模型。皮毛的细节真的好像是模特想象出来的。

![](img/c6163861d0a99406fb1f618dcdddfebb.png)

Super resolution on an image from the Div2K validation dataset, example 9

例 10 来自在不同类别的图像上训练的模型。这似乎真的令人印象深刻的锐化周围的结构和灯光。

![](img/c66c82f11d51096babb96cc15d55e7af.png)

Super resolution on an image from the Div2K validation dataset, example 10.

示例 11 来自在不同类别的图像上训练的模型。羽毛的改进和变尖非常明显。

![](img/c27dc97a7c834e73265617233fd87f4a.png)

Super resolution on an image from the Div2K validation dataset, example 11.

示例 12 来自在不同类别的图像上训练的模型。这种内部图像几乎在所有地方都得到了微妙的改善。

![](img/01f7cde980774ed0bba964e18a6cdae9.png)

Super resolution on an image from the Div2K validation dataset, example 12.

示例 13 来自在不同类别的图像上训练的模型。这是本节的最后一个例子，一个复杂的图像已经被锐化和改进。

![](img/13c289189ea4c4c57b872bd5af41a93d.png)

Super resolution on an image from the Div2K validation dataset, example 13.

# 该模型的预测具有超高的分辨率

以上所有图像都是在训练期间或训练结束时对验证图像集进行的改进。

经过训练的模型已用于创建超过 100 万像素的放大图像，以下是一些最佳示例:

在该第一示例中，以高 JPEG 质量(95)保存的 256 像素正方形图像被输入到模型中，该模型将图像放大到 1024 像素正方形图像，执行 X4 超分辨率:

![](img/7630322b9ca38638231ac5a08b8a7ae2.png)

Super resolution using deep learning, example 1

上面的图片集不一定能准确预测，查看我的公共 Google drive 文件夹中的完整 PDF:
[https://drive.google.com/open?id = 1g 0 o7 ul 7 zllxki _ 0 gsz 2 C4 qo 4 f 71 S6 F5 I](https://drive.google.com/open?id=1g0o7uL7ZlLxKI_0GSz2C4qO4f71s6F5i)

在下一个例子中，以低 JPEG 质量保存的 512 像素图像(30)被输入到模型中，该模型将图像放大到 1024 像素的正方形图像，对较低质量的源图像执行 X2 超分辨率。在这里，我相信模型的预测看起来比目标地面真实图像更好，这是惊人的:

![](img/b51d2e4f6c3807aef6c39d09f940f08a.png)

Super resolution using deep learning, example 2

上面的图片集不一定能准确预测，请查看我的公共 Google drive 文件夹中的全尺寸 PDF:
【https://drive.google.com/open? id = 1 fro 6n 7d qfh qzw 5-oTGTMgjUTMQGsGeaD

用最基本的术语来说，这个模型:

*   接受图像作为输入
*   让它通过一个训练有素的数学函数，这是一种神经网络
*   输出与输入尺寸相同或更大的图像，这是对输入尺寸的改进。

这建立在杰瑞米·霍华德和雷切尔·托马斯在 Fastai 课程中建议的技术之上。它使用 Fastai 软件库、PyTorch 深度学习平台和 CUDA 并行计算 API。

Fastai 软件库打破了很多复杂深度学习入门的障碍。因为它是开源的，所以如果需要的话，很容易定制和替换你的架构的元素来适应你的预测任务。这个图像生成器模型是建立在 Fastai U-Net 学习器之上的。

该方法使用以下内容，下面将进一步解释其中的每一项:

*   一种交叉连接的 U-Net 架构，类似于 DenseNet
*   基于 ResNet-34 的编码器和基于 ResNet-34 的解码器
*   用 ICNR 初始化进行像素混洗放大
*   从预训练的 ImageNet 模型进行迁移学习
*   基于来自 VGG-16 模型的激活、像素损失和克矩阵损失的损失函数
*   区别学习率
*   渐进调整大小

这个模型或数学函数有超过 4000 万个参数或系数，允许它尝试执行超分辨率。

# 剩余网络

ResNet 是一种卷积神经网络(CNN)架构，由一系列残差块(ResBlocks)组成，如下所述，通过跳跃连接将 ResNet 与其他 CNN 区分开来。

当最初设计 ResNet 时，它以显著的优势赢得了当年的 ImageNet 竞争，因为它解决了渐变消失的问题，即随着层数的增加，训练速度变慢，准确性没有提高，甚至变得更差。正是网络跳过连接完成了这一壮举。

这些在下面的图表中显示，并在描述 ResNet 中的每个 ResBlock 时进行更详细的解释。

![](img/c1020bc2f00ab75471466dfd7e9123e6.png)

Left 34 Layer CNN, right 34 Layer ResNet CNN. Source Deep Residual Learning for Image Recognition: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

## 残余块(ResBlocks)和密集块

如果卷积网络在靠近输入的层和靠近输出的层之间包含较短的连接，则卷积网络可以训练得更深入、更准确、更有效。

如果您将损失面(模型预测的变化损失的搜索空间)可视化，这看起来就像下图中左侧图像所示的一系列山丘和山谷。最低的损失就是最低点。研究表明，一个较小的最优网络可以被忽略，即使它是一个较大网络的一部分。这是因为损失面太难导航。这意味着向模型中添加层会使预测变得更糟。

![](img/3ae27fc4e586ea68ebbc913680280a46.png)

Loss surface with and without skip connections. Source: Visualising loss space in Neural networks: [https://arxiv.org/abs/1712.09913](https://arxiv.org/abs/1712.09913)

一个非常有效的解决方案是在网络各层之间增加交叉连接，允许在需要时跳过大部分。这创建了一个损失表面，看起来像右边的图像。这对于用最佳权重训练模型以减少损失要容易得多。

![](img/3076ba578d460f1abaf23282d665406f.png)

A ResBlock within a ResNet. Source: Deep Residual Learning for Image Recognition: [https://arxiv.org/abs/1512.03385](https://arxiv.org/abs/1512.03385)

每个 ResBlock 从其输入有两个连接，一个经过一系列卷积、批量规格化和线性函数，另一个连接跳过这一系列卷积和函数。这些被称为身份连接、交叉连接或跳过连接。两个连接的张量输出相加在一起。

## 稠密连接的卷积网络和稠密块

在 ResBlock 提供张量相加的输出的情况下，这可以被改变为张量连接。随着每个交叉/跳跃连接，网络变得更加密集。然后，ResBlock 变成 DenseBlock，网络变成 DenseNet。

这允许计算跳过架构中越来越大的部分。

![](img/c6cfb5f428947f197442c42222955b38.png)

DenseBlocks within a DenseNet. Source: Densely Connected Convolutional Networks: [https://arxiv.org/pdf/1608.06993.pdf](https://arxiv.org/pdf/1608.06993.pdf)

由于串联，DenseBlocks 与其他架构相比会消耗大量内存，非常适合较小的数据集。

# u 型网

U-Net 是为生物医学图像分割开发的卷积神经网络架构。已经发现 u-网对于输出与输入大小相似并且输出需要该量的空间分辨率的任务非常有效。这使得它们非常适合于创建分割蒙版和图像处理/生成，如超分辨率。

当卷积神经网络通常与用于分类的图像一起使用时，使用一系列每次减小网格大小的两个步长的卷积，图像被获取并向下采样到一个或多个分类中。

为了能够输出与输入大小相同或更大的生成图像，需要有一个上采样路径来增加网格大小。这使得网络布局类似于 U 形，U 形网络下采样/编码器路径形成 U 形的左侧，上采样/解码器路径形成 U 形的右侧

对于上采样/解码器路径，几个转置卷积实现这一点，每个卷积在现有像素之间和周围添加像素。基本上执行下采样路径的相反过程。上采样算法的选项将在后面进一步讨论。

请注意，该模型的基于 U-Net 的架构也有交叉连接，这将在后面详细说明，这些不是原始 U-Net 架构的一部分。

![](img/e36e2f5c666fd8262fb082170e0e85c4.png)

A U-Net network architecture. Source: [http://deeplearning.net/tutorial/_images/unet.jpg](http://deeplearning.net/tutorial/_images/unet.jpg)

原来的研究可在这里:[https://arxiv.org/abs/1505.04597](https://arxiv.org/abs/1505.04597)

## 上采样/转置卷积

网络的解码器/上采样部分(U 的右手部分)中的每个上采样需要在现有像素周围以及现有像素之间添加像素，以最终达到期望的分辨率。

这个过程可以从论文“深度学习卷积算法指南”中可视化如下，其中在像素之间添加零。蓝色像素是原始的 2x2 像素扩展到 5x5 像素。在外部添加 2 个像素的填充，并在每个像素之间添加一个像素。在这个例子中，所有新像素都是零(白色)。

![](img/ec93f50278a58d2697b4e89b7f0c3b83.png)

Adding pixels around and between the pixels. Source: A guide to convolution arithmetic for deep learning: [https://arxiv.org/abs/1603.07285](https://arxiv.org/abs/1603.07285)

这可以通过使用像素的加权平均(使用双线性插值)对新像素进行一些简单的初始化来改善，否则会不必要地使模型更难学习。

在这个模型中，它使用了一种改进的方法，称为像素混洗或带有 ICNR 初始化的亚像素卷积，这导致像素之间的间隙被更有效地填充。这在论文“使用高效亚像素卷积神经网络的实时单幅图像和视频超分辨率”中有所描述。

![](img/8560e501935f12c3ff326a24350c07e1.png)

Pixel shuffle. Source: Real-Time Single Image and Video Super-Resolution Using an Efficient Sub-Pixel Convolutional Neural Network, source: [https://arxiv.org/abs/1609.05158](https://arxiv.org/abs/1609.05158)

像素混洗以因子 2 放大，使图像的每个通道中的维度加倍(在网络的该部分的当前表示中)。然后执行复制填充以在图像周围提供额外的像素。然后，执行平均池以平滑地提取特征并避免由许多超分辨率技术产生的棋盘图案。

在添加了这些新像素的表示之后，随着路径继续通过网络的解码器路径，随后的卷积改善了这些像素内的细节，然后再进行另一个步骤并使维度加倍。

## u 网和精细图像细节

当使用唯一的 U-Net 架构时，预测往往缺乏细节，为了帮助解决这个问题，可以在网络的块之间添加交叉或跳过连接。

不是像在 ResBlock 中那样每两个卷积添加一个跳过连接，而是跳过连接从下采样路径中相同大小的部分跨越到上采样路径。这些是上图中显示的灰色线条。

原始像素通过跳过连接与最终的 ResBlock 连接，以允许在知道输入到模型中的原始像素的情况下进行最终计算。这导致输入图像的所有细节都在 U-Net 的顶部，输入几乎直接映射到输出。

U-Net 块的输出被连接起来，使它们更类似于 DenseBlocks 而不是 ResBlocks。但是，有两个跨距卷积可以减小网格大小，这也有助于防止内存使用量增长过大。

# ResNet-34 编码器

ResNet-34 是一个 34 层 ResNet 架构，它被用作 U-Net(U 的左半部分)的下采样部分中的编码器。

在将 ResNet-34 编码器转换为具有交叉连接的 U-Net 的情况下，Fastai U-Net 学习器在配备有编码器架构时将自动构建 U-Net 架构的解码器端。

为了使图像生成/预测模型知道如何有效地执行其预测，如果使用预训练的模型，将大大加快训练时间。然后，该模型具有需要检测和改进的特征种类的初始知识。当照片被用作输入时，使用在 ImageNet 上预先训练的模型和权重是一个很好的开始..用于 pyTorch 的预训练 ResNet-34 可从卡格尔:[https://www.kaggle.com/pytorch/resnet34](https://www.kaggle.com/pytorch/resnet34)获得

# 损失函数

损失函数基于论文《实时风格传输和超分辨率的损失》中的研究以及 Fastai 课程(v3)中所示的改进。

本文重点研究特征损失(文中称为感知损失)。这项研究没有使用 U-Net 架构，因为当时机器学习社区还不知道它们。

![](img/3733f02dc75e5861c6830c2067b5feb6.png)

Source: Convolutional Neural Network (CNN) Perceptual Losses for Real-Time Style Transfer and Super-Resolution: [https://arxiv.org/abs/1603.08155](https://arxiv.org/abs/1603.08155)

此处使用的模型使用与论文相似的损失函数进行训练，使用 VGG-16，但也结合了像素均方误差损失和克矩阵损失。Fastai 团队发现这非常有效。

## VGG-16

VGG 是 2014 年设计的另一个 CNN 架构，16 层版本用于训练该模型的损失函数。

![](img/5935cad94bff00a4018a595b000abce4.png)

VGG-16 Network Architecture. Source: [https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png](https://neurohive.io/wp-content/uploads/2018/11/vgg16-1-e1542731207177.png)

VGG 模式。在 ImageNet 上预先训练的网络用于评估发电机模型的损耗。通常这将被用作分类器来告诉你图像是什么，例如这是一个人，一只狗还是一只猫。

VGG 模型的头部被忽略，损失函数使用网络主干中的中间激活，其代表特征检测。网络的头部和主干将在后面的培训部分进一步介绍。

![](img/e3c70cdb912861b9425f714e4791362e.png)

Different layers in VGG-16\. Source: [https://neurohive.io/wp-content/uploads/2018/11/vgg16.png](https://neurohive.io/wp-content/uploads/2018/11/vgg16.png)

这些激活可以通过查看 VGG 模型找到所有的最大池层。这些是检测网格大小变化和特征的地方。

在下图中可以看到可视化各种图像激活的热图。这显示了在网络的不同层中检测到的各种特征的例子。

![](img/51a4ca4bc3ae65ebea26753a392b57b3.png)

Visualisation of feature activations in CNNs. Source: page 4 of [https://arxiv.org/pdf/1311.2901.pdf](https://arxiv.org/pdf/1311.2901.pdf)

该超分辨率模型的训练使用基于 VGG 模型激活的损失函数。损失函数在整个训练过程中保持固定，不像 GAN 的关键部分。

特征图有 256 个 28×28 的通道，用于检测毛发、眼球、翅膀和类型材料等特征以及许多其他类型的特征。使用基本损失的均方误差或最小绝对误差(L1)误差来比较(目标)原始图像和生成图像在同一层的激活。这些是特征损失。该误差函数使用 L1 误差。

这使得损失函数能够了解目标地面真实影像中的特征，并评估模型预测的特征与这些特征的匹配程度，而不仅仅是比较像素差异。

# 培训详情

训练过程从如上所述的模型开始:使用基于在 ImageNet 上预训练的 VGG-16 架构的损失函数结合像素损失和 gram 矩阵，在 ImageNet 上预训练的基于 ResNet-34 架构的 U-Net。

## 培训用数据

幸运的是，在大多数应用程序中，可以创建几乎无限量的数据作为训练集。如果采集了一组高分辨率图像，可以将这些图像编码/调整大小为较小的图像，这样我们就有了具有低分辨率和高分辨率图像对的训练集。然后，我们的模型预测可用于评估高分辨率图像。

低分辨率图像最初是一半尺寸的目标/地面真实图像的副本。然后，使用双线性变换对低分辨率图像进行初始放大，以使其与目标图像具有相同的尺寸，从而输入到基于 U-Net 的模型中。

在这种创建训练数据的方法中采取的操作是模型学习适应的操作(颠倒该过程)。

可以通过以下方式进一步扩充训练数据:

*   在一定范围内随机降低图像质量
*   随机选择农作物
*   水平翻转图像
*   调整图像的照明
*   添加透视扭曲
*   随机添加噪声
*   在图像上随机打孔
*   随机添加覆盖的文本或符号

下面的图像是数据扩充的一个例子，所有这些都是从同一个源图像生成的:

![](img/8522b367894bddd60e248df18a4befcf.png)

Example of data augmentation

将每个图像的质量降低和噪声改变为随机的改进了结果模型，允许它学习如何改进所有这些不同形式的图像退化，并更好地概括。

## 功能和质量改进

基于 U-Net 的模型增强了放大图像中的细节和特征，通过包含大约 4000 万个参数的函数生成了改进的图像。

## 训练模特的头部和骨干

这里使用的三种方法特别有助于训练过程。这些是渐进调整大小、冻结然后解冻主干中权重的梯度下降更新和区别学习率。

该模型的架构分为两部分，主干和头部。

主干是 U-Net 的左侧部分，是基于 ResNet-34 的网络的编码器/下采样部分。头部是 U-Net 的右侧部分，即网络的解码器/上采样部分。

主干已经基于在 ImageNet 上训练的 ResNet34 预先训练了权重，这就是迁移学习。

头部需要训练其权重，因为这些层的权重被随机初始化以产生期望的最终输出。

在最开始，网络的输出基本上是像素的随机变化，而不是使用 ICNR 初始化的像素混洗子卷积，其被用作网络的解码器/上采样路径中每个升级的第一步。

一旦经过训练，骨架顶部的头部允许模型学习用骨架中预先训练的知识做一些不同的事情。

**冻住脊梁，练好脑袋**

网络主干中的权重被冻结，因此最初只训练头部中的权重。

学习率查找器运行 100 次迭代，并绘制损失与学习率的关系图，选择最陡斜率附近朝向最小损失的点作为最大学习率。可选地，可以使用比最低点小 10 倍的速率来查看是否表现得更好。

![](img/85c51d9596fccd72094332ff28ba8778.png)

Learning rate against loss, optimal slope with backbone frozen

“适合一个周期”政策用于改变学习速度和动力，在 Leslie Smith 的论文中有详细描述

[https://arxiv.org/pdf/1803.09820.pdf](https://arxiv.org/pdf/1803.09820.pdf)和西尔万·古格的[https://sgugger.github.io/the-1cycle-policy.html](https://sgugger.github.io/the-1cycle-policy.html)

## 渐进调整大小

首先在大量较小的映像上进行训练，然后扩展网络和训练映像，这样会更快。将图像从 64px 乘以 64px 放大并改善为 128px 乘以 128px 比在更大的图像上执行该操作要容易得多，在更大的数据集上也要快得多。这被称为渐进式调整大小，它也有助于模型更好地概括，因为它看到更多不同的图像，不太可能过度拟合。

这种渐进式调整大小的方法是基于 Nvidia 与 progressive GANs 的出色研究:[https://research . Nvidia . com/sites/default/files/pubs/2017-10 _ Progressive-Growing-of/karras 2018 iclr-paper . pdf](https://research.nvidia.com/sites/default/files/pubs/2017-10_Progressive-Growing-of/karras2018iclr-paper.pdf)。这也是 Fastai 用来在 ImageNet 上击败科技巨头的方法:[https://www.fast.ai/2018/08/10/fastai-diu-imagenet/](https://www.fast.ai/2018/08/10/fastai-diu-imagenet/)

该过程是在较大的批次中用小图像进行训练，然后一旦损失降低到可接受的水平，则创建新的模型，该模型接受较大的图像，从在较小的图像上训练的模型转移学习。

随着训练图像大小的增加，批次大小必须减小以避免耗尽内存，因为每个批次包含更大的图像，每个图像中的像素是四倍。

请注意，输入图像中的缺陷是随机添加的，以提高模型的恢复属性，并帮助它更好地概括。

从训练集中分离出来的验证集的示例以一些渐进的大小显示在这里:

在每个图像尺寸下，执行 10 个时期的一个循环的训练。这是冷冻的骨干重物。

图像尺寸被加倍，并且对于通过网络的较大图像的路径，用额外的网格尺寸来更新模型。重要的是要注意砝码的数量不会改变。

步骤 1:从 32 像素乘 32 像素放大到 64 像素乘 64 像素。使用 1e-2 的学习率。

![](img/0919ddf76bbdffbae16b051a3f07dce2.png)

Super resolution to 64px by 64px on a 32px by 32px image from the validation set. Left low resolution input, middle super resolution models prediction, right target/ground truth.

步骤 2:从 64 像素乘 64 像素放大到 128 像素乘 128 像素。使用 2e-2 的学习速率。

![](img/0afc69822c14f909aa584bf34511e9c1.png)

Super resolution to 128px by 128px on a 64px by 64px image from the validation set. Left low resolution input, middle super resolution models prediction, right target/ground truth

步骤 3:从 128 像素乘 128 像素放大到 256 像素乘 256 像素。使用了 3e-3 和 1e-3 之间的区别学习率。

![](img/30c726ce2654d2d9f6401cdeaa9ba5e9.png)

Super resolution to 256px by 256px on a 128px by 128px image from the validation set. Left low resolution input, middle super resolution models prediction, right target/ground truth

步骤 4:从 256 像素乘 256 像素放大到 512 像素乘 512 像素。使用 1e-3 和之间的区别学习率。

![](img/47a0722899e73e7d9458094a16c01b68.png)

Super resolution to 512px by 512px on a 256px by 256px image from the validation set. Left low resolution input, middle super resolution models prediction, right target/ground truth

## 解冻主干

主干被分成两层组，头部是第三层组。

然后，整个模型的权重被解冻，并且该模型用判别学习率来训练。这些学习率在第一层组中要小得多，然后在第二层组中增加，最后一层组在头部再次增加。

在脊柱和头部解冻的情况下，再次运行学习率查找器。

![](img/2c9a2bfed167d2c447938f506f383edc.png)

Learning rate against loss with backbone and head unfrozen

使用 1e-6 和 1e-4 之间的区别学习率。头部的学习速率仍然比先前的学习周期低一个数量级，在先前的学习周期中，只有头部被解冻。这允许对模型进行微调，而不会有损失已有精度的风险。这被称为学习率退火，当我们接近最佳损失时，学习率降低。

在更大的输入图像上继续训练将提高超分辨率的质量，但是批量大小必须保持缩小以适应内存限制，并且训练时间增加，并且达到了我的训练基础设施的极限。

所有训练都是在 Nvidia Tesla K80 GPU 上进行的，内存为 12GB，从开始到结束不到 12 个小时，逐步调整大小。

# 结果

训练的渐进调整大小部分中的以上图像显示了基于深度学习的超分辨率在改善细节、去除水印、缺陷和赋予缺失的细节方面是多么有效。

接下来的三个基于来自 Div2K 数据集的图像的图像预测都通过相同的训练模型对其执行了超分辨率，这表明深度学习超分辨率模型可能能够普遍应用。

注意:这些来自实际的 Div2K 训练集，尽管该集被分成我自己的训练和验证数据集，并且模型在训练期间没有看到这些图像。后面还有来自实际 Div2K 验证集的更多示例。

左:256 x 256 像素输入。中间:来自模型的 512 x 512 预测。右:512 x 512 像素地面真实目标。看着火车前面的通风口，细节改进很清楚，非常接近地面真实目标。

![](img/b83ab22ad93bcfa09a35eb4daef6f470.png)

256 by 256 pixel super resolution to 512 by 512 pixel image, example 1

左:256 x 256 像素输入。中间:来自模型的 512 x 512 预测。右:512 x 512 像素地面真实目标。下面这个图像预测中的特征改进是相当惊人的。在我早期的训练尝试中，我几乎得出结论，人类特征的超分辨率将是一项过于复杂的任务。

![](img/96c81a6f27b7154e845049e26dd39dc7.png)

256 by 256 pixel super resolution to 512 by 512 pixel image, example 2

左:256 x 256 像素输入。中间:来自模型的 512 x 512 预测。右:512 x 512 像素地面真实目标。请注意白色的“安全出口”文字和镶板线是如何改进的。

![](img/ebf6e4ed7d47d39b8b169db27780d920.png)

256 by 256 pixel super resolution to 512 by 512 pixel image, example 3

## Div2K 验证集上的超分辨率

来自官方 Div2K 验证集的超分辨率示例。这里有 PDF 版本:[https://drive.google.com/open?id = 1 ylselpp _ _ emdywihpmlhw 4 fxjn _ LybkQ](https://drive.google.com/open?id=1ylselPp__emdYwIHpMlhw4fxjN_LybkQ)

![](img/1e26b455046558d3e05a5a2825edbf7b.png)

Model prediction comparison on the Div2K validation dataset

## 牛津 102 Flowers 数据集上的超分辨率

超分辨率来自于花的图像数据集上的单独训练模型，我认为这非常出色，许多模型预测实际上看起来比在验证集上真正执行超分辨率的地面事实更清晰(训练期间看不到的图像)。

![](img/9221b974cc5ad8e6f1a1d3dca13f45ef.png)

Validation results upscaling images from the Oxford 102 Flowers dataset consisting of 102 flower categories

## 牛津-IIIT Pet 数据集上的超分辨率

下面的例子来自一个单独的训练模型，放大了狗的低分辨率图像，令人印象深刻，同样来自验证集，创建了更精细的皮毛细节，锐化了眼睛和鼻子，并真正改善了图像中的特征。大多数放大的图像接近地面真实情况，当然比双线性放大的图像好得多。

![](img/897bab4bc49eaa0385d6a53262e39e25.png)

Validation results upscaling images from Oxford-IIIT Pet dataset, a 37 category pet dataset with roughly 200 images for each class.

我相信这些结果是令人印象深刻的，模型必须发展出一种“知识”,即照片/图像的原始主题中的一组像素必须是什么。

它知道某些区域是模糊的，并且知道重建模糊的背景。

如果模型在损失函数的特征激活上表现不佳，它就无法做到这一点。实际上，该模型已经逆向工程了匹配这些像素的特征，以匹配损失函数中的激活。

# 限制

对于要由模型学习的恢复类型，它必须作为要解决的问题存在于训练数据中。当在一个训练过的模型的输入图像上打孔，而这个模型不知道如何处理它们，就让它们保持不变。

需要在图像上产生幻觉的特征，或者至少是相似的特征，必须存在于训练集中。如果模型是在动物上训练的，那么模型不太可能在完全不同的数据集类别上表现良好，例如房间内部或花。

在特写人脸上训练的模型的超分辨率结果不是特别令人信服，尽管在 Div2K 训练集中的一些例子上确实看到了特征上的良好改进。特别是在 X4 超分辨率中，虽然特征比最近邻插值更加锐化，但是特征呈现出几乎是绘制的/艺术的效果。对于分辨率非常低的图像或有很多压缩伪像的图像，这可能仍然是更好的选择。这是我打算继续探索的一个领域。

# 结论

使用损失函数训练的基于 U-Net 深度学习的超分辨率可以很好地执行超分辨率，包括:

*   将低分辨率图像放大到更高分辨率的图像
*   提高图像质量并保持分辨率
*   移除水印
*   从图像中消除损坏
*   移除 JPEG 和其他压缩伪像
*   给灰度图像着色(另一项正在进行的工作)

对于要由模型学习的恢复类型，它必须作为要解决的问题存在于训练数据中。在经过训练的模型的输入图像中打孔，模型不知道如何处理它们，就让它们保持不变，而当打孔被添加到训练数据中时，经过训练的模型可以很好地恢复它们。

这里展示的所有图像超分辨率的例子都是来自我训练的模型的预测。

下面有五个例子，我认为模型的预测(中间)与目标(右边)一样好或非常接近，目标是来自验证集的原始地面真实图像。该模型适用于动物特征，如毛皮和眼睛，眼睛是一个非常困难的任务，以锐化和增强。

![](img/8e1cdb1befc8ac9785344d612cd530a4.png)

Super resolution concluding example 1

![](img/28042dbf2723467e1ff9cc183b68260d.png)

Super resolution concluding example 2

![](img/615f91222985a9614be5494b4355e866.png)

Super resolution concluding example 3

![](img/e4d36fd156a0672f4986a1c92640ebc5.png)

Super resolution concluding example 4

![](img/967d4cf7880d1aac1cb4af0f7bdbb673.png)

Super resolution concluding example 5

还有最后一个来自验证集的例子，在我看来，模型的预测(中间)比目标(右边)更好，即来自验证集的原始地面真实图像。

![](img/de208165e7cf78f6489ebe42694f4b4f.png)

Super resolution concluding example 6, model prediction possibly better than the ground truth target?

# 后续步骤

我热衷于将这些技术应用于不同主题的图像和不同的领域。如果你认为这些超分辨率技术可以帮助你的行业或项目，那么请联系我们。

我计划将这个模型转移到一个生产 web 应用程序中，然后可能转移到一个移动 web 应用程序中。

我正在对 Image Net 数据集的较大子集进行训练，该数据集包含许多类别，以产生一个有效的通用超分辨率模型，该模型对任何类别的图像都表现良好。我还在训练我在这里训练的相同数据集的灰度版本，其中模型正在给图像着色。

我计划尝试像 ResNet-50 这样的模型架构，以及一个带有初始主干的 ResNet 主干。

# 伦理问题

通过产生在诸如安全镜头、航空摄影或类似的类别中不存在的幻觉细节，然后从低分辨率图像生成图像可能会使其远离原始的真实主题。

想象一下，如果面部特征被细微地改变，但足以通过面部识别来识别一个实际上不在那里的人，或者航拍照片被改变，足以通过另一种算法将一座建筑物识别为另一个样子。多样化的训练数据应该有助于避免这种情况，尽管随着超分辨率方法的改进，这是一个问题，因为缺乏机器学习研究社区历史上使用的多样化训练数据。

# 法斯泰

感谢 Fastai 团队，没有你们的课程和软件库，我怀疑我是否能够进行这些实验并了解这些技术。