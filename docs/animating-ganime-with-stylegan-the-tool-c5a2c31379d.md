# 使用 StyleGAN 制作 gAnime 动画:工具

> 原文：<https://towardsdatascience.com/animating-ganime-with-stylegan-the-tool-c5a2c31379d?source=collection_archive---------9----------------------->

## 开源 GAN 研究工具的深入教程

![](img/22602d300a0a5129387b5471fffc06d0.png)

Visualization of feature map 158 at a layer with resolution 64x64

# 0.前言

这是我个人项目中的一个研究工具的教程/技术博客。虽然博客的很大一部分假设您在阅读时可以访问该工具，但我试图包括足够多的截图，即使您没有时间亲自尝试，也应该清楚它是如何工作的。

在本教程中，我们将与一个经过训练的 StyleGAN 模型进行交互，以创建如下动画(的帧):

![](img/83e89693b35ddbb7762d462ea976e4ae.png)

Spatially isolated animation of hair, mouth, and eyes

在上面的动画中，改变嘴、眼睛和头发的变换大多是独立的。这通常比用 GANs 制作说话动画的其他方法(据我所知)更可取，这可能会导致脱发等副作用:

![](img/d1cf44a18e6d954d7b29daae735bfecf.png)

Another animation we’ll create to demonstrate how changes in the ‘mouth’ attribute can influence other parts of an image. Note the hair thickening and thinning along the edges.

我们还将通过使用网络中各层的特征图来构建简单的启发式面部特征检测器:

![](img/f4125a08708264b29c294aee99568e9d.png)![](img/a629365f2010d8d92db108955a6ee14b.png)![](img/70a7990a006b59012b98f23f518e9a42.png)

Using feature maps at various layers to detect mouths, eyes, and cheeks

然后，这些检测器可用于自动进行有意义的修改，以便隔离图像的各个部分:

![](img/f6cbdef6c70032c4845b27fea8c2cf9e.png)

These images generated were in a single batch without human intervention or a labeled dataset

这些步骤都不需要训练集的标签，但是需要一些手工操作。

# 1.介绍

您可以从以下链接之一下载该工具的编译版本:

 [## GanStudio_x64_v1.zip

### 编辑描述

drive.google.com](https://drive.google.com/file/d/1cv2SiWQKtlC-XCAeAiGHh2C8XCAZe9xd/view?usp=sharing) [](https://mega.nz/#!VCIRyIRI!t_g2OQYkuqtPAdd5wgsSHUEYhYI47ip84jydGZMI-bg) [## 非常

### MEGA 提供免费云存储，具有方便、强大的永远在线隐私功能。立即申领您的免费 50GB

mega .新西兰](https://mega.nz/#!VCIRyIRI!t_g2OQYkuqtPAdd5wgsSHUEYhYI47ip84jydGZMI-bg) 

sha 256:ec2a 11185290 c 031 b 57 c 51 EDB 08 BF 786298201 EB 36 f 801 b 26552684 c 43 BD 69 c 4

它配备了一个在动画数据集(T2 的许多项目都基于该数据集)上训练的模型。不幸的是，由于数据集的性质，缺乏性别多样性。我目前正在尝试训练一个模型，以产生更高质量的男性形象，但这需要一段时间才能完成。数据集还包含 NSFW 影像，虽然我生成了数千幅影像，但从未遇到过任何 NSFW，我并没有检查训练集中的每一幅影像(对早期图层中的要素地图进行大规模修改可能会增加风险)。如果你遇到问题，你可以在 https://github.com/nolan-dev/GANInterface[提出问题，我会尽力回应。](https://github.com/nolan-dev/GANInterface)

这个博客有两个部分:基础教程和高级教程。基础教程演示了如何使用该工具，并且不需要太多的技术知识就可以完成(尽管它为感兴趣的人提供了一些技术解释)。高级教程演示了如何定制工具，并且更具技术性——例如，您应该熟悉卷积神经网络中的[特征映射](/applied-deep-learning-part-4-convolutional-neural-networks-584bc134c1e2)。

我在以前的博客中介绍了这个工具，并分享了源代码，但是让它工作起来很复杂，需要一个用我的自定义 StyleGAN 实现训练的模型。我希望提供一个工具的编译版本和一个预先训练好的模型能让它更容易试用。该工具是上一篇博客讨论的更大项目(包括 StyleGAN 的重新实现)的一部分，但是阅读它并不是这篇博客的先决条件。如果你感兴趣，这里有一个链接:

[](/animating-ganime-with-stylegan-part-1-4cf764578e) [## 用 StyleGAN 制作 gAnime 动画:第 1 部分

### 介绍与创成式模型交互的工具

towardsdatascience.com](/animating-ganime-with-stylegan-part-1-4cf764578e) 

由于这是一个研究工具，我一直在定期添加和减少功能，以更好地了解模型如何工作以及与它交互的最佳方式。有许多次要功能，但主要功能包括:

*   修改生成图像的潜在向量，以便在图像之间进行插值，表达某些特征，并在质量和变化之间进行权衡(截断技巧)
*   修改特征地图以改变图像中的特定位置:这可用于动画
*   读取和处理特征地图，以自动检测有意义的特征
*   通过创建批处理作业来自动化上述所有操作

和上一篇博客一样，我写这篇博客的目的是获得其他人对这个话题的看法，详细描述我在这个项目上的工作经历，并接受建设性的批评/纠正。本博客的教程格式旨在缓解该工具不发达的 UI，并使其有可能在不处理混乱的源代码的情况下使用。不幸的是，它仅适用于 Windows，但它已在免费的 Windows AWS 实例上进行了测试(Microsoft Windows Server 2019 Base，但图像生成会很慢)。

# 2.基础教程

在我们开始之前，有一个小提示:我在写这篇文章的时候修改了这个工具，所以一些截图与当前版本略有不同，但是一切都在大致相同的地方。

下载并打开上面链接的 zip 文件后，你会看到几个文件/文件夹:

![](img/c78ba85ee63e836df50831fe185b693e.png)

我将在高级部分(3.3)更详细地解释其中的一些，但是此时唯一重要的文件是 GanStudio.exe。当您运行它时，单击免责声明的 ok(希望不要出现任何错误)，您将看到如下内容:

![](img/ae10b8df3ec263351eeb95419667d8ed.png)

由于 UI 的复杂性，在本教程中我第一次引用该工具的一部分时，我会有一个附近的截图，相关部分用红色标出。许多 UI 元素都有工具提示。

**设置:**

使用这个工具涉及到与 windows 资源管理器的交互，用“大”或“超大”图标查看生成的文件是最容易的。通过右键单击资源管理器窗口并选择“查看”来选择其中之一:

![](img/b8a8d4d0848721c0f0a07b2098cf729e.png)

在许多情况下，按修改日期对图像进行排序也很有帮助，这也可以通过右键菜单实现:

![](img/a3e3b7ff6b41ff0b90632de7b904f47b.png)

**生成新图像**

要测试图像生成，请单击生成新图像(3)。这将产生一个新的潜在代码，并显示它的图像。注意生成第一幅图像通常比生成后续图像需要更长的时间。

![](img/8bf31849de5a97f3ab4a570c4eaf18a6.png)

该图像是随机创建的，但被插值以接近“平均”图像。这导致更高质量的图像，但减少了变化。如果你把质量滑块(1)放在上图的同一个地方，你的图像很可能是相似的:一个棕色头发，紫色和/或蓝色眼睛的女孩。

**加载图像**

为了使本教程中的图像与该工具将产生的图像保持一致，我提供了一个示例图像。点击“导入图像”(如上，2)。这将在“肖像”目录中创建一个打开文件对话框。导航至“肖像”上方的目录，并选择“教程文件”:

![](img/363f6491af81a81ea68d39958562916f.png)

双击 sample_01.png 进行加载。您生成的所有图像都保存在“肖像”文件夹中，您可以使用这种方法再次加载它们。

GANs 通常不能加载任意图像，但是这个工具会将生成图像的潜在代码附加到它写入磁盘的每个 PNG 文件中。“导入图像”按钮读取写入您选择的图像的潜在代码。只要工具加载了生成图像的模型，它就能够重新创建图像。

**修改属性**

![](img/8feef2f04a8b019a9a8e67ee37a48d12.png)

Hover over attribute label that have been cut off to see full name

要开始修改属性，请选择“Attributes_0”选项卡(如上)。属性包括头发/眼睛颜色、背景强度、配饰和嘴部状态(微笑/张开)。向右移动对应于属性的滑块将增加该属性对图像的影响，向左移动将减少所述影响。他们中的一些人比其他人工作得更好。选择位置后，按“更新此图像”(如上)。以下是一些例子:

![](img/f020502481b7f90dc8265f9a39e6180e.png)

Left to right/top to bottom: Positive open_mouth, positive skin_tone, negative background and negative black_hair, positive blonde_hair and negative smile

以这种方式修改属性的一个缺点是它们并不总是空间隔离的；修改只影响图像一部分的属性也会影响其他部分。如果我们想要创建动画，这尤其成问题。要查看实际问题，请执行以下步骤(以下截图供参考):

1.  向下滚动到“张开嘴”滑块
2.  把它移到右边
3.  按“更新此图像”。嘴现在应该微微张开
4.  选择*批次- >属性- >光谱*
5.  选择“确定”以生成 5 幅图像，选择“否”以“滑过 0？”提示。

![](img/eb70ffe2c003f55a320008bb0bbed97e.png)

这将产生 5 个图像，其中“张嘴”属性从 0 移动到滑块上的选定位置:

![](img/788bd145851a68f8ae1070b0b95f425f.png)

将这些内容放入 gif 生成器会生成以下动画:

![](img/d1cf44a18e6d954d7b29daae735bfecf.png)

Frame order based on the first number in the filename: 0, 1, 2, 3, 4, 3, 2, 1

正如你所看到的，即使我们只选择了一个与嘴相关的属性，整个图像的特征都会改变。

**修改具体位置**

在这一节中，我们将只对嘴进行修改，而不改变其他特征。不幸的是，此时使用导入图像按钮导入的图像将不会反映这里所做的更改。

重复*加载图像*一节中的说明，回到基础图像(或重置属性滑块并更新)。我们使用“空间”选项卡(如下)来修改图像的孤立部分。

![](img/59421c4a47f8c664bfe02806d34655ea.png)

UI 很复杂，但是对于这一部分，我们将只使用几个部分。我们需要做的第一件事是指出我们想要改变图像的哪一部分:

1.  选择“嘴巴张开”选项卡。
2.  放开之前，单击、按住并拖动光标穿过嘴巴

这就在嘴的周围做了一个选择，并确保我们对图像的改变只会影响所选的区域。

![](img/c1b5dcb0572e13ba8fa28dafe0a976df.png)

Click on locations within these squares to create them on your image, or click and drag across this portion of the image

这将产生一个浅绿色的正方形，除非选择了“在可视化中用绿色替换蓝色”。我将在本教程中选择这个选项来提高可见度，并希望当我们开始处理表示负面影响的红框时，对色盲更加友好。

![](img/2483d0a767d1b183cf6887743ce3964f.png)

如果您错误地选择了一个位置，您可以在选择时按住“control”来删除方块:

![](img/70fa14f9c5cae40aec124b371558ca36.png)

Remove an undesired selection. Click and drag with ctrl held to remove multiple selections.

以下是您可以在图像上“绘制”的所有方式。其中一些现在还不需要，但以后会有用:

1.  左键单击产生一个大小取决于所选分辨率的框(稍后将详细介绍)。该框表示在该位置的积极影响，根据您的设置，该框将为绿色或蓝色。
2.  右键单击在该位置产生一个具有负面影响的红色框
3.  按住 Ctrl 键并单击可擦除某个位置的方框
4.  执行上述任一操作时，单击并拖动鼠标以绘制一个大矩形区域
5.  如果未选中“无选择重叠”,您可以多次选择同一位置以增加选择的幅度(正或负)。这表现为更高的颜色强度和更厚的方框。

选择嘴部后，向右移动“嘴部张开”标签下方的滑块，直到左下方的数字大约为 100。此滑块是“要素地图乘数滑块”，当它向右(积极影响)或向左(消极影响)移动时，会对活动选项卡产生指数影响。将滑块左下方的数字设置为 100 左右，选择“更新此图像”:

![](img/61432562e8f1c0abd7a4ad356bc8cc46.png)

这应该会产生以下图像:

![](img/851d648f5bc2291183fac3350cd828ef.png)

正如标签的名字所暗示的，这开了口。让我们尝试用这种方法制作动画。select*Batch->Fmap->combinator IC*(我会在高级教程中详述为什么这么叫):

![](img/15108a325c5345d2393e5623a9686eef.png)

为要生成的图像选择 5:

![](img/7146a82556b8b4d2ccb4dfae25ca61fa.png)

选择 0 作为起点。该批将由 5 个图像组成，滑块在起点和终点之间有规则的间距(0，20，40，80，100)。因为在此图像中嘴默认是闭合的，所以起点 0(无影响)表示闭合。

![](img/29ea5882affa01b4785052a8b6599759.png)

这将产生比属性方法具有更少空间纠缠的 5 个图像:

![](img/69e9ee5e770b1153b406a4583b3390a6.png)

gif 生成器产生以下内容:

![](img/0853a37683ca5a420655ef9f498ed0f3.png)

其他选项卡也可以使用相同的过程，并且可以组合不同的选项卡。如果你使用的是*组合*批处理生成器，你需要将除激活标签之外的所有标签的乘数条保持为零，以避免产生多个标签的组合。这可以通过在更改活动标签的乘数之前按下“全部设置为零”来完成:

![](img/4a943098274c1272582048d2fc41f86b.png)

这里列出了一些可能的变化。请注意，使用大乘数很可能会产生奇怪的伪像:

1.  红色或蓝色的眼睛

![](img/8506d0039ca530937bea420309499e0f.png)

起点和终点如下:

![](img/757f2624467d8221995b2e72adbd540e.png)

The dialog actually says Start Point, this is an old screenshot. The start point corresponds to the first image that gets generated in the batch, and the end point corresponds to the last.

生产:

![](img/c959b5cc41052f67474ad8caf9ab0131.png)

2.脸红

设置:

![](img/f9d32e523a3cb481c24343469b571243.png)

If the screenshot and the tool disagree, believe the tool. This is an old screenshot (should say Start Point instead of End Point)

动画:

![](img/0b4736d55b54c11bbff6e686e78a5b31.png)

3.束发带

设置:

![](img/6d6889571167e8b5c1b4f0709abb29a4.png)

I actually had the variable for this prompt called ‘startValue’ originally, don’t know why I made the prompt say End Point. You may have guessed this is an old screenshot

动画:

![](img/f8e35f4a3ccdb75737c228e886f33a50.png)

4.头发

设置:

![](img/6b8b8f19a63c8cbf597b8fa1c93697b9.png)

You’d think it would make sense to start at the low number and end at the high number, but actually this is an old screenshot and End Point should be Start Point. It will start at 100 and shift down to -257

动画:

![](img/5d6f4a685ab31520686f583e3b513027.png)

These settings slightly influence the mouth because the effective receptive field of convolutions late in the network cover a lot of spatial locations in early layers.

5.ice_is_nice(？？？)

设置:

![](img/859d05bd6ef6f5cd6b3b0e7b9dc77bc2.png)

We’re selecting the entire image here. For scenarios like this, I’d advocate for the click and drag method over clicking 4096 times with extreme precision. Also: it should say Start Point instead of End Point

动画:

![](img/af85abe62d9c5855d83a7582c592abe7.png)

Example of a modification late in the network: details are influenced, but overall structure stays the same

**其他可以尝试的东西:**

*   批量生成新图像，质量和属性栏位于不同位置:

![](img/0224fd8b7923d97a24d7a5cf0c822790.png)

*   使用“设置为基础图像”(如下)使质量条在当前图像(而不是平均图像)和新图像之间插值。当与 *Batch- > New latents* (上图)结合使用时，这对于查找与当前图像略有不同的新图像非常有用。

![](img/c9464ccc9881b8cfade1d5b70996164e.png)

*   使用“切换上一张”在当前图像和上一张图像之间快速切换，以检查更改。

![](img/a5a1686c4078ba0387b402572a9a5109.png)

Focus on the red box, not the blue box around ‘Toggle Show Selection’ which I clicked earlier. At the moment toggle show selection doesn’t even work.

*   使用 *Misc- >在两幅图像之间插值*生成两幅现有图像之间的图像光谱。

![](img/61dde664b6cc652a67bd1fe67c5b3707.png)

# 3.高级教程

本节假设您对卷积神经网络和 StyleGAN 有所了解。

## 3.1 寻找新功能

“空间”部分中的选项卡(嘴巴张开、发带等)对应于添加到特定图层的特定要素地图的值。尝试选择 mouth_open 选项卡。在选项卡上方的组合框中，它应该显示分辨率:16x16。StyleGAN 中的早期图层具有低分辨率的要素地图，而后期图层具有高分辨率的要素地图(分辨率通常会翻倍)。由于我们生成的图像是 256x256 像素，对应于 16x16 的层在网络中处于早期。要查看 mouth_open 选项卡修改了哪些特征映射，请在选中“过滤零点”的情况下按“查看所有 Fmap Mults ”,然后选择“特征映射输入”选项卡:

![](img/f5d08158f831dd34e1ed527e6b5689c1.png)

这意味着通过单击在图像中选择的空间位置会乘以-2，然后乘以要素地图乘数滑块，结果会添加到分辨率为 16x16 的图层上的要素地图 33 中。

一些选项卡会影响多个要素地图:

![](img/51e38bd3c927676387017f673b318764.png)

The feature maps influenced by the ‘hairband’ tab, viewed by clicking on ‘View All Fmap Mults’

我通过摆弄这个工具手动找到了这些乘数。我用了两种方法:

1.  为了修改图像中的现有属性(例如，嘴)，我通过修改不同的特征图并查看哪一个产生了期望的结果来使用试错法。
2.  要添加一个属性(发带)，我会查看当该属性存在时哪些特征地图是活动的。

在接下来的两节中，我将介绍这些方法的例子。

**方法一:修改嘴**

假设网络可以生成嘴巴张开和闭合的图像(这是欺骗鉴别者所必需的),并且每一层的特征图都是最终图像的表示，那么修改特征图可以用于张开或闭合嘴巴是有意义的。但是，不能保证修改单个要素地图会导致有意义的变化-我们可能需要修改许多要素地图的组合才能获得所需的结果。也就是说，单一要素地图更容易使用(至少在当前工具下)，因此查看每个要素地图如何影响图像可以作为一个起点。

以下是我如何找到特征图 33 来张开/闭上嘴巴。首先，我用“添加标签”按钮添加了一个 16x16 的标签(如下)。我选择这个分辨率是因为它能产生合理的嘴巴大小的盒子。较小的分辨率将改变嘴以外的区域，而较大的分辨率通常导致比张开或闭合嘴更精细的变化(此时分辨率的选择是启发式的)。通过再次单击“查看所有 Fmap Mults ”,我们看到没有为新选项卡设置任何功能图。然后我将滑块移动到 190 左右，这也是一个基于过去模型经验的启发式决定。最后，和我们之前做的一样，我选择了包含嘴的两个盒子。

![](img/fb1c3d7be073462207484b1f97e7f2da.png)

Adding a new tab, which initially does not influence any feature maps

然后，我选择*批处理- > Fmap - >轴对齐*，并选择 512 张图像进行生成。

![](img/fcd90fd07c077718e113001d9f18d23c.png)

这实际上会产生 1024 个图像，因为对于每个特征图，它会将乘数栏中指定的值(在本例中为 190)添加到图像中标记的空间位置(嘴)。批量生成会弹出一个窗口，显示已经生成的图像数量，并允许您中断该过程。点击“生成图像”旁边的计数器，打开它们被写入的目录:

![](img/fb205e5b52a00328868a940706bc3f6b.png)![](img/b54aae0f75fae3fb11ceef199a41e77a.png)

The number prepended to the file names is the feature map that was modified

以‘33 _ n _ sample’(n 代表负)开头的样本明显有开口，而‘33 _ p _ sample’没有。这意味着当从嘴周围的特征图 33 中减去 190 时，嘴张开了。

我使用“设置 Fmap”框(如下)将特征映射 33 设置为-1。这使得向右移动滑块将打开嘴巴(这感觉比将特征地图 33 设置为 1 并使标签名为“mouth_close”)更直观，我使用“重命名标签”按钮将标签重命名为 mouth_open。重命名选项卡旁边的保存选项卡按钮可用于保存选项卡。

![](img/e65706af0888496c4d5a4f002e17b08a.png)

**方法二:加一个发带**

这种方法依赖于找到具有所需属性的现有图像。这需要一个图像库来处理，可以用 *Batch- > New latents* 生成。在这种情况下，我通常会将质量条移过中间一点，以确保有合理的变化量。

![](img/bcbe44d14be45ece433c2a6f38397bf7.png)

可能需要几百个样本才能得到几个带发带的样本。我在 tutorial_files 目录中添加了一个，我将在本教程中加载它(sample_02.png)。加载之前，创建一个 16x16 的选项卡，并确保“Fmaps To Get”设置为“All”或“Current Tab”(如下)。创建新图像时，这些选项从网络获取并存储额外的输出:当前选项卡或所有选项卡的特征映射。这可能会减慢速度，所以它不是默认选项(而且在撰写本文时，它有 0.5%的机会导致崩溃，与大批量相关)。

![](img/f1ae9f22b71f502e21d6f502446404e9.png)

然后，执行以下操作:

1.  选择发带周围的框
2.  在“添加视图特征映射按钮”下，选择“按选择的相似性排序”
3.  按“更新此图像”

![](img/ef9bb525cf4a796bf9b5e66e221dfc1f.png)

这将向“功能图输出”选项卡添加一组按钮:

![](img/41809108c63bf18b5b2198eb5e9405d7.png)

这些按钮对应于特征地图。它们按特征图的点积大小和图像上的选择(展平后)进行排序。这使得发带周围具有较大幅度的特征贴图会在按钮列表中较早显示。

这是第 310 张特征图的一个例子。蓝色对应正值，红色对应负值。某个位置的绝对值越大，在该位置绘制的方框就越厚、越饱和。

![](img/86ab331a03c0efecf90083a0537c4b30.png)

该特征图似乎在发带周围以及嘴周围具有大的正值。虽然它显然不仅仅用于发带，但我们可以尝试修改它，看看结果是什么。将该选项卡的特征映射 310 设置为 1，擦除发带周围的选择(ctrl+单击并拖动)，并再次加载 sample_01.png。

![](img/7fb4fed7ae4ea9d4f8e8598b56822d98.png)

尝试选择头发，将权重增加到 100 左右，并更新图像:

![](img/a210f571ee49204555839dc42fd1e377.png)

……没什么变化。然而，由于我们没有产生任何奇怪的人造物，将星等增加到 100 以上可能不会有什么坏处。

![](img/f98b32e942b8747b4d0abe66122552ac.png)

大约 600 年，我们得到了看起来像一个发带。我最初的假设是，我们需要使用大星等的一个原因是因为发带不常见。

对于包含在工具中的发带标签，我设置了几个其他的特征贴图，在示例图像中，这些贴图在发带周围是活跃的，设置为 3。将它们设置为大于 1 的数字有助于规范化选项卡的预期范围，因此将乘数设置为 100 左右应该可以表达所需的属性。

![](img/f31b353140f1558c21ed74585e0e5e09.png)

## 3.2 自动特征检测

我们修改属性的方法的一个问题是，它需要手动选择我们想要改变的方块。与在‘嘴部张开’方向上修改潜在向量相比，这无疑是不利的，因为在‘嘴部张开’方向上修改潜在向量不需要我们知道嘴部的位置(即使它也修改非嘴部特征)。这使得该方法无法很好地扩展；虽然比绘图容易，但每次修改仍然需要人工干预。然而，正如我们在上一节中看到的，一些特征地图与属性的位置相关联:例如，特征地图 310 可能被用于一般地检测发带。让我们看看能否找到一种方法，仅使用特征图的线性组合来检测图像中的嘴。

首先，重复用于激活发带周围的特征贴图的过程，只是这次选择嘴:

![](img/b6798eec1511b921b60338977020a4b4.png)

像以前一样，我们可以点击一个按钮来显示一个特征图:

![](img/8f8b3a1e2ce5c3597dd837363f16c101.png)

但是，在查看和比较大量要素地图时，这种方法有点慢。相反，在“从输出添加”按钮下方的文本框中键入 20，然后按按钮。然后，按“查看多个 Fmaps”。

![](img/ebcc752a90e061f0f375c09c242fc8db.png)

带有 20 个按钮的“从输出添加”将前 20 个按钮的特征地图添加到“fmap”文本框，“查看多个 fmap”并排显示它们。

![](img/2192805f18a6e550fcdc823d7dc033af.png)

对应于特征图 234、34、370 和 498 的前 4 个(以及其他几个)看起来都像是嘴巴检测器。然而，我们不知道他们是否会持续检测新图像的嘴部。为了测试这一点，我们可以生成几个新的图像，质量条位于中心右侧，以获得适当的方差。首先确保“记录要素地图”已选中。使用“重置历史记录”清除现有记录。此外，确保“Fmaps To Get”设置为“Current Tab”(该工具不会记录所有分辨率的历史记录，只会记录与当前选项卡相对应的分辨率)。然后，我们可以通过使用*批- >新潜在客户*生成许多新图像，该工具将记录它们的特征图。

![](img/9e5b700c25db320c60292e4f59c793b4.png)

在这种情况下，我将生成 10 个新的图像，这将是不同于你的任何一个。要查看所有图像的相同特征图，请在 Fmaps 框中输入特征图，然后选择“查看历史”。我将对 234、34 和 370 中的每一个都这样做。

![](img/7bfc1d2206c1d593f2fd51d64c97b432.png)

234:

![](img/a1a46a0795ca18f4ecd22f2404588cbc.png)

34:

![](img/e025c6bb7fb93f47224f2e377f9ee906.png)

370:

![](img/f4125a08708264b29c294aee99568e9d.png)

嘴的位置变化不大也无妨，但这些特征地图似乎确实能可靠地跟踪它。

相同的过程可以应用于其他层的其他属性。以下是一些例子:

眼睛:

![](img/a629365f2010d8d92db108955a6ee14b.png)

脸颊:

![](img/70a7990a006b59012b98f23f518e9a42.png)

背景:

![](img/5981d77b6c59c8f4554971556da61209.png)

在许多情况下，我通过组合多个特征地图来获得最一致的结果。为此，我让 python 脚本可以从工具中调用(这对未来的特性也很有用)，因为我更愿意用 numpy 进行多维数据处理。该工具使用 PATH 环境变量来查找 pythonw.exe，因此需要在运行该工具之前进行设置。脚本功能是该工具的最新特性，甚至比其他功能开发得更少。这里有一个例子:

![](img/04ef2c896d3c20b75253c7e94bc934bd.png)

spatial_map.py 存储在“脚本”目录中，您需要安装其依赖项才能使用它。该工具将路径传递到写入“脚本要素地图”中指定的要素地图的目录。然后，它组合这些特征地图并输出结果，该结果由工具读取并用于在图像中进行选择。这里有一个例子，使用了我们之前发现的一些与嘴部位置相关的特征地图:

![](img/861ab432cf390e0d2ca715d3fd2f5ac8.png)

移动嘴部滑块到~100 并更新图像，像平常一样张开嘴。

![](img/87e616ac13204b17b7a98eb1a8a9d748.png)

这让我们可以自动创建具有特定属性的图像。

![](img/2a5b55a5e3df1fecc35dff2e425693ad.png)

通过选择“生成时运行脚本”下的“运行并修改”，将 mouth_open 设置为 100 左右，并生成新的 latents，我们可以确保新图像有一个张开的嘴巴。通过将 mouth_open 设置为-100，并将该设置应用于包含张嘴图像的目录，我们可以生成同样的闭嘴图像。

![](img/029f53008a6328e8ee1a23a718737483.png)![](img/0ab3d46a288dd48c6c893265e6411e22.png)

将设置应用到目录可以通过*杂项- >将当前滑块应用到目录*并选择带有生成图像的目录来完成:

![](img/90e7105e86288731f650404dad130ff0.png)

或者，这可以使用*批处理- > Fmap- >新的潜在组合*一次完成:

![](img/82c325829a30596d0e6a1cdd82b0dc07.png)![](img/f6cbdef6c70032c4845b27fea8c2cf9e.png)

请注意，该选项将基于所有乘数为非零的选项卡进行组合修改。例如，如果“头发”选项卡有一个非零乘数，我们选择生成 3 个“头发”图像和 2 个“嘴巴张开”，它将创建每个图像的 6 个变体，具有以下属性:

*   短发，闭着嘴
*   短发，张着嘴
*   中等长度的头发，闭着嘴
*   中等长度的头发，嘴巴张开
*   长头发，闭着嘴
*   长发，张嘴

关于脚本还有几点:

*   您可以在要素地图前添加一个乘数，以更改各种地图组合时的影响。

![](img/b92ab80d4e8d850de1afc1768a51d463.png)

*   用于检测面部特征的特征图的分辨率不需要与我们输入数据的分辨率相匹配。脚本应该根据需要调整大小:

![](img/c66b502b39052d2c4997f11cd62834f6.png)

*   脚本还有其他潜在用途，例如记录要素地图的表示，以便在工具关闭后进行分析。

这涵盖了该工具的大部分功能。在下一节中，我将更详细地讨论这个架构。

## 3.3 工具详情

![](img/b1d3b145ede724cb2097f0b23e403ac2.png)

From TensorFlow python implementation to generating images with the tool

该工具有 4 个主要组件:

1.  TensorflowInterface.dll:使用 tensorflow C api 加载模型
2.  GanTools.dll:将 TensorflowInterface.dll 加载到 C#环境中
3.  GanStudio.exe:加载 GanTools.dll 以便与模型交互
4.  数据文件:graph.pb(模型)和 data _[graph . Pb 的 MD5 中的文件，这些文件包含特定于模型的信息并被加载以帮助交互。

这些都可以在 zip 文件中看到:

![](img/c78ba85ee63e836df50831fe185b693e.png)

图像被写入“肖像”目录，标记为收藏夹的图像被写入“收藏夹”目录，而“脚本”目录包含该工具可以加载的 python 脚本。其余的文件是工具的其他组件使用的库。

数据目录包含几个文件:

![](img/224b3ee4c05a47a16cb79e3b05f14e43.png)

*   saved_fmaps 包含工具启动时加载的选项卡(嘴巴张开、发带、腮红等)数据。选项卡数据包括修改了哪些要素地图、脚本信息和乘数。
*   attributes.csv 包含“Attributes_0”选项卡用来以有意义的方式修改潜在向量的向量。
*   average_latent.csv 包含通过对映射网络生成的大约 10，000 个潜在值进行平均而生成的平均(中间)潜在值。质量/唯一性栏使用它来应用截断技巧。
*   latents.csv 包括所有生成图像的潜在值。在我开始将潜在值附加到图像之前，它更重要，但它仍然可以作为备份。

所有这些都是特定于模型的，这就是为什么我通过将 graph.pb 文件的散列附加到目录名来将目录绑定到模型的原因。

TensorflowInterface.dll 是加载 graph.pb 的组件，它使用[tensorflow.dll](https://storage.googleapis.com/tensorflow/libtensorflow/libtensorflow-cpu-windows-x86_64-1.14.0.zip)与之交互。它通过导出 3 个主要函数来实现这一点:

![](img/056ce5f2961e723ea0110133339210b9.png)

这种交互的大部分是使用模型中张量的名称来完成的。虽然源代码是可用的，但我认为获得这些名称(以及更好地理解模型)的最好方法之一是通过像 [netron](https://github.com/lutzroeder/netron) 这样的工具。在下图中，netron 用于检查添加操作的输入，该操作合并了此工具所做的要素地图修改。

![](img/34a8de8e0094d2b9f8dcba1a4c9a35c0.png)

Operations to allow adding/subtracting values in feature maps (screenshot of [Netron](https://github.com/lutzroeder/netron))

# 4.结论

结合本系列的前一篇博客，我已经写了一份相当全面的记录，记录了到目前为止我为这个项目所做的工作。这些博客没有详细介绍的两个例外是生成 512x256(高 x 宽)图像的能力和我用更多男性图像训练的模型。

本博客中使用的模型实际上可以产生 512x256 的图像，但是我在网络早期就切掉了底部:

![](img/0c2dce9b13ec7b6526cb0a900cdf377c.png)

Using netron to view graph.pb

这提高了图像创建速度，减小了 UI 的大小，但最重要的是，它保持了图像适合所有观众。

我最初认为有男性图像的模型更糟糕，因为图像的方差增加了，平均质量降低了，为了包含大量的男性图像，我必须降低“最喜欢的”阈值(见原始博客)。然而，当我更新我的 StyleGAN 重新实现的代码时，我注意到我在实现样式混合时引入了一个 bug ( [这里是修正](https://github.com/nolan-dev/stylegan_reimplementation/commit/da7c6abd039f1547f3d60c38be356b160e1dd120))。该错误意味着用于从中间潜在到风格转换的权重在两个自适应实例规范化层之间共享。这降低了网络的容量，也可能是新模型比旧模型更差的原因。这是需要处理的最令人讨厌的错误类型之一:它不会阻止模型工作，但它会以一种直到模型完成训练才真正注意到的方式降低性能，并且性能的降低可能归因于其他因素。安德烈·卡帕西在这一系列推文中说得很好:

对于个人项目，包括这篇博客中讨论的工具，我更喜欢使用“快速修复”的方法(恶意软件分析让我几乎享受到了调试的乐趣)。然而，这对于实现深度学习模型不起作用，尽管我在使用 TensorFlow 时倾向于更慢地移动以避免这样的错误，但有时如果我在项目的不同部分花费了太多时间，它们仍然会出现。这是我正在积极尝试改进的地方，我发现一个有用的策略是在 TensorBoard 或 netron 中查看模型的图形，以获得不同的视角。

由于这个工具主要是基于查看和修改特征地图，我有兴趣使它适用于除了 GANs 之外的生成模型。如果可以将任意图像作为输入的模型可以像 StyleGAN 一样在其内部表示中演示相同类型的无监督面部特征检测，那将是一件好事。我还想将这种动画制作方法与[视频到视频合成](https://nvlabs.github.io/few-shot-vid2vid/)进行比较，并对使用生成模型制作动画的其他工作进行更多研究。