# 手部关键点检测

> 原文：<https://towardsdatascience.com/hand-keypoints-detection-ec2dca27973e?source=collection_archive---------7----------------------->

*用小训练数据集检测手部图像上的关键点位置。*

H 训练一个网络来准确预测手指和手掌线的位置需要多少标记图像？我受到了[这篇博文](/fun-with-small-image-data-sets-part-2-54d683ca8c96)的启发，在这篇博文中，作者报告了 97.5%的分类准确率来分类一个人是否戴着眼镜，而每类只有 135 张训练图像。对于我的任务，从 15 个不同的人那里得到的 60 张带标签的图片能有多准确？

![](img/46d07229a704875bc8fef5bf10362c82.png)

12 points to detect.

目的是在给定未标记的手部图像的情况下，精确估计 12 个点的 x 和 y 坐标。8 个点位于指尖和四个手指的根部，另外 4 个点沿手掌的中心线等距分布，第一个点和最后一个点正好位于中心线的起点和终点。

这些图片是我的朋友们的，他们友好地同意参加，每人最多 6 张图片，左手 3 张，右手 3 张。这些照片是用不同的智能手机在不同的白色背景和不同的光照条件下拍摄的。60 张图片都是我手动标注的。

对于神经网络处理，我使用了基于 PyTorch 的 fastai 库 v1.0.42。Jupyter 笔记本做 IDE，我笔记本的 NVidia GTX 960M 4Gb VRAM 做训练用。我下面分享的结果的总训练时间是 25 小时，非常合理的时间，因为这种 GPU 远远不是当今市场上最好的硬件！

该项目的主题是数据增强，幸运的是 fastai 提供了高效的图像转换算法，以及定义这些算法的干净 API。让我们深入细节。

## 数据和模型

标记的数据被分成 51 个训练图像和 9 个验证图像。验证包括出现在列车组中的人的 3 个图像，但也包括既不出现在列车组中也不与列车中的任何人共享相同背景\摄像机的两个人的 6 个图像。在预处理中，所有右侧图像都水平翻转。

![](img/e202901b646c3dfcf299b180c7c1f732.png)

All train images with labels.

在如此小的数据集上进行训练，数据增强是必要的，我对进入神经网络的每张图像都使用了随机缩放、包裹、旋转、亮度和对比度转换。fastai 库允许很容易地定义它，并且同样的仿射变换也应用于标签点。根据训练集的样本均值和方差对图像进行归一化，分别针对每个 RGB 通道进行归一化，并将其大小调整为 4:3 的比例，更具体地说是 384×288 像素。听起来要定义很多东西？令人惊讶的是，整个数据定义可以归结为下面一小段代码。

```
transforms = get_transforms(do_flip=False, max_zoom=1.05, max_warp=0.05, max_rotate=5, p_affine=1, p_lighting=1)data = (PointsItemList.from_folder(path=PATH/'train', recurse=False)
   .split_by_idxs(train_idx, valid_idx)
   .label_from_func(get_y_func, label_cls=PointsLabelList)
   .transform(transforms, tfm_y=True, size=(384,288),
              padding_mode='border', resize_method=ResizeMethod.PAD)
   .databunch(bs=6, num_workers=0, no_check=True)
   .normalize(stats))
```

该型号是带有定制头标准 resnet34 主干。从 resnet34 中删除了最后两个分类层，取而代之的是，我添加了 1x1 卷积以减少通道数，然后是两个完全连接的层。第二个完全连接的层输出 24 个激活，在通过 tanh 激活函数后，这些激活表示 12 个关键点的 x 和 y 坐标。

但有时用暗语说更好:

```
head_reg = nn.Sequential(
    nn.Conv2d(512,64,kernel_size=(1,1)),
    nn.BatchNorm2d(64),
    nn.ReLU(),
    Flatten(),
    nn.Linear(64 * 12 * 9, 6144),
    nn.ReLU(),
    nn.Linear(6144, 24),
    Reshape(-1,12,2),
    nn.Tanh())
learn = create_cnn(data, arch, metrics=[my_acc,my_accHD], loss_func=F.l1_loss, custom_head=head_reg)
```

上面的自定义头定义使用常规 PyTorch 语法和模块，除了我写的整形模块，它只是…嗯，整形张量。这种整形是必需的，因为我的标签坐标由 fastai 内部表示为 12 乘 2 张量，它需要匹配。此外，标签被重新调整为[-1；1]范围，这就是为什么 tanh 激活函数在这里是合适的。

优化目标是最小化列车组的 L1 损失。

还有两个额外的准确性指标来判断网络的性能和进度。在验证集上测量，这两个度量计算预测坐标落在实际标签的 0.1 和 0.01 误差内(分别对于第一和第二度量)的比例。这里标签的范围也是[-1；1]，并且给定 384×288 像素的图像尺寸，可以容易地计算出第二(高精度)度量允许高度和宽度的最大误差分别为 1.92 和 1.44 像素。

## 神经网络训练

NN 训练是通过运行这行代码 40 次来完成的:

```
learn.fit_one_cycle(cyc_len = 100, max_lr = 1e-4)
```

除了使用 Adam optimizer 运行 100 个时期的常规训练之外，该 fastai 方法具有有趣的学习速率和动量策略，fastai 使用该策略在广泛的应用中提供更快的收敛。在 Sylvain Gugger 的博客文章中可以看到更多的细节。我发现它对于我的模型来说开箱即用。对于每 100 个历元周期，50 个历元后的误差比开始时高，但在周期结束时总是有所改善。看看下图中的典型错误发展。

![](img/61663baf2d5dc7d3502fcbe9a1b69a96.png)

Learning rate (left) and momentum (right) changing across 100 epochs, 8 batches in every epoch.

![](img/3cd44f7bad9fd499534b7fc0b08e5a2a.png)

Losses for epochs 2500 to 2600, 8 batches per epoch. More data were added at epoch 2500.

这个学习速率和动量过程被称为 1 周期策略。据称，它也有助于对抗过度拟合，而且它似乎比我尝试的其他(公认有限的)选项收敛得更快。

为了理解不同变化的影响，我将培训分为 5 个步骤:

1.  1500 个时期，resnet34 个主干层冻结在 ImageNet 预训练值上，并且仅训练自定义头层。仅使用 35 个训练图像的子集。
2.  300 个纪元，**解冻**骨干层。
3.  700 个纪元，增加了**更多数据增强**。具体来说，最大缩放 5%到 10%，最大扭曲 5%到 20%，最大旋转 5 度到 10 度。
4.  500 个历元，从 4 个额外的人添加 **16 个图像**到训练集。使总的训练集大小达到 51。
5.  1000 个周期，每个周期**降低 20%的学习率**，最后一个周期达到 1e-5 左右的学习率。记住，每个周期是 100 个纪元。

下图总结了进度:

![](img/f33c890a532a324614bcb9ab83e7b960.png)

Loss and accuracy metrics during the training on 4000 epochs.

这 5 个步骤中的每一步都对模型进行了额外的改进。数据扩充中的更多转换尤其重要，对验证集误差的改善有显著贡献。解冻和更多的数据也为验证集提供了很好的改进。另一方面，学习率的降低显著改善了训练集误差，而验证集误差则停滞不前。过度适应在这里是一个真正的问题，使用较小的学习率会使情况变得更糟。

总的来说，在训练期间，网络看到了 147k 个不同的变换图像，并且训练花费了 25.5 小时。

## 讨论结果

虽然训练集的最终平均 L1 误差是 0.46 像素，但是验证集的最终平均误差是 1.90 像素。此外，训练集的这个分数是针对经变换的图像的，而验证图像是未经变换的(更容易)。这是一个明显的过度拟合。

尽管如此，结果还是相当不错的，下图显示了验证集推理。请注意，绿点是实际的标签，红点是最终模型的预测。观察结果，似乎该模型使其预测更加全局化，并且在点之间更加相互依赖，而不是给予局部边缘更多的权重。

![](img/2ca4c38aaa8323242e9f9fa6518a55bc.png)

Validation set final results. Images 1, 2 and 3 are of people present in the train set, but different images. Images 4 to 6 and 7 to 9 are two people not appearing in the train set. Green points are actual labels and red points are predicted ones.

模型改进的明确方向是更多的数据和更智能的数据增强。平均误差仍然需要小 4-5 倍，才能达到良好的/生产水平的质量。

## 什么没用？

在不同位置添加下降层和应用权重衰减没有用。这可能是因为 1 周期策略的高学习率本身就是一种正则化形式，不需要更多的正则化。

替代 Adam 的其他优化方法没有显示出任何改进，或者取得了更差的结果。为不同的层组选择不同的学习速率也是一个死胡同。

如果从自定义头中删除 BatchNorm 层，即使学习率为 1e-5，模型也不再是可训练的。

我尝试了另外两种主干模型架构， [Darknet](https://docs.fast.ai/vision.models.html) 和 [U-Net](https://docs.fast.ai/vision.models.unet.html#vision.models.unet) ，它们有不同的定制头，但在我的实验中，它们的效果不如更简单的 resnet34。

![](img/a96b57ed1c3832debe9eeacb4fc14bdb.png)

最后，fastai 库在这一点上只有仿射变换(平行线映射成平行线)和透视变换(直线映射成直线)。鉴于数据扩充在这个项目中的重要性，我实现了一个额外的转换，见图片。然而，由于某种原因，它并没有导致改善。

## 结论

仅用 51 幅训练图像，所讨论的模型在独立图像上达到了相当好的预测精度。更多的数据和更多的数据扩充被证明可以提高精度，并在某种程度上成功地对抗过拟合。

Fastai 库是这个项目的合适工具，它提供了以下好处:

1.  简洁但灵活的数据和模型定义
2.  一系列用于数据扩充的内置仿射和透视变换，也可以自动变换标注点
3.  智能学习率和动量策略似乎给出了更快的收敛并减少了过拟合