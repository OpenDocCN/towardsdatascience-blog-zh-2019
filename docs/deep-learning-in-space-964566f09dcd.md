# 太空中的深度学习

> 原文：<https://towardsdatascience.com/deep-learning-in-space-964566f09dcd?source=collection_archive---------12----------------------->

## 人工智能和机器学习如何支持航天器对接。

![](img/080753c85d6d1d83e9d83b8250d43ae3.png)

Magellan space probe meant to map the surface of Venus, simulated in orbit around earth using Unity (credits and info at the end).

人工智能无处不在。家电、汽车、娱乐系统，凡是你能想到的，它们都在包装人工智能能力。航天工业也不例外。

在过去的几个月里，我一直在开发一个机器学习应用程序，通过简单的摄像头视频来帮助卫星对接。如果你想知道深度学习、神经网络和 Tensorflow 对卫星对接有多有用，请继续阅读。

在这篇博文中，我将向你介绍我学到的方法、工作原理、结果和教训。我一点也不想挑战物体识别的技术水平。但是，当追溯我的脚步时，我意识到我学到了很多。因此，我希望我的故事对你有用，并激励你创建自己的机器学习应用程序。

如果你想跳过阅读直接进入代码，GitHub 上什么都有:[https://github.com/nevers/space-dl](https://github.com/nevers/space-dl)

> 请注意，这篇文章假设读者熟悉张量流、卷积神经网络(CNN)的基本原理以及如何使用反向传播来训练它们。

## 信用

*   非常感谢 [Rani Pinchuk](https://www.linkedin.com/in/first) 的支持，无数次的讨论，由此产生的见解以及花费在繁琐的标注所有训练数据上的时间。
*   OS-SIM 设施的训练和评估图像由 [DLR](https://www.dlr.de) 提供。
*   麦哲伦太空探测器，由 Xavier Martinez Gonzalez 使用 Unity 模拟。

## 索引

1.  [数据集准备](https://medium.com/p/964566f09dcd#be6d)
2.  [模型原理](https://medium.com/p/964566f09dcd#3dda)
3.  [损失函数](https://medium.com/p/964566f09dcd#5817)
4.  [迁移学习](https://medium.com/p/964566f09dcd#c63b)
5.  [结果](https://medium.com/p/964566f09dcd#fbc9)
6.  [惨痛的教训](https://medium.com/p/964566f09dcd#a37a)
7.  [硬件](https://medium.com/p/964566f09dcd#238a)
8.  [结论和后续步骤](https://medium.com/p/964566f09dcd#5c2b)
9.  [学分](https://medium.com/p/964566f09dcd#f888)

[Intermezzo——对象分割、检测和定位。](https://medium.com/p/964566f09dcd#d8d1)

Intermezzo——张量流估值器与人工抽象。

# 数据集准备

知道卫星的详细尺寸，目标是创建一种算法，可以准确预测其姿态和与相机的相对距离。这个项目的数据集是从安装在[德国航天中心](https://www.dlr.de) OS-SIM 设施的机械臂上的卫星实物模型中创建的。手臂模拟各种动作，同时摄像机记录视频。

![](img/9d00898d155caa5c89b9ac4b13af76bf.png)

The satellite mockup captured by the video camera on the robotic arm. Source: OSM-SIM facility [DLR](https://www.dlr.de).

我决定集中精力寻找卫星的尖端。如果我能准确地定位它，我相信我可以对模型上的至少两个其他标签做同样的事情。(卫星的“尖端”实际上是其对接机构的一部分。)给定这 3 个(或更多)点和卫星的 3D 模型，然后我可以重建卫星的姿态和相对于相机的相对位置。

相机记录了 14，424 张未标记的图像，我想用它们来训练和评估一个神经网络。我的一个担忧是，我将不得不花很长时间在每张图片上手动标注小费。幸运的是，我了解到 OpenCV 的优秀图像标记工具: [CVAT](https://github.com/opencv/cvat) 。

> 使用 [CVAT](https://github.com/opencv/cvat) 你可以批量导入所有你想要添加注释的图像，将它们作为电影播放，并插入相隔许多帧的注释。它还允许在多人之间分割工作，它甚至有一个很好的 docker-compose 文件，允许你点击一个按钮来运行它。

CVAT 节省了大量的时间和工作:只花了几个小时就在 14，424 张图片上标注了提示。(其实这个工作我不能拿[的功劳](https://medium.com/p/964566f09dcd#5206)。)对于卫星的线性运动，我们只需标注开始和结束位置，CVAT 将在它们之间插入并添加所有标签。如果你需要视频或图像注释工具，强烈推荐 CVAT。

![](img/cdcf69e444f185461326c779e6bc2b90.png)

Annotating the tip of the satellite using boxes in OpenCV’s CVAT.

然而，有一些改进的机会，或者更确切地说，是我希望拥有的功能。例如，CVAT 不支持点之间的插值。作为一个变通办法，所有的注释都必须用方框来代替点。(框的左上角坐标用于匹配笔尖的位置。)此外，任何未加注释的帧，即提示不可见的帧，都不包括在 XML 输出中。

XML output from CVAT after annotating the images.

为了使这个 XML 文件适合于训练和评估模型，必须对它进行后处理，使其成为正确的格式。有趣的是:这个看似琐碎的任务，实际上需要相当多的迭代才能完成。我经常不得不回去修改标签、添加新标签、更新输出格式等等。对我来说，这是一个教训。

> 将原始数据和注释转换成适合训练和评估的数据集的代码是代码库的重要组成部分。它不仅仅是一堆晦涩难懂的一次性终端命令。你应该尊重它，因为它是剧本的一部分，允许你重现你的结果和你的文档。对你的代码进行版本化，审查它，对你的数据集版本使用[语义版本化](https://semver.org/),最重要的是，通过压缩数据集并提供下载，使其他处理相同问题的人能够容易地使用数据集。

一旦我有了数据集构建脚本的基线，同事们就能够重用它并分享他们的更改。我已经在我们公司引入了 [Nexus](https://www.sonatype.com/nexus-repository-oss) ，我们现在使用它来分发来自 Java、Python 和 Docker 的所有代码工件，包括数据集等等。

数据集构建脚本还允许对不同版本的数据集进行快速实验:

*   应用数据增强:旋转，模糊，锐化。
*   尝试不同的培训和评估方法。
*   将数据定制为适合您的模型的表示形式。
*   将数据转换和打包成合适的格式。

这最后一点值得多加注意。因为我使用的是 [Tensorflow](https://tensorflow.org) ，所以我想使用 TFRecords 数据格式。不仅仅是因为它很好地集成到了 TF Dataset API 中，更重要的是因为我认为这种二进制数据格式从磁盘读取会更有效。下面是我如何使用 Python 多重处理将图像和标签转换成 TFRecords 文件的代码摘录。(我想使用多线程，但是……在 Python-land 中，线程并不酷，而且 [GIL](https://wiki.python.org/moin/GlobalInterpreterLock) 也这么说。)

Convert images and labels to a TFRecords file using multi-processing in Python.

在创建了 TFRecords 文件之后，我创建了这个脚本来测试和比较从 TFRecords 文件中读取 13，198 个训练图像与简单地从磁盘中读取每个图像并动态解码它们所花费的时间。令人惊讶的是，TFRecords 数据格式并没有真正提高读取训练数据集的速度。下面的时序输出显示，从 TFRecords 文件中顺序读取比从磁盘中读取每个图像并即时解码要慢*。差别很小，但我肯定 TFRecords 会更快。*

*如果您真的想提高数据导入管道的性能，可以考虑并行处理和数据的[预取](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#prefetch)。通过在解析数据集时简单地设置[TF . data . dataset . map num _ parallel _ calls](https://www.tensorflow.org/api_docs/python/tf/data/Dataset#map)参数，从 TFRecords 文件中并行读取这些完全相同的图像要比顺序读取快 2 倍。从磁盘上读取每张图片并即时解码甚至比 T8 快 3 倍。然而，在并行示例中，读取 TFRecords 文件几乎比动态读取图像慢 2 倍。又不是我所期待的。如果有人可以指出这个问题并分享他们与 TFRecords 的经历，我会很高兴。*

> *最后，结合并行解析和预取使我能够消除训练期间的任何 CPU 瓶颈，并将平均 GPU 利用率从 75%提高到 95%以上，这是用 [nvidia-smi](https://developer.nvidia.com/nvidia-system-management-interface) 命令测量的。*

*以下是脚本在我的旧 2011 iMac(2.7 GHz 英特尔酷睿 i5)上运行时的时序输出:*

***13198 幅图像的顺序解析:***

*   *TFRecords 数据集:50.13s*
*   *普通 PNG 文件数据集:49.46 秒*

***并行解析 13198 幅图像:***

*   *TFRecords 数据集:26.78s*
*   *普通 PNG 文件数据集:15.96 秒*

# *示范原则*

*最近，我在Coursera 上完成了[吴恩达的深度学习专精](https://www.coursera.org/specializations/deep-learning)。(Ng 的发音有点像《歌》结尾的 [n 音](https://www.quora.com/How-does-Andrew-Ng-prefer-his-name-to-be-pronounced)。)这五门课程涵盖了深度学习和神经网络的核心概念，包括卷积网络、RNNs、LSTM、Adam、Dropout、BatchNorm、Xavier/He 初始化等等。该课程还详细介绍了医疗保健、自动驾驶、手语阅读、音乐生成和自然语言处理的实际案例研究。除了安德鲁惊人的成绩之外，他还是一位伟大的老师，我必须说这是一次美妙的经历。我可以向任何想进入深度学习的人推荐这门课程。*

*在涵盖卷积神经网络的第四门课程中，他对使用 [YOLO 算法](https://arxiv.org/pdf/1506.02640.pdf)的物体检测做了精彩的[解释](https://youtu.be/9s_FpMpdYW8)(你只看一次)。该算法执行实时对象检测，如下图所示。*

*![](img/bffeab17421a39d6d61ac35be2ca4fd4.png)*

*Object detection using YOLO. Source: [https://pjreddie.com/darknet/yolo/](https://pjreddie.com/darknet/yolo/)*

> *“YOLO 是最有效的对象检测算法之一，它包含了整个计算机视觉文献中与对象检测相关的许多最佳想法。”— [吴恩达](https://youtu.be/9s_FpMpdYW8?t=385)*

*就这样，我忍不住实现了我自己的算法的天真版本。我不会在这篇文章中解释最初 YOLO 论文的全部工作原理和细节，因为有太多优秀的博客文章都是这么做的。(像[这个](https://blog.paperspace.com/how-to-implement-a-yolo-object-detector-in-pytorch/)举个例子。)相反，我将着重于我如何使用 YOLO 来解决我的具体本地化问题。*

## *物体分割、检测和定位。*

*物体分割、检测和定位是有区别的。对象分割的目的是找到各种形状的片段，这些片段给出图像中要检测的对象的轮廓的逐像素描述。对象检测是在给定图像中的一个或多个对象周围找到矩形边界框的过程。目标定位是寻找一个或多个目标的位置。*

*![](img/d973c630da82d706fc2ee78c70dc1cf7.png)**![](img/9576cc9febb2bb6c8fff0d606ee4663c.png)**![](img/cd157b3cf7da09b5b884879bdb90a93b.png)*

*Object segmentation, detection and localization from left to right.*

*该算法的主要原理很简单:获取输入图像并应用大量的卷积层，每个卷积层都有自己的一组滤波器。每组卷积层都会降低图像的特征空间或分辨率。请记住，卷积保持空间局部性，因为每一层都有与其相邻层的局部连接模式。因此，输出层中的每个元素代表输入处原始图像的一小块区域。每个卷积步长的滤波器可以从 64 到 1024 或者甚至 4096 变化。然而，在最终的输出层中，过滤器的数量减少到 3 个。换句话说，输出层有 3 个通道，每个通道将被训练为针对图像中特定区域的不同目的而激活:*

*   *通道 1 —预测位:表示在 0 和 1 之间卫星提示出现在图像的该区域中的机会。*
*   *通道 2-相对 X 位置:尖端的垂直位置(如果可用)，相对于该区域的左上原点。*
*   *通道 3 —相对 Y 位置:与通道 2 相同，但不同。*

*看看下面的图片，这是我试图描绘的概念。*

*![](img/fce6f894c7766169b5ea1c4f7906d519.png)*

*The input image in the top layer is dimensionally reduced to the output layer at the bottom (omitting the convolutional layers in between). The grey lines between the input and output layer show how each neuron along the depth dimension (or per channel) is dedicated to a specific region of the image. Per region, the output volume predicts whether a tip is visible and its X and Y coordinate relative to the origin of that region. In an ideal scenario, a prediction would have all elements set to zero except for the highlighted volume where the tip is visible.*

*在我的算法的第一个天真的版本中，我没有花很多时间来找出解决我的问题的完美的 CNN 模型架构。相反，我想专注于代码中允许我训练和评估模型的部分。因此，我简单地实现了与最初 YOLO 论文中的架构图相同的模型布局(如下)。*

*![](img/79556b8cbdad9243c2fb3912074d3fae.png)*

*YOLO v1 CNN model (source: [https://arxiv.org/pdf/1506.02640.pdf](https://arxiv.org/pdf/1506.02640.pdf))*

*这就是我简单的头脑如何将这些层解释成代码。*

*My naive interpretation of the Yolo v1 model.*

*我很高兴我没有在模型上花太多时间，因为建立损失函数和训练/评估需要花费更多的时间。此外，现在有太多优秀的模特，很难被击败。例如，看看 [Tensorflow Hub](https://tfhub.dev/) ，或者看看[在 Keras](https://keras.io/applications/) 提供的模型。出于这个原因，我并不太关心模型的性能。相反，我的主要目标是让算法的所有活动部分都工作起来:输入端的数据集管道、可训练模型、损失函数和评估指标。*

# *损失函数*

*为了计算损耗，我的第一步是将所有标签(基本上是卫星尖端的 x，y 位置)转换成输出体积，如上图所示。这是我想出来的。或者，如果您喜欢跳过大部分代码，只需查看脚本第 20 行的简单示例。*

*Code excerpt that parses a given label (i.e. the x and y position of the tip of the satellite) into a volume similar to the output of the model.*

*下一步是将给定的解析标签与模型的输出进行比较，并设置允许梯度下降以优化模型参数的损失函数。我尝试了许多替代方案，摆弄着均方误差和均方对数误差。最后，我决定使用[交叉熵损失](https://ml-cheatsheet.readthedocs.io/en/latest/loss_functions.html?highlight=cross-entropy)(如果你愿意，也可以使用对数损失)，因为它对于概率值在 0 到 1 之间的分类任务特别有效，比如预测损失。*

*损失函数本身是两部分的加权和:*

*   *预测损失:模型预测输出体积中每个盒子是否有卫星提示的程度。我给这个预测损失的权重是 5，因为它是获得正确预测的主要因素。*
*   *XY-loss:如果盒子里有尖端，模型预测尖端位置的准确性如何。如果图中没有可用的提示，损失函数的这一部分应该为零，这样只有预测损失决定最终损失。我给这个预测损失的权重是 1。*

*看看下面实现这个损失函数的代码。这样，我就可以使用 Adam 优化器来训练模型了，万岁！*

*The loss function of the model.*

> *事后想来，写这个的时候，我意识到这个损失函数还是可以改进很多的。如果图像中有一个尖端，则为输出体积中的每个框计算 XY 损耗。这意味着 XY 损耗也被考虑到所有没有可见尖端的盒子中，这不是我想要的。因此，XY-loss 主要被训练来探测背景，而不是卫星提示…哎呀。此外，XY 损失不像预测损失那样是一项分类任务。因此，使用均方误差或类似策略来计算可能更好。有趣的是:这个损失函数表现很好。所以，这实际上是个好消息:它可以表现得更好:)*

# *迁移学习*

*一旦我有了模型和损失函数，运行和训练正常，我想把我对 YOLO 模型的天真解释换成一个经过战斗考验和预先训练的版本。因为我只有有限的数据集，所以我假设需要迁移学习来解决这个问题。*

*一种选择是简单地从 [Tensorflow Hub](https://tfhub.dev/) 中挑选一个模型。然而，TensorFlow 使得[使用这些模型](https://medium.com/tensorflow/introducing-tensorflow-hub-a-library-for-reusable-machine-learning-modules-in-tensorflow-cdee41fa18f9)太容易了，我想走一条更具挑战性的路线，这样我可以学到更多。我决定使用原作者的最新版本的 YOLO 模型，因为它是为[暗网](https://pjreddie.com/darknet/)编写的，我想了解如何将该模型导入 Tensorflow。*

*当我开始研究[最新的 YOLO 模型定义](https://github.com/pjreddie/darknet/blob/master/cfg/yolov3-spp.cfg)时，我很快意识到我需要解析定义文件中的每一个部分并将其映射到正确的 Tensorflow 层。也许我应该对我的愿望更加小心，因为这是一项乏味又费时的工作。幸运的是，我找到了这个将 YOLO 模型定义转换成 Keras 模型的脚本[,我可以使用 Tensorflow 加载它。](https://github.com/qqwweee/keras-yolo3/blob/master/convert.py)*

*迁移学习是指重用在不同但相似的数据集上预先训练的现有模型的部分层权重，并仅重新训练剩余的层。一旦我加载了所有的 252 个模型层，我必须弄清楚哪些层(及其相关权重)我想要保持不变，哪些层我想要重新训练以及它们的每个维度。为此，我编写了一个简短的脚本[，将 Keras 模型绘制成图像](https://github.com/nevers/space-dl/blob/master/print_layer_shape.py)[并从给定的图层列表中计算尺寸。](https://github.com/nevers/space-dl/blob/master/model.png)*

*使用这个脚本，我可以简单地[预览整个模型布局](https://github.com/nevers/space-dl/blob/master/model.png)，包括所有的图层名称。然后，我在模型的正中间手动选择了一个图层:“add_19”。在我的实现中，使用 layer.trainable 属性，前半部分的所有层的权重保持不变，后半部分的所有层的权重被重新训练。模型中的最后一层是“conv2d_75 ”,它有 255 个通道。我添加了一个额外的卷积层，内核/过滤器大小为 3，以减少模型输出并使其符合我的最终目标。*

*Loading the YOLO model in Keras, enabling transfer learning and matching the output layer dimensionality to match the labels and loss function.*

# *结果*

*首先，让我们看看迁移学习是如何影响结果的。看看下面的图片。重用 YOLO 模式的前半部分，重新训练后半部分，会产生巨大的不同。事实上，结果是无可比拟的。没有迁移学习，损失函数停滞在 80 左右，而有了迁移学习，损失函数立即下降到几乎为零。*

*![](img/d554ca7632b26960f53ddc9e05f66ff7.png)*

*Model loss function output per training step. The dark blue line shows the loss without transfer learning or simply a randomly initialized model. The light blue line shows the loss when half of the model reuses weights from the YOLO model.*

*下图显示了不使用迁移学习时的模型输出。请注意，该模型能够过滤掉背景并专注于提示，但永远无法做出准确的预测。*

*![](img/f562e74cdf4b408a134e881e00a18d7a.png)*

*Model prediction output when not using transfer learning. Each output volume of the model is shown as a semi-transparent box with a color that ranges from (very transparent) blue (indicating a low chance of a tip present in that box) to green and then to red (indicating a high chance).*

*![](img/569c126a6c9e3b63de7c8815de5b7b74.png)*

*The model prediction output visualized for the same image over 42 training epochs when not using transfer learning. Notice how the model learns how to filter out the background, but never succeeds into narrowing down on the tip.*

*这是整个评估数据集的样子，仍然没有迁移学习。*

*Video animation of the model predictions for the evaluation data without transfer learning.*

*然而，这是在启用迁移学习的情况下，整个评估数据集的情况。*

*Video animation of the model predictions for the evaluation data using transfer learning.*

*很明显，迁移学习对结果有着巨大的影响。因此，本文的其余部分和结果假设迁移学习是启用的。*

*除了损失函数的输出，模型的性能以 4 种方式测量:*

1.  ***metrics/dist_mean:** 对于模型正确预测存在吸头的所有样本，从预测到标签的平均距离是多少(以像素为单位)。*
2.  ***accuracy/point_mean:** 对于模型正确预测存在吸头的所有样本，这些样本中距离标记吸头 10 个像素以内的样本所占的百分比。*
3.  ***accuracy/prob_mean:** 模型预测小费存在的准确程度。即预测位必须高于 0.5。*
4.  ***准确性/总体均值:**正确预测样本的百分比。即，如果没有尖端，模型也预测同样的情况，并且如果有尖端，它在离标签 10 个像素之内。*

*以下是对 2885 个样本的评估数据集进行约 8 个小时的模型训练后得出的评估结果。*

1.  ***度量/距离 _ 平均值:**1.352 像素*
2.  ***准确率/点均值:** 98.2%*
3.  ***准确率/概率均值:** 98.7%*
4.  ***准确率/总体均值:** 98.7%*

> *下面你可以在 Tensorboard 上看到这些数字随时间的变化。简单来说，算法平均差一个像素。*

*![](img/4565d3eccb5242ee2d13654cd0b8a7eb.png)**![](img/075592eac2f995f6e150e5a6e06c032c.png)*

*Four evaluation metrics and the loss function calculated after every training epoch during a training period of 8 hours.*

*在 2885 个评估样本中，有 32 张图片的预测是错误的。当我看着它们的时候，有 28 张图片的尖端位置被相当精确地探测到了，但是这个模型根本就没有足够的信心说有一个尖端。即，预测器比特没有超过 0.5，但是选择了正确的盒子。这里有一个例子。*

*![](img/32335e631a61e7991f18a861368afbfe.png)*

*The model predicts the tip within 10px but the confidence level is just below 0.5 and therefore it is marked as an erroneous prediction. It’s so close to 0.5 that when rounding the predictor bit, it yields exactly 0.50.*

*剩下的四个负面预测更有意思。它们大多被贴错了标签，或者至少是模棱两可的。当尖端隐藏在物体后面，但人类仍然容易定位时，一些图像被不一致地标记。这正是模型所捕捉到的。下面显示了两个示例:尖端隐藏在对象后面，并被标记为没有可见的尖端。而模型预测存在尖端并且在正确的位置。*

*![](img/6444a684b97029e0380e7834c10a509b.png)**![](img/39bbfaf51a458e2e9578e541cc21e055.png)*

*Examples of a negative prediction where the tip of the satellite is hidden behind an object. These images are labeled as not having a visible tip (hence the label -1, -1), whilst the model is still able to predict a correct position.*

## *intermezzo——张量流估值器与人工抽象。*

*Tensorflow 包括评估器，以保护开发人员免受样板代码的影响，并将他们的代码引导到一个可以轻松扩展到多台机器的结构中。我总是使用评估者，并假设我对他们的忠诚会得到高效率、干净代码和免费特性的回报。在 Tensorflow 1.12 中，这些假设的一部分是正确的，但我仍然决定创建自己的抽象。下面我解释一下原因。*

*为了确保一致性，每次您调用 estimator 时，estimator 都会从磁盘重新加载模型。{train()，predict()，evaluate()}。(train_and_evaluate 方法只是一个调用 estimator.train 和 estimator.evaluate 的循环)如果您有一个大模型(这很常见)，并且您希望在同一台机器上交错训练和评估，重新加载模型确实会减慢训练过程。*

*评估人员重新加载模型的原因是为了确保分布模型时的一致性。这是他们背后设计理念的很大一部分，但正如你可以在这里读到的，经济放缓确实会导致挫折。此外，并不是每个人都需要也没有闲心拥有一大群 GPU，或者更重要的是，有时间让他们的模型并发，因为这需要仔细的(重新)设计和努力。Tensorflow 确实有一个 [InMemoryEvaluatorHook](https://www.tensorflow.org/versions/r1.9/api_docs/python/tf/contrib/estimator/InMemoryEvaluatorHook) 来克服这个问题。我试过了，效果不错，但感觉更像是一种变通办法，而不是真正的解决办法。*

*此外，当我尝试从估计器模型函数中加载我的 Keras 模型时，我花了一些时间才意识到[必须在每次训练或评估调用后手动清除 Keras 模型](https://github.com/tensorflow/tensorflow/issues/14356)。真尴尬。*

*这些东西并不是真正的亮点，但加上学习 Tensorflow 如何工作的冲动，它们足以说服我创建自己的微抽象。*

*随着 Tensorflow 2.0 的出现，我相信大部分我纠结的事情都会迎刃而解。Keras 将被集成到 Tensorflow 的核心，并成为其主要接口之一。估值器仍然是首选。如果你想了解更多关于 Tensorflow 2.0 的信息，请查看这个[博客](https://medium.com/tensorflow/whats-coming-in-tensorflow-2-0-d3663832e9b8)和这个[视频](https://youtu.be/k5c-vg4rjBw)。*

# *惨痛的教训*

*我不敢相信我在做这个的时候犯了多少错误。有些很傻，很容易得到，但有些真的很难发现。以下是我学到的一些可能对你有用的经验:*

*   *双重、三重和四重检查你的评估/培训指标的语义、解释和正确性。例如，我的模型从一开始就获得了 100%的准确率。这并不是因为模型超级准确，而是因为这个指标只考虑了那些模型正确预测有小费的样本。如果 10000 个样本中只有 5 个样本检测到正确的 tip，100%的准确性仍然意味着在 10px 内只检测到 5 个图像。*
*   *特别是 tf.metrics API 不止一次愚弄了我。明智地使用 tf.metrics。它们用于评估，即汇总多个批处理操作和整个评估数据集的结果。确保在适当的时候重置它们的状态。*
*   *如果在 Tensorflow 中使用批量定额，不要忘记在训练期间更新移动平均值和方差。这些更新操作自动存储在 tf 中。UPDATE_OPS 集合，所以不要忘记运行它们。*

*Two code examples on how to update moving mean and variance when performing batch norm in Tensorflow.*

*   *编写单元测试作为一个健康检查，或者至少将您快速和肮脏的测试脚本保存到一个单独的文件中，以供以后参考。彻底测试损失函数尤其有意义。*
*   *每次训练模型时，请确保所有输出指标和其他数据都保存在一个唯一的、带有时间标签的目录中。此外，存储 git 标签(例如 heads/master-0-g5936b9e)。这样，无论何时你搞乱了模型，它都会帮助你恢复到以前的工作版本。*

*Example code on how to write the git description to a file.*

*   *将您的指标写入 Tensorboard，用于培训和评估。这是非常值得的，因为可视化让你对你的工作表现有了深入的了解。这有一些挑战，但反过来你可以更快地迭代和测试你的想法。*
*   *在 TensorBoard 中跟踪所有可训练变量，以帮助您尽早发现爆炸或消失的梯度。这里有一些如何做到这一点的灵感。*

*Example code on how to visualize the mean value and a histogram for each trainable variables in the model and an overall histogram for all trainable variables.*

*   *尝试自动并定期暂停训练过程以评估模型。确保在 Tensorboard 中将训练曲线和评估曲线渲染到同一个图形中。这样，您就可以可视化模型在训练过程中从未见过的数据上的性能，并在发现问题时立即停止。请注意，您不能通过简单地重用同一个标签在同一个图中显示多个汇总。Tensorboard 将通过在标签名称中添加“_1”来自动使这些摘要具有唯一性，从而强制它们显示在单独的绘图中。如果您想解决这个限制，您可以自己生成总结协议缓冲区，然后手动将它们添加到总结中。FileWriter()。有点尴尬，但很管用。*

*Example on how to save a metric with the tag “metrics/loss” during evaluation whilst, during training, a metric with the very same tag was used. This allows having both the training and evaluation curves shown on the same graph in Tensorboard.*

*   *监控 GPU 利用率和内存消耗，并尝试获得尽可能高的 GPU 利用率。如果你使用 NVIDIA 显卡，你可以使用 nvidia-smi 命令来完成。您还可以使用 htop 监控 CPU 和内存消耗。*

# *五金器具*

*   *NVIDIA Geforce RTX2080TI (11GB，4352 个 Cuda 内核，600 瓦，INNO3D 游戏 OC X3)*
*   *超微 X9DRG-QF 双 CPU 主板*
*   *2 枚英特尔至强 E5–2630(12 核)*
*   *三星 860 EVO SD(500 克)*
*   *128 克内存*

# *结论和下一步措施*

*当我开始写这篇文章的时候，我的目标是一个简短但内容丰富的信息。我一点也不知道，这将是这样一个怪物的职位，很抱歉。这个项目让我接触了监督学习算法的许多方面，这可能解释了它的大小。*

*利用给定的评估数据，该模型能够在 98%的时间内精确定位卫星的尖端，误差为一个像素。我对这些结果很满意。然而，如果我想将这款车型投入生产，我还有很多事情要做。当前的数据集太有限，无法训练一个鲁棒的模型:图像彼此非常相似，并且只覆盖了少量不同的姿态。由于我没有机会获得更多的数据，一位同事决定帮助并使用 unity 渲染卫星图像。这篇文章顶部的麦哲伦太空探测器的图片就是这个数据集的一个例子。我意识到这不会产生一个现实的模型，但它将允许我继续我的学习过程。*

*此外，仅识别卫星的一个点不足以精确计算相对于观测相机的姿态和距离。理论上，必须跟踪至少 3 个点，而在实践中，该模型应该预测更多的点以获得可靠的输出。*

*您已经完成了这篇文章，感谢您的阅读！我希望它能激励你从事自己的人工智能项目和挑战。*