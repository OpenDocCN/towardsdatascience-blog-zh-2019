# 使用 NVIDIA flownet2-pytorch 实现生成光流

> 原文：<https://towardsdatascience.com/generating-optical-flow-using-nvidia-flownet2-pytorch-implementation-d7b0ae6f8320?source=collection_archive---------5----------------------->

## 视频分类算法中使用的光流文件创建指南

![](img/71edc921aeb1895babe3a36f8f4bbe3b.png)

这篇博客最初发表于[blog.dancelogue.com](https://blog.dancelogue.com/generating-optical-flow-using-flownet-for-human-action-deep-learning-algorithms/)。在之前的[帖子](https://blog.dancelogue.com/what-is-optical-flow-and-why-does-it-matter/)中，进行了光流的介绍，以及基于 [FlowNet 2.o 论文](https://arxiv.org/abs/1612.01925)的光流架构概述。本博客将重点深入光流，这将通过从标准 Sintel 数据和自定义舞蹈视频生成光流文件来完成。将使用 [NVIDIA flownet2-pytorch](https://github.com/NVIDIA/flownet2-pytorch) 代码库的 [fork](https://github.com/dancelogue/flownet2-pytorch) 进行，该代码库可在 [Dancelogue 关联回购](https://github.com/dancelogue/flownet2-pytorch)中找到。

本博客的目标是:

*   启动并运行 flownet2-pytorch 代码库。
*   下载相关数据集，如原始存储库中提供的示例所述。
*   生成光流文件，然后研究光流文件的结构。
*   将流文件转换为颜色编码方案，使其更易于人类理解。
*   将光流生成应用于舞蹈视频并分析结果。

# 系统需求

flownet2-pytorch 实现被设计为与 GPU 一起工作。不幸的是，这意味着如果你没有访问权限，就不可能完全关注这个博客。为了缓解这个问题，我们提供了模型生成的样本数据，并允许读者继续阅读博客的其余部分。

本教程的其余部分是使用 ubuntu 18.04 和 NVIDIA GEFORCE GTX 1080 Ti GPU 进行的。Docker 是必需的，并且必须支持 GPU，这可以通过使用[NVIDIA-docker](https://github.com/NVIDIA/nvidia-docker)r 包来实现。

# 下载代码库和数据集

这里列出了继续写博客所需的所有代码和数据(下载数据是自动进行的，因此读者不必手动完成，请参见入门部分):

*   这个博客的代码库可以从下面的 [repo](https://github.com/dancelogue/flownet2-pytorch) 中克隆。
*   点击下面的[链接](http://files.is.tue.mpg.de/sintel/MPI-Sintel-complete.zip)可以下载 Sintel 数据，压缩后的文件是 5.63 GB，解压后增加到 12.24 GB。
*   可以下载自定义数据，包括[样本光流](https://dancelogue.s3.amazonaws.com/open_source/datasets/generating-optical-flow-using-flownet-for-human-action-deep-learning-algorithms/000835.flo) `.flo`文件，从样本光流文件生成的[颜色编码方案](https://dancelogue.s3.amazonaws.com/open_source/datasets/generating-optical-flow-using-flownet-for-human-action-deep-learning-algorithms/000835.flo.png)，进行光流的[舞蹈视频](https://dancelogue.s3.amazonaws.com/open_source/datasets/generating-optical-flow-using-flownet-for-human-action-deep-learning-algorithms/sample-video.mp4)，舞蹈视频的[光流视频表示](https://dancelogue.s3.amazonaws.com/open_source/datasets/generating-optical-flow-using-flownet-for-human-action-deep-learning-algorithms/sample-optical-flow-video.mp4)。

完成这篇博客所需的内存空间大约是 32 GB。其原因将在后面解释。

# 叉子的差异

如前所述，创建了原始 flownet2-pytorch 的一个分支，这是因为在撰写本博客时，原始存储库在构建和运行 docker 映像时出现了问题，例如 python 包版本问题、c 库编译问题等。这些更新包括:

*   通过修复 python 包版本、更新 cuda 和 pytorch 版本、运行相关层的自动构建和安装、添加 ffmpeg、添加第三方 github 包来修改 Dockerfile，该第三方 github 包将允许流文件的读取、处理和转换到颜色编码方案。
*   为数据集和训练模型编写下载脚本以便更容易开始，其灵感来自 NVIDIA 的 [vid2vid](https://github.com/NVIDIA/vid2vid) 存储库。

# 入门指南

考虑到这一点，我们开始吧。第一件事是从[https://github.com/dancelogue/flownet2-pytorch](https://github.com/dancelogue/flownet2-pytorch)中克隆原始存储库的 dancelogue 分支。然后使用以下命令运行 docker 脚本:

```
bash launch_docker.sh
```

设置应该需要几分钟时间，之后，应该将终端上下文更改为 docker 会话。

接下来是下载相关的数据集，初始设置所需的所有数据都可以通过在 docker 上下文中运行以下命令来获得:

```
bash scripts/download.sh
```

这会将`FlowNet2_checkpoint.pth.tar`模型权重下载到 models 文件夹，并将`MPI-Sintel`数据下载到 datasets 文件夹。这是必需的，以便遵循《flownet2-pytorch 入门指南》中指示的推理示例的说明。定制的舞蹈视频以及样本光流`.flo`文件也被下载。

本博客中的其余命令已经自动化，可以通过以下方式运行:

```
bash scripts/run.sh
```

# 运行推理示例

运行原始推理示例的命令如下:

```
python main.py --inference --model FlowNet2 --save_flow \ 
--inference_dataset MpiSintelClean \
--inference_dataset_root /path/to/mpi-sintel/clean/dataset \
--resume /path/to/checkpoints
```

但是，根据 fork，这已修改为:

```
python main.py --inference --model FlowNet2 --save_flow \ 
--inference_dataset MpiSintelClean \
--inference_dataset_root datasets/sintel/training \
--resume checkpoints/FlowNet2_checkpoint.pth.tar \
--save datasets/sintel/output
```

让我们来分解一下:

*   `--model`表示要使用的型号变体。从[之前的博客](https://blog.dancelogue.com/what-is-optical-flow-and-why-does-it-matter/)我们看到这个可以是`FlowNetC`、`FlowNetCSS`或`FlowNet2`，但是对于这个博客它被设置为`FlowNet2`。
*   `--resume`参数指示训练模型权重的位置。已使用下载脚本将其下载到检查点文件夹中。请注意，训练模型权重有一定的许可限制，如果您需要在本博客之外使用它们，您应该遵守这些限制。
*   `--inference`参数简单地意味着，基于由来自训练数据的模型权重定义的学习能力，你能告诉我关于新数据集的什么。这不同于训练模型，其中模型权重将改变。
*   `--inference_dataset`表示将输入何种类型的数据。在当前情况下，它是由`MpiSintelClean`指定的 sintel。更多选项可以在[https://github . com/dance Logue/flownet 2-py torch/blob/master/datasets . py](https://github.com/dancelogue/flownet2-pytorch/blob/master/datasets.py)中找到，并被定义为类，例如`FlyingChairs`。还有一个`ImagesFromFolder`类，这意味着我们可以输入自定义数据，例如来自视频的帧，我们可以从中得出推论。
*   `--inference_dataset_root`表示将用于推理过程的数据的位置，该数据已被下载并解压缩到`datasets/sintel`文件夹中。
*   `--save_flow`参数表示推断的光流应该保存为`.flo`文件。
*   `--save`参数指示推断的光流文件以及日志应该保存到的位置。这是一个可选字段，默认为`work/`位置。

运行上述命令将生成的光流文件保存到`datasets/sintel/output/inference/run.epoch-0-flow-field`文件夹中。生成的光流文件的扩展名为`.flo`，是流场的表示。

# 分析和可视化光流文件

现在光流文件已经生成，是时候分析结构了，以便更好地理解结果，并将其转换为流场颜色编码方案。本节使用的样本流文件可从以下[链接](https://dancelogue.s3.amazonaws.com/open_source/datasets/generating-optical-flow-using-flownet-for-human-action-deep-learning-algorithms/000835.flo)下载。

# 分析流文件

将光流文件加载到 numpy 是一个相当简单的过程，可以按如下方式进行:

```
path = Path('path/to/flow/file/<filename>.flo')
with path.open(mode='r') as flo:
    np_flow = np.fromfile(flo, np.float32)
    print(np_flow.shape)
```

上面的语法基于 python3，其中文件被加载到一个缓冲区中，然后被送入 numpy。接下来的事情是试图理解由打印语句实现的流文件的基本特性。假设您使用提供的样本流文件，print 语句应该输出`(786435,)`。这意味着对于每个流文件，它包含一个数组，数组中有 786453 个元素。单个流文件的内存占用大约为 15.7 MB，尽管看起来很小，但增长非常快，尤其是在查看具有数千帧的视频时。

在继续之前，我们需要看看 http://vision.middlebury.edu/flow/code/flow-code/README.txt[中定义的光流规范。我们关心的是以下内容:](http://vision.middlebury.edu/flow/code/flow-code/README.txt)

```
".flo" file format used for optical flow evaluationStores 2-band float image for horizontal (u) and vertical (v) flow components.
Floats are stored in little-endian order.
A flow value is considered "unknown" if either |u| or |v| is greater than 1e9. bytes  contents 0-3     tag: "PIEH" in ASCII, which in little endian happens to be the float 202021.25
          (just a sanity check that floats are represented correctly)
  4-7     width as an integer
  8-11    height as an integer
  12-end  data (width*height*2*4 bytes total)
          the float values for u and v, interleaved, in row order, i.e.,
          u[row0,col0], v[row0,col0], u[row0,col1], v[row0,col1], ...
```

基于上述规范，下面的代码将允许我们正确地读取流文件(借用自[https://github . com/georgegach/flowiz/blob/master/flowiz/flowiz . p](https://github.com/georgegach/flowiz/blob/master/flowiz/flowiz.py#L25)y)。

```
with path.open(mode='r') as flo:
  tag = np.fromfile(flo, np.float32, count=1)[0]
  width = np.fromfile(flo, np.int32, count=1)[0]
  height = np.fromfile(flo, np.int32, count=1)[0] print('tag', tag, 'width', width, 'height', height) nbands = 2
  tmp = np.fromfile(flo, np.float32, count= nbands * width * height)
  flow = np.resize(tmp, (int(height), int(width), int(nbands)))
```

基于光流格式规范，希望上面的代码对正在发生的事情更有意义，即我们得到标签，然后是宽度，接着是高度。打印语句的输出是`tag 202021.25 width 1024 height 384`。从给定的规范中，我们可以看到标签匹配健全性检查值，流文件的宽度是 1024，高度是 384。请注意，在读取文件缓冲区并将其加载到 numpy 时，正确的顺序很重要，这是因为 python 中读取文件的方式(字节是顺序读取的)，否则标签、高度和宽度会混淆。现在我们已经有了宽度和高度，我们可以读取其余的光流数据，并将其调整为更熟悉的形状，这是使用`np.resize`方法完成的。

理解流向量如何被调整大小的一个快速方法是将它们打印到终端，这是通过运行以下代码来完成的:

```
>> print(flow.shape)
(384, 1024, 2)>> print(flow[0][0])
[-1.2117167 -1.557275]
```

正如我们所料，新表示的形状意味着高度为 384，宽度为 1024，并且具有由 2 个值组成的位移向量。关注位置`0, 0`处的像素，我们可以看到该点的位移向量似乎指向左侧和底部，即 x，y 图的左下象限，这意味着我们预计该位置的颜色代码是浅蓝色，甚至是基于下面给出的颜色编码方案的绿色。

![](img/4c6696b9b53a093c1eefb8911db8faa7.png)

# 可视化流文件

有不少开源代码库是为了可视化光流文件而编写的。为此选择的那个可以在 github 库[https://github.com/georgegach/flowiz](https://github.com/georgegach/flowiz)中找到。这样做的原因是，它允许从颜色编码方案中生成视频剪辑，这在稍后阶段将是有用的。假设使用了本教程开头提供的 docker 上下文，可以使用以下命令来生成光流的彩色编码图像文件。

```
python -m flowiz \
datasets/sintel/output/inference/run.epoch-0-flow-field/*.flo \
-o datasets/sintel/output/color_coding \
-v datasets/sintel/output/color_coding/video \
-r 30
```

这将获取光流文件并生成图像文件，其中的位移向量用颜色编码，如下所示。

![](img/ee9e2653da065c1cdbf41fb3a50b14ac.png)

为了理解颜色编码方案，请查看[之前关于光流的博客](https://blog.dancelogue.com/what-is-optical-flow-and-why-does-it-matter/)。在位置 0，0，即图像的右下部分，我们确实可以看到浅蓝色，这是我们从位移向量中预期的颜色，即它是指向左侧和底部的向量的颜色。

# 将光流应用于舞蹈视频

在这一节中，我们将使用一个舞蹈视频，并从中生成光流文件。舞蹈视频是:

它包括一个真实世界中的舞蹈编排课程。

# 生成帧

当 flownet 代码库接收图像时，我们需要做的第一件事是将视频转换成帧，这可以通过以下使用 ffmpeg 的命令来完成。

```
ffmpeg -i datasets/dancelogue/sample-video.mp4 \
datasets/dancelogue/frames/output_%02d.png
```

它将在帧文件夹中按顺序输出帧，顺序很重要，因为 flownet 算法使用相邻图像来计算图像之间的光流。生成的帧占用 1.7 GB 的内存，而视频只有 11.7 MB，每帧大约 2 MB。

# 产生光流

光流表示可以通过运行以下命令来生成。

```
python main.py --inference --model FlowNet2 --save_flow \
--inference_dataset ImagesFromFolder \
--inference_dataset_root datasets/dancelogue/frames/ \
--resume checkpoints/FlowNet2_checkpoint.pth.tar \
--save datasets/dancelogue/output
```

这类似于我们在 sintel 数据集上运行的推理模型，不同之处在于从`--inference_dataset`参数变为`ImagesFromFolder`，并在[代码库](https://github.com/dancelogue/flownet2-pytorch/blob/master/datasets.py#L320)中定义。`--inference_dataset_root`是生成的视频帧的路径。生成的光流文件占用 14.6 GB 的存储器，这是因为对于这个例子，每个光流文件大约为 15.7 MB。

# 生成颜色编码方案

生成颜色编码方案的命令是:

```
python -m flowiz \
datasets/dancelogue/output/inference/run.epoch-0-flow-field/*.flo \
-o datasets/dancelogue/output/color_coding \
-v datasets/dancelogue/output/color_coding/video \
-r 30
```

这使用了 [flowviz 库](https://github.com/georgegach/flow2image)和 ffmpeg。它不仅将光流颜色编码生成为`.png`文件，而且`-v -r 30`参数在`30 fps`从图像文件生成视频。生成的彩色编码帧占用 422 MB 的内存，其中包括一个 8.7 MB 的视频文件，如果你正在浏览这个博客，它的名称是`000000.flo.mp4`。

# 结果

生成的光流的视频表示如下:

舞蹈动作的要点可以从生成的视频中看出，不同的颜色表示动作的方向。然而，尽管视频中没有明显的运动，但可以看到有很多背景噪音，特别是在中心舞者周围。不幸的是，不清楚为什么会这样。

# 尺寸影响

当运行 flownet 算法时，需要注意大小含义，例如，一个 11.7 MB 的视频在提取时会生成一个 1.7 GB 的单个帧文件。然而，当生成光流时，这变成包含所有光流表示的 14.6 GB 文件。这是因为每个光流文件占用大约 15.7 MB 的存储器，然而每个图像帧占用 2 MB 的存储器(对于所提供的例子的情况)。因此，当运行光流算法时，需要注意计算需求与空间的权衡。在为视频构建深度学习系统时，这种权衡将影响架构，这意味着要么以计算时间为代价按需(即，懒惰地)生成光流文件，要么以存储空间为代价提前生成所有需要的格式和表示并将其保存到文件系统。

# 结论

我们已经看到了如何使用 NVIDIA 的 flownet2-pytorch 实现的分支来生成光流文件，并对光流文件有了一个大致的了解。下一篇博客将介绍如何使用光流表示来理解视频内容，并将重点放在 2 流网络上。

如果您有任何问题或需要澄清的事情，您可以在[https://mbele.io/mark](https://mbele.io/mark)和我预约时间

# 参考

*   [https://blog . dance Logue . com/什么是光流，为什么有关系/](https://blog.dancelogue.com/what-is-optical-flow-and-why-does-it-matter/)
*   [https://blogs . NVIDIA . com/blog/2016/08/22/差异-深度学习-训练-推理-ai/](https://blogs.nvidia.com/blog/2016/08/22/difference-deep-learning-training-inference-ai/)
*   [https://stack overflow . com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy](https://stackoverflow.com/questions/28013200/reading-middlebury-flow-files-with-python-bytes-array-numpy)