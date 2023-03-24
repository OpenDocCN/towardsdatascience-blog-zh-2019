# 将简单的深度学习模型从 PyTorch 转换为 TensorFlow

> 原文：<https://towardsdatascience.com/converting-a-simple-deep-learning-model-from-pytorch-to-tensorflow-b6b353351f5d?source=collection_archive---------3----------------------->

![](img/8877d9a0ec989904c6a990a19ca8e595.png)

Reference: [https://towardsdatascience.com/applied-deep-learning-part-1-artificial-neural-networks-d7834f67a4f6](/applied-deep-learning-part-1-artificial-neural-networks-d7834f67a4f6)

**简介**

TensorFlow 和 PyTorch 是两个比较流行的深度学习框架。有些人更喜欢 TensorFlow 以获得部署方面的支持，有些人更喜欢 PyTorch，因为它在模型构建和培训方面具有灵活性，而没有使用 TensorFlow 所面临的困难。使用 PyTorch 的缺点是，使用该框架构建和训练的模型不能部署到生产中。(2019 年 12 月更新:据称 PyTorch 的后续版本对部署有更好的支持，但我相信这是有待探索的其他事情。)为了解决部署使用 PyTorch 构建的模型的问题，一种解决方案是使用 ONNX(开放式神经网络交换)。

正如 ONNX 的[关于第](https://onnx.ai/about)页所解释的，ONNX 就像一座桥梁，将各种深度学习框架连接在一起。为此，ONNX 工具支持模型从一个框架到另一个框架的转换。到撰写本文时为止，ONNX 仅限于更简单的模型结构，但以后可能会有进一步的补充。本文将说明如何将一个简单的深度学习模型从 PyTorch 转换为 TensorFlow。

**安装必要的软件包**

首先，我们需要安装 PyTorch、TensorFlow、ONNX 和 ONNX-TF(将 ONNX 模型转换为 TensorFlow 的包)。如果在 Linux 中使用 *virtualenv* ，你可以运行下面的命令(如果你安装了 NVidia CUDA，用 *tensorflow-gpu* 替换 *tensorflow* )。请注意，截至 2019 年 12 月，ONNX 尚不支持 TensorFlow 2.0，因此请注意您安装的 TensorFlow 版本。

```
source <your virtual environment>/bin/activate
pip install tensorflow==1.15.0# For PyTorch, choose one of the following (refer to [https://pytorch.org/get-started/locally/](https://pytorch.org/get-started/locally/) for further details)
pip install torch torchvision # if using CUDA 10.1
pip install torch==1.3.1+cu92 torchvision==0.4.2+cu92 -f [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html) # if using CUDA 9.2
pip install torch==1.3.1+cpu torchvision==0.4.2+cpu -f [https://download.pytorch.org/whl/torch_stable.html](https://download.pytorch.org/whl/torch_stable.html) # if using CPU onlypip install onnx# For onnx-tensorflow, you may want to refer to the installation guide here: [https://github.com/onnx/onnx-tensorflow](https://github.com/onnx/onnx-tensorflow)
git clone [https://github.com/onnx/onnx-tensorflow.git](https://github.com/onnx/onnx-tensorflow.git)
cd onnx-tensorflow
pip install -e ..
```

如果使用 Conda，您可能希望改为运行以下命令:

```
conda activte <your virtual environment>
conda install -c pytorch pytorchpip install tensorflow==1.15.0pip install onnx# For onnx-tensorflow, you may want to refer to the installation guide here: [https://github.com/onnx/onnx-tensorflow](https://github.com/onnx/onnx-tensorflow)
git clone [https://github.com/onnx/onnx-tensorflow.git](https://github.com/onnx/onnx-tensorflow.git)
cd onnx-tensorflow
pip install -e ..
```

我发现使用 pip 安装 TensorFlow、ONNX 和 ONNX-TF 将确保这些包相互兼容。但是，也可以使用其他方式安装软件包，只要它们能在您的机器上正常工作。

要测试软件包是否已正确安装，可以运行以下命令:

```
python
import tensorflow as tf
import torch
import onnx
from onnx_tf.backend import prepare
```

如果您没有看到任何错误消息，这意味着软件包安装正确，我们可以开始了。

在这个例子中，我使用了 Jupyter Notebook，但是转换也可以在. py 文件中完成。要安装 Jupyter Notebook，您可以运行以下命令之一:

```
# Installing Jupyter Notebook via pip
pip install notebook# Installing Jupyter Notebook via Conda
conda install notebook
```

**构建、培训和评估示例模型**

接下来要做的是在 PyTorch 中获得一个可用于转换的模型。在这个例子中，我生成了一些模拟数据，并使用这些数据来训练和评估一个简单的多层感知器(MLP)模型。下面的代码片段展示了如何导入已安装的包，以及如何生成和准备数据。

然后，我为简单的 MLP 模型创建了一个类，并定义了层，这样我们就可以指定任意数量和大小的隐藏层。我还定义了一个二进制交叉熵损失和 Adam 优化器，用于计算训练期间的损失和权重更新。下面的代码片段展示了这个过程。

在构建模型并定义损失和优化器之后，我使用生成的训练集对模型进行了 20 个时期的训练，然后使用测试集进行评估。模型的测试损失和准确性不好，但这在这里并不重要，因为这里的主要目的是展示如何将 PyTorch 模型转换为 TensorFlow。下面的片段显示了培训和评估过程。

在训练和评估模型之后，我们需要保存模型，如下所示:

**将模型转换为张量流**

现在，我们需要转换*。pt* 文件到了一个*。onnx* 文件使用 *torch.onnx.export* 函数。这里我们需要注意两件事:1)我们需要定义一个虚拟输入作为导出函数的输入之一，2)虚拟输入需要具有形状(1，单个输入的维度)。例如，如果单个输入是具有形状(通道数、高度、宽度)的图像数组，那么伪输入需要具有形状(1，通道数、高度、宽度)。需要虚拟输入作为所得张量流模型的输入占位符)。下面的代码片段显示了以 ONNX 格式导出 PyTorch 模型的过程。我还将输入和输出名称作为参数，以便在 TensorFlow 中更容易进行推理。

拿到*后。onnx* 文件，我们需要使用 onnx-TF 的*后端*模块中的 *prepare()* 函数将模型从 ONNX 转换为 TensorFlow。

**在 TensorFlow 中做推理**

有趣的部分来了，这是为了看看合成的张量流模型是否能按预期进行推理。从*加载张量流模型。pb* 文件可以通过定义以下函数来完成。

定义了加载模型的函数后，我们需要启动一个 TensorFlow graph 会话，为输入和输出指定占位符，并将输入输入到会话中。

上面代码片段的输出如下所示。占位符的名称对应于在`torch.onnx.export`功能中指定的名称(用粗体表示)。

```
(<tf.Tensor 'Const:0' shape=(50,) dtype=float32>,)
(<tf.Tensor 'Const_1:0' shape=(50, 20) dtype=float32>,)
(<tf.Tensor 'Const_2:0' shape=(50,) dtype=float32>,)
(<tf.Tensor 'Const_3:0' shape=(50, 50) dtype=float32>,)
(<tf.Tensor 'Const_4:0' shape=(1,) dtype=float32>,)
(<tf.Tensor 'Const_5:0' shape=(1, 50) dtype=float32>,)
(**<tf.Tensor 'input:0' shape=(1, 20) dtype=float32>,)**
(<tf.Tensor 'flatten/Reshape/shape:0' shape=(2,) dtype=int32>,)
(<tf.Tensor 'flatten/Reshape:0' shape=(1, 20) dtype=float32>,)
(<tf.Tensor 'transpose/perm:0' shape=(2,) dtype=int32>,)
(<tf.Tensor 'transpose:0' shape=(20, 50) dtype=float32>,)
(<tf.Tensor 'MatMul:0' shape=(1, 50) dtype=float32>,)
(<tf.Tensor 'mul/x:0' shape=() dtype=float32>,)
(<tf.Tensor 'mul:0' shape=(1, 50) dtype=float32>,)
(<tf.Tensor 'mul_1/x:0' shape=() dtype=float32>,)
(<tf.Tensor 'mul_1:0' shape=(50,) dtype=float32>,)
(<tf.Tensor 'add:0' shape=(1, 50) dtype=float32>,)
(<tf.Tensor 'Relu:0' shape=(1, 50) dtype=float32>,)
(<tf.Tensor 'flatten_1/Reshape/shape:0' shape=(2,) dtype=int32>,)
(<tf.Tensor 'flatten_1/Reshape:0' shape=(1, 50) dtype=float32>,)
(<tf.Tensor 'transpose_1/perm:0' shape=(2,) dtype=int32>,)
(<tf.Tensor 'transpose_1:0' shape=(50, 50) dtype=float32>,)
(<tf.Tensor 'MatMul_1:0' shape=(1, 50) dtype=float32>,)
(<tf.Tensor 'mul_2/x:0' shape=() dtype=float32>,)
(<tf.Tensor 'mul_2:0' shape=(1, 50) dtype=float32>,)
(<tf.Tensor 'mul_3/x:0' shape=() dtype=float32>,)
(<tf.Tensor 'mul_3:0' shape=(50,) dtype=float32>,)
(<tf.Tensor 'add_1:0' shape=(1, 50) dtype=float32>,)
(<tf.Tensor 'Relu_1:0' shape=(1, 50) dtype=float32>,)
(<tf.Tensor 'flatten_2/Reshape/shape:0' shape=(2,) dtype=int32>,)
(<tf.Tensor 'flatten_2/Reshape:0' shape=(1, 50) dtype=float32>,)
(<tf.Tensor 'transpose_2/perm:0' shape=(2,) dtype=int32>,)
(<tf.Tensor 'transpose_2:0' shape=(50, 1) dtype=float32>,)
(<tf.Tensor 'MatMul_2:0' shape=(1, 1) dtype=float32>,)
(<tf.Tensor 'mul_4/x:0' shape=() dtype=float32>,)
(<tf.Tensor 'mul_4:0' shape=(1, 1) dtype=float32>,)
(<tf.Tensor 'mul_5/x:0' shape=() dtype=float32>,)
(<tf.Tensor 'mul_5:0' shape=(1,) dtype=float32>,)
(<tf.Tensor 'add_2:0' shape=(1, 1) dtype=float32>,)
**(<tf.Tensor 'output:0' shape=(1, 1) dtype=float32>,)**
```

如果一切顺利， *print(output)* 的结果应该与前面步骤中的 *print(dummy_output)* 的结果相匹配。

**结论**

ONNX 可以非常简单，只要你的模型不太复杂。这个例子中的步骤将对具有单个输入和输出的深度学习模型起作用。对于具有多个输入和/或输出的模型，通过 ONNX 进行转换更具挑战性。因此，转换多个输入/输出模型的示例必须在另一篇文章中完成，除非以后有新版本的 ONNX 可以处理这样的模型。

包含所有代码的 Jupyter 笔记本可以在这里找到。