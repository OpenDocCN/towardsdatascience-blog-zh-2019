# TensorFlow 实践教程:使用 ImageNet 数据集从头开始训练 ResNet-50

> 原文：<https://towardsdatascience.com/hands-on-tensorflow-tutorial-train-resnet-50-from-scratch-using-the-imagenet-dataset-850aa31a39c0?source=collection_archive---------6----------------------->

![](img/1994030d5cc04a1154b1ac705d394c62.png)

在这篇博客中，我们给出了如何在 TensorFlow 中训练 ResNet 模型的快速指南。虽然官方的 TensorFlow 文档确实有您需要的基本信息，但它可能不会马上完全有意义，并且可能有点难以筛选。

我们在这里展示了一个循序渐进的培训过程，同时记录了最佳实践、技巧、诀窍，甚至是我们在开展培训过程中遇到并最终克服的一些挑战。

我们涵盖您需要做的一切，从启动 TensorFlow、下载和准备 ImageNet，一直到记录和报告培训。所有的实验和训练都是在 [Exxact Valence 工作站](https://www.exxactcorp.com/Exxact-VWS-1542881-DPN-E1542881)上完成的，使用了 2 个英伟达 RTX 2080 Ti GPU。

# 有什么意义？我就不能用迁移学习吗？

是的，但是本教程是使用大型数据集(ImageNet)从头开始训练大型神经网络的好练习。虽然迁移学习是一件美妙的事情，并且您可以下载 ResNet-50 的预培训版本，但以下是您可能想要进行此培训练习的一些令人信服的原因:

1.  如果你完成了本教程，你已经有效地训练了一个神经网络，可以作为一个通用的图像分类器。
2.  有了合适的流程，您可以根据自己的数据训练网络。例如，假设您想要训练一个可以对医学图像进行分类的网络。如果图像经过适当的预处理，根据您的数据训练的网络应该能够对这些图像进行分类。
3.  如果您有许多独特的训练数据，从头开始训练网络应该比一般的预训练网络具有更高的准确性。
4.  您可以专门针对您的数据调整训练参数。
5.  在预训练的模型上，检查点是脆弱的，并且不能保证与未来版本的代码一起工作。

虽然迁移学习是一种强大的知识共享技术，但知道如何从头开始训练仍然是深度学习工程师的必备技能。所以现在，让我们开始吧。

# 步骤 1)运行 TensorFlow Docker 容器。

首先，您需要启动 TensorFlow 环境。我们喜欢和 Docker 一起工作，因为它给了我们极大的灵活性和可复制的环境。打开一个终端窗口，让我们开始吧！

*注意:一定要指定你的-v 标签，以便在容器内创建一个交互卷。*

```
nvidia-docker run -it -v /data:/datasets tensorflow/tensorflow:nightly-gpu bash
```

**或**如果您计划在 docker 容器中启动 Tensorboard，请确保指定 **-p 6006:6006** 并使用以下命令。

```
nvidia-docker run -it -v /data:/datasets -p 6006:6006 tensorflow/tensorflow:nightly-gpu bash
```

# 步骤 2)下载并预处理 ImageNet 数据集。

我们决定包括这一步，因为它似乎会引起一点混乱。注意:当你做这一步时，你要确保你有 300+ GB 的存储空间(正如我们发现的),因为下载&预处理步骤需要这个！

2.1)对于第一个子步骤，如果您的环境中没有 git，您需要安装它。

```
apt-get install git
```

2.2)其次，您必须将 TPU 回购克隆到您的环境中(不，我们没有使用谷歌的 TPU，但是这里包含了基本的预处理脚本！)

```
git clone [https://github.com/tensorflow/tpu.git](https://github.com/tensorflow/tpu.git)
```

2.3)第三，您需要安装 GCS 依赖项(即使您没有使用 GCS，您仍然需要运行它！)

```
pip install gcloud google-cloud-storage
```

2.4)最后，您将需要运行 **imagenet_to_gcs.py** 脚本，该脚本从 Image-Net.org 下载文件，并将它们处理成 TFRecords，但不将它们上传到 gcs(因此有了**‘nogcs _ upload’**标志)。同样 **'local_scratch_dir='** 应该指向您想要保存数据集的位置。

```
python imagenet_to_gcs.py --local_scratch_dir=/data/imagenet --nogcs_upload
```

*注意:ImageNet 非常大，根据您的连接情况，可能需要几个小时(可能是一夜)才能下载完整的数据集！*

# 步骤 3)下载 TensorFlow 模型。

这一步是显而易见的，如果你没有模型，克隆回购使用:

```
git clone [https://github.com/tensorflow/models.git](https://github.com/tensorflow/models.git)
```

# 步骤 4)导出 PYTHONPATH。

将 PYTONPATH 导出到计算机上 models 文件夹所在的文件夹。下面的命令是模型在我的机器上的位置！确保用模型文件夹的数据路径替换 **'/datasets/models'** 语法！

```
export PYTHONPATH="$PYTHONPATH:/datasets/models"
```

# 第 5 步)安装依赖项(您差不多准备好了！)

导航到 models 文件夹(如果您还不在那里)并运行以下命令

```
pip install --user -r official/requirements.txt
```

**或**如果你使用 Python3

```
pip3 install --user -r official/requirements.txt
```

***重要提示:你已经准备好训练了！根据我们的经验，为了让训练脚本正确运行，您需要从验证文件夹中复制(或移动)数据，并将其移动到训练文件夹中！！！***

# 步骤 6)设置训练参数，训练 ResNet，坐好，放松。

运行训练脚本 python**imagenet _ main . py**，设置训练参数。下面是我用来训练 ResNet-50 的，120 个训练历元对这个练习来说太多了，但我们只是想推动我们的 GPU。根据您的计算能力，在完整数据集上训练可能需要几天时间！

```
python imagenet_main.py --data_dir=/data/imagenet/train --num_gpus= 2 --batch_size=64 --resnet_size= 50 --model_dir=/data/imagenet/trained_model/Resnet50_bs64 --train_epochs=120
```

***关于训练参数的注意事项:注意有许多不同的选项可以指定，包括:***

上面提到的只是模型训练可用的一些选项。请参见[**resnet _ run _ loop . py**](https://github.com/tensorflow/models/blob/master/official/resnet/resnet_run_loop.py)查看选项的完整列表(您必须仔细阅读代码)。

# 你完了！现在让我们在 TensorBoard 中查看结果！

您也可以使用 TensorBoard 查看您的结果:

```
tensorboard --logdir=/data/imagenet/trained_model/Resnet50_bs64
```

# 张量板输出

如果您正确运行了上面的步骤(并使用了相似的参数)，您应该会得到类似的结果。注意，这些结果与官方 TensorFlow 结果相当。让我们看看你能做什么！

**精度**

![](img/2319e1839a1f0247a1a13ee1251751d7.png)

**训练 _ 精度 _1**

![](img/a5824856ab13c425ff6e72966edb63be.png)

**准确度 _ 最高 _5**

![](img/8d6ce67e689a698053e2ebcea107984b.png)

**train_accuracy_top_5_1**

![](img/3bc1326b66d46291eebd1744e44af20c.png)

**损失**

![](img/99c1f6238056fe8240a937565b8e9415.png)

**L2 _ 损失**

![](img/592f43c13689da6a1321c52a37cf6b53.png)

**cross_entropy_1**

![](img/dcfdcfeb56ba7f0ab057f48c82f8c3e4.png)

**learning_rate_1**

![](img/ff978ff11c22291cffdebf3054597f73.png)

**秒**

![](img/d8d0f3f1e550312c76b05820f7405754.png)

# 最后的想法

差不多就是这样！如果您在 ResNet 培训中有任何问题，请告诉我们。还有，你在 TensorFlow 中训练模型的时候用了哪些技巧和窍门？让我知道！

*原载于 2019 年 3 月 26 日*[*https://blog.exxactcorp.com*](https://blog.exxactcorp.com/deep-learning-with-tensorflow-training-resnet-50-from-scratch-using-the-imagenet-dataset/)*。*