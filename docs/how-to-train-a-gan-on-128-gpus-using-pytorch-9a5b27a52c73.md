# 如何使用 PyTorch 在 128 GPUs 上训练 GAN

> 原文：<https://towardsdatascience.com/how-to-train-a-gan-on-128-gpus-using-pytorch-9a5b27a52c73?source=collection_archive---------12----------------------->

![](img/e85b70b14306a49c0328d6940ba2c066.png)

如果你对 GANs 感兴趣，你会知道生成好看的输出需要很长时间。通过分布式培训，我们可以大幅缩短时间。

在另一个教程中，我介绍了你可以做的 9 件事来加速你的 PyTorch 模型。在本教程中，我们将实现一个 GAN，并使用分布式数据并行在 32 台机器(每台机器有 4 个 GPU)上训练它。

# 发电机

![](img/f26a76785789b4f511cfdaa13c682978.png)

首先，我们需要定义一个生成器。该网络将随机噪声作为输入，并从噪声索引的潜在空间生成图像。

这个生成器也会有自己的优化器

```
opt_g = torch.optim.Adam(self.generator.parameters(), lr=lr, betas=(b1, b2))
```

# 鉴别器

![](img/bdf0b40c678e83dfd4337737a0486268.png)

Real or fake pirate?

鉴别器只是一个分类器。它接受一幅图像作为输入，并判断它是否是真实的。

鉴别器也有自己的优化器。

```
opt_d = torch.optim.Adam(self.discriminator.parameters(), lr=lr, betas=(b1, b2))
```

# 甘培训

甘训练有许多变化，但这是最简单的，只是为了说明主要的想法。

你可以把它看作是发生器和鉴别器之间的交替训练。所以，批次 0 训练发生器，批次 1 训练鉴别器，等等…

为了训练发电机，我们执行以下操作:

Generator Training

从上面我们看到，我们采样一些正常的噪声，并把它给鉴别器，因为它是假的，我们生成假标签，要求鉴别器分类为假的。

我们反向传播上述内容，然后继续下一批训练鉴别器的内容。

为了训练鉴别器，我们进行如下操作

Discriminator Training

鉴别器计算出 2 个损耗。首先，它能多好地检测真实的例子？第二，它能多好地检测假例子？全损刚好是这两个的平均值。

# 照明模块

![](img/302f7ea113b1839064d96053437e194f.png)

PyTorch Lightning

为了在 128 个 GPU 上训练这个系统，我们将在 PyTorch 之上使用一个名为 [PyTorch-Lightning](https://github.com/williamFalcon/pytorch-lightning) 的轻量级包装器，它可以自动完成我们在这里没有讨论的所有事情(训练循环、验证等……)。

这个库的美妙之处在于，你唯一需要定义的是一个 [LightningModule](https://williamfalcon.github.io/pytorch-lightning/LightningModule/RequiredTrainerInterface/) 接口中的系统，你可以获得免费的 GPU 和集群支持。

这是作为[照明模块](https://williamfalcon.github.io/pytorch-lightning/LightningModule/RequiredTrainerInterface/)的整个系统

这个抽象非常简单。培训的实质总是发生在 training_step 中，因此您可以查看世界上的任何存储库，并知道在哪里发生了什么！

让我们一步一步来。

首先，我们定义这个系统将在 __init__ 方法中使用的模型。

接下来，我们定义我们希望系统的输出在。正向法。这意味着，如果我从服务器或 API 运行这个模型，我总是给它噪声，并得到一个采样图像。

现在是系统的核心部分。我们在 training_step 方法中定义任何系统的复杂逻辑，在本例中是 GAN 训练。

我们还缓存我们为鉴别器生成的图像，并记录每一批的示例。

最后，我们配置我们想要的优化器和数据。

就是这样！总之，我们必须明确我们关心的事情:

1.  数据
2.  涉及的模型(初始)
3.  涉及优化人员
4.  整个系统的核心训练逻辑(训练 _ 步骤)
5.  对于可能需要计算精度或使用不同数据集的其他系统，有一个可选的 validation_step。

# 在 128 个 GPU 上进行培训

![](img/33a07180829005da7e034fe16c3a87ab.png)

这部分现在其实很琐碎。定义了 GAN 系统后，我们可以简单地将它传递给一个 [Trainer](https://williamfalcon.github.io/pytorch-lightning/Trainer/) 对象，并告诉它在 32 个节点上训练，每个节点有 4 个 GPU。

现在我们向 SLURM 提交一个作业，它有以下标志:

```
# SLURM SUBMIT SCRIPT
#SBATCH --gres=gpu:4
#SBATCH --nodes=32
#SBATCH --ntasks-per-node=4
#SBATCH --mem=0
#SBATCH --time=02:00:00 # activate conda env
conda activate my_env # run script from above
python gan.py 
```

我们的模型将使用所有 128 个 GPU 进行训练！

在后台，Lightning 将使用 DistributedDataParallel，并配置一切为您正确工作。分布式数据并行在[本教程](/9-tips-for-training-lightning-fast-neural-networks-in-pytorch-8e63a502f565)中有深入解释。

在高层次上，DistributedDataParallel 为每个 GPU 提供数据集的一部分，在该 GPU 上初始化模型，并在训练期间仅同步模型之间的梯度。所以，更像是“分布式数据集+梯度同步”。

# 但是我没有 128 个 GPU

别担心！你的模型不需要改变。只需从训练器中删除 nb_gpu_nodes 参数，即可使用机器上的所有 4 个 gpu:

然后在配有 4 个 GPU 的机器上运行脚本。

[完整的代码可在这里](https://github.com/williamFalcon/pytorch-lightning/blob/master/examples/templates/gan.py)。

这段 GAN 代码改编自[这个棒极了的 GAN 库](https://github.com/eriklindernoren/PyTorch-GAN/tree/master/implementations/gan)，并被重构为使用 PyTorch Lightning