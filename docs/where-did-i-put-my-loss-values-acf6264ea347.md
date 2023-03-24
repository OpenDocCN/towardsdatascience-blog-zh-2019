# 我把损失值放在哪里了？

> 原文：<https://towardsdatascience.com/where-did-i-put-my-loss-values-acf6264ea347?source=collection_archive---------23----------------------->

## 如何将 PyTorch 中训练深度学习模型时生成的指标和键值添加到您的检查点。

![](img/e0f2967bc726fc5c9e5b3e7c5656d357.png)

保存和加载 [PyTorch](https://pytorch.org/) 模型非常简单直观。

保存模型权重:

```
torch.save(model.state_dict(), PATH)
```

负载模型重量:

```
model = TheModelClass()
model.load_state_dict(torch.load(PATH))
```

这个 PyTorch [教程](https://pytorch.org/tutorials/beginner/saving_loading_models.html)还解释了如何保存优化器状态和其他关于这个主题的好技巧。

# 但是其他参数呢？

上面的保存和加载示例是加载您的模型的好方法，以便在测试时使用它进行推理，或者如果您使用预训练的模型进行微调。

现在，假设你在一个非常大的数据集上训练一个非常深的模型，这将花费很多时间，如果你使用[云实例](https://aws.amazon.com/ec2/)，这也需要很多钱。一个省钱的好办法是使用[现货实例](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-spot-instances.html)，这样会有很大的折扣。在训练过程中，您可能会丢失实例。仅仅使用模型权重和优化器状态来恢复训练是不够的。您还需要测量的所有指标，例如每个集合的损失和精度值、验证集合的最佳 top-k 精度、到停止点为止的时期数、样本数或迭代数，以及您跟踪的任何其他特殊值。

为了在您恢复培训后继续报告[学习进度曲线](https://en.wikipedia.org/wiki/Learning_curve)，以及基于[停止改进](https://pytorch.org/docs/stable/optim.html#how-to-adjust-learning-rate)的验证损失等其他动态决策，这些都是必需的。

完全恢复一个学习实验也很重要，因为它具有可重复性，发表一篇论文和一个代码库。此外，能够从收敛曲线上的任何“点”开始，在改变或不改变任何超参数的情况下继续学习过程对于新任务或领域的研究是至关重要的。

我正在使用 PyTorch，处理大型视频数据集，例如 [Kinetics](https://deepmind.com/research/open-source/open-source-datasets/kinetics/) 。在为我的研究寻找完整的检查点解决方案的过程中，我开始为 PyTorch 开发这样的检查点处理程序，最近发布了 Python 包索引(PyPI)的一个包:[https://pypi.org/project/pytorchcheckpoint/](https://pypi.org/project/pytorchcheckpoint/)
和一个 GitHub 库:
[https://github.com/bomri/pytorch-checkpoint](https://github.com/bomri/pytorch-checkpoint)。

为了安装软件包:

`pip install pytorchcheckpoint`

然后，在培训代码开头的某个地方启动课程:

```
from pytorchcheckpoint.checkpoint import CheckpointHandler
checkpoint_handler = CheckpointHandler()
```

现在，除了保存您的模型权重和优化器状态之外，您还可以在学习过程的任何步骤中添加任何其他值。

例如，为了节省您可以运行的类的数量:

```
# saving
checkpoint_handler.store_var(var_name='n_classes', value=1000)# restoring
n_classes = checkpoint_handler.get_var(var_name='n_classes')
```

此外，您可以存储值和指标:

*   每套:培训/验证/测试
*   对于每个时期/样本/迭代次数

例如，每个历元的训练集和验证集的最高 1 精度值可以通过使用以下来存储:

```
# train set - top1
checkpoint_handler.store_running_var_with_header(header=’train’, var_name=’top1', iteration=0, value=80)
checkpoint_handler.store_running_var_with_header(header=’train’, var_name=’top1', iteration=1, value=85)
checkpoint_handler.store_running_var_with_header(header=’train’, var_name=’top1', iteration=2, value=90)
checkpoint_handler.store_running_var_with_header(header=’train’, var_name=’top1', iteration=3, value=91)# valid set - top1
checkpoint_handler.store_running_var_with_header(header=’valid’, var_name=’top1', iteration=0, value=70)
checkpoint_handler.store_running_var_with_header(header=’valid’, var_name=’top1', iteration=1, value=75)
checkpoint_handler.store_running_var_with_header(header=’valid’, var_name=’top1', iteration=2, value=80)
checkpoint_handler.store_running_var_with_header(header=’valid’, var_name=’top1', iteration=3, value=85)
```

保存和加载完整的检查点只需一行代码:

```
# save checkpoint
checkpoint_handler.save_checkpoint(checkpoint_path=path, iteration=25, model=model)# load checkpoint
checkpoint_handler = checkpoint_handler.load_checkpoint(path)
```

您可以查看 py torch-check point[README](https://github.com/bomri/pytorch-checkpoint#pytorch-checkpoint)以获得更多有用的示例。

所以下次你开始训练的时候，确保你手头有这些损失值以备不时之需。