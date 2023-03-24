# 可重复的模型训练:深度潜水

> 原文：<https://towardsdatascience.com/reproducible-model-training-deep-dive-2a4988d69031?source=collection_archive---------14----------------------->

> 可重复的研究很容易。只需在某处记录您的参数和指标，修复种子，您就可以开始了

*—我，大约两周前。*

哦，天啊，我错了。

有很多关于可重复研究的研讨会、教程和会议。

大量的实用程序、工具和框架被用来帮助我们做出良好的可复制的解决方案。

然而，仍然存在问题。这些陷阱在一个简单的辅导项目中并不明显，但在任何真正的研究中必然会发生。很少有人谈论它们，所以我想分享我关于这些话题的知识。

在这篇文章中，我将讲述一个关于我对能够持续训练模型的追求的故事(所以每次跑步都给相同的重量)。

# 意想不到的问题

从前，我有一个与计算机视觉(笔迹作者识别)相关的项目。

在某个时候，我决定花时间重构代码和整理项目。我将我的大型 Keras 模型分成几个阶段，为每个阶段设计测试集，并使用 ML Flow 来跟踪每个阶段的结果和性能(这很难，但这是另一个故事了)。

经过一周左右的重构，我已经构建了一个很好的管道，捕获了一些 bug，设法摆弄了一下超参数，并略微提高了性能。

然而，我注意到一件奇怪的事情。我修复了所有随机种子，正如许多向导建议的那样:

```
def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)
    session_conf = tf.ConfigProto(intra_op_parallelism_threads=1, inter_op_parallelism_threads=1)
    sess = tf.Session(graph=tf.get_default_graph(), config=session_conf)
    K.set_session(sess)
```

但是由于某种原因，连续两次使用相同的超参数得到了不同的结果。

由于无法跟踪项目中的问题，我决定用一个小模型制作一个脚本来重现这个问题。

我定义了一个简单的神经网络:

```
def create_mlp(dim):
    model = Sequential()
    model.add(Dense(8, input_dim=dim))
    model.add(Dense(1))
    return model
```

因为数据在这里并不重要，所以我生成了一些随机数据来处理。之后，我们准备训练我们的模型:

```
model = create_mlp(10)
init_weights = np.array(model.get_weights()[0]).sum()
model.compile(optimizer=keras.optimizers.RMSprop(lr=1e-2),
              loss=keras.losses.MSE)
model.fit(Xs, Ys, batch_size=10, epochs=10)
```

训练后，我们可以检查再现性。*assert _ same _ cross _ runs*是一个简单的函数，它检查传递的值在两次运行之间是否相同(通过将值写入文件来完成):

```
assert_same_across_runs("dense model data", Ys.sum())
assert_same_across_runs("dense model weight after training", 
    init_weights)
assert_same_across_runs("dense model weight after training", 
    np.array(model.get_weights()[0]).sum())
```

[**全脚本**](https://github.com/rampeer/rampeer.github.io/blob/master/sources/reproducibility/reproducibility_dense.py?source=post_page---------------------------)

我运行了几次。每次执行的模型权重完全相同。

奇怪！我通过插入卷积层给模型增加了一点复杂性:

```
def create_nnet(dim):
    input = Input(shape=dim)
    conv = Conv2D(5, (3, 3), activation="relu")(input)
    flat = Flatten()(conv)
    output = Dense(1)(flat)
    return Model([input], [output])
```

[**全脚本**](https://github.com/rampeer/rampeer.github.io/blob/master/sources/reproducibility/reproducibility_cnn.py?source=post_page---------------------------)

训练过程非常相似:我们创建模型，生成一些数据，训练我们的模型，然后检查指标。

瞧，它碎了。每次运行脚本时，都会打印不同的数字。

这个问题不会发生在没有 GPU 的机器上。如果你的机器有 GPU，你可以通过从控制台设置 *CUDA_VISIBLE_DEVICES* 环境变量为""，在脚本中隐藏它。

快速调查发现了丑陋的事实:再现性存在[问题](https://machinelearningmastery.com/reproducible-results-neural-networks-keras/?source=post_page---------------------------)，一些层使模型不可再现，至少在默认情况下(对所有“你因为使用 Keras 而受苦”的人来说，Pytorch 有一个[类似的问题](https://pytorch.org/docs/stable/notes/randomness.html?source=post_page---------------------------)

# 为什么会这样？

一些复杂的操作没有明确定义的子操作顺序。
比如卷积就是一堆加法，但是这些加法的顺序并没有定义。

因此，每次执行都会导致不同的求和顺序。因为我们使用有限精度的浮点运算，卷积产生的结果略有不同。

是的，求和的顺序很重要，(a+b)+c！= a+(b+c)！你甚至可以自己检查:

```
import numpy as np
np.random.seed(42)xs = np.random.normal(size=10000)
a = 0.0
for x in xs:
    a += xb = 0
for x in reversed(xs):
    b += xprint(a, b)
print("Difference: ", a - b)
```

应该打印

```
-21.359833684261957 -21.359833684262377
Difference: 4.192202140984591e-13
```

没错，就是“就 4e-13”。但是因为这种不精确发生在深度神经网络的每一层，并且对于每一批，这种误差随着层和时间而累积，并且模型权重显著偏离。因此，连续两次运行的损失可能如下所示:

![](img/195de5bac03af4d0fbee04dd33d5d8ae.png)

# 重要吗？

有人可能会说，运行之间的这些小差异不应该影响模型的性能，事实上，固定随机种子和精确的再现性并不那么重要。

嗯，这个论点有可取之处。随机性影响权重；因此，模型性能在技术上取决于随机种子。与添加新功能或改变架构相比，改变种子对准确性的影响应该较小。此外，因为随机种子不是模型的重要部分，所以对不同的种子多次评估模型(或者让 GPU 随机化)并报告平均值和置信区间可能是有用的。

然而，在实践中，很少有论文这样做。相反，模型与基于点估计的基线进行比较。此外，还有[担心](https://arxiv.org/abs/1709.06560?source=post_page---------------------------)论文报告的改进比这种随机性少*，并且不恰当的汇总会扭曲结果。*

*更糟糕的是，这会导致代码中不可约的随机性。您不能编写单元测试。这通常不会困扰数据科学家，所以想象一种情况。*

*你已经找到了一篇描述一个奇特模型的论文——并且它已经清晰地组织了开源实现。你下载代码和模型，开始训练它。几天后(这是一个非常奇特的模型)
你测试这个模型。它不起作用。是超参数的问题吗？您的硬件或驱动程序版本？数据集？也许，回购作者是骗子，问题出在存储库本身？你永远不会知道。*

*在实验完全*重现之前，故障排除是极其麻烦的，因为你不知道管道的哪一部分出现了问题。**

**因此，拥有复制模型的能力似乎非常方便。**

# **哦不。我们该怎么办？**

**求和的顺序未定义？好吧。我们可以自己定义，也就是把卷积重写为一堆求和。是的，它会带来一些开销，但是它会解决我们的问题。**

**令人高兴的是，CuDNN 已经有了大多数操作的“可再现”实现(而且确实更慢)。首先，你不需要自己写任何东西，你只需要告诉 CuDNN 使用它。其次，CuDNN 在堆栈中的位置
很低——因此，您不会得到太多的开销:**

**![](img/07a4aff5f3b8aedbb07ee66649114d87.png)**

**在你的代码和硬件之间有一个分层的管道**

**因为我们不直接与 CuDNN 交互，所以我们必须告诉我们选择的库使用特定的实现。换句话说，我们必须打开“我可以接受较慢的训练，让我每次跑步都有稳定的结果”的标志。**

**不幸的是，Keras 还没有那个功能，如
[这些](https://github.com/tensorflow/tensorflow/issues/18096?source=post_page---------------------------) [问题](https://github.com/tensorflow/tensorflow/issues/12871?source=post_page---------------------------)中所述。**

**看来是 PyTorch 大放异彩的时候了。它具有支持使用 CuDNN 确定性实现的设置:**

# **火炬能拯救我们吗？**

**让我们写一个简单的单卷积网络，用随机数据进行训练。确切的架构或数据并不重要，因为我们只是在测试再现性。**

```
**class Net(nn.Module):
    def __init__(self, in_shape: int):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(3, 5, 3)
        self.hidden_size = int((in_shape — 2) * (in_shape — 2) / 4) * 5
        self.fc1 = nn.Linear(self.hidden_size, 1)def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        x = x.view(-1, self.hidden_size)
        x = F.relu(self.fc1(x))
        return x**
```

***fix_seeds* 函数也被修改为包括**

```
**def fix_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(42)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False**
```

**同样，我们将使用合成数据来训练网络。**

**初始化后，我们确保权重之和等于特定值。类似地，在训练网络之后，我们检查模型权重。如果有任何差异或随机性，脚本会告诉我们。**

**[**全脚本**](https://github.com/rampeer/rampeer.github.io/blob/master/sources/reproducibility/reproducibility_cnn_torch.py?source=post_page---------------------------)**

```
**python3 reproducibility_cnn_torch.py**
```

**给出一致的结果，这意味着在 PyTorch 中训练神经网络给出完全相同的权重。**

**有一些警告。首先，这些设置会影响训练时间，但是[测试](https://github.com/rampeer/rampeer.github.io/blob/master/sources/reproducibility/reproducibility_cnn_torch_timing.py?source=post_page---------------------------)显示的差异可以忽略不计( *1.14 +- 0.07* 秒对于非确定性版本； *1.17 +- 0.08* 秒为确定性一)。其次，CuDNN 文档[警告我们](https://docs.nvidia.com/deeplearning/sdk/cudnn-developer-guide/index.html?source=post_page---------------------------#reproducibility)有几种算法没有再现性保证。这些算法通常比它们的确定性变体更快，但是如果设置了标志，PyTorch 就不会使用它们。最后，有些模块是[不确定的](https://discuss.pytorch.org/t/non-deterministic-behavior-of-pytorch-upsample-interpolate/42842?u=sbelharbi&source=post_page---------------------------)(我自己无法重现这个问题)。**

**无论如何，这些问题很少成为障碍。所以，Torch 有内置的再现性支持，而 Keras 现在没有这样的功能。**

***但是等等！*有[一些](https://www.kaggle.com/c/statoil-iceberg-classifier-challenge/discussion/45663?source=post_page---------------------------) [猜测](https://www.kaggle.com/c/jigsaw-unintended-bias-in-toxicity-classification/discussion/90946?source=post_page---------------------------#524750)由于 Keras 和 Torch 共享后端，在 PyTorch 中设置这些变量会影响 Keras。如果变量影响某些全过程参数，则可能出现这种情况。这样，我们就可以利用这些标志，并使用 Keras 轻松地编写模型。
首先，让我们看看这些标志是如何工作的。**

**我们再深入一点！从 Python 端
[设置确定性/基准标志调用 C 源代码](https://github.com/pytorch/pytorch/blob/3a0b27b73d901ab99b6c452b7e716058311e3372/torch/backends/cudnn/__init__.py?source=post_page---------------------------#L481)中定义的函数。这里，状态[存储在内部](https://github.com/pytorch/pytorch/blob/3a0b27b73d901ab99b6c452b7e716058311e3372/aten/src/ATen/Context.cpp?source=post_page---------------------------#L67)。
之后，该标志影响卷积期间算法的[选择。换句话说，设置这个标志不会影响全局状态(例如，环境变量)。](https://github.com/pytorch/pytorch/blob/85528feb409d2a44e2a35637e0768d6de8d92039/aten/src/ATen/native/cudnn/Conv.cpp?source=post_page---------------------------#L460)**

**因此，PyTorch 设置似乎不会影响 Keras 内部(实际的[测试](https://github.com/rampeer/rampeer.github.io/blob/master/sources/reproducibility/reproducibility_cnn_import_test.py) [证实了这一点)。](https://rampeer.github.io/2019/06/12/keras-pain.html)**

**补充:一些用户报告说，对于一些层(CuDNNLSTM 是一个明显的例子)和一些硬件和软件的组合，这个黑客可能会工作。这绝对是一个有趣的效果，但是我不鼓励在实践中使用它。使用一种有时有效有时无效的技术会破坏可复制学习的整体理念。**

# **结论**

**修复随机种子非常有用，因为它有助于模型调试。此外，运行之间减少的度量方差有助于决定某个特性是否会提高模型的性能。**

**但有时这还不够。某些模型(卷积、递归)需要设置额外的 CuDNN 设置。Keras 还没有这样的功能，而 PyTorch 有:**

```
**torch.backends.cudnn.deterministic = True
torch.backends.cudnn.benchmark = False**
```