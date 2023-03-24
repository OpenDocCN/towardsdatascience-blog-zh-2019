# PyTorch BigGraph 简介—带示例

> 原文：<https://towardsdatascience.com/introduction-to-pytorch-biggraph-with-examples-b50ddad922b8?source=collection_archive---------6----------------------->

![](img/5c832fbef978a82bf9d9bba783b26b63.png)

Network Photo by [Alina Grubnyak](https://unsplash.com/@alinnnaaaa?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/network?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

PyTorch BigGraph 是一个为机器学习创建和处理大型图形嵌入的工具。目前在基于图的神经网络中有两种方法:

*   直接使用图形结构，并将其输入神经网络。然后在每一层都保留图形结构。graphCNNs 使用这种方法，例如参见[我的帖子](https://medium.com/@svenbalnojan/using-graph-cnns-in-keras-8b9f685c4ed0)或[这篇文章](https://arxiv.org/abs/1609.02907)。
*   但是大多数图表都太大了。所以创建图的大嵌入也是合理的。然后把它作为传统神经网络的特征。

PyTorch BigGraph 处理第二种方法，下面我们也将这样做。仅供参考，让我们先讨论一下尺寸。图通常由它们的[邻接矩阵](https://en.wikipedia.org/wiki/Adjacency_matrix)编码。如果你有一个有 3000 个节点的图，每个节点之间有一条边，那么你的矩阵中就有大约 10000000 个条目。即使这很少，但根据上面链接的论文中的[,这显然会使大多数 GPU 崩溃。](https://arxiv.org/abs/1609.02907)

如果你想一想推荐系统中常用的图表，你会发现它们通常要大得多。现在已经有一些关于 BigGraph 的方式和原因的优秀帖子，所以我不会在这方面花更多时间。我对将 BigGraph 应用于我的机器学习问题很感兴趣，为此我喜欢举最简单的例子并让事情运转起来。我构建了两个例子，我们将一步一步来看。

整个代码都经过了重构，可以在 [GitHub](https://github.com/sbalnojan/biggraph-examples) 获得。它改编自 BigGraph 存储库中的示例。

第一个示例是 LiveJournal 图表的一部分，数据如下所示:

```
# FromNodeId ToNodeId0 1
0 2
0 3
...
0 10
0 11
0 12
...
0 46
1 0
...
```

第二个例子是简单的带边的 8 个节点:

```
# FromNodeId ToNodeId
0   1
0   2
0   3
0   4
1   0
1   2
1   3
1   4
2   1
2   3
2   4
3   1
3   2
3   4
3   7
4   1
5   1
6   2
7   3
```

# **嵌入 LiveJournals 图形的一部分**

BigGraph 是为机器的内存限制而设计的，所以它是完全基于文件的。您必须触发进程来创建适当的文件结构。如果你想再次运行一个例子，你必须删除检查点。我们还必须预先分成训练和测试，再次以文件为基础。文件格式为 TSV，用制表符分隔值。

让我们开始吧。第一段代码声明了两个助手函数，取自 BigGraph 源代码，设置了一些常量并运行文件分割。

helper functions and random_split_file call.

这通过创建两个文件 *data/example_1/test.txt* 和 *train.txt* 将边分成测试和训练集。接下来，我们使用 BigGraphs 转换器为数据集创建基于文件的结构。我们将“分区”成 1 个分区。为此，我们已经需要部分配置文件。这是配置文件的相关部分，I/O 数据部分和图形结构。

```
entities_base = 'data/example_1' def get_torchbiggraph_config(): config = dict(       
         # I/O data
        entity_path=entities_base,
        edge_paths=[],
        checkpoint_path='model/example_1', # Graph structure
        entities={
            'user_id': {'num_partitions': 1},
        },
        relations=[{
            'name': 'follow',
            'lhs': 'user_id',
            'rhs': 'user_id',
            'operator': 'none',
        }],
...
```

这告诉 BigGraph 在哪里可以找到我们的数据，以及如何解释我们的制表符分隔的值。有了这个配置，我们可以运行下一个 Python 代码片段。

convert data to _partitioned data.

结果应该是数据目录中的一堆新文件，即:

*   两个文件夹 *test_partitioned，train_partitioned*
*   h5 格式的边缘每个文件夹一个文件，用于快速部分读取
*   *dictionary.json* 文件包含“user_ids”和新分配的 id 之间的映射。
*   entity_count_user_id_0.txt 包含实体计数，在本例中为 47。

dictionary.json 对于稍后将 BigGraph 模型的结果映射到我们想要的实际嵌入非常重要。准备够了，我们来训练嵌入。看一下 *config_1.py* ，它包含三个相关的部分。

```
 # Scoring model - the embedding size
        dimension=1024,
        global_emb=False, # Training - the epochs to train and the learning rate
        num_epochs=10,
        lr=0.001, # Misc - not important
        hogwild_delay=2,
    ) return config
```

为了进行训练，我们运行以下 Python 代码。

train the embedding.

通过这段代码，我们可以根据测试集上预先安装的一些指标来评估模型。

evaluate the embedding.

现在让我们尝试检索实际的嵌入。同样，因为一切都是基于文件的，所以它现在应该位于 *models/* 文件夹中的 h5 位置。我们可以通过在字典中查找用户 0 的映射来加载用户 0 的嵌入，如下所示:

output the embedding.

现在让我们转到第二个例子，一个构造好的例子，我们希望可以在这个例子上做一些有用的事情。liveJournal 数据实在太大了，无法在合理的时间内浏览一遍。

# **对构建示例的链接预测和排序**

好的，我们将重复第二个例子的步骤，除了我们将产生一个 10 维的嵌入，所以我们可以查看和使用它。此外，对我来说，10 维对 8 个顶点来说已经足够了。我们在 *config_2.py* 中设置这些东西。

```
entities_base = 'data/example_2'
 def get_torchbiggraph_config():
     config = dict(
        # I/O data
        entity_path=entities_base,
        edge_paths=[],
        checkpoint_path='model/example_2', # Graph structure
        entities={
            'user_id': {'num_partitions': 1},
        },
        relations=[{
            'name': 'follow',
            'lhs': 'user_id',
            'rhs': 'user_id',
            'operator': 'none',
        }], # Scoring model
        dimension=10,
        global_emb=False, # Training
        num_epochs=10,
        lr=0.001, # Misc
        hogwild_delay=2,
    )
     return config
```

然后，我们像以前一样运行相同的代码，但是一次完成，处理不同的文件路径和格式。在这种情况下，我们在数据文件顶部只有 3 行注释:

作为最终输出，你应该得到一堆东西，特别是所有的嵌入。让我们做一些嵌入的基本任务。当然，我们现在可以使用它，并将其加载到我们喜欢的任何框架中， *keras* ， *tensorflow* ，但是 BigGraph 已经为常见任务带来了一些实现，如**链接预测**和**排名**。所以让我们试一试。第一个任务是**链路预测**。我们预测实体 **0-7** 和**0-1**的得分，因为我们从数据中知道**0-1**的可能性更大。

作为比较器，我们加载了“DotComparator ”,它计算两个 10 维向量的点积或标量积。结果显示输出的数字很小，但至少 score_2 比 score_1 高得多，正如我们所预期的那样。

最后，作为最后一段代码，我们可以生成一个相似项目的排名，它使用与前面相同的机制。我们使用标量积来计算嵌入到所有其他实体的距离，然后对它们进行排序。

在这种情况下，顶级实体的顺序是 0、1、3、7……如果你观察数据，就会发现这似乎非常正确。

# 更有趣

这是我能想到的最基本的例子。我没有在 freebase 数据或 LiveJournal 数据上运行原始示例，只是因为它们需要相当长的训练时间。您可以在这里找到代码和参考资料:

*   PyTorch BigGraph 的 [GitHub 库](https://github.com/facebookresearch/PyTorch-BigGraph)
*   [GitHub 库](https://github.com/sbalnojan/biggraph-examples)带示例代码
*   [https://arxiv.org/pdf/1903.12287.pdf](https://arxiv.org/pdf/1903.12287.pdf)，a .勒尔等人。艾尔。(2019)，PyTorch-BigGraph:大规模图嵌入系统。
*   [https://arxiv.org/abs/1609.02907](https://arxiv.org/abs/1609.02907)，T. N .基普夫，m .韦林(2016)，利用图卷积网络的半监督分类。

# 你可能遇到的问题

我在 mac 上运行代码，遇到了三个问题:

*   说明*“lib *…”的错误..原因:未找到映像:“*解决方案是安装缺少的部分，例如使用*“brew install libomp”*
*   然后我遇到了一个错误*“attribute error:模块‘torch’没有属性' _ six '”*，这可能只是因为不兼容的 python & torch 版本。反正我是从 *python 3.6 &火炬 1.1*=>*python 3.7&火炬 1。X* 解决了我的问题。
*   在您继续之前，检查 train.txt 和 test.txt，我在测试时看到那里有一些丢失的新行。

希望这有所帮助，并且玩起来很有趣！