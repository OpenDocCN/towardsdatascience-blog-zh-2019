# 用 Keras 和 tf 微调 BERT。组件

> 原文：<https://towardsdatascience.com/fine-tuning-bert-with-keras-and-tf-module-ed24ea91cff2?source=collection_archive---------4----------------------->

在这个实验中，我们将预训练的 BERT 模型检查点转换为可训练的 Keras 层，我们使用它来解决文本分类任务。

我们通过使用 tf 来实现这一点。模块，这是一个简洁的抽象，旨在处理预先训练好的 Tensorflow 模型。

导出的模块可以很容易地集成到其他模型中，这有助于使用强大的神经网络架构进行实验。

![](img/ea1e2026e79c229e9fbbb5a79500614a.png)

这个实验的计划是:

1.  获得预训练的 BERT 模型检查点
2.  定义 tf 的规格。组件
3.  导出模块
4.  构建文本预处理管道
5.  实现自定义 Keras 层
6.  训练一个 Keras 模型来解决句子对分类任务
7.  保存和恢复
8.  优化用于推理的训练模型

# 这本指南里有什么？

本指南是关于将预先训练的 Tensorflow 模型与 Keras 集成。它包含两个东西的实现:一个 BERT [tf。模块](https://www.tensorflow.org/api_docs/python/tf/Module)和构建在它上面的 Keras 层。它还包括微调(见下文)和推理的例子。

# 需要什么？

对于熟悉 TensorFlow 的读者来说，完成本指南大约需要 60 分钟。代码用 tensorflow==1.15.0 测试。

# 好吧，给我看看代码。

这个实验的代码可以在 T2 的实验室里找到。独立版本可以在[库](https://github.com/gaphex/bert_experimental)中找到。

# 步骤 1:获得预训练模型

我们从一个预先训练好的基于 BERT 的检查点开始。在这个实验中，我们将使用由谷歌预先训练的英语模型[。当然，在构建 tf.Module 时，您可以使用更适合您的用例的模型。](https://storage.googleapis.com/bert_models/2018_10_18/uncased_L-12_H-768_A-12.zip)

# 步骤 2:构建一个 tf。组件

tf。模块旨在提供一种简单的方法来操作 Tensorflow 中预训练机器学习模型的可重用部分。谷歌在 [tf 维护了一个此类模块的精选库。轮毂](https://www.tensorflow.org/hub)。然而，在本指南中，我们将自己从头开始构建一个。

为此，我们将实现一个包含模块内部工作的完整规范的 *module_fn* 。

我们从定义输入占位符开始。BERT 模型图由通过*配置路径*传递的配置文件创建。然后提取 we 模型输出:最终的编码器层输出被保存到 *seq_output* 并且汇集的“CLS”令牌表示被保存到 *pool_output。*

此外，额外的资产可以与模块捆绑在一起。在这个例子中，我们将一个包含单词表的 *vocab_file* 添加到模块资产中。因此，词汇文件将与模块一起导出，这将使它成为自包含的。

最后，我们定义了签名，它是输入到输出的特定转换，向消费者公开。人们可以把它看作是与外界的一个模块接口。

这里添加了两个签名。第一个将原始文本特征作为输入，并将计算后的文本表示作为输出返回。另一个不接受输入，返回词汇表文件的路径和小写标志。

# 步骤 3:导出模块

既然已经定义了 *module_fn* ，我们就可以用它来构建和导出模块了。将 *tags_and_args* 参数传递给 *create_module_spec* 将导致两个图变量被添加到模块中:用 tags*{“train”}*进行训练，以及用一组空的 tags 进行推断。这允许控制退出，这在推理时被禁用，而在训练期间被启用。

# 步骤 4:构建文本预处理管道

BERT 模型要求将文本表示为包含*input _ id*、 *input_mask* 和*segment _ id*的 3 个矩阵。在这一步中，我们构建了一个管道，它接受一个字符串列表，并输出这三个矩阵，就这么简单。

首先，原始输入文本被转换成 InputExamples。如果输入文本是由特殊的“|||”序列分隔的句子对，则句子被拆分。

然后，使用来自原始存储库的*convert _ examples _ to _ features*函数，将 InputExamples 标记化并转换为 InputFeatures。之后，特征列表被转换成带有*特征 _ 到 _ 数组*的矩阵。

最后，我们将所有这些放在一个管道中。

全部完成！

# 步骤 5:实现 BERT Keras 层

使用 tf 有两种方法。带有 Keras 的模块。

第一种方法是用 [hub 包裹一个模块。角斗士](https://www.tensorflow.org/hub/api_docs/python/hub/KerasLayer)。这种方法很简单，但是不太灵活，因为它不允许在模块中加入任何定制的逻辑。

第二种方法是实现包含该模块的自定义 Keras 层。在这种情况下，我们可以完全控制可训练变量，并且可以将池化操作甚至整个文本预处理管道添加到计算图中！我们将走第二条路。

为了设计一个定制的 Keras 层，我们需要编写一个继承自 tf.keras.Layer 的类，并覆盖一些方法，最重要的是*构建*和*调用。*

*build* 方法创建模块的资产。它从实例化来自 *bert_path* 的 BERT 模块开始，该路径可以是磁盘上的路径或 http 地址(例如，对于来自 tf.hub 的模块)。然后建立可训练层的列表，并填充该层的可训练权重。将可训练权重的数量限制在最后几层可以显著减少 GPU 内存占用并加快训练速度。它还可能提高模型的准确性，特别是在较小的数据集上。

*build_preprocessor* 方法从模块资产中检索单词表，以构建步骤 4 中定义的文本预处理管道。

*initialize_module* 方法将模块变量加载到当前的 Keras 会话中。

大多数有趣的事情都发生在*调用*方法内部。作为输入，它接受一个张量 *tf。字符串*，使用我们的预处理管道将其转换成 BERT 特征。使用 *tf.numpy_function 将 python 代码注入到图中。*然后将特征传递给模块，并检索输出。

现在，根据在 *__init__，*中设置的*池*参数，额外的变换被应用于输出张量。

如果 *pooling==None* ，则不应用池，输出张量具有形状 *[batch_size，seq_len，encoder_dim]* 。此模式对于解决令牌级任务非常有用。

如果 *pooling=='cls '，*仅检索对应于第一个 *'CLS'* 标记的向量，并且输出张量具有形状 *[batch_size，encoder_dim]* 。这种池类型对于解决句子对分类任务很有用。

最后，如果 *pooling=='mean '，*所有记号的嵌入都是平均池化的，并且输出张量具有形状 *[batch_size，encoder_dim]* 。这种模式对于句子表达任务特别有用。它的灵感来自[伯特即服务](https://github.com/hanxiao/bert-as-service)的 REDUCE_MEAN 池策略。

BERT 层的完整列表可以在[库](https://github.com/gaphex/bert_experimental/blob/master/finetuning/bert_layer.py)中找到。

# 第六步:句子对分类

现在，让我们在真实数据集上尝试该图层。对于这一部分，我们将使用 [Quora 问题对](https://www.quora.com/q/quoradata/First-Quora-Dataset-Release-Question-Pairs)数据集，它由超过 400，000 个潜在的问题重复对组成，标记为语义等价。

建立和训练句子对分类模型很简单:

顺便说一句，如果你不喜欢在图形中进行预处理，你可以通过设置 *do_preprocessing=False* 来禁用它，并使用 3 个输入来构建模型。

仅微调最后三层就可以获得 88.3%的验证准确率。

```
Train on 323432 samples, validate on 80858 samples
Epoch 1/5323432/323432 [==============================] - 3197s 10ms/sample - loss: 0.3659 - acc: 0.8255 - val_loss: 0.3198 - val_acc: 0.8551
Epoch 2/5
323432/323432 [==============================] - 3191s 10ms/sample - loss: 0.2898 - acc: 0.8704 - val_loss: 0.2896 - val_acc: 0.8723
Epoch 3/5
323432/323432 [==============================] - 3231s 10ms/sample - loss: 0.2480 - acc: 0.8920 - val_loss: 0.2833 - val_acc: 0.8765
Epoch 4/5
323432/323432 [==============================] - 3205s 10ms/sample - loss: 0.2083 - acc: 0.9123 - val_loss: 0.2839 - val_acc: 0.8814
Epoch 5/5
323432/323432 [==============================] - 3244s 10ms/sample - loss: 0.1671 - acc: 0.9325 - val_loss: 0.2957 - val_acc: 0.8831
```

# 步骤 7:保存和恢复模型

模型权重可以通过常规方式保存和恢复。模型架构也可以序列化为 json 格式。

如果到 BERT 模块的**相对**路径不变，从 json 重建模型将会工作。

# 步骤 8:优化推理

在某些情况下(例如，当服务时)，人们可能想要优化训练模型以获得最大的推理吞吐量。在 TensorFlow 中，这可以通过“冻结”模型来实现。

在“冻结”过程中，模型变量由常数代替，训练所需的节点从计算图中删除。生成的图形变得更加轻量级，需要更少的 RAM，并获得更好的性能。

我们冻结训练好的模型，并将序列化的图形写入文件。

现在我们还原一下冻结图，做一些推断。

为了运行推理，我们需要得到图的输入和输出张量的句柄。这部分有点棘手:我们在恢复的图中检索所有操作的列表，然后手动获取相关操作的名称。列表是排序的，所以在这种情况下，只需进行第一个和最后一个操作。

为了得到张量名称，我们将 **":0"** 附加到 op 名称上。

我们注入到 Keras 层中的预处理函数是不可序列化的，并且没有在新图中恢复。不过不用担心——我们可以简单地用相同的名称再次定义它。

最后，我们得到了预测。

```
array([[9.8404515e-01]], dtype=float32)
```

# 结论

在这个实验中，我们创建了一个可训练的 BERT 模块，并用 Keras 对其进行了微调，以解决句子对分类任务。

通过冻结训练好的模型，我们消除了它对自定义层代码的依赖性，使它变得可移植和轻量级。

## 本系列中的其他指南

1.  [用云 TPU 从头开始预训练 BERT】](/pre-training-bert-from-scratch-with-cloud-tpu-6e2f71028379)
2.  [用 BERT 和 Tensorflow 构建搜索引擎](/building-a-search-engine-with-bert-and-tensorflow-c6fdc0186c8a)
3.  [用 Keras 和 tf 微调 BERT。模块](/fine-tuning-bert-with-keras-and-tf-module-ed24ea91cff2)【你在这里】
4.  [使用 BERT 和表示学习改进句子嵌入](/improving-sentence-embeddings-with-bert-and-representation-learning-dfba6b444f6b)