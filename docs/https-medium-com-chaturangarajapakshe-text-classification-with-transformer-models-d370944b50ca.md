# 变压器模型文本分类实践指南(XLNet，BERT，XLM，RoBERTa)

> 原文：<https://towardsdatascience.com/https-medium-com-chaturangarajapakshe-text-classification-with-transformer-models-d370944b50ca?source=collection_archive---------0----------------------->

## 关于使用 Transformer 模型进行文本分类任务的分步教程。学习如何使用 Pytorch-Transformers 库加载、微调和评估文本分类任务。包括伯特、XLNet、XLM 和罗伯塔模型的现成代码。

![](img/978b6877f9c363804ad749f5a4cc1e94.png)

Photo by [Arseny Togulev](https://unsplash.com/@tetrakiss?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 更新通知

请考虑使用 [Simple Transformers](https://github.com/ThilinaRajapakse/simpletransformers) 库，因为它易于使用，功能丰富，并且定期更新。这篇文章仍然是 BERT 模型的参考，可能有助于理解 BERT 是如何工作的。然而，[简单变形金刚](https://github.com/ThilinaRajapakse/simpletransformers)提供了更多的功能，更简单的调音选项，而且快速易用！下面的链接应该可以帮助你快速入门。

1.  [二进制分类](/simple-transformers-introducing-the-easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3?source=---------16------------------)
2.  [多级分类](https://medium.com/swlh/simple-transformers-multi-class-text-classification-with-bert-roberta-xlnet-xlm-and-8b585000ce3a?source=---------15------------------)
3.  [多标签分类](/multi-label-classification-using-bert-roberta-xlnet-xlm-and-distilbert-with-simple-transformers-b3e0cda12ce5?source=---------13------------------)
4.  [命名实体识别(词性标注)](/simple-transformers-named-entity-recognition-with-transformer-models-c04b9242a2a0?source=---------14------------------)
5.  [问题解答](/question-answering-with-bert-xlnet-xlm-and-distilbert-using-simple-transformers-4d8785ee762a?source=---------12------------------)
6.  [句子对任务和回归](https://medium.com/swlh/solving-sentence-pair-tasks-using-simple-transformers-2496fe79d616?source=---------9------------------)
7.  [对话式人工智能](/how-to-train-your-chatbot-with-simple-transformers-da25160859f4?source=---------6------------------)
8.  [语言模型微调](https://medium.com/skilai/language-model-fine-tuning-for-pre-trained-transformers-b7262774a7ee?source=---------4------------------)
9.  [ELECTRA 和语言模型培训从零开始](/understanding-electra-and-training-an-electra-language-model-3d33e3a9660d?source=---------2------------------)
10.  [可视化模型训练](https://medium.com/skilai/to-see-is-to-believe-visualizing-the-training-of-machine-learning-models-664ef3fe4f49?source=---------10------------------)

**py torch-Transformers(现在是 Transformers)库在这篇文章写完之后有了很大的发展。我推荐使用**[**simple Transformers**](/simple-transformers-introducing-the-easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3)**，因为它与变形金刚库保持同步，而且更加用户友好。虽然本文中的思想和概念仍然有效，但是代码和 Github repo 不再被积极地维护。**

1.  [语言模型微调](https://medium.com/skilai/language-model-fine-tuning-for-pre-trained-transformers-b7262774a7ee?source=---------4------------------)
2.  [ELECTRA 和语言模型培训从零开始](/understanding-electra-and-training-an-electra-language-model-3d33e3a9660d?source=---------2------------------)
3.  [可视化模型训练](https://medium.com/skilai/to-see-is-to-believe-visualizing-the-training-of-machine-learning-models-664ef3fe4f49?source=---------10------------------)

*我强烈推荐为本文克隆*[*Github repo*](https://github.com/ThilinaRajapakse/pytorch-transformers-classification)*，并在遵循指南的同时运行代码。它应该有助于你更好地理解指南和代码。阅读很棒，但编码更好* ***r.*** 😉

特别感谢拥抱脸的 Pytorch-Transformers 库让变形金刚模型变得简单有趣！

# 1.介绍

Transformer 模型已经席卷了自然语言处理的世界，转换(对不起！)领域突飞猛进。几乎每个月都会出现新的、更大的、更好的模型，为各种各样的任务设定了新的性能基准。

这篇文章的目的是作为一个直接的指南，来使用这些令人敬畏的模型进行文本分类任务。因此，我不会谈论网络背后的理论，或者它们是如何工作的。如果你对深入了解变形金刚的本质感兴趣，我推荐杰伊·阿拉玛的图解指南。

这也是对我之前关于使用 BERT 进行二进制文本分类的指南的更新。我将使用与上次相同的数据集(Yelp 评论),以避免下载新的数据集，因为我很懒，而且我的互联网很糟糕。更新背后的动机有几个原因，包括对我在上一篇指南中使用的 HuggingFace 库的更新，以及多个新的 Transformer 模型的发布，这些模型成功地将 BERT 从其高位上拉了下来。

背景设置好了，让我们看看我们将要做什么。

1.  使用 HuggingFace 提供的 Pytorch-Transformers 库设置开发环境。
2.  正在转换*。csv* 数据集到*。HuggingFace 库使用的 tsv* 格式。
3.  设置预训练模型。
4.  将数据转换为要素。
5.  微调模型。
6.  评价。

我将使用两个 Jupyter 笔记本，一个用于数据准备，一个用于培训和评估。

# 2.各就各位

## 让我们设置环境。

1.  强烈建议在安装和使用各种 Python 库时使用虚拟环境。我个人最喜欢的是 Anaconda，但是你可以用任何你想用的东西。
    `conda create -n transformers python pytorch pandas tqdm jupyter
    conda activate transformers
    conda install -c anaconda scikit-learn
    pip install pytorch-transformers
    pip install tensorboardX` *请注意，本指南中使用的其他软件包可能没有安装在这里。如果遇到缺失的软件包，只需通过 conda 或 pip 安装它们。*
2.  Linux 用户可以使用 shell 脚本[下载并提取 Yelp 评论极性数据集。其他人可以在 fast.ai 手动下载](https://github.com/ThilinaRajapakse/pytorch-transformers-classification/blob/master/data_download.sh)[这里](https://course.fast.ai/datasets)还有，[直接下载链接](https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz)。
    我把`train.csv`*`test.csv`*文件放在一个名为`*data*` *的目录下。* `<starting_directory>/data/`**

# **3.预备开始**

## **是时候为变压器模型准备好数据了。**

**大多数在线数据集通常是`.csv`格式。遵循规范，Yelp 数据集包含两个`csv`文件`train.csv`和`test.csv`。**

**开始我们的第一个(数据准备)笔记本，让我们用熊猫加载`csv`文件。**

**然而，这里使用的标签打破了常规，是 1 和 2，而不是通常的 0 和 1。我完全赞成一点叛逆，但这只是让我反感。我们来修正一下，让标签为 0 和 1，分别表示差评和好评。**

**在 Pytorch-Transformer 模型的数据准备好之前，我们需要做一些最后的润饰。数据需要在`tsv`格式，有四列，没有标题。**

*   **guid:行的 id。**
*   **label:行的标签(应为 int)。**
*   **字母:所有行都有相同字母的列。不用于分类但仍然需要。**
*   **文本:行的文本。**

**所以，我们把数据整理一下，保存成`tsv`格式。**

**这标志着数据准备笔记本的结束，我们将继续下一部分的培训笔记本。**

# **4.走吧。(差不多)**

## **从文本到特征。**

**在开始实际训练之前，我们需要将数据从文本转换成可以输入神经网络的数值。在变压器模型的情况下，数据将被表示为`InputFeature`对象。**

**为了让我们的数据转换器做好准备，我们将使用文件`utils.py`中的类和函数。(稳住自己，一墙代码来袭！)**

**让我们看看重要的部分。**

**`InputExample`类代表我们数据集的一个样本；**

*   **`guid`:唯一 ID**
*   **`text_a`:我们的实际文本**
*   **`text_b`:不用于分类**
*   **`label`:样品的标签**

**`DataProcessor`和`BinaryProcessor`类用于从`tsv`文件中读入数据并转换成`InputExamples`。**

**`InputFeature`类表示可以提供给转换器的纯数字数据。**

**三个函数`convert_example_to_feature`、`convert_examples_to_features`、`_truncate_seq_pair`用于将`InputExamples`转换为`InputFeatures`，最终发送给变压器模型。**

**转换过程包括*标记化*，将所有句子转换为给定的*序列长度*(截断较长的序列，填充较短的序列)。在标记化期间，句子中的每个单词被分成越来越小的标记(单词片段)，直到数据集中的所有标记都被转换器识别。**

**作为一个人为的例子，假设我们有单词*理解*。我们使用的转换器没有理解的*令牌，但是它有理解*和 *ing 的*令牌。*然后，*理解*这个词就会被分解成令牌*理解*和 *ing。**序列长度*是序列中此类令牌的数量。***

*`convert_example_to_feature`函数获取*单个*数据样本，并将其转换为`InputFeature`。`convert_examples_to_features`函数接受一个*列表*的例子，并通过使用`convert_example_to_feature`函数返回一个`InputFeatures`的*列表*。之所以有两个独立的函数，是为了让我们在转换过程中使用多重处理。默认情况下，我已经将进程计数设置为`cpu_count() - 2`，但是您可以通过为`convert_examples_to_features`函数中的`process_count`参数传递一个值来更改它。*

*现在，我们可以转到我们的培训笔记本，导入我们将使用的材料并配置我们的培训选项。*

*仔细阅读`args`字典，记下您可以为训练配置的所有不同设置。在我的情况下，我使用 fp16 训练来降低内存使用和加快训练速度。如果你没有安装 Nvidia Apex，你将不得不通过设置为 False 来关闭 fp16。*

*在本指南中，我使用序列长度为 128 的 XL-Net 模型。*请参考*[*Github repo*](https://github.com/ThilinaRajapakse/pytorch-transformers-classification)*了解*的完整可用型号列表。*

*现在，我们准备加载我们的模型进行训练。*

*Pytorch-Transformers 库最酷的地方在于，你可以使用上面的任何一个`MODEL_CLASSES`，只需改变参数字典中的`model_type`和`model_name`。所有模型的微调和评估过程基本相同。拥抱脸万岁！*

*接下来，我们有定义如何加载数据、训练模型和评估模型的函数。*

*最后，我们做好了标记数据和训练模型的一切准备。*

# *5.走吧。(真的)*

## *培训。*

*从这里开始应该相当简单。*

*这将把数据转换成特征并开始训练过程。转换后的要素将被自动缓存，如果您以后想要运行相同的实验，可以重用它们。但是，如果您更改了类似于`max_seq_length`的东西，您将需要重新处理数据。改变使用的模型也是如此。要重新处理数据，只需在`args`字典中将`reprocess_input_data`设置为`True`。*

**作为对比，我在 RTX 2080 上花了大约 3 个小时来训练这个数据集。**

*一旦训练结束，我们就可以挽救一切。*

# *6.回首*

## *评价。*

*评估也很容易。*

*在没有任何参数调整的情况下，使用一个训练时期，我的结果如下。*

```
*INFO:__main__:***** Eval results  *****
INFO:__main__:  fn = 1238
INFO:__main__:  fp = 809
INFO:__main__:  mcc = 0.8924906867291726
INFO:__main__:  tn = 18191
INFO:__main__:  tp = 17762*
```

*不算太寒酸！*

# *7.包裹*

*Transformer 模型在处理各种各样的自然语言处理任务方面表现出了令人难以置信的能力。在这里，我们已经了解了如何使用它们来完成最常见的任务之一，即序列分类。*

*HuggingFace 的 Pytorch-Transformers 库使得驾驭这些庞大模型的能力变得几乎微不足道！*

# *8.最后的想法*

*   *当使用自己的数据集时，我建议编辑`data_prep.ipynb`笔记本，将数据文件保存为`tsv`文件。在大多数情况下，只要确保将包含*文本*和*标签*的正确列传递给`train_df`和`dev_df`构造函数，就可以让事情运行起来。您也可以在`utils.py`文件中定义自己的类，该类继承自`DataProcessor`类，但我觉得第一种方法更简单。*
*   *请使用 [Github repo](https://github.com/ThilinaRajapakse/pytorch-transformers-classification) 而不是从这里复制粘贴。任何修复或额外的功能都将被添加到 Github repo 中，除非是重大变更，否则不太可能添加到这里。使用 Gists 将代码嵌入到介质中，因此，它们不会自动与回购代码同步。*
*   *如果您需要支持，或者您发现了一个 bug，在 Github repo 上提出一个问题可能会得到比这篇文章上的评论更快的回复。在这里很容易错过评论，缺乏评论/聊天线索也很难跟进。另外，如果是在 Github 上，而不是在 Medium response 上，其他人可能会更容易找到同样问题的答案。*