# 简单变压器——变压器模型的命名实体识别

> 原文：<https://towardsdatascience.com/simple-transformers-named-entity-recognition-with-transformer-models-c04b9242a2a0?source=collection_archive---------7----------------------->

## 简单的变形金刚是“它只是工作”的变形金刚库。使用 Transformer 模型进行命名实体识别，只需 3 行代码。是的，真的。

![](img/278a18c2189278da16d229d14c5b0e0b.png)

Photo by [Brandi Redd](https://unsplash.com/@brandi1?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/book-glasses?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

# 前言

简单变形金刚库是为了使变形金刚模型易于使用而设计的。变形金刚是非常强大的(更不用说巨大的)深度学习模型，在处理各种各样的自然语言处理任务方面取得了巨大的成功。Simple Transformers 使 Transformer 模型的应用程序能够用三行代码完成序列分类任务(最初是二进制分类，但不久后增加了多类分类)。

我很高兴地宣布，Simple Transformers 现在支持命名实体识别，这是另一个常见的 NLP 任务，以及序列分类。

*其他功能链接:*

*   [*二元序列分类用的简单变形金刚*](/simple-transformers-introducing-the-easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3)
*   [*用简单变压器进行多类序列分类*](https://medium.com/swlh/simple-transformers-multi-class-text-classification-with-bert-roberta-xlnet-xlm-and-8b585000ce3a)

[简单变形金刚](https://github.com/ThilinaRajapakse/simpletransformers)库是在优秀的[变形金刚](https://github.com/huggingface/transformers)库的基础上，通过抱脸的方式构建的。拥抱脸变形金刚图书馆是为研究人员和其他需要广泛控制事情如何完成的人准备的图书馆。当你需要偏离常规，做不同的事情，或者完全做新的事情时，这也是最好的选择。简单的变形金刚简单多了。

# 介绍

你想尝试这个绝妙的想法，你想卷起袖子开始工作，但是成千上万行看起来神秘(但很酷)的代码甚至会让一个资深的 NLP 研究员感到害怕。简单变形金刚背后的核心思想是，使用变形金刚并不需要很困难(或令人沮丧)。

简单的转换器抽象出所有复杂的设置代码，同时尽可能保留灵活性和配置空间。Transformer 模型只需要三行代码，一行用于初始化，一行用于训练，一行用于评估。

这篇文章演示了如何使用简单的变压器执行 NER。

*所有源代码可在*[*Github Repo*](https://github.com/ThilinaRajapakse/simpletransformers)*上获得。如果您有任何问题或疑问，这是解决它们的地方。请务必检查一下！*

# 装置

1.  从[这里](https://www.anaconda.com/distribution/)安装 Anaconda 或 Miniconda 包管理器
2.  创建新的虚拟环境并安装所需的软件包。
    `conda create -n transformers python pandas tqdm`
    `conda activate transformers`
    如果使用 cuda:
    `conda install pytorch cudatoolkit=10.0 -c pytorch`
    其他:
    `conda install pytorch cpuonly -c pytorch`
    `conda install -c anaconda scipy`
    `conda install -c anaconda scikit-learn`
    `pip install transformers`
    `pip install tensorboardx` `pip install seqeval`
3.  **安装*简单变压器***
    `pip install simpletransformers`

# 使用

为了演示命名实体识别，我们将使用 [CoNLL](https://www.clips.uantwerpen.be/conll2003/ner/) 数据集。获得这个数据集可能有点棘手，但我在 Kaggle 上找到了一个版本，它可以满足我们的目的。

## 数据准备

1.  从 [Kaggle](https://www.kaggle.com/alaakhaled/conll003-englishversion/download) 下载数据集。
2.  将文本文件解压到`data/`目录。(它应该包含 3 个文本文件`train.txt, valid.txt, test.txt`。我们将使用`train`和`test`文件。您可以使用`valid`文件来执行超参数调整，以提高模型性能。

*简单变形金刚的 NER 模型既可以使用* `*.txt*` *文件，也可以使用熊猫* `*DataFrames*` *。关于* `*DataFrames*` *的使用示例，请参考* [*回购*](https://github.com/ThilinaRajapakse/simpletransformers) *单据中的 NER 最小启动示例。*

使用自己的数据集时，输入文本文件应该遵循 CoNLL 格式。文件中的每一行都应该包含一个单词及其相关的标记，每个标记之间用一个空格隔开。简单的转换器假定一行中的第一个“单词”是实际的单词，一行中的最后一个“单词”是它的指定标签。为了表示一个新的句子，在前一个句子的最后一个单词和下一个句子的第一个单词之间添加一个空行。然而，在使用定制数据集时，使用`DataFrame`方法可能更容易。

## 神经模型

我们创建了一个`NERModel`,可用于 NER 任务中的训练、评估和预测。一个`NERModel`对象的完整参数列表如下。

*   `model_type`:型号类型(伯特、罗伯塔)
*   `model_name`:默认的变压器模型名称或包含变压器模型文件的目录路径(pytorch_nodel.bin)。
*   `labels`(可选):所有命名实体标签的列表。如果没有给出，将使用["O "，" B-MISC "，" I-MISC "，" B-PER "，" I-PER "，" B-ORG "，" I-ORG "，" B-LOC "，" I-LOC"]。
*   `args`(可选):如果未提供此参数，将使用默认参数。如果提供的话，它应该是一个包含应该在默认参数中更改的参数的字典。
*   `use_cuda`(可选):如果可用，使用 GPU。设置为 False 将强制模型仅使用 CPU。

要加载一个先前保存的模型而不是默认模型，您可以将`model_name` 更改为包含已保存模型的目录路径。

```
model = NERModel(‘bert’, ‘path_to_model/’)
```

一个`NERModel`包含一个 python 字典`args`,它有许多属性提供对超参数的控制。在[回购文件](https://github.com/ThilinaRajapakse/simpletransformers)中提供了每一个的详细描述。默认值如下所示。

当创建一个`NERModel`或调用它的`train_model`方法时，只要传入一个包含要更新的键值对的`dict`就可以修改这些属性。下面给出一个例子。

## 训练模型

正如承诺的那样，训练可以在一行代码中完成。

`train_model`方法将在每第 n 步创建一个模型的检查点(保存)，其中 n 是`self.args['save_steps']`。训练完成后，最终模型将保存到`self.args['output_dir']`。

加载保存的模型如下所示。

## 评估模型

同样，评估只是一行代码。

这里，三个返回值是:

*   `result`:包含评估结果的字典。(eval_loss，precision，recall，f1_score)
*   `model_outputs`:原始模型输出列表
*   `preds_list`:预测标签列表

这里给出我得到的评测结果，供参考。

```
{'eval_loss': 0.10684790916955669, 'precision': 0.9023580786026201, 'recall': 0.9153082919914954, 'f1_score': 0.9087870525112148}
```

对于使用默认超参数值的单次运行来说，还不算太差！

## 把所有的放在一起

## 预测和测试

在实际应用中，我们通常不知道真正的标签应该是什么。要对任意示例执行预测，可以使用`predict`方法。这个方法与`eval_model`方法非常相似，除了它接受一个文本列表并返回一个预测列表和一个模型输出列表。

```
predictions, raw_outputs = model.predict(["Some arbitary sentence"])
```

# 包扎

Simple Transformers 提供了一种快速简单的方法来执行命名实体识别(以及其他令牌级分类任务)。借用伯特背后的人的一句台词，简单变形金刚“概念简单，经验丰富”。