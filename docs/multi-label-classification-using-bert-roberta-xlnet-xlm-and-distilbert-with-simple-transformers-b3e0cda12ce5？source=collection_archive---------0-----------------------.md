# 使用 BERT、RoBERTa、XLNet、XLM 和 DistilBERT 以及简单变压器进行多标签分类

> 原文：<https://towardsdatascience.com/multi-label-classification-using-bert-roberta-xlnet-xlm-and-distilbert-with-simple-transformers-b3e0cda12ce5?source=collection_archive---------0----------------------->

## 了解如何使用 Transformer 模型，仅用简单的 Transformer 的 3 行代码来执行多标签分类。

![](img/a0ea16ccf9b7b335a2a94056f350ac53.png)

Photo by [russn_fckr](https://unsplash.com/@russn_fckr?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/choice?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

# 前言

[简单变形金刚](https://github.com/ThilinaRajapakse/simpletransformers)库是在优秀的[变形金刚](https://github.com/huggingface/transformers)库的基础上通过抱脸的方式构建的。你们太不可思议了！

简单变压器现在支持:

*   [二进制分类](/simple-transformers-introducing-the-easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3)
*   [多类分类](https://medium.com/swlh/simple-transformers-multi-class-text-classification-with-bert-roberta-xlnet-xlm-and-8b585000ce3a)
*   [命名实体识别](/simple-transformers-named-entity-recognition-with-transformer-models-c04b9242a2a0)(以及类似的令牌级任务)
*   **多标签分类**

还有很多正在筹备中。

# 介绍

变压器模型和迁移学习方法继续以惊人的速度推动自然语言处理领域向前发展。然而，最先进的性能往往是以大量(复杂)代码为代价的。

Simple Transformers 避免了所有的复杂性，让您开始关注重要的事情，训练和使用 Transformer 模型。绕过所有复杂的设置、样板文件和其他常见的不愉快，在一行中初始化一个模型，在下一行中训练，然后用第三行进行评估。

本指南展示了如何使用简单的变压器来执行多标签分类。在多标签分类中，每个样本可以具有来自给定标签集的任意标签组合*(无、一个、一些或全部)*。

*所有的源代码都可以在*[*Github Repo*](https://github.com/ThilinaRajapakse/simpletransformers)*上找到。如果您有任何问题或疑问，这是解决它们的地方。请务必检查一下！*

# 装置

1.  从[这里](https://www.anaconda.com/distribution/)安装 Anaconda 或 Miniconda 包管理器。
2.  创建新的虚拟环境并安装软件包。
    `conda create -n simpletransformers python pandas tqdm`
    `conda activate simpletransformers`
    如果使用 cuda:
    `conda install pytorch cudatoolkit=10.0 -c pytorch`
    其他:
    `conda install pytorch cpuonly -c pytorch`
    `conda install -c anaconda scipy`
    `conda install -c anaconda scikit-learn`
    `pip install transformers`
    `pip install seqeval`
    `pip install tensorboardx`
3.  如果您使用 fp16 培训，请安装 Apex。请遵循此处的说明[。(从 pip 安装 Apex 给一些人带来了问题。)](https://github.com/NVIDIA/apex)
4.  安装简单变压器。
    `pip install simpletransformers`

# 多标签分类

为了演示多标签分类，我们将使用 Kaggle 的[毒性评论](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/)数据集。从上面的链接下载数据集，并将`csv`文件放在`data/`目录中。

## 数据准备

数据集中的评论已经根据下面的标准进行了标注。

*   `toxic`
*   `severe_toxic`
*   `obscene`
*   `threat`
*   `insult`
*   `identity_hate`

数据集包含每个标准的一列，用布尔值 1 或 0 表示注释是否包含相应的毒性。

然而，简单的 Transformers 需要一个包含多热点编码标签列表的列`labels`,以及一个包含所有文本的列`text`(废话！).

让我们将`df`分成训练和评估数据集，这样我们就可以轻松地验证模型。

现在数据集已经可以使用了！

## 多标签分类模型

这创建了一个`MultiLabelClassificationModel`,可用于多标签分类任务的训练、评估和预测。第一个参数是 *model_type，*第二个是 *model_name* ，第三个是数据中标签的个数。

*   `model_type`可能是`['bert', 'xlnet', 'xlm', 'roberta', 'distilbert'].`中的一个
*   关于可用于`model_name`的预训练模型的完整列表，请参考[当前预训练模型](https://github.com/ThilinaRajapakse/simpletransformers#current-pretrained-models)。

`args`参数接受一个可选的 Python 字典，其中包含超参数值和配置选项。我强烈推荐在这里查看所有选项。

默认值如下所示。

要加载一个先前保存的模型而不是默认模型，您可以将 *model_name* 更改为包含已保存模型的**目录**的路径。

```
model = MultiLabelClassificationModel('xlnet', 'path_to_model/', num_labels=6)
```

# 培养

这将根据训练数据训练模型。您还可以通过将包含相关属性的`dict`传递给`train_model`方法来更改超参数。请注意，这些修改**将持续**甚至在训练完成后。

`train_model`方法将在每第 *n* 步创建一个模型的检查点(保存)，其中 *n* 为`self.args['save_steps']`。训练完成后，最终模型将保存到`self.args['output_dir']`。

# 估价

`eval_model`方法用于对评估数据集进行评估。这个方法有三个返回值。

*   result:以`dict`的形式给出评估结果。默认情况下，对于多标注分类，仅报告标注等级平均精度(LRAP)。
*   model_outputs:评估数据集中每个项目的模型输出的一个`list`。如果您需要每个类别的概率，而不是单个预测，这将非常有用。请注意，`sigmoid`函数已应用于每个输出，以压缩 0 和`.`之间的值
*   错误预测:每个错误预测的第`InputFeature`个`list`。文本可以从`InputFeature.text_a`属性中获得。*(*`*InputFeature*`*类可以在* `*utils.py*` *文件中的* [*回购*](https://github.com/ThilinaRajapakse/simpletransformers) *)*

您还可以包括评估中使用的其他指标。只需将度量函数作为关键字参数传递给`eval_model`方法。度量函数应该接受两个参数，第一个是真实标签，第二个是预测。这遵循了 sklearn 标准。

*确保度量函数与多标签分类兼容。*

## 预测/测试

虽然当我们知道正确的标签并且只需要评估模型的性能时,`eval_model`是有用的，但是我们很少在现实世界的任务中知道真正的标签(我确信这里面有一些深刻的哲学)。在这种情况下，`predict`方法就派上了用场。它类似于`eval_model`方法，除了它不需要真正的标签并返回预测和模型输出。

我们可以在有毒评论数据集中提供的测试数据上进行尝试。

将此提交给 Kaggle 使我获得了 0.98468 的**分数，再次证明了自变形金刚和迁移学习出现以来 NLP 取得了多大的进步。请记住，我在这里没有做太多的超参数调优！**

## 结论

伯特及其衍生产品太棒了！我希望简单的变形金刚有助于在使用它们的过程中消除一些障碍。