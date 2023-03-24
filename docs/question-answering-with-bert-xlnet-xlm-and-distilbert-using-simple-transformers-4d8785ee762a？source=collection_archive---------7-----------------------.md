# 伯特、XLNET、XLM 和迪沃伯特使用简单变压器回答问题

> 原文：<https://towardsdatascience.com/question-answering-with-bert-xlnet-xlm-and-distilbert-using-simple-transformers-4d8785ee762a?source=collection_archive---------7----------------------->

## 问题:如何使用变形金刚进行问答？回答:简单的变形金刚，咄！(看到我在那里做了什么吗？)

![](img/4770949eeebdc954d653ef57b1288d28.png)

Photo by [Camylla Battani](https://unsplash.com/@camylla93?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 自然语言处理中的问题回答

> 背景:问答(QA)是信息检索和自然语言处理(NLP)领域中的一门计算机科学学科，它涉及到构建自动回答人类用自然语言提出的问题的系统。
> 
> **人类**:什么是问答系统？
> **系统**:自动回答人类用自然语言提出的问题的系统

QA 在大量的任务中有应用，包括信息检索、实体提取、聊天机器人和对话系统等等。虽然回答问题可以有多种方式，但最常见的问答方式可能是从给定的上下文中选择答案。换句话说，系统将从正确回答问题的上下文中挑选一个*范围*的文本。如果无法从上下文中找到正确答案，系统将只返回一个空字符串。

使用预先训练的 Transformer 模型进行迁移学习在 NLP 问题中已经变得无处不在，问答也不例外。考虑到这一点，我们将使用 BERT 来处理问答任务！

我们将使用[简单变形金刚](https://github.com/ThilinaRajapakse/simpletransformers)库来轻松处理变形金刚模型。

[](https://github.com/ThilinaRajapakse/simpletransformers) [## ThilinaRajapakse/简单变压器

### 这个库是基于 HuggingFace 的变形金刚库。简单的变形金刚让你快速训练和…

github.com](https://github.com/ThilinaRajapakse/simpletransformers) 

简单的变形金刚是建立在高超的[拥抱脸变形金刚](https://github.com/huggingface/transformers)库之上的。

# 设置

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

# 数据集

我们将使用[斯坦福问答数据集(SQuAD 2.0)](https://rajpurkar.github.io/SQuAD-explorer/) 来训练和评估我们的模型。SQuAD 是一个阅读理解数据集，也是 QA 模型的标准基准。该数据集在网站上公开提供。

下载数据集并将文件(train-v2.0.json，dev-v2.0.json)放在`data/`目录中。

# 数据准备

为了在简单的转换器中执行 QA，数据必须以正确的格式保存在 JSON 文件或 Python 字典列表中。

如果使用 JSON 文件，这些文件应该包含一个字典列表。字典代表一个上下文及其相关的问题。

每个这样的字典包含两个属性，`"context"`和`"qas"`。

*   `context`:提问的段落或文本。
*   `qas`:问答列表。

问题和答案被表示为字典。`qas`中的每个字典都有以下格式。

*   `id` : (string)问题的唯一 ID。在整个数据集中应该是唯一的。
*   `question`:(字符串)一个问题。
*   `is_impossible` : (bool)表示是否能从上下文中正确回答问题。
*   `answers`:(列表)问题正确答案列表。

单个答案由具有以下属性的字典表示。

*   `answer`:(字符串)问题的答案。必须是上下文的子字符串。
*   `answer_start` : (int)上下文中答案的起始索引。

我们可以很容易地将球队数据转换成这种格式。

# 问答模型

Simple Transformers 有一个类，可以用于每个受支持的 NLP 任务。这个类的对象用于执行训练、评估(当基本事实已知时)和预测(当基本事实未知时)。

这里，我们创建一个`QuestionAnsweringModel`对象，并设置超参数来微调模型。第一个参数是*型号类型*，第二个是*型号名称。*

`args`参数接受一个可选的 Python 字典，其中包含超参数值和配置选项。我强烈推荐在这里查看所有选项。

默认值如下所示。

要加载一个先前保存的模型而不是默认模型，您可以将 *model_name* 更改为包含已保存模型的**目录**的路径。

```
model = QuestionAnsweringModel('bert', 'path_to_model/')
```

# 培养

训练模型是一行程序！只需将`train_data`传递给`train_model`函数。

您还可以通过向`train_model`方法传递一个包含相关属性的`dict`来更改超参数。注意，这些修改**将持续**甚至在训练完成后。

`train_model`方法将在每 *n* 步创建一个模型的检查点(保存)，其中 *n* 为`self.args['save_steps']`。训练完成后，最终模型将保存到`self.args['output_dir']`。

# 估价

dev 数据的正确答案没有在小队数据集中提供，但是我们可以将我们的预测上传到[小队](https://rajpurkar.github.io/SQuAD-explorer/)网站进行评估。或者，您可以将训练数据分成训练和验证数据集，并使用`model.eval_model()`方法在本地验证模型。

对于本指南，我将简单地把预测上传到小队。

分解这些代码，我们读入开发数据，将其转换成正确的格式，获得模型预测，最后以所需的提交格式写入 JSON 文件。

# 结果

使用这些超参数获得的结果如下所示。

```
"exact": 67.24500968584182, 
"f1": 70.47401515405956, 
"total": 11873, 
"HasAns_exact": 64.1025641025641, 
"HasAns_f1": 70.56983500744732, 
"HasAns_total": 5928, 
"NoAns_exact": 70.3784693019344, 
"NoAns_f1": 70.3784693019344, 
"NoAns_total": 5945
```

小队 2.0 是一个具有挑战性的基准，这反映在这些结果中。一些超参数调整应该会提高这些分数。同样，使用一个`large`模型而不是`base`模型也会显著提高结果。