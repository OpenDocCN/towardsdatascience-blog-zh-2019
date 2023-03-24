# 简单的变压器——介绍使用伯特、罗伯塔、XLNet 和 XLM 的最简单方法

> 原文：<https://towardsdatascience.com/simple-transformers-introducing-the-easiest-bert-roberta-xlnet-and-xlm-library-58bf8c59b2a3?source=collection_archive---------6----------------------->

## 想要为 NLP 使用 Transformer 模型吗？一页页的代码让你沮丧？不再是了，因为简单的变形金刚正在工作。只需 3 行代码就可以启动、训练和评估变形金刚！

![](img/7c479c5160293291ab3d4ea6e98f1f38.png)

# 前言

简单的变形金刚库是通过拥抱脸作为优秀的 T2 变形金刚库的包装而构建的。我永远感谢在拥抱脸的人们所做的艰苦工作，使公众能够方便地访问和使用变形金刚模型。没有你们我真不知道该怎么办！

# 介绍

我相信可以公平地说，Transformer 模型的成功在推进自然语言处理领域取得了惊人的成就。他们不仅在许多他们被设计解决的 NLP 任务上表现出惊人的飞跃，预先训练过的变形金刚在迁移学习上也表现得出奇的好。这意味着任何人都可以利用训练这些模型的长时间和令人难以置信的计算能力来执行无数种 NLP 任务。你不再需要谷歌或脸书的雄厚财力来构建一个最先进的模型来解决你的 NLP 问题了！

或者人们可能希望如此。事实是，让这些模型发挥作用仍然需要大量的技术知识。除非你在深度学习方面有专业知识或至少有经验，否则这似乎是一个令人生畏的挑战。我很高兴地说，我以前关于变形金刚的文章(这里是[这里是](/https-medium-com-chaturangarajapakshe-text-classification-with-transformer-models-d370944b50ca)和[这里是](https://medium.com/swlh/a-simple-guide-on-using-bert-for-text-classification-bbf041ac8d04))似乎已经帮助了很多人开始使用变形金刚。有趣的是，我注意到不同背景的人(语言学、医学、商业等等)都在尝试使用这些模型来解决他们自己领域的问题。然而，为了使变压器适应特定的任务，需要克服的技术障碍并非微不足道，甚至可能相当令人沮丧。

# 简单变压器

这个难题是我决定开发一个简单的库来使用 Transformers 执行(二进制和多类)文本分类(我见过的最常见的 NLP 任务)的主要动机。我们的想法是让它尽可能简单，这意味着抽象出许多实现和技术细节。库的实现可以在 [Github](https://github.com/ThilinaRajapakse/simpletransformers) 上找到。我强烈建议您查看它，以便更好地了解一切是如何工作的，尽管使用该库并不需要了解内部细节。

为此，我们编写了简单的 Transformers 库，只需 3 行代码就可以初始化 Transformer 模型，在给定的数据集上进行训练，并在给定的数据集上进行评估。让我们看看这是怎么做的，好吗？

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
    `pip install tensorboardx`
3.  安装*简单变压器*。
    

# 使用

快速浏览一下如何在 Yelp 评论数据集上使用这个库。

1.  下载 [Yelp 评论数据集](https://s3.amazonaws.com/fast-ai-nlp/yelp_review_polarity_csv.tgz)。
2.  提取`train.csv`和`test.csv`并放入目录`data/`中。

*(Bash 用户可以使用* [*这个脚本*](https://github.com/ThilinaRajapakse/pytorch-transformers-classification/blob/master/data_download.sh) *来下载数据集)*

这里没有什么特别的，我们只是以正确的形式得到数据。对于任何数据集，这都是您必须做的。

*   为训练和评估部分创建两个熊猫`DataFrame`对象。
*   每个`DataFrame`应该有两列。第一列包含您想要训练或评估的文本，数据类型为`str`。第二列有相应的标签，数据类型为`int`。
    *更新:现在建议将列命名为* `*labels*` *和* `*text*` *而不是依赖于列的顺序。*

数据整理好了，就该训练和评估模型了。

就是这样！

为了对其他文本进行预测，`TransformerModel`附带了一个`predict(to_predict)`方法，它给出了一个文本列表，返回模型预测和原始模型输出。

有关所有可用方法的更多详细信息，请参见 [Github repo](https://github.com/ThilinaRajapakse/simpletransformers) 。repo 还包含一个使用该库的最小示例。

# 默认设置以及如何更改它们

下面给出了使用的默认参数。这些都可以通过将包含相应键/值对的 dict 传递给 TransformerModel 的 init 方法来覆盖。*(见下面的例子)*

```
self.args = {
   'model_type':  'roberta',
   'model_name': 'roberta-base',
   'output_dir': 'outputs/',
   'cache_dir': 'cache/', 'fp16': True,
   'fp16_opt_level': 'O1',
   'max_seq_length': 128,
   'train_batch_size': 8,
   'eval_batch_size': 8,
   'gradient_accumulation_steps': 1,
   'num_train_epochs': 1,
   'weight_decay': 0,
   'learning_rate': 4e-5,
   'adam_epsilon': 1e-8,
   'warmup_ratio': 0.06,
   'warmup_steps': 0,
   'max_grad_norm': 1.0, 'logging_steps': 50,
   'evaluate_during_training': False,
   'save_steps': 2000,
   'eval_all_checkpoints': True,
   'use_tensorboard': True, 'overwrite_output_dir': False,
   'reprocess_input_data': False,
}
```

要覆盖其中任何一个，只需向 TransformerModel 传递一个带有适当的键/值对的字典。

关于每个参数的解释，请参考 [Github repo](https://github.com/ThilinaRajapakse/simpletransformers) 。

*更新:当前参数见* [*文档*](https://simpletransformers.ai/docs/usage/#configuring-a-simple-transformers-model) *。*

# 结论

那都是乡亲们！据我所知，使用变压器模型的最简单方法。