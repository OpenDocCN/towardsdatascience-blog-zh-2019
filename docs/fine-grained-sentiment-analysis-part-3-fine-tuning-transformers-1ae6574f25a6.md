# 细粒度情感分析(第 3 部分):微调变压器

> 原文：<https://towardsdatascience.com/fine-grained-sentiment-analysis-part-3-fine-tuning-transformers-1ae6574f25a6?source=collection_archive---------10----------------------->

## PyTorch 中使用预训练变压器的动手迁移学习

![](img/ade0661d5dd594530a2c286332b81709.png)

Source: [Pixabay](https://pixabay.com/illustrations/light-bulbs-light-bulb-light-energy-1125016/)

这是 Python 中细粒度情感分析系列的第 3 部分。[第 1 部分](/fine-grained-sentiment-analysis-in-python-part-1-2697bb111ed4)和[第 2 部分](/fine-grained-sentiment-analysis-in-python-part-2-2a92fdc0160d)涵盖了对斯坦福情感树库细粒度(SST-5)数据集上六种不同分类方法的分析和解释。在这篇文章中，我们将看看如何通过建立一个基于 **transformer-** 的模型和应用迁移学习来改善过去的结果，这是一种最近已经[统治 NLP 任务排行榜](https://hackingsemantics.xyz/2019/leaderboards/)的强大方法。

从本系列的[第一篇文章](/fine-grained-sentiment-analysis-in-python-part-1-2697bb111ed4)中，以下分类准确度和 F1 分数是在 SST-5 数据集上获得的:

![](img/97182a9043c8db347f89d0bc8f779a5f.png)

在下面的章节中，我们将讨论关键的培训、评估和解释步骤，这些步骤说明了为什么 transformers 比上面列出的方法更适合这项任务。

# 是什么让变形金刚如此强大？

transformer 架构的核心是以下关键思想，这些思想使其非常适合解释自然语言中的复杂模式:

*   **自我关注**:这是一种机制，变形金刚用它来表示输入序列的不同位置，并学习它们之间的关系。
*   **多头注意**:变形金刚将自我注意的任务委托给多个“头”，即[从不同的表征子空间共同注意来自序列](http://nlp.seas.harvard.edu/2018/04/03/attention.html#training-loop)中不同位置的信息。这允许他们使用*无监督学习*有效地扩展(大型数据集)。
*   **迁移学习**:最近领先的基于 transformer 的方法是通过*迁移学习*实现的——即使用从之前的设置(例如无监督训练)中提取的知识，并将其应用于不同的设置(例如情感分类或问题回答)。这是通过两个阶段的过程实现的:*预调整*，随后是*微调*，或*适应。*

![](img/96e1f7ac01511faf3dae13cb6e4f57c8.png)

Source: [“The State of Transfer Learning in NLP” by Sebastian Ruder](http://ruder.io/state-of-transfer-learning-in-nlp/)

本质上，最近所有基于 transformer 的顶级方法( [GPT](https://s3-us-west-2.amazonaws.com/openai-assets/research-covers/language-unsupervised/language_understanding_paper.pdf) 、[伯特](https://arxiv.org/pdf/1810.04805.pdf)、 [XLNet](https://arxiv.org/pdf/1906.08237.pdf) )都使用这种顺序迁移学习方法。在内部，他们在预训练阶段使用来自大型语料库的未标记数据训练一个*语言模型*，随后一个额外的特定任务模块(附在语言模型的下游)根据定制数据进行微调。下面是这种方法在多个基准数据集上的可视化。

![](img/0f0e066fdadcce54b2892622550016d1.png)

Source: [Devlin et al. 2019](https://arxiv.org/pdf/1810.04805.pdf)

# 预培训目标的重要性

预训练步骤包括以无监督的方式训练语言模型——这决定了模型如何从给定的训练语料库中学习语法、语义和上下文信息。最近的证据(来自 OpenAI 的 GPT-2 等模型)表明，如果有足够的数据和计算，真正大型的语言模型将会学到很多关于语言语法的知识！对于本文后面描述的 transformer 模型，下面的预处理目标值得注意。

*   **OpenAI GPT** 在预训练过程中使用了一个*从左到右*语言建模目标——也就是说，它学习从左到右预测序列中最有可能的下一个单词，就像它们在自然语言中出现一样(至少对于英语来说)。这种模型通常被称为*经典*语言模型，或者 ***因果*** 语言模型。这里的“因果”一词指的是这样一个事实:一个标记出现在特定位置的可能性是由出现在它之前的标记序列引起的。
*   **伯特**在其核心使用一个*屏蔽*语言模型，通过在预训练期间随机屏蔽一个序列的 15%的标记获得——这允许模型学习如何预测在前一个标记之前或之后出现的标记(它被双向训练*，不像 GPT)。除了屏蔽之外，BERT 还使用了下一句预测目标，该模型学习预测一句话是否在前一句话之后出现。与因果语言建模相比，这种方法需要较长的训练时间，因为屏蔽一部分标记会导致较小的训练信号。*
*   ***XLNet** 使用*置换*语言建模目标——与 BERT 不同，它在预训练期间随机屏蔽序列中的每个标记(不仅仅是 15%)。通过这种方式，模型学习预测两个方向上序列的*随机记号*，允许模型学习记号之间的*依赖性*(不仅仅是在给定序列中哪些记号最有可能)。可以想象，这在训练时间方面更加昂贵，并且需要更大的语言模型来获得良好的基线。*

*在这篇文章的其余部分，我们将使用一个使用**因果**语言建模目标(类似于 GPT/GPT-2，但比它小得多)训练的变压器模型。*

# *制造变压器*

*有了这些背景知识，我们可以继续写一些代码了！*

## *读取数据集*

*使用 Pandas 对 SST-5 数据集进行了一些基本的预处理。注意，类标签递减 1(在范围[0，1，2，3，4]内)，因为 PyTorch 期望标签是零索引的。*

## *对数据进行令牌化和编码*

*处理数据的第一步是使用单词块标记化器进行标记化— [ [参见本文第 4.1 节](https://arxiv.org/pdf/1609.08144.pdf)了解更多详情]。我们使用 HuggingFace 的 `[pytorch_transformers](https://huggingface.co/pytorch-transformers/model_doc/bert.html?highlight=berttokenize#pytorch_transformers.BertTokenizer)` [库](https://huggingface.co/pytorch-transformers/model_doc/bert.html?highlight=berttokenize#pytorch_transformers.BertTokenizer)中实现的`BertTokenizer` [。接下来，标记化的文本被编码成整数序列，由我们的 transformer 模型处理。随后，创建一个](https://huggingface.co/pytorch-transformers/model_doc/bert.html?highlight=berttokenize#pytorch_transformers.BertTokenizer) [PyTorch](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) `[DataLoader](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader)`将样本装载到批次中进行训练。*

*[Oliver Atanaszov 写了一个非常好的 TextProcessor 类，封装了标记化、编码和数据准备步骤(用于 PyTorch)。这个类利用多个 CPU 内核来加速编码过程，所以在这里对它进行了改编和重用。](https://medium.com/swlh/transformer-fine-tuning-for-sentiment-analysis-c000da034bb5)*

*请注意，在本例中，我们将序列的最大长度固定为 256——理论上，BERT 和类似模型编码器可以处理的[最大序列长度为 512，但由于 SST-5 是一个样本相对较短的相对较小的基准数据集，我们将最大令牌序列长度截断为 256，以减少内存使用和模型大小。](https://github.com/google-research/bert/blob/master/README.md)*

*一个特殊的分类标记`‘[CLS]’`被添加到每个序列的末尾——这个标记在分类任务中被用作每个序列表示的集合，以了解该序列属于哪个类。对于短于 256 的序列，添加一个填充标记`‘[PAD]’`以确保所有批次在训练期间保持相同的大小加载到 GPU 内存中。*

## *构建变压器模型*

*用于分类任务的变压器组的一般结构如下所示。这是 Vaswani 等人用于机器翻译的原始版本的修改形式。*

*![](img/de8c0118c34853d2e247ac4d0aaaf404.png)*

*Source: [Child et al.](https://arxiv.org/pdf/1904.10509.pdf)*

*在 PyTorch 代码中，上面的结构如下所示。*

*定义了一个从 PyTorch 的`nn.module`继承而来的基类`Transformer`。输入序列(在我们的例子中，是用于情感分类的文本)通过对序列的标记和位置嵌入求和而被馈送到变换器块。每个连续的变压器组由以下模块组成:*

*   ***层标准化**:本质上，标准化应用于训练期间的小批量，以使优化更容易(从数学上讲)——与未标准化的网络相比，这提高了模型的性能。层标准化允许通过改变维度来选择任意大小的小批量，在该维度上为每个批量计算批量统计数据(平均值/方差)——经验表明，这可以提高递归神经网络(RNNs)中顺序输入的性能。要获得图层规范化的直观解释，请阅读 Keita Kurita 的这篇博客文章。*
*   ***自我关注**:使用 PyTorch 的`MultiHeadAttention`模块封装变形金刚的自我关注逻辑——即变形金刚对位置信息进行编码并在训练过程中从中学习的能力。*
*   ***漏失**:神经网络中减少过拟合的经典正则化技术——这是通过[在训练过程中引入随机噪声](https://machinelearningmastery.com/dropout-for-regularizing-deep-neural-networks/)来实现的。*
*   ***前馈模块**:执行前向传递，包括另一层归一化，带有一个隐藏层和一个非线性(通常为 [ReLU](https://machinelearningmastery.com/rectified-linear-activation-function-for-deep-learning-neural-networks/) 或 [GELU](https://datascience.stackexchange.com/questions/49522/what-is-gelu-activation) )，然后是 dropout。*

*在我们网络的前馈模块中定义了两个“掩码”。*

*   ***填充掩码**:这屏蔽了之前引入的填充标记(`[‘PAD’]`)，使每个批次的每个序列长度相同。这样做告诉模型在推断过程中屏蔽掉这些标记，以便它们被自我关注模块忽略。填充遮罩特定于每个批次。*

*![](img/c21b536a47038cc07a02d894dc5609f3.png)*

*Padding mask: [NAACL 2019 transferlearning tutorial slides](https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/preview?pru=AAABaz2o8Jk*rdDLH7fXP7h4HQFLtzvHNQ#slide=id.g5888218f39_177_4)*

*   ***注意屏蔽**:由于该方法使用因果语言模型，所以包括了注意屏蔽——这按照因果语言模型屏蔽了下面输入中的先前标记。通过将主对角线以上的所有元素的值设置为负无穷大，使用上三角矩阵(`[torch.triu](https://pytorch.org/docs/stable/torch.html#torch.triu)`)为所有批次指定相同的掩码。*

*![](img/e9db4d389983181bae561cc34bea98fa.png)*

*Attention mask: [NAACL 2019 transferlearning tutorial slides](https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/preview?pru=AAABaz2o8Jk*rdDLH7fXP7h4HQFLtzvHNQ#slide=id.g5888218f39_177_4)*

## *添加分类标题*

*模型的下游部分在现有转换器的顶部使用线性分类层。`TransformerWithClfHead`类继承自基础`Transformer`类，并将`CrossEntropyLoss`指定为要优化的损失函数。线性层的大小为`[embedding_dimensions, num_classes]` —在这种情况下，对于现有的预训练模型和 SST-5 数据集，为 410×5。*

*从分类层中提取原始输出，即逻辑值，并将其提供给 softmax 函数，以生成分类概率向量(1×5)作为输出。*

## *训练模型*

*从由 HuggingFace、提供的[预训练模型初始化模型的权重，并且在 SST-5 数据集上运行训练脚本[](https://github.com/huggingface/naacl_transfer_learning_tutorial/blob/master/utils.py) `[training/train_transformer.py](https://github.com/prrao87/fine-grained-sentiment/blob/master/training/train_transformer.py)`。*

*以下超参数用于训练模型——注意嵌入维度的数量、注意头的数量等。没有显式设置—这些是从预训练模型继承的。在 3 个训练时期之后，对模型进行检查点检查，并保存其配置参数。*

*![](img/7dcbfaf57555c39924f5ba6d8d114520.png)*

*💡**线性预热时间表**:来自 `[pytorch-ignite](https://pytorch.org/ignite/contrib/handlers.html#ignite.contrib.handlers.param_scheduler.PiecewiseLinear)`的[分段线性时间表不是设置一个恒定的学习速率，而是定义为在训练的早期阶段提高学习速率，然后线性下降到零。这通常是一种在迁移学习过程中确保知识良好迁移的好方法(类似于](https://pytorch.org/ignite/contrib/handlers.html#ignite.contrib.handlers.param_scheduler.PiecewiseLinear) [ULMFiT，Howard and Ruder，2018](https://arxiv.org/pdf/1801.06146.pdf) )中的“倾斜三角形学习率”。*

*💡**梯度累积**:正如 Thomas Wolf 在他的文章“ [*训练神经网络的实用技巧*](https://medium.com/huggingface/training-larger-batches-practical-tips-on-1-gpu-multi-gpu-distributed-setups-ec88c3e51255) ”中描述的那样，模拟较大批量而不会遇到 GPU 内存问题的一个好方法是累积梯度。这是通过对来自多个反向传播步骤的梯度张量求和，然后*调用优化器使损失最小化来实现的——注意，损失还需要除以累积步骤的数量。这样做可以让我们用比 GPU 内存中实际容纳的更大的批量进行训练，从而改善模型的学习。**

## *模型推理*

*在本节中，我们将通过一个示例，逐行深入研究如何使用一个经过训练的模型对我们的 SST-5 情感数据集进行推断。输入文本被标记化，转换成整数 id，并作为适当形状的张量馈送给模型，如下面的笔记本所示。*

*这个过程使用文件`[classifiers.py](https://github.com/prrao87/fine-grained-sentiment/blob/master/classifiers.py)`中的`TransformerSentiment`类封装，以将 SST-5 数据集读入 Pandas 数据帧，并评估转换器的性能。*

# *估价*

*在 SST-5 测试上运行经过训练的 transformer 模型([在这个 Google drive 链接](https://drive.google.com/open?id=1_6PoXN5THRB9Px-r7GwVx-8H7fNwNAIH)中可用)，我们可以看到因果 transformer 将分类准确率提高了近 50%！*

*![](img/6e8edb0479d6ccf0c0056a6145043b48.png)*

*transformer 的 macro-F1 分数也比其他基于嵌入的方法(FastText 和 Flair)有所提高。*

*![](img/57a781aaadf6f46d0a222e513f40445f.png)*

*Each cell in the confusion matrix shows the **percentage** of predictions made for the corresponding true label.*

*下一个最好的模型 Flair 的混淆矩阵被放在变压器的混淆矩阵旁边进行比较。*

*![](img/8bc2169f55ce574201572843fd66a572.png)*

*转换器确实做出了许多属于类别标签 2 和 4 的错误预测，但是，与所有其他方法相比，它获得了更多属于少数类别(1 和 3)的正确标签。它能够用如此有限数量的训练样本合理地分类这些少数类的事实证明了使用预训练语言模型和迁移学习进行情感分类任务的能力。*

# *解释变压器的预测*

*按照本系列的第 2 部分中所示的模型解释方法，我们使用我们训练好的 transformer 模型对 SST-5 测试集中的特定文本示例进行 LIME explainer 可视化。用于生成以下可视化效果的代码可在文件`[explainer.py](https://github.com/prrao87/fine-grained-sentiment/blob/master/explainer.py)`中获得——经过训练的模型文件也可在这个 Google drive 链接中获得[。](https://drive.google.com/open?id=1_6PoXN5THRB9Px-r7GwVx-8H7fNwNAIH)*

***例一:***不是恐怖，只是平庸得可怕。**

*![](img/b1bdeef2c861f2929adb43130dc21647.png)*

***True: 1 — Predicted: 1***

*变压器似乎正确地将单词“*可怕的*”和“*一般的*”识别为促成该示例具有类别标签 1(即，强烈否定)的两个最重要的特征。*而非*这个词将预测概率推离 1(大概是为了标注 2 或 3)，但其效果没有句子中的否定词那么强。副词“*可怕地*”在将这个句子归类为强烈否定中也起了很小的作用，这意味着该模型对副词等修饰语如何改变句子中的情感程度有所了解。*

*例 2:演员阵容都很出色……但这部电影本身只是有点迷人。*

*![](img/7009f006e550ddd7647ee4b0c621aa8c.png)*

***True: 3 — Predicted: 3***

*在此示例中，很明显，单词“ *but* ”在变压器对类别标签 3(中性)的预测中具有最大的权重。这很有意思，因为虽然在这句话中有很多词表现出不同程度的极性("*优秀的*"、"*仅仅是"*、"*温和的*"和"*迷人的*")，但是"*但是*"这个词在这句话中充当了一个关键的修饰语——它标志着句子在后半部分从强烈肯定到轻微否定的过渡。再一次，转换者似乎知道修饰语(在这种情况下是连词)如何改变句子的整体情感程度。*

*另一个有趣的观察是单词“ *cast* ”在被预测为中性的情绪(类别 3)中不起作用。**所有之前使用的方法([参见第 2 部分](/fine-grained-sentiment-analysis-in-python-part-2-2a92fdc0160d))，包括 Flair，都错误地将单词“ *cast* 的特征重要性学习为有助于情感——因为这个单词在训练数据中多次出现。转换器的因果语言模型的潜在功能有助于它在单词和预测的情感标签之间建立*更有意义的关联，而不管该单词在训练集中出现了多少次。****

# *结束语*

*通过这个系列，我们探索了 Python 中的各种 NLP 方法，用于在斯坦福情感树库(SST-5)数据集上进行细粒度分类。虽然这个数据集极具挑战性，并给现有的文本分类方法带来了许多问题，但很明显，与其他方法相比，NLP 中的当前行业标准——结合迁移学习的 transformers 在分类任务上表现出内在的优异性能。这主要是由于转换器的底层语言模型表示，这给予它更多的上下文意识和对训练词汇的更好的句法理解。*

*下图总结了 SST-5 数据集上准确性和 F1 分数的连续改进(使用越来越复杂的模型)。*

*![](img/4d0f365fa1b7888f562a3932a65395f8.png)*

## *附加实验*

*为了从转换器中挤出更多的性能，可以尝试一些额外的实验(除了增加模型中的参数数量之外):*

*   ***特定领域数据**:目前的方法使用因果语言模型(即 HuggingFace 的预训练模型)，该模型有 5000 万个可训练参数。如 [NAACL 教程](https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/preview?pru=AAABaz2o8Jk*rdDLH7fXP7h4HQFLtzvHNQ#slide=id.g5888218f39_177_4)中所述，该模型在 Wikitext-103(即来自维基百科文章的 1.03 亿个标记)上进行了预训练。用更相关的特定领域数据(例如几十万条电影评论)来增加预训练可以帮助模型更好地理解 SST-5 中发现的典型词汇。这一步虽然很昂贵——因为它需要重新训练语言模型——但它是一次性的步骤([,可以使用 HuggingFace 的存储库](https://github.com/huggingface/naacl_transfer_learning_tutorial)中的 `[pretraining_train.py](https://github.com/huggingface/naacl_transfer_learning_tutorial)` [文件来完成),并且可以在分类准确性方面产生显著的下游改进。](https://github.com/huggingface/naacl_transfer_learning_tutorial)*
*   ***掩蔽语言建模**:使用因果语言模型作为预训练目标，虽然训练成本更低，但可能不是获得高精度分类结果的最佳方式。将掩蔽语言建模作为预训练目标进行实验(同时用电影评论增加训练数据)可以产生更准确的语言模型。事实上， [NAACL 教程](https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/preview?pru=AAABaz2o8Jk*rdDLH7fXP7h4HQFLtzvHNQ#slide=id.g5888218f39_177_4)表明，与因果语言模型相比，使用掩蔽语言建模目标的预训练产生了更低的[](/perplexity-intuition-and-derivation-105dd481c8f3)*复杂度，这意味着该模型可以在情感分类等下游任务中表现得更好。**
*   ****超参数调整**:试验更大的批量(用于训练和测试)、增加梯度累积步骤和查看不同的学习率预热时间表可以在测试集上产生额外的性能增益。**

**如本系列所述，从头开始构建情感分析框架的目的是建立对我们能力的信心，即*评估*和*解释*不同的机器学习技术，与我们的问题陈述和可用数据相关。当然，利用大型预训练的基于 transformer 的模型，如 BERT-large、XLNet 或 RoBERTa，可以显著提高真实数据集的性能——然而，重要的是平衡使用这些模型的巨大计算成本与使用具有良好预训练目标和干净、*良好注释的*、*领域特定的*训练数据的更小、更简单的模型。随着未来几个月越来越多的 NLP 技术在变形金刚的巨大成功上展开，未来只会有有趣的时代！**

## **代码和训练模型**

*   **使用 transformer 模型对 SST-5 进行训练和预测的代码在这个项目的 GitHub repo 中。**
*   **经过训练的变形金刚模型[可以在这个 Google drive 链接](https://drive.google.com/open?id=1_6PoXN5THRB9Px-r7GwVx-8H7fNwNAIH)中找到。**

**请随意使用 transformer 重现结果并做出您自己的发现！**

## **承认**

**这篇文章使用的所有代码都是从以下链接中的优秀示例代码改编而来的:**

*   **[用于 NAACL 2019 迁移学习 NLP 教程的 GitHub 资源库](https://github.com/huggingface/naacl_transfer_learning_tutorial)**
*   **[Google Colab 笔记本](https://colab.research.google.com/drive/1iDHCYIrWswIKp-n-pOg69xLoZO09MEgf)展示变形金刚模型实验**
*   **[本文由 Oliver Atanaszov](https://medium.com/swlh/transformer-fine-tuning-for-sentiment-analysis-c000da034bb5) 在变压器微调上发表**

## **进一步阅读**

**为了更详细地深入了解 NLP 中的迁移学习，强烈建议浏览 Sebastian Ruder、Matthew Peters、Swabha Swayamdipta 和 Thomas Wolf 的 [NAACL 2019 教程幻灯片](https://docs.google.com/presentation/d/1fIhGikFPnb7G5kr58OvYC3GN4io7MznnM0aAgadvJfc/preview?pru=AAABaz2o8Jk*rdDLH7fXP7h4HQFLtzvHNQ#slide=id.g5888218f39_177_4)。**