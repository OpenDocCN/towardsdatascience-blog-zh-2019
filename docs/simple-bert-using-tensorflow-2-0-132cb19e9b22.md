# 使用 TensorFlow 2.0 的简单 BERT

> 原文：<https://towardsdatascience.com/simple-bert-using-tensorflow-2-0-132cb19e9b22?source=collection_archive---------4----------------------->

## 在 15 行代码中使用 BERT 和 TensorFlow Hub。最后更新时间:2020 年 11 月 15 日。

这个故事展示了一个使用 TensorFlow 2.0 嵌入 BERT [1]的简单例子。最近发布了 TensorFlow 2.0，该模块旨在使用基于高级 Keras API 的简单易用的模型。伯特之前的用法在[的长篇笔记本](https://colab.research.google.com/github/google-research/bert/blob/master/predicting_movie_reviews_with_bert_on_tf_hub.ipynb#scrollTo=LL5W8gEGRTAf)中有描述，它实现了一个电影评论预测。在这个故事中，我们将看到一个使用 Keras 和最新的 [TensorFlow](https://www.tensorflow.org/) 和 [TensorFlow Hub](https://www.tensorflow.org/hub) 模块的简单 BERT 嵌入生成器。

Google Colab 上新的[更新版本在这里](https://colab.research.google.com/drive/1PJOmDL7oN_NmLRRbdhQPcABculw8Gbzk?usp=sharing)(2020–11–15)。老版本在这里[有。](https://colab.research.google.com/drive/1hMLd5-r82FrnFnBub-B-fVW78Px4KPX1)

我之前的文章使用了`[bert-embedding](https://pypi.org/project/bert-embedding/)` 模块，使用预先训练的无案例 BERT 基本模型生成句子级和标记级嵌入。这里，我们将只通过几个步骤来实现这个模块的用法。

![](img/8edf3510a9fa88809aaa000c285ed788.png)

[TensorFlow 2.0](https://www.tensorflow.org/)

# 更新 2020 年 11 月 15 日:HUB 上的新模型版本 v3

对于 TensorFlow Hub 上的新模型版本 v3，它们包括一个[预处理器模型](https://tfhub.dev/tensorflow/bert_en_uncased_preprocess/1)来实现本故事中描述的步骤。Hub 版本还更改为使用字典输入和输出变量，因此如果您想以原始故事中描述的方式实现，请在使用较新的模型版本时考虑到这一点。

有了新版本，我们有 3 个步骤要遵循:1)从 TF、TF-Hub 和 TF-text 导入正确的模块和模型；2)将输入加载到预处理器模型中；3)将预处理的输入加载到 BERT 编码器中。

BERT with TensorFlow HUB — 15 lines of code (from the [official HUB model example](https://tfhub.dev/tensorflow/bert_en_uncased_L-12_H-768_A-12/3))

我没有更新 Colab，而是用上面的例子创建了一个新的笔记本。我把原始版本留在这里，因为我相信它有助于理解预处理器模型的步骤。使用不同版本时，请注意轮毂模型导入结束时的版本(`/3`)。

# 模块导入

我们将使用最新的 TensorFlow (2.0+)和 TensorFlow Hub (0.7+)，因此，它可能需要在系统中进行升级。对于模型创建，我们使用高级 Keras API 模型类(新集成到 tf.keras 中)。

BERT 记号赋予器仍然来自 BERT python 模块(bert-for-tf2)。

Imports of the project

# 模型

我们将基于 TensorFlow Hub 上的示例实现一个模型。在这里，我们可以看到 `bert_layer` 可以像其他 Keras 层一样用于更复杂的模型。

该模型的目标是使用预训练的 BERT 来生成嵌入向量。因此，我们只需要 BERT 层所需的输入，并且模型只将 BERT 层作为隐藏层。当然，在 BERT 层内部，还有一个更复杂的架构。

`hub.KerasLayer`函数将预训练的模型作为 Keras 层导入。

BERT embedding model in Keras

# 预处理

BERT 层需要 3 个输入序列:

*   标记 id:针对句子中的每个标记。我们从伯特词汇词典中恢复它
*   mask ids:for each token，用于屏蔽仅用于序列填充的标记(因此每个序列都具有相同的长度)。
*   Segment ids:对于单句序列为 0，如果序列中有两个句子，并且是第二个，则为 1(更多信息请参见原始论文或 GitHub 上 BERT 的相应部分:`[run_classifier.py](https://github.com/google-research/bert/blob/master/run_classifier.py)`中的`convert_single_example`)。

Functions to generate the input based on the tokens and the max sequence length

# 预言；预测；预告

通过这些步骤，我们可以为我们的句子生成 BERT 上下文化嵌入向量！不要忘记添加`[CLS]`和`[SEP]`分隔符标记以保持原始格式！

Bert Embedding Generator in use

# 作为句子级嵌入的混合嵌入

原始论文建议使用`[CLS]`分隔符作为整个句子的表示，因为每个句子都有一个`[CLS]`标记，并且因为它是一个上下文化的嵌入，所以它可以表示整个句子。在我之前的作品中，也是用这个 token 的嵌入作为句子级的表示。来自 TensorFlow Hub 的`bert_layer`返回一个不同的合并输出，用于表示整个输入序列。

为了比较这两种嵌入，我们使用余弦相似度。汇集嵌入和第一个标记嵌入在例句中的区别*“这是一个好句子。”*为 0.0276。

# BERT 中的偏差

当有人使用预先训练好的模型时，调查它的缺点和优点是很重要的。模型就像数据集一样有偏差，因此，如果使用有偏差的预训练模型，新模型很可能会继承缺陷。如果你使用 BERT，我建议你阅读我关于 BERT 中[偏差的帖子。](/racial-bias-in-bert-c1c77da6b25a)

# 摘要

这个故事介绍了一个简单的、基于 Keras 的 TensorFlow 2.0 对 BERT 嵌入模型的使用。像阿尔伯特这样的其他模型也可以在 TensorFlow Hub 上[找到。](https://tfhub.dev/)

[这个故事的所有代码都可以在 Google Colab 上获得。](https://colab.research.google.com/drive/1hMLd5-r82FrnFnBub-B-fVW78Px4KPX1)

# 参考

[1] Devlin，j .，Chang，M. W .，Lee，k .，& Toutanova，K. (2018 年)。 [Bert:用于语言理解的深度双向转换器的预训练。](https://arxiv.org/abs/1810.04805) *arXiv 预印本 arXiv:1810.04805* 。

# 用伯特的故事学习 NMT

1.  [BLEU-BERT-y:比较句子得分](https://medium.com/@neged.ng/bleu-bert-y-comparing-sentence-scores-307e0975994d)
2.  [嵌入关系的可视化(word2vec，BERT)](https://medium.com/@neged.ng/visualisation-of-embedding-relations-word2vec-bert-64d695b7f36)
3.  [机器翻译:简要概述](/machine-translation-a-short-overview-91343ff39c9f)
4.  [使用 BERT 识别单词的正确含义](/identifying-the-right-meaning-of-the-words-using-bert-817eef2ac1f0)
5.  [机器翻译:对比 SOTA](/machine-translation-compare-to-sota-6f71cb2cd784)
6.  [使用 TensorFlow 2.0 的简单 BERT](/simple-bert-using-tensorflow-2-0-132cb19e9b22)