# 用于语音识别的数据扩充

> 原文：<https://towardsdatascience.com/data-augmentation-for-speech-recognition-e7c607482e78?source=collection_archive---------7----------------------->

## 自动语音识别(ASR)

![](img/2964ffffbd1cb9bbc1a35b0a236f2758.png)

Photo by [Edward Ma](https://unsplash.com/@makcedward?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

这个故事发表在 [Dev.to](https://dev.to/makcedward/data-augmentation-for-speech-recognition-bfc) 和 Medium 上。

语音识别的目的是将音频转换成文本。这项技术广泛应用于我们的生活中。[谷歌助手](https://en.wikipedia.org/wiki/Google_Assistant)和[亚马逊 Alexa](https://en.wikipedia.org/wiki/Amazon_Alexa) 就是将我们的声音作为输入并转换成文本以理解我们意图的一些例子。

与其他自然语言处理问题一样，关键挑战之一是缺乏足够数量的训练数据。它导致过多或难以处理看不见的数据。谷歌大脑(Google Brain)和人工智能(AI)团队通过引入几种用于语音识别的数据增强方法来解决这个问题。本故事将讨论 [SpecAugment:一种用于自动语音识别的简单数据增强方法](https://arxiv.org/pdf/1904.08779.pdf) (Park 等人，2019 年)，并将涵盖以下内容:

*   数据
*   体系结构
*   实验

# 数据

为了处理数据，波形音频转换成声谱图，并馈入神经网络产生输出。执行数据扩充的传统方式通常应用于波形。Park 等人采用了另一种方法，即操纵声谱图。

![](img/bc55c141a897d36032e93a4498065c18.png)

Waveform audio to spectrogram ([Google Brain](https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html))

给定一个声谱图，你可以把它看作一幅图像，其中 x 轴是时间，而 y 轴是频率。

![](img/b06b142e122fd434e3ee0723fc7485a1.png)

Spectrogram representation ([librosa](https://librosa.github.io/librosa/generated/librosa.feature.melspectrogram.html))

直观地说，它提高了训练速度，因为没有波形数据到频谱图数据之间的数据转换，而是增加了频谱图数据。

Park 等人推出了用于语音识别中数据增强的`SpecAugment`。有三种基本方法来扩充数据，即时间弯曲、频率屏蔽和时间屏蔽。在他们的实验中，他们将这些方法结合在一起，并引入了 4 种不同的组合，即 LibriSpeech basic (LB)、LibriSpeech double (LD)、Switchboard mild (SM)和 Switchboard strong (SS)。

## ***时间扭曲***

将选择一个随机点，并向左或向右弯曲一段距离 W，该距离从 0 到沿该线的时间弯曲参数 W 的均匀分布中选择。

## 频率掩蔽

a 频道[f0，F0+f]被屏蔽。F 选自 0 至频率掩模参数 F 的均匀分布，f0 选自(0，νF ),其中ν为频率通道数。

## 时间掩蔽

t 个连续的时间步长[t0，t0+t]被屏蔽。T 选自 0 至时间屏蔽参数 T 的均匀分布，t0 选自[0，τT]。

![](img/5914e197178bc2c87e8a47183c5d6eff.png)

From top to bottom, the figures depict the log mel spectrogram of the base input with no augmentation, time warp, frequency masking and time masking applied. (Park et al., 2019)

## 基本增强政策的组合

通过结合频率屏蔽和时间屏蔽的增强策略，引入了 4 种新的增强策略。而符号表示:

*   w:时间弯曲参数
*   f:频率屏蔽参数
*   mF:应用的频率屏蔽数量
*   t:时间屏蔽参数
*   mT:应用时间屏蔽的次数

![](img/e4c06248ab8203bda7d80b5f23c8fd05.png)

Configuration for LB, LD, SM and SS (Park et al., 2019)

![](img/2ae5e64b8b4dc2b2183787d7446e17de.png)

From top to bottom, the figures depict the log mel spectrogram of the base input with policies None, LB and LD applied. (Park et al., 2019)

# 体系结构

## 听、听、拼(LAS)网络架构

Park 等人使用 LAS 网络架构来验证使用和不使用数据增强的性能。它包括两层卷积神经网络(CNN)，注意力和堆叠双向 LSTMs。由于本白皮书的目标是数据扩充，并且利用模型来查看模型的影响，因此您可以从[这里](https://arxiv.org/pdf/1508.01211.pdf)深入研究 LAS。

## 学习费率表

学习率时间表成为决定模型性能的关键因素。类似于[倾斜三角形学习率(STLR)](/multi-task-learning-in-language-model-for-text-classification-c3acc1fedd89) ，应用非静态学习率。学习速率将呈指数衰减，直到它达到其最大值的 1/100，并在该点之后保持恒定。一些参数表示为:

*   sr:加速步骤(从零学习率开始)完成
*   si:指数衰减的步骤开始
*   sf:指数衰减的步骤停止。

另一种学习速率调度是统一标签平滑。正确的类别标签被赋予置信度 0.9，而其他标签的置信度相应地增加。参数表示为:

*   噪声:变权噪声

在后面的实验中，定义了三个标准的学习速率表:

1.  B(asic): (sr，snoise，si，sf ) = (0.5k，10k，20k，80k)
2.  d(double):(Sr，snoise，si，sf ) = (1k，20k，40k，160k)
3.  L(ong): (sr，snoise，si，sf ) = (1k，20k，140k，320k)

## 语言模型

LM 用于进一步提高模型性能。一般来说，LM 被设计成在给定前一个记号的结果的情况下预测下一个记号。一旦预测到新标记，在预测下一个标记时，它将被视为“前一个标记”。这种方法在很多现代的 NLP 模型中都有应用，比如[伯特](/how-bert-leverage-attention-mechanism-and-transformer-to-learn-word-contextual-relations-5bbee1b6dbdb)和 [GPT-2](/too-powerful-nlp-model-generative-pre-training-2-4cc6afb6655) 。

# 实验

模型性能通过[单词错误率](https://en.wikipedia.org/wiki/Word_error_rate) (WER)来衡量。

从下图中，“Sch”表示学习率计划，而“Pol”表示扩充政策。我们可以看到，具有 6 个 LSTM 层和 1280°嵌入向量的 LAS 表现出最好的结果。

![](img/1c794856907a9ba53b0f68be6d950719.png)

Evaluation of LibriSpeech (Park et al., 2019)

通过将 LAS-6–1280 与 SpecAugment 一起使用，与其他模型和没有数据扩充的 LAS 相比，可获得最佳结果。

![](img/2fc33ede01839fcc30472ac88dde6051.png)

Comparing SpecAugment method in LibriSpeech 960h (Park et al., 2019)

在配电盘 300h 中，LAS-4–1024 被用作基准。我们可以看到 SpecAugment 确实有助于进一步提高模型性能。

![](img/500e1300f54bad2de71749ab3cd643b4.png)

Comparing SpecAugment method in Switchboard 300h (Park et al., 2019)

# 拿走

*   时间扭曲并没有显著提高模型性能。如果资源有限，这种方法将被放弃。
*   标签平滑导致训练不稳定。
*   数据扩充将过拟合问题转化为欠拟合问题。从下图中，您可以注意到，没有增强(无)的模型在训练集中表现接近完美，而在其他数据集中没有类似的结果。

![](img/b873eb4656dc8b934977f8fd73404450.png)

*   为了便于语音识别的数据扩充， [nlpaug](https://github.com/makcedward/nlpaug) 现在支持 SpecAugment 方法。

# 关于我

我是湾区的数据科学家。专注于数据科学、人工智能，尤其是 NLP 和平台相关领域的最新发展。欢迎在 [LinkedIn](https://www.linkedin.com/in/edwardma1026) 上与 [me](https://makcedward.github.io/) 联系，或者在 [Medium](http://medium.com/@makcedward/) 或 [Github](https://github.com/makcedward) 上关注我。

# 延伸阅读

*   [自然语言处理中的数据扩充](/data-augmentation-in-nlp-2801a34dfc28)
*   [文本的数据扩充](/data-augmentation-library-for-text-9661736b13ff)
*   [音频数据增强](/data-augmentation-for-audio-76912b01fdf6)
*   [谷歌正式发布 SpecAugment】](https://ai.googleblog.com/2019/04/specaugment-new-data-augmentation.html)
*   [倾斜三角形学习率(STLR)](/multi-task-learning-in-language-model-for-text-classification-c3acc1fedd89)
*   [变压器的双向编码器表示](/how-bert-leverage-attention-mechanism-and-transformer-to-learn-word-contextual-relations-5bbee1b6dbdb)
*   [创成式预培训 2](/too-powerful-nlp-model-generative-pre-training-2-4cc6afb6655)

# 参考

*   D.朴正熙、陈伟雄、张宇人、赵超群、左宗棠、朱布克及黎庆伟。 [SpecAugment:一种用于自动语音识别的简单数据扩充方法](https://arxiv.org/pdf/1904.08779.pdf)。2019
*   W.Chan，N. Jaitly，Q. V. Le 和 O. Vinyals .[听，听，拼](https://arxiv.org/pdf/1508.01211.pdf)。2015