# 简单文本生成

> 原文：<https://towardsdatascience.com/simple-text-generation-d1c93f43f340?source=collection_archive---------9----------------------->

## 如何教机器“说话”

![](img/7b21d6ee57a9febacdd79ee358759851.png)

Photo by [Kaitlin Dowis](https://unsplash.com/photos/3YnT86K0CdE) on Unsplash

如果你想看完整的代码，这里有 [Github 链接](https://github.com/jeremyrchow/text-generation-kaggle)。

# 目标

在本文中，我将简要介绍一种使用 Keras 和 Tensorflow 在 Python 中编码和训练文本生成模型的简单方法。我们的目标是训练一个模型来模仿它被训练的文本的说话风格。在这种情况下，我们的数据集是埃德加·爱伦·坡恐怖故事中的 7009 句话。

# 整理训练数据

训练任何 NLP 模型的第一步是单词的标记化。这里我们使用 Keras Tokenizer，它执行以下操作:

1.  删除标点符号
2.  将所有文本设置为小写
3.  将单词拆分成列表中的单个元素，然后为每个单词分配一个唯一的整数
4.  用整数替换该单词的所有实例。

令牌化是为嵌入层准备数据所必需的(参见下面的模型架构部分)

使用滑动窗口设置培训数据。我们正在使用一个相当基本的策略，使用前面的 19 个单词来预测第 20 个单词。这些数字的选择没有任何具体的理由，肯定可以修改以改善。

# 模型架构

**1。嵌入层**

嵌入层是任何寻求理解单词的深度学习模型的关键层。从数学的角度来看，嵌入层所做的是将向量从高维空间(数万或更多，我们的 vocab 的原始大小)带到低维空间(我们希望用来表示数据的向量的数量，在 Word2Vec、Fasttext 等模型中通常为 100–300)。

然而，它以这样一种方式做到了这一点，即具有相似含义的单词具有相似的数学值，并且存在于与其含义相对应的空间中。可以对这些向量进行数学运算，例如，‘王’减去‘人’可能等于‘皇族’。

**2。两个堆叠的 LSTM 层**

我不打算在这里深入 LSTM 的细节，但有大量的好文章来看看他们是如何工作的。一些需要注意的事项:

*   康奈尔大学的这些人表示，当应用于语音识别时，堆叠的 LSTM 可能比单个 LSTM 层中的附加单元增加更多的深度。虽然我们的应用程序不完全相同，但它在使用 LSTM 的程序来尝试识别语言模式方面是相似的，所以我们将尝试这种架构。
*   第一 LSTM 层必须将`return sequences`标志设置为真，以便将序列信息传递给第二 LSTM 层，而不仅仅是其结束状态

**3。具有 ReLU 激活的密集(回归)层**

LSTM 的输出是“隐藏层”。通常在 LSTM 后应用密集图层来进一步捕捉线性关系，但这并不是明确必需的。

**4。激活 Softmax 的密集层**

这一层对于将上面几层的输出转换成我们整个词汇表中的实际单词概率是必要的。它使用了 [softmax 激活函数](https://en.wikipedia.org/wiki/Softmax_function)，这是一个将所有输入单词概率从(-∞，∞)转换为(0，1)的函数。这允许我们选择或生成最可能的单词。

一旦我们建立了我们的模型架构，我们就可以训练它了！

(注意，我们可以使用检查点来保持我们的进度，以防我们的训练被中断。)

# 部署

训练完成后，我们现在有了一个可以生成文本的模型。然而，我们需要给它一个起点。为此，我们编写了一个函数，它接受一个字符串输入，将其标记化，然后用零填充它，以便它适合我们 19 长的预测窗口。

# 结果

下面是给定输入字符串后的结果。注意，我们没有用任何标点符号训练我们的模型，所以句子的划分留给读者去解释。

```
Input String:  First of all I dismembered the corpse.

Model  3 :
first of all i dismembered the corpse which is profound since in the first place he appeared at first so suddenly as any matter no answer was impossible to find my sake he now happened to be sure it was he suspected or caution or gods and some voice held forth upon view the conditions
```

虽然这不是最有意义的，但它实际上似乎是可读的，并且在正确的位置有正确的单词类型。给定一组更大的训练数据，这种方法和模型可以用来生成更容易理解的文本(当然，最有可能的是通过一些参数调整)。

为了量化性能，您可以使用原始数据实现典型的训练测试分割，并根据预测和目标之间的余弦相似性对预测进行评级。

# 谢谢大家！

今天就到这里，感谢阅读。如果你想了解更多，这里有一些我做过的其他机器学习项目，你可以阅读一下:

[量化聊天室毒性](/quantifying-chatroom-toxicity-e755dd2f9ccf)

[面向成长的流媒体推荐系统](/building-a-growth-focused-game-recommender-for-twitch-streamers-7389e3868f2e)