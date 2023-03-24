# 什么是自然语言处理(NLP)？狭隘的解释

> 原文：<https://towardsdatascience.com/what-is-natural-language-processing-nlp-a-bitesize-explanation-cb27ea446305?source=collection_archive---------28----------------------->

NLP 一直被认为是一种神奇的灵丹妙药或数字巴别塔，并且有许多相互矛盾的想法。我甚至看到数据科学家争论是否只有特定的机器学习算法才算 NLP，或者是否真的是方法使它成为 NLP。因此，我在这里简单解释一下它的含义。

![](img/94ed58cfb875b42a192f3a94bdff3ae4.png)

Photo by [Mark Rasmuson](https://unsplash.com/@mrasmuson?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 那么什么是 NLP 呢？

自然语言处理(NLP)有着非常有趣的历史，它本身也是一门非常古老的学科，在更复杂的方法实现之前，一些最早的例子是[手写的规则](https://en.wikipedia.org/wiki/Natural_language_processing)。我经常被问到的一个问题是“什么是 NLP？”我认为答案取决于你想让它做什么。NLP 通常被分解成不同的需求，我在这里找到了一个很好的列表[七种不同的用途:](https://machinelearningmastery.com/applications-of-deep-learning-for-natural-language-processing/)

*   文本分类—例如垃圾邮件过滤
*   语言建模—例如拼写检查
*   语音识别——将我们的口头语音分解成文本格式，然后可以被其他算法接受
*   标题生成—描述图片的内容
*   机器翻译——将一种语言转换成另一种语言
*   文档汇总—文档的抽象创建
*   问答——理解一个问题，然后给出一个人类可读的答案

令人惊讶的是，你可以为这些的不同部分使用不同的模型。例如，传统的分类问题使用支持向量机或决策树来处理文本的[单词包](https://en.wikipedia.org/wiki/Bag-of-words_model)(单词包最简单的形式是单词在一段文本中出现的频率计数，因此失去了排序和语法)，但被更复杂的算法(例如[深度学习](https://medium.com/dair-ai/deep-learning-for-nlp-an-overview-of-recent-trends-d0d8f40a776d))慢慢取代， 但实际上这取决于手头的任务，对于简单的问题，一个单词包方法和一个 SVM 可以在更短的时间内比一个复杂的[单词嵌入](https://machinelearningmastery.com/what-are-word-embeddings/)(每个单词的向量，包含上下文信息和与其他单词的关联)构建的数据集做得更好，这个数据集被输入到一个专门的神经网络中。

NLP 不是一项简单的任务，只是试图分解和理解人类语言的复杂性，而是关于理解我们如何说话，它可以完成任务，有时会给人留下它像人类一样理解你所说的话的印象。我这么说是因为机器永远不会像我们一样理解我们的语言，一个很好的例子是，当脸书试图让聊天机器人相互交流时，他们迅速将英语语言改变到人们不再理解的水平(但他们似乎理解)。

NLP 算法可能会遇到的另一个问题是多义性(同一个词有多种含义，英语中的 [40%](https://www.thoughtco.com/polysemy-words-and-meanings-1691642) 估计就是这样)、同音异义词(听起来相同但拼写不同的词)、同形异义词(拼写相同但不同的词)等问题，这些问题导致模型为其特定的目标领域进行训练(尽管正在努力处理[这个](https://techcrunch.com/2018/06/15/machines-learn-language-better-by-using-a-deep-understanding-of-words/))。

这样的单词或句子的例子有(许多摘自[这里是](http://www-users.york.ac.uk/~ez506/downloads/L140%20Handout%20-%20homonyms.pdf)):

*   对于语义分析，在工程师报告中，单词“点火”在木材(坏)和锅炉行业(好)中有不同的含义
*   对于多义词来说，动词“get”可以表示根据上下文变成或理解
*   [谐音](http://www-users.york.ac.uk/~ez506/downloads/L140%20Handout%20-%20homonyms.pdf):“爬下梯子”或者“买了羽绒被”。拼写和发音相同但不同的东西
*   [同形词](https://examples.yourdictionary.com/examples-of-homographs.html):“轴”。如果是砍木头的斧头的复数或者是一个图形的轴的复数

所以总的来说，NLP 是很难的，而且还涵盖了大量围绕使用人类语言的各种各样的问题。所以下次 Alexa 或 Siri 出错时，只要想想它试图做什么就行了。

一个轻松有趣的事实是，任何想要通过[图灵测试](https://en.wikipedia.org/wiki/Turing_test)的人工智能都需要 NLP(因为它需要产生与人类书面文本无法区分的对话文本)。