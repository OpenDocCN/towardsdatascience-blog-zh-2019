# 每个自然语言处理(NLP)从业者必须知道的资源集合

> 原文：<https://towardsdatascience.com/a-collection-of-must-known-pre-requisite-resources-for-every-natural-language-processing-nlp-a18df7e2e027?source=collection_archive---------17----------------------->

## 一个你绝不会错过的终极指南！！

> 嘿，你是一个想进入自然语言处理世界的新手，还是一个对网络上的大量信息感到困惑并且不知道从哪里开始的普通自然语言处理实践者？放松，我也一样，直到我决定花大量的时间在一个地方收集所有需要的资源。

经过去年以来从多种来源的彻底阅读，这里是我编译的最好的学习资源版本，可以帮助任何人开始他们进入 NLP 迷人世界的旅程。有各种各样的任务属于更广泛的自然语言处理领域，如机器翻译、问答、文本摘要、对话系统、语音识别等。然而，要在这些领域中的任何一个领域工作，基本的必备知识是相同的，我将在这篇博客中简要讨论。(**注意:*如果任何链接已经过期，请在评论区告诉我。*** )

简单说一下**免责声明**关于内容:
1。我将要讨论的内容大部分属于现代的自然语言处理，而不是经典的自然语言处理技术。
2。任何人都不可能去翻遍所有可用的资源。我已经尽了最大的努力。
3。我假设读者对至少相当数量的关于机器学习(ML)和深度学习(DL)算法的知识感到满意。
4。对于我将要涉及的所有主题，我主要引用了博客或视频方面的最佳资源。读者可以很容易地找到每个单独主题的研究论文。我觉得上面提到的博客足以让任何人充分理解各自的主题。

> *下面是我通往 NLP 世界的路线图:* **1。单词嵌入— Word2Vec，GloVe，FastText
> 2。语言模型& RNN
> 3。上下文单词嵌入— ELMo
> 4。NLP 中的迁移学习— ULMFiT
> 5。句子嵌入
> 6。Seq2Seq &注意机构
> 7。变形金刚
> 8。OpenAI GPT &伯特
> 9。GPT-2，XLNet
> 10。总结**

我们来简单总结一下以上 10 个话题:

# **1。单词嵌入— Word2Vec、GloVe、FastText**

当我们开始学习 NLP 时，首先想到的是我们如何将单词表示成数字，以便任何 ML 或 DL 算法都可以应用于它。这就是矢量/嵌入这个词发挥作用的地方。顾名思义，这里的目的是将任何给定的单词作为输入，并输出一个表征该单词的有意义的向量表示。基于诸如 Word2Vec、GloVe、FastText 之类的底层技术，存在不同的方法来获得这种表示

## **Word2Vec:**

从这个话题开始，我建议读者在 YouTube 上免费观看斯坦福 CS224N: NLP 与深度学习| Winter 2019 的讲座 1 和 2。

> [***&list = ploromvodv 4 rohcuxmzknm 7j 3 fvwbby 42 z&index = 1***](https://www.youtube.com/watch?v=8rXD5-xhemo&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z&index=1)

这两个讲座形成了一个关于语义词表示的坚实背景。除此之外，您还将了解 Word2Vec 和 GloVe model 工作中涉及的详细数学知识。一旦你对此感到满意，我将向你推荐一些我认为对这个话题最有用的博客。在这些博客中，你可以找到一些帮助你更好理解的例子和图像。

> [***http://mccormickml . com/2016/04/19/word 2 vec-tutorial-the-skip-gram-model/***](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)
> [***http://mccormickml . com/2017/01/11/word 2 vec-tutorial-part-2-negative-sampling/***](http://mccormickml.com/2017/01/11/word2vec-tutorial-part-2-negative-sampling/)
> [***http://jalammar.github.io/illustrated-word2vec/***](http://jalammar.github.io/illustrated-word2vec/)

我希望这些阅读足以让你对 Word2Vec 有一个坚实的理解。让我们继续前进。

## **手套:**

在斯坦福深度学习自然语言处理(2017 年冬季)的第 3 讲中，GloVe 得到了更好的解释

> [***https://www.youtube.com/watch?v=ASn7ExxLZws***](https://www.youtube.com/watch?v=ASn7ExxLZws)

除此之外，下面的博客可以帮助你清楚地了解这个话题及其背后的数学原理。

> [**【slide share id = 229369562&doc = paperpassedgloveglobalvectorsforwordrepresentationexplained-200228052056&type = d】**](https://www.slideshare.net/NikhilJaiswal3/paper-dissected-glove-global-vectors-for-word-representation-explained-machine-learning-explained)
> 
> [**【slide share id = 229369559&doc = emnlpwhatisloveparti-towardsdatascience-200228052054&type = d】**](https://www.slideshare.net/NikhilJaiswal3/emnlp-what-is-glo-ve-part-i-towards-data-science)
> 
> [**【slide share id = 229369555&doc = emnlpwhatislovepartii-towardsdatascience-200228052050&type = d】**](https://www.slideshare.net/NikhilJaiswal3/emnlp-what-is-glo-ve-part-ii-towards-data-science)
> 
> [**【slide share id = 229369551&doc = emnlpwhatislovepartiii-towardsdatascience-200228052047&type = d】**](https://www.slideshare.net/NikhilJaiswal3/emnlp-what-is-glo-ve-part-iii-towards-data-science)

我希望你到目前为止已经理解了 GloVe 是如何利用全局统计信息而不是 Word2Vec 来优化一个完全不同的目标的。

## **快速文本:**

FastText 是由脸书研究团队创建的一个库，用于高效学习单词表示和句子分类。它支持类似于 Word2Vec 的训练 CBOW 或 Skip Gram 模型，但它对单词的 n-gram 表示进行操作。通过这样做，它有助于通过利用字符级信息找到罕见单词的向量表示。

请参考以下链接，以便更好地理解:

> [***https://towardsdatascience . com/fast text-under-the-hood-11 EFC 57 B2 B3***](/fasttext-under-the-hood-11efc57b2b3)[***https://arxiv.org/pdf/1607.04606v1.pdf***](https://arxiv.org/pdf/1607.04606v1.pdf)[***https://arxiv.org/pdf/1607.01759.pdf***](https://arxiv.org/pdf/1607.01759.pdf)

如果您已经完成了上面提到的要点，那么您现在至少对单词嵌入方法有了更深的理解。是时候进入 NLP 的主干了——语言模型。

# 2.语言模型(LM)和 RNN

语言模型是我们日常使用的东西。一个这样的场景发生在用手机、Gmail 或 LinkedIn 发短信的时候。LM 为您提供了您希望进一步键入的最有可能的建议。简单地说，LM 就是预测下一个单词是什么的任务。我关于语言模型是自然语言处理的支柱的论点是因为所有当前的迁移学习模型都依赖于语言模型作为基础任务。在你即将到来的旅程中，你会进一步了解这些。但在此之前，我们先来看看了解 LM 的资源。

像往常一样，我的第一个建议是浏览斯坦福大学关于这个特定主题的精彩讲座。CS224N 的第 6 讲很好地涵盖了这个话题。它让你一瞥 LM 在神经网络之前是如何发展的，以及神经网络基本上 RNN 给它带来了什么优势。另外，如果你想重温一下关于 RNNs 的知识，请参考第 7 课。

> [***https://www.youtube.com/watch?v=iWea12EAu6U&list = ploromvodv 4 rohcuxmzknm 7j 3 fvwbby 42 z&index = 6***](https://www.youtube.com/watch?v=iWea12EAu6U&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z&index=6)
> [***https://www.youtube.com/watch?v=QEw0qEa0E50&list = ploromvodv 4 rohcuxmzknm 7j 3 fvwbby 42 z&index = 7***](https://www.youtube.com/watch?v=QEw0qEa0E50&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z&index=7)

另外，如果你觉得对 RNNs 的内部运作不太了解，你可以去参加我在 Udemy 上学的一门很棒的课程

> **深度学习:高级 NLP 和 RNNs**
> [***https://www.udemy.com/course/deep-learning-advanced-nlp/***](https://www.udemy.com/course/deep-learning-advanced-nlp/)

这是我在网上大量的在线课程中发现的最有用的课程之一。在本课程中，您可以通过展开 rnn、将其转换为双向等方式来理解 rnn 的工作原理。此外，你可以学习在 Keras 中对这些模型进行编码——这是最简单的深度学习框架之一。

# **3。上下文单词嵌入— ELMo**

猜猜什么单词嵌入又回来了！！等等，有一个术语叫“语境”，它不同于我们之前研究过的方法。好吧，那我们为什么不把这个话题和第一个话题放在一起研究。嗯，只是因为我们需要 LM 的知识来理解这个题目。是的，正如我之前提到的，在我们的旅程中，我们遇到了 LM 的第一个应用。相信我到最后，你会同意我把 LM 授予 NLP 的中坚力量。说够了，让我们进入我们当前的主题——上下文单词嵌入

来自语言模型的嵌入(ELMo)使用 LM 来获得单个单词的嵌入。到目前为止，我们对任何输入单词都只有一个嵌入，比如说银行。现在假设我有两个不同的句子——我去银行取钱，站在河边。在这两个句子中，单词 bank 的意思完全不同，因此它们肯定有不同的向量表示。这就是语境嵌入的目的。ELMo 是一种基于多层双向 LSTM 模型的方法，用于获得上下文单词嵌入。请浏览以下博客来了解它们。

> [***https://mlexplained . com/2018/06/15/paper-parsed-deep-contextized word-representations-explained/***](https://mlexplained.com/2018/06/15/paper-dissected-deep-contextualizedword-representations-explained/)[***https://www . slide share . net/shuntaroy/a-review-of-deep-contexternalized-word-representations-Peters-2018***](https://www.slideshare.net/shuntaroy/a-review-of-deep-contextualized-word-representations-peters-2018)

希望以上两个资源足以帮助你更好的了解 ELMo。是时候向前迈进了…

# 4.自然语言处理中的迁移学习— ULMFiT

在过去的一年里，迁移学习彻底改变了自然语言处理领域。大多数当前正在开发的算法都利用了这种技术。在对计算机视觉领域做出重大贡献之后，迁移学习终于让 NLP 从业者欢欣鼓舞。
用于文本分类的通用语言模型微调(ULMFiT)就是这样一种方法，应该归功于这种奇妙的变化。
ULMFiT 引入了多种方法来有效利用模型在预训练期间学习到的大量内容——不仅仅是嵌入，也不仅仅是
情境化嵌入。ULMFiT 引入了一个语言模型和一个过程，以便针对各种任务有效地微调该语言模型。
最后，预训练和微调概念开始在 NLP 领域显示其魔力。ULMFiT 论文还介绍了不同的技术，如区别微调和倾斜三角形学习率，这些技术有助于改进迁移学习方法的使用方式。

准备探索这些令人兴奋的术语，然后保持冷静，参考以下博客:

> [***http://NLP . fast . ai/class ification/2018/05/15/introducing-ulm fit . html***](http://nlp.fast.ai/classification/2018/05/15/introducing-ulmfit.html)[***https://ahmedhanibrahim . WordPress . com/2019/07/01/a-study-on-cove-context 2 vec-elmo-ulm fit-and-Bert/***](https://ahmedhanibrahim.wordpress.com/2019/07/01/a-study-on-cove-context2vec-elmo-ulmfit-and-bert/)

现在，你一定很熟悉乌尔菲特了。我们旅程的下一步是句子嵌入。

# 5.句子嵌入

学习了足够多的单词嵌入。句子呢？我们能获得类似于一个词的句子的某种表示吗？一种非常幼稚但强大的基线方法是平均句子的单词向量(所谓的单词袋方法)。除此之外，可以有基于无监督、有监督和多任务学习设置的不同方法。

无监督的方案学习句子嵌入作为学习的副产品，以预测连贯的连续句子。这里的主要优势是，你可以获得大量无人监管的数据，因为互联网上充满了文本类型的东西。跳过思维向量和快速思维向量是在无监督环境中开发的两种成功的方法。
另一方面，监督学习需要为给定任务标注的标签数据集。完成这项任务让你学会一个好的句子嵌入。脸书研究小组的 InferSent 就是这样一个有趣的方法。现在，为了解决无监督嵌入和有监督嵌入之间的冲突，多任务学习机制应运而生。多任务学习的几个建议已经发表，如 MILA/MSR 的通用句子表示，谷歌的通用句子编码器等。

兴奋的来到这个世界。探索&探索提到的链接:

> [***https://medium . com/hugging face/universal-word-sentence-embeddings-ce 48 DDC 8 fc 3a***](https://medium.com/huggingface/universal-word-sentence-embeddings-ce48ddc8fc3a)[***https://ai . Google blog . com/2018/05/advances-in-semantic-textual-similarity . html***](https://ai.googleblog.com/2018/05/advances-in-semantic-textual-similarity.html)[](/deep-transfer-learning-for-natural-language-processing-text-classification-with-universal-1a2c69e5baa9)

# **6.Seq2Seq &注意机制**

**已经学习了 RNN 模型的变体，并且对单词和句子嵌入有了很好的理解，现在是时候前进到一个令人兴奋的 NLP 架构，称为序列 2 序列模型(Seq2Seq)。这种体系结构用于各种 NLP 任务，如神经机器翻译、文本摘要、会话系统、图像字幕等。序列到序列模型是一种模型，它采用一个项目序列(单词、字母、图像特征等)并输出另一个项目序列。理解这些模型的最好方法是借助可视化，这是我想向你推荐的我最喜欢的 NLP 作者的博客之一。他不是别人，正是杰伊·阿拉姆马。相信我，你会喜欢浏览他的每一个博客。他用来解释这些术语的努力是杰出的。点击下面的链接进入这个美丽的世界。**

> **[***http://jalammar . github . io/visualizing-neural-machine-translation-mechanics-of-seq 2 seq-models-with-attention/***](http://jalammar.github.io/visualizing-neural-machine-translation-mechanics-of-seq2seq-models-with-attention/)**

**我想我确实需要向你进一步解释 Seq2Seq，因为到现在为止，你一定很熟悉它。然而，现在我想再次向您推荐斯坦福讲座，以了解有关统计和神经机器翻译的更多信息。了解 Seq2Seq 将有助于你流畅地进行这些讲座。此外，注意力是最重要的话题之一，在那里详细讨论。此外，您还将了解用于评估 NMT 模型的波束搜索解码& BLEU 度量。**

> **敬请参考 CS224N 讲座—8
> [**https://www.youtube.com/watch?v=XXtpJxZBa2c&list = ploromvodv 4 rohcuxmzknm 7j 3 fvwbby 42 z&index = 8**](https://www.youtube.com/watch?v=XXtpJxZBa2c&list=PLoROMvodv4rOhcuXMZkNm7j3fVwBBY42z&index=8)**

# **7.变形金刚(电影名)**

**该是野兽——变形金刚的时候了。虽然 LSTM 模型是革命性的 NLP 行业，它是变压器，开发出了开箱即用，作为 RNN 模型的改进替代品。
Transformer 是一个使用注意力来提高这些模型训练速度的模型。
转换器有助于并行化。论文中提出的变形金刚就是你所需要的全部注意力。由于并行化的性质，它将我们从 RNN 模型中涉及的重复连接中解放出来。它不仅有助于减少训练时间，而且在各种 NLP 任务上有很大的提高。它类似于 Seq2Seq 架构，但它只依赖于注意力机制及其变体。再说一次，理解这个话题的最佳博客是 Jay Alammar 的博客。事实上，如前所述，您可以关注他的所有博客来了解这些先进的 NLP 技术。**

> **[***http://jalammar.github.io/illustrated-transformer/***](http://jalammar.github.io/illustrated-transformer/)**

**除此之外，如果你想从实现的角度理解这篇论文，那么请参考哈佛大学 NLP 小组的这篇精彩的博客。**

> **[***https://nlp.seas.harvard.edu/2018/04/03/attention.html***](https://nlp.seas.harvard.edu/2018/04/03/attention.html)**

**如果你已经成功理解了以上两篇博客，那么给自己一个大拇指吧！！相信我，这不是一件容易的事。
现在让我们探索一下研究人员是如何利用这种更新的架构来构建像伯特、GPT-2 等这样的艺术模型的。**

# **8.开放 GPT &伯特公司**

**迁移学习又回来了，但现在当然是用变形金刚。很简单，如下:利用变压器解码器的堆栈来建立一个新的模型称为 GPT 或利用编码器部分的变压器来建立一个惊人的模型称为伯特。相信我，即使你是 NLP 领域的新手，并且在过去的一年里一直在听 NLP 流行语，伯特和 GPT 也是这个列表中的佼佼者。**

**生成性预训练(GPT)的目标与 ULMFit 相似，即在 NLP 中应用迁移学习。但是有一个主要的区别。是的，你说对了——用变形金刚代替 LSTM。除此之外，在培训目标上也有一些不同，你可以通过下面提到的博客来了解。总而言之，GPT 的总体思想是训练用于语言建模任务的变换器解码器，也称为预训练。一旦它被预先训练，我们就可以开始使用它来完成下游任务。可以有许多输入转换来处理各种这样的任务。**

**NLP 最火的词来了——BERT。**

**主要目标是建立一个基于 transformer 的模型，它的语言模型既取决于左上下文，也取决于右上下文。这是 GPT 的局限性，因为 GPT 只训练了一个正向语言模型。现在，为了实现双向调节的目标，BERT 使用了变压器的编码器部分。为了在计算注意力分数时看不到未来的单词，它使用了一种叫做掩蔽的特殊技术。根据这篇论文的作者，这种掩蔽技术是这篇论文的最大贡献。除了处理多个句子之间关系的掩蔽目标之外，预训练过程还包括一个额外的任务:给定两个句子(A 和 B)，B 是否可能是 A 后面的句子？**

**好吧，如果你对上面的一些术语感到有负担，并且想要对它们有更深入的了解，放松就好。所有这些术语在下面的博客中都有很好的解释:**

> **[](http://jalammar.github.io/illustrated-bert/)*[***http://jalammar . github . io/a-visual-guide-to-using-BERT-for-the-first-time/***](http://jalammar.github.io/a-visual-guide-to-using-bert-for-the-first-time/)[***https://mlexplained . com/2019/01/07/paper***](https://mlexplained.com/2019/01/07/paper-dissected-bert-pre-training-of-deep-bidirectional-transformers-for-language-understanding-explained/)***

# ***9.GPT-2，XLNet***

***GPT-2 只不过是 GPT 的继任者，参数是它的 10 倍以上，训练数据量也是它的 10 倍以上。由于担心该技术的恶意应用，作者最初没有发布更大的训练模型，这成为一个有争议的话题。
XLNet 是广义自回归模型。它在 20 个任务上超过了 BERT，通常是大幅度超过。这是自然语言处理中迁移学习的新方法。为了更广泛地了解 GPT-2 和 XLNet，请参考以下博客。***

> ***[](http://jalammar.github.io/illustrated-gpt2/)*[***https://openai.com/blog/better-language-models/***](https://openai.com/blog/better-language-models/)[***https://towardsdatascience . com/openai-GPT-2-理解-语言-生成-通过-可视化-8252 f683 B2 F8***](/openai-gpt-2-understanding-language-generation-through-visualization-8252f683b2f8)****

# ********10 摘要********

******最后，您已经按照我们提议的计划完成了学习 NLP 先决条件的整个过程。向你致敬！！！由于大量的研究人员积极从事这一领域的工作，模型不断被涉及。因此，几乎每个月，你都会看到一篇新的论文，它超越了之前的技术水平。因此，在这个快速变化的世界中前进的唯一方法是通过定期浏览研究论文和这些信息丰富的博客来了解最新知识。******

> ******如果你想回忆整个旅程，可以浏览一下下面提到的博客。
> [***https://lilian Weng . github . io/lil-log/2019/01/31/generalized-language-models . html***](https://lilianweng.github.io/lil-log/2019/01/31/generalized-language-models.html)******

******在这里，我还列出了一些我认为在学习 NLP 领域的新主题时最有用的最佳博客:******

> ******[**【https://medium.com/huggingface】**](https://medium.com/huggingface)[***http://jalammar.github.io/***](http://jalammar.github.io/)[***https://ruder.io/******https://mlexplained.com/***](https://ruder.io/)******

******我希望我的资源对读者有所帮助。如前所述，任何人都不可能涵盖所有主题。建议总是受欢迎的，引用一些更好的博客或我错过的重要话题。我希望任何对这些先决条件了如指掌的人都能胜任任何 NLP 任务。到那时，干杯， ***尽情享受！！！*********