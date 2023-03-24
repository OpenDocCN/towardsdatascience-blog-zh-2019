# 在互联网上，没有人知道你是一只狗，或人工智能

> 原文：<https://towardsdatascience.com/the-next-step-for-deep-fake-news-deep-fake-comments-1c423450663a?source=collection_archive---------32----------------------->

## 不可避免地滑向网络社区无政府状态

![](img/4da0e5f31cbb9291b3ff26ef97994c91.png)

Photo by [Andy Kelly](https://unsplash.com/@askkell?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

我今天看到了下面这条推文，我想谈谈它。这是一篇关于根据标题和文章生成深度虚假评论的新论文。已经有很多关于这一点的伦理含义的讨论。由于这是一篇由中国研究人员开发的论文，作为一名中国人，我想我可能会对它有更多的了解，因为我更了解中国的在线社区以及这种新技术的可能用途。

# 像审查，但更具侵略性

从坏的方面来说，产生深度虚假评论有点与审查相反。它不是屏蔽/删除你**不想要的信息**，而是生成/添加你**想要的信息**。审查有时不容易发现，比如你的内容在 YouTube 上不被推荐，或者你的推文由于某种原因在 Twitter 上很难搜索到。这些都发生在后台，通常由算法完成。你不知道到底发生了什么。很多时候，审查有似是而非的可否认性。另外，如果你不把内容放上去，没人能审查你，所以这是被动的。但是深度假评论就不一样了。它就在你的面前，每个人都可以看到它，他们不必等你发布任何东西来采取行动。它会淹没你的频道或时间线，使真实的信息不那么明显。与审查相比，审查是一种塑造公众意见的被动方式，深度虚假评论是一种非常积极的方式，可以创造一些“人为趋势”，以努力改变人们对某些事情的看法。它可以是一些电影评论，也可以是一些关于社会事件的报道来描述发生了什么。尽管如此，潜在的影响是巨大的。

# 粒度细，但规模大

![](img/5d4b8b6c91f12778b846ab61f21920c2.png)

Photo by [hue12 photography](https://unsplash.com/@hue12_photography?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

eep 虚假评论可能正处于发展的早期阶段，但它有可能在质量上变得更加精细，在数量上变得更加庞大。NLP 最近的发展经常在具有更少数据和更快的训练模型中利用迁移学习。研究人员可以利用当前[预先训练的模型，如在维基百科](https://www.kdnuggets.com/2017/11/building-wikipedia-text-corpus-nlp.html)上训练的模型作为基础，微调更具体的模型，以更少的数据更快地获得最先进的结果。预先训练的模型已经知道英语作为一种语言总体上是如何工作的，只需要学习特定类型的英语是如何说的(例如，Reddit 或 IMDB 电影评论)。利用迁移学习，深度虚假评论具有为许多利基快速开发多种模型的潜力，并实现细粒度的质量，在不同领域生成非常相关和“真实”的评论。

此外，由于它不依赖于人工干预(就像中国互联网上的“[五毛钱派对](https://en.wikipedia.org/wiki/50_Cent_Party)”)，理论上你可以有数千个脚本，运行风格略有不同的模型，并生成试图推动相同议程的评论。想想谷歌的定向广告能做得多好，你就会知道这种“定向评论”有多大的潜力。我甚至会宣称它在某种意义上可能是某种“武器”。

在一个没有人知道你是一只狗的互联网里，人们不再相信文章，因为它可以被精心制作来说服你购买一些产品或推动一些议程，但人们总的来说仍然相信评论，认为它更人性化、更隐私，因此更值得信任。现在随着假货评论的深入，连评论都不可信了。那网上还有什么是合法的？

# 还没有失去所有的希望

Google Duplex is very close to passing the Turing Test

阿文讨论了深度虚假评论的所有可怕迹象。也不是没有弱点。众所周知，如果不能形成高质量的对话，单靠评论是不会有太大影响的。真正打动人的是思想和情感的交流。一个评论，无论多么“真实”或“相关”，都不会产生最佳效果。当真实的人对评论进行回复，并期望得到一个聪明或有力的回复时，算法很可能会失败，至少现在是这样。这就是为什么很多聊天机器人或“Siri”之类的语音助手还没有成为主流。能够应对这一挑战就是说人工智能已经通过了[图灵测试](https://en.wikipedia.org/wiki/Turing_test)，这是一个非常高的门槛，我不相信我们还没有到那一步。[谷歌的 Duplex 目前是最接近的](/did-google-duplex-beat-the-turing-test-yes-and-no-a2b87d1c9f58)，但仍不完全是。

所以就目前而言，我认为该算法可能会带来很多麻烦，但不能真正打动人们，产生非常深刻的影响。

> 还没有。

# 我们能做什么

老实说，我不知道这个问题的答案。我们可以监管这种技术的开发和发布，或者我们可以培养自律，就像 OpenAI 在他们著名的 GPT-2 模型中所做的那样(这是一个相应的举措，应该得到鼓励，尽管远远不能解决更大的问题)。

另一种方法是接受算法将以某种方式发展，并尝试开发一种反人工智能来检测深层假货，就像脸书和谷歌现在正在做的一样。

如果我们能发现深度造假，我们就能审查深度造假，对吗？**对吧？**

你认为处理这件事的最好方法是什么？请留下你的评论(请勿深假！)下面。

注:还可以参考[吴恩达](https://medium.com/u/592ce2a67248?source=post_page-----1c423450663a--------------------------------)的[拿](https://info.deeplearning.ai/the-batch-tesla-acquires-deepscale-france-backs-face-recognition-robots-learn-in-virtual-reality-acquirers-snag-ai-startups)在 deeplearning.ai 的'[批](https://www.deeplearning.ai/thebatch/)，颇有见地。

欢迎任何反馈或建设性的批评。你可以在 Twitter [@lymenlee](https://twitter.com/lymenlee) 或者我的博客网站[wayofnumbers.com](https://wayofnumbers.com/)上找到我。