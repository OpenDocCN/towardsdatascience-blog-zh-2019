# 熊猫和模特的下一步

> 原文：<https://towardsdatascience.com/next-steps-with-pandas-and-modeling-5e6e36672e39?source=collection_archive---------12----------------------->

![](img/52c388c30dffd1e76957f09c41f69b6f.png)

Not for wildlife enthusiasts, but data scientists. Copyright [Chi King](https://commons.wikimedia.org/wiki/File:Pandas!!_(GIANT_PANDA-WOLONG-SICHUAN-CHINA)_(2150601789).jpg) under the [Creative Commons Attribution 2 Generic License](https://creativecommons.org/licenses/by/2.0/deed.en). No changes were made to this image.

到目前为止，您已经学习了机器学习的基础知识，以及一点 Python 3 和 Pandas。下面是一些后续步骤和免费资源，让你开始行动。我会继续在这里添加我想到的信息，或者评论中的建议。

*   阅读[熊猫文档](https://pandas.pydata.org/pandas-docs/stable/)

此时，您不应该像阅读一本书一样阅读文档(尽管如果这对您有用，您可以这样做)。自上而下浏览文档，熟悉各种可用的主题。

在查看堆栈溢出之前，特别是如果您想提出问题，请查看文档。我更喜欢查找我当前的疑问或问题，并通读相关部分。

例如，我经常不得不编写自己的日期时间解析器，并将其传递给 *read_csv* 。我还传递了自己的数据类型列表，以获得正确的数据框。文档中有这两种情况的示例。

文档还包括[食谱](http://pandas-docs.github.io/pandas-docs-travis/user_guide/cookbook.html)，非常值得浏览。

*   将 Pandas 用于大型数据集

在创建了我的 Pandas 数据框之后，我使用 Pandas 创建了大约 30 GB 的数据集。这个过程包括导入、清理、合并 Pandas 数据帧和旋转(如果需要的话)。在 Pandas 中使用大型数据集有多种策略。查看文档和[这个关于栈溢出](https://stackoverflow.com/questions/14262433/large-data-work-flows-using-pandas)的问题。小心栈溢出的老问题，因为它们可能涉及过时的特性或熊猫编码风格。

用于 Tensorflow 的 Dask[和](https://dask.org/) [Dask-ml](https://ml.dask.org/tensorflow.html) 可能是你的下一步，还有[分布式 Tensorflow](https://github.com/tensorflow/examples/blob/master/community/en/docs/deploy/distributed.md) 。在走这条路之前，确保你能彻底证明资源的花费是合理的。制定一份书面计划，并在做出最终决定前征求反馈。

*   用 [tsfresh](https://tsfresh.readthedocs.io/en/latest/) 提取日期/时间特征

不要试图自己构建依赖于时间的特征(星期几等)。

*   通过指定 dtypes 减少内存使用

不要只接受熊猫的推断。检查数据类型，并仔细查看没有按预期显示的列。如果已知一个列只包含浮点数或整数，并且它显示为**对象**，那么请仔细检查它。

如果您知道某个特定的列是一个范围有限的整数，比如 1–3，请指定数据类型。例如

并且要小心具有混合数据类型的列。 [Pandas 使用数据块](https://pandas.pydata.org/docs/user_guide/io.html#specifying-column-data-types)而不是整列来推断数据类型。所以仔细检查导入数据的数据类型是必要的。

*   汤姆·奥格斯佩格的《现代熊猫》

我从阅读汤姆的书中受益匪浅，这本书是免费的。不过，我劝你还是付他点什么吧，要看你自己的情况。

*   克里斯·阿尔邦和[技术笔记](https://chrisalbon.com/)

另一个优秀的免费资源。当你遇到问题或疑问时，请在这里查阅。

*   学习贝叶斯方法

正如 Cal Davidson Pilon 在他的书 [*概率编程和针对黑客的贝叶斯方法*](http://camdavidsonpilon.github.io/Probabilistic-Programming-and-Bayesian-Methods-for-Hackers/) 中指出的，贝叶斯方法在统计学书籍中经常受到冷遇。贝叶斯方法优雅、有用，是思考某些问题的一种非常自然的方式。我们现在都有计算资源，如数字海洋、自动气象站等。所以贝叶斯方法现在已经可以实际学习和使用了。

Pilon 首先通过编码，然后是理论来介绍贝叶斯方法，这不仅是开发 Python 技能的极好方法，也是获得工作模型的极好方法。

艾伦·唐尼的《思维贝叶斯》和《思维统计》都是免费的，也值得一读。

*   最好的计算和学习资源

*最佳*永远是一个相对名词。这就是为什么 Stack 的人会问很多封闭的问题，询问使用哪种技术或者哪种技术是 T2 最好的。 [Tim Dettmers](https://timdettmers.com/2019/04/03/which-gpu-for-deep-learning/) 发表了一篇评估各种 NVIDIA 卡的性价比的优秀博文。挑选一台符合你预算的二手电脑，但在购买之前，确保你了解你需要的[计算能力](https://developer.nvidia.com/cuda-gpus)。

最新的驱动程序更新可能不支持旧的 AMD 卡。

如果你正在寻找便宜的培训或教程资源，不要忽视甲骨文、英伟达、英特尔或 AMD 的开发者网站。例如，英伟达提供相关技术的在线课程，如 OpenACC。

*   参与挑战

例如， [Kaggle 数据清理挑战](https://www.kaggle.com/rtatman/data-cleaning-challenge-json-txt-and-xls?utm_medium=email&utm_source=mailchimp&utm_campaign=5DDC-data-cleaning-R)可以锻炼你的技能。创建几个笔记本在 GitHub 上展示总是一件好事，可以传递给潜在的雇主，或者作为在当地 PyData meetup 上准备演讲的提纲。

*   参加您当地的 [PyData](https://pydata.org/) 聚会

请在评论中留下优质资源。我一直在寻找对这份文件的补充。

*熊猫形象版权所有* [*驰王*](https://commons.wikimedia.org/wiki/File:Pandas!!_(GIANT_PANDA-WOLONG-SICHUAN-CHINA)_(2150601789).jpg)*[*知识共享署名 2*](https://creativecommons.org/licenses/by/2.0/deed.en)*通用许可。没有对此图像进行任何更改。**