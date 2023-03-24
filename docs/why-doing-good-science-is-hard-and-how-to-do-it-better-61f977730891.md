# 为什么做好科学很难，如何做得更好

> 原文：<https://towardsdatascience.com/why-doing-good-science-is-hard-and-how-to-do-it-better-61f977730891?source=collection_archive---------23----------------------->

![](img/05d83a47e5d77774b583d8e0ff0048b9.png)

Photo by [Steve Johnson](https://unsplash.com/photos/Kr8Tc8Rugdk?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/difficult?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

 [## 想在数据科学方面变得更好吗？

### 当我在我发布独家帖子的媒体和个人网站上发布新内容时，请单击此处获得通知。](https://bobbywlindsey.ck.page/5dca5d4310) 

做好科学很难，很多实验都失败了。虽然科学方法有助于减少不确定性并带来发现，但它的道路充满了坑坑洼洼。在本帖中，您将了解常见的 p 值曲解、p 黑客以及执行多重假设检验的问题。当然，不仅提出了问题，还提出了潜在的解决方案。在这篇文章结束时，你应该对假设检验的一些陷阱，如何避免它们，以及为什么做好科学如此困难有一个很好的理解。

# p 值曲解

有许多方法可以曲解 p 值。根据定义，假设零假设为真，p 值是获得至少与实际观察到的一样极端的检验统计的概率。

什么是 p 值*而不是*:

*   衡量效果的大小或证据的强度
*   干预有效的机会
*   零假设为真或为假的陈述
*   另一种假设为真或为假的陈述

如果你想衡量证据的强度或效果的大小，那么你需要计算效果的大小。这可以通过皮尔逊 r 相关、标准均值差或其他方法来实现。建议在您的研究中报告效应大小，因为 p 值将告诉您实验结果与随机预期不同的可能性，而不是实验治疗的相对大小或实验效应的大小。

p 值也不能告诉你干预有效的几率，但是计算精度可以，而且基础利率会影响这个计算。如果干预的基本比率很低，即使假设检验显示出具有统计显著性的结果，这也为许多假阳性的机会打开了大门。例如，如果干预有效的机会是 65%，那么*仍然*只有 65%的机会干预实际上是有效的，而留下 35%的错误发现率。忽视基本利率的影响被称为[基本利率谬误](http://neglecting/%20the%20impact%20of%20base%20rates%20on%20hypothesis%20tests%20is%20known%20as%20the%20base%20rate%20fallacy)，这种情况比你想象的更常见。

最后，p 值也不能告诉你一个假设是真还是假。统计学是一个推理框架，没有办法确切知道某个假设是否正确。记住， [*科学上没有所谓的证明*](https://bobbywlindsey.com/math/2019/02/13/how-to-really-prove-something/) 。

# 黑客问题

作为一名科学家，在建立假设检验时，你的一个自由度是决定在你检验的数据中包含哪些变量。在一定程度上，您的假设将影响您可能在数据中包含哪些变量，在用这些变量测试假设后，您可能会得到大于 5%的 p 值。

此时，您可能会尝试在数据中使用不同的变量并重新测试。但是如果你尝试了足够多的变量组合，并测试了每种情况，你很可能会得到 5%或更低的 p 值，正如这个应用程序在[这篇 538 博客文章](https://fivethirtyeight.com/features/science-isnt-broken/#part1)中展示的那样。它被称为 *p-hacking* ，它可以让你在竞争性替代假设下实现 5%或更低的 p 值。

这至少有几个问题:

*   由于你可以在竞争的替代假设下获得统计上显著的 p 值，作为你选择包括在测试中的数据的结果，p-hacking 不能帮助你更接近你正在研究的事物的真相。更糟糕的是，如果这样的结果被公布，并且这项研究变成了传统智慧，那么[将很难被移除](https://xkcd.com/386/)。
*   随着假设检验次数的增加，假阳性率(即错误地称无效发现为显著)也会增加。
*   你可能会成为确认偏差的受害者，忽略其他假设测试的结果，只考虑与你的信念一致的测试结果。
*   由于许多期刊要求出版的 p 值为 5%或更低，这就促使你用 p-hack 的方式达到这个 5%的阈值，这不仅造成了伦理上的困境，也降低了研究的质量。

# 应对 P-Hacking

为了帮助减少 p-hacking，您应该公开研究期间探索的假设数量、所有数据收集决策、所有进行的统计分析和所有计算的 p 值。如果您进行了多重假设检验，但没有强有力的依据预期结果具有统计学意义，正如在基因组学中可能发生的情况，在基因组学中可以测量和检验数百万个遗传标记的基因型，您应该验证是否存在某种控制家族错误率或错误发现率的方法(如下一节所述)。否则，这项研究可能没有意义。

报告假设检验的功效也是一个好主意。也就是说，报告 *1 —当假设为假*时，不拒绝零假设的概率。请记住，功效会受到样本大小、显著性水平、数据集中的可变性以及真实参数是否远离零假设假设的参数的影响。简而言之，样本量越大，功效越大。显著性水平越高，权力越大。数据集中的可变性越低，功效就越大。并且真实参数离零假设假设的参数越远，功效越大。

# 用 Bonferroni 修正检验多重假设

由于假阳性的概率随着假设检验次数的增加而增加，因此有必要尝试并控制这种情况。因此，您可能希望控制所有假设测试中出现一个或多个假阳性的概率。这有时被称为*家庭错误率*。

对此进行控制的一种方法是将显著性水平设置为 *α/n* ，其中 *n* 是假设检验的次数。这种校正被称为 [Bonferroni 校正](https://en.wikipedia.org/wiki/Bonferroni_correction)，确保整个系列的错误率小于或等于 *α* 。

然而，这种修正可能过于严格，尤其是当你在进行许多假设检验的时候。原因是因为你在控制家庭的错误率，你也可能会错过一些存在于更高显著性水平的真正的积极因素。显然，在提高假设检验的能力(即，当替代假设为真时，提高拒绝零假设的概率)和控制假阳性之间，需要找到一个平衡点。

# 用 Benjamini-Hochberg 程序检验多个假设

您可以尝试控制错误发现率，而不是尝试控制家族错误率，错误发现率是所有被识别为具有统计显著性结果但实际上没有统计显著性结果的假设检验的比例。换句话说，误发现率等于 *FP/(FP + TP)* 。

控制误发现率应该有助于您识别尽可能多的具有统计显著性结果的假设检验，但仍要尽量保持相对较低的误报比例。像控制假阳性率的 *α* 一样，我们同样使用另一个显著性水平 *β* ，它控制假发现率。

你可以用来控制错误发现率的程序叫做[本杰明-霍克伯格程序](https://en.wikipedia.org/wiki/False_discovery_rate)。你首先选择一个 *β* ，其显著性水平为误发现率。然后计算执行的所有零假设检验的 p 值，并从最低到最高排序，其中 *i* 是列表中 p 值的索引。现在找到最大 p 值的索引 *k* ，使其小于或等于 *i/m*β* ，其中 *m* 是执行的零假设检验的次数。所有 p 值指数为 *i ≤ k* 的零假设检验都被 Benjamini-Hochberg 程序视为具有统计显著性。

# 结论

正如你所看到的，做好科学不仅仅是进行零假设检验，并在你得到小于或等于 5%的 p 值时发表你的发现。有多种方法可以曲解 p 值，调整数据以获得您确信的假设的正确 p 值，以及使用不同的数据样本进行足够多的测试，直到获得所需的 p 值。

但是现在你已经意识到了这些坑洼，并且掌握了一些避免它们的方法，我希望它能帮助你提高研究的质量，让你更接近真相。

*如果你喜欢我在这里写的东西，一定要看看我的* [*个人博客*](https://www.bobbywlindsey.com) *，那里有我在媒体上看不到的文章。*

# 参考

*   [避免 p 值坑洞的 5 个小技巧](https://blogs.plos.org/absolutely-maybe/2016/04/25/5-tips-for-avoiding-p-value-potholes/)
*   [权力的定义](https://newonlinecourses.science.psu.edu/stat414/node/304/)
*   [美国统计协会关于 p 值的声明](http://web9.uits.uconn.edu/lundquis/ASA%20statement%20on%20p%20values.pdf)
*   [对错误发现率和 p 值错误解释的调查](https://royalsocietypublishing.org/doi/full/10.1098/rsos.140216)
*   [误报率对误发现率](https://stats.stackexchange.com/questions/336455/fpr-false-positive-rate-vs-fdr-false-discovery-rate)
*   [控制错误发现率:一种实用且强大的多重测试方法](http://www.math.tau.ac.il/~ybenja/MyPapers/benjamini_hochberg1995.pdf)
*   [重要性测试中的权力介绍—可汗学院](https://www.khanacademy.org/math/ap-statistics/tests-significance-ap/error-probabilities-power/v/introduction-to-power-in-significance-tests)
*   [敏感性和特异性——维基百科](https://en.wikipedia.org/wiki/Sensitivity_and_specificity)
*   [p 值和基础利率谬误——统计有误](https://www.statisticsdonewrong.com/p-value.html)
*   [如何从已发表的研究中计算效应大小:一种简化的方法](http://www.bwgriffin.com/gsu/courses/edur9131/content/Effect_Sizes_pdf5.pdf)
*   [多重比较问题—维基百科](https://en.wikipedia.org/wiki/Multiple_comparisons_problem)

*原载于 2019 年 2 月 25 日*[*bobbywlindsey.com*](https://www.bobbywlindsey.com/2019/02/25/good-science-is-hard/)*。*