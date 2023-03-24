# 为数据科学家简单解释 p 值

> 原文：<https://towardsdatascience.com/p-value-explained-simply-for-data-scientists-4c0cd7044f14?source=collection_archive---------2----------------------->

![](img/36c4dd5d0cf2c30e765fd0be106b440c.png)

Our Null and Alternate Hypothesis in the battle of the century. Image by [Sasin Tipchai](https://pixabay.com/users/sasint-3639875/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=1822701) from [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=1822701)

## 没有统计学家的自命不凡和数据科学家的冷静

最近，有人问我如何用简单的术语向外行人解释 p 值。我发现很难做到这一点。

p 值总是很难解释，即使对了解它们的人来说也是如此，更不用说对不懂统计学的人了。

我去维基百科找了些东西，下面是它的定义:

> 在统计假设检验中，对于给定的统计模型，p 值或概率值是指当零假设为真时，统计汇总(如两组之间的样本均值差异)等于或大于实际观察结果的概率。

我的第一个想法是，他们可能是这样写的，所以没有人能理解它。这里的问题在于统计学家喜欢使用大量的术语和语言。

***这篇文章是关于以一种简单易懂的方式解释 p 值，而不是像统计学家*** 那样自命不凡。

# 现实生活中的问题

在我们的生活中，我们当然相信一件事胜过另一件事。

从**的*明显可以看出*的**像——地球是圆的。或者地球绕着太阳转。太阳从东方升起。

不确定性程度不同的**——运动减肥？或者特朗普会在下一次选举中获胜/失败？或者某种特定的药物有效？还是说睡 8 小时对身体有好处？**

**前一类是事实，后一类因人而异。**

**所以，如果我来找你说运动不影响体重呢？**

**所有去健身房的人可能会对我说不那么友好的话。但是有没有一个数学和逻辑的结构，在其中有人可以反驳我？**

**这让我们想到了假设检验的概念。**

# **假设检验**

**![](img/5920b6f4a6918977ebbcb1bf73c71239.png)**

**Exercising doesn’t reduce weight?**

**所以我在上面例子中的陈述——运动不影响体重。这个说法是我的假设。暂且称之为 ***无效假设*** 。目前，这是我们认为真实的现状。**

**那些信誓旦旦坚持锻炼的人提出的 ***替代假设*** 是——锻炼确实能减肥。**

**但是我们如何检验这些假设呢？我们收集数据。我们收集了 10 个定期锻炼超过 3 个月的人的减肥数据。**

> **失重样品平均值= 2 千克**
> 
> **样品标准偏差= 1 千克**

**这是否证明运动确实能减肥？粗略地看一下，似乎锻炼确实有好处，因为锻炼的人平均减掉了 2 公斤。**

**但是你会发现，当你进行假设检验时，这些明确的发现并不总是如此。如果锻炼的人体重减轻了 0.2 公斤会怎么样。你还会这么肯定运动确实能减肥吗？**

**那么，我们如何量化这一点，并用数学来解释这一切呢？**

**让我们建立实验来做这件事。**

# **实验**

**让我们再次回到我们的假设:**

*****H :*** 练功不影响体重。或者相当于𝜇 = 0**

*****Hᴬ:*** 运动确实能减肥。或者相当于𝜇 > 0**

**我们看到 10 个人的数据样本，我们试图找出**

**观察到的平均值(锻炼者的体重减轻)= 2 公斤**

**观察到的样品标准偏差= 1 千克**

**现在问我们自己一个好问题是— ***假设零假设为真，观察到 2 kg 或者比 2 kg 更极端的样本均值的概率是多少？*****

**假设我们可以计算出这个值——如果这个概率值很小(小于阈值),我们拒绝零假设。否则，我们无法拒绝我们的无效假设。 ***为什么拒绝失败而不接受？*** *我* 这个后面会回答。**

**这个概率值实际上就是 p 值。简单地说，如果我们假设我们的零假设是正确的，它只是观察我们所观察到的或极端结果的概率。**

*****统计学家称该阈值为显著性 level(𝜶，在大多数情况下，𝜶取为 0.05。*****

*****那么我们怎么回答:*** 假设原假设为真，得到 2 kg 或大于 2 kg 的值的概率是多少？**

**这是我们最喜欢的分布，图中的正态分布。**

# **正态分布**

**假设我们的零假设为真，我们创建重量损失样本平均值的抽样分布。**

*****中心极限定理:*****中心极限定理**简单来说就是，如果你有一个均值为μ、标准差为σ的总体，并从总体中随机抽取样本，那么**样本**均值的**分布**将近似为正态分布**以均值为总体均值**、**标准差σ/√n **。**其中σ是样本的标准差，n 是样本中的观察次数。****

**现在我们已经知道了由零假设给出的总体均值。所以，我们用它来表示均值为 0 的正态分布。其标准偏差由 1/√10 给出**

**![](img/802df1b851881828623216c4b5d7d591.png)**

**The sampling distribution is a distribution of the mean of samples.**

**事实上，这是总体样本均值的分布。我们观察到一个特殊的平均值，即 2 公斤。**

**现在我们可以使用一些统计软件来找出这条特定曲线下的面积:**

```
**from scipy.stats import norm
import numpy as npp = 1-norm.cdf(2, loc=0, scale = 1/np.sqrt(10))
print(p)
------------------------------------------
1.269814253745949e-10**
```

**因此，这是一个非常小的概率 p 值(<significance level="" of="" for="" the="" mean="" a="" sample="" to="" take="" value="" or="" more.=""></significance>**

**And so we can reject our Null hypothesis. And we can call our results statistically significant as in they don’t just occur due to mere chance.**

# **The Z statistic**

**You might have heard about the Z statistic too when you have read about Hypothesis testing. Again as I said, terminology.**

**That is the extension of basically the same above idea where we use a standard normal with mean 0 and variance 1 as our sampling distribution after transforming our observed value x using:**

**![](img/0cc03bddcda517a77886e8c246cdd706.png)**

**This makes it easier to use statistical tables. In our running example, our z statistic is:**

```
**z = (2-0)/(1/np.sqrt(10))
print(z)
------------------------------------------------------
6.324555320336758**
```

**Just looking at the Z statistic of > 6 应该让您知道观察值至少有六个标准差，因此 p 值应该非常小。我们仍然可以使用以下公式找到 p 值:**

```
**from scipy.stats import norm
import numpy as npp = 1-norm.cdf(z, loc=0, scale=1)
print(p)
------------------------------------------------------
1.269814253745949e-10**
```

**正如你所看到的， ***我们使用 Z 统计量得到了相同的答案。*****

# **一个重要的区别**

**![](img/ca0855bbb78e41a84de41cc109912a10.png)**

**Our Jurors can never be definitively sure so they don’t accept they just reject.**

**我们之前说过，我们拒绝我们的零假设，因为我们有足够的证据证明我们的零假设是错误的。**

**但是如果 p 值高于显著性水平。然后我们说，我们不能拒绝零假设。为什么不说接受零假设呢？**

**最直观的例子就是使用审判法庭。在审判法庭上，无效假设是被告无罪。然后我们看到一些证据来反驳零假设。**

**如果我们不能反驳无效的假设，法官不会说被告没有犯罪。法官只是说，根据现有的证据，我们无法判定被告有罪。**

**推动这一观点的另一个例子是:假设我们正在探索外星球上的生命。而我们的零假设( ***H*** )就是这个星球上没有生命。我们漫游了几英里，寻找那个星球上的人/外星人。如果我们看到任何外星人，我们可以拒绝零假设，支持替代方案。**

**但是如果我们没有看到任何外星人，我们能肯定地说这个星球上没有外星生命或者接受我们的无效假设吗？也许我们需要探索更多，或者也许我们需要更多的时间，我们可能已经找到了外星人。所以，在这种情况下，我们不能接受零假设；我们只能拒绝它。或者用 [Cassie Kozyrkov 的](https://medium.com/hackernoon/statistical-inference-in-one-sentence-33a4683a6424)话来举例，我们可以说 ***“我们没学到什么有趣的东西”。*****

> **在 STAT101 课上，他们会教你在这种情况下写一段令人费解的段落。(“我们未能拒绝零假设，并得出结论，没有足够的统计证据来支持这个星球上存在外星生命。”)我深信这种表达的唯一目的就是让学生的手腕紧张。我总是允许我的本科生这样写:我们没有学到任何有趣的东西。**

**![](img/1548448d08ab56255e817ce456a9e54d.png)**

**Riddikulus: Hypothesis testing can make the null hypothesis look ridiculous using p-values (The Wand)**

*****本质上，假设检验就是检查我们的观察值是否让零假设看起来很荒谬*** 。如果是，我们拒绝零假设，称我们的结果具有统计学意义。除此之外，我们没有学到任何有趣的东西，我们继续我们的现状。**

# **继续学习**

**如果你想了解更多关于假设检验、置信区间以及数字和分类数据的统计推断方法，Mine etinkaya-Rundel 在 coursera 上教授[推断统计学](https://coursera.pxf.io/DVZ13d)课程，这是最简单不过的了。她是一个伟大的导师，很好地解释了统计推断的基础。**

**谢谢你的阅读。将来我也会写更多初学者友好的帖子。在 [**媒体**](https://medium.com/@rahul_agarwal?source=post_page---------------------------) 关注我或者订阅我的 [**博客**](https://mlwhiz.ck.page/a9b8bda70c) 了解他们。一如既往，我欢迎反馈和建设性的批评，可以通过 Twitter [@mlwhiz](https://twitter.com/MLWhiz?source=post_page---------------------------) 联系**

**此外，一个小小的免责声明——在这篇文章中可能会有一些相关资源的附属链接，因为分享知识从来都不是一个坏主意。**