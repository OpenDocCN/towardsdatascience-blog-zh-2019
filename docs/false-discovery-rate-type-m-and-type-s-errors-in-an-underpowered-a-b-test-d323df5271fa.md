# 不充分 A/B 测试中的错误发现率、M 型和 S 型错误

> 原文：<https://towardsdatascience.com/false-discovery-rate-type-m-and-type-s-errors-in-an-underpowered-a-b-test-d323df5271fa?source=collection_archive---------33----------------------->

如果我们转向模拟，我们可以很容易地理解在不充分的统计测试中的“统计显著性”发现如何能够提供关于观察到的效应的再现性、其大小以及其符号的错误结论。

![](img/bb0167727a91dbb647c6f294e66238f1.png)

Harmelen, Netherlands

# 错误的答案

假设我们正在对两个版本的登录页面进行 A/B 测试，我们观察到 4.32%的差异。观察值是随机产生的几率有多大？

我们运行一个双尾 t 检验，对于 p 值，我们得到 0.0139。由于我们决定显著性水平为α = 0.05，我们宣布差异具有统计显著性，并继续实现版本 B 作为我们的登录页面。却看到生产中的表现变成了上面观察到的效果的四分之一。

![](img/b834215d0bde53e9e1c869ddd902a389.png)

**Figure 1.** An example of an A/B test output. Data is simulated using the beta distribution. True means for two distribution are 1.5%(A) and 2.8%(B)

发生了什么事？如果我们将 p 值解释为:“鉴于两个版本之间的真实差异为零，观察到差异 4.32%的概率为 1.39%”，我们将完全错误。

# 假阳性率不应与假发现率混淆

如果我们能够不止一次地运行上面的 t 检验，比如说几万次，我们会对给出错误答案的可能性有一个更清晰的了解，就像上面的例子一样。幸运的是，比如 Rstudio，让我们可以通过模拟来做到这一点。

我们做一万次 t 检验。首先，我们需要转化率 A 和 b 的两个分布，我取[贝塔分布](https://en.wikipedia.org/wiki/Beta_distribution)。这两个分布将由它们的形状参数定义，我将设置为:

*   版本 A

```
# conversions (i.e the number of visitors who subscribed)
> shape1 <- 2 
# visitors who chose not to subscribe (complement of shape1)
> shape2 <- 131
```

*   版本 B

```
# conversions (i.e the number of visitors who subscribed)
> shape1 <- 4
# visitors who chose not to subscribe (complement of shape1)
> shape2 <- 137
```

*请注意，我选择这些分布作为基础事实。意思是，差异的真实均值是 1.3%。图 1 中的例子。是从这些中取样的，因此转换率不同。*

如果两个参数足够大且彼此接近，则贝塔分布接近正态分布。但是我们的访问者数量很少，相对于选择不订阅的访问者数量，订阅者的数量更成问题。也就是说，t 检验中假设的两个条件中有一个不满足。然而，这是我们的模拟，我们的游乐场。让我们看看如果我们假设两个样本接近正态会发生什么。

![](img/91a18a197674bb35cb38af8f272c7033.png)

**Figure 2.** Density plots after 10,000 simulated t-tests. Distribution of the null hypothesis is presented in grey. Simulated data with the mean being the true difference between versions A and B in blue. Red dotted lines represent 95% confidence interval of the null hypothesis. The full grey line represents the example observed effect size 4.32%.

图二。显示真实效果大小有多小和多嘈杂。它几乎与零假设重叠。小的真实效应大小加上噪声测量总是产生不可靠的测试。下面，图 3。展示了我们 t-test 的威力。只有 11%。假设零假设是错误的，只有 11%的 t 检验会正确地拒绝它。这是一个动力不足测试的极端例子。

![](img/13e11d7ea6d8c355a9d4a64009e3e365.png)![](img/0e33743d98ef8b47e41004fe8c84a4af.png)

Figure 3\. Left: the red shaded area represents the power of a t-test where the true effect size is 1.3%. The fraction, equal to the power, of tests that show real effect is the **true positive rate**. Right: the grey shaded area represents the p-value (**false positive rate**) for the example effect size.

为了使用我们测试的能力来计算真正的阳性率，我们需要假设显示真实效果的测试分数(零假设是假的)。这个分数无法精确测量。基本上是贝叶斯先验。一般来说，我们必须在实验开始前假设它。我将使用 10%，因为我们的 10，000 次 t 检验中有 1，064 次显示差异为正，并且其 95%的置信区间不包含零。因此，我们有以下内容:

```
# number of **true positive** : 
> true_positive <- 10000 * 0.1 * 0.11
> true_positive
[1] **110**# number of **false positive** : 
> false_positive <- 10000 * 0.9 * 0.05
> false_positive
[1] **450**# where 10000 is the number of simulated samples
#      0.1 is the fraction of samples that contain the real effect
#      0.11 is the power
#      0.9 is the fraction of samples that don't contain the real #effect
#     0.05 is the significance level
```

现在，我们可以将假发现率计算为假阳性与阳性发现总数的比率:

```
**# false discovery rate**: 
> false_discovery_rate <- false_positive / (false_positive + true_positive)
> false_discovery_rate
[1] 0.8036
```

因为我们测试的功效如此之低，80%的时间我们的“统计显著性”发现将是错误的。因此，在我们的例子中，观察到的差异为 4.32%，p 值为 0.0139，我们犯第一类错误的概率不是 1.39%，而是 80%。

# M 型(夸张)和 S 型(符号)错误

在动力不足的测试中，问题不会以错误的发现而结束。在图 4 中，我们可以看到 10，000 次 t 检验的 p 值分布。其中 11050 个等于或小于 0.05。这是意料之中的，因为功率是 11%。发现网站两个版本之间显著差异的平均值为 4.08。这意味着，平均而言，t 检验会产生比真实效应大小大 3 倍的显著结果。更进一步，在这 11，050 个差值中，有 57 个有错误的符号。我们有很小的机会观察到我们网站的版本 A 比版本 B 表现得更好！

![](img/92be188b94e28a24ea9cf344965a326e.png)![](img/0b50b5a66657d3e1b6901df89a984ca2.png)

Figure 4\. Left: the distribution of the 10,000 p-values. Right: observed effect sizes that correspond to p-value ≤ 0.05

R 中有一个库，可以用来计算这两个误差和乘方。叫做**逆向设计**。我们的真实效应大小和合并标准误差的输出为:

```
> library(retrodesign)
> retrodesign(0.013, 0.017)
$power
[1] 0,1156893$typeS
[1] 0,02948256$exaggeration
[1] 3,278983
```

# 能做些什么？

动力不足的统计测试总是误导人。为了避免对计算出的 p 值得出错误的结论，在进行 A/B 测试之前必须做一些事情:

*   必须设定真实效应大小的估计值。由于偏倚和较大效应具有较小 p 值的趋势，观察到的效应大小往往被夸大。
*   基于该估计和测得的噪声，必须设置适当的样本大小，以实现可接受的功率水平。如上所述，低功率使得测试不可重复。
*   显著性级别应设置为小于 0.003 的值。根据 James Berger [5，6]，p 值=0.0027 对应于 4.2%的假发现率，这接近于 0.05 显著性水平的假阳性率。

*这些模拟的代码可以在我的* [*GitHub*](https://github.com/DarioMakaric/rsimulations-ab-test) *上获得。*

*这些形状参数会给我们带来较大的标准误差和较低的功耗，这正是我们在这个例子中所需要的。由于样本量不足或数据偏斜导致的非正态数据经常出现。转换率样本分布在没有仔细检查条件的情况下被草率地假设为接近正态。如例子所示，这会导致错误的发现。*

*retrodesign 库是由 Andrew Gelman 和约翰·卡林在他们的论文中创建的[1]。*

# 参考

[1]安德鲁·盖尔曼和约翰·卡林(2014)。超越功率计算:评估 S 型(符号)和 M 型(量值)误差。心理科学透视。第九卷第六期第 641 至 651 页。
([DOI:10.1177/1745691614551642](https://journals.sagepub.com/doi/10.1177/1745691614551642)

[2]大卫·科尔昆(2014 年)。对错误发现率和 p 值误解的调查。R. Soc。打开 sci。1: 140216.[http://dx.doi.org/10.1098/rsos.140216](http://dx.doi.org/10.1098/rsos.140216)

[3]克里斯·斯图基奥(2013 年)。使用贝叶斯规则分析转换率(贝叶斯统计教程)。[https://www . chriss tuccio . com/blog/2013/Bayesian _ analysis _ conversion _ rates . html](https://www.chrisstucchio.com/blog/2013/bayesian_analysis_conversion_rates.html)

[4]彼得·博登(2014 年)。差点让我被解雇。[https://blog . sumall . com/journal/optimize ly-get-me-fired . html](https://blog.sumall.com/journal/optimizely-got-me-fired.html)

[5]塞尔克 T，巴亚里 MJ，伯杰乔。(2001).检验精确零假设的 p 值校准。我是。统计。55, 62–71.([DOI:10.1198/000313001300339950](https://www.tandfonline.com/doi/abs/10.1198/000313001300339950))

[6]伯格·乔，塞尔克·t .(1987 年)。检验一个点零假设:p 值和证据的不可调和性。J. Am。统计。协会 82，
112–122。([DOI:10.1080/01621459 . 1987 . 10478397](https://www.tandfonline.com/doi/abs/10.1080/01621459.1987.10478397))