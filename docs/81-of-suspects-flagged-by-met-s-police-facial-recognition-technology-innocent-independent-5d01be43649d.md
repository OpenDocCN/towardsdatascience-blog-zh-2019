# 使用朴素贝叶斯挖掘“81%的‘嫌疑人’被 Met 的警方面部识别技术标记为无辜”

> 原文：<https://towardsdatascience.com/81-of-suspects-flagged-by-met-s-police-facial-recognition-technology-innocent-independent-5d01be43649d?source=collection_archive---------16----------------------->

天空新闻:伦敦

![](img/aec07edffcb3af6c4ef2dcc669efb657.png)

Photo by [Daniel Lincoln](https://unsplash.com/@danny_lincoln?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/police-tape?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

最近我在我的[朴素贝叶斯](/naive-bayes-and-disease-detection-ffefe2cc5c01)帖子中写道“人群中疾病的发病率是许多人忽视的关键变量”。

今天在浏览新闻时，我偶然看到了天空新闻的文章，并注意到他们已经落入了这个陷阱。

> 研究人员发现这个有争议的系统 81%是不准确的

**嫌疑人在人群中的出现率是一个关键变量，而天空新闻却忽略了这个变量**

这是为什么呢？让我们试着改变人群中嫌疑人的发生率，看看为什么它如此重要。

```
**System 1**: Randomly tosses a coin to assess guilt, accuses 50%
**System 2**: Error rate of 1 in 1,000**Crowd A**: 32,000 suspects;      0 innocent people
**Crowd B:** 0 suspects; 32,000 innocent people
```

哪个系统比较好？

```
**System 1** on **Crowd A** - 16000 suspects;         100% accurate!
**System 2** on **Crowd B** - 32 mistakes;           100% inaccurate!
```

人群中嫌疑人的发生率有很大的不同。天空新闻忽略了人群组成，只测量准确性，因此通过这种方法**系统 1** 的硬币投掷获胜。

因此，让我们试着将我之前的朴素贝叶斯文章中的理论引入到这个现实世界的例子中。

首先，我们需要从这句话开始:

> 他们发现，在 42 个匹配中，只有 8 个被证实是正确的——错误率为 81%。42 人中有 4 人因为被人群吸收，一直没有找到，所以无法验证是否匹配。

数学很简单:

![](img/0cb03aced1506fe7cf31c3fdf9de7a54.png)

但是*“42 人中有 4 人是永远找不到的人”*所以不知道为什么会被统计。我们不知道这四个是不正确的，正确的还是混合的。在我看来，这些例子需要被丢弃，留给我们 38 个匹配中的 8 个或者 79%不正确，但是我离题了。

我们需要关于系统性能的更多细节。在文章中我们可以找到一个[警队](https://getyarn.io/yarn-clip/f89ce89e-3450-417e-a8de-5783353e3d85)误差估计为千分之一:

> 该部队坚称，他们的技术只会在千分之一的情况下出错——但他们使用了不同的测量方法来得出这一结论。

因为他们引用了一个数字。没有出现假阴性/阳性，所以我们假设两者是一样的。根据贝叶斯理论，这些数字是什么？尽管我不同意，我还是会用他们的 19%或 0.19:

```
TP = True Positive = 0.999
FP = False Negative = 0.001 P(B ∣ A) P(A)
P(A ∣ B) =  ──────────────
                P(B)
A = Suspect
B = Positive Neoface matchP(A ∣ B) = Probability a person is a suspect given a match = 0.19
P(B ∣ A) = Probability of a match given a suspect = TP = 0.999
P(A) = Probability of a person in the crowd being a criminal
P(B) = Probability of a Neoface match = FP × (1-P(A)) + TP × P(A)
```

插入公式和数值，求解 P(A)，用铅笔算出来([绝对不用 Wolfram Alpha](https://www.wolframalpha.com/input/?i=solve+c%3D(t*a)%2Fd+for+a+where+f%3D0.001,+t%3D0.999,+c%3D0.19,+d%3D(f*(1-a)%2Bt*a)) )你得到:

![](img/4576f14d7ec6cc64f8fe8260fe0c0c60.png)

19/80938

或者说 4000 分之一。这是对随机人群中被通缉嫌疑人数量的合理估计吗？英国的监狱人口是千分之一。这使得警方已知的嫌疑人数量与囚犯数量处于同一数量级，但却少了 4 倍。似乎很合理。

让我们对 32，000 人的人群进行一次健全性检查，我们估计 4000 人中有 1 人是嫌疑人，并且该系统(据称)在检测他们方面有 99.9%的可靠性。所以 8 个嫌疑犯中有 8 个被发现了。它还(声称)在拒绝非嫌疑人方面有 99.9%的可靠性，因此 31，992 个非嫌疑人中的 32 个将被错误地检测到。让我们扩展一下前面的例子:

```
**System 1**: Randomly tosses a coin to assess guilt, accuses 50%
**System 2**: Error rate of 1 in 1,000**Crowd A**: 32,000 suspects;      0 innocent people
**Crowd B:** 0 suspects; 32,000 innocent people
**Crowd C**:      8 suspects; 31,992 innocent people**System 1** on **Crowd A** - 16000 suspects;         100% accurate!
**System 2** on **Crowd B** - 32 mistakes;            100% inaccurate!
**System 2** on **Crowd C** - 8 suspects, 32 mistakes; 80% inaccurate!
```

我们兜了一圈，又回到了 80%不准确的头条数字。所以下面的事情可以同时成立:

1.  该系统的假阴性和假阳性率为 0.1%
2.  在 32000 人的人群中，40%或 80%的人会被错误标记

如果在随机人群中以 4000 分之一的概率出现嫌疑人，以上两件事都可能成立。

**人群中嫌疑人的发生率是一个关键变量**

在文章的后面，我们发现了以下内容:

> Met 更喜欢通过将成功和不成功的匹配与面部识别系统处理的面部总数进行比较来测量准确性。根据这一指标，错误率仅为 0.1%。

这并不奇怪，也很合理。在伦敦，随机人群中嫌疑人的频率可能极低，因此，即使是高性能的面部检测系统*对那些没有受过贝叶斯定理教育的人来说也表现不佳。*

自然，机器学习研究人员已经发现了衡量表现的这个问题，并通过混合使用[精度、Recall 等人的](https://en.wikipedia.org/wiki/Precision_and_recall)衡量标准解决了这个问题，但 F1 分数为我们提供了一个很好的单一衡量标准:

![](img/01fd8bfb8084a35dbfb33f2d3987ebf8.png)

F1 Score

```
 2TP                 2 × 0.999
 F1 =  ─────────────  =  ───────────────────────── = 0.999
       2TP + FP + FN     2 × 0.999 + 0.001 + 0.001
```

最大可能的 F1 分数是 1，因此这是一个高性能的系统，但当试图检测大海捞针或随机人群中的嫌疑人时，您仍然会遇到很多误报。这是一项艰巨的任务。

我看了半打新闻来源，但只发现同样的故事重复，没有关键的分析。

[天空新闻](https://news.sky.com/story/met-polices-facial-recognition-tech-has-81-error-rate-independent-report-says-11755941) [卫报](https://www.theguardian.com/technology/2019/jul/03/police-face-calls-to-end-use-of-facial-recognition-software) [ABC 新闻](https://abcnews.go.com/International/80-facial-recognition-suspects-flagged-londons-met-police/story?id=64129255) [布莱巴特](https://www.breitbart.com/tech/2019/07/05/report-81-of-suspects-flagged-by-uk-police-facial-recognition-are-innocent/) [麻省理工科技评论](https://www.technologyreview.com/f/613922/london-polices-face-recognition-system-gets-it-wrong-81-of-the-time/) [镜报](https://www.mirror.co.uk/tech/facial-recognition-used-police-forces-17389070)