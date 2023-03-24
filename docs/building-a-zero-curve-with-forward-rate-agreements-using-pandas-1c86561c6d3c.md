# 使用熊猫构建远期汇率协议的零曲线

> 原文：<https://towardsdatascience.com/building-a-zero-curve-with-forward-rate-agreements-using-pandas-1c86561c6d3c?source=collection_archive---------13----------------------->

![](img/76395bf02f0b1a0c2a751801e4c9a627.png)

Photo by [Markus Spiske](https://unsplash.com/@markusspiske?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

在金融界，如果你想给一种工具定价，并计算出从 t0(现在)到 t(n)的未来价值，你需要使用现货收益率曲线。在专业交易者中，现货收益率曲线被称为零曲线。

如果你现在有 1000 美元可以投资，你可以去银行存一张一年期的存单，很容易得到即期汇率。CD 利率是你的基准。如果你投资了回报比 CD 低的东西(假设风险相对相同),你知道你最好把钱放在 CD 里。这很容易。

但是，如果你知道从现在起一年后你会有 1000 美元，并且想投资一年，那该怎么办呢？你不能走进一家银行，试图锁定一年后的利率。银行不会也不可能告诉你未来的利率是多少。也许银行的不同部门可以，但不是你走进的那个部门，因为目标客户是不同的。

事实上，银行确实知道未来的利率是多少。这就是 FRA。

FRA，或未来利率协议，是双方之间的协议，如果你借出你的钱，你将在期限结束时获得指定的利息和本金。

在本文中，我们将使用 Pandas 建立一个基于 FRAs(远期利率协议)的零曲线。有了这条零曲线，你可以很容易地对某样东西进行定价，从一天到未来十年的任何天数。

为简单起见，我们使用的 FRA 是一年期。实际上，欧洲美元期货(FRA)可以是一个月也可以是三个月。请注意，所有的利率，无论其期限，总是被称为年利率。

```
import pandas as pdfra = pd.DataFrame({
    'Description': ['2019', '2020', '2021', '2022', '2023'],
    'start_date': pd.date_range('2019-01-01', periods=5, freq='YS'),
    'end_date': pd.date_range('2019-01-01', periods=5, freq='Y'),
    'rate': [2.2375, 2.4594, 2.6818, 2.7422, 2.6625]
})
```

这里我们有截至 2019 年 1 月 1 日的 5 年 FRA。如果你在 2019-01-01 存入这笔钱，你将在第一年年底获得 2.2375%的利息，第二年获得 2.4594%，以此类推。

![](img/38bca262447ce0721fef77a9ea29a4a2.png)

我们知道每年的利率，但如果我们想知道最终的复利，我可以向你保证答案不仅仅是把利率相加。下面我们要做的是，年复一年的复合增长。我们来分解一下步骤。

步骤 1:计算从 M 到 N 期间的增长，因此，`mxn_growth`。

```
fra['mxn_growth'] = 1 + fra['rate'] / 100
```

第二步:复合增长是前期复合增长乘以本期增长。因为增长是从时间 0 开始复合的，所以我们称之为`0xn_growth`。

```
fra['0xn_growth'] = fra['mxn_growth'].cumprod()
```

步骤 3:每个 FRA 都是一年期，但增长是前几年的复合增长。

```
fra['years'] = 1
fra['cummulative_years'] = fra['years'].cumsum()
```

第四步:最后我们知道总年数`cummulative_years`的总增长`0xn_growth`。我们只需要把它标准化为年率。正如我们前面指出的，利率总是被称为年利率。

```
fra['zero_rate'] = fra['0xn_growth'] ** (1 / fra['cummulative_years']) - 1
```

![](img/26722f80bf01ffc46130d617319f654d.png)

这就是了。你想知道 2020 年末你的回报是多少，2.3484%是你的利率。

等等！你可能会说，这太简单了。

是的，事实上，请记住，我们有意将每个 FRA 设为一年期。如果它们像欧洲美元期货一样是一个月或三个月呢？你会怎么做？我会把这个挑战留给你。如果你想直接得到答案，请阅读[复利的数学没有你想象的那么简单(困难)](https://medium.com/@5amfung/the-math-of-compounding-interest-is-not-as-simple-difficult-as-you-think-ee2da04a7609)。

这篇文章也有 [Juypter 笔记本格式](https://gist.github.com/5amfung/66e0a6cb267eb026dbcaa1c37e719941)。随便玩吧。