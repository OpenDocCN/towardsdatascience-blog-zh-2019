# 作为数据科学家的商业智能—第 1 部分

> 原文：<https://towardsdatascience.com/business-intelligence-as-a-data-scientist-part-1-2974655766c5?source=collection_archive---------33----------------------->

我在英国的分析行业呆过一段时间。多年来，商业智能一直主导着分析市场，这是正确的。在我看来，BI 似乎是对分析感兴趣的人和会写一点代码(SQL！).回到 2015 年左右，我希望进入职业生涯的下一个阶段，并决定我基本上有三条道路。我可以:

晋升为光荣的商务智能团队的光荣商务智能经理

作为数据仓库开发人员或 DBA，逐步晋升为 SQL 大师

完全偏离这一切，成为一个统计怪胎。我不确定这对我的商务智能事业有什么帮助，我只是喜欢和数字打交道

这三个选项都不吸引我。但这没关系——因为 2015 年正是数据科学在英国蓬勃发展的时候！数据科学让我着迷——我的职业生涯一直在处理表格数据集，有列和行，人们编写模型来预测 JPEG 中的对象是猫还是狗。我马上抓住了这个机会。我获得了数据科学硕士学位，有了几年的 DS 经验，现在我发现自己又进入了 BI 领域。为什么？因为商业智能很重要。如果你在一家不做 BI 的公司做 DS，那么这绝对是你应该考虑的事情。

我以前从来没有写过博客，甚至没有写过文章，但是在这个系列中，我将作为一个同时从事两种职业的人，把我的想法写在纸上。BI 和 DS 可以从彼此身上学到很多东西——它们有着内在的联系——不要让任何人告诉你它们不是！

第 1 部分—品牌

![](img/09043a4d4eb19f332b753fb974b7cc28.png)

毫无疑问，我看到的数据科学家最忽视的报告元素是品牌。这里有一些突破:如果你试图用你惊人的蒙特卡罗模拟吸引非技术业务用户，你需要让它看起来很好，否则他们不会感兴趣。当然，我同意 MCS 最重要的元素是运行模拟后获得的统计数据和知识——当然，任何模拟的数字只有在模型设计正确的情况下才有用——但如果因为它看起来太复杂而没有人使用它，那还有什么意义呢？业务用户习惯于以某种方式查看他们的报告，使您的工作符合公司标准绝对是您的工作。

让我们玩一些代码。我们将编写一个蒙特卡洛模拟来玩 *n* 轮盘赌游戏，看看我们是否赢了钱(剧透:我们没有！).这个模型的交付将是 R 闪亮的，我们将讨论使用默认格式交付这个模型和我们良好的公司品牌之间的区别。所有代码都存放在我的 [Github](https://github.com/shaun-parker/shiny-template) 上。

顺便说一句，很难说蒙特卡洛模拟属于商业智能的范畴——我只是认为它们很有趣。不过，R Shiny 完全可以用作 BI 交付系统——有一些软件包可以帮助编写，例如，可以在整个企业中推广的 KPI 仪表板。BI 和 DS 之间的界限已经模糊了！

![](img/c2ed226af5d27d322e33ccd6a0309c40.png)

Although neither profession will help you win at Roulette —at least you’ll be able to explain why you lost.

所以——让我们设定一些基本规则。我的轮盘上有 37 个数字，从 1 到 36，加上房子 0。每个数字和任何其他数字一样有相同的变化，所以我们可以使用 1:37 范围内的随机数发生器(计算机 RNG 实际上不是随机的是一个[整件事](https://qrng.anu.edu.au/)——与本文无关，但如果你喜欢这种东西，这是一个有趣的阅读)。

我们来玩 100 个游戏，看看哪个数字赢的最多。

```
numbers = 1:37 # 36 numbers on a roulette wheel, including 0 = 37n_games = 100 # how many games do you want to play?# Create an empty matrix, n_games long and 37 columns widemat <- matrix(data = 0, nrow=n_games, ncol = max(numbers))# Each game is a row, and the winning number gets a 1 in its cellfor (i in 1:n_games) {current_number = sample(x = numbers, size = 1) #what number came in?mat[i,current_number] <- 1 #whatever number came in, replace it's value with 1}# Create dataframe of winning numberswinners = data.frame("number" = numbers - 1,"wins" = colSums(mat))# Chart itlibrary(ggplot2)ggplot(data=winners, aes(x=number, y=wins)) +geom_bar(stat="identity") +theme_minimal() +ggtitle(paste("Which numbers came in after", n_games, "games of Roulette?"))
```

![](img/e914a050fc73e47406c67de0ac5a22f6.png)

Frequency of winning numbers in our random (not really) number generator

在我们的 100 场比赛中，32 号出现的次数最多(8 次)，12 号和 35 号之间零胜。零号房只出现过一次——对于我虚构的赌场客人来说，这是个不错的结果。

我们开始下注吧。我们将以 1，000 英镑开始，每局下注 10 英镑。根据我在第一次模拟中的结果，我将玩 100 场游戏，每场游戏都赌 32 号。轮盘赌的赔率是 35 比 1，我们就用这个。如果我的号码中了，从 10 英镑的赌注中，我将获得 360 英镑，即 10*35 +我原来的 10 英镑赌注。在 500 场比赛结束时，我们最终获得的启动资金是多于还是少于 1000 英镑？

我们可以构建一个闪亮的应用程序来玩 100 次游戏 *x* 。有很多方法可以做到这一点——我创建的“默认”非股票应用程序如下所示:

![](img/dd14c7b270aba6e8145c3c624001e3f4.png)

我使用了优秀的 [shinydashboard](https://rstudio.github.io/shinydashboard/) 作为我的应用程序的基础框架。滑块让我们玩更多的游戏，我们的中奖号码总是 32。第一张图表显示，每玩一次游戏，我们口袋里的钱就会减少，当数字 32 出现时，我们口袋里的钱就会增加。红色和蓝色显示了 100 场比赛后的净盈利/亏损，底部的图表显示了 RNG 中 32 场比赛的出现频率。

100 场打了 100 次，我们平均留下的现金是 1024.9。一万个游戏玩 100 次呢？

![](img/ee603d8d201a3de22eba24ad66205dc0.png)

The more games played, the less money we leave the casino with

我们玩得越多，输得越多——这当然是赌场赚钱的方式。我玩这个很开心，但是这篇文章是关于品牌的，不是蒙特卡洛模拟！最终，我们将向非技术业务用户展示我们的应用，我向你保证，如果我们稍微清理一下视觉效果，MCS 背后的数学/过程会更容易解释。

我喜欢在 HTML CSS 中做尽可能多的事情，在 ui 中做尽可能少的事情。r 脚本。我绝对不是 HTML 专家，但构建一个与同事共享的模板足够简单，它将处理大量标准格式。每当需要新的格式元素时，可以(也应该)添加这个模板——我的 CSS 只包含这个应用程序所需的最低限度。任何未指定的都将使用 shinydashboard 预设，随着新元素的添加，这些预设可能需要更改。

下面是应用了品牌的仪表板的 a 版本。

![](img/3acdd27d17b8070679217b47e527e36e.png)

Same simulation, better visuals

我们来说说这是怎么回事。我们已经添加了一个漂亮的植物背景(假设这是我们公司的标志)，shinydashboard 元素已经格式化以适应背景主题，我们制作的公司字体已经应用到每个地方，ggplots 也已经主题化。我们还添加了一个新的图表来显示所有模拟游戏的收敛平均值——这显示了一个下降趋势，意味着我们玩得越多，损失的钱就越多。

我使用[sessions.edu 颜色计算器](https://www.sessions.edu/color-calculator/)根据植物背景中的绿色获得补色的 HTML 代码——这些是符合我们公司品牌的美学上令人愉悦的颜色。[色彩理论](https://en.wikipedia.org/wiki/Color_theory)是另一个完整的东西——如果你试图改善你的数据可视化，值得一读。

如上所述，大部分品牌工作都是在 css 文件中完成的。下面我的目录结构显示了存储在 www 文件夹中的 css 文件，它调用 images 目录中的文件。我还存储了数据的硬拷贝——我喜欢这样做是为了重现，以防我们需要将它发送到任何地方。

![](img/5ec4a752197a6fa21f65ba53395dc9c6.png)

Directory tree for the casino app

这些事情总是这样——创建这些应用程序没有对错之分。这篇文章中概述的方法对我来说很有效，但我很想看看数据科学社区中的其他人是如何做品牌实践的。蒙特卡洛模拟本质上并不复杂，但很难向非数据人员解释——标记和格式化您的输出将使解释任务变得容易得多！

这就是第 1 部分！你可以在 [Github](https://github.com/shaun-parker/) 和 [LinkedIn](https://www.linkedin.com/in/shaun-parker-56353886/) 上找到我——第二部分将与品牌推广 ggplot2 有关，还没有完全解决！