# 理解数据

> 原文：<https://towardsdatascience.com/what-is-data-8f94ae3a56b4?source=collection_archive---------1----------------------->

## 关于信息、内存、分析和分布的思考

Here’s the audio version of the article, read for you by the author.

我们感官感知的一切都是数据，尽管它在我们大脑湿物质中的存储还有待改进。写下来更可靠一点，尤其是当我们在电脑上写下来的时候。当那些笔记组织良好时，我们称之为 ***数据*** ...虽然我见过一些非常混乱的电子涂鸦得到相同的名字。我不知道为什么有些人把 ***data*** 这个词读起来像是有个大写的 D 在里面。

> 为什么我们用大写 D 来读数据？

我们需要学会对数据采取不敬的务实态度，因此本文旨在帮助初学者了解幕后情况，并帮助从业者向表现出数据崇拜症状的新手解释基础知识。

![](img/b9634cd7471b808fda17a37f49fdc735.png)

# 感觉和感官

如果你从网上购买数据集开始你的旅程，你有忘记它们来自哪里的危险。我将从绝对的零开始向你展示你可以随时随地制作数据。

这里有一些常年住在我的储藏室，安排在我的地板上。

![](img/0f46f3e6e7ba01b85bf456e3ffa40c43.png)

My life is pretty much a Marmite commercial. Three sizes; Goldilocks would be happy here.

这张照片是数据——它被存储为你的设备用来向你展示美丽色彩的信息。(如果你很想知道当你看到矩阵时图像是什么样的，看看[我的监督学习介绍](http://bit.ly/quaesita_slkid)。)

让我们从我们看到的东西中找到一些意义。我们有无限的选择去关注和记忆什么。这是我看食物时看到的一些东西。

![](img/21cd32b223cbde9377ca5c1fb1a021ce.png)

There’s no universal law that says that this, the weight in grams, is the best thing to pay attention to. We‘re allowed to prefer volume, price, country of origin, or anything else that suits our mood.

如果你闭上眼睛，你还记得你刚刚看到的每一个细节吗？没有吗？我也没有。这差不多就是我们收集数据的原因。如果我们能在头脑中完美地记住并处理它，那就没有必要了。互联网可以是一个洞穴里的隐士，记录人类所有的推文，完美地呈现我们数十亿张猫的照片。

# 书写和耐久性

因为人类的记忆是一个漏桶，所以用我们在黑暗时代上学时统计数据的方式记下信息将是一种进步。是的，我的朋友们，我还有纸在这里的某个地方！让我们记下这 27 个[数据点](http://bit.ly/quaesita_slkid)。

![](img/e89823b8ec816cab672d11b32f30417d.png)

This is data. Remind me why we’re worshipping it? Data are always a caricature of reality made to the tastes of a human author. This one’s full of subtle choice — shall we record dry weight or wet weight? What to do with volume units? Also, I might have made mistakes. If you inherit my data, you can’t trust your eyes unless you know what exactly happened in the data collection.

这个版本的伟大之处——相对于我的海马体或地板上的东西——在于它更耐用、更可靠。

> 人类的记忆是一个漏桶。

我们认为记忆革命是理所当然的，因为它始于几千年前，商人需要一个可靠的记录，谁卖给谁多少蒲式耳的东西。花一点时间来意识到拥有一个比我们的大脑更好地存储数字的通用书写系统是多么的荣耀。当我们记录数据时，我们产生了对我们丰富感知的现实的不忠实的破坏，但在那之后，我们可以以完美的保真度将结果的未被破坏的副本转移给我们物种的其他成员。文笔惊人！存在于我们体外的思想和记忆。

> 当我们分析数据时，我们在访问别人的记忆。

担心机器胜过我们的大脑？连纸都能做到！这 27 个小数字对你的大脑存储来说是一个很大的提升，但如果你手边有一个书写工具，耐用性是有保证的。

虽然这是一个耐久性的胜利，但用纸工作是令人讨厌的。例如，如果我突发奇想，把它们从最大到最小重新排列，会怎么样？纸，给我一个更好的顺序！没有？可恶。

# 电脑和魔咒

你知道软件最棒的是什么吗？咒语真的有效！所以让我们从纸张升级到电脑。

![](img/c79abdfd4d52dd2b9696c92e32efe3bd.png)

Ah, spreadsheets. Baby’s first data wrangling software. If you meet them early enough, they seem friendly by dint of mere exposure. Spreadsheets are relatively limited in their functionality, though, which is why data analysts prefer to strut their stuff in Python or R.

电子表格让我不冷不热。与现代数据科学工具相比，它们非常有限。我更喜欢在 R 和 Python 之间摇摆，所以这次让我们试试 R。你可以在浏览器中用 Jupyter 跟随[:点击](http://bit.ly/jupyter_try) [*【带 R】*框](http://bit.ly/jupyter_try)，然后点击几次剪刀图标，直到所有内容都被删除。恭喜，只花了 5 秒钟，你就可以粘贴我的代码片段并运行它们了。

```
weight <- c(50, 946, 454, 454, 110, 100, 340, 454, 200, 148, 355, 907, 454, 822, 127, 750, 255, 500, 500, 500, 8, 125, 284, 118, 227, 148, 125)
weight <- weight[order(weight, decreasing = TRUE)]
print(weight)
```

你会注意到，如果你是新来的，R 的 abracadabra 对你的数据排序并不明显。

嗯,“abracadabra”这个词本身就是如此，电子表格软件中的菜单也是如此。你知道这些事情只是因为你接触过它们，而不是因为它们是普遍规律。要用电脑做事情，你需要向你的常驻占卜师要魔法单词/手势，然后练习使用它们。我最喜欢的圣人叫做互联网，他知道所有的事情。

![](img/eb223ed0a5877a20dd7b8e64ca576ca3.png)

Here’s what it looks like when you run that code snippet in Jupyter in your browser. I added [comments](http://bit.ly/code_comments) to explain what each line does because I’m polite sometimes.

为了加快你的巫师训练，不要只是粘贴咒语——试着改变它们，看看会发生什么。比如上面片段中把 TRUE 变成 FALSE 会有什么变化？

你这么快就得到答案是不是很神奇？我喜欢编程的一个原因是它是魔法和乐高的结合。

> 如果你曾经希望你能变魔术，那就学着写代码吧。

简单地说，编程是这样的:向互联网询问如何做某件事，用你刚刚学到的神奇单词，看看当你调整它们时会发生什么，然后像乐高积木一样把它们放在一起，完成你的命令。

# 分析和总结

这 27 个数字的问题在于，即使它们被排序，对我们来说也没有多大意义。当我们阅读它们的时候，我们会忘记刚刚读过的内容。这是人类的大脑。告诉我们去读一个由一百万个数字组成的有序列表，我们最多能记住最后几个。我们需要一种快速的方法来分类和总结，这样我们就可以掌握我们正在看的东西。

这就是[分析](http://bit.ly/quaesita_datasci)的用途！

```
median(weight)
```

用正确的咒语，我们可以立即知道体重的中位数是多少。(Median 的意思是“中间的东西”。)

![](img/2394dedb7e395be9fe744e94f9d017d6.png)

This is for the three of you who share my taste in [movies](http://bit.ly/fish_called_wanda).

结果答案是 284g。谁不爱瞬间的满足感？有各种各样的汇总选项:*(min()、max()、mean()、median()、mode()、variance()* …都试试吧！或者试试这个神奇的单词，看看会发生什么。

```
summary(weight)
```

对了，这些东西叫做 ***统计*** 。统计数据是任何一种将你的数据混在一起的方式。这不是统计学领域的内容——这里有一段 8 分钟的学术介绍。

![](img/31c1172c91bf2c88b6b9be2b442781d2.png)

# 绘图和可视化

这一部分不是关于那种涉及世界统治的阴谋(请继续关注那篇文章)。就是用图片总结数据。事实证明，一张照片可能比一千个单词更有价值——每个数据点一个单词，然后更多。(这样的话我们就做一个只值 27 个砝码的。)

![](img/61b051bed1920643a348261a0dd2d2e3.png)

Tip jars are nature’s [bar charts](http://bit.ly/bar_wiki), pun intended. More height means more popularity in that category. Histograms are almost the same thing, except that the categories are ordered.

如果我们想知道权重在我们的数据中是如何*分布的——例如，是 0 到 200g 之间的项目多，还是 600g 到 800g 之间的项目多？— a ***直方图*** 是我们最好的朋友。*

*![](img/faa13ead6276faa5202a40e682ec7d12.png)*

*Nature’s histogram.*

*直方图是总结和显示样本数据的方法之一。对于更受欢迎的数据值，它们的块更高。*

> *把条形图和直方图想象成流行度竞赛。*

*要在电子表格软件中制作一个，神奇的咒语是在各种菜单上点击一长串。在 R 中，速度更快:*

```
*hist(weight)*
```

*这是我们的一行程序带给我们的:*

*![](img/02d304976418ef45994eb6dd81af7c38.png)*

*This is one ugly histogram — but then I’m used to the finer things in life and know the [beauty of what you can do with a few more lines of code in R](http://bit.ly/histogram_tutorial) . Eyesore or not, it’s worth knowing how easy the basics are.*

*我们在看什么？*

*在水平轴上，我们有垃圾箱(或者小费罐，如果你喜欢的话)。默认情况下，它们被设置为 200g 的增量，但我们稍后会更改。纵轴是计数:有多少次我们看到重量在 0 克到 200 克之间？剧情说 11。600g 到 800g 之间怎么样？只有一个(如果没记错的话，那是食盐)。*

*我们可以选择我们的容器大小——我们没有修改代码得到的默认容器是 200g 容器，但是也许我们想使用 100g 容器来代替。没问题！受训的魔术师可以修改我的咒语来发现它是如何工作的。*

```
*hist(weight, col = "salmon2", breaks = seq(0, 1000, 100))*
```

*结果如下:*

*![](img/4c567b2e42f7e501fc61f9d09aef1831.png)*

*现在我们可以清楚地看到，两个最常见的类别是 100–200 和 400–500。有人在乎吗？大概不会。我们这样做只是因为我们可以。另一方面，一个真正的分析师擅长快速浏览数据的科学和寻找有趣的金块的艺术。如果他们的手艺[好](http://bit.ly/quaesita_hallows)，他们就值他们[重量的黄金](http://bit.ly/quaesita_hero)。*

# *什么是发行版？*

*如果这 27 项是我们关心的所有东西，那么我刚刚做的这个直方图样本也正好是人口分布。*

*这基本上就是*的分布:如果你将 *hist()* 应用于整个[人口](http://bit.ly/quaesita_statistics)(所有你关心的信息)，而不仅仅是[样本](http://bit.ly/quaesita_statistics)(你手头上碰巧有的数据)，你就会得到这个直方图。有几个脚注，比如 y 轴上的刻度，但是我们会把它们留到另一篇博文中——请不要伤害我，数学家们！**

**![](img/e4397ef25321ae30dc061cc1f6b55320.png)**

**A distribution gives you popularity contest results for your whole [population](http://bit.ly/quaesita_popwrong). It’s basically the population histogram. Horizontal axis: population data values. Vertical axis: relative popularity.**

**如果我们的人口都是 T21 的包装食品，那么分布将会是他们体重的直方图。这种分配仅仅作为一种理论概念存在于我们的想象中——一些包装食品已经湮没在时间的迷雾中。即使我们想，我们也不能建立那个数据集，所以我们能做的最好的事情就是用一个好的样本来猜测它。**

# **什么是数据科学？**

**有各种各样的观点，但我喜欢这个定义:“ [**数据科学**](http://bit.ly/quaesita_datasci) **是使数据有用的学科**”它的三个子领域涉及挖掘大量信息以获得灵感([分析](http://bit.ly/quaesita_analysts))，基于有限信息明智地做出决策([统计](http://bit.ly/quaesita_statistics))，以及使用数据中的模式来自动化任务( [ML/AI](http://bit.ly/quaesita_emperor) )。**

> **所有的数据科学都归结为一点:知识就是力量。**

**宇宙中充满了等待收获和利用的信息。虽然我们的大脑在驾驭现实方面令人惊叹，但它们不太擅长存储和处理某些类型的非常有用的信息。**

**这就是为什么人类首先求助于泥板，然后求助于纸，最终求助于硅。我们开发了快速查看信息的软件，如今知道如何使用它的人自称为数据科学家或 T2 数据分析师。真正的英雄是那些构建工具，让这些从业者更好更快地掌握信息的人。顺便说一句，即使是[互联网也是一种分析工具](http://bit.ly/quaesita_versus)——我们只是很少这样想，因为即使是孩子也能做那种数据分析。**

**![](img/9ed1fc142d2f5388636005dfd3be6f83.png)**

# **面向所有人的内存升级**

**我们感知的一切都储存在某个地方，至少是暂时的。数据没有什么神奇的，除了它比大脑管理的更可靠。有些信息是有用的，有些是误导的，其余的都在中间。数据也是如此。**

> **我们都是数据分析师，一直都是。**

**我们认为我们惊人的生物能力是理所当然的，并夸大了我们与生俱来的信息处理能力和机器辅助能力之间的差异。不同之处在于耐用性、速度和规模……但常识性的规则同样适用于这两者。为什么那些规则在一个等式的第一个符号就被抛弃了？**

**![](img/7fa3bc9aed7154f7a18f5a4b20456ec9.png)**

**Still looking for Data to pronounce with a capital D? [Well, there it sits.](http://bit.ly/startrek_capitaldata)**

**我很高兴我们庆祝信息作为进步的燃料，但崇拜数据是神秘的东西对我来说没有意义。最好简单地谈论数据，因为我们都是数据分析师，而且一直都是。让我们让每个人都能这样看待自己！**

# **感谢阅读！YouTube AI 课程怎么样？**

**如果你在这里玩得开心，并且你正在寻找一个为初学者和专家设计的有趣的应用人工智能课程，这里有一个我为你制作的娱乐课程:**

**Enjoy the entire course playlist here: [bit.ly/machinefriend](http://bit.ly/machinefriend)**

# **喜欢作者？与凯西·科兹尔科夫联系**

**让我们做朋友吧！你可以在 [Twitter](https://twitter.com/quaesita) 、 [YouTube](https://www.youtube.com/channel/UCbOX--VOebPe-MMRkatFRxw) 、 [Substack](http://decision.substack.com) 和 [LinkedIn](https://www.linkedin.com/in/kozyrkov/) 上找到我。有兴趣让我在你的活动上发言吗？使用[表格](http://bit.ly/makecassietalk)取得联系。**