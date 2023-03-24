# 整理的魔力

> 原文：<https://towardsdatascience.com/the-magic-of-tidying-up-d160cb7a5ef6?source=collection_archive---------14----------------------->

![](img/75f481f81ef3facbb208c8389f9cbaf1.png)

A World of Chaotic Data

我们如何处理数据，使其可读、易于理解、可用于发现特征之间可能的相关性，并基于清理后的数据进行预测？无论数据的用途是什么，都需要在有效使用之前对其进行清理和组织。我们今天要用的数据集是足球数据集。(我最喜欢的运动！)随着比赛期间电视上抛出如此多的体育统计数据，了解这些统计数据的来源是一个很好的话题。

![](img/ba2fa4fdf2559853a0d84fef9f04083d.png)

你有没有想过体育赛事的解说员是如何在比赛过程中得出运动员的具体数据的？是的，显而易见的答案是有人在幕后向他们提供信息。但是，那个人从哪里获得信息呢？人们不会在体育赛事期间坐在那里手动查看统计数据。如果必须手动查找相关的事件和数字，很少有人会在正确的情况下找到正确的统计数据。

找到数字也是不够的。需要多种统计数据来与过去的比赛进行比较，以了解运动员是否打破了世界纪录或甚至打破了他们自己的纪录。像现在这样的情况发生过多少次？例如，在规定时间内，下雨的时候，有多少次罚球入网？这是一个相当模糊的统计数据，但信不信由你，评论家们已经提出了比这个例子更疯狂的统计数据。

那么，所有这些统计数据是从哪里来的呢？虽然我们知道是与数据打交道的人，但我指的是一个更具体的问题。处理数据的人是如何得到这些统计数据并提供给评论员的。这就是我们将在这里学到的东西。

有多种组织数据的方法。今天我们将使用 Python 中的一个名为 Pandas 的库。不是我们在电视或变焦镜头中看到的可爱、毛茸茸、黑白相间的熊。Pandas 是一个 Python 包，设计用于数据科学。

我们需要做的第一件事是导入熊猫库。当我们导入它时，我们将它加载到内存中，以便我们在这个项目中工作时可以访问它。当我们导入它时，我们使用 pd 来缩短它。像其他编程语言一样，Python 也有编码约定。

![](img/be84b7b2af9fd54f19806544e4f3f147.png)

Simply import pandas as pd and we can use it further down in the code.

一旦库被导入，我们不需要再次导入它。我们必须向警方报告。在 Python 中命名变量时，约定是全部使用小写字母。

以下代码行将数据集的值赋给变量“soccer”。为了加载数据集，我们使用 pd 和函数“read_csv”来引用我们的 pandas 库。CSV 代表什么？它代表“逗号分隔值”, url 或原始数据集用引号括起来。

![](img/61c0d2ef0635acff95a4adf6232f5d03.png)

The raw dataset assigned to the variable **soccer.** The command below it will give us the first five rows of the dataset.

![](img/208417f30f12134e80329186d5148e69.png)

**soccer.head()** gives us the first 5 rows of the dataframe. That can give us a great amount of information. If we want to see more rows at one time, we can just put a number inside the parentheses.

请注意，数据帧的索引从零开始。我们也可以通过使用 tail 而不是 head 来查看数据帧的最后 5 行。 ***足球.尾巴()。在挑选这些数据之前，我还想知道一件事。我想知道这里总共有多少行和列。下面是我们如何做到这一点。***

![](img/1e272a9107c7f30a1c1412ff16995f0c.png)

This tells us there are 16,781 rows and 19 columns

你认为我们接下来应该看什么？我们想确保数据是可读和可用的。现在，我看到很多写着*南*的价值观。我们需要找出这些值中有多少是存在的，以便能够决定如何处理它们。这可以通过下面的代码来完成。

![](img/0d0079c41c83daa4eaadb8f877b7b508.png)

In every cell there is a True or False. True means there is a null value. False means there is data in that cell.

该数据框包含许多行，因此逐个像元地查看会非常繁琐和耗时。有一种更有效的方法来确定每一列中 null 值的数量。下面是我们如何做到这一点。

![](img/5b227533d9aee28989546d49baa61d87.png)

Clumped together but still not organized

![](img/55ab61d119bd79b6ac3704375eeda70e.png)

> 总共有 16，781 行。我们看到相当多的列中有大量的空值。与总行数相比，这个数字很大，这可能使得这些数据用处不大。我们可以去掉那些柱子。有几种方法可以做到这一点。最安全的方法是创建一个新的数据帧。它让我们的足球数据框架保持原样，这样我们可以在需要时再次引用它。以下是如何仅使用对我们的工作更有用的列来制作副本。

![](img/34472b07ebd2625b55c9e89d4a870d74.png)

Wow! That looks much easier to work with doesn’t it? We still have Null Values here though and we need to address those. Let’s see where they are again. How do we find those? Remember, we have a new variable for the dataframe we’re using.

![](img/9829b06b2e11b8ddc380bd3c14100b4f.png)

Much easier to work with!

![](img/6a4a923813ba5f402099004d561eabf2.png)

我们拥有的行数没有改变。当我们选择不使用它们时，只有列的数量减少了。与总行数相比，这些空值很小。在这里，我们需要决定，我们只是摆脱这些行吗？我们用不同的值填充它们吗？如果是的话，我们选择什么样的价值？

我们看到我们要处理的第一列是日期。你认为日期重要吗？有可能。我们不确定我们到底要找什么。我之前提到的是评论员在比赛中使用统计数据。如果我们把数据用于这个目的，日期就非常重要。但是，16，781 行中的 282 行是最少的。和所有的空值一样，我们可以去掉它们，或者把它们改成平均值、众数、中位数。该决定因人而异，取决于您正在处理的数据。让我们快速看一下这个数据中的最小值、最大值、平均值、众数、中位数和四分位数。当我们这样做时，我们将只获得那些包含数字列的值。接下来我们将讨论分类列。

![](img/a7d9011942b7273eeae4ea6b47185a9e.png)

这里季节的年份没有空值。主客场进球数有相同数量的空值。用平均值替换这些空值不会对这些列的数据产生太大影响，所以这是一个很好的值。对于主场进球，平均值是 1.880657。对于取得的客场进球，平均值为 1.230657。我们可以在 pandas 中使用一个名为。replace()将每列中的所有空值替换为它们的平均值。

![](img/cff4c0b4080260f07d1e666d44ac68f9.png)

We also know the mean of vgoal. We can do the exact same thing to get rid of the null values in that column. If we check for null values again right after we do this, you’ll see they are gone.

![](img/a0aa3193a30248cd8682c01be49d7e5c.png)

我们现在只剩下 Date、visitor、ft 和 tie 列需要处理。记得原始数据帧是 16781 行。在 16，000 多个值中，tie 列中只有三个空值，删除这三行可能不会对我们的数据产生太大影响。因此，我们将删除这三行。我们该拿其他人怎么办？日期是游戏的确切数据。我们有另一个名为“季节”的列，没有空值。你认为日期重要吗？这也取决于我们想从这个数据中了解什么。访客栏代表与主队比赛的客队。由于这是一个字符串，而不是数字，我们不能像用数字那样取平均值。

在这一盘的 16，781 场比赛中，客队有 163 个值缺失。这还不到游戏总数的 1%。我们也可以删除这些行。你认为这会对我们的总体数据和结论产生影响吗？我们可以用模式(访问者栏中最常玩的球队)填充空值。最终，决定权在数据科学家。哪个是更好的选择？我们应该删除空值吗？我们应该用其他信息填充它们吗？如果我们填满了它们，我们用什么来填满它们呢？这些都是你应该问自己的好问题。在这种情况下，我将用访问者列的模式填充它们。

你不需要先找到模式。通过使用下面的代码，可以用模式替换空值。我们现在已经将 visitor 列重新分配给了它自己，但是用模式填充了空值。

![](img/12c8711ddf443f1711031f9cf070513d.png)

我们只需要考虑日期和最终分数。最终得分有 283 个空值。我们可以用 0–0 分数(分数的模式)替换这些行，删除这些行，或者选择不同的值来填充它们。对于最终得分(FT)，我将使用最终得分的模式再次填充它们。

让我们同时处理日期。我们在日期中有 282 个空值，但是在季节中从来没有空值。这意味着我们知道比赛发生在哪个赛季，只是有一些具体的日期，比赛发生在我们不知道的地方。具体日期重要吗？这可能取决于我们想知道什么。我已经展示了必须用不同的值填充空值。在这里，我将删除日期列，因为我对至少拥有游戏发生的赛季感到满意。

在填写最终分数并删除日期后，我们应该没有空值了！

![](img/72f806a1c94eb8b050ac1980958ffce7.png)

Yay! No more null values! dropping the column just used ‘drop’ after the data frame name. Specified what column to drop, and the specified the axis. axis=1 is for columns. axis=0 is for rows.

正是在这一点上，我们可以使用数据来找出额外的信息。我可能会问的一个问题是，“赢得比赛和在你的主场比赛之间有关联吗？”根据赛季的不同，比赛输赢有什么趋势吗？哪个队赢了最多的比赛？从这一小组数据中可以找到很多问题的答案。想象一下，我们可以从不同来源的大量数据中发现什么。我们可以回答曾经被认为不可能得到答案的问题。电脑如此强大，让我们有能力回答各种问题。它还赋予我们训练计算机完成特定任务的能力。未来的话题。

这是如何清理小型数据集的一个非常基本的指南。这是一个开始的好地方。如果我激发了你的兴趣，你可以访问网上的大量资料来了解更多。而且，如果你发现自己彻底着迷了，Lambda School 有一个数据科学/机器学习项目，它会让你大吃一惊！