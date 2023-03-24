# 我在数据科学硕士课程中学到的 7 件事

> 原文：<https://towardsdatascience.com/top-7-things-i-learned-on-my-data-science-masters-bd89fdbab769?source=collection_archive---------1----------------------->

## 尽管我还在学习，这里还是列出了我学到的最重要的东西(到目前为止)。

其中一些你已经很熟悉了，但是我不建议跳过它们——另一种观点总是很有用的。

![](img/c328f2bb3d21ad3e9eae73bbfe629002.png)

Photo by [Charles DeLoye](https://unsplash.com/@charlesdeloye?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 1.总是咨询领域专家

我说的“总是”是指如果你有机会这样做的话。

这是最先学会的事情之一。我们被介绍给这个家伙，当谈到数据世界时，他就像一个摇滚明星，更准确地说是搅动建模。听到这句话的时候，可能是我关于数据科学的第一个神话被打破的时候。

你可以在这里阅读更多关于他和整个案件的信息:

[](/attribute-relevance-analysis-in-python-iv-and-woe-b5651443fc04) [## Python-IV 和 WoE 中的属性相关性分析

### 最近我写了一篇关于递归特征消除的文章，这是我最常用的特征选择技术之一…

towardsdatascience.com](/attribute-relevance-analysis-in-python-iv-and-woe-b5651443fc04) 

最初，我认为数据科学家是一种罕见的物种，只要提供正确的数据，他们几乎可以做任何事情。但是在大多数情况下，这个**与事实**相去甚远。是的，你可以用一切来分析一切，这样做你会发现一些有趣的东西，但这真的是对你时间的最佳利用吗？

问问你自己，

> 问题是什么？X 如何连接到 Y？为什么？

了解这些将会引导你找到解决问题的好方向。这正是领域专家派上用场的时候。

另一件真正重要的事情是**特色工程**。同一位教授强调指出，您可以使用领域专家进行特征工程过程。如果你花一分钟思考一下，这是有道理的。

# 2.你将花费大部分时间准备数据

是的，你没看错。

我参加数据科学硕士课程的最大原因之一是**机器学习**——我不太关心数据是关于什么的，以及数据是如何收集和准备的。由于这种态度，当学期开始时，我有点震惊和失望。

当然，如果你在一家既有数据工程师又有机器学习工程师的大公司工作，这可能不适用于你，因为你一天的大部分时间都在做机器学习。但如果不是这样，你只会花大约 15%的时间进行机器学习。

这其实很好。机器学习并不那么有趣。在你跳到评论区之前，听我说完。

[https://giphy.com/gifs/breaking-bad-huh-say-what-l41YkFIiBxQdRlMnC](https://giphy.com/gifs/breaking-bad-huh-say-what-l41YkFIiBxQdRlMnC/links)

我认为机器学习并不那么有趣的原因是，对于大多数项目来说，它归结为尝试几种学习算法，然后优化最佳算法。
如果您在此之前没有做好工作，因此在数据准备过程中，您的模型很可能会很糟糕，您对此无能为力，除了调整超参数、调整阈值等。

这就是为什么**数据准备和探索性分析才是王道**，机器学习只是它之后自然而然的事情。

一旦我意识到我已经失去了对机器学习的大部分宣传。我发现我更喜欢数据收集和可视化，因为我在那里学到了最多的数据。

特别是，我真的很喜欢网络抓取，因为相关的数据集很难找到。如果这听起来像是你喜欢东西，请看看这篇文章:

[](/no-dataset-no-problem-scrape-one-yourself-57806dea3cac) [## 没有数据集？没问题。自己刮一个。

### 使用 Python 和 BeautifulSoup 的强大功能来收集对您重要的数据。

towardsdatascience.com](/no-dataset-no-problem-scrape-one-yourself-57806dea3cac) 

# 3.不要多此一举

图书馆的存在是有原因的。**行动之前先谷歌一下**。

我会给你看一个你可能没有犯过的“错误”的小例子，但它会帮助你理解这一点。
关于计算中位数的两种方法。中位数定义为:

> 数字排序列表的中间部分。

因此，要计算它，您必须实现以下逻辑:

*   对输入列表排序
*   检查列表的长度是偶数还是奇数
*   如果是偶数，中位数是两个中间数的平均值
*   如果是奇数，中值就是中间的数字

幸运的是，有像 [Numpy](https://numpy.org/) 这样的库，它可以帮你完成所有繁重的工作。看看下面的代码就知道了，前 17 行指的是自己计算中位数，后两行用 Numpy 的力量达到了同样的效果:

Median calculation — [https://gist.github.com/dradecic/7f295913c01172ffebe84052c8158703](https://gist.github.com/dradecic/7f295913c01172ffebe84052c8158703)

正如我所说的，这只是一个你自己可能没有做过的微不足道的例子。但是想象一下，你写了多少行代码都是徒劳的，因为你没有意识到已经有一个这样的库了。

# 4.掌握 lambdas 和列表理解

虽然不是数据科学的特定内容，但我会说我一直使用 ***列表理解*** 进行特性工程，使用 ***lambda*** 函数进行数据清理和准备。

下面是特征工程的一个简单例子。给定一个字符串列表，如果给定的字符串包含一个问号(？)否则为 0。你可以看到，无论有没有列表理解，你都可以做到这一点( ***提示*** *:它们可以节省大量时间*):

List Comprehension Example — [https://gist.github.com/dradecic/9f23eb0c8073ecc8957f8fd533388cef](https://gist.github.com/dradecic/9f23eb0c8073ecc8957f8fd533388cef)

现在对于 ***lambdas*** ，假设你有一个你不喜欢格式的电话号码列表。基本上你要把' **/** '换成' **-** '。这几乎是一个微不足道的过程，只要你的数据集是*Pandas*data frame 格式:

Lambdas — [https://gist.github.com/dradecic/68e81f6610b26fe8da68e25d217c5052](https://gist.github.com/dradecic/68e81f6610b26fe8da68e25d217c5052)

花点时间想想如何将这些应用到您的数据集。很酷，对吧？

# 5.了解你的统计数据

如果你没有生活在岩石下，你就知道统计学在数据科学中的重要性。这是你必须发展的基本技能。

让我引用 T2 的话:

> 统计用于处理现实世界中的复杂问题，以便数据科学家和分析师可以在数据中寻找有意义的趋势和变化。简而言之，通过对数据进行数学计算，统计可用于从数据中获得有意义的见解。[1]

从我迄今为止从我的硕士课程中学到的关于统计学的知识来看，你有必要了解它，以便能够提出正确的问题。

如果你的统计技能生疏了，我强烈建议你看看 YouTube 上的 [StatQuest](https://www.youtube.com/user/joshstarmer) 频道，更准确地说是基于统计的播放列表:

Statistics Playlist

# 6.学习算法和数据结构

如果你不能提供解决方案，问正确的问题(见第 5 点)就没有意义，对吗？

我因为忽视算法和数据结构而感到内疚，因为我认为只有软件工程师才应该担心这些。至少可以说，我大错特错了。

我并不是说你必须在睡觉的时候知道如何编写二进制搜索算法，但只是一个基本的理解会帮助你更清楚地了解如何用代码思考——因此，如何编写代码来完成工作，而且尽可能快地完成工作。

对于一个没有计算机科学背景的人来说，我强烈推荐这门课程:

[](https://www.udemy.com/course/python-for-data-structures-algorithms-and-interviews/) [## 学习 Python 的数据结构、算法和面试

### 请注意:如果你是一个完全的 PYTHON 初学者，请查看我的另一门课程:完整的 PYTHON BOOTCAMP 来学习…

www.udemy.com](https://www.udemy.com/course/python-for-data-structures-algorithms-and-interviews/) 

此外，确保查看面试问题——它们非常有用！

# 7.超出范围

永远做那个努力工作的人。**有回报**。

至少在我的情况下，我的小组是根据其中一门课的最初表现来评估的。这不是关于谁知道的最多，因为这在第一学期会是一件愚蠢的事情，而是关于谁会表现出职业道德和纪律。

因为那时我没有全职工作，所以我为了这个项目拼命工作。因为我做了，而别人没有，所以我被指派了一个全方位的数据科学项目，这个项目将持续两年，并将为我的硕士论文服务。

是的，我可以把它写进我的简历。

所以，牺牲几周的个人生活值得吗？你自己判断吧，但我会说是的。

喜欢这篇文章吗？成为 [*中等会员*](https://medium.com/@radecicdario/membership) *继续无限制学习。如果你使用下面的链接，我会收到你的一部分会员费，不需要你额外付费。*

[](https://medium.com/@radecicdario/membership) [## 通过我的推荐链接加入 Medium-Dario rade ci

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

medium.com](https://medium.com/@radecicdario/membership) 

# 参考

[1][https://www . edu reka . co/blog/math-and-statistics-for-data-science/](https://www.edureka.co/blog/math-and-statistics-for-data-science/)