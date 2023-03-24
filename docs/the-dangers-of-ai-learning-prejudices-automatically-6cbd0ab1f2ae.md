# 人工智能自动学习偏见的危险

> 原文：<https://towardsdatascience.com/the-dangers-of-ai-learning-prejudices-automatically-6cbd0ab1f2ae?source=collection_archive---------28----------------------->

阅读新闻，你会读到机器学习和人工智能领域取得的巨大成功和发展。但是仔细研究一下，你会发现数据科学已经发现了一些过去的错误。

![](img/cdb074162dc6386848a29d47edab3ae4.png)

Photo by [Trym Nilsen](https://unsplash.com/@trymon?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 机器学习中的习得性偏差

我认为这将是一个很好的报道领域，因为最近的 [BBC 新闻文章](https://www.bbc.co.uk/news/technology-48222017)强调，由于英国警方采用的面部识别软件是针对主要是欧洲白人的，该系统将遭受更高的错误率，任何人都不在这个群体之内。[的文章](https://www.bbc.co.uk/news/technology-48222017)强调了这可能会导致黑人和少数民族更高的误判率。

可悲的是，这不是一次性的案例，如果你在新闻中搜索一下，你会发现几篇谈论机器学习偏见问题的文章。

*   [2013 年](https://www.theguardian.com/news/datablog/2013/aug/14/problem-with-algorithms-magnifying-misbehaviour)一所医学院在第一轮面试选拔中使用之前的申请结果来给简历打分。后来的研究表明，对于非欧洲发音的名字和女性来说，这种现象会有所减弱。
*   2018 年[亚马逊](https://phys.org/news/2018-11-amazon-sexist-hiring-algorithm-human.html)因明显偏向男性而关闭了自动招聘。

事实上，算法中明显的性别歧视和种族歧视是一些人争论了很久的事情(特别是在医学领域，这里是[这里是](https://qz.com/1367177/if-ai-is-going-to-be-the-worlds-doctor-it-needs-better-textbooks/)和[这里是](https://journalofethics.ama-assn.org/article/can-ai-help-reduce-disparities-general-medical-and-mental-health-care/2019-02)),归结为数据本身(有时是算法设计者)包含算法发现的偏见。对于上述许多技术和高等医学领域，男性占主导地位，该算法将其视为成功的有力指标并加以利用(当它对过去的数据进行预测时，通过使用它，它得出了很好的准确性，这只会加强它)。

# 为什么这很重要？

出于任何伦理或道德原因，我们都不希望这种行为成为首要原因。这对数据科学的健康和地位也很重要，因为它的一个被吹捧的优势是，它将人从等式中移除，从而做出无偏见的卓越决策。然而，如果结果是我们在教它我们自己的偏见，并潜在地[放大它们](https://medium.com/@laurahelendouglas/ai-is-not-just-learning-our-biases-it-is-amplifying-them-4d0dee75931d)。

如果人们对机器学习失去信心，开始不信任它，那么整个领域的未来都会因此受损。

> 人比机器更能原谅人的错误。

# 你能做什么来阻止它？

你能做什么？这很大程度上取决于观察你的算法随着时间的推移表现如何(它是否正确工作？)并检查你得到的数据中可能包含的潜在偏见。你本质上追求的是一个[代表性样本](https://en.wikipedia.org/wiki/Sampling_%28statistics%29)，不是老的有偏见的人口，而是你想要的人口。因此，举例来说，如果你看一下旧的录取数据，就会发现它明显偏向男性，因此，当时被录取的申请人中就有偏见。然而，对于未来，你不希望这样，所以历史数据需要纠正。

校正的基本方法是对有偏见的群体的数据进行加权，删除算法可能使用的特征(例如性别),或者您甚至可以对有偏见的群体进行降采样，以便获得相等的数量。也有一些更先进的方法，一些研究人员开发了保留性别信息但消除了[偏差的方法。](https://venturebeat.com/2018/09/07/researchers-develop-a-method-that-reduces-gender-bias-in-ai-datasets/)

# 拿走

> 数据是解决您的问题的正确数据吗？是否包含不良偏见？

然而，随着越来越多的地方提供[伦理课程](https://www.edx.org/course/ethics-and-law-in-data-and-analytics-2)，机器学习带来了更多关于机器学习如果使用不当可能造成的损害的意识，以及关于如何纠正机器学习的[建议](https://www.ft.com/content/d2a1ab08-f63e-11e7-a4c9-bbdefa4f210b)，这并不完全是悲观的，但也有一些论点认为，即使有偏见的模型仍然可能比他们取代的有偏见的人[更好](https://phys.org/news/2018-11-amazon-sexist-hiring-algorithm-human.html)，即使它仍然不可取。我将让你决定那一个。