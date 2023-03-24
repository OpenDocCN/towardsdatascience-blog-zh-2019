# 用通俗易懂的语言描述机器学习

> 原文：<https://towardsdatascience.com/machine-learning-in-plain-language-817c88a92dbb?source=collection_archive---------33----------------------->

## 初学者机器学习的简明介绍。

就像生物一样，机器也可以学习做事。在这种情况下，机器是*学生*，而世界或另一个存在——无论是自然的还是人工的——是*老师*。

根据 Tom Mitchell 1997 年的有影响力的书*机器学习*，机器学习有三个主要部分:

*   需要学习的任务。
*   学习任务的观察。
*   任务进展如何。

要学习的任务包括预测知识或行动。这些任务执行得有多好是由预测知识的准确性和行动产生的奖惩决定的。

根据 Mitchell *的说法，当作为学生的机器通过观察更多的世界而更好地完成任务时，机器学习*就会发生。

![](img/ba9225dd65f923f6df6a22bd8acfbb71.png)

[Artificial Intelligence & AI & Machine Learning](https://www.flickr.com/photos/mikemacmarketing/30212411048) by [www.vpnsrus.com](http://www.vpnsrus.com) is licensed under the [Attribution 2.0 Generic (CC BY 2.0)](https://creativecommons.org/licenses/by/2.0/)

# 学习类型

学生可以根据观察到的世界预测新知识。这可能包括对事物进行分类或评估，或者理解行动可能产生的好的或坏的结果。

## 学习预测知识

知识包括对世界当前*状态*的描述和基于该状态的预测知识。状态可以包括多条信息。例如，在电子邮件上注明包括发件人的电子邮件地址和电子邮件的正文。预测的知识将包括基于状态的电子邮件是否是垃圾邮件。

学生可以根据世界或老师给出的完整例子来预测知识。完整的例子是那些包括*前因*和*后果*知识的例子。结果知识跟随先前知识。例如，电子邮件文本的后续知识(先行知识)可能是它被分类为垃圾邮件或不是垃圾邮件。使用这些完整的例子，学生发现模式或一般关系，称为*假设*，可以用来估计未来的后续知识，只有先行知识。这种学习形式被称为*监督学习*。

![](img/89dc1060d9bbdbc533692e8133f55c68.png)

Image via [https://pxhere.com/en/photo/1575601](https://pxhere.com/en/photo/1575601) is licensed under the [CC0 1.0 Universal (CC0 1.0)
Public Domain Dedication](https://creativecommons.org/publicdomain/zero/1.0/)

学生也可以在没有包含结果性知识的例子的情况下预测知识。在这种情况下，学生在观察中发现模式和共性，可以用来估计随后的知识。这叫做*无监督学习。*例如，学生可以有一组对象照片，并根据照片中的共同特征(或状态)将该组对象分类为不同类型的对象。但是类别事先并不知道。这个例子是一个叫做*聚类*的无监督学习的例子。

![](img/b3e2f37e9d777d5b4e84d77dd6865691.png)

[Polaroids](https://www.flickr.com/photos/louisephotography/4649041863) by [Louise McLaren](https://www.flickr.com/photos/louisephotography/) is licensed under the [Attribution 2.0 Generic (CC BY 2.0)](https://creativecommons.org/licenses/by/2.0/)

## 学习行动

学生也可以根据可能得到的奖励或惩罚来采取行动。行动任务是通过所谓的*强化学习*来学习的。这包括学习*状态*(可能是目标)的可取性，以及可能导致状态被实现的行动。学生学习一个*策略*——在某些*状态*(即情况)下要做的动作。

例如，一个学生可以在迷宫中导航，到达终点的奖励与超时的惩罚。学生通过从一个位置移动到另一个位置来行动——从一个状态到另一个状态。期望的最终目标或状态是走出迷宫并获得奖励。

老师给予奖励或惩罚，他可以是另一个存在，也可以是外部世界本身。众生可以发奖；外部世界会带来惩罚，比如下雨或其他自然灾害。学生们学会寻找奖励，避免惩罚，不管它们来自何方。

![](img/f404db67b98b223609f3635eb2a25139.png)

[The hedge maze at Traquair House](https://www.flickr.com/photos/12173006@N08/2937635448) by [marsroverdriver](https://www.flickr.com/people/12173006@N08) is licensed under the [Creative Commons](https://en.wikipedia.org/wiki/en:Creative_Commons) [Attribution-Share Alike 2.0 Generic](https://creativecommons.org/licenses/by-sa/2.0/deed.en) license.

# 推理和知识结构

这三种学习形式从具体的例子中建立更多的一般知识——归纳推理的一种形式。概括储存在记忆中，并在新的情况下用于预测新的知识或行动——一种演绎推理的形式。它们包含两部分知识:前件和后件。例如，来自强化学习的策略具有作为当前情况的前因和作为要做什么的后果。

# 实践中的机器学习:算法

我们可以通过用*机器学习算法*给它们编程来制造学习机器——一系列接收输入并产生输出的指令。例如，监督学习算法将数据作为输入，并输出前件和后件知识之间的关系。这些算法可以从简单的指令系列到更复杂的甚至模仿大脑的指令系列——这些算法被称为*神经网络*。

# 学习是人工智能的一部分

为了有效地发挥作用，学生需要将预测知识和行动结合起来。例如，监督学习可以用于识别国际象棋中的情况——在人工智能中称为*状态*——如处于“检查”状态。但是强化学习被用来找出在那种状态下最有效的行动来获得奖励——在这种情况下，就是赢得游戏。

已经制造出具有人工智能的机器来处理这些形式的学习，但是还没有一种机器能与人类的能力相匹敌。