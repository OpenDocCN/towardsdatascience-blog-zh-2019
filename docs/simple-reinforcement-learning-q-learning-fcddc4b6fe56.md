# 简单强化学习:Q-学习

> 原文：<https://towardsdatascience.com/simple-reinforcement-learning-q-learning-fcddc4b6fe56?source=collection_archive---------1----------------------->

![](img/43bd5d5e4a6edd0945a443e064a27b20.png)

Typical Exploring Image for RL - Credit [@mike.shots](https://www.instagram.com/mike.shots/)

# **简介**

我在参加强化学习课程时学到的最喜欢的算法之一是 q-learning。可能是因为它对我来说最容易理解和编码，但也因为它似乎有意义。在这篇快速的帖子中，我将讨论 q-learning，并提供理解该算法的基本背景。

# **什么是 q 学习？**

Q-learning 是一种非策略强化学习算法，它寻求在给定的当前状态下采取最佳行动。它被认为是不符合策略的，因为 q-learning 函数从当前策略之外的动作中学习，比如采取随机动作，因此不需要策略。更具体地说，q-learning 寻求学习一种使总回报最大化的策略。

# 什么是‘Q’？

q-learning 中的“q”代表质量。在这种情况下，质量代表了一个给定的行为在获得未来回报方面的有用程度。

# **创建一个 q 表**

当进行 q-learning 时，我们创建一个所谓的 *q-table* 或遵循`[state, action]`形状的矩阵，并将我们的值初始化为零。然后，在一集之后，我们更新并存储我们的 *q 值*。这个 q 表成为我们的代理根据 q 值选择最佳行动的参考表。

```
import numpy as np# Initialize q-table values to 0Q = np.zeros((state_size, action_size))
```

# **Q-学习和更新**

下一步只是让代理与环境交互，并更新我们的 q 表`Q[state, action]`中的状态动作对。

*采取行动:探索或利用*

代理以两种方式之一与环境交互。第一种是使用 q 表作为参考，并查看给定状态的所有可能动作。然后，代理根据这些操作的最大值选择操作。这就是所谓的 ***利用*** ，因为我们利用现有的信息做出决定。

第二种采取行动的方式是随机行动。这叫做 ***探索*** 。我们不是根据最大未来回报来选择行动，而是随机选择行动。随机行动很重要，因为它允许代理探索和发现新的状态，否则在利用过程中可能不会被选择。您可以使用 epsilon ( *ε* )来平衡探索/利用，并设置您希望探索和利用的频率值。这里有一些粗略的代码，将取决于如何设置状态和动作空间。

```
import random# Set the percent you want to explore
epsilon = 0.2if random.uniform(0, 1) < epsilon:
    """
    Explore: select a random action """
else:
    """
    Exploit: select the action with max value (future reward) """
```

*更新 q 表*

更新发生在每个步骤或动作之后，并在一集结束时结束。在这种情况下，完成意味着代理到达某个端点。例如，终端状态可以是任何类似于登录到结帐页面、到达某个游戏的结尾、完成某个期望的目标等的状态。在单个情节之后，代理将不会学到太多，但是最终通过足够的探索(步骤和情节)，它将收敛并学习最优 q 值或 q 星(`Q∗`)。

以下是 3 个基本步骤:

1.  代理从一个状态(s1)开始，采取一个动作(a1)并收到一个奖励(r1)
2.  代理通过随机(epsilon，ε)引用具有最高值(max) **或**的 Q 表来选择动作
3.  更新 q 值

以下是 q-learning 的基本更新规则:

```
# Update q valuesQ[state, action] = Q[state, action] + lr * (reward + gamma * np.max(Q[new_state, :]) — Q[state, action])
```

在上面的更新中，有几个变量我们还没有提到。这里发生的是，我们根据贴现后的新值和旧值之间的差异来调整 q 值。我们使用 gamma 来贴现新值，并使用学习率(lr)来调整步长。以下是一些参考资料。

**学习率:** `lr`或学习率，常被称为*α*或α，可以简单定义为你接受新值相对于旧值的程度。上面我们取新旧之间的差值，然后用这个值乘以学习率。这个值然后被加到我们先前的 q 值上，这实质上使它朝着我们最新更新的方向移动。

**γ:**`gamma`或 *γ* 是贴现因子。它用于平衡当前和未来的奖励。从上面的更新规则中，您可以看到我们将折扣应用于未来的奖励。通常，该值可以在 0.8 到 0.99 的范围内。

**奖励:** `reward`是在给定状态下完成某个动作后获得的数值。奖励可以发生在任何给定的时间步或只在终端时间步。

**Max:** `np.max()`使用 numpy 库，并获取未来奖励的最大值，并将其应用于当前状态的奖励。这是通过未来可能的回报来影响当前的行为。这就是 q-learning 的妙处。我们将未来奖励分配给当前行动，以帮助代理在任何给定状态下选择最高回报的行动。

**结论**

就这样，简短而甜蜜(希望如此)。我们讨论过 q-learning 是一种非策略强化学习算法。我们使用一些基本的 python 语法展示了 q-learning 的基本更新规则，并回顾了算法所需的输入。我们了解到，q-learning 使用未来的奖励来影响给定状态下的当前行为，从而帮助代理选择最佳行为，使总奖励最大化。

关于 q-learning 还有很多，但希望这足以让你开始并有兴趣学习更多。我在下面添加了几个资源，它们对我学习 q-learning 很有帮助。尽情享受吧！

**资源**

1.  使用 [OpenAI Gym taxi](https://www.learndatasci.com/tutorials/reinforcement-q-learning-scratch-python-openai-gym/) 环境的绝佳 RL 和 q 学习示例
2.  [强化学习:导论](http://www.incompleteideas.net/book/RLbook2018trimmed.pdf)(萨顿的免费书籍)
3.  Quora [Q 学习](https://www.quora.com/How-does-Q-learning-work-1)
4.  维基百科 [Q-learning](https://en.wikipedia.org/wiki/Q-learning)
5.  大卫·西尔弗关于 RL 的[讲座](http://www0.cs.ucl.ac.uk/staff/d.silver/web/Teaching.html)