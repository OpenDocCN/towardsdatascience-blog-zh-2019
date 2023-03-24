# 无梯度强化学习:用遗传算法进化智能体

> 原文：<https://towardsdatascience.com/reinforcement-learning-without-gradients-evolving-agents-using-genetic-algorithms-8685817d84f?source=collection_archive---------7----------------------->

在假期里，我想提高我的强化学习技能。对这个领域一无所知，我上了一门[课程](https://www.udemy.com/reinforcement-learning-with-pytorch/)，在那里我接触到了 Q-learning 和它的“深度”等价物(深度 Q 学习)。那是我接触 OpenAI 的[健身房](https://gym.openai.com/)的地方，那里有几个环境供代理人玩耍和学习。

课程仅限于深度 Q 学习，所以我自己读了更多。我意识到现在有更好的算法，如政策梯度及其变体(如行动者-批评家方法)。如果这是你第一次参加强化学习，我推荐以下资源，它们对建立良好的直觉很有帮助:

*   安德烈·卡帕西的[深度强化学习:来自像素的 Pong】。这是一个经典的教程，引发了人们对强化学习的广泛兴趣。**必读**。](http://karpathy.github.io/2016/05/31/rl/)
*   [通过 Q-Learning](https://medium.freecodecamp.org/diving-deeper-into-reinforcement-learning-with-q-learning-c18d0db58efe) 深入强化学习。这篇文章对最基本的 RL 算法 : Q-learning 做了一个很好的、有插图的概述。
*   [各种强化学习算法介绍](/introduction-to-various-reinforcement-learning-algorithms-i-q-learning-sarsa-dqn-ddpg-72a5e0cb6287)。RL 算法世界的**漫游**。(有趣的是，我们在这篇文章中将要讨论的算法——遗传算法——不在列表中。
*   [直觉 RL:优势介绍——演员——评论家(A2C)](https://hackernoon.com/intuitive-rl-intro-to-advantage-actor-critic-a2c-4ff545978752) 。**一个非常非常好的漫画**(是的，你没听错)关于 RL 目前最先进的算法。

我感到很幸运，作为一个社区，我们分享了这么多，以至于在对强化学习一无所知的几天内，我能够复制仅仅 3 年前的艺术状态:[使用像素数据玩雅达利游戏。](https://www.nature.com/articles/nature14236)这里有一个我的代理人(绿色)只用像素数据和 AI 玩 Pong 的快速视频。

![](img/6b0658af1ad14a33cf726f13e4e9cdcf.png)

This sort of feels like a personal achievement!

# 强化学习的问题是

我的代理使用策略梯度算法进行了很好的训练，但这需要在我的笔记本电脑上进行整整 2 天 2 夜的训练。即使在那之后，它也没有完美地播放。

我知道总的来说，两天并不算多，但我很好奇为什么强化学习的训练如此缓慢。通过阅读和思考，我意识到**强化学习缓慢的原因是因为梯度(几乎)不存在，因此不是很有用**。

梯度有助于监督学习任务，如图像分类，它提供了关于如何改变网络参数(权重、偏差)以获得更高精度的有用信息。

![](img/1ed097dffd25e47d6f2489047233f185.png)

Imagine this surface representing error for different combinations of weights and biases (the lower the better). Starting from a randomly initialized point (of weights and biases), we want to find the values that minimize the error (the lowest point). Gradients at each point represent the direction of downhill, so finding the lowest point is equivalent to following the gradient. (Wait, did I just describe the [stochastic gradient descent](https://medium.com/@hakobavjyan/stochastic-gradient-descent-sgd-10ce70fea389) algo?)

在图像分类中，在每次小批量训练之后，反向传播为网络中的每个参数提供了清晰的梯度(方向)。然而，在强化学习中，梯度信息只是在环境给予奖励(或惩罚)时偶尔出现。大多数时候，我们的代理人在采取行动时并不知道这些行动是否有用。梯度信息的缺乏使得我们的误差范围看起来像这样:

![](img/5fb3e9dde42af2a6a0243eb135b125bf.png)

Image via the excellent blog post [Expressivity, Trainability, and Generalization in Machine Learning](https://blog.evjang.com/2017/11/exp-train-gen.html?spref=tw)

奶酪的表面代表我们的代理人网络的参数，无论代理人做什么，环境都不会给予奖励，因此没有梯度(即，因为没有错误/奖励信号，我们不知道下一次在哪个方向改变参数以获得更好的性能)。表面上的几个孔代表对应于与表现良好的代理相对应的参数的低误差/高回报。

你现在看到政策梯度的问题了吗？一个随机初始化的代理很可能位于平面上(而不是一个洞)。如果一个随机初始化的代理在一个平坦的表面上，它很难达到更好的性能，因为没有梯度信息。**因为(错误)表面是平的，一个随机初始化的代理或多或少做了一次随机游走，并在很长一段时间内被坏策略所困**。(这就是为什么我花了几天时间来训练一个代理。*提示*:也许政策梯度法不比随机搜索好？)

正如标题为[为什么 RL 有缺陷](https://thegradient.pub/why-rl-is-flawed/)的文章明确指出的:

> 如果你想学的棋盘游戏是围棋，你会如何开始学习？你会阅读规则，学习一些高水平的策略，回忆你过去如何玩类似的游戏，得到一些建议…对吗？事实上，这至少部分是因为 AlphaGo Zero 和 OpenAI 的 Dota 机器人的从头学习限制，与人类学习相比，它并不真正令人印象深刻:它们依赖于比任何人都多得多的游戏数量级和使用更多的原始计算能力。

我认为这篇文章切中要害。RL 是低效的，因为它没有告诉代理它应该做什么。不知道该做什么的代理开始做随机的事情，只是偶尔环境会给出奖励，现在代理必须弄清楚它采取的数千种行动中的哪一种导致了环境给出奖励。人类不是这样学习的！我们被告知需要做什么，我们发展技能，奖励在我们的学习中扮演相对次要的角色。

![](img/9e9ec6bcd69bf0ebc919cd9e4e46542a.png)

If we trained children through policy gradients, they’ll always be confused about what they did wrong and never learn anything. (Photo via [Pixabay](https://pixabay.com/en/portrait-child-hands-hide-317041/))

# 强化学习的无梯度方法

当我在探索基于梯度的 RL 方法的替代方法时，我偶然发现了一篇题为[深度神经进化:遗传算法是训练用于强化学习的深度神经网络的有竞争力的替代方法](https://arxiv.org/abs/1712.06567)。这篇论文结合了我正在学习的东西(强化学习)和我一直很兴奋的东西(进化计算)，所以我着手实现论文中的算法并进化出一个代理。

请注意，严格来说，我们甚至不必实现遗传算法。如上所述，同一篇论文发现**甚至完全随机的搜索也能发现好的代理商**。这意味着，即使你继续随机生成代理，最终，你会找到一个表现良好的代理(有时比策略梯度更快)。我知道，这很疯狂，但这只是说明了我们最初的观点，即 RL 从根本上是有缺陷的，因为它几乎没有什么信息可供我们用来训练算法。

## 什么是遗传算法？

这个名字听起来很花哨，但实际上，这可能是你能设计出的探索风景的最简单的算法。考虑一个通过神经网络实现的环境中的代理(如 Pong)。它获取输入层中的像素，并输出可用动作的概率(向上、向下或不做任何动作)。

![](img/f83d0a719a21a4b5dcd6d8963227036c.png)

Image via [Deep Reinforcement Learning: Pong from Pixels](http://karpathy.github.io/2016/05/31/rl/)

我们在强化学习中的任务是找到神经网络(权重和偏差)的参数(权重和偏差)，这些参数使代理更经常获胜，从而获得更多奖励。到目前为止还好吗？

**遗传算法的伪代码**

*   简单地把代理想象成一个有机体，
*   参数将是指定其行为(策略)的基因
*   奖励将指示有机体的适应性(即，奖励越高，生存的可能性越高)
*   在第一次迭代中，您从带有随机初始化参数的 *X* 个代理开始
*   他们中的一些人可能会比其他人表现得更好
*   就像现实世界中的自然进化一样，你实现了*适者生存:*简单地选取最适的 10%的代理，并为下一次迭代复制它们，直到下一次迭代又有了 *X* 个代理。击杀最弱的 90%(如果这听起来很病态，可以把击杀功能改名为 give- [木沙](https://en.wikipedia.org/wiki/Moksha)！)
*   在复制前 10%最适合的代理的过程中，向其参数添加一个微小的随机高斯噪声，以便在下一次迭代中，您可以探索最佳代理参数周围的邻域
*   让表现最好的代理*保持原样*(不添加噪声)，这样你就可以一直保留最好的，以防止高斯噪声导致性能下降

就是这样。你已经理解了遗传算法的核心。遗传算法有许多(奇特的)变种，其中两个代理之间有各种各样的(肮脏的)性(性重组),但关于深度神经进化的论文用上面的伪代码实现了香草遗传算法，这也是我在代码中实现的。(你可以在我的 Github 库上用代码访问[Jupyter 笔记本)。](https://github.com/paraschopra/deepneuroevolution)

![](img/6ec8d732a167d14c02caf10dc208c44e.png)

Yellower regions are regions with lower error (higher rewards/performance). Blue dots are all agents. Green ones are the top 10% and the red dot is the best of the best. Notice how gradually the entire population moves towards the region with the lowest error. (Image via [Visual Guide to Evolutionary Strategies](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/))

关于进化算法如何工作的更多可视化，我强烈推荐阅读这篇做得非常好的帖子:[进化策略可视化指南](http://blog.otoro.net/2017/10/29/visual-evolution-strategies/)。

## 用于实现强化学习的深度神经进化的代码

我 t̸r̸a̸i̸n̸e̸d̸进化出了一个移动小车上的代理平衡杆(又名 [CartPole-v0](https://gym.openai.com/envs/CartPole-v0/) )。下面是完整的代码:[https://github.com/paraschopra/deepneuroevolution](https://github.com/paraschopra/deepneuroevolution)

使用 PyTorch，我们通过 2 个隐藏层的神经网络(希望保留“深层”部分:)对代理进行参数化，对于 CartPole，一个层的网络也可以做得很好。

这是进化的主循环:

这段代码基本上是不言自明的，并且遵循了我在本文前面写的伪代码。将细节映射到伪代码:

*   我们的人口规模是 500 ( *num_agents* )，我们在第一次迭代中随机生成代理( *return_random_agents* )
*   从 500 个中，我们只选择前 20 个作为父代( *top_limit* )
*   我们想要运行循环的最大迭代次数是 1000 次(*代*)。尽管通常对于 CartPole 来说，在几次迭代中就能找到一个表现良好的代理。
*   在每一代中，我们首先运行所有随机生成的代理，并在 3 次运行中获得它们的平均性能(曾经可能是幸运的，所以我们想要平均)。这是通过 *run_agents_n_times* 函数完成的。
*   我们按照奖励(绩效)的降序对代理进行排序。排序后的索引存储在 *sorted_parent_indexes* 中。
*   然后，我们选择前 20 名代理，并在其中随机选择，为下一次迭代生成子代理。这发生在 *return_children* 函数中(见下文)。该函数在复制代理时向所有参数添加一个小的高斯噪声，但保留一个最佳的精英代理(不添加任何噪声)。
*   现在将子代理作为父代理，我们再次迭代并运行整个循环，直到完成所有 1000 代或者我们找到具有良好性能的代理(在 Jupyter 笔记本中，我打印了前 5 名代理的平均性能。当它足够好时，我手动中断循环)

***return_children*** 函数是这样的:

您可以看到，首先，它从前 20 个代理中选择一个随机代理(索引包含在 *sorted_parents_indexes* 中)，然后调用 *mutate* 函数添加随机高斯噪声，然后将其添加到 *children_agents* 列表中。在第二部分中，它调用 *add_elite* 函数将最佳代理添加到 *children_agents* 列表中。

下面是 ***变异*** 的功能:

你可以看到，我们迭代所有参数，并简单地添加高斯噪声(通过 *np.random.randn()* )乘以一个常数( *mutation_power* )。乘法因子是一个超参数，大致类似于梯度下降中的学习速率。(顺便说一下，这种遍历所有参数的方法速度慢，效率低。如果你知道 PyTorch 中更快的方法，请在评论中告诉我)。

最后，函数 ***add_elite*** 如下:

这段代码简单地选取前 10 名代理并运行 5 次，以根据平均表现双重确定哪一个是精英(通过 *return_average_score* )。然后它通过 *copy.deepcopy* 复制精英，并原样返回(无突变)。如前所述，精英确保我们总是有一个我们以前最好的副本，它防止我们随机漂移(通过突变)到一个没有好代理的区域。

**就是这样！让我们看看进化后的钢管舞特工的表现。**

![](img/614af501a127a09e9c128a5089800c1e.png)

You, Sir, are a product of evolution.

遗传算法并不完美。例如，在添加高斯噪声时，没有关于如何选择乘法因子的指导。你只需要尝试一堆数字，看看哪一个有效。而且，在很多情况下，可能会彻底失败。我多次尝试为 Pong 开发一个代理，但它非常慢，我放弃了。

我联系了深度神经进化论文的作者，[费利佩这样的](https://twitter.com/felipesuch?lang=en)。他提到，对于一些游戏(包括 Pong)来说，神经进化失败了，但对于其他游戏(如 [Venture](https://en.wikipedia.org/wiki/Venture_(video_game)) )来说，它比政策梯度要快得多。

## 你会进化成什么样？

我的知识库中用于深度神经进化的[代码足够通用，可以与 PyTorch 中实现的任何神经网络一起工作。我鼓励你尝试各种不同的任务和架构。如果你成功了，失败了或者卡住了，请告诉我！](https://github.com/paraschopra/deepneuroevolution)

祝你好运进化你自己的世界。

*PS:查看我以前的实践教程 a)* [*从小语料库中生成哲学*](/generating-new-ideas-for-machine-learning-projects-through-machine-learning-ce3fee50ec2) *和 b)* [*贝叶斯神经网络*](/making-your-neural-network-say-i-dont-know-bayesian-nns-using-pyro-and-pytorch-b1c24e6ab8cd)

感谢 Nirant Kasliwal 和 Felipe 审阅这篇文章的草稿。

## 在 Twitter 上关注我

我定期发关于人工智能、深度学习、创业公司、科学和哲学的推特。跟着我上[https://twitter.com/paraschopra](https://twitter.com/paraschopra)

[](https://twitter.com/paraschopra) [## Paras Chopra (@paraschopra) |推特

### Paras Chopra 的最新推文(@paraschopra)。@Wingify |的创始人兼董事长写道…

twitter.com](https://twitter.com/paraschopra)