# 四月版:强化学习

> 原文：<https://towardsdatascience.com/april-edition-reinforcement-learning-badbb1726722?source=collection_archive---------14----------------------->

## 如何建立一个真正的人工智能代理

![](img/046cb2c15b8eb2efd46ff68a54eb6b25.png)

强化学习(RL)是指面向目标的算法，其中“代理”学习完成一个特定的目标或目的，同时在许多步骤中最大化一组“奖励”。RL 代理人是人们在描述电影中描绘的“人工智能”时经常想到的。RL 代理通常从空白开始，在正确的条件下，当它了解其环境时，可以实现惊人的性能。

当 RL 算法做出错误的决定时，它们会受到惩罚，而当它们做出正确的决定时，它们会受到奖励，因此出现了术语“强化学习”。建立环境，选择合适的算法/策略，设计奖励函数，为实现代理的预期行为提供激励——这些都是使强化学习成为一个迷人而复杂的领域的一些方面。

强化学习的应用在[多个行业](/applications-of-reinforcement-learning-in-real-world-1a94955bcd12)各不相同，包括[机器人](/robotic-control-with-graph-networks-f1b8d22b8c86)、[聊天机器人](/training-a-goal-oriented-chatbot-with-deep-reinforcement-learning-part-i-introduction-and-dce3af21d383)和[自动驾驶汽车](/learning-to-drive-smoothly-in-minutes-450a7cdb35f4)。大多数强化学习的新生都是通过使用 OpenAI Gym 框架来了解 RL 的。TDS 作者，[霍纳塔斯·菲格雷多](https://towardsdatascience.com/@jonathas.mpf)和[维哈尔·鞍马](https://towardsdatascience.com/@vihar.kurama)写了关于 [OpenAI](/openai-gym-from-scratch-619e39af121f) 和 [RL 的优秀文章，让你开始使用 Python](/reinforcement-learning-with-python-8ef0242a2fa2) 。一旦你在 RL 方面打下了基础，那么看看更复杂的应用程序，比如[训练代理打网球](/training-two-agents-to-play-tennis-8285ebfaec5f)或者[使用 RL 训练聊天机器人](/training-a-goal-oriented-chatbot-with-deep-reinforcement-learning-part-i-introduction-and-dce3af21d383)。

— [Hamza Bendemra](https://medium.com/u/1ede83301f25?source=post_page-----badbb1726722--------------------------------) ，编辑助理，致力于数据科学。

## [强化学习在现实世界中的应用](/applications-of-reinforcement-learning-in-real-world-1a94955bcd12)

由 [Garychl](https://medium.com/u/e75d60a48e51?source=post_page-----badbb1726722--------------------------------) — 13 分钟读完

虽然卷积神经网络(CNN)和递归神经网络(RNN)因其在计算机视觉(CV)和自然语言处理(NLP)中的应用而对企业变得越来越重要，但强化学习(RL)作为计算神经科学的框架来模拟决策过程似乎被低估了。

## [OpenAI 健身房从零开始](/openai-gym-from-scratch-619e39af121f)

到[霍纳塔斯·菲格雷多](https://medium.com/u/e4becf60c47f?source=post_page-----badbb1726722--------------------------------) — 10 分钟读完

有很多工作和教程解释了如何使用 OpenAI Gym toolkit，以及如何使用 Keras 和 TensorFlow 来训练使用一些现有 OpenAI Gym 结构的现有环境。然而在本教程中，我将解释如何从头开始创建一个 OpenAI 环境，并在其上训练一个代理。

## [用 Python 进行强化学习](/reinforcement-learning-with-python-8ef0242a2fa2)

由维哈尔·鞍马 — 11 分钟阅读

强化是一类机器学习，其中代理通过执行动作来学习如何在环境中行为，从而得出直觉并看到结果。在本文中，您将学习理解和设计一个强化学习问题，并用 Python 来解决。

## [训练机器人打网球](/training-two-agents-to-play-tennis-8285ebfaec5f)

托马斯·特雷西 — 16 分钟阅读

这篇文章探索了我在 Udacity 的深度强化学习纳米学位的[最终项目](https://github.com/tommytracey/DeepRL-P3-Collaboration-Competition)中的工作。我的目标是帮助其他机器学习(ML)学生和专业人士，他们正处于在强化学习(RL)中建立直觉的早期阶段。

## [几分钟学会平稳驾驶](/learning-to-drive-smoothly-in-minutes-450a7cdb35f4)

由安东宁·拉芬 — 11 分钟阅读

在这篇文章中，我们将看到如何在几分钟内训练一辆自动驾驶赛车，以及如何平稳地控制它。这种基于强化学习(RL)的方法，在这里的模拟(驴车模拟器)中提出，被设计成适用于现实世界。它建立在一家名为 [Wayve.ai](https://wayve.ai/) 的专注于自动驾驶的初创公司的工作基础上。

## [用图形网络控制机器人](/robotic-control-with-graph-networks-f1b8d22b8c86)

通过[或 Rivlin](https://medium.com/u/d6ea8553654c?source=post_page-----badbb1726722--------------------------------) — 9 分钟读取

正如任何对技术感兴趣的人无疑都知道的那样，机器学习正在帮助改变不同行业的许多领域。在过去的几年里，由于深度学习算法，计算机视觉和自然语言处理等事情发生了巨大的变化，这种变化的影响正在渗透到我们的日常生活中。

## 用深度强化学习训练一个目标导向的聊天机器人([第一部分](/training-a-goal-oriented-chatbot-with-deep-reinforcement-learning-part-i-introduction-and-dce3af21d383)、[第二部分](https://medium.com/@maxbrenner110/training-a-goal-oriented-chatbot-with-deep-reinforcement-learning-part-ii-dqn-agent-f84122cc995c)、[第三部分](https://medium.com/@maxbrenner110/training-a-goal-oriented-chatbot-with-deep-reinforcement-learning-part-iii-dialogue-state-d29c2828ce2a)、[第四部分](https://medium.com/@maxbrenner110/training-a-goal-oriented-chatbot-with-deep-reinforcement-learning-part-iv-user-simulator-and-a0efd3829364)、[第五部分](https://medium.com/@maxbrenner110/training-a-goal-oriented-chatbot-with-deep-reinforcement-learning-part-v-running-the-agent-and-63d8cd27d1d))

通过[最大布伦纳](https://medium.com/u/e83c3988e008?source=post_page-----badbb1726722--------------------------------) — 10 分钟读取

在这个系列中，我们将学习面向目标的聊天机器人，并用 python 训练一个具有深度强化学习的聊天机器人！一切从零开始！这个系列教程的代码可以在这里找到。

我们也感谢最近加入我们的所有伟大的新作家，[马科斯·特雷维索](https://medium.com/u/664d4f028ac1?source=post_page-----badbb1726722--------------------------------)，[大卫·康弗](https://medium.com/u/e6cc63aba9b8?source=post_page-----badbb1726722--------------------------------)，[让·克利斯朵夫·b·卢瓦索](https://medium.com/u/147ab927857?source=post_page-----badbb1726722--------------------------------)，[莫里茨·基尔希特](https://medium.com/u/e9624bd27418?source=post_page-----badbb1726722--------------------------------)，[卡尔·温梅斯特](https://medium.com/u/56d387899b8b?source=post_page-----badbb1726722--------------------------------)，[卡梅隆·布朗斯坦](https://medium.com/u/e235be22f216?source=post_page-----badbb1726722--------------------------------)，[阿曼·迪普](https://medium.com/u/2f104346e4c3?source=post_page-----badbb1726722--------------------------------)，[雅各布·戴维斯](https://medium.com/u/ecf6f07b3a6a?source=post_page-----badbb1726722--------------------------------)，[詹姆斯·迪特尔](https://medium.com/u/557cc66a323a?source=post_page-----badbb1726722--------------------------------)，[扎伊我们邀请你看看他们的简介，看看他们的工作。](https://medium.com/u/e4c8fc17bd10?source=post_page-----badbb1726722--------------------------------)