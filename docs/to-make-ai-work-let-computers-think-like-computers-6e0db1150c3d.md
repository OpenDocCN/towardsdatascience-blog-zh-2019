# 让人工智能发挥作用——让计算机像计算机一样思考

> 原文：<https://towardsdatascience.com/to-make-ai-work-let-computers-think-like-computers-6e0db1150c3d?source=collection_archive---------31----------------------->

人工智能研究人员渴望复制人类思维。但我们已经看到，人工智能的真正关键是让机器以自己的方式超越他人。

由(www.percepto.co)[Percepto](http://www.percepto.co)的首席技术官 Sagi Blonder 提供

人工智能(AI)这个词在今天的科技(和日常生活)世界中是不可避免的。高端、最先进的人工智能应用确实正在改变企业[分析数据](https://www.cio.com/article/3342421/enterprise-ai-data-analytics-data-science-and-machine-learning.html)的方式，或者[工业产品](https://www.newequipment.com/research-and-development/what-generative-design-and-why-its-future-manufacturing)的设计和生产方式。

也就是说，什么是人工智能，什么不是人工智能可能会令人困惑。对人工智能应该如何工作的根本误解可能会阻碍真正自主机器的进展。原因？我们需要学会区分人工智能和人工智能，前者依靠人类来准确定义机器应该如何学习，后者由人类向机器提出问题，然后让机器根据自己的经验和能力来解决这个问题。

例如，让谷歌助手[开玩笑](https://www.digitaltrends.com/mobile/funny-things-to-ask-google-assistant/)或帮助 Spotify 和网飞学习你喜欢和不喜欢什么的人工智能就是前者的一个例子。这是机器学习，人工智能的一个分支，但本质上是一种超级聪明的方法，让机器编程来执行任务。

人类驱动的人工智能的另一个例子是[神经网络](https://en.wikipedia.org/wiki/Neural_network)——其操作模仿人脑。这些技术正被用来创造惊人的专家机器，这些机器被训练来完成一项单一的任务，并且成功的水平不断提高。例如，[最近的一项研究](https://www.aidoc.com/blog/clinical_study/detection-of-intracranial-haemorrhage-on-ct-of-the-brain-using-a-deep-learning-algorithm/)显示，一种基于神经网络的算法在脑部 CT 图像中对颅内出血进行分类的成功率相当于一名四年的放射科住院医师。这些解决方案非常可靠，FDA 已经批准使用图像分析算法来帮助诊断，甚至自动确定病人护理的优先顺序。

然而，这些机器仍然按照我们教它们的方式做我们教它们做的事情。

这就是问题的要点。因为把一个能产生艺术的机器和一个有创造力的机器，或者一个知道因果的机器和一个有[常识](https://www.paulallen.com/ai-common-sense-project-alexandria/)的机器混为一谈是很危险的。*只有当我们让机器像机器一样思考，而不是像制造它们的人一样思考，真正的机器自主才能实现。*

**这看起来怎么样？**

举个最近的例子，想想谷歌子公司 DeepMind 开发的国际象棋算法 [AlphaZero](https://en.wikipedia.org/wiki/AlphaZero) 。为了探索国际象棋本身的复杂性，AlphaZero 开发了罕见的[棋步——以一种完全独特的风格下棋，并持续获胜。AlphaZero 的兄弟](https://www.nytimes.com/2018/12/26/science/chess-artificial-intelligence.html) [AlphaGo](https://www.scmp.com/tech/enterprises/article/2095929/alphago-vanquishes-worlds-top-go-player-marking-ais-superiority) 掌握了古老而复杂得可笑的围棋——采用了一种独创的策略和看不见的战术，让世界知名的大师们瞠目结舌(并一直被打败)。

在更实际的层面上，自动驾驶汽车的出现提出了有趣的人工智能挑战。考虑“驾驶员直觉”的问题。有经验的人类司机通常能够预测汽车何时要变道，即使在转向灯激活之前。我们人类能够利用我们的经验来直觉行为，这些行为可能在物理上并不明显。我们是如此无意识地这样做，以至于我们中的许多人无法用语言准确地表达我们是如何知道我们所知道的——我们只是知道而已。

这种直觉——可以说对安全驾驶和其他危险的现实世界任务至关重要——可以被分解并教给机器吗？可以用来历不明的信息注释一个图像吗？解决这个问题的一个常见方法是使用强化学习——通过改进自己的技术来训练机器达到某个目标，而不是通过从其他人那里学习样本。然而，通过强化的方式训练自动驾驶汽车也有其自身的挑战。一个是模拟质量的内在限制，因为我们无法真正让自动驾驶汽车在真实的道路上学习如何驾驶。给汽车的感官数据加上标签，描述道路上还没有发生的事情，又怎么样呢？我们如何标记无关的数据——例如，传感器看不到的汽车，隐藏在卡车后面或框架外的汽车。而没有这个出发点，机器又怎么能学会寻找汽车存在的暗示呢？

简单的答案是*我们无法教会一台机器* *直觉*。我们可能永远也不会制造出像人类一样思考的机器，这也不是我们必须追求的目标。在实验室和现实世界中已经证明了自己的是，让机器以它能理解的方式理解给定的任务——让机器利用它们的优势来补偿不是人类。

例如，[一项新的研究](https://ai.googleblog.com/2018/04/seeing-more-with-in-silico-labeling-of.html)表明，一台经过训练的细胞分类机器学会了在不添加荧光标签的情况下做到这一点，荧光标签用于使细胞特征对人眼来说显而易见。通过在添加侵入材料之前拍摄细胞图像，并用后验知识进行标记，机器找到了一种检测人类无法检测到的东西的方法。澄清一下，*没人教机器怎么做这个。它看着数据和自己的能力，无视建造它的人类的限制，独自解决问题。*

**底线**

当我们放下我们狭隘的学习和理解概念，接受不同的有机体有不同的运作方式时，真正的自主机器就能实现。鉴于今天的技术限制(仍然有很多)，当我们放弃生产像人一样思考的机器的目标，让机器成为机器时，我们将在机器自主方面取得更有效的结果。