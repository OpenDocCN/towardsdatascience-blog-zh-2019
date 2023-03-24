# 当自动化反噬时

> 原文：<https://towardsdatascience.com/when-automation-bites-back-8541b061d5ee?source=collection_archive---------21----------------------->

## 不诚实的自动化业务以及背后的工程师、数据科学家和设计师如何修复它

![](img/206bde5d6b18a2accd1a567e362f3c00.png)

Photo courtesy of [Nicolas Nova](https://www.flickr.com/photos/nnova/13734129414).

“飞行员持续战斗，直到飞行结束”，调查 2018 年 10 月 29 日坠毁的狮航 610 航班的负责人 Nurcahyo Utomo 上尉说，机上 189 人遇难。对黑匣子的分析显示，这架波音 737 飞机的机头被反复压下，显然是由接收到错误传感器读数的自动系统造成的。在悲剧发生前的 10 分钟内，飞行员曾 24 次尝试手动拉起机头。他们与故障的防失速系统进行了斗争，他们不知道如何为特定版本的飞机解除该系统。

这种人类与顽固的自动化系统斗争的戏剧性场景属于流行文化。在 1968 年的科幻电影《2001:太空漫游》的著名场景中，宇航员戴夫要求哈尔(启发式编程算法计算机)打开飞船上的舱门，哈尔反复回应，“*对不起，戴夫，恐怕我做不到*。

# 1.自动化的商品化

令人欣慰的是，数字自动化的当代应用是局部的，并没有像哈尔那样采取“人工一般智能”的形式。然而，曾经专门应用于驾驶舱等关键环境中自动化人类工作的计算任务已经进入人们的日常生活(例如，自动寻路、智能恒温器)，这些技术通常用于更琐碎但非常有利可图的目标(例如，定向广告，优先选择 YouTube 上的下一个视频)。

> “令我担忧的是，许多工程师、数据科学家、设计师和决策者将数字摩擦带入了人们的日常生活，因为他们没有采用方法来预见他们工作的限制和影响”

曾经依赖基于作者对世界的理解的编程指令的自动化系统，现在也从传感器和人类活动的数据集中找到的模式来模拟它们的行为。随着这些机器学习技术的应用变得广泛，数字自动化正成为一种商品，其系统在互联网规模上执行一项任务，而无需深入理解人类环境。这些系统被训练来完成“一个”工作，但有证据表明，当事情没有按预期发展时，它们的行为，如 HAL 或波音 737 防失速系统，可能会违背用户的意图。

# 2.笨拙的边缘

在不久的将来的实验室，像 [#TUXSAX](http://tuxsax.nearfuturelaboratory.com/) 和[好奇的仪式](http://curiousrituals.nearfuturelaboratory.com/)最近的视觉民族志揭示了自动化商品化的一些含义。在导致狮航 610 航班坠毁的完全不同规模的戏剧性后果中，这些观察强调了一些数字解决方案如何让人们有一种被“锁定”的感觉，没有从顽固行为中脱离的“退出”键。这些数字摩擦中的绝大多数都会在人们的日常生活中引发无害的微挫折。它们通过校准不良的系统和忽视边缘情况的设计表现出来。例如，经常会遇到语音助手无法理解某种口音或发音，或者导航系统由于位置不准确、过时的道路数据或不正确的交通信息而误导驾驶员。

Curious rituals is a fiction that showcases the gaps and junctures that glossy corporate videos on the “future of technology” do not reveal. Source: [Curious Rituals](http://curiousrituals.nearfuturelaboratory.com/¨).

这些笨拙的自动化可以减轻，但不会消失，因为不可能为所有意想不到的限制或后果设计应急计划。然而，其他类型的顽固自主行为被有意设计为商业模式的核心，以人类控制换取便利。

# 3.不诚实的自动化行业

许多自动化日常任务的技术使组织能够降低成本并增加收入。科技行业的一些成员利用这些新的技术能力将客户或员工锁定在他们没有合法需求或愿望的行为中。这些系统通常被设计成抵制用户的需求，并且很难脱离。让我给你举几个我称之为“不诚实的自动性”的例子:

## 3.1.数据肥胖

自动云备份系统已经成为操作系统的默认功能。他们将个人照片、电子邮件、联系人和其他数字生活的存储具体化。他们的商业模式鼓励客户无休止地积累更多内容，而没有一个明确的替代方案来促进数据的适当卫生(即，还没有人提出“Marie Kondo for Dropbox”)。不管提供商的承诺如何，人们越来越难以从云存储服务中清理他们的数字生活。

![](img/f339c7d6fbe720cb8b09ce9130003cac.png)

Upgrade your storage to continue backing up: an automatic cloud backup system that locks in its user, leaving no alternative to the accumulation of content.

## 3.2.系统老化

今天的应用程序自动更新通常会增加对资源和处理能力的需求，以进行表面上的改进，这几乎是在故意试图让硬件过时，让软件更难操作。在多年逍遥法外之后，现在反对系统报废的意识更强了，因为这是一种浪费，而且剥削消费者。

![](img/42e16e729fc051e2ce6a6f2e042c3b23.png)

## 3.3.数字注意力

随着互联网上的内容呈指数级增长，(社交)媒体公司越来越依赖自动化来过滤信息并将其发送给每个用户。例如，YouTube 自动为 15 亿用户播放数十亿个视频。这些算法旨在促进更高参与度的内容，并倾向于引导人们反对他们的兴趣。

鉴于这些笨拙和不诚实的自动化例子，我担心的是，许多工程师、数据科学家、设计师和决策者将这些摩擦带入人们的日常生活，因为他们没有采用方法来预见他们工作的限制和影响。除了高效解决方案的工程设计之外，自动化还要求专业人员思考他们实践的基础和结果，这些超越了他们组织的任何关键绩效指标。

# 4.人性化自动化设计

自动化的设计不是要消除人类的存在。它是关于人性化、尊重和信任的系统的设计，这些系统自动化了人类活动的某些方面。当[与该领域的数据科学家、设计师和工程师](/the-mindset-for-innovation-with-data-science-fc51605a4867)合作时，我们设想系统超越“用户”和要自动化的“任务”的范围。我鼓励团队 a)从过去中学习 b)评论现在，c)讨论未来。让我解释一下:

## 4.1.从过去吸取教训

当谈到自动化时，学术界和工业界的知识获取并不是分开的追求。在过去的 50 年里，研究机构对自动化人工任务和决策的含义进行了大量的研究。关键发现有助于在关键环境中节省资金，并防止大量致命错误(例如在驾驶舱中)。

今天，这种知识并没有转化为日常任务。例如，许多工程师或数据科学家不掌握由科学和技术研究或人机交互研究社区理论化的自动化偏差(即人类倾向于支持来自自动化决策系统的建议)或自动化自满(即人类对监控自动化结果的关注减少)等概念。可悲的是，只有少数组织推动聚集学者、艺术家、工程师、数据科学家和设计师的平台。处于数字化进程中的行业将从这种专业人员的相互交流中大大受益，这些专业人员从他们学科之外已经出现的考虑中学习。

## 4.2.批判现在

我认为，参与人类活动自动化业务的专业人员应该是他们的同行所部署的解决方案的坚持不懈的关键评审者。他们应该成为今天人们如何处理现代生活中出现的笨拙、不诚实、令人讨厌、荒谬和任何其他尴尬的数字技术的跟踪者。

#TUXSAX is an invitation to engage with these knotty, gnarled edges of technology. It provides some raw food for thoughts to consider the mundane frictions between people and technologies. Do we want to mitigate, or even eliminate these frictions? Source: [Documenting the State of Contemporary Technology](https://medium.com/@girardin/documenting-the-state-of-contemporary-technology-ddf2cc6c8de4).

当被恰当地记录下来时，这些观察为众多“天真的乐观主义”和科技行业迷人的乌托邦式愿景提供了一种补充形式的灵感。它们为专业人员提供了质疑自动化可能存在偏见的目标的材料。此外，他们为定义组织中可实现的目标做好准备(例如，smart/intelligent 是什么意思？，如何衡量效率？，什么必须变得易读？).

## 4.3.辩论未来

在今天的互联网中，即使是最简单的应用程序或连接对象的设计也已成为一项复杂的工作。它们建立在分割的操作系统、众多协议、版本、框架和其他可重用代码包之上。数字摩擦的缓解超出了保证应用程序健全性的“质量保证”团队的范围。它们也是关于记录对技术生存环境的影响、意想不到的后果和“假设”情景。

![](img/436b241b7db0227c13edd0636a815633.png)

It’s easy to get all Silicon Valley when drooling over the possibility of a world chock-full of self-driving cars. However, when an idea moves from speculation to designed product it is necessary to consider the many facets of its existence — the who, what, how, when, why of the self-driving car. To address these questions, we took a sideways glance at it by forcing ourselves to write the quick-start guide for a typical self-driving car. Source: [The World of Self-Driving Cars](https://medium.com/design-fictions/the-world-of-self-driving-cars-b1f7ade18931).

一般来说，设计虚构是一种引发对话和预测有关人类活动自动化的更大问题的方法。例如，我们制作了亚马逊 Helios: Pilot 的[快速入门指南，这是一部虚构的自动驾驶汽车。在那个项目中，我们确定了涉及自动驾驶汽车人性方面的关键系统，并以一种非常有形、引人注目的方式为设计师、工程师和其他任何参与自动化系统开发的人带来了这种体验。通过集体创作,《快速入门指南》成为了一个图腾，任何人都可以通过它来讨论后果、提出设计考虑和形成决策。](http://qsg.nearfuturelaboratory.com/)

# 5.信托业务

像许多技术进化一样，日常生活的自动化不会没有为了方便而交易控制的摩擦。然而，后果比减轻边缘案件更大。它们反映了人类、组织或社会的选择。部署系统的选择误导了他们与人民和社会利益相冲突的意图。

马克·魏泽在 90 年代关于普适计算的开创性工作中，强烈影响了当前计算领域的“第三次浪潮”，当时技术已经退居人们生活的背景。科技行业的许多专业人士(包括我)都接受他对冷静技术的描述，即“*告知但不要求我们关注或注意*”然而，魏泽和其他许多人(包括我)没有预料到的是一个不诚实的自动化或解决方案的行业，当事情没有按计划进行时，它们会违背用户的意图。我们也没有真正预料到自动化会在多大程度上反作用于部署自动化的组织，而这些组织遭到了客户、社会和决策者的强烈反对。

这些暗示为任何参与数字自动化业务的组织提供了超越纯技术和商业的替代范例。例如，一种促进尊重(过度高效)、清晰(过度冷静)和诚实(过度聪明)技术的范式。当专业人员(例如，工程师、数据科学家、设计师、决策者、高管)游离于他们的实践之外，运用批判性思维来揭露不诚实的行为，并使用虚构来做出考虑超出“用户”和要自动化的“任务”范围的影响的决策时，这些就是出现的价值类型。

我认为，在自动化行业中，维持现状、不发展成信任企业的组织可能最终需要处理声誉受损及其对内部价值观、员工道德、收入以及最终利益相关者信任的影响。

我积极参与人文科技的发展。如果你想用其他范式或不同方法来设计尊重人民和社会利益的技术，请随时发表评论或[联系我](https://girardin.org/fabien)。

感谢 Nicolas Nova 和 Jose Antonio Rodriguez 对本文提出的周到建议。

*原载于 2019 年 1 月 16 日*[*blog.nearfuturelaboratory.com*](http://blog.nearfuturelaboratory.com/2019/01/16/when-automation-bites-back/)*。*