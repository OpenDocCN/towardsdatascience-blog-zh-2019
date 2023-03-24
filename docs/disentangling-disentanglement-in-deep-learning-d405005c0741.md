# 解开缠绕:NEURIPS 的脚注— 2019

> 原文：<https://towardsdatascience.com/disentangling-disentanglement-in-deep-learning-d405005c0741?source=collection_archive---------12----------------------->

![](img/1106724d49ddcfbf13278b64c4c4fea1.png)

Credit: ‘ Striving for Disentanglement’ by Simon Greig — [https://www.flickr.com/photos/xrrr/500039281/](https://www.flickr.com/photos/xrrr/500039281/)

TL；博士:解开，解开。通过这篇博文，我打算尝试总结今年在 2019 年温哥华 NEURIPS-2019 上发表的关于深度学习中的解纠缠的十几篇论文。

充满论文摘要和备忘单的配套 Github repo:[https://Github . com/vinay prabhu/distanglement _ neur IPS _ 2019/](https://github.com/vinayprabhu/Disentanglement_NEURIPS_2019/tree/master/Figures)

# 背景:表象学习中的解开

在会议周的周四晚上，当我在温哥华会议中心巨大的东展厅的海报会议上闲逛时，我意识到我在过去几天里偶然发现了可能是第五张海报，这需要对作者们工作的一个解开框架进行分析。

![](img/b5c2d1b340db9b71d40d7fb5b325cdb6.png)

Fig 1: (Yet another) Poster on disentanglement at this year’s NEURIPS

快速查看一下[的进程](https://papers.nips.cc/book/advances-in-neural-information-processing-systems-32-2019)，我得到了这个令人震惊的统计数据:今年共有十几篇标题为“**解开**”的论文被接受。在众多的车间里，我至少还偶然发现了一些。(2017 年 NEURIPS 研讨会期间有 20 多篇论文和演讲，主题是“学习解开表征:从感知到控制”——[https://sites.google.com/view/disentanglenips2017](https://sites.google.com/view/disentanglenips2017)，我们今年还举办了一场挑战研讨会:[https://www . ai crowd . com/challenges/neur IPS-2019-解开表征-挑战](https://www.aicrowd.com/challenges/neurips-2019-disentanglement-challenge))

我第一次感受到这个术语在统计学习中的用法是在我在 CMU 大学博士旅程的最后阶段(大约 2013 年)，当时我读了 Yoshua Bengio 的《[深度学习表征:展望](https://arxiv.org/pdf/1305.0445.pdf)》一书，他在书中强调了“成为”的必要性..学习*理清*观察数据背后的变异因素。(我多么希望他仍然撰写这样的单作者论文)

事实证明，也许让物理学家很懊恼的是，如果你正致力于从 MNIST 上的数字类型中梳理出视觉风格，或者从西里巴上的面部形状中分离出人体和面部特征图像中的形状和姿势，或者努力解开两种组成化合物的混合比例和环境因素(例如为微结构生长而生成的图像中的热波动)的影响，你就在*理清头绪。*

对于这个术语的确切含义或捕捉其范围的度量标准似乎没有达成共识，斯塔法诺·索阿托在 IPAM 的演讲中的这张相当有趣/尖刻的幻灯片证实了这一点(参考下面的播放列表)

![](img/21eb2b0d2634d99f0b278c786b021f9b.png)

Fig 2: Invariance and disentanglement in deep representations

也就是说，这不是一个存在仅仅少量的经验实验的案例，这些实验都使用他们自己定制的解开概念。事实上，人们已经提出了相当严格的框架，利用来自变分推理、Shannonian 信息论、群论和矩阵分解等领域的强大工具。Deepmind 对这一问题的群论处理似乎已经成为一个首选框架。如果你正在寻找一个简洁的 3 分钟回顾，请参考这个[视频](https://www.youtube.com/watch?v=PeZIo0Q_GwE&t=420s)，我在西蒙斯学院的一个研讨会上看到的(大约 7 分钟)。(可以在[这里](https://www.youtube.com/watch?v=XNGo9xqpgMo)找到来自 Deepmind group 的主要作者之一的非常详细的演讲)

![](img/7738cdc36beb3d360f233f6d561ad06a.png)

Fig 3: Group theoretic framework for disentanglement

## 对提交的论文的鸟瞰

下面的图 4 是 12 篇论文的鸟瞰图。我粗略地将它们分成两个小节，这取决于*的主要*感知的论文目标(从我的拙见来看)是分析和/或评论一个预先存在的框架的属性，还是利用一个框架并将其应用于一个有趣的问题领域。请记住，这无疑是一个相当简单的分类，对于面向应用的论文是否对所使用的框架进行了评论和分析，或者分析/评论论文是否不包括现实世界的应用，这并没有很大的指导意义。

![](img/149dbd8564d35ff7956e271b8ba6edc5.png)

Fig 4: Disentanglement papers categorization (NEURIPS -2019)

(可以在这里找到论文链接的 pdf 版本:[https://github . com/vinay prabhu/distanglement _ neur IPS _ 2019/blob/master/distanglement _ papers _ tree-diagram . pdf](https://github.com/vinayprabhu/Disentanglement_NEURIPS_2019/blob/master/Disentanglement_papers_tree-diagram.pdf))

## 他们所说的解开是什么意思？

为了总结这些论文中使用解纠缠的上下文，我创建了一个查找表(见表 1)。在那些作者明确没有专门的小节来定义的情况下，我临时拼凑并提取了要点(因此有了“即兴解释”)。

![](img/8cc99ea7056d0cd1dee2ca029ca51ff0.png)

Table-1(a) Disentanglement context in the application papers

![](img/2be7ec31ca74b917de8ad078c82e09fd.png)

Table-1(b) Disentanglement context in the analysis papers

## 可复制性和开源代码:

鉴于开源用于产生结果的代码的强劲增长趋势，12 个作者组中的 10 个也共享了他们的 github repos。下面的表 2 显示了这一点:

Table-2: Papers and the open-source code links

## 现在怎么办？一些想法..

[这里有一些涂鸦，试图让我自己更认真地工作。请半信半疑地接受这些或 12:)

1:调查报告，详细说明要使用的定义、框架和指标。

2:使用卡纳达语-MNIST 语数据集解开作者/写作风格/原籍国。(65 名印度本土志愿者和 10 名美国非本土志愿者)

[https://github.com/vinayprabhu/Kannada_MNIST](https://github.com/vinayprabhu/Kannada_MNIST)

3:有点令人惊讶的是，没有人尝试抛出一个 K 用户干扰信道模型进行纠缠，并看看类似 https://arxiv.org/pdf/0707.0323.pdf 的干扰对齐技巧是否适用于类似 Dsprites 的数据集

4:从步态表征中分离鞋类型、口袋和设备位置

5:连接与[(高光谱)解混](http://www.ee.cuhk.edu.hk/~wkma/publications/slides-%20HU-%20%20CWHISPERS%202015.pdf) /盲源分离和解纠缠表示学习相关的工作主体。

# 资源列表:

充满论文摘要和备忘单的配套 github repo。

[](https://github.com/vinayprabhu/Disentanglement_NEURIPS_2019) [## vinayprabhu/distanglement _ neur IPS _ 2019

### TL；DR - On 解开解开论文的动物园在这次 NEURIPS(2019)期间，我遇到了大量的…

github.com](https://github.com/vinayprabhu/Disentanglement_NEURIPS_2019) 

## A.数据集入门:

[1][https://www.github.com/cianeastwood/qedr](https://www.github.com/cianeastwood/qedr)

[https://github.com/deepmind/dsprites-dataset](https://github.com/deepmind/dsprites-dataset)

[https://github.com/rr-learning/disentanglement_dataset](https://github.com/rr-learning/disentanglement_dataset)

(neur IPS 2019:distanglement 的主要道具也向组织者挑战他们共享的资源！)

链接:[https://www . ai crowd . com/challenges/neur IPS-2019-distanglement-challenge](https://www.aicrowd.com/challenges/neurips-2019-disentanglement-challenge)

 [## Google-research/distanglement _ lib

### 此时您不能执行该操作。您已使用另一个标签页或窗口登录。您已在另一个选项卡中注销，或者…

github.com](https://github.com/google-research/disentanglement_lib/tree/master/disentanglement_lib/evaluation/metrics) 

## B.视频播放列表:

[1] Y. Bengio 的《从深度学习解开表征到更高层次的认知》

https://www.youtube.com/watch?v=Yr1mOzC93xs&t = 355 秒

[2]\贝塔-VAE(deep mind):【https://www.youtube.com/watch?v=XNGo9xqpgMo】T2

[3]柔性公平表征解缠学习:【https://www.youtube.com/watch?v=nlilKO1AvVs】T4&t = 27s

[4]用于姿态不变人脸识别的解纠缠表征学习 GAN:[https://www.youtube.com/watch?v=IjsBTZqCu-I](https://www.youtube.com/watch?v=IjsBTZqCu-I)

【5】深层表象中的不变性与解纠缠(趣谈)[https://www.youtube.com/watch?v=zbg49SMP5kY](https://www.youtube.com/watch?v=zbg49SMP5kY)

(来自 NEURIPS 2019 作者)
[1]审计模型预测论文:[https://www.youtube.com/watch?v=PeZIo0Q_GwE](https://www.youtube.com/watch?v=PeZIo0Q_GwE)

[2]对 Olivier Bachem 的 Twiml 采访(在 NEURIPS-19 上有 3 篇关于该主题的论文):[https://www.youtube.com/watch?v=Gd1nL3WKucY](https://www.youtube.com/watch?v=Gd1nL3WKucY)

## C.备忘单

![](img/7112225472aac6ede6a2301c89b50622.png)

Cheat sheet-1: All the abstracts! (Print on A3/2)

![](img/b16ad622c5c5ac4097eaa9db7be029d9.png)

Cheat sheet-2: All the essences!