# 你想在 NumPy 文档中看到什么？

> 原文：<https://towardsdatascience.com/what-do-you-want-to-see-in-the-numpy-docs-de73efb80375?source=collection_archive---------21----------------------->

## NumPy 和 SciPy 与谷歌文档季的幕后

![](img/09b909bb66c26bdae7fa6067cf7c4ad2.png)

Photo by [Chevanon Photography](https://www.pexels.com/@chevanon?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) from [Pexels](https://www.pexels.com/photo/red-betta-fish-1335971/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)

在经历了许多焦虑之后，我欣喜若狂地发现我被 NumPy 选中参加[Google Docs 季](https://opensource.googleblog.com/2019/08/season-of-docs-announces-technical.html)！！！！！

GIF via [GIPHY](https://giphy.com/gifs/happy-excited-cartoons-F6PFPjc3K0CPe)

[Docs](https://developers.google.com/season-of-docs/)季开始了！！！

## 谷歌文档季？

谷歌做了一件惊人的事情，创造了文档季。它为技术作者与开源组织合作创造了真正的机会。

“博士季节”是一个为期三个月的辅导项目。它将技术作者与开源组织配对。作家有机会与知名和备受推崇的组织合作。开源组织(他们通常没有技术作者的预算)有机会与有经验的技术作者一起改进和扩展他们现有的文档。

这太不可思议了。

> Docs 季的目标是为技术作者和开源项目提供一个框架，让他们共同努力实现改进开源项目文档的共同目标。对于不熟悉开放源码的技术作者来说，该计划提供了一个为开放源码项目做贡献的机会。对于已经在开源领域工作的技术作者来说，该程序提供了一种潜在的新的合作方式。Docs 季也给开源项目一个机会，让更多的技术写作社区参与进来。
> 
> 在这个项目中，技术作者花了几个月的时间与开源社区密切合作。他们将他们的技术写作专业知识带到项目的文档中，同时了解开源项目和新技术。
> 
> 开源项目与技术作者一起改进项目的文档和过程。他们可能会一起选择构建一个新的文档集，或者重新设计现有的文档，或者改进并记录开源社区的贡献过程和入职体验。
> 
> 我们一起提高公众对开源文档、技术写作以及我们如何合作以造福全球开源社区的意识。
> 
> ~ [谷歌文档季简介](https://developers.google.com/season-of-docs/docs/)

这是双赢！

![](img/627bddbb679308f6ff48003500412c52.png)

Photo by [It’s me, Marrie](https://www.pexels.com/@it-s-me-marrie-1418249?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) from [Pexels](https://www.pexels.com/photo/pembroke-welsh-corgi-sticking-its-tongue-out-2737392/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)

## 什么是 NumPy？

在最基本的层面上， [NumPy](https://www.numpy.org/) 是数值，或者说是数值( **Num** ) Python ( **Py** )。

来自官方文件:

> “NumPy 是 Python 中科学计算的基础包。它是一个 Python 库，提供了一个多维数组对象、各种派生对象(如掩码数组和矩阵)以及一系列用于数组快速操作的例程，包括数学、逻辑、形状操作、排序、选择、I/O、离散傅立叶变换、基本线性代数、基本统计操作、随机模拟等等。”

这是一个非常重要的开源 Python 库。它是 Python 中科学计算的核心库。它在数据科学、机器学习、深度学习、人工智能、计算机视觉、科学、工程等领域都很有用。它增加了对大型多维数组和矩阵的支持，以及可以对数组进行操作的大量高级数学函数。

NumPy(数字)的祖先最初是由[吉姆·胡古宁](https://en.wikipedia.org/wiki/Jim_Hugunin)创造的。到 2000 年，人们对为科学和技术计算创造一个完整的环境越来越感兴趣。2001 年，Travis Oliphant、Eric Jones 和 Pearu Peterson 合并了他们编写的代码，并将结果包称为 SciPy。2005 年，[特拉维斯·奥列芬特](https://en.wikipedia.org/wiki/Travis_Oliphant)创建了 NumPy。他通过将 Numarray 的特性整合到 Numeric 中并做了大量的修改来做到这一点。2005 年初，他希望围绕一个单一的阵列包来统一社区。因此，他在 2006 年发布了 NumPy 1.0。这个项目是[科学计划](https://en.wikipedia.org/wiki/SciPy)的一部分。为了避免安装大的 SciPy 包只是为了获得一个数组对象，这个新的包被分离出来并被称为 NumPy。

## 什么是 SciPy？

是科学(**Sci**Python(**Py)**！ [SciPy](https://www.scipy.org/) 是一个免费的开源 Python 库。它用于科学计算和技术计算。它包含了用于[优化](https://en.wikipedia.org/wiki/Optimization_%28mathematics%29)、[线性代数](https://en.wikipedia.org/wiki/Linear_algebra)、[积分](https://en.wikipedia.org/wiki/Integral)、[插值](https://en.wikipedia.org/wiki/Interpolation)、[特殊函数](https://en.wikipedia.org/wiki/Special_functions)、 [FFT](https://en.wikipedia.org/wiki/Fast_Fourier_transform) 、[信号](https://en.wikipedia.org/wiki/Signal_processing)、[图像处理](https://en.wikipedia.org/wiki/Image_processing)、 [ODE](https://en.wikipedia.org/wiki/Ordinary_differential_equation) 解算器等科学与工程中常见任务的模块。SciPy 使用 NumPy 数组作为基本的数据结构。它有科学编程中各种常用任务的模块。这些任务包括积分(微积分)、常微分方程求解和信号处理。

SciPy 构建在 [NumPy](https://en.wikipedia.org/wiki/NumPy) 数组对象上。这是 NumPy 堆栈的一部分。该堆栈包括像 [Matplotlib](https://en.wikipedia.org/wiki/Matplotlib) 、 [Pandas](https://en.wikipedia.org/wiki/Pandas_%28software%29) 和 [SymPy](https://en.wikipedia.org/wiki/SymPy) 这样的工具，以及一组不断扩展的科学计算库。它的用户来自科学、工程等所有领域。Python 拥有最大的科学用户群体之一，如果不是最大的话。类似的社区还有 R，Julia，Matlab。

*还跟我在一起？*

![](img/dcf6bd04f6a08ed56a6e4d9039bb4775.png)

Photo by [Passerina](https://www.pexels.com/@passerina-523993?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) from [Pexels](https://www.pexels.com/photo/tilt-shift-lens-of-yellow-napped-amazon-1257855/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)

## 该过程

谷歌在 2019 年 3 月宣布了 Docs 季。4 月，开源组织有机会申请成为该计划的一部分。谷歌于 4 月 30 日公布了入选组织。技术作者可以查看 45 个组织的列表，并选择他们感兴趣的项目。他们最多可以提交三份项目建议书。从 5 月 29 日至 6 月 28 日，技术写作申请开放！申请截止日期过后，每个组织选择他们有兴趣指导的技术写作项目。

[8 月 6 日，谷歌公布了被接受的写作项目！](https://opensource.googleblog.com/2019/08/season-of-docs-announces-technical.html)

该计划收到了来自近 450 名技术作家的 700 多份技术写作项目建议书。每个组织能够为一个被批准的项目选择一个技术作者。然而，NumPy/SciPy 团队决定更进一步，在《Docs》季之外，为另外三名作者争取资金。该团队如此坚信推进他们的文档，以至于他们找到了额外的资金。这使得他们在与《Docs》一季相同的条件下，增加了三名作家。

## 经费从哪里来？

NumPy 收到了两笔赠款，这是一种一揽子交易(你可以在这里[和](https://www.moore.org/grant-detail?grantId=GBMF5447)[这里](https://bids.berkeley.edu/news/bids-receives-sloan-foundation-grant-contribute-numpy-development)了解它们)。摩尔和斯隆基金会向伯克利数据科学研究所(BIDS)拨款 130 万美元，支持 NumPy 的开发。资助期从 2018 年 4 月到 2020 年 10 月。NumPy 指导委员会成员 Stéfan van der Walt 同意从这笔赠款中提供资金。)

Ralf Gommers 是 NumPy 和 SciPy 的核心程序员之一，也是 Quansight 实验室的负责人。拉尔夫是一个不可思议的人，他对《医生的季节》有这样的评价:

> “当我第一次看到 Docs 季的公告时，我喜欢这个计划的想法——对我个人来说，与一位科技作家合作将是一种有趣的新体验，对 NumPy 和 SciPy 来说可能有很大的好处。所以我花了很多精力写了一个非常吸引人的想法页面，然后跟进那些表现出兴趣的作者。我大概接到了 10 个视频电话，以及更多的电子邮件。
> 
> 然后，事实证明，兴趣很大，申请人和提案的质量真的很高。我开始思考如何不仅让一个或两个为期 3 个月的项目运行，而且如何让这些作家参与进来，让他们享受足够的体验，以便在项目结束后留下来。我想到的一件事是，人们喜欢和志同道合的人一起工作。然而，我们还没有技术人员——给 NumPy 和 SciPy 各增加一个可能还不够。所以我决定开始建立一个文档团队。想法和人都在那里，所以接下来需要的是资金。
> 
> NumPy 有一笔很大的活动资金，所以我和 Stéfan 讨论了将这笔资金用于 Docs 项目额外一季的可能性。斯蒂芬很棒，他也看到了提议的项目和建立一个作家团队的价值。所以他同意为此预留一些资金。所以我们今天来了——兴奋地开始吧！"

## 编剧是谁？

为 NumPy/SciPy 文档项目选择的作者是惊人的，你需要知道他们是谁！

## 玛雅·格沃兹兹

SciPy 在 Docs 季节选择的官方技术作家是 Maja Gwozdz。她的项目提案叫做“面向用户的文档和彻底的重构”你可以[在这里](https://developers.google.com/season-of-docs/docs/participants/)阅读所有相关内容，但本质上，Maja 打算对现有文档进行重构，以便不同需求的用户可以轻松访问。

玛雅做了一些惊人的研究，你可以在这里找到[。她不仅对 SciPy 有丰富的经验，而且她很清楚优秀的文档和指南会带来多大的不同。](https://lmu-munich.academia.edu/MajaGwóźdź)

## 安妮·邦纳

你真诚的(耶！)是 NumPy 的官方选择，项目提案是“让‘基础’更基础一点:改进 NumPy 的入门部分。”因为没有什么比帮助初学者理解复杂的信息和技术更让我高兴的了，NumPy 是一个完美的挑战！

我很高兴能够深入研究介绍性的 NumPy 材料，为没有经验的人创造更容易理解的东西。NumPy 处于这样一个有趣的位置:它非常复杂，但对于对数据感兴趣的初学者来说，它也是最重要的库之一。我将在 NumPy 中创建基本概念的初级文档，这些文档可以作为想使用*NumPy，而不一定要学习它的人的垫脚石。*

## ***更新* * *

初学者指南正在进行中，我希望得到您的反馈！看看这里的中等形式，让我知道你的想法！

[](/the-ultimate-beginners-guide-to-numpy-f5a2f99aef54) [## NumPy 初学者终极指南

### 开始使用 NumPy 需要知道的一切

towardsdatascience.com](/the-ultimate-beginners-guide-to-numpy-f5a2f99aef54) 

## 谢哈尔·拉贾克

[Shekhar Rajak](https://medium.com/@Shekharrajak) 被选为“Numpy.org 重新设计和高级文档重组的最终用户焦点”他的项目目标包括:

*   为 www.numpy.org[设计和开发更好的 UI](http://www.numpy.org)
*   增强和修改[www.numpy.org](http://www.numpy.org)的内容:NumPy 用户指南、NumPy 基准测试、F2Py 指南、NumPy 开发者指南、构建和扩展文档、NumPy 参考、关于 NumPy、报告 bug 以及所有其他与开发相关的页面。
*   增加了关于何时使用 NumPy，何时使用 XND，Dask 数组 Python 库的内容，提供了类似的 API。
*   保存 Python API 文档。

## 布兰登·大卫

Brandon David 因其项目“改进 scipy.stats 的文档”而入选。Brandon 计划填充缺失的函数，并添加示例和内部链接。他的目标是消除歧义，解决 GitHub 上的问题。

## 克里斯蒂娜·李

克里斯蒂娜·李因她的提案“科学文档:设计、可用性和内容”而被选中她是最近加入的，我期待着很快与你分享她的作品！

## [Harivallabha Rangarajan](https://medium.com/@harivallabharangarajan)

[Harivallabha Rangarajan](https://medium.com/@harivallabharangarajan) 计划以任何可能的方式为文档做出贡献，并补充为文档季选择的作者的工作。他对为 scipy.stats 模块编写端到端教程特别感兴趣。他写道，“拥有更全面的教程将有助于用户更好地了解可用方法在管道中的使用方式和位置。”

**欢迎来到博士季！！！**

参与 NumPy 和 SciPy 的内部工作是令人难以置信的。到目前为止，我们一直在参加团队会议，了解核心成员，学习工作流程。我迫不及待地想让你们了解我们项目的进展。

![](img/45932e9f14c0674eab1a9fe722bbc238.png)

Photo by [Pineapple Supply Co.](https://www.pexels.com/@psco?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels) from [Pexels](https://www.pexels.com/photo/photo-of-pineapple-wearing-black-aviator-style-sunglasses-and-party-hat-1071878/?utm_content=attributionCopyText&utm_medium=referral&utm_source=pexels)

## 参与进来！

既然您已经了解了写作方面的主要参与者，那么不要害怕联系我们，让我们知道您是否想在官方文档中看到相关信息！谁知道呢，我们也许能给你你想看的东西。

如果你对加入开源组织的想法感兴趣，那就去那里开始分享吧！不要等待邀请。现在就开始投稿吧！每个人都有责任让科技世界变得更加精彩。

如果您有兴趣为开源组织做贡献，但不知道如何开始使用 GitHub，您可能想看看这篇文章:

[](/getting-started-with-git-and-github-6fcd0f2d4ac6) [## Git 和 GitHub 入门:完全初学者指南

### Git 和 GitHub 基础知识，供好奇和完全困惑的人使用(加上最简单的方法来为您的第一次公开…

towardsdatascience.com](/getting-started-with-git-and-github-6fcd0f2d4ac6) 

感谢阅读！和往常一样，如果你对这些信息做了什么很酷的事情，请在下面的评论中让所有人都知道，或者联系 LinkedIn [@annebonnerdata](https://www.linkedin.com/in/annebonnerdata/) ！