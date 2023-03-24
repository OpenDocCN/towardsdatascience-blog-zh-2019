# 为什么 AI/机器学习在企业中停滞不前？

> 原文：<https://towardsdatascience.com/why-has-cloud-native-ai-machine-learning-stalled-in-the-enterprise-21cfaeb29551?source=collection_archive---------28----------------------->

主要是因为合理的云安全考虑。而 AI/ML 的数据胃口引发了与企业安全的严重冲突。有哪些问题？

![](img/5a3f4e2a0845a10e0dd732e0d6975076.png)

Photo by [Nareeta Martin](https://unsplash.com/@splashabout?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/slow?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

# 为什么企业在云中采用 AI/ML 停滞不前？

简而言之，因为材料，真实的，有效的数据安全问题。不幸的是，这些担忧与 AI/ML 对数据的贪婪胃口直接冲突。在冲突得到解决之前，云原生 AI/ML 在企业中的采用将会停滞不前。

最近关于 Capital One 最近的数据泄露及其成本的头条新闻强调了这一点。它甚至可能让像 Capital 这样的重度云消费者猜测他们的云数据安全和隐私状况。

由于潜在的巨额损失和罚款，可以预料，许多企业在将少量敏感数据迁移到云中时都会非常谨慎。

然而，尽管有潜在的风险，公司继续将目标对准 AI/ML 云平台上可用的高级计算能力。因此，像之前的通用计算一样，人工智能/人工智能在云上的进军现在正在开始，早期采用者正在建立云原生人工智能/人工智能能力的滩头阵地。

但是使用云的早期企业 AI/ML 采用者看到一个主要障碍反复出现:云数据安全和隐私问题。事实上，AI/ML 的独特特征加剧了这些担忧，毕竟，AI/ML 的成功与大量数据的可用性直接相关，其中大部分可能是机密或非常敏感的数据。

*在这些严重的、实质性的、有效的云数据安全问题得到解决之前，云上的企业 AI/ML 采用将停滞不前，被降级为小项目，交付缓慢，不幸的是，提供的价值有限。*

在本文中，我将讨论 AI/ML 的独特特征，这些特征导致了数据安全问题，阻碍了云原生 AI/ML 的广泛采用。

# AI/ML 向云的进军本来应该很简单

“理论”是，向云原生 AI/ML 进军应该很简单。毕竟，互联网巨头已经展示了如何应用基于云的人工智能/人工智能来解决以前难以解决的问题，例如提供比人类更好的机器视觉，实时音频和文本翻译。

毫不奇怪，企业认为使用相同的云功能、相同的开源软件以及相同的丰富网络和计算能力将有助于轻松实现云原生 AI/ML。

然而，在实践中，企业遇到了一些障碍。在最近的一份报告中，德勤指出，尽管有很大的兴趣，“43%的公司表示他们对潜在的人工智能风险有重大或极端的担忧。”而且，在这些担忧中，“排在首位的是网络安全漏洞(49%的人将其列为三大担忧之一)。”事实上，“对大多数国家来说，对潜在人工智能风险的担忧超过了他们应对这些风险的信心。”

但是，这些担忧与安全性旨在保护的东西(即数据)交织在一起。不幸的是，大多数企业可能没有做好准备。根据 Irving Wladawsky-Berger 在《华尔街日报》的一篇文章中的说法，只有“18%的人说他们的公司制定了访问和获取人工智能工作所需数据的战略”。

*事实证明，企业采用云原生人工智能/人工智能并不那么简单。事实上，云上的企业 AI/ML 采用正在停滞。在云上采用企业 AI/ML 时，有一个棘手的挑战是一个基本问题:数据安全和隐私。*

![](img/d3503f850acf5ad98fa80127b9e6c926.png)

Photo by [Srh Hrbch](https://unsplash.com/@srhhrbch?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/security-camera?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

# AI/ML 的独特需求是重新思考云安全策略的催化剂

AI/ML 在企业中有一些独特的特征:首先，数据科学家——实际上是 AI/ML 从业者——需要频繁访问敏感数据。无论它被认为是私有的(客户数据)还是机密的(财务数据)，或者介于两者之间，底线是企业将从对其唯一的数据中获得最大价值。对于企业来说是唯一的数据，根据定义，这些数据几乎都是敏感的，必须受到保护。过去，对敏感数据的访问通常受到严格控制。

其次，数据科学家要求敏感数据“畅通无阻”。过去，数据存储在被锁定的生产系统中，只有在受控和特定的情况下应用程序才能访问，即使可以访问，生产数据也被严重屏蔽。

但是，今天的数据科学家有一个基本需求，即以无屏蔽的形式访问生产数据。简而言之，如果数据科学家需要某个数据属性或特征来进行模型开发、培训或验证，那么该数据必须在处理时“清晰地”(无屏蔽地)可用。其含义很清楚:如果培训活动要利用云资源，那么数据必须“明文”驻留在云上。

第三，AI/ML 因数据而兴盛。然而，更多的数据——尤其是云中更敏感的数据——大大增加了风险。显而易见的含义是，如果发生数据泄露，那么这将是一个巨大的数据泄露，并带来相应的巨大成本。

最后，保护云中的数据非常困难。必须建立安全边界。必须在企业及其云租赁之间创建安全的链接。对数据的访问必须在一定的粒度级别上进行仲裁，这需要复杂的治理和基于角色的访问控制(RBAC)机制。

这只是典型问题的一小部分。经验表明，这些问题，以及建立和实施解决这些问题的强大数据安全和隐私政策的相关高成本，是在云上广泛采用 AI/ML 的重大障碍。

*显然，AI/ML 的独特需求与所需数据的敏感性相结合，迫使企业不得不采取行动——需要一套新的安全策略来保护云中的敏感数据。*

![](img/6693b33e10199de81e2563f5d4f3eaf9.png)

Photo by [Randy Colas](https://unsplash.com/@randycolasbe?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/police?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

# 云安全策略必须应对一系列新的威胁

在某些方面，这些天的头条新闻几乎像一本漫画书，其中我们的英雄——企业——不断受到攻击。就像漫画书一样，大多数攻击都被击退了，但也有一些总能成功。下面是企业必须应对的一些威胁(见图 1)。

![](img/1f4427dcc1f4554547b1d462866cf809.png)

Figure 1

场景 1:首先想到的是恶意黑客的攻击，他们试图渗透到企业云租赁中。关于这一点[已经写了很多](https://en.wikipedia.org/wiki/List_of_data_breaches)，所以不需要多说什么，除了损害可能是广泛的:想想最近的优步黑客事件，据 [TechCrunch](https://techcrunch.com/2018/10/25/uber-hackers-indicted-lynda-breach/) 报道，两名黑客绕过安全控制，“窃取了数百万用户的数据”。

场景 2:管理企业云租赁的员工可能会无意中出错，这可能是由于基础架构配置错误或未打补丁，从而导致未被发现的数据泄露。例如，在 Capital One，泄露 1 亿客户数据的大规模数据泄露事件是由错误配置的防火墙引起的。类似地，Equifax 的数据泄露导致 1.48 亿客户的数据泄露，这是由[未打补丁的](https://epic.org/privacy/data-breach/equifax/)软件造成的。

场景 3:企业需要处理由于潜在的松散控制而导致的敏感数据的无意泄露。最近[关于英国 Monzo 银行的头条新闻](https://www.wired.co.uk/article/monzo-pin-security-breach-explained)强调“该银行近 50 万的客户被要求重置他们的个人识别码，因为这些信息被保存在一个员工可以访问的不安全文件中。”

场景 4:有权访问敏感数据的员工可能会非法泄露这些数据。环球邮报[报道](https://www.theglobeandmail.com/business/article-desjardins-group-suffers-massive-data-breach-of-29-million-members-by/)影响 290 万客户的 Desjardins 集团数据泄露是由一名违规员工引起的，该员工涉嫌窃取和暴露敏感数据，包括个人信息，如姓名、出生日期、社会保险号(类似于美国社会保险号)以及电子邮件、电话和家庭地址。

场景 5:企业还需要考虑第三方或云供应商对其云基础设施做出的意外更改。最近，一家领先的云供应商对导致客户数据泄露的[违规事件](https://www.eweek.com/security/microsoft-s-cloud-email-breach-is-a-cause-for-concern)承担了责任。值得庆幸的是，这种情况很少发生，但仍然值得关注。

# 云上的企业 AI/ML 可能已经停滞了，但还是有解决方案的

回到我最初的问题:为什么云上的企业 AI/机器学习停滞不前了？简而言之，主要是由于有效的数据安全问题。而且随着企业 AI/ML 不断需求大量敏感的。如果有人能从中得出什么结论，那就是赌注每天都在变得越来越大。

解决方案是有的，但业界已经承认，过去的解决方案可能不会像需要的那样有效。事实上，[据麦肯锡的](https://www.mckinsey.com/business-functions/digital-mckinsey/our-insights/cloud-adoption-to-accelerate-it-modernization)称，“大多数传统 IT 环境采用基于边界的‘城堡和护城河’方法来实现安全性，而云环境更像是现代酒店，通过钥匙卡可以进入特定楼层和房间”。

*在下一篇* [*文章*](/rethinking-enterprise-data-security-and-privacy-for-cloud-native-ai-machine-learning-f2df9009bc6d) *中，我将分享一些与建立云原生 AI/ML 功能相关的个人经验。特别是，我将讨论“城堡和护城河”方法的缺点，然后解释麦肯锡倡导的“现代酒店房间”和“钥匙卡”概念如何解决阻碍企业在云中采用 AI/ML 的许多安全问题。*