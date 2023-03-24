# 零信任和零泄漏策略如何实现人工智能机器学习

> 原文：<https://towardsdatascience.com/how-zero-trust-and-zero-leakage-strategies-enable-ai-machine-learning-31dbaf597247?source=collection_archive---------19----------------------->

## 最近备受瞩目的数据泄露事件让 AI/ML 在云上止步不前。以下是零信任和零泄漏策略如何解决这些问题。

![](img/9709233835c7b7d15dbd15f4ce86c721.png)

Photo by [Bernard Hermant](https://unsplash.com/@bernardhermant?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/trust?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

# 迈向更安全的企业云租赁

越来越多的证据表明，云上的企业人工智能/人工智能正在[拖延](/why-has-cloud-native-ai-machine-learning-stalled-in-the-enterprise-21cfaeb29551)。罪魁祸首:真实有效的安全顾虑。尽管如此，AI/ML 的商业利益是压倒性的，对 AI/ML 的需求似乎是无法满足的。

但 AI/ML 对数据(其中许多是敏感数据)的不懈需求，加剧了人们对数据安全的担忧。不幸的是，旧的安全方法似乎不太管用。最近来自 Capital One 的[数据泄露](https://www.nytimes.com/2019/07/29/business/capital-one-data-breach-hacked.html)和来自 Desjardins Group 的[数据泄露](https://www.theglobeandmail.com/business/article-desjardins-group-suffers-massive-data-breach-of-29-million-members-by/)似乎强化了这一点。

这使企业陷入困境。美其名曰，一股不可阻挡的力量(业务需求)遇到了一个不可动摇的对象(企业安全否决)。

然而，企业正在反击，引入新技术，如[基于身份的安全](/why-identity-management-is-a-prerequisite-for-enterprise-ai-ml-on-the-cloud-408919055596)，这些技术已经能够解决许多安全问题。但是需要做更多的工作。

*在本文中，我将描述一种“零信任”和“零泄漏”的安全方法，这种方法建立在基于身份的安全等技术之上，以提供额外的(可能是必要的)功能来保护 AI/ML 赖以发展的敏感数据。通过采用“零信任”和“零泄漏”的安全方法，企业有很好的机会再次加快 AI/ML 在云上的采用。*

# 企业需求推动了 AI/ML 云的采用

今天，AI/ML 正在扩展内部数据中心的能力。简而言之，他们没有计算资源(例如，GPU 农场、按需分配的容量等。)AI/ML 所要求的。

不足为奇的是，许多企业现在已经得出结论，通过云可以获得一个更具可伸缩性、按需且经济高效的解决方案。迈向云原生 AI/ML 的旅程正在进行中。

或者看起来是这样。早期采用者——那些现在在云上有一些 AI/ML 实践经验的企业——已经提出了有效和严重的安全问题。有些甚至有相当公开和负面的经历(比如 [Capital One](https://www.nytimes.com/2019/07/29/business/capital-one-data-breach-hacked.html) 和 [Desjardins Group](https://www.theglobeandmail.com/business/article-desjardins-group-suffers-massive-data-breach-of-29-million-members-by/) )。

不幸的是，这些担忧阻止了之前企业大规模使用云原生 AI/ML 的宏伟目标。

![](img/120e7c549281a6f88e0884a37aca8cc3.png)

Photo by [Artem Sapegin](https://unsplash.com/@sapegin?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/assumption?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

# 有缺陷的假设是数据泄露的根本原因

直到最近，企业安全性一直专注于确保企业数据中心周围的网络安全。当时，这是有意义的，因为大多数应用程序和资产几乎只在数据中心内工作。此外，从应用程序开发、运营和安全管理的角度来看，它使管理变得更加容易。所谓的双赢局面。

不幸的是，默认的假设——这些旧技术将在未来应用——已经被证明是非常不正确的。更具体地说，旧的基于网络边界的安全方法不适用于云。最近的数据泄露也印证了这一事实。哪里出了问题？

三个关键问题动摇了这一假设。

首先，今天的敌人更加聪明，他们获得了更好的工具，导致网络边界变得更加漏洞百出。优步最近在云上的[数据泄露](https://www.washingtonpost.com/news/innovations/wp/2017/11/21/uber-waits-a-year-to-reveal-massive-hack-of-customer-data/)无疑证明了这一点。

第二，在安全和企业云租赁方面可能会出现错误。不幸的是，只需一个配置错误，就会导致一个受威胁的端点，从而开启企业的大范围云租赁。似乎即使是最优秀、最有经验的云团队也是“人质熵”——安全配置确实不可避免地会随着时间的推移而漂移，并且不可避免地会导致违规。或许 Capital One 最近在云上的[体验](https://www.nytimes.com/2019/07/29/business/capital-one-data-breach-hacked.html)是这种情况最令人心酸的例子。

最后，云的本质也产生了一个安全问题:默认情况下，许多资源被设计为“开放的”，或者在创建时就可以通过互联网访问。这要求企业安全团队为更大范围的云组件提供安全性，不幸的是，一个错误就可能导致灾难性的数据泄露。

有鉴于此，现代云安全方法现在可以说是在网络边界可能被攻破的明确假设下设计的。

# 保护云安全需要新的技术

有几个主要概念是安全企业云租赁的基础。

首先,“零信任”安全方法规定，除非提供明确的身份验证和授权，否则在企业的云租赁中不能访问任何内容。这种方法与[基于身份的安全方法](/why-identity-management-is-a-prerequisite-for-enterprise-ai-ml-on-the-cloud-408919055596)紧密集成在一起(可能没有它就无法实现)。

第二，“零泄漏”策略，即数据不能以任何未经授权的方式在企业云租赁之外传输，无论是意外还是有意。这解决了两种情况:第一，它阻止了未授权的黑客进入系统获取任何信息；其次，它可以防止被授权访问敏感数据的员工意外或恶意地允许数据泄露企业云租赁。

![](img/5cb5b033852e5c80808eafa6dbf4d239.png)

Photo by [Clem Onojeghuo](https://unsplash.com/@clemono2?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/door?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

# 零信任——从不信任，总是验证

[零信任](https://hackernoon.com/the-rise-of-zero-trust-architecture-17464e6cbf30)是一种基于两个前提的安全方法:“永远不信任”和“永远验证”。第一个是“永远不要信任”，默认情况下，在企业的安全边界内(在我们的例子中，是企业的云租用)没有任何东西是可访问的。第二个是“总是验证”，规定只有那些具有明确验证的凭证的人才被允许访问资源。

应用于云的零信任架构使用身份和相关凭据以及设备声明/信息(例如，IP 地址)来验证用户身份和验证授权，然后再提供对企业云租户中的数据和应用程序的访问。

对于 AI/ML，零信任确保驻留在云上的敏感数据只能由数据科学家访问，这些数据科学家经过身份验证并有权访问和使用关键敏感数据。认识到有效的安全问题是在云上采用 AI/ML 云的主要障碍，实现这种能力形成了再次加速云 AI/ML 计划的基础。

# 零信任实现注意事项

通常情况下，没有任何安全控制是孤立的:“零信任”最好使用一些补充技术来实现。以下是用于实施整体“零信任”解决方案的一些注意事项:

*   **过渡到基于身份的安全方法**[](/why-identity-management-is-a-prerequisite-for-enterprise-ai-ml-on-the-cloud-408919055596)**:该方法确保在所有企业资源中使用一致的身份，包括数据中心和云上的资源，这是“零信任”的“始终验证”部分的基础**
*   ****实施粒度执行技术**:应使用所有可用信息来验证请求/代理，不仅包括身份，还包括位置和设备信息，这些信息有助于确定:在高安全性场景中，应考虑多因素身份认证来提供额外的验证级别**
*   ****建立严格的支持性治理政策**:虽然“治理”可能是一个过重的术语，但就我们的目的而言，“需要知道”和“最小特权”原则应该得到严格的监控和执行**
*   ****协调角色和访问控制**:建立一个跨越企业数据中心和企业云租户的 RBAC(基于角色的访问控制)方案，从而提供一致的机制来实施授权、资源许可和安全控制**
*   ****检查并记录所有流量**:应扫描并验证在云租赁内流动的信息，并对照企业安全控制进行扫描和验证；应记录所有流量，以支持任何必要的调查或安全分析活动**
*   ****安全流程的集中管理**:零信任和相关的身份管理流程应集中管理，以确保安全控制的设计、实施和治理保持一致，从而最大限度地减少出现安全错误的机会**
*   ****监控配置变更和配置漂移**:熵——变更发生的必然性——将影响企业云租用的安全配置；为了解决这种情况，应监控所有配置更改，以确保尽快检测到危及零信任模型的意外或未经授权的更改并进行补救**

**![](img/0bb7e12a3ec7de43d3a2116cda4f79f4.png)**

**Photo by [Yogesh Pedamkar](https://unsplash.com/@yogesh_7?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/pipeline?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)**

# **零泄漏政策**

**云的相对可扩展性和计算能力显然有其优势。而 AI/ML 贪婪的数据需求和与之相称的计算能力需求是云的理想匹配。**

**但是云的设计也考虑到了“开放性”。默认情况下，可以通过互联网访问资源。在安全措施到位的地方，黑客现在拥有更多的工具来破坏这些安全措施。显然，如果没有适当的注意，这种开放性也可能成为安全隐患。**

**虽然 AI/ML 是云的理想候选，但它对数据的贪婪胃口(其中许多数据通常非常敏感)在任何情况下都不能离开企业的云租户，无论是意外还是有意的。**

**那么，企业如何确保数据不会因为某个恶意员工或者某个无意的事故而逃脱企业的边界呢？**

**如前所述，基于身份的安全和“零信任”策略对保护资源大有帮助。**

**但是当它涉及到消除**数据泄露**时，第一个也是最明显的起点是解决基本的云数据卫生问题。以下建议可作为这方面的起点:**

*   ****对所有静态(存储)和动态(传输)数据进行加密:**这将确保数据具有基本的保护级别，使非预期的访客无法查看或更改数据。根据企业的安全状况，大多数云供应商现在允许企业管理自己的密钥，从而确保即使是云供应商也无法访问企业的数据**
*   ****提供基本的网络安全、日志记录和监控**:注意云上的网络安全点不一定是保护资源(是的，它执行这个角色，但是零信任和零泄漏，以及相关的身份管理方法做得更好)。相反，它的作用主要是确保网络健康，以及为网络管理、检查和记录建立一个分界点；在这种情况下，这些成熟的技术防火墙方案、网络日志管理和网络分段可以应用于企业的云租用**

**但仅此并不能阻止“泄漏”，或者更具体地说，数据泄漏到批准的企业边界之外。**

**考虑一个必须解决的实际和真实的场景:当一个有权访问非常敏感的数据的授权员工(比如一个数据科学家)决定非法共享这些数据时，会发生什么？它确实发生了，而且可能比我们意识到的更频繁。**

**最明显的解决方案可能是锁定员工的设备(例如笔记本电脑或台式机)。可能设备 USB 和外部设备被锁定。并且为了停止任何基于网络的数据传输，可能会限制设备的网络接入。可以说这种方法是可行的，但是在大型企业用户群中执行起来非常复杂，管理起来更加困难。**

**一个更好更简单的机会是使用“安全门户”。这个“安全门户”相当于一个浏览器窗口，可以访问云租户，但禁用了允许数据移动的特定功能(通常是“剪切-粘贴”和下载的功能)。实际上，“安全门户”充当了访问云租用的安全观察口。**

**对于数据科学家和他们的组织来说，这提供了两个世界的最佳选择:不仅是一个功能相对完整的访问云原生 AI/ML 功能的视口(几乎所有的云功能都可以通过这种方式访问)，而且是一种保证消除数据泄漏的机制。**

**这些“安全门户”功能非常常见，由许多供应商提供:思杰有许多支持这一功能的[产品](https://www.citrix.com/downloads/citrix-receiver/)，微软([远程桌面](https://www.microsoft.com/en-us/p/microsoft-remote-desktop/9wzdncrfj3ps?activetab=pivot:overviewtab)，以及 VMWare ( [地平线](https://www.vmware.com/ca/products/horizon.html))也是如此。**

**![](img/75cc39274c1c81d66b7afe60361ddfd6.png)**

**Photo by [Samuel Zeller](https://unsplash.com/@samuelzeller?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/cloud?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)**

# **AI/ML 是重新思考云安全的催化剂**

**AI/ML 的商业利益是压倒性的，对 AI/ML 的需求似乎是无法满足的。但越来越多的证据表明，云上的企业人工智能/人工智能正在停滞不前。和真正的安全关切是主要的罪魁祸首。**

**不幸的是，旧的安全方法似乎不能很好地工作。**

**因此，这是[新安全方法](/rethinking-enterprise-data-security-and-privacy-for-cloud-native-ai-machine-learning-f2df9009bc6d)的催化剂。例如，基于身份的安全性有望解决这些问题，现在已经在企业中站稳脚跟。但是企业意识到需要做更多的工作。**

**这就是“零信任”和“零泄漏”的由来。越来越多的证据表明，零信任和零泄漏策略对解决许多剩余的安全问题大有帮助。**

***事实上,“零信任”和“零泄漏”策略现在不仅被视为先决条件，而且可能是改变企业传统上谨慎的安全态势的催化剂，使企业在云上采用 AI/ML 的速度再次加快。***