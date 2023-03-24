# 网页抓取合法吗？

> 原文：<https://towardsdatascience.com/is-web-crawling-legal-a758c8fcacde?source=collection_archive---------6----------------------->

![](img/b7417049045c7a8697310effcab7fc9e.png)

Photo by [Sebastian Pichler](https://unsplash.com/@pichler_sebastian?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/legal?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

网络爬行，也称为网络抓取、数据抓取或蜘蛛，是一种用于从网站抓取大量数据的计算机程序技术，其中规则格式的数据可以被提取并处理成易于阅读的结构化格式。

# Web 爬行通常用于:

网络爬行基本上就是互联网的功能。例如，SEO 需要创建网站地图，并允许 Google 抓取他们的网站，以便在搜索结果中获得更高的排名。许多咨询公司会雇用专门从事网络搜集的公司来丰富他们的数据库，以便为他们的客户提供专业服务。

在数字化时代，很难确定网络抓取的合法性。

# 为什么网络爬行有负面含义:

网页抓取可用于恶意目的，例如:

1.  搜集私人或机密信息。
2.  无视网站的条款和服务，未经所有者允许就刮。
3.  数据请求的滥用方式会导致 web 服务器在额外的高负载下崩溃。

**请注意，在以下情况下，负责任的数据服务提供商将拒绝您的请求:**

1.  数据是私人的，需要用户名和密码
2.  TOS(服务条款)明确禁止网页抓取行为
3.  这些数据是受版权保护的

# 有哪些理由可以用来告人？

*   [违反《计算机欺诈和滥用法案》(CFAA)。](https://www.nacdl.org/cfaa/)
*   [违反数字千年版权法(DMCA)](https://en.wikipedia.org/wiki/Digital_Millennium_Copyright_Act)
*   [侵犯动产](http://cyberlaw.stanford.edu/blog/2016/02/digital-trespass-what-it-and-why-you-should-care)
*   [挪用。](https://definitions.uslegal.com/m/misappropriation-law/)

你的**“刚刮了一个网站”**如果使用不当，可能会造成意想不到的后果。

# HiQ vs LinkedIn

你大概听说过 2017 年的 [HiQ vs Linkedin 案](https://www.reuters.com/article/us-microsoft-linkedin-ruling/u-s-judge-says-linkedin-cannot-block-startup-from-public-profile-data-idUSKCN1AU2BV)。HiQ 是一家数据科学公司，向企业人力资源部门提供抓取的数据。Linkedin 随后发出停止信，要求停止 HiQ 抓取行为。HiQ 随后提起诉讼，要求 Linkedin 阻止他们的访问。结果，法院做出了有利于 HiQ 的裁决。这是因为 HiQ 在没有登录的情况下从 Linkedin 上的公共档案中抓取数据。也就是说，抓取互联网上公开分享的数据是完全合法的。

让我们再举一个例子来说明在什么情况下网络抓取是有害的。易贝诉投标人边缘案。如果你是出于自己的目的进行网络爬行，这是合法的，因为它属于合理使用原则。如果你想把搜集来的数据用于其他人，特别是商业目的，事情就复杂了。引自 Wikipedia.org，100 F.Supp.2d 1058。2000 年)，是一个领先的案例适用于动产入侵理论的在线活动。2000 年，在线拍卖公司易贝成功利用“侵犯动产”理论获得了一项初步禁令，禁止拍卖数据聚合网站 Bidder's Edge 使用“爬虫”从易贝网站收集数据。该意见是将“侵犯动产”适用于在线活动的一个主要案例，尽管其分析在最近的判例中受到了批评。

只要你不是以一个破坏性的速度爬行，并且源代码是公开的，你应该没问题。

我建议你检查一下你计划抓取的网站，看看是否有任何与抓取他们的知识产权相关的服务条款。如果上面写着“禁止刮擦或爬行”，你应该尊重它。

## 建议:

1.  小心地刮擦，在开始刮擦之前检查“Robots.txt”
2.  保守一点。过分地要求数据会加重互联网服务器的负担。合乎道德的方式是温和。没人想让服务器崩溃。
3.  明智地使用数据。不要复制数据。您可以从收集的数据中获得洞察力，并帮助您的业务增长。
4.  在你开始抓取之前，先联系网站的所有者。
5.  不要随意把刮来的数据传给任何人。如果它是有价值的数据，请确保其安全。