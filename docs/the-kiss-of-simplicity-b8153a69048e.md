# 简单之吻

> 原文：<https://towardsdatascience.com/the-kiss-of-simplicity-b8153a69048e?source=collection_archive---------18----------------------->

## 做得更少但做得更好

![](img/a81691f0c9d7a0b0da91bb5f5ffeaf6f.png)

Photo by [Kyle Cesmat](https://unsplash.com/@kylecesmat?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

想象一个男人出现在一个正式的舞会上。他穿着一套剪裁完美的西装，夹克是亮粉色，带有红色方格图案。他的裤子是绿色的，带有紫色的竖条纹。这套衣服是一位著名设计师的作品，花了不少钱。这个人的脖子上还挂着一条金项链，双手戴着红宝石戒指。然而，在他身后，是一个穿着黑色礼服的女人，除了一对珍珠耳环之外，没有任何配饰。想一想这两位客人。

你认为这两者中哪一个更优雅？

IT 行业非常了解第一种类型。穿西装的人相当于一个花费了公司数百万美元的系统。该系统由来自不同提供商的几个组件组成，这些组件通常无法正常通信，而且往往只是在你需要向老板传递紧急信息的时候才会停止工作。

# 科技中的优雅

Unix 哲学[更接近](https://en.wikipedia.org/wiki/Unix_philosophy)穿黑裙子的女士，专注于使用工具或编写程序，做一件事，但做得完美。

当我们问[谷歌](https://www.google.com/search?q=elegance)什么是“优雅”它提供了两个。

> 1."外表或举止优雅时尚的品质."
> 
> 2.“令人愉快的巧妙和简单的品质；整洁。”

维基百科显示了类似的东西，

> "优雅是一种美，表现出不同寻常的有效性和简单性."

为什么我会在一个科技博客上进入优雅？因为正是我们这些工程师应该为过于复杂的系统、超出预算的项目以及围绕在我们工作周围的普遍丑陋(更好的说法是“技术债务”)而受到指责。我们大多数人都知道 KISS 原则，但是每当需要在代码中多实现一个特性时，我们经常会忘记它。

![](img/c219f994b9ef9cae834a5c795461dc31.png)

Photo by [Dan Gold](https://unsplash.com/photos/yOcNQ54cA4w?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

# 是什么阻碍了优雅？

紧张的时间限制是优雅解决方案的主要杀手。简单不等于“快”或“容易”通常情况下，是相反的。简单的设计需要对所有预期的用例进行仔细的思考和分析，以提出一个清晰简洁的想法。

虽然您可以在没有适当设计和没有对可能的功能范围进行太多考虑的情况下构建软件，但是维护代码将是一件痛苦的事情。因为每次出现一个 bug，它很可能是由您缺乏适当的边缘情况处理引起的。如果没有一个函数应该或不应该做什么的明确范围，您很可能会在代码的某个地方添加一个条件子句。

但是在十次这样的情况之后，你的代码看起来不再像最初发布时那样了。现在是一堆模糊的条件，没有人知道应用程序的预期行为是什么。有时候，乍一看完全不相关的东西的变化会导致不同子系统的错误。然后每个人都抓抓头，悄悄地恢复最后一次提交。是时候采用另一种方法了。

# 为什么复杂更容易实现

优雅的解决方案需要专注和观察。他们需要分析、与客户的良好沟通以及深思熟虑的范围。如果你没有时间弄清楚所有可能的输入参数，或者如果你只是因为缺乏适当的设计而不能弄清楚它们，并且客户不确定他将如何使用产品，那么你最终会在解决问题的过程中跳过一些步骤。

但是还有另一个复杂性的来源。而这一个更难克服。它是大量现成的组件和抽象等着你去使用。前端开发尤其脆弱。使用普通的 HTML5、CSS 和 Javascript 编写一个吸引人的 web 应用程序并不是一件容易的事情。

这就是为什么我们决定委托第三方来实施所有的基本细节，这样我们就可以专注于重要的事情。我们选择一个框架，寻找更多的模块，最终得到臭名昭著的 3GB `node_modules`目录。只要这些模块中的抽象层符合我们使用它们的方式，一切都应该没问题。通常情况下，我们无法就框架中的某些内容达成一致，因此我们最终会编写一个糟糕的解决方案或一个特例来让它工作。

我希望我能分享一个可行的方法来处理泄漏的抽象，但是我不能。我没有！但是我知道我们不能停止使用它们。这将是巨大的资源浪费。

![](img/0e33af4b62b15c493ec638eff47ac6c0.png)

Photo by [John Barkiple](https://unsplash.com/@barkiple?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 如何保持简单

**你写手机应用？**与其创建自己的后端，不如使用后端即服务，比如 [Firebase](https://firebase.google.com/) 。

**你想创建一个登陆页面或博客吗？**使用静态站点生成器(如 [Jekyll](https://jekyllrb.com/) 或 [Gatsby](https://www.gatsbyjs.org/) )和静态文件托管(如 [Netlify](https://www.netlify.com/) )。

你想要一个 CMS 吗？ [检出 Contentful 或 DatoCMS。](https://www.datocms.com/)

厌倦了跟踪 web 服务的 SSL 证书吗？使用自动复习器，如 [AWS ACM](https://aws.amazon.com/certificate-manager/) 、 [Traefik](https://aws.amazon.com/certificate-manager/) 、 [Caddy](https://caddyserver.com/) 或[僵尸 Nginx](https://github.com/dlabspl/zombie-nginx) 。

**希望您的代码“按需”运行，而不是支付固定价格的 VPC 实例？**尝试功能即服务(或无服务器)解决方案。

哦，对了，除非你有非常精致的品味，否则**使用 AWS RDS/Cloud SQL/Azure SQL 这样的托管数据库服务是个不错的主意。**

实现简单就是减少组件的数量而不是积累。

# 保持简单的好处

更多的移动部件通常意味着更多的错误。在一个经典的博客引擎示例中，您可以预期*至少会有*以下问题:

*   数据库性能差。
*   博客引擎性能差(内存/CPU 不足)。
*   网络吞吐量不足。
*   磁盘空间不足。
*   您为最悲观的用例过度配置，因此您在其余时间超额支付了费用。
*   部署失败。
*   数据库迁移可能会失控。
*   VPC 可能会随着备份一起消失得无影无踪。

我不是在这里卖给你一个[保护费](https://www.youtube.com/watch?v=DNj1dXi-z0M)，但是如果你选择一个静态站点生成器和一个静态文件托管，这些问题大部分都会消失。您还获得了一个免费的连续交付管道，其中每个源代码更改都会导致页面上可见的更改。既简单又功能齐全！

# 寻找灵感

有些人非常认真地对待 KISS 原则，并将其应用到他们的项目中。如果你这样做了，我很高兴听到你关于通过简单来保持解决方案优雅的建议。请随意使用下面的评论部分来分享一些优雅项目的伟大范例，简单减少的建议，或者关于应该避免什么的警告。

> 如果你喜欢我创造的东西，考虑订阅 Bit 更好。这是一个社区时事通讯，推荐书籍、文章、工具，有时还有音乐。

*原载于 2018 年 8 月 24 日*[*https://www.iamondemand.com*](http://www.iamondemand.com/blog/the-kiss-of-simplicity/)*。*