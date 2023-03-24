# 用 OCR 防止同形异义字符攻击。

> 原文：<https://towardsdatascience.com/homoglyph-attack-prevention-with-ocr-a6741ee7c9cd?source=collection_archive---------12----------------------->

![](img/f952b91c95e9ede3202ebdb250db5d0d.png)

TLDR；这篇文章描述了什么是[同形异义攻击](https://en.wikipedia.org/wiki/Homoglyph)以及如何用认知服务来防止它们。

# 入门指南

这个故事的代码可以在 github 上找到。

[](https://github.com/aribornstein/HomoglyphAttackPreventionService) [## aribornstein/homoglyphatckpreventionservice

### 一种防止同形异义攻击的服务。通过以下方式为 aribornstein/homoglyphatckpreventionservice 的发展做出贡献…

github.com](https://github.com/aribornstein/HomoglyphAttackPreventionService) 

Azure 上的一键式部署说明可以在下面找到。

# 什么是同形攻击？

![](img/c88e1a80d578d8c17e4f3401f71948af.png)

在[正字法](https://en.wikipedia.org/wiki/Orthography)和[印刷术](https://en.wikipedia.org/wiki/Typography)中，**同形异义字**是两个或更多[字符](https://en.wikipedia.org/wiki/Character_(computing))中形状看起来相同或非常相似的一个。通俗地说，同形字就是任何看起来与另一个字符相似的字符，比如上图**中的 **S** 和 **$** 。**

语言模型经常容易受到使用同形符号的混淆攻击，这是由于它们编码文本的方式。例如，在 Unicode 和 Ascii 中，相同的字符代码在不同的字体中看起来不同，模型将努力学习它们的相似性。

 [## 面向初学者的字符编码

### 什么是字符编码，我为什么要关心？

www.w3.org](https://www.w3.org/International/questions/qa-what-is-encoding) 

为了说明这一点，让我们来看看下面的短语:

> 我从令人敬畏的云情感分析模型中得到了$hitty 结果。

上面这句话清楚地表明了消极情绪。单词 *$hitty* 是亵渎单词*shipped*的同音字混淆。

让我们看看互联网上最流行的四种云情感分析服务是如何处理这种攻击的。

![](img/51d367474545b4b6ff8cc4fded1ceb44.png)

Clockwise, Azure Text Analytics and GCP Natural Language correctly classify the original sentiment but fail on the obfuscated text, IBM Watson fails to correctly classify sentiment of either text and AWS Comprehend does not provide a demo with out an AWS account but also fails on the example sentence.

正如我们所看到的，Azure 文本分析和 GCP 自然语言正确地对原始情感进行了分类，但都没有对模糊文本进行分类。IBM Watson 未能正确地对两种文本的情感进行分类。AWS comprehensive 不提供没有 AWS 订阅的演示，也不能使用例句。

虽然上面的$和 S 的例子可能看起来有些武断，但维基百科下面的拉丁字母和西里尔字母展示了这种攻击是如何有效和难以检测的。

![](img/ca1364bcfc4fd71846faa656ee8f037b.png)

这给用例带来了各种各样的问题，在这些用例中，这种攻击可以利用应用程序或对应用程序造成伤害，例如试图躲避假新闻检测器的机器人。

# 如何防范同形异义攻击？

![](img/2a5f9ea2315f9603ef189309c4fd000c.png)

Azure Computer Vision Correctly Reveals the Homoglyph when the text is represented as an image.

在与我来自 BIU NLP 实验室的朋友 Amit Moryossef 交谈后，我们意识到我们可以使用 OCR 系统来防止同形异义字符攻击。

使用 Azure 计算机视觉服务，我用上面的句子测试了这个理论，它正确地使用图像域上下文从同形词 **$hitty** 中提取单词**shipped**。

使用这种能力，我编写了下面的开源容器服务，它将:

1.  将一个给定的文本作为输入

2.将文本转换为图像

3.使用 OCR 处理图像

4.返回删除了同形异义字符的正确文本。

Docker 服务是云不可知的。为了方便起见，我向 Azure 提供了一个单击部署选项。

如果您已有 Azure 订阅，您可以通过单击下面的按钮来自动部署服务。

![](img/997f54de93f84163e351e4b522c51a65.png)

[Click here to get started!](https://portal.azure.com/#create/Microsoft.Template/uri/https%3A%2F%2Fraw.githubusercontent.com%2Faribornstein%2FHomoglyphAttackPreventionService%2Fmaster%2Fazuredeploy.json)

否则你可以在这里获得一个[免费 Azure 账户，然后点击上面的部署按钮。](https://azure.microsoft.com/offers/ms-azr-0044p/?WT.mc_id=medium-blog-abornst)

[](https://azure.microsoft.com/offers/ms-azr-0044p/?WT.mc_id=medium-blog-abornst) [## Azure 免费试用版|微软 Azure

### 开始您的免费 Microsoft Azure 试用，并获得 200 美元的 Azure 点数，以任何方式使用。运行虚拟机…

azure.microsoft.com](https://azure.microsoft.com/offers/ms-azr-0044p/?WT.mc_id=medium-blog-abornst) 

如果您有任何问题、评论或希望我讨论的话题，请随时在 [Twitter](https://twitter.com/pythiccoder) 上关注我。再次感谢 Amit Moryossef 和 BIU NLP 实验室的惊人灵感，以及 Iddan Sachar 在他的帮助下调试 ARM 的一键式部署。

# 使用服务

要使用该服务，只需向该服务发送一个最多 200 个字符的 URL 编码查询字符串，该服务非常适合验证推文。下面是一个使用 curl 的调用示例，请务必使用您自己的服务端点。

![](img/7806f33f7d35e6332d505f4737678815.png)

# 后续步骤

虽然该服务在删除同形异义词方面表现很好，但仍有少数情况下会失败。

![](img/ee4b367170b67b6c07fbc3c66f1ed1b1.png)

Example Failure Case

未来的工作将探索使用一个更加定制的方法来解决这个问题，但这种方法非常适合非常小的努力。

# 额外资源

[](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/#text?WT.mc_id=blog-medium-abornst) [## 使用计算机视觉 API | Microsoft Azure 进行图像处理

### 使用 Microsoft Azure 的计算机视觉 API 分析图像并提取您需要的数据。查看手写 OCR…

azure.microsoft.com](https://azure.microsoft.com/en-us/services/cognitive-services/computer-vision/#text?WT.mc_id=blog-medium-abornst) [](https://azure.microsoft.com/en-us/services/app-service/containers/?WT.mc_id=blog-medium-abornst) [## 应用服务—容器的 Web 应用| Microsoft Azure

### 带上您自己的容器，使用 web app 在几秒钟内部署到应用服务，作为运行在 Linux 上的 Web App…

azure.microsoft.com](https://azure.microsoft.com/en-us/services/app-service/containers/?WT.mc_id=blog-medium-abornst) [](https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/?WT.mc_id=blog-medium-abornst) [## 文本分析 API | Microsoft Azure

### 使用 Azure 文本分析 API 将非结构化文本转化为有意义的见解。获取情感分析，关键词…

azure.microsoft.com](https://azure.microsoft.com/en-us/services/cognitive-services/text-analytics/?WT.mc_id=blog-medium-abornst) 

# 关于作者

亚伦(阿里) 是一个狂热的人工智能爱好者，对历史充满热情，致力于新技术和计算医学。作为微软云开发倡导团队的开源工程师，他与以色列高科技社区合作，用改变游戏规则的技术解决现实世界的问题，然后将这些技术记录在案、开源并与世界其他地方共享。