# 深度造假——今日新闻热门话题

> 原文：<https://towardsdatascience.com/deepfake-trending-topic-in-the-news-today-f2607a28fb8f?source=collection_archive---------29----------------------->

## 是时候质疑你看到的每一张照片和视频了！

![](img/30638a0b35429de19515174c4bf579bd.png)

Photo of a woman (Photo by [Andrey Zvyagintsev](https://unsplash.com/@zvandrei?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText))

> "[亚马逊联合脸书和微软打击 deep fakes](https://thenextweb.com/artificial-intelligence/2019/10/22/amazon-joins-facebook-and-microsoft-to-fight-deepfakes/)"—2019 年 10 月 22 日的下一个网络
> 
> “[对 deepfakes 的战争:亚马逊用 100 万美元的云信用支持微软和脸书](https://www.zdnet.com/article/war-on-deepfakes-amazon-backs-microsoft-and-facebook-with-1m-in-cloud-credits/)”—ZDNet，2019 年 10 月 22 日
> 
> “[什么是‘深度假货’，它们有多危险](https://www.cnbc.com/2019/10/14/what-is-deepfake-and-how-it-might-be-dangerous.html)”—《美国消费者新闻与商业频道》2019 年 10 月 13 日报道
> 
> [“人工智能无法保护我们免受 deepfakes 的影响，新报告认为”](https://www.theverge.com/2019/9/18/20872084/ai-deepfakes-solution-report-data-society-video-altered) — The Verge 于 2019 年 9 月 18 日

我注意到“ **DeepFake** ”这个词一次又一次地出现在我的新闻提要中，在我第六次或第七次遇到它之后，我开始想这是不是我应该知道的事情。我的意思是，如果像亚马逊、脸书和微软这样的大型科技公司正在共同努力打击 DeepFake，而美国消费者新闻与商业频道正在报告其危险，那一定是很严重的事情。

事实证明，使用 *DeepFake* ，你可以创建一个甚至不存在的人的面部图像，操纵视频来交换人们的面部，并使用某人的声音编造演讲，甚至无需他们同意。试试这个 hispersonedoesnotexist.com 网站，你会大吃一惊。这个网站生成一个甘想象的人的形象(根据网站左下角的注释)。下面是该网站生成的一些人脸图像。我打赌你不会猜到这些图像是假的，我也不会。

![](img/0902b126acd8e1d49f0220867b7339b4.png)![](img/085ceb1765bbc1527d8fc77f505d10a8.png)![](img/5e2623ed2803066a7caddec5c25b3d53.png)

Face images created using thispersonedoesnotexist.com

![](img/521874032ea58cad895f1be362bc05d7.png)

Note on thispersonedoesnotexist.com

*DeepFakes* 是由一种被称为生成对抗网络(GANs)的深度学习技术创建的，其中使用了两种机器学习模型来使假货更加可信。通过研究一个人的图像和视频，以训练数据的形式，第一个模型创建一个视频，而第二个模型试图检测它的缺陷。这两个模型携手合作，直到他们创造出一个可信的视频。

当谈到无监督学习时，DeepFake 打开了一个全新的世界，这是机器学习的一个子领域，机器可以学习自学，并且有人认为，当谈到自动驾驶汽车检测和识别道路上的障碍以及 Siri、Cortana 和 Alexa 等虚拟助手学习更具对话性时，它有很大的前景。

真正的问题是，像其他技术一样，它有可能被滥用。利用 *DeepFake* 作为报复的一种形式，通过在色情中使用它，操纵政治家的照片和影响政治运动只是冰山一角。

> 它引发了一个严重的问题，即我们是否可以信任以图像、视频或声音形式呈现给我们的任何数字信息。

据 [BBC 新闻](https://www.bbc.com/news/technology-49961089)， *DeepFake* 视频在短短九个月内翻了一番，其中 96%是色情性质的。这对 *DeepFake* 的受害者意味着什么？谁应该为创建和分享这种虚假媒体并诽谤另一个人负责？

科技巨头们意识到了 *DeepFake* 的危害，已经主动站出来反对 *DeepFake* 的滥用。脸书和微软在 9 月份宣布了一项竞赛[deep face Detection Challenge(DFDC)](https://deepfakedetectionchallenge.ai/)，以帮助检测 *DeepFakes* 和被操纵的媒体，脸书正在通过创建一个大型的 *DeepFake* 视频数据集来提供帮助，亚马逊正在为这项价值 100 万美元的挑战提供云资源。这项比赛将于 12 月启动，总奖金为 1000 万美元。

另一方面，谷歌也在创建一个 DeepFake 视频数据集，以便 Munchin 技术大学、那不勒斯大学 Federico II 和埃尔兰根-纽伦堡大学的研究人员可以将其用于 FaceForensics 项目，这是一个检测被操纵的面部图像的项目。

总而言之，技术正一天天变得越来越聪明，DeepFake 的创造证明了机器已经发展到这样一个水平，它们可以合成照片、视频和语音，甚至我们都无法将它们与真的区分开来。DeepFake 有一定的好处，其中之一是为自动驾驶汽车的改进创造更多的训练数据。但是，就像所有其他技术一样，人们正在滥用它来制作虚假视频，并在社交媒体上分享错误信息。科技巨头们已经加紧对付这一问题，希望在不久的将来，我们将拥有一个可以检测深度造假的系统。

我们能做些什么来帮助防止 DeepFake 被滥用，并确保它以安全的方式使用？ ***在下面留下你的想法作为评论。***

[**点击这里**](https://medium.com/@sabinaa.pokhrel) 阅读我其他关于 AI/机器学习的帖子。

*跟我上* [***中***](https://medium.com/@sabinaa.pokhrel)**和/或*[***LinkedIn***](https://www.linkedin.com/in/sabinapokhrel)***。****

***来源:***

*[](https://www.theverge.com/2019/9/20/20875362/100000-fake-ai-photos-stock-photography-royalty-free) [## 100，000 张人工智能生成的免费头像让股票照片公司受到关注

### 使用人工智能来生成看起来令人信服，但完全虚假的人的照片变得越来越容易。现在…

www.theverge.com](https://www.theverge.com/2019/9/20/20875362/100000-fake-ai-photos-stock-photography-royalty-free) [](https://www.theverge.com/tldr/2019/2/15/18226005/ai-generated-fake-people-portraits-thispersondoesnotexist-stylegan) [## ThisPersonDoesNotExist.com 利用人工智能生成无尽的假脸

### 人工智能产生虚假视觉效果的能力还不是主流知识，但一个新的网站…

www.theverge.com](https://www.theverge.com/tldr/2019/2/15/18226005/ai-generated-fake-people-portraits-thispersondoesnotexist-stylegan) [](https://www.theverge.com/2019/9/18/20872084/ai-deepfakes-solution-report-data-society-video-altered) [## 新报告称，人工智能无法保护我们免受深度欺诈

### 来自数据和社会的一份新报告对欺骗性修改视频的自动化解决方案提出了质疑，包括…

www.theverge.com](https://www.theverge.com/2019/9/18/20872084/ai-deepfakes-solution-report-data-society-video-altered) [](https://www.zdnet.com/article/deepfakes-for-now-women-are-the-main-victim-not-democracy/) [## Deepfakes:目前，女性，而不是民主，是主要的受害者

### 虽然 2020 年美国总统选举让立法者对人工智能生成的假视频感到紧张，但一项由…

www.zdnet.com](https://www.zdnet.com/article/deepfakes-for-now-women-are-the-main-victim-not-democracy/) [](https://www.forbes.com/sites/daveywinder/2019/10/08/forget-2020-election-fake-news-deepfake-videos-are-all-about-the-porn/#7ea9952563f9) [## 忘掉假新闻吧，Deepfake 视频其实都是关于未经同意的色情内容

### “深度造假”对公平选举，特别是 2020 年美国总统大选的威胁越来越大…

www.forbes.com](https://www.forbes.com/sites/daveywinder/2019/10/08/forget-2020-election-fake-news-deepfake-videos-are-all-about-the-porn/#7ea9952563f9)  [## Deepfakes 可能会打破互联网

### 我们可能已经在一个深度造假的世界里游泳了，而公众甚至不会知道。毕竟，我们要怎么做呢？那些…

www.cpomagazine.com](https://www.cpomagazine.com/cyber-security/deepfakes-could-break-the-internet/) [](https://www.zdnet.com/article/war-on-deepfakes-amazon-backs-microsoft-and-facebook-with-1m-in-cloud-credits/) [## 向假货宣战:亚马逊用 100 万美元的云信用支持微软和脸书

### AWS 全力支持即将到来的 Deepfake 检测挑战赛，这是一项奖金高达 1000 万美元的比赛…

www.zdnet.com](https://www.zdnet.com/article/war-on-deepfakes-amazon-backs-microsoft-and-facebook-with-1m-in-cloud-credits/) [](https://www.cnbc.com/2019/10/14/what-is-deepfake-and-how-it-might-be-dangerous.html) [## 什么是“深度假货”,它们有多危险

### “Deepfakes”被用来在虚假视频中描绘他们实际上并没有出现的人，并可能潜在地影响…

www.cnbc.com](https://www.cnbc.com/2019/10/14/what-is-deepfake-and-how-it-might-be-dangerous.html) [](https://thenextweb.com/artificial-intelligence/2019/10/22/amazon-joins-facebook-and-microsoft-to-fight-deepfakes/) [## 亚马逊联合脸书和微软打击 deepfakes

### Deepfakes 今年遇到了严重的问题，大公司现在开始关注。亚马逊宣布…

thenextweb.com](https://thenextweb.com/artificial-intelligence/2019/10/22/amazon-joins-facebook-and-microsoft-to-fight-deepfakes/) [](https://www.csoonline.com/article/3293002/deepfake-videos-how-and-why-they-work.html) [## Deepfake 视频:它们是如何工作的，为什么工作，以及有什么风险

### Deepfakes 是假的视频或音频记录，看起来和听起来就像真的一样。曾经是…的辖区

www.csoonline.com](https://www.csoonline.com/article/3293002/deepfake-videos-how-and-why-they-work.html) [](https://www.bbc.com/news/technology-49961089) [## Deepfake 视频'九个月翻倍'

### 新的研究显示，所谓的深度假视频创作激增，在线数量几乎…

www.bbc.com](https://www.bbc.com/news/technology-49961089)*