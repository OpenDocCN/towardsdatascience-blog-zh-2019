# 虚拟助手中技能发现的演变

> 原文：<https://towardsdatascience.com/the-evolution-of-skill-discovery-in-virtual-assistants-229c9363ac58?source=collection_archive---------20----------------------->

![](img/b82d0ae86de7f143f29e7ce427d803b0.png)

Skill discovery in virtual assistants (Illustration: [Luniapilot](http://www.luniapilot.com/))

这篇博文是来自 [**Embodied AI**](http://www.embodiedai.co/) **，**的一期，由 [TwentyBN](https://20bn.com/) 撰写的关于 AI 化身、虚拟生物和数字人类背后的最新新闻、技术和趋势的双周刊。

上周五，化身 AI 在几个朋友的公寓里喝着酒，突然瞥见角落里放着一个新买的 Google Home。他对所有人工智能的好奇心占据了他的头脑，他问他们用新技术做什么。“除了播放音乐，”他们回答说，“我们还问它时间、天气，以及……开灯和关灯。”

和很多消费者一样，Embodied AI 的朋友们遇到了基于语音的智能技术中的 ***技能发现问题*** 。虽然智能扬声器可以做的不仅仅是报告天气、开灯和点餐——仅谷歌助手就有 4200 件事情——但它不能有效地传达它可以帮助我们的无数方式。由于技能发现是使虚拟助手更有效和更像人类的一个关键因素，我们将探索技能发现的演变，包括它的挑战，目前的进展，以及它将如何在未来继续发展。

# 技能发现:2 项挑战和 8 项建议

[**Benedict Evans**](https://www.ben-evans.com/)(Andre essen Horowitz)将智能音箱中的技能发现称为[基本的 UX 难题](https://www.ben-evans.com/benedictevans/2019/1/29/is-alexa-working):例如，Alexa 的纯音频界面很方便，直到你希望它向用户逐一背诵其 [80，000 项技能](https://voicebot.ai/2019/01/31/amazon-announces-80000-alexa-skills-worldwide-and-jeff-bezos-earnings-release-quote-focuses-solely-on-alexa-momentum/)。

有两个因素使得技能发现特别具有挑战性:

*   可用性:虚拟助理的技能正在迅速扩展。 [Voicebot](https://voicebot.ai/2019/02/15/google-assistant-actions-total-4253-in-january-2019-up-2-5x-in-past-year-but-7-5-the-total-number-alexa-skills-in-u-s/) 报告称，自 2018 年以来，谷歌助手的能力增长了 2.5 倍，达到 4253 个动作，Alexa 的能力增长了 2.2 倍，达到近 8 万个动作。
*   **启示:**用户不确定他们的虚拟助手有什么能力，导致期望偏差，他们中的许多人忽视了使用互联网来了解他们助手的技能的全部范围。

![](img/25a3e33cacb03730116fbb056004d9c7.png)

Alexa and her many skills (Credit: Ryen W. White, Communications of the ACM)

## 虚拟演讲者帮助人们发现他们所能做的 8 种方式

在这篇[文章](http://ryenwhite.com/papers/WhiteCACM2018a.pdf)， [**Ryen W. White**](https://www.microsoft.com/en-us/research/people/ryenw/) (微软研究院)列出了虚拟扬声器的 8 项改进，以便用户可以更容易地发现他们日常需要的新技能:

*   **积极主动:**让虚拟助理主动利用他们的技能吸引用户，而不是被动的、用户发起的交互。
*   **时机决定一切:**在需要的时候向用户提供技能建议，以确保这些能力在将来更容易被记住。
*   **使用情境和个人信号:**利用用户情境和个人信号的组合，包括长期习惯和模式
*   **检查附加信号:**还不可访问的上下文和个人信息，例如只能用视觉观察的人类活动，是用于个性化推荐的未开发的机会。
*   **考虑隐私和效用:**在正确的时间提供正确的帮助，并通过推荐解释主动将其归因于许可的数据访问。
*   **允许多个推荐:**当推荐模型的置信度低于某个阈值时，建议多个技能，在该阈值时通常会建议一个确定的技能。
*   **利用配套设备:**允许通过 WiFi 或蓝牙连接访问各种屏幕，如智能手机、平板电脑或台式电脑，以丰富上下文并帮助提供更相关的技能建议。
*   **支持持续学习:**根据以前的活动模式提出新的技能。

# 提高技能发现的研究进展

最近，亚马逊推出了[**Alexa Conversations**](https://developer.amazon.com/blogs/alexa/post/44499221-01ff-460a-a9ee-d4e9198ef98d/introducing-alexa-conversations-preview)，这是一种深度学习方法，允许开发人员以更少的努力、更少的代码行和更少的训练数据更有效地提高技能发现。虽然仍处于“预览”阶段，但 Alexa Conversations 已经[在为智能扬声器培养技能的开发人员中引起了相当大的兴奋](https://voicebot.ai/2019/06/06/alexa-conversations-to-automate-elements-of-skill-building-using-ai-and-make-user-experiences-more-natural-while-boosting-discovery/)。

![](img/9fe1cb86d5ad524ff8cbeb61e1237e70.png)

Alexa Conversations claims to utilize machine learning to predict a user’s true goal from the dialogue and proactively enable the conversation flow across skills. (Credit: [Alexa Blog](https://developer.amazon.com/blogs/alexa/post/9615b190-9c95-452c-b04d-0a29f6a96dd1/amazon-unveils-novel-alexa-dialog-modeling-for-natural-cross-skill-conversations))

本质上，Alexa Conversations 旨在建立一个更自然和流畅的互动，在一个单一的技能范围内，Alexa 和它的用户之间。在未来的版本中，该软件有望在一次对话中引入多种技能。它还声称能够处理模糊的引用，例如，“附近有意大利餐馆吗？”(哪里附近？)，以及从一种技能过渡到另一种技能时的上下文保留，例如在建议附近的餐馆时记住某个电影院的位置。

在 6 月份亚马逊的 re:MARS AI 和 ML 大会上，Alexa 副总裁兼首席科学家 Rohit Prasad 提到 [Alexa Conversations 的机器学习功能](https://developer.amazon.com/blogs/alexa/post/9615b190-9c95-452c-b04d-0a29f6a96dd1/amazon-unveils-novel-alexa-dialog-modeling-for-natural-cross-skill-conversations)可以帮助它从对话的方向预测客户的真实意图和目标，从而在对话过程中主动实现多种技能的流动。如果这些承诺得到满足，与 Alexa 的命令-查询交互肯定会开始感觉更像自然的人类交互。

# 未来:看见和具体化的虚拟助手

Alexa 团队取得的进展确实令人兴奋，但对话式人工智能并不是唯一有改进空间的领域。在 Embodied AI，我们支持将对话式人工智能和视频理解集成到拟人化的具体化助手中，这是通过在现有的扬声器界面上添加摄像头和屏幕实现的。

随着我们对自然语言处理和计算机视觉的理解不断进步，没有理由将虚拟助手局限于音频。最近发布的[亚马逊 Echo Show 5](https://www.theverge.com/2019/5/29/18643451/amazon-echo-show-5-alexa-screen-price-availability-89) 、[脸书门户](https://portal.facebook.com/)和[谷歌的 Nest Hub Max](https://www.theverge.com/2019/5/7/18301161/google-nest-hub-max-camera-home-announcement-io-2019-keynote) ，所有这些都带有摄像头和屏幕，已经预示着该行业向虚拟助手的发展，有一天虚拟助手可以看到和被看到。人们可以合理地推测，大型科技公司正在致力于视觉化和具体化的虚拟助理，以在不久的将来取代他们的智能扬声器。这是他们现有产品线的自然延伸。

具有照相机、屏幕和拟人化实施例的虚拟助理的好处包括:

*   **多模式 I/O:** 配备语音 I/O 和视频 I/O 的虚拟助理不再局限于音频，而是拥有更高的智能和更具吸引力的图形用户界面。
*   **改进的技能发现体验:**利用计算机视觉捕捉目前纯音频设备未利用的上下文和个人信号，允许从用户发起的交互过渡到主动协助。
*   **陪伴而不是仆人:**有了数字化、类人的身体，虚拟助理将不再被视为仆人，而是帮手。虽然这不会直接改善技能发现，但它丰富了虚拟助理的整体体验。

![](img/b4b35bf7d3dccae4e16158e91e83514d.png)

Roland Memisevic, TwentyBN’s CEO, believes that computer vision, by unlocking context awareness for virtual assistants, will shift the assistant paradigm from query-response to memory-infused companionship. (Credit: LDV Capital)

TwentyBN 的首席执行官 Roland Memisevic 设想了一个未来，我们与虚拟助手的对话将不会像现在的智能扬声器那样感觉像打电话:

> *"* 化身不一定需要唤醒词，但可以一直在这里，看到和听到，特别是当他们是边缘驱动的并且没有隐私问题时。使用计算机视觉来解锁虚拟助手的上下文感知，我们将把助手范式从查询-响应转变为注入记忆的陪伴。问我们未来的同伴他们有什么技能，就像问你最好的朋友他们是否呼吸氧气一样可笑。

# *“嘿，谷歌，让我们结束吧！”*

*或许在不久的将来，在另一个周五的晚上，化身人工智能将重访他朋友的屋顶公寓，并发现一个新的虚拟助理，一个不仅精通对话，而且拥有理解上下文和识别语言无法捕捉的需求的眼睛的虚拟助理。也许它甚至可能有一个数字化的人体，成为一个虚拟的朋友，分享并增加柏林仲夏夜晚的活跃气氛。*

*化身 AI 伸手从他的酒杯中抿了一口，当发现它是空的时，听到他朋友的谷歌助手从角落里喊道，“我想我们准备好再来一瓶白葡萄酒了！”*

**作者* [*那华*](https://twitter.com/nahuakang) *编辑* [*大卫*](https://twitter.com/david_greenberg)*[*将*](https://www.linkedin.com/in/william-hackett-95950517a/)*[*莫里茨*](https://twitter.com/muellerfreitag) *，以及* [*萨克*](https://medium.com/@isaacwu_10508) *。插图由* [*氹欞侊*](http://www.luniapilot.com/) *组成。****

***[**Embodied AI**](http://www.embodiedai.co/)**是一份关于 AI 化身和虚拟生物背后的最新新闻、技术和趋势的双周刊。**订阅下面:*******

*****[](http://www.embodiedai.co/) [## 人工智能化身时事通讯

### 具体化的人工智能是权威的人工智能化身时事通讯。报名参加最新新闻、技术…

www.embodiedai.co](http://www.embodiedai.co/)*****