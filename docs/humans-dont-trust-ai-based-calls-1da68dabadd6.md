# 人类不信任基于人工智能的呼叫

> 原文：<https://towardsdatascience.com/humans-dont-trust-ai-based-calls-1da68dabadd6?source=collection_archive---------24----------------------->

## 同样糟糕的客服，现在有了人工智能

## 人类代表也可能很糟糕，但不要试图用人工智能欺骗人们

![](img/2b95351fba0cdc42138c6cfba94d2461.png)

客服的 AI“管用”，但人家不喜欢。我最近为了一份关于[对话式人工智能和消费者信任状态](https://clutch.co/developers/artificial-intelligence/resources/conversational-ai-voice-technology-survey)的报告，与[**【clutch.co】**](http://clutch.co)的员工就此进行了深入的交谈。

研究结果很有趣。例如:

“73%的人不信任谷歌 Duplex 等人工智能语音技术来打简单的电话，尽管随着使用的增加，信任可能会建立起来”

在 B2C 领域，我对此有更多的想法，我想在这里提出来。开始了。

在**星巴克**，你支付额外的费用，因为他们是“高端”的，并且做一些事情，比如记住你的名字并写在你的杯子上。但是在许多商店，给你饮料的人和为你点菜的人不是同一个人。因此，这导致了这样一种情况，服务器在人群中寻找“丹尼尔”，希望找到正确的那个。这真的比一个无名无姓的机器以半价卖给你同样的 goop 好吗？

更让我对最近的客户服务经历感到愤怒的是，最近我在蒂姆·霍顿的汽车通道上，当我在开车离开之前把信用卡放回钱包时，窗口的人说了声“再见”，这是一个毫不隐晦的信号，促使我继续前进。顺便说一下，正好有一辆车在我后面等着，所以我觉得没有必要在付完饮料钱后 1 秒钟就开走。他们想让我像抢劫一样离开。

因此，人性化的客户服务可能会非常令人恼火。为什么不用基于人工智能的点餐系统来代替这些人呢？技术是存在的。考虑一下 **Google Duplex** ，它通过像真人一样的行为，代表 Google Assistant 用户给商店打电话预约或订购披萨:

Google Duplex: A.I. Assistant Calls Local Businesses To Make Appointments

当 Duplex 问世时，许多评论者强烈反对谷歌通过伪装成人来欺骗人们。他们是对的。我喜欢那一年[谷歌停止使用座右铭“不要作恶”](https://gizmodo.com/google-removes-nearly-all-mentions-of-dont-be-evil-from-1826153393)他们开始推出双面打印。谷歌可能在欺骗比萨饼制造商和发廊，但我更担心的是那些真正的坏演员将这项技术用作现代自动呼叫机来吸引消费者。我认为[莱奥·格雷布勒](https://medium.com/u/136fa39ffeba?source=post_page-----1da68dabadd6--------------------------------)说得最好，这远远超过了欺骗消费者:即使是用不幸的人作为测试假人来训练这些系统也是非常粗略的。我曾经和 Leor 一起研究过语音交互代理，从早期的文本到语音和语音到文本的转换，整个领域都在以疯狂的速度发展，以至于普通人都没有意识到他们正处于语音交互代理的新时代。

[Leor Grebler](https://medium.com/u/136fa39ffeba?source=post_page-----1da68dabadd6--------------------------------) talking about his voice interaction technology way back in 2012\. It’s now 2019!

与仅仅一两年前的系统语音质量相比，这项技术非常非常强大。看看以下来自 VoiceNet 的语音样本:[https://deep mind . com/blog/wave net-generative-model-raw-audio/](https://deepmind.com/blog/wavenet-generative-model-raw-audio/)

事情变得越来越有趣，这对消费者来说不会没有痛苦。举例来说，一位密友在过去几周有如下经历:

一名销售人员代表她的银行打电话给她。这是一个真正的销售电话，而不是一个骗局。与她通话的代理人有着人类应有的语调和热情，嘲笑她的 T2 闲聊，并停下来获取更多信息，正如你所料，但当她被问及一个人们通常不会问的奇怪问题时，通话速度慢了很多，然后一名带有印度口音的代理人加入了通话，接管了通话。该人士马上明确表示:“你一直在和我说话，但我们使用人工智能软件来克服口音障碍”。我的朋友很奇怪，挂了电话。这完全违反了。更糟糕的是，人工代理可能是同时拨打 50 个电话的备用系统。也许更多。

在我家，我们有一位年长的亲戚，自 2012 年以来一直在不要打电话的名单上，她在早上 6 点、7 点，甚至晚上 11 点通过她的座机接到机器人电话。这些都是老派录制的 robocalls，但是真的很让人气愤。一点都不酷。基本都是机械化骚扰。像这样的系统只会变得更糟，因为坏人得到了人工智能语音代理的东西。我在亚马逊上看了一下，有一些阻止垃圾电话的硬件设备，但切断电话线和反垃圾手机应用程序是更好的策略。

这是我拼凑的一个小东西，用来证明这项技术是切实可行的。它只会持续一个月左右，但它证明了这一点。打电话:**+1 213–224–2234**，你会听到我制作的一个基本会话代理。我给代理打电话的录音嵌在这里:

Me calling an artificial intelligence system on the phone, to show the basic capability.

像这样的系统有很多问题。请注意，我演示的系统预先告诉您它只能做某些事情，因此用户(调用者)更清楚会发生什么和正在发生什么。

在我与 Darryl Praill 的谈话中，我们讨论了人工智能如何不会取代销售人员，因为消费者讨厌被欺骗，而且现在的工具经常没有被正确应用(在道德上？)来帮助消费者。

My talk with [Darryl Praill](https://medium.com/u/6428b8b4730c?source=post_page-----1da68dabadd6--------------------------------) of VanillaSoft.com fame, about artificial intelligence in sales.

我提到的[新调查报告](https://clutch.co/developers/artificial-intelligence/resources/conversational-ai-voice-technology-survey)的主要观点是:“消费者对谷歌 Duplex 等对话式人工智能工具表现出不信任，但随着使用的增加，信任将会建立。”离合器是华盛顿特区的一家 B2B 评级和评论公司。为了获得数据，离合器与全球数字解决方案公司 Ciklum 合作，调查了 500 多人对对话式人工智能的信任程度。调查告诉我们这项技术将会被使用。很多。所以系好安全带接受消费者的投诉。

一个机器人打电话卖东西已经够烦人的了，但这种人工智能呼叫技术并不像一个非法来电者试图诈骗你。相反，这是一个恼人的信任危机业务工具，可以扩展，并且很容易证明成本合理。这是伪装的机器人电话，还打了类固醇。即使人类介入捕捉系统不知道该做什么的情况，使用技术来欺骗人们也不是一个好主意。信任将是一个问题。

我朋友的爸爸告诉了她一个巧妙的方法。简单地问打电话的人是不是人。许多这样的系统被设计成将你拉向人类作为回应。我从我在一次会议上遇到的一位银行业人士那里得知，咒骂系统可能会让系统挂断你的电话，而用愤怒/吼叫的语气说话可能会让你遇到真人。但我不是一个伟大的演员，所以我会坚持“再见”。

在这篇文章中，你看到了我对人类和人工智能代理的一些负面体验和极度沮丧。如果你喜欢这篇文章，那么看看我过去最常读的一些文章，比如“[如何给一个人工智能项目定价](https://medium.com/towards-data-science/how-to-price-an-ai-project-f7270cb630a4)”和“[如何聘请人工智能顾问](https://medium.com/towards-data-science/why-hire-an-ai-consultant-50e155e17b39)”嘿，[看看我们的时事通讯](http://eepurl.com/gdKMVv)！

下次见！

——丹尼尔
[Lemay.ai](https://lemay.ai)
[丹尼尔@lemay.ai](mailto:daniel@lemay.ai)