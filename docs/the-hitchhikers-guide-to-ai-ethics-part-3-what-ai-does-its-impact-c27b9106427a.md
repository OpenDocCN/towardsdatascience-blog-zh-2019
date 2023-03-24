# 人工智能伦理指南第 3 部分:人工智能做什么及其影响

> 原文：<https://towardsdatascience.com/the-hitchhikers-guide-to-ai-ethics-part-3-what-ai-does-its-impact-c27b9106427a?source=collection_archive---------9----------------------->

## 探索人工智能伦理问题的 3 集系列

![](img/48147c6ed3b1bd9d93b71cdbe33796f4.png)

“I don’t know what I’m doing here, do you?” (Image by [Rock’n Roll Monkey](https://unsplash.com/@rocknrollmonkey?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText))

# 故事到此为止

在第一部分，我探索了人工智能的伦理是什么和为什么。[第二部分看什么是人工智能的伦理](/the-hitchhikers-guide-to-ai-ethics-part-2-what-ai-is-c047df704a00)。在第 3 部分，我以对人工智能做什么和人工智能影响什么的伦理的探索结束了这个系列。无论是安全还是公民权利，无论是对人类行为的影响还是恶意使用的风险，这些话题都有一个共同的主题——需要重新审视技术专家在处理他们所创造的东西的影响方面的作用；超越“避免伤害”和“尊重隐私”等宽泛的原则，建立因果关系，并确定我们独特的优势或劣势。

*我很早就有一种感觉，第三部分将很难公正地作为一个不到 10 分钟的帖子。但是三部分刚刚好，超过 10 分钟太长；所以当我试图证明直觉是错的并且失败的时候，请耐心等待！让我们一起探索。*

# 人工智能做什么

人工智能的能力将随着时间的推移而提高，人工智能应用将充斥我们的世界。这不一定是一件坏事，但它确实创造了一种迫切的需求，即评估 AI 做什么以及它如何影响人类；从我们的安全，到我们与机器人的互动，到我们的隐私和代理。

那么 AI 是做什么的？人工智能使用大量计算和一些规则来分析、识别、分类、预测、推荐，并在允许的情况下为我们做决定。做出能够永久改变人类生活进程的决定是一项巨大的责任。AI 准备好了吗？是吗？在缺乏内在道德偏见的情况下，人工智能系统可以用来帮助我们或伤害我们。我们如何确保人工智能不会造成或导致伤害？

![](img/e400fb01960b72d956aad69f9fb57455.png)

Distilling the Harms of Automated Decision-Making ([Future of Privacy Forum Report](https://fpf.org/2017/12/11/unfairness-by-algorithm-distilling-the-harms-of-automated-decision-making/))

## 安全

人工智能中的安全可以理解为“人工智能必须[不会导致事故](https://arxiv.org/pdf/1606.06565v1.pdf)，或者表现出非故意或有害的行为”。身体伤害是显而易见的，无人驾驶汽车的安全问题也是众所周知的。但是，如何在一个自治系统中建模并实现安全性呢？

在基于规则的系统中，给定的输入总是产生相同的结果，安全问题可以通过严格的测试和操作程序来解决。这种方法只适用于人工智能。

> 自主决策需要在不确定的情况下自动评估安全性，以便可预测地预防伤害。

让我们打开这个。人类不会在真空中做决定。我们的行动不仅由外部触发因素决定，还取决于我们的[意图、规范、价值观和偏见](https://hrilab.tufts.edu/publications/aaai17-alignment.pdf)。我们认为安全的东西也会随着时间和环境而变化。考虑在车流中穿梭，将某人紧急送往医院。你会做吗？我猜你说，看情况。

对于一个硬件-软件组合来说，要做出正确的调用，需要它能够**响应出现的上下文**，能够**在其环境中对这种不确定性**建模，并且**与什么是“正确的”**保持一致。与“正确的”目标保持一致，又名 [**价值-与**](https://futureoflife.org/2017/02/03/align-artificial-intelligence-with-human-values/) **、**保持一致是 AI 中安全的一个关键主题。问题是，[自治系统如何追求与人类价值观一致的目标](https://distill.pub/2019/safety-needs-social-scientists/)？更重要的是，鉴于人类可以持有相互冲突的价值观，[系统与谁的价值观一致](https://www.newyorker.com/science/elements/a-study-on-driverless-car-ethics-offers-a-troubling-look-into-our-values)？

## 网络安全和恶意使用

![](img/2573e6c3ef721c702787dedd7c622245.png)

Building ethics into machines. Yes/No? Image credit: [Iyad Rahwan](http://www.mit.edu/~irahwan/cartoons.html)

虽然人工智能越来越多地通过检测和防止入侵来实现网络安全，但它本身也容易受到游戏和恶意使用的影响。在一个数据驱动、高度网络化、永远在线的世界里，风险是巨大的。除了经典的威胁，人工智能系统还可以通过毒害输入数据或修改目标函数来造成伤害。

降低成本和提高音频/视频/图像/文本生成能力也推动了人工智能系统和[社会工程](https://searchsecurity.techtarget.com/definition/social-engineering)的发展。当技术做技术的时候，谁来承担这种滥用的负担？由于担心被恶意使用，开放人工智能的[决定](https://openai.com/blog/better-language-models/)不发布他们的文本生成模型 GPT-2，这引起了人工智能研究人员的强烈反应。通过[推特](https://twitter.com/Smerity/status/1096247080941629441)，博客[例如 1](https://www.fast.ai/2019/02/15/openai-gp2/) ，[例如 2](https://anima-ai.org/2019/02/18/an-open-and-shut-case-on-openai/) 和[辩论](https://twimlai.com/twiml-talk-234-dissecting-the-controversy-surrounding-openais-new-language-model/)很明显，确定“做正确的事情”是困难的，人工智能研究人员尚未在前进的道路上达成一致。与此同时，黑帽研究人员和坏演员没有这样的冲突，并继续有增无减。

## 隐私、控制和监视

沿着伤害和滥用技术的路线，是人工智能被重新利用的能力，甚至是有意设计的监视能力。构建这种工具的伦理考虑是什么？到底该不该建？让我们后退一步，理解为什么这很重要。

请人们描述隐私，你会得到多种定义。这是合理的，因为隐私是一种社会建构，随着时间的推移而演变，并受到文化规范的影响。虽然定义隐私很难，但识别侵犯隐私的行为却很直观。当你在机场被单独搜身时，你会觉得受到了侵犯。你没有什么可隐瞒的，但是你觉得被侵犯了！因为在许多定义的背后是人类最基本的东西——尊严和控制。

在所描述的案例中，你理解了违规行为并继续遵守，为了更大的利益(公共安全)放弃了隐私。我们一直都这样。现在考虑一个数字化、大数据、人工智能的世界，在这个世界里，侵犯隐私既不直接也不明显，放弃隐私的风险来自包装在便利和个性化中的礼物。个人的、私人的、安全的、开放的、同意的概念都变得混乱，不利于普通用户。正是在这一点上，技术专家占据优势，可以在保护隐私方面发挥作用。

以面部识别技术为例，这是迄今为止最致命的侵犯隐私的技术。CCTV、Snapchat/Instagram stories、脸书直播等看似无害的技术，都在推动一种文化，在这种文化中，记录他人感觉很正常。在有钱可赚的时候，企业继续推销“方便和个性化”的产品。[自拍到签到](https://qz.com/1623799/facial-recognition-is-coming-to-cruise-ships/)、[拇指支付](https://usa.visa.com/visa-everywhere/security/biometric-payment-card.html)、[眨眼解锁](https://www.digitaltrends.com/home/iris-scanners-can-lock-and-unlock-your-doors/)、 [DNA 祭祖游](https://press.airbnb.com/heritage-travel-on-the-rise/)，都让收集和凝聚让你，你。**同时，AI 可以进行面部分析、皮肤纹理分析、步态识别、语音识别和情感识别。所有** [**未经个人许可或合作**](https://www.le-vpn.com/face-recognition-privacy/) **。把所有这些加起来，你会不成比例地强化国家/企业而非个人。虽然中国的监控制度听起来有些极端，但面部识别对执法和公共安全至关重要的观点却很普遍。尽管有许多偏见，美国也经常在执法中使用 T21 的面部识别，除了在科技之都。事实上“安全”的诱惑是如此强烈，基于人工智能的跟踪包括面部识别现在正被用在儿童身上，尽管误报对年轻人的心灵有害。作为技术人员，我们应该从哪里开始划分界限呢？**

## 人与人工智能的互动

在我们有了谷歌主页几个月后，我 4 岁的孩子大声宣称“谷歌无所不知”。不用说，一个关于 Google Home 如何知道它知道什么以及它肯定不是一切的长对话接踵而至！令我沮丧的是，他看起来不太相信。人类的声音回应“嘿谷歌，我的纸杯蛋糕在哪里”，“嘿谷歌，你今天刷牙了吗”，“嘿谷歌，给我讲个笑话”对他这个年龄的孩子来说太真实了；而像机器、程序和训练这样的术语，我们应该叫它什么，人工的。

人工智能在我孩子的生活中扮演的角色[好的或坏的](https://www.cnn.com/2018/10/16/tech/alexa-child-development/index.html)是不可低估的。大多数父母使用技术，包括智能音箱，不知道它如何工作或何时不工作。但问题是，他们不应该知道。再次，考虑技术专家，效果和有利位置。

> 算法对我们精神和情感福祉的影响，无论是积极的还是消极的，也是令人担忧的原因。

我最近[分享了一个故事](https://twitter.com/purpultat/status/1122380606355021824)某人被一个心理健康 app 通知救了；在其他案例中[算法将某人推向自残](https://www.wired.com/story/when-algorithms-think-you-want-to-die/)；与此同时，依靠 Alexa 来对抗孤独、依靠毛茸茸的机器海豹来治疗、依靠玻璃管雕像来陪伴的例子也存在。

> 这种对人工智能的依赖将我们从本质上是社区的失败中拯救出来，在某些情况下是医疗保健的失败，这让我感到害怕和悲伤。

# 人工智能有什么影响

人工智能越来越多地影响着一切，但重要的是要突出二阶和三阶效应。不管是不是有意的，这些影响都是复杂的、多维的、大规模的。理解它们需要时间和专业知识，但意识到这一点是有价值的第一步，也是我的目标。

## 自动化、失业、劳动力趋势

围绕 AI 的新闻周期在“AI 将拯救我们”和“AI 将取代我们”之间交替。工厂工人被机器人取代的故事，人工智能创造了数百万个工作岗位的故事，以及 T4 人工智能中隐形劳动的危险的故事，都描绘了人工智能时代工作的未来。没关系，正如这份布鲁金斯报告建议的那样，“所有这些主要研究报告的重大劳动力中断的事实应该被认真对待”。

谈到人性，我是一个乐观主义者——我相信，只要我们愿意，我们可以战胜一切。人工智能引发的失业问题是，我们会做得足够快吗？那些风险最大的人会找到生存所需的手段和资源吗？仅仅生存就足够了吗？目标感、生产力和尊严呢？人工智能会给所有人提供这些还是仅仅给那些有足够特权去追求它的人？人工智能会加剧拥有者因拥有 T9 而拥有更多的恶性循环吗？

很明显，人工智能将打破劳动力的格局。[人工智能伙伴关系](https://www.partnershiponai.org/compendium-synthesis/)、[布鲁金斯](https://www.brookings.edu/research/automation-and-artificial-intelligence-how-machines-affect-people-and-places/)、[奥巴马白宫](https://obamawhitehouse.archives.gov/sites/default/files/whitehouse_files/microsites/ostp/NSTC/preparing_for_the_future_of_ai.pdf)报告就谁将受到影响以及如何受到影响提供了有益的见解。但是，还不完全清楚这种变化会发生得多快，以及我们是否正在尽全力为此做准备。

## 民主和民权

> “权力总会学习，强大的工具总会落入它的手中。”——Zeynep Tufecki，发表于[麻省理工学院技术评论](https://www.technologyreview.com/s/611806/how-social-media-took-us-from-tahrir-square-to-donald-trump/)

无论是中国的大规模监控还是公共话语的系统性劫持，人工智能在强者手中的影响已经显而易见。虽然人工智能不是他们的唯一原因，但它推动强大力量的独特方式是人工智能研究人员必须应对的问题。

互联网，尤其是推动其发展的盈利性公司，造就了一种造假文化。假的人，假的对话:在 2013 年的某个时候，YouTube 上一半的人是冒充真人的机器人。一半。年复一年，只有不到 60%的网络流量来自人类。虽然 YouTube 和脸书等公司声称对其“平台”上的内容保持中立，但实际上它们最大化了消费，这导致一些内容比其他内容提供得更多。当机器人或不良行为者生成为病毒式传播定制的内容时，平台会满足他们的要求。这对我们如何消费和处理信息、谁掌握着我们的权力、我们信任谁以及我们如何行动意味着什么？Danah Boyd， [Data & Society](https://datasociety.net/) 的创始人说，这种[对基于人工智能的推荐引擎](https://points.datasociety.net/the-fragmentation-of-truth-3c766ebb74cf)的操纵导致了真相的破碎，并最终失去信任，失去社区。这种知情、信任的地方社区的丧失削弱了民主的力量。随着民主遭受打击，结构性偏见扩大，公民权利的自由行使不再是所有人都能一律享有的。

## 人与人之间的互动

人工智能如何重塑人类互动关系到我们个人和集体的福祉。早期迹象令人不安:[性别化的人工智能促进了刻板印象和歧视](https://unesdoc.unesco.org/in/documentViewer.xhtml?v=2.1.196&id=p::usmarcdef_0000367416&file=/in/rest/annotationSVC/DownloadWatermarkedAttachment/attach_import_77988d38-b8bd-4cc1-b9b4-3cc16d631bf9%3F_%3D367416eng.pdf&locale=en&multi=true&ark=/ark:/48223/pf0000367416/PDF/367416eng.pdf#%5B%7B%22num%22%3A384%2C%22gen%22%3A0%7D%2C%7B%22name%22%3A%22XYZ%22%7D%2C0%2C842%2Cnull%5D)，自然语言人工智能导致了[礼貌的丧失](https://www.theatlantic.com/family/archive/2018/04/alexa-manners-smart-speakers-command/558653/)以及[当人工智能调解互动时信任度降低](https://osf.io/qg3m2/)。这就引出了一个更基本的问题，即今天的狭义人工智能，以及在某一点上的 AGI，将如何影响我们去爱、去同情、去信任和归属的能力？

耶鲁大学教授尼古拉斯·克里斯塔基斯的一项实验表明，人类的群体动力可以通过引入类人机器人来改变。当自私的搭便车机器人加入群体时，一个合作以最大化集体回报的群体完全停止合作。对环境信任的减少改变了我们建立联系和合作的方式。

![](img/f3619c304069a2c628e33a85e34eb718.png)

Humanity is Human Interdependency (image src: [deposit photos](https://depositphotos.com/stock-photos/holding-hands.html))

尼古拉斯·克里斯塔基斯说:“随着人工智能渗透到我们的生活中，我们必须面对它可能会阻碍我们的情感，抑制深层的人类联系，让我们彼此之间的关系变得更少互惠，更浅，或更自恋。”这种阻碍也延伸到了道德上。正如肌肉不被使用会浪费肌肉一样，道德肌肉需要真实世界的互动来增强力量。那么，如果一个社会的典型决策是由隐藏在脱离其来源的数据背后的计算做出的，会发生什么？我们是否失去了感同身受的能力？我们对不公平变得不敏感了吗？我们能够经常练习道德判断以获得实践智慧吗？[圣克拉拉大学的哲学教授香农·瓦勒](https://www.shannonvallor.net/)，称之为道德去技能化(在这里[详细阐述](https://www.scu.edu/ethics/focus-areas/technology-ethics/resources/artificial-intelligence-decision-making-and-moral-deskilling/))。这种去技能化使得人类需要做出的一些决定变得更加困难，通常是在更关键和冲突的情况下(例如，作为陪审员)。

# 连续函数不连续人类

我需要提醒这里的读者，我正在以一种深刻反思的状态结束这个系列，前面的总结反映了这一点。从第一部分中对人工智能的伦理景观的[调查开始，到第二部分](/ethics-of-ai-a-comprehensive-primer-1bfd039124b0)中对人工智能是什么以及人工智能在第三部分中的作用和影响的[深入探究，再到我对心理学、社会学、人类学和技术的阅读，我痛苦地意识到我们对人类和人性的理解是多么的不足。](/the-hitchhikers-guide-to-ai-ethics-part-2-what-ai-is-c047df704a00)

> 也许，做人就是站在别人的立场上，理解他们所经历的事情的严重性，并尽你所能找到帮助他们的最佳方式。人性就是相信别人也会为你做同样的事。作为技术人员，我们制造产品，我们相信它们会帮助人们，我们定义衡量标准来表明它们确实有帮助，但是我们经常不能理解当它们没有帮助时的严重性。

为人类建造人工智能的伦理要求理解人类，以及他们所有的不连续性。这不是一个指标驱动的、有时限的、高回报、高增长的项目，但却非常有价值。

这是探索人工智能伦理的 3 部分系列的第 3 部分。 [*第一部*](/ethics-of-ai-a-comprehensive-primer-1bfd039124b0?source=friends_link&sk=02e78d6fe2c2c82b000b47230193d383) *点击这里。* [*点击此处查看第二部分*](/the-hitchhikers-guide-to-ai-ethics-part-2-what-ai-is-c047df704a00) *。非常感谢* [*雷切尔·托马斯*](https://twitter.com/math_rachel)*[*卡蒂克·杜赖萨米*](https://www.linkedin.com/in/karthik-duraisamy-66705025/) *和* [*斯里拉姆·卡拉*](https://twitter.com/skarra) *对初稿的反馈。**