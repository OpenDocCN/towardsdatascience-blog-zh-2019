# 大数据有多大？

> 原文：<https://towardsdatascience.com/how-big-is-big-data-3fb14d5351ba?source=collection_archive---------7----------------------->

我们已经永远进入了数据时代。我们在线上甚至线下所做的一切都会在数据中留下痕迹——从 cookies 到我们的社交媒体档案。那么到底有多少数据呢？我们每天处理多少数据？欢迎来到齐塔字节时代。

![](img/c8a75aa9d286effbd86ec45904af2d65.png)

IBM Summit supercomputer

# 1.齐塔字节时代

数据是用比特和字节来衡量的。一位包含值 0 或 1。八位构成一个字节。然后我们有千字节(1000 字节)、兆字节(1000 字节)、千兆字节(1000 字节)、太字节(1000⁴字节)、千兆字节(1000⁵字节)、千兆字节(1000⁶字节)和千兆字节(1000⁷字节)。

[思科估计](https://blogs.cisco.com/sp/the-zettabyte-era-officially-begins-how-much-is-that)2016 年，我们的年度互联网总流量超过了 zettabyte，这是我们在万维网上上传和共享的所有数据，其中大部分是文件共享。zettabyte 是存储容量的度量单位，它等于 1000⁷(10 亿字节)。一个 zettabyte 等于一千 EB、十亿 TB 或一万亿 GB。换句话说——太多了！尤其是如果我们考虑到互联网还不到 40 岁。思科还估计，到 2020 年，年流量将增长到 2 千兆字节以上。

互联网流量只是全部数据存储的一部分，其中还包括所有个人和商业设备。对 2019 年我们现在拥有的总数据存储容量的估计各不相同，但已经在 10-50 吉字节[的范围内](https://www.northeastern.edu/levelblog/2016/05/13/how-much-data-produced-every-day/)。到 2025 年，据[估计](https://www.networkworld.com/article/3325397/storage/idc-expect-175-zettabytes-of-data-worldwide-by-2025.html)将增长到 150-200 兆字节。

毫无疑问，数据创建只会在未来几年加速，因此您可能会想:数据存储有任何限制吗？不完全是，或者更确切地说，是有极限的，但是是如此遥远，以至于我们不会很快接近它们。例如，仅仅一克 DNA 就可以存储 700 万亿字节的数据，这意味着我们可以在 1500 千克的 DNA 上存储我们现在拥有的所有数据——密集包装，可以放入一个普通的房间。然而，这与我们目前所能制造的相差甚远。正在制造的[最大硬盘](https://www.notebookcheck.net/Western-Digital-unveils-new-15-TB-hard-drive-the-largest-HDD-yet.346781.0.html)有 15tb，[最大 SSD 达到](https://www.zdnet.com/article/worlds-largest-ssd-hits-100tb/)100tb。

术语[大数据](https://en.wikipedia.org/wiki/Big_data)是指对于普通计算设备来说太大或太复杂而无法处理的数据集。因此，这是相对于市场上可用的计算能力。如果你看一下最近的[数据历史](https://www.forbes.com/sites/gilpress/2013/05/09/a-very-short-history-of-big-data/#1acdf20d65a1)，那么在 1999 年，我们总共有 1.5 的数据，1gb 被认为是大数据。早在 2006 年，总数据量估计就达到了 160 年内增长了 1000%。在我们的 Zettabyte 时代，1gb 已经不再是真正的大数据，谈论从至少 1tb 开始的大数据是有意义的。如果我们用更数学的术语来说，那么谈论大数据似乎很自然，因为数据集超过了世界上创建的总数据除以 1000。

# 2.千万亿次浮点运算

为了让数据变得有用，仅仅存储它是不够的，你还必须访问和处理它。人们可以通过每秒指令数(IPS)或每秒浮点运算数(FLOPS)来衡量计算机的处理能力。虽然 IPS 比 FLOP 更广泛，但它也不太精确，并且取决于所使用的编程语言。另一方面，FLOPS 很容易想象，因为它们与我们每秒可以做的乘法/除法次数直接相关。例如，一个简单的手持计算器需要几个 FLOPS 才能正常工作，而大多数现代 CPU 都在 20–60 GFLOPS 的范围内(gigaFLOPS = 1000 FLOPS)。IBM 在 2018 年制造的破纪录的[计算机](https://www.top500.org/lists/2018/06/)达到了 122.3 petaFLOPS (1000⁵ FLOPS)，比一台普通 PC(峰值性能 [200 petaflops](https://en.wikipedia.org/wiki/Summit_(supercomputer)) )快了几百万次。

GPU 的浮点计算性能更好，达到数百 GFLOPS(大众市场设备)。当你研究专门化的建筑时，事情变得有趣了。最新的趋势是构建硬件来促进机器学习，最著名的例子是谷歌的 TPU[，它达到了](https://arstechnica.com/information-technology/2017/05/google-brings-45-teraflops-tensor-flow-processors-to-its-compute-cloud/) 45 万亿次浮点运算(1000⁴浮点运算)，可以通过云访问。

如果你需要进行大型计算，而你自己又没有超级计算机，那么退而求其次的办法就是租一台，或者在云上进行计算。[亚马逊](https://aws.amazon.com/hpc/)为你提供高达 1 petaFLOPS 的 P3，而[谷歌](https://cloud.google.com/tpu/)提供速度高达 11.5 petaFLOPS 的一组 TPU。

# 3.人工智能和大数据

让我们把它们放在一起:你有数据，你有与之匹配的计算能力，所以为了获得新的见解，是时候使用它们了。要真正从两者中受益，你必须求助于机器学习。人工智能处于数据使用的前沿，有助于预测天气、交通或健康(从发现新药到癌症的早期检测)。

人工智能需要训练来执行专门的任务，看看需要多少训练才能达到峰值性能是计算能力与数据的一个很好的指标。OpenAI 在 2018 年有一份[出色的报告，评估了这些指标，并得出结论，自 2012 年以来，以 petaflops/day (petaFD)衡量的人工智能训练每 3.5 个月翻一倍。一个 petaFD 包括在一天中每秒执行 1000⁵神经网络运算，或者总共大约 10 次⁰运算。这一指标的伟大之处在于，它不仅考虑了网络的架构(以所需操作数量的形式)，还将其与当前设备上的实施(计算时间)联系起来。](https://blog.openai.com/ai-and-compute/)

你可以通过查看下面的图表来比较在人工智能的最新进展中使用了多少 petaFD:

![](img/ef24b1e9f2a4ec8188b2f189773578f6.png)

chart by OpenAI

毫无疑问，领先的是 DeepMind 的 [AlphaGo Zero](https://deepmind.com/blog/alphago-zero-learning-scratch/) ，使用了超过 1,000 petaFD 或 1 exaFD。就资源而言，到底有多少？如果你要用同样的硬件来复制你自己的培训，你很容易就会花费近 300 万美元[，正如这里详细估算的](https://www.yuzeh.com/data/agz-cost.html)。根据上面的图表，较低的估计是，1,000 petaFD 至少相当于使用最好的亚马逊 P3 1000 天。如果当前价格为每小时 31.218 美元，则 31.218 美元 x 24 小时 x 1，000 天= 749，232 美元。这是最低的界限，因为它假设一个神经网络操作是一个浮点操作，并且您在 P3 上获得的性能与在 DeepMind 使用的不同 GPU/TPU 上获得的性能相同。

这说明 AI 需要大量的力量和资源来训练。有一些机器学习的最新进展的例子，当时在计算能力或数据方面不需要太多，但大多数情况下，额外的计算能力是非常有用的。这就是为什么建造更好的超级计算机和更大的数据中心是有意义的，如果我们想发展人工智能，从而发展我们的整个文明。你可以想象类似于[大型强子对撞机](https://en.wikipedia.org/wiki/Large_Hadron_Collider)的超级计算机——你建造越来越大的对撞机，这样你就可以获得关于我们宇宙的更深层次的真相。计算能力和人工智能也是如此。我们不了解我们自己的智力或我们如何执行创造性任务，但增加 FLOPS 的规模可以帮助解开这个谜。

拥抱 Zettabyte 时代！并且更好的快速从中获利，因为 Yottabyte 时代已经不远了。