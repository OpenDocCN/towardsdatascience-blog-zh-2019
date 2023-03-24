# 比特币区块链浅探第一部分——共识

> 原文：<https://towardsdatascience.com/a-shallow-dive-into-bitcoins-blockchain-part-1-consensus-48f62355681b?source=collection_archive---------35----------------------->

从自然物体到硬币，到黄金支持的纸币，再到纸币，再到数字货币。无论是哪种形式，货币都是一种交换手段，一种支付方式，一种价值标准，一种财富储存手段。

如今，大多数交易都变得数字化了。例如，当你用你的卡从一个商人那里买东西时会发生这样的事情:

1.通过将您的 PIN 输入到商家的机器中，您将获得授权

2.交易和卡数据被发送到商家的银行

3.商户银行(下图中的收单机构)将详细信息转发给你的卡组织(Visa、Mastercard 等)。)

4.然后卡组织联系您的银行(下图中的发卡银行)以确保卡是有效的并且里面有钱

5.一旦商家收到 OK 信息，他们就可以给你商品/服务

![](img/5007b7d703dc565c0342e5c14a125f91.png)

Acquirer is Merchant’s Bank. Issuing Bank is the Customer’s bank. [Source](https://wallethub.com/edu/cc/credit-card-transaction/25511/)

6.在一天结束时，招商银行将所有有效的交易数据发送给卡组织(例如 Visa)进行结算

7.卡组织将所有商户的交易转发给客户的银行(发卡银行)

8.客户的银行需要将资金转移到卡组织

9.卡组织支付给商户银行的费用减去交换费

10.商户银行将资金存入商户账户，扣除手续费

![](img/01c327004db5c90879a828342849fde3.png)

Acquirer is Merchant’s Bank. Issuing Bank is the Customer’s bank. [Source](https://wallethub.com/edu/cc/credit-card-transaction/25511/)

卡组织充当金融机构的中间人。他们正在促进人们银行账户之间的交易。这就像他们在银行之上，给你一张卡就能安全存取你的资金。

然而，这篇文章分析了我们如何在自己之间安全地交易。

让我们试着建立一个不依赖于任何组织的系统。

第一种方法是建立一个公共场所，保存我们之间发生的所有交易。例如，某处公共服务器上的数据库分类帐。因此，无论何时发生交易，我们都会将其保存在数据库中。数字钱包会自动将交易细节发送到服务器。

但是您可以很快看到这种方法带来的问题。谁将负责这样的数据库服务器？有权访问服务器的用户将有权随意添加、编辑或删除交易。

为了消除这种信任，我们可以让网络的每个参与者在他们的机器上保留一份数据库分类帐的副本。

但这里有另一个问题。**参与者如何保持这个分类账同步？你怎么能让每个人都同意同一个账本呢？**

为了应对这一挑战，使用一种被称为工作证明(PoW)的共识算法来替代中央机构。

*旁注:
战俘的术语是“采矿”。采矿的目的不是为了发现新的硬币，而是为了保证区块链的安全。*

基本思想是信任对其进行了最多计算工作的分类帐。

请记住，从参与者的角度来看，他不能相信任何人！

在比特币网络上，矿工将交易集中在一个区块中，并向其余参与者广播。然而，并不是所有的参与者都需要创建积木(成为一名矿工)。参与者也可以只听和验证新交易的有效性；他们感兴趣的交易。

例如，手机上的钱包应用程序也可以参与，但不一定支持完整挖掘参与者运行的功能(这包括验证和广播所有新创建的交易和块，创建新块等)。这将需要大量的计算能力和千兆字节的空间。

![](img/b4550bb7c56da11b3eb9388a4fe5d38b.png)

Alice, running a full mining node, just found a new block and broadcasted it to the rest of the network. Bob, being a participant in the network, listening for new blocks of transactions from the miners, receives the newly created block. [Source](https://www.youtube.com/watch?v=bBC-nXj3Ng4)

假设你是这个网络的参与者。你怎么能相信矿工会做好他们的工作？换句话说，确保您从他们那里收到的事务块是有效的，同时，其他人也记录了相同的事务块。

你信任矿工的方式是通过验证他们在创建这些块时投入了一些计算工作(PoW)。这个挖掘过程将在后面更详细地讨论。

糟糕的演员也可以付出一些努力，给你盖帽，对吗？

正确！但是这里有一个警告:

因为为这些块创建 PoW 需要时间和精力，所以不良行为者不能轻易欺骗整个网络。

假设您收到一个块，其中包含一个坏的矿工事务。过了一会儿，你听到另一个来自一个好矿工的阻塞，但是有一个冲突的事务。从你的角度来看，你真的不知道两个街区该相信哪一个。

关键点是保留两个块并创建两个单独的区块链(一个分叉事件)并继续监听新的块；连同他们力量。

有关更多详细信息，请参见该动画的标题:

![](img/e0cf01cf4bebd844c1e5f553c77abdba.png)

In this example, Alice is trying to cheat the network by trying to convince it that her version of the chain is the valid one (and thus the transactions in it). Bob, who is a participant in the network, is trying to understand which blockchain is the correct one. [Source](https://www.youtube.com/watch?v=bBC-nXj3Ng4)

叉子不一定是坏事。网络里每天都在发生。这是全球网络传输延迟的自然结果。即两个不同的挖掘器同时发现并发送一个块(当创建一个块时，每个挖掘器将包括不同的事务或不同顺序的事务，这可能发生在两个不同的挖掘器同时发现它们的块的随机数的情况下)。但只要所有节点都等待并选择最长的链，网络最终会收敛到一致的状态(通过网络的共识)。

![](img/0e3d9e99ba3e845d5d76f65ba4527a85.png)

Visualization of a blockchain fork event: two blocks propagate at the same time. Some nodes receive block “triangle” first and some receive block “upside-down triangle” first. [Source](https://github.com/bitcoinbook/bitcoinbook/blob/develop/images/mbc2_1004.png)

因此，如果坏矿工的区块最终没有做到这一点，并说服网络的其余部分，所有的努力都浪费了，在他们用来制作它的资源中。这也是阻止坏演员作弊的因素之一。

# **矿工需要做什么样的计算工作，才能证明自己在区块的创建上付出了一些努力？**

为了理解这一点，让我们先解释一下**加密哈希函数**。

首先要注意的是，数字世界中的一切都可以归结为比特(0 和 1)。

一个**加密散列函数**是一个数学函数，它接受任何大小的位(这是输入消息——文本、图像、数字),并输出固定长度的 256 位。这个输出被称为消息的**哈希值、摘要**或简称为**哈希**。

加密哈希函数具有以下属性:

1.  在不改变哈希值的情况下修改消息是不可行的。如果您稍微改变输入消息，即使是一位，产生的哈希值也会完全改变！
2.  不可能找到两个具有相同哈希值的不同消息。相同的输入消息(比特组)将总是给出相同的散列值(256 比特)，而两个不同的消息(比特组)将给出不同的散列值(256 比特)。
3.  从消息的哈希值生成消息是不可行的。有了 256 位哈希值，您就可以返回并找到用作输入的消息。

![](img/181c514978683078030d288d9c4496e3.png)

Hashing comparison. Note that for readability hexadecimal notation is used for the resulting hash. One hex value represents 4 bits. 64 hex values times 4 bits = 256 bits. [Source](https://freecontent.manning.com/cryptographic-hashes-and-bitcoin/)

比特币使用 [SHA256](https://en.bitcoinwiki.org/wiki/SHA-256) 哈希函数。

这样的函数如何证明参与者确实花费了大量的计算能力？换句话说，我们如何知道一个矿工实际上做了特定事务块所需的工作？

规则是，挖掘者必须找到一个名为 **nonce** 的数字，当该数字被添加到事务块中并被散列在一起时，将产生一个 256 位散列(名为**块散列**),该散列以特定数量的 0 开始。前导 0 的数量由**难度等级**定义。

记住，在数字世界里，一切都是用比特来表示的。

区块链中的每个块包含关于该块的各种信息(交易、创建时间等。).该信息也表示为一组比特。这些位对于每个块都是唯一的，因为它有唯一的创建时间和一组事务。

> 对于矿工来说，很难找到额外数量的位( **nonce** 数),当添加到现有块的位中，并使用 SHA256 哈希函数哈希在一起时，将产生一个具有特定 0 集的哈希值(由难度级别定义)。生成与特定目标匹配的哈希值的唯一方法是反复尝试，随机修改输入(通过尝试不同的随机数)，直到所需的哈希值偶然出现。

花点时间再读一遍。那是工作证明的基石。

![](img/5f98cc5000a081e0881cc96c19824f52.png)

Hashing numbers to find the nonce that satisfies the difficulty level. [Source](https://www.youtube.com/watch?v=bBC-nXj3Ng4)

这个数字(nonce)作为证据，证明为了产生与特定目标匹配的结果散列，做了一定的工作。因此工作证明这个术语。

使用这种方法的好处是，每个参与者都可以很容易地验证他们做了大量的工作，而不必做同样的努力。拥有 1。块的数据，2。随机数和 3。由此产生的散列，任何人都可以使用相同的函数快速验证其正确性。

满足难度等级的方块被认为是有效的。

因此，如果一个坏的参与者将要改变一个事务(实际上是消息中的一些位)，那么**块散列**将会不同，并且很可能不是以所需数量的 0 开始，因此，使该块无效。

**块散列**充当数字指纹，或者块标识符。它唯一地标识一个块，并且可以由任何节点通过简单地散列该块的数据(随机数、事务、难度、时间戳等)而独立地导出。).

*我没有提到的一件事是，一个糟糕的矿工不可能改变任何交易。还有一点额外的安全性，我将在第 2 部分的* [***中讨论——事务***](https://medium.com/@alymbouras/a-shallow-dive-into-bitcoins-blockchain-part-2-transactions-d4ee83067bae) *。但目前来看，这是理所当然的。*

有了这个额外的安全措施，一个坏矿工只能改变他自己的一个交易。

因此，为了欺骗网络，他将不得不去链中的特定块(保存他的事务的块)改变对他有利的事务，并重新做寻找满足该块难度级别的**块散列**的工作。将这样的块传播到网络的其余部分将被认为是完全有效的。

除非我们再增加一点安全措施。

所有的块都是相互连接的！通过散列前一个块的散列，而不仅仅是事务、nonce 和时间戳，我们得到了所有块之间的紧密关系。

![](img/383db8821a3abb47c5a79780b9e983cc.png)

Blocks’ contents. All transactions (Tx0, Tx1, Tx2, Tx3) can be represented with just one hash number, called the [Merkle Root](https://en.bitcoinwiki.org/wiki/Merkle_tree) (Tx_Root in the diagram). [Source](https://bitcoin.stackexchange.com/questions/12427/can-someone-explain-how-the-bitcoin-blockchain-works)

所以，如果你要改变任何一个方块，或者试着交换两个方块的顺序，它会改变后面的方块。由于块上的单个更改会更改其哈希，因此它会更改下一个块的哈希，依此类推，使其后的所有块无效。这将需要重做所有的工作，为这些块中的每一个找到一个新的特殊数字，使它们的散列以特定的数字 0 开始。这需要大量的工作！

*附注:
在撰写本文时，网络的哈希速率约为每秒 65 万亿次哈希。一个坏演员将需要至少有一半来欺骗网络(被称为多数或 51%攻击)。因为攻击者可以比网络的其他部分更快地生成块，所以他可以简单地坚持他的私有分支，直到它变得比诚实网络构建的分支更长。然而，当前网络的散列率使得它在经济上是不可能的(所需的机器和能量花费了数百万美元)。而这仅仅是为了改变他的一笔交易。所以，最好值得！*

# **难度等级**

前面我说过，PoW 是要找到一个特殊的数字(nonce ),以便块的散列以一个特定的数字 0 开始。

前导 0 越多，挖掘者就越难找到满足该散列的随机数。这是因为可接受的结果散列的范围变小了。

但是，为什么一开始就让矿工们为难呢？

目标是将整个网络开采一个区块的难度保持在 10 分钟以内。

这是快速确认时间(交易结算)和分叉概率之间的设计折衷。更快的封锁时间会使交易清算更快，但会导致更频繁的区块链分叉，而更慢的封锁时间会减少分叉的数量，但会使结算更慢。

难度参数每 2016 块动态调整，以保持 10 分钟的块间隔目标。因此，公式是，对于每 2016 个区块，计算矿工的平均区块创建时间，如果该时间超过 10 分钟，难度参数将被调整。例如，如果网络平均花费 12.5 分钟建造一个街区(比 10 分钟的目标多 20%)，那么对于接下来的 2016 个街区，难度将降低 20%。块创建时间的这一修订遵循了矿工散列能力(本质上是硬件)的改进。

*旁注:*

今天，许多矿工在一个矿池中合作，竞争开采一个区块。他们分担寻找解决方案的工作，分享报酬。这是集中了整个挖掘思想，但它也有一些 [*优点*](https://www.investopedia.com/tech/are-large-mining-pools-bad-cryptocurrencies/) *。但是，无论如何，随着当前金融机构因内幕交易而受到密切关注，大型矿池也因不当行为而受到关注。任何不当行为都会让他们失去信誉，甚至更糟，失去用户对比特币的信心，将整个项目推入深渊。这会让他们破产。*

如今，矿池就像一家银行，保护着人们的资金安全。但是，说到底，大多数都是中央集权的公司！😉对等池也存在。但他们是少数。

![](img/3f5ae67ecfe763add3efc6a6ef02d17d.png)

有兴趣了解更多信息吗？阅读我的文章 [**第二部分—交易**](https://medium.com/@alymbouras/a-shallow-dive-into-bitcoins-blockchain-part-2-transactions-d4ee83067bae) 了解更多知识。

在社交媒体上关注我:

[](https://www.linkedin.com/in/andreas-lymbouras-51211246/) [## Andreas Lymbouras -应用程序开发人员-法国巴黎银行| LinkedIn

### 加入 LinkedIn 经验丰富的应用程序开发人员，并展示其在银行业的工作经历。熟练的…

www.linkedin.com](https://www.linkedin.com/in/andreas-lymbouras-51211246/) [](https://twitter.com/alymbouras) [## 安德烈亚斯·林布拉斯(@alymbouras) |推特

### Andreas Lymbouras 的最新推文(@alymbouras)。软件开发商区块链工程师

twitter.com](https://twitter.com/alymbouras)