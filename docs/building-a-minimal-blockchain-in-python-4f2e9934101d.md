# 用 Python 构建一个最小的区块链

> 原文：<https://towardsdatascience.com/building-a-minimal-blockchain-in-python-4f2e9934101d?source=collection_archive---------4----------------------->

## 通过编码了解区块链

![](img/f367da870da2cf1ae6953d1d20df08a0.png)

Photo by [Shaojie](https://unsplash.com/@neural_notworks?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

区块链不仅仅是比特币或其他加密货币。它也用于数据库，如健康记录。

与某种流行的观念相反，区块链与数据加密无关。事实上，区块链中的所有数据都是透明的。它的特别之处在于它(在一定程度上)防止了回溯和数据篡改。让我们使用 Python 实现一个最小的区块链。下面是我如何构建一个最小的区块链，代码可以在 GitHub 上找到。

因为这是区块链的最小实现，所以在任何分布式网络上都没有算法或工作证明。

# 散列法

我们想要一个可以代表一个数据块的“键”。我们想要一个很难伪造或暴力破解的密钥，但是很容易验证。这就是哈希的用武之地。散列是满足以下性质的函数 H(x ):

*   相同的输入`x`总是产生相同的输出`H(x)`。
*   不同(甚至相似)的输入`x`应该产生 ***完全*** 不同的输出`H(x)`。
*   从输入`x`中获得`H(x)`在计算上很容易，但要逆转这个过程却很难，即从已知散列`H`中获得输入`x`。

这就是谷歌如何存储你的“密码”，而不是实际存储你的密码。他们存储你的密码散列`H(password)`，这样他们可以通过散列你的输入和比较来验证你的密码。没有进入太多的细节，我们将使用 SHA-256 算法散列我们的块。

![](img/0b7939f30ac14e624004dc0a67ca2bf1.png)

Photo by [CMDR Shane](https://unsplash.com/@cmdrshane?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 最小块

我们来做一个名为`MinimalBlock()`的对象类。它是通过提供一个`index`、一个`timestamp`、一些你想要存储的`data`和一个叫做`previous_hash`的东西来初始化的。前一个哈希是前一个块的哈希(键)，它充当指针，这样我们就知道哪个块是前一个块，从而知道块是如何连接的。

换句话说，`Block[x]`包含索引`x`、时间戳、一些数据和前一个块 x-1 `H(Block[x-1])`的散列。现在这个块已经完成了，它可以被散列以生成`H(Block[x])`作为下一个块中的指针。

![](img/77e4e3d0430d88d10e9a6494bd42b21c.png)

Photo by [Aiden Marples](https://unsplash.com/@mraidenmarples?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 最小链

区块链本质上是一个区块链，通过存储前一个区块的哈希来建立连接。因此，可以使用 Python 列表实现一个链，而`blocks[i]`表示第{ i }个块。

当我们初始化一个链时，我们用函数`get_genesis_block()`自动分配一个第 0 块(也称为 Genesis 块)给这个链。这一块标志着你的链的开始。注意`previous_hash`在创世纪块中是任意的。添加块可以通过调用`add_block()`来实现。

![](img/2f4106c458a96a1d202bc442acc8bd6c.png)

Photo by [Rustic Vegan](https://unsplash.com/@rusticvegan?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 数据验证

数据完整性对数据库非常重要，区块链提供了一种验证所有数据的简单方法。在函数`verify()`中，我们检查以下内容:

*   `blocks[i]`中的索引是`i`，因此没有丢失或额外的块。
*   计算块哈希`H(blocks[i])`，并与记录的哈希进行交叉检查。即使块中的单个位被改变，计算出的块散列也会完全不同。
*   验证`H(blocks[i])`是否正确存储在下一个程序块的`previous_hash`中。
*   通过查看时间戳来检查是否有回溯。

# 分支

在某些情况下，您可能想从一个链中分支出来。这被称为分叉，如代码中的`fork()`所示。你可以复制一个链(或者链的根),然后分道扬镳。在 Python 中使用`deepcopy()`至关重要，因为 Python 列表是可变的。

![](img/69e173bc7b87e69c5fe9421b16742ee4.png)

Photo by [elCarito](https://unsplash.com/@elcarito?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 外卖

对一个块进行哈希运算会创建一个块的唯一标识符，并且一个块的哈希构成下一个块的一部分，以在块之间建立链接。只有相同的数据才会创建相同的哈希。

如果您想修改第三个块中的数据，第三个块的散列值会改变，第四个块中的`previous_hash`也需要改变。`previous_hash`是第 4 个块的一部分，因此它的散列也会改变，依此类推。

## 相关文章

感谢您的阅读！如果您对 Python 感兴趣，请查看以下文章:

[](/5-python-features-i-wish-i-had-known-earlier-bc16e4a13bf4) [## 我希望我能早点知道的 5 个 Python 特性

### 超越 lambda、map 和 filter 的 Python 技巧

towardsdatascience.com](/5-python-features-i-wish-i-had-known-earlier-bc16e4a13bf4) [](/visualizing-bike-mobility-in-london-using-interactive-maps-for-absolute-beginners-3b9f55ccb59) [## 使用交互式地图和动画可视化伦敦的自行车移动性

### 探索 Python 中的数据可视化工具

towardsdatascience.com](/visualizing-bike-mobility-in-london-using-interactive-maps-for-absolute-beginners-3b9f55ccb59) 

*最初发布于*[*edenau . github . io*](https://edenau.github.io)*。*