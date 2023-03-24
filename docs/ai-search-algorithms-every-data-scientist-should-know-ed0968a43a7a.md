# 每个数据科学家都应该知道的人工智能搜索算法

> 原文：<https://towardsdatascience.com/ai-search-algorithms-every-data-scientist-should-know-ed0968a43a7a?source=collection_archive---------11----------------------->

![](img/5eeb8d039fda432df0880a07122e675b.png)

TL；下面的帖子概述了人工智能中的一些关键搜索算法，为什么它们很重要，它们有什么用途。

虽然近年来，[搜索和规划算法](https://en.wikipedia.org/wiki/Automated_planning_and_scheduling)已经让位于机器和深度学习方法，但更好地理解这些算法可以提高模型的性能。此外，随着量子计算等更强大的计算技术的出现，基于搜索的人工智能很可能会卷土重来。

# 什么是 AI 中的搜索算法？

在我们开始之前，让我们定义一下人工智能中的搜索是什么意思。

![](img/85547312fd2160c26421b4375a562ba4.png)

A search algorithm is not the same thing as a search engine.

人工智能中的**搜索**是通过**过渡**到**中间状态**从**起始状态**导航到**目标状态**的过程。

几乎任何人工智能问题都可以用这些术语来定义。

*   **状态** —问题的潜在结果
*   **转换** —在不同状态之间移动的行为。
*   **开始状态—** 从哪里开始搜索。
*   **中间状态**——我们需要转换到的起始状态和目标状态之间的状态。
*   **目标状态—** 停止搜索的状态。
*   **搜索空间**——状态的集合。

# 不知情的搜索

![](img/03184b66f6bcdb7207d55cd689ae5403.png)

当没有关于在州之间导航的成本的信息时，使用无信息搜索。

对于不知情的搜索，有三种主要的经典算法:

*   **DFS —** 使用 [LIFO 堆栈](https://en.wikipedia.org/wiki/Stack_(abstract_data_type))遍历搜索空间，以确定下一个节点。**优点**:适合深度图形，内存效率高。**缺点:**会卡在循环中。

[](https://en.wikipedia.org/wiki/Depth-first_search) [## 深度优先搜索-维基百科

### 需要额外的引用来验证。通过增加对可靠来源的引用来改进这篇文章。无来源…

en.wikipedia.org](https://en.wikipedia.org/wiki/Depth-first_search) 

*   **IDFS —** 遍历搜索空间，使用一个 [*LIFO 堆栈*](https://en.wikipedia.org/wiki/Stack_(abstract_data_type)) *，*来确定下一个节点，当它到达某个深度时，它清除堆栈，增加深度限制，并再次开始搜索。

[](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search) [## 迭代深化深度优先搜索-维基百科

### 需要额外的引用来验证。通过增加对可靠来源的引用来改进这篇文章。无来源…

en.wikipedia.org](https://en.wikipedia.org/wiki/Iterative_deepening_depth-first_search) 

*   **BFS-** 使用一个 [*队列 FIFO*](https://en.wikipedia.org/wiki/Queue_(abstract_data_type)) 遍历搜索空间以确定下一个节点。

[](https://en.wikipedia.org/wiki/Breadth-first_search) [## 广度优先搜索-维基百科

### 需要额外的引用来验证。通过增加对可靠来源的引用来改进这篇文章。无来源…

en.wikipedia.org](https://en.wikipedia.org/wiki/Breadth-first_search) 

# 知情搜索

![](img/b9a3a3c408807956d497f29e969986fc.png)

当我们知道成本或对各州之间的成本有一个可靠的估计时，就使用知情搜索。

**UCF-** 用一个[优先级队列](https://en.wikipedia.org/wiki/Priority_queue)和一个分数遍历搜索空间。给定状态的分数是通过沿着它被发现的路径从父状态到达该状态的成本来计算的。

[](https://algorithmicthoughts.wordpress.com/2012/12/15/artificial-intelligence-uniform-cost-searchucs/) [## 人工智能-统一成本搜索(UCS)

### 在这篇文章中，我将讨论在加权图中寻找最短路径的统一成本搜索算法…

algorithmicthoughts.wordpress.com](https://algorithmicthoughts.wordpress.com/2012/12/15/artificial-intelligence-uniform-cost-searchucs/) 

**A* —** 用优先级队列和分数遍历搜索空间。一个状态的得分是通过沿着它被发现的路径从父状态到达该状态的成本，结合给定状态的启发式值来计算的。

试探法的*容许值*必须符合以下两个性质。

*   一个状态的启发式值必须小于从一个状态到目标节点的最低成本路径。
*   试探值必须小于到状态的路径和每个父节点的试探值之间的成本值。

[](https://en.wikipedia.org/wiki/A*_search_algorithm) [## 搜索算法-维基百科

### 斯坦福研究所(现在的斯坦福国际研究所)的彼得·哈特、尼尔斯·尼尔森和伯特伦·拉斐尔首先发表了…

en.wikipedia.org](https://en.wikipedia.org/wiki/A*_search_algorithm) 

**IDA***——IDFS 版的 A *

[](https://en.wikipedia.org/wiki/Iterative_deepening_A*) [## 迭代深化 A* -维基百科

### 迭代深化 A* ( IDA*)是一种图遍历和路径搜索算法，可以找到 a…

en.wikipedia.org](https://en.wikipedia.org/wiki/Iterative_deepening_A*) 

# 本地搜索

![](img/7343f72f260fa7bfede7072fcf4020a9.png)

当有不止一个可能的目标状态，但是一些结果比其他结果更好，并且我们需要发现最好的结果时，我们使用局部搜索算法。大量用于机器学习算法的优化。

**爬山-** 一种贪婪的搜索方法，基于最小值损害下一个状态，直到达到局部最大值。

[](https://en.wikipedia.org/wiki/Hill_climbing) [## 爬山-维基百科

### 需要额外的引用来验证。通过增加对可靠来源的引用来改进这篇文章。无来源…

en.wikipedia.org](https://en.wikipedia.org/wiki/Hill_climbing) 

**模拟退火-** 从爬山开始，直到达到局部最大值，然后使用温度函数来决定是停止还是继续处于更差的状态，希望找到更好的状态

[](https://en.wikipedia.org/wiki/Simulated_annealing) [## 模拟退火-维基百科

### 这种在模拟退火算法中实现的缓慢冷却的概念被解释为…

en.wikipedia.org](https://en.wikipedia.org/wiki/Simulated_annealing) 

**GSAT**—CNF 领域爬山的实现。为每个可能的参数选择一组随机的布尔值，如果这些值匹配所有的前提条件，则返回目标状态，否则我们翻转这些值，以满足目标状态的最大数量的可能前提条件，然后为我们在最后一次迭代中翻转的每个布尔值重复这个过程。

 [## WalkSAT -维基百科

### WalkSAT 首先选择一个对当前赋值不满意的子句，然后翻转该子句中的一个变量…

en.wikipedia.org](https://en.wikipedia.org/wiki/WalkSAT) 

**遗传搜索** -生成初始种群状态，使用适应度函数删除阈值以下具有最低值的状态。随机组合幸存者，然后变异一对夫妇位，并评估适应度和重复。

[](https://en.wikipedia.org/wiki/Genetic_algorithm) [## 遗传算法-维基百科

### 在计算机科学和运筹学中，遗传算法(GA)是一种元启发式算法，其灵感来自于…

en.wikipedia.org](https://en.wikipedia.org/wiki/Genetic_algorithm) 

**波束搜索** -使用模型当前和先前输出的前 n 个似然值执行统一成本搜索。

 [## 波束搜索-维基百科

### 在计算机科学中，波束搜索是一种启发式搜索算法，它通过扩展最有希望的…

en.wikipedia.org](https://en.wikipedia.org/wiki/Beam_search) 

**蒙特卡罗搜索—** 一种随机搜索算法，当终止时将返回正确搜索结果的最佳估计。蒙特卡罗算法总是很快，但只是可能正确。

![](img/442e4f32a61ef9cd254bf998db866796.png)[](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) [## 蒙特卡洛树搜索-维基百科

### 在计算机科学中，蒙特卡罗树搜索(MCTS)是一种启发式搜索算法，用于某些类型的决策…

en.wikipedia.org](https://en.wikipedia.org/wiki/Monte_Carlo_tree_search) 

**拉斯维加斯搜索**是一种随机搜索算法，与蒙特卡洛不同，它只会在找到正确结果时返回。拉斯维加斯的算法总是正确的，但可能只是很快。

```
*// Las Vegas algorithm*
2 repeat:
3     k = RandInt(n)
4     **if** A[k] == 1,
5         **return** k;
```

 [## 拉斯维加斯算法-维基百科

### 在计算中，拉斯维加斯算法是一种随机算法，总能给出正确的结果；也就是说，它总是…

en.wikipedia.org](https://en.wikipedia.org/wiki/Las_Vegas_algorithm) 

**大西洋城搜索** —是一种有界概率多项式时间搜索算法，结合了拉斯维加斯和蒙特卡洛搜索算法的优点和缺点。

 [## 大西洋城算法-维基百科

### 大西洋城算法是一种概率多项式时间算法，其正确回答率至少为 75%。

en.wikipedia.org](https://en.wikipedia.org/wiki/Atlantic_City_algorithm) 

# 后续步骤

如果你喜欢这篇文章，请马上关注新内容，并在 medium 或 twitter 上关注我。要开始试验这些算法，请查看 [Azure 笔记本](https://docs.microsoft.com/en-us/azure/notebooks/?WT.mc_id=medium-blog-abornst)以及 Azure 上 [CosmosDB](https://docs.microsoft.com/en-us/azure/cosmos-db/?WT.mc_id=medium-blog-abornst) 的图形数据库特性。

如果你还没有，你可以在下面免费订阅 Azure。

[](https://azure.microsoft.com/en-us/offers/ms-azr-0044p/?WT.mc_id=medium-blog-abornst) [## 立即创建您的 Azure 免费帐户| Microsoft Azure

### 开始享受 12 个月的免费服务和 200 美元的信用点数。立即使用 Microsoft Azure 创建您的免费帐户。

azure.microsoft.com](https://azure.microsoft.com/en-us/offers/ms-azr-0044p/?WT.mc_id=medium-blog-abornst) 

# 关于作者

亚伦(阿里)博恩施泰因 是一个狂热的人工智能爱好者，对历史充满热情，致力于新技术和计算医学。作为微软云开发倡导团队的开源工程师，他与以色列高科技社区合作，用改变游戏规则的技术解决现实世界的问题，然后将这些技术记录在案、开源并与世界其他地方共享。