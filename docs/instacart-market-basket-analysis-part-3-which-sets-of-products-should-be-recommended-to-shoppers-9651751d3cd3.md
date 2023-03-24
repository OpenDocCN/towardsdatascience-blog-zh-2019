# Instacart 市场购物篮分析第三部分:应向购物者推荐哪几组产品？

> 原文：<https://towardsdatascience.com/instacart-market-basket-analysis-part-3-which-sets-of-products-should-be-recommended-to-shoppers-9651751d3cd3?source=collection_archive---------22----------------------->

## [深入分析](https://medium.com/towards-data-science/in-depth-analysis/home)

## 基于土星云的 Instacart 产品关联规则挖掘

欢迎来到充满希望的数据科学和机器学习平台时代！这些平台具有开箱即用的优势，因此团队可以从第一天就开始分析数据。使用开源工具，你需要手工组装很多部件，可以这么说，任何做过 DIY 项目的人都可以证明，理论上比实践上容易得多。明智地选择一个数据科学和 ML 平台(指的是一个灵活的、允许合并和继续使用开源的平台)可以让企业两全其美:尖端的开源技术和对数据项目的可访问、可治理的控制。

![](img/66bcc8529f39839b6f44f8522ab52029.png)

最终，数据科学和 ML 平台是关于时间的。当然，这意味着在流程的所有部分节省时间(从连接数据到构建 ML 模型到部署)。但这也是为了减轻人工智能入门的负担，让企业现在就开始投入，而不是等到技术确定下来，人工智能世界变得更加清晰(因为剧透警告:这可能永远不会发生)。开始人工智能之旅令人生畏，但数据科学和 ML 平台可以减轻这一负担，并提供一个框架，允许公司边走边学习。

# **土星云**

[土星云](https://www.saturncloud.io/?source=jl-3)允许数据科学家轻松地在云上提供和托管他们的工作，而不需要专门的开发人员。然后，数据科学家可以在 Juptyer 笔记本中工作，该笔记本位于您指定的服务器上，由系统创建。所有软件、网络、安全和图书馆的设置都由土星云系统自动处理。然后，数据科学家可以专注于实际的数据科学，而不是围绕它的乏味的基础设施工作。

您还可以使用链接与公众或团队成员共享您的 Juypter 笔记本。这消除了理解如何使用 GitHub 进行基础数据科学项目的需要。如果您确实知道如何使用 GitHub，它仍然提供了一种快速方便的方法来测试和开发代码。因此，数据科学家可以专注于数据科学，并以比其他方式更快的速度完成项目。

![](img/e20042bf5c5c01cbcfd53569befdf6c1.png)

这是 3 篇文章系列的最后一篇，我将举例说明如何使用 Saturn Cloud 来应对 Instacart 市场购物篮分析挑战。在[第 1 部分](/instacart-market-basket-analysis-part-1-which-grocery-items-are-popular-61cadbb401c8)中，我探索了数据集，以更细致地了解 Instacart 平台上客户购物行为的细节。在《T4》第二部中，我对客户进行了细分，以找出是什么因素将他们区分开来。

# **动机**

我希望使用 **Apriori 算法**在 Python 中运行关联分析，以导出形式为 ***{A} - > {B}*** 的规则。然而，我很快发现它不是标准 Python 机器学习库的一部分。尽管存在一些实现，但我找不到一个能够处理大型数据集的实现。在我的例子中，“大”是一个包含 3200 万条记录的 orders 数据集，包含 320 万个不同的订单和大约 50K 个不同的项目(文件大小刚刚超过 1 GB)。

因此，我决定自己实现算法来生成那些简单的 ***{A} - > {B}*** 关联规则。因为我只关心理解任何给定的项目对之间的关系，所以使用先验知识来得到大小为 2 的项目集就足够了。我经历了各种迭代，将数据分成多个子集，这样我就可以在内存小于 10 GB 的机器上运行 cross-tab 和 combinations 等功能。但是，即使使用这种方法，在我的内核崩溃之前，我也只能处理大约 1800 个项目……这时，我了解了 Python 生成器的奇妙世界。

![](img/2db890b9281995c2662b724069c3cdc6.png)

## **Python 生成器**

简而言之，生成器是一种特殊类型的函数，它返回可迭代的项目序列。然而，与一次返回所有值的常规函数不同，生成器一次生成一个值。要获得集合中的下一个值，我们必须请求它——要么通过显式调用生成器的内置“next”方法，要么通过 for 循环隐式调用。

这是生成器的一个重要特性，因为这意味着我们不必一次将所有的值都存储在内存中。我们可以一次加载并处理一个值，完成后丢弃该值，然后继续处理下一个值。这个特性使得生成器非常适合创建项目对并计算它们的共现频率。

这里有一个具体的例子来说明我们正在努力实现的目标:

*   *获取给定订单的所有可能商品对*

```
order 1:  apple, egg, milk —-> item pairs: {apple, egg}, {apple, milk}, {egg, milk}order 2:  egg, milk --> item pairs: {egg, milk}
```

*   *统计每个项目对出现的次数*

```
eg: {apple, egg}: 1{apple, milk}: 1{egg, milk}: 2
```

下面是实现上述任务的生成器:

```
def get_item_pairs(order_item): # For each order, generate a list of items in that order for order_id, order_object in groupby(orders, lambda x: x[0]): item_list = [item[1] for item in order_object] # For each item list, generate item pairs, one at a time for item_pair in combinations(item_list, 2): yield item_pair
```

上面编写的 *get_item_pairs()* 函数为每个订单生成一个商品列表，并为该订单生成商品对，一次一对。

*   第一个项目对被传递给计数器，计数器记录一个项目对出现的次数。
*   获取下一个项目对，并再次传递给计数器。
*   这个过程一直持续到不再有项目对。
*   使用这种方法，我们最终不会使用太多内存，因为在计数更新后，项目对会被丢弃。

## **Apriori 算法**

**Apriori** 是一种在事务数据库上进行频繁项集挖掘和关联规则学习的算法。它通过识别数据库中频繁出现的单个项目，并将其扩展到越来越大的项目集，只要这些项目集在数据库中出现得足够频繁。

Apriori 使用“自下而上”的方法，一次扩展一个频繁子集(这一步被称为*候选生成*)，并根据数据测试候选组。当没有找到进一步的成功扩展时，算法终止。

![](img/e12c72a48fcf1cfaa42ae81be9d35f26.png)

Apriori 使用广度优先搜索和哈希树结构来有效地计算候选项目集。它从长度为(k — 1)的项集生成长度为 k 的候选项集。然后，它修剪具有不频繁子模式的候选。根据向下闭包引理，候选集包含所有频繁 k 长项目集。之后，它扫描事务数据库以确定候选项中的频繁项集。

下面是一个先验的例子，假设最小发生阈值为 3:

```
order 1: apple, egg, milkorder 2: carrot, milkorder 3: apple, egg, carrotorder 4: apple, eggorder 5: apple, carrotIteration 1:  Count the number of times each item occursitem set      occurrence count{apple}              4{egg}                3{milk}               2{carrot}             2=> {milk} and {carrot} are eliminated because they do not meet the minimum occurrence threshold. Iteration 2: Build item sets of size 2 using the remaining items from Iteration 1item set           occurence count{apple, egg}             3=> Only {apple, egg} remains and the algorithm stops since there are no more items to add.
```

如果我们有更多的订单和商品，我们可以继续迭代，构建包含 2 个以上元素的商品集。对于我们试图解决的问题(即:寻找项目对之间的关系)，实现先验以获得大小为 2 的项目集就足够了。

# **关联规则挖掘**

一旦使用 apriori 生成了项目集，我们就可以开始挖掘关联规则。假设我们只查看大小为 2 的项目集，我们将生成的关联规则将采用{ A }--> { B }的形式。这些规则的一个常见应用是在推荐系统的领域，其中购买了商品 A 的顾客被推荐商品 b。

![](img/931a4613c7782fbe18cbb4b4db883286.png)

以下是评估关联规则时要考虑的 3 个关键指标:

## **1 —支架**

这是包含项目集的订单的百分比。在上面的示例中，总共有 5 个订单，{apple，egg}出现在其中的 3 个订单中，因此:

```
support{apple,egg} = 3/5 or 60%
```

apriori 所需的最小支持阈值可以根据您对领域的了解来设置。例如，在这个杂货数据集中，由于可能有数千个不同的商品，而一个订单只能包含这些商品中的一小部分，因此将支持阈值设置为 0.01%可能是合理的。

## **2 —置信度**

给定两个商品 A 和 B，置信度度量商品 B 被购买的次数的百分比，给定商品 A 被购买。这表示为:

```
confidence{A->B} = support{A,B} / support{A}
```

置信值的范围是从 0 到 1，其中 0 表示在购买 A 时从不购买 B，1 表示每当购买 A 时总是购买 B。请注意，置信度是有方向性的。这意味着，我们还可以计算出商品 A 被购买的次数百分比，假设商品 B 被购买:

```
confidence{B->A} = support{A,B} / support{B}
```

在我们的例子中，假设苹果被购买，鸡蛋被购买的次数百分比是:

```
confidence{apple->egg} = support{apple,egg} / support{apple}= (3/5) / (4/5)= 0.75 or 75%
```

这里我们看到所有包含 egg 的订单也包含 apple。但是，这是否意味着这两个项目之间有关系，或者它们只是偶然以相同的顺序同时出现？为了回答这个问题，我们来看看另一个衡量标准，它考虑了这两个项目的受欢迎程度。

## **3 —升降机**

给定两个项目 A 和 B，lift 表示 A 和 B 之间是否存在关系，或者这两个项目是否只是偶然地(即:随机地)以相同的顺序一起出现。与置信度度量不同，其值可能随方向而变化(例如:置信度{A->B}可能不同于置信度{B->A}), lift 没有方向。这意味着升力{A，B}总是等于升力{B，A}:

```
lift{A,B} = lift{B,A} = support{A,B} / (support{A} * support{B})
```

在我们的例子中，我们计算升力如下:

```
lift{apple,egg} = lift{egg,apple}= support{apple,egg} / (support{apple} * support{egg})= (3/5) / (4/5 * 3/5)= 1.25
```

理解 lift 的一种方法是将分母视为 A 和 B 以相同顺序出现的可能性，如果它们之间没有关系的话。在上面的例子中，如果 apple 出现在 80%的订单中，egg 出现在 60%的订单中，那么如果它们之间没有关系，我们会期望它们在 48%的时间里以相同的顺序一起出现(即:80% * 60%)。另一方面，分子代表苹果和鸡蛋以相同顺序出现的频率。在这个例子中，这是 60%的时间。取分子并除以分母，我们得到苹果和鸡蛋实际上以相同的顺序出现的次数比它们之间没有关系时多多少次(即:它们只是随机出现在一起)。

总之，lift 可以采用以下值:

*   **lift = 1** 暗示 A 和 B 之间没有关系(即:A 和 B 只是偶然出现在一起)
*   **lift > 1** 暗示 A 和 B 之间存在正相关关系(即:A 和 B 一起出现的次数比随机出现的次数多)
*   **lift < 1** 暗示 A 和 B 之间存在负相关关系(即:A 和 B 一起出现的频率低于随机出现的频率)

在这个例子中，苹果和鸡蛋一起出现的次数比随机出现的次数多 1.25 倍，所以我断定它们之间存在正相关关系。有了 apriori 和关联规则挖掘的知识，让我们深入到数据和代码中，看看我能展现什么关系！

# **insta cart 数据上的关联规则挖掘**

云中木星笔记本入门使用土星云非常直观。笔记本运行后，我可以轻松地从笔记本内部与公众分享。为了证明这一点，你可以在这里查看我的关联规则挖掘笔记本[中的完整代码。](https://www.saturncloud.io/published/khanhnamle1994/instacart-notebooks/notebooks/Association-Rule-Mining.ipynb?source=jl-3)

在数据预处理步骤之后，我编写了几个助手函数来辅助主要的关联规则函数:

*   **freq()** 返回项目和项目对的频率计数。
*   **order_count()** 返回唯一订单的数量。
*   **get_item_pairs()** 返回生成项目对的生成器，一次一个。
*   **merge_item_stats()** 返回与项目相关联的频率和支持。
*   **merge_item_name()** 返回与项目相关的名称。

下面的 GitHub Gist 中显示了大的 **association_rules()** 函数:

Full Link: [https://gist.github.com/khanhnamle1994/934d4c928c836d38879d3dd6637d9904](https://gist.github.com/khanhnamle1994/934d4c928c836d38879d3dd6637d9904)

此时，我能够利用 Saturn Cloud 的 GPU 功能来处理我的工作负载。本质上，[土星云](https://www.saturncloud.io/?source=jl-3)让我只需点击一下鼠标就能部署 Spark 或 Dask 集群。这简化了处理非常昂贵的计算的问题，只需点击一下鼠标就可以实现分布式计算。考虑到这种关联规则挖掘算法相当昂贵，访问 GPU 在这里是一个明显的赢家。

在最小支持度为 0.01 的**订单**数据帧上调用 **association_rules()** 函数后，得到如下结果(正在挖掘的 7 个规则样本):

![](img/803e9b56aa58bb0dd02b0d315070ae08.png)

从上面的输出中，我观察到顶级关联并不令人惊讶，一种口味的商品与同一商品系列中的另一种口味一起购买(例如:草莓 Chia 农家奶酪与蓝莓 Acai 农家奶酪，鸡肉猫粮与土耳其猫粮，等等)。如上所述，关联规则挖掘的一个常见应用是在推荐系统领域。一旦项目对被确定为具有积极的关系，就可以向顾客提出建议以增加销售。希望在这个过程中，我们还可以向顾客介绍他们以前从未尝试过甚至想象不到的东西！

# **结论**

当试图启动数据科学团队项目时，开发运维可能会非常困难。在幕后，有很多工作要做，以建立数据科学家用于工作的实际平台:创建服务器，安装必要的软件和环境，设置安全协议，等等。使用 Saturn Cloud 托管 Jupyter 笔记本电脑，同时还负责版本管理，并能够根据需要进行扩展，这可以极大地简化您的生活，缩短上市时间，降低成本和对专业云技能的需求。

我希望你喜欢这个由 3 部分组成的系列，通过 Instacart 市场购物篮分析挑战来了解使用 Saturn Cloud 的好处。作为数据科学和机器学习平台运动的一部分，当团队不必在管理、组织或重复任务上花费宝贵的时间时，像土星云这样的框架可以打开真正的数据创新之门。事实是，在人工智能时代，任何规模的企业都无法承受没有数据科学平台的工作，该平台不仅可以支持和提升他们的数据科学团队，还可以将整个公司提升到最高水平的数据能力，以实现最大可能的影响。