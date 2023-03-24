# 利用多臂 Bandit 解决推荐系统的冷用户问题

> 原文：<https://towardsdatascience.com/solving-cold-user-problem-for-recommendation-system-using-multi-armed-bandit-d36e42fe8d44?source=collection_archive---------2----------------------->

## 本文全面概述了如何使用 Multi-Armed Bandit 向新用户推荐电影

![](img/8e0903a4083dc50cacf8b0253db34b26.png)

Umm not the cold user we are referring to

作者:Animesh Goyal，Alexander Cathis，Yash Karundia，Prerana Maslekar

# **简介**

在紧张忙碌的工作一天后，你是否经常会觉得接下来我该看什么？至于我——是的，而且不止一次。从网飞到 Prime Video，考虑到现代消费者对个性化内容的巨大需求，构建健壮的电影推荐系统的需求极其重要。

一旦回到家，坐在电视机前似乎是一种徒劳的练习，没有控制力，也不记得我们消费的内容。我们倾向于选择一个智能平台，它能理解我们的口味和偏好，而不只是自动运行。

在这篇文章中，我试图建立一个推荐系统，在最短的时间内向你推荐最好的电影。该推荐系统也可以用于推荐广泛的项目。例如，它可以用来推荐产品、视频、音乐、书籍、新闻、脸书朋友、衣服、Twitter 页面、Android/ios 应用程序、酒店、餐馆、路线等。

# **推荐系统的类型**

![](img/9256f75d88570fda4a1340b21e01bf29.png)

Types of Recommendation system

# **现有解决方案**

## *1。基于流行度的推荐系统*

顾名思义，基于流行度的推荐系统顺应了这一趋势。它基本上使用了现在流行的东西。例如，如果任何产品通常被每个新用户购买，那么它可能会向刚刚注册的用户推荐该产品。

基于流行度的推荐系统的问题在于，这种方法不能实现个性化，也就是说，即使你知道用户的行为，你也不能相应地推荐项目。

## *2。基于内容的推荐系统*

![](img/dfeca210f3ebe6611d698396ae741b87.png)

Content-based recommendation system

基于内容的过滤使用该技术来分析一组文档和先前由用户评级的项目描述，然后基于那些评级项目的特征来建立用户兴趣的简档或模型。使用该简档，推荐系统可以过滤出适合用户的建议。

基于内容的推荐系统的问题是，如果内容不包含足够的信息来精确地区分项目，推荐将不会精确地结束

## *3。基于协作的推荐系统*

![](img/97a77ee031a06ff16a8dbf262c47c6ef.png)

Collaborative based recommendation system

基于协作的推荐系统背后的关键思想是相似的用户分享相同的兴趣，并且相似的项目被用户喜欢。基于协作的推荐系统有两种类型:基于用户的和基于项目的。我们将使用基于用户的过滤过程。

但是这些都无法解决冷用户的问题。

# **问题—冷启动**

那么，什么是冷启动问题呢？这个术语来源于汽车。当天气真的很冷时，发动机的启动会出现问题，但一旦达到最佳工作温度，它就会平稳运行。对于推荐引擎,“冷启动”仅仅意味着环境还不是引擎提供最佳可能结果的最佳条件。我们的目标是尽量缩短加热发动机的时间。冷启动的两个不同类别:产品冷启动和用户冷启动。在这个博客中，我们集中讨论用户冷启动问题。

为了解决这个问题，我们引入了使用多臂土匪的概念

# **多臂土匪(MAB)**

![](img/5de49e3183964f2790f2882dacd30841.png)

Multi-Armed Bandit Problem

多臂土匪问题是一个经典的问题，它模拟了一个代理人(或计划者或中心)，该代理人希望最大化其总报酬，同时希望获得新知识(“探索”)并基于现有知识优化他或她的决策(“利用”)。MAB 问题描述了这样一个场景:游戏者面临着探索和剥削之间的权衡，前者乐观地拉动探索较少的手臂以寻找具有更好回报的手臂，后者拉动已知最好的手臂直到当前时刻，以获得最大回报。

# **MAB 为冷用户**

我们的目标是使用不同的 bandit 算法为用户探索/利用最佳推荐。有几种 MAB 算法，每一种都在不同程度上倾向于开发而不是探索。最流行的三种是ε贪婪、Thompson 抽样和置信上限 1 (UCB-1):

## *1。ε贪心*

![](img/622db7ddbd0738f36efd5d3e14c051f8.png)

Epsilon Greedy Approach

Epsilon Greedy，顾名思义，是三种 MAB 算法中最贪婪的。在ε贪婪实验中，常数ε(值在 0 和 1 之间)由用户在实验开始前选择。当将联系人分配到活动的不同变量时，随机选择的变量被选择ε次。其余的 1-ε时间，选择已知收益最高的变量。ε越大，该算法越有利于探索。对于我们所有的例子，ε设置为 0.1。

## *2。置信上限(UCB*

![](img/db8a7560d30f9c2715d4ced55a66b99c.png)

Upper confidence bound approach

对于活动的每个变体，我们将确定置信上限(UCB ),它代表我们对该变体的可能收益的最高猜测。该算法会将联系人分配给具有最高 UCB 的变量。

每个变量的 UCB 是根据变量的平均收益以及分配给变量的联系数计算的，公式如下:

![](img/9ba225de51c238c229c87d54549287d4.png)

详细描述:[https://www.youtube.com/watch?v=RPbtzWgzD9M](https://www.youtube.com/watch?v=RPbtzWgzD9M)

## *3。汤普森采样*

![](img/a6c01b8f2ca8d2d66430a85719bbb3b8.png)

Thompson Sampling Approach

相比之下，汤普森抽样是一种更有原则的方法，它可以在边际情况下产生更平衡的结果。对于每个变体，我们使用观察到的结果建立真实成功率的概率分布(出于计算原因，最常见的是 beta 分布)。对于每个新联系人，我们从对应于每个变体的 beta 分布中采样一个可能的成功率，并将该联系人分配给具有最大采样成功率的变体。我们观察到的数据点越多，我们对真实的成功率就越有信心，因此，随着我们收集到更多的数据，抽样成功率将越来越有可能接近真实的成功率。

详细描述:[https://www.youtube.com/watch?v=p701cYQeqew](https://www.youtube.com/watch?v=p701cYQeqew)

# **方法论**

![](img/944af5b3a59f64ebc8897683f49baf10.png)

Infrastructure for solving MAB cold start problem

我们已经使用了[电影镜头](https://grouplens.org/datasets/movielens/)数据集来解决包含 4 个文件的 MAB 问题。这些文件被合并并用于协同过滤。

![](img/ef45f28c8785e4ea0d5f11ca937a5d9c.png)

Sparse Matrix from User Ratings

![](img/d7afcb2c1ecc708b92fe8c2cf7ffee54.png)

Matrix after application of Collaborative Filtering and Clustering

从上述步骤中获得聚类后，我们将它们按降序排序，并首先推荐每个聚类中评分最高的项目，这有助于更快地了解用户的偏好。例如，上述矩阵中的对应θ1 将具有等于{i5，i4，i1，i3，i2，i6，i7}的项目排序列表，而对于θ2，我们将具有{i4，i2，i3，i5，i6，i1，i7}。

该方法被应用于向最初是新用户的用户做出推荐，这些新用户保持冷淡，直到他们提供了关于推荐项目的一定量的偏好/反馈。确定用户何时不再冷的阈值可以是不同的。在这篇博客中，我们使用 NDCG 指标(排名列表的大小)测量了 5、10、15、40、100 条推荐后的性能。一旦用户不再感冒，系统就可以切换到个性化预测模型。

# **奖励功能**

![](img/4ebb80f4bfae444c0b98176289fec240.png)

Reward function

奖励函数被定义为用户提供的反馈与用户提供的最高评级的比率。这样，当一个用户对一个推荐的商品给出较高的反馈时，我们就会有较高的奖励。例如，假设冷用户进入系统并使用 UCB 算法，从θ1 中选择项目 i5。如果用户给出的反馈是 4，那么对于聚类θ1，这部电影的回报将是 4/5 = 0.8。在下一个建议中，如果选择了群集θ2，则选择了项目 i4，并且它收到的反馈为 1，则群集θ2 的奖励为 1/5 = 0.2。此时，聚类θ1 的平均回报为 0.8，聚类θ2 的平均回报为 0.2。下一个建议是从θ1 开始的 i1，θ1 代表此时平均奖励最高的手臂。

# **评估指标**

我们将使用 NDCG(标准化折扣累积收益)，它更重视早期的准确推荐，而不是后期的推荐。由于我们专注于对冷用户的推荐，我们重视早期的准确结果。

![](img/f60056df49c2c91221096daf0d5bd160.png)

在哪里

> DCG(u):目标用户 u 的预测排名的折扣累积收益
> 
> DCG∫(u):地面真相
> 
> n:结果集中的用户数量
> 
> r (u，1):首先推荐给用户 u 的项目的评级(根据用户反馈)
> 
> r (u，t):用户对依次推荐的项目的反馈
> 
> t:推荐时间
> 
> t:排名列表的大小

# **结果**

下表提供了等级规模为 5、10、15、40 和 100 的 NDCG 分数。汤普森的表现一直比贪婪和 UCB 要好。预热时间更短，并且能够快速提供更好的结果。另一方面，UCB 在不同的 N 和 t 值下表现最差

![](img/6fcd8142f63e79c1209eb8fb63e910f7.png)

NDCG Results for varying rank-size and number of users

# **嵌入网络**

虽然嵌入网络通常用于不同的问题，但我们希望看到它们在冷启动问题上的有效性。嵌入网络是学习特征之间的嵌入的神经网络。有了这些知识，它应该能够填充空值。因此，对于电影镜头数据集，它可以嵌入用户和电影之间的关系，然后可以填充用户没有为电影提供的缺失评级。虽然我们刚刚用协同过滤执行了这项任务，但使用神经网络可以学习复杂的非线性特征。

![](img/cf4b9d33b10aea54fa699550bfddeb08.png)

Embedding network

因此，如上所示的嵌入网络可用于填充以下嵌入的缺失值:

![](img/20c5dbc022de2ee9531d2a2e4248e480.png)

一般嵌入式网络用于有温暖用户的推荐系统。这样，网络就不必猜测太多。对于热情的用户来说，学习到的关系有望被很好地建立，并且不会发生变化。通过这种设置，嵌入式神经网络可以在密集的数据集上训练一次，学习非常复杂的关系，并给出受过良好教育的建议。然而，我们的问题没有温暖用户的奢侈，导致一个大的数据集。

对于冷启动问题，我们必须向没有提供反馈的用户提供好的建议。本质上，他们的数据是空的，对于这个用户没有任何可以推断的关系。对于这个问题，使用嵌入网络的简单解决方案可以是首先给出一个随机的推荐，并且对于每个后续的推荐:接受反馈，重新训练，对预测的评级进行排序，并且返回最高的预测推荐。然而，这将导致疯狂的计算成本和延迟的推荐，这可能给用户体验带来灾难性的后果。

我们的想法是，即使嵌入网络不知道冷用户，用户和项目之间的数据集关系的网络知识仍然可以提供一些价值。因此，我们希望分析性能，如果对于每个用户请求，我们考虑多个建议。这些推荐可以来自全局推荐器，例如随机、全局平均或最受欢迎。它们也可能来自 MAB 算法，如汤普森算法、ε-贪婪算法和 UCB 算法。我们可以首先让嵌入式网络估计这些推荐中哪一个得分最高，而不是直接反馈单个算法推荐。最高的推荐将被提供给用户。根据计算和时间限制，网络可以根据用户的反馈进行重新训练。

这个想法在一个玩具例子中得到了验证，这个例子重用了前面章节中描述的 MAB 评估框架。由于时间限制，嵌入式网络几乎没有接受任何训练(5 个时期)，也没有超参数调整。它也没有根据用户反馈进行再培训。对于每个用户请求，它会考虑 3 个随机推荐，并给用户一个它认为得分最高的。然而，在这个 15 次试验和 5 个随机用户的小测试中，它给出了比 UCB 算法更好的结果。必须注意，严格使用随机建议会从下面所示的有限测试中获得更好的结果:

![](img/262e551cb4012de3520fcc44bd622981.png)

虽然这些结果并不是突破性的，但应该指出的是，进一步的实验有机会展示更好的性能。更多的训练，实际上调整超参数，根据用户反馈重新训练网络，以及使用其他采样推荐器(而不仅仅是随机的)都有可能极大地提高性能。

# **结论和未来工作**

尽管由于缺乏对用户的了解，冷启动用户在推荐系统中造成了独特的问题，但 MAB 在推荐电影方面做得相当好，并且随着数据输入不断发展。总而言之，我们的贡献是:

> 多臂土匪问题模型选择的形式化
> 
> 使用 UCB、汤姆逊抽样和ε贪婪算法，这是一种为没有先验边信息的用户进行推荐的有效方法
> 
> 基于电影镜头数据集的 NDCG 实证评估
> 
> 推荐系统中嵌入网络的有效性评估

对于未来的工作，我们可以尝试

> 使用不同的 bandit 算法作为武器，正如我们看到的，epsilon-greedy 在一开始执行得更好，而 Thompson 在更多次迭代中执行得更好
> 
> 扩展这个实现以在每次迭代中提供多个推荐——目前，最好的电影被反复推荐
> 
> 我们依靠 NDCG 评分来评估我们的方法，主要是因为我们在调查推荐质量。总的来说，它表现出很高的准确性，但我们可能会检查其他指标，以确保公平的评估
> 
> 调整嵌入网络的超参数，根据用户反馈重新训练网络，以及使用其他采样推荐器(而不仅仅是随机的)

非常感谢您的阅读，我们希望您能从项目中学到一些东西！

请随意查看:

[*Github 资源库本帖*](https://github.com/animeshgoyal9/Multi-Armed-Bandit-for-cold-user-in-Recommendation-System)

[*我的其他中等帖子*](https://medium.com/@animeshgoyal)

[*我的 Linkedin 个人资料*](https://www.linkedin.com/in/animesh-goyal-b33080101/)

**参考文献**

[https://medium . com/datadriveninvestor/how-to-build-a-recommender-system-RS-616 c 988d 64 b 2](https://medium.com/datadriveninvestor/how-to-built-a-recommender-system-rs-616c988d64b2)

[https://hal.inria.fr/hal-01517967/document](https://hal.inria.fr/hal-01517967/document)

[http://klerisson.github.io/papers/umap2017.pdf](http://klerisson.github.io/papers/umap2017.pdf)

[https://www . analyticsvidhya . com/blog/2018/06/comprehensive-guide-recommendation-engine-python/](https://www.analyticsvidhya.com/blog/2018/06/comprehensive-guide-recommendation-engine-python/)

[https://towards data science . com/variable-implementations-of-collaborative-filtering-100385 c 6 dfe 0](/various-implementations-of-collaborative-filtering-100385c6dfe0)

[https://medium . com/the-andela-way/foundations-of-machine-learning-singular-value-decomposition-SVD-162 AC 796 c 27d](https://medium.com/the-andela-way/foundations-of-machine-learning-singular-value-decomposition-svd-162ac796c27d)

[https://sebastianraschka . com/Articles/2014 _ PCA _ step _ by _ step . html](https://sebastianraschka.com/Articles/2014_pca_step_by_step.html)

[https://medium . com/datadriveninvestor/how-to-build-a-recommender-system-RS-616 c 988d 64 b 2](https://medium.com/datadriveninvestor/how-to-built-a-recommender-system-rs-616c988d64b2)

[https://medium . com/@ iliazaitsev/how-to-implementation-a-recommendation-system with-deep-learning-and-py torch-2d 40476590 F9](https://medium.com/@iliazaitsev/how-to-implement-a-recommendation-system-with-deep-learning-and-pytorch-2d40476590f9)

[https://github . com/dev for fu/py torch _ playground/blob/master/movie lens . ipynb](https://github.com/devforfu/pytorch_playground/blob/master/movielens.ipynb)

https://nipunbatra.github.io/blog/2017/recommend-keras.html

[https://towards data science . com/neural-network-embeddings-explained-4d 028 E6 f 0526](/neural-network-embeddings-explained-4d028e6f0526)

[https://medium . com/hey car/neural-network-embeddings-from-inception-to-simple-35 e 36 CB 0 c 173](https://medium.com/heycar/neural-network-embeddings-from-inception-to-simple-35e36cb0c173)