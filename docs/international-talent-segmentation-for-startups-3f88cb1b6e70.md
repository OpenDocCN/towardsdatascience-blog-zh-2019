# 雇用海外人才的艺术

> 原文：<https://towardsdatascience.com/international-talent-segmentation-for-startups-3f88cb1b6e70?source=collection_archive---------24----------------------->

![](img/fc9297f023c3bbdd012bff3375448181.png)

Photo by [Brooke Cagle](https://unsplash.com/@brookecagle?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/office?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

*使用 K-Modes 的无监督学习来识别区域技术中心*

1.  **设置**

在美国招聘既昂贵又耗时。在某些情况下，小企业主[花费大约 40%的时间](https://www.entrepreneur.com/article/217866)在非创收任务上，比如招聘。另外，[麻省理工学院估计](http://web.mit.edu/e-club/hadzima/how-much-does-an-employee-cost.html)每雇佣一名美国新员工，公司就要花费大约 1.4 倍的基本工资。对于初创公司来说，这些影响会更加明显，因为他们通常都缺乏时间和资金。

雇佣远程团队让初创公司能够接触到更大的人才库，并有机会以本地化的价格支付工资。那么，在时间和资源有限的情况下，初创公司如何为他们所需的专业技能找到远程办公室的最佳位置呢？是否有某些国家专门从事特定的技术？中国和印度仍然是最佳观察地吗？

**2。数据**

我开始通过使用 [2019 Stack Overflow 开发者调查](https://insights.stackoverflow.com/survey/2019)来回答这些问题，其中包括大约 40，000 名非美国受访者。随着他们的位置，每个回答者报告了他们积极使用的技术栈和框架，如 Java、Python、Angular 和 Django。

在投资离岸办公室之前，初创公司需要确保有足够的人才库满足他们公司的技术和经验要求。为了充分评估本地人才库的质量，我向数据集添加了维度。我为每个回答者的经验、教育和薪水设定了权重，以确定他们作为潜在候选人的素质——从而将他们从他们工作的环境中区分出来。

**3。算法**

我二进制化了包括被告技术堆栈的列。在处理结束时，我的数据看起来像这样:

![](img/0cf07f802733ccb5eceee98482e78b88.png)

Example of binarized data

对二进制数据进行聚类需要专门的算法。我使用了 K-Means 的一个表亲，称为 K-Modes，它允许我以这种当前形式对我的数据进行聚类。快速复习一下——K-Means 是一种分区算法，它根据现有的(和不可见的！)在 *K* 簇中连续数据的相似性。

1.  *K* -随机初始化质心，分配点，
2.  基于度量(例如欧几里德距离)将点分配给最近的质心，
3.  重新计算作为每个聚类的平均值计算的质心，
4.  将点重新分配到最近的质心，
5.  重复步骤 2 至 4，直到这些点不再被指定给不同的聚类。

结果是数据的分组，其中聚类内的对象的分离被最小化。

顾名思义，K-Means 使用平均值来计算质心。然而，K-Means 算法只能用于连续数据，而不能用于分类数据。例如，如果我们对应答者 1 和应答者 2 都取欧几里德距离(如上所示),那么当我们知道这是不正确的时候，K-Means 会将两个应答者分配到同一个聚类中。

那么，对于这种二值化的数据，我们如何计算不同开发者技能之间的距离呢？解决方案在于 K-Modes 算法，该算法使用*相异度*而不是每个数据点到质心的距离。在这种情况下，每个数据点和每个质心的相异度或“距离”可以定义为他们不同意的技术堆栈的数量。当数据点和质心在技术堆栈上一致时，这将使“距离”更低，当它们发散时，将使“距离”更高。

当计算新的质心时，K-Modes 算法也偏离 K-Means。K-Modes 计算的不是类中的平均值，而是类的模式。因此，在我的工作中，我感兴趣的是根据受访者使用的特定技术对他们进行聚类。

K-Modes 是一个相对较新的算法，来自于 1998 年的一篇论文，还不是 scikit-learn 包的一部分。你可以在 [Github](https://github.com/nicodv/kmodes) 上找到 K-Modes 和它的堂兄 K-Prototypes，以便安装和文档。

**4。实施**

现在我有了一个可以处理我的数据的算法，我需要决定聚类的数量。我在 K-Modes 包中使用了 Jaccard 相异度函数，该函数测量我的集群之间的相异度。

众所周知，Jaccard 距离是 1 减去交集的大小，再除以并集的大小，如下所示:

![](img/080efd2828f163c2a5642225059f1f77.png)

Jaccard Distance Formula

与 K-Means 剪影分数一样，我将“肘”方法与 Jaccard 相异分数一起应用，以找到最适合我的模型的聚类数。我在 40 个星团里发现了那个手肘。

![](img/ad3fc03539af779450872d845242786c.png)

**5。应用程序**

现在，我已经用 *k=40* 运行了我的模型，我希望能够理解我的集群的地理性质——直观地显示我的集群中开发人员的地域性。特别是，我想为早期创业公司开发一个工具，使用一线工具来定位离岸办公室。

为此，我构建了一个 Flask 应用程序，它接受每个受访者的堆栈、教育、经验和工资等参数，并返回满足这些约束的集群的交互式地图。

下面视频中的演示采用了 Hadoop、Spark 和 Postgres 的参数，其中受访者至少拥有学士学位和至少 4 年的工作经验，并且收入低于 7.5 万美元。有了这些参数，我的模型告诉我，我应该在爱沙尼亚、波兰和芬兰开始寻找，因为那里有有这种经历的人。

![](img/acc16f6dffc9ab0926182509624cca04.png)

Search for Data Engineer by tech stack

然而，如果我需要用 Java、Kotlin 和 C#的特定技术栈建立一个专门用于 Android 应用程序开发的远程办公室，我的模型建议首先在中美洲和南美洲寻找。

![](img/5bfd407ac57b0fea70bd051368062d38.png)

Search for Android Developer by tech stack

**6。结论**

伟大的天才无处不在。公司在为他们需要的特定人才寻找离岸办公室时，可以更准确地锁定目标地区。这个模型和应用程序是帮助追求离岸战略的公司迈出建立远程团队的第一步的初始工具。

**7。额外资源**

[领英](https://www.linkedin.com/in/kathmbell/)、[引用](https://amva4newphysics.wordpress.com/2016/10/26/into-the-world-of-clustering-algorithms-k-means-k-modes-and-k-prototypes/comment-page-1/#targetText=k%2Dmodes%20is%20an%20extension%20of%20k%2Dmeans.&targetText=And%20instead%20of%20means%2C%20it,since%20they%20act%20as%20centroids.)、[引用](https://shapeofdata.wordpress.com/2014/03/04/k-modes/)