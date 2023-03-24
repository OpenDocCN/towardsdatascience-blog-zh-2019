# 基于机器学习的高效军事部署建模——R

> 原文：<https://towardsdatascience.com/modelling-efficient-military-deployments-with-machine-learning-k-means-clustering-in-r-7dc4ab6a945c?source=collection_archive---------19----------------------->

**(如果使用智能手机，这篇文章最好在横向模式下查看)**

拉丁美洲和加勒比海地区的武装部队面临着必须执行多方面任务的挑战。在内乱加剧的时候，他们需要开展维持和平行动，武器换毒品交易引发的帮派战争要求进行反叛乱式的部署，季节性自然灾害往往需要他们的服务，以支持极端条件下的基本服务。在资源有限的情况下，需要利用一切机会防止不必要的开支，同时保持效力。

在本帖中，我将展示如何在拉丁美洲和加勒比海的海军部队中应用 K-means 聚类算法，以安排高效的海军部署并减少不必要的行动。

# 准备数据

对于这个例子，我模拟了 200 个数据点，这些数据点代表了需要在加勒比海部署海军资源的事件的位置。这些数据有一个时间戳，指示每个事件在 24 小时时钟周期中发生的时间。重点关注的特定区域约为 7500 平方英里，位于以下区域之间:

纬度— N18*00 '和 N17*20'
,经度— W78*20 '和 W76*00 '

(**牙买加**南海岸附近的加勒比海)

我们的任务是确定任何容易发生事故的地点和时间，如果是这样的话，向这些地区分配更多的军事资源，同时减少向很少发生事故的地点的部署。

```
#You will require these packages to execute the code in this example 
#effectively.
library("plotly", lib.loc="~/R/win-library/3.6")
library("dplyr", lib.loc="~/R/win-library/3.6")
library("ggplot2", lib.loc="~/R/win-library/3.6")
```

现在来模拟数据点。

```
#Generate data for example
set.seed(11)
LatCL1 <- round(rnorm(20, mean = 17.9, sd =  0.1), 2)set.seed(12)
LonCL1 <- round(rnorm(20, mean = -78.37, sd = 0.2), 2)
CL1 <- data.frame(LonCL1, LatCL1)
names(CL1) <- c('Longitude', 'Latitude')set.seed(21)
LatCL2 <- round(rnorm(50, mean = 17.73, sd =  0.033), 2)set.seed(22)
LonCL2 <- round(rnorm(50, mean = -78, sd = 0.25), 2)
CL2 <- data.frame(LonCL2, LatCL2)
names(CL2) <- c('Longitude', 'Latitude')set.seed(31)
LatCL3 <- round(rnorm(130, mean = 17.64, sd =  0.075), 2)set.seed(32)
LonCL3 <- round(rnorm(130, mean = -76.64, sd = 0.3), 2)
CL3 <- data.frame(LonCL3, LatCL3)
names(CL3) <- c('Longitude', 'Latitude')CL <- rbind(CL1, CL2, CL3)
```

我们已经生成了 200 个数据点，这些数据点代表了记录事件的位置，例如遇险呼叫、海盗船目击报告(是的，加勒比海真的有海盗)、海上失踪的水手(请求搜索和救援服务)、拦截毒品/武器走私者等…

# 将数据可视化

我们现在可以将它们绘制在图表上，看看我们是否可以直观地识别数据点之间的任何明显趋势。

```
CL <- rbind(CL1, CL2, CL3)
plot(CL, xlim = c(-78.3, -76),  ylim = c(17.3, 18), main = 'Events')
```

![](img/997295119192107b478c4d175098cec4.png)

似乎有两组事件聚集在一起(分别位于-77.4 经度线的两侧)。

接下来，我们将实现 K-means 算法来验证我们的观察是否在数学上得到支持。由于聚类数“K”是模型的一个超参数，因此遵循我们对 K = 2 的初始观察是合理的。但是，我们知道运营部署安排在三个轮班日左右，我们将设置 K = 3 来测试我们是否可以识别三个相应的集群，每个轮班一个集群。之后，我们将使用 ggplot2 可视化集群。

```
#Clustering with K=three 
set.seed(101)
TwoD_Clust <- kmeans(CL, 3, iter.max = 10)#Assign the cluster to the varibale
CL$Group <- as.factor(TwoD_Clust$cluster)ggplot(CL, aes(x=CL$Longitude, y=CL$Latitude, color=CL$Group)) +
  labs(x ='Longitude', y ='Latitude', title = 'Tri-Clustered Events') + 
  geom_point()
```

![](img/6da078df804be9cb35e80b127c4ed713.png)

在 K = 3 的情况下，该算法仍然能够识别高度可区分的组，但是这两个组(1 & 3)看起来是强制的。让我们回到 K = 2，并把结果可视化。

```
#Clustering with K=two
set.seed(100)
TwoD_Clust <- kmeans(CL, 2, iter.max = 10)#Assign the cluster to the varibaleCL$Group <- as.factor(TwoD_Clust$cluster)ggplot(CL, aes(x=CL$Latitude, y=CL$Longitude, color=CL$Group)) +
  labs(x ='Longitude', y ='Latitude', title = 'Bi-Clustered Events') + 
  geom_point()
```

![](img/4b4037aa4288b2db6973dcce5d4b1846.png)

# 探索数据并获得创造力

这些组看起来仍然更合适，幸运的是我们还没有将第三个变量(事件发生的时间)添加到我们的数据中。可能添加第三个变量将有助于我们能够更好地区分各组。接下来，我们将把所有三个变量经度、纬度和时间编译成一个数据帧，并使用 Plotly 软件包进行 3D 可视化，以制作一个交互式 3D 绘图。

```
#Adding the third variable 'Time'
set.seed(41)
TCL1 <- data.frame(round(rnorm(20, mean = 10, sd = 1), 2))
names(TCL1) <- c('ETime')set.seed(42)
TCL2 <- data.frame(round(rnorm(50, mean = 23.5, sd = 1), 2))
names(TCL2) <- c('ETime')set.seed(43)
TCL3 <- data.frame(round(rnorm(130, mean = 3, sd = 1), 2))
names(TCL3) <- c('ETime')Event_Times <- rbind(TCL1, TCL2, TCL3)#Compiling all the data into a dataframe
Event_Data <- data.frame(rbind(CL1, CL2, CL3), Event_Times)#Using Plotly to visualise three dimensions
plot_ly(x =Event_Data$Longitude, y = Event_Data$Latitude, z= Event_Data$ETime,
        type = 'scatter3d', mode = 'markers')
```

Rotate the Graph to view from different perspectives. Interactive 3D Scatter-plot

您可以与可视化交互并旋转可视化，以从不同的角度查看集群的外观。使用这个交互式 3D 散点图，我们现在可以清楚地识别三组事件。虽然 2D 图掩盖了数据的可变性，并建议部署应集中在两个特定区域，但现在很明显，在规划常规部署时，“部署时间”应是一个重要的考虑因素，理想情况下，可用的人力资源应分成三个小组，每个小组在特定时间部署。

为了进一步验证我们的观察，我们将再次进行另一个 K 均值分析，K= 3。利用 K-means 函数的输出，我们还能够对数据中可能出现的任何趋势进行更详细的分析。

```
#Clustering the data with three variables and K=3
ThreeD_Clust <- kmeans(Event_Data, 3, iter.max = 10)Event_Data$Group <- as.factor(ThreeD_Clust$cluster)D32<- plot_ly(x = Event_Data$Longitude, y = Event_Data$Latitude, z= Event_Data$ETime,
        type = 'scatter3d', mode = 'markers', color = Event_Data$Group) %>% 
  layout(title = '3D Clusters', scene = list(
    xaxis = list(title ='Longitude'),
    yaxis = list(title ='Latitude'),
    zaxis = list(title ='Time Axis')
    ))
```

Rotate the Graph to view from different perspectives. Interactive Output of K-Means Cluster with K=3

您可以与可视化交互并旋转可视化，以从不同的角度查看集群的外观。该算法很好地确定了三个不同的集群，可用于确定有效作战部署的最佳时间和位置。

# 分析和解释

这项工作还没有完成。接下来，我们必须对算法的输出进行分析，以将其结果正式化，并将其转换为可用于自信地部署军事资源的可操作信息。

```
#Viewing the output of the kmeans functionThreeD_ClustK-means clustering with 3 clusters of sizes 130, 20, 50Cluster means: Longitude Latitude     ETime Group
1 -76.64654 17.64231  3.068308     1
2 -78.43600 17.86550 10.427000     3
3 -77.96940 17.73420 23.464600     2Clustering vector:
  [1] 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 2 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3
[42] 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 3 1 1 1 1 1 1 1 1 1 1 1 1
[83] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
[124] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1
[165] 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1 1Within cluster sum of squares by cluster:
  [1] 134.22748  19.67019  68.32914(between_SS / total_SS =  98.6 %)Available components:

  [1] "cluster"      "centers"      "totss"        "withinss"     "tot.withinss"
[6] "betweenss"    "size"         "iter"         "ifault"
```

从输出/结果中，三条信息对我们的用例特别重要:1)聚类平均值或“质心”。2)每个聚类的大小 3)平方和。

1.  我们已经确定了三个集群，它们确定了资源应该部署到的适当时间和地点，以便期望最高的产出回报。以下是要点:

W-76*.64，N17*.64 凌晨 3 点左右
W-78*.43，N17*.86 上午 10:45 左右
W-77*.96，N17*.73 晚上 11:45 左右。

我们现在可以在这些地点和时间规划部署，以获得最大的效率。

2.集群 1、2 和 3 的大小分别为 130、20 和 50(比率为 65:10:25)。这些数字表示每个集群中可能发生的事件的相对数量。有了这些信息，我们就能够将一个 100 人的军事人员小组适当地分配如下:第 1 组 65 人，第 2 组 20 人，第 3 组 25 人。这减少了因疏忽造成的人员不足和过度部署。

3.类内平方和表示类的可变性。基本上，模型与数据的吻合程度，以及特定值与平均值相差很大的可能性。大面积的平方表明，尽管部署将集中在特定的地点和时间，但他们仍然必须在这些中心区域进行长时间的巡逻。我们的组间 SS /总 SS = 98.6%，表明该模型非常适合！

这种技术在现实生活中应用的一个例子可以在[这里](https://www.researchgate.net/publication/331834584_Clustering-based_task_coordination_to_search_and_rescue_teamwork_of_multiple_agents)找到

# 边注

通常建议在执行机器学习算法之前对数据进行归一化/缩放，因为变量范围的差异会影响模型的性能。我可以证实，在这种情况下，情况并非如此，也没有证据表明发生了这种情况。事实上，缩放后的模型往往表现更差！(你可以自己重新运行代码，评估缩放后的结果)——这是对机器学习科学中存在的艺术的微妙提醒。此外，将质心的初始随机迭代次数设置为 10 似乎足以产生一个稳定的模型，该模型在多次重新运行算法后具有一致的结果。时间戳被转换为表示一天 24 小时中的小时，纬度/经度值从度*分钟格式转换为度*和小数。