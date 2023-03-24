# 与游戏网站相比，游戏玩家如何评价视频游戏。

> 原文：<https://towardsdatascience.com/how-gamers-rate-video-games-compared-to-gaming-websites-a07d5c9f308a?source=collection_archive---------26----------------------->

![](img/b3e456e0d3d4b099c13b903acb870976.png)

## 洞察玩家对游戏和网站的不同看法

我记得当我拿到我的 Wii 时那种兴奋的感觉。我打开盒子，把游戏机连接到电视上。我打开了控制台。然后，我把 Wii Sports 的游戏插入 CD 槽。一个显示 Wii Sports 的新图标被创建在屏幕*的多个框中的一个框中。*

我点击它，它把我带到一个屏幕，让我从多种体育游戏中进行选择，如拳击、网球、高尔夫和棒球。然后，我开始玩这些游戏。玩这些游戏不仅仅是按下按钮在电视屏幕上创造运动，还可以完成不同的任务。

当我玩这些游戏的时候，我感到娱乐和受教育的同时。当我学习棒球中的一垒、二垒、三垒和全垒打时，我把它作为一种放松的方式。因此，我开始将游戏视为一种生活方式。

游戏绝对是一项引人入胜的活动，可以让你在网上和网下都玩得开心。游戏产业也生产很多游戏。当他们制作这些游戏时，他们试图从他们的观众那里得到反馈。

此外，网站评论家和游戏玩家都对这些游戏进行评级。在看了 YouTuber 对视频游戏的评论，并看到它们在网站上的评级后，我对双方的差异感到惊讶。

因此，我很想知道**游戏玩家对视频游戏和网站**的评价有什么不同。

# 数据集

我寻找代表我所寻找的最佳数据集，我在 Kaggle 上找到了一个 [**这里**](https://www.kaggle.com/floval/12-000-video-game-reviews-from-vandal) 。我下载上传到 Jupyter 笔记本进行编码。

我清理了数据集，删除了不必要的列，留下了重要的列，如平台、网站评级和用户评级。用于执行该清洁过程的代码是:

```
df_game = df_data_1[['platform', 'website_rating', 'user_rating']]
```

包含 android、iPhone 和 PC 游戏的行被删除，因为该研究主要与主机游戏相关。删除这些行的代码是:

```
df_game = df_game[df_game['platform'] != 'Android']
df_game = df_game[df_game['platform'] != 'iPhone'] 
df_game = df_game[df_game['platform'] != 'PC']
```

清理数据集后，创建了一个新的数据集。下面是新形成的数据集的快照:

![](img/6f38597603e0e741fc08c1d48fb8b59d.png)

A dataset of the gaming platforms with their respective website ratings and gamers’ ratings

# 最高等级的游戏

数据集被重新排列了两次。首先，找出哪个游戏在网站评分中是最高的。第二次是找出游戏玩家中评分最高的游戏。

![](img/ffa78bc8fed747b79410a63962dc0cf3.png)

Top 10 highest rated games based on website rating

![](img/00e10f28113266163cdf4e808f73aea1.png)

Top 10 highest rated games based on gamers

根据网站评论，有史以来评分最高的游戏是 Wii 的 Xenoblade Chronicles、PS2 的《侠盗猎车手:圣安地列斯》、DreamCast 的《大都会街头赛车》、DreamCast 的《神木》和任天堂 64 的****的**塞尔达Majora Mask。有史以来基于游戏玩家评分最高的游戏是 PS3 的游戏《XCOM:敌人在和愤怒的小鸟三部曲》。**

# **游戏玩家评分与网站评分**

**首先构建了散点图，以查看游戏玩家的评级和网站评级之间的密切关系。**

**![](img/b1ceabb592a6047231f421ab9d191d88.png)**

**A scatter plot of gamers’ ratings vs website ratings**

**![](img/37eb27d9e1d85d4315caf0cf67e49f99.png)**

**Correlation coefficient of website rating and gamers’ rating**

**根据散点图，它们具有中度正相关关系，相关系数为 0.477。这表明两党对大多数游戏的评级并不完全一致。**

**然后，绘制了一个图表来显示每个平台的平均玩家评分和网站评分之间的差异。这是图表:**

**![](img/9f2d0518e8d7ed35c39854acaba9bbcf.png)**

**Website ratings vs Gamers’ ratings by platform**

**根据图表，游戏玩家的评分用红条表示，而网站的评分用蓝条表示。游戏玩家和网站平均评分最高的游戏主机是任天堂 64，网站和游戏玩家的评分分别为 9.2 和 8.13。游戏玩家对游戏的平均评分最低的游戏主机是任天堂 DS，评分为 7.19。网站有史以来平均评分最低的游戏控制台是 N-Gage，评分为 6.33。**

# ****结论****

**根据最高评级游戏的结果、条形图和散点图，可以准确地说，网站评论家和游戏玩家对视频游戏应该获得的评级意见不一。**

**用于进行这项研究的完整版本代码可以在[**这里**](https://github.com/MUbarak123-56/DataBEL/blob/master/Gaming%20Visualization.ipynb) **看到。****