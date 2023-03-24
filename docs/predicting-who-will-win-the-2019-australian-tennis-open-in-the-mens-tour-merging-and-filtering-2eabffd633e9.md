# R 简介—合并和过滤数据—第 1 部分

> 原文：<https://towardsdatascience.com/predicting-who-will-win-the-2019-australian-tennis-open-in-the-mens-tour-merging-and-filtering-2eabffd633e9?source=collection_archive---------20----------------------->

通过对 2019 年澳网男子巡回赛数据的筛选和合并进行数据理解。

![](img/0df61dd4b9e022e1c9bb7f12e68f95d2.png)

Photo by [Christopher Burns](https://unsplash.com/photos/YSfTcJZR-ws?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/tennis?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

你知道当澳大利亚网球公开赛造访墨尔本的时候是夏天，每个人都为罗杰和塞丽娜的到来而兴奋。

# 问题

我有兴趣预测谁可能赢得 2019 年澳大利亚网球公开赛男子巡回赛。我希望他退役前是罗杰·费德勒。

# 资料组

我从网站[http://www.tennis-data.co.uk/data.php](http://www.tennis-data.co.uk/data.php)选择了从 2000 年到 2019 年 1 月的男子网球锦标赛历史成绩，这些成绩以 CSV 文件的形式提供。

# 数据字典

网球数据附带的注释解释了由[网球博彩](http://www.tennis-data.co.uk/notes.txt)提供的数据的属性:

```
Key to results data:

ATP = Tournament number (men)
WTA = Tournament number (women)
Location = Venue of tournament
Tournament = Name of tournament (including sponsor if relevant)
Data = Date of match (note: prior to 2003 the date shown for all matches played in a single tournament is the start date)
Series = Name of ATP tennis series (Grand Slam, Masters, International or International Gold)
Tier = Tier (tournament ranking) of WTA tennis series.
Court = Type of court (outdoors or indoors)
Surface = Type of surface (clay, hard, carpet or grass)
Round = Round of match
Best of = Maximum number of sets playable in match
Winner = Match winner
Loser = Match loser
WRank = ATP Entry ranking of the match winner as of the start of the tournament
LRank = ATP Entry ranking of the match loser as of the start of the tournament
WPts = ATP Entry points of the match winner as of the start of the tournament
LPts = ATP Entry points of the match loser as of the start of the tournament
W1 = Number of games won in 1st set by match winner
L1 = Number of games won in 1st set by match loser
W2 = Number of games won in 2nd set by match winner
L2 = Number of games won in 2nd set by match loser
W3 = Number of games won in 3rd set by match winner
L3 = Number of games won in 3rd set by match loser
W4 = Number of games won in 4th set by match winner
L4 = Number of games won in 4th set by match loser
W5 = Number of games won in 5th set by match winner
L5 = Number of games won in 5th set by match loser
Wsets = Number of sets won by match winner
Lsets = Number of sets won by match loser
Comment = Comment on the match (Completed, won through retirement of loser, or via Walkover)
```

# 在 Mac Book Pro 中将 csv 文件合并成一个数据文件

我在 Youtube 上看到了一个来自[的 Trent Jessee](https://www.youtube.com/watch?v=5c_VKhYSTjA) 的教程，它帮助我在 Macbook Pro 上将 2000 年至 2019 年的多个 csv 文件合并成一个文件。

#打开终端会话，进入
cd 桌面

#输入 cd 和保存 csv 文件的文件夹名称
cd CSV

#用此命令合并文件
cat *。csv >合并. csv

# 将您的数据加载或导入 R

```
*# Set the working directory*
setwd("~/Desktop/ATP")

*# Read the dataframe into Rstudio as a csv file.* 
tennis_data <- read.csv("merged.csv",stringsAsFactors = FALSE, header = TRUE)

*# Review the first 5 observations*
head(tennis_data)##   ATP Location                         Tournament   Date        Series
## 1   1 Adelaide Australian Hardcourt Championships 1/3/00 International
## 2   1 Adelaide Australian Hardcourt Championships 1/3/00 International
## 3   1 Adelaide Australian Hardcourt Championships 1/3/00 International
## 4   1 Adelaide Australian Hardcourt Championships 1/3/00 International
## 5   1 Adelaide Australian Hardcourt Championships 1/3/00 International
## 6   1 Adelaide Australian Hardcourt Championships 1/3/00 International
##     Court Surface     Round Best.of       Winner          Loser WRank
## 1 Outdoor    Hard 1st Round       3   Dosedel S.    Ljubicic I.    63
## 2 Outdoor    Hard 1st Round       3   Enqvist T.     Clement A.     5
## 3 Outdoor    Hard 1st Round       3    Escude N.  Baccanello P.    40
## 4 Outdoor    Hard 1st Round       3   Federer R. Knippschild J.    65
## 5 Outdoor    Hard 1st Round       3  Fromberg R.  Woodbridge T.    81
## 6 Outdoor    Hard 1st Round       3 Gambill J.M.     Arthurs W.    58
##   LRank W1 L1 W2 L2 W3 L3 W4 L4 W5 L5 Wsets Lsets   Comment X X.1 X.2 X.3
## 1    77  6  4  6  2 NA NA NA NA NA NA     2     0 Completed              
## 2    56  6  3  6  3 NA NA NA NA NA NA     2     0 Completed              
## 3   655  6  7  7  5  6  3 NA NA NA NA     2     1 Completed              
## 4    87  6  1  6  4 NA NA NA NA NA NA     2     0 Completed              
## 5   198  7  6  5  7  6  4 NA NA NA NA     2     1 Completed              
## 6   105  3  6  7  6  6  4 NA NA NA NA     2     1 Completed
```

##根据 2000 年至 2019 年的数据预测谁将赢得 2019 年澳大利亚网球公开赛

#加载包

图书馆(dplyr)

#设置工作目录

setwd(" ~/桌面/ATP ")

#将数据帧作为 csv 文件读入 Rstudio。

tennis _ data

#回顾前 5 条观察结果。最佳实践是使用 tail()来查看最后 5 次观察。

主管(网球 _ 数据)

# 查看合并数据的结构

在 R 中，我们可以探索数据的结构，检查数据的属性，在预处理之前观察原始数据。

str(网球 _ 数据)

在合并的数据文件中有 52383 行和 83 列。

```
## 'data.frame':    52383 obs. of  83 variables:
##  $ ATP       : int  1 1 1 1 1 1 1 1 1 1 ...
##  $ Location  : chr  "Adelaide" "Adelaide" "Adelaide" "Adelaide" ...
##  $ Tournament: chr  "Australian Hardcourt Championships" "Australian Hardcourt Championships" "Australian Hardcourt Championships" "Australian Hardcourt Championships" ...
##  $ Date      : chr  "1/3/00" "1/3/00" "1/3/00" "1/3/00" ...
##  $ Series    : chr  "International" "International" "International" "International" ...
##  $ Court     : chr  "Outdoor" "Outdoor" "Outdoor" "Outdoor" ...
##  $ Surface   : chr  "Hard" "Hard" "Hard" "Hard" ...
##  $ Round     : chr  "1st Round" "1st Round" "1st Round" "1st Round" ...
##  $ Best.of   : int  3 3 3 3 3 3 3 3 3 3 ...
##  $ Winner    : chr  "Dosedel S." "Enqvist T." "Escude N." "Federer R." ...
##  $ Loser     : chr  "Ljubicic I." "Clement A." "Baccanello P." "Knippschild J." ...
##  $ WRank     : chr  "63" "5" "40" "65" ...
##  $ LRank     : chr  "77" "56" "655" "87" ...
##  $ W1        : chr  "6" "6" "6" "6" ...
##  $ L1        : chr  "4" "3" "7" "1" ...
##  $ W2        : int  6 6 7 6 5 7 6 7 2 6 ...
##  $ L2        : int  2 3 5 4 7 6 1 6 6 7 ...
##  $ W3        : int  NA NA 6 NA 6 6 NA NA 6 6 ...
##  $ L3        : int  NA NA 3 NA 4 4 NA NA 1 4 ...
##  $ W4        : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ L4        : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ W5        : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ L5        : int  NA NA NA NA NA NA NA NA NA NA ...
##  $ Wsets     : int  2 2 2 2 2 2 2 2 2 2 ...
##  $ Lsets     : int  0 0 1 0 1 1 0 0 1 1 ...
##  $ Comment   : chr  "Completed" "Completed" "Completed" "Completed" ...
```

# 数据框的尺寸

数据帧的尺寸包括-52383 行和 83 列。

dim(网球 _ 数据)

# 检查是否有丢失的值

有相当多的缺失值。评估后可能需要对缺失值进行插补。

is.na(网球 _ 数据)

# 过滤并子集化 R 中的数据

我需要过滤我的数据，以便更易于管理，我对分析 2000 年至 2019 年的澳大利亚网球公开赛很感兴趣。

#我们只想查看结果数据的前 26 列

姓名(网球 _ 数据)[1:26]

#我们想要过滤澳大利亚网球公开赛的数据，以便我们可以使用数据的子集:

aust _ Open

#查看仅与澳大利亚网球公开赛相关的数据子集的结构:

str(aust_open)

```
## 'data.frame':    2413 obs. of  26 variables:
##  $ ATP       : int  6 6 6 6 6 6 6 6 6 6 ...
##  $ Location  : chr  "Melbourne" "Melbourne" "Melbourne" "Melbourne" ...
##  $ Tournament: chr  "Australian Open" "Australian Open" "Australian Open" "Australian Open" ...
##  $ Date      : chr  "1/17/00" "1/17/00" "1/17/00" "1/17/00" ...
##  $ Series    : chr  "Grand Slam" "Grand Slam" "Grand Slam" "Grand Slam" ...
##  $ Court     : chr  "Outdoor" "Outdoor" "Outdoor" "Outdoor" ...
##  $ Surface   : chr  "Hard" "Hard" "Hard" "Hard" ...
##  $ Round     : chr  "1st Round" "1st Round" "1st Round" "1st Round" ...
##  $ Best.of   : int  5 5 5 5 5 5 5 5 5 5 ...
##  $ Winner    : chr  "Agassi A." "Alami K." "Arazi H." "Behrend T." ...
##  $ Loser     : chr  "Puerta M." "Manta L." "Alonso J." "Meligeni F." ...
##  $ WRank     : chr  "1" "35" "41" "106" ...
##  $ LRank     : chr  "112" "107" "111" "28" ...
##  $ W1        : chr  "6" "6" "6" "6" ...
##  $ L1        : chr  "2" "4" "3" "2" ...
##  $ W2        : int  6 7 7 4 6 6 6 6 6 5 ...
##  $ L2        : int  2 6 6 6 4 1 1 4 4 7 ...
##  $ W3        : int  6 7 6 6 6 6 6 NA 6 6 ...
##  $ L3        : int  3 5 2 7 4 4 4 NA 4 3 ...
##  $ W4        : int  NA NA NA 6 0 NA 7 NA NA 7 ...
##  $ L4        : int  NA NA NA 3 6 NA 6 NA NA 5 ...
##  $ W5        : int  NA NA NA 6 6 NA NA NA NA NA ...
##  $ L5        : int  NA NA NA 0 4 NA NA NA NA NA ...
##  $ Wsets     : int  3 3 3 3 3 3 3 2 3 3 ...
##  $ Lsets     : int  0 0 0 2 2 0 1 0 0 1 ...
##  $ Comment   : chr  "Completed" "Completed" "Completed" "Completed" ...
```

# 编写并导出您的数据框架

现在，我对仅包含澳大利亚公开赛的数据感到满意，我将编写文件并将其导出为 csv 文件，这样我就可以用它来预处理我的数据以及在 R 和 Tableau 中可视化数据。

```
*# Save the dataframe to a csv file to write the csv file into R working folder:*write.csv(aust_open,file = "aust_open.csv", row.names = FALSE)
```

我的下一篇文章将包括:

*   **预处理 R 中的数据**
*   **R 中的数据可视化，查看数据的样子**
*   **R 中数据的探索性数据分析**
*   **缺失数据的处理**

感谢阅读，敬请关注，编码快乐！