# 具有改进的协同过滤的闪亮产品推荐器

> 原文：<https://towardsdatascience.com/a-r-shiny-application-with-improved-collaborative-filtering-48a82955da6a?source=collection_archive---------10----------------------->

## 我对购物篮分析的看法——第 3 部分，共 3 部分

![](img/db3e297fea3ee769f839e26ab11f9288.png)

Photo by [NeONBRAND](https://unsplash.com/@neonbrand?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 概观

最近我想学习一些新的东西，并挑战自己进行端到端的**市场篮子分析**。为了继续挑战自己，我决定将我的努力成果展示给数据科学界。

这是**的第三个也是最后一个帖子**:

> ***Part 1*** *:(可在此处找到*[](/clean-a-complex-dataset-for-modelling-with-recommendation-algorithms-c977f7ba28b1)**)探索并清理适合用推荐算法建模的数据集* ***Part 2****:(可在此处找到*[](/market-basket-analysis-with-recommenderlab-5e8bdc0de236)**)应用各种产品推荐模型***

# **介绍**

**在我为这个项目进行研究的过程中，我遇到了几件作品，所有这些都为我闪亮的应用程序提供了巨大的灵感和洞察力。特别要提到的是 *Jekaterina Novikova，她的 [**电影推荐系统**](https://jeknov.shinyapps.io/movieRec) 的博士*，以及*迈克尔·哈斯勒教授*，他创造了一个 [**笑话推荐系统**](https://mhahsler-apps.shinyapps.io/Jester/) 。他们分别使用了`MovieLens`和`Jester`数据集，这是**推荐者实验室**的默认设置。**

**然而，两位作者都使用了 **recommenderlab** 作为他们闪亮应用的引擎，众所周知，这需要很长时间来计算大型数据集的预测。Jekaterina 为使计算更易于管理而采用的解决方案是对数据集进行采样，以减小评级矩阵的大小。然而，这可能证明对预测准确性有害。**

**幸运的是，我遇到了由 Philipp Spachtholz 开发的这个出色的 [**Kaggle 内核**](https://www.kaggle.com/philippsp/book-recommender-collaborative-filtering-shiny) ，他不仅对一个非 Kaggle 数据集进行了显著的分析，而且对我来说至关重要的是，他还使用一个快得多的协作过滤代码构建了一个基于 Shiny 的 [**图书推荐系统**](https://philippsp.shinyapps.io/BookRecommendation/) 。**

***Philipp* 从这篇由[**smart cat Consulting**](https://www.smartcat.io/)撰写的[博客文章](https://www.smartcat.io/blog/2017/improved-r-implementation-of-collaborative-filtering/)中获得了灵感，这篇文章描述了如何使用改进的协同过滤代码以及包含在配套 [Github](https://github.com/smartcat-labs/collaboratory) 库中的所有相关功能。特别地，存储库包括用于计算相似性矩阵的`simililarity_measures.R`函数，以及具有协同过滤算法和预测函数的`cf_algorithm.R`文件。**

**在这个项目的最后一部分，我将描述如何在我闪亮的实现中使用这些函数。**

# **加载包**

```
**library(tidyverse)            
library(knitr)
library(Matrix)
library(recommenderlab)**
```

# **数据**

**在这一节中，我使用第二部分[](/market-basket-analysis-with-recommenderlab-5e8bdc0de236)**中的`retail`数据集来准备闪亮应用程序所需的数据文件。**请注意，在第 2 部分的**中，我执行了额外的格式化步骤，删除了多次包含相同商品的订单。****

```
****glimpse(retail)
## Observations: 517,354
## Variables: 10
## $ InvoiceNo   <dbl> 536365, 536365, 536365, 536365, 536365,
## $ StockCode   <chr> "85123A", "71053", "84406B", "84029G",
## $ Description <fct> WHITE HANGING HEART T-LIGHT HOLDER, 
## $ Quantity    <dbl> 6, 6, 8, 6, 6, 2, 6, 6, 6, 32, 6, 6, 8, 6,
## $ InvoiceDate <dttm> 2010-12-01 08:26:00, 2010-12-01 08:26:00,
## $ UnitPrice   <dbl> 2.55, 3.39, 2.75, 3.39, 3.39, 7.65, 4.25,
## $ CustomerID  <dbl> 17850, 17850, 17850, 17850, 17850, 17850,
## $ Country     <fct> United Kingdom, United Kingdom, Un...
## $ Date        <date> 2010-12-01, 2010-12-01, 2010-12-01,
## $ Time        <fct> 08:26:00, 08:26:00, 08:26:00, 08:26:00,****
```

****对于应用程序部署，我需要创建 2 个数据文件:`past_orders_matrix`和`item_list`。****

****`past_orders_matrix`是包含过去订单历史的用户项目稀疏矩阵。所有计算都需要这个闪亮的`server.R`文件。****

```
****past_orders_matrix <- retail %>%
    # Select only needed variables
    select(InvoiceNo, Description) %>%     # Add a column of 1s
    mutate(value = 1) %>% # Spread into user-item format
    spread(Description, value, fill = 0) %>%
    select(-InvoiceNo) %>%    # Convert to matrix
    as.matrix() %>%

    # Convert to class "dgCMatrix"
    as("dgCMatrix")****
```

****我保存文件，以便在应用程序中使用。****

```
****saveRDS(past_orders_matrix, file = "past_orders_matrix.rds")****
```

****`item_list`列出了所有可供购买的产品。这将输入闪亮的`ui.R`文件，使产品列表可供选择。****

```
****# Creating a unique items list
item_list <- retail  %>% 
    select(Description) %>% 
    unique()****
```

****我保存列表，以便在应用程序中使用。****

```
****saveRDS(item_list, file = "item_list.rds")****
```

# ****改进的协同过滤****

****为了展示`Improved Collaborative Filtering`是如何工作的，我将在 [**第 2 部分**](/market-basket-analysis-with-recommenderlab-5e8bdc0de236)`item-based CF`中找到的表现最好的模型装配在同一个定制订单上。我也在做同样的事情，用`recommenderlab`来比较这两种方法的性能。****

****首先，我使用同样的 6 个随机选择的产品重新创建了定制订单。****

```
****customer_order <- c("GREEN REGENCY TEACUP AND SAUCER",
                     "SET OF 3 BUTTERFLY COOKIE CUTTERS",
                     "JAM MAKING SET WITH JARS",
                     "SET OF TEA COFFEE SUGAR TINS PANTRY",
                     "SET OF 4 PANTRY JELLY MOULDS")****
```

****接下来，我将把`new_order`放入一个 user_item 矩阵格式中。****

```
****# put in a matrix format
    new_order <- item_list %>% # Add a 'value' column with 1's for customer order items
    mutate(value = 
               as.numeric(Description %in% customer_order)) %>% # Spread into sparse matrix format
    spread(key = Description, value = value) %>% # Change to a matrix
    as.matrix() %>%    # Convert to class "dgCMatrix"
    as("dgCMatrix")****
```

****然后，我将`new_order`添加到`past_orders_matrix`作为它的第一个条目。****

```
****# binding 2 matrices
   all_orders_dgc <- t(rbind(new_order,past_orders_matrix))****
```

****现在，我需要设置一些改进的 CF 工作所需的参数。****

```
****# Set range of items to calculate predictions for
   items_to_predict <- 1:nrow(all_orders_dgc)  # I select them all# Set current user to 1, which corresponds to new_order
   users <- c(1)# Set prediction indices
   prediction_indices <- as.matrix(expand.grid(items_to_predict, 
                                                   users = users))****
```

****我加载算法实现和相似性计算。****

```
****# Load algorithm implementations and similarity calculations
source("cf_algorithm.R")
source("similarity_measures.R")****
```

****最后，我可以用`Improved CF`来拟合`item-based CF`模型，并检查运行时间。****

```
****start <- Sys.time()
recomm <- predict_cf(all_orders_dgc, 
                       prediction_indices,
                       "ibcf", FALSE, cal_cos, 3, 
                       FALSE, 4000, 2000)
end <- Sys.time()
cat('runtime', end - start) ## runtime 0.630003****
```

****哇！真是快如闪电！****

****现在让我们用`recommenderlab`运行`item-based CF`模型并比较性能。****

```
****# Convert `all_orders` to class "binaryRatingMatrix"
    all_orders_brm <- as(all_orders_dgc, "binaryRatingMatrix")# Run run IBCF model on recommenderlab
start <- Sys.time()
recomm <- Recommender(all_orders_brm, 
                      method = "IBCF",  
                      param = list(k = 5))
end <- Sys.time()
cat('runtime', end - start)## runtime 12.75939****
```

****速度增加了大约 20 倍，这与菲利普·斯帕茨在他的工作中所看到的一致。这对闪亮的应用程序来说是相当有前途的！****

# ****部署应用程序****

****对于应用程序的部署，我采用了 [*概念验证*](https://en.wikipedia.org/wiki/Proof_of_concept) 的方法。我的重点是执行速度，以及如何正确完成本文中显示的所有计算，从而为应用程序的`server`端提供动力。这反映在目前的极简用户界面上，它仅仅以产品选择和*完成购买*操作按钮为特色。****

****这里有一个 [**产品推荐者**](https://diegousai.shinyapps.io/Product_Recommender/) 的链接供你细读。****

****我将继续在用户界面上工作，增加改善客户体验的功能，使它更像一个最终产品。****

****以下是我目前正在考虑的一些想法:****

1.  ****添加**产品价格**和选择**产品数量**的能力****
2.  ****添加**产品图片** —有 4000 种商品可供选择，这本身就是一个迷你项目！****
3.  ****能够使用产品名称的第一个字母**从列表中选择项目******
4.  ****使用`shinyJS`和/或`HTML tags`增强用户界面的视觉效果****
5.  ****考虑到目前并非所有产品组合都返回建议，研究如何实施混合方法****

# ****评论****

****我真的很喜欢这个市场篮分析项目。推荐系统是一个迷人的研究领域，有着真实世界的应用，我觉得我只是触及了这个主题的表面。****

****我也非常感谢学习闪亮应用程序开发的诀窍，这比我最初想的要简单得多:反应性是一个让你头脑清醒的关键概念，它迫使你从用户界面和服务器是一个硬币的两面的角度来思考。****

****对我来说，主要的考虑是**潜力是巨大的**:即使是在线的小公司也能从实施最基本的推荐系统中受益。只需几行代码，就可以改善客户体验，提高客户忠诚度，促进销售。****

# ****代码库****

****完整的 R 代码可以在[我的 GitHub 简介](https://github.com/DiegoUsaiUK/Market_Basket_Analysis)上找到****

# ****参考****

*   ****有关推荐的 lab 包，请参见:[https://cran.r-project.org/package=recommenderlab](https://cran.r-project.org/package=recommenderlab)****
*   ****有关推荐者实验室软件包简介，请参见:[https://cran . r-project . org/web/packages/re commender lab/vignettes/re commender lab . pdf](https://cran.r-project.org/web/packages/recommenderlab/vignettes/recommenderlab.pdf)****
*   ****关于 SmartCat 改进的协作过滤器，请参见:[https://www . SmartCat . io/blog/2017/Improved-r-implementation-of-Collaborative-filtering/](https://www.smartcat.io/blog/2017/improved-r-implementation-of-collaborative-filtering/)****

*****原载于 2019 年 4 月 15 日*[*https://diegousei . io*](https://diegousai.io/2019/04/market-basket-analysis-part-3-of-3/)*。*****