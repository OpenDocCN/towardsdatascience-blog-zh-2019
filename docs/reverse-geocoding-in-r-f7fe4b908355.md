# 在 R 中反向地理编码

> 原文：<https://towardsdatascience.com/reverse-geocoding-in-r-f7fe4b908355?source=collection_archive---------16----------------------->

## 没有谷歌或必应 API 的免费

![](img/f07d104d13c0e7d96cebe02222ace733.png)

Photo by [Lonely Planet](https://unsplash.com/@lonely_planet?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

随着我继续写论文，我在执行 R 和 Python 中各种包的简单脚本时遇到了一些小故障。这个周末给自己的任务是反地理编码~ 100 万经纬度坐标。我在 R 中发现了一个名为 revgeo 的很棒的包，并认为这很容易实现。我只需要指定提供者和 API 键。谷歌和必应限制每天免费查询的数量，所以这不是一个可行的选择，但光子没有！唯一需要注意的是，像地址名称这样的详细位置并不总是可用的。以下是如何使用 revgeo 软件包的示例:

```
library(revgeo)revgeo(longitude=-77.0229529, latitude=38.89283435, provider = 'photon', output=’frame’)
```

那么问题出在哪里呢？嗯，正如光子网页上所说:

> 您可以在您的项目中使用 API，但是请注意公平——大量使用将会受到限制。我们不保证可用性和使用可能会在未来发生变化。

我不确定在 Photon API 变慢之前需要多少次查询，但重要的是要注意我们向他们的服务器发送了多少请求。我决定从 500，000 坐标开始逆向地理编码，但这并不奏效。我运行了代码，离开了一段时间，当我回来时，我看到节流已经开始，所以我需要调整代码。此外，R 抛出了一个错误`cannot allocate vector of size x.x Gb`，这意味着我的可用内存已经耗尽。

在这一点上，我有两个问题:1)节流和 2)内存分配。对于第 1 期，我需要在代码中加入睡眠时间，并处理已经子集化的数据帧的较小子集。对于第 2 期，我在 stackoverflow 上找到了一个有用的建议:

[](https://stackoverflow.com/questions/10917532/memory-allocation-error-cannot-allocate-vector-of-size-75-1-mb) [## 内存分配“错误:无法分配大小为 75.1 Mb 的向量”

### 在对一些模拟代码进行矢量化的过程中，我遇到了内存问题。我使用的是 32 位 R 版本 2.15.0(通过…

stackoverflow.com](https://stackoverflow.com/questions/10917532/memory-allocation-error-cannot-allocate-vector-of-size-75-1-mb) 

帮助我的一个解决方案是运行`memory.limit(size = _ _ _ _ _ _)`。此外，我使用`rm()`命令删除代码中不再需要的数据帧，并使用`gc()`命令进行垃圾收集。如下所示，我加载了名为`main`的大约一百万坐标的数据帧。我将数据细分为只有 100，000 行。正如您稍后将看到的，我在 while 循环中进一步对数据进行了子集化，以避免内存分配问题。

```
library(revgeo)# the dataframe called 'main' is where the 1 million coordinate points reside.main <- readRDS("main.rds"))main_sub <- main[0:100000,] # Working with a smaller initial subsetrm(main)
gc()
```

下面是完整的代码。该脚本包含了与本文主题无关的其他操作，但我想在这里发布它，以便您可以看到全貌，并希望在反向地理编码中获得一些有用的提示。

```
# Step 1: Create a blank dataframe to store results.
data_all = data.frame()start <- Sys.time()# Step 2: Create a while loop to have the function running until the # dataframe with 100,000 rows is empty.
while (nrow(main_sub)>0) {# Step 3: Subset the data even further so that you are sending only # a small portion of requests to the Photon server. main_sub_t <-  main_sub[1:200,]# Step 4: Extracting the lat/longs from the subsetted data from
# the previous step (Step 3). latlong <- main_sub_t %>% 
    select(latitude, longitude) %>% 
    unique() %>% 
    mutate(index=row_number()) # Step 5: Incorporate the revgeo package here. I left_joined the 
# output with the latlong dataframe from the previous step to add 
# the latitude/longitude information with the reverse geocoded data.cities <- revgeo(latlong$longitude, latlong$latitude, provider =  'photon', output = 'frame')) %>% 
    mutate(index = row_number(),country = as.character(country)) %>%
    filter(country == 'United States of America') %>% 
    mutate(location = paste(city, state, sep = ", ")) %>% 
    select(index, location) %>% 
    left_join(latlong, by="index") %>% 
    select(-index) # Removing the latlong dataframe because I no longer need it. This 
# helps with reducing memory in my global environment.
rm(latlong) # Step 6: Adding the information from the cities dataframe to 
# main_sub_t dataframe (from Step 3).

  data_new <- main_sub_t %>% 
    left_join(cities, by=c("latitude","longitude")) %>% 
    select(X, text, location, latitude, longitude) # Step 7: Adding data_new into the empty data_all dataframe where 
# all subsetted reverse geocoded data will be combined.

  data_all <- rbind(data_all,data_new) %>% 
    na.omit() 
# Step 8: Remove the rows that were used in the first loop from the # main_sub frame so the next 200 rows can be read into the while # loop.

  main_sub <- anti_join(main_sub, main_sub_t, by=c("X"))
  print(nrow(main_sub))

# Remove dataframes that are not needed before the while loop closes # to free up space.
  rm(data_sub_t)
  rm(data_new)
  rm(latlong_1)
  rm(cities)

  print('Sleeping for 10 seconds')
  Sys.sleep(10)

}
end <- Sys.time()
```

实现这个代码后，大概花了 4 个小时反推地理编码 10 万坐标。在我看来，如果我有一百万个坐标要转换，这不是一个可行的选择。我可能不得不寻找另一种方法来实现我的目标，但我认为这对那些数据集较小的人会有所帮助。

感谢阅读和快乐编码！