# 在创建一个好的回归模型时建立你的直觉

> 原文：<https://towardsdatascience.com/build-your-intuition-in-creating-a-good-regression-model-90fd5aa97058?source=collection_archive---------29----------------------->

## 你可以通过分类来辨别那是猫还是不是猫。但是，要回答那只猫有多“猫”，回归是唯一的方法。

![](img/12303b96d827be0c3f14755f54aff4ca.png)

Photo by [Erik-Jan Leusink](https://unsplash.com/@ejleusink?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 分类与回归

两个大问题，两种不同的原因。

热狗还是不是热狗，猫还是不是猫，诈骗还是不是诈骗。这些问题是分类的常见用例。

通过查看示例，您可以很容易地得出结论:分类问题就是将一些数据集划分为几个类或组。从某种意义上说，组的数量是有限的。

正如您所猜测的，另一方面，回归是将这些数据分成无限组。

例如，拥有一组关于房屋面积的数据。通过查看数据，你可以预测每套房子的价格。因为价格几乎是无限的，所以你可以根据这些数据建立一个回归模型。

那么，如何开始创建回归模型呢？

# 定义问题

用例子做任何事情总是更好，因为理论不会让你走那么远。

我们来发现一些问题。

我发现了这个:

[**纽约市出租车费用预测**](https://www.kaggle.com/c/new-york-city-taxi-fare-prediction/data)

这是一个很好的开始问题，可以训练你建立良好回归模型的直觉。

基本上，你会得到一堆关于在纽约从一个地方到另一个地方需要付多少钱的数据。根据这些数据，你需要预测你需要支付多少打车费用。

他们在问题中陈述的指标是，在不做所有必要的机器学习事情的情况下，你可以得到 5 到 8 美元的 RMSE(均方根误差)。所以，让我们建立一个能给你更少误差的东西。

开始吧！

# 数据

下载数据并打开 train.csv(是的，它很大)，您将看到大约 5500 万行包含实际出租车费用的数据。

每行将由 7 列组成。

1.  **接送日期**，当出租车开始行驶时
2.  **皮卡 _ 经度**，出租车乘坐皮卡的经度
3.  **皮卡 _ 纬度**，乘坐皮卡的纬度
4.  **落客 _ 经度**，落客的经度
5.  **落客 _ 纬度**，落客的纬度
6.  **乘客计数**，船上乘客的数量
7.  **fare_amount** ，我们要预测的那个。

我们走吧！

# 逻辑

对于本教程，我将使用 Python 和几个标准库，如 Numpy、Pandas 和 LGBM

```
import numpy as np 
import pandas as pd
import scipy as scipy
import datetime as dt
from sklearn.model_selection import train_test_split
import lightgbm as lgbm
import os
import gc
```

我导入了 GC。那是一个用 python 收集垃圾的库。垃圾收集器的 GC。

*为什么要导入？*

看看数据，对于本教程，我的计算机无法加载文件中的所有行，我最多只能加载 2200 万行。甚至，手动使用垃圾收集器也有所帮助，这样我就可以释放内存来加载和处理所有这 2200 万行。

```
# Reading Data
train_df =  pd.read_csv('train.csv', nrows = 22000000)# Drop rows with null values
train_df = train_df.dropna(how = 'any', axis = 'rows')# Drop invalid rows
train_df = train_df[(train_df.fare_amount > 0)  & (train_df.fare_amount <= 500) & ((train_df.pickup_longitude != 0) & (train_df.pickup_latitude != 0) & (train_df.dropoff_longitude != 0) & (train_df.dropoff_latitude != 0))]
```

读取并删除所有缺少值或值无效的行。

# 直觉

现在你已经准备好了数据。下一步是什么？

像这样的问题，你需要几个新的特性(数据中只有 6 个可用)。你可以试着用这些数据来训练模型，但是你给的信息对模型来说太少了。

特征工程是解决方案。

它基本上是使用现有的功能创建或修改数据。这样做将增加您对数据的理解，最终在训练阶段帮助您的模型。

比如经纬度，直接看的话可能没有任何意义。但是，什么对你有意义呢？**距离**！正确。

让我们准备一个函数来计算拾取点和衰减点之间的距离。

```
# To Compute Haversine distance
def sphere_dist(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return distance along great radius between pickup 
    and dropoff coordinates.
    """
    #Define earth radius (km)
    R_earth = 6371 #Convert degrees to radians
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon =
            map(np.radians, [pickup_lat, pickup_lon,dropoff_lat,        
                dropoff_lon]) #Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = dropoff_lon - pickup_lon

    #Compute haversine distance
    a = np.sin(dlat/2.0)**2 + np.cos(pickup_lat) * 
        np.cos(dropoff_lat) * np.sin(dlon/2.0)**2
    return 2 * R_earth * np.arcsin(np.sqrt(a))
```

这就是哈弗辛公式，一个计算两个纬度和经度之间距离的函数。

然而，在导航中，除了距离之外，你经常需要计算方位，也称为方位角或在北方和游乐设备之间移动的角度。

```
def sphere_dist_bear(pickup_lat, pickup_lon, dropoff_lat, dropoff_lon):
    """
    Return distance along great radius between pickup and dropoff 
    coordinates.
    """

    #Convert degrees to radians
    pickup_lat, pickup_lon, dropoff_lat, dropoff_lon = 
    map(np.radians, [pickup_lat, pickup_lon, dropoff_lat, 
        dropoff_lon]) #Compute distances along lat, lon dimensions
    dlat = dropoff_lat - pickup_lat
    dlon = pickup_lon - dropoff_lon

    #Compute the bearing
    a = np.arctan2(np.sin(dlon * 
        np.cos(dropoff_lat)),np.cos(pickup_lat) * 
        np.sin(dropoff_lat) - np.sin(pickup_lat) * 
        np.cos(dropoff_lat) * np.cos(dlon))
    return a
```

你可以在这里阅读哈弗森距离[，在这里](https://en.wikipedia.org/wiki/Haversine_formula)阅读方位[。](https://en.wikipedia.org/wiki/Bearing_(navigation))

让我们添加数据

```
train_df['distance'] = sphere_dist(train_df['pickup_latitude'], 
                                   train_df['pickup_longitude'], 
                                   train_df['dropoff_latitude'], 
                                   train_df['dropoff_longitude']) 

train_df['bearing'] = sphere_dist_bear(train_df['pickup_latitude'], 
                                       train_df['pickup_longitude'], 
                                       train_df['dropoff_latitude'], 
                                      train_df['dropoff_longitude'])
```

接下来，您可能想要将纬度和经度转换为弧度。因为用弧度计算会给你已经标准化的输入。弧度参数的最大值是 2π弧度。

```
def radian_conv(degree):
    """
    Return radian.
    """
    return  np.radians(degree)train_df['pickup_latitude'] =  
                       radian_conv(train_df['pickup_latitude'])
train_df['pickup_longitude'] = 
                       radian_conv(train_df['pickup_longitude'])
train_df['dropoff_latitude'] = 
                       radian_conv(train_df['dropoff_latitude'])
train_df['dropoff_longitude'] = 
                       radian_conv(train_df['dropoff_longitude'])
```

现在，还有什么？

您可能希望从 **datetime** 特性中提取详细的日期和时间数据。

```
def add_datetime_info(dataset):
    #Convert to datetime format
    dataset['pickup_datetime'] = pd.to_datetime(dataset['pickup_datetime'],format="%Y-%m-%d %H:%M:%S UTC")

    dataset['hour'] = dataset.pickup_datetime.dt.hour
    dataset['day'] = dataset.pickup_datetime.dt.day
    dataset['month'] = dataset.pickup_datetime.dt.month
    dataset['weekday'] = dataset.pickup_datetime.dt.weekday
    dataset['year'] = dataset.pickup_datetime.dt.year

    return datasettrain_df = add_datetime_info(train_df)
```

这些是您可能希望看到的常见功能，因为出租车费用可能是季节性的。让我们将所有这些要素放入数据集中。

# 开始变得有创造力

从这一点来说，您已经准备好了，因为您可能已经从初始特征中提取了所有可能的特征。

这一次，你可能想在餐桌上增加一些创意。

给自己一个问题。

为什么人们在纽约打车？

1.  可能他们刚到机场就去市区了。在这种情况下，如果乘坐距离机场较近，而乘坐距离机场较远，情况可能会有所不同。
2.  人们到处都乘出租车。市内，市外，靠近市中心，远离市中心。但是大概票价会因为你离纽约市中心有多远而有所不同。
3.  自由女神像？

![](img/22d8432d8b99a4062721a938fcd6a517.png)

Photo by [AussieActive](https://unsplash.com/@aussieactive?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

让我们计算一下乘车点和这些点之间的距离。

```
def add_airport_dist(dataset):
    """
    Return minumum distance from pickup or dropoff coordinates to 
    each airport.
    JFK: John F. Kennedy International Airport
    EWR: Newark Liberty International Airport
    LGA: LaGuardia Airport
    SOL: Statue of Liberty 
    NYC: Newyork Central
    """
    jfk_coord = (40.639722, -73.778889)
    ewr_coord = (40.6925, -74.168611)
    lga_coord = (40.77725, -73.872611)
    sol_coord = (40.6892,-74.0445) # Statue of Liberty
    nyc_coord = (40.7141667,-74.0063889) 

    pickup_lat = dataset['pickup_latitude']
    dropoff_lat = dataset['dropoff_latitude']
    pickup_lon = dataset['pickup_longitude']
    dropoff_lon = dataset['dropoff_longitude']

    pickup_jfk = sphere_dist(pickup_lat, pickup_lon, 
                       jfk_coord[0], jfk_coord[1]) 
    dropoff_jfk = sphere_dist(jfk_coord[0], jfk_coord[1], 
                       dropoff_lat, dropoff_lon) 
    pickup_ewr = sphere_dist(pickup_lat, pickup_lon, 
                       ewr_coord[0], ewr_coord[1])
    dropoff_ewr = sphere_dist(ewr_coord[0], ewr_coord[1], 
                       dropoff_lat, dropoff_lon) 
    pickup_lga = sphere_dist(pickup_lat, pickup_lon, 
                       lga_coord[0], lga_coord[1]) 
    dropoff_lga = sphere_dist(lga_coord[0], lga_coord[1], 
                       dropoff_lat, dropoff_lon)
    pickup_sol = sphere_dist(pickup_lat, pickup_lon, 
                       sol_coord[0], sol_coord[1]) 
    dropoff_sol = sphere_dist(sol_coord[0], sol_coord[1], 
                       dropoff_lat, dropoff_lon)
    pickup_nyc = sphere_dist(pickup_lat, pickup_lon, 
                       nyc_coord[0], nyc_coord[1]) 
    dropoff_nyc = sphere_dist(nyc_coord[0], nyc_coord[1], 
                       dropoff_lat, dropoff_lon)

    dataset['jfk_dist'] = pickup_jfk + dropoff_jfk
    dataset['ewr_dist'] = pickup_ewr + dropoff_ewr
    dataset['lga_dist'] = pickup_lga + dropoff_lga
    dataset['sol_dist'] = pickup_sol + dropoff_sol
    dataset['nyc_dist'] = pickup_nyc + dropoff_nyc

    return datasettrain_df = add_airport_dist(train_df) 
```

这取决于你的想象力，你的特征可能不像我创造的。你可能对影响出租车费用的因素有另一种直觉，比如离商业区或住宅区的距离。或者你想象的任何东西。

# 该训练了

现在，在您准备好数据之后，让我们来训练模型。

```
train_df.drop(columns=['key', 'pickup_datetime'], inplace=True)

y = train_df['fare_amount']
train_df = train_df.drop(columns=['fare_amount'])x_train,x_test,y_train,y_test = train_test_split(train_df, y, test_size=0.10)del train_df
del y
gc.collect()
```

将数据以 90:10 的比例分割成训练验证分割。获得训练数组后，不要忘记删除数据帧。它占用了大量资源。

```
params = {
        'boosting_type':'gbdt',
        'objective': 'regression',
        'nthread': 4,
        'num_leaves': 31,
        'learning_rate': 0.05,
        'max_depth': -1,
        'subsample': 0.8,
        'bagging_fraction' : 1,
        'max_bin' : 5000 ,
        'bagging_freq': 20,
        'colsample_bytree': 0.6,
        'metric': 'rmse',
        'min_split_gain': 0.5,
        'min_child_weight': 1,
        'min_child_samples': 10,
        'scale_pos_weight':1,
        'zero_as_missing': True,
        'seed':0,
        'num_rounds':50000
    }train_set = lgbm.Dataset(x_train, y_train, silent=False,categorical_feature=['year','month','day','weekday'])valid_set = lgbm.Dataset(x_test, y_test, silent=False,categorical_feature=['year','month','day','weekday'])model = lgbm.train(params, train_set = train_set, num_boost_round=10000,early_stopping_rounds=500,verbose_eval=500, valid_sets=valid_set)
```

它将永远训练，直到你的模型不能再优化结果。这将使你达到大约 25，000 次迭代，并给你 **$3.47966** 均方根误差。

这是一个巨大的提升，可以为你节省 1.5 美元到 4.5 美元的出租车费。你可以用它买一份简单的快餐。Lol。

![](img/7de79ac5e9798a92a5572b3916b7af86.png)

Photo by [Jen Theodore](https://unsplash.com/@jentheodore?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 结论

一开始直觉可能很难。但是你知道，通常是常识帮助你度过所有这些。通过了解基层的数据情况，可以让你比以前更直观一点。

别忘了，继续努力！

干杯