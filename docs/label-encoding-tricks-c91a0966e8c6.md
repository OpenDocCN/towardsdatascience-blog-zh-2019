# 释放数据的价值

> 原文：<https://towardsdatascience.com/label-encoding-tricks-c91a0966e8c6?source=collection_archive---------23----------------------->

![](img/4e13f2d17bc1691ccad5fb1436895f51.png)

Photo by [Sereja Ris](https://unsplash.com/@kimtheris?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 标签编码—通过编码释放您的数据

在这篇文章中，我分享了一些处理标签的技巧。我提出方法来**揭示**你的字符串数据的内部值。第一部分回顾了经典的方法，下面几节讨论更具体的方法。

**如何处理:
1。标签
2。颜色
3。位置
4。周期性特征**

# 1.如何转换标签

![](img/5bdd718f1d04dce8305be9b300a7fd6d.png)

Photo by [Max Baskakov](https://unsplash.com/@snowboardinec?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

大多数机器学习算法无法处理带标签的特征。字符串列很难转换成数值。

让我们考虑以下卡特彼勒特性数据集:

```
id    hair     breed
1     naked    sphynx
2     short    siamois
3     naked    sphynx
4     angora   ankara
..    ...      ...
```

## 列`hair`和`breed`包含字符串格式的数据**。**

要将`hair`特征转换成数值，可以给每个头发标签设置一个数字。例如`naked`变成 0，`short`变成 1 等等；

```
id    hair   breed
1      0     sphynx
2      1     siamois
3      0     sphynx
4      2     ankara
..    ...    ...
```

**这种标签编码方法在头发值之间创建了一个层次。**/`naked`小于`1` / `short`和`2` / `angora`。这个等级在这里不是问题，因为它反映了猫毛的长度。

由于猫品种之间没有****的特定顺序**，应用以前的标签编码方法会误导算法。在这种情况下， **One Hot Encoding** 方法最合适。它用布尔值(**0-假**/**1-真**)按品种创建一个列。结果是:**

```
id  hair  sphynx  siamois  ankara
1    0      1        0        0
2    1      0        1        0
3    0      1        0        0
4    2      0        0        1
.   ...    ...      ...      ...
```

**通过使用虚拟矩阵，**每个品种都被认为是独立的。这种标签编码变体不涉及层次结构。****

**![](img/72b1cb27ef5d402ce06d09b90fdec09d.png)**

**Photo by [Linnea Sandbakk](https://unsplash.com/@linneasandbakk?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)**

## ****小心，一个热点编码一个栏目会产生更多可供学习的功能。如果**品种**的数量与数据集中猫的数量相比较小，则应使用此方法。****

**注意:创建虚拟对象会给你的数据集带来冗余。事实上，如果一只猫既不是斯芬克斯猫也不是暹罗猫，那么它必然是安卡拉猫。因此，可以安全地删除一个虚拟列，而不会丢失任何信息:**

```
id   hair   sphynx   siamois
1     0      1        0         
2     1      0        1         
3     0      1        0         
4     2      0        0         
..   ...    ...      ...
```

****奖金。**认为不同品种的猫相互独立简化了现实。例如，一只猫可能一半是暹罗猫，一半是安卡拉猫。在我们的编码数据集中，我们可以通过在 siamois 和 ankara 列中写入 0.5 来轻松表示这种混合。**

```
id   hair   sphynx   siamois
1     0      1        0         
2     1      0        0.5        
3     0      1        0         
4     2      0        0         
..   ...    ...      ...
```

## **第一部分介绍了编码标签的经典方法。**但是更具体的技术可能更适合。这取决于标签信息以及您需要如何处理数据。F** 跟随技术**可能会有更好的效果。****

# **2.如何转换颜色**

**假设我们现在有了每只猫的眼睛颜色:**

**![](img/e6ebc2f846ed33a052bd23e88133f699.png)**

**Photo by [Antonio Lapa](https://unsplash.com/@alcqlapa?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)**

```
id   eyecolor
1    light_green    
2    dark_cyan
3    medium_sea_green 
4    yellow_green
..   ...
```

**这一次，以前的方法不太适合表达颜色标签中的价值。相反，我们可以将颜色列分为红绿蓝三种色调:**

```
id  eye_red  eye_green  eye_blue
1    173       237         150    
2    70        138         139
3    105       178         117 
4    168       205         67
..   ...       ...         ...
```

**一般来说，颜色有几种编码方式。人们可能需要添加透明度信息。根据数据的使用方式，可以用 HSB(色调饱和亮度)而不是 RGB 进行编码。**

# **3.如何转换位置**

**现在我们考虑猫的位置。我们有他们居住地的城市名。**

**![](img/42eeb78becc0764bec805864e1910338.png)**

**Photo by [Anthony DELANOIX](https://unsplash.com/@anthonydelanoix?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)**

```
id   city
1    Paris
2    New-York
3    Cairo
4    New-York
..   ...
```

**这里，我们可以用城市的 GPS(纬度/经度)十进制度坐标来代替城市名称，而不是使用常规的标签编码方法:**

```
id  latitude   longitude
1   48.85341   2.3488
2   37.6       -95.665
3   30.06263   31.24967
4   37.6       -95.665
..  ...        ...
```

**如您所见，我们根据地理属性对位置进行了编码。**

**根据我们的需要，城市可以用其他指标来表征。例如，如果我们需要预测猫的预期寿命，那么 ***城市发展指数*** ，或者 ***平均家庭收入*** 可能更适合。**

**这些翻译带来了其他类型的信息。**人们可以结合指标来丰富数据集。****

# **4.如何转换周期性特征**

**最后但同样重要的是，让我们关注循环特性。我们现在有了猫外出夜游的平均时间:**

**![](img/e75a3af2884b381561c452e07a673ba3.png)**

**Photo by [Agê Barros](https://unsplash.com/@agebarros?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)**

```
id   avg_night_walk_time
1     11:30 PM
2     00:12 AM
3     10:45 PM
4     01:07 AM
..    ...
```

**首先让我们把它转换成十进制小时:**

```
id   avg_night_walk_time
1     23.5
2     0.2
3     22.75
4     1.12
..    ...
```

**所以现在我们有了`0`和`23.99`之间的时间。作为人类，我们知道 23:59 和 00:00 很接近，但它没有出现在数据中。要改变它，我们可以用余弦和正弦来转换时间。它将数据投影到一个圆形空间中，在该空间中 *00:00 AM* 和 *11:59 PM* 靠得很近。**

```
id   avg_nwt_cos   avg_nwt_sin
1    0.99          -0.13
2    0.99          0.05
3    0.94          -0.32
4    0.95          0.28
..   ...           ...
```

**通过转换时间，我们失去了一些人类的理解，但我们可以在任何时候将余弦和正弦数据转换回小时。这种转换并不意味着信息的丢失。**

**这种循环转换也应该应用于角度和周期过程。**

## **正确处理周期性特征会大大增加数据集的价值。当你用小时、工作日、月、角度工作时，不要忘记它...**

***PS:在本文中，我没有谈论特性缩放，因为我关注的是数据转换。如有必要，考虑在转换后缩放/居中数据。***