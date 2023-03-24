# Python 列表没有 Argmax 函数

> 原文：<https://towardsdatascience.com/there-is-no-argmax-function-for-python-list-cd0659b05e49?source=collection_archive---------10----------------------->

## 有三种方法可以解决它

![](img/80581e1de53d9c6c914debbf29aac381.png)

Photo by [Talles Alves](https://unsplash.com/@deffyall?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

**亲爱的 Python 程序员们，**

给定一个 Python 列表`l = [50, 99, 67, 99, 48]`，如何找到列表的 ***argmax*** ？原来 Python list 没有内置的 ***argmax*** 函数！你不能只做`argmax(l)`或`max(l).index`。我能想到的第一件事是将列表转换成一个 *numpy* 数组，或者 *pandas* DataFrame，或者甚至 *Tensorflow* tensor，如果你认为可以过度使用的话:

```
import numpy as np
l_np = np.asarray(l)
print(l_np.argmax())import pandas as pd
l_pd = pd.DataFrame({'l': l})
print(l_pd.idxmax())import tensorflow as tf
l_tf = tf.constant(l)
with tf.Session() as sess:
    print(sess.run(tf.argmax(l_tf)))
```

![](img/c07d9f3b4dd11bd652d118a43e761642.png)

他们都返回`1`，就像他们应该的那样。如果多个项目达到最大值，函数将返回遇到的第一个项目。但是将一个对象转换成另一种数据类型感觉像作弊，而且还需要更多的执行时间。以下是几种修复方法:

# 1.传统 C 逻辑

构建一个 for 循环，手动检查每一个元素，它工作得非常好。但是，它由多行可读性不太好的代码组成。与下面讨论的替代方案相比，它还需要稍微多一点的计算时间。

```
index, max_val = -1, -1
for i in range(len(l)):
    if l[i] > max_val:
        index, max_val = i, l[i]
print(index)
```

![](img/190b4936d6d060b9b476aa6a4f4beb69.png)

Photo by [Nikhil Mitra](https://unsplash.com/@nikhilmitra?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 2.用计数器枚举

我们可以利用 Python 中`enumerate`函数的内置计数器。注意`max((x,i) for i,x in enumerate(l))[1]`返回**最后一个**最大项的索引，但是可以通过

```
-max((x,-i) for i,x in enumerate(l))[1]
```

但这绝不是可读的。我们还可以通过使用`zip`来模仿`enumerate`函数，使其更具可读性:

```
max(zip(l, range(len(l))))[1]
```

![](img/213b40b8a3012f3686cfb0d7ca1489b3.png)

Photo by [Tomas Sobek](https://unsplash.com/@tomas_nz?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 3.更改密钥

这可能是最可读的黑客。原来我们还是可以使用默认的`max`功能。但是我们没有将列表作为参数传递，而是将索引列表作为参数传递，并将一个函数作为“key”传递。该函数将索引映射到列表中相应的元素。

```
f = lambda i: l[i]
max(range(len(l)), key=f)
```

![](img/7931bbd359fe505326a8dab71e237fdb.png)

Photo by [Silas Köhler](https://unsplash.com/@silas_crioco?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

好了，Python 列表上执行 ***argmax*** 的三种方法。

## 相关文章

感谢您的阅读！如果您对 Python 感兴趣，请查看以下文章:

[](/5-python-features-i-wish-i-had-known-earlier-bc16e4a13bf4) [## 我希望我能早点知道的 5 个 Python 特性

### 超越 lambda、map 和 filter 的 Python 技巧

towardsdatascience.com](/5-python-features-i-wish-i-had-known-earlier-bc16e4a13bf4) [](/visualizing-bike-mobility-in-london-using-interactive-maps-for-absolute-beginners-3b9f55ccb59) [## 使用交互式地图和动画可视化伦敦的自行车移动性

### 探索 Python 中的数据可视化工具

towardsdatascience.com](/visualizing-bike-mobility-in-london-using-interactive-maps-for-absolute-beginners-3b9f55ccb59) 

*原载于我的博客*[*edenau . github . io*](https://edenau.github.io)*。*