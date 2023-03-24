# 请求原谅比请求许可更容易:一个简短的 Python 例子。

> 原文：<https://towardsdatascience.com/easier-to-ask-for-forgiveness-than-permission-a-short-python-example-e115566a3b8c?source=collection_archive---------29----------------------->

一个简单的概念，可以帮助您避免重复解析所有给定的数据，而您只需要用一个小样本来交叉检查它们。

![](img/f00b097c403c8233ffc4b14830c0c7fa.png)

Photo by [Pisit Heng](https://unsplash.com/@pisitheng?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

想象一下，你已经记下了你看过的电影，以及每部电影你看了多少次。

出于某种奇怪的原因，您将注释保存在 python 字典中:-)，
其中键是电影标题，值是您观看电影的次数。

```
movies = {
...
'The Godfather' : 5,
'Pulp Fiction' : 2,
...
}
```

在某个时候，你决定你只想在你的字典里保留黑帮电影，而不是别的。
所以你得用黑帮电影清单来反复核对你的字典。

一种方法是获取字典键，并查看它们是否包含在给定的黑帮列表中。
这个结果到 **O(k*n)** 动作总计，如果我们在黑帮名单里有 **k 值**，在电影字典里有 **n 值**。

```
gangster_list = [..] # k valuesfor key in movies.keys(): # n values
    if movies[key] in gangster_list:
        continue
    else:
        del movies[key]# O(k*n) actions in total
```

但是如果黑帮电影排行榜包含了 100 万部电影，其中你看过的黑帮电影接近大黑帮排行榜的末尾呢？

假设你已经看了 10 部黑帮电影。
上面的‘in’检查将导致对你所观看的每部电影的几乎整个黑帮列表进行解析，结果是:
O([列表中的 1m 个值] * [观看的 10 部黑帮电影]) = O(10m)个动作。

反而是**请求原谅比请求允许更容易！**

我们可以用 O(k)行动的成本从大黑帮名单中产生一个虚拟字典:

```
gangster_dic = {}
for movie in gangster_list:
    gangster_dic[movie] = None
```

而黑帮字典的内容将会是:

```
{ # k keys without values
...
'Goodfellas' : None,
'The Untouchables' : None,
...
}
del gangster_list # To balance used memory
```

现在，我们可以直接搜索每个观看过的电影名称:

```
for key in movies.keys(): # n values
    # O(1) action
    try:
        gangster_dic[key]
        # do nothing: this is a gangster movie
    except KeyError:
        del movies[key]
        # delete the movie, since we did not find it in the   
        # gangster movies dictionary
```

如果我们失败了，没什么大不了的；我们处理异常。

**这其实是我们想要的:**为了例外发生而删除了一部我们看过的电影，那不是黑帮片。

总的来说，动作的成本是:
O(k)用于字典，加上
O(n * 1)用于检查所观看的电影在黑帮电影集合中是否存在，
导致总共 **O(k+m)** 个动作。

如果我们再次假设，k=1m，n=10，那么我们总共得到(1m+10) = O(1m)个动作。

我们将所需行动从最初的 1000 万减少到了 100 万！或者总的来说需要的动作少了#n 倍！

**`请求原谅比请求许可容易`**这个概念多次对我派上用场。我希望你也会发现它很有用！