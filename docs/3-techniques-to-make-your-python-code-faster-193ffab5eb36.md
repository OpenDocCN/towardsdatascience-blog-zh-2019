# 提高 Python 代码速度的 3 种技巧

> 原文：<https://towardsdatascience.com/3-techniques-to-make-your-python-code-faster-193ffab5eb36?source=collection_archive---------6----------------------->

## 你甚至不会流汗

![](img/d0e03e0ade3e74c434c3e7ef18ea5e98.png)

[source](https://pixabay.com/images/id-332857/)

在这篇文章中，我将分享你在日常脚本中可能用到的 3 种 Python 效率技巧，以及如何衡量 2 种解决方案之间的性能提升。我们开始吧！

# 我们如何比较两个候选解决方案的性能？

性能可能指解决方案中的许多不同因素(例如，执行时间、CPU 使用率、内存使用率等)。).不过，在本文中，我们将重点关注执行时间。

新解决方案在执行时间上的改进可以像除法一样简单地计算出来。也就是说，我们将旧的(或未优化的)解决方案的执行时间除以新的(或优化的)解决方案:tell/Tnew。这个指标通常被称为[加速](https://en.wikipedia.org/wiki/Speedup)。例如，如果我们的加速因子为 2，我们改进的解决方案将花费原始解决方案一半的时间。

为了比较我们函数的性能，我们将创建一个函数来接收这两个函数，计算它们的执行时间，并计算获得的加速比:

```
import timedef compute_speedup(slow_func, opt_func, func_name, tp=None):
  x = range(int(1e5))
  if tp: x = list(map(tp, x)) slow_start = time.time()
  slow_func(x)
  slow_end = time.time()
  slow_time = slow_end - slow_start opt_start = time.time()
  opt_func(x)
  opt_end = time.time()
  opt_time = opt_end - opt_start speedup = slow_time/opt_time
  print('{} speedup: {}'.format(func_name, speedup))
```

为了获得有意义的结果，我们将使用一个相对较大的数组(100.000 个元素)，并将其作为参数传递给两个函数。然后，我们将使用时间模块计算执行时间，并最终交付获得的加速。

请注意，我们还传递了一个可选参数，允许我们更改列表元素的类型。

# 1.避免用+运算符连接字符串

您可能会发现的一种常见情况是必须用多个子部分组成一个字符串。Python 有一个方便的+运算符，允许我们以如下方式轻松地连接字符串:

```
def slow_join(x):
  s = ''
  for n in x:
    s += n
```

尽管对我们来说这是一个干净的方法，Python 的字符串是不可变的，因此不能被修改。这意味着每次我们使用+操作符时，Python 实际上是基于两个子字符串创建一个新字符串，并返回新字符串。考虑一下，在我们的例子中，这个操作将被执行 100.000 次。

这种方法显然是有代价的，我们可以使用 [join()](https://docs.python.org/3/library/stdtypes.html#str.join) 找到一个更便宜的解决方案，如下例所示:

```
def opt_join(x):
  s = ''.join(x)
```

这个解决方案采用子字符串数组，并用空字符串分隔符将它们连接起来。让我们检查一下我们的性能改进:

```
compute_speedup(slow_join, opt_join, 'join', tp=str)
```

我得到了 7.25 倍的加速系数！考虑到实现这项技术所需的少量工作，我认为还不错。

# 2.使用地图功能

当我们需要对列表中的每个元素进行操作时，我们通常可以这样做:我们应用生成器理解并处理当前元素。然后，我们可以在必要时迭代它:

```
def slow_map(x):
  l = (str(n) for n in x)
  for n in l:
    pass
```

然而，在许多情况下，您可能更喜欢使用 [Python 的内置映射函数](https://docs.python.org/3/library/functions.html#map)，它对 iterable 中的每个元素应用相同的操作并产生结果。它可以简单地实现如下:

```
def opt_map(x):
  l = map(str, x)
  for n in l:
    pass
```

是时候检查一下我们在执行时间上提高了多少了！如下运行我们的 compute_speedup 函数:

```
compute_speedup(slow_map, opt_map, 'map')
```

我获得了 155 的加速。可能有人会说理解更易读，但是我会说，适应 [map 的](https://docs.python.org/3/library/functions.html#map)语法也不需要什么，至少在简单的场景中是这样(例如，不需要 iterable 上的任何条件)。

# 3.避免重新评估函数

每当您发现自己在循环块中的元素上重复使用相同的函数时，例如:

```
y = []
for n in x:
  y.append(n)
  y.append(n**2)
  y.append(n**3)
```

…或者只是在循环块中使用一次这样的函数，但是是在一个很大的列表上，例如下面的情况:

```
def slow_loop(x):
  y = []
  for n in x:
    y.append(n)
```

…您可以利用另一种优化技术。

如果您以前将函数作为变量保存，并在循环块中重用它，则可以节省重新计算函数的成本。以下片段显示了这种行为:

```
def opt_loop(x):
  y = []
  append = y.append
  for n in x:
    append(n)
```

注意，如果需要将当前元素添加到不同的列表中，就必须为每个列表的 append 函数创建一个新变量。

让我们使用 compute_speedup 来检查加速:

```
compute_speedup(slow_loop, opt_loop, 'loop')
```

在这种情况下，我获得了 2.07 的加速比！同样，我们不需要做任何重大的改变来获得这样的改进。

—

**想要更多提高效率的技巧吗？看看这些文章吧！**

[3 个简单的 Python 效率技巧](/3-simple-python-efficiency-tips-f7c35b511503)

[在 Python 中寻找性能瓶颈](/finding-performance-bottlenecks-in-python-4372598b7b2c)