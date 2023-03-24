# Python 中的装饰函数

> 原文：<https://towardsdatascience.com/decorating-functions-in-python-619cbbe82c74?source=collection_archive---------13----------------------->

## 什么是高阶函数？

![](img/b5ed6096e70def6dd9f1ed553476d4de.png)

[source](https://pixabay.com/images/id-1149619/)

在这篇文章中，我将解释什么是 Python decorators，以及如何实现它。我们开始吧！

# 什么是室内设计师？

装饰器只不过是一种调用高阶函数的舒适方式。你可能已经看过很多次了，它是关于那些“@dostuff”字符串的，你会不时地在函数的签名上面找到它们:

```
@dostuff
def foo():
  pass
```

好的，但是现在，什么是高阶函数？高阶函数就是任何一个以一个(或多个)函数作为参数和/或返回一个函数的函数。例如，在上面的例子中，“@dostuff”是修饰者，“dostuff”是高阶函数的名称，“foo”是被修饰的函数和高阶函数的参数。

安全了吗？太好了，让我们开始实现我们的第一个装饰器吧！

# 打造我们的第一个室内设计师

为了开始实现我们的装饰器，我将向您介绍 [functools](https://docs.python.org/3/library/functools.html) : Python 的高阶函数模块。具体来说，我们将使用[包装](https://docs.python.org/3/library/functools.html#functools.wraps)函数。

让我们创建一个高阶函数，它将打印被修饰函数的执行时间:我们称它为“timeme”。这样，每当我们想要计算函数的执行时间时，我们只需要在目标方法的签名上添加装饰符“@timeme”。让我们开始定义“timeme”的签名:

```
def timeme(func):
  pass
```

如前所述，一个高阶函数将另一个函数(修饰函数)作为它的参数，所以我们在它的签名中包含了“func”。现在，我们需要添加一个包含计时逻辑的包装函数。为此，我们将创建一个“包装器”函数，它将被 functools 的[包装器](https://docs.python.org/3/library/functools.html#functools.wraps)函数包装:

```
from functools import wrapsdef timeme(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    pass return wrapper
```

注意,“timeme”返回函数“wrapper ”,该函数除了打印执行时间之外，还将返回修饰函数的结果。

现在，让我们通过实现计时逻辑来完成包装器:

```
from functools import wraps
import timedef timeme(func):
  @wraps(func)
  def wrapper(*args, **kwargs):
    print("Let's call our decorated function")
    start = time.time()
    result = func(*args, **kwargs)
    print('Execution time: {} seconds'.format(time.time() - start))
    return result
  return wrapper
```

注意，修饰函数“func”是用它的位置和关键字参数执行的。我添加了一些打印消息，供您观察执行顺序。好吧！让我们试一试，我将创建一个简单的函数，带有一条打印消息，用“timeme”来修饰:

```
@timeme
def decorated_func():
  print("Decorated func!")
```

如果您运行它，您将看到如下内容:

```
Let's call our decorated function
Decorated func!
Execution time: 4.792213439941406e-05 seconds
```

如您所见，第一个打印消息被放在高阶函数中。然后，我们调用修饰的函数并打印它自己的消息。最后，我们计算执行时间并打印出来。