# 何时以及如何使用 Python mock

> 原文：<https://towardsdatascience.com/when-and-how-to-use-python-mock-b626e5f9bf7?source=collection_archive---------15----------------------->

![](img/58b881230ab7e8c5bd333225049c372e.png)

Photo by [Suzanne D. Williams](https://unsplash.com/@scw1217?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

## 使用 mock 减少您的测试执行时间

## 议程

这篇文章将涵盖何时以及如何使用`unittest.mock`库。

Python 文档恰当地描述了模拟库:

```
unittest.mock allows you to replace parts of your system under test with mock objects and make assertions about how they have been used.
```

## 何时使用模拟库

简短的回答是“大多数时候”。下面的例子将使这一主张更加清晰:

在 shell 上定义以下三个函数。

```
In [1]: def cow():
   ...:     print("cow")
   ...:     return {'cow': 'moo'}
   ...:In [2]: def dog():
   ...:     print("dog")
   ...:     return {'dog': 'bark'}
   ...:In [3]: def animals():
   ...:     data = cow()
   ...:     data.update(dog())
   ...:     data['pig'] = 'oink'
   ...:     return data
   ...:
```

让我们执行`animals`。素食者会杀了我；)

```
In [4]: animals()
cow
dog
Out[4]: {'cow': 'moo', 'dog': 'bark', 'pig': 'oink'}
```

输出确认了从`animals()`调用了`cow`和`dog`。

让我们为`cow`写一个测试。

```
In [5]: def test_cow():
   ...:     assert cow() == {'cow': 'moo'}
   ...:
```

让我们执行`test_cow`来确保`cow`按预期运行。

```
In [6]: test_cow()
cow
```

让我们同样测试`dog`。

```
In [7]: def test_dog():
   ...:     assert dog() == {'dog': 'bark'}
   ...:In [8]: test_dog()
dog
```

下面给`animals`加个测试。

```
In [9]: def test_animals():
   ...:     assert animals() == {'dog': 'bark', 'cow': 'moo', 'pig': 'oink'}
   ...:In [10]: test_animals()
cow
dog
```

从打印的声明中可以看出，`cow()`和`dog()`是从`test_animals()`开始执行的。

测试`animals`时不需要执行`cow`和`dog`，因为`cow` 和`dog`已经单独测试过了。

在测试`animals`时，我们只想确保`cow`和`dog`会被执行。我们不希望真正的处决发生。

`cow`和`dog`是微小函数。从`test_animals`开始的执行目前不是什么大事。如果函数`cow`和`dog`很大，`test_animals`可能需要很长时间才能完成。

如果我们使用`unitest.mock.patch`，这种情况是可以避免的。

让我们修改`test_animals`如下所示:

```
In [17]: from unittest.mock import patchIn [18]: [@patch](http://twitter.com/patch)('__main__.cow')
    ...: [@patch](http://twitter.com/patch)('__main__.dog')
    ...: def test_animals(patched_dog, patched_cow):
    ...:     data = animals()
    ...:     assert patched_dog.called is True
    ...:     assert patched_cow.called is True
    ...:
```

执行` test_animals()`。

```
In [19]: test_animals()
```

我们再也看不到`cow`和`dog`的打印声明。这确认了`cow`和`dog`没有被执行。

我们来解剖一下`test_animals`。

`test_animals`已装饰有`@patch`。函数`dog`作为参数传递给`@patch`。

由于`test_animals`已经被修饰，所以在`test_animals`的上下文中，实际的函数`dog`已经被替换为一个`unittest.mock.Mock`实例。这个`Mock`实例被称为`patched_dog`。

由于`animals()`是在`test_animals()`的上下文中执行的，所以没有从`animals`实际调用`dog`。取而代之的是`patched_dog`的称呼。

`Mock`实例有一个名为`called`的属性，如果从测试中的函数调用模拟实例，该属性将被设置为 true。我们断言模拟实例`patched_dog`已经被调用。

如果从`animals`到`dog`的调用被移除/注释，那么`test_animals`将失败。

```
In [20]: def animals():
    ...:     data = cow()
    ...:     #data.update(dog())
    ...:     data['pig'] = 'oink'
    ...:     return data
    ...:In [21]: test_animals()
---------------------------------------------------------------------------
AssertionError                            Traceback (most recent call last)
<ipython-input-21-f8c77986484f> in <module>
----> 1 test_animals()/usr/local/Cellar/python/3.6.5/Frameworks/Python.framework/Versions/3.6/lib/python3.6/unittest/mock.py in patched(*args, **keywargs)
   1177
   1178                 args += tuple(extra_args)
-> 1179                 return func(*args, **keywargs)
   1180             except:
   1181                 if (patching not in entered_patchers and<ipython-input-18-a243d0eea2e8> in test_animals(patched_dog, patched_cow)
      3 def test_animals(patched_dog, patched_cow):
      4     data = animals()
----> 5     assert patched_dog.called is True
      6     assert patched_cow.called is True
      7AssertionError:
```

这证实了测试提供了正确的覆盖范围，而没有实际执行`cow`和`dog`。

## 在模拟实例上设置 return_value

取消注释代码。

`animals`中除了调用`cow`和`dog`还有额外的代码。`animals`也将猪添加到数据中。

我们来测试一下`animals`的附加代码。

```
In [43]: [@patch](http://twitter.com/patch)('__main__.cow')
    ...: [@patch](http://twitter.com/patch)('__main__.dog')
    ...: def test_animals(patched_dog, patched_cow):
    ...:     patched_cow.return_value = {'c': 'm'}
    ...:     patched_dog.return_value = {'d': 'b'}
    ...:     data = animals()
    ...:     assert patched_dog.called is True
    ...:     assert patched_cow.called is True
    ...:     assert 'pig' in data
```

让我们执行测试

```
In [45]: test_animals()
```

我们所有的断言都通过了，因为没有断言错误。

调用`Mock`函数的默认返回值是另一个`Mock`实例。因此，当在`test_animals`上下文中从`animals`调用`patched_dog`时，它会返回一个模拟实例。我们不希望它返回一个模拟实例，因为`animals`的附加代码期望它是一个字典。

我们在`patched_cow`上设置一个`return_value`作为字典。同样适用于`patched_dog`。

现在`animals`的附加代码也被测试覆盖。

## 另一个例子

让我们定义一个函数来测试 url 是否有效。这个靠 Python 的`requests`。

```
In [59]: import requestsIn [60]: def is_valid_url(url):
    ...:     try:
    ...:         response = requests.get(url)
    ...:     except Exception:
    ...:         return False
    ...:     return response.status_code == 200
```

下面给`is_valid_url`加个测试。

```
In [69]: def test_is_valid_url():
    ...:     assert is_valid_url('[http://agiliq.com'](http://agiliq.com')) is True
    ...:     assert is_valid_url('[http://agiliq.com/eerwweeee'](http://agiliq.com/eerwweeee')) is False # We want False in 404 pages too
    ...:     assert is_valid_url('[http://aeewererr.com'](http://aeewererr.com')) is FalseIn [70]: test_is_valid_url()
```

您会注意到您的测试在进行网络调用时有多慢。

让我们通过利用`patch`和`Mock`使我们的测试更快来解决这个问题。

```
In [71]: [@patch](http://twitter.com/patch)('__main__.requests')
    ...: def test_is_valid_url(patched_requests):
    ...:     patched_requests.get.return_value = Mock(status_code=200)
    ...:     assert is_valid_url('[http://agiliq.com'](http://agiliq.com')) is True
    ...:     patched_requests.get.return_value = Mock(status_code=404)
    ...:     assert is_valid_url('[http://agiliq.com/eerwweeee'](http://agiliq.com/eerwweeee')) is False # We want False in 404 pages too
    ...:     patched_requests.get = Mock(side_effect=Exception())
    ...:     assert is_valid_url('[http://aeewererr.com'](http://aeewererr.com')) is False
    ...:In [72]: test_is_valid_url()
```

你应该注意到了速度的提升。

## 摘要

模拟应该在下列情况下使用:

*   来模拟任何已经单独测试过的依赖项。`cow`是`animals`的属地，所以我们嘲讽了`cow`。
*   来模拟网络通话。`requests` 或`send_mail`应该被嘲讽以避免网络调用。
*   模拟任何大型函数以减少测试执行时间。

## 签署

希望这篇文章有用。

如果你喜欢这篇文章，考虑发送至少 50 个掌声:)