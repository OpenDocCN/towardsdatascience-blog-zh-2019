# 一口大小的蟒蛇食谱——第 2 卷

> 原文：<https://towardsdatascience.com/bite-sized-python-recipes-vol-2-385d00d17388?source=collection_archive---------32----------------------->

## Python 中有用的小函数的集合

![](img/ae1397bfb43bc856eef33ce98de6dc66.png)

Photo by [Jordane Mathieu](https://unsplash.com/@mat_graphik?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

*免责声明:*这是我在网络上找到的一些有用的小函数的集合，主要在堆栈溢出或 Python 的文档页面上。有些可能看起来微不足道，但无论如何，我都在我的项目中使用过它们，我认为它们值得分享。你可以在我试图保持更新的笔记本中找到它们(以及一些额外的评论)。如果你感兴趣，你可以在这里查看我的第一篇关于小型函数的博客！

除非必要，我不打算过度解释这些功能。那么，让我们开始吧！

**返回可迭代**的前 *N* 项

```
**import** itertools **as** it**def** first_n(iterable, n):
    *""" If n > len(iterable) then all the elements are returned. """***return** list(it.islice(iterable, n))
```

*示例:*

```
>>> d1 = {3: 4, 6: 2, 0: 9, 9: 0, 1: 4}
>>> first_n(d1.items(), 3)
[(3, 4), (6, 2), (0, 9)]
>>> first_n(d1, 10)
[3, 6, 0, 9, 1]
```

检查一个可迭代的所有元素是否都相同

```
**import** itertools **as** it**def** all_equal(iterable):
    *""" Returns True if all the elements of iterable are equal to each other. """*g = it.groupby(iterable)
    **return** next(g, **True**) **and not** next(g, **False**)
```

*例如:*

```
>>> all_equal([1, 2, 3])
False
>>> all_equal(((1, 0), (True, 0)))
True
>>> all_equal([{1, 2}, {2, 1}])
True
>>> all_equal([{1:0, 3:4}, {True:False, 3:4}])
True
```

当您有一个序列时，下面的替代方法通常会更快。(如果您正在处理非常大的序列，请确保自己进行测试。)

```
**import** itertools **as** it**def** all_equal_seq(sequence):
    *""" Only works on sequences. Returns True if the sequence is empty or all the elements are equal to each other. """***return not** sequence **or** sequence.count(sequence[0]) == len(sequence)
```

例如:你有一份卡车清单，可以查看它们是在仓库里还是在路上。随着时间的推移，每辆卡车的状态都会发生变化。

```
**import** random
random.seed(500)*# Just creating an arbitrary class and attributes* **class** Truck:
    **def** __init__(self, id):
        self.id = id
        self.status = random.choice((**'loading-unloading'**, **'en route'**))**def** __repr__(self):
        **return f'P{**str(self.id)**}'** trucks = [Truck(i) **for** i **in** range(50)]
```

早上你查看了一下，发现第一辆卡车是`en route`。你听说另外三个人也离开了仓库。让我们验证一下:

```
>>> all_equal_seq([t.status **for** t **in** trucks[:4]])
True
```

**用**和`**None**`求和

当你有`numpy`阵列或者`pandas`系列或者数据帧时，选择是显而易见的:`[numpy.nansum](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nansum.html)`或者`[pandas.DataFrame/Series.sum](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.sum.html)`。但是如果你不想或者不能使用这些呢？

```
**def** sum_with_none(iterable):**assert not** any(isinstance(v, str) **for** v **in** iterable), **'string is not allowed!'** **return** sum(filter(**None**, iterable))
```

这是因为`[filter](https://docs.python.org/3/library/functions.html#filter)`将`None`视为身份函数；即删除 iterable 的所有 [falsy](https://stackoverflow.com/a/39984051) 元素。

*例如:*

```
>>> seq1 = [None, 1, 2, 3, 4, 0, True, False, None]
>>> sum(seq1)
**TypeError**: unsupported operand type(s) for +: 'int' and 'NoneType'>>> sum_with_none(seq1)  # Remember True == 1
11
```

**检查 *N* 或更少的项目是否真实**

```
**def** max_n_true(iterable, n):
    *""" Returns True if at most `n` values are truthy. """***return** sum(map(bool, iterable)) <= n
```

*例如:*

```
>>> seq1 = [None, 1, 2, 3, 4, 0, True, False, None]
>>> seq2 = [None, 1, 2, 3, 4, 0, True, False, 'hi']
>>> max_n_true(seq1, 5)
True
>>> max_n_true(seq2, 5)  # It's now 6
False
```

**检查 Iterable 中是否只有一个元素为真**

```
**def** single_true(iterable):
    *""" Returns True if only one element of iterable is truthy. """*i = iter(iterable)
    **return** any(i) **and not** any(i)
```

函数的第一部分确保迭代器有任何[真值](https://stackoverflow.com/a/39984051)值。然后，它从迭代器中的这一点开始检查，以确保没有其他真值。

*示例:*使用上面的几个函数！

```
*# Just creating an arbitrary class and attributes* **class** SampleGenerator:
    **def** __init__(self, id, method1=**None**, method2=**None**, method3=**None**,
                 condition1=**False**, condition2=**False**,
                 condition3=**False**):
        *"""
        Assumptions:
        1) One and only one method can be active at a time.
        2) Conditions are not necessary, but if passed, maximum one can have value.
        """

        # assumption 1* **assert** single_true([method1, method2, method3]), **"Exactly one method should be used"** *# assumption 2* **assert** max_n_true([condition1, condition2, condition3], 1), **"Maximum one condition can be active"**self.id = id
```

下面的第一个样本(`sample1`)是有效的，但是其他样本违反了至少一个假设，导致了一个`AssertionError`(为了避免混乱，我没有在这里显示任何一个。)在笔记本上运行它们，亲自查看错误。

```
>>> sample1 = SampleGenerator(1, method1=**'active'**)  *# Correct* >>> sample2 = SampleGenerator(2, condition2=**True**)  *# no method is active* >>> sample3 = SampleGenerator(3, method2=**'active'**, method3=**'not-active'**)  *# more than one method has truthy value* >>> sample4 = SampleGenerator(4, method3=**'do something'**, condition1=**True**, condition3=**True**)  *# multiple conditions are active* >>> sample5 = SampleGenerator(5)  *# nothing is passed*
```

**写入 CSV 时跳过冗余标题**

假设你需要运行一系列的模拟。在每次运行结束时(可能需要几个小时)，您记录一些基本的统计数据，并希望创建或更新一个用于跟踪结果的`restults.csv`文件。如果是这样，您可能希望在第一次之后跳过将头写入文件。

首先，让我们创建一些数据来使用:

```
**import** pandas **as** pd
**import** random*# An arbitrary function* **def** gen_random_data():
    demands = [random.randint(100, 900) **for** _ **in** range(5)]
    costs = [random.randint(100, 500) **for** _ **in** range(5)]
    inventories = [random.randint(100, 1200) **for** _ **in** range(5)]
    data = {**'demand'**: demands, 
            **'cost'**: costs, 
            **'inventory'**: inventories} **return** pd.DataFrame(data)*# Let's create a few df* df_list = [gen_random_data() **for** _ **in** range(3)]
```

现在，让我们假设我们需要在`df_list`数据帧一创建好就把它们写入`orders.csv`。

```
**import** osfilename = **'orders.csv'
for** df **in** df_list:
    df.to_csv(filename, index=**False**, mode=**'a'**, 
              header=(**not** os.path.exists(filename)))
```

如果您不需要一次循环一个相似的数据帧，下面的替代方法是将它们写入文件的简洁方法:

```
pd.concat(df_list).to_csv(‘orders2.csv’, index=False)
```

**将 CSV 文件转换为 Python 对象**

假设您需要创建 Python 对象的集合，其中它们的属性来自 CSV 文件的列，并且文件的每一行都成为该类的一个新实例。然而，假设您事先不知道 CSV 列是什么，因此您不能用期望的属性初始化该类。

下面，您可以看到实现这一点的两种方法:

```
**class** MyClass1(object):
    **def** __init__(self, *args, **kwargs):
        **for** arg **in** args:
            setattr(self, arg, arg) **for** k, v **in** kwargs.items():
            setattr(self, k, v) **class** MyClass2:
    **def** __init__(self, **kwargs):
        self.__dict__.update(kwargs)
```

在`MyClass1`中，我们可以同时传递`args`和`kwargs`，而在`MyClass2`中，我们利用了特殊的`[__dict__](https://stackoverflow.com/a/19907498)`属性。

*示例:*让我们使用这两种实现将上面示例中的`orders.csv`文件转换成对象。

```
**import** csvfilename = **'orders.csv'** class1_list = []
class2_list = []**with** open(filename) **as** f:
    reader = csv.DictReader(f)**for** row **in** reader:class1_list.append(MyClass1(**row))
        class2_list.append(MyClass2(**row))# Let's check the attributes of the first row of class1_list
>>> print(f'first row = {vars(class1_list[0])}')
first row = {'demand': '821', 'cost': '385', 'inventory': '1197'}
```

暂时就这样了。如果你也有一些经常使用的小函数，请告诉我。我会尽量在 GitHub 上保持[笔记本](https://github.com/ekhoda/utilities/blob/master/Bite-Sized%20Recipes.ipynb)的最新状态，你的笔记本也可以放在那里！

*我可以在* [*Twitter*](https://twitter.com/EhsanKhoda) *和*[*LinkedIn*](https://www.linkedin.com/in/ehsankhodabandeh/)*上联系到。*