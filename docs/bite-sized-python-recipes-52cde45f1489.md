# 一口大小的蟒蛇食谱

> 原文：<https://towardsdatascience.com/bite-sized-python-recipes-52cde45f1489?source=collection_archive---------12----------------------->

## Python 中有用的小函数的集合

![](img/1fe1d42c4778b40fc5b8381ff7a4811d.png)

Photo by [Jordane Mathieu](https://unsplash.com/@mat_graphik?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

*免责声明:*这是我在网上找到的一些有用的小函数的集合，主要在 Stack Overflow 或者 Python 的文档页面上。有些人可能会看，但无论如何，我都在我的项目中使用过它们，我认为它们值得分享。你可以在我试图保持更新的笔记本中找到所有这些，以及一些额外的评论。

除非必要，我不打算过度解释这些功能。那么，让我们开始吧！

**从两个列表中创建字典:**

```
>>> prod_id = [1, 2, 3]
>>> prod_name = ['foo', 'bar', 'baz']
>>> prod_dict = dict(zip(prod_id, prod_name))>>> prod_dict
{1: 'foo', 2: 'bar', 3: 'baz'}
```

**从列表中删除重复项并保留顺序:**

```
>>> from collections import OrderedDict>>> nums = [1, 2, 4, 3, 0, 4, 1, 2, 5]
>>> list(OrderedDict.fromkeys(nums))
[1, 2, 4, 3, 0, 5]*# As of Python 3.6 (for the CPython implementation) and
# as of 3.7 (across all implementations) dictionaries remember
# the order of items inserted. So, a better one is:*
>>> list(dict.fromkeys(nums))
[1, 2, 4, 3, 0, 5]
```

**创建多级嵌套字典:**

创建一个字典作为字典中的值。本质上，它是一本多层次的字典。

```
from collections import defaultdict**def** multi_level_dict():
    *""" Constructor for creating multi-level nested dictionary. """* **return** defaultdict(multi_level_dict)
```

*例 1:*

```
>>> d = multi_level_dict()
>>> d['a']['a']['y'] = 2
>>> d['b']['c']['a'] = 5
>>> d['x']['a'] = 6>>> d
{**'a'**: {**'a'**: {**'y'**: 2}}, **'b'**: {**'c'**: {**'a'**: 5}}, **'x'**: {**'a'**: 6}}
```

*例二:*

给出了一个产品列表，其中每个产品需要从其原产地运送到其配送中心(DC)，然后到达其目的地。给定这个列表，为通过每个 DC 装运的产品列表创建一个字典，这些产品来自每个始发地，去往每个目的地。

```
**import** random
random.seed(20)# Just creating arbitrary attributes for each Product instance
**class** Product:
    **def** __init__(self, id):
        self.id = id
        self.materials = random.sample(**'ABCD'**, 3)self.origin = random.choice((**'o1'**, **'o2'**))
        self.destination = random.choice((**'d1'**, **'d2'**, **'d3'**))
        self.dc = random.choice((**'dc1'**, **'dc2'**))

    **def** __repr__(self):
        **return f'P{**str(self.id)**}'**products = [Product(i) **for** i **in** range(20)]# create the multi-level dictionary
**def** get_dc_origin_destination_products_dict(products):
    dc_od_products_dict = multi_level_dict()
    **for** p **in** products:
        dc_od_products_dict[p.dc][p.origin].setdefault(p.destination, []).append(p)
    **return** dc_od_products_dictdc_od_orders_dict = get_dc_origin_destination_products_dict(products)
>>> dc_od_orders_dict
{**'dc1'**: {**'o2'**: {**'d3'**: [P0, P15],
                **'d1'**: [P2, P9, P14, P18],
                **'d2'**: [P3, P13]},
         **'o1'**: {**'d1'**: [P1, P16],
                **'d3'**: [P4, P6, P7, P11],
                **'d2'**: [P17, P19]}},
 **'dc2'**: {**'o1'**: {**'d1'**: [P5, P12], 
                **'d3'**: [P10]},
         **'o2'**: {**'d1'**: [P8]}}}
```

请注意，当您运行以上两个示例时，您应该在输出中看到`defaultdict(<function __main__.multi_level_dict()>...)`。但是为了结果的易读性，这里删除了它们。

**返回嵌套字典最内层的键和值:**

```
**from** collections **import** abc

**def** nested_dict_iter(nested):
    *""" Return a generator of the keys and values from the innermost layer of a nested dict. """* **for** key, value **in** nested.items():
        *# Check if value is a dictionary* **if** isinstance(value, abc.Mapping):
            **yield from** nested_dict_iter(value)
        **else**:
            **yield** key, value
```

关于此功能，有几点需要说明:

*   `nested_dict_iter`函数返回一个[生成器](https://wiki.python.org/moin/Generators)。
*   在每个循环中，字典值被递归地检查，直到到达最后一层。
*   在条件检查中，为了通用性，使用了`[collections.abc.Mapping](https://docs.python.org/3/library/collections.abc.html#collections.abc.Mapping)`而不是`dict`。这样就可以检查容器对象，比如`dict`、`collections.defaultdict`、`collections.OrderedDict`和`collections.Counter`。
*   为什么是`yield from`？简短而不完整的回答:它是为需要从生成器内部调用生成器的情况而设计的。我知道一个简短的解释不能做到任何公正，所以检查[这个 so 线程](https://stackoverflow.com/questions/9708902/in-practice-what-are-the-main-uses-for-the-new-yield-from-syntax-in-python-3)以了解更多信息。

*例 1:*

```
>>> d = {'a':{'a':{'y':2}},'b':{'c':{'a':5}},'x':{'a':6}}
>>> list(nested_dict_iter(d))
[('y', 2), ('a', 5), ('a', 6)]
```

*示例 2:* 让我们从上面的`dc_od_orders_dict`中检索键和值。

```
>>> list(nested_dict_iter(dc_od_orders_dict))
[('d3', [P0, P15]),
 ('d1', [P2, P9, P14, P18]),
 ('d2', [P3, P13]),
 ('d1', [P1, P16]),
 ('d3', [P4, P6, P7, P11]),
 ('d2', [P17, P19]),
 ('d1', [P5, P12]),
 ('d3', [P10]),
 ('d1', [P8])]
```

**多个集合的交集:**

```
def get_common_attr(attr, *args):
    """ intersection requires 'set' objects """ return set.intersection(*[set(getattr(a, attr)) for a in args])
```

*例:*找出前 5 个`products`中的共同组成材料(如果有的话)。

```
>>> get_common_attr(**'materials'**, *products[:5])
{'B'}
```

**第一场比赛:**

从符合条件的 iterable 中查找第一个元素(如果有的话)。

```
first_match = next(i **for** i **in** iterable **if** check_condition(i))# Example:
>>> nums = [1, 2, 4, 0, 5]
>>> next(i for i in nums if i > 3)
4
```

如果没有找到匹配，上面的实现抛出一个`StopIteration`异常。我们可以通过返回一个默认值来解决这个问题。既然来了，就让它成为一个函数吧:

```
**def** first_match(iterable, check_condition, default_value=**None**):
    **return** next((i **for** i **in** iterable **if** check_condition(i)), default_value)
```

*例如:*

```
>>> nums = [1, 2, 4, 0, 5]
>>> first_match(nums, lambda x: x > 3)
4
>>> first_match(nums, lambda x: x > 9) # returns nothing
>>> first_match(nums, lambda x: x > 9, 'no_match')
'no_match'
```

**动力组:**

集合 ***S*** 的幂集是 ***S*** 的所有子集的集合。

```
import itertools as it**def** powerset(iterable):s = list(iterable)
    **return** it.chain.from_iterable(it.combinations(s, r)
                                  **for** r **in** range(len(s) + 1))
```

*例如:*

```
>>> list(powerset([1,2,3]))
[(), (1,), (2,), (3,), (1, 2), (1, 3), (2, 3), (1, 2, 3)]
```

**定时器装饰器:**

显示每个类/方法/函数的运行时。

```
**from** time **import** time
**from** functools **import** wraps

**def** timeit(func):
    *"""* **:param** *func: Decorated function* **:return***: Execution time for the decorated function
    """* @wraps(func)
    **def** wrapper(*args, **kwargs):
        start = time()
        result = func(*args, **kwargs)
        end = time()
        print(**f'{**func.__name__**} executed in {**end - start**:.4f} seconds'**)**return** result

    **return** wrapper
```

*例如:*

```
**import** random# An arbitrary function
@timeit
**def** sort_rnd_num():
    numbers = [random.randint(100, 200) **for** _ **in** range(100000)]
    numbers.sort()
    **return** numbers>>> numbers = sort_rnd_num()
sort_rnd_num executed in 0.1880 seconds
```

**计算文件中的总行数:**

```
**def** file_len(file_name, encoding=**'utf8'**):
    **with** open(file_name, encoding=encoding) **as** f:
        i = -1
        **for** i, line **in** enumerate(f):
            **pass
    return** i + 1
```

*举例:*你当前目录的 python 文件有多少行代码？

```
>>> from pathlib import Path>>> p = Path()
>>> path = p.resolve()  # similar to os.path.abspath()
>>> print(sum(file_len(f) for f in path.glob('*.py')))
745
```

**只是为了好玩！创建长标签:**

```
>>> s = "#this is how I create very long hashtags"
>>> "".join(s.title().split())
'#ThisIsHowICreateVeryLongHashtags'
```

# **以下不是一口大小的食谱，但不要被这些错误咬到！**

注意不要混淆可变和不可变对象！
*示例:*用空列表作为值初始化字典

```
>>> nums = [1, 2, 3, 4]
# Create a dictionary with keys from the list. 
# Let's implement the dictionary in two ways
>>> d1 = {n: [] for n in nums}
>>> d2 = dict.fromkeys(nums, [])
# d1 and d2 may look similar. But list is mutable.
>>> d1[1].append(5)
>>> d2[1].append(5)
# Let's see if d1 and d2 are similar
>>> print(f'd1 = {d1} \nd2 = {d2}')
d1 = {1: [5], 2: [], 3: [], 4: []} 
d2 = {1: [5], 2: [5], 3: [5], 4: [5]}
```

不要在遍历列表时修改它！

*示例:*从列表中删除所有小于 5 的数字。

*错误实现:*迭代时移除元素！

```
nums = [1, 2, 3, 5, 6, 7, 0, 1]
for ind, n in enumerate(nums):
    if n < 5:
        del(nums[ind])# expected: nums = [5, 6, 7]
>>> nums
[2, 5, 6, 7, 1]
```

*正确实施:*

使用列表理解创建一个新列表，只包含您想要的元素:

```
>>> id(nums)  # before modification 
2090656472968
>>> nums = [n for n in nums if n >= 5]
>>> nums
[5, 6, 7]
>>> id(nums)  # after modification
2090656444296
```

你可以在上面看到,`[id](https://docs.python.org/3/library/functions.html#id)(nums)`在前面和后面被检查，以表明实际上这两个列表是不同的。因此，如果在其他地方使用该列表，并且改变现有列表很重要，而不是创建一个同名的新列表，则将它分配给切片:

```
>>> nums = [1, 2, 3, 5, 6, 7, 0, 1]
>>> id(nums)  # before modification 
2090656472008
>>> nums[:] = [n for n in nums if n >= 5]
>>> id(nums)  # after modification
2090656472008
```

目前就这样了(查看第二个小型博客[这里](/bite-sized-python-recipes-vol-2-385d00d17388))。如果你也有一些经常使用的小函数，请告诉我。我会尽量让[的笔记本](https://github.com/ekhoda/utilities/blob/master/Bite-Sized%20Recipes.ipynb)在 GitHub 上保持最新，你的也可以在那里结束！

*我可以在* [*Twitter*](https://twitter.com/EhsanKhoda) *和*[*LinkedIn*](https://www.linkedin.com/in/ehsankhodabandeh)*上联系到。*