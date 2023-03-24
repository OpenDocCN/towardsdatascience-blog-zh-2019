# 让你看起来像詹姆斯·布朗的五个蟒蛇动作

> 原文：<https://towardsdatascience.com/five-python-moves-that-make-you-look-funcy-like-james-brown-7da78a9c4050?source=collection_archive---------15----------------------->

![](img/0bef1f789d1d46efad313b9048a3d9db.png)

Photo by [Start Digital](https://unsplash.com/@startdig?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/funky?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

在我们作为数据科学家的日常生活中，我们经常使用各种 Python 数据结构，如列表、集合或字典，或者更一般地说，我们使用 *iterables* 和 *mappings* 。有时，转换或操作这些数据结构会变得非常代码化。这可能导致代码不可读，并增加引入错误的机会。幸运的是，有一个叫做 [funcy](https://funcy.readthedocs.io/en/stable/) 的简洁的 Python 模块可以帮助我们简化这些任务。在这篇短文中，我向您展示了普通的 Python 代码和相应的 funcy 函数，这些函数允许您用可读性更好的代码更高效地完成五个不同的任务。读完这篇文章后，我相信你会变得更有趣:)

## 省略/投影

有时您得到了一个字典，而您想继续使用该字典的一个子集。例如，假设您正在构建一个 rest API 端点，并且您只想返回模型属性的子集。为此，funcy 提供了两个功能，即*省略*和*项目*

```
**from** funcy **import** project, omitdata = {**"this"**: 1, **"is"**: 2, **"the"**: 3, **"sample"**: 4, **"dict"**: 5}**# FUNCY** omitted_f = omit(data, (**"is"**, **"dict"**))
**# PLAIN PYTHON**
omitted_p = {k: data[k] 
             **for** k **in** set(data.keys()).difference({**"is"**, **"dict"**})
            }**# FUNCY** projected_f = project(data, (**"this"**, **"is"**))
**# PLAIN PYTHON** projected_p = {k: data[k] **for** k **in** (**"this"**, **"is"**)}
```

使用 funcy，您不仅需要键入更少的字符，还可以获得可读性更好、更少出错的代码。但是为什么我们需要两个功能呢？如果您想要保留的关键点数量小于您想要移除的关键点数量，选择*项目*否则选择*忽略*。

## 展平嵌套的数据结构

假设你有一个嵌套的数据结构，比如一个列表和列表的列表，你想把它变成一个列表。

```
**from** funcy **import** lflattendata = [1, 2, [3, 4, [5, 6]], 7, [8, 9]]**# FUNCY**
flattened_f = lflatten(data)**# PLAIN PYTHON**
**def** flatter(in_):
    for e in in_:
        **if** isinstance(e, list):
            **yield** **from** flatter(e)
        **else**:
            **yield** eflattend_p = [e **for** e **in** flatter(data)]
```

如您所见，funcy 版本只有一行代码，而普通 Python 版本看起来相当复杂。我也花了一些时间来想出解决方案，我仍然没有 100%的信心。所以，我会坚持使用 funcy:)除了 list 版本 **l** flatten，funcy 还为 iterables 提供了一个更通用的版本，称为 flatten，没有 l 前缀。你会发现对于不同的函数。

## 分成块

假设您有一个包含 *n* 个条目的 iterable，并且您想要将它分成包含 *k < n* 个元素的块。如果 n 不能被 k 整除，则最后一个块可以小于 k。这就像有一个 n 个样本的训练集，您希望将其分成大小为 k 的批次来进行批处理

```
**from** funcy **import** lchunks
data = list(range(10100))**# FUNCY**
**for** batch **in** lchunks(64, data):
    # process the batch
    pass**# PLAIN PYTHON
from** typing **import** Iterable, Any, List**def** my_chunks(batch_size:int, data:Iterable[Any])->List[Any]:
    res = []
    **for** i, e **in** enumerate(data):
        res.append(e)
        **if** (i + 1) % batch_size == 0:
            **yield** res
            res = []
    **if** res:
        yield res

**for** batch **in** my_chunks(64, data):
    # process the batch
    **pass**
```

注意，我使用了 *lchunks* 版本来进行列表分区，而不是更通用的 *chunks* 版本来对可重复项进行分区。您所要做的就是传递您想要的批处理/块大小，以及您想要分区的 iterable。还有另一个 funcy 函数，叫做*分区*，它只返回那些正好有 k 个条目的块。因此，如果 n 不能被 k 整除，它将忽略最后一个。

## 组合多个词典

假设您有多个保存不同日期但相同对象的数据的字典。您的目标是将所有这些字典合并成一个字典，并使用特定的函数合并来自相同关键字的数据。在这里，funcy 的 *merge_with* 函数派上了用场。你只需要传入合并函数和所有你想合并的字典。

```
**from** funcy **import** merge_with, lcat
d1 = {1: [1, 2], 2: [4, 5, 6]}**# FUNCY VERSION**
merged_f = merge_with(lcat, d1,d2)**# PYTHON VERSION**
**from** itertools **import** chain
**from** typing **import** Callable, Iterable, Any, Dict
**def** _merge(func: Callable[[Iterable], Any], *dics:List[Dict])->Dict:
    # Get unique keys
    keys = {k **for** d **in** dics **for** k **in** d.keys()}
    return {k: func((d[k] **for** d **in** dics **if** k **in** d)) **for** k **in** keys}merged_p = _merge(lambda l: list(chain(*l)), d1, d2)
```

还有一个函数 *join_with* ，它类似于 merge_with，但不是将每个字典作为单个参数传递，而是传递一个可迭代的字典。哦，我“偶然”潜入了另一个 funcy 函数 *lcat，*它将不同的列表合并成一个。

## 缓存属性

最后但同样重要的是，完全不同但非常有用的东西； *cached_prope* rty 装饰器。顾名思义，它使您能够创建只执行一次的属性，而不是缓存该执行的结果并在所有后续调用中返回该结果。我在构建数据集类时经常使用它，因为它给了我非常干净和易读的接口，同时减少了加载时间。

```
**from** funcy **import** cached_property
**import** pandas **as** pd**# Funcy Version**
**class** DatasetsF:
    @cached_property
    **def** movies(self) -> pd.Dataframe:
        **return** pd.read_csv(**"the_biggest_movie_file.csv"**)**# PYTHON VERSION**
**class** DatasetsP:
    **def** __init__(self):
        self._movies = None@property
    **def** movies(self) -> pd.Dataframe:
        **if** self._movies **is** None:
            self._movies = pd.read_csv(**"the_biggest_movie_file.csv"**)
        **return** self._movies
```

# 结论

在这篇文章中，我向您介绍了 funcy，向您展示了它提供的一个非常小但很方便的功能子集。要快速了解所有功能，请查看此备忘单。我希望这篇文章能激励你学习一些 funcy 动作。它比我在这里向你展示的要多得多。感谢您的关注，如果有任何问题、意见或建议，请随时联系我。