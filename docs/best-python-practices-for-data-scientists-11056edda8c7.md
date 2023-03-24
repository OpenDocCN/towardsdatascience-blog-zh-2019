# 数据科学家的最佳 Python 实践

> 原文：<https://towardsdatascience.com/best-python-practices-for-data-scientists-11056edda8c7?source=collection_archive---------2----------------------->

## 让我们看看生产级代码的一些行业标准。

![](img/6a594628f37ea90227b634c650a02943.png)

**Photo by:** [Matthew Henry](https://burst.shopify.com/@matthew_henry)

人们常说，一开始从事数据科学家工作的人不会写出干净的代码，这是有原因的。像 EDA、特性选择和预处理这样的大部分先决工作都是在 Jupyter 笔记本上完成的，我们不太关心代码。

将职业从软件开发转向数据科学的人更擅长编写生产级代码。他们知道如何正确处理所有的错误案例、文档、模块化等等。很多公司有时候会要求 X 年的软件开发，因为他们知道软件开发人员可以写出更好的代码。

数据科学家为生产模型而编写的大部分代码并不遵循 PEP8 这样的行业标准。PEP 代表 Python 增强提案。A **PEP** 是一个文档，描述了为 **Python** 提出的新特性，并为社区记录了 **Python** 的各个方面，比如设计和风格。我们在 python 中看到的每一个特性，都是由致力于 python 开发的成员(这些人是来自谷歌、微软和其他大型跨国公司的开发人员)首先提出，然后进行评审的。

但是为什么我们需要遵循这些指导方针呢？

> *正如吉多·范·罗苏姆(Python 的创始人)所说，“代码被阅读的次数比它被编写的次数多得多。”您可能要花几分钟，或者一整天，编写一段代码来处理用户认证。一旦你写了，你就不会再写了。但是你一定要再读一遍。这段代码可能仍然是您正在进行的项目的一部分。每次你回到那个文件，你都必须记住代码做了什么，为什么要写它，所以可读性很重要。*

现在，让我们来看一些 PEP8 指南。

> 命名风格

1.  变量、函数、方法、包、模块

*   `lower_case_with_underscores`

2.类别和例外

*   `CapWords`

3.受保护的方法和内部函数

*   `_single_leading_underscore(self, ...)`

4.私有方法

*   `__double_leading_underscore(self, ...)`

5.常数

*   `ALL_CAPS_WITH_UNDERSCORES`

6.首选反向符号

```
elements = ...
active_elements = ...
defunct_elements ...
```

> 刻痕

使用 4 个空格，不要使用制表符。

> 进口

1.  进口顺序如下

```
1\. python inbuilt packages import2\. third party packages import3\. local imports
```

> 线长度

尝试断开任何超过 80 个字符的行。

```
even_numbers = [var for var in range(100)
                if var % 2 == 0]
```

有时，不可能再换行了，尤其是在方法链接的情况下。

> 真空

1.  在字典中，冒号和键值之间要留有空格。

```
names = {'gagan': 123}
```

2.在赋值和算术运算的情况下，在运算符之间留出空间。

```
var = 25
math_operation_result = 25 * 5
```

3.当作为参数传递时，运算符之间不要留空格

```
def count_even(num=20):
    pass
```

4.逗号后留空格。

```
var1, var2 = get_values(num1, num2)
```

> 证明文件

遵循 [PEP](http://www.python.org/dev/peps/pep-0257/) 257 的 docstring 指南，学习如何记录你的 python 程序。

1.  对于显而易见的函数，使用单行文档字符串。

```
"""Return the pathname of ``foo``."""
```

2.多行文档字符串应包括

```
Summary lineUse case, if appropriateArgsReturn type and semantics, unless None is returned
```

例子

```
class Car:
    """A simple representation of a car.

    :param brand: A string, car's brand.
    :param model: A string, car's model.
    """
    def __init__(self, brand, model):
        self.brand = brand
        self.model = model
```

## 结论

当你写代码时，简单是你首先应该想到的。以后你可能需要参考你的代码或者其他人的代码，这应该不需要太多的努力就能理解。

## 参考

1.  [https://sphinxcontrib-Napoleon . readthedocs . io/en/latest/example _ Google . html # example-Google](https://sphinxcontrib-napoleon.readthedocs.io/en/latest/example_google.html#example-google)
2.  【https://gist.github.com/sloria/7001839 
3.  [https://realpython.com/python-pep8/](https://realpython.com/python-pep8/)