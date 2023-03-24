# 从 Python 2 切换到 Python 3 的 5 个简单技巧

> 原文：<https://towardsdatascience.com/5-easy-tips-for-switching-from-python-2-to-3-6100f717610f?source=collection_archive---------16----------------------->

![](img/c44f368f5526b12172842e2be93679ab.png)

[Source](https://pixabay.com/photos/person-mountain-top-achieve-1245959/)

> 我为学习者写了一份名为《强大的知识》的时事通讯。每一期都包含链接和最佳内容的关键课程，包括引文、书籍、文章、播客和视频。每一个人都是为了学习如何过上更明智、更快乐、更充实的生活而被挑选出来的。 [**在这里报名**](https://mighty-knowledge.ck.page/b2d4518f88) 。

末日即将来临……

不，这不是世界末日…..是 Python 2 的[结尾！](https://pythonclock.org/)

Python 2 将正式退役，2020 年 1 月 1 日之后****将不再维护**。这个消息对一些人(老一代 Python 2 用户)来说是可怕的，但对另一些人(酷的 Python 3 用户)来说却是巨大的安慰。**

**那么这对那些用 Python 编码的人来说意味着什么呢？你是被迫切换到 Python 3 的吗？**

**实际上，答案是肯定的*是的*。**

**许多主要的 Python 项目已经签署了一份保证书，承诺完全迁移到 Python 3，放弃 Python 2。我们讨论的是一些非常常见的应用中的真正重量级产品:**

*   ****数据科学**:熊猫、Numpy、Scipy 等**
*   ****机器学习** : TensorFlow、Scikit-Learn、XGBoost 等**
*   ****网**:请求，龙卷风**
*   **[**多多多**](https://python3statement.org/)**

**不用说，随着所有这些主要库全面转向 Python 3，更新您的代码库变得非常必要。**

**在本文中，我将与您分享从 Python 2 转换到 Python 3 的 5 个简单技巧，这将使转换平稳进行。**

# **⑴进口**

**Python 2 使用相对导入，而 Python 3 现在使用绝对导入来使事情更加明确。**

**考虑以下示例文件夹结构:**

```
py_package
├── main_code.py
└── helper_code.py
```

**对于 Python 2 的相对导入，import 语句是相对于当前文件编写的。例如，如果我们想在主代码中导入一些助手代码，我们可以这样做:**

```
from helper_code import help_function
```

**Python 3 不再支持这种风格的导入，因为它不明确您想要“相对”还是“绝对”`helper_code`。如果你的计算机上安装了一个名为`helper_code`的 Python 包，你可能会得到错误的包！**

**Python 3 现在要求您使用 *explicit imports* ，它明确指定了您想要使用的模块的位置，相对于您当前的工作目录。因此，导入您的助手代码如下所示:**

```
from .helper_code import help_function
```

**注意包名旁边的新`.`,它指定包驻留在当前文件夹中。你也可以做类似于`..helper_code`的事情，它告诉 Python`helper_code`包比当前的`main_code.py`包高一个目录。要点是 Python 3 要求您指定确切的包位置。**

# **(2)打印报表**

**可怕的 Python 打印语句！**

**我们大多数人都曾遇到过与 Python 打印语句相关的错误。这一改变将 100%要求任何使用 Python 2 打印语句的人切换到 Python 3 风格。**

```
# Python 2 style
print "Hello, World!"# Python 3 style 
print("Hello, World!")
```

**基本上，你必须在你所有的代码中为所有的打印语句加上括号。幸运的是，一旦你的 print 语句有了括号，它们就可以在 Python 2 和 3 中使用了。**

# **(3)整数除法**

**在 Python 2 中，您可以使用`/`操作符对整数进行除法运算，结果会四舍五入到最接近的整数。如果你希望结果是一个浮点数，你首先要把数字转换成浮点数，然后*执行除法。***

**Python 3 通过使用两种不同的操作符:`/`用于浮点除法，`//`用于整数除法，从而避免了必须进行显式转换以转换成浮点的需要。**

```
### Python 2
a = 3 // 2                   # Result is 1
b = 3 / 2                    # Result is 1
c = float(3) / float(2)      # Result is 1.5### Python 3
a = 3 // 2                   # Result is 1
b = 3 / 2                    # Result is 1.5
```

**Python 3 中显式操作符的使用使得代码更加简洁，可读性更好——我们确切地知道`//`是整数,`/`是浮点数，而不必进行显式类型转换。**

**新的除法运算符肯定是要小心的。如果您在从 Python 2 到 3 的过程中忘记了从`/`转换到`//`进行整数除法运算，那么您将在原来拥有整数的地方得到浮点数！**

# **(4)字符串格式**

**在 Python 3 中，字符串格式的语法更加清晰。在 Python 2 中，您可以像这样使用`%`符号:**

```
"%d %s" % (int_number, word)
"%d / %d = %f" % (10, 5, 10 / 5)
```

**关于那些 Python 2 语句，有一些事情需要注意:**

*   **变量的类型是显式指定的(%d 表示 int，%s 表示 string，%f 表示 float)**
*   **字符串语句和变量是分开编写的，并排放在一起**
*   **风格本身和 Python 2 印花很像；没有括号，插入变量的%符号等**

**Python 3 现在使用所谓的*字符串格式化程序*，它包含在 Python 字符串类`str.format()`中。字符串格式化程序允许您通过位置格式化将字符串中的元素连接在一起。**

```
"{} {}".format(int_number, word)
"{} / {} = {}".format(10, 5, 10 / 5)
```

**有了新的`format()`函数，根本不需要指定类型，一切都写在一个干净的语句中。你所要做的就是确保这些变量是有序的，一切都会顺利进行。**

# **(5)返回可迭代对象而不是列表**

**在 Python 3 中，许多内置函数现在返回一个*迭代器*而不是一个列表。这种变化的主要原因是迭代器通常比列表更有效地消耗内存。**

**当你使用迭代器时，元素是在需要的基础上*创建并存储在内存中*。这意味着，如果你必须使用迭代器创建 10 亿个浮点数，你只能一次一个地将它们存储在内存中。如果你想创建一个有 10 亿浮点数的列表，你必须确保你有足够的内存，因为数据会被一次性存储在内存中。**

**在许多情况下，使用迭代器和列表的行为是一样的。这方面的一个例子是当循环通过`range()`函数的输出时。**

```
# Python 2 range() function returns a list
for i in range(10):
    print(i)# Python 3 range() function returns an iterator
for i in range(10):
    print(i)
```

**注意语法是完全相同的，即使当我们使用 Python 3 时，我们有一个迭代器。需要注意的一点是，如果你确实特别想要一个列表，你所要做的就是直接转换类型:`list(range(10))`。**

**Python 3 中一些更常用的返回迭代器而不是列表的函数和方法有:**

*   **zip()**
*   **地图()**
*   **过滤器()**
*   **Python dict()的方法:`.keys()`、`.values(),`、**

# **喜欢学习？**

**在 twitter 上关注我，我会在这里发布所有最新最棒的人工智能、技术和科学！也在 LinkedIn[上与我联系](https://www.linkedin.com/in/georgeseif/)！**