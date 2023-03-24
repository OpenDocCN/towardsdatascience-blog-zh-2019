# 对我来说 __init__ 是什么？

> 原文：<https://towardsdatascience.com/whats-init-for-me-d70a312da583?source=collection_archive---------2----------------------->

## 设计 Python 包导入模式

我最近有几次关于 Python 打包的谈话，特别是关于构造`import`语句来访问包的各种模块。这是我在[组织](https://deppen8.github.io/posts/2018/09/python-packaging/) `[leiap](https://deppen8.github.io/posts/2018/09/python-packaging/)` [包](https://deppen8.github.io/posts/2018/09/python-packaging/)时不得不做的大量调查和实验。尽管如此，我还没有看到各种场景中最佳实践的好指南，所以我想在这里分享一下我的想法。

# 导入模式

![](img/6c306c02dcb79adc1f85f56eabd1a369.png)

Photo by [Mick Haupt](https://unsplash.com/@rocinante_11?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

设计用户如何与模块交互的关键是软件包的`__init__.py`文件。这将定义用`import`语句引入名称空间的内容。

# 模块

出于几个原因，将代码分成更小的模块通常是个好主意。主要地，模块可以包含与特定一致主题相关的所有代码(例如，所有 I/O 功能)，而不会被与完全不同的东西(例如，绘图)相关的代码弄得混乱。由于这个原因，大班得到一个专用模块是很常见的(例如，`geopandas`中的`geodataframe.py`)。其次，将代码划分成适当的逻辑单元会使其更容易阅读和理解。

然而，对于*开发者*来说好的模块结构对于*用户*来说可能是也可能不是好的模块结构。在某些情况下，用户可能不需要知道包下面有各种模块。在其他情况下，可能有充分的理由让用户明确地只要求他们需要的模块。这就是我在这里想要探索的:不同的用例是什么，它们需要包开发人员采用什么方法。

# 一个示例包

Python 包有多种结构，但是让我们在这里创建一个简单的演示包，我们可以在所有的例子中使用它。

```
/src
    /example_pkg
        __init__.py
        foo.py
        bar.py
        baz.py
    setup.py
    README.md
    LICENSE
```

它由三个模块组成:`foo.py`、`bar.py`和`baz.py`，每个模块都有一个单独的函数，打印该函数所在模块的名称。

## foo.py

```
def foo_func():
    print(‘this is a foo function’)
```

## bar.py

```
def bar_func():
    print(‘this is a bar function’)
```

## baz.py

```
def baz_func():
    print(‘this is a baz function’)
```

# 你的杂货店守则

现在是承认谈论`import`语句和包结构可能很难理解的时候了，尤其是在文本中。为了让事情更清楚，让我们把 Python 包想象成一个杂货店，把用户想象成购物者。作为开发商，你是商店的所有者和管理者。你的工作是想出如何建立你的商店，让你为你的顾客提供最好的服务。您的`__init__.py`文件的结构将决定设置。下面，我将介绍建立该文件的三种可选方法:普通商店、便利商店和在线商店。

# **综合商店**

![](img/3da4e788ee8cd77185f98ca7a7283e6e.png)

Photo by [Mick Haupt](https://unsplash.com/@rocinante_11?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/visual/9b8309e3-2c1c-4c5b-b161-fd7dc6e587ca?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

在这个场景中，用户可以在`import example_pkg`上立即访问一切。在他们的代码中，他们只需要键入包名和他们想要的类、函数或其他对象，而不管它位于源代码的哪个模块中。

这个场景就像一个旧时代的普通商店。顾客一进门，就能看到所有商品毫不费力地摆放在商店里的箱子和货架上。

## 在幕后

```
# __init__.py
from .foo import *
from .bar import *
from .baz import *
```

## 用户实现

```
import example_pkgexample_pkg.foo_func()
example_pkg.bar_func()
example_pkg.baz_func()
```

## **优点**

*   例如，用户不需要知道模块名或记住哪个功能在哪个模块中。他们只需要包名和函数名。在综合商店里，所有的产品都以最小的标识陈列着。顾客不需要知道去哪个通道。
*   导入顶级包后，用户可以访问任何功能。所有的东西都陈列出来了。
*   Tab 补全只需`example_pkg.<TAB>`就能给你一切。制表就像杂货店的杂货商，他知道所有东西的确切位置，并且乐于提供帮助。
*   当新特性被添加到模块中时，你不需要更新任何`import`语句；它们将自动包含在内。在一般的商店里，没有花哨的招牌可以换。只要在架子上放一个新的项目。

## **缺点**

*   要求所有函数和类必须唯一命名(即在`foo`和`bar`模块中都没有名为`save()`的函数)。你不想把苹果放在两个不同的箱子里，让你的顾客感到困惑。
*   如果包很大，它会给名称空间增加很多东西，并且(取决于很多因素)会减慢速度。一家普通商店可能有许多个人顾客可能不想要的小杂物。这可能会让您的客户不知所措。
*   需要更多的努力和警惕来让一些元素远离用户。例如，您可能需要使用下划线来防止函数导入(例如，`_function_name()`)。大多数普通商店没有一个大的储藏区来存放扫帚和拖把之类的东西；这些项目对客户是可见的。即使他们不太可能拿起扫帚开始扫你的地板，你也可能不希望他们这样做。在这种情况下，您必须采取额外的措施来隐藏这些供应品。

## **建议**

*   当很难预测典型用户的工作流程时使用(例如像`pandas`或`numpy`这样的通用包)。这是一般商店的“一般”部分。
*   当用户可能经常在不同模块之间来回切换时使用(例如，`leiap`包)
*   当函数名和类名描述性很强且容易记忆，而指定模块名不会提高可读性时使用。如果你的产品是你熟悉的东西，比如水果和蔬菜，你就不需要很多标牌；顾客会很容易发现问题。
*   仅使用几个模块。如果有许多模块，新用户在文档中找到他们想要的功能会更加困难。如果你的综合商店太大，顾客将无法找到他们想要的东西。
*   在可能频繁添加或移除对象时使用。在普通商店添加和移除产品很容易，不会打扰顾客。

## **众所周知的例子**

*   `pandas`
*   `numpy`(增加了复杂性)
*   `seaborn`

# **便利店**

![](img/47b7f006d95e2d32aa4cffc426bcc0cb.png)

Photo by Caio Resende from Pexels

到目前为止，最容易阅读和理解的是一般商店场景的变体，我称之为便利店。代替`from .module import *`，您可以在`__init__.py`中用`from .module import func`指定导入什么。

便利商店和普通商店有许多共同的特点。它的商品选择相对有限，可以随时更换，麻烦最小。顾客不需要很多标牌就能找到他们需要的东西，因为大多数商品都很容易看到。最大的区别是便利店的订单多一点。空盒子、扫帚和拖把都放在顾客看不见的地方，货架上只有待售的商品。

## 在幕后

```
# __init__.py
from .foo import foo_func
from .bar import bar_func
from .baz import baz_func
```

## 用户实现

```
import example_pkgexample_pkg.foo_func()
example_pkg.bar_func()
example_pkg.baz_func()
```

## **优点**

分享普通商店的所有优势，并增加:

*   更容易控制哪些对象对用户可用

## **缺点**

*   如果有许多功能多样的模块，结果会非常混乱。像普通商店一样，过于杂乱的便利店会让顾客难以浏览。
*   当新特性被添加到一个模块时(即新的类或函数)，它们也必须被显式地添加到`__init__.py`文件中。现代 ide 可以帮助检测遗漏的导入，但是仍然很容易忘记。你的便利店有一些最小的标志和价格标签。当你改变书架上的东西时，你必须记得更新这些。

## **建议**

我将在综合商店的建议中增加以下内容:

*   当您的模块或多或少由一个`Class`(例如`from geopandas.geodataframe import GeoDataFrame`)组成时，这尤其有用
*   当有少量对象要导入时使用
*   当您的对象有明确的名称时使用
*   当您确切知道用户需要哪些对象，不需要哪些对象时，请使用
*   当您不希望频繁添加大量需要导入的新模块和对象时，请使用。

## **众所周知的例子**

*   `geopandas`

# **网上购物**

![](img/bbd36ac35a618f2cee96d329a4b375ce.png)

Photo by [Pickawood](https://unsplash.com/@pickawood?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

任何在网上买过杂货的人都知道，订购正确的产品可能需要顾客付出一些努力。你必须搜索产品、选择品牌、选择想要的尺寸等等。然而，所有这些步骤都可以让你从一个几乎无限的仓库里买到你想要的东西。

在 Python 包的情况下，在某些情况下，避免简单地导入整个包的便利性，而是迫使用户更清楚地知道导入的是什么部分，可能会更谨慎。这使得作为开发人员的您可以在不影响用户的情况下在包中包含更多的内容。

## 在幕后

```
# __init__.py
import example_pkg.foo
import example_pkg.bar
import example_pkg.baz
```

## 用户实现

在这种情况下，用户可以采用(至少)三种不同的方法。

```
import example_pkgexample_pkg.foo.foo_func()
example_pkg.bar.bar_func()
example_pkg.bar.baz_func()
```

或者

```
from example_pkg import foo, bar, bazfoo.foo_func()
bar.bar_func()
baz.baz_func()
```

或者

```
import example_pkg.foo as ex_foo
import example_pkg.bar as ex_bar
import example_pkg.baz as ex_bazex_foo.foo_func()
ex_bar.bar_func()
ex_baz.baz_func()
```

## **优点**

*   简化了`__init__.py`文件。仅在添加新模块时需要更新。更新你的网上商店相对容易。您只需更改产品数据库中的设置。
*   它是灵活的。它可用于仅导入用户需要的内容或导入所有内容。网上商店的顾客可以只搜索他们想要或需要的东西。当你需要的只是一个苹果时，就没有必要再去翻“水果”箱了。但是如果他们真的想要“水果”箱里的所有东西，他们也可以得到。
*   别名可以清理长的 package.module 规范(如`import matplotlib.pyplot as plt`)。虽然网上购物一开始会很痛苦，但如果你把购物清单留到以后再用，购物会快很多。
*   可以有多个同名的对象(例如在`foo`和`bar`模块中都被称为`save()`的函数)

## **缺点**

*   一些导入方法会使代码更难阅读。例如，`foo.foo_func()`并不表示`foo`来自哪个包。
*   可读性最强的方法(`import example_pkg`，没有别名)可能会产生很长的代码块(例如`example_pkg.foo.foo_func()`)，使事情变得混乱。
*   用户可能很难找到所有可能的功能。在你的网上杂货店，购物者很难看到所有可能的商品。

## **建议**

*   当您有一个复杂的模块系列时使用，其中的大部分任何一个用户都不会需要。
*   当`import example_pkg`导入大量对象并且可能很慢时使用。
*   当您可以为不同类型的用户定义非常清晰的工作流时使用。
*   当您希望用户能够很好地浏览您的文档时使用。

## **例题**

*   `matplotlib` *
*   `scikit-learn` *
*   `bokeh` *
*   `scipy` *

*这些包实际上在它们的`__init__.py`文件中使用了不同方法的组合。我在这里包括它们是因为对于用户来说，它们通常是按菜单使用的(例如，`import matplotlib.pyplot as plt`或`import scipy.stats.kde`)。

# 结论

我概述的三个场景当然不是 Python 包的唯一可能的结构，但是我希望它们涵盖了任何从博客中了解到这一点的人可能会考虑的大多数情况。最后，我将回到我之前说过的一点:对于*开发者*来说好的模块结构对于*用户*来说可能是也可能不是好的模块结构。无论你做什么决定，不要忘记站在用户的角度考虑问题，因为那个用户很可能就是你。