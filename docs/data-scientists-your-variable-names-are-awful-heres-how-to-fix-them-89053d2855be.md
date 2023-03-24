# 数据科学家:你的变量名很糟糕。以下是修复它们的方法。

> 原文：<https://towardsdatascience.com/data-scientists-your-variable-names-are-awful-heres-how-to-fix-them-89053d2855be?source=collection_archive---------2----------------------->

![](img/64e37511eefcb8df60eda1b42a94d53d.png)

Wading your way through data science code is like hacking through a jungle. [(Source)](https://www.pexels.com/photo/inside-forest-photography-904788/)

## 一种大大提高代码质量的简单方法

快速，下面的代码是做什么的？

```
for i in range(n):
    for j in range(m):
        for k in range(l): 
            temp_value = X[i][j][k] * 12.5
            new_array[i][j][k] = temp_value + 150
```

这是不可能的，对吗？如果你试图修改或调试这段代码，你会不知所措，除非你能读懂作者的想法。即使你是作者，在编写这段代码几天后，你也不会知道它做了什么，因为使用了*无用的变量名*和 [*【神奇的】数字*](https://en.wikipedia.org/wiki/Magic_number_(programming)#Unnamed_numerical_constants) 。

使用数据科学代码时，我经常看到类似上面(或者更糟)的例子:带有变量名如`X, y, xs, x1, x2, tp, tn, clf, reg, xi, yi, ii`和无数未命名常量值的代码。坦率地说，数据科学家(包括我自己)很不擅长给变量命名，甚至不知道如何给它们命名。

随着我从为一次性分析编写面向研究的[数据科学代码](https://github.com/WillKoehrsen/Data-Analysis)成长为生产级代码(在 [Cortex Building Intel](https://cortexintel.com) )，我不得不通过*抛弃来自数据科学书籍、课程和实验室的*实践来改进我的编程。可以部署的机器学习代码与数据科学家被教授的编程方式之间存在许多差异，但我们将从两个具有重大影响的常见问题开始:

*   **无用/混乱/模糊的变量名**
*   **未命名的“神奇”常数数字**

这两个问题都导致了数据科学研究(或 Kaggle 项目)和生产机器学习系统之间的脱节。是的，你可以在运行一次的 Jupyter 笔记本*中摆脱它们，但是当你有任务关键的机器学习管道每天运行数百次而没有错误时，你必须编写 [*可读和可理解的*](https://softwareengineering.stackexchange.com/questions/162923/what-defines-code-readability) 代码。幸运的是，我们数据科学家可以采用来自软件工程的最佳实践，包括我们将在本文中讨论的那些。*

**注意:**我关注 Python，因为它是目前为止[行业数据科学中使用最广泛的语言。Python 中的](https://insights.stackoverflow.com/survey/2019)(详见[此处](https://stackoverflow.com/questions/159720/what-is-the-naming-convention-in-python-for-variable-and-function-names)):

*   变量/函数名为`lower_case`和`separated_with_underscores`
*   命名常量在`ALL_CAPITAL_LETTERS`中
*   班级在`CamelCase`

# 命名变量

在命名变量时，有三个基本概念需要记住:

1.  **变量名必须描述变量所代表的信息。变量名应该用文字明确地告诉你变量代表什么。**
2.  你的代码被阅读的次数会比它被编写的次数多。优先考虑你的代码有多容易理解，而不是写得有多快。
3.  采用标准的命名惯例，这样你就可以做出一个全局决定，而不是多个局部决定。

这在实践中是什么样子的？让我们来看看对变量名的一些改进:

*   `X`和`y`。如果你已经看过几百次了，你就知道它们是特性和目标，但是对于阅读你的代码的其他开发人员来说，这可能并不明显。相反，使用描述这些变量代表什么的名称，例如`house_features`和`house_prices`。
*   `value`。值代表什么？可能是一个`velocity_mph`、`customers_served`、`efficiency`、`revenue_total`。像`value`这样的名字没有告诉你变量的用途，很容易混淆。
*   `temp`。即使你只是使用一个变量作为临时值存储，也要给它一个有意义的名字。也许这是一个需要转换单位的值，所以在这种情况下，要明确:

```
# Don't do this
temp = get_house_price_in_usd(house_sqft, house_room_count)
final_value = temp * usd_to_aud_conversion_rate# Do this instead
house_price_in_usd = get_house_price_in_usd(house_sqft, 
                                            house_room_count)
house_price_in_aud = house_price_in_usd * usd_to_aud_conversion_rate
```

*   如果你使用像`usd, aud, mph, kwh, sqft`这样的缩写，确保你提前建立这些缩写。与团队中的其他人就常用缩写达成一致，并写下来。然后，在代码审查中，确保执行这些书面标准。
*   `tp`、`tn`、`fp`、`fn`:避免机器学习特定缩写。这些值代表`true_positives`、`true_negatives`、`false_positives`和`false_negatives`，所以要明确。除了难以理解之外，较短的变量名可能会输入错误。当你指的是`tn`的时候用`tp`太容易了，所以把整个描述写出来。
*   上面是一个优先考虑*代码的易读性*而不是你能写多快的例子。阅读、理解、测试、修改和调试写得不好的代码要比写得好的代码花费更长的时间。总的来说，通过尝试更快地编写代码——使用更短的变量名——你实际上会*增加*你程序的开发时间！如果你不相信我，回到你 6 个月前写的一些代码，并尝试修改它。如果你发现自己试图破译你的代码，这表明你应该集中精力在更好的命名约定。
*   `xs`和`ys`。这些通常用于绘图，在这种情况下，这些值代表`x_coordinates`和`y_coordinates`。然而，我已经看到这些名称用于许多其他任务，因此*通过使用描述变量*用途的特定名称来避免混淆，例如`times`和`distances`或`temperatures`和`energy_in_kwh`。

## 什么导致了不好的变量名？

命名变量的大多数问题源于

*   希望变量名简短
*   将公式直接翻译成代码

关于第一点，虽然像 Fortran 这样的语言[确实限制了变量名的长度(6 个字符)，但是现代编程语言没有限制，所以不要觉得被迫使用人为的缩写。也不要使用过长的变量名，但是如果你不得不偏向某一方，那么就以可读性为目标。](https://web.stanford.edu/class/me200c/tutorial_77/05_variables.html)

关于第二点，当你写方程或使用模型时——这是学校忘记强调的一点——记住字母或输入代表真实世界的值！

让我们看一个既犯错误又如何改正的例子。假设我们有一个多项式方程，可以从模型中找到房子的价格。您可能想直接用代码编写数学公式:

![](img/a6b33df0257cf8678745709ad2b4d663.png)

```
temp = m1 * x1 + m2 * (x2 ** 2)
final = temp + b
```

这是看起来像是机器为机器写的代码。虽然计算机最终会运行你的代码，但它会被人类阅读更多次，所以要写人类能理解的代码！

要做到这一点，我们需要考虑的不是公式本身——如何建模——而是正在建模的真实世界的对象——什么是什么。让我们写出完整的方程(这是一个很好的测试，看看你是否理解模型):

![](img/f7df536f4a265b753688dd2b4231ba6a.png)

```
house_price = price_per_room * rooms + \
              price_per_floor_squared * (floors ** 2)
house_price = house_price + expected_mean_house_price
```

如果你在给你的变量命名时有困难，这意味着你对模型或者你的代码不够了解。我们编写代码来解决现实世界的问题，我们需要理解我们的模型试图捕捉什么。描述性变量名让你在比公式更高的抽象层次上工作，帮助你专注于问题领域。

# 其他考虑

命名变量要记住的重要一点是一致性计数。与变量名保持一致意味着你花在命名上的时间更少，而花在解决问题上的时间更多。当您将聚合添加到变量名时，这一点很重要。

## 变量名中的聚合

所以你已经掌握了使用描述性名称的基本思想，将`xs`改为`distances`、`e`改为`efficiency`、`v`改为`velocity`。现在，当你取平均速度时会发生什么？这应该是`average_velocity`、`velocity_mean`还是`velocity_average`？两个步骤可以解决这个问题:

1.  首先，确定常见的缩写:`avg`表示平均值，`max`表示最大值，`std`表示标准差等等。确保所有团队成员都同意并记下这些内容。
2.  把缩写放在名字的末尾。这将最相关的信息，即由变量描述的实体，放在了开头。

遵循这些规则，您的聚合变量集可能是`velocity_avg`、`distance_avg`、`velocity_min`和`distance_max`。规则二。是一种个人选择，如果你不同意，没关系，只要你坚持应用你选择的规则。

当你有一个代表物品数量的变量时，一个棘手的问题就出现了。你可能很想使用`building_num`，但这是指建筑物的总数，还是某一特定建筑物的具体指数？为了避免歧义，使用`building_count`指代建筑物的总数，使用`building_index`指代特定的建筑物。你可以把它应用到其他问题上，比如`item_count`和`item_index`。如果你不喜欢`count`，那么`item_total`也是比`num`更好的选择。这种方法*解决了歧义*并保持了将聚合放在名字末尾的一致性*。*

## 循环索引

由于某些不幸的原因，典型的循环变量变成了`i`、`j`和`k`。这可能是数据科学中比任何其他实践更多错误和挫折的原因。将无信息的变量名与嵌套循环结合起来(我见过嵌套循环包括使用`ii`、`jj`，甚至`iii`)，你就有了不可读、易错代码的完美配方。这可能有争议，但是我从不使用`i`或任何其他单个字母作为循环变量，而是选择描述我正在迭代的内容，例如

```
for building_index in range(building_count):
   ....
```

或者

```
for row_index in range(row_count):
    for column_index in range(column_count):
        ....
```

这在你有嵌套循环时特别有用，这样你就不必记得`i`是代表`row`还是`column`或者是`j`还是`k`。您希望将您的精神资源用于计算如何创建最佳模型，而不是试图计算数组索引的特定顺序。

(在 Python 中，如果你没有使用循环变量，那么使用`_`作为占位符。这样，您就不会对是否使用索引感到困惑。)

## 更多要避免的名字

*   避免在变量名中使用数字
*   避免英语中常见的拼写错误
*   避免使用带有不明确字符的名称
*   避免使用意思相似的名字
*   避免在名字中使用缩写
*   避免听起来相似的名字

所有这些都坚持优先考虑读时可理解性而不是写时便利性的原则。编码主要是一种与其他程序员交流的方法，所以在理解你的计算机程序方面给你的团队成员一些帮助。

# 永远不要使用神奇的数字

一个[幻数](https://en.wikipedia.org/wiki/Magic_number_(programming))是一个没有变量名的常量值。我看到这些用于转换单位、改变时间间隔或添加偏移量等任务:

```
final_value = unconverted_value * 1.61final_quantity = quantity / 60value_with_offset = value + 150
```

(这些变量名都很差！)

幻数是错误和混乱的主要来源，因为:

*   只有一个人，作者，知道它们代表什么
*   更改该值需要查找所有使用该值的位置，并手动键入新值

不使用幻数，我们可以定义一个转换函数，接受未转换的值和转换率作为*参数*:

```
def convert_usd_to_aud(price_in_usd,            
                       aud_to_usd_conversion_rate):
    price_in_aus = price_in_usd * usd_to_aud_conversion_rate
```

如果我们在一个程序的许多函数中使用转换率，我们可以在一个位置定义一个名为常量的[:](http://wiki.c2.com/?NamedConstants)

```
USD_TO_AUD_CONVERSION_RATE = 1.61price_in_aud = price_in_usd * USD_TO_AUD_CONVERSION_RATE
```

(在我们开始项目之前，我们应该与团队的其他成员确定`usd` =美元，`aud` =澳元。记住标准！)

这是另一个例子:

```
# Conversion function approach
def get_revolution_count(minutes_elapsed,                       
                         revolutions_per_minute):
    revolution_count = minutes_elapsed * revolutions_per_minute # Named constant approach
REVOLUTIONS_PER_MINUTE = 60revolution_count = minutes_elapsed * REVOLUTIONS_PER_MINUTE
```

使用在一个地方定义的`NAMED_CONSTANT`使得改变值更加容易和一致。如果转换率发生变化，您不需要搜索整个代码库来更改所有出现的代码，因为它只在一个位置定义。它也告诉任何阅读你的代码的人这个常量代表什么。如果名字描述了*参数所代表的*，函数参数也是一个可接受的解决方案。

作为幻数风险的真实例子，在大学期间，我参与了一个建筑能源数据的研究项目，这些数据最初每隔 15 分钟出现一次。没有人过多考虑这种变化的可能性，我们用神奇的数字 15(或 96 表示每日观察次数)编写了数百个函数。在我们开始以 5 分钟和 1 分钟的间隔获取数据之前，这一切都很好。我们花了几周的时间修改所有的函数来接受区间的一个参数，但即使如此，我们仍然要与几个月来使用幻数所导致的错误作斗争。

真实世界的数据习惯于在你身上变化——货币之间的兑换率每分钟都在波动——硬编码成特定的值意味着你将不得不花费大量的时间来重写代码和修复错误。编程中没有“魔法”的位置，即使在数据科学中也是如此。

## 标准和惯例的重要性

采用标准的好处是，它们让你做出一个全球性的决定，而不是许多地方性的决定。不要在每次命名变量的时候都选择聚合的位置，而是在项目开始的时候做一个决定，并在整个过程中一致地应用它。目标是花更少的时间在与数据科学无关的问题上:命名、格式、风格——而花更多的时间解决重要问题(比如使用[机器学习解决气候变化](https://arxiv.org/abs/1906.05433))。

如果你习惯独自工作，可能很难看到采用标准的好处。然而，即使是独自工作，你也可以练习定义自己的约定，并坚持使用它们。你仍然会从更少的小决策中获益，当你不可避免地需要在团队中发展时，这是一个很好的实践。每当一个项目中有一个以上的程序员时，标准就成了必须的！

你可能不同意我在这篇文章中所做的一些选择，这没关系！*采用一套一致的标准比精确选择使用多少空格或变量名的最大长度更重要。*关键是不要在偶然的困难上花费太多时间，而是要专注于本质的困难。(大卫·布鲁克斯[有一篇关于我们如何从解决软件工程中的偶然问题到专注于本质问题的优秀论文](http://faculty.salisbury.edu/~xswang/Research/Papers/SERelated/no-silver-bullet.pdf)。

# 结论

记住我们所学的，我们现在可以回到我们开始的初始代码:

```
for i in range(n):
    for j in range(m):
        for k in range(l): 
            temp_value = X[i][j][k] * 12.5
            new_array[i][j][k] = temp_value + 150
```

把它修好。我们将使用描述性的变量名和命名的常量。

现在我们可以看到，这段代码正在对一个数组中的像素值进行归一化，并添加一个常量偏移量来创建一个新的数组(忽略实现的低效率！).当我们将这些代码交给我们的同事时，他们将能够理解并修改这些代码。此外，当我们回过头来测试代码并修复错误时，我们会准确地知道我们在做什么。

这个话题很无聊吗？也许这有点枯燥，但是如果你花时间阅读软件工程，你会意识到区分最好的程序员的是重复实践诸如好的变量名、保持例程简短、测试每一行代码、重构等平凡的技术。这些是将代码从研究阶段转移到生产阶段所需要的技术，一旦你到了那里，你会发现让你的模型影响现实生活中的决策一点也不无聊。

在本文中，我们介绍了一些改进变量名的方法。

## 需要记住的要点

1.  变量名应该描述变量所代表的实体。
2.  优先考虑你的代码易于理解的程度，而不是你写代码的速度。
3.  在整个项目中使用一致的标准，以最小化小决策的认知负担。

## 具体要点

*   使用描述性变量名称
*   使用函数参数或命名常数，而不是“神奇”的数字
*   不要使用机器学习专用的缩写
*   用变量名描述一个方程或模型代表什么
*   将聚合放在变量名的末尾
*   用`item_count`代替`num`
*   使用描述性循环索引代替`i`、`j`、`k`。
*   在整个项目中采用命名和格式约定

(如果你不同意我的一些具体建议，那没关系。更重要的是，你使用一种标准的方法来命名变量，而不是教条地使用确切的约定！)

我们可以对数据科学代码进行许多其他更改，以使其达到生产级别(我们甚至没有讨论函数名！).我很快会有更多关于这个主题的文章，但与此同时，请查看[“从代码完成开始构建软件的注意事项”](/notes-on-software-construction-from-code-complete-8d2a8a959c69)。要深入了解软件工程最佳实践，请阅读史蒂夫·麦康奈尔的 [*代码全集*](https://www.microsoftpressstore.com/store/code-complete-9780735619678) 。

为了发挥其真正的潜力，数据科学将需要使用允许我们构建健壮的软件产品的标准。对我们来说幸运的是，软件工程师已经想出了这些最佳实践的大部分，并在无数的书籍和文章中详细介绍了它们。现在该由我们来阅读和执行它们了。

我写关于数据科学的文章，欢迎建设性的评论或反馈。你可以在推特上找到我。如果在提高底线的同时帮助世界对你有吸引力，那就去看看 Cortex Building Intelligence 的[职位空缺。](https://get.cortexintel.com/careers/)我们帮助世界上一些最大的办公楼节省了数十万美元的能源成本，同时减少了碳足迹。