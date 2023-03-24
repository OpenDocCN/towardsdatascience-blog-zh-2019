# R 中的数据类型

> 原文：<https://towardsdatascience.com/data-types-in-r-8124c3b2afe6?source=collection_archive---------34----------------------->

## 了解 R 中最常见的数据类型

![](img/db4f4d9ed01745c37f94d60011e3c49c.png)

Photo by [Luke Chesser](https://unsplash.com/@lukechesser?utm_source=medium&utm_medium=referral)

# 介绍

T 这篇文章展示了 r 中不同的数据类型。要从统计学的角度了解不同的变量类型，请阅读“[变量类型和示例](https://www.statsandr.com/blog/variable-types-and-examples/)”。

# R 中存在哪些数据类型？

R 中有 6 种最常见的数据类型:

1.  数字的
2.  整数
3.  复杂的
4.  性格；角色；字母
5.  因素
6.  逻辑学的

R 中的数据集通常是这 6 种不同数据类型的组合。下面我们将逐一探讨每种数据类型的更多细节，除了“复杂”数据类型，因为我们关注的是主要的数据类型，这种数据类型在实践中很少使用。

# 数字的

R 中最常见的数据类型是数字。如果值是数字或者值包含小数，则变量或序列将存储为数字数据。例如，默认情况下，以下两个系列存储为数字:

```
# numeric series without decimals
num_data <- c(3, 7, 2)
num_data## [1] 3 7 2class(num_data)## [1] "numeric"# numeric series with decimals
num_data_dec <- c(3.4, 7.1, 2.9)
num_data_dec## [1] 3.4 7.1 2.9class(num_data_dec)## [1] "numeric"# also possible to check the class thanks to str()
str(num_data_dec)##  num [1:3] 3.4 7.1 2.9
```

换句话说，如果你给 R 中的一个对象分配一个或几个数字，默认情况下它将存储为数字(带小数的数字)，除非另外指定。

# 整数

整数数据类型实际上是数值数据的一种特殊情况。整数是没有小数的数字数据。如果您确定您存储的数字永远不会包含小数，则可以使用此选项。例如，假设您对 10 个家庭样本中的孩子数量感兴趣。该变量是一个离散变量(如果您不记得什么是离散变量，请参见[变量类型](https://www.statsandr.com/blog/variable-types-and-examples/)的提示)，并且永远不会有小数。因此，由于使用了`as.integer()`命令，它可以存储为整数数据:

```
children##  [1] 1 3 2 2 4 4 1 1 1 4children <- as.integer(children)
class(children)## [1] "integer"
```

请注意，如果您的变量没有小数，R 会自动将类型设置为整数而不是数字。

# 性格；角色；字母

存储文本时使用数据类型字符，在 r 中称为字符串。在字符格式下存储数据的最简单方法是在文本段周围使用`""`:

```
char <- "some text"
char## [1] "some text"class(char)## [1] "character"
```

如果您想强制将任何类型的数据存储为字符，您可以使用命令`as.character()`来完成:

```
char2 <- as.character(children)
char2##  [1] "1" "3" "2" "2" "4" "4" "1" "1" "1" "4"class(char2)## [1] "character"
```

注意`""`里面的一切都会被认为是字符，不管看起来像不像字符。例如:

```
chars <- c("7.42")
chars## [1] "7.42"class(chars)## [1] "character"
```

此外，只要变量或向量中至少有一个字符值，整个变量或向量都将被视为字符:

```
char_num <- c("text", 1, 3.72, 4)
char_num## [1] "text" "1"    "3.72" "4"class(char_num)## [1] "character"
```

最后但同样重要的是，虽然空格在数字数据中无关紧要，但在字符数据中却很重要:

```
num_space <- c(1)
num_nospace <- c(1)
# is num_space equal to num_nospace?
num_space == num_nospace## [1] TRUEchar_space <- "text "
char_nospace <- "text"
# is char_space equal to char_nospace?
char_space == char_nospace## [1] FALSE
```

从上面的结果可以看出，字符数据中的一个空格(即在`""`中)使它成为 R！

# 因素

因子变量是字符变量的特例，因为它也包含文本。但是，当唯一字符串的数量有限时，会使用因子变量。它通常代表一个[分类变量](https://www.statsandr.com/blog/variable-types-and-examples/)。例如，性别通常只有两个值，“女性”或“男性”(将被视为一个因素变量)，而名字通常有很多可能性(因此将被视为一个字符变量)。要创建因子变量，使用`factor()`功能:

```
gender <- factor(c("female", "female", "male", "female", "male"))
gender## [1] female female male   female male  
## Levels: female male
```

要了解因子变量的不同级别，使用`levels()`:

```
levels(gender)## [1] "female" "male"
```

默认情况下，级别按字母顺序排序。您可以使用`factor()`函数中的参数`levels`对等级重新排序:

```
gender <- factor(gender, levels = c("male", "female"))
levels(gender)## [1] "male"   "female"
```

字符串可以用`as.factor()`转换成因子:

```
text <- c("test1", "test2", "test1", "test1") # create a character vector
class(text) # to know the class## [1] "character"text_factor <- as.factor(text) # transform to factor
class(text_factor) # recheck the class## [1] "factor"
```

字符串已经被转换为因子，如其类型为`factor`的类所示。

# 逻辑学的

逻辑变量是只有两个值的变量；`TRUE`或`FALSE`:

```
value1 <- 7
value2 <- 9# is value1 greater than value2?
greater <- value1 > value2
greater## [1] FALSEclass(greater)## [1] "logical"# is value1 less than or equal to value2?
less <- value1 <= value2
less## [1] TRUEclass(less)## [1] "logical"
```

也可以将逻辑数据转换成数字数据。使用`as.numeric()`命令从逻辑转换为数字后，`FALSE`值等于 0，`TRUE`值等于 1:

```
greater_num <- as.numeric(greater)
sum(greater)## [1] 0less_num <- as.numeric(less)
sum(less)## [1] 1
```

相反，数字数据可以转换为逻辑数据，所有值的`FALSE`等于 0，所有其他值的`TRUE`。

```
x <- 0
as.logical(x)## [1] FALSEy <- 5
as.logical(y)## [1] TRUE
```

感谢阅读。我希望这篇文章能帮助你理解 R 中的基本数据类型及其特殊性。如果您想从统计学的角度了解更多关于不同变量类型的信息，请阅读文章“[变量类型和示例](https://www.statsandr.com/blog/variable-types-and-examples/)”。

和往常一样，如果您有与本文主题相关的问题或建议，请将其添加为评论，以便其他读者可以从讨论中受益。

**相关文章:**

*   [安装和加载 R 包的有效方法](https://www.statsandr.com/blog/an-efficient-way-to-install-and-load-r-packages/)
*   我的数据符合正态分布吗？关于最广泛使用的分布以及如何检验 R 中的正态性的注释
*   [R 中的 Fisher 精确检验:小样本的独立性检验](https://www.statsandr.com/blog/fisher-s-exact-test-in-r-independence-test-for-a-small-sample/)
*   [R 中独立性的卡方检验](https://www.statsandr.com/blog/chi-square-test-of-independence-in-r/)
*   [如何在简历中创建时间线](https://www.statsandr.com/blog/how-to-create-a-timeline-of-your-cv-in-r/)

*原载于 2019 年 12 月 30 日*[*https://statsandr.com*](https://statsandr.com/blog/data-types-in-r/)*。*