# RTest:R 包的漂亮测试

> 原文：<https://towardsdatascience.com/rtest-pretty-testing-of-r-packages-50f50b135650?source=collection_archive---------24----------------------->

## specflow 和 cucumber.io 支持非编码人员解释 R 包的测试报告，此外还允许非编码人员创建测试用例。简单 r 包验证的一步。

![](img/c0bffa0b5a8a81e6924a27fe44cd53b7.png)

by startupstockphotos [http://startupstockphotos.com/post/143841899156](http://startupstockphotos.com/post/143841899156)

## 目录

*   [为什么要 RTest？](#5cfa)
*   【RTest 有什么特别之处？
*   [RTest](#618a)的测试实现示例。
*   [延伸阅读](#85b2)

# 为什么是 RTest？

![](img/2b83d447148314320344db3789890321.png)

在 R 中测试似乎很简单。从使用`usethis::test_name("name")`开始，然后在[*test that*](http://testthat.r-lib.org/)*中用类似`expect_equal`的功能编写测试代码。网上可以找到很多教程，甚至有一整本书都是关于“[测试 R 代码](https://www.amazon.de/gp/product/1498763650/ref=as_li_tl?ie=UTF8&camp=1638&creative=6742&creativeASIN=1498763650&linkCode=as2&tag=zappingseb-21&linkId=ca15df0c474acdcf98c8750db741c4e3">Testing R Code (Chapman &amp; Hall/Crc the R)</a><img src="//ir-de.amazon-adsystem.com/e/ir?t=zappingseb-21&l=am2&o=3&a=1498763650)”的。可悲的是，这不是我能走的路。正如我几次提到的，我在一个受到严格监管的 T21 环境中工作。在这样的环境中，你的测试不仅会被编码员检查，也会被不会编码的人检查。你的一些测试甚至会由不会编码的人来编写。像 specflow 或 cucumber 这样的东西会真正帮助他们编写这样的测试。但是这些在 r 中是不存在的。另外，这些人不能阅读命令行测试报告。你可以训练他们这样做，但我们认为为我们和他们提供一个漂亮的测试环境更容易，这个环境叫做 **RTest** 。*

*如果你想了解更多关于开发这样一个环境的原因，你可以阅读文章:[为什么我们需要对一种编程语言进行人类可读的测试。](https://medium.com/datadriveninvestor/why-do-we-need-human-readable-tests-for-a-programming-language-1786d552f450)*

# *RTest 有什么特别之处？*

*为了解释我们将哪些特性放入 [*RTest*](https://github.com/zappingseb/RTest) 中，我将开始描述一个基本的测试工作流程。*

*1 测试代码从写代码开始。你的 R 包将包含函数、类和方法。应对这些进行测试。*

*2 编写测试现在主要包括这样的调用:*

```
*my_function(x,y){sums_up(x,y) return(z)}x=3
y=4
z=7stopifnot(my_function(x,y)==z)*
```

*很容易看出，如果你的函数`my_function`不能将两个值相加，你的测试将会失败。您将创建一组这样的测试，并将它们存储在您的包的一个单独的文件夹中，通常称为`tests`。*

*3 之后你可以运行所有这样的测试。您可以在`tests`文件夹中包含一个脚本，或者使用 *testthat* 并运行`testthat::test_dir()`。*

*4 如果你的一个测试失败了，脚本会停止并告诉你哪个测试在控制台中失败了。这描述了下图所示的 4 个步骤。*

*![](img/20775c4a910dc4666e7b689599a86f11.png)*

*RTest 现在的特别之处在于两个主要步骤。*

1.  *测试的定义*
2.  *测试执行的[报告](#b317)*

*对于测试的**定义**，我们决定使用 XML。为什么是 XML？XML 不仅仅比纯 R 代码更容易阅读，它还有一个特性，叫做 XSD；“XML 架构定义”。我们创建的每个 XML 测试用例都可以立即对照开发人员设计的模式进行检查。它也可以与我们自己的`Rtest.xsd`进行核对。这意味着测试人员可以在执行测试用例之前仔细检查它们。这为我们节省了大量的时间，并且给所有的测试用例一个固定的结构。*

***报告**是用 HTML 实现的。这是因为 HTML 自带了许多用于报告的特性。它允许测试结果着色，链接到测试用例并包含图像。 *RTest* 和*test*之间的 HTML 报告的主要区别在于，RTest 报告每个应该执行的测试，而不仅仅是失败的测试。测试报告还将包括由函数调用创建的值和作为参考给出的值。读者可以看到比较是否正确。这样，测试报告包含的信息比*测试和*控制台日志包含的信息多得多。*

# *使用 RTest 的测试实现示例*

*请注意，整个示例存储在一个 [github gist](https://gist.github.com/zappingseb/0f5dabe94c7d284bc543469c50a4213c) 中。如果你喜欢这个例子，请列出要点。*

1.  *给定一个对两列求和的**函数**:*

```
*my_function <- function(data = data.frame(x=c(1,2),y=c(1,2))){stopifnot(dim(data)[2]==2)data[,"sum"] <- apply(data,1,function(x){sum(x)})return(data)}*
```

*2.我们希望有一个成功的和一个不成功的测试。两者在 XML 文件中都有三个部分:*

```
*<params><reference><testspec>*
```

*`params`账户输入参数*

*`reference`为输出数据帧*

*`testspec`测试是否静默运行，容差是多少*

*对于成功的测试，我们的测试应该是这样的:*

*您可以立即看到 RTest 的一个特殊功能。它允许为多个测试使用数据集，我们将这些数据集存储在`input-data`标签中。这可以节省文件中的空间。这里将使用数据集`test01`。此外，可以为每个测试给出测试描述。对于存储在 XML 中的每个 data.frame，列的类型可以在`col-defs`中给出。这些都是数字。*

*这里给出了`input-data`:*

*这是一个数据帧，其中 *x* 列只携带 1，而 *y* 列只携带 2。该测试应创建一个数据帧，每行的 sum 列为 3。*

*我们可以通过改变`reference`标签让测试失败，而不是在`sum`列中只有 3，我们可以添加一个 3.5 让测试失败。整个测试用例可以在有 90 行的 [github gist 中找到](https://gist.github.com/zappingseb/0f5dabe94c7d284bc543469c50a4213c)。*

*3.测试用例的**执行**只是一行代码。您应该在 XML 文件目录中有您的工作目录，并且应该在全局环境中定义`my_function`。*

```
*RTest.execute(getwd(),"RTest_medium.xml")*
```

*4.**测试报告**现在包含一个成功测试和一个失败测试。两者都将被可视化:*

*![](img/be301029551426f3ebf64ff06924ce6b.png)*

*General test outcome in RTest test report*

*所有测试都有附加信息。对于失败的测试，我们将总和设置为 3.5 而不是 3。它在表的末尾报告:*

*![](img/6890250662fa3fab4a5790861544482b.png)*

*example of a failed data frame comparison in RTest*

*此外，报告还包含测试运行环境的信息:*

*![](img/071f4f854b6a6d30d0856feb6f80a702.png)*

*System information for an RTest test report*

*就是这样。现在你可以用 RTest 测试任何包。*

# *进一步阅读*

*   *[RTest github 知识库](https://github.com/zappingseb/RTest)*
*   *[RTest 文档网站](https://zappingseb.github.io/RTest/articles/RTest.html)*
*   *为什么我们需要对编程语言进行人类可读的测试？*
*   *作者的 LinkedIn p 年龄*