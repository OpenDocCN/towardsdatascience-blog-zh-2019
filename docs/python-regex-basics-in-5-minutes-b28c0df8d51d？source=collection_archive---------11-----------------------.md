# Python 正则表达式基础 5 分钟

> 原文：<https://towardsdatascience.com/python-regex-basics-in-5-minutes-b28c0df8d51d?source=collection_archive---------11----------------------->

## 对 Regex 世界的轶事介绍

![](img/0059313dfddc1358dd6d6ea0affc21e2.png)

当我开始网络抓取时，我经常会有奇怪字符的字符串，如\n\nnn 等等。我用替换功能把它去掉了。我构建了类似. str.replace('文本之墙'，''). str.replace('文本之墙'，' ')的结构。结果是 1)可读性差，2)总是只有一个术语的单独解决方案，以及 3)非常低效的过程。所以我开始使用正则表达式来清理我的数据。在下面的文章中，我想和你分享我的发现和一些技巧。祝阅读愉快。

## 场景 1:从网络搜集的数据中提取信息

Python 提供了用于处理正则表达式的`re`库。让我们导入它并创建一个字符串:

```
import re
line = "This is a phone number: (061) — 535555\. And this is another: 01–4657289, 9846012345"
```

`re`有四种不同的功能:`findall, search, split`和`sub`。我们将从`findall`开始学习基础知识。在我们的字符串中，显然有三个不同的电话号码，我们为我们的营销部门丢弃了它们(该字符串是来自 [stackoverflow](https://stackoverflow.com/questions/44229314/regex-scrap-phone-number) 的真实世界的例子)。我们的同事不需要整个字符串，而是需要一个列表，每个数字分开。第一步，我们尝试从字符串中提取每个数字:

```
foundNum = re.findall('\d', line)
print("The numbers found are: ", foundNum)
```

输出:

```
The numbers found are:  ['0', '6', '1', '5', '3', '5', '5', '5', '5', '0', '1', '4', '6', '5', '7', '2', '8', '9', '9', '8', '4', '6', '0', '1', '2', '3', '4', '5']
```

如你所见,`findall`有两个参数。第一项`\d`是我们要找的表达式。在这种情况下，数字。第二项是包含我们感兴趣的模式的字符串。嗯，我们的市场部不能用这个打电话给任何客户，所以我们必须寻找不同的方法。

```
foundNum = re.findall('\(\d', line)
print("The numbers found are: ", foundNum)
```

输出:

```
The numbers found are:  ['(0']
```

现在我们寻找括号和数字的组合。有一个匹配是第一个电话号码的开头。如果我们得出这个结果，营销部门将不会再向我们要求任何东西…好的，下一次尝试:

```
foundNum = re.findall("\(*\d*\d", line)
print("The phone numbers found are: ", foundNum)
```

输出:

```
The phone numbers found are:  ['(061) - 535555', '01 - 4657289', '9846012345']
```

现在我们有了完美的解决方案。我们在表达式之间添加了一个列表，字符串必须以数字开头和结尾。术语`[- \d()]`表示字符“-”、“”、数字或括号必须在数字之间。完美！我们的市场部对此会很高兴，可以打电话给我们的客户。快乐结局！

## 场景 2:清理和匿名化客户数据

我们在刮削项目中的出色表现之后，刚刚放了一天假。然而，在我们早上查看邮件后，我们注意到了另一个请求。我们销售部的同事有一些包含括号中表达式的字符串。出于隐私原因，括号内的术语必须删除。好了，我们已经学了很多，我们开始吧:

```
string_df = '500€ (Mr. Data) Product Category 1'
re.findall(r"\([a-zA-Z .]*\)", string_df)
```

输出:

```
['(Mr. Meier)']
```

好极了。这很容易。我们刚刚检查了一个以括号开始和结束的字符串，并包含了一个包含字母字符和一个点的列表。当我们开始写一封关于我们的好结果的邮件时，我们注意到我们不需要搜索这个术语。相反，我们必须删除这个表达式。我们后退一步，引入一个新函数`sub`。`re.sub`需要三个参数。第一个参数是我们要寻找的表达式，第二个参数是我们要用来替换旧参数的术语，最后一个参数是我们要使用的字符串:

```
re.sub(r"\([a-zA-Z .]*\)", "-XY-", string_df)
```

我们用表达式“-XY-”替换了这个名称。现在，销售部门可以将数据发送给我们的仪表板供应商。又一个幸福的结局。

## 场景 3:将字符串分成几部分

我们开始对正则表达式感到舒服。如果我们称自己为专家，那就有点言过其实了，但我们为每个问题都找到了解决方案，不是吗？在我们帮助了其他部门的同事后，我们专注于自己的工作。我们收到了新的数据进行探索性的数据分析。像往常一样，我们查看数据帧的第一行。第二列看起来很奇怪:

```
strange_cell = '48 Mr.Data 57'
```

根据我们以前的经验，我们得出第一个数字代表 Data 先生购买的单位数量。第二个数字告诉我们客户 id。我们需要在三个独立的栏目中提供这些信息，但是如何提供呢？Regex 提供了另一个名为`split`的函数:

```
re.split(" ", strange_cell)
```

输出:

```
['48', 'Mr.Data', '57']
```

这正是我们要找的。有了这么多快乐的结局，你会认为你在看一部迪斯尼电影，对吗？

## 简短的课程

在漫长而成功的一天后，我们收到了市场部的另一封邮件，他们要求修改我们的第一个任务:他们只需要字符串中的第一个电话号码，因为这是业务电话号码:

```
line = "This is a phone number: (061) — 535555\. And this is another: 01–4657289, 9846012345"
```

使用`searchall`功能，我们考虑每一场比赛。一个简单的解决方法是，我们只给出列表的第一个元素:

```
foundNum = re.findall(r"\(*\d[- \d()]*\d", line)[0]
print("The phone numbers found are: ", foundNum)
```

输出:

```
The phone numbers found are:  (061) - 535555
```

另一个更好的解决方案是使用第四个正则表达式函数`search`，它返回字符串中的第一个匹配:

```
foundNum = re.search(r"\(*\d[- \d()]*\d", line)
print("The phone numbers found are: ", foundNum)
```

输出:

```
The phone numbers found are:  <re.Match object; span=(24, 38), match='(061) - 535555'>
```

## 结论

我希望你喜欢阅读并理解为什么`Regex`对数据科学家如此重要。在数据清理和数据转换过程中，修改字符串是一种强有力的方法。如果你处理数据，你需要处理和转换字符串的技能。`Regex`是完美的解决方案。对于不同表达的概述，我推荐以下页面:

[](https://www.w3schools.com/python/python_regex.asp) [## Python 正则表达式

### 正则表达式是构成搜索模式的一系列字符。正则表达式可以用来检查是否…

www.w3schools.com](https://www.w3schools.com/python/python_regex.asp)  [## 正则表达式 HOWTO - Python 2.7.17 文档

### 该模块是在 Python 1.5 中添加的，提供了 Perl 风格的正则表达式模式。Python 的早期版本…

docs.python.org](https://docs.python.org/2/howto/regex.html) 

[如果您喜欢中级和高级数据科学，并且还没有注册，请随时使用我的推荐链接加入社区。](https://medium.com/@droste.benedikt/membership)