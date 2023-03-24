# 正则表达式

> 原文：<https://towardsdatascience.com/regular-expressions-f3ee2213b085?source=collection_archive---------22----------------------->

![](img/141ae36622e21513d6379674becd2af9.png)

Stanley Cole Kleene, 1978\. Photograph by Konrad Jacobs, Erlangen. Copyright: MFO, [https://opc.mfo.de/detail?](https://opc.mfo.de/detail?photo_id=2122)

在计算机科学、数理逻辑和语言学中，正则表达式指的是提取和处理文本模式的技术集合。正则表达式有时被称为*正则表达式*，它是一组精确的字符，用来对目标文本中的字符组模式进行操作。例如，用户输入到微软 Word 中熟悉的`find`和`find and replace`功能中的文本作为正则表达式；查询的文本序列的每个文档实例被定位、突出显示、计数，然后被单独或成批处理。

在计算中，复杂的正则表达式技术提供了比那些在`find`中使用的更复杂的功能。事实上，模式匹配正则表达式的理论是由美国数学家斯坦利·科尔·克莱尼——阿隆佐·邱奇的学生——在 20 世纪 50 年代初首次系统化的，这比现代编程的出现早了几十年。然而，20 世纪 70 年代 Unix 和 Unix 邻近操作系统的发展使得正则表达式无处不在。实现正则表达式的两个主要语法标准之一在当代编程语言中仍然存在:继承自 POSIX 的标准和继承自 Perl 的标准。

Perl 是 Python 内置`re`模块的祖先。使用`re`，Python 程序员可以编写以各种方式作用于字符串的字符串，而不是简单地将输入字符串中的单个字符与文本中的其他字符串进行元素匹配。其中，正则表达式搜索中最常用的技术之一叫做*通配符*。

```
>import re
>
>string = 'This is a wildcard.'
>x = re.findall('wil.*', string)
>
>print(x)['wildcard.']
```

通配符——有时称为占位符——在上面的代码中由字符对`.*`表示。这个`.*`指示解释器在目标文本中搜索它左边的任何字符，然后它右边的任何字符可以跟随它。在本例中，通配符的功能类似于 Bash 中文件搜索时的 tab 补全，它允许用户只键入文件的前几个字母，然后让 autocomplete 处理其余部分。在 SQL 中，通配符只是星号`*`，前面没有`.`。

```
>string = 'This is a wildcard.'
>x = re.findall('wil.', string)
>
>print(x)['wild']
```

如果我们删除`*`但保留`.`，单个通配符将表示在该位置允许任何一个字符。换句话说，`.`将告诉解释器在目标文本中搜索它左边的字符序列，并打印它和紧随其后的下一个字符。正则表达式理论中的单个字符被称为*原子*。添加更多的`.`原子字符将打印序列后面更多的字母。

```
>string = 'This is a wildcard.'
>x = re.findall('wil...', string)
>
>print(x)['wildca']
```

我们也可以在表达式中间使用通配符。

```
>string = 'This is a wildcard.'
>x = re.findall('wil....d', string)
>
>print(x)['wildcard']
```

通配符在正则表达式理论中被称为*元字符*，在`re`中有很多这样的元字符。另一个特别有用的元字符允许我们搜索目标文本*是否以正则表达式的*开头。这个正则表达式用`^`表示。其他元字符包括以`$`、非此即彼`|`结尾，以及特定数量的实例`{}`。

```
>string = 'This is a wildcard.'
>x = re.findall('^This', string)
>
>print(x)['This']
```

除了元字符之外，`re`还有大量的*特殊字符*、*、*，它们允许表达式对模式进行不同类型的检索或操作。这些特殊字符各由`\`表示；这与该字符在 Python 字符串中的用法一致。这些特殊字符之一就是`d`。`d`返回目标文本中所有匹配的整数。

```
>string = '1These2are3not4numbers5.'
>x = re.findall('\d', string)
>
>print(x)['1', '2', '3', '4', '5']
```

与此相反，我们可以通过调用`\D`返回所有非数字字符。

```
>string = '1These2are3not4numbers5.'
>x = re.findall('\D', string)
>
>print(x)['T', 'h', 'e', 's', 'e', 'a', 'r', 'e', 'n', 'o', 't', 'n', 'u', 'm', 'b', 'e', 'r', 's', '.']
```

我们也可以将正则表达式视为集合。这是通过将我们的表达式放在括号`[]`中来实现的。下面，我们在目标文本中搜索表达式中的任何字符，并在每次匹配时返回该字符:

```
>string = 'I am looking for all five vowels.'
>x = re.findall('[aeiou]', string)
>
>print(x)['a', 'o', 'o', 'i', 'o', 'a', 'i', 'e', 'o', 'e']
```

与此相反，如果我们以`^`开始表达式，那么集合的补集——正则表达式中除以外的任何字符——都将被打印出来。

```
>string = 'I am looking for all five vowels.'
>x = re.findall('[^aeiou]', string)
>
>print(x)['I', ' ', 'm', ' ', 'l', 'k', 'n', 'g', ' ', 'f', 'r', ' ', 'l', 'l', ' ', 'f', 'v', ' ', 'v', 'w', 'l', 's', '.']
```

使用连字符时，正则表达式将返回任何没有指定范围的小写字母的字符的匹配。

```
>string = 'I am looking for all five vowels.'
>x = re.findall('[a-l]', string)
>
>print(x)['a', 'l', 'k', 'i', 'g', 'f', 'a', 'l', 'l', 'f', 'i', 'e', 'e', 'l']
```

这里给出的每个例子都使用了`findall()`方法。但是我们也可以`split()`、`sub()`、`search()`、`span()`、`match()`、`group()`。因此，这仅仅是对`re`的一些概念基础的简单介绍。此外，Python 中的正则表达式超越了`re`固有的功能。例如，一个流行的第三方包`regex`提供了更多的正则表达式方法，并且向后兼容原生的`re`。