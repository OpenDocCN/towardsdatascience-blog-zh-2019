# 关于正则表达式你需要知道的

> 原文：<https://towardsdatascience.com/regular-expressions-in-python-a212b1c73d7f?source=collection_archive---------19----------------------->

## 我的常用备忘单。

# 第一部分:揭开正则表达式的神秘面纱

![](img/4a1e302fe3a598f37963591ca3c059b5.png)

Source: [XKCD](https://xkcd.com/1171/)

## 什么是正则表达式，为什么你应该知道它们？

正则表达式是一种用字符序列匹配模式的通用方法。它们定义了一个搜索模式，主要用于字符串的模式匹配或字符串匹配，即“查找和替换”之类的操作。

假设你要搜索一个语料库，并返回其中提到的所有电子邮件地址。对于一个人来说，这是一个非常简单的任务，因为每个电子邮件地址都可以概括为一个字母数字字符串，其中包含一些允许的特殊字符，如。(点)、_(下划线)、-(连字符)，后面跟一个' @ {域名}。' com '。

但是如何将这样的模板传递给计算机呢？电子邮件地址可以是可变长度的，没有固定的数字/特殊字符位置(可以在开头、结尾或中间的任何位置)，这一事实并没有使计算机的任务变得更容易。

这就是正则表达式帮助我们的地方！通过使用它们，我们可以很容易地创建一个通用的模板，它不仅能被计算机理解，还能满足我们所有的限制。

![](img/a9fd2b1e380476588edec3299c44426b.png)

Source: [XKCD](https://xkcd.com/208/)

## 正则表达式这个术语是从哪里来的？

*正则表达式*这个术语来源于数学和计算机科学理论，在这里它反映了数学表达式的一个特质叫做 ***正则性*** 。最早的 grep 工具使用的文本模式是数学意义上的正则表达式。虽然这个名字已经被记住，但是现代的 Perl 风格的正则表达式(我们仍然在其他编程语言中使用)在数学意义上根本不是正则表达式。

它们通常被称为 **REs** ，或**正则表达式**，或**正则表达式**模式。

# 如何写正则表达式？

## 基础知识:

**#1** :大部分角色与自己匹配。因此，如果您必须为单词“wubbalubadubdub”编写正则表达式，那么它的正则表达式应该是“wubbalubadubdub”。请记住，REs 是区分大小写的。

但是有一些名为 ***的元字符*** 与它们本身并不匹配，而是表示一些不寻常的东西应该匹配，或者它们通过重复它们或改变它们的意思来影响 re 的其他部分。这些元字符使正则表达式成为强大的工具。

**#2** : **复读机*，+，{}** 。很多时候你会遇到这样的情况，你不知道一个字符会重复多少次，或者是否会重复。复读机让我们能够处理这种情况。

**星号(*):** 表示前面的字符出现 0 次或更多次。前面字符的出现次数没有上限。

```
In ‘a*b’, ‘a’ precedes the *, which means ‘a’ can occur 0/0+ times. So it will match to ‘b’, ‘ab’, ‘aab’, ‘aaab’, ‘aa..(infinite a’s)..b’.
```

**加号(+):** 与*相同，但暗示前面的字符至少出现一次。

```
‘a+b’ will match to ‘ab’, ‘aab’, ‘aaab’, ‘aaa…aab’ but not just ‘b’.
```

**花括号{}** :用于定义前一个字符出现的范围。

```
‘a{3},b’: will match ‘aaab’ only. 
‘a{min,}b’: restricting minimum occurrences of ‘a’, no upper limit restrictions. 
‘a{3,}b’ will match ‘aaab’, ‘aaaab’, ‘aaa…aab’.
‘a{min, max}b’: ‘a’ should occur at least min times and at most max times. ‘a{3,5}b’ will match ‘aaab’, ‘aaaab’, ‘aaaaab’.
```

**#3 通配符(。)**:知道字符串中字符的顺序并不总是可行的。在这些情况下，通配符非常有用。
这意味着任何字符都可以在序列中占据它的位置。

```
 ‘.’ will match all strings with just one character.
‘.*’ will match all possible strings of any length.
‘a.b’ will match ‘aab’, ‘abb’, ‘acb’, ‘adb’, …. ‘a b’ [a(space)b], ‘a/b’ and so on. Basically, any sequence of length 3 starting with ‘a’ and ending with ‘b’.
```

**#4 可选字符(？):**有时候一个词可能有多种变体。就像“color”(美国英语)和“color”(英国英语)是同一个单词的不同拼法，都是正确的，唯一的区别是在英国版本中多了一个“u”。
那个“？”意味着前一个字符可能出现在最终字符串中，也可能不出现。在某种程度上，'？'意味着出现 0 或 1 次。

```
‘colou?r’ will match both ‘color’ and ‘colour’.
‘docx?’ will match both ‘doc’ and ‘docx’.
```

Caret(^):暗示一个字符串的第一个字符。

```
‘^a.*’ will match all strings starting with ‘a’.
‘^a{2}.*’ will match all strings starting with ‘aa’.
```

**#6 美元($):** 它暗示一个字符串的最后一个字符。

```
‘.*b$’ will match all strings ending with a ‘b’.
```

**#7 字符类([])** :通常情况下，对于一个字符串中的一个特定位置，会有不止一个可能的字符。为了容纳所有可能的字符，我们使用字符类。它们指定了您希望匹配的一组字符。字符可以单独列出，也可以通过给出两个字符并用'-'分隔来表示一个字符范围。

```
‘[abc]’: will match ‘a’, ‘b’, or ‘c’. 
‘[^abc]’:*Negation* will match any character except ‘a’, ‘b’, and ‘c’. **Note** -- Not to be confused with caret where ^ implies begins with. If inside a character class ^ implies negation.
Character range: ‘[a-zA-Z]’ will match any character from a-z or A-Z. 
```

`\d`匹配任意十进制数字；这相当于类[0–9]。
`\D` 匹配任何非数字字符；这相当于类[^0–9].
`\s` 匹配任何空白字符；这相当于类[\t\n\r\f\v]。
`\S` 匹配任何非空白字符；这相当于 class[^\t\n\r\f\v].
`\w` 匹配任何字母数字字符；这相当于类[a-zA-Z0–9]。
`\W` 匹配任何非字母数字字符；这相当于类[^a-za-z0–9].

**#9 分组字符:**正则表达式的一组不同符号可以组合在一起，作为一个单元，表现为一个块，为此，需要将正则表达式括在括号( )中。

```
‘(ab)+’ will match ‘ab’, ‘abab’,’abab…’.
‘^(th).*” will match all string starting with ‘th’.
```

**#10 竖线(| ):** 匹配由(|)分隔的任何一个元素。

```
th(e|is|at) will match words - the, this and that.
```

**#11 转义符(\):** 如果你想匹配一个元字符本身，也就是匹配' * '和' * '而不使用通配符，该怎么办？
你可以在这个字符前加一个反斜杠(\)来实现。这将允许使用特殊字符而不调用它们的特殊含义。

```
\d+[\+-x\*]\d+ will match patterns like "2+2" and "3*9" in          "(2+2)*3*9".
```

# 为电子邮件地址编写正则表达式

为了为任何电子邮件地址编写正则表达式，我们需要记住以下约束:

*   它只能以字母开头。***【^([a-za-z】)***
*   它的长度可变，可以包含任何字母数字字符和“.”, '_', '-'.***(【a-zA-Z0–9 _ \-\。]*)***
*   它应该有一个“@”后跟一个域名。***@***([a-zA-Z0–9 _ \-\。]+)
*   它应该以点号结尾，通常有 2/3 字符长。
    ***。([a-zA-Z]){2，3 } $***

我们最后一个电子邮件地址的正则表达式:
^([a-za-z])([a-za-z0–9_\-\.]*)@([a-zA-Z0–9 _ \-\。]+)\.([a-zA-Z]){2，3}$

# 正则表达式的实际应用

正则表达式在各种各样的文本处理任务中非常有用，更普遍的是在数据不需要是文本的字符串处理中非常有用。常见的应用包括数据验证、数据抓取(尤其是 web 抓取)、数据争论、简单解析、生成语法突出显示系统以及许多其他任务。

虽然正则表达式在互联网搜索引擎上很有用，但是根据正则表达式的复杂性和设计，在整个数据库中处理它们会消耗过多的计算机资源。尽管在许多情况下，系统管理员可以在内部运行基于正则表达式的查询，但大多数搜索引擎并不向公众提供正则表达式支持。

# 进一步阅读:

[本教程的第 2 部分](https://medium.com/@ria.kulshrestha16/regular-expressions-in-python-92d09c419cce)到此为止，我们在 python 中通过他们的 *re* 模块使用 REs。还讨论了编写正则表达式时面临的一些复杂问题，即“反冲瘟疫”。那里见！👋

# 参考资料:

*   [维基百科:正则表达式](https://en.wikipedia.org/wiki/Regular_expression)
*   [极客 forgeeks](https://www.geeksforgeeks.org/write-regular-expressions/)
*   [Python 文档](https://docs.python.org/3/howto/regex.html)
*   Steven Levithan，Jan Goyvaerts 的正则表达式食谱

## 我写的其他文章，我认为你可能会喜欢:D

*   [8 分钟内学会 Git！](/git-help-all-2d0bb0c31483)

> 我很高兴你坚持到了这篇文章的结尾。*🎉*
> 我希望你的阅读体验和我写这篇文章时一样丰富。*💖*
> 
> 请点击这里查看我的其他文章[。](https://medium.com/@ria.kulshrestha16)
> 
> 如果你想联系我，我会选择 Twitter。