# [NLP]基础:理解正则表达式

> 原文：<https://towardsdatascience.com/nlp-basics-understanding-regular-expressions-fc7c7746bc70?source=collection_archive---------8----------------------->

## 你唯一需要的向导

![](img/d69aa5e7c36d48ee90952f7d1607c6b1.png)

Photo by [travelnow.or.crylater](https://unsplash.com/@travelnow_or_crylater?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

当我开始学习自然语言处理时，正则表达式真的感觉像一门外语。我努力理解语法，花了几个小时写一个正则表达式来返回我正在寻找的输入。很自然的，我尽可能的远离他们。

但事实是，作为一名数据科学家，总有一天你会接触到正则表达式。它们构成了自然语言处理中基本技术的一部分，学习它们将使你成为一个更高效的程序员。

所以是时候坐下来开始了。把学习正则表达式想象成一堂语法课:它们很痛苦，起初看起来不可理解，但一旦你理解并学会了，你会感到如释重负。我向你保证，到头来也没那么难。

*注意，我将在本文中使用的编程软件是 R.*

# 展开正则表达式

简而言之，正则表达式是给函数的“指令”,告诉它如何匹配或替换一组字符串。

让我们从正则表达式的一些基础开始，这是一些你应该知道的基本语法。

**括号** []用于指定字符的析取。例如，使用括号放 W 或 W 允许我返回大写的 W 或小写的 W。

```
/[wW]oodchuck/ --> Woodchuck or woodchuck 
/[abc]/ --> ‘a’, ‘b’, or ‘c’
/[1234567890]/ --> any digit
```

如果你添加一个**破折号**，你就指定了一个范围。例如，将 A-Z 放在括号中允许 R 返回一个大写字母的所有匹配。

```
/[A-Z]/ → machtes an upper case letter
/[a-z]/ → matches a lower case letter
/[0–9]/ → matches a single digit
```

^可以用于否定，或者仅仅表示^.

```
/[ˆA-Z]/ --> not an upper case letter
/[ˆSs]/ --> neither ‘S’ nor ‘s’
/[ˆ\.]/ --> not a period
/[eˆ]/ --> either ‘e’ or ‘ˆ’
/aˆb/ --> the pattern ‘aˆb’
```

**问号**？标记上一个表达式的可选性。例如，将一个。在土拨鼠结束时，返回土拨鼠(不带 s)和土拨鼠(带 s)的结果。

```
/woodchucks?/ --> woodchuck or woodchucks
/colou?r/ --> color or colour
```

可以用**期**。指定两个表达式之间的任意字符。例如，输入 beg.n 将返回 begin 或 begin 这样的单词。

```
/beg.n/ --> Match any character between beg and n (e.g. begin, begun)
```

***或+** 用户允许您添加 1 个或多个先前字符。

```
 oo*h! → 0 or more of a previous character (e.g. ooh!, ooooh!)
o+h! → 1 or more of a previous character (e.g. ooh!, ooooooh!)
baa+ → baa, baaaa, baaaaaa, baaaaaaa
```

**锚**用于断言关于字符串或匹配过程的一些东西。因此，它们不是用于特定的单词或字符，而是帮助进行更一般的查询，正如您在下面的示例中所看到的。

```
 . → any character except a new line
\\w → any word character
\\W → anything but a word character
\\d → any digit character
\\D → anything but a digit character
\\b → a word boundary
\\B → anything but a word boundary
\\s → any space character
\\S → anything but a space character
```

**POSIX 字符类别**有助于匹配特定的字符类别，如数字。换句话说，它使一个小的字符序列匹配一个更大的字符集。

```
 [[:alpha:]] → alphabetic characters
[[:digit:]] → digits
[[:punct:]] → punctuation
[[:space:]] → space, tab, newline etc.
[[:lower:]] → lowercase alphatic characters
[[:upper:]] → upper case alphabetic characters
```

# strsplit()、grep()和 gsub()

这是行动开始的地方。在处理字符串时，很可能必须使用命令 strsplit()、grep()和 gsub()来激活您希望 R 返回给您的输入。

## **strsplit(x，split)**

下面的例子展示了一种使用 strsplit 来拆分句子中的单词的方法，在这个例子中是破折号“…”内的所有单词。

```
richard3 <- “now is the winter of our discontent”
strsplit(richard3, “ “) # the second argument specifies the space
```

## **grep(pattern，x，ignore.case = FALSE，value = FALSE)**

grep 允许你“抓取”你想要的单词或单词集，这取决于你设置的匹配模式。例如，在下面的代码中，我让 R 返回包含单词“both”的字符串。

```
grep.ex <- c(“some like Python”, “some prefer R”, “some like both”)grep(“both”, grep.ex) # in numerical form
grep(“both”, grep.ex, value=TRUE) #prints the character string itself
```

## gsub(pattern，replacement，x，ignore.case= FALSE)

例如，gsub 允许你用一个词替换另一个词。在这种情况下，我选择在下面的字符串中用超人代替罗密欧。

```
r_and_j <- “O Romeo, Romeo, wherefore art thou Romeo?”
gsub(“Romeo”, “Superman”, r_and_j, ignore.case = FALSE)
```

# 应用了正则表达式

让我们开始将这些命令应用于正则表达式。下面我给你看几个例子。

## 1.grep(模式，x，ignore.case =假，值=假)

```
dollar <- c(“I paid $15 for this book.”, “they received $50,000 in grant money”, “two dollars”)
```

注意，在上面的例子中，你有三个不同的句子，其中两个使用了$符号，一个使用了单词“美元”。仅使用$来匹配您的模式将会返回所有三个句子。然而，如果你在$前加上\\你可以指定你只需要使用$符号的句子。

```
grep(“$”, dollar) 
grep(“\\$”, dollar)
```

下面还有几个例子，说明如何使用 grep 来匹配单词“ashes”、“shark”、“bash”的正则表达式:

```
# matches all three vector elements
grep(“sh”, c(“ashes”, “shark”, “bash”), value=TRUE) # matches only “shark”
grep(“^sh”, c(“ashes”, “shark”, “bash”), value=TRUE) # matches only “bash”
grep(“sh$”, c(“ashes”, “shark”, “bash”), value=TRUE) 
```

关于“失态”、“擦破”、“粉笔”三个词:

```
quant.ex <- c(“gaffe”, “chafe”, “chalk”, “encyclopaedia”, “encyclopedia”)# Searching for one or more occurences of f and we want to see the value not only the index (that’s why we put value = TRUE)
grep(“f+”, quant.ex, value=TRUE) # one or two “f”
grep(“f{1,2}”, quant.ex, value=TRUE) # at least one “f”
grep(“f{1,}”, quant.ex, value=TRUE)
```

## 2.gsub(pattern，replacement，x，ignore.case= FALSE)

在下面的例子中，你可以用 gsub 替换句子中的单词。

```
ex.sentence <- “Act 3, scene 1\. To be, or not to be, that is the Question:”
```

如果您还记得在本文第一部分中学习的正则表达式，您应该能够猜出 R 将返回哪种输入。否则，我会将它添加到下面的代码中，以便您更好地理解使用正则表达式返回所需结果的所有不同方式。

```
gsub(“.”, “*”, ex.sentence, ignore.case=TRUE, perl=TRUE)
[1] "**********************************************************"gsub(“\\w”, “*”, ex.sentence, ignore.case=TRUE, perl=TRUE)
[1] "*** *, ***** *. ** **, ** *** ** **, **** ** *** ********:"gsub(“\\W”, “*”, ex.sentence, ignore.case=TRUE, perl=TRUE)
[1] "Act*3**scene*1**To*be**or*not*to*be**that*is*the*Question*"gsub(“\\d”, “*”, ex.sentence, ignore.case=TRUE, perl=TRUE)
[1] "Act *, scene *. To be, or not to be, that is the Question:"gsub(“\\D”, “*”, ex.sentence, ignore.case=TRUE, perl=TRUE)
[1] "****3********1********************************************"gsub(“\\b”, “*”, ex.sentence, ignore.case=TRUE, perl=TRUE)
[1] "*Act* *3*, *scene* *1*. *To* *be*, *or* *not* *to* *be*, *that* *is* *the* *Question*:"gsub(“\\B”, “*”, ex.sentence, ignore.case=TRUE, perl=TRUE)
[1] "A*c*t 3,* s*c*e*n*e 1.* T*o b*e,* o*r n*o*t t*o b*e,* t*h*a*t i*s t*h*e Q*u*e*s*t*i*o*n:*"gsub(“\\s”, “*”, ex.sentence, ignore.case=TRUE, perl=TRUE)
[1] "Act*3,*scene*1.*To*be,*or*not*to*be,*that*is*the*Question:"gsub(“\\S”, “*”, ex.sentence, ignore.case=TRUE, perl=TRUE)
[1] "*** ** ***** ** ** *** ** *** ** *** **** ** *** *********"
```

否则，试着猜测最后一个问题的答案:

```
letters.digits <- “a1 b2 c3 d4 e5 f6 g7 h8 i9”
gsub(“(\\w)(\\d)”, “\\2\\1”, letters.digits)
```

就是这样！我希望你喜欢这篇文章，并且我已经设法使正则表达式对你来说更容易理解。

*我经常写关于数据科学和自然语言处理的文章。在* [*Twitter*](https://twitter.com/celine_vdr) *或*[*Medium*](https://medium.com/@celine.vdr)*上关注我，查看更多类似的文章或简单地更新下一篇文章。* ***感谢阅读！***

PS:最后一个的答案是“1a 2b 3c 4d 5e 6f 7g 8h 9i”。换句话说，您已经使用正则表达式重写了所有的字母和数字。很酷不是吗？