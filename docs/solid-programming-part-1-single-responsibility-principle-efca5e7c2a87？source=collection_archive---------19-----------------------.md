# 可靠编程:单一责任原则

> 原文：<https://towardsdatascience.com/solid-programming-part-1-single-responsibility-principle-efca5e7c2a87?source=collection_archive---------19----------------------->

![](img/aa98c2b697a5287a4eba8b604501ecf1.png)

坚实的原则是软件工程中最有价值的原则之一。它们允许编写干净、可伸缩且易于扩展的代码。在这一系列的文章中，我将解释每一个原则是什么，以及为什么应用它是重要的。

有些人认为 SOLID 只适用于 OOP，而实际上它的大部分原理可以用于任何范式。

实线中的“s”代表**单责任**。很多程序员新手的错误就是写复杂的函数和做很多事情的类。然而，根据单一责任原则，一个模块、一个类或一个函数必须只做一件事。换句话说，他们必须只有一个责任。这样代码更健壮，更容易调试、阅读和重用。

让我们来看看这个函数，它将一个单词和一个文件路径作为参数，并返回该单词在文本中出现的次数与单词总数的比率。

```
def percentage_of_word(search, file):
    search = search.lower()
    content = open(file, "r").read()
    words = content.split()
    number_of_words = len(words)
    occurrences = 0
    for word in words:
        if word.lower() == search:
            occurrences += 1
    return occurrences/number_of_words
```

该代码在一个函数中做许多事情:读取文件，计算总字数，单词出现的次数，然后返回比率。

如果我们想遵循单一责任原则，我们可以用下面的代码来代替它:

```
def read_localfile(file):
    '''Read file'''

    return open(file, "r").read()

def number_of_words(content):
    '''Count number of words in a file'''

    return len(content.split())

def count_word_occurrences(word, content):
    '''Count number of word occurrences in a file'''

    counter = 0
    for e in content.split():
        if word.lower() == e.lower():
            counter += 1
    return counter

def percentage_of_word(word, content):
    '''Calculate ratio of number of word occurrences to number of    
       all words in a text'''

    total_words = number_of_words(content)
    word_occurrences = count_word_occurrences(word, content)
    return word_occurrences/total_words

def percentage_of_word_in_localfile(word, file):
    '''Calculate ratio of number of word occurrences to number
       of all words in a text file'''

    content = read_localfile(file)
    return percentage_of_word(word, content)
```

现在每个函数只做一件事。第一个读取文件。第二个计算总字数。有一个函数可以计算一个单词在文本中出现的次数。另一个函数计算单词出现的次数与单词总数的比率。如果要得到这个比率，我们更愿意传递文件路径而不是文本作为参数，有一个专门的函数。

那么，以这种方式重组代码，我们能得到什么呢？

*   这些功能很容易**重用**，并且可以根据任务进行混合，从而使代码很容易**扩展**。例如，如果我们想要计算包含在 AWS S3 桶而不是本地文件中的文本中的单词的频率，我们只需要编写一个新函数`read_s3`，其余的代码无需修改就可以工作。
*   代号是**干**。没有重复的代码，所以如果我们需要修改其中一个函数，我们只需要在一个地方做。
*   代码**干净、有条理，非常容易阅读和理解**。
*   我们可以分别为每个函数编写**测试**，这样就更容易调试代码了。你可以在这里检查这些功能[的测试。](https://github.com/AnnaLara/SOLID_blogposts/blob/master/tests.py)

# GitHub 中的代码

这篇文章中的代码和测试可以在 GitHub 中找到:
[https://github.com/AnnaLara/SOLID_blogposts](https://github.com/AnnaLara/SOLID_blogposts)

*原载于 2019 年 9 月 21 日*[*https://dev . to*](https://dev.to/annalara/solid-programming-part-1-single-responsibility-principle-1ki6)*。*