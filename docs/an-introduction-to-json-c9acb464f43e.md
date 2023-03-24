# JSON 简介

> 原文：<https://towardsdatascience.com/an-introduction-to-json-c9acb464f43e?source=collection_archive---------8----------------------->

## 完全的初学者指南

![](img/734b192f092b3639747f205951c81c89.png)

Created by [Katerina Limpitsouni](https://twitter.com/ninalimpi)

如果你对数据科学或软件工程或任何相关领域感兴趣，你可能会遇到“JSON”这个术语，如果你是新手，你可能会感到困惑。在这篇文章中，我将尝试在没有任何先验知识的情况下介绍 JSON，并用简单的例子解释 JSON 的概念。让我们开始吧。

**JSON 是什么？**

![](img/9d8fd5fc10835f6908b43e8a94beadd4.png)

Created by [Katerina Limpitsouni](https://twitter.com/ninalimpi)

JSON 代表 *JavaScript 对象符号。*不要被行话冲昏了头脑，理解和使用起来其实很简单。正如单词“notation”可能暗示的那样， *JSON 只是一种独立于平台的表示数据的方式*——这只是意味着它有点像数据的 PDF(在不同平台如移动、桌面和 web 上是一样的)。JSON 格式是由[道格拉斯·克洛克福特](https://www.crockford.com/about.html)指定的，文件扩展名是*。json* 。

因此，数据的 PDF 跨平台传输，并保持表示和存储的一致性。如今它被广泛使用，因此在数据科学和软件工程领域至关重要。

**为什么是 JSON？**

![](img/4ce24b4cd4c2f7c266f76fd3ad60c4e4.png)

Created by [Katerina Limpitsouni](https://twitter.com/ninalimpi)

JSON 听起来很酷，但背后的动机或目的是什么？

如上所述，JSON 独立于平台，是每天交换大量数据的计算机之间数据传输格式的主要选择之一。

JSON 的替代品是 XML(可扩展标记语言)，但 JSON 在许多方面更好。虽然两者都是人类可读和机器可读的，但 JSON 更容易阅读，计算机处理起来也更快。此外，JSON 是用 JavaScript 解析器(内置于大多数 web 浏览器中)处理(或解析)的，而 XML 需要一个单独的 XML 解析器。这就是“JavaScirpt”在“JSON”中发挥作用的地方

你可以在这里和这里阅读更多关于[和](https://www.w3schools.com/js/js_json_xml.asp)[的区别。](https://stackoverflow.com/questions/2620270/what-is-the-difference-between-json-and-xml)

**如何 JSON？**

![](img/6b96ff7b0b268e8d240c3423e82e74cf.png)

Created by [Katerina Limpitsouni](https://twitter.com/ninalimpi)

在理解了什么是 JSON 及其背后的基本原理之后，现在可以开始编写一些 JSON 代码了。JSON 的语法与 JavaScript 非常相似，所以如果你以前有过 JavaScript 的经验，应该会很熟悉。

让我们通过一个例子来理解 JSON 的编写。假设你是你所在街区的领导，并且维护着一个所有人的数据库。考虑一个场景，当 Jodhn Appleseed 先生搬到你家附近时。您可能希望存储像他的名、姓、出生日期、婚姻状况等信息。这个就用 JSON 吧！

当你写 JSON 的时候，你本质上是一个撮合者！*是的，真的！*但你匹配的不是人，而是数据。在 JSON 中，数据存储为键-值对——每个数据项都有一个键，通过这个键可以修改、添加或删除数据项。

让我们从添加名字和姓氏开始:

```
{
    "first name":"John",
    "last name":"Appleseed"
}
```

正如您所注意到的，左列中的值是键(“名字”、“姓氏”)，右列中的值是各自的值(“John”、“Appleseed”)。冒号将它们分开。用双引号括起来的值是字符串类型，这基本上意味着它们应该是文本，而不是引用其他内容(例如，数字、文件另一部分中的变量等)。).

注意，在 JSON 中，所有的键都必须是字符串，所以必须用双引号括起来。此外，除了最后一个键-值对，每个键-值对后面都有一个逗号，表示正在记录一个新项目。

现在，让我们加上他的年龄:

```
{
    "first name":"John",
    "last name":"Appleseed",
    "age":30
}
```

请注意，数字 30 没有双引号。直观地说，这种数据的数据类型是*数，*，因此你可以对它们进行数学运算(当你检索信息时)。在 JSON 中，这种数据类型(数字)可以采用任何数值——十进制、整数或任何其他类型。请注意，当我在“Appleseed”下面添加另一个项目时，我是如何在它后面添加一个逗号的。

让我们现在做一些有趣的事情。让我们试着添加他的房子，这将有它的地址，所有者信息和城市。但是我们如何为 John 将这些东西添加到 JSON 文件中呢？像所有者信息这样的东西是房子的属性，而不是 John 的属性，所以直接将这些信息添加到 John 的 JSON 文件中是没有意义的。不用担心，JSON 有一个有趣的数据类型来处理这个问题！

在 JSON 中，值也可以是对象(也是键值对)。这个对象就像另一个 JSON 文件——用花括号括起来，包含键值对，只是它在我们原始的 JSON 文件中，而不是有自己的文件。

```
{
    "first name" : "John",
    "last name" : "Appleseed",
    "age" : 30, 
    "house" : { 
            "address":{
                "house no":22,
                "street":"College Ave East",
                "city":"Singapore",
            },
            "owner":"John Appleseed"
            "market price($)":5000.00
    }
}
```

您可能注意到，房子是一个对象，它包含关键字*地址*、*所有者*和*市场价格*。*地址*中的数据也是一个对象，包含关键字*门牌号*、*街道*和*城市*。因此，可以在对象中嵌套对象，这样可以更清晰地表示数据，如上所示。

现在，让我们添加他的朋友的信息。我们可以通过添加“朋友 1”和名字，“朋友 2”和名字等等来做到这一点，但是这很快就会变得很无聊。JSON 提供了一种有效存储这些信息的数据类型，称为*数组*。它是项目的有序集合，可以是任何数据类型。

假设他有三个朋友:查尔斯、马克和达伦。我们的 JSON 文件现在看起来像这样:

```
{
    "first name":"John",
    "last name":"Appleseed",
    "age":30, 
    "house" : { 
            "address":{
                "house no":22,
                "street":"College Ave East",
                "city":"Singapore",
            },
            "owner":"John Appleseed"
            "market price($)":5000.00
    },
    "friends":[
        "Charles",
        "Mark",
        "Darren"
     ]
}
```

请注意，数组用方括号括起来，除了最后一项，我们将每一项都写在一个新行中，后跟一个逗号。新的一行不是必需的，但是它有助于提高代码的可读性。

最后，让我们加上他的婚姻状况。我们可以做类似于`"married":"yes"`的事情，但是 JSON 为所有二分法选择提供了一种特殊的数据类型:布尔值。它只能有两个值:真或假。直觉上，不可能两者同时。假设约翰是个单身汉。让我们将这最后一条信息添加到我们的文件中！我们的文件现在看起来像这样:

Final JSON File for John Appleseed

并且，您已经熟悉了 JSON。在本文中，我们了解了 JSON 是什么，它为什么有用，以及如何使用 JSON。在此过程中，我们学习了 JSON 数据类型(字符串、数字、数组、对象和布尔)。

上面的文件是 GitHub 的要点，你可以点击这个[链接来下载](https://gist.github.com/raivatshah/ab2bb7dc2e3a2d4937e5fadd7385972e)这个文件，并使用它，添加更多的信息或为新人制作 JSON 文件。你可以检查你的代码是否是有效的 JSON，也可以用这个[工具](https://jsonformatter.curiousconcept.com/)格式化它。

我希望这篇文章能帮助您了解 JSON。回复这个故事，让我知道你的旅程如何。