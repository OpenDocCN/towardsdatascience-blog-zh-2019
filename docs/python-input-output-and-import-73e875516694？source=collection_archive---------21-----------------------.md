# Python 输入、输出和导入

> 原文：<https://towardsdatascience.com/python-input-output-and-import-73e875516694?source=collection_archive---------21----------------------->

在本教程中，让我们了解 python 中使用的输入和输出内置函数，我们还将学习如何导入库并在程序中使用它们。

![](img/cb6b1b724ca359f30702d3770610150e.png)

Image Credits: [Data Flair](https://data-flair.training/blogs/python-built-in-functions/)

在开始之前，让我们了解什么是内置函数？

> **作为高级语言的一部分而提供的、可以由简单的** [**引用**](https://www.yourdictionary.com/reference) **来执行的任何功能，无论是否有** [**参数**](https://www.yourdictionary.com/arguments)**——信用:**[**your dictionary**](https://www.yourdictionary.com/built-in-function)

python 中有许多内置函数。记住每个内置函数的名称和语法是非常不可能的。下面是 python 中所有内置函数的文档，按字母顺序由[python.org](https://docs.python.org/3/library/functions.html)—**提供，我称之为内置函数的备忘单，请看下面:**

 [## 内置函数- Python 3.8.0 文档

### Python 解释器内置了许多随时可用的函数和类型。它们被列出…

docs.python.org](https://docs.python.org/3/library/functions.html) 

`input()`和`print()`内置函数是执行标准输入和输出操作最常用的函数之一。python 中没有`output()`函数，而是使用`print()`来执行标准的输出操作。您也可以在我的 [GitHub 资源库](https://github.com/Tanu-N-Prabhu/Python/blob/master/Python_Input%2C_Output_and_Import.ipynb)上找到本教程的文档，如下所示:

[](https://github.com/Tanu-N-Prabhu/Python/blob/master/Python_Input%2C_Output_and_Import.ipynb) [## Tanu-N-Prabhu/Python

### 您现在不能执行该操作。您使用另一个选项卡或窗口登录。您在另一个选项卡上注销，或者…

github.com](https://github.com/Tanu-N-Prabhu/Python/blob/master/Python_Input%2C_Output_and_Import.ipynb) 

# 使用打印功能的输出操作

python 中的 print 函数通常用于在屏幕上打印输出。顾名思义**打印**意为*要打印的东西***在哪里？***屏幕上。*现在你不用每次都写打印函数的整个主体(内容)，我个人甚至不知道打印函数的主体是什么样子的。你所要做的就是通过传递参数来调用函数。你在打印函数里面写的东西都叫做参数。让我们看看如何把东西打印到屏幕上。

```
variable = "Welcome to Python Tutorials"
print(variable)**Welcome to Python Tutorials**
```

就像我说的，变量是和打印函数一起传递的参数，你不需要写打印函数的主体，你只需要调用打印函数。print 函数的实际语法如下所示:

```
**print(value, …, sep=’ ‘, end=’\n’, file=sys.stdout, flush=False)**Prints the values to a stream, or to sys.stdout by default.     Optional keyword arguments: **file:**  a file-like object (stream); defaults to the current sys.stdout.     
**sep: **  string inserted between values, default a space.    
**end: **  string appended after the last value, default a newline.     **flush:** whether to forcibly flush the stream.
```

你也可以试试这个，只需调用 **help(print)** 帮助功能就会为你提供你需要的帮助(咄；))

现在让我们看看如何在打印函数中传递多个参数，结果会是什么？

```
variable = "Welcome to Python Tutorials"
variable2 = ",This is a good place to learn programming"print(variable, variable2)**Welcome to Python Tutorials ,This is a good place to learn programming**
```

您可以在 print 函数中传递尽可能多的参数，但是要确保将变量彼此分开。

让我们在打印函数中使用“ **sep** ”(分隔符)并查看输出。

```
print(1, 2, 3, 4, 5, sep="--->")**1--->2--->3--->4--->5**
```

“ **end** ”参数用于将一个字符附加到我们在打印函数内部传递的字符串的末尾。

```
print(1, 2, 3, 4, 5, end=" This is the end")**1 2 3 4 5 This is the end**
```

## 格式化输出

可能在某些情况下需要格式化输出，在这种情况下，我们可以使用`str.format()`方法。

```
variable = "Python"
variable2 = "Programming"print("I love {0} and {1}".format(variable,variable2))**I love Python and Programming**
```

此外，format()方法可以做很好的工作，例如向整数字符串添加逗号分隔符，如下所示:

```
print(format(1234567, ",d"))**1,234,567**
```

类似地，我们可以使用“ **%** ”操作符格式化输出，如下所示:

```
number = 12.3456
print("The value of x is %1.2f" %number)**The value of x is 12.35**print("The value of x is %1.5f" %number)**The value of x is 12.34560**
```

# Python 输入函数

过去，我们在主程序中硬编码变量的值。但有时我们需要听取用户的意见。这可以通过使用输入功能来实现。

输入功能顾名思义就是接受用户的输入。输入函数的语法是:

```
input(prompt = ‘ ’)
```

这里的提示是屏幕上显示的提示。

```
number = input("Enter the number of your choice: ")
print("The number that you have entered is: %s" %number)**Enter the number of your choice: 100 
The number that you have entered is: 100**
```

这里的提示是“**输入你选择的数字**”，这里你可能会奇怪为什么我输入整数的时候用了“ **%s** ”。这是因为在 Python 中所有的变量都存储为字符串，要显式地对其进行类型转换，可以使用 int(variable_name)或 float(variable_name) 等等。去试试，然后告诉我。

# Python 导入

有时，我们需要从不同的模块中导入一些库(模块是包含定义和语句的 python 文件)。在这些情况下，我们可以通过使用`import`关键字和模块名来导入这些方法或模块。

考虑一个例子，认为你想看“圆周率”的值是多少，你能做的只是而不是做(22/7，或者背 3.1423……我不知道更进一步)，只要导入数学模块，调用数学方法，你的工作就完成了。

```
import math
print(math.pi)**3.141592653589793**
```

同样使用`from`关键字，我们可以从模块中访问特定的属性。例如:

```
from math import pi
print(pi)**3.141592653589793**
```

更像是“从这个模块导入这个函数、方法或者属性”。

“Python 输入、输出和导入”教程到此结束。希望你喜欢。如果你有任何意见或建议，请在下面的评论区告诉我。直到那时再见！！！。