# 使用 Cython 可以将 Python 代码的速度提高 30 倍以上

> 原文：<https://towardsdatascience.com/use-cython-to-get-more-than-30x-speedup-on-your-python-code-f6cb337919b6?source=collection_archive---------0----------------------->

![](img/46d4548c74f21127059f5539bed60bc0.png)

Cython will give your Python code super-car speed

> 想获得灵感？快来加入我的 [**超级行情快讯**](https://www.superquotes.co/?utm_source=mediumtech&utm_medium=web&utm_campaign=sharing) 。😎

Python 是社区最喜欢的编程语言！这是迄今为止最容易使用的方法之一，因为代码是以直观、人类可读的方式编写的。

然而，你经常会一遍又一遍地听到对 Python 的相同抱怨，尤其是来自 C 代码大师的抱怨: *Python 很慢。*

他们没有错。

相对于很多其他编程语言，Python *比较慢*。[基准测试游戏](https://benchmarksgame-team.pages.debian.net/benchmarksgame/fastest/gpp-python3.html?source=post_page---------------------------)有一些比较各种编程语言在不同任务上速度的坚实基准。

我以前写过几个不同的方法可以加快速度:

(1)使用[多处理库](/heres-how-you-can-get-a-2-6x-speed-up-on-your-data-pre-processing-with-python-847887e63be5)来使用所有的 CPU 内核

(2)如果您使用 Numpy、Pandas 或 Scikit-Learn，请使用 [Rapids 来加速 GPU](/heres-how-you-can-accelerate-your-data-science-on-gpu-4ecf99db3430) 上的处理。

如果您正在做的事情实际上可以并行化，例如数据预处理或矩阵运算，那就太棒了。

但是如果你的代码是纯 Python 呢？如果你有一个大的 for 循环，而你只有*有*可以使用，并且不能放入一个矩阵中，因为数据必须在*序列*中处理，那该怎么办？有没有办法加速 Python *本身*？

这就是 Cython 加速我们的原始 Python 代码的原因。

# Cython 是什么？

就其核心而言，Cython 是 Python 和 C/C++之间的中间步骤。它允许您编写纯 Python 代码，只需稍加修改，然后直接翻译成 C 代码。

您对 Python 代码所做的唯一调整是向每个变量添加类型信息。通常，我们可以像这样在 Python 中声明一个变量:

```
x = 0.5
```

使用 Cython，我们将为该变量添加一个类型:

```
cdef float x = 0.5
```

这告诉 Cython，我们的变量是浮点型的，就像我们在 c 中做的一样。使用纯 Python，变量的类型是动态确定的。Cython 中类型的显式声明使得到 C 的转换成为可能，因为显式类型声明是必需的+。

安装 Cython 只需要一行 pip:

```
pip install cython
```

# Cython 中的类型

使用 Cython 时，变量和函数有两组不同的类型。

对于变量，我们有:

*   cdef int a、b、c
*   cdef char *s
*   **cdef 浮点型** x = 0.5(单精度)
*   **cdef double** x = 63.4(双精度)
*   **cdef 列表**名称
*   **cdef 词典**进球 _for_each_play
*   **cdef 对象**卡片 _ 卡片组

请注意，所有这些类型都来自 C/C++！对于我们拥有的功能:

*   **def** —常规 python 函数，仅从 Python 调用。
*   **cdef** — Cython only 不能从纯 python 代码访问的函数，即必须在 Cython 内调用
*   **cpdef** — C 和 Python。可以从 C 和 Python 中访问

有了对 Cython 类型的了解，我们就可以开始实施我们的加速了！

# 如何用 Cython 加速你的代码

我们要做的第一件事是建立一个 Python 代码基准:一个用于计算数字阶乘的 for 循环。原始 Python 代码如下所示:

我们的 Cython 相同的功能看起来非常相似。首先，我们将确保我们的 Cython 代码文件有一个`.pyx`扩展名。代码本身唯一的变化是我们声明了每个变量和函数的类型。

注意这个函数有一个`cpdef`来确保我们可以从 Python 中调用它。也看看我们的循环变量`i`是如何拥有类型的。你需要为函数中的所有变量**设置类型，以便 C 编译器知道使用什么类型！**

接下来，创建一个`setup.py`文件，该文件将 Cython 代码编译成 C 代码:

并执行编译:

```
python setup.py build_ext --inplace
```

嘣！我们的 C 代码已经编译好了，可以使用了！

你会看到在你的 Cython 代码所在的文件夹中，有运行 C 代码所需的所有文件，包括`run_cython.c`文件。如果你很好奇，可以看看 Cython 生成的 C 代码！

现在我们已经准备好测试我们新的超快的 C 代码了！查看下面的代码，它实现了一个速度测试来比较原始 Python 代码和 Cython 代码。

代码非常简单。我们以与普通 Python 相同的方式导入文件，并以与普通 Python 相同的方式运行函数！

Cython 几乎可以在任何原始 Python 代码上获得很好的加速，根本不需要太多额外的努力。需要注意的关键是，你经历的循环越多，你处理的数据越多，Cython 就能提供越多的帮助。

查看下表，它显示了 Cython 为我们提供的不同阶乘值的速度。我们通过 Cython 获得了超过 36 倍的速度提升！

# 喜欢学习？

在 twitter 上关注我，我会在这里发布所有最新最棒的人工智能、技术和科学！也在 [LinkedIn](https://www.linkedin.com/in/georgeseif/) 上和我联系吧！