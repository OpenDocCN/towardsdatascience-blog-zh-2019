# 学习足够有用的 Python 第 2 部分

> 原文：<https://towardsdatascience.com/learn-enough-python-to-be-useful-part-2-34f0e9e3fc9d?source=collection_archive---------17----------------------->

## 如何使用 if __name__ == "__main__ `”`

本文是帮助您熟悉 Python 脚本的系列文章之一。它面向数据科学家和任何 Python 编程新手。

![](img/6696acd6dc996ca852e1ddffbf956b1b.png)

cat_breed = maine_coon

`if __name__ == "__main__":`是你在 Python 脚本中看到的那些经常不被解释的东西之一。你可能已经看过上百个关于 Python 列表理解的教程，但是 Python 脚本约定却没有得到足够的重视。在这篇文章中，我将帮助你把这些点联系起来。

![](img/679a88a46dfc3899ecd3311371a9eed3.png)

state = maine

`if __name__ == "__main__":`确保 *if* 块中的代码只有在找到它的文件被直接运行时才运行。让我们来分析一下这是如何工作的。

# 这是什么？

每个 Python 文件都有一个名为`__name__`的特殊属性。

测试一下。创建一个以`print(__name__)`为唯一内容的 Python 文件 *my_script.py* 。然后在您的终端中运行`python my_script.py`。`__main__`你将会看到。

注意，*名字*两边都是双下划线。那些被称为 *dunders* 的是*双下划线*。他们也被称为*魔法方法*或*特殊方法*。

您可能熟悉来自 *__init__* 方法的 dunders，该方法用于初始化您创建的类的对象。

# 为什么要用 __main__？

如果您将文件作为模块导入，则代码不会运行。只有当该文件作为自身运行时，代码才会运行。

看看这个例子。代码可从 [GitHub](https://github.com/discdiver/name-equals-main) 获得。

```
# my_script
print(f"My __name__ is: {__name__}")def i_am_main():
    print("I'm main!")def i_am_imported():
    print("I'm imported!")if __name__ == "__main__":
    i_am_main()
else:
    i_am_imported()
```

使用`python my_script.py`在终端中运行程序会导致

```
My __name__ is: __main__
I'm main!
```

好的。这正是我们所期待的。😀`__name__`被设置为`__main__`,因为文件本身被执行。

让我们看看如果导入文件会发生什么。用`python`在终端中启动一个 IPython 会话，然后用`import my_script`导入文件(用 no。py 扩展名)。这会产生以下输出:

```
My __name__ is: my_script
I'm imported!
```

狂野！文件导入后，`__name__`成为文件名`my_script`！又不是`*__*main*__*`！

所执行的功能是 *else* bloc: `i_am_imported`中的功能。

当您创建一个真正的模块导入到您的程序中时，这个功能是很有用的。测试时也很方便。在这里阅读更多。

![](img/e1c84c220828b08d6cf0564ffd37ef64.png)

another_cat = maine_coon

## 包装

希望这个解释能让你在使用 Python 脚本时更加得心应手。去和`if __name__ == "__main__":`玩玩，找个人解释一下，巩固你的知识。

如果您对 Python 脚本的更多技巧感兴趣，请查看我写的这篇关于`argparse`的文章。

[](/learn-enough-python-to-be-useful-argparse-e482e1764e05) [## 学习足够有用的 Python:arg parse

### 如何在脚本中加入命令行参数

towardsdatascience.com](/learn-enough-python-to-be-useful-argparse-e482e1764e05) 

我正在撰写更多关于跳出 Jupyter 笔记本的文章，所以请关注[我](https://medium.com/@jeffhale)并加入我的[数据惊人邮件列表](https://dataawesome.com)，确保你不会错过它们！🎉

我希望这个指南对你有所帮助。如果你有，请在你最喜欢的社交媒体渠道上分享，这样其他人也可以找到它。😃

[![](img/ba32af1aa267917812a85c401d1f7d29.png)](https://dataawesome.com)

感谢阅读！👏