# 如何增强你的熊猫工作流程

> 原文：<https://towardsdatascience.com/how-to-supercharge-your-pandas-workflows-ae642b03ba52?source=collection_archive---------16----------------------->

![](img/571fd1577abb0abef8387843b72e0e90.png)

Photo by [Bill Jelen](https://unsplash.com/@billjelen?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

当我处理结构化数据时，Pandas 是我的首选工具。我认为这并不奇怪，因为 Pandas 是最流行的用于数据操作、探索和分析的 Python 库。它提供了很多现成的功能。此外，还存在各种其他模块来进一步增强和扩展 Pandas 核心。在这篇文章中，我想和你分享我用来增强我的熊猫工作流程的工具和技术。

# 增压步骤

## 准备

为了跟随例子，你需要 Python 3.7+和[熊猫](https://pandas.pydata.org/)、 [funcy](https://funcy.readthedocs.io/) 和 [seaborn](https://seaborn.pydata.org/) 。你可以在我的[***GitHub repo***](https://github.com/Shawe82/awesome-pandas)上找到所有的例子。

一如既往，我推荐使用[poem](https://poetry.eustace.io/)来管理您的 Python 包和环境。你可以查看[这篇文章](/how-to-setup-an-awesome-python-environment-for-data-science-or-anything-else-35d358cc95d5)了解如何设置它。作为一种快捷方式，我建议使用 pip 或 pipx 将其安装在您的机器上。

现在，让我们快速创建一个名为 *awesome-pandas* 的诗歌项目，在这里我们实现示例并添加必要的包

```
poetry new awesome-pandas
cd awesome-pandas
poetry add pandas seaborn funcy
```

现在我们有了一个独立的 Python 环境，安装了我们需要启动的所有东西。太好了，我们可以开始增压我们的熊猫工作流程了！

## 探索性数据分析

不管你有什么样的数据集，你首先要做的可能是对其中发生的事情有一个想法和理解。这个过程被称为探索性数据分析，EDA。

数据集中有多少要素和样本？那些特征是什么？它们有哪些数据类型？是否有任何缺失值？它们是如何分布的？有什么关联吗？这些只是人们在查看新数据集时通常会遇到的许多问题中的一部分。

对于熊猫来说，获取这些信息的通常嫌疑人是函数 *head* 、 *info* 、*description*和 *corr、*等等。因此，对于您的 EDA，您可以从一个接一个地调用这些函数开始。接下来，您可以从控制台或 Jupyter 笔记本上读取结果，并将结果关联起来。这个过程对我来说听起来很乏味。此外，它不包括任何可以很容易解释的视觉效果。创建它们需要额外的步骤和代码。以结构化和整合的方式获取所有这些信息不是很好吗？理想情况下只调用一个函数？

> 这里是熊猫的光辉之地！

[**Pandas-profiling**](https://pandas-profiling.github.io/pandas-profiling/docs/)是一个扩展 Pandas DataFrame API 的库。它将常见的第一个 EDA 步骤合并到一个函数调用中。这些包括 ***数据类型、*** ***缺失值、相关性、分位数和描述性统计以及直方图*** 等。所有这些信息被组合成一个单一的输出。*您可以通过一个配置文件* *来配置* [*分析什么以及结果如何呈现。听起来很棒，不是吗？让我们简单看一下。*](https://github.com/pandas-profiling/pandas-profiling/blob/master/pandas_profiling/config_default.yaml)

首先，我认为将它添加到我们的示例环境中是有意义的

```
poetry add pandas-profling
```

现在，您所要做的就是导入 pandas-profiling 并在 DataFrame 对象上调用 profile_report。如果你用的是 Jupyter Notebook，这个足以看到输出。当使用 IDE 或编辑器时，您必须将报告保存为 HTML 文件，并在浏览器中打开它。我不是一个笔记本电脑爱好者，所以我必须做更多的工作

这就是为什么我想出了下面的函数。它使我能够检查只调用单个函数的报告，就像 Jupyter 人一样。

现在，我们拥有和 Jupyter 人几乎一样的舒适。

但是，我希望*和*一样舒适，并使用一个 ***专用数据框架 API*** 来创建和打开我的报告。

## 扩展熊猫 API

Pandas 已经提供了一套丰富的现成方法。但是，您可能希望根据需要扩展和自定义索引、系列或数据框架对象。幸运的是，从 0.23 版本开始，Pandas 允许你通过自定义的 ***访问器*** 来实现。如果你想在 Pandas < 0.23 中添加一个自定义的访问器，你可以使用 [Pandas-flavor](https://zsailer.github.io/software/pandas-flavor/) 。

编写自定义访问器相当简单。让我们用上一节中的例子来看看它；*在浏览器中为数据帧创建并打开 pandas-profiling 报告。*

我们从一个 Python 类*报告*开始，我们必须用`@pandas.api.extensions.register_dataframe_accessor("report")`来修饰它。class' `__init__`'方法必须将 DataFrame 作为其唯一的参数。您将该对象存储在一个实例变量中，并使用它来应用您自定义方法。自定义方法是您的类的实例方法。

作为一个例子，我们将一个方法 *show_in_browser* 添加到类*报告中。我们可以通过`df.report.show_in_browser()`在数据帧 *df* 上调用它，为此，我们只需导入包含类报告的模块。*

您可以使用传递给装饰器的名称来访问您的自定义方法。举个例子，如果我们用“ *wicked* ”交换“ *report* ”，而不改变其他任何东西，我们必须调用`df.wicked.show_in_browser()`在浏览器中打开我们的报表。

说够了，下面是代码，它给我们带来了和 Jupyter 人一样的舒适

注意，我通过添加`__call__`使类 a 成为可调用的。为什么这很有用？因为现在你可以调用`df.report()`来执行一个函数，就是在这里获取报告。如果你只想给你的对象添加一个函数，并且不想调用`df.report.report().`之类的函数，这就非常方便了

关于自定义访问器的最后一件有用的事情是，ide 可以识别它们并启用自动完成功能。这增强了整体开发体验。

现在我们知道了我们的数据是什么样子，以及如何扩展 Pandas 对象。所以，让我们继续有趣的事情，看看如何改善处理体验。

## 跟踪应用调用的进度

假设您想对数据帧应用一个函数。根据数据帧的大小或函数的复杂程度，执行`df.apply(func)`可能需要很长时间。所以你可能会坐在电脑前等待它结束。不幸的是，你没有得到任何反馈，不知道你要等多久，也不知道它到底有没有用。你可能只听到一个响亮的风扇和你的电脑正在升温。如果你像我一样，你会在那一点上变得紧张和不耐烦。得到有事情发生的反馈不是更好吗？或者更好的是，得到一个关于事情完成需要多长时间的预测？

> Tqdm 是你的朋友！

[Tqdm](https://tqdm.github.io/) 是一个 Python 模块，它让你*用很少的代码和很小的执行开销就能给你的循环添加智能进度条*。除此之外，它还提供了剩余执行时间的预测。而且，对本文来说最重要的是，它附带了一个熊猫集成。让我们看看那是什么样子。

和往常一样，我们首先将模块添加到我们的示例环境中

```
poetry add tqdm
```

现在你要做的就是*导入 tqdm* 和调用 *tqdm.pandas* 向熊猫注册。现在您可以在 DataFrame 对象上调用 *progress_apply* 而不是 *apply* 。下面是使用一些无意义的示例数据的完整代码

在我的电脑上，产生的进度条看起来像

```
Hello Medium: 42%|████▏ | 4177/10000 [00:01<00:02, 2232.11it/s]
```

你现在知道你在这个过程中的位置，并且估计你还剩多少时间来喝完你的咖啡。

## 增加申请电话

在上一节中，我们已经看到了如何获得关于执行长时间运行的应用函数的反馈。这让我们可以在继续工作之前估计我们可以喝多少咖啡。然而，有些人说你不应该喝太多咖啡，因为它不健康。因为我们都想保持健康，所以减少等待时间，从而减少我们喝咖啡的数量是一件好事。即使你不喝咖啡，更少的等待时间意味着更多的生产时间。我们都想拥有它。

> Swifter 提升您的数据框架应用调用。

[更快](https://github.com/jmcarpenter2/swifter)使*更容易*以 ***最快*** ***可用*** 方式将任何功能应用到您的熊猫系列或数据帧。它首先尝试以矢量化的方式运行函数。如果失败，swifter 会决定是执行 Dask 并行处理更快还是使用标准 Pandas apply 更快。所有这些都是通过一个函数调用自动完成的。这是一个很好的抽象层次！

让我们快速看一下代码。首先，我们将 swifter 添加到我们的示例环境中(我想这会变得很无聊，但我想保持这种结构)

```
poetry add swifter
```

我们现在只需要导入得更快，并且可以在我们的数据帧或系列对象上调用`df.swifter.apply(func)`。搞定了。Swifter 还带有 tqdm 进度条实现。要调用它，您需要调用`df.swifter.progressbar(True,”Hello Medium").apply(func)` Sweat！

当你看到你必须做什么来使用 swifter 时，你能猜出它是如何注册到熊猫对象的吗？如果你的答案是“它使用熊猫扩展 API”，那你就对了。哦，这一切是如此完美地结合在一起:)

最后，让我们比较一下 swifter apply 和 Pandas apply 对上述虚拟数据的执行时间

```
df = pd.DataFrame(np.random.rand(10000, 10))
%timeit df.swifter.apply(lambda s:s**1/2, axis=1)
**29.1 ms** ± 118 µs per loop
%timeit df.apply(lambda s:s**1/2, axis=1)
**2.96 s** ± 6.39 ms per loop
```

使用 swifter 的 29.1 毫秒比使用正常应用的 2.96 秒少得多。具体来说，大约快 100 倍。

我相信您也可以手动实现这种性能，但 swifter 可以为您实现。

# 包裹

在这篇文章中，我向你展示了如何增压你的熊猫工作流程和经验。总之，对你来说最重要的三点是

*   ***使用 pandas-profiling 对 pandas 数据帧执行 EDA。***
*   利用 Pandas 扩展 API 根据您的需要定制 Pandas 对象。
*   ***使用 tqdm 和 swifter 来增强和加速对熊猫对象应用函数。***

感谢您关注这篇文章。一如既往，如有任何问题、意见或建议，请随时联系我。