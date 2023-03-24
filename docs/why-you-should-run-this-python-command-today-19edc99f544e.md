# 为什么今天应该运行这个 Python 命令

> 原文：<https://towardsdatascience.com/why-you-should-run-this-python-command-today-19edc99f544e?source=collection_archive---------17----------------------->

## 现实世界中的数据科学

## 随着 Python 2.7 即将寿终正寝，运行这个命令来看看您将会受到怎样的影响是至关重要的

![](img/d1de8fdf5141871ebf17a1621838f8d5.png)

The end is near! Photo by [NASA](https://unsplash.com/photos/rTZW4f02zY8?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/explosion?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText).

# 我们所知的世界末日

玛雅人对世界末日的预测并不遥远。尽管他们错误地认为由于太阳耀斑和其他宇宙异常现象，世界将在 2012 年终结，但他们确实准确地假设了 21 世纪初的大规模混乱。然而，预测的大规模混乱的原因与现实相去甚远。

回到现实，是时候讨论一下软件世界即将到来的末日了 Python 2.7 的死亡。对于那些不知道的人来说，Python 最流行的版本将于 2020 年 1 月 1 日到达其生命周期的终点。你说的生命终结是什么意思？当 Python 版本(这也适用于其他形式的软件)达到这个里程碑时，开发人员同意结束该特定分支的所有未来开发工作，即使主要的安全问题已经确定。

这有什么关系？Python 2.7 是在近十年前发布的，所以它肯定获得了没有任何错误或漏洞的硬化版本的好处，对吗？嗯，不完全是…尽管它的任期很长，但直到今天，新的错误和问题仍然在[官方 Python 问题追踪器](https://bugs.python.org/)上出现。明年 1 月，如果发现另一个问题，将不会发布进一步的更新，这可能会使代码无限期地易受攻击。

也许你不访问数据库，操纵关键任务数据，或者在你的代码中暴露用户注入。相反，你有一个单纯的数据科学项目，只是进行预测或运行分析，你不关心这些漏洞？即使是对安全问题“免疫”的代码仍然会受到 Python 2.7 版本的影响，因为许多流行的库都在不断地放弃对该版本的支持。

正如你在 Python 2 日落页面上的[项目时间表](https://python3statement.org)中所看到的，大多数顶级 Python 包都计划在 2020 年 1 月前停止支持，其中一些已经是 Python 3 独有的。在这些包中有几个重要的数据科学库，如 **Scipy** 、 **scikit-learn** 、 **Numpy** 、 **Pandas** 和 **Matplotlib** 。即使我们做了一个[错误的]假设，认为这些包永远不会遇到安全问题，它们总是在更新以提高性能，改进 API，并添加有用的新功能。任何坚持 Python 2.7 的代码都会错过这些重要的更新。

为什么大家不升级到 Python 3，然后继续自己的生活呢？不幸的是，Python 2 和 Python 3 本身并不兼容，在 Python 2 中工作的代码不一定能在 Python 3 中工作。对于大多数开发人员来说，这是最大的入门障碍之一，因为过渡到 Python 的最新版本可能需要对代码进行彻底的返工，这是不可取的。

在过去的几年里，许多操作系统供应商试图通过在系统范围内安装新版本的 Python 来取代过时的版本，从而帮助淘汰 Python 2.7。虽然供应商的这一举动应该受到欢迎，但许多开发人员和用户发现，由于语言的不同默认版本，他们多年来无缝工作的 Python 项目在操作系统升级后突然中断。

综上所述，这个世界上所有的好事都是要结束的，即使是臭名昭著的 Python 2.7。然而，随着这一结束，随之而来的是潜在的破坏，因为一些程序将不再在新的系统上工作。此外，任何漏洞的补丁都将不复存在，新功能将被遗忘，支持将受到冷落。然而，通过确保您的项目是最新的和经得起未来考验的，这种严峻的未来是可以预先避免的。

![](img/127041b9d3ed13ad03063e3e5a63a5a9.png)

Photo by [Chris Ried](https://unsplash.com/@cdr6934?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 检查你自己

随着悲观情绪的消失，您今天应该做些什么来看看 Python 2.7 的终结会对您产生怎样的影响呢？第一步是运行一个非常简单的 Python 命令，它会告诉您需要知道的一切:

```
python --version
```

假设没有抛出错误，上面的命令将列出您的系统正在使用的默认 Python 版本，比如`2.7.15`或`3.7.3`。如果命令返回`3.*something*`,并且您现有的所有代码都像预期的那样工作，那么您就可以继续您的幸福生活，享受 Python 3 新的精彩更新了！然而，如果您看到`2.7.*something*`(或者更糟，`2.6`或更低)，我强烈建议您进一步深入您的系统，确定您的开发环境是否存在未来维护问题和漏洞的风险。

![](img/e0861f8d8fe00828454606b38c15f27c.png)

Photo by [Hush Naidoo](https://unsplash.com/@hush52?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

## 确定原因

现在，仅仅因为系统报告了版本 2.7 或类似的版本，并不意味着这就是你的 Python 程序正在使用的版本。例如，如果您正在使用虚拟环境或 Anaconda 来本地化您的 Python 可执行文件，那么如果您在所有项目中使用该语言的新版本，您可能仍然没问题。如果这些项目中的任何一个仍然指向旧版本的 Python，那么升级系统将是明智的。

如果虚拟环境和 Anaconda 对您来说毫无意义，那么您可能只是在使用操作系统附带的默认 Python 版本。如果是这样的话，那么我强烈建议您更新您的环境，无论是项目的本地环境还是全局环境。

![](img/2ea734ddd0bdbbbe32cdc227b953f834.png)

Photo by [Fancycrave](https://unsplash.com/@fancycrave?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 修复问题

如果你还在读这篇文章，你可能正在使用旧版本的 Python，并且正在寻求升级方面的帮助。我建议按照以下常规步骤更新到最新最棒的版本:

*   安装新版本的 Python
*   升级所有依赖包
*   更新代码以兼容 Python 3
*   测试更新
*   部署最终的代码

让我们更深入地了解每一步！

## 安装新的 Python 版本

在您的系统上实际安装最新版本的 Python 有几种不同的选择。其中一些包括使用虚拟项目本地化，比如 Anaconda 和虚拟环境。官方的 Anaconda 文档对安装和使用它有很好的指导[这里](https://docs.anaconda.com/anaconda/install/linux/)。

虚拟环境也可以使用下面的指南[来安装，但是请注意，您需要先下载所需版本的 Python 并安装到您的系统中。更多关于下载和安装 Python 的信息可以在](https://virtualenv.pypa.io/en/stable/installation/)[这里](https://www.python.org/downloads/)找到。

当然，您可以选择简单地在系统范围内安装一个新版本的 Python，忘记项目本地化，尽管我建议使用上述方法之一来确保依赖关系特定于每个项目。

另一种选择是使用 Docker 容器，并在定制的 Python 容器中运行应用程序。虽然我喜欢容器化项目，但这不一定是最简单、最合理和最实用的解决方案。

## 升级所有依赖包

由于许多 Python 包开始放弃对 2.7 的支持，项目完全有可能期望依赖包的旧版本能够正常工作。如果您的项目的依赖项列在目录中的一个`requirements.txt`文件中(如果还没有，我强烈建议这样做)，明智的做法是检查所有列出的包，如果可能的话尝试升级它们。这确保了您的依赖项与 Python 的最新版本保持同步，即使 Python 2 中的所有内容都是最新的。

## 更新代码以兼容 Python 3

幸运的是，Python 2 和 3 的代码基础之间没有太大的区别，但是某些命令会使工作代码在新版本上立即失败。最大的例子之一就是`print`命令。在 Python 2 中，`print`是一个特殊的语句，允许我们写

```
print "Hello world!"
```

Python 3 中的行为发生了变化。print 语句现在是一个实际函数，所以上面的 print 语句现在必须写成

```
print("Hello World!")
```

另一个重要的问题是整数除法的处理。在 Python 2 中，两个整数相除总会返回一个整数。例如

```
3 / 2 = 1
```

尽管正确答案是 1.5，Python 2 认为这是一个整数除以一个整数等于一个整数，因此是 1。

然而，Python 3 在这种情况下会返回一个浮点数。相同的语句将计算为

```
3 / 2 = 1.5
```

这可能会对结果产生非常显著的影响，尤其是对于依赖数学运算的数据科学项目。

虽然这些只是版本 2 和版本 3 之间许多差异中的一小部分，但它们通常是升级后错误的最大元凶。从使用的角度来看，其他一些主要的区别可以在[这篇博文](https://www.differencebetween.com/difference-between-python-2-and-vs-3/)中找到。

幸运的是，我们创造了一些有用的工具来帮助从 2 到 3 的过渡。其中一个工具被恰当地命名为`2to3`,它是大多数 Python 安装所附带的。在一个特定的文件上运行这个程序将会输出为了与 Python 3 兼容而需要做的修改。虽然这个程序对于转换代码来说是一个有价值的资源，但它并不完美，所以我建议回顾一下这些更改，以确保一切都合乎逻辑，并且是必要的更新。

## 测试更新

Python 和程序的依赖项随代码一起更新后，就该测试这些变化了，以确保新版本的一切都如预期的那样工作。希望您的项目包含某种测试(如果没有，我强烈建议至少编写单元测试，以确保您的代码按预期工作)，您可以用 Python 3 运行这些测试来验证代码。

运行测试套件(如果适用)之后，运行一些模拟生产用例的实时测试。希望这个生产测试遍历大部分(如果不是全部)代码库，以验证一切都按预期运行，没有问题。如果一切看起来都很美好，你的代码[可能]已经准备好了！

## 部署最终的代码

如果你能走到这一步，恭喜你！您应该已经成功地将您的项目从 Python 2 转换到 Python 3，并确保您的代码是经得起未来考验的。此时，将变更推送到生产环境，更新任何上游存储库，并验证所有开发环境都使用最新版本的 Python 可能是安全的。您的代码现在应该能够利用 Python 3 提供的对其前身的大量改进，同时还创建了一个稳定的代码库，将在未来许多年得到支持。

![](img/2c5ae06edd5a7f7892fb78779308ca57.png)

Photo by [Alice Achterhof](https://unsplash.com/@alicegrace?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 收尾

如果您遵循了上面的所有步骤，那么您现在应该只在项目中使用 Python 3。正如整篇文章中提到的，升级失败会带来潜在的问题。有时升级并不总是可能的，例如生产系统需要特定的操作系统或系统范围的 Python 版本。几个不同的解决方案是使用虚拟环境或 Anaconda 来安装不同版本的 Python，或者用 Docker 封装应用程序，以便它将使用预期的 Python 版本，而不管您运行在哪个操作系统上。

随着 2.7 版本的结束，很难说接下来的几个月会发生什么。通过主动更新您的项目，您可以在里程碑之后避免任何潜在的陷阱。

2.7 是世界上许多 Python 开发者使用的第一个版本，并且在过去几年中已经成为深度和机器学习项目爆炸的主要容器。但是，即使是最大的火也会随着时间的流逝而熄灭，软件世界的猛兽也是如此。谢谢你的回忆。愿 3 的未来光明昌盛。