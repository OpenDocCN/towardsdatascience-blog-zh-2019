# 在 R 中小心地附加包

> 原文：<https://towardsdatascience.com/attach-packages-mindfully-in-r-b9c8151b3fb4?source=collection_archive---------23----------------------->

## **使用显式名称空间减少冲突，使您的源代码更加简洁，未来更加安全**

![](img/3705f1a263474c2f32860354c8b6440e.png)

R packages exporting the same symbol lead to conflicts when imported into the global namespace ([Photo](https://pixabay.com/photos/package-packaging-shipping-carton-1511683/) by [Siala](https://pixabay.com/users/siala-719262/) on [pixabay](https://pixabay.com))

当通过 library()调用在脚本顶部附加几个库时，R 中的一个常见问题就会出现。将所有符号加载到全局 R 环境中通常会导致名称冲突。最后导入的包赢得了战斗——**顺序**很重要，根据这个顺序，库出现在脚本的顶部！

R 是一种快速发展的语言，有一个生动的社区贡献了无数的包(c.f. [R 包文档](https://rdrr.io/))。昨天的脚本很可能明天就不工作了，因为一个更新的包现在导出了一个与另一个全局附加包冲突的函数。

所以我在这里提出**注意使用**将包附加到全局 R 环境中。尽量减少 library()调用，对其他包改用 requireNamespace()。使用**显式名称空间**比如

```
requireNamespace("lubridate") # at the top of your file
# ... some code
lubridate::ymd(20190130)
```

代替

```
library(lubridate) # at the top of your file
# ... some code
ymd(20190130)
```

requireNamespace()加载包，将其留在自己的名称空间中。如果缺少所需的包，调用会产生异常。

这可能看起来很尴尬。但是:它使代码更加简洁，并且在其他软件开发领域(例如 Python)被视为一种好习惯。**否**全局附件的候选对象可能是很少使用的函数或包，从而导致名称冲突。通常，我用重载操作符全局导入包，比如 dplyr 的% > %操作符。

**例如:**我的一个脚本头看起来像这样:

```
library(dplyr)
library(tidyr)
library(readr)
library(ggplot2)
requireNamespace(“lubridate”)
requireNamespace(“glue”)# here comes my code
```

使用 data()调用的 package 参数从未附加的包中加载数据:

```
requireNamespace("psd")
data(magnet, package = "psd")
```

## 为什么 requireNamespace()很重要

**不**在顶部附加包的缺点是你的脚本可能会遇到这样的错误

```
Error in loadNamespace(name) : there is no package called ‘lubridate’
```

如果当前的 R 安装缺少所需的包，则在稀有程序路径的深度。这让人很不舒服，也很意外！

在像 R 或 Python 这样的动态类型语言中，确保软件完整性具有挑战性。编译语言在更早的时候抱怨缺少库:在编译时(如果静态链接)或者至少在启动应用程序时(如果动态链接)。

**警告:**有些 R 包在没有全局附加它们的名称空间的情况下无法工作——所以在这些情况下，你必须使用 library()来代替。这个问题的一个例子是 [**psd** 包](https://cran.r-project.org/package=psd)(版本 1.2.0):

```
data(magnet, package = "psd")
x <- magnet$clean
spec <- psd::pspectrum(x, plot = TRUE)
```

给出错误

```
Stage  0 est. (pilot)
environment  ** .psdEnv **  refreshed
detrending (and demeaning)
Error in psdcore.default(as.vector(X.d), X.frq = frq, ...) :!is.null(ops) is not TRUE
```

## 结论

通过使用显式名称空间编写健壮而简洁的 R 脚本。此外，R 脚本对其他人来说可读性更好:函数在哪个包中被调用是很清楚的！