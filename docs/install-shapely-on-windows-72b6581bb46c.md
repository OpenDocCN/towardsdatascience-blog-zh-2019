# 在 Windows 上安装 Shapely

> 原文：<https://towardsdatascience.com/install-shapely-on-windows-72b6581bb46c?source=collection_archive---------7----------------------->

![](img/89294d49ec4b512a04ac7708f2730fbe.png)

Shapely 是一个 Python 包，它充满了用几何图形处理数据的各种可能性。
但是，如果你在 Windows 操作系统上工作，安装 Shapely 并不是一件简单的事情。

幸运的是，一旦你知道了细节，就没那么难了。

1.  F **查明您使用的是 32 位还是 64 位 Windows**。转到`Settings => System => About => System Type`。
2.  **找出你的 python 版本**。打开命令提示符，输入`python --version`并记住前两个数字。例如，我的版本是`Python 3.7.3`，所以我应该记得号码`37`。
3.  `pip install wheel`
4.  在此[处](https://www.lfd.uci.edu/~gohlke/pythonlibs/#shapely)和**下载项目 1-2 对应的车轮**。例如，我有 **64** 位操作系统和 Python**3.7**0.3，所以我需要下载文件`Shapely-1.6.4.post2-cp**37**-cp**37**m-win_amd**64**.whl`。
5.  `pip install <path-to-Downloads>\<filename_from_item_4>`。例如，我输入了
    `pip install C:\Users\Dalya\Downloads\Shapely-1.6.4.post2-cp**37**-cp**37**m-win_amd**64**.whl`。

就是这样！

# 使用 conda 安装

这真的很简单！
关于为什么`conda install shapely`有时不工作，以及为什么需要第一行的更多信息，请到[这里](https://conda-forge.org/)。

```
conda config --add channels conda-forge
conda install shapely
```

就是这样！