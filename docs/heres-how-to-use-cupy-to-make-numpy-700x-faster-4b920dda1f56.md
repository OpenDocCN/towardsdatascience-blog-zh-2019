# 以下是如何使用 CuPy 让 Numpy 快 10 倍以上

> 原文：<https://towardsdatascience.com/heres-how-to-use-cupy-to-make-numpy-700x-faster-4b920dda1f56?source=collection_archive---------1----------------------->

## 是时候来点 GPU 的威力了！

![](img/5f5b2ddfb649018b7526776aa90dae4f.png)

> 想获得灵感？快来加入我的 [**超级行情快讯**](https://www.superquotes.co/?utm_source=mediumtech&utm_medium=web&utm_campaign=sharing) 。😎

Numpy 是 Python 社区的一份礼物。它允许数据科学家、机器学习从业者和统计学家以简单高效的方式处理矩阵格式的大量数据。

即使就其本身而言，Numpy 在速度上已经比 Python 有了很大的进步。每当您发现您的 Python 代码运行缓慢时，尤其是如果您看到许多 for 循环，将[数据处理转移到 Numpy](/one-simple-trick-for-speeding-up-your-python-code-with-numpy-1afc846db418) 并让其矢量化以最高速度完成工作总是一个好主意！

尽管如此，即使有这样的加速，Numpy 也只是在 CPU 上运行。由于消费类 CPU 通常只有 8 个内核或更少，因此并行处理的数量以及所能实现的加速是有限的。

这就是我们的新朋友 CuPy 的切入点！

# 什么是 CuPy？

CuPy 是一个通过利用 CUDA GPU 库在 Nvidia GPUs 上实现 Numpy 阵列的库。通过这种实现，由于 GPU 拥有许多 CUDA 核心，可以实现卓越的并行加速。

![](img/8697aaa0de2493a626816ca090134199.png)

CuPy 的界面是 Numpy 的一面镜子，在大多数情况下，它可以作为一个直接的替代品。只需将 Numpy 代码替换为兼容的 CuPy 代码，就可以实现 GPU 加速。CuPy 将支持 Numpy 拥有的大多数数组操作，包括索引、广播、数组数学和各种矩阵转换。

如果你有一些特定的还不被支持的东西，你也可以写定制的 Python 代码来利用 CUDA 和 GPU 加速。所需要的只是一小段 C++格式的代码，CuPy 会自动进行 GPU 转换，非常类似于[使用 Cython](/use-cython-to-get-more-than-30x-speedup-on-your-python-code-f6cb337919b6) 。

要开始使用 CuPy，我们可以通过 pip 安装库:

```
pip install cupy
```

# 使用 CuPy 在 GPU 上运行

对于这些基准测试，我将使用具有以下设置的 PC:

*   i7–8700k CPU
*   1080 Ti GPU
*   32 GB DDR 4 3000 MHz 内存
*   CUDA 9.0

一旦安装了 CuPy，我们可以像导入 Numpy 一样导入它:

```
*import* numpy *as* np
*import* cupy *as* cp
*import* time
```

对于其余的代码，在 Numpy 和 CuPy 之间切换就像用 CuPy 的`cp`替换 Numpy 的`np`一样简单。下面的代码为 Numpy 和 CuPy 创建了一个包含 10 亿个 1 的 3D 数组。为了测量创建数组的速度，我使用了 Python 的本地`time`库:

```
### Numpy and CPU
s = time.time()
**x_cpu = np.ones((1000,1000,1000))**
e = time.time()
print(e - s)### CuPy and GPU
s = time.time()
**x_gpu = cp.ones((1000,1000,1000))
cp.cuda.Stream.null.synchronize()**
e = time.time()
print(e - s)
```

那很容易！

注意我们是如何在 cupy 数组初始化之后添加额外的一行的。这样做是为了确保我们的代码在进入下一行之前在 GPU 上完成执行。

令人难以置信的是，尽管这只是创建数组，但 CuPy 的速度还是快得多。Numpy 在 1.68 秒内创建了 10 亿个 1 的数组，而 CuPy 只用了 0.16 秒；这是 10.5 倍的加速！

但是我们仍然可以做得更多。

让我们试着在数组上做一些数学运算。这次我们将整个数组乘以 5，并再次检查 Numpy 与 CuPy 的速度。

```
### Numpy and CPU
s = time.time()
**x_cpu *= 5**
e = time.time()
print(e - s)### CuPy and GPU
s = time.time()
**x_gpu *= 5
cp.cuda.Stream.null.synchronize()** e = time.time()
print(e - s)
```

在这种情况下，CuPy **撕碎** Numpy。Numpy 拿了 0.5845 而 CuPy 只拿了
0.0575；这是 10.17 倍的加速！

现在，让我们尝试使用多个数组并进行一些操作。下面的代码将完成以下工作:

1.  将数组乘以 5
2.  将数组本身相乘
3.  将数组添加到自身

```
### Numpy and CPU
s = time.time()
**x_cpu *= 5
x_cpu *= x_cpu
x_cpu += x_cpu**
e = time.time()
print(e - s)### CuPy and GPU
s = time.time()
**x_gpu *= 5
x_gpu *= x_gpu
x_gpu += x_gpu
cp.cuda.Stream.null.synchronize()** e = time.time()
print(e - s)
```

在这种情况下，Numpy 在 CPU 上执行过程用了 1.49 秒，而 CuPy 在 GPU 上执行过程用了 0.0922 秒；一个更适度但仍然很棒的 16.16 倍加速！

# 总是超级快吗？

使用 CuPy 是将 GPU 上的 Numpy 和 matrix 运算加速许多倍的好方法。值得注意的是，您将获得的加速在很大程度上取决于您正在使用的阵列的大小。下表显示了我们改变正在处理的数组大小时的速度差异:

当我们达到大约 1000 万个数据点时，速度会急剧加快，当我们超过 1 亿个数据点时，速度会快得多。低于这个，Numpy 其实更快。此外，请记住，更多的 GPU 内存将帮助您处理更多的数据，所以重要的是要看看您的 GPU 是否有足够的内存来容纳足够的数据。

# 喜欢学习？

在 [twitter](https://twitter.com/GeorgeSeif94) 上关注我，我会在那里发布所有最新最棒的人工智能、技术和科学！也在 LinkedIn 上与我联系！