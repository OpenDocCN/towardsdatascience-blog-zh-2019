# NumPy 入门

> 原文：<https://towardsdatascience.com/getting-started-with-numpy-59b22df56729?source=collection_archive---------10----------------------->

![](img/6ca5a70d1c76a8a10afac25631495fd2.png)

NumPy 代表***Num****erical****Py****thon*它是 Python 中的一个核心 ***科学计算* *库*** 。它提供了高效的 ***多维数组对象*** 和各种操作来处理这些数组对象。

在这篇文章中，你将了解到
1。安装数字
2。在 NumPy
3 中创建数组。NumPy 数组上的基本操作

## 安装 Numpy

1.  Mac 和 Linux 用户可以通过 pip 命令安装 NumPy:

```
pip install numpy
```

2.因为 windows 没有像 linux 或 mac 那样的包管理器，所以你可以从这里下载 NumPy。一旦你下载了合适的**。从链接的 whl** 文件中，打开命令提示符。导航到下载的目录。whl 文件。最后，使用以下命令安装它:

```
pip install name_of_the_file.whl
```

注意:如果您正在使用 Anaconda，您不需要安装 NumPy，因为它已经与 Anaconda 一起安装了。然而，您可以通过命令在 Anaconda 中安装任何包/库:

```
conda install name_of_the_package
# conda install numpy
```

要在我们的程序中使用 Numpy 库，你只需要导入它。

```
import numpy as np
```

## NumPy 中的数组简介

NumPy 数组是 ***同质网格*** 的值。数组的维数在 NumPy 中称为 ***轴*** 。轴数称为 ***秩*** 。给出数组沿每个维度的大小的非负整数元组称为其 ***形状*。**例如，考虑下面的 2D 阵列。

```
[[11, 9, 114]
 [6, 0, -2]]
```

1.  这个数组有两个轴。长度 2 的第一轴和长度 3 的第二轴。
2.  秩=轴数= 2。
3.  形状可以表示为:(2，3)。

## 在 NumPy 中创建数组

要创建一个数组，可以使用 ***数组****numpy 的方法。*

```
*# Creating 1D array
a = np.array([1, 2, 3])# Creating 2D array
b = np.array([[1,2,3],[4,5,6]])*
```

*创建 NumPy 数组的函数:*

```
*a = np.zeros((2,2))     # Create an array of all zerosb = np.ones((1,2))      # Create an array of all onespi = 3.14
c = np.full((2,2), pi)  # Create a constant array of pid = np.eye(3)           # Creates a 3x3 identity matrixe = np.random.random((2,2))  # Create an array of random values*
```

*为了创建数字序列，NumPy 提供了一个类似于 range 的函数，它返回数组而不是列表。*

1.  ****arange*** :返回给定区间内间隔均匀的值。步长是指定的。*
2.  ****linspace*** :返回给定间隔内间隔均匀的值。num 返回元素的数量。*

```
*A = np.arange(0, 30, 5)   # Creates [ 0, 5, 10, 15, 20, 25]B = np.linspace(1, 15, 3) # Creates [ 1.0,  8.0, 15.0]*
```

*你可以用 ***重塑*** 的方法重塑一个数组。考虑一个形状为(a1，a2，a3，…，an)的数组。我们可以改变它的形状并将其转换成另一个形状为(b1，b2，b3，…..，bM)。*

*唯一的条件是:(***)a1 * a2 * a3…* aN) = (b1 *b2 * b3 …* bM )***
即两个数组中的元素数量必须相同。*

## *访问数组元素:切片*

*就像 Python 列表一样，NumPy 数组也可以切片。由于数组可能是多维的，因此必须为数组的每个维度指定一个切片。例如*

```
*a = np.array([[1,2,3,4], [5,6,7,8], [9,10,11,12]])*
```

*这将创建一个如下所示的数组*

```
*[[1  2  3  4]
 [5  6  7  8]
 [9 10 11 12]]*
```

*现在让我们假设你想从中访问元素。*

```
*# Accessing an element
b = a[rowIndex, colIndex]# Accessing a block of elements together
c = a[start_row_index:end_row_index, start_col_index:end_col_index]*
```

***注 1:**Python 中的 Index 从 0 开始。
**注 2:** 每当我们将元素块指定为*start _ row _ index:end_row_index*时，它就意味着
【start _ row _ index，end _ row _ index】*

## *NumPy 数组上的基本操作*

*基本算术运算*

```
*# Create a dummy array for operations
a = np.array([1, 2, 5, 3])# Add 3 to every element
a += 3# Subtract 5 from every element
a -= 5# Multiply each element by 7
a *= 7# Divide each element by 6
a /= 6# Squaring each element
a **= 2# Taking transpose
a = a.T*
```

*其他一些有用的功能*

```
*# Create a dummy array
arr = np.array([[1, 5, 6], [4, 7, 2], [3, 1, 9]])# maximum element of array
print(arr.max())# row-wise maximum elements
arr.max(axis=1)# column wise minimum elements
arr.min(axis=0)# sum of all array elements
arr.sum()# sum of each row
arr.sum(axis=1)# cumulative sum along each row
arr.cumsum(axis=1)*
```

*两个 NumPy 数组上的操作*

```
*a = np.array([[1, 2], [3, 4]])b = np.array([[4, 3], [2, 1]])print(a+b)      # [[5, 5], [5, 5]]print(a-b)      # [[-3, -1], [1, 3]]print(a*b)      # [[4, 6], [6, 4]]print(a.dot(b)) # [[8, 5], [20, 13]]*
```

*NumPy 提供了很多数学函数，比如 sin，cos，exp 等。这些函数也对数组进行元素操作，产生一个数组作为输出。*

```
*a = np.array([0, np.pi/2, np.pi])print(a)print(np.sin(arr))     *# sin of each element*print(np.cos(arr))     *# cosine of each element*print(np.sqrt(arr))     *# square root of each element*print(np.exp(arr))     *# exponentials of each element*print(np.log(arr))     *# log of each element*print(np.sum(arr))     *# Sum of elements*print(np.std(arr))     *# standard deviation**
```

*NumPy 中的排序数组*

```
*a = np.array([[1, 4, 2], [3, 4, 6], [0, -1, 5]])# array elements in sorted order
print(np.sort(a, axis=None))# sort array row wise
print(np.sort(a, axis=1))*
```

*这就是 NumPy 的故事。我们已经讨论了很多重要的概念。我希望它能让你明白很多。*