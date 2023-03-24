# Python 中的奇异值分解示例

> 原文：<https://towardsdatascience.com/singular-value-decomposition-example-in-python-dab2507d85a0?source=collection_archive---------1----------------------->

![](img/c2cce57edf3d06cec9c3adadb2523239.png)

[https://www.pexels.com/photo/woman-writing-on-a-whiteboard-3862130/](https://www.pexels.com/photo/woman-writing-on-a-whiteboard-3862130/)

奇异值分解(SVD)有着广泛的应用。这些包括降维、图像压缩和数据去噪。本质上，SVD 表明一个矩阵可以表示为其他三个矩阵的乘积。用数学术语来说，SVD 可以写成如下形式:

![](img/60c08fb1cf12f683a61358561c3a26e2.png)

其中 ***n*** 为行数(即样本数) ***p*** 代表维数。

假设我们有一个矩阵 ***一个*** 。

![](img/9f5d8b232366a15fba7fd1d7abb427eb.png)

为了确定关联的*矩阵，我们必须先求出矩阵*的特征向量乘以矩阵*的转置。***

***![](img/e10b7cbcfa33ad69c12f00c4e515551f.png)***

***因为***

***![](img/910a4c541f77eafd1d7b48d98def4940.png)***

***回想一下特征向量和特征值的定义:***

***![](img/da1a69eb7af94ee17b976a129d9a67fc.png)***

***后者也可以表示为:***

***![](img/4504fddb61046bc2dcd6818b016c7c0d.png)***

***因此，回到我们的例子:***

***![](img/1debc84cf66ac59ba6cde531f797576f.png)***

***我们得到了一个四次多项式。***

***![](img/36fd30d6b1db6db9aff3c25c3389fc7b.png)******![](img/f24bf65f2d6af059b748ad7f98678569.png)******![](img/0bf90b5050cb3611292123c4536e3c13.png)******![](img/08cb5865b91ba059c3bd274695fde780.png)***

***求解后，我们用其中一个特征值代替λ。***

***![](img/42d9b9b1cde662432f5e8062e8388224.png)******![](img/8b86474e78e0752c92fbc0271c7a25c0.png)***

***在将矩阵乘以向量 ***x*** 之后，我们得到如下:***

**![](img/3f7f3609d9a42a9822377b154602d8d2.png)**

**在解方程时，我们得到:**

**![](img/754e8f879ed84c751624804cd8676cec.png)**

**我们把特征向量代入**的列。****

**![](img/9ceb8085fdca74cbc6de22e0133603cb.png)**

**然后，我们对矩阵*乘以矩阵*的转置重复同样的过程。****

***![](img/393558f8b9826491a7a047304a164c18.png)***

***求解后，我们得到了 ***V*** 的表达式。***

**![](img/90c25ee0504dfad2fba8b4ec922bbbf4.png)**

**最后， ***S*** 是一个对角矩阵，其值是任一个的特征值的平方根**

**![](img/82e9acc3d8ed1031c3b72a3ae4f9c842.png)****![](img/e5305fb76c3b29a36ae0bc13f9ac1b4e.png)****![](img/c798ebecf68a6b0265c07519022e481a.png)**

**这就是事情变得有趣的地方。我们知道所有三个矩阵的乘积等价于左边的矩阵。**

**![](img/a4cbdb2343fa52bc95ec1c6d673a3f9f.png)**

**我们可以排除特征，但仍然保持原始矩阵的近似值。假设矩阵*是组成图像的列和行或像素的数据集，我们可以在理论上使用新形成的矩阵来训练模型，并达到相当的(如果不是更好的话)(由于维数灾难)精确度。***

***![](img/c800c9a8e4856c84823b69a8ecf979c0.png)******![](img/e40b1c9d4ed45075fffbc4a9f72aa2a7.png)***

# ***密码***

***让我们看看如何在 Python 中应用奇异值分解。首先，导入以下库。***

```
***import numpy as np
from sklearn.datasets import load_digits
from matplotlib import pyplot as plt
from sklearn.decomposition import TruncatedSVD
float_formatter = lambda x: "%.2f" % x
np.set_printoptions(formatter={'float_kind':float_formatter})
from sklearn.ensemble import RandomForestClassifier***
```

***在接下来的教程中，我们将尝试对手写数字进行分类。幸运的是，`scikit-learn`库提供了一个包装器函数，用于将数据集导入我们的程序。***

```
***X, y = load_digits(return_X_y=True)***
```

***该数据集包含 1797 幅 8×8 的图像。如果你指定了`return_X_y=True`,这个函数将把像素作为一维数组返回。***

```
***X.shape***
```

***![](img/7308c224ed51c04b9c27f33f3302896e.png)***

******y*** 包含每个数字的标签。***

```
**y**
```

**![](img/b28ad3ff8beb7016e9b4c5976c2c086d.png)**

**让我们来看看第一个数字。正如我们所看到的，它只是一个长度为 64 的数组，包含像素亮度。**

```
**image = X[0]**
```

**![](img/672f68f4e3f7128b2c3611f49f0bfb07.png)**

**如果我们想使用`matplotlib`查看图像，我们必须首先重塑数组。**

```
**image = image.reshape((8, 8))plt.matshow(image, cmap = 'gray')**
```

**![](img/19e02b5e42e9c92f808468b5528ea573.png)**

**接下来，我们将使用奇异值分解来看看我们是否能够仅使用每行的两个特征来重建图像。函数返回的 **s** 矩阵必须使用`diag`方法转换成对角矩阵。默认情况下，`diag`将创建一个相对于原始矩阵为 *n x n* 的矩阵。这导致了一个问题，因为矩阵的大小不再遵循矩阵乘法的规则，其中一个矩阵中的列数必须与另一个矩阵中的行数相匹配。因此，我们创建一个新的 *m x n* 矩阵，并用对角矩阵填充它的第一个 *n x n* 部分。**

```
**U, s, V = np.linalg.svd(image)S = np.zeros((image.shape[0], image.shape[1]))S[:image.shape[0], :image.shape[0]] = np.diag(s)n_component = 2S = S[:, :n_component]
VT = VT[:n_component, :]A = U.dot(Sigma.dot(VT))print(A)**
```

**![](img/94cd4447feb66744f117f9f728b9a9f1.png)**

```
**plt.matshow(A, cmap = 'gray')**
```

**![](img/5b937ab634ed9783d30bb69620d9b360.png)**

**我们可以通过取 **U** 和 **S** 矩阵的点积得到缩减的特征空间。**

```
**U.dot(S)**
```

**![](img/37d026aab37ebea8a197c472f7f372b2.png)**

## **原始与缩减的特征空间**

**让我们比较随机森林模型在使用原始手写数字训练时和使用从奇异值分解获得的缩减特征空间训练时的准确性。**

**我们可以通过查看公开得分来衡量模型的准确性。如果你对 OOB 的概念不熟悉，我鼓励你看看兰登森林的这篇文章。**

```
**rf_original = RandomForestClassifier(oob_score=True)rf_original.fit(X, y)rf_original.oob_score_**
```

**![](img/7e5a7e53ba15e5d3ba21f5675f4f08b3.png)**

**接下来，我们用 2 个组件创建并装配一个`TruncatedSVD`类的实例。值得一提的是，与前面的例子不同，我们使用的是 2/64 特性。**

```
**svd = TruncatedSVD(n_components=2)X_reduced = svd.fit_transform(X)**
```

**精简数据集中的每个图像(即行)包含 2 个特征。**

```
**X_reduced[0]**
```

**![](img/47da4ffa927d39f2873fbeb6243729fd.png)**

**看一看图像，很难区分图像由什么数字组成，它很可能是 5 而不是 0。**

```
**image_reduced = svd.inverse_transform(X_reduced[0].reshape(1,-1))image_reduced = image_reduced.reshape((8,8))plt.matshow(image_reduced, cmap = 'gray')**
```

**![](img/0fa1dc17d60459cba6244a3583d7971b.png)**

**在精简的数据集上训练随机森林分类器后，我们获得了 36.7%的微弱准确率**

```
**rf_reduced = RandomForestClassifier(oob_score=True)rf_reduced.fit(X_reduced, y)rf_reduced.oob_score_**
```

**![](img/667b42cd286559bed2fb169b6e807ae8.png)**

**我们可以通过取`explained_variance_ratio_`属性的和得到总方差解释。我们通常希望达到 80%到 90%的目标。**

```
**svd.explained_variance_ratio_.sum()**
```

**![](img/353bcf4c507c31b0c43487cee336fa7e.png)**

**让我们再试一次，只是这一次，我们使用 16 个组件。我们查看包含在 16 个特征中的信息量。**

```
**svd = TruncatedSVD(n_components=16)X_reduced = svd.fit_transform(X)svd.explained_variance_ratio_.sum()**
```

**![](img/4598773dab6464d494ccfbe19ce8ff39.png)**

**我们获得了与使用原始图像训练的模型相当的准确度，并且我们使用了 16/64=0.25 的数据量。**

```
**rf_reduced = RandomForestClassifier(oob_score=True)rf_reduced.fit(X_reduced, y)rf_reduced.oob_score_**
```

**![](img/f8642bd2d55d89c8ebc856b03a6c8112.png)**