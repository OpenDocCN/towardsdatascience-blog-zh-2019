# 使用 R 中的阿尔法形状算法估计 3D 图像的体积

> 原文：<https://towardsdatascience.com/estimating-the-volume-of-a-3d-image-using-the-alpha-shape-algorithm-in-r-61169ac00020?source=collection_archive---------21----------------------->

## 在这篇文章中，我将描述一种估计 3D 图像中物体体积的方法。我在这里展示的方法——阿尔法形状算法——最近在黑客马拉松 [Hacktaferme](http://www.acta.asso.fr/actualites/communiques-de-presse/articles-et-communiques/detail/a/detail/les-5-laureats-dhacktaferme-2019-1117.html) 上获得了三等奖。

# 本文概述:

1.解释估计 3D 点云的体积的困难以及由阿尔法形状/凹面外壳算法提出的解决方案。

2.展示如何导入和可视化。R 中的 obj 图像并访问数据点矩阵。

3.将 alpha shape 算法应用于。obj 图像来估计体积和理解 alpha 参数的效果。

4.结论和替代解决方案

# 1.1.为什么我们需要一种算法来估算 3D 形状的体积？

起初，估计点云的体积似乎很简单。如果我们有一个如下所示的 3D 立方体，那么我们可以使用我们在物理课上学过的公式。

![](img/49c62d341c8ceda198fe709dc70396c3.png)

Usual 3D Cube that we work with

![](img/beeef557478dd90d13d05f359d8e3e91.png)

Usual Formula for Estimating 3D Volume

如果我们有一个不同的形状，我们可以用已知的体积公式，用立方体或其他几何形状来填充它，这些的总和就是总体积，对吗？

![](img/d6a7f126815af06b5a296ce4c698a460.png)

Estimating volume on a complicated 2D shape by cutting the area in pieces.

但是在现实生活中，我们没有立方体。还有，我们没有形状。我们实际上有一个点云，这是实际的困难:找到点云的形状，这样我们可以把它切成几块，并计算每块的体积。

![](img/d9d1f623ffea3cbe7124c222b6ea28b5.png)

In an image, we have a point cloud, not a pre-defined shape. This is because of the functioning of a 3D camera which takes many points on a surface rather than complete shapes.

绘制点云轮廓有两种方法:凸形和凹形。凸形意味着一个形状不会向内弯曲，或者我们可以说不可能通过穿过形状的外部来连接形状内部的两个点。凹面是相反的:它向内弯曲，我们可以在它里面的两点之间画一个点，这个点穿过它的外面。

![](img/b9f99546947e65806c23cd87c2f20cf1.png)

Concave shape (left) vs a convex shape (right).

凸轮廓是一个相对容易的问题:以最小的体积(或 2D 的表面)在所有点周围取凸的形状。但是凹轮廓估计是困难的。困难在于同一个点云可能有多个凹轮廓，并且没有数学方法来定义哪个轮廓是最好的！

![](img/a908fa118bc9845370eb277bf58b21d7.png)

The convex contour has a fixed solution, the concave contour has not.

由于凹轮廓允许向内的线，最小凹轮廓形状可以在该点云中找到几乎为 0 表面的形状，这将是一个巨大的低估。另一方面，我们看到凸轮廓是一个严重的高估。

我们需要调整凹度的大小，我在这里给出的解决方案是阿尔法形状算法。

# 1.2.阿尔法形状算法是如何工作的？

为了计算点云的 alpha 形状，我们需要定义一个 alpha。alpha 为 1 将导致凸包。那么α越接近零，我们形状的轮廓就越灵活。

## 步骤 1:使用 Delauney 三角测量法寻找凸包

在这些点之间绘制三角形，以便这些三角形之间没有重叠(这就是所谓的 Delauney 三角剖分)。这给了我们凸包(迄今为止，凸包是我称之为轮廓的官方术语)。

## 步骤 2:使用一个半径为 alpha 的圆过滤这个结果

那些三角形的边都有一定的长度。我们知道凸包太大了，所以我们需要删除一些三角形边界来制作一个更好的凸包。

准确的选择标准是:围绕三角形的每条边画一个尽可能大的圆。如果
1，选择边框。这个圆不包含另一个点
2。这个圆的半径小于α

![](img/d7ddcc5ad9284e5226354a48b126c504.png)

Finding the alpha shape for the shape on the left (already applied Delauney Triangulation) using a large alpha (middle) and a small alpha (right). The large alpha has resulted here in the convex solution (middle) and the smaller alpha has given a concave hull (right). Another (small) alpha could have given a different concave hull.

# 2.在 R 中导入、可视化和访问. obj 3D 图像

为了在实践中应用凹壳算法，我们需要将 3D 图像导入到我们的编程环境中。在这个例子中，我将使用现有的 R 库。

要导入图像文件，可以使用 readobj 包中的 read.obj 函数，如左图所示。

Executing this code on a 3D image will open a viewer window to inspect your 3D object.

然后使用 *rgl* 库来获得可视化效果。首先，使用函数 *tinyobj2shapelist3d* 将您的 *obj* 对象转换为 *shapelist* ，然后使用函数 *shade3d* 生成一个 3d 查看器窗口，您可以使用鼠标来旋转和扭曲图像。

![](img/66ea636baae6bc94547379afa2b8c8c1.png)

The viewer in R gives this interactive representation of the .obj 3D image. In my example it were 3D pictures of cows.

来访问您的。将 obj 文件导入 R 后，执行以下命令:

现在我们有了数据点的矩阵(3D 点云),我们可以开始使用 Alpha Shape 算法来估计它周围的凹面外壳，并从那里计算体积。

# 3.应用 alpha 形状算法并理解 alpha 参数

为了估计凹面外壳及其体积，我们使用 R 包 *alphashape3d* ，它有一个函数 *ashape3d* 在给定 xyz 矩阵和 alpha 值的情况下创建 alpha 形状，此外，还有一个 *plot* 函数显示发生了什么，还有一个 v*volume _ ashape3d*给我们估计的体积。

完整代码如下所示，alpha 值为 0.35:

![](img/fbcb702cac37012d4e785f3fa96d7874.png)

Two concave hulls of a .obj 3D image using the alpha shape algorithm with on the left alpha = 0.05 and on the right alpha = 0.35.

现在，您可以调整 alpha 参数，以制作出您需要的形状。一种方法是看形状，看哪一个最合适。另一种方法是使用验证数据来校准 alpha 形状算法:测试不同的 alpha 值，看看哪个值给出了最正确的体积估计。

你甚至可以考虑使用网格搜索来优化 alpha，这就是我们在黑客马拉松中所做的。我们的目标是从 3D 图像开始估计一头奶牛的体重，因此我们对总共 74 张图像进行了从 0 到 1、步长为 0.05 的阿尔法网格搜索，以选择与观察到的奶牛体重相关性最大的阿尔法。这导致了 0.35 的 alpha，这并没有导致体积的最佳估计，但它确实给了我们一个有用的变量，用于进一步的机器学习。

# 4.结论和替代解决方案

在本文中，我解释了 alpha shape 算法，并展示了如何使用它从点云中估算 3D 对象的体积。这种方法的最大优点是它允许凹形，而凸形方法是不可能的。

阿尔法形状必须校准，我已经提出了几个想法来做到这一点，即使没有校准数据。尽管校准 alpha 需要一些时间，但这种方法对于建模 3D 形状来说是一个很大的附加值，并且是 3D 图像处理的一个很好的切入点。祝你好运！