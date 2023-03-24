# 逆投影变换

> 原文：<https://towardsdatascience.com/inverse-projection-transformation-c866ccedef1c?source=collection_archive---------0----------------------->

![](img/80699cd0b0cad00adc4bddee6220fab9.png)

Fig 1: 3D points back-projected from a RGB and Depth image

# 深度和反向投影

当相机捕捉到一个场景的图像时，我们**会丢失深度信息**，因为 3D 空间中的对象和点被映射到 2D 图像平面上。这也称为投影变换，其中世界上的点被转换为 2d 平面上的像素。

但是，如果我们想做*逆呢？*也就是说，我们希望在只给定 2D 图像的情况下恢复和重建场景。为此，我们需要知道每个对应像素的深度或 Z 分量。深度可以表示为如图 2(中间)所示的图像。较亮的强度表示较远的点。

理解深度知觉在许多计算机视觉应用中是必不可少的。例如，能够测量自主车辆的深度可以更好地做出决策，因为代理完全知道其他车辆和行人之间的间距。

![](img/5840abeb089d737ba2a007c3f2dcabdc.png)

Fig 2: (from left) RGB, Depth, 3D back-projected points

## **透视视角下的推理难度**

考虑上面的图 2，仅给出 RGB 图像。很难判断左侧车道上两辆车之间的绝对距离。此外，我很难确定左边的树是离房子很近还是非常远。上述现象是**透视投影的结果，**透视投影要求我们依靠各种线索来对距离进行良好的估计。在这篇文章中，我讨论了一些你可能感兴趣的[问题:)](https://medium.com/@daryl.tanyj/depth-estimation-1-basics-and-intuition-86f2c9538cd1)

然而，如果我们在深度图的帮助下将其重新投影回 3d(图 2。右)，我们可以准确地定位树木，并认为他们实际上是远离建筑物。是的，当一个物体被另一个物体遮挡时，我们真的不擅长计算相对深度。主要的收获是只看图像，很难辨别深度。

估计深度的问题是一项正在进行的研究，多年来取得了很大进展。已经开发了许多技术，最成功的方法来自使用立体视觉确定深度[1]。并且在最近几年，使用深度学习的深度估计已经显示出令人难以置信的性能[2]，[3]。

在本文中，我们将浏览并理解执行从 2D 像素坐标到 3D 点的反投影的数学和概念。然后，我将通过 Python 中的一个简单示例来展示实际的投影。[此处有代码](https://github.com/darylclimb/cvml_project/tree/master/projections)。我们将假设提供深度图来执行 3D 重建。我们将经历的概念是相机校准参数、使用内蕴及其逆的投影变换、帧之间的坐标变换。

![](img/dfd6f9a51fb57b1fb04379b171d95048.png)

3D reconstructed point clouds from the scene in Fig 2

# 针孔摄像机模型的中心投影

![](img/b144c08c403df86ff3c0b48ed0ebdd8e.png)

Fig 3: Camera Projective Geometry

首先，理解相机投影的几何模型是核心思想。我们最终感兴趣的是深度，参数 **Z.** 这里，我们考虑最简单的没有歪斜或失真因子的针孔相机模型。

3D 点被映射到图像平面(u，v) = f(X，Y，Z)。描述这种转换的完整数学模型可以写成 **p = K[R|t] * P.**

![](img/adca3c0db40a80d1ed0b157f1203cfb4.png)

Fig 4: Camera Projective Model

在哪里

*   p 是图像平面上的投影点
*   k 是摄像机内部矩阵
*   [R|t]是描述世界坐标系中的点相对于摄像机坐标系的变换的外部参数
*   p，[X，Y，Z，1]表示在欧几里得空间中预定义的世界坐标系中表示的 3D 点
*   纵横比缩放，s:控制当焦距改变时，像素如何在 x 和 y 方向缩放

**内在参数矩阵**

矩阵 **K** 负责将 3D 点投影到图像平面。为此，必须将下列量定义为

*   焦距(fx，fy):测量图像平面 wrt 到相机中心的位置。
*   主点(u0，v0):图像平面的光学中心
*   倾斜因子:如果图像平面轴不垂直，则偏离正方形像素。在我们的例子中，这被设置为零。

求解所有参数的最常见方法是使用[棋盘法](https://opencv-python-tutroals.readthedocs.io/en/latest/py_tutorials/py_calib3d/py_calibration/py_calibration.html)。其中，借助于 PnP、直接线性变换或 RANSAC，通过匹配和求解未知参数来获得若干 2D-3D 对应，以提高鲁棒性。

确定了所有未知数后，我们可以通过应用逆运算来最终恢复 3D 点(X，Y，Z)。

## **反投影**

考虑图 4 中的等式。假设(X，Y，Z，1)在摄像机坐标系中。即我们不需要考虑外部矩阵[R|t]。扩展该等式将得到

![](img/69301c0f05411a2786bd18c375d27f39.png)

Fig 5: Equation mapping 3D to 2D point

3D 点可以通过深度图给定的 Z 值以及对 X 和 y 的求解来恢复。如果需要，我们可以进一步将这些点转换回世界坐标系。

# **逆投影示例**

让我们通过一个简单的例子来理解这些概念。我们将使用如图 1 所示的 RGB 和深度图像。图片是从模拟器[卡拉](http://carla.org/)的汽车上安装的摄像机中获取的。深度图存储为 float32，对于无限远的深度值，编码最大为 1000 米。

**视场的固有参数**

代替使用棋盘确定内部参数，可以计算针孔照相机模型的焦距和光学中心。所需信息是以像素为单位的**成像传感器高度和宽度**以及垂直和水平方向的有效**视野**。相机制造商通常会提供这些。在我们的例子中，我们将在垂直和水平方向都使用+-45 度。我们将比例因子设置为 1。

参考图 3，焦距(fx，fy)和主点(u0，v0)可以使用简单的三角学来确定。我把它留给你作为一个练习来推导，或者你可以在代码中查找它！

现在，我们可以计算如下的倒数

*   获得摄像机固有参数 K
*   求 K 的倒数
*   应用图 5 中的公式，Z 为深度图中的深度。

```
# Using Linear Algebra
cam_coords = K_inv @ pixel_coords * depth.flatten()
```

编写步骤 3 的一种更慢但更直观的方式是

```
*cam_points = np.zeros((img_h * img_w, 3))
i = 0
# Loop through each pixel in the image
for v in range(height):
    for u in range(width):
        # Apply equation in fig 5
        x = (u - u0) * depth[v, u] / fx
        y = (v - v0) * depth[v, u] / fy
        z = depth[v, u]
        cam_points[i] = (x, y, z)
        i += 1*
```

你会得到同样的结果！

# 结论

好了，我已经讲了做反投影所需的基本概念。

反投影到 3D 形成了经由结构形式运动的 3D 场景重建的基础，其中从移动的相机捕捉若干图像，以及其已知或计算的深度。此后，匹配和拼接在一起，以获得对场景结构的完整理解。

[](https://github.com/darylclimb/cvml_project/tree/master/projections/inverse_projection) [## darylclimb/cvml_project

### 使用计算机视觉和机器学习的项目和应用

github.com](https://github.com/darylclimb/cvml_project/tree/master/projections/inverse_projection) 

# **正投影:俯视图(可选)**

使用 3D 表示的点，一个有趣的应用是将其投影到场景的自上而下的视图中。对于移动机器人来说，这通常是一种有用的表示，因为障碍物之间的距离保持不变。此外，它易于解释和利用来执行路径规划和导航任务。为此，我们需要知道这些点参考的坐标系。

我们将使用如下定义的右手坐标系

![](img/f400e3f89af0744a708d5c70e09adff7.png)

[Fig 6: Camera coordinate and pixel coordinate](https://www.mathworks.com/help/vision/gs/coordinate-systems.html)

对于这个简单的例子，你认为点应该投影到哪个平面？

![](img/e16cb17ce6dd76481d9cc129ee763039.png)

Fig 7: Top-down view

如果你的猜测是在 y= 0 的平面上，那么你是对的，因为 y 代表相机坐标系定义的高度。我们简单地折叠投影矩阵中的 y 分量。

查看下图，您可以轻松测量所有车辆和物体之间的间距。

![](img/4020d7d74e744e88a0ffe0406999f027.png)

Fig 8: Birds Eye View projection

## 参考

[1]赫希米勒，H. (2005 年)。通过半全局匹配和互信息进行准确有效的立体处理。 *CVPR*

[2]周廷辉、马修·布朗、诺亚·斯内夫利和大卫·劳。来自视频的深度和自我运动的无监督学习。2017 年在 CVPR

[3]克莱门特·戈达尔、奥辛·麦克·奥德哈和加布里埃尔·J·布罗斯托。具有左右一致性的无监督单目深度估计。2017 年在 CVPR。