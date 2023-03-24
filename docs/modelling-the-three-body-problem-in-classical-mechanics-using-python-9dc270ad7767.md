# 用 Python 在经典力学中模拟三体

> 原文：<https://towardsdatascience.com/modelling-the-three-body-problem-in-classical-mechanics-using-python-9dc270ad7767?source=collection_archive---------7----------------------->

## 引力基础概述，Scipy 中的 odeint 解算器和 Matplotlib 中的 3D 绘图

![](img/c4ac437ce2ecbd1e844fe5565ecdf09e.png)

Image by [Kevin Gill](https://www.flickr.com/photos/kevinmgill/) on [Flickr](https://www.flickr.com/photos/kevinmgill/16532908079/)

# 1.介绍

我最近读了中国作家刘的科幻小说《三体》。在书中，他描述了一个虚构的外星文明，生活在一个被三颗恒星环绕的名为 Trisolaris 的星球上。由于三颗恒星的存在，你认为他们的存在会与我们的存在有多大的不同？耀眼的阳光？持续的夏天？事实证明，事情要糟糕得多。

我们很幸运生活在一个只有一颗主要恒星的太阳系中，因为这使得恒星(我们的太阳)的轨道是可预测的。将恒星的数量增加到两颗，系统仍然保持稳定。它有，我们称之为的解析解——也就是说，我们可以求解描述它的方程组，得到一个精确给出系统从 1 秒到一百万年的时间演化的函数。

然而，当你添加第三个身体时，一些不寻常的事情发生了。系统变得混乱不堪，高度不可预测。它没有解析解(除了少数特殊情况),它的方程只能在计算机上数值求解。它们会突然从稳定变为不稳定，反之亦然。生活在这样一个混乱世界中的三索拉人发展出了在“混乱时代”让自己“脱水”并冬眠，在“稳定时代”醒来并平静生活的能力。

书中对恒星系统有趣的形象化描述启发我阅读了引力中的 n 体类问题以及用于解决它们的数值方法。这篇文章涉及到理解这个问题所需的引力的几个核心概念和求解描述这个系统的方程所需的数值方法。

通过本文，您将了解到以下工具和概念的实现:

*   使用 **Scipy** 模块中的 **odeint** 函数在 Python 中解微分方程。
*   使方程无量纲化
*   在 **Matplotlib** 中制作 3D 图

# 2.引力基础

## 2.1 牛顿引力定律

牛顿万有引力定律说，任意两个点质量之间都有引力(称为万有引力)，其大小**与它们质量的乘积**成正比，**与它们之间距离的平方成反比。**下面的等式用向量的形式表示这个定律。

![](img/457e03700236103485fb70f8c02ec4b4.png)

这里， *G* 是万有引力常数， *m₁* 和 *m₂* 是两个物体的质量， *r* 是它们之间的距离。单位矢量远离物体 *m₁* 指向 *m₂* ，力也作用在同一个方向。

## 2.2 运动方程

根据**牛顿第二运动定律**，物体上的净力产生了物体动量的净变化——简单来说，**力就是质量乘以加速度**。因此，将上面的方程应用到质量为 *m₁* 的物体上，我们得到下面的物体运动微分方程。

![](img/e5e9d148ef85ed12e6db5816c0221b59.png)

请注意，我们将单位向量分解为向量 *r* 除以其大小 *|r|* ，从而将分母中的 *r* 项的幂增加到 3。

现在，我们有了一个**二阶微分方程**来描述两个物体由于重力而产生的相互作用。为了简化它的解，我们可以把它分解成两个一阶微分方程。

物体的加速度是物体速度随时间的变化，因此位置的**二阶微分可以用速度**的**一阶微分代替。类似地，**速度**可以表示为**位置的一阶微分。****

![](img/e674ed42a7d53605fe83b4e2e2e1f8ae.png)

索引 *i* 用于待计算位置和速度的物体，而索引 *j* 用于与物体 *i* 相互作用的另一物体。因此，对于一个两体系统，我们将求解这两个方程的两个集合。

## 2.3 质心

另一个需要记住的有用概念是系统的质心。质心是系统所有质量矩的**总和为零的点**——简单来说，你可以把它想象成系统整体质量平衡的点。

有一个简单的公式可以求出系统的质心和速度。它包括位置和速度向量的质量加权平均值。

![](img/db29fca54efe8f388cf9654f3bccda95.png)

在建立三体系统的模型之前，让我们首先建立一个两体系统的模型，观察它的行为，然后将代码扩展到三体系统。

# 3.两体模型

## 3.1 半人马座阿尔法星系

两体系统的一个著名的现实世界例子可能是半人马座阿尔法星系。它包含三颗恒星——**半人马座α星 A、半人马座α星 B** 和**半人马座α星 C** (通常称为比邻星)。然而，由于比邻星与其他两颗恒星相比质量小得可以忽略不计，半人马座阿尔法星被认为是一个**双星系统**。这里要注意的重要一点是，在多体系统中考虑的物体都有相似的质量。因此，太阳-地球-月亮不是一个三体系统，因为它们没有相等的质量，地球和月亮也不会显著影响太阳的路径。

![](img/84c9040a9882873b8cc079fdcb3033f1.png)

The Alpha Centauri binary star system captured from the Paranal Observatory in Chile by [John Colosimo](https://www.eso.org/public/images/colosimo-130704-011/)

## 3.2 无量纲化

在我们开始解这些方程之前，我们必须先把它们无量纲化。那是什么意思？我们将方程中所有有量纲(分别像 **m、m/s、kg** 的量)的量(像**位置、速度、质量**等等)转换成大小接近 1 的无量纲量。这样做的原因是:

*   在微分方程中，不同的项可能有不同的数量级(从 0.1 到 10 ⁰).如此巨大的差距可能导致数值方法收敛缓慢。
*   如果所有项的幅度变得接近于 1，那么所有的**计算将变得比幅度不对称地大或小时更便宜**。
*   **您将获得一个相对于标尺的参考点**。例如，如果我给你一个量，比如 4×10 ⁰千克，你可能无法计算出它在宇宙尺度上是大还是小。然而，如果我说是太阳质量的 2 倍，你将很容易理解这个量的意义。

为了无量纲化方程，**将每个量除以一个固定的参考量**。例如，将质量项除以太阳的质量，位置(或距离)项除以半人马座阿尔法星系统中两颗恒星之间的距离，时间项除以半人马座阿尔法星的轨道周期，速度项除以地球围绕太阳的相对速度。

当你用参考量除每一项时，你也需要乘以它以避免改变等式。所有这些项连同 *G* 可以组合成一个常数，比如说等式 1 的 *K₁* 和等式 2 的 *K₂* 。因此，无量纲化方程如下:

![](img/37815928cad49a348801243d6a36b03d.png)

术语上的横条表示术语是无量纲的。这些是我们将在模拟中使用的最终方程。

## 3.3 代码

让我们从导入模拟所需的所有模块开始。

```
#Import scipy
import scipy as sci#Import matplotlib and associated modules for 3D and animations
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from matplotlib import animation
```

接下来，让我们定义用于无量纲化方程的常数和参考量以及净常数 *K₁* 和 *K₂* 。

```
#Define universal gravitation constant
G=6.67408e-11 #N-m2/kg2#Reference quantities
m_nd=1.989e+30 #kg #mass of the sun
r_nd=5.326e+12 #m #distance between stars in Alpha Centauri
v_nd=30000 #m/s #relative velocity of earth around the sun
t_nd=79.91*365*24*3600*0.51 #s #orbital period of Alpha Centauri#Net constants
K1=G*t_nd*m_nd/(r_nd**2*v_nd)
K2=v_nd*t_nd/r_nd
```

是时候定义一些参数来定义我们试图模拟的两颗恒星了——它们的**质量**、**初始位置**和**初始速度**。注意这些参数都是无量纲的，所以半人马座阿尔法星 A 的质量定义为 1.1(表示太阳质量的 1.1 倍，这是我们的参考量)。速度是任意定义的，没有一个物体能逃脱彼此的引力。

```
#Define masses
m1=1.1 #Alpha Centauri A
m2=0.907 #Alpha Centauri B#Define initial position vectors
r1=[-0.5,0,0] #m
r2=[0.5,0,0] #m#Convert pos vectors to arrays
r1=sci.array(r1,dtype="float64")
r2=sci.array(r2,dtype="float64")#Find Centre of Mass
r_com=(m1*r1+m2*r2)/(m1+m2)#Define initial velocities
v1=[0.01,0.01,0] #m/s
v2=[-0.05,0,-0.1] #m/s#Convert velocity vectors to arrays
v1=sci.array(v1,dtype="float64")
v2=sci.array(v2,dtype="float64")#Find velocity of COM
v_com=(m1*v1+m2*v2)/(m1+m2)
```

我们现在已经定义了模拟所需的大部分主要量。我们现在可以继续准备 scipy 中的 **odeint** 解算器来解我们的方程组。

为了求解任何一个 ODE，你需要**方程**(当然！)、**一组初始条件**和**时间跨度**来求解方程。 **odeint** 解算器也需要这三个基本要素。这些方程是通过函数来定义的。该函数接受一个包含所有因变量(这里是位置和速度)的数组和一个包含所有自变量(这里是时间)的数组。它返回数组中所有微分的值。

```
#A function defining the equations of motion 
def TwoBodyEquations(w,t,G,m1,m2):
    r1=w[:3]
    r2=w[3:6]
    v1=w[6:9]
    v2=w[9:12] r=sci.linalg.norm(r2-r1) #Calculate magnitude or norm of vector dv1bydt=K1*m2*(r2-r1)/r**3
    dv2bydt=K1*m1*(r1-r2)/r**3
    dr1bydt=K2*v1
    dr2bydt=K2*v2 r_derivs=sci.concatenate((dr1bydt,dr2bydt))
    derivs=sci.concatenate((r_derivs,dv1bydt,dv2bydt))
    return derivs
```

从代码片段中，您可能能够非常容易地识别微分方程。其他零零碎碎的是什么？请记住，我们正在解决 3 维的方程，所以每个位置和速度矢量将有 3 个组成部分。现在，如果考虑上一节给出的两个向量微分方程，需要求解向量的所有三个分量。因此，对于单个物体，你需要解 6 个标量微分方程。对于两个物体，你得到了，12 个标量微分方程。所以我们制作了一个大小为 12 的数组 *w* ,用来存储两个物体的位置和速度坐标。

在函数的末尾，我们连接或加入所有不同的导数，并返回一个大小为 12 的数组*deriv*。

困难的工作现在完成了！剩下的就是将函数、初始条件和时间跨度输入到 **odeint** 函数中。

```
#Package initial parameters
init_params=sci.array([r1,r2,v1,v2]) #create array of initial params
init_params=init_params.flatten() #flatten array to make it 1D
time_span=sci.linspace(0,8,500) #8 orbital periods and 500 points#Run the ODE solver
import scipy.integratetwo_body_sol=sci.integrate.odeint(TwoBodyEquations,init_params,time_span,args=(G,m1,m2))
```

变量 *two_body_sol* 包含关于两体系统的所有信息，包括位置矢量和速度矢量。为了创建我们的情节和动画，我们只需要位置向量，所以让我们把它们提取到两个不同的变量。

```
r1_sol=two_body_sol[:,:3]
r2_sol=two_body_sol[:,3:6]
```

是时候剧情了！这就是我们将利用 Matplotlib 的 3D 绘图功能的地方。

```
#Create figure
fig=plt.figure(figsize=(15,15))#Create 3D axes
ax=fig.add_subplot(111,projection="3d")#Plot the orbits
ax.plot(r1_sol[:,0],r1_sol[:,1],r1_sol[:,2],color="darkblue")
ax.plot(r2_sol[:,0],r2_sol[:,1],r2_sol[:,2],color="tab:red")#Plot the final positions of the stars
ax.scatter(r1_sol[-1,0],r1_sol[-1,1],r1_sol[-1,2],color="darkblue",marker="o",s=100,label="Alpha Centauri A")
ax.scatter(r2_sol[-1,0],r2_sol[-1,1],r2_sol[-1,2],color="tab:red",marker="o",s=100,label="Alpha Centauri B")#Add a few more bells and whistles
ax.set_xlabel("x-coordinate",fontsize=14)
ax.set_ylabel("y-coordinate",fontsize=14)
ax.set_zlabel("z-coordinate",fontsize=14)
ax.set_title("Visualization of orbits of stars in a two-body system\n",fontsize=14)
ax.legend(loc="upper left",fontsize=14)
```

最后的图非常清楚地表明，轨道遵循一种可预测的模式，正如两体问题的解决方案所预期的那样。

![](img/13de0500ae4a1c61d7095686f953ad73.png)

A Matplotlib plot showing the time evolution of the orbits of the two stars

这里有一个动画展示了轨道的逐步演变。

An animation made in Matplotlib that shows the time evolution step-by-step (code not given in article)

我们还可以做一个可视化，那是从质心的参考系。上面的图像是从空间中某个任意的静止点拍摄的，但是如果我们从系统的质量中心观察这两个物体的运动，我们将会看到一个更加清晰的图像。

首先让我们在每个时间步找到质心的位置，然后从两个物体的位置向量中减去这个向量，找到它们相对于质心的位置。

```
#Find location of COM
rcom_sol=(m1*r1_sol+m2*r2_sol)/(m1+m2)#Find location of Alpha Centauri A w.r.t COM
r1com_sol=r1_sol-rcom_sol#Find location of Alpha Centauri B w.r.t COM
r2com_sol=r2_sol-rcom_sol
```

最后，我们可以使用用于绘制前一个视图的代码，通过改变变量来绘制后面的视图。

![](img/17de6d442f448fd1d10473fccb07e830.png)

A Matplotlib plot showing the time evolution of the orbits of the two stars as seen from the COM

如果你坐在通讯器前观察这两个天体，你会看到上面的轨道。从这个模拟中还不清楚，因为时间尺度非常小，但即使这些轨道保持旋转非常轻微。

现在很清楚，它们遵循非常可预测的路径，你可以使用一个函数——也许是椭球的方程——来描述它们在空间中的运动，正如两体系统所预期的那样。

# 4.三体模型

## 4.1 代码

现在，为了将我们先前的代码扩展到三体系统，我们必须对参数进行一些添加——添加第三体的质量、位置和速度向量。让我们假设第三颗星的质量等于太阳的质量。

```
#Mass of the Third Star
m3=1.0 #Third Star#Position of the Third Star
r3=[0,1,0] #m
r3=sci.array(r3,dtype="float64")#Velocity of the Third Star
v3=[0,-0.01,0]
v3=sci.array(v3,dtype="float64")
```

我们需要更新规范中的质心公式和质心速度公式。

```
#Update COM formula
r_com=(m1*r1+m2*r2+m3*r3)/(m1+m2+m3)#Update velocity of COM formula
v_com=(m1*v1+m2*v2+m3*v3)/(m1+m2+m3)
```

对于一个三体系统，我们将需要修改运动方程，以包括另一个物体的存在所施加的额外引力。因此，我们需要在 RHS 上添加一个力项，以表示每一个其他物体对所讨论的物体施加的力。在三体系统的情况下，一个物体将受到其余两个物体施加的力的影响，因此**两个力项将出现在 RHS** 上。它在数学上可以表示为。

![](img/d1ad618902733d8ec02aecc85ae86452.png)

为了在代码中反映这些变化，我们需要创建一个新函数来提供给 **odeint** 求解器。

```
def ThreeBodyEquations(w,t,G,m1,m2,m3):
    r1=w[:3]
    r2=w[3:6]
    r3=w[6:9]
    v1=w[9:12]
    v2=w[12:15]
    v3=w[15:18] r12=sci.linalg.norm(r2-r1)
    r13=sci.linalg.norm(r3-r1)
    r23=sci.linalg.norm(r3-r2)

    dv1bydt=K1*m2*(r2-r1)/r12**3+K1*m3*(r3-r1)/r13**3
    dv2bydt=K1*m1*(r1-r2)/r12**3+K1*m3*(r3-r2)/r23**3
    dv3bydt=K1*m1*(r1-r3)/r13**3+K1*m2*(r2-r3)/r23**3
    dr1bydt=K2*v1
    dr2bydt=K2*v2
    dr3bydt=K2*v3 r12_derivs=sci.concatenate((dr1bydt,dr2bydt))
    r_derivs=sci.concatenate((r12_derivs,dr3bydt))
    v12_derivs=sci.concatenate((dv1bydt,dv2bydt))
    v_derivs=sci.concatenate((v12_derivs,dv3bydt))
    derivs=sci.concatenate((r_derivs,v_derivs))
    return derivs
```

最后，我们需要调用 **odeint** 函数，并向其提供上述函数和初始条件。

```
#Package initial parameters
init_params=sci.array([r1,r2,r3,v1,v2,v3]) #Initial parameters
init_params=init_params.flatten() #Flatten to make 1D array
time_span=sci.linspace(0,20,500) #20 orbital periods and 500 points#Run the ODE solver
import scipy.integratethree_body_sol=sci.integrate.odeint(ThreeBodyEquations,init_params,time_span,args=(G,m1,m2,m3))
```

与两体模拟一样，我们需要提取所有三体的位置坐标以进行绘制。

```
r1_sol=three_body_sol[:,:3]
r2_sol=three_body_sol[:,3:6]
r3_sol=three_body_sol[:,6:9]
```

可以使用上一节中给出的代码进行一些修改来制作最终的绘图。这些轨道没有可预测的模式，你可以从下面混乱的图表中观察到。

![](img/f2cb83bd063feb0f4e9871571d10c7de.png)

A Matplotlib plot showing the time evolution of the orbits of the three stars

一个动画会让混乱的情节变得更容易理解。

An animation made in Matplotlib that shows the time evolution step-by-step (code not given in article)

这是另一个初始配置的解决方案，你可以观察到，这个解决方案最初似乎是稳定的，但后来突然变得不稳定。

An animation made in Matplotlib that shows the time evolution step-by-step (code not given in article)

你可以试着改变初始条件，看看不同的解决方案。近年来，由于更强大的计算能力，人们发现了许多有趣的三体解，其中一些似乎是周期性的——就像 8 字形解，其中所有三个物体都在平面 8 字形路径上移动。

一些供进一步阅读的参考资料:

1.  一篇关于三体的数学描述的小论文。
2.  一篇关于平面受限三体解研究的[论文](http://www.maths.usyd.edu.au/u/joachimw/thesis.pdf)(包括图 8 和希尔解)。

我没有在本文中包括动画的代码。如果你想了解更多，你可以[发邮件](mailto:gauravsdeshmukh@outlook.com)给我或者[在推特上联系我。](https://twitter.com/intent/follow?screen_name=ChemAndCode)