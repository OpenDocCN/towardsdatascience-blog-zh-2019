# 使用 R 和 Python 的介绍

> 原文：<https://towardsdatascience.com/an-introduction-to-working-with-r-and-python-1c51fac0b16f?source=collection_archive---------9----------------------->

> T 他的文章旨在介绍如何在 Python 中使用 R

![](img/1d49e17ae509a612e047789007ac450d.png)

Image by Mitchell Luo

当我还是一名大学生时，统计学课程(生存分析、多变量分析等)是在 r 中教授的。然而，由于我希望学习数据科学，所以我选择了 Python，因为它对我来说似乎很“怪异”。

由于只使用 Python，我偶然发现需要实现一些统计技术，如离群值的 Grubb 检验、模拟的马尔可夫链蒙特卡罗或合成数据的贝叶斯网络。因此，本文旨在作为一篇介绍性的指南，将 R 融入到您作为 Python 数据科学家的工作流程中。如果您想作为一名 R 数据科学家将 Python 集成到您的工作流程中，reticulate 包很有用，请查看[1]。

# rpy2

我们选择 rpy2 框架，其他选项有 [pyRserve](https://pypi.org/project/pyRserve/) 或 [pypeR](http://bioinfo.ihb.ac.cn/softwares/PypeR/) ，因为它运行嵌入式 R。换句话说，它允许 Python 和 R 对象之间通过 rpy2.robjects 进行通信，稍后我们将看到一个将 pandas 数据帧转换为 R 数据帧的特定示例。如果您在以下任何步骤中遇到困难，请阅读[官方](https://rpy2.github.io/doc/latest/html/index.html)文档或参考资料。

我们将介绍在 Python 中开始使用 R 的三个步骤。最后，我们将做一个实际的例子，并涵盖 rpy2 包允许您处理的更多功能。

1.  安装 R 包。
2.  从 r 导入包和函数。
3.  将 pandas 数据帧转换为 R 数据帧，反之亦然。
4.  实际例子(运行贝叶斯网络)。

但是首先，我们应该安装 rpy2 软件包。

```
# Jupyter Notebook option
!pip install rpy2
# Terminal option
pip install rpy2
```

## 1.安装 R 包

在 R 中，安装包是通过从 CRAN mirrors 下载然后在本地安装来实现的。与 Python 模块类似，可以安装并加载包。

```
# Choosing a CRAN Mirror
import rpy2.robjects.packages as rpackages
utils = rpackages.importr('utils')
utils.chooseCRANmirror(ind=1)# Installing required packages
from rpy2.robjects.vectors import StrVector
packages = ('bnlearn',...,'other desired packages')
utils.install_packages(StrVector(packages))
```

通过在*`chosseCRANmirror`*中选择`ind = 1`*，我们保证自动重定向到离我们位置最近的服务器。现在，我们将讨论第二步。***

## ***2.导入包和函数***

***这里，我们将导入在实际例子中执行贝叶斯网络所需的库和函数。***

```
***# Import packages
from rpy2.robjects.packages import importr
base, bnlearn = importr('base'), importr('bnlearn')# Import Functions
bn_fit, rbn = bnlearn.bn_fit, bnlearn.rbn
hpc, rsmax2, tabu = bnlearn.hpc, bnlearn.rsmax2, bnlearn.tabu***
```

***为了导入任何函数，在每个包的字典中查看“rpy2”键是很方便的，例如，在我们运行的 bnlearn 上查看可导入的函数:***

```
***bnlearn.__dict__['_rpy2r']Output:
...
...
'bn_boot': 'bn.boot',
  'bn_cv': 'bn.cv',
  'bn_cv_algorithm': 'bn.cv.algorithm',
  'bn_cv_structure': 'bn.cv.structure',
  'bn_fit': 'bn.fit',
  'bn_fit_backend': 'bn.fit.backend',
  'bn_fit_backend_continuous': 'bn.fit.backend.continuous',
...
...***
```

***有关如何导入函数 checkout [4]或[5]的更多信息。***

## ***3.将 pandas 数据帧转换为 R 数据帧，反之亦然***

***就我个人而言，我认为这个功能允许您将可伸缩性(python)与统计工具(R)结合起来。举一个个人的例子，当我使用多处理 python 库实现并行计算时，除了 statsmodels Python 包的函数之外，我还想尝试 forecast R 库中的`auto.arima()`函数来进行预测。因此，`robjects.conversion`允许人们融合两种编程语言的精华。***

```
***# Allow conversion
import rpy2.robjects as ro
from rpy2.objects import pandas2ri
pandas2ri.activate()# Convert to R dataframe
r_dt = ro.conversion.py2rpy(dt) # dt is a pd.DataFrame object# Convert back to pandas DataFrame        
pd_dt = ro.conversion.rpy2py(r_dt)***
```

***当激活熊猫转换(`pandas2ri.activate()`)时，许多 R 到熊猫的转换将自动完成。然而，对于显式转换，我们调用`py2rpy`或`rpy2py`函数。***

## ***4.贝叶斯网络的实际例子***

***除了蒙特卡罗方法，贝叶斯网络也是模拟数据的一种选择。然而，到目前为止，Python 中还没有可用于此任务的库。因此，我选择了 bnlearn 包，它可以学习贝叶斯网络的图形结构，并从中进行推理。***

***在下面的例子中，我们使用混合算法(`rsmax2`)来学习网络的结构，因为它允许我们使用基于约束和基于分数的算法的任意组合。然而，根据问题的性质，你应该选择正确的启发式算法，可用算法的完整列表见[7]。一旦网络被学习，我们用`rbn`函数从贝叶斯网络中模拟 n 个随机样本。最后，我们执行一个 try-except 结构来处理特定类型的错误。***

```
***r_imputados = robjects.conversion.py2rpy(imputados)                

try:   
    # Learn structure of Network
    structure = rsmax2(data, restrict = 'hpc', maximize = 'tabu')       

    fitted = bn_fit(structure, data = data, method = "mle")                                               

    # Generate n number of observations
    r_sim = rbn(fitted, n = 10)

except rpy2.rinterface_lib.embedded.RRuntimeError:
    print("Error while running R methods")***
```

***当我们不希望函数失败或做一些意想不到的事情时，就会发生 RunTimeError 。在这种情况下，我们捕获这个错误是因为这是一种在出错时通知用户这不是另一种错误的方式(完整的异常参见[9])。作为一个例子，我得到了在运行`rsmax2` 函数时没有找到`hybrid.pc.filter` hybrid.pc.filter 的错误。***

## ***更多功能***

***使用 rpy2 低级接口和高级接口，您可以做更多的事情。例如，您可以用 R 调用 python 函数，让我们看看如何通过共轭梯度法找到四维 Colville 函数的最小值。***

```
*****from** **rpy2.robjects.vectors** **import** FloatVector
**from** **rpy2.robjects.packages** **import** importr
**import** **rpy2.rinterface** **as** **ri** stats = importr('stats')

*# Colville f: R^4 ---> R*
**def** Colville(x):
    x1, x2, x3, x4 = x[0], x[1], x[2], x[3]

    **return**   100*(x1**2-x2)**2 + (x1-1)**2+(x3-1)**2 + 90*(x3**2-x4)**2 + 10.1*((x2-1)**2 + (x4-1)**2) + 19.8*(x2-1)*(x4-1)

*# Expose function to R*
Colville = ri.rternalize(Colville)

*# Initial point*
init_point = FloatVector((3, 3, 3, 3))

*# Optimization Function*
res = stats.optim(init_point, Colville, method = c("CG"))***
```

> ****参考文献:****

**[1]马特·布朗。“使用 reticulate 包从 R 运行 Python 代码”。r 酒吧。网址:[https://r studio-pubs-static . S3 . amazonaws . com/407460 _ 396 f 867 ce 3494d 479 FD 700960879 e22c . html](https://rstudio-pubs-static.s3.amazonaws.com/407460_396f867ce3494d479fd700960879e22c.html)**

**[2] Ajay Ohri。“结合使用 Python 和 r: 3 种主要方法”。KDnuggets。网址:[https://www . kdnugges . com/2015/12/using-python-r-together . html](https://www.kdnuggets.com/2015/12/using-python-r-together.html)**

**[3] Rpy2 官方文档。网址:【https://rpy2.github.io/doc/latest/html/index.html **

**[4][https://stack overflow . com/questions/59462337/importing-any-function-from-an-r-package-into-python/59462338 # 59462338](https://stackoverflow.com/questions/59462337/importing-any-function-from-an-r-package-into-python/59462338#59462338)**

**[5][https://stack overflow . com/questions/49776568/calling-functions-from-in-r-packages-in-python-using-importr](https://stackoverflow.com/questions/49776568/calling-functions-from-within-r-packages-in-python-using-importr)**

**[6][https://stack overflow . com/questions/47306899/how-do-I-catch-an-rpy 2-rinter face-rruntimeerror-in-python](https://stackoverflow.com/questions/47306899/how-do-i-catch-an-rpy2-rinterface-rruntimeerror-in-python)**

**[7]bn 学习官方文档。[http://www . bn learn . com/documentation/man/structure . learning . html](http://www.bnlearn.com/documentation/man/structure.learning.html)**

**[8]丹尼尔·厄姆。“贝叶斯网络与 bnlearn 包的例子”。网址:[http://gradient descending . com/Bayesian-network-example-with-the-bn learn-package/](http://gradientdescending.com/bayesian-network-example-with-the-bnlearn-package/)**

**[9] Python 3.8 内置异常。URL:[https://docs . python . org/3.8/library/exceptions . html # runtime error](https://docs.python.org/3.8/library/exceptions.html#RuntimeError)**

**[10]罗伯特，克里斯蒂安；卡塞拉，乔治。介绍蒙特卡罗方法。2010**

**[11] Nagarajan，Radhakrishnan 马可·斯库塔里；太棒了，索菲。贝叶斯网络。2013**