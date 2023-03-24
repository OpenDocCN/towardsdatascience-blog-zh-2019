# 用 Python 编程识别支持/阻力趋势线

> 原文：<https://towardsdatascience.com/programmatic-identification-of-support-resistance-trend-lines-with-python-d797a4a90530?source=collection_archive---------1----------------------->

# 背景

确定支撑位和阻力位的趋势线分析传统上是由经济学家在图表(如特定证券的收盘价图表)上手工画线来完成的。这种任务的计算机化自动化在大量的图书馆中还没有得到很好的实现。这种任务的开始需要得到[趋势线](https://en.wikipedia.org/wiki/Trend_line_(technical_analysis))的客观定义:

“在[金融](https://en.wikipedia.org/wiki/Finance)中，**趋势线**是[证券](https://en.wikipedia.org/wiki/Security_(finance))价格运动的边界线。当在**至少三个或更多价格支点**之间画一条对角线时，它就形成了。可以在任意两点之间画一条线，但是在测试之前它不能作为趋势线。因此需要第三点，即测试。”

几乎所有琐碎的尝试都达不到三点或三点以上的要求。因此，我们将讨论实现这一点的每个方面背后的技术细节。这可以进一步用于识别各种图表模式，例如三角形，包括三角形、楔形、三角旗和平行形状，例如旗帜、双/三重顶/底、头和肩、矩形，尽管所有这些通常只需要 2 个点，但是如果在适当的地方找到 3 个点，无疑将具有加强的指示。这个有趣的话题留到以后讨论。

# 设置

我们将使用 [Python](https://www.python.org/) 3(尽管从 2.7 开始的任何版本都足够了)来完成这项任务，因为它有很好的库来处理数据集，并且非常容易使用。首先，必须获取股票数据。对于这一系列的例子，标准普尔 P500(股票代码: [^GSPC](https://finance.yahoo.com/quote/%5EGSPC/) )的普遍指数将被用来具有实际的相关性。

Python 库 [yfinance](https://github.com/ranaroussi/yfinance) 可以非常容易地获取整个历史数据。例如，取略少于 4 年的数据或最近 1000 个点是方便的。它会将数据作为 [pandas](https://pandas.pydata.org/) DataFrame 返回，以便于索引和其他操作；

```
import numpy as np
import pandas as pd
import yfinance as yf #pip install yfinance
tick = yf.Ticker('^GSPC')
hist = tick.history(period="max", rounding=True)
#hist = hist[:'2019-10-07']
hist = hist[-1000:]
h = hist.Close.tolist()
```

为了获得可重复的结果，数据集是在 2019 年 10 月 7 日截止时采集的，并显示了对此效果的评论过滤。

# 轴心点

第一个问题是确定支点。我们的点数将是给定时间的收盘价。我们可以将图表中的这些点称为波峰和波谷，或者称为局部最大值和局部最小值。

## 所有波峰和波谷朴素方法

有一种简单的方法可以做到这一点，因为枢轴点要求前后的点都比当前点低或高。一个简单的方法有严重的缺点，但是，如果价格连续两天保持不变，就不会检测到波峰或波谷。然而，没有发生这种情况的指数可以相对容易地计算出来:

```
minimaIdxs = np.flatnonzero(
 hist.Close.rolling(window=3, min_periods=1, center=True).aggregate(
   lambda x: len(x) == 3 and x[0] > x[1] and x[2] > x[1])).tolist()
maximaIdxs = np.flatnonzero(
 hist.Close.rolling(window=3, min_periods=1, center=True).aggregate(
   lambda x: len(x) == 3 and x[0] < x[1] and x[2] < x[1])).tolist()
```

## 所有波峰和波谷处理连续重复的方法

当然，折叠价格保持不变的位置和具有不同索引检索的先前代码将克服这一缺点，并产生所有这些位置，同时识别折叠的平坦区间中的单个时间。在上述计算之前删除连续的重复项非常简单:`hist.Close.loc[hist.Close.shift(-1) != hist.Close]`。但是，索引需要重新计算，因为它们可能已经改变:

```
hs = hist.Close.loc[hist.Close.shift(-1) != hist.Close]
x = hs.rolling(window=3, center=True)
     .aggregate(lambda x: x[0] > x[1] and x[2] > x[1])
minimaIdxs = [hist.index.get_loc(y) for y in x[x == 1].index]
x = hs.rolling(window=3, center=True)
     .aggregate(lambda x: x[0] < x[1] and x[2] < x[1])
maximaIdxs = [hist.index.get_loc(y) for y in x[x == 1].index]
```

这两种方法都可以使用列表和循环来完成，而不是使用 Pandas 滚动窗口技术，并且实际上通过简单的实现更改会快得多。尽管 Pandas 提供了简短优雅的代码片段，但利用它们并不总是最有效的，尤其是使用回调的聚合函数。

## 数值微分法

然而，解决这个问题的一个更好的方法是使用收盘价的[数字导数](https://en.wikipedia.org/wiki/Numerical_differentiation)来识别点。一阶导数是收盘价的变化率或有效动量或速度，而二阶导数表示一阶导数或其加速度的变化率。正态微分在这里并不适用，因为离散时间序列需要离散的数值分析工具。有几个优点，包括[数值导数](https://en.wikipedia.org/wiki/Numerical_differentiation)将通过考虑从要计算变化率的点开始的给定范围内的所有点来平滑数据。例如，5 点模板方法考虑了点本身之前和之前的一些增量，以及点本身之前和之前的双重增量。findiff 库使这种计算变得简单而精确，即使使用更高阶的近似方法:

```
from findiff import FinDiff #pip install findiff
dx = 1 #1 day interval
d_dx = FinDiff(0, dx, 1)
d2_dx2 = FinDiff(0, dx, 2)
clarr = np.asarray(hist.Close)
mom = d_dx(clarr)
momacc = d2_dx2(clarr)
```

通过计算一阶和二阶导数，有效地实现了一定程度的平滑，给出了显著的波峰和波谷。它们是一阶导数为 0 的地方，因为没有动量表示方向发生了变化。正的或负的二阶导数分别表示波谷或波峰，因为向上与向下的加速度表示方向相反。然而，0 的精确一阶导数是非常不可能的。相反，实际上，0 一侧的值后面跟着另一侧的值，因为 0 导数点出现在两天之间。因此，基于这一点，较高或较低的收盘价将在发生 0 交叉的两点之间选择:

```
def get_extrema(isMin):
  return [x for x in range(len(mom))
    if (momacc[x] > 0 if isMin else momacc[x] < 0) and
      (mom[x] == 0 or #slope is 0
        (x != len(mom) - 1 and #check next day
          (mom[x] > 0 and mom[x+1] < 0 and
           h[x] >= h[x+1] or
           mom[x] < 0 and mom[x+1] > 0 and
           h[x] <= h[x+1]) or
         x != 0 and #check prior day
          (mom[x-1] > 0 and mom[x] < 0 and
           h[x-1] < h[x] or
           mom[x-1] < 0 and mom[x] > 0 and
           h[x-1] > h[x])))]
minimaIdxs, maximaIdxs = get_extrema(True), get_extrema(False)
```

相当长的讨论，并且如果所有的最小值和最大值点都是期望的，那么可以组合来自朴素方法的指数，因为它将捕获在动量计算期间被平滑的那些指数。但是对于趋势线来说，突出的支点通常是理想的，其他的大多是嘈杂的或不相关的。这是一个小图，显示了 10 天的速度和加速度值以及确定的支点。

![](img/f21f075195a4be3c549e4e84f5bf3a42.png)

Closing Price with Pivot Points, Momentum, Acceleration

现在，对于那些不希望将这种计算仅仅留给 findiff 这样的库的数学好奇者，我将建议如何计算这一数据中的单个点(2019-09-27 收于 2961.79，其先前为 2977.62，其后续为 2976.74)。它计算自己的系数来启用[高阶方法](https://en.wikipedia.org/wiki/Numerical_differentiation#Higher-order_methods)。

大多数人将导数理解为 y 相对于 x 的变化(δ—△)。对于连续线，这仅仅是线的斜率，对于后续点，通过取 y 值的差来计算是微不足道的。但是导数实际上是当△x 趋近于 0 时△x 对△y 的极限。对离散数据点进行这样的限制实际上需要扩大被观察的点的数量。

这意味着，这里存在技术上的数据泄漏，尽管这是一个无关紧要的小问题，因为未来的值是根据过去的数据来考虑的。这很微妙，也不是特别重要，但是如果在数值导数的时间间隔窗口内使用最近的数据来寻找趋势线，也许提到的不同技术会更好。我们在谈论多少天？这取决于准确性，默认情况下，窗口的每一边只有 1 天，除了最左边和最右边的值会提前或滞后 2 天。

因此，我将给出显示一个中心点的手工计算的代码:

```
import findiff
coeff = findiff.coefficients(deriv=1, acc=1)
print(coeff)
```

这将产生:

```
{‘center’: {‘coefficients’: array([-0.5, 0\. , 0.5]),
‘offsets’: array([-1, 0, 1])},
‘forward’: {‘coefficients’: array([-1.5, 2\. , -0.5]),
‘offsets’: array([0, 1, 2])},
‘backward’: {‘coefficients’: array([ 0.5, -2\. , 1.5]),
‘offsets’: array([-2, -1, 0])}}
```

所以 findiff 对所有的中心点使用-0.5，0，0.5 的系数和-1，0，1 的时间差。当然，在最右边和最左边，它使用向前和向后的值。计算所有的导数很简单。计算所谓的[有限差分系数](https://en.wikipedia.org/wiki/Finite_difference_coefficient)和窗口大小的细节不在这里的范围内，但是有[表格和容易计算它们的方法](https://en.wikipedia.org/wiki/Finite_difference_coefficient)。

```
hist = hist[:'2019–10–07']
day = 23043 # September 27, 2019 per example (-7 or 7th last point)
sum([coeff[‘center’][‘coefficients’][x] *
     hist.Close[day + coeff[‘center’][‘offsets’][x]]
     for x in range(len(coeff[‘center’][‘coefficients’]))])
```

结果如图所示:

*   -0.44000000000005457=-2977.62/2+2976.74/2

对于加速度:

```
coeff=findiff.coefficients(deriv=2, acc=1)
print(coeff)
```

产量:

```
{‘center’: {‘coefficients’: array([ 1., -2., 1.]),
‘offsets’: array([-1, 0, 1])},
‘forward’: {‘coefficients’: array([ 2., -5., 4., -1.]),
‘offsets’: array([0, 1, 2, 3])},
‘backward’: {‘coefficients’: array([-1., 4., -5., 2.]),
‘offsets’: array([-3, -2, -1, 0])}}
```

再说一遍:

```
sum([coeff[‘center’][‘coefficients’][x] *
     hist.Close[day + coeff[‘center’][‘offsets’][x]]
     for x in range(len(coeff[‘center’][‘coefficients’]))])
```

我们还看到了期望值:

*   30.779999999999745=2977.62–2961.79*2+2976.74

精确度可以提高，虽然看起来好像使用了默认值 1，所以它不是我给你指出的 5 点模板，但它有自己的技术来生成偏移和系数，维基百科也有一些细节。已经划分的 5 点模板系数可以通过`print(findiff.coefficients(deriv=1, acc=3))`找到。然后，它会向前和向后看 2 天，而不是只看 1 天，其中的`[1/12, -2/3, 0, 2/3, -1/12]`完全相同，只是顺序相反。注意，`acc=3`可以作为附加参数传递给 FinDiff 构造函数，以获得更高的准确性。

# 趋势线方法

最终得到枢轴点后，选择其中的 2 个将非常容易，因为任何一对最小值或最大值点将构成一条线，总共有`n(n-1)`条线，这些线都是完美的拟合线，因为基本几何学表明 2 个唯一的点是一条线的一个描述。然而，可以通过`n(n-1)(n-2)`方式选择 3 个点，并且这种枚举的算法复杂度为 O(n)。这些点不会总是在一条直线上，所以也会有一定程度的误差。误差的程度将是重要的，因为我们需要过滤掉不正确的点集，并包括正确的点集。

基本几何现在将发挥作用，因为线可以更方便地表示为斜率和截距，因此可以很容易地计算线上任何位置的点。回想一下，对于 2 个点，具有 x 轴和 y 轴的二维平面中直线的斜率 m 定义为 y 的变化除以 x 的变化，截距 b 通过使用其中一个点来拟合标准直线方程来计算。幸运的是，我们可以使用斜率截距符号，因为在时间序列数据中没有无限的斜率，每个时间只有一个点。否则，可以使用带有角度-距离的极坐标符号。根据勾股定理给出的直角三角形，两点之间的距离是从由这些点形成的矩形的对角线开始的直线距离。

对于 3 个点，如果我们将直线拟合到 2 个点，我们可以计算从该直线到 y 轴上第 3 个点的距离大小。然而，这个距离作为误差度量是相当任意的，因为它会根据我们从 3 个可能的选择中选择哪 2 个点而不同，另外 2 个是:第一个和最后一个或第二个和第三个。所有这些公式及其示例都以可视化方式给出:

![](img/5ffc6ca26e9a6d9aa7f1f1f7c72667c9.png)

Closing Price Points Demonstrating Line Calculations

给定 3 个点，可以找到最佳拟合线，这正是[线性回归](https://en.wikipedia.org/wiki/Regression_analysis#Linear_regression)所计算的。它基于最小化误差找到最适合 3 个或更多点的线。误差通常是残差的平方和，其中残差等于上面给出的距离度量。所需的主要等式是直线的斜率，根据该斜率，使用 y 和 x 的平均值按照上述方法计算截距。

给出了两个标准误差，一个是斜率误差，另一个是截距误差。对于斜率，误差基于残差平方和(SSR ),部分通过除以点数减去 2 来计算，其中在这种情况下点数为 3，使得该项消失，因此它仅仅是 y 差的平方和的平方根除以 x 差的平方和。截距误差是由 x 值调整的斜率误差。斜率误差是误差的充分指标，因此将应用它。应该注意的是，SSR、斜率和截距误差假设点的方差是恒定的，但实际情况可能并非如此。假设标准斜率和截距误差呈正态分布，这也可能不成立，这将表明 68%的点在该误差值的正负范围内，根据[标准偏差钟形曲线](https://en.wikipedia.org/wiki/Normal_distribution#Standard_deviation_and_coverage)，两倍误差为 95%，三倍误差为 99.7%。既然方差不能保证，也不是正态分布，为什么还要使用 SSR 衍生的东西呢？因为我们不是假设分布，而仅仅是将误差值与固定百分比误差进行比较，所以我们的用例使斜率误差足以进行选择。SSR 仍然可以保留，因为它也可以用于相同的目的，尽管具有非常相似的结果。然而，对期望值和实际值做一个简单的直线平均是最简单的，也是可以接受的。

![](img/f171aff3a6a3cef905b87e0bd7830197.png)

Closing Price Points Demonstrating Linear Regression

在 Python 中，3 点变量可以有效地编码为:

```
def get_bestfit3(x0, y0, x1, y1, x2, y2):
  xbar, ybar = (x0 + x1 + x2) / 3, (y0 + y1 + y2) / 3
  xb0, yb0, xb1, yb1, xb2, yb2 =
    x0-xbar, y0-ybar, x1-xbar, y1-ybar, x2-xbar, y2-ybar
  xs = xb0*xb0+xb1*xb1+xb2*xb2
  m = (xb0*yb0+xb1*yb1+xb2*yb2) / xs
  b = ybar - m * xbar
  ys0, ys1, ys2 =
    (y0 - (m * x0 + b)),(y1 - (m * x1 + b)),(y2 - (m * x2 + b))
  ys = ys0*ys0+ys1*ys1+ys2*ys2
  ser = np.sqrt(ys / xs)
  return m, b, ys, ser, ser * np.sqrt((x0*x0+x1*x1+x2*x2)/3)
```

然而，出于一般性考虑，但以速度为代价，我们将对任意数量的点实现这一点，因为更好的趋势线可能具有甚至多于 3 个点，因为 3 只是最小值:

```
def get_bestfit(pts):
  xbar, ybar = [sum(x) / len (x) for x in zip(*pts)]
  def subcalc(x, y):
    tx, ty = x - xbar, y - ybar
    return tx * ty, tx * tx, x * x
  (xy, xs, xx) =
    [sum(q) for q in zip(*[subcalc(x, y) for x, y in pts])]
  m = xy / xs
  b = ybar - m * xbar
  ys = sum([np.square(y - (m * x + b)) for x, y in pts])
  ser = np.sqrt(ys / ((len(pts) - 2) * xs))
  return m, b, ys, ser, ser * np.sqrt(xx / len(pts))
```

更普遍的是， [numpy](https://numpy.org/) 库有 [polyfit](https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html) 和 [poly1d](https://docs.scipy.org/doc/numpy/reference/generated/numpy.poly1d.html) 函数，它们可以对任何多项式做同样的事情，在这种情况下，多项式是 1 的线或次数。我们将使用它来计算平均支撑线和平均阻力线，分别基于所有的局部最小值和最大值点。斜率和截距由 [polyfit](https://docs.scipy.org/doc/numpy/reference/generated/numpy.polyfit.html) 和 [poly1d](https://docs.scipy.org/doc/numpy/reference/generated/numpy.poly1d.html) 返回，尽管 [poly1d](https://docs.scipy.org/doc/numpy/reference/generated/numpy.poly1d.html) 提供了更简单的操作和使用，因此证明了返回的误差仅仅是剩余误差的平方和，但用于比较的目的是一致的。虽然适用于单一用途，如整体平均值，但这种通用函数可能没有上面写的优化函数快，因此需要练习编写它们。当然，至少有 2 个点必须符合这条线。

```
ymin, ymax = [h[x] for x in minimaIdxs], [h[x] for x in maximaIdxs]zmin, zmne, _, _, _ = np.polyfit(minimaIdxs, ymin, 1, full=True)  #y=zmin[0]*x+zmin[1]
pmin = np.poly1d(zmin).c
zmax, zmxe, _, _, _ = np.polyfit(maximaIdxs, ymax, 1, full=True) #y=zmax[0]*x+zmax[1]
pmax = np.poly1d(zmax).c
print((pmin, pmax, zmne, zmxe))
```

如果更喜欢按照文档使用数值更稳定的代码，那么也有[多项式. fit](https://docs.scipy.org/doc/numpy/reference/generated/numpy.polynomial.polynomial.Polynomial.fit.html) 版本:

```
p, r = np.polynomial.polynomial.Polynomial.fit
  (minimaIdxs, ymin, 1, full=True) #more numerically stable
pmin, zmne = list(reversed(p.convert().coef)), r[0]
p, r = np.polynomial.polynomial.Polynomial.fit
  (maximaIdxs, ymax, 1, full=True) #more numerically stable
pmax, zmxe = list(reversed(p.convert().coef)), r[0]
```

但是，由于误差的绝对值是相对于时间段和该时间段内的价格范围而言的，因此误差不能一致地应用于所有证券。所以首先应该计算一个合适的比例:`scale = (hist.Close.max() — hist.Close.min()) / len(hist)`。(如果仅仅取剩余误差的平方根并除以(n-2 ),那么这里除以`len(hist)`是不必要的。)那么趋势线函数的参数`errpct` 将是简单的百分比误差，例如 0.5%=0.005，其中`fltpct=scale*errpct`。应该处理其他细微差别，例如斜率 0 不是作为系数返回的，必须手动填充。

## 朴素方法

该方法将简单地枚举 3 个点的所有组合，以找到相关的误差，并过滤掉误差值太大的那些。当然，O(n)并不理想，如果数据集足够大，实际上也不可行。但是实际上总的最小值和总的最大值点不会大到不可能的程度。例如，100 个轴心点就是 100*100*100=1，000，000 或一百万次计算。显然，用 4 或 5 分来加强这一点将开始变得不切实际，因为复杂性的顺序去了 O(n⁴)或 O(n⁵).因此，需要一种不同的策略。

```
def get_trend(Idxs):
  trend = []
  for x in range(len(Idxs)):
    for y in range(x+1, len(Idxs)):
      for z in range(y+1, len(Idxs)):
        trend.append(([Idxs[x], Idxs[y], Idxs[z]],
          get_bestfit3(Idxs[x], h[Idxs[x]],
                       Idxs[y], h[Idxs[y]],
                       Idxs[z], h[Idxs[z]])))
  return list(filter(lambda val: val[1][3] <= fltpct, trend))
mintrend, maxtrend = get_trend(minimaIdxs), get_trend(maximaIdxs)
```

## 排序斜率法

幸运的是，这些点的某些属性可以相互关联，例如直线的斜率(或与原点的角度)。通过对 2 个点的所有组合进行 O(n)遍历，并计算它们形成的线的斜率，可以生成每个点的斜率列表。对于像合并排序这样的高效排序算法，对列表进行排序的最坏情况复杂度是 O(n log n ),我们需要对所有 n 个列表进行排序，复杂度是 O(n log n)。排序后的斜率列表中的相邻点可以被认为是从 3 到连续的，然而许多点继续满足过滤标准。一旦匹配，该组将被删除，搜索将继续。这部分算法也是 O(n)。(最大的复杂性因素通常是唯一要考虑的因素，以保持公式简化，从而不添加其他 2 O(n)。)

这是一种近似算法，并不详尽，因为当点之间的距离很大时，按斜率排序并不能保证相邻的斜率值具有最佳拟合。然而，在实践中，这种情况很少发生。

```
def get_trend_opt(Idxs):
  slopes, trend = [], []
  for x in range(len(Idxs)): #O(n^2*log n) algorithm
    slopes.append([])
    for y in range(x+1, len(Idxs)):
      slope = (h[Idxs[x]] - h[Idxs[y]]) / (Idxs[x] - Idxs[y])
      slopes[x].append((slope, y))
  for x in range(len(Idxs)):
    slopes[x].sort(key=lambda val: val[0])
    CurIdxs = [Idxs[x]]
    for y in range(0, len(slopes[x])):
      CurIdxs.append(Idxs[slopes[x][y][1]])
      if len(CurIdxs) < 3: continue
      res = get_bestfit([(p, h[p]) for p in CurIdxs])
      if res[3] <= fltpct:
        CurIdxs.sort()
        if len(CurIdxs) == 3:
          trend.append((CurIdxs, res))
          CurIdxs = list(CurIdxs)
        else: CurIdxs, trend[-1] = list(CurIdxs), (CurIdxs, res)
      else: CurIdxs = [CurIdxs[0], CurIdxs[-1]] #restart search
  return trend
mintrend, maxtrend =
  get_trend_opt(minimaIdxs), get_trend_opt(maximaIdxs)
```

事实上，许多 3 分的匹配会立即被 4 分甚至更多的点所取代，因为不出所料，在真实的证券或指数数据中确实会出现趋势。

## 霍夫线变换法

存在尝试解决线寻找算法的替代方法，并且一种来自图像处理，其中在图像中寻找线是计算机视觉中常见且重要的任务。这样做的一种方法是霍夫线变换，该变换寻找以特定角度穿过线的点，同时根据找到的点的数量对它们进行评分。这个算法也有局限性。不幸的是，它使用大量的内存来跟踪所有的直方图，并且它的准确性是基于尝试了多少个角度。尝试的角度越多，速度就越慢。记忆基于图像的对角线尺寸(用`width` 乘`height` 尺寸)和角度数量(`numangles`或`ceil(sqrt(width*width+height*height)) * numangles`)。为了获得好的结果，缩放图像是必要的。事实上，`arctan(2/width)`和`arctan(2/height)`中较小的一个将是寻找所有 3 点可能性的最小角度，因为它是垂直线或水平线之间的最小增量。通过将算法固定为 90 度和-90 度之间的 360*5 个可能角度，我们可以使用`int(np.ceil(2/np.tan(np.pi / (360 * 5))))`来找到最大图像尺寸，并且如果图像超出界限，就仅仅缩放这个量。

我们首先将时间序列数据改编成图像。这需要将价格值离散化，可以通过乘以 100 来消除金额的小数部分。该图像只需要按时间长度和最低和最高价格在这段时间的大小。整个图像将被初始化为黑色，白色点将被设置在最小值或最大值点出现的适当位置。

```
def make_image(Idxs):
  max_size = int(np.ceil(2/np.tan(np.pi / (360 * 5)))) #~1146
  m, tested_angles =
    hist.Close.min(), np.linspace(-np.pi / 2, np.pi / 2, 360*5)
  height = int((hist.Close.max() - m + 0.01) * 100)
  mx = min(max_size, height)
  scl = 100.0 * mx / height
  image = np.zeros((mx, len(hist))) #in rows, columns or y, x
  for x in Idxs:
    image[int((h[x] - m) * scl), x] = 255
  return image, tested_angles, scl, m
```

霍夫变换作为一种点输入算法，在 Python 中很容易实现。对于所有角度和所有点，计算到垂直于穿过该点的角度的直线的距离。对于每个角度，到这条垂直线的距离是累积的一个点，其中具有相同几何距离的不同点必须位于一条直线上。

![](img/5c79670fe72e1023d462ba9b1d3c3b87.png)

Closing Price Points Demonstrating Hough transform accumulation of rho-theta for 2 point line whose origin is based as the first day and minimal price shown

这导致 O(n*m)算法，其中 m 是角度的数量，而存储器使用需要 m 倍于 2 维点空间的对角线长度:

```
def hough_points(pts, width, height, thetas):
  diag_len = int(np.ceil(np.sqrt(width * width + height * height)))
  rhos = np.linspace(-diag_len, diag_len, diag_len * 2.0)
  # Cache some resuable values
  cos_t = np.cos(thetas)
  sin_t = np.sin(thetas)
  num_thetas = len(thetas)
  # Hough accumulator array of theta vs rho
  accumulator =np.zeros((2 * diag_len, num_thetas), dtype=np.uint64)
  # Vote in the hough accumulator
  for i in range(len(pts)):
    x, y = pts[i]
    for t_idx in range(num_thetas):
      # Calculate rho. diag_len is added for a positive index
      rho=int(round(x * cos_t[t_idx] + y * sin_t[t_idx])) + diag_len
      accumulator[rho, t_idx] += 1
  return accumulator, thetas, rhos
```

保持记忆的霍夫变换不返回任何特定的点信息，这对于可视化的目的是非常有用的。因此，我们可以计算所有点到直线的距离，对其进行排序，并尽可能多地选取误差容差范围内的点。一个点到一条线的[距离用于确定要增加的正确累加器。记住，垂直斜率的斜率是斜率的负倒数，所以它们的乘积等于-1。相对于新点的点构造一条垂直线，并计算交点。证明的其余部分是使用点和交点之间的距离导出的。请注意，分子仅仅是 y 和预期 y 之差，而分母保持不变，因为我们只考虑一个直线斜率。显示霍夫变换的图也给出了明确的公式。](https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line)

```
def find_line_pts(Idxs, x0, y0, x1, y1):
  s = (y0 - y1) / (x0 - x1)
  i, dnm = y0 - s * x0, np.sqrt(1 + s*s)
  dist = [(np.abs(i+s*x-h[x])/dnm, x) for x in Idxs]
  dist.sort(key=lambda val: val[0])
  pts, res = [], None
  for x in range(len(dist)):
    pts.append((dist[x][1], h[dist[x][1]]))
    if len(pts) < 3: continue
    r = get_bestfit(pts)
    if r[3] > fltpct:
      pts = pts[:-1]
      break
    res = r
  pts = [x for x, _ in pts]
  pts.sort()
  return pts, res
```

我们也可以使用名为 [hough_line](https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.hough_line) 的 [scikit-image](https://scikit-image.org/) 库的 Hough line 变换函数。需要注意的是，Python 中同样可用的 [OpenCV](https://opencv.org/) (计算机视觉)库也有同样的功能。这也可以用少量代码手工实现，因为算法并不特别复杂，但是库被优化并且更快。请注意，累加器中只有 2 个点就足以让算法返回它们，因此我们将过滤 3 个或更多相关点。有用的细节是它反转轴，并且具有大于但不等于的阈值参数。其内部功能有一些细微的差异，特别是在累加器中选择局部最大值方面，因此结果不会完全相同。这导致执行计算的两种不同方法(一种是点优化，另一种是将点转换为图像以用于库):

```
def houghpt(Idxs):
  max_size = int(np.ceil(2/np.tan(np.pi / (360 * 5)))) #~1146
  m, tested_angles =
    hist.Close.min(), np.linspace(-np.pi / 2, np.pi / 2, 360*5)
  height = int((hist.Close.max() - m + 1) * 100)
  mx = min(max_size, height)
  scl = 100.0 * mx / height
  acc, theta, d = hough_points(
    [(x, int((h[x] - m) * scl)) for x in Idxs], mx, len(hist),
    np.linspace(-np.pi / 2, np.pi / 2, 360*5))
  origin, lines = np.array((0, len(hist))), []
  for x, y in np.argwhere(acc >= 3):
    dist, angle = d[x], theta[y]
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    y0, y1 = y0 / scl + m, y1 / scl + m
    pts, res = find_line_pts(Idxs, 0, y0, len(hist), y1)
    if len(pts) >= 3: lines.append((pts, res))
  return lines
mintrend, maxtrend = houghpt(minimaIdxs), houghpt(maximaIdxs)def hough(Idxs): #pip install scikit-image
  image, tested_angles, scl, m = make_image(Idxs)
  from skimage.transform import hough_line, hough_line_peaks
  h, theta, d = hough_line(image, theta=tested_angles)
  origin, lines = np.array((0, image.shape[1])), []
  for pts, angle, dist in
                   zip(*hough_line_peaks(h, theta, d, threshold=2)):
    y0, y1 = (dist - origin * np.cos(angle)) / np.sin(angle)
    y0, y1 = y0 / scl + m, y1 / scl + m
    pts, res = find_line_pts(Idxs, 0, y0, image.shape[1], y1)
    if len(pts) >= 3: lines.append((pts, res))
  return lines
mintrend, maxtrend = hough(minimaIdxs), hough(maximaIdxs)
```

## 概率霍夫线变换方法

正常霍夫线识别所使用的精确角度和所需的高处理和存储要求使得对于具有少量点的任务来说有些不切实际。点数越多，效果越好。然而，这种现有方法更多的是精确的练习。在实践中，使用概率霍夫线变换，其中以随机方式进行搜索，并且使用参数来过滤结果，包括点数的阈值。另一个库函数[probabilical _ Hough _ line](https://scikit-image.org/docs/dev/api/skimage.transform.html#skimage.transform.probabilistic_hough_line)用于此目的:

```
def prob_hough(Idxs): #pip install scikit-image
  image, tested_angles, scl, m = make_image(Idxs)
  from skimage.transform import probabilistic_hough_line
  lines = []
  for x in range(hough_prob_iter):
    lines.append(probabilistic_hough_line(image, threshold=2,
                 theta=tested_angles, line_length=0,
      line_gap=int(np.ceil(np.sqrt(
        np.square(image.shape[0]) + np.square(image.shape[1]))))))
  l = []
  for (x0, y0), (x1, y1) in lines:
    if x0 == x1: continue
    if x1 < x0: (x0, y0), (x1, y1) = (x1, y1), (x0, y0)
    y0, y1 = y0 / scl + m, y1 / scl + m
    pts, res = find_line_pts(Idxs, x0, y0, x1, y1)
    if len(pts) >= 3: l.append((pts, res))
  return l
mintrend, maxtrend = prob_hough(minimaIdxs), prob_hough(maximaIdxs)
```

这里仍然只有返回的线的 2 个点，但是阈值已经内置并为我们完成了。不幸的是，由于具有随机性的概率算法的性质，每次运行的结果都会不同。这通常使得该方法不特别适合于寻找所有趋势线。因为它可以运行多次，所以可以这样做，直到运行了一定次数或者找到了一定数量的行，以大大增加找到所有行的概率。因此，参数`hough_prob_iter` 指定了运行它的迭代次数，例如 10 次迭代，以增加足够行数的可能性。

请注意，由于美元在使用中，因此比硬编码 100 或 1/0.01 更好的是有一个`hough_scale`参数。

我们现在有 5 种不同的方法来寻找趋势线。

# 寻找最佳趋势线

不幸的是，仅仅因为局部最小值或最大值点在一条线上，它们可能不总是产生最好的趋势线。这是因为什么被认为是最好的往往是主观的。然而，有一些方法可以把它变成一个客观的定义。一种方法是查看价格穿过趋势线的频率，或者趋势线和价格数据穿过最早和最晚点的面积。误差百分比仅用于确定什么可能是趋势线，什么可能不是趋势线，而不是趋势线是否有用。有了足够的数据，可以发现相隔多年的线，不管它们是否是巧合，这些线都不太可能有用。通常绘制的趋势线包含信号的一侧或另一侧，因此基于区域的方法至少是合理的。一个[黎曼和](https://en.wikipedia.org/wiki/Riemann_sum)提供了一种积分技术，可以用来计算这个面积。事实上，这个应用程序很简单，因为它只是将两者相减，并将所有小于或大于 0 的值相加。具体来说，这将是一个右黎曼和，它使用每个时间点的函数值，而不是中点或前面的最大值或最小值。最后，用它除以天数，给出了每天趋势误差的度量，可以选择最佳趋势线。

![](img/40487d3a63a26fdbe1277f11db8201a5.png)

Closing Price with Resistance and Area

```
def measure_area(trendline, isMin):
  base = trendline[0][0]
  m, b, ser =
    trendline[1][0], trendline[1][1], h[base:trendline[0][-1]+1]
  return sum([max(0, (m * (x+base) + b) - y
               if isMin else y - (m * (x+base) + b))
              for x, y in enumerate(ser)]) / len(ser)
mintrend = [(pts, (res[0], res[1], res[2], res[3], res[4],
             measure_area((pts, res), True)))
            for pts, res in mintrend]
maxtrend = [(pts, (res[0], res[1], res[2], res[3], res[4],
             measure_area((pts, res), False)))
            for pts, res in maxtrend]
mintrend.sort(key=lambda val: val[1][5])
maxtrend.sort(key=lambda val: val[1][5])
print((mintrend[:5], maxtrend[:5]))
```

另一个问题是，由于各种原因，这些算法即使是最好的算法在有效实现时也会很慢。进一步的优化通常是可能的，特别是通过消除 pandas 和转向具有快速 C 实现的列表和 numpy 数组。

然而，真正的问题是趋势线到底是如何被使用的？它们是短期的还是长期的，是当前的还是历史的？这一点非常重要，以至于必须处理计算窗口趋势线的想法。这里到底是怎么做窗户的？

第一个想法是选择一个窗口，比如一个季度或一年，因为窗口太短将不会产生足够有用的支点。然后，窗口必须滚动通过所有的数据，但是，不能以这种方式，其效率低下或重复工作，也不能错过可能的趋势线。一旦确定了窗口大小，就可以通过两倍窗口大小的搜索来开始工作。将窗口大小加倍可以确保找到跨越窗口边界的趋势线。则以窗口大小的增量在所有数据中搜索双窗口大小。因此，如果您的窗口是一年，您搜索过去两年，然后从过去一年之前的 3 年开始搜索，然后从之前的 4 年到之前的 2 年，等等。窗口的一部分将会被搜索两次，许多相同的或相同趋势的延伸将会出现。这在视觉上也会更令人愉快，因为趋势线画得太远会破坏情节。

因此，最后，找到的趋势线必须智能地合并在一起，以获得最终的集合。这可以通过遍历包含每个给定点的所有趋势线并应用前面讨论的斜率排序方法来完成。这对于霍夫变换方法来说是必要的，因为它们会产生冗余的点集。

```
def merge_lines(Idxs, trend):
  for x in Idxs:
    l = []
    for i, (p, r) in enumerate(trend):
      if x in p: l.append((r[0], i))
    l.sort(key=lambda val: val[0])
    if len(l) > 1: CurIdxs = list(trend[l[0][1]][0])
    for (s, i) in l[1:]:
      CurIdxs += trend[i][0]
      CurIdxs = list(dict.fromkeys(CurIdxs))
      CurIdxs.sort()
      res = get_bestfit([(p, h[p]) for p in CurIdxs])
      if res[3] <= fltpct: trend[i-1], trend[i], CurIdxs =
        ([], None), (CurIdxs, res), list(CurIdxs)
      else: CurIdxs = list(trend[i][0]) #restart search from here
  return list(filter(lambda val: val[0] != [], trend))
mintrend, maxtrend = merge_lines(minimaIdxs, mintrend),
                     merge_lines(maximaIdxs, maxtrend)
```

从这个讨论中得到的主要收获是客观看待你的最佳想法，要知道不是所有的人都同意任何一个客观的定义。这有助于在头脑风暴阶段不走任何捷径，以创造一个全面和适当的方法。

# 可视化结果

上面完成的合并过程实际上可以找到趋势线，这些趋势线恰好在相距很远的窗口中具有点，这将大大延长它们，这可能是所期望的，也可能不是所期望的。然而，出于显示的目的，对发现的趋势线重新开窗可能是理想的。不管它们是如何合并的，它们都可以被它们的窗口分开。一般来说，我们希望只画出有限范围内的未来趋势线，以避免弄乱图表。

![](img/dd079255acdad170083feea8ade09dcc.png)

Closing Price with best 2 Support/Resistance Trend Lines by error

请注意图中的底部支撑线从金融角度来看是正确和有用的，因为没有价格跌破它的情况。其他支撑线和阻力线只选择了半突出的峰值，根据趋势线的定义是合理的，但几乎没有什么是有用的。所以误差最小的最好趋势线，不一定是最好的支撑线和阻力线。再次测量该线上方形成的区域对股价构成阻力，该线下方对股价构成支撑，这似乎是关键。

![](img/5fa8ac956bd150552222f6f07998804d.png)

Closing Price with best 2 Support/Resistance Trend Lines by smallest wrong side area per day

这些趋势线对于它们所包含的点的范围来说是非常好的。不显著的局部极值有时仍然会产生有限用途的趋势线，如平坦的阻力线。此外，非常短期的趋势，如非常陡峭的支撑线，可能与未来无关，可以使用进一步的试探法来尝试消除它们，尽管在短期内它可能被认为是有用的。然而，在第三点之后，由于这是一个恢复期，市场逆转，当它被发现时，它的用处已经过时了。另一个最好的支撑线和阻力线绝对是分析师希望看到的。简单地不画出未来可能也有帮助，但对于最近的窗口，整个目的是向未来投射。

最后，不是显示整个周期的 2 条最好的支撑线/阻力线，而是为了显示和推断的目的，可以对最终结果重新开窗，并且可以使用来自每个窗口的 2 条最好的线。

# 用作机器学习的特征

趋势线可用于推断机器学习的特征数据，例如使用 LSTM(长短期记忆)或 GRU(门控递归单元)神经网络，甚至作为 g an(生成对抗网络)的一部分来预测时间序列，以提及一些现实的现代想法。也许它们将首先通过使用 PCA(主成分分析)或自动编码器与其他特征一起被减少。

但是，为了防止数据泄露，找到的任何趋势线都只能用于预测趋势线上最后一个点之后的未来点。在最终点或之前的任何时间，都是数据泄漏的典型例子，其中未来值被用来将值泄漏到过去。即使选择最佳趋势线，也必须仅使用最终点之前的数据仔细完成，尽管可视化没有这样的限制。注意，在处理像时间序列预测这样的精确任务时，即使是典型的编程错误，如计算中的 1 误差，也是致命的错误。记住数字微分的细节，在趋势出现后，根据准确性处理前一两天的数据时，数字微分也是向前看的。假设趋势在最后一个最小值或最大值点后一两天开始会更好，以避免这种泄漏，这取决于趋势的使用方式。

# 把所有的放在一起

trendln 有一个包含所有代码的 [GitHub 存储库。这也可以作为一个](https://github.com/GregoryMorse/trendln) [PyPI 包 trendln](https://pypi.org/project/trendln/) 使用，可以很容易地安装和使用。

依赖项包括 numpy。绘图需要 matplotlib 和熊猫。对于基于图像的霍夫趋势线和概率霍夫趋势线，需要 scikit-image。对于数值微分，findiff 是必需的。本文中生成所有图像的代码也仅作为参考资料。

# 结论

我们已经分析并演示了如何使用包括数值微分在内的不同方法来寻找支点。此外，我们还展示了如何测量线中的误差，以及如何使用各种趋势线识别方法，包括使用排序斜率、霍夫线变换和概率霍夫线变换。现在可以收集和研究支撑和阻力的正确趋势线。它们可以进一步用于更复杂的分析，或用于预测证券定价的变动。

此外，鉴于它将大量信息结合到一个资源中，并对任务的每个方面进行了全面分析，因此有分享这一点的动机，而大多数声称的解决方案甚至没有满足趋势线的基本定义。

如果你喜欢这篇文章，请让我知道，并随时提问。请继续关注刚才提到的形状识别技术。