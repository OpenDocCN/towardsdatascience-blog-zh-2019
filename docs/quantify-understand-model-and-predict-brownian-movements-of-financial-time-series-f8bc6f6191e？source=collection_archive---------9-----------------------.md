# 量化、理解、建模和预测金融时间序列

> 原文：<https://towardsdatascience.com/quantify-understand-model-and-predict-brownian-movements-of-financial-time-series-f8bc6f6191e?source=collection_archive---------9----------------------->

## 确定时间序列动量的赫斯特指数

## 动量识别和计量经济学方法

![](img/30051f7826db4fae304bffab029dc7b8.png)

Image by author

准确预测股票价格是一项艰巨的任务。由于内在的复杂行为，不规则的时间行为在股票市场价格中无处不在。在这里，我将展示一个预测模型框架，用三阶段方法预测金融市场的未来回报。

1.  *分形建模和递归分析，并检验有效市场假说以理解时间行为，从而考察自回归特性。*
2.  *在 VAR 环境下进行格兰杰因果检验，以探索因果互动结构，并确定预测分析的解释变量。*
3.  *ML 和 ANN 算法学习内在模式和预测未来运动。*

我将使用 2000 -2019 年以来的贵金属(黄金和白银)股票价格、能源股票(原油)价格和 S&P500 指数。目标是估计未来收益以及这些产品与 S&P 指数的关系。

## 数据预处理和描述性统计:

在做了所有必要的数据净化后，用以前的值填充丢失的值，重命名列，索引日期等。下面是我们可以做进一步分析和调查的输出。数据净化需要花费大量的时间和精力。因此，我将跳过对数据处理过程的解释，而是专注于这里的目标。

```
if __name__ == "__main__": 
  df = pd.read_excel('brownian data.xlsx', index_col=0)
  # removing unnecessary columns
  df = df.drop(['Date', 'Year', 'NaturalGasPrice'], axis=1) 
  # converting 'DatesAvailable' to datetime
  df['DatesAvailable']= pd.to_datetime(df['DatesAvailable']) 
  # setting 'DatesAvailable' as index column
  df.set_index('DatesAvailable', inplace=True) 
  df1 = pd.read_excel('S&P Futures.xlsx', parse_dates = True,
                      index_col=0)
  DF = df1.loc['20190729':'20000104']
  data = [DF,df] # combining dataframes
  result = pd.concat(data, axis=1)
  # renaming column
  result = result.rename(columns = {"Price": "SPPrice"}) 
  DATA = result.fillna(method ='pad') 
  # filling the missing values with previous ones
  DATA.index.name = 'Date'
  print(DATA.head())
```

![](img/2ff1dd1544a476d71fe5b1b2f841b00d.png)

```
DATA.plot(subplots=True, layout = (2,2),
          figsize = (15,6), sharex = False, grid =True)
plt.tight_layout()
plt.show()
```

![](img/29e24a90a8bd27cf13f87abb28295dfc.png)

所有系列都采用每日收盘价。这些图显示了 S&P 指数的明显趋势；对于本质上相当随机的商品。视觉上可以假设，没有一个序列是稳定的，商品具有布朗运动。为了便于理解，结果以表格的形式给出。

![](img/6b7a9702884e74088ab3b31da7aa8658.png)

偏斜度和峰度值证实没有一个序列遵循正态分布。百分位数和标准差表明指数和黄金价格有很大的差距。如果是由随机波动引起的，这一系列中黄金的巨大价差可能会使准确预测变得困难。

![](img/681824a7df3303f694da3fa883f54ad8.png)

Mann-Whitney test

Mann-Whitney U 检验的零假设是数据样本的分布之间没有差异。

![](img/1fc869917e9723478cea2d0daff8a871.png)

Unit root test Original series

检验统计证实没有一个序列符合正态分布。这也是非参数和非平稳序列的指示，并证明了高级 ML 和 DNN 算法用于预测建模练习的部署。然而，如果时间序列遵循布朗运动，我们不会得到一个稳定的结果。所以，检验 RWH(随机漫步假说)是很重要的。

![](img/5e7a7aa10bec520a145ae3b082fc0c29.png)

Unit root test 1st order difference

结果表明该序列是一阶平稳的。因此，我将考虑回报序列进行计量经济分析，以评估因果关系的方向。

## 非线性动力学:

在这里，我们将执行分形建模和递归量化分析(RQA)来检查 RWH，并获得关于时间演化模式的更深入的见解。让我们看看 FD(分形维数)，R/S(重标范围)，H(赫斯特指数)和 RQA(递归量化分析)。

让我们来了解一下[赫斯特指数](https://en.wikipedia.org/wiki/Hurst_exponent)。赫斯特指数被用来衡量时间序列的长期记忆。赫斯特指数的目标是为我们提供一个标量值，帮助我们识别一个序列是均值回复、随机游走还是趋势。这是一个统计推断。特别是:

*   h< 0:5 — Anti-persistent time-series which roughly translates to mean reverting.
*   H = 0:5 — The time series is random walk and prediction of future based on past data is not possible.
*   H >0:5——持久的时间序列，大致可以解释为趋势。

FD = ln(N)/ln(1/d)，N =圈数，D =直径。这个等式显示了圆的数量与直径的关系。对于时间序列，FD 的值介于 1 和 2 之间。布朗运动的 FD 是 1.5。If 1.5 < FD < 2, then a time series is an anti-persistent process, and if 1 < FD < 1.5, then the series is a long memory process (persistent). H is related to FD (FD =2-H) and a characteristic parameter of long-range dependence. In case of H, the value of 0.5 signifies no long-term memory, < 0.5 means anti-persistence and > 0.5 表示该过程是跨时间相关的，并且是持久的。R/S 是 FD 的中心工具，其计算方法是将其平均调整累积离差序列的范围除以时间序列本身的标准离差(1)。它是表征时间序列散度的度量，定义为给定持续时间(T)的均值中心值范围除以该持续时间(2)的标准偏差。

> 递归量化分析(RQA)可以处理序列中的非平稳性，有助于理解隐藏在金融市场中的复杂动态。

现在，让我们通过计算 REC、DET、TT 和 LAM 等指标来使用量化技术。

## **计算 H:**

```
if __name__ == "__main__":
  index = DATA[['SPIndex']]
  lag1, lag2  = 2,20
  lags = range(lag1, lag2)
  tau = [sqrt(std(subtract(index[lag:], index[:-lag]))) for lag in lags]
  m = polyfit(log(lags), log(tau), 1)
  hurst = m[0]*2

  gold = DATA[['GoldPrice']]
  lag1, lag2  = 2,20
  lags = range(lag1, lag2)
  tau = [sqrt(std(subtract(gold[lag:], gold[:-lag]))) for lag in lags]
  m = polyfit(log(lags), log(tau), 1)
  hurst = m[0]*2 crude = DATA[['CrudeOilPrice']]
  lag1, lag2  = 2,20
  lags = range(lag1, lag2)
  tau = [sqrt(std(subtract(crude[lag:], crude[:-lag]))) for lag in lags]
  m = polyfit(log(lags), log(tau), 1)
  hurst = m[0]*2 silver = DATA[['SilverPrice']]
  lag1, lag2  = 2,20 
  lags = range(lag1, lag2)
  tau = [sqrt(std(subtract(silver[lag:], silver[:-lag]))) for lag in lags]
  m = polyfit(log(lags), log(tau), 1)
  hurst = m[0]*2print('*'*60)
print( 'hurst (Index), 2-20 lags = ',hurst[0])
print('*'*60)
print( 'hurst (Crude), 2-20 lags = ',hurst[0])
print('*'*60)
print( 'hurst (Gold), 2-20 lags = ',hurst[0])
print('*'*60)
print( 'hurst (Silver), 2-20 lags = ',hurst[0])
```

![](img/9f1676f211c00b11d7548b81dbb2e3e4.png)![](img/2eda693758cea1b1121cdcbc8cb6d23d.png)

```
np.random.seed(42)
random_changes = 1\. + np.random.randn(5019) / 1000.
DATA.index = np.cumprod(random_changes)
H, c, result = compute_Hc(DATA.SPIndex, kind='price', 
                          simplified=True)plt.rcParams['figure.figsize'] = 10, 5
f, ax = plt.subplots()
_ = ax.plot(result[0], c*result[0]**H)
_ = ax.scatter(result[0], result[1])
_ = ax.set_xscale('log')
_ = ax.set_yscale('log')
_ = ax.set_xlabel('log(time interval)')
_ = ax.set_ylabel('log(R/S ratio)')
print("H={:.3f}, c={:.3f}".format(H,c))
```

![](img/4f30d3fc83bbe5a66cde4ca12a72da8a.png)

赫斯特指数“H”是每个范围的对数(R/S)与每个范围的对数(大小)的曲线斜率。这里 log(R/S)是因变量或 y 变量，log(size)是自变量或 x 变量。这个值表明我们的数据是持久的。然而，我们正在研究一个小数据集，不能从输出中得出 H 值明显更高的结论，尤其是商品值(0.585)，但给定的时间序列具有一定程度的可预测性。RQA 将帮助我们理解可预测性的程度。

## **计算 RQA:**

```
time_series = TimeSeries(DATA.SPIndex, embedding_dimension=2,
                         time_delay=2)
settings = Settings(time_series, 
                    analysis_type=Classic,                                 
                    neighbourhood=FixedRadius(0.65), 
                    similarity_measure=EuclideanMetric, 
                    theiler_corrector=1)computation = RQAComputation.create(settings,verbose=True)
result = computation.run()
result.min_diagonal_line_length = 2
result.min_vertical_line_length = 2
result.min_white_vertical_line_lelngth = 2
print(result)
```

![](img/5c0a4deba0c3e06253a512431ea0a1b0.png)![](img/6a3d3224aa5b7cd46456f82d8836278c.png)

在这里，我们看到，所有时间序列的 RR(重复率)值并没有偏高，表明周期性程度较低。DET (S&P，原油和白银)和拉姆值原油，黄金和白银)较高，支持确定性结构。然而，这也证实了高阶确定性混沌的存在。

## 计量经济学方法:

这里采用了一系列测试程序来探索变量之间的因果相互作用结构，并确定预测分析的解释变量。

![](img/52ff9b16ee8ca7c8b3c967cc1e3dafcf.png)

皮尔森相关性显示了 S&P 和黄金、原油和黄金、原油和白银以及黄金和白银之间的显著相关性。进行瞬时相位同步(IPA)和格兰杰因果检验。

> 实证研究表明，即使两个变量之间有很强的相关性，也不能保证因果关系。它不提供关于两个信号之间的方向性的信息，例如哪个信号在前，哪个信号在后。

现在，我将使用格兰杰因果检验来检查因果关系，以确定预测因素。VAR 被认为是进行格兰杰因果关系检验的一种手段。在风险值模型中，每个变量被建模为其自身的过去值和系统中其他变量的过去值的线性组合。我们有 4 个相互影响的时间序列，因此，我们将有一个由 4 个方程组成的系统。

1.  Y1，t = α1 + β11，1Y1，t-1 + β12，1Y2，t-1 + β13，1Y3，t-1 + β14，1Y4，t-1 + ε1，t
2.  Y2，t = α2 + β21，1Y1，t-1 + β22，1Y2，t-1 + β23，1Y3，t-1 + β24，1Y4，t-1 + ε2，t
3.  Y3，t = α3 + β31，1Y1，t-1 + β32，1Y2，t-1 + β33，1Y3，t-1 + β34，1Y4，t-1 + ε3，t
4.  Y4，t = α4 + β41，1Y1，t-1 + β42，1Y2，t-1 + β43，1Y3，t-1 + β44，1Y4，t-1 + ε4，t

这里，Y{1，t-1，Y{2，t-1}，Y{3，t-1}，Y{4，t-1}分别是时间序列 Y1，Y2，Y3，Y4 的第一滞后。上述方程被称为 VAR (1)模型，因为每个方程都是一阶的，也就是说，它包含每个预测值(Y1、Y2、Y3 和 Y4)的一个滞后。因为方程中的 Y 项是相互关联的，所以 Y 被认为是内生变量，而不是外生预测因子。为了防止结构不稳定的问题，我使用了 VAR 框架，根据 AIC 选择滞后长度。

```
# make a VAR model
model = VAR(DATA_diff)
model.select_order(12)
x = model.select_order(maxlags=12)
x.summary()
```

![](img/80ee8cce1c1f6ecfb38717814685a99f.png)

Lag selection (VAR)

滞后 4 时获得的最低 AIC 值。所以，因果关系分析是在此基础上进行的。

```
A = model.fit(maxlags=4, ic=’aic’) # pass a maximum number of lags and the order criterion to use for order selectionR1=A.test_causality(‘Index’, [‘CrudeOilPrice’, ‘GoldPrice’, ‘SilverPrice’], kind=’f’)R2=A.test_causality(‘CrudeOilPrice’, [‘Index’, ‘GoldPrice’, ‘SilverPrice’], kind=’f’)R3=A.test_causality(‘GoldPrice’, [‘Index’, ‘CrudeOilPrice’, ‘SilverPrice’], kind=’f’)R4=A.test_causality(‘SilverPrice’, [‘Index’, ‘GoldPrice’, ‘CrudeOilPrice’], kind=’f’)R5=A.test_causality(‘Index’, [‘CrudeOilPrice’], kind=’f’)
```

![](img/168b9d54ab11313ea09a566ddcbd3574.png)

Granger Casuality test result

从上表中我们可以看出，S&P 指数并没有引起系列中其他股票的价格，但是其他股票的价格对 S&P 指数有显著的影响。因此，这里的因果结构是单向的。

```
granger_test_result = grangercausalitytests(DATA_diff[[‘Index’, ‘CrudeOilPrice’]].values,maxlag=4)granger_test_result = grangercausalitytests(DATA_diff[[‘Index’, ‘GoldPrice’]].values,maxlag=4)granger_test_result = grangercausalitytests(DATA_diff[[‘Index’, ‘SilverPrice’]].values,maxlag=4)granger_test_result = grangercausalitytests(DATA_diff[[‘CrudeOilPrice’, ‘SilverPrice’]].values,maxlag=4)granger_test_result = grangercausalitytests(DATA_diff[[‘CrudeOilPrice’, ‘GoldPrice’]].values,maxlag=4)granger_test_result = grangercausalitytests(DATA_diff[[‘GoldPrice’, ‘SilverPrice’]].values,maxlag=4)
```

![](img/d97b6868bb8b5576a548cfa4eb39c562.png)

接下来，估算 IR(脉冲响应),以评估一种资产对另一种资产的冲击影响。

![](img/ec56a26cc9fc9ec010448d11e40b4e99.png)

IR 图显示了给定时期内预期的冲击水平，虚线代表 95%的置信区间(低估计值和高估计值)。此外，所有四个系列的回报率的方差百分比在短期内很大程度上可以由它们自己来解释。从短期来看，自身运动的冲击往往会在波动中扮演重要角色。通过格兰杰因果关系发现对其他序列有影响的序列，对方差也有边际影响。因此，因果分析的总体结果通过方差分解的结果得到验证。因果互动的结果有助于理解相互关系的结构。为了进一步证明，预测误差方差分解(FEVD)执行。基本思想是分解方差-协方差矩阵，使σ= PP 1，其中 P 是具有正对角元素的下三角矩阵，通过 Chelsi 分解获得。

![](img/369889e0f1a8357660826ab00243d589.png)

Forecast error variance decomposition

FEVD 表示在自动回归中每个变量贡献给其他变量的信息量。它决定了每个变量的预测误差方差中有多少可以用对其他变量的外生冲击来解释。所有三个股票指数的回报率的方差百分比在短期内很大程度上由它们自己来解释。

但是，格兰杰因果关系不一定反映真实的因果关系。因此，为了严谨起见，使用 IPA(瞬时相位同步)从统计上验证了资产之间同步的存在。这里 IPA 用于视觉模式识别。它通过增量移动一个时间序列向量并重复计算两个信号之间的相关性来进行测量。中间的峰值相关性表示两个时间序列在那个时间最同步。带通滤波后通过希尔伯特变换实现的频域变换。

在执行任何相位同步之前，预处理阶段需要对金融数据进行去趋势分析，并去除高频噪声成分。滤波级将 SST 用作带通滤波器(SST 是一种块算法，因此数据首先被加窗)，仅应用于去趋势信号的低频振荡。带通滤波的一个重要标准是确保最终信号尽可能窄。

Pearson r 是对全局同步性的度量，它将两个信号之间的关系简化为单个值。然而，我们在这里使用瞬时相位同步测量来计算两个信号之间的瞬时同步，而不像滚动窗口相关那样任意决定窗口大小。

![](img/26864227a8bf9b1f7d107ddabdd177f7.png)

S&P Index & Crude Oil

![](img/dc57116f5a98d6d8ce8d91424fe1a99f.png)

S&P Index and Gold

![](img/7c37a8ea77a794a0a372b5c67e01369a.png)

S&P Index and Silver

![](img/fe9444872770069ddf125e272486bebf.png)

Crude Oil & Gold

![](img/3f24c9cf938504601dbc083387157f34.png)

Crude Oil & Silver

![](img/f6f6ec86b3cb087b889db26a364d9544.png)

Gold & Silver

每个图由过滤的时间序列(顶部)、每个时刻每个信号的角度(中间一行)和 IPS 测量值(底部)组成。当两个信号同相排列时，它们的角度差为零。通过信号的希尔伯特变换计算角度。这里，我模拟了一个相位同步的信号。对于每个图，信号 y1 和 y2 都处于 30Hz (30 天)。我在数据中加入了随机噪声，看看它如何改变结果。波动相位同步图表明 IPS 对噪声敏感，并强调了滤波和选择分析频带的重要性。在对波形进行实际处理之前，使用简单的移动平均和带通滤波器来消除信号中的噪声。然而，IPA 产生了一个全球指标。因此，在将信号分解成多分量的过程中，一个关键的标准是确保相关的频率在本地有效。同步的存在和持续时间反映了金融市场的动态。

# 预测模型

解释性结构的选择是预测建模的关键。一些广泛使用的技术指标是 SMA，MACD 和 CCI 用于趋势，STD 和 BB 用于波动，RSI 和 WR 用于动量。因此，使用原油、黄金和白银作为输入变量的多变量框架对模型进行了测试，这些变量会显著影响 S&P 指数和一系列技术指标，如下所示。

![](img/000f96fcf37b37f05cfcc6f2afcb1825.png)

原始数据集被随机分成 85%的训练数据和 15%的验证数据。对 kNN、DT、RF、GBM、SVN 和 DNN 进行了算法搜索，以选择稳健的模型在给定数据集上进行训练。

![](img/2f40d9a27fda216943b614ecbf3f8b24.png)

目标是获得最低的错误，根据这一逻辑，可以看出 GBM/XGB 在训练和测试数据上都显示最低的错误率。这证明了使用 GBM 算法开发给定时间序列的预测模型的合理性。DNN 以其处理连续和大容量数据的能力而闻名。因此，DNN 也被选中开发一个能够处理高速大数据的混合预测模型。

我用五重 CV 技术进行了重型性能调整，以获得两种算法(XGB 和 DNN)的最佳性能。最后使用堆叠集成技术，将两者相结合以获得更好的预测性能。

![](img/ec32de5a008084ddb20bb7dda3efa40b.png)

发现原油和白银的平均均方误差、RMSE 和平均平均误差值较小，表明该模型对所有原油和白银的有效预测能力。这一结果也支持了这样一个事实，即从历史上看，黄金在股市崩盘期间往往具有弹性，两者呈负相关。这里也没有显示高 MSE 值的异常。让我们看看我们的功能是如何优先排序的。

![](img/be9b7c86e001b59f95138a19cfb96815.png)

Feature importance-XGB

![](img/0c7341a3a9b98448865b0229f9ef66d1.png)

Feature importance-DNN

这些特征是根据它们的相对强度和重要性顺序绘制的。这意味着几乎所有选择的特征，对模型的建立都有一定程度的重要性。

# 结论

在这里，我解释了精选股票和 S&P 的主要特征，并随后深入研究了它们之间的因果关系和预测分析。这里解释了一些关键的技术指标。这种预测结构是计量经济学模型和机器学习算法的结合。虽然我用 ML 算法测试过，但是，技术指标也可以用于成功的交易策略。随后必须对验证集进行前滚验证。

**接我这里**[](https://www.linkedin.com/in/saritmaitra/)***。***

**参考:**

**(1)曼德尔布罗，B. B .，&沃利斯，J. R. (1969)。分数高斯噪声的计算机实验:第 2 部分，重新标度的范围和光谱。水资源研究，5(1)，242–259 页。**

**(2)赫斯特，H. E .布莱克，r . p .&斯迈卡，Y. M. (1965)。长期储存:康斯特布尔的实验研究。英国伦敦。**