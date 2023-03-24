# 预测洛杉矶的电力需求——超越政府

> 原文：<https://towardsdatascience.com/predicting-electricity-demand-in-la-outperforming-the-government-a0921463fde8?source=collection_archive---------13----------------------->

# 介绍

这是一篇小博文，描述了我为 [Springboard 的数据科学职业轨道](https://www.springboard.com/workshops/data-science-career-track/)所做的第一个顶点项目。我比较了一些根据天气特征预测电力需求的机器学习模型。我发现大多数模型对电力需求的预测比政府预测要准确 5%。查看我的 [GitHub 库](https://github.com/rvgramillano/springboard_portfolio/tree/master/Electricity_Demand)，获取该项目的代码、图表和数据集。

# 数据收集和分析

为了训练和测试我的模型，我使用了 2015 年 7 月至 2018 年 9 月洛杉矶约 27，000 个小时的电力需求测量值。该数据由洛杉矶水电部记录，并通过[能源信息管理局的公共 API](https://www.eia.gov/opendata/) 发布。

对于天气数据，我从美国国家海洋和大气管理局(NOAA)网站检索每小时的当地气候数据(LCD) [。在 LCD 数据集中，我们有熟悉的测量方法，如压力和温度，以及不太直接相关的量，如风向。增加了一个额外的功能:每小时的冷却和加热程度。加热/冷却度天数分别是低于/高于 65 华氏度(负值设置为零)的度数，代表该温度所需的加热/冷却总量。我们特别预计 CDD 将成为洛杉矶电力需求的强劲代表，因为洛杉矶的主要能源消耗在夏季会降温。**图 1** 显示了 2015 年 7 月至 2018 年 9 月 LA 的电力需求时间流程。请注意夏季的需求高峰。](https://www.ncdc.noaa.gov/cdo-web/datatools/lcd)

![](img/c679a704b13171b545cb8515aafbca8b.png)

**Figure 1**. Electricity demand versus time in LA. Note the sinusoidal pattern, with the summer season experiencing more demand because of air conditioning.

经过一天的初步清理和准备，我进行了一些探索性的数据分析，以查看独立变量(天气)和非独立变量(电力)之间的初始关系。**图 2** 显示了各特性与电力需求之间的[皮尔逊相关系数](https://en.wikipedia.org/wiki/Pearson_correlation_coefficient)和**图 3**r[值*和*值](https://en.wikipedia.org/wiki/Coefficient_of_determination)。我们发现，日冷却度天数与电力需求有很强的正相关关系——绝大多数电力使用是在洛杉矶的夏季用于冷却。

```
# Code to produce figure 2
import pandas# df is our main dataset with weather + electricity data
print('DEMAND CORRELATIONS (PEARSON) FOR LA')
print(df.corr()['demand'].sort_values(ascending=False)[1:])
```

![](img/7d86d542e162df0b9c0b6f5dd78274c0.png)

**Figure 2**. Pearson correlation coefficients for all weather features with electricity demand. Cooling degrees is a strong predictor of electricity demand.

```
# Code to produce figure 3
import scipy
import pandas as pd# get r^2 values per column and print
demand_r = {}
for col in df.columns:
 if col != 'demand':
  slope, intercept, r_value, p_value, std_err = scipy.stats.stats.linregress(df['demand'], df[col])
  demand_r[col] = r_value**2print('DEMAND CORRELATIONS (r^2) FOR LA')
demand_r_df = pd.DataFrame({'col': demand_r.keys(), 'r^2': demand_r.values()})
print(demand_r_df.sort_values(by='r^2', ascending=False))
```

![](img/df32754986e1031f42ea93510ff509af.png)

**Figure 3**. *r²* values for all weather features with electricity demand.

我们的一些天气特征有共线性；有三个压力特征(气象站、海平面、高度计)和另外三个温度特征(湿球、干球、露点)。[共线性增加了预测建模的难度](https://en.wikipedia.org/wiki/Multicollinearity)，因此我们删除了具有最低 *r* 值的两个温度和压力特征。

由于人们在白天和晚上的行为是不同的，不管那天有多热或多冷，我添加了一个“hourlytimeofday”列，代表白天或晚上(早上 6 点到下午 6 点之间为 0，其他时间为 1)。温度数据对于识别加热和冷却峰值很重要，但是回归可以开始依赖温度作为白天的代理。这个特征大大增加了我们的多元回归 r，表明了它的重要性。

在使用其他机器学习技术之前，我们对所有特征与电力需求进行多重普通最小二乘(OLS)回归，以隔离不重要的特征。**图 4** 显示了结果。高 p 值特征证实了我们的直觉——加热对洛杉矶并不重要。天空条件，即云覆盖和风速，直观上不是预测电力需求的重要特征。这些不太有用的要素由高 p 值表示，p 值大于 0.1 的任何要素都将从数据集中删除。

![](img/d41a4495e705e21b505dee75f48f6a6d.png)

**Figure 4**. Multivariate OLS regression for all the features + constant. Shown are a list of coefficients with errors, t-statistics, confidence intervals.

# 机器学习建模

现在我们为机器学习建模准备数据集。我们的第一步是将时间序列数据转换成适合监督学习的形式。从这篇博文中大量提取[，我转换了我们的数据集，这样我使用了前一时间步的所有**特征(即电力需求加上所有天气特征)来预测当前时间步*的电力需求*。*由于前一小时的电力需求似乎是当前电力需求的相关代理，因此该过程似乎是直观的。在以这种格式重新构建我们的数据后，我们将数据分成训练和测试集，然后开始评估我们的机器学习模型。为了使分析与我们后来使用的递归神经网络和长短期记忆一致，我们将第一年的数据指定为训练集，其余数据指定为测试集(约 32%训练和约 68%测试)。这种分割避免了过度拟合，并且在训练和测试上更传统的 80/20 分割产生了几乎相同的模型性能。建模的一般过程是:1 .)在所有功能上使用***](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)**[最小-最大缩放器](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)以使它们更具可比性，2。)适合训练集，以及 3 .)计算测试集上的相关度量，并在最后比较所有模型。在我们的分析中，我们没有对模型进行超参数调整(sklearn 的默认设置),但仍然取得了非常显著的结果。**

*我使用 sklearn 模块训练和测试了大量的标准机器学习模型，如上所述。建模的具体过程和结果可以在[本 IPython 笔记本](https://nbviewer.jupyter.org/github/rvgramillano/springboard_portfolio/blob/master/Electricity_Demand/modeling/modeling.ipynb)中找到。对于直接开箱即用的模型，我获得了令人印象深刻的准确性，在我尝试的一半以上的模型中， *r* 值超过 0.95。此外，训练和测试的 r 值相差不超过百分之几，这表明我没有过度适应。*

*除了开箱即用的模型，我们还使用了循环神经网络(RNN)和长短期记忆(LSTM)。本 IPython 笔记本中的[详细介绍了 RNNs 和 LSTM 的基础知识以及建模过程。使用这个模型，我获得了与我之前测试的最好的机器学习模型相当的性能。**图 5** 显示了使用 LSTM 建模的训练集和测试集的损失(平均绝对误差)与历元的关系。测试损失仍然略高于训练损失，表明模型没有过度拟合。](https://nbviewer.jupyter.org/github/rvgramillano/springboard_portfolio/blob/master/Electricity_Demand/modeling/lstm.ipynb)*

```
*# Code to produce figure 5
from keras.models import Sequential
from keras.layers import Dense, LSTM
import matplotlib.pyplot as plt# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y, epochs=50, batch_size=72, validation_data=(test_X, test_y), verbose=2, shuffle=False)
# plot history
plt.plot(history.history['loss'], label='train')
plt.plot(history.history['val_loss'], label='test')
plt.xlabel('Epoch')
plt.ylabel('Loss (MAE)')
plt.legend()
plt.show()*
```

*![](img/63e3b604a73925a85ba7774c206525ed.png)*

***Figure 5**. Loss measured in mean-absolute error (MAE) versus epoch for both training and testing of our LSTM network. Testing error remains slightly above training, indicating that we are not overfitting. The LSTM network was optimized using [the Adam implementation of stochastic gradient descent](http://ruder.io/optimizing-gradient-descent/index.html#adam) with MAE loss.*

*EIA 还提供了一天前的电力需求预测，我们希望将其与我们的模型进行比较。EIA 用各种方法[预测需求](https://www.eia.gov/todayinenergy/detail.php?id=27192)。天气预报很大程度上取决于天气预报和一周中的第[天](https://www.eia.gov/todayinenergy/detail.php?id=37572)。使用相同的测试集，我们发现 EIA 的前一天需求预测的准确性比我们的所有模型都差，只有一个模型除外，这表明了简单、开箱即用的机器学习模型在回归问题上的能力。*

*结果汇总在按 *r* 排序的**表 1** 中。总的来说，集成机器学习模型比其他模型表现得更好，尽管简单的线性回归本身表现得非常好。所有模型的训练和测试误差保持在彼此的百分比水平之内，这表明我们没有过度拟合数据。由于交叉验证方法在处理相关的时间流数据时变得更加棘手，我们在分析中省略了交叉验证，以保持所有模型的可比性和一致性。我们发现梯度增强模型具有最好的性能，尽管前五个模型的性能都相似(它们之间的性能差异约为 1%)。*

*![](img/d6c5265c6d549996098a6fa2ee801fa1.png)*

***Table 1**. Table of results of machine learning models. The best performing model (gradient boosting) is highlighted.*

*选择了性能最佳的模型(梯度推进)后，我们现在想看看哪些特性在预测电力需求时最为重要。这显示在**图 6** 中。[“特征重要性被计算为通过到达该节点的概率加权的节点杂质的减少。节点概率可以通过到达该节点的样本数除以样本总数来计算](https://medium.com/@srnghn/the-mathematics-of-decision-trees-random-forest-and-feature-importance-in-scikit-learn-and-spark-f2861df67e3)前一小时的需求是预测当前小时需求的最重要特征，但如我们所料，其他替代因素如降温天数也是一个强有力的预测因素。*

*![](img/0bbf41d0eac0816b2eaf0619fd876a0b.png)*

***Figure 6.** Feature importance for each feature in the gradient boosting model. Demand at the previous time step is four times more important than cooling degree days. Reframing our supervised learning problem in this way greatly improved the predictive power of the models.*

# *结论*

*我们表明，通过机器学习，超越 EIA 的日前电力需求预测实际上相当容易。值得注意的是，除了一个模型(k-NN)之外，我们的所有模型都比 EIA 的模型表现得更好。来自公共数据源的相对简单的数据采集、清理和机器学习建模导致比传统方法更精确的建模。我们报告说，与 EIA 的模型相比，总体性能提高了约 5%;能源公司可以利用这些模型更好地应对高需求电力高峰，并最终增加利润。既然已经确定了最佳模型，那么可以执行额外的超参数调整和交叉验证技术来提高性能，尽管这种提高很可能是微不足道的，并且由于我们的模型已经优于 EIA 的模型，所以我们在没有调整的情况下得出结论。[请查看这个项目的 GitHub](https://github.com/rvgramillano/springboard_portfolio/tree/master/Electricity_Demand)以获得更详细的报告、代码、情节和文档。*