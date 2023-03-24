# 瑞士的道路交通事故预测——脸书预言家和 LSTM 神经网络的简单比较。

> 原文：<https://towardsdatascience.com/road-accidents-in-switzerland-forecasting-a-brief-comparison-between-facebook-prophet-and-lstm-6514f585681?source=collection_archive---------19----------------------->

多年来，预测未来的能力只保留给少数人，他们的工具仅限于水晶球、手掌和塔罗牌。但是在过去的 50 年里，新的工具已经出现，现在更多的人可以进行预测，这很棒！

在本文中，我将通过一个简单的例子向您展示如何执行基本的时间序列预测。我们将使用开源库**脸书先知**和使用 Keras / Tensorflow 的 **LSTM** 神经网络来分析、可视化和预测瑞士的道路事故。

我在这篇文章中使用的 jupyter 笔记本可以在我的 [github](https://github.com/figuetbe/ChRoadAccidents) 上找到。

![](img/6f0e46a3cb27f0ee15089e07f8ee8f81.png)

[2011–2018] Accidents location in Switzerland

# 数据集:

对于这篇博客文章，我们将使用“Strassenverkehrsunfallorte”数据集，可从以下地址获得:[https://open data . Swiss/en/dataset/strassenverkehrsunfalle-MIT-personenschaden](https://opendata.swiss/en/dataset/strassenverkehrsunfalle-mit-personenschaden)

**数据集描述:**2011 年至 2018 年道路交通伤害事故的匿名可视化和定位。为道路交通事故提供关于年、月、工作日、事故时间、道路类型、事故类型、严重程度类别和位置的信息。

# 可视化:事故地点

如果可能的话，我总是试图用一个漂亮的可视化来开始一个新的数据科学项目，幸运的是这个项目适合这个目的。

实际上，在考虑预测之前，我对这个数据集感兴趣的原因是它有可能被用来使用 [**数据阴影**](https://datashader.org/) 生成一个“漂亮的”可视化。对于那些熟悉这个库的人来说，你可能已经看到了它在纽约出租车数据集或 OpenSky 网络数据集上的使用，它能够处理数十亿个数据点。

![](img/c15cd643618e45ffdcb1adab3dfae8f1.png)

[2011–2018] Accidents location in Switzerland with colors depending on severity

# 事故预测

虽然原始数据集是一个时间序列，其中每行代表一次事故，但我对它进行了处理，这样我们现在就有了一个时间序列，它可以计算每小时的事故数量，这就是我们将尝试预测的情况。

## 时间序列预测:

说到预测时间序列，我想到了一些流行的方法:

*   简单基线:平均、平滑…
*   自回归:SARIMA 模型(季节性自回归综合移动平均模型)
*   神经网络/深度学习:CNN，LSTM

几乎所有这些方法都需要一点努力来调整它们并找到最佳参数。最酷的事情是，脸书的工程师(可能厌倦了从头开始使用新模型)推出了一个开源库，名为[**Prophet**](https://facebook.github.io/prophet/)**:**“Prophet 是一个基于加法模型预测时间序列数据的程序，其中非线性趋势符合每年、每周和每天的季节性，加上假日效应。它最适用于具有强烈季节效应的时间序列和几个季节的历史数据。Prophet 对缺失数据和趋势变化具有稳健性，通常能很好地处理异常值。

## 数据集操作:

这里的目标不是展示先知或 LSTM 神经网络的所有能力，而是操纵一个简单的数据集，并向您展示如何塑造它来预测未来😮

我们的原始数据集包含每个已登记事故的一行，我们将对其进行操作，以计算在您选择的时间增量(dt)内发生的事故数量。我们将获得一个数据集，它仍然是一个时间序列，但现在每个“时间戳”在一个选定的频率上只有一行，还有一列“n_accident ”,包含在选定的时间跨度内登记的事故数量。

# 脸书先知:

## 先知事故的季节性:

一旦获得新的数据集，使用 Prophet 很容易获得预测和季节性分析。

以下是您可以获得的季节性曲线图类型:

![](img/f235ba0ac414de30c6c2e36f6d3f4a5b.png)

Seasonality of number of accidents per hours

y 轴表示每个时间戳的事故数量(1H 的分辨率)。

正如我们所见，事故数量在 2011 年至 2013 年间有所下降，但自 2016 年以来又有所上升。

我们可以从该图中提取的其他信息是，就事故以及一天中 16-18 小时的时间跨度而言，6 月似乎是最关键的一个月。

## 使用 Prophet 进行事故预测:

由于我们有 2011 年至 2018 年的数据，我们将只使用 2011 年至 2017 年期间来预测 2018 年，然后能够评估预测的准确性。我们不再试图预测每小时的事故数量，而是转换我们的数据集来预测每天的事故数量。

![](img/ddf4095162b4dd8dbfa58b1d0646c3a6.png)

Daily accidents number forecast for 2018

仅通过查看图表很难评估准确性，因此我们需要在本文的下一部分查看一些指标。

# 长短期记忆神经网络

## 使用深度学习和 LSTM 神经网络的事故预测；

LSTM 代表长期短期记忆。这是一种递归神经网络，LSTM 的强大之处在于它有一种记忆，能够记住/忘记模式。

我们将用过去的 1000 天来预测接下来的 365 天。为此，我们根据 2011 年到 2017 年底的数据训练模型，然后我们试图预测 2018 年瑞士的事故。

我们使用步行前进模式。假设我们想要使用最后 3 个元素[1，2，3，4，5，6]来预测下面序列的下一个元素。我可以设计多行的训练数据集，例如:

```
[1, 2, 3] -> 4
[2, 3, 4] -> 5
[3, 4, 5] -> 6
```

为此，我使用了这篇非常有用的文章:[https://machine learning mastery . com/convert-time-series-supervised-learning-problem-python/](https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/)

## 神经网络结构；

我使用 keras 来建模神经网络，我们有一个由 50 个 LSTM 细胞组成的层(我使用了 CuDNNLSTM 层，这是一个充分利用 GPU 的 LSTM 的快速实现)和一个密集层，其中神经元的数量等于我们想要做的预测的数量。

Neural network architecture using keras

## 培训和结果:

我们使用 33%的数据作为验证和 val_loss 上的模型检查点回调，以避免网络对训练数据的过度拟合。

![](img/f1bb16941eccea7e572ef2ae607c05f8.png)

2018 daily accidents forecast using LSTM

正如我们在上面的图中看到的，它抓住了趋势，但仅此而已，我们看到的振荡并没有真正抓住数据中任何有趣的部分。我确信我们可以通过微调 LSTM 超参数得到更好的结果，但这不是本文的目的。

# 脸书预言家 vs. LSTM 预测准确性

为了判断我们预测的准确性，我们将检查两个模型在不同指标下的表现。我选择了 3 个不同的指标:均方根误差[RMSE]、平均绝对误差[MAE]和平均百分比误差[MPE]。

2018 forecasting metrics

正如我们所见，prophet 在 3 个指标中的每一个指标上都表现得更好。这展示了这个工具的威力。以三分之一的努力，我们获得了比用 LSTM 神经网络更好的预测。下次你想预测一些时间序列时，你可能想先试试 Prophet。

但是不要误解我的意思，当你想使用多种特征或更复杂的数据进行预测时，LSTM 是非常强大的，当然，根据你的应用，它可能是你要考虑的“工具”。