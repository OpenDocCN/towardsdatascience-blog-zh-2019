# 为什么绿色和银色可能成为新的黑色和金色

> 原文：<https://towardsdatascience.com/why-green-silver-just-might-be-the-new-black-gold-1694ac5562bf?source=collection_archive---------7----------------------->

![](img/ab759863db819f9eb0c0bf40881d3f21.png)

(CREDIT: Author on [Canva](http://canva.com))

## 使用典型的数据科学家工具箱对白银进行基本面分析。

# 背景

白银的故事很长，在过去的几千年里，这种白色金属与人类一起享受了一段相当喧嚣的浪漫史——尽管它似乎一直处于黄金的次要地位。

这篇文章试图对贵金属进行一些彻底的分析，同时也作为一种“数据科学”教程——因为我将记录我在进行分析时经历的许多步骤，并在适用的情况下提供代码样本。笔记本和一些数据可以在这个库中找到[。对于这篇文章的压缩版本，点击](https://github.com/peter-stuart-turner/silver-blog-post-medium)[这里](https://medium.com/welded-thoughts/5-reasons-why-green-silver-may-be-the-new-black-gold-f13633b079cf)。

# 内容——这篇文章的内容

![](img/0c81549ae2f308326cdc78aedd645ee6.png)

What to Expect In This Post (CREDIT: Author on [Canva](http://canva.com))

# 需求分析

银作为一种贵金属很有意思，因为它的需求远远超出了仅仅作为珠宝和价值储存手段的用途——相反，它的物理特性使它在工业中备受追捧。本节将主要关注当前和近期对白银的需求。

## 概述和细分

为了有一点基础，让我们看看来自[白银协会](https://www.silverinstitute.org/)的 [2019 年世界白银调查](https://www.silverinstitute.org/wp-content/uploads/2019/04/WSS2019V3.pdf)的要点——与白银需求有关:

*   2018 年实物总需求增长 4%，至 10.335 亿金衡盎司(3.2146 万吨)，创下三年新高
*   这主要是由于对珠宝和银器的需求增加(4%)，以及对硬币和金条的需求增加(20%)，这抵消了工业需求的减少(1%)
*   工业需求的减少主要是由于光伏(太阳能电池板)行业的减少(-9%)，这是因为银负载的持续节约(由于价格原因，制造商正试图使用越来越少的银)

[CPM 白银年鉴](https://www.cpmgroup.com/store/cpm-silver-yearbook-2019/)对国际白银市场进行了详细的统计和分析，过去 43 年全球白银加工需求如下:

![](img/a3d2c0b1695c7a1e6a259510bd141105.png)

Silver Fabrication Demand by Year since 1977 ([GoldSilver Blog Post](https://goldsilver.com/blog/silver-in-charts-supplydemand-crunch-after-years-of-the-opposite/))

*(图片摘自* [*这篇博文*](https://goldsilver.com/blog/silver-in-charts-supplydemand-crunch-after-years-of-the-opposite/) *使用 CPM 白银年鉴的数据)。*

很明显，制造业的总需求已经连续六年呈上升趋势。现在，这一切都很好，但为什么会增加呢？为了了解这一点，**让我们深入了解这一需求:**

白银协会提供了更详细的全球白银供需数据，但仅限于 2009 年至 2018 年。

![](img/f4509ffb2a6d99f3e060748df0c494e1.png)

World Silver Demand and Silver Closing Price, Per Year (Source: [The Silver Institute](https://www.silverinstitute.org/silver-supply-demand/))

他们提供的数据实际上是 JPEG 图像，所以我必须手动将其记录到 Excel 中，然后再用 Python 读取。相当麻烦，但我已经把这个文件添加到了这篇文章的[回购中，以拯救可怜的灵魂，使其在未来不再经历类似的考验。我从一个网站上刮下来的白银收盘价(见上一节)。](https://github.com/peter-stuart-turner/silver-blog-post-medium/)

我使用熊猫、Plotly 和袖扣来绘制这些数据，对于上面的图表，我实际上使用了在线图形设计工具 [Canva](https://www.canva.com/) ，来结合白银价格和需求图表(我知道这也可以用 Plotly 来完成，但我不知道如何使用袖扣来轻松完成)。我很喜欢 Plotly，我一定会推荐它；Cufflinks 是一个帮助 Plotly 轻松使用熊猫数据框的库。

```
import pandas as pd
# Standard plotly imports
import plotly.plotly as py
import plotly.graph_objs as go
from plotly.offline import iplot, init_notebook_mode
# Using plotly + cufflinks in offline mode
import cufflinks
cufflinks.go_offline(connected=True)
init_notebook_mode(connected=True)# Reading in data
demand = pd.read_excel('../data/silver_institute_supply_and_demand.xlsx', 'Demand')
demand.set_index('Year', inplace=True)
demand.fillna(0, inplace=True)# Plotting - Total Supply per Year
demand[['Total Physical Demand']]\
.iplot(kind='bar', xTitle='Year',  color='#139A43',
                  yTitle='Demand (million ounces)', 
       title='Total Demand for Silver, Per Year', width=4)# Plotting - Total Demand per Year & by Sector
demand.loc[:,'Jewelry':'Industrial Fabrication'].iplot(kind='bar', xTitle='Year', 
                  yTitle='Demand (million ounces)', 
       title='Total Demand for Silver, Per Year and By Sector', width=4)# Plotting - Total Industrial Demand per Year & by Industry
demand.loc[:,'…of which Electrical & Electronics':'…of which Other Industrial*']\
.iplot(kind='bar', xTitle='Year', 
                  yTitle='Demand (million ounces)', 
       title='Total Industrial Demand for Silver, Per Year and By Industry', width=4)
```

![](img/4c3c679525f14455ad95d2d12d6e95b5.png)

Global Silver Demand by Sector, by Year (CREDIT: Author on Jupyter Notebook)

![](img/7b0bab896f6a649e80665fc31df1a9e7.png)

Total Industrial Silver Demand by Industry, by Year (CREDIT: Author on Jupyter Notebook)

从上面的图表中可以看出什么？嗯——我从他们那里得到的是:

*   最大比例的白银需求来自工业应用——这与大多数其他贵金属形成鲜明对比
*   在工业需求中，最大的比例是由电气和电子行业驱动的(占所示期间总需求的 43.64%)
*   电气和电子产品需求呈上升趋势(自 2016 年起)
*   光伏需求从 2014 年到 2017 年呈上升趋势，但在 2018 年大幅下降——这似乎是白银研究所在他们的[报告](https://www.silverinstitute.org/wp-content/uploads/2019/04/WSS2019V3.pdf)中谈论的内容(见上文)

![](img/40b7444e276ad04cb0cd19f4aa7e25fb.png)

Silver Industrial Demand, By Year Since 1960 ([GoldSilver Blog Post](https://goldsilver.com/blog/silver-in-charts-supplydemand-crunch-after-years-of-the-opposite/))

为什么与其他贵金属相比，白银的工业需求如此之高？为什么会增加？

**银作为一种元素的科学属性有很大关系，Ag:**

![](img/2a9d83836f262e4aced720b70d753b95.png)

Thermal & Electrical Conductivities of Several Metals (Left), and Silver Nanoparticles Biocide Solution (Right) (CREDIT: Author on [Canva](http://canva.com))

在地球表面的所有金属中，银具有最高的导电性和导热性。不仅如此，这种金属还是一种众所周知的杀菌剂——意思是:它能杀死细菌。(注意电阻率是电导率的[倒数](https://www.cliffsnotes.com/cliffsnotes/subjects/math/in-math-what-does--i-reciprocal-i-mean))。

> *“在地球表面的所有元素中，银的导电性和导热性最高。”*

在本需求分析的下几节中，我们将调查和分析白银在几个主要行业中的确切用途。

## 电气和电子行业的白银

银在金属中无与伦比的导电性能意味着它不容易被其他更便宜的金属取代。银也是一种**贵金属**，这意味着它在潮湿的空气中耐腐蚀和抗氧化——不像大多数其他金属。

考虑到这些事实，电子产品遥遥领先于工业白银的头号消费者就不足为奇了。

![](img/cd9ddeaaeb6c1ef960423bf425808d7e.png)

Many electronic applications make use of Silver pastes (CREDIT: Author on [Canva](http://canva.com))

银可以直接在银矿中开采，也可以作为铅锌矿的副产品开采。熔炼和精炼将银从矿石中提炼出来，之后通常将其加工成条状或颗粒状。电子产品只需要最高纯度的银——意味着没有污染物(99.9%的银)。银在硝酸中的溶解产生硝酸银，然后可以形成薄片或粉末——然后可以制作成隐形眼镜或银浆。总的来说，这些最终产品是直接用于电子产品的。使用它们的例子很多，但包括:

*   许多汽车的后除霜器，
*   印刷电子和纳米银(广泛用于电动汽车的再生制动)，
*   发光二极管(可以使用银电极)，
*   电池技术(使用氧化锌或银锌合金)，
*   超导体(当与银配对时，结果可以比单独的超导体更快地传输电流)

## 光伏太阳能中的银

如前所述，银是一种独特的贵金属，因为超过一半的供应量实际上用于工业目的。这主要是因为银具有优异的反射和导电性能。银的一个有趣的应用是在太阳能光伏产业。**银被广泛用于太阳能电池板的生产**。

![](img/541a6d6c30b16e75af0e54e615476516.png)

Silver is used extensively to produce solar panels (CREDIT: Author on [Canva](http://canva.com) using their library)

事实上，含银浆料对光伏电池和 90%的晶体硅光伏电池都至关重要。该浆料用于太阳能电池中，以收集和传输电子，电子的运动是由阳光通过光电效应引起的。实际上，银有助于传导电子，从而产生电流；这些电能可以被储存或消耗。银的优异电阻率增加了捕获的潜在阳光、传导的能量，并最终增加了太阳能电池产生的总功率——因此太阳能电池的效率在某种程度上与其银含量成比例。

鉴于面对气候变化，全球需要追求更可持续的能源，太阳能以及太阳能电池生产的未来确实是光明的。

两个因素对太阳能导致的白银需求的无节制增长构成了明确的威胁:即**节约**和**其他技术相关的效率提高**。

**节约**是制造商试图减少白银用量以降低成本的过程。事实上，根据代表白银协会的报告，到 2028 年，光伏行业生产导电银浆所需的银量可能会减少近一半，从 2016 年的每电池 130 毫克减少到 65 毫克。银的含量已经下降了很多，从 2007 年的 400 毫克下降到 2016 年的 130 毫克。即使银含量下降，太阳能电池的产量实际上还是增长了，而且预计还会继续增长；从现在的 4.7 瓦到 2030 年的 6 瓦。这种增长是由于效率的提高以及太阳能电池板设计和制造方面的其他技术进步。

![](img/38e657bc55364403c0a80574552fdbd8.png)

Solar PV Fabrication and Thrifting, Per Year ([The Silver Institute](https://www.silverinstitute.org/wp-content/uploads/2019/04/WSS2019V3.pdf))

尽管如此，银无与伦比的导电特性意味着，事实上，太阳能电池生产中可能的银负载减少量存在“物理极限”。简而言之，总有一天，效率损失会超过使用更便宜的原材料(铜、铝)替代白银所带来的好处。

> 尽管如此，银无与伦比的导电特性意味着，事实上，太阳能电池生产中可能的银负载减少量存在“物理极限”。简而言之，总有一天，效率损失会超过使用廉价原材料带来的好处

上述报告预测，光伏行业的需求约为每年 7000 万至 8000 万盎司，到 2020 年代中期将降至 5000 万至 5500 万盎司。只有到 2030 年，需求才有望恢复，达到每年约 6600 万盎司。世界银行的报告呼应了类似的观点，预测太阳能光伏发电将导致需求下降。

我认为，这些实体没有考虑到的是:不要低估在对气候变化的持续恐惧中长大的一代人的力量，当他们到了成为市场驱动者的年龄时，他们作为一个集体会做些什么。

***但是让我们用一点分析来解决这个问题:***

下面是两张图表。第一个是每年全球可再生能源份额图，第二个是每年全球太阳能总消耗量图:

![](img/39b14c3eb07e948d7d4129a849d6033f.png)

Total Share of Global Energy that is Renewable (Split Between Solar+Wind, and Other Forms of Renewable Electricity) (CREDIT: Author on Jupyter Notebook)

![](img/3126d0f92dbd8af9c3467628ad3672ee.png)

Total World Solar Energy Consumption, Per Year ([Our World In Data](https://ourworldindata.org/renewable-energy))

很明显，太阳能的增长是指数级的——我们看到的绿色革命肯定不会阻止我们在未来使用更多的太阳能。

脸书有一个名为“先知”的非常棒的开源库，它非常适合单变量时间序列预测。它也非常容易使用，让我们把我们的全球太阳能消耗数据扔进 Prophet。

Excerpt of Github Gist — See [here](https://gist.github.com/peter-stuart-turner/a5a10c07bbb0ee5beadc77fed95a7974)

Github Gists 似乎不再恰当地嵌入到媒体中——所以跟随[链接](https://gist.github.com/peter-stuart-turner/a5a10c07bbb0ee5beadc77fed95a7974)正确地查看代码(或者检查[笔记本](https://github.com/peter-stuart-turner/silver-blog-post-medium))结果预测如下所示(下载数据集[这里](https://ourworldindata.org/renewable-energy)):

![](img/c858ccf633c33e4384c76a3645fa2a1d.png)

Forecasted Global Solar Consumption Using FbProphet — y is still in TWh (CREDIT: Author on Jupyter Notebook using Facebook’s Prophet library)

为了得到节省的数据，我使用了一个在线工具从 jpeg 图(如上图所示)中映射出来。我已经在这篇文章的[资源库中提供了它。数据范围仅为 2009 年至 2018 年。如果我们把这个也扔进 Prophet，我们得到如下预测:](https://github.com/peter-stuart-turner/silver-blog-post-medium)

![](img/036e34fdefb1c57d48a07a75cd6650b4.png)

Forecasted Silver Loadings in grams/cell ([Gist](https://gist.github.com/peter-stuart-turner/af01d1bad7610f3215272d27e52d50e2)) (CREDIT: Author on Jupyter Notebook using Facebook’s Prophet library)

嗯。我们知道银负载量不能降到零以下，我们也没有以任何方式嵌入我们的假设，即“太阳能电池生产中可能的银负载量减少量存在物理限制”——因此这在我们的预测中显然不明显。

白银协会在他们的报告中指出，他们预计节约将会继续，到 2028 年，所用的银负载量可能会降至每电池 65 毫克——现在让我们调整我们的数据(和预测)以包括以下条目:

![](img/3bbdfdd10c758d3797d44366545b2fd8.png)

Forecasted Silver Loadings in grams/cell (Taking into account Silver Institute Prediction) (CREDIT: Author on Jupyter Notebook using Facebook’s Prophet library)

老实说，这不是很有帮助，但我必须说:这些数据看起来非常像可以用指数函数来拟合，幸运的是 [Scipy](https://www.scipy.org/) 有一个非常酷的函数可以在瞬间为我们做到这一点——使用[最小二乘](https://www.investopedia.com/terms/l/least-squares-method.asp)回归。

不过，在我们这样做之前，让[归一化](https://en.wikipedia.org/wiki/Normalization_(statistics))我们的银负载数据集——从 0 到 1。我们可以使用 S [klearn 的](https://scikit-learn.org/)和[最小最大缩放器](https://scikit-learn.org/stable/modules/generated/sklearn.preprocessing.MinMaxScaler.html)来实现。

```
import datetime
import pandas as pd
from sklearn import preprocessingest_thrifting = pd.read_excel('../data/silver_solar_thrifting_estimates.xlsx')# Normalize Estimated Silver 
min_max_scaler = preprocessing.MinMaxScaler()
thrifting_scaled = min_max_scaler.fit_transform(est_thrifting)scaled_thrifting_df = pd.DataFrame(columns=['Year', 'Silver Loadings (Scaled)'], data=thrifting_scaled)scaled_thrifting_df.set_index('Year').plot(color='green')
```

我们的标准化数据如下所示:

![](img/79ca4eff39edeec2ca555af8def6bb5d.png)

Silver Loadings per Year (Scaled from 0 to 1) (CREDIT: Author on Jupyter Notebook)

现在我们可以使用 Scipy 的 [curve_fit](https://docs.scipy.org/doc/scipy/reference/generated/scipy.optimize.curve_fit.html) 函数来拟合数据的以下函数(指数曲线)。

![](img/03c97769b2322eafe8d1acc0fd6f3655.png)

Exponential Function Formula (CREDIT: Author on [Canva](http://canva.com))

```
import numpy as np
import scipy.optimize as optimize
import matplotlib.pyplot as pltxdata = scaled_thrifting_df['Year'].values
ydata = scaled_thrifting_df['Silver Loadings (Scaled)'].values
x = np.array(xdata)
y = np.array(ydata)def f(x, a, b, c):
    return a * np.exp(-b * x) + cpopt, pcov = optimize.curve_fit(f,x,y, bounds=([0,0,-0.15], [10,10, -0.06]))
# popt, pcov = optimize.curve_fit(f,x,y)
x1 = np.linspace(0,2.5,num =50)fig_size = plt.rcParams["figure.figsize"]
fig_size[0] = 12
fig_size[1] = 8
plt.rcParams["figure.figsize"] = fig_size
plt.plot(x1, f(x1, *popt), 'g',)
plt.plot(x,y,'ro')
plt.show()
```

请注意，我限制了 c 的界限，以便得到一个至少趋向于每细胞小于 65 毫克的值的曲线(由 Silver Institute 在 2028 年估计的负载量)。结果如下:

![](img/a3390b3f0fad55d5531438dbfeffe8fd.png)

Exponential Curve fitted to Normalized Silver Loadings Data (CREDIT: Author on [Canva](http://canva.com))

在上图中，x 轴上的 2.00 对应于 2047 年。银负载量趋向于 0.045 左右的值——或 45 毫克/电池。 ***所以这就是我们的【生理极限】*** 。

现在——当然，这条曲线有点笨拙，我调整了界限，这样它基本上把图表降低到了一个“看起来差不多”的值，甚至有些保守。

> 现在，这变得非常有趣，当你把两个: 结合起来的时候，就是

*![](img/eb2aa38fd9923bf7767ffe122e302e1b.png)*

*(Normalized) Solar Demand Forecast Using Only Silver Loadings and Solar Consumption (CREDIT: Author on [Canva](http://canva.com))*

**我不会在这里张贴上图的代码，只是因为它进行了大量的调整、缩放、创建新的数据框架等等(我的熊猫技能可能也有所欠缺)。但是，包含所有代码的单独笔记本是可用的* [***这里是***](https://github.com/peter-stuart-turner/silver-blog-post-medium/blob/master/notebooks/solar-forecasts.ipynb) *。拜托，伙计们，* ***检查一下我对这一切的逻辑，并在评论中回复我****——我很容易搞砸了一些事情，如果是这样的话，我很想知道。**

*当然，这整个过程是假设太阳能银需求与太阳能消耗和每个太阳能电池的银负载成唯一和直接的比例关系——这似乎是有意义的:当然，太阳能银需求与“每个电池中的银量”以及“太阳能电池板消耗的能量”成比例，对吗？*

*一条不归路。如果这个图表可信的话，看起来我们正处于太阳能白银需求的转折点。如果太阳能电池板中继续使用的银量确实存在物理限制，那么似乎——假设太阳能消耗的增加等于或类似于我们的预测——太阳能行业对银的需求将在未来开始相对快速地攀升，特别是当它达到这一限制时。*

> *看起来，我们正处于太阳能白银需求的转折点。*

*上图似乎也与过去的 [CPM 白银年鉴](https://www.cpmgroup.com/store/cpm-silver-yearbook-2019/)数据相对应(见 [GoldSilver 的博客文章](https://goldsilver.com/blog/silver-in-charts-supplydemand-crunch-after-years-of-the-opposite/)):*

*![](img/08200bb560af04699bbcedc05b198af8.png)*

*Actual Silver Demand for Solar Panels per Year(GoldSilver blog, CPM Silver Yearbook)*

## *水处理中的银*

***生物杀灭剂**在欧洲法律中被定义为“旨在破坏、阻止、使无害或对任何有害生物产生控制作用的化学物质或微生物”。*

*![](img/c873c10c085ab774141ee9210dd2b4d2.png)*

*Silver may have increased future applications in water treatment, but this is speculatory at best (CREDIT: Author on [Canva](http://canva.com))*

***银是一种广为人知的广谱杀菌剂**，被证明能有效消除越来越多的细菌威胁，包括军团病、大肠杆菌、链球菌和 MSRA；这些微生物占了我们日常接触的细菌的很大一部分。由于这些性质，银长期以来一直用于水处理，在世界各地的许多过滤器中销售。*

*尽管世卫组织世界卫生组织已经宣布，在其目前的应用中，银是*“不是饮用水的有效消毒剂”*，纳米技术领域的发展和银纳米粒子的使用可能会在未来几年内看到这种变化。最近的研究表明，银(银)纳米粒子片的银损失低于世卫组织和环境保护局(EPA)提出的饮用水标准，结论是通过沉积有银纳米粒子的纸过滤可以是有效的应急水处理。除此之外，最近的研究调查了将银用作航天器水系统杀菌剂的可能性。事实上，美国宇航局正在考虑将银作为未来探索的杀菌剂。*

> *最近的研究表明，银(银)纳米粒子片的银损失低于世卫组织和环境保护局(EPA)提出的饮用水标准，结论是通过沉积有银纳米粒子的纸过滤可以是有效的应急水处理。*

*总结一下银在水处理中的应用；如果你像 Michael Burry(《大空头》故事背后的人)一样，认为未来将会出现可获得淡水的严重短缺，并且同样的未来将会看到太空探索的急剧增长，你可能只是想在这种迷人的金属上对冲你的赌注。*

*![](img/a1625b15698c88ba98e82a3e2cc4f5d9.png)*

*Christian Bale as Michael Burry in ‘The Big Short’. Burry has been known to be concentrating on freshwater, as he thinks freshwater reserves per capita will decrease in the future — and it will thus become an ever more economically valuable commodity. (CREDIT: Author on [Canva](http://canva.com))*

# *供应分析*

*对白银供应的分析即将出炉，供应数据当然也很有趣。*

## *概述和细分*

*全球白银供应总量如下图所示:*

*![](img/8c9b27811b72eb1bad4eff85e70dcf5e.png)*

*Total Global Silver Supply by Year since 1977 ([GoldSilver Blog Post](https://goldsilver.com/blog/silver-in-charts-supplydemand-crunch-after-years-of-the-opposite/))*

*同样，白银协会的数据更加细致:*

```
*supply = \ pd.read_excel('../data/silver_institute_supply_and_demand.xlsx', 'Supply')
supply.set_index('Year', inplace=True)
supply.fillna(0, inplace=True)supply[['Total Supply']]\
.iplot(kind='bar', xTitle='Year', color='#1C7C54',
                  yTitle='Supply in Tons', 
       title='Total Supply of Silver, Per Year', width=4)supply[['Mine Production','Net Government Sales','Scrap', 'Net Hedging Supply']]\
.iplot(kind='bar', xTitle='Year', 
                  yTitle='Supply in Tons', 
       title='Total Supply of Silver, Per Year', width=4)*
```

*![](img/354c5e1288fc69b55d93df5d9f198315.png)*

*Total Silver Supply (Silver Institute) (CREDIT: Author on Jupyter Notebook)*

*![](img/93e59d5778954c0a748de87cd073cd9e.png)*

*Total Supply by Year and Source (CREDIT: Author on Jupyter Notebook)*

****到目前为止的一些要点:****

*   *全球大部分白银供应来自地下(矿山)*
*   *无论是矿山供应还是废钢供应(白银的两大来源) ***似乎都在下降****——****但是为什么呢？****

*来自矿山的银供应量的减少似乎是由于银价造成的**。我使用了 Python 库 [*请求*](https://pypi.org/project/requests/) 和 [*BeautifulSoup*](https://pypi.org/project/beautifulsoup4/) 来抓取历史白银价格，然后用 Plotly 绘图:***

```
*import requests
from bs4 import BeautifulSoup## Silver 
res = requests.get("[https://www.macrotrends.net/1470/historical-silver-prices-100-year-chart](https://www.macrotrends.net/1470/historical-silver-prices-100-year-chart)")
soup = BeautifulSoup(res.content, 'lxml')
tables = soup.find_all('table')
headers = ['Year', 'Average Closing', 'Year Open', 'Year High', 'Year Low', 'Year Close', 'Annual % Change']
df_silver = pd.DataFrame(pd.read_html(str(tables[0]))[0])
df_silver.columns = headersdef convert_to_float_if_applicable(cell):
    if isinstance(cell, str) and '$' in cell:
        try:
            cell = float(cell.replace('$', '').replace(',',''))
            return cell
        except:
            return cell
    elif isinstance(cell, str) and '%' in cell:
        try:
            cell = float(cell.replace('%', ''))
            return cell
        except:
            return cell
    else:
        return celldf_silver = df_silver.applymap(lambda cell: convert_to_float_if_applicable(cell))
df_silver.rename(columns={'Average Closing': 'Average Closing, Silver'}, inplace=True)df_silver.set_index('Year')[['Average Closing, Silver']]\
.iplot(kind='line', xTitle='Year', color='#C0C0C0',
                  yTitle='Average Closing Price in Dollars ($)', 
       title='Average Closing Prices for Silver', width=4)*
```

*![](img/f1e70ca79321a4ab1244f0defe260787.png)*

*Historical Silver Prices per year since 1969 ([macrotrends.net](http://macrotrends.net))*

*低银价意味着银矿一直难以盈利，导致它们投入勘探项目和增加产量的资金越来越少。请注意下图中*十大白银生产国名单中的每个国家*开采的白银都在减少。*

*![](img/67cd09b463ea4b962e6c12657aa27fdb.png)*

*Mine Production from Top 10 Silver Producing Countries ([CPM Yearbook](https://www.cpmgroup.com/store/cpm-silver-yearbook-2019/), [GoldSilver Blog Post](https://goldsilver.com/blog/silver-in-charts-supplydemand-crunch-after-years-of-the-opposite/))*

*此外，即使白银价格飙升，银矿也需要一段时间，甚至几年的时间来增加产量——这种“滞后”可能意味着白银需求高而供应不足的时期会延长。*

> *“……即使白银价格飙升，银矿也需要一段时间，甚至几年才能增加产量”*

****但是等等，还有。****

*根据 GoldSilver 今年早些时候的博客文章，白银的“矿山产能”——预计从新矿进入市场的白银量——已经急剧下降(引用一句话，跌落悬崖)；自 2013 年以来，它已经下跌了近 90%，这也是白银价格低迷的结果(低价格意味着人们不太可能开始新的银矿)。*

*此外，根据 CPM 年鉴[和](https://www.cpmgroup.com/store/cpm-silver-yearbook-2019/)，白银废料**也有所下降**(20 年来的最低点)——同样，低白银价格似乎是罪魁祸首，因为废料经销商发现越来越难赚钱。*

*![](img/5014ddbc44f0ad2287341595aa50589b.png)*

*Global Silver Scrap Supply Per Year ([CPM Yearbook](https://www.cpmgroup.com/store/cpm-silver-yearbook-2019/), [GoldSilver Blog Post](https://goldsilver.com/blog/silver-in-charts-supplydemand-crunch-after-years-of-the-opposite/))*

*现在，如果全球白银需求增加，但供应却停滞不前，那么白银将从何而来？再次引用戈德西尔弗的话:“不是来自政府储备。政府不再囤积白银。没有一个国家在其货币中使用它，所以他们不再囤积它”。*

*![](img/25ff943b248f4c074efff3d6cbba4cab.png)*

*Global Government Silver Inventories Per Year ([CPM Yearbook](https://www.cpmgroup.com/store/cpm-silver-yearbook-2019/), [GoldSilver Blog Post](https://goldsilver.com/blog/silver-in-charts-supplydemand-crunch-after-years-of-the-opposite/))*

*![](img/ef3a1d174fc146fceaa6b8be9a41ea7a.png)*

*Top Silver Producing Countries and Companies ([The Silver Institute World Silver Survey 2019](https://www.silverinstitute.org/wp-content/uploads/2019/04/WSS2019V3.pdf))*

# *地缘政治和经济*

*虽然前面的章节已经通过[供需理论](investopedia.com/terms/l/law-of-supply-demand.asp)的透镜观察了白银，并特别关注了白银的工业应用，但这一节将采取更为[的宏观经济](investopedia.com/macroeconomics-4689798)方法，并探索可能会在未来显著影响白银价格的几个想法和潜在事件。*

## *“指数泡沫”和另一场衰退？*

*据亿万富翁投资者兼对冲基金经理[雷伊·达里奥](https://en.wikipedia.org/wiki/Ray_Dalio)称，经济以一种简单而机械的方式运行。他在一个精彩的 YouTube 视频中解释了这个概念，但本质上他说经济是由三种主要力量驱动的*

1.  *生产率增长*
2.  *短期债务周期*
3.  *长期债务周期*

*![](img/002f97e4ccaa9c33aa66879796fec040.png)*

*According to Ray Dalio, The Economy is the Result of the Accumulation of Three Forces (CREDIT: [Ray Dalio on YouTube](https://www.youtube.com/watch?v=PHe0bXAIuk0&t=926s))*

*根据 Dalio 的说法，“债务波动”发生在两个大周期(长期和短期)，分别为 75-100 年和 5-8 年；在这个时间点(2019 年 12 月)，我们正处于与 1935 年至 1940 年相似的时期，这一时期呈现出以下共同特征:*

*   *巨大的财富差距(最富有的 0.1%人口的净资产现在相当于最贫穷的 90%人口的总和)*
*   *于是，我们就有了[](https://en.wikipedia.org/wiki/Populism)**的民粹主义和社会动荡***
*   ***由于前两次经济衰退(20 世纪 90 年代和 2008 年)，我们的货币政策也不太有效，因此降低利率和量化宽松作为扭转下一次经济衰退的手段不会那么有效***
*   ***我们正进入经历“紧缩周期”(类似于 1937 年)的阶段***

***用 Dalio 的话[来说](https://www.youtube.com/watch?v=4eBoFSVakdo):“下一次衰退将会发生，可能是在几年内”，而且将会是一次社会和政治问题将会很严重的衰退——这使得中央银行使用的“杠杆”(利率、量化宽松)的有效性下降，这将意味着我们将很难“扭转”下一次衰退。这将是一个艰难的决定。***

> ***“下一次经济低迷可能会在未来几年内发生”——雷伊·达里奥，2019 年 9 月***

***除此之外，像迈克尔·伯利(Michael Burry)这样的人(成功预测 2008 年房地产市场泡沫的人)最近一直在谈论 ***我们目前可能正处于一个指数泡沫*** *。* Burry 指出，“被动投资正在推高股票和债券价格，就像 10 多年前抵押债务债券对次级抵押贷款的作用一样……就像大多数泡沫一样，持续时间越长，崩溃就越严重”。***

> ***“被动投资正在抬高股票和债券价格，就像 10 多年前抵押债务债券对次级抵押贷款的影响一样，”Burry 告诉彭博新闻—迈克尔·伯里，2019 年 9 月***

***现在我不知道我们是否真的处于泡沫之中。也不知道下一次低迷/衰退还有多远。但我认为我们还没有最后一次。在经济低迷时期会发生什么？答案是投资者涌向贵金属——比如白银——这将意味着白银需求的巨大增长。***

## ***亚洲***

***国家**中国**和**印度**值得在这篇文章中特别提及。当谈到白银时，这两个国家似乎尤其值得关注。在过去的 18 年里，中国对白银加工的需求增长了 420%——通货膨胀率低于美国。中国也是 2018 年白银需求的主要贡献者， ***占新增太阳能电池板安装量*** 的近一半，报告称，白银产量较 2017 年增长 2%，实现 1.149 亿金衡盎司；Silvercorp 的 Ying 矿区获得了最高的改进(产量和品位)。***

***![](img/7a6833fc4d4128423864142c36a81f11.png)***

***Chinese Silver Fabrication Demand Per Year, ([CPM Yearbook](https://www.cpmgroup.com/store/cpm-silver-yearbook-2019/), [GoldSilver Blog Post](https://goldsilver.com/blog/silver-in-charts-supplydemand-crunch-after-years-of-the-opposite/))***

***![](img/86c6e54fbe28e1dca97e705a9f37f521.png)***

***Annual Chinese Industrial Silver Fabrication (Left) and Asian Mine Production (Right), [World Silver Survey](https://www.silverinstitute.org/wp-content/uploads/2019/04/WSS2019V3.pdf)***

***![](img/dcb61b513d80df75707948b7d24f1f39.png)***

***Silver Supply Chain in Global PV Industry (2018), The Silver Institute***

***![](img/b5cf9195393d5d6782c65b6deae10e14.png)***

***Top Exporters of Photosensitive/photovoltaic/LED Semiconductor Devices (2017), [Observatory of Economic Complexity](https://oec.world/en/visualize/geo_map/hs92/export/show/all/854140/2017/)***

***![](img/04549e7a56e07038de3cab8c569974b7.png)***

***Top Importers of Photosensitive/photovoltaic/LED Semiconductor Devices (2017), [Observatory of Economic Complexity](https://oec.world/en/visualize/geo_map/hs92/export/show/all/854140/2017/)***

***从上述图表来看，中国似乎将受益于电子和可再生能源驱动的白银需求的大幅增长(假设需求确实增长)。***

***投资中国的白银供应端(矿山和废料公司)可能确实是个好主意——因为这是中国最廉价满足其需求的地方。***

***![](img/e5c5eab1620bc57c6eaf1f1a2144fa28.png)***

***Top Exporters of Silver, [Observatory of Economic Complexity](https://oec.world/en/profile/hs92/7106/)***

***![](img/ba00f7ae1f6c824ddf2fc5be24acd0ba.png)***

***Top Importers of Silver, [Observatory of Economic Complexity](https://oec.world/en/profile/hs92/7106/)***

***另一方面，印度似乎对 ***白银作为价值储存手段更感兴趣。*** 看上面的树形图，印度是 2017 年白银的最大进口国。纵观整个世界，在连续两年下降后，2018 年全球 ***实物白银购买总量增长了 5%***至 1.61 亿金衡盎司，而这一需求飙升在很大程度上归因于印度。印度的实物白银购买量增长了 53%，也主导了银锭贸易，进口量增长至 36%(2015 年以来的最高水平)。***

***2018 年，印度的银首饰制造量同比增长 16%，至 7，650 万金衡盎司(2，378 吨)。 ***未来几年，印度可能确实是全球白银需求的重要驱动力。******

# ***关于未来和其他花絮***

***在这一部分，我决定把其他所有东西都塞进去——那些似乎不适合放在其他地方的东西。我们将看看白银未来可能的应用，以及一些技术分析(黄金/白银比率)。***

## ***在未来观看***

*   ***电动汽车***
*   ***银纳米粒子与应急水处理***
*   ***航天工业中的白银***

***![](img/e9158b83fd46457b6cfe97a3435d5486.png)***

***Silver Demand in the Automotive Industry, Silver Institute***

***由于其优异的导电性，银被广泛用于电动动力系统和其他应用中，这些应用越来越多地出现在混合动力内燃机(ICE)汽车和电动汽车(ev)中。此外——关于水处理——纳米技术可能会创造一种环境，在这种环境中，银在紧急水处理中可能非常有效，这就是为什么美国宇航局将其列为未来太空探索的生物杀灭剂。***

## ***金银比率***

***[金银比率](https://www.sbcgold.com/blog/what-is-the-gold-silver-ratio-and-why-does-it-matter/)代表购买一枚等量黄金所需的白银数量。有趣的是，这一比率接近历史最高水平，这表明白银相对于黄金越来越被低估。***

***![](img/f89877f8d6243ad9c6abebd81628afbf.png)***

***The Ratio of the Gold Price to the Silver Price, Per Year Since 1969 (CREDIT: Author in Jupyter notebook)***

***上面的图表是使用下面的代码绘制的，基本上是使用 [Plotly](https://plot.ly/) 和[袖扣](https://plot.ly/python/v3/ipython-notebooks/cufflinks/)来绘制黄金&白银价格的比率(从互联网上搜集的)。***

```
***import pandas as pd
import requests
from bs4 import BeautifulSoup
## Silver 
res = requests.get("[https://www.macrotrends.net/1470/historical-silver-prices-100-year-chart](https://www.macrotrends.net/1470/historical-silver-prices-100-year-chart)")
soup = BeautifulSoup(res.content, 'lxml')
tables = soup.find_all('table')
headers = ['Year', 'Average Closing', 'Year Open', 'Year High', 'Year Low', 'Year Close', 'Annual % Change']
df_silver = pd.DataFrame(pd.read_html(str(tables[0]))[0])
df_silver.columns = headers## Gold
res = requests.get("[https://www.macrotrends.net/1333/historical-gold-prices-100-year-chart](https://www.macrotrends.net/1333/historical-gold-prices-100-year-chart)")
soup = BeautifulSoup(res.content, 'lxml')
tables = soup.find_all('table')
headers = ['Year', 'Average Closing', 'Year Open', 'Year High', 'Year Low', 'Year Close', 'Annual % Change']
df_gold = pd.DataFrame(pd.read_html(str(tables[0]))[0])
df_gold.columns = headersdef convert_to_float_if_applicable(cell):
    if isinstance(cell, str) and '$' in cell:
        try:
            cell = float(cell.replace('$', '').replace(',',''))
            return cell
        except:
            return cell
    elif isinstance(cell, str) and '%' in cell:
        try:
            cell = float(cell.replace('%', ''))
            return cell
        except:
            return cell
    else:
        return celldf_silver = df_silver.applymap(lambda cell: convert_to_float_if_applicable(cell))
df_gold = df_gold.applymap(lambda cell: convert_to_float_if_applicable(cell))
df_gold.rename(columns={'Average Closing': 'Average Closing, Gold'}, inplace=True)
df_silver.rename(columns={'Average Closing': 'Average Closing, Silver'}, inplace=True)
precious_metals_avg_cls = pd.merge(df_silver, df_gold, how='inner', on = 'Year')
precious_metals_avg_cls.set_index('Year', inplace=True)ratio_gold_to_silver = precious_metals_avg_cls
ratio_gold_to_silver['Ratio of Gold to Silver'] = precious_metals_avg_cls['Average Closing, Gold'] \
/ precious_metals_avg_cls['Average Closing, Silver']precious_metals_avg_cls[['Ratio of Gold to Silver']]\
.iplot(kind='line', xTitle='Year', color='#1C7C54',
                  yTitle='Ratio of Gold Closing Price, to Silver Closing Price', 
       title='Ratio of Gold Price to Silver Price', width=4)***
```

***就这些了，伙计们！感谢你的阅读，如果你已经做到了这一步，那么恭喜你——我喜欢做这个，希望你们中的一些人能从中发现一些价值。请不要把这篇文章当成强硬的金融/投资建议——我不是一个有经验的投资者，白银甚至可能是一个糟糕的投资。我很想知道这种金属在未来会发生什么。***

******再次感谢阅读！******

***![](img/c25f5d62b39ea06b25ce991ab09f65b9.png)***

***(CREDIT: Author in Jupyter on [Canva](http://canva.com))***

# ***参考***

*   ***[http://www . silver institute . org/WP-content/uploads/2017/08/eosilverdemandreportdec 2016 . pdf](http://www.silverinstitute.org/wp-content/uploads/2017/08/EOSilverDemandReportDec2016.pdf)***
*   ***[https://www . silver institute . org/WP-content/uploads/2019/04/WSS 2019 v3 . pdf](https://www.silverinstitute.org/wp-content/uploads/2019/04/WSS2019V3.pdf)***
*   ***[https://www . jmbullion . com/investing-guide/types-physical-metals/silver-solar-demand/](https://www.jmbullion.com/investing-guide/types-physical-metals/silver-solar-demand/)***
*   ***[https://www . PV-magazine . com/2018/05/11/silver-prices-to-drop-4-toz-by-2030-world-bank-says/](https://www.pv-magazine.com/2018/05/11/silver-prices-to-drop-by-4-toz-by-2030-world-bank-says/)***
*   ***[https://www . who . int/water _ sanitary _ health/publications/silver-02032018 . pdf？ua=1](https://www.who.int/water_sanitation_health/publications/silver-02032018.pdf?ua=1)***
*   ***[https://www . PV-magazine . com/2018/07/06/amount-of-silver-institute-says/](https://www.pv-magazine.com/2018/07/06/amount-of-silver-needed-in-solar-cells-to-be-more-than-halved-by-2028-silver-institute-says/)***
*   ***[https://www.sbcgold.com/blog/silver-supply-demand/](https://www.sbcgold.com/blog/silver-supply-demand/)***
*   ***[https://seeking alpha . com/article/4044219-足够-银-电力-世界-甚至-太阳能-电力-效率-四倍](https://seekingalpha.com/article/4044219-enough-silver-power-world-even-solar-power-efficiency-quadruple)***
*   ***[https://gold silver . com/blog/silver-in-charts-supply demand-crunch-after-years of-the-opposite/](https://goldsilver.com/blog/silver-in-charts-supplydemand-crunch-after-years-of-the-opposite/)***
*   ***[https://www . science daily . com/releases/2019/04/190417102750 . htm](https://www.sciencedaily.com/releases/2019/04/190417102750.htm)***
*   ***[https://www.statista.com/topics/1335/silver/](https://www.statista.com/topics/1335/silver/)***
*   ***[https://www . macro trends . net/1470/historical-silver-prices-百年走势图](https://www.macrotrends.net/1470/historical-silver-prices-100-year-chart)***
*   ***【https://www.silverinstitute.org/silver-supply-demand/ ***
*   ***[https://oec.world/en/](https://oec.world/en/profile/hs92/7106/)***