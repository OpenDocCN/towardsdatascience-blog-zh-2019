# 用深度学习对抗污染

> 原文：<https://towardsdatascience.com/fighting-pollution-with-deep-learning-694dd6259b36?source=collection_archive---------29----------------------->

![](img/1e4c0cedad950d3ab7c5f7ac743f8324.png)

A brick kiln spewing smoke. Photo by [koushik das](https://unsplash.com/@7890857439kd?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/delhi?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

作者:[罗希特·辛格](https://medium.com/@rohitgeo)；[桑迪普·库马尔](https://medium.com/@sandeepgadhwal1)

每年 11 月，新德里都会被厚厚的烟雾笼罩。作为回应，政府采取了一些照本宣科的措施，比如偶尔关闭学校，实施臭名昭著的[单双号限行计划](https://en.wikipedia.org/wiki/Odd%E2%80%93even_rationing#India)，迫使一半城市的汽车离开街道。德里的污染有几个原因:邻近邦的季节性残茬燃烧、车辆排放、以及发电厂和遍布首都地区的砖窑排放的烟雾。抗击污染需要多管齐下的方法——仅靠政府政策是不够的，它们需要与实地行动相结合。

# 照片不会说谎

我们以砖窑为例。在我们解决它们造成的污染之前，我们需要确切地知道有多少这样的窑和它们的位置，它们的数量是在增加还是在减少，以及有多少按照法律的要求采用了减少排放的技术。**卫星图像与深度学习相结合可以回答这些问题，增强问责制并推动实地成果。这篇博文描述了我们如何做到这一点。**

# 砖窑

随着快速城市化，对砖的需求不断增加。在亚洲的许多地方，制砖是一项非常庞大的传统产业。印度的制砖业虽然没有组织，但规模巨大。印度是世界上第二大砖块生产国——中国以 54%的份额占据主导地位[1]。

![](img/1a6f402494b80230d114ffd8b1be84ce.png)

Brick kilns are a major contributor to air pollution in North India. Photo by Ishita Garg

砖窑是印度北部空气污染的主要原因。生产砖块会导致有害气体和颗粒物质的排放，包括一氧化碳和硫氧化物(SOx)等有毒化学物质。接触这些排放物对健康极其有害，并会影响儿童的身体和智力发育。

印度的大多数砖窑都是**固定烟囱公牛沟窑** (FCBTK)型。然而，有一种更新的窑炉设计，被称为**之字形窑炉**，它更节能，造成的空气污染更少。旧设计的窑炉可以转换成之字形设计，从而提高效率——但是，改造需要成本，这导致这项技术的采用速度较慢。

![](img/a90ef3af888f88f14cdcc1f702be83ed.png)

Source: [Towards Cleaner Brick Kilns in India](https://shaktifoundation.in/wp-content/uploads/2014/02/CATF-2013-Towards-Cleaner-Brick-Kilns-in-India.pdf) by Sameer Maithel

**中央污染控制委员会(CPCB)于 2017 年 6 月发布了一项** [**指令**](https://www.cseindia.org/content/downloadreports/9387) **，强制要求全印度所有的砖窑转换成锯齿形设计**。该指令明确指出未经许可的砖窑将被关闭[2]。尽管有指令，仍有许多砖窑没有按照规定的设计规范运行。

在卫星图像中，FCBTK 窑看起来是椭圆形的，而新设计的锯齿形窑是矩形的(只有空气以锯齿形流动！)这可以用来从航空/卫星图像中识别窑的类型。

![](img/2f6dd4392bfcf8eeb9603496a04b94b3.png)

We can identify Zigzag (left) and FCBTK kilns (right) from their layout, as seen in satellite imagery.

在这篇文章中，我们描述了我们如何使用深度学习来检测德里周围的所有砖窑，绘制它们的地图，并根据它们的设计进行分类。这将有助于发现那些没有遵守政府指令的窑炉，并在执法方面大有作为，同时增加问责制和透明度。

# 使用 ArcGIS 进行深度学习

深度学习是一种久经考验的卫星图像对象检测方法。我们使用以下步骤对砖窑进行检测和分类:

*   使用 ArcGIS Pro 收集训练数据
*   使用 arcgis.learn 训练深度学习模型
*   使用 ArcGIS Pro 部署训练好的模型

最后，我们创建了一个 ArcGIS Operations 仪表盘来传达分析结果。

*我们使用 ESRI 世界影像图层来训练模型，为了进行对比分析，我们使用了 2014 年的同一图层(此历史影像可在 Esri 的 Living Atlas 上找到，可使用* [*wayback 影像工具*](https://livingatlas.arcgis.com/wayback/) *进行浏览)。*

# 收集培训数据

我们使用 ArcGIS Pro 在 Esri World 影像上标记了两种砖窑的位置。我们创建了一个表示砖窑位置的点要素类，并设置了一个指示砖窑类型的属性字段(0=FCBTK/Oval design。1 =锯齿形/矩形)。为了简化我们的工作，我们只标记了窑的中心位置——我们只对它们的位置感兴趣，而不是精确的大小。

![](img/4dbc2fbc5dfa0d8e04e500784a9487d6.png)

Marking locations of brick kilns and their types (red=FCBTK, blue=Zigzag) using ArcGIS Pro. This served as training data that our AI model `learnt` from.

# 导出培训数据

![](img/90e149d09b8543706c8675eb50ac4601.png)

这些数据被用来训练深度学习模型，以检测图像中的砖窑。我们使用了 [ArcGIS Pro](https://app.reviewnb.com/Esri/arcgis-python-api/pull/506/files/#%22https://pro.arcgis.com/en/pro-app/tool-reference/image-analyst/export-training-data-for-deep-learning.htm%22) 中提供的“导出深度学习训练数据”工具来导出包含多个砖窑示例及其在每个芯片中的位置的图像芯片。

我们选择每个窑炉位置周围的缓冲半径为 75 米，因为每个窑炉大约是该尺寸的两倍(即大约 150 米长)。

需要根据将要训练的模型类型来选择元数据格式。在这种情况下，我们必须训练一个对象检测模型。“Pascal 可视对象类”格式是一种用于对象检测任务的流行元数据格式。

在“环境”选项卡中，我们可以调整“单元大小”参数，以便每个芯片可以容纳两到三个砖窑。对于这个项目，我们使用不同的单元大小参数值导出芯片。这个技巧使我们能够通过从相同数量的标记数据引导中创建更多的训练芯片来增加训练数据。此外，它还帮助我们的模型学会为砖窑创建更合适的边界框。如果我们只输入一个单元大小的模型数据，它将总是为每个窑预测相同的大小(大约 150m ),因为这是它已经看到的所有数据。

# 砖窑检测人员培训

我们使用 Jupyter 笔记本和 ArcGIS API for Python 中的`arcgis.learn`模块来训练模型。`arcgis.learn`模块建立在 [fast.ai](https://docs.fast.ai/) 和 [PyTorch](https://pytorch.org/) 之上，只需几行代码就能训练出高度精确的模型。安装和设置环境的详细文档可在[此处](https://developers.arcgis.com/python/guide/install-and-set-up/)获得。

我们训练的模型类型是 SingleShotDetector，之所以这样叫是因为它能够一眼找到图像(芯片)中的所有对象。

```
**from** **arcgis.learn** **import** SingleShotDetector, prepare_data
```

# 数据扩充

我们使用`prepare_data()`函数对训练数据进行各种类型的转换和扩充。这些增强使我们能够用有限的数据训练更好的模型，并防止模型过度拟合。`prepare_data()`取 3 个参数。
`path`:包含训练数据的文件夹路径。
`chip_size`:与导出培训数据时指定的相同。
对于这个项目，我们在 11GB 内存的 GPU 上使用了 64 个的批处理大小。

此函数返回 fast.ai 数据束，在下一步中用于训练模型。

```
**from** **arcgis.learn** **import** SingleShotDetector, prepare_datadata = prepare_data(path=r'data\training data 448px 1m',
                    chip_size=448, 
                    batch_size=64)
```

## 从您的训练数据中可视化一些样本

为了理解训练数据，我们将在`arcgis.learn`中使用`show_batch()`方法。`show_batch()`从训练数据中随机选取几个样本，并将其可视化。

```
data.show_batch(rows=5)
```

![](img/6c785e9487aae31a107d77900cc41ed2.png)

Some random samples from our training data have been visualized above.

上面的图像芯片标出了砖窑的边界框。标有 0 的方框是椭圆形(FCBTK)砖窑，标有 1 的是锯齿形砖窑。

# 加载单发探测器模型

下面的代码实例化了一个`SingleShotDetector`模型——它基于一个流行的**对象检测**模型，它的缩写形式“SSD”更为人所知。该模型返回检测到的特征的类型和边界框。

```
model = SingleShotDetector(data)
```

# 找到一个最佳的学习速度

一个新初始化的深度学习模型就像一个刚出生的孩子的大脑。它不知道从什么开始，并通过查看它需要学习识别的对象的几个示例来学习。如果它学习的速度非常慢，它需要很长时间才能学会任何东西。另一方面，如果孩子很快就下结论(或者，在深度学习术语中具有“高学习率”)，它将经常学习错误的东西，这也不好。

类似地，深度学习模型需要用学习率来初始化。这是一个重要的*超参数*，其值应在学习过程开始前设置[3]。学习率是一个关键参数，它决定了我们如何根据损失梯度调整网络的权重[4]。

`arcgis.learn`利用 fast.ai 的学习率查找器为训练模型找到最佳学习率。我们可以使用`lr_find()`方法来找到能够足够快地训练我们的模型的最佳学习速率。

```
model.lr_find()
```

![](img/27803d70edf0f53bb3dbb3786a6e9484.png)

根据上面的学习率图，我们可以看到 lr_find()为我们的训练数据建议的学习率大约是 1e-03。我们可以用它来训练我们的模型。在最新发布的`arcgis.learn`中，我们甚至可以在不指定学习率的情况下训练模型。它在内部使用学习率查找器来查找最佳学习率并使用它。

# 符合模型

为了训练模型，我们使用`fit()`方法。首先，我们将使用 10 个纪元来训练我们的模型。Epoch 定义了模型暴露于整个训练集的次数。

```
model.fit(epochs=10, lr=0.001)
```

![](img/62d776652f6da0c245d002010ba11312.png)

`fit()`方法的输出给出了训练集和验证集的损失(或错误率)。这有助于我们评估模型对未知数据的泛化能力，并防止过度拟合。在这里，只有 10 个时期，我们看到了合理的结果-训练和验证损失都大幅下降，表明模型正在学习识别砖窑。

下一步是保存模型，以便以后进一步训练或推断。默认情况下，模型将保存到本笔记本开头指定的数据路径中。

# 保存模型

我们将把我们训练的模型保存为'**深度学习包**'('。dlpk’格式)。深度学习包是用于在 ArcGIS 平台上部署深度学习模型的标准格式。

我们可以用`save()`的方法来保存训练好的模型。默认情况下，它保存在我们的培训数据文件夹中的'**模型**'子文件夹中。

```
model.save('ssd_brick-kiln_01')
```

# 加载一个中间模型来进一步训练它

为了重新训练一个保存的模型，我们可以使用下面的代码再次加载它，并按照上面的解释进一步训练它。

```
*# model.load('ssd_brick-kiln_01')*
```

# 可视化验证集中的结果

查看模型 viz-a-viz 地面真相的结果是一个很好的实践。下面的代码选取了随机样本，并向我们并排展示了基本事实和模型预测。这使我们能够在笔记本中预览模型的结果。

```
model.show_results(rows=8, thresh=0.2)
```

![](img/9ebd40b62d9df9b78ddd09251a20abe1.png)

Detected brick kilns, ground truth on left, and the predictions on the right.

在这里，来自训练数据的地面实况的子集与来自模型的预测一起被可视化。正如我们所看到的，我们的模型运行良好，预测与地面实况相当。

# 砖窑模型部署与检测

我们可以使用保存的模型，使用 [**ArcGIS Pro**](https://pro.arcgis.com/en/pro-app/tool-reference/image-analyst/detect-objects-using-deep-learning.htm) 和[**ArcGIS Enterprise**](https://developers.arcgis.com/rest/services-reference/detect-objects-using-deep-learning.htm)中提供的“**使用深度学习检测对象**”工具来检测对象。对于该项目，我们使用 ESRI 世界影像图层来检测砖窑。

![](img/fa01969c4a5ff9d14e2e840a8d2bc33e.png)

Detected brick kilns in ArcGIS Pro. See the [web layer](https://geosaurus.maps.arcgis.com/home/webmap/viewer.html?webmap=711b1725f6334aeca2151734b37d3c50) with the detected kilns

该工具返回一个地理要素图层，可使用[定义查询](https://pro.arcgis.com/en/pro-app/help/mapping/layer-properties/definition-query.htm)和[非最大值抑制工具](https://pro.arcgis.com/en/pro-app/tool-reference/image-analyst/non-maximum-suppression.htm)进一步细化该图层。该图层也已在线发布并作为托管要素图层共享[。](https://geosaurus.maps.arcgis.com/home/webmap/viewer.html?webmap=711b1725f6334aeca2151734b37d3c50)

一旦我们有了一个经过训练的模型，我们就可以用它来以完全自动化的方式检测感兴趣的对象，只需查看图像即可。为了进行对比分析，我们使用相同的模型在 2014 年的旧版本 [ESRI 世界影像图层上检测砖窑，该图层是使用](https://www.arcgis.com/home/item.html?id=903f0abe9c3b452dafe1ca5b8dd858b9) [wayback 影像工具](https://livingatlas.arcgis.com/wayback/)找到的。

![](img/304a8b63d2688f16ce87ec9895055478.png)

[Web map](https://geosaurus.maps.arcgis.com/home/webmap/viewer.html?webmap=711b1725f6334aeca2151734b37d3c50) showing the detected brick kilns.

# 使用 ArcGIS Dashboard 交流结果

数据科学可以帮助我们从数据中获得洞察力，但将这些洞察力传达给利益相关者和决策者或许同样重要，如果不是更重要的话。

以仪表板的形式显示结果可以回答关键问题，提高透明度和问责制，并推动实地成果。我们使用 ArcGIS Operations Dashboard 创建了以下仪表盘来传达我们的分析结果:

![](img/27e642f101764ce983412ef53e2d8c6d.png)

[Online Dashboard](https://geosaurus.maps.arcgis.com/apps/opsdashboard/index.html#/b934421b79fc4be08c345609860a5a6e) showing growth in number of brick kilns and very poor (less than 14%) compliance with order mandating conversion to Zigzag design

从仪表板上可以明显看出，砖窑的数量在过去几年中有所增加，我们在将它们改造成新设计方面还有很长的路要走。不到 14%的人转向了污染更少的锯齿形设计。像这样有真实数据支持的信息产品可以增加透明度、分配责任和推动实地成果。

# 参考

[1]谢赫·阿费法。(2014).砖窑:大气污染的原因。

[2]https://www.cseindia.org/content/downloadreports/9387

[3][https://en . Wikipedia . org/wiki/Hyperparameter _(机器学习](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))

[4][https://towards data science . com/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d 4059 c1c 10](/understanding-learning-rates-and-how-it-improves-performance-in-deep-learning-d0d4059c1c10)

# 加入我们吧！

在新德里的 **ESRI 研发中心，**我们正在应用前沿的人工智能和深度学习技术来革新地理空间分析，并从图像和位置数据中获得洞察力。我们正在寻找数据科学家和产品工程师。在线申请[这里](https://www.esri.com/en-us/about/careers/job-search#@careerPath=@location=IN-DL-New%20Delhi@jobSearch=)！