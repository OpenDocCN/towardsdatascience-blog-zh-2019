# 火箭(数据)科学

> 原文：<https://towardsdatascience.com/rocket-data-science-d81a4375d5aa?source=collection_archive---------10----------------------->

过去几年，中东集团相当吵闹。作为一个被许多敌人包围的小国，以色列一直处于冲突之中。简言之(根据以色列外交部的说法):

**来自东北**——因“隔壁”叙利亚战争引发的边境事件

**来自北方**——来自黎巴嫩边境的真主党威胁

**来自南方**——ISIS 在埃及边境附近的敌对活动

**来自西方**——伊斯兰圣战组织/哈马斯试图从加沙地带发动恐怖袭击

**来自东部**——朱迪亚和萨马拉暴力再起

据谷歌称，自 2001 年以来，巴勒斯坦武装分子从加沙地带向以色列发动了数千次火箭/迫击炮袭击，作为持续的巴以冲突的一部分。

如果你在以色列生活了几个月以上，你很可能熟悉臭名昭著的警报“红色代码”(希伯来语:צבע אדום)，这意味着由于火箭发射，人们必须立即找到避难所。

撇开政治不谈，我认为尝试对火箭发射进行可视化建模(通过使用警报日志)可能是一个很酷的项目，由于我在以前的文章中被多次问到，我决定尝试并详细说明我在处理这样一个项目时使用的方法和工具。

# **A 部分—获取相关数据**

第一步是在谷歌上搜索以色列警报的某种“记录”数据。

上面提到的搜索引导我进入“Home Front Command”网站，在那里我可以查询以色列过去的警报:

![](img/f98a20577a94bb61a9bf7fbc660a9561.png)

*Home Front Command’s site*

结果产生了一长串过去的警报:

![](img/172d436da0a0754c24d7298b81e5539b.png)

*Small portion of the data*

就我个人而言，我更喜欢使用纯文本/表格数据，快速浏览 Chrome 的 DevTools 可以发现查询请求/响应的 API(这只是使用搜索按钮时“在幕后”发出的网络请求)，这使得可以立即获取纯文本答案:

![](img/5630bb93a7db00cc3cb8b34426dc7b00.png)

*The query URL*

![](img/166a272721ff8176d4fde83027920ac5.png)

*The website’s text response*

总之，我提取了从 2014 年 7 月(可用的较早日期)到 2019 年 5 月的警报历史。

# **B 部分—丰富数据**

首先，原始数据没有指定确切的位置，而只是“区号”(由于历史和技术原因)，所以我必须以某种方式构建“区号到城市列表转换器”。

尽管没有公开的“字典”来记录后方司令部的代号，但谷歌搜索发现，每个代号在维基百科上都有一个独特的页面，囊括了该地区城市的完整列表:

![](img/af297be82e4cf96beabbd1362ec9c2f4.png)

*Google search results show a re-occurring pattern of Wikipedia page per area code*

因此，使用一个简短的 Python 脚本使我能够获得每个区域的城市列表(没什么特别的，使用 GET request per area，并提取页面主体中的所有 URL 名称)。

第二，我不得不[分解](https://medium.com/@sureshssarda/pandas-splitting-exploding-a-column-into-multiple-rows-b1b1d59ea12e)数据集，以使每个警报作为一个独特的行出现在对等城市，这样它就可以在以后作为独立的数据呈现。

最后，我使用开源的[地理 API](https://opencagedata.com/) 将城市名称转换成 GPS 坐标。

总而言之——这是数据经历的过程:

![](img/2eda06bc78b5f80ba98499fd962e8648.png)

The procedure — visually

# **C 部分—数据清理/完整性**

在这一部分中，我必须进行一些合理性检查，包括:

1.手动修复错误刮取的值——因为刮取永远不会完美。

2.手动修复不可能的 GPS 坐标-由于模糊的城市名称，一些城市似乎不在以色列境内。

3.删除不相关的数据-警报系统测试或国家危机演习等事件也包含在数据集中，因此特定的时间/日期被删除(它们很容易过滤掉，因为它们总是在特定的月份在所有城市同时发生)。

4.标记不适用数据—例如由于缺乏维基百科/谷歌中的信息，一些地区无法翻译成城市列表——不幸的是，这些数据没有出现在项目中。

# **D 部分—可视化数据**

我第一个也是最喜欢的可视化聚合工具是 Tableau，使用这个工具可以非常容易地呈现数据:

![](img/1a679a0a968c5ce2d10855467f914e9f.png)

*Heat map of “Code Red” alarms over the past 5 years*

![](img/3d40e943e8723e7f19087a6ee2c8cbbe.png)

*Heat map of “Code Red” alarms per month (interesting to notice the increase during summer periods of July-August, the Israeli summer)*

![](img/637a972cba35930e4a416a53ba55118a.png)

*Full Heat map of “Code Red” alarms per distinct months. note: data is available only from July 2014*

除此之外，一个朋友向我介绍了一个神奇的开源工具，叫做[开普勒](https://kepler.gl/)，用于地理空间数据，在其中你可以看到数据的动画，甚至可以自己重塑它:

![](img/81c2ce34a62117495a25312b511c3b4d.png)

Top siren activity

(note when exploring the animated data: while the launch destination is an exact GPS coordinate, the launch source was added only for aesthetic purposes and it is not accurate)

bird’s eye view of alarms in the past 5 years

你也可以在下面的链接中使用这些数据，自己制作其他很酷的动画:

[](https://kepler.gl/demo/map?mapUrl=https://dl.dropboxusercontent.com/s/ynd93swqccgnebr/keplergl_gxj4nuk.json) [## 火箭(数据)科学— Yoav Tepper

### Kepler.gl 是一个强大的基于网络的地理空间数据分析工具。基于高性能渲染引擎和…

开普勒](https://kepler.gl/demo/map?mapUrl=https://dl.dropboxusercontent.com/s/ynd93swqccgnebr/keplergl_gxj4nuk.json) 

# **E 部分——留在上下文中**

这一部分与数据无关，我只是想利用这个平台来强调，即使一些数据可能看起来很迷人/美丽，但人们永远不要忘记背后的故事。在这种情况下，尽管我很喜欢这个项目，但我希望这些数据一开始就不存在。

希望你觉得这篇文章有趣/有用，一如既往地欢迎你的反馈。