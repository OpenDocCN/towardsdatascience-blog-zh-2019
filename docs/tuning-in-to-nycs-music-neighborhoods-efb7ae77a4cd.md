# 收听纽约的音乐街区

> 原文：<https://towardsdatascience.com/tuning-in-to-nycs-music-neighborhoods-efb7ae77a4cd?source=collection_archive---------17----------------------->

## K-Means 通过音乐档案聚类邻居

# 摘要

机器学习允许创建能够识别多维数据集中的模式的计算模型。这个项目旨在利用 Foursquare 的“Places API”和一种叫做“k-means clustering”的机器学习算法来识别具有类似“音乐档案”的“纽约市”社区。

# 介绍

## 背景

音乐是一种艺术形式，已经并将永远深深地嵌入城市、社区和更广泛的人群的文化活动中。音乐是一种交流和表达的方式，有时甚至是抗议的方式，它有能力和平地将大量志同道合的人聚集在一起，影响流行文化，用一首令人难忘的歌词催眠你，让你在淋浴时潜意识地连续几周唱歌，即使在有意识地对自己这样做感到失望之后…..我跑题了……

## 问题

城市在某种程度上是由音乐实体组成的，如唱片店、乐器供应商、音乐厅、圆形剧场等等，它们不仅满足当地居民的音乐需求，也满足来自世界各地的游客的需求。对于较大的城市，音乐实体可以分散开来，从而形成一个随着时间的推移而发展和变化的时尚利基社区生态系统。这种生态系统通常由寻找酷音乐场景的人通过自然生活体验(流浪/浪荡)或以互联网评论、评论和与现实生活中的人交谈的形式推荐来了解。

这个项目旨在量化一个大都市纽约市的社区的“音乐档案”,以识别相似音乐场景的集群。

## 利益相关者

不同的团体可能对能够基于可用音乐商店的类型来量化邻域相似性的模型感兴趣。这种模型将能够通知那些喜欢住在音乐发生的地方的租房者和购房者，他们的下一个家就在合适的位置。未来的音乐场所初创企业可以利用该模型来识别缺乏现场音乐场所的社区，并确保他们投资于一个未饱和的地区。未来的音乐零售商，如唱片和乐器销售商，同样可以利用这种模式来确保他们开展的业务中竞争对他们有利。

# 方法学

## 数据源

*NYU 空间数据仓库:*我使用 NYU 空间数据仓库托管的“2014 年纽约市街区名称”数据集作为街区名称和相关位置质心的基础[0]。下图显示了此信息的一个示例:

![](img/c2aed65b0f9d3d41979d5c7caa23a1f4.png)

DataFrame created from NYU’s ‘2014 New York City Neighborhood Names’

*Foursquare—‘Places API’*:我将使用 four square 的‘Places API’来获取与‘场馆’(four square 定义的)相关的数据，这些数据被归类为与音乐有某种关联[1]。值得注意的是，Foursquare 将“地点”定义为一个人们可以去或签到的地方，并且“地点”不一定是音乐地点，而是可以是任何机构，如餐馆或零售商店。每个 Foursquare“地点”被分配一个“类别”，每个“类别”与一个特定的“类别 ID”相关联。右图显示了 Foursquare 提供的“categoryID”值，该值将用于获取纽约市内与音乐相关的场所:

![](img/c77fc0525a34e60d96887636234e4c2a.png)

Foursquare Music-Related Venue CategoryIDs

## 资料检索

*街区名称&位置质心数据:*由 NYU 空间数据仓库托管的“2014 年纽约市街区名称”数据集可轻松下载为 JSON 文件并导入 Jupyter 笔记本:

![](img/4d0ec0e7c19315eb9544c51b0a8f496c.png)

Importing the newyork_data.json fie

然后，将与每个邻域相关的“区”、“邻域”、“纬度”和“经度”值从 JSON 转换为 Pandas 数据框架，作为分析的基础。

![](img/e6558b566decedfec4e569cfecb428a9.png)

Creating a DataFrame out of the JSON data

*Foursquare 音乐相关场地数据* : 如本报告数据来源部分所述，Foursquare 拥有众多的“[场地类别](https://developer.foursquare.com/docs/resources/categories)，用于标识各类场地。对“API . four square . com/v2/ventures/search？”的“get”请求提供类别 ID 的端点将返回该类别的地点。下面的示例代码向 Foursquare 发送一个“get”请求，请求一个属于“音乐商店”类别的场所(categoryID = ' 4 BF 58 DD 8d 48988 D1 Fe 941735 '):

![](img/032cc2004ad416fc4b8537a57a518bd2.png)

Example Foursquare Places API Request

![](img/9069dd66f99da8eef9bdace112b88c4c.png)

通过递归地发送“get”请求到前面提到的端点，确保结果特定于具有音乐相关“类别 id”的场所，创建了与每个纽约市街区相关的音乐相关场所的初步数据集。对于每个邻域，我们可以在一个“get”请求中包含所有选定的类别 id，方法是将它们作为逗号分隔值传递。下面显示了一个创建所需 url 的函数和一个示例:

![](img/e334a28789672168c76c111799b565a5.png)

Dynamically creating API request URLs

下面的函数递归地向 Foursquare 发送一个“get”请求，请求所有与音乐相关的地点。在遍历 NYU 数据集中的每个邻域时，该函数将每个与音乐相关的地点条目附加到一个列表中，并且在遍历每个邻域后，创建所有结果的数据帧。对于数据集中的每个条目，包括邻居名称和位置，以及地点名称、位置和类别。

![](img/6afd185d9c7c7f3bcabe6b78ad9768d1.png)![](img/67d3cfea02d4549785cab04bce8d20de.png)![](img/6af1acedcd9d385b88d04fe730dc6dde.png)

Recursively retrieving music-related venues for each New York City neighborhood

最终的初步场馆数据框架包括从 Foursquare 中抽取的 9，442 个场馆:

![](img/654f7cdbda5ac75dd8deb799372bc2d3.png)

9,442 venues were pulled from Foursquare

由于我对超出 Foursquare 的 API 速率限制有一些问题，在获得初步数据集后，将一个副本保存到 csv，以便将来的开发不需要从 Foursquare 重新请求信息。

![](img/ae875dd487661e35ac401fb90a266c5e.png)

Write data to csv

![](img/a9777d29492283cdf9c5213663ea6819.png)

Sample of csv file

# 探索性数据分析

下面的一系列图片旨在捕捉我探索从 Foursquare 检索的数据的过程，以更好地了解在我的请求期间实际上是什么样的场地。在一个完美的世界中，每个条目都与音乐相关，并且位于纽约市，但是这需要被验证。以下问题说明了如何对初步场馆数据进行预处理，如本报告的数据预处理部分所示。

## **问题**:场馆位于哪些州？

**回答**:从 API 请求中提取的大多数条目都包含一个等于“纽约”或“NY”的“州”参数某些条目包含等于“CA”、“MA”和“NJ”的“state”参数，需要删除。

![](img/963d4eb984be0035ab10ac8bceddd008.png)

Showing the Venue State counts

## **问**:参赛作品属于哪个场馆类别？

**答**:这个数据集中有 149 个独特的场馆类别。有些类别与音乐无关，这是因为使用了 Foursquare 定义的更高级别的“场地类别”。与音乐无关的类别将被删除。

![](img/7775dd69dc0a4fe32ce5237b4fcb4a55.png)

Showing the Venue Category counts

## **问题**:有多少场馆没有填写“城市”参数？

**答**:下图显示有不少场馆没有填写“城市”参数。起初，我认为这不会是一个问题，因为我仍然有一个与每个条目相关联的纬度和经度。经过进一步分析，确定没有“城市”参数的条目不再是活跃的机构，因此将被删除。

![](img/8308af4069d7b184010c605baa137218.png)

Showing venues that do not have a Venue City parameter

## **问题**:数据集中有空值吗？

**回答**:没有。

![](img/b4e31fd5b945ffbf17cfee30adccd80b.png)

Checking for null values in the data

## **问题**:检索到多少个独特场馆？

**答**:初步数据集中，唯一场馆名称比条目总数少。这意味着存在与多个邻近区域相关联的场所，这是由于 API 请求中半径被设置为 1000 米而导致查询重叠的结果。这是可以接受的，因为场地在邻近区域质心的步行距离内，可以影响邻近区域的场景。

![](img/b1d46c6204e2b8ec957b4147a04a63d4.png)

Checking for duplicate Venue Names

# 数据预处理

## 数据清理

根据上面探索性数据分析部分中列出的答案清理初步数据集。首先，位于“纽约”或“NY”以外的州的场馆被移除。“地点州”等于“纽约”的条目被更改为“纽约”

![](img/9602e7587668571b02cc4bb0523b48cb.png)

Removing venues not in New York state

Foursquare 返回的没有“场馆城市”并给予“不适用”待遇的条目也被删除:

![](img/367b6fe56164d51baca25c328c26ecb4.png)

Removing venues with no Venue City parameter

基于包括在初步数据集中的独特场所类别，创建了音乐相关场所类别的列表。这个列表用于过滤掉混入我们请求中的与音乐无关的条目。

![](img/ff8d71c98f303d963cca0c43edbd6edf.png)

Removing venues that are not music-related

下图显示了 ny _ music _ venues 数据帧中的条目总数和唯一条目数。如前所述，一些场馆被分配给多个街区，因为场馆位于街区质心位置的 1000 米以内。

![](img/7df85dae3387405db15a9e2768384004.png)

Checking for duplicate venues

## 一键编码场馆类别

为了使用 Foursquare 的类别值来找到基于音乐场所的相似邻居，使用 Pandas 的“get_dummies”功能创建了每个条目的一个热编码表示。结果是纽约市音乐相关场所的数据框架，其中条目场所类别由匹配场所类别列中的值 1 表示，如下所示:

![](img/8c8c587da02aa7661e3a680191958365.png)

One-Hot-Encoding categorical variables

## 数据可视化

使用 one hot encoded DataFrame 确定了纽约市每个场馆类别和街区的场馆数量:

![](img/90d4b2d3eb9e7a803e33a37ed0b4565a.png)

Total amount of venues of each category in each neighborhood

使用上面显示的场馆数量的数据框架，为选定的场馆类别创建了水平条形图，以帮助可视化每个特定场馆最多的前 25 个社区。使用以下循环和 matplotlib:

![](img/e64e137d1f5cd8085b8b031ddd3d4ea6.png)

Code for recursively plotting top neighborhoods with venues of particular category

拥有最多“音乐厅”的社区

![](img/ff4204e7586bfc75fd96cb589ca9194c.png)

拥有最多“音乐场所”的社区

![](img/4c2d5ed62034b178232f9081264f213e.png)

拥有最多“夜总会”的社区

![](img/f461d7f3be9f1a1248244a0b148d0371.png)

“爵士俱乐部”最多的街区

![](img/5dfe10466b67add621427cd0a2da7b95.png)

“钢琴酒吧”最多的街区

![](img/cb760b08f34eabefb6aee15e99912a9c.png)

## 特征生成

纽约市音乐相关场所的编码数据集随后被用于量化每个街区的音乐概况。对于每个场馆类别，计算了场馆在每个社区的分布百分比。然后，该信息将用于使 K-Means 聚类算法适合数据，以努力确定相似音乐场所简档的邻域。

首先，确定每个类别的场馆总数:

![](img/a716730204a19f21666e4a9a849ef2df.png)

Creating a dictionary of venue category and total count

最后，根据场馆类别，计算每个街区的场馆相对于数据集中场馆总数的百分比。很明显，Astoria 的“Lounge”列中显示的值表示位于 Astoria 的休息室在数据集中所占的百分比。

![](img/799f28950ccae602ee76040de98ec675.png)

Percentage of entities of particular venue in a particular neighborhood

根据上述内容，创建了一个数据框架，显示每个街区的前五大音乐场所类别:

![](img/29acd937b0288a9585e25359cd3b77ef.png)![](img/7694606b1489aa020f0a1b6597366779.png)

Showing the top five venue categories per neighborhood

# 结果

## 集群建模

Scikit-learn 的 K-Means 聚类用于根据音乐场所百分比确定相似的邻域。下图显示了正在缩放的数据和正在创建的 K 均值模型:

![](img/92fa5be5c2aaa6a4045531ec750dc2a6.png)

Clustering neighborhood venue data

一个新的数据框架是通过合并邻居位置数据和聚类标签和顶级场馆类别创建的。

![](img/48c57c83b8eef8d8b792b78a4d9d26a5.png)

Merging neighborhood location and cluster data

## 集群可视化

以下代码通过基于分类标签对每个邻域点进行着色，使用 follow 来可视化相似音乐配置文件的邻域:

![](img/33c8fe329b1b4c6c8f4bf5fc101beb42.png)

Code to generate a folium leaflet map with neighborhoods colored by cluster

![](img/88b1b8b83b89832bd67883d5f7cf7aef.png)

Map of New York City showing clusters

## 聚类评估

下面的代码遍历并打印每个分类的结果:

![](img/b02ec1b8c245742dce1b990c428decb4.png)

Code to iterate through and print each cluster

由此产生的集群可以在本文档附录的集群部分看到。每个聚类显示一个邻域列表及其各自的顶级场馆类别。我们可以将得到的聚类与数据可视化部分中的条形图进行比较，并根据与音乐相关的场地计数来判断聚类是否正确地对邻近区域进行了分组。

有趣的是，一些集群非常小，有时只包含一个邻居，并且似乎已经确定了一个利基音乐简档。这方面的例子有:

*集群 1 —康尼岛—音乐节(康尼岛音乐节)*

*第二组团——林肯广场——歌剧院(大都会歌剧院)*

诸如集群 4 和集群 7 的其他集群非常大，并且看起来是具有诸如音乐场所、夜总会、摇滚俱乐部、休息室等各种现场音乐类型场所的分组邻居。

集群 9、11、12、13 和 14 的集群间第一大场馆类别都是相同的；爵士俱乐部、录音室、摇滚俱乐部、爵士俱乐部和钢琴俱乐部。有趣的是，集群 9 和集群 12 都主要对爵士乐感兴趣，但由于它们的其他顶级场所类别不同，这意味着不同的音乐概况，所以它们被不同地分组。

# 结论

机器学习和聚类算法可以应用于多维数据集，以发现数据中的相似性和模式。使用*高质量*场地位置数据，可以生成具有相似音乐简档或任何简档的邻居的聚类。关于*高质量*有一个前言，因为分析模型的好坏取决于对它们的输入(垃圾输入，垃圾输出)。幸运的是，Foursquare 提供了一个强大的“位置 API”服务，虽然(正如我们所看到的)并不完美(没有什么是完美的)，但可以在类似的研究和模型制作中加以利用。

这个项目并没有完成，可以用不同的方式进行扩展。Foursquare 的 API 可以被进一步询问，以检索和考虑纽约市更多与音乐相关的场所。可以获得音乐相关场所的新数据集，并可能与从 Foursquare 检索到的数据集合并。DBSCAN 聚类算法在保持密集聚类和忽略异常值方面更好，可以实现该算法并与 KMeans 进行比较。该聚类模型可以成为推荐系统的基础，该推荐系统旨在向用户提供相似音乐档案的邻居。

我期待着在未来继续探索和利用音乐相关的数据集。

GitHub 项目:[https://github.com/cascio/IBM_Data_Science_Capstone](https://github.com/cascio/IBM_Data_Science_Capstone)

个人领英:[https://linkedIN.com/in/mscascio](https://linkedIN.com/in/mscascio)

个人网站:[https://cascio.co](https://cascio.co)

推特:【https://twitter.com/MCasiyo 

# 参考

[0] — [2014 纽约市街区名称— NYU 空间数据仓库](https://geo.nyu.edu/catalog/nyu_2451_34572)

[1]—[‘Places API’文档— Foursquare](https://developer.foursquare.com/docs/api/endpoints)

# 附录

## 簇

> 群集 0:

![](img/aeef925fe8cb66fdbb1c37059d568e25.png)

> 群组 1:

![](img/18bbfe5ac23bcad33735254de6a24b69.png)

> 第二组:

![](img/5c707c1548b75ab25ac04d18c96b25fb.png)

> 第三组:

![](img/4a77c0bb54b5dbfe3fd35b1dc2217fcf.png)

> 第 4 组:

![](img/d0731c21d4cdee00113cad987cabdaae.png)![](img/740a9524046c46622a3d075579ce5024.png)![](img/cd7093532c47f21d025f12068764406d.png)![](img/da2aa1bbe780668c918e89bde53fb552.png)

> 第 5 组:

![](img/dab954e16c52abcebbe025e821e31e73.png)

> 第 6 组:

![](img/38d87502726c86331d561f4ad6c650a2.png)

> 第 7 组:

![](img/c9f51b45bd3e50737b630cfaa60d6d0c.png)![](img/fb17aa67e7beb0c3d55065823651af4c.png)![](img/c9a9893d5183a66d0a8a041a2035c1ea.png)![](img/37250e55fbd3fbd07a9ad2fccb0b6203.png)![](img/bd1b37335c4f51424fa05f9a8583048d.png)![](img/77b0c55213f43ec875c6c7b5798da4c9.png)![](img/a32b44f09fe73feb0acc5b4675eea0fe.png)![](img/73e1e353d9263a2e2b648916fde01cf3.png)![](img/f2517de742163444212d239b5e9eaa36.png)

> 第 8 组:

![](img/f3b32092327a4f3582d5fa9fe3f82ee6.png)

> 第 9 组:

![](img/673e11cf7ff9b8a66ca629f8b3fafcaa.png)

> 第 10 组:

![](img/feefeb14ce2c5a19085fecf018e56efe.png)

> 第 11 组:

![](img/7d08c1072ad632dc6e61899ff52d9f3b.png)

> 第 12 组:

![](img/e8bd412fd34443396e8aae385263b2e3.png)

> 第 13 组:

![](img/ee4ef1ab535f6baed8c77775afab5022.png)

> 第 14 组:

![](img/1a0eb7cfdcb1441c304caab9fc66272f.png)