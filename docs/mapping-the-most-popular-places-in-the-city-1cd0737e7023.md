# 绘制城市中最受欢迎的地方

> 原文：<https://towardsdatascience.com/mapping-the-most-popular-places-in-the-city-1cd0737e7023?source=collection_archive---------19----------------------->

## 使用蟒蛇，熊猫和树叶

![](img/cadcc6ac06628b20a06ddc6b6b81d7ee.png)

“Mass” by Gaetano Virgallito. Licensed under CC BY-ND 2.0

当“数据科学”一词出现时，一些与分析人类行为、大城市、服务消费等相关的研究引起了我的兴趣。上次工作时，我为巴西纳塔尔的优步服务制作了一张[地图，对如何更好地服务中心街区很感兴趣。这一次，人们的兴趣来自于谷歌的一个地点搜索功能:“大众时报”。](https://github.com/mrmorais/uber-map-natal)

![](img/3e1ec52c1573cbbb7c4519dd9e006e8e.png)

谷歌虽然聪明，但它可以向我们展示(有时甚至是“直播”)你搜索的地方有多忙。所以…

如果我根据热门时间比较几个地方呢？并发现特定类型的地方比其他地方更受欢迎？使用这种数据可以提出许多问题。

那么，为什么不从纳塔尔市中心(我居住的城市)以 15 英里为半径获取不同类型的地方，然后获取每个地方的流行时间数据，并用这些数据绘制一个交互式地图呢？这就是我们在这里做的。

我会尽量解释，用足够的细节不要让这篇文章讨厌，我是怎么做的，以及这个实验如何能被复制。这里用到的技术有 Python，pandas，叶子库。如果你想了解更多，代码可以在 [GitHub](https://github.com/mrmorais/behavior-map-natal) 上找到，比如[interactives Colab notebooks](https://colab.research.google.com/drive/1TW67ytLbvg6WOGxVy8rPXpfNsie420pg)你可以上传自己的数据库并生成地图。

## 谷歌，谷歌…

在绘制地图之前，我们必须提取我们需要的数据。这将需要两种类型的数据:纳塔尔的一组地点和这些地点中的每一个地点，关于流行时刻的数据。

我们将使用 [Google Places API](https://developers.google.com/places/web-service/intro) 来获取位置。这个 API 提供了几种服务，对于所有这些服务，您都需要一个[访问令牌](https://developers.google.com/places/web-service/get-api-key)来消费数据。

我使用了“[附近搜索](https://developers.google.com/places/web-service/search#PlaceSearchRequests)”来寻找纳塔尔市中心附近半径 15 英里的地方。脚本通过类型(健身房、超市、银行……)搜索这些地方——我为每种类型定义了我想要的地方的数量；这样我会得到最相关的结果。然后，该脚本保存。将数据集放置为`places.csv`的 csv 文件。

现在我们有地方了。通过我设置的过滤器和界限，我们在纳塔尔有 800 个地方。下一步是捕捉所有这些流行的时间。在 Places API 文档中查找该主题，您会注意到没有提到任何“流行时间”(至少直到今天，2019 年 5 月。)但是用谷歌快速搜索，我找到了让我们获得这项服务的[回购](https://github.com/m-wrzr/populartimes)。回购协议中说，谷歌确实允许这种查询，但不是免费的，这个 API 调用被 SKU 称为“查找当前位置”。你可以用每月分配的预算给这个 API 打 5000 个电话。

使用该 API，`get_places_popular_moments.py`读取 places 数据集并在其上包含热门时刻，生成新的`places_with_moments.csv`数据集。现在，我们应该有 800 个地方，但我们只有 300 个，因为没有每个地方的“流行时间”。

到目前为止，我们的数据集看起来像这样:

![](img/7bf46684d458503925d29592405867d5.png)

对于每个工作日列，有 24 个位置，具有一天中每个小时的力矩值。

## 用 follow 和 Kepler 生成地图

现在，我们希望将这些数据可视化在一个漂亮的交互式地图中。我们将使用基于 fleet . js 的 Python 库——来可视化地图。

这是纳塔尔周一上午 10 点的热门地点地图。圆圈的颜色表示地点的类型和大小，以及时刻值。

![](img/e3c470008184010b358312dc0ac2e443.png)

生成这个地图代码是这样的:

```
def generate_map(weekday, hour, types=places_types):
  natal = folium.Map(location=[-5.831308, -35.20470], zoom_start=13)
  ptypes_groups = {} for ptype in types:
    ptypes_groups[ptype] = FeatureGroup(name=ptype) for index, place in natal_places.iterrows():
    moments = json.loads(place[weekday])
    if (place.type in types):
      folium.Circle(location=[place.lat, place.lng],
                radius=int(moments[hour])*3,
                fill_color=colors[places_types.index(place.type)],
                fill_opacity=0.6).add_to(ptypes_groups[place.type])
  for ptype, group in ptypes_groups.items():
    group.add_to(natal) LayerControl().add_to(natal)
  return natal
```

它将星期几、小时和可选的地点类型子集作为参数。它做的第一件事是创建一个新的地图`natal`，并为每个想要的类型创建一个`FeatureGroup`。这个资源允许我们在地图上创建一个带有切换类型选项的菜单。

遍历数据集上的所有项目，我们将每个项目添加到相应的特征组。

深入到 colab 笔记本中，您会看到对数据集进行了修改，使其适合与 [Kepler.gl](https://kepler.gl/) 一起使用，这是一个强大的地理空间分析 web 工具。开普勒给了我们比叶更多的资源，而且性能非常好。所以我们可以这样做:

![](img/91b1f4e79c3885316f059db33d0e631e.png)

Anitation made with Kepler

上面的动画是使用开普勒生成的，让我们可以看到纳塔尔在这一周的移动情况。开普勒还允许我们导出环境，使其可以在线访问，在[这个地址](https://mrmorais.github.io/others/behavior-map-natal/keplergl.html)你可以访问上面的地图。

## 我们还能做什么？

乍一看，关于流行时间的信息没什么意义。谷歌说的“地方 100 拥挤”是什么意思？这是这个地方的人口密度等级？这些流行的时代数据到底意味着什么？

通过更好地分析一些值，我意识到“momet”的范围是从 0 到 100，这导致了诸如“通常不太忙”或“通常有点忙”之类的分类。但我真正的问题是“这些信息有多大意义？”

为了更好地理解它，我采用了另一种方法，对在数据集中找到一些有用的信息感兴趣。这是我的发现:

![](img/bb20cbd54029aa8e786d6b8df220b44d.png)

这个不错。图表中显示的健身房流量证实了一个常识:健身房周一更忙，周五没那么忙。高峰时间大约是晚上 20 点

![](img/24c6f3a7a9734356f7d91627c13bf42f.png)

大多数图表在午餐时间没有高峰时间，这与机场不同，机场在下午 12 点更忙

![](img/1e0a3dde997b3b3edc1b1cc2bc74328e.png)

星期五下午，医疗诊所有所减少。

![](img/fce6acb5ee120d7f2e0632c10974a64d.png)

纳塔尔的人们喜欢在星期五下午去理发店。

这些是从我们获得的数据中可以推断出的一些有趣的信息。总的来说，我很高兴这些数据与现实相符，这让我相信这确实是“流行时代”数据中的一个含义。这些结果可能会引导我们进行有趣的分析，这对理解纳塔尔，当然还有其他城市的行为是有意义的。