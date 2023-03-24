# 如何在 Airbnb 上实现利润最大化？基于数据的主机方法。

> 原文：<https://towardsdatascience.com/how-to-maximize-profits-on-airbnb-data-based-approach-for-hosts-beaf08f26941?source=collection_archive---------24----------------------->

![](img/cafac90beced1de4a681634205c409bb.png)

Foto credit unsplash.com/[@filios_sazeides](http://twitter.com/filios_sazeides)

可能每个人都听说过或使用过 Airbnb——在世界各地安排寄宿家庭的在线社区市场。Airbnb 将主人与游客联系起来，并负责包括支付在内的整个交易过程。也许你想在度假时在 Airbnb 上列出你的公寓，以赚取额外的钱。也许你已经在 Airbnb 上列出了你的房产，但你的利润有限，没有人联系你。你的价格太高了吗？Airbnb 不会告诉你这个。如何用你的财产赚更多的钱？如何确定你的 Airbnb 列表的最佳价格？数据可以帮助你深入了解 Airbnb 市场，以及在为你的 Airbnb 房产制定价格策略之前应该考虑什么。

我们使用了西雅图 Airbnb 的[公开数据](https://www.kaggle.com/airbnb/seattle/data)，这给我们的分析带来了限制。这个内幕可能只在西雅图或美国西海岸类似人口的任何城市有效(西雅图市 70 万；西雅图市区 3 . 000 . 000；).我们的方法通过每人每晚的价格来比较不同的 Airbnb 属性。

# 哪些属性增加了我的 Airbnb 挂牌价格？

像酒店行业一样，五星级设施的成本更高，但也提供更高的舒适度、质量和服务，Airbnb 上的价格也存在类似的现象。大楼里的电梯(20%)、前门的门卫(12%)、健身房(12%)或游泳池(8%)会对价格产生积极影响。不足为奇的是，在市中心(16%)、安妮女王(9%)或国会山(8%)等热门地区，价格更高。此外，客人对酒店位置的良好评价(12%)会影响价格，甚至比良好的清洁度评分(4%)更重要。

![](img/9d5e0744442817c8d897ba7ad0b76f69.png)

Price per person along Seattle districts.

像船一样过夜的不寻常的航空旅馆对价格有正面影响(8 %)。这可能是对主人的一个建议，让他们的位置看起来特别或独特。

但是，如果主人不能搬到游客最喜欢的地方，也不能改变她居住的建筑的属性，那该怎么办呢？数据还显示，安装空调(8%)、真正的床(6%)、有线电视(11%)或普通电视(7%)和热管(6%)可能会有所回报。这些小小的改进产生了积极的影响，提高了标准和价格。

每人的整体价格还取决于清洁费(35 %)和保证金(20%)。向客人收取额外的清洁和保安费用可能是增加利润的好主意，但可能会阻止客人预订。我们的数据方法没有回答这个问题。

拥有一个以上列表(8 %)的主机似乎要价更高，这导致了一个假设，即更有经验的主机更了解市场。

严格的取消政策(13%)，要求客人的电话验证(8%)或客人的个人资料图片(7 %)对价格有正面影响，可以从两个方面解读。首先，受欢迎的主人可能会收取更高的价格，但也希望确保他们的租金收入，而不会让善变的客人在最后一刻取消订单。更有可能的是，受欢迎的主持人收紧了他们的预订政策。第二，拥有更昂贵、更豪华、标准更高的房子的人，可能会更愿意让经过核实、值得信任的陌生人进入他们的房子。

```
cleaning_fee                                        0.350421
security_deposit                                    0.203517
Elevator in Building                                0.198643
neighbourhood_group_cleansed_Downtown               0.162595
cancellation_policy_strict                          0.130720
Doorman                                             0.121563
review_scores_location                              0.121196
Gym                                                 0.120789
Cable TV                                            0.113371
neighbourhood_group_cleansed_Queen Anne             0.090053
calculated_host_listings_count                      0.083392
Pool                                                0.083153
property_type_Boat                                  0.079727
neighbourhood_group_cleansed_Capitol Hill           0.078682
require_guest_profile_picture                       0.077501
Air Conditioning                                    0.076503
TV                                                  0.075863
Kitchen                                             0.071869
require_guest_phone_verification                    0.069068
bathrooms                                           0.067893
bed_type_Real Bed                                   0.063524
Hot Tub                                             0.060124
review_scores_rating                                0.057456
property_type_Condominium                           0.051757
review_scores_cleanliness                           0.047834
extra_people                                        0.043730
neighbourhood_group_cleansed_Cascade                0.042380
property_type_Bed & Breakfast                       0.039962
```

# 哪些属性会降低我的 Airbnb 标价？

西雅图的一些地区不是很受欢迎:雷尼尔谷(-9%)、德尔里奇(-8%)、北门(-8%)、比肯山(-6%)、苏厄德公园(-6%)或其他一些不太出名的街区(-7 %)。

酒店类型私人房间(-17 %)或房子(-13 %)允许的客人数量少(-16 %)，床位数量少(-9%)似乎比宽敞的酒店更便宜。

住在酒店的宠物(-8%)也可能会吓退一些客人。有些客人可能对狗或猫过敏，遗憾的是，并不是每个人都喜欢动物。

令人惊讶的是，价格越高，主机收到的评论越少(-12 %)和每月评论越少(-22 %)。原因可能是越贵的房产每月的预订率越低。一个主持人的预约越少，她收到的评论就越少。由于该分析不包括时间因素，我们必须进一步调查，哪种情况更有希望:一个月内以更低的价格接待更多的客人，还是以更高的价格接待更少的客人。

![](img/c7ebbc6913aef02356f2eebac8af7ed2.png)

Distribution for number of reviews, price per person and reviews per month.

```
neighbourhood_group_cleansed_Seward Park           -0.057727
Shampoo                                            -0.058516
neighbourhood_group_cleansed_Beacon Hill           -0.059966
instant_bookable                                   -0.063909
neighbourhood_group_cleansed_Other neighborhoods   -0.070834
neighbourhood_group_cleansed_Northgate             -0.075342
Pets live on this property                         -0.078505
neighbourhood_group_cleansed_Delridge              -0.083484
beds                                               -0.087425
neighbourhood_group_cleansed_Rainier Valley        -0.088460
Free Parking on Premises                           -0.098190
number_of_reviews                                  -0.121019
property_type_House                                -0.126896
accommodates                                       -0.159946
room_type_Private room                             -0.168101
reviews_per_month                                  -0.219961
```

# 如何让我的 Airbnb 利润最大化？

在调查了数据之间的相关性之后，我们创建了一个模型，该模型生成了对人均价格最有影响的属性。以下是前 10 名。

1.  **点评评分“位置”**:楼盘广告语*“位置，位置，位置！”*在 Airbnb 环境中似乎也是如此。一个位置好的 Airbnb 为主机获得更多利润。
2.  **查看评分“评级”:**毫不奇怪，良好的总体评级会产生新的预订并使利润最大化。
3.  **大楼内的电梯:**客人不喜欢楼梯，似乎会为使用电梯支付更多费用。
4.  **主持人是超级主持人:**为了利益最大化，成为超级主持人是好的。如果主人满足四个标准，Airbnb 会自动分配 superhost 徽章:A)主人一年至少入住 10 次，B)快速回复客人，C)至少有 80%的五星评论，D)主人应该很少取消预订。以超主机奖为目标是值得的。
5.  **位置“市中心”:**为了利润最大化，最好将位于市中心的物业上市。
6.  **计算主机挂牌数:**越有经验的主机越了解市场。偶尔在 Airbnb 上列出位置可能不是实现利润最大化的好主意。
7.  **点评评分“清洁度”**:客人期望有一定程度的清洁度，并愿意为高标准的完美入住支付更多费用。
8.  **点评评分“准确性”:**对客人坦诚是有回报的。如果住宿无法描述，图片或问题没有反映在列表中，客人不会喜欢。
9.  浴室:毫不奇怪，更高的标准，比如更多的浴室，会给主人带来更多的利润。
10.  取消政策严格:更有可能的是，受欢迎的主持人收紧了他们的预订政策。有了稳定的预订，主人就可以选择希望谁住在她的房子里。

我们的方法仅限于时间部分。我们只分析了特定时间点的数据，而不是长期数据。

这篇文章是作为完成 Udacity 数据科学纳米学位的一部分提交的。请在以下链接中找到 Github 库:[https://github.com/astraetluna/airbnb-blogpost](https://github.com/astraetluna/airbnb-blogpost)