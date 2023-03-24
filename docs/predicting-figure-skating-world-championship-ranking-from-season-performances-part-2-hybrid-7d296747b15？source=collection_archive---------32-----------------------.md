# 从赛季表现预测花样滑冰世锦赛排名

> 原文：<https://towardsdatascience.com/predicting-figure-skating-world-championship-ranking-from-season-performances-part-2-hybrid-7d296747b15?source=collection_archive---------32----------------------->

## 体育分析

## 第 2 部分:混合模型

![](img/12c46e807b50a7b435f918bc59e8f61e.png)

*   *要看我为这个项目写的代码，可以看看它的 Github* [*回购*](https://github.com/dknguyengit/skate_predict)
*   *对于项目的其他部分:* [*第一部分*](/predicting-figure-skating-championship-ranking-from-season-performances-fc704fa7971a?source=friends_link&sk=7e6b2992c6dd5e6e7e1803c574b4236d) *、第二部分* [*第三部分*](https://medium.com/@seismatica/predict-figure-skating-world-championship-ranking-from-season-performances-8af099351e9c?source=friends_link&sk=48c2971de1a7aa77352eb96eec77f249) *、* [*第四部分*](https://medium.com/@seismatica/predict-figure-skating-world-championship-ranking-from-season-performances-a4771f2460d2?source=friends_link&sk=61ecc86c4340e2e3095720cae80c0e70) *、* [*第五部分*](https://medium.com/@seismatica/predict-figure-skating-world-championship-ranking-from-season-performances-7461dc5c0722?source=friends_link&sk=fcf7e410d33925363d0bbbcf59130ade) *、* [*第六部分*](https://medium.com/@seismatica/predict-figure-skating-world-championship-ranking-from-season-performances-d97bfbd37807)

# 背景

在项目的第一部分，我试图根据运动员在该赛季以前的比赛项目中获得的分数来预测年度花样滑冰世界锦标赛中的**排名**。主要策略是将**滑手效应**(每个滑手的内在能力)与**事件效应**(一个事件对滑手表现的影响)分开，以便建立更准确的排名。

针对这个排名问题提出了两个模型，其中一对运动员的赛季得分可以近似为:

*   一个**基线得分**，它在所有赛季得分中保持不变
*   潜在的**事件分数**，该分数在所有参加该事件的选手中是不变的
*   一个潜在的**滑冰者分数**，该分数在该滑冰者参加的所有比赛中都是不变的

这些潜在得分可以相加在一起(**加法模型**)，或者相乘(**乘法模型**)以接近赛季得分:

![](img/d3c339cf2cf82fb39ef4f2e97b1941bf.png)

在任一情况下，通过重写上述公式，可以容易地找到潜在分数，使得它们成为简单的线性模型，其回归系数是潜在分数本身。因此，这些分数可以简单地通过求解线性回归的[正规方程](https://en.wikipedia.org/wiki/Linear_least_squares#Derivation_of_the_normal_equations)来得到。从那里，每个模型的预测排名只是潜在滑手分数的排名，从最高到最低。

# 问题

将这两个模型预测的排名与根据赛季平均水平简单排名的基线模型进行比较。用于评估这些模型的度量是相对于实际世界锦标赛排名的[肯德尔排名相关系数](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient)(也称为肯德尔的τ):如果模型的预测排名具有更接近 1 的肯德尔τ，则模型更好，这意味着预测排名与当年世界锦标赛中的实际排名更相关。

![](img/016d221263e2ccbf5560cc7b2a49d8aa.png)

Baseline: season average model. The 95%-confidence intervals are built using a t-distribution with 9 degrees of freedom (since n=10 years).

从以上关于这两个模型在训练集中选择的 10 年(共 14 年)的报告来看，这两个模型比赛季平均值的基线模型更好地逼近赛季得分，因为它们与基线模型之间的[均方根误差](https://en.wikipedia.org/wiki/Root-mean-square_deviation) (RMSE)差异的 95%置信区间小于零。

然而，在更重要的 Kendallτ度量中，遗憾的是，与基线模型相比，这些模型没有提供该度量的任何显著增加，因为对于男性和女性溜冰者，它们与基线之间的 Kendallτ差异的 95%置信区间包含零。

# 混合模型

鉴于这两个模型在排名预测方面表现平平，我提出了一个新模型，它是这两个模型的奇怪混合:

![](img/4a919348a3f9971c42fce8d38fbf9602.png)

*   给定事件中给定选手的分数
*   `θ_baseline`:基线得分(整个赛季得分不变)
*   `θ_event`:该项目潜在得分(运动员之间不变)
*   `θ_skater`:溜冰者潜在得分(跨事件恒定)

根据模型公式，给定溜冰者在赛季期间的给定事件中获得的分数可以近似为该溜冰者的潜在分数乘以该事件的潜在分数，类似于乘法模型。但是，该产品随后会添加到基线得分中，类似于加法模型。

由于有些年份加法模型比乘法模型具有更高的肯德尔τ，而有些年份则相反，因此混合模型可能提供比单独的任一模型更好的预测排名。你可以在这个项目的 Github [repo](https://github.com/dknguyengit/skate_predict/) 里找到这个 [Jupyter 笔记本](https://github.com/dknguyengit/skate_predict/blob/master/analysis_part2.ipynb)里关于混动车型的分析。

## 寻找潜在分数

由于混合模型中加法和乘法的奇怪组合，我们既不能将其重新表述为线性模型，也不能取公式的对数将其转换为线性模型(类似于我们对乘法模型所做的)。

尽管如此，寻找该模型的最佳参数的策略仍然与前两个模型相同:最小化赛季中所有事件-滑冰运动员对的预测分数和真实分数之间的平方差之和:

![](img/7d76919af79d80449f98350d5386e7d2.png)

*   `J`:模型的目标函数；继续以 2017 赛季的男选手为例，这将是 1 个基线分数、9 个潜在事件分数(以蓝色突出显示)和 62 个潜在选手分数(以黄色突出显示)的函数。
*   `ŷ_e,s`:来自混合模型的项目-滑冰运动员对的预测得分
*   `y_e,s`:在赛季中记录事件-溜冰者对的真实分数
*   `θ_baseline`:基线得分
*   `θ_e`:事件潜在得分`e`
*   `θ_s`:滑手潜在得分`s`

## 梯度下降算法

有了上面的目标函数，我们可以使用 good ol '[gradient descent](https://en.wikipedia.org/wiki/Gradient_descent)来找到模型的最佳基线、事件和选手分数。让我们看看每个参数的梯度是什么样的，首先是基准分数**(`θ_baseline`):**

**![](img/86a9a08b490f9a64c660bc075e12d118.png)**

**根据上面的公式，目标函数相对于基线得分的梯度简单地是赛季中所有得分的真实得分和预测得分之间的差的总和(从这里开始将被称为残差)。**

**因此，给定基线、项目和运动员分数的一些当前值，我们总是可以计算基线分数的梯度，并根据该梯度更新基线分数(因为我们希望最小化目标函数):**

**![](img/38917f5258d020ca66b070b1043c8691.png)**

**与梯度相乘的额外项α被称为[学习速率](https://en.wikipedia.org/wiki/Learning_rate)，并控制梯度下降算法运行的速度。低α可能会使算法收敛太慢，而高α可能会使更新太大，参数不断从一个极端摆动到另一个极端，这在我这个项目中发生过几次。**

**给定事件的**潜在得分可以类似地更新，以俄罗斯的事件(潜在得分`θ_RU`)为例:****

**![](img/2a5b8f8266e1c79d21f74729bd2c0ee0.png)**

**当将目标函数与该事件的潜在得分进行微分时，所有不涉及该事件的平方差都将消失，这就只剩下涉及来自该事件的参加滑手的项，包括哈维尔、费尔南德斯(`θ_FERNANDEZ`)和亚历山大、马约罗夫(`θ_MAJOROV`)。使用链式法则，我们可以看到，梯度只不过是参与该事件的每个溜冰者的残差乘以该溜冰者各自的潜在得分，然后将这些乘积加在一起。**

**使用这个梯度，这个俄罗斯事件的潜在得分可以使用梯度下降来更新:**

**![](img/a77a041a107e80b288df5fe316dba175.png)**

**最后，对于溜冰者的**潜在得分，例如 MURA 的高仁(`θ_MURA`)，目标函数相对于该潜在得分的梯度得分为:****

**![](img/6151bc474237a6d1a10a6e1d12e05b8c.png)**

**2017 赛季，他参加了 2 场比赛，一场在加拿大，一场在法国，分别有潜在分数`θ_CA`和`θ_FR`。因此，梯度只不过是每个事件的残差乘以该事件的潜在得分，然后将这些乘积相加。因此，可以使用梯度下降来更新该溜冰者的潜在得分:**

**![](img/1332aa170531a44f4a11e55374b472a1.png)**

**简而言之，寻找基线、项目和运动员分数的梯度下降算法可以概括为以下步骤:**

**![](img/97afee4946aec0f9e7b309bf9f07cef8.png)**

**在第二步中，可以通过检查模型的 RMSE(即平均残差平方的平方根)是否已经稳定到最小值来监控收敛。梯度下降收敛后得到的运动员分数将被用于世界锦标赛的运动员排名。**

# **混合模型编码**

**给定上面的梯度下降算法，让我们看一个如何用 Python 编码的小例子。下面，我们有一个数据框(`season_scores`)，包含 4 名滑冰运动员(马约罗夫、费尔南德斯、葛、MURA)和 3 项赛事(加拿大、法国、俄罗斯)的 7 个赛季成绩:**

**![](img/f2dba115326a3814b0a298e5bef77a63.png)**

**首先，我们使用 pandas 的`pivot_table`函数将这个长表格式转换成数据透视表格式，将运动员姓名作为行索引，将事件名称作为列。将赛季得分转换成这种数据透视表格式的原因很快就会明了。**

```
season_pivot = pd.pivot_table(season_scores[['name', 'event', 'score']], values='score', index='name', columns='event')
```

**下面是我们最终将得到的 4×3 的赛季得分数据透视表。注意不是所有的滑手都参加所有的项目，所以数据透视表中有缺失值，用`NaN`(不是数字)表示。**

**![](img/3fd07382b8f4f9e152eee19212bb8659.png)**

**最后，在我们开始梯度下降之前，让我们把熊猫数据帧`season_pivot`转换成一个 4×3 的 numpy 矩阵`true_scores`，这样我们可以更容易地操作它。这实际上将删除所有的行和列名称，所以让我们存储运动员名称(行)和事件名称(列)，这样我们在运行梯度下降后仍然可以识别它们。**

```
skater_names = list(season_pivot.index) 
# ['Alexander, MAJOROV', 'Javier, FERNANDEZ', 'Misha, GE', 'Takahito, MURA']event_names = list(season_pivot.columns) 
# ['CA', 'FR', 'RU']true_scores = season_pivot.values
# array([[   nan,    nan, 192.14],
#        [   nan, 285.38, 292.98],
#        [226.07, 229.06,    nan],
#        [222.13, 248.42,    nan]])
```

## **第一步:初始化基线、潜在项目和潜在选手得分**

**首先，让我们使用 numpy 的带有特定种子的`[RandomState](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.RandomState.html)`对象初始化梯度下降的基线分数(这样每次运行代码时结果都是一致的)，并调用该对象的`[random_sample](https://docs.scipy.org/doc/numpy-1.15.0/reference/generated/numpy.random.RandomState.random_sample.html#numpy.random.RandomState.random_sample)`方法返回一个介于 0 和 1 之间的随机数。初始值没有理由在这个范围内。我只是觉得这样会更好看。**

```
random_state = np.random.RandomState(seed=42)
baseline = random_state.random_sample() # 0.866
```

**现在，让我们初始化潜在事件和选手分数。我不再逐一初始化它们，而是将所有选手的分数初始化为大小为 4×1 ( `skater_scores`)的列向量，将所有事件的分数初始化为大小为 1×3 ( `event_scores`)的行向量，其条目也是 0 到 1 之间的随机数，再次使用`random_sample`方法。**

```
skater_scores = random_state.random_sample(4, 1))
# array([[0.37454012],
#        [0.95071431],
#        [0.73199394],
#        [0.59865848]])event_scores = random_state.random_sample((1, 3))
# array([[0.15601864, 0.15599452, 0.05808361]])
```

**以向量形式初始化这些潜在得分的原因是，通过将这些向量相乘(使用矩阵乘法运算符`@`)并在此基础上添加基线，可以一次性计算出每对运动员的预测得分:**

```
predicted_scores = skater_scores @ event_scores + baseline
# array([[0.52284634, 0.42976104, 1.19802617],
#        [0.48872716, 0.41705697, 1.00857581],
#        [0.46792756, 0.40931237, 0.89308382],
#        [0.39887817, 0.38360225, 0.50967974]])
```

**为了更清楚地说明这一点，下图显示了两个潜在向量(大小分别为 4×1 和 1×3)与添加在顶部的基线相乘，以形成每个项目-选手对的预测得分的 4×3 矩阵。用红色突出显示的是一对运动员(CA-MAJOROV ),这是通过乘以潜在向量中的相应条目计算出来的，添加的基线在顶部。**

**![](img/2bbeb007d3332bb079b8c0b26f681db6.png)**

****Dotted line:** implied indices of skaters and events, even though they don’t exist in the numpy array**

**接下来，由于已经计算了每对运动员的真实分数(`true_scores`)和预测分数(`predicted_scores`),我们可以很容易地计算出赛季中每个分数的残差。**

```
residuals = predicted_scores - true_scores
```

**回想一下，我们的`true_scores` numpy 矩阵包含赛季中不存在的赛事-运动员配对的`NaN`值。因此，当计算残差时，这些对的相应残差也是`NaN`。**

**![](img/e63326093052d8076a87e46d63307b53.png)**

## **步骤 2a:计算基线、项目和运动员分数的梯度**

**残差矩阵中`NaN`的存在使得计算梯度变得非常方便，因为我们不必跟踪哪个运动员在赛季中存在或不存在。例如，先前导出的基线梯度是该季节中所有现有事件-滑冰运动员对{e，s}的残差之和:**

**![](img/86a9a08b490f9a64c660bc075e12d118.png)**

**因此，**基线梯度**可以简单地实现为:**

```
baseline_gradient = np.nansum(residuals) # -1692
```

**`[np.nansum](https://docs.scipy.org/doc/numpy/reference/generated/numpy.nansum.html)`函数有效地跳过了不存在的`NaN`值(通过将它们视为零)，这符合我们的目的，即只对赛季中现有的事件-选手对的残差求和。**

**为了计算所有潜在事件分数的**梯度，`np.nansum`函数因其`axis`参数而再次大有帮助:****

```
event_gradients = np.nansum(residuals * skater_scores, axis=0, keepdims=True)
```

**让我们分解这行代码，看看它是如何工作的:**

*   **`residuals * skater_scores`:通过使用`*`操作，我们简单地将`residual`矩阵与步骤 1 中初始化的`skater_scores`逐元素相乘。**

**然而，由于两个数组的大小不同:矩阵的大小为 4×3 (4 个运动员乘 3 个事件)，而列向量的大小为 4×1(4 个运动员的潜在得分)，numpy 的[广播](https://docs.scipy.org/doc/numpy/user/basics.broadcasting.html)开始运行。实际上，numpy 将水平复制`skater_scores`向量 3 次，使其与`residuals`矩阵的维数相同(4×3)，如下图所示:**

**![](img/b51f7a873b4e7ce07031219e31e73132.png)**

****Highlighted in red:** relevant values for the Russian eve`nt (RU). **Dotted arrows:** numpy broadcasting of latent skater scores`**

**注意，在广播之后，两个矩阵(现在大小相似)之间的逐元素乘法将在与`residuals`矩阵中的值相同的位置留下`NaN`值。这允许`np.nansum`来计算相关的事件梯度，但这次该函数有两个新参数:**

*   **`axis=0`:这允许在`residuals * skater scores`矩阵的行(代表运动员)之间计算总和。实际上，这将对参加每个项目的所有选手的剩余分数和潜在分数的乘积求和，从而得出该项目潜在分数的梯度(见上图)**

**![](img/a619133327953073698edc6e24ce803f.png)**

****Highlighted in red:** relevant values for the Russian eve`nt (RU)`**

*   **`keepdims=True`:对 4×3 `residuals * skater scores`矩阵的行求和将产生一个 1×3 维的行向量，它正好包含 3 个事件的梯度。从技术上讲，这是 numpy 中 shape (1，3)的一个二维数组，然而默认情况下，`np.nansum`将结果折叠成 shape (3，)的一个一维数组。因此，设置`keepdims=True`将使事件梯度保持在 2-D 中，因为稍后将从相同形状(1，3)的`event_score`数组中减去该梯度。**

**计算所有潜在溜冰者分数的**梯度与计算事件梯度相同，除了我们将残差(`residuals`大小为 4×3)乘以事件分数(`event_scores`大小为 1×3，将垂直广播 4 次)。****

```
skater_gradients = np.nansum(residuals * event_scores, axis=1, keepdims=True)
```

**![](img/753011862824b41a9de47d5f4b703fb9.png)**

****Highlighted in red:** relevant values for the skater Takahito, MURA. `**Dotted arrows:** numpy broadcasting of latent skater scores`**

**乘法的结果将通过使用带有参数`axis=1`的`np.nansum`跨列(每个选手参加的项目)求和，以获得选手梯度:**

```
skater_gradients = np.nansum(residuals * event_scores, axis=1, keepdims=True)
```

**![](img/8fd2ff6eae1fd7b8acac182e2901e2fa.png)**

****Highlighted in red:** relevant values for the skater Takahito, MURA**

## **步骤 2b:使用梯度更新基线、项目和运动员分数**

**一旦计算出梯度，使用它们来更新基线、项目和选手的潜在得分只涉及简单的减法(使用学习率`alpha=0.0005`):**

```
alpha = 0.0005
baseline = baseline - alpha * baseline_gradient
event_scores = event_scores - alpha * event_gradients
skater_scores = skater_scores - alpha * skater_gradients
```

**![](img/eec8c489a934e202ff0415dacdb08325.png)**

**使用一个简单的`for`循环，我们可以多次重复步骤 2，在计算梯度和使用它们更新潜在得分之间交替进行:**

```
alpha = 0.0005for i in range(1000):
    # 2a. Calculate gradients
    predicted_scores = skater_scores @ event_scores + baseline
    residuals = predicted_scores - true_scores

    baseline_gradient = np.nansum(residuals)
    event_gradients = np.nansum(residuals * skater_scores, axis=0, keepdims=True)
    skater_gradients = np.nansum(residuals * event_scores, axis=1, keepdims=True)

    # 2b. Update latent scores using gradients
    baseline = baseline - alpha * baseline_gradient
    event_scores = event_scores - alpha * event_gradients
    skater_scores = skater_scores - alpha * skater_gradients
```

**梯度下降算法的收敛可以通过计算每个循环后的 RMSE`np.sqrt(np.nanmean(residuals**2))`来监控，并检查它与前一个循环的不同程度。经过 1000 次迭代后，模型的 RMSE 为 4.30，迭代与迭代之间的差异为 10e-9。这表明该算法已经很好地真正收敛。**

**最后，梯度下降收敛后的潜在选手分数被转换成熊猫系列，这允许早先存储的选手名字被添加回来。请注意，在将二维`skater_scores`数组(4，1)转换成序列之前，需要用`[ravel](https://docs.scipy.org/doc/numpy/reference/generated/numpy.ravel.html)`将它折叠成一维数组(4，)。此外，调用`sort_values(ascending=False)`从最高到最低对运动员的潜在得分进行排序，因为最初的目标是通过这些潜在得分对运动员进行排名。**

```
skater_scores = pd.Series(skater_scores.ravel(), index=skater_names).sort_values(ascending=False)
# Javier, FERNANDEZ     16.701678
# Takahito, MURA        14.101404
# Misha, GE             13.587320
# Alexander, MAJOROV    10.678634
```

# **结果**

**当在 2017 赛季期间将梯度下降算法应用于男性滑冰运动员时，我们可以通过每次迭代的残差和 RMSE 来监控算法的进展。此外，每次迭代中的潜在选手分数可用于建立临时排名，其与世界锦标赛排名的相关性由肯德尔的 tau 测量。在算法期间监控的所有指标(学习率α = 0.0005)显示在下面的动画仪表板中:**

**![](img/1264ea2e2861b39a58c33ace3c26eff2.png)**

****Left:** heat map of residuals, along with baseline, event, and skater scores at each iteration. Skaters are ordered based on their predicted ranks after 150 iterations. **Right (top to bottom):** RMSE, Kendall’s tau, and comparison between predicted ranking and world ranking at each iteration (colors are based on world ranking).**

**从上面的仪表板中，我们可以看到:**

*   **随着梯度下降算法的运行，残差(预测得分-真实得分)一般会减少，从它们的热图越来越亮就可以看出。我第一次看的时候，觉得很酷！**
*   **因此，RMSE 也会随着迭代次数的增加而降低(见右上图)。事实上，RMSE 甚至在 20 次迭代后就显著降低，之后开始稳定。**
*   **在预测排名方面，我们还注意到，随着梯度下降的进行，它开始越来越好地与世界冠军排名相关联(见右下图):连接两个排名的线开始变得越来越不纠结。**
*   **然而，可以看出，在算法开始附近有一个时刻(大约 25 次迭代左右),两个等级之间的相关性“看起来”最好，这也对应于在同一次迭代中肯德尔τ的峰(中右图)。这意味着，混合模型不需要尽可能好地逼近赛季得分，以便获得运动员的良好预测排名；否则，它可能会过度拟合赛季得分，从而损害预测排名的准确性。**
*   **这种过度拟合的潜在证据意味着我们应该惩罚混合模型(类似于我们在前一部分对线性模型所做的)。避免过度拟合的另一个策略是当预测等级的肯德尔τ是最高的，也就是驼峰所在的位置时，过早地停止梯度下降。在机器学习的说法中，这被称为[提前停止](https://en.wikipedia.org/wiki/Early_stopping)，这两种减少模型过度拟合的策略将在稍后探讨。**

**以下是梯度下降收敛后的潜在滑手分数；只有获得 2017 年世界锦标赛资格的 24 名选手将被显示，从最高到最低的选手分数排列如下:**

**![](img/0bdf323429738689bd479a104996595a.png)**

**Latent skater scores from hybrid model for male skaters in 2017 season (with associated rank)**

# **模型评估**

**在以学习速率`alpha=0.0005`和 10，000 次迭代对混合模型运行梯度下降后，模型的最终 RMSE 为 8.86，迭代与迭代之间的差异为 1.7e-8，对于收敛来说足够小。**

**将该 RMSE 与项目第一部分中的线性模型进行比较时——加法模型为 8.84，乘法模型为 8.88——我们可以看到，混合模型在逼近赛季得分的程度上正好位于之前的模型之间。这也优于季节平均值的基线模型，其 RMSE 为 10.3。**

**所有型号的每对运动员与真实分数的偏差如下所示。该图中所有 3 个模型之间预测得分的接近程度证实了它们之间相似的 RMSEs。**

**![](img/fc2de6103d3d15f8879220fd1f54383f.png)**

**Deviation from true score for each event-skater pair of the season for all models**

**就预测排名而言，混合排名再次与早期排名几乎相同(见下图)。事实上，相比于乘法排名，混合排名只是颠倒了只有一对滑手的顺序:`Kevin, REYNOLDS`和`Maxim, KOVTUN`，用红色突出显示。**

**![](img/4c643d4994eb37bd8772efb6839622c7.png)**

**不幸的是，这种逆转与最终的世界冠军结果不符:`Kevin, REYNOLDS`最终排名高于`Maxim, KOVTUN`。结果，混合模型在所有 276 个有序对中得到 238 对，比乘法模型少一对。这意味着混合模型的肯德尔τ值略低，为 0.724，而乘法模型和加法模型的肯德尔τ值均为 0.732。然而，这仍然略高于 2017 赛季男子滑冰运动员赛季平均基线模型的肯德尔τ值 0.695(或 276 对中的 234 对)。**

**在评估训练集中所有 10 年的混合模型之前，让我们首先看看我们如何能够像前面承诺的那样使它对赛季分数的过度拟合更少。**

# **典型惩罚**

## **理论**

**回想一下梯度下降试图最小化的混合模型的目标函数:**

**![](img/7d76919af79d80449f98350d5386e7d2.png)**

**为了惩罚混合模型，我们需要找到一种方法来人为地使潜在的事件和运动员分数变小(同时保持基线分数不变，以便赛季分数仍然是相当接近的)。这样做的一种方法是这样修改目标函数:**

**![](img/d22b0dfb25896443f157f4b2fc8c5d89.png)**

**The 1/2 factor for the squared terms will make later derivations easier**

**上述目标函数末端的附加平方项表明:除了最小化季节分数的残差平方，我们还希望最小化潜在事件和潜在选手分数的平方(分别为`θ_e`和`θ_s`)，以使这些潜在分数保持较小。潜在得分被惩罚的程度由惩罚参数λ控制:λ越高，潜在得分的平方越小，这些潜在得分就越小。**

**我们可以相对于这些潜在得分对修改的目标函数进行微分，以在梯度下降期间获得梯度公式(步骤 2a):**

**![](img/05a33a166cb273bacc5cbbd17f6929ec.png)**

**Notice that the gradient formula for baseline score stays the same**

**从上面的梯度公式可以看出，对原梯度下降公式唯一的修改是分别在项目和滑手梯度公式的末尾带有λ的项。**

## **编码**

**由于这些简单的修改，将这些术语合并到梯度下降的 Python 实现中变得非常容易。下面是惩罚模型的梯度下降步骤 2 的代码(使用惩罚参数`lambda_param=10`)，与原始模型的变化以粗体突出显示。请注意，对于我们的玩具示例，我们在收敛后获得的惩罚模型的潜在溜冰者分数小于原始模型的分数，这意味着惩罚确实有效。**

```
alpha = 0.0005
**lambda_param = 10**for i in range(1000):
    # 2a. Calculate gradients
    predicted_scores = skater_scores @ event_scores + baseline
    residuals = predicted_scores - true_scores

    baseline_gradient = np.nansum(residuals)
    event_gradients = np.nansum(residuals * skater_scores, axis=0, keepdims=True) **+ lambda_param * event_scores**
    skater_gradients = np.nansum(residuals * event_scores, axis=1, keepdims=True) **+ lambda_param * skater_scores**

    ### 2b. Update latent scores using gradients
    baseline = baseline - alpha * baseline_gradient
    event_scores = event_scores - alpha * event_gradients
    skater_scores = skater_scores - alpha * skater_gradientsskater_scores = pd.Series(skater_scores.ravel(), index=skater_names).sort_values(ascending=False)
# Javier, FERNANDEZ     16.108775
# Takahito, MURA        13.429787
# Misha, GE             12.893933
# Alexander, MAJOROV     9.784296
```

## **估价**

**下面是当惩罚参数λ增加(增加 10 倍)时，混合模型的 RMSE、肯德尔τ和预测排名如何变化的图表:**

**![](img/4781271c46cfdb0440c6ecb05e7b8323.png)**

**Learning rate α = 0.0005, Iterations = 10,000**

**从上图可以看出:**

*   **随着惩罚参数λ的增加，模型的 RMSE 如预期一样增加，特别是从λ=10 开始，RMSE 几乎是基线模型的两倍(RMSE=10.3)。**
*   **然而，当我们试图惩罚模型时，预期的肯德尔τ值的增加并没有实现。λ越高，模型的预测排名就变得越差，这可以从肯德尔的τ的减少中得到证明，并且随着λ的增加，预测排名与世界冠军排名相比变得越来越“凌乱”。**

**当通过绘制不同λ值的基线模型(季节平均值)上的 RMSE 和肯德尔τ的平均差异来检查这种惩罚对训练集中所有 10 年的影响时，出现了相同的画面:**

**![](img/02073ab6b43d953bf1333b872007d518.png)**

*   **如左图所示，当λ增加时，混合模型相对于基准模型的 RMSE 平均差异(10 年以上)通常会增加，尤其是在λ=10 之后。这意味着当惩罚越多，该模型越接近赛季得分。然而，这种行为是意料之中的，因为毕竟惩罚的主要目的是减少对赛季分数的过度拟合。**
*   **然而，一般来说，混合模型相对于基线模型的肯德尔τ的平均差异随着λ的增加而下降。换句话说，模型给出的预测排名越差，惩罚就越多。这令人失望，但并不令人惊讶，因为惩罚(通过最小化潜在得分的平方)在项目的第一部分对线性模型有同样的影响。**
*   **此外，与之前的模型类似，与基线模型相比，无惩罚混合模型(λ=0)给出了更差的平均排名预测。然而，这种差异微乎其微:平均而言，Kendall tau 仅下降 0.001，并且不具有统计学意义:当λ=0 时，与基线模型相比，Kendall tau 平均差异的 95%置信区间很容易包含零水平线。**

**有了这些黯淡的模型惩罚结果，让我们看看是否可以用另一种方式减少模型过度拟合，这有望为混合模型产生更好的预测排名。**

# **提前停止**

## **理论**

**如前所述，在这种情况下提前停止的想法非常简单:当预测排名的肯德尔τ值最高时，我们只是提前停止梯度下降。实际上，这将防止潜在分数变得太大，就好像我们在原始混合模型中让梯度下降一直收敛一样。**

**因此，早期停止将使模型在预测赛季得分(更高的 RMSE)方面不如收敛混合模型准确。然而，类似于模型惩罚，这可能会使模型不太适应训练数据，并更准确地对溜冰者进行排名(更高的肯德尔τ)。这在前面的动画仪表盘中可以看到，它监控梯度下降迭代中的这些指标，我在下面的静态版本中复制了这些指标。**

**![](img/8688bf48e52dd41bdc242f5a07328b95.png)**

**RMSE and Kendall’s tau across iterations of gradient descent (for male skaters in 2017 season)**

**然而，与模型惩罚相比，提前停止提供了一些额外的优势:**

*   **对于提前停止，模型仅运行—并且提前停止—一次，相比之下，对于模型惩罚，模型运行多次，每次使用不同的惩罚参数λ。**
*   **对于模型惩罚，我们首先需要猜测惩罚参数λ的一些值来运行模型(例如当λ增加 10 倍时，如前面所示)。如果某个值λ提高了模型的肯德尔τ，我们需要测试这个λ的邻近值，看看是否有任何额外的提高。相比之下，对于早期停止，我们总是可以找到 Kendall 最高的单个迭代，并让该迭代成为停止点。换句话说，在早期停止中，没有“四处寻找”最佳模型[超参数](https://en.wikipedia.org/wiki/Hyperparameter_(machine_learning))的忙乱。**

## **编码**

**在我们的梯度下降的 Python 实现中，早期停止仅仅意味着将步骤 2 的循环次数限制到一个小的数目——例如 `for i in range(10)`。之后，可以计算 10 次迭代后的预测排名的肯德尔τ，我们继续尝试其他少量的迭代，并选择具有最高肯德尔τ的一个。**

**然而，执行早期停止的更实际的方法是运行大量迭代(比如 1000 次)的梯度下降，监视每次迭代的 Kendallτ(类似于我们之前对仪表板动画所做的)，并追溯性地查明 Kendallτ最高的迭代。如果在 Kendall 的 tau 中有几个最大值，我们选择最早的迭代，因为我们希望在较少的模型过度拟合方面出错。**

## **估价**

**![](img/a74954df4c0aed2c86cc7fd8523e6c1d.png)**

**Each dot represents the stopping point (highest Kendall’s tau) for each year. For reference, curves for 2017 are highlighted in red.**

**当监测每年梯度下降迭代中的 RMSE 和肯德尔τ值时(见附图)，我们可以看到:**

*   **在几乎所有的 10 年中，如果梯度下降很早就停止，甚至在第 40 次迭代之前，梯度下降就达到最大肯德尔τ(见下图)。相比之下，RMSE 通常在这些点之后继续下降(见上图)，这是有道理的，因为梯度下降的收敛过早地停止了。**
*   **10 年中只有一年在 40 次迭代后出现最大肯德尔τ，即 2014 年的第 60 次迭代。然而，这可能是侥幸，因为上图显示，在第 40 次迭代之前的几次迭代中，今年的肯德尔τ已经非常接近最大值。**
*   **然而，尽管大多数年份的理想停止点低于 40 次迭代，但我们可以看到，在一些情况下，一年的理想停止点可能是另一年的灾难(尝试在每个购物点画一条垂直线，并在该迭代中查看其他年份的肯德尔τ)。鉴于肯德尔τ在梯度下降的早期迭代中高度不稳定，这一点尤其正确。**

**因此，我们必须通过在每次迭代中取平均值来平滑所有 10 年中的肯德尔τ，并找出哪个迭代具有最高的跨年度平均值。更准确地说，对于每次迭代，我们对混合模型的肯德尔τ与每年的基线模型的差异进行平均(超过 10 年)。这使我们能够直接比较早期停止对基线模型的影响，下图(棕色部分)突出显示了这一点:**

**![](img/cc31e68ec9c1ad0ee57e266db4af9c0b.png)**

**Difference in Kendall’s tau of hybrid model over baseline model (season averages) at each iteration of gradient descent**

**从上图中，我们可以看出:**

*   **在梯度下降的第 47 次迭代时，混合模型相对于基线模型的平均肯德尔τ差异(超过 10 年)达到最大值。**
*   **然而，快速检查表明，对于大多数年来说，这个迭代已经过了他们的停止点，这意味着他们的肯德尔的 tau 已经走下坡路相当一段时间了。此外，虽然有些年份的肯德尔τ在第 47 次迭代时显示出与基线相比的显著提高(有些年份的肯德尔τ提高了近 0.1)，但仍有一些年份的肯德尔τ无论是否提前停止，与基线模型相比始终表现不佳。**
*   **因此，毫不奇怪，在 10 年的训练集中，即使是最好的早期停止点，肯德尔的 tau 也只比赛季平均值的基线模型提高了 0.005。更糟糕的是，这种改善在统计学上毫无意义，多年来高度可变的 Kendall tau 改善超过基线就是证明。这相当于一个非常宽的 95%置信区间，很容易包含零水平线(见上图)。**

**简而言之，即使提前停止似乎也无助于提高混动车型的预测排名。然而，在我们完全放弃混合模型之前，让我们看看我们是否可以以某种方式结合两种策略来减少模型过度拟合——模型惩罚和提前停止——作为改进该模型的最后努力。**

# **结合模型惩罚和提前停止**

**将模型惩罚和早期停止相结合无非是在计算梯度(模型惩罚)的同时包括惩罚参数λ，同时将梯度下降的迭代次数限制为较小的数目(早期停止)。因此，除了限制梯度下降算法步骤 2 的迭代次数外，Python 代码对模型惩罚没有任何改变——例如，只有`for i in range(10)`。**

**对于模型参数的每个组合(梯度下降的迭代次数以及惩罚参数λ),我们可以测量混合模型相对于基线模型的肯德尔τ的平均差异(超过 10 年),并将其绘制在下面的热图中:**

**![](img/14981548df589595f1d8baaa1baf4d95.png)**

**Color intensity signifies how much higher on average the Kendall’s tau of hybrid model over the baseline model is**

**从上面的热图可以看出:**

*   **即使模型被罚(λ>0)，梯度下降的理想停止点仍在 40–50 次迭代附近徘徊，因为与基线相比，这是肯德尔τ平均最高的区域。**
*   **然而，随着模型越来越多地被惩罚(增加λ)，当与模型惩罚相结合时，早期停止似乎越来越无效。这在λ=10 及以上的贫瘠荒地上看得最清楚，除了一个小岛在λ=100 时迭代次数接近 80 次。**
*   **早期停止迭代和λ之间的关系可以以不同的方式可视化，首先通过挑选每个λ的 Kendall(在基线模型上)具有最高平均差的迭代(该λ的理想停止点),并将其性能与如果没有执行早期停止(即如果允许模型完全收敛)的性能进行比较。**

**![](img/61f1dee3295e46e947632720b985acec.png)**

**The small numbers on the graph show the ideal iteration for early stopping at each λ**

**以上是男性和女性溜冰者的对比图，显示了非常有趣的模式:**

*   **对于女性滑冰运动员来说，与之前讨论的男性运动员类似，单独的惩罚往往会降低她们预测排名的准确性(右图中的蓝线):随着λ的增加，肯德尔τ与基线模型的平均差异越小。**
*   **对于男性和女性来说，提前停止可以使肯德尔的 tau 值比基线模型有最大的提升:男性平均提升 0.005，而女性则提升了 0.009。诚然，这些改善在统计上并不显著，这可以从它们包含水平线为零的宽置信区间中看出。**
*   **此外，这种通过提前停止的改进适用于每一级惩罚，尤其是当模型被高度惩罚(高λ)时。相比之下，即使早期停止也仍然无法挽救一个高度惩罚的模型，正如当λ增加时早期停止模型的 Kendall tau 改进普遍降低所证明的那样(两个图中的紫色线)。**

# **结论**

**从上面的分析可以清楚地看出，对于这个问题，提前停止不仅比模型惩罚(如前所述)更容易实现，而且与后者相比，它还具有明显的性能优势。因此，对于两种性别，只选择早期停车来改进混合模式。以下是将其应用于 10 年训练集时的最终结果:**

**![](img/34bd3117380d0179cdf8b99b41bf515f.png)**

*   **请注意，即使早期停止模型与基线模型相比在肯德尔τ方面提供了边际改善，这些改善在很大程度上是不显著的。这可以从肯德尔τ与基线模型的平均差异中看出:这种差异的 95%置信区间对于两种性别来说仍然低于零。**
*   **此外，由于世界锦标赛排名本身被用于选择最佳模型参数，即梯度下降应该停止时的迭代，模型本身很可能过度拟合 10 年训练集中的数据。因此，在项目的[最后部分](https://medium.com/@seismatica/predict-figure-skating-world-championship-ranking-from-season-performances-d97bfbd37807?source=friends_link&sk=2f7deacde0e597d10fe5761b611bce12)，我将评估这些提前停止的模型(以及其他模型)与测试集中剩余的 4 年作为最终基准。**

# **资源**

**在项目的这一部分开发的混合模型最初是受推荐系统中使用的各种矩阵分解技术的启发。一般来说，这些技术会将许多用户对许多商品(例如，在亚马逊上)留下的评级矩阵“分解”成特定于用户的矩阵和特定于商品的矩阵。当这两个矩阵相乘时，它们可以近似原始评级矩阵。**

**同样的原理在我们的混合模型中成立:评级矩阵只不过是所有事件-溜冰者对的季节分数的数据透视表，而用户特定和项目特定的矩阵只不过是溜冰者和事件潜在分数向量。因此，当乘以`skater_scores @ season_scores`(加上顶部的基线分数)时，我们可以近似得出赛季的`true_scores`。您可以查看这篇非常可读的关于推荐系统矩阵分解的[文章](https://datajobs.com/data-science-repo/Recommender-Systems-%5BNetflix%5D.pdf)，其中也包括梯度下降算法的推导，该算法与本项目中使用的算法非常相似，通常被称为 [FunkSVD](https://sifter.org/~simon/journal/20061211.html) 。**

**也就是说，我们目前的混合模型只涉及一个单一因素:每个运动员和每个项目只由一个潜在的分数来表示。相比之下，推荐系统中的矩阵分解往往涉及多个因素。在项目的[下一部分](https://medium.com/@seismatica/predict-figure-skating-world-championship-ranking-from-season-performances-8af099351e9c?source=friends_link&sk=48c2971de1a7aa77352eb96eec77f249)中，我将描述多因子矩阵分解如何应用于我们的排序问题。**