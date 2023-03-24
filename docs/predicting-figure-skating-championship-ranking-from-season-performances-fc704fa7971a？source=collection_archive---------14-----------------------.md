# 从赛季表现预测花样滑冰世锦赛排名

> 原文：<https://towardsdatascience.com/predicting-figure-skating-championship-ranking-from-season-performances-fc704fa7971a?source=collection_archive---------14----------------------->

## 体育分析

## 第 1 部分:线性模型

![](img/569fd1c16bdac69d9f8dd73db52c31be.png)

*   *要看我为这个项目写的代码，可以看看它的 Github* [*回购*](https://github.com/dknguyengit/skate_predict)
*   *对于项目的其他部分:第一部分，* [*第二部分*](/predicting-figure-skating-world-championship-ranking-from-season-performances-part-2-hybrid-7d296747b15?source=friends_link&sk=86881d127654ece260be2e3029dfbad2)*[*第三部分*](https://medium.com/@seismatica/predict-figure-skating-world-championship-ranking-from-season-performances-8af099351e9c?source=friends_link&sk=48c2971de1a7aa77352eb96eec77f249)*[*第四部分*](https://medium.com/@seismatica/predict-figure-skating-world-championship-ranking-from-season-performances-a4771f2460d2?source=friends_link&sk=61ecc86c4340e2e3095720cae80c0e70)*[*第五部分*](https://medium.com/@seismatica/predict-figure-skating-world-championship-ranking-from-season-performances-7461dc5c0722?source=friends_link&sk=fcf7e410d33925363d0bbbcf59130ade)*[*第六部分*](https://medium.com/@seismatica/predict-figure-skating-world-championship-ranking-from-season-performances-d97bfbd37807)****

# **背景**

**几年前我开始关注花样滑冰，现在已经成为这项运动的忠实粉丝。与许多运动类似，花样滑冰比赛分为两大类:**

*   **每年 10 月至次年 3 月的一系列小型赛事，包括:在 6 个不同国家(美、加、俄、法、中、日)举行的[大奖赛](https://en.wikipedia.org/wiki/ISU_Grand_Prix_of_Figure_Skating)系列赛，大奖赛系列赛 6 名最佳选手之间的[大奖赛决赛](https://en.wikipedia.org/wiki/Grand_Prix_of_Figure_Skating_Final)比赛，欧洲运动员的[欧洲锦标赛](https://en.wikipedia.org/wiki/European_Figure_Skating_Championships)，以及其他大洲运动员的另一场比赛，适当的名称为[四大洲](https://en.wikipedia.org/wiki/Four_Continents_Figure_Skating_Championships)。当然，每四年，人们熟悉的冬季奥运会会在二月举行。**
*   **一年一度的[世界锦标赛](https://en.wikipedia.org/wiki/World_Figure_Skating_Championships):这是本赛季的盛大赛事，通常在三月底举行，为本赛季画上圆满的句号。全世界大约有 24 名选手将参加每个项目(男、女)的比赛。**

# **问题**

**每年，在世界锦标赛之前，互联网上都充斥着对谁将登上冠军领奖台的预测。做出预测的显而易见的来源是赛季早期比赛中运动员的分数，根据这些分数对运动员进行排名的显而易见的方法是比较每个运动员在他或她之前所有比赛中的平均分数。**

**然而，这种方法的一个潜在问题是分数是跨不同事件平均的，没有两个事件是相同的。首先，每个项目的评判小组大不相同。另一方面，可能有一些其他因素可以系统地提高或降低某个项目的运动员分数，例如冰的情况，项目发生的时间，甚至海拔高度(运动员可能不适应高海拔的项目)。正如下面 2017 赛季男子滑冰运动员的方框图所示，每个项目的得分中心和分布可能会明显不同:**

**![](img/337d801b5b1aedfeaf0189b47b91a9bc.png)**

**Box plot of skater scores for 2017 season events**

**因此，一个排名模型可以梳理出**运动员效应**，每个运动员的内在能力，从**事件效应**，一个事件对运动员表现的影响，可能会对谁在世界锦标赛中表现更好或更差做出更好的预测。**

**在这个项目中，我们将探索几个这样的模型来实现这个目标。你正在阅读的第一部分涉及简单的线性模型，而项目的后续部分将涵盖更复杂的模型。**

# **数据**

## **数据抓取**

**好消息是:从 2005 年(现行评分系统实施的那一年)开始的所有比赛的分数都可以通过国际滑冰联盟的官方网站获得。分数包含在普通 HTML 页面的表格中，所以使用 [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) 来抓取它们就足够了。**

**坏消息是:每个赛事都有自己的分数网页。正如十多年来不同活动组织者建立的网页所预期的那样，它们有轻微但非常令人沮丧的不同链接模板和 HTML 格式，从 2005 年的页面和 2018 年的页面可以看出。**

**![](img/e3b37f6b3bf67d91137113b7409f172f.png)**

****Left:** score page in 2005\. **Right:** score page in 2018.**

**因此，我们非常小心地确保所有的分数都是准确的。例如，下面是几个链接模板，用于捕捉所有网页的事件分数:**

```
**'http://www.isuresults.com/results/gp{0}{1}/CAT00{3}RS.HTM',
'http://www.isuresults.com/results/gp{0}20{1}/CAT00{3}RS.HTM',
'http://www.isuresults.com/results/gp{0}{1}{2}/CAT00{3}RS.HTM',
'http://www.isuresults.com/results/season{1}{2}/gp{0}{1}{2}/CAT00{3}RS.HTM',
'http://www.isuresults.com/results/season{1}{2}/gp{0}{1}{2}/data0{3}90.htm',
'http://www.isuresults.com/results/season{1}{2}/gp{0}20{1}/CAT00{3}RS.HTM',
'http://www.isuresults.com/results/season{1}{2}/gp{0}20{1}/data0{3}90.htm'**
```

**链接样板中的占位符有助于识别特定事件，包括:**

*   **`{0}`:赛事类型(如`'usa'`代表美国大奖赛，`'gpf'`代表大奖赛决赛，以此类推)**
*   **`{1}`:季节的开始年份(如`'06'`)**
*   **`{2}`:季节的结束年(如`'07'`)**
*   **`{3}`:对应评分页面的选手性别(`'1'`为男性，`'2'`为女性)**

**例如，如果`{0}='gpf'`、`{1}='06'`、`{2}='07'`、`{3}='1'`，那么相应的模板就变成了[http://www.isuresults.com/results/gpf0607/CAT001RS.HTM](http://www.isuresults.com/results/gpf0607/CAT001RS.HTM)，这是 2006-2007 赛季男子滑冰大奖赛总决赛的比分网页。如果一个模板不起作用，scraper 会沿着列表向下尝试另一个模板，直到找到一个有效的链接。**

## **数据清理**

**一旦所有事件的分数被清除，它们将通过以下步骤被清除:**

1.  ****删除所有多余的空格**，包括[不换行空格](https://en.wikipedia.org/wiki/Non-breaking_space)如`\xa0`**
2.  ****将溜冰者的名字从姓氏中分离出来**。由于名称顺序的不一致，这是必要的:一些页面将名称存储为`First Name LAST NAME`，而一些存储为`LAST NAME First Name`。因此，将名和姓分开可以使运动员在他或她参加的所有比赛中被完全识别。值得庆幸的是，所有选手的姓氏都是大写的，只需要一些简单的正则表达式就可以进行拆分。例如，匹配运动员姓氏的正则表达式的一部分是`[A-ZÄÖÜ-]+`，它允许像带重音符号的`PÖYKIÖ`或带连字符的`GILLERON-GORRY`这样的姓氏被捕获。**
3.  ****标准化备选名称拼写**。由于运动员的数量足够少，通过他们的名字进行人工检查可以识别出运动员名字在多个项目中的不同拼写，例如名字的`Min-Jeong`和`Min-Jung`，或者姓氏的`TUKTAMYSHEVA`和`TUKTAMISHEVA`。从下面的列表中可以看出，可供选择的拼写大多是有多个名字(有时用连字符连接)的韩国滑冰运动员，夹杂着多音节的俄罗斯名字和充满元音的北欧名字。**

**![](img/712b4a145745bd9a98871fab6ff6aa88.png)**

**Alternative spellings of skaters’ names (blue: male, pink: female)**

## **数据分割**

**在 14 年的粗略分数中，随机 10 年的分数将用于训练和验证不同的排名模型，而剩余 4 年的分数将用于在最后对模型进行基准测试。这个项目的后面部分将专注于更复杂的选手排名模型，我将利用 4 年的测试时间在项目的最后部分对所有模型进行基准测试。**

# **排名度量**

**在我们比较不同的模型对世界锦标赛选手的排名之前，让我们首先定义我们如何评价一个排名比另一个更好或更差。**

**我们知道，一个好的预测排名是一个与最终世界锦标赛结果非常接近的排名。可以衡量两个排名之间相似性的一个简单度量就是 [**肯德尔排名相关系数**](https://en.wikipedia.org/wiki/Kendall_rank_correlation_coefficient) (也叫肯德尔的τ系数)。让我们通过一个简单的例子来看看这个指标是如何工作的:**

```
**Predicted ranking   Actual ranking  
  1\. AARON            1\. BRUCE                 
  2\. BRUCE            2\. AARON                   
  3\. CRAIG            3\. CRAIG**
```

**上表显示了一些预测排名(左)和世界锦标赛后的实际排名(右)，每个从最好到最差，为 3 个虚构的选手。比较这两个排名，我们可以看到预测的排名错误地将 Aaron 置于 Bruce 之上。然而，克雷格的排名是正确的:他在两个排名中都低于其他两名选手。**

**肯德尔τ系数(𝜏)可以用下面的公式量化上述描述:**

**![](img/ab14a03d68c67a1eb485cac7e27dd143.png)**

**n: number of ranked skaters**

**该公式的工作原理如下:**

*   **根据`AARON > BRUCE > CRAIG`的预测排名，可以生成有序的选手对，其中选手对中第一个选手的排名高于第二个选手。对于 3 个溜冰者(n = 3)的排名，我们最终得到 3×(3–1)/2 = 3 个有序对。它们是:**

```
**(AARON, BRUCE), (AARON, CRAIG), (BRUCE, CRAIG)**
```

*   **在这 3 对有序的组合中，第一对`(AARON, BRUCE)`不再保持`BRUCE > AARON > CRAIG`的冠军排名，而后面两对仍然保持。换句话说，**一致对**——在两种排序中都成立的有序对——的数量是 2，而**不一致对**——在一种排序中成立但在另一种排序中不成立的有序对——的数量是 1。**
*   **因此，这两个排名的肯德尔τ系数为:**

**![](img/2f4063b266436b020b38d0542865e01b.png)**

**结果，在两个排序中存在的有序对越多，这两个排序之间的肯德尔τ系数就越高。两个相同的排名将有𝜏 =1，而两个完全相反的排名(想想:一个排名从最高到最低，而另一个从最低到最高)将有𝜏 = -1。因此，当与同赛季的世界冠军排名进行比较时，肯德尔的τ系数较高(更接近 1)的预测排名将优于系数较低的预测排名。**

## **履行**

**在 Python 中，`scipy`库内置了`[scipy.stats.kendalltau](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.kendalltau.html)`函数来计算系数。然而，“从零开始”实现这种计算非常容易:`itertools`模块中的`[combinations](https://docs.python.org/2/library/itertools.html#itertools.combinations)`函数将从每个排名中生成有序对，两个排名之间的一致对只不过是两组有序对的交集(通过`&`运算符):**

```
**from itertools import combinations
ranking1 = ['AARON', 'BRUCE', 'CRAIG']
ranking2 = ['BRUCE', 'AARON', 'CRAIG']# Generate set of ordered pairs from each ranking
ordered_pairs1 = set(combinations(ranking1, 2))
# {('AARON', 'BRUCE'), ('AARON', 'CRAIG'), ('BRUCE', 'CRAIG')}
ordered_pairs2 = set(combinations(ranking2, 2))
# {('AARON', 'CRAIG'), ('BRUCE', 'AARON'), ('BRUCE', 'CRAIG')}# Calculate number of total, concordant & discordant pairs and tau
n_pairs = len(ordered_pairs1) # 3
n_concordant_pairs = len(ordered_pairs1 & ordered_pairs2) # 2
n_discordant_pairs = n_pairs - n_concordant_pairs # 1# Calculate Kendall's tau from pair counts
tau = (n_concordant_pairs - n_discordant_pairs) / n_pairs # 0.33**
```

# **季节平均模型(基线)**

**正如开始提到的，这是预测世界锦标赛上运动员排名的最简单和最明显的方法，可以通过以下方式实现:**

1.  **平均每位选手在本赛季参加的所有比赛中的得分**
2.  **根据这些赛季平均水平，从最高到最低对运动员进行排名，并使用该排名作为世界锦标赛的预测排名**

**以 2017 年男子滑冰运动员的成绩为例，我们可以看到，使用赛季平均值预测的排名与世界锦标赛的最终排名具有相对较强的相关性(见下面的排行榜对比)。和任何运动一样，都有很大的惊喜:`Denis, TEN`排名下降了 10 位，从第 6 位降至第 16 位，而`Brendan, KERRY`上升了 7 位，从第 22 位升至第 15 位。然而，快速浏览一下，赛季中顶级选手的表现往往比低级选手更稳定，这可以从排行榜顶部与底部相比相对较少的排名中看出。**

**![](img/a568cd1924f34d16731f43205de95941.png)**

**Ranking by season average vs. world ranking (for male skaters in the 2017 season). Colors are based on the world championship ranking.**

**24 个滑手，2017 年世锦赛实际排名有 24×23/2 = 276 个可能的有序对。在这 276 对中，季节平均模型正确预测了 234 对(如`Yuzuru, HANYU > Javier, FERNANDEZ`)，剩下 42 对预测错误(如`Boyang, JIN > Nathan, CHEN)`)。这转化为肯德尔τ分数为(234 - 42)/276 = 0.695。对于基线模型来说还不算太差！**

# **加性模型**

## **理论**

**让我们从季节平均值的基线模型转移到其他模型，这些模型有望将事件影响与运动员影响分开。一个简单的这样的模型是假设在给定的事件中给定的溜冰者的分数是以下的**加**:**

*   **赛季中所有分数的基准分数**
*   ****潜在的**事件分数**对于该事件是唯一的，但是对于该事件的所有运动员是相同的****
*   ****一个潜在的滑冰者分数对于该滑冰者来说是唯一的，但是对于该滑冰者参加的所有项目来说是相同的****
*   ****零均值高斯噪声****

******这个加法模型可以用数学方法表示为:******

******![](img/51962cee397e76c48899a425da19e7f0.png)******

*   ******`y_event-skater`:给定运动员在给定比赛项目中的得分******
*   ******`θ_baseline`:基线得分(整个赛季得分不变)******
*   ******`θ_event`:潜在事件分数(运动员之间不变)******
*   ******`θ_skater`:潜在选手得分(跨赛事不变)******
*   ******`N(0, σ²)`:均值为零的高斯噪声******

******乍一看，似乎没有简单的方法来学习这些潜在分数，因为每对运动员都有不同的潜在分数组合。换句话说，不存在机器学习问题所预期的所有数据点的固定特征集。******

******但是，模型公式可以重写为以下形式:******

******![](img/80dd8fc39b54d975cb38db2498b6cc7c.png)******

******这种形式的模型公式中的新术语是:******

*   ******`θ_e`:事件的潜在得分`e`******
*   ******`I(e = event)`:如果事件`e`是给定事件，则为 1，否则为 0******
*   ******`θ_s`:滑手潜在得分`s`******
*   ******`I(s = skater)` : 1 如果滑冰者`s`是给定的滑冰者，否则为 0******

******以这种方式书写，很容易看出，对于给定的一对运动员，只有该运动员和该项目的潜在得分将计入该对运动员的实际得分，这与前面的公式完全相同。这是因为在上述公式中，那些潜在得分的系数`I(e = event)`和`I(s = skater)`将为 1，而其他滑手和项目的潜在得分的系数将为 0。******

******同样，以这种方式书写，每个运动员的分数可以简单地认为是基线分数和零均值高斯噪声之上的二元变量`I(s = skater)`和`I(e = event)`的线性组合。换句话说，问题就变成了一个简单的二元特征线性回归，其回归系数的确是所有项目和滑手的潜在得分——分别为`θ_e`和`θ_s`——回归截距不过是基线得分`θ_baseline`。******

## ******编码******

******我们通过一个小例子(使用 2017 赛季的男性得分)来看看这个线性回归模型是如何编码的:******

*   ******清理完数据后，我们以下面的数据框`season_scores`结束，它代表了本赛季所有赛事选手的分数(当然，除了世界锦标赛)。对于 2017 赛季，有来自 9 项赛事的 62 名选手的 120 个分数。******

******![](img/443868b7003817e86d8af958e2601c66.png)******

*   ******接下来，我们使用 panda 的`[get_dummies](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.get_dummies.html)`函数将这个数据帧转换成包含虚拟特征的数据帧`dummies`，虚拟特征就是上面提到的二进制特征。******

```
******dummies = pd.get_dummies(season_scores[['name', 'event']], prefix=['', ''], prefix_sep='', drop_first=True)******
```

******请注意，我们必须使用函数的`drop_first=True`参数，因为包含所有溜冰者和事件的*二元变量将导致二元特征完全[共线](https://en.wikipedia.org/wiki/Multicollinearity)，这对于线性回归是不允许的；这有时被称为[虚拟变量陷阱](https://www.algosome.com/articles/dummy-variable-trap-regression.html)。实际上，这种争论将会抛弃一个运动员和一个项目的二元特征。由于有 62 名溜冰者和 9 个事件，我们的`dummies`数据框将有 61 个溜冰者二进制列和 8 个事件二进制列，总共有 69 个二进制要素。可以通过下面的方法找到被丢弃的运动员和事件的身份:*******

```
******# Get unique names of skaters and events from original data frame
unique_skaters = season_scores['name'].unique()
unique_events = season_scores['event'].unique()# Get unique names of skaters and events from dummies data frame 
dummies_skater_count = len(unique_skaters) - 1
dummies_skaters = dummies.columns[:dummies_skater_count]
dummies_events = dummies.columns[dummies_skater_count:]# Dropped skater is 'Adam, RIPPON'
dropped_skater = list(set(unique_skaters) - set(dummies_skaters))[0]# Dropped event is '4C'
dropped_event = list(set(unique_events) - set(dummies_events))[0]******
```

******删除这两列后，得到的`dummies`数据表如下左图所示，其中 120 行对应于该赛季的 120 个唯一得分，69 列为二进制哑变量。所有 1 的列被附加到数据表的左侧，以对应基线得分，该基线得分出现在所有赛季得分中。因此，基线分数将只是回归系数中的一个，它现在有 70 个。******

******![](img/ab17ec0c43e28b79958ec2dc64a5de7f.png)******

********Left:** feature matrix (predictors). Center: coefficient vector. **Right:** score vector (response). Highlighted in red is how one row of the feature matrix can make a dot product with the coefficient vector to approximate the season score of an event-skater pair (CA-Yuzuru, HANYU).******

******创建这个 120×70 数据表的原因是，它可以表示一个二进制特征矩阵`X`，当乘以表示所有 70 个回归系数(包括基线/截距)的向量`θ`时，它可以近似该赛季的所有得分——长度为 120 的向量`y`。换句话说，线性回归问题可以用矩阵形式表示为:******

******![](img/d85d4fd2a4b4f409db92ecb5c7fdb755.png)******

******其中`ε`表示对每个赛季得分有贡献的高斯噪声。因此，回归系数的最小二乘估计可以通过求解线性回归的[标准方程](https://en.wikipedia.org/wiki/Linear_least_squares#Derivation_of_the_normal_equations)来轻松计算:******

******![](img/bb9be9e26fca77b3ed424e08bcfc02ad.png)******

******由于每个季节的数据都很小，如微小的 120×70 要素矩阵所示，因此在 Python 中可以很容易地求解法线方程，以返回回归系数的精确值(与梯度下降之类的东西相反)。这个的代码如下，用`.T`做矩阵转置，`@`做矩阵乘法，`np.linalg.inv`做矩阵求逆。******

```
******# Create numpy feature matrix and response vector
X = dummies.values
y = season_scores['score'].values# Append column of all 1's to the left (0th-index) of feature matrix
X = np.insert(X, obj=0, values=1, axis=1)# Apply normal equation to find linear regression coefficients (including intercept)
coefs = np.linalg.inv(X.T @ X) @ (X.T @ y)******
```

******从通过解正规方程找到的回归系数中，可以提取基线分数、溜冰者分数和项目分数。请注意，我们需要添加回运动员和事件的分数，这些分数在回归之前被删除；他们的分数为零，因为通过放弃他们，他们的分数已经被吸收到基线分数中。******

```
******# Get baseline, skater & event scores from regression coefficients
baseline_score = coefs[0]
skater_scores = pd.Series(coefs[1:dummies_skater_count+1], index=dummies_skaters)
event_scores = pd.Series(coefs[dummies_skater_count+1:], index=dummies_events)# Add back scores of dropped skater and dropped event
skater_scores[dropped_skater] = 0
event_scores[dropped_event] = 0# Sort skater scores and event scores from highest to lowest
skater_scores.sort_values(ascending=False, inplace=True)
event_scores.sort_values(ascending=False, inplace=True)******
```

## ******结果******

******![](img/651c8f26180470910ad862057bed3d78.png)******

******一旦对潜在的溜冰者分数进行了排序，基于加法模型的预测排名就像将具有最高潜在分数的溜冰者排名为第一，将具有第二高潜在分数的溜冰者排名为第二一样简单，以此类推。******

******此外，这些潜在分数可以被解释为一个运动员与被淘汰的运动员相比好或差多少，一旦去除了事件影响(根据加法模型)，参考分数为零。******

******另外，请注意，在已经计算出潜在分数的 62 名滑手中，只有 24 人获得了 2017 年世界锦标赛的参赛资格。因此，我们只根据那些滑手的预测排名来评估模型。******

## ******估价******

******首先，让我们看看加法模型在逼近赛季中的实际得分时表现如何，因为这是回归的主要目标。下面是一个图表，显示了本赛季每对赛事选手的预测分数，每个分数都按其潜在成分(基线分数+赛事分数+选手分数)与该对选手的真实分数进行了细分。******

******![](img/951806ff4e191138f84c147df6b2cee4.png)******

******Each tick represents a skater. Each gray bar represents the discrepancy between predicted and true season scores.******

******上图显示，预测得分与赛季中的真实得分非常接近，与真实得分的最大偏差(也称为[残差](https://en.wikipedia.org/wiki/Errors_and_residuals))不到 30 分。我们可以根据预测得分绘制这些残差，如果使用了季节平均模型，还可以绘制相同的残差。此外，我们可以绘制这些残差的直方图和 [QQ 图](https://en.wikipedia.org/wiki/Q%E2%80%93Q_plot)，以确认线性回归模型的高斯噪声假设是否合理。******

******![](img/aa7cbd819c0412206fce67a4840ee45d.png)******

********Top:** residual vs. predicted score for season average and additive models. **Bottom:** histogram and QQ plot of additive models’s residuals******

******从上面的情节可以看出:******

*   ******来自加法模型的残差在整个预测分数范围内是相当随机的。这加强了随机噪声的模型假设，与不同分数范围的噪声不同(也称为[异方差](https://en.wikipedia.org/wiki/Heteroscedasticity))，其在残差与预测分数图中具有特征性的“扇形”形状。******
*   ******从残差的直方图和 QQ 图来看，残差的分布非常接近高斯分布。这证明了线性模型的零均值高斯噪声假设。******
*   ******然而，似乎有异常大量的零残差，从直方图的高峰以及 QQ 图中间的扭结可以看出。经过一些检查，这些零残差原来只属于在赛季中只参加一个项目的运动员。这是有意义的，例如，如果梯度下降被用来寻找潜在得分:这样的滑冰运动员的潜在得分只对单个项目的预测得分有贡献，而整个赛季的其他预测得分不变。因此，在梯度下降过程中，潜在的运动员分数可以自由地自我调整，使得该项目-运动员对的真实分数和预测分数之间的差，即残差变为零。******
*   ******最后，将加法模型的残差与平均模型的残差进行比较，结果显示它们在规模上非常相似。事实上，线性模型的[均方根误差](https://en.wikipedia.org/wiki/Root-mean-square_deviation) (RMSE)为 8.84 点，比季节平均模型的 10.3 点要好一些。******

******其次，也是更重要的，让我们看看加法模型在预测世界锦标赛的最终排名方面做得如何(与运动员赛季平均水平的基线模型相比)。细心的读者会注意到，之前显示的加法模型得出的排名与赛季平均排名非常相似，这在将这两个排名相互比较时得到了证实，同时也与世界锦标赛排名进行了比较:******

******![](img/033c468de8148cc75a313f861a21831e.png)******

********Left:** predicted ranking from additive (linear) model. **Center:** predicted ranking from season average model. **Right:** actual ranking from 2017 world championship. Colors are based on the world championship ranking.******

******在比较这两种模型的预测排名时，有一些有趣的情况(以红色突出显示):******

*   ******还有一些滑手按季均排名更准确，比如`Moris, KVITELASHVILI`(加法排名第 16，季均排名第 12，世界排名第 13)。******
*   ******相比之下，也有一些从加性模型预测的排名更准确，比如`Paul, FENTZ`(加性第 20，季均第 17，世界第 20)。******
*   ******当然，由于体育运动的不可预测性，有些运动员的两个模型都没有提供好的预测:`Boyang, JIN`被预测会在两个排名中名列第七，但却出人意料地在世界锦标赛中获得了第三名。******

******总的来说，在世界锦标赛的 276 个有序对中，有 239 个是加法模型正确得到的，这转化为肯德尔的 tau 分数为 0.732。与具有 234 个正确对的季节平均模型(肯德尔的τ= 0.695)相比，相加模型与基线模型相比似乎提供很少的预测改进。因此，让我们转向下一款有望表现更好的车型。******

# ******乘法模型******

## ******理论******

******我们可以将这些潜在得分**乘以**，再乘以零均值高斯噪声的指数，从而得出赛季得分，而不是将赛季中的项目-运动员得分视为由一些潜在得分相加而成(基线+项目+运动员)。这种乘法关系可以用数学方法表示为:******

******![](img/b7ad65db0ba08d04d3c7d6bddf1a90be.png)******

******乘法模型相对于加法模型的一个潜在优势是，潜在分数可以在该模型中相互作用。例如，在附加模型中，一个给定的溜冰者的潜在效应保持不变，不管他或她参加什么项目。相比之下，乘法模型中的溜冰者分数充当基线和事件分数之上的“乘数”:如果你愿意，可以把这想象为溜冰者在“利用”每个事件方面有多好。******

******我们在公式末尾乘以高斯噪声的指数的原因是因为可以在整个公式中取对数，这导致:******

******![](img/c81a5270d00deea5a1fb9ecb0095fe7d.png)******

******换句话说，与加法模型一样，乘法模型只不过是一个具有二元特征的线性回归问题。然而，不同之处在于，响应现在是赛季得分的对数，回归系数现在是潜在得分的对数。******

## ******编码******

******由于模型核心的线性回归保持不变，乘法模型的编码很大程度上与加法模型相同，除了在最后解正规方程时:我们用`X.T @ np.log(y)`代替了`X.T @ y`。******

```
******coefs = np.linalg.inv(X.T @ X) @ (X.T @ np.log(y))******
```

******另一个小的修改发生在回归系数被获得之后，我们需要取它们的指数来得到潜在的基线、事件和选手分数。关于如何从回归系数中提取`baseline_score`、`event_scores`和`skater_scores`，请参考加性模型。******

```
******baseline_score = np.exp(baseline_score)
event_scores = np.exp(event_scores)
skater_scores = np.exp(skater_scores)******
```

## ******结果******

******以下是乘法模型恢复的选手潜在得分:******

******![](img/57ca3b71f8ab087d8ef65f2a45e14726.png)******

******最优秀的滑手，`Yuzuru, HANYU`“乘数”为 1.15；这意味着对于任何一个项目，在基线得分和项目得分相乘后，他的预测得分将是该数字的 1.15 倍。垫底的是排名最差的滑手`Julian Zhi Jie, YEE`，乘数为 0.77。******

## ******估价******

******尽管它们的形式不同，乘法模型为赛季中的每一对运动员提供了与加法模型非常相似的预测。当绘制两个模型的真实得分和预测得分之间的偏差以及赛季平均值的基线模型时，可以在下面看到这一点。因此，两个模型的 RMSE 实际上是不可区分的:加法模型为 8.84，乘法模型为 8.88(而季节平均模型为 10.3)。******

******![](img/74ebc8c58f59a42db868d8ef3df33223.png)******

******Deviation from true score for each event-skater pair of the season for all 3 models (average, additive, multiplicative)******

******在预测世界锦标赛的最终排名方面，这两个模型再次几乎相同(参见它们与最终世界排名的比较):******

******![](img/41450f25fd2d29be2b9c179d6bb2dbbb.png)******

******Predicted rankings of multiplicative and additive models compared to the 2017 world ranking. Colors are based on the world championship ranking.******

******2017 赛季的两个排名只有两个不同，在上图中用红色突出显示:******

*   ******乘法排名预测`Nathan, CHEN`排名高于`Javier, FERNANDEZ`，而加法排名预测相反。事实证明，在最终的世界锦标赛中，加法排名是正确的。******
*   ******乘法排名预测`Kevin, REYNOLDS`排名高于`Maxim, KOVTUN`，而加法排名预测相反。然而，在这种情况下，乘法排名在最终的世界锦标赛中确实是正确的。******

******因此，乘法排序与世界排序具有完全相同的一致对数量(276 对中的 239 对)和与加法排序相同的肯德尔τ(0.732)，这仍然高于季节平均值基线模型的肯德尔τ(0.695)。换句话说，至少在 2017 赛季，两个模型在预测世界冠军的最终排名方面似乎没有什么表现差异。******

# ******典型惩罚******

******假设模型选择(加法或乘法)似乎不会显著影响预测性能，让我们看看如何以另一种方式改进任一模型。******

******具有讽刺意味的是，改进这些模型的一个常用方法是**惩罚**它们(也称为[规则化](https://en.wikipedia.org/wiki/Regularization_(mathematics)))。对于这个问题，罚分可以理解为人为地将潜在事件和滑手分数向零收缩。一方面，这种萎缩无疑会使模型在预测赛季得分方面不太准确。另一方面，它可以防止模型过度适应赛季分数，赛季分数可能会很嘈杂，可能不会反映溜冰者的真实能力(可能他们在那天早些时候食物中毒)。因此，这可能会更好地捕捉运动员的真实能力，并对他们进行更准确的排名。******

******对于这个问题，我会选择 [**岭回归**](https://en.wikipedia.org/wiki/Tikhonov_regularization) 来惩罚两个模型，原因有二:******

*   ******岭回归及其近亲(例如[套索](https://en.wikipedia.org/wiki/Lasso_(statistics))和[弹性网](https://en.wikipedia.org/wiki/Elastic_net_regularization)回归)通常用于正则化线性回归问题。它也非常容易实现并合并到目前使用的正规方程中，如下所示。******
*   ******但是，与类似的正则化类型(如 lasso)相比，岭回归的一个优点是回归系数永远不会完全收缩为零。换句话说，多个滑手之间永远不会有平手，都是零潜分，适合这个排名问题。******

## ******理论******

******让我们看看如何将岭回归纳入用于线性回归的正规方程。回想一下加法模型的正规方程的解:******

******![](img/aa07b006dd85ed864899875ca58be6fe.png)******

*   ******`θ`:回归系数(70: 1 基线+ 8 项+ 61 名选手)******
*   ******`X`:二元特征表(120×70)******
*   ******`y`:响应向量(120)。对于乘法模型，这变成了`log(y)`******

******我们可以修改岭回归的这个解决方案，使回归系数向零收缩(下面公式的推导见最后的参考资料部分):******

******![](img/244dee57cb410302ca9e2f171812d939.png)******

******这个等式中的额外项是:******

*   ******`λ`:惩罚参数(标量)******
*   ******`I`:单位矩阵(70×70 矩阵，其条目除了对角线上的 1 之外都是 0，每个对角线条目对应一个回归系数)。注意，通常情况下，我们不会惩罚回归的截距。因此，单位矩阵`I`的第一个对角元素将是 0 而不是 1。******

******惩罚参数λ越高，模型将变得越惩罚，即潜在事件和运动员分数进一步向零收缩。相比之下，当λ = 0 时，模型成为原始的、未受惩罚的版本。******

## ******编码******

******给定上面的公式，为加性模型的岭回归编写正规方程的解再简单不过了，因为我们已经为同一个方程的正规版本编写了代码。******

```
******# Create 70x70 identity matrix
I = np.identity(n=X.shape[1])# Set first diagonal entry of identity matrix to zero
I[0, 0] = 0# Solving normal equation for ridge regression
lambda_reg = 0.01
coefs = np.linalg.inv(X.T @ X + lambda_reg * I) @ (X.T @ y)******
```

## ******结果******

******以 2017 赛季的加性模型为例，当λ增加 10 倍时，潜在项目和滑手分数一般会向零收缩。相比之下，基线分数相对不受惩罚参数的影响(这本来就不是它的本意):******

******![](img/cf0c277cb51f12ae8af48624562b98cb.png)******

******Effect of λ (penalty parameter) on baseline, skater, and event scores of additive model in 2017 season******

## ******估价******

******当潜在事件和运动员分数向零收缩时，模型在逼近赛季分数时自然会变差。因此，模型 RMSE 将增加(见下图):在极端情况下，当λ = 10000 时，RMSE 为 35.6，这比仅仅取每个溜冰者的季节平均值(RMSE = 10.3)要差得多，更不用说原始的、未被惩罚的模型(RMSE = 8.8)。******

******![](img/7f8cdacda3ae45bf2cd3b52a0d543644.png)******

******Effect of λ (penalty parameter) on RMSE, Kendall’s tau, and predicted ranking of additive model in 2017 season. Colors are based on world championship ranking.******

******然而，这种极端的萎缩会导致更好地预测世界锦标赛的排名吗？唉，没有。从上图来看，在所有 276 个有序配对中，极度惩罚模型只正确获得了 231 对，低于未惩罚模型的 239 对，甚至低于季节平均模型的 234 对！通过绘制当λ增加时预测排名如何变化的图表(见上文)，我们可以看到，直到λ = 0.1，惩罚模型的预测排名与原始模型相比没有太大变化，并且一致对的数量保持在 239。然而，在此之后，惩罚模型的排名开始有点混乱，这与和谐对数量的下降相一致(从λ = 0.1 时的 239 到λ = 1 时的 235)。******

******简而言之，惩罚加性模型造成了双重打击:它不仅使模型在逼近赛季得分方面变得更差，而且在预测世界冠军排名方面也变得更差。******

******然而，这只是单个赛季(2017 年)的结果。让我们在训练样本中对所有 10 年应用相同的模型(加法和乘法)。对于每个模型，我们从季节平均值的基线模型计算它们在 RMSE 和肯德尔τ中的差异。然后，我们使用 9 个自由度的 t 分布(因为 n=10 年)绘制这些差异的平均值，以及它们的 95%置信区间。这与执行[配对 t-检验](https://en.wikipedia.org/wiki/Student%27s_t-test#Unpaired_and_paired_two-sample_t-tests)来比较每个模型与基线相比的 RMSE 和肯德尔τ的平均差异是一样的。******

******![](img/d6c8bfdac42adcf9662bd2369d897b4c.png)******

******Effect of λ (penalty parameter) on differences (Δ) in RMSE and Kendall’s tau from baseline model, averaged over the 10-year training sample******

******从上面的图表中，我们可以看到:******

*   ******与季节平均值的基线模型相比，加法和乘法模型实际上以相同的量影响 RMSE，并且平均以相似的量影响肯德尔τ。******
*   ******就 RMSE(左图)而言，两个模型将 RMSE 从基线降低了λ < 0.1\. However, this in itself is of little value since approximating season scores is not our goal for this project. Furthermore, the RMSE naturally increases as λ increases, especially for λ > 0.1 的统计显著量，而 RMSE 从季节平均值的基线模型显著上升。******
*   ******就肯德尔τ(右图)而言，模型受到的惩罚越多(λ越大)，与基线相比，它们的肯德尔τ平均越差，不幸的是，这与我们之前从 2017 赛季得到的结果一致。更糟糕的是，从λ = 1 开始，这些模型开始从基线模型减少显著数量的肯德尔τ:注意肯德尔τ从基线的差异的 95%置信区间如何下降到水平线以下的零。因此，我们应该为这个项目选择非惩罚模型。******

******然而，即使使用未加惩罚的加法和乘法模型(λ = 0)，上图显示，平均而言，与季节平均值的基线模型相比，这些模型实际上降低了肯德尔τ😱当然，肯德尔τ差异的 95%置信区间表明，这种与基线的差异在统计学上不显著，因为它们包含了 0 处的水平线。但是，让我们分别绘制这三个模型每年的表现，以便我们可以更详细地比较这三个模型:******

******![](img/43712cea9a150dec114e365301d5f1b9.png)******

******Each gray line represents a year in the training set, with 2017 highlighted in red******

*   ******就 RMSE(左)而言，与训练样本中每年的季节平均模型相比，加法和乘法模型确实降低了 RMSE。然而，与 2017 赛季一样，这两个模型在 RMSE 方面没有太大差异，即它们同样接近赛季得分。******
*   ******然而，就肯德尔的τ(右)而言，画面就不那么清晰了。正如 2017 赛季早些时候所示，加法和乘法模型预测最终的世界冠军排名都优于赛季平均模型。然而，有相当多的年份，这两种模型给出的预测都比季节平均模型差。平均所有 10 年的训练集，这并不奇怪，他们的综合表现与基线相比是如此黯淡。******
*   ******请注意，到目前为止，所有的结果都来自男性选手。对于女滑手来说，情况基本保持不变，从下面的对比图可以看出。******

******![](img/c896411df896bf4dad735f90c29bcf91.png)******

******Comparison between three models for female skaters******

******以下是关于两个模型(加法和乘法)的 RMSE 和肯德尔τ性能的最终报告，包括平均值、与季节平均值基线模型的平均差异，以及 10 年训练样本中这些差异的 95%置信区间:******

******![](img/55e7874f86a79afa2d9acf2b9dc77e0d.png)******

******对于女性滑冰运动员，我们可以看到肯德尔τ(与季节平均值的基线模型相比)差异的 95%置信区间包含加法和乘法模型的零。换句话说，在从基线预测世界冠军排名时，两个模型都没有提供任何统计上的显著改进。******

# ******结论******

******尽管 2017 赛季的结果最初很乐观，但事实证明，为了预测最终的世界冠军排名，该项目期间提出的两个模型(加法和乘法)并不比简单地根据赛季平均水平对滑手进行排名更好。因此，根据[奥卡姆剃刀](https://en.wikipedia.org/wiki/Occam%27s_razor)，没有令人信服的理由选择这些更复杂的模型而不是更简单的季节平均模型。******

******一个问题仍然存在:考虑到这些模型中潜在的事件效应和运动员效应之间看似合理的相互作用，为什么它们在预测世界锦标赛排名时没有显示出显著的改进？******

******一个答案是体育运动太不可预测了。当然，这些模型可以很好地预测过去*发生的*事件的分数，正如他们的小 RMSE 事件所证明的那样，但是很难说在*未来的*事件中，没有人会受伤，或者有时差反应，或者只是在冰上撞了一个怪胎。这种不可预测性可以通过所有三种模型每年高度可变的肯德尔τ来反映:对于男性滑冰运动员，这可以在 0.5 到 0.8 以上的范围内(或者女性为 0.4 到 0.7)。换句话说，仅仅因为某年的世界锦标赛恰好有利于某个模特，并不意味着下个赛季不会有任何时髦的事情发生。******

******![](img/2e070c0500ff0fa148a2e1119ce462fc.png)******

******Ice, as they say, is slippery******

# ******资源******

******尽管与基线模型相比没有获得显著的改进，但这个项目让我知道，即使是最简单的模型，如线性回归，也可以以创造性的方式使用，如找到潜在的因素来解决这种情况下的排名问题。以下是一些有助于理解本文理论的资源:******

*   ******对于法方程解的推导(对于非惩罚和惩罚模型)，麻省理工学院开放式课件上的机器学习[课程](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-867-machine-learning-fall-2006/)的这个[讲义](https://ocw.mit.edu/courses/electrical-engineering-and-computer-science/6-867-machine-learning-fall-2006/lecture-notes/lec5.pdf)非常详细。注释还解释了惩罚参数λ的重要性:在贝叶斯设置中，较高的λ意味着潜在得分的高斯先验更集中于零。因此，这些分数的估计值也更有可能接近于零，正如在这个项目中所看到的那样。******
*   ******对于乘法模型，似乎可以用在[时序分析](http://www-ist.massey.ac.nz/dstirlin/CAST/CAST/Hmultiplicative/multiplicative1.html)中。此外，从这个讲座[幻灯片](http://statmath.wu.ac.at/~fruehwirth/Oekonometrie_I/Folien_Econometrics_I_teil2.pdf)来看，它似乎也可以被称为对数线性模型。然而，这些模型似乎包括取预测值的对数，而不是取感兴趣系数的对数(这正是本项目所做的)。******