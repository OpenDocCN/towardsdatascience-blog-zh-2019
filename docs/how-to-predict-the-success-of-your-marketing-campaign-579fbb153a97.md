# 如何预测你的营销活动的成功

> 原文：<https://towardsdatascience.com/how-to-predict-the-success-of-your-marketing-campaign-579fbb153a97?source=collection_archive---------2----------------------->

## 线性、树、森林和支持向量回归:比较、源代码和即用型应用程序

![](img/7a045c1668151809a3f35684b2353c2a.png)

Photo by [Anika Huizinga](https://unsplash.com/@iam_anih?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

在这篇文章中，我将带你经历一个建立、训练和评估数字营销活动中广告投放数量预测模型的过程。所有这些技术都可以类似地应用于其他回归问题，尤其是预测各种活动绩效指标。这些预测可用于**在推出**之前评估未来的营销活动，以及**确定最佳参数，包括此类活动的时间表和预算规模**。您可以使用自己的活动数据或提供的样本数据集来编写 Python 代码。在所有源代码旁边，我还提供了一个简单的应用程序来预测基于购买的数字营销活动的印象、点击和转换。

app:【predictor.stagelink.com
代码:[github.com/kinosal/predictor](https://github.com/kinosal/predictor)

# 故事大纲

1.  要求
2.  定义你的目标
3.  获取数据集
4.  第一眼
5.  预处理您的数据
6.  训练你的模特
7.  评估您的模型
8.  预测下一次活动的结果
9.  额外收获:现成的训练模型

# 要求

我们将使用过去营销活动的数据来预测未来营销活动的结果。一般来说，数据越多，即活动越多，预测就越准确。确切的数字取决于你的活动的同质性，但是你可能需要至少几百个活动的数据。此外，由于我们将使用监督学习技术，您需要相同的输入，即尺寸或特征，用于您想要估计结果的未来活动。

如果您现在手头没有合适的数据集，不要担心:您可以从这里下载一个 CSV 文件，其中包含我将在本文中使用的示例:

[https://github . com/kinosal/predictor/blob/master/model/impressions . CSV](https://github.com/kinosal/predictor/blob/master/model/impressions.csv)

# **明确你的目标**

当我们提到活动的成功或结果时，我们实际上指的是什么？这个显然要看你的具体情况。在这篇文章中，我们将尝试预测单个活动的印象数。类似地，点击和转化可以被预测来完成经典的营销漏斗:

![](img/997f0a2cd05316ffed7cf1529acebe84.png)

# **获取数据集**

我们看到了几个过去的活动，每个活动都提供了一个观察值或表中的一行，该表具有多个维度或列，包括我们要预测的因变量以及多个解释自变量或特征:

![](img/8010312cedea04f8d2183d863d4cba0a.png)

由于我们希望预测结果的活动存在于未来，因此这种情况下的特征不包括任何先前的性能数据，而是活动的不同可观察质量。由于我们通常事先不知道哪些功能将成为良好的预测工具，我建议也使用那些看起来与您的活动关系不大的变量，并投入一些时间来寻找或构建新功能。虽然也有关于[减少特征空间](https://en.wikipedia.org/wiki/Dimensionality_reduction)的争论，但这通常仍然可以在稍后阶段处理。

你可以用一个非常简单的功能加载 CSV 并保存到一个[熊猫](https://pandas.pydata.org)数据框中:

```
import pandas as pd
data = pd.read_csv('impressions.csv')
```

# **第一眼**

在建立和训练预测模型之前，我总是先看一眼数据，以了解我在处理什么，并找出潜在的异常。我们将使用样本数据来预测营销活动的印象数，因此“impressions.csv”包含每个活动的一行，每个活动的印象总数以及指标和分类特征，以帮助我们预测未来活动的印象数。我们将通过加载数据并显示其形状、列和前 5 行来确认这一点:

```
>>> data.shape
(241, 13)>>> data.columns
Index(['impressions', 'budget', 'start_month', 'end_month',
       'start_week', 'end_week', 'days', 'region', 'category',
       'facebook', 'instagram', 'google_search', 'google_display'],
      dtype='object')>>> data.head(5)
impressions budget start_month ... google search google_display
9586        600    7           ... 1             0
...
```

第一列包含从属(待预测)变量“印象”,而总共有 241 个记录(行)的 12 个特征列。我们还可以使用 *data.describe()* 来显示每个指标列的计数、平均值、标准差、范围和四分位数。

我们可以进一步观察到，我们正在处理十个数字特征和两个分类特征，而四个数字列是二进制的:

![](img/a5ca624827074c9213bae2ebd17a06a2.png)

现在让我们绘制数字特征的直方图。我们将使用两个非常方便的数据可视化库， [Matplotlib](https://matplotlib.org) 和 [Seaborn](https://seaborn.pydata.org) (构建于 Matplotlib 之上):

```
import matplotlib.pyplot as plt
import seaborn as snsquan = list(data.loc[:, data.dtypes != 'object'].columns.values)
grid = sns.FacetGrid(pd.melt(data, value_vars=quan),
                     col='variable', col_wrap=4, height=3, aspect=1,
                     sharex=False, sharey=False)
grid.map(plt.hist, 'value', color="steelblue")
plt.show()
```

![](img/830929e59d2407acb8f72cb77594e89f.png)

作为最后一瞥，我们将看看数字特征之间的基本线性相关性。首先，让我们用 Seaborn 热图来想象一下:

```
sns.heatmap(data._get_numeric_data().astype(float).corr(),
            square=True, cmap='RdBu_r', linewidths=.5,
            annot=True, fmt='.2f').figure.tight_layout()
plt.show()
```

![](img/62fc9aecf7caf54269c3eaadb907e80f.png)

此外，我们还可以输出每个特征与因变量的相关性:

```
>>> data.corr(method='pearson').iloc[0].sort_values(ascending=False)
impressions       1.000000
budget            0.556317
days              0.449491
google_display    0.269616
google_search     0.164593
instagram         0.073916
start_month       0.039573
start_week        0.029295
end_month         0.014446
end_week          0.012436
facebook         -0.382057
```

这里我们可以看到，印象数与预算金额和活动持续时间(天数)正相关，与使用脸书作为渠道的二元选项负相关。然而，这仅仅向我们展示了一种成对的线性关系，并且只能作为一种粗略的初始观察。

# **预处理您的数据**

在我们开始构建预测模型之前，我们需要确保我们的数据是干净的和可用的，因为这里适用于:“垃圾进，垃圾出。”

在这种情况下，我们很幸运地获得了一个结构相当良好的数据集，但我们仍然应该针对即将面临的挑战进行快速预处理:

1.  只保留因变量大于零的行，因为我们只想预测大于零的结果(理论上值等于零是可能的，但它们对我们的预测没有帮助)。
2.  检查缺少数据的列，并决定是删除还是填充它们。这里，我们将删除丢失数据超过 50%的列，因为这些特性不会给模型增加太多。
3.  检查缺少值的行，并决定是删除还是填充它们(不适用于示例数据)。
4.  将罕见的分类值(例如，份额小于 10%)放入一个“其他”桶中，以防止我们的模型过度适应这些特定事件。
5.  将分类数据编码到[一次性](https://en.wikipedia.org/wiki/One-hot)虚拟变量中，因为我们将使用的模型需要数字输入。有各种方法对分类数据进行编码，这篇[文章](/smarter-ways-to-encode-categorical-data-for-machine-learning-part-1-of-3-6dca2f71b159)提供了一个很好的概述，以防你有兴趣了解更多。
6.  指定因变量向量和自变量矩阵。
7.  将数据集分为训练集和测试集，以便在训练后正确评估模型的拟合度。
8.  根据我们将要构建的一个模型的需要缩放特征。

下面是完整的预处理代码:

# 训练你的模特

最后，我们可以继续构建和训练多个回归变量，以最终预测结果(因变量的值)，即所讨论的营销活动的印象数。我们将尝试四种不同的监督学习技术——线性回归、决策树、随机森林(决策树)和支持向量回归——并将使用 [Scikit-learn](https://scikit-learn.org) 库提供的相应类来实现这些技术，该库已经用于在预处理期间缩放和拆分数据。

我们可以使用更多的模型来开发回归器，例如[人工神经网络](https://en.wikipedia.org/wiki/Artificial_neural_network)，它可能会产生更好的预测器。然而，本文的重点是以直观和可解释的方式解释这种回归的一些核心原则，而不是产生最准确的预测。

## **线性回归**

![](img/98e7aba485659fe826c81dd508b5da8e.png)

[https://towardsdatascience.com/introduction-to-linear-regression-in-python-c12a072bedf0](/introduction-to-linear-regression-in-python-c12a072bedf0)

使用 Scikit-learn 构建线性回归器非常简单，只需要两行代码，从 Scikit 的线性模型类中导入所需的函数并将其赋给一个变量:

```
from sklearn.linear_model import LinearRegression
linear_regressor = LinearRegression(fit_intercept=True, normalize=False, copy_X=True)
```

我们希望保留默认参数，因为我们需要计算截距(当所有特征都为 0 时的结果),并且我们不需要偏好可解释性的标准化。回归器将通过最小化误差平方和(即预测结果与真实结果的偏差)来计算自变量系数和截距，这被称为[普通最小二乘法](https://en.wikipedia.org/wiki/Ordinary_least_squares)。

我们还可以输出系数及其各自的 p 值，输出的概率独立于(此处也不相关于)特定特征(这将是系数等于 0 的零假设)，因此是统计显著性的度量(越低越显著)。

在我们之前的“第一眼”中已经可视化了数字特征之间的相关性，我们期望特征“预算”、“天数”和“facebook”携带相对较小的 p 值，其中“预算”和“天数”具有正系数，“facebook”具有负系数。 [statsmodels](https://www.statsmodels.org/stable/index.html) 模块提供了一种输出这些数据的简单方法:

```
model = sm.OLS(self.y_train, sm.add_constant(self.X_train)).fit()
print(model.summary())
```

![](img/4aefb653d079f3697251c6ea35f23a02.png)

这里的 p 值是使用基于 [t 分布](https://en.wikipedia.org/wiki/Student%27s_t-distribution)的 t 统计或得分计算的。该摘要还为我们提供了整个模型的准确性或拟合优度的第一个提示，通过测定输入变量解释的输出中方差份额的决定系数 R 平方进行评分，此处为 54.6%。

然而，为了比较所有模型并适应我们的特殊挑战，我们将使用一种不同的评分方法，我称之为“平均相对准确度”，定义为 1 -平均百分比误差= 1 -平均值(|(预测-真值)/真值|)。如果真值为 0，则该度量明显是未定义的，但是在我们的情况下，这是不相关的，因为我们在预处理步骤中检查该条件(见上文),并且我们将因此获得与准确性的直观定义相匹配的良好可解释性。我们将使用五重交叉验证计算所有模型的得分，随机将数据集拆分五次，并取每个得分的平均值。Scitkit-learn 也为此提供了一种简便的方法:

```
linear_score = np.mean(cross_val_score(estimator=linear_regressor,
                       X=X_train, y=y_train, cv=5,
                       scoring=mean_relative_accuracy))
```

我们获得的线性回归的训练分数是 0.18；因此，我们能够用这个模型产生的最佳拟合结果只有 18%的预测准确度。让我们希望其他模型能够超越这一点。

## **决策树**

![](img/00bf1180cc69869db1555c9f8d31365a.png)

[https://becominghuman.ai/understanding-decision-trees-43032111380f](https://becominghuman.ai/understanding-decision-trees-43032111380f)

接下来是从一个决策树中得到的回归量。这里，我们将使用一个 Scikit-learn 函数，它比线性模型多了几个参数，即所谓的超参数，包括一些我们尚不知道所需设置的参数。这就是为什么我们要引入一个叫做网格搜索的概念。网格搜索也可从 Scikit-learn 获得，当训练预测模型并返回最佳参数(即产生最高分数的参数)时，它允许我们定义要测试的参数网格或矩阵。通过这种方式，我们可以测试[决策树模型](https://scikit-learn.org/stable/modules/generated/sklearn.tree.DecisionTreeRegressor.html)的所有可用参数，但我们将关注其中的两个参数，即衡量将一个分支分成两个分支的质量的“标准”和树的一个叶子(最终节点)的最小样本(数据点)数量。这将有助于我们找到具有训练数据的良好模型，同时限制过拟合，即不能从训练数据推广到新样本。从现在开始，我们也将为所有的随机计算设置一个等于 1 的随机状态，这样你将会得到相同的编码值。其余的工作类似于我们之前构建的线性回归:

```
tree_parameters = [{'min_samples_leaf': list(range(2, 10, 1)),
                    'criterion': ['mae', 'mse'],
                    'random_state': [1]}]
tree_grid = GridSearchCV(estimator=DecisionTreeRegressor(),
                         param_grid=tree_parameters,
                         scoring=mean_relative_accuracy, cv=5,
                         n_jobs=-1, iid=False)
tree_grid_result = tree_grid.fit(X_train, y_train)
best_tree_parameters = tree_grid_result.best_params_
tree_score = tree_grid_result.best_score_
```

从我们定义的网格中选择的最佳参数包括均方误差，作为确定每个节点最佳分割的标准，以及每个叶片的最少九个样本，产生 67%的平均相对(训练)准确度，这已经比线性回归的 18%好得多。

决策树的一个优点是我们可以很容易地形象化和直观地理解模型。使用 Scikit-learn 和两行代码，您可以生成拟合决策树的[点表示](https://en.wikipedia.org/wiki/DOT_(graph_description_language))，然后您可以将其转换为 PNG 图像:

```
from sklearn.tree import export_graphviz
export_graphviz(regressor, out_file='tree.dot', 
                feature_names=X_train.columns)
```

![](img/462fc791c642f68be203ddc28374e3c4.png)

正如您所看到的，所有 16 个特性中只有 4 个用于构建这个模型:budget、days、category_concert 和 start_month。

## **随机森林**

单一决策树的主要挑战在于在每个节点找到最佳分裂，并过度适应训练数据。当将多个树组合成随机森林集合时，这两种情况都可以得到缓解。这里，森林的树将在数据的不同(随机)子集上被训练，并且树的每个节点将考虑可用特征的(再次随机)子集。

随机森林回归器的构建几乎与决策树一模一样。我们只需要添加树的数量，这里称为估计量，作为一个参数。由于我们不知道最佳数字，我们将在网格搜索中添加另一个元素来确定最佳回归量:

```
forest_parameters = [{'n_estimators': helpers.powerlist(10, 2, 4),
                      'min_samples_leaf': list(range(2, 10, 1)),
                      'criterion': ['mae', 'mse'],
                      'random_state': [1], 'n_jobs': [-1]}]
forest_grid = GridSearchCV(estimator=RandomForestRegressor(),
                           param_grid=forest_parameters,
                           scoring=mean_relative_accuracy, cv=5,
                           n_jobs=-1, iid=False)
forest_grid_result = forest_grid.fit(X_train, y_train)
best_forest_parameters = forest_grid_result.best_params_
forest_score = forest_grid_result.best_score_
```

根据我们定义的网格搜索，森林模型的最佳参数包括平均绝对误差标准、3 个最小叶子样本大小和 80 个估计量(树)。与单个决策树相比，通过这些设置，我们可以再次将训练准确率提高到 70%。

## **支持向量回归机**

我们要构建的最后一个回归变量基于[支持向量机](https://en.wikipedia.org/wiki/Support-vector_machine)，这是一个由 [Vladimir Vapnik](https://en.wikipedia.org/wiki/Vladimir_Vapnik) 在 20 世纪 60 年代到 90 年代开发的美丽的数学概念。不幸的是，解释它们的内部工作超出了本文的范围。尽管如此，我还是强烈建议去看看；一个很好的入门资源是温斯顿教授在麻省理工学院的演讲。

一个非常基本的总结:支持向量回归机试图将给定样本拟合到由线性边界定义的直径的多维(按照特征数量的顺序)超平面中，同时最小化误差或成本。

尽管这种类型的模型与决策树和森林有着本质的不同，但 Scikit-learn 的实现是相似的:

```
svr_parameters = [{'kernel': ['linear', 'rbf'],
                   'C': helpers.powerlist(0.1, 2, 10),
                   'epsilon': helpers.powerlist(0.01, 2, 10),
                   'gamma': ['scale']},
                  {'kernel': ['poly'],
                   'degree': list(range(2, 5, 1)),
                   'C': helpers.powerlist(0.1, 2, 10),
                   'epsilon': helpers.powerlist(0.01, 2, 10),
                   'gamma': ['scale']}]
svr_grid = GridSearchCV(estimator=SVR(),
                        param_grid=svr_parameters,
                        scoring=mean_relative_accuracy, cv=5,
                        n_jobs=-1, iid=False)
svr_grid_result = svr_grid.fit(X_train_scaled, y_train_scaled)
best_svr_parameters = svr_grid_result.best_params_
svr_score = svr_grid_result.best_score_
```

我们可以再次使用网格搜索来找到一些模型参数的最佳值。这里最重要的是将样本变换到更高维度的特征空间的核，在该特征空间中，数据可以被线性分离或近似，即通过上述超平面。我们正在测试一个线性核，一个多项式核和一个径向基函数。ε为 0.08，即预测与真实值的最大(缩放)距离，其中没有与其相关联的错误，并且惩罚参数 C 为 12.8，线性核表现最佳，达到 23%的(缩放)训练精度。

# **评估您的模型**

在我们根据手头的训练数据确定了模型的最佳参数后，我们可以使用这些参数来最终预测测试集的结果，并计算它们各自的测试精度。首先，我们需要用训练数据的期望超参数来拟合我们的模型。这一次，我们不再需要交叉验证，并将使模型适合完整的训练集。然后，我们可以使用拟合回归来预测训练和测试集结果，并计算它们的准确性。

```
training_accuracies = {}
test_accuracies = {}
for regressor in regressors:
    if 'SVR' in str(regressor):
        regressor.fit(X_train_scaled, y_train_scaled)
        training_accuracies[regressor] = hel.mean_relative_accuracy(
            y_scaler.inverse_transform(regressor.predict(
                X_train_scaled)), y_train)
        test_accuracies[regressor] = hel.mean_relative_accuracy(
            y_scaler.inverse_transform(regressor.predict(
                X_test_scaled)), y_test)
    else:
        regressor.fit(X_train, y_train)
        training_accuracies[regressor] = hel.mean_relative_accuracy(
            regressor.predict(X_train), y_train)
        test_accuracies[regressor] = hel.mean_relative_accuracy(
            regressor.predict(X_test), y_test)
```

结果如下:

训练精度:线性 0.34，树 0.67，森林 0.75，SVR 0.63
测试精度:线性 0.32，树 0.64，森林 0.66，SVR 0.61

我们最好的回归器是随机森林，最高测试精度为 66%。这似乎有点过了，因为它与训练精度的偏差相对较大。随意试验超参数的其他值，以进一步改进所有模型。

在最终保存我们的模型以对新数据进行预测之前，我们将使它适应所有可用的数据(训练和测试集)，以尽可能多地纳入信息。

# **预测结果**

现在我们有了一个模型，可以预测未来营销活动的结果。我们只需对它调用 predict 方法，传递一个特定的特征向量，并将收到我们训练回归元的度量的相应预测。我们还可以将现有数据集的真实印象与基于新模型的预测进行比较:

![](img/c9dcbda1e242aaf416a2c6f40d2f6458.png)

预测值与实际值的平均相对偏差为 26%，因此我们达到了 74%的准确率。只有 14%的中值偏差甚至更小。

# 结论

我们能够构建和训练回归器，使我们能够根据历史营销活动数据预测未来营销活动的印象数(以及其他模拟方式的绩效指标)。

> 随机森林模型的预测精度最高。

我们现在可以使用这些预测来评估一个新的营销活动，甚至在它开始之前。此外，这使我们能够确定最佳参数，包括我们活动的时间表和预算规模，因为我们可以用这些特性的不同值来计算预测。

# **奖励:现成的训练模型**

你手头还没有数据来为你计划的数字营销活动建立一个准确的预测模型？别担心:我已经训练了多个模型，用 1000 多个活动的数据来预测印象、点击和购买。通过不同模型的组合，这些预测的准确率高达 90%。在 predictor.stagelink.com[](https://predictor.stagelink.com)**你会发现一个简单的应用程序，只需几个输入就能预测你未来活动的结果。这些模型主要根据推广活动门票销售的数字营销活动的数据进行训练，因此这可能是它们表现最好的地方。**

**![](img/e8c93967cb2d249c462d6cb5a7f1d502.png)**

**predictor.stagelink.com**

**除此之外，你可以在我的 Github 上找到所有用于讨论营销业绩预测的代码:[**github.com/kinosal/predictor**](https://github.com/kinosal/predictor)**

**感谢您的阅读-我期待您的任何反馈！**