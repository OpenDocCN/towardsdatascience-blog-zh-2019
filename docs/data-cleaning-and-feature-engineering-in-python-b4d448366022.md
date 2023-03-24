# Python 中的数据清理和要素工程

> 原文：<https://towardsdatascience.com/data-cleaning-and-feature-engineering-in-python-b4d448366022?source=collection_archive---------5----------------------->

## 为预测旧金山房价建立更好的机器学习模型

![](img/210775aa6d551c9a92243ce1b18c85e5.png)

房价数据为机器学习提供了很好的介绍。任何买了房子甚至租了公寓的人都很容易理解它的特点:更大的空间，更多的房间，通常会导致更高的价格。

因此，开发一个模型应该很容易——但有时并不容易，不是因为机器学习很难，而是因为数据很乱。此外，在同一城市的不同社区，即使只有一英里远，完全相同的房子也可能有明显不同的价格。处理这种情况的最佳方法是设计数据，以便模型可以更好地处理这种情况。

由于寻找数据可能是机器学习中最困难的问题，我们将使用 Github 上另一个数据科学项目的一个很好的样本集，这是旧金山的一组房价，主要是在过去几年里，从*旧金山纪事报*房屋销售清单中刮出来的。这个数据集可以在这里找到:[https://github . com/Ruichang 123/Regression _ for _ house _ price _ estimation/blob/master/final _ data . CSV](https://github.com/RuiChang123/Regression_for_house_price_estimation/blob/master/final_data.csv)

首先，我们将从文件的本地副本加载数据。

```
**import** **pandas** **as** **pd** housing = pd.read_csv("final_data.csv")
```

现在，让我们来看一下这个数据集的一些图表，这些图表按照最后的销售价格绘制了房间总数。

```
**import** **matplotlib.pyplot** **as** **plt** x = housing['totalrooms']
y = housing['lastsoldprice']
plt.scatter(x,y)
plt.show()
```

![](img/6bb0aac17665bce1a3fd93717533f1d2.png)

右下角的那个点是异常值。它的值如此极端，以至于扭曲了整个图表，以至于我们甚至看不到主数据集的任何变化。这将扭曲任何在这个数据集上训练机器学习算法的尝试。我们需要更仔细地观察这个数据点，并考虑如何处理它。如果我们按照房间总数对数据进行排序，这是我们上面的一个轴，应该会很突出。

```
housing['totalrooms'].sort_values()
```

结果如下:

```
7524        1.0
11223       1.0
3579        1.0
2132        1.0
5453        1.0
2827        1.0
            ...2765       23.0
8288       24.0
9201       24.0
6860       24.0
4802       26.0
8087       26.0
11083      27.0
2601       28.0
2750       28.0
10727      28.0
11175      33.0
8300       94.0
8967     1264.0
Name: totalrooms, Length: 11330, dtype: float64
```

事实上，这个数据点确实很突出。这是列表中的最后一个值，这是一个有 1264 个房间的房子！这是非常可疑的，特别是因为情节显示它有一个相当低的价格。至少它与其他数据大相径庭。显示 94 个房间的前一个值可能也是这种情况。我们可以用下面的命令仔细看看这两栋房子，通过它们的数字标识符把它们拉出来。

首先让我们来看看这座据称有 1264 个房间的房子:

```
df = pd.DataFrame(housing)
df.iloc[[8967]]
```

![](img/90e095b361ac8ded407290a87c5ed8df.png)

这个查询显示出更可疑的东西，即“finishedsqft”字段是*也是* 1264.0 *。*换句话说，这显然只是一个错误，可能是数据输入上的错误——当创建原始数据集时，有人意外地对 finishedsqft 和 totalrooms 使用了相同的值。

现在，让我们看看它前面的值，有 94 个房间:

```
df.iloc[[8300]]
```

![](img/22664df1060453dc11cd6a700420e71c.png)

这个据称有 94.0 个房间的家只有两个卧室和两个浴室！同样，这是一个错误。不清楚这是怎么进来的，但我们可以非常肯定，如果我们去这所房子，它没有 94 个房间，只有两个卧室和两个浴室。我们需要消除这两个数据点，但首先让我们再来看一张完成的图表:

```
x = housing['finishedsqft']
y = housing['lastsoldprice']
plt.scatter(x,y)
plt.show()
```

![](img/1c210176eda7177806d77d836398e34e.png)

右下角还有一个异常值。让我们仔细看看这些数据:

```
housing['finishedsqft'].sort_values()
```

这是结果

```
1618         1.0
3405         1.0
10652        1.0
954          1.0
11136        1.0
5103         1.0
916          1.0
10967        1.0
7383         1.0
1465         1.0
8134       243.0
7300       244.0
...
9650      9699.0
8087     10000.0
2750     10236.0
4997     27275.0
Name: finishedsqft, Length: 11330, dtype: float64
```

第一，出乎意料，有十套房子挂牌 1.0 平方英尺。这显然是错误的。请注意，这些不可能在图表中看到，我们必须查看实际值。此外，上述结果显示最大的房子为 27，275.0 平方英尺。事实证明，这是一栋只有 2.0 间卧室和 2.0 间浴室的房子，尽管它的上市面积为 27，275 平方英尺，所以这几乎肯定是一个错误，或者至少是一个极端的异常值。让我们排除所有这些异常值，再看一下图表。

```
housing = housing.drop([1618])     
housing = housing.drop([3405])     
housing = housing.drop([10652])     
housing = housing.drop([954])     
housing = housing.drop([11136])     
housing = housing.drop([5103])     
housing = housing.drop([916])     
housing = housing.drop([10967])     
housing = housing.drop([7383])     
housing = housing.drop([1465])     
housing = housing.drop([8967])     
housing = housing.drop([8300])     
housing = housing.drop([4997])       
x = housing['finishedsqft']
y = housing['lastsoldprice']
plt.scatter(x,y)
plt.show()
```

![](img/99ff6a938852fada3d68d74f2fdbd429.png)

这看起来好多了。这里仍然可能有一些异常值，如果我们真的想的话，我们可以更仔细地调查它们，但在这个视图中没有任何单个数据点会扭曲图表，可能也没有(我们可以看到的)会扭曲机器学习模型。

现在我们已经清理了数据，我们需要做一些特性工程。这包括将数据集中的值转换为机器学习算法可以使用的数值。

以“lastsolddate”值为例。在当前数据集中，这是一个“mm/dd/yyyy”形式的字符串我们需要将它改为一个数值，这可以通过下面的 Pandas 命令来实现:

```
housing['lastsolddateint'] = pd.to_datetime(housing['lastsolddate'], format='%m/**%d**/%Y').astype('int')
housing['lastsolddateint'] = housing['lastsolddateint']/1000000000
housing = housing[housing['lastsolddateint'].notnull()]
```

现在让我们为我们的数据创建一个检查点，以便我们以后可以引用它。

```
clean_data = housing.copy()
```

此外，还有一些我们不能或不应该使用的字段，因此我们将删除它们。

我更喜欢创建函数来做这种我们可能会反复做的工作，正如我们将在下面看到的，以便在我们尝试不同的假设时简化代码。

出于多种原因，我们删除了 remove_list 中的列。其中一些是我们无法处理的文本值(info，address，z_address，zipcode，zpid)。纬度和经度字段在某种形式上可能是有用的，但是对于这个例子来说，它可能会使事情变得复杂——但是没有理由不在将来尝试它。zestimate 和 zindexvalue 字段实际上是由其他数据科学技术(可能来自 Zillow)生成的，因此使用它们就是欺骗！最后，我们将删除 usecode(例如，house，condo，mobile home ),它可能非常有用，但我们不会在本例中使用它。

```
**def** drop_geog(data, keep = []):     
    remove_list = ['info','address','z_address','longitude','latitude','neighborhood','lastsolddate','zipcode','zpid','usecode', 'zestimate','zindexvalue']
    **for** k **in** keep:
        remove_list.remove(k)
    data = data.drop(remove_list, axis=1)
    data = data.drop(data.columns[data.columns.str.contains('unnamed',case = **False**)],axis = 1)
     **return** data housing = drop_geog(housing)
```

现在我们已经清理了数据，让我们看看一些算法是如何管理使用它的。我们将使用 scikit-learn。

首先，我们需要将数据分成测试集和训练集，再次使用我们以后可以重用的函数。这确保了当我们测试数据时，我们实际上是在用以前从未见过的数据测试模型。

```
**from** **sklearn.model_selection** **import** train_test_split

**def** split_data(data):
    y = data['lastsoldprice']
    X = data.drop('lastsoldprice', axis=1)
    *# Return (X_train, X_test, y_train, y_test)*
    **return** train_test_split(X, y, test_size=0.2, random_state=30)housing_split = split_data(housing)
```

我们先试试线性回归。

```
**import** **sys**
**from** **math** **import** sqrt
**from** **sklearn.metrics** **import** mean_squared_error, mean_absolute_error, r2_score
**from** **sklearn.model_selection** **import** GridSearchCV
**import** **numpy** **as** **np**

**from** **sklearn.linear_model** **import** LinearRegression

**def** train_eval(algorithm, grid_params, X_train, X_test, y_train, y_test):
    regression_model = GridSearchCV(algorithm, grid_params, cv=5, n_jobs=-1, verbose=1)
    regression_model.fit(X_train, y_train)
    y_pred = regression_model.predict(X_test)
    print("R2: **\t**", r2_score(y_test, y_pred))
    print("RMSE: **\t**", sqrt(mean_squared_error(y_test, y_pred)))
    print("MAE: **\t**", mean_absolute_error(y_test, y_pred))
    **return** regression_model

train_eval(LinearRegression(), {}, *housing_split)
```

此`train_eval`函数可用于任何任意的 scikit-learn 算法，用于训练和评估。这是 scikit-learn 的一大好处。函数的第一行包含一组我们想要评估的超参数。在这种情况下，我们传入`{}`,这样我们就可以在模型上使用默认的超参数。该函数的第二行和第三行执行实际工作，拟合模型，然后对其运行预测。然后打印报表显示一些我们可以评估的统计数据。让我们看看我们如何公平。

```
R2: 0.5366066917131977
RMSE: 750678.476479495
MAE: 433245.6519384096
```

第一个得分 R 也称为决定系数，是对模型的一般评估，它显示了可以由特征解释的预测中的变化百分比。一般来说，R 值越高越好。另外两个统计是均方根误差和平均绝对误差。这两个只能与其他模型上相同统计的其他评估相关联地进行评估。话虽如此，一个 0.53 的 R，和其他几十万的统计数据(对于可能价值一两百万的房子来说)并不是很好。我们可以做得更好。

让我们看看其他几个算法的表现。首先，K-最近邻(KNN)。

```
**from** **sklearn.neighbors** **import** KNeighborsRegressor
knn_params = {'n_neighbors' : [1, 5, 10, 20, 30, 50, 75, 100, 200, 500]}
model = train_eval(KNeighborsRegressor(), knn_params, *housing_split)
```

如果线性回归是平庸的，KNN 是可怕的！

```
R2: 0.15060023694456648
RMSE: 1016330.95341843
MAE: 540260.1489399293
```

接下来我们将尝试决策树。

```
**from** **sklearn.tree** **import** DecisionTreeRegressor

tree_params = {}
train_eval(DecisionTreeRegressor(), tree_params, *housing_split)
```

这就更惨了！

```
R2: .09635601667334437
RMSE: 1048281.1237086286
MAE: 479376.222614841
```

最后，我们来看看《随机阿甘正传》。

```
**from** **sklearn** **import** ensemble
**from** **sklearn.ensemble** **import** RandomForestRegressor
**from** **sklearn.datasets** **import** make_regression

forest_params = {'n_estimators': [1000], 'max_depth': [**None**], 'min_samples_split': [2]}
forest = train_eval(RandomForestRegressor(), forest_params, *housing_split)
```

这个好一点，但我们还可以做得更好。

```
R2: 0.6071295620858653
RMSE: 691200.04921061
MAE: 367126.8614028794
```

我们如何改进这些结果？一种选择是尝试其他算法，也有很多，有些会做得更好。但是我们实际上可以通过使用特征工程的数据来微调我们的结果。

![](img/373199e4f509e340537e68e3b69d182a.png)

让我们重新考虑一下数据中的一些特征。邻居是一个有趣的领域。这些值类似于“波特雷罗山”和“南海滩”这些不能简单地排序(从最贵到最便宜的邻域)，或者至少，这样做不一定会产生更好的结果。但是我们都知道，两个不同小区的同一个房子，会有两个不同的价格。所以我们想要这些数据。我们如何使用它？

Python 的 Pandas 库为我们提供了一个简单的工具来创建这些值的“一次性编码”。这将采用“neighborhood”这一列，并为原始“neighborhood”列中的每个值创建一个新列。对于这些新行中的每一行(具有新的列标题名称，如“Portrero Hill”和“South Beach”)，如果一行数据在原始列中具有该邻域的值，则它被设置为 1，否则它被设置为 0。机器学习算法现在可以构建与该邻域相关联的权重，如果数据点在该邻域中(如果该列的值为 1)，则应用该权重，否则不应用该权重(如果该列的值为 0)。

首先，我们需要检索我们的检查点数据，这次保留“neighborhood”字段。

```
housing_cleaned = drop_geog(clean_data.copy(), ['neighborhood'])
```

现在我们可以为“neighborhood”字段创建一个一次性编码。

```
one_hot = pd.get_dummies(housing_cleaned['neighborhood'])
housing_cleaned = housing_cleaned.drop('neighborhood',axis = 1)
```

我们将保留“one_hot”值，稍后再添加它。但首先，我们必须做两件事。我们需要将数据分成训练集和测试集。

```
(X_train, X_test, y_train, y_test) = split_data(housing_cleaned)
```

最后一步，我们需要对数据进行缩放和居中。

```
**from** **sklearn.preprocessing** **import** StandardScalerscaler = StandardScaler()
scaler.fit(X_train)
X_train[X_train.columns] = scaler.transform(X_train[X_train.columns])
X_train = X_train.join(one_hot)
X_test[X_test.columns] = scaler.transform(X_test[X_test.columns])
X_test = X_test.join(one_hot)

housing_split_cleaned = (X_train, X_test, y_train, y_test)
```

让我们稍微解释一下这个步骤。

首先，我们应用 StandardScaler()。对于每列中的所有数据点，此函数通过减去列的平均值并除以列的标准偏差来缩放和居中数据。这将所有数据标准化，使每一列呈正态分布。它还会缩放数据，因为有些字段会在 0 到 10，000 之间变化，如“finishedsqft”，而其他字段只会在 0 到 30 之间变化，如房间数。缩放将把它们都放在相同的比例上，这样一个特性就不会仅仅因为它具有较高的最大值而任意地扮演比其他特性更大的角色。对于一些机器学习算法，正如我们将在下面看到的，这对于获得哪怕是半个像样的结果都是至关重要的。

其次，需要注意的是，我们必须在训练特性 X_train 上“安装”缩放器。也就是说，我们获取训练数据的平均值和标准偏差，用这些值拟合 scaler 对象，然后使用拟合的 scaler 转换训练数据和测试数据。我们不希望*在测试数据上安装*缩放器，因为那样会将测试数据集中的信息泄露给训练好的算法。我们可能最终得到看起来比实际情况更好的结果(因为算法已经在测试数据上进行了训练)或看起来更差的结果(因为测试数据是根据它们自己的数据集进行缩放的，而不是根据测试数据集)。

现在，让我们用新设计的功能重建我们的模型。

```
model = train_eval(LinearRegression(), {}, *housing_split_cleaned)
```

现在，在线性回归下，我们拥有的最简单的算法，结果已经比我们之前看到的任何结果都要好。

```
R2: 0.6328566983301503
RMSE: 668185.25771193
MAE: 371451.9425795053
```

下一个是 KNN。

```
model = train_eval(KNeighborsRegressor(), knn_params, *housing_split_cleaned)
```

这是一个巨大的进步。

```
R2: 0.6938710004544473
RMSE: 610142.5615480896
MAE: 303699.6739399293
```

决策树:

```
model = train_eval(DecisionTreeRegressor(), tree_params,*housing_split_cleaned)
```

还是很糟糕，但比以前好了。

```
R2: 0.39542277744197274
RMSE: 857442.439825675
MAE: 383743.4403710247
```

最后，随机福里斯特。

```
model = train_eval(RandomForestRegressor(), forest_params, *housing_split_cleaned)
```

这又是一个不错的进步。

```
R2: 0.677028227379022
RMSE: 626702.4153226872
MAE: 294772.5044353021
```

从额外的特征工程到尝试额外的算法，可以利用这些数据做更多的事情。但是，从这个简短的教程中得到的教训是，寻求更多的数据或在文献中寻找更好的算法并不总是正确的下一步。首先从一个更简单的算法中获取尽可能多的数据可能更好，这不仅是为了比较，也是因为数据清理可能会在未来带来回报。

最后，尽管它很简单，K-最近邻可以是相当有效的，只要我们用适当的方式对待它。