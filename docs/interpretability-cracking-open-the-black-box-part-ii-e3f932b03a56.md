# 可解释性:打开黑匣子——第二部分

> 原文：<https://towardsdatascience.com/interpretability-cracking-open-the-black-box-part-ii-e3f932b03a56?source=collection_archive---------35----------------------->

## 杂质的平均减少，排列的重要性？LOOC 重要性、PDP 图等

![](img/1db841a2fbd9f5475f284421b6fc5fe5.png)

付费墙是否困扰着你？ [*点击这里*](/interpretability-cracking-open-the-black-box-part-ii-e3f932b03a56?source=friends_link&sk=62f9d9b76ae4f3960194ed434d018c42) *可以绕过去。*

在本系列的[最后一篇文章](https://deep-and-shallow.com/2019/11/13/interpretability-cracking-open-the-black-box-part-i/)中，我们定义了什么是可解释性，并查看了几个可解释性模型以及其中的怪癖和“陷阱”。现在，让我们更深入地研究事后解释技术，当您的模型本身不透明时，这种技术很有用。这与大多数真实世界的用例产生了共鸣，因为无论我们喜欢与否，我们都可以通过黑盒模型获得更好的性能。

# 数据集

在本练习中，我选择了[成人数据集，也称为人口普查收入](https://archive.ics.uci.edu/ml/datasets/adult)数据集。**人口普查收入**是一个非常受欢迎的数据集，它有人口统计信息，如年龄、职业，还有一列告诉我们特定人的收入是否为 50k。我们使用该列来运行使用随机森林的二元分类。选择随机森林有两个原因:

1.  随机森林和梯度增强树是最常用的算法之一。这两种算法都来自于决策树集成算法家族。
2.  我想讨论一些特定于基于树的模型的技术。

![](img/3c487c262f99fa82196b10cb986966ec.png)![](img/c6203e7ef85fd4209fc8a54b8d4c830d.png)

# 事后口译

现在，让我们看看做事后解释的技术，以理解我们的黑盒模型。在博客的其余部分，讨论将基于机器学习模型(而不是深度学习)，并将基于结构化数据。虽然这里的许多方法都是模型不可知的，但由于有许多特定的方法来解释深度学习模型，特别是在非结构化数据上，我们将这一点置于我们当前的范围之外。(可能是另一篇博客，改天。)

数据预处理

*   将目标变量编码成数字变量
*   处理缺失值
*   通过组合几个值，将*婚姻状况*转换为二元变量
*   放弃*教育*，因为*教育 _ 数字*给出了相同的信息，但以数字格式表示
*   放弃*资本收益*和*资本损失*，因为他们没有任何信息。90%以上都是零
*   放弃 *native_country* 是因为高基数和偏向我们
*   因为与婚姻状况有很多重叠，所以放弃了关系

对随机森林算法进行了调整，并对数据进行了训练，取得了 83.58%的性能。考虑到根据你建模和测试的方式，最好的分数从 78-86%不等，这是一个不错的分数。但是对于我们的目的来说，模型性能已经足够了。

# 1.杂质的平均减少

这是迄今为止最流行的解释基于树的模型及其集合的方式。这很大程度上是因为 Sci-Kit Learn 及其简单易用的实现。拟合随机森林或梯度推进模型并绘制“要素重要性”已成为数据科学家中使用和滥用最多的技术。

当在任何集合决策树方法(随机森林、梯度推进等)中创建决策树时，通过测量特征在减少不确定性(分类器)或方差(回归器)方面的有效性来计算特征的杂质重要性的平均减少。).

该技术的**优点**是:

*   获取特征重要性的快速简单的方法
*   在 Sci-kit 学习和 R 中决策树实现中很容易获得
*   向外行人解释是非常直观的

## 算法

*   在构建树的过程中，无论何时进行分割，我们都会跟踪哪个特征进行了分割，分割前后的基尼系数是多少，以及它影响了多少样本
*   在树构建过程的最后，您可以计算归因于每个特征的基尼系数的总增益
*   在随机森林或梯度增强树的情况下，我们对集合中的所有树平均这个分数

## 履行

默认情况下，Sci-kit Learn 在基于树的模型的“特征重要性”中实现这一点。因此检索它们并绘制出前 25 个特征是非常简单的。

```
feat_imp = pd.DataFrame({'features': X_train.columns.tolist(), "mean_decrease_impurity": rf.feature_importances_}).sort_values('mean_decrease_impurity', ascending=False) feat_imp = feat_imp.head(25) feat_imp.iplot(kind='bar', y='mean_decrease_impurity', x='features', yTitle='Mean Decrease Impurity', xTitle='Features', title='Mean Decrease Impurity', )
```

![](img/f4814c219ed427b293ebffd30deae4c0.png)

[Click here for full interactive plot](https://chart-studio.plot.ly/~manujosephv/25)

我们还可以检索并绘制每种树的杂质平均减少量的箱线图。

```
# get the feature importances from each tree and then visualize the distributions as boxplots all_feat_imp_df = pd.DataFrame(data=[tree.feature_importances_ for tree in rf], columns=X_train.columns) order_column = all_feat_imp_df.mean(axis=0).sort_values(ascending=False).index.tolist() 
all_feat_imp_df[order_column[:25]].iplot(kind=’box’, xTitle = 'Features’, yTitle=’Mean Decease Impurity’)
```

![](img/96719aba0a0921ab0c506e9a9b1e7f31.png)

[Click for full interactive plot](https://chart-studio.plot.ly/~manujosephv/27)

## 解释

*   前 4 个特征是*婚姻状况、教育程度、年龄、*和*工作时间*。这很有道理，因为它们与你的收入有很大关系
*   注意到 fnlwgt 和 random 这两个特性了吗？它们比一个人的职业更重要吗？
*   这里的另一个警告是，我们将热门特征视为独立的特征，这可能与职业特征的排名低于随机特征有一定关系。当考虑特性重要性时，处理热点特性是另外一个话题

我们来看看 *fnlwgt* 和*随机*是什么。

*   *fnlwgt* 数据集的描述是一个冗长而复杂的描述，描述了人口普查机构如何使用抽样来创建任何特定人口社会经济特征的“加权记录”。简而言之，这是一个抽样体重，与一个人的收入无关
*   而*随机*也就顾名思义。在拟合模型之前，我用随机数做了一列，称之为 random

当然，这些特征不可能比职业、工作阶级、性别等其他特征更重要。如果是这样的话，那就有问题了。

## 包里的小丑，又名“抓到你了”

当然…有。杂质测量的平均减少是特征重要性的**偏差测量。它偏爱连续的特性和高基数的特性。2007 年，施特罗布尔*等人* [1]也在[的《随机森林变量重要性度量的偏差:图解、来源和解决方案](https://link.springer.com/article/10.1186%2F1471-2105-8-25)中指出“*布雷曼的原始随机森林方法的变量重要性度量……在潜在预测变量的度量尺度或类别数量*发生变化的情况下是不可靠的。”**

我们来试着理解一下为什么会有偏差。还记得杂质平均减少量是如何计算的吗？每次在要素上分割一个结点时，都会记录基尼系数的下降。并且当一个特征是连续的或者具有高基数时，该特征可能比其他特征被分割更多次。这扩大了该特定特征的贡献。我们这两个罪魁祸首的特征有什么共同点——它们都是连续变量。

# 2.删除列重要性，也称为遗漏一个协变量(LOOC)

Drop Column feature importance 是查看特性重要性的另一种直观方式。顾名思义，这是一种迭代删除特性并计算性能差异的方法。

该技术的优点是:

*   给出了每个特征的预测能力的相当准确的图像
*   查看特性重要性的最直观的方法之一
*   模型不可知。可适用于任何型号
*   它的计算方式，会自动考虑模型中的所有相互作用。如果要素中的信息被破坏，它的所有交互也会被破坏

## 算法

*   使用您训练过的模型参数，在 OOB 样本上计算您选择的指标。您可以使用交叉验证来获得分数。这是你的底线。
*   现在，从您的训练集中一次删除一列，重新训练模型(使用**相同的参数**和**随机状态**)并计算 OOB 分数。
*   *重要性= OOB 得分—基线*

## 履行

```
def dropcol_importances(rf, X_train, y_train, cv = 3): 
    rf_ = clone(rf) 
    rf_.random_state = 42 
    baseline = cross_val_score(rf_, X_train, y_train,     scoring='accuracy', cv=cv) 
    imp = [] 
    for col in X_train.columns: 
        X = X_train.drop(col, axis=1) 
        rf_ = clone(rf) 
        rf_.random_state = 42 
        oob = cross_val_score(rf_, X, y_train, scoring='accuracy', cv=cv)
        imp.append(baseline - oob) 
    imp = np.array(imp) 
    importance = pd.DataFrame( imp, index=X_train.columns)     importance.columns = ["cv_{}".format(i) for i in range(cv)] 
    return importance
```

让我们做一个 50 倍交叉验证来估计我们的 OOB 分数。(我知道这是多余的，但让我们保留它，以增加我们箱线图的样本)像以前一样，我们绘制准确性的平均下降以及箱线图，以了解跨交叉验证试验的分布。

```
drop_col_imp = dropcol_importances(rf, X_train, y_train, cv=50) drop_col_importance = pd.DataFrame({’features’: X_train.columns.tolist(), "drop_col_importance": drop_col_imp.mean(axis=1).values}).sort_values(’drop_col_importance’, ascending=False) drop_col_importance = drop_col_importance.head(25) drop_col_importance.iplot(kind=’bar’, y=’drop_col_importance’, x=’features’, yTitle=’Drop Column Importance’, xTitle=’Features’, title=’Drop Column Importances’, ) all_feat_imp_df = drop_col_imp.T order_column = all_feat_imp_df.mean(axis=0).sort_values(ascending=False).index.tolist() 
all_feat_imp_df[order_column[:25]].iplot(kind=’box’, xTitle = 'Features’, yTitle=’Drop Column Importance’)
```

![](img/ca42047d8530eb539728f471afe5f93f.png)

[Click for full interactive plot](https://plot.ly/~manujosephv/33/)

![](img/36b2cc3586ec83455e39f520db339136.png)

[Click for full interactive plot](https://plot.ly/~manujosephv/35/)

## 解释

*   排名前 4 位的特征仍然是*婚姻状况、教育程度、年龄、*和*工作时间*。
*   fnlwgt 在列表中被向下推，现在出现在一些热门编码职业之后。
*   *random* 仍然占据着很高的排名，将自己定位在*工作时间*之后

正如所料， *fnlwgt* 远没有杂质重要性的平均降低让我们相信的那么重要。*随机*的高位置让我有点困惑，我重新运行了重要性计算，将所有独一无二的特性视为一个。即删除所有职业列并检查职业的预测能力。当我这么做的时候，我可以[看到*随机*和 *fnlwgt* 排名低于*职业*和*工作类*](https://plot.ly/~manujosephv/37/) 。冒着让帖子变得更大的风险，让我们改天再调查吧。

那么，我们有完美的解决方案吗？这些结果与杂质的平均减少一致，它们有一致的意义，并且它们可以应用于任何模型。

## 群里的小丑

这里的难点在于所涉及的**计算**。为了执行这种重要性计算，您必须多次训练模型，针对您拥有的每个功能训练一次，并针对您想要进行的交叉验证循环次数重复训练。即使你有一个训练时间不到一分钟的模型，当你有更多的特征时，计算时间也会激增。为了给你一个概念，我花了 **2 小时 44 分钟**来计算 36 个特性和 50 个交叉验证循环的特性重要性(当然，这可以通过并行处理来改进，但你要明白这一点)。如果你有一个需要两天时间训练的大型模型，那么你可以忘记这个技巧。

我对这种方法的另一个担心是，因为我们每次都用新的特性集来重新训练模型，所以我们没有进行公平的比较。我们删除一列并再次训练模型，如果可以，它会找到另一种方法来获得相同的信息，当存在共线特征时，这种情况会放大。因此，我们在调查时混合了两种东西——功能的预测能力和模型自我配置的方式。

# 3.排列重要性

置换特征重要性被定义为当单个特征值被随机打乱时模型分数的减少[2]。这项技术衡量的是置换或打乱特征向量时的性能差异。关键的想法是，一个特性是重要的，如果这个特性被打乱，模型性能就会下降。

该技术的**优点**是:

*   非常直观。如果一个特性中的信息被打乱，性能会下降多少？
*   模型不可知。尽管该方法最初是由 Brieman 为 Random Forest 开发的，但它很快就适应了模型不可知的框架
*   它的计算方式，会自动考虑模型中的所有相互作用。如果要素中的信息被破坏，它的所有交互也会被破坏
*   该模型不需要重新训练，因此我们节省了计算

## 算法

*   使用度量、训练模型、特征矩阵和目标向量来计算基线得分
*   对于特征矩阵中的每个特征，制作特征矩阵的副本。
*   打乱特征列，将其传递给训练好的模型以获得预测，并使用度量来计算性能。
*   重要性=基线-得分
*   重复 N 次以获得统计稳定性，并在所有试验中取平均重要性

## 履行

置换重要性至少在 python 的三个库中实现——[Eli 5](https://eli5.readthedocs.io/en/latest/autodocs/sklearn.html#module-eli5.sklearn.permutation_importance)、 [mlxtend](http://rasbt.github.io/mlxtend/user_guide/evaluate/feature_importance_permutation/) ，以及 Sci-kit Learn 的一个[开发分支](https://scikit-learn.org/dev/modules/generated/sklearn.inspection.permutation_importance.html#sklearn.inspection.permutation_importance)。我选择 mlxtend 版本完全是为了方便。根据施特罗布尔*等人*【3】，“原始【排列】重要性……具有更好的统计特性”与通过除以标准偏差来归一化重要性值相反。我已经检查了 mlxtend 和 Sci-kit Learn 的源代码，它们没有正常化它们。

```
from mlxtend.evaluate import feature_importance_permutation #This takes sometime. You can reduce this number to make the process faster num_rounds = 50 
imp_vals, all_trials = feature_importance_permutation( predict_method=rf.predict, X=X_test.values, y=y_test.values, metric='accuracy', num_rounds=num_rounds, seed=1)permutation_importance = pd.DataFrame({’features’: X_train.columns.tolist(), "permutation_importance": imp_vals}).sort_values(’permutation_importance’, ascending=False) permutation_importance = permutation_importance.head(25) permutation_importance.iplot(kind=’bar’, y=’permutation_importance’, x=’features’, yTitle=’Permutation Importance’, xTitle=’Features’, title=’Permutation Importances’, )
```

我们还绘制了所有试验的箱线图，以了解偏差。

```
all_feat_imp_df = pd.DataFrame(data=np.transpose(all_trials), columns=X_train.columns, index = range(0,num_rounds)) order_column = all_feat_imp_df.mean(axis=0).sort_values(ascending=False).index.tolist() 
all_feat_imp_df[order_column[:25]].iplot(kind='box', xTitle = 'Features', yTitle='Permutation Importance')
```

![](img/f2ad457f2449d7415d43a13fc7a2a323.png)

[Click for full interactive plot](https://plot.ly/~manujosephv/29/)

![](img/78364b9ca00b3dc47580867f9a735e47.png)

[Click for full interactive plot](https://plot.ly/~manujosephv/31/)

## 解释

*   前四名保持不变，但是前三名(*婚姻状况、教育程度、年龄*)在排列重要性上更加明显
*   *fnlwgt* 和 *random* 甚至没有进入前 25 名
*   成为一名执行经理或专业教授与你是否能挣到 5 万英镑有很大关系
*   总而言之，这与我们对这一过程的心理模型产生了共鸣

功能重要性方面一切都好吗？我们有没有最好的方法来解释模型在预测中使用的特征？

## 包里的小丑

我们从生活中知道，没有什么是完美的，这种技术也是如此。它的致命弱点是特征之间的相关性。就像 drop column importance 一样，这种技术也会受到特性之间的相关性的影响。施特罗布尔*等人*在《随机森林的条件变量重要性》[3]中指出，“*排列重要性高估了相关预测变量的重要性。*“特别是在树的集合中，如果有两个相关变量，一些树可能会选择特征 A，而另一些树可能会选择特征 B。在进行此分析时，在没有特征 A 的情况下，选择特征 B 的树会工作良好并保持高性能，反之亦然。这将导致相关特征 A 和 B 都具有夸大的重要性。

该技术的另一个缺点是该技术的核心思想是置换特征。但这本质上是一种随机性，我们无法控制。正因为如此，结果**可能**变化很大。我们在这里看不到它，但是如果箱线图显示一个特性在不同试验中的重要性有很大的变化，我会在我的解释中保持警惕。

![](img/d5def3c93d77bf4bf61523fb7bcad934.png)

Phi_k [Correlation Coefficient](https://phik.readthedocs.io/en/latest/introduction.html) [7](in-built in pandas profiling which considers categorical variables as well)

这种技术还有另一个缺点，在我看来，这是**最令人担心的**。Giles Hooker 等人[6]说，*“当训练集中的特征表现出统计相关性时，排列方法在应用于原始模型时可能会产生很大的误导。”*

我们来考虑一下*职业*和*学历*。我们可以从两个角度理解这一点:

1.  **逻辑**:你想想，*职业*和*学历*有确定的依赖关系。如果你受过足够的教育，你只能得到几份工作，从统计学上来说，你可以在它们之间画出平行线。因此，如果我们改变这些列中的任何一个，它将创建没有意义的功能组合。一个学历*学历*第十*职业*第*职业*第*Prof-specialty*的人没有太大意义，不是吗？因此，当我们评估模型时，我们评估的是无意义的情况，这些情况混淆了我们用来评估特性重要性的度量标准。
2.  **数学** : *职业*和*学历*有很强的统计相关性(从上面的相关图可以看出)。因此，当我们置换这些特征中的任何一个时，我们是在迫使模型探索高维特征空间中看不见的子空间。这迫使模型进行外推，而这种外推是误差的重要来源。

Giles Hooker 等人[6]提出了结合 LOOC 和置换方法的替代方法，但所有替代方法的计算量都更大，并且没有具有更好统计特性的强有力的理论保证。

## 处理相关特征

在识别高度相关的特征之后，有两种处理相关特征的方法。

1.  将高度相关的变量组合在一起，仅评估该组中的一个特征作为该组的代表
2.  当您置换列时，请在一次试验中置换整组特征。

*注意:第二种方法与我建议的处理一次性变量的方法相同。*

## 旁注(培训或验证)

在讨论删除列重要性和排列重要性时，您应该想到一个问题。我们将测试/验证集传递给计算重要性的方法。为什么不是火车组？

这是应用这些方法的一个灰色地带。这里没有对错之分，因为两者都有支持和反对的理由。在《可解释机器学习》一书中，Christoph Molnar 提出了训练集和验证集都适用的案例。

测试/验证数据的案例是显而易见的。出于同样的原因，我们不能通过训练集中的误差来判断模型，我们也不能根据训练集中的性能来判断特征的重要性(特别是因为重要性与误差有着内在的联系)。

训练数据的情况是反直觉的。但是如果你仔细想想，你会发现我们想要衡量的是模型如何使用这些特性。有什么比训练模型的训练集更好的数据来判断这一点呢？另一个无关紧要的问题是，我们应该在所有可用的数据上训练模型，在这样一个理想的场景中，将没有测试或验证数据来检查性能。在可解释机器学习[5]中，第 5.5.2 节详细讨论了这个问题，甚至用一个过度拟合 SVM 的合成例子。

归根结底，你是想知道模型依靠什么特征来进行预测，还是想知道每个特征对未知数据的预测能力。例如，如果您在特性选择的上下文中评估特性的重要性，在任何情况下都不要使用测试数据(您会过度调整您的特性选择以适应测试数据)

# 4.部分依赖图(PDP)和个体条件期望图(ICE)

到目前为止，我们回顾的所有技术都着眼于不同特性的相对重要性。现在让我们稍微换个方向，看看一些探索特定特征如何与目标变量交互的技术。

部分依赖图和个别条件期望图帮助我们理解特征和目标之间的功能关系。它们是一个给定变量(或多个变量)对结果的边际效应的图形可视化。Friedman(2001)在他的开创性论文*贪婪函数逼近:梯度推进机*【8】中介绍了这一技术。

部分相关图显示平均效应，而 ICE 图显示单个观察值的函数关系。PD 图显示平均效应，而 ICE 图显示效应的分散性或异质性。

这种技术的**优势**是:

*   计算非常直观，易于用通俗的语言解释
*   我们可以理解一个特征或特征组合与目标变量之间的关系。即它是线性的、单调的、指数的等等。
*   它们易于计算和实现
*   它们给出了因果解释，与特征重要性风格解释相反。但我们必须记住的是，模型如何看待世界以及现在的现实世界的因果解释。

## 算法

让我们考虑一个简单的情况，其中我们为单个特征 *x* 绘制 PD 图，具有唯一值{x1，x2，…..xk}。PD 图可构建如下:

*   对于{1，2，3，…的 I 元素。k}
*   复制训练数据并用 x(i)替换原始值 *x*
*   使用已训练的模型为整个训练数据的已修改副本生成预测
*   将对 x(i)的所有预测存储在类似映射的数据结构中

对于 PD 图:

*   计算{1，2，3，…k}的 I 元素的每个 x(i)的平均预测值
*   绘制对{ x { I }，平均值(所有带有 x(i)的预测值)

对于冰图:

*   画出所有的对{x(i)，f(x(i)，其余的特征(n)}其中 n 元素为{1，2，3，…。N}
*   在实践中，我们不是取某个特征的所有可能值，而是为连续变量定义一个区间网格，以节省计算时间。
*   对于分类变量，这个定义也成立，但是我们不会在这里定义一个网格。取而代之的是，我们采用类别中的所有唯一值(或属于分类特征的所有一次性编码变量),并使用相同的方法计算 ICE 和 PD 图。
*   如果过程你还不清楚，建议看看这篇[中帖](/introducing-pdpbox-2aa820afd312)(作者是 PDPbox，一个用于绘制 PD 图的 python 库。

## 履行

我发现在 [PDPbox](https://github.com/SauceCat/PDPbox) 、[滑手](https://github.com/oracle/Skater)和 [Sci-kit Learn](https://scikit-learn.org/stable/modules/partial_dependence.html) 中实现的 PD 剧情。以及 [PDPbox](https://github.com/SauceCat/PDPbox) 、 [pyCEbox](https://github.com/AustinRochford/PyCEbox) 和 [skater](https://github.com/oracle/Skater) 中的冰剧情。在所有这些中，我发现 PDPbox 是最完美的。它们还支持 2 个可变 PDP 图。

```
from pdpbox import pdp, info_plots pdp_age = pdp.pdp_isolate( model=rf, dataset=X_train, model_features=X_train.columns, feature='age' ) #PDP Plot 
fig, axes = pdp.pdp_plot(pdp_age, 'Age', plot_lines=False, center=False, frac_to_plot=0.5, plot_pts_dist=True,x_quantile=True, show_percentile=True) 
#ICE Plot 
fig, axes = pdp.pdp_plot(pdp_age, 'Age', plot_lines=True, center=False, frac_to_plot=0.5, plot_pts_dist=True,x_quantile=True, show_percentile=True)
```

![](img/410e7e1243664bf128f95fc935f01a43.png)![](img/6f1b4170b719ca948524d9ca4fc3b1d1.png)

让我花点时间解释一下情节。在 x 轴上，您可以找到您试图理解的特征的值，即年龄。在 y 轴上你会发现预测。在分类的情况下，它是预测概率，而在回归的情况下，它是实值预测。底部的条形表示不同分位数中训练数据点的分布。这有助于我们判断推论的正确性。点数很少的部分，模型看到的例子很少，解释可能很棘手。PD 图中的单线显示了特征和目标之间的平均函数关系。ICE 图中的所有线条显示了训练数据中的异质性，即训练数据中的所有观察值如何随着不同的年龄值而变化。

## 解释

*   *年龄*与一个人的赚钱能力有很大程度上的单调关系。年龄越大，收入越有可能超过 5 万英镑
*   冰图显示了很大的分散性。但是所有这些都显示了我们在 PD 图中看到的同样的行为
*   训练观察在不同的分位数之间相当平衡。

现在，让我们举一个具有分类特征的例子，比如*教育*。PDPbox 有一个非常好的特性，它允许您传递一个特性列表作为输入，它将计算它们的 PDP，将它们视为分类特性。

```
# All the one-hot variables for the occupation feature occupation_features = ['occupation_ ?', 'occupation_ Adm-clerical', 'occupation_ Armed-Forces', 'occupation_ Craft-repair', 'occupation_ Exec-managerial', 'occupation_ Farming-fishing', 'occupation_ Handlers-cleaners', 'occupation_ Machine-op-inspct', 'occupation_ Other-service', 'occupation_ Priv-house-serv', 'occupation_ Prof-specialty', 'occupation_ Protective-serv', 'occupation_ Sales', 'occupation_ Tech-support', 'occupation_ Transport-moving'] #Notice we are passing the list of features as a list with the feature parameter 
pdp_occupation = pdp.pdp_isolate( model=rf, dataset=X_train, model_features=X_train.columns, feature=occupation_features ) #PDP 
fig, axes = pdp.pdp_plot(pdp_occupation, 'Occupation', center = False, plot_pts_dist=True) 
#Processing the plot for aesthetics 
_ = axes['pdp_ax']['_pdp_ax'].set_xticklabels([col.replace("occupation_","") for col in occupation_features]) axes['pdp_ax']['_pdp_ax'].tick_params(axis='x', rotation=45) 
bounds = axes['pdp_ax']['_count_ax'].get_position().bounds axes['pdp_ax']['_count_ax'].set_position([bounds[0], 0, bounds[2], bounds[3]]) _ = axes['pdp_ax']['_count_ax'].set_xticklabels([])
```

![](img/64ef906888fb673514124ca041dab948.png)

## 解释

*   大多数职业对你的收入影响很小。
*   其中最突出的是行政管理、专业教授和技术支持
*   但是，从分布情况来看，我们知道几乎没有技术支持的培训示例，因此我们对此持保留态度。

多个特征之间的相互作用

理论上可以为任意数量的特征绘制 PD 图，以显示它们的相互作用。但实际上，我们只能做两个人，最多三个人。让我们来看看两个连续特征*年龄*和*教育*之间的交互图(教育和年龄不是真正连续的，但由于缺乏更好的例子，我们选择它们)。

有两种方法可以绘制两个特征之间的 PD 图。这里有三个维度，特征值 1、特征值 2 和目标预测。或者，我们可以绘制一个三维图或一个二维图，第三维用颜色表示。我更喜欢二维图，因为我认为它比三维图以更清晰的方式传达信息，在三维图中，你必须查看三维形状来推断关系。PDPbox 实现了二维交互图，既有等高线图也有网格图。等值线最适用于连续要素，格网最适用于分类要素

```
# Age and Education inter1 = pdp.pdp_interact( model=rf, dataset=X_train, model_features=X_train.columns, features=['age', 'education_num'] ) fig, axes = pdp.pdp_interact_plot( pdp_interact_out=inter1, feature_names=['age', 'education_num'], plot_type='contour', x_quantile=False, plot_pdp=False ) axes['pdp_inter_ax'].set_yticklabels([edu_map.get(col) for col in axes['pdp_inter_ax'].get_yticks()])
```

![](img/699750f01882d4e89fd30eb50c4c1f8c.png)

## 解释

*   尽管我们在观察孤立时观察到了与年龄的单调关系，但现在我们知道这并不普遍。例如，请看第 12 级教育右边的等高线。与一些大学及以上的线相比，这是相当平坦的。它真正表明的是，你获得超过 50k 的概率不仅随着年龄的增长而增加，而且还与你的教育程度有关。随着年龄的增长，大学学位能确保你增加收入的潜力。

这也是一个非常有用的技术来调查你的算法中的偏见(伦理的那种)。假设我们想看看在*性别*维度上的算法偏差。

```
#PDP Sex pdp_sex = pdp.pdp_isolate( model=rf, dataset=X_train, model_features=X_train.columns, feature='sex' ) fig, axes = pdp.pdp_plot(pdp_sex, 'Sex', center=False, plot_pts_dist=True) _ = axes['pdp_ax']['_pdp_ax'].set_xticklabels(sex_le.inverse_transform(axes['pdp_ax']['_pdp_ax'].get_xticks())) # marital_status and sex inter1 = pdp.pdp_interact( model=rf, dataset=X_train, model_features=X_train.columns, features=['marital_status', 'sex'] ) fig, axes = pdp.pdp_interact_plot( pdp_interact_out=inter1, feature_names=['marital_status', 'sex'], plot_type='grid', x_quantile=False, plot_pdp=False ) axes['pdp_inter_ax'].set_xticklabels(marital_le.inverse_transform(axes['pdp_inter_ax'].get_xticks())) axes['pdp_inter_ax'].set_yticklabels(sex_le.inverse_transform(axes['pdp_inter_ax'].get_yticks()))
```

![](img/c094aea5b77023b30baf02f3c81d7031.png)

*   如果我们只看性别的概率分布图，我们会得出这样的结论:不存在真正的基于人的性别的歧视。
*   但是，你只要看看和 marriage _ status 的互动情节就知道了。在左手边(已婚)，两个方块有相同的颜色和价值，但在右手边(单身)有女性和男性的区别
*   我们可以得出结论，单身男性比单身女性更有机会获得超过 50k 的收入。(虽然我不会基于此发动一场反对性别歧视的全面战争，但这绝对会是调查的起点。

## 群里的小丑

这种方法的最大缺陷是假设特征之间的独立性。LOOC 重要性和排列重要性中存在的相同缺陷。适用于 PDP 和 ICE 图。[累积局部效应](https://christophm.github.io/interpretable-ml-book/ale.html)图是这个问题的解决方案。ALE 图通过计算——也是基于特征的条件分布——预测中的**差异而不是平均值**来解决这个问题。

总结一下每种类型的图(PDP，ALE)如何计算某个网格值 v 下某个特性的效果:
**部分相关图**:“让我向你展示当每个数据实例都有那个特性的值 v 时，模型平均预测的结果。我忽略了值 v 是否对所有数据实例都有意义。”
**ALE plots** :“让我向您展示模型预测如何在 v 周围的一个小特征“窗口”中针对该窗口中的数据实例发生变化。”

在 python 环境中，没有好的稳定的 ALE 库。我只发现了一个 [ALEpython](https://github.com/blent-ai/ALEPython) ，它还在开发中。我设法得到了一个年龄的 ALE 图，你可以在下面找到。但是当我尝试交互图时出错了。它也不是为分类特征开发的。

![](img/a42c81b87f13afd0703e7f2418082d6d.png)

这就是我们再次中断的地方，把剩下的东西推到下一篇博文。在下一部分，我们来看看石灰，SHAP，锚，等等。

完整的代码可以在我的 [Github](https://github.com/manujosephv/interpretability_blog) 中找到

**博客系列**

*   [第一部分](https://link.medium.com/M0YZX7zbA1)
*   第二部分
*   第三部分—即将推出

## 参考

1.  施特罗布尔特区，Boulesteix，AL。，Zeileis，a .等,《BMC 生物信息学》( 2007 年)8: 25。[https://doi.org/10.1186/1471-2105-8-25](https://doi.org/10.1186/1471-2105-8-25)
2.  长度布雷曼，“随机森林”，机器学习，45(1)，5–32，2001 年。[https://doi.org/10.1023/A:1010933404324](https://doi.org/10.1023/A:1010933404324)
3.  施特罗布尔，c .，Boulesteix，a .，Kneib，T. *等*随机森林的条件变量重要性。 *BMC 生物信息学* **9，**307(2008)doi:10.1186/1471–2105–9–307
4.  特伦斯·帕尔、凯雷姆·图尔古特鲁、克里斯多夫·西萨尔和杰瑞米·霍华德，“[小心默认随机森林重要性](https://explained.ai/rf-importance/#4)”
5.  Christoph Molnar，“[可解释的机器学习:使黑盒模型可解释的指南](https://christophm.github.io/interpretable-ml-book/)”
6.  Giles Hooker，Lucan Mentch，“请停止置换特征:一种解释和替代方案”， [arXiv:1905.03151](https://arxiv.org/abs/1905.03151) 【统计。我]
7.  米（meter 的缩写））Baak，R. Koopman，H. Snoek，S. Klous，“具有皮尔逊特征的分类变量、序数变量和区间变量之间的新相关系数，[arXiv:1811.11440](https://arxiv.org/abs/1811.11440)【stat .我]
8.  杰罗姆·h·弗里德曼[贪婪函数近似法:一种梯度推进机](https://projecteuclid.org/euclid.aos/1013203451)。安。统计学家。29 (2001 年)，第 5 号，1189-1232。doi:10.1214/aos/1013203451
9.  亚历克斯·戈尔茨坦*等*“窥视黑盒内部:用个体条件期望图可视化统计学习”， [arXiv:1309.6392](https://arxiv.org/abs/1309.6392) 【统计。美联社]

*原载于 2019 年 11 月 16 日*[*http://deep-and-shallow.com*](https://deep-and-shallow.com/2019/11/16/interpretability-cracking-open-the-black-box-part-ii/)*。*