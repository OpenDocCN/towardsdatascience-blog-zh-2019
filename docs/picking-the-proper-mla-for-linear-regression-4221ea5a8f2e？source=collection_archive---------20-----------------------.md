# 为线性回归选择合适的 MLA

> 原文：<https://towardsdatascience.com/picking-the-proper-mla-for-linear-regression-4221ea5a8f2e?source=collection_archive---------20----------------------->

## 普通最小二乘法失败的地方，决策树成功了

![](img/86e25c292677d59c1cfefb1ca8c7bbf7.png)

[Decision Tree Regression](https://www.google.com/url?sa=i&source=images&cd=&ved=2ahUKEwjX0P7rm-PkAhWjTt8KHYaKDAMQjRx6BAgBEAQ&url=https%3A%2F%2Fdocumentation.sas.com%2F%3FdocsetId%3Dstathpug%26docsetTarget%3Dstathpug_hpsplit_examples03.htm%26docsetVersion%3D15.1%26locale%3Den&psig=AOvVaw1ZfxcplwgyLeiTv7BDFCzk&ust=1569200040973408)

在我的[上一篇文章](https://medium.com/@imamun/linear-regression-of-selected-features-132cc6c4b600)中，我谈到了选择特定的特征，并在普通的最小二乘公式中使用它们来帮助预测分子相互作用属性的标量耦合常数。我用一个相对较差的预测算法结束了它，并决定这是由于由我所有的 EDA 计算确定的不重要的特征。我将张贴我以前的图片以及只有重要特征的输出视觉效果。

![](img/6885088d05ad513692cf126b5d25ba60.png)![](img/6fda3d96461749fae4d8efaac6497f16.png)

你能看出区别吗？他们使用两种不同的功能。第一个包含权重为 150 或以上的所有特性，而第二个包含权重为 1500 或以上的特性。那时我意识到这只是一个糟糕的工作模式。

鉴于我之前的特征重要性是不正确的，我决定研究一个模型，该模型考虑了多个特征，并且通过主成分分析只保留重要的特征。这让我想到了决策树回归算法。现在，当大多数人谈论决策树时，他们指的是分类，但由于我试图寻找一个连续变量，它必须是一个回归。

首先，我加载了使用 pandas 从大规模数据库中创建的子样本 csv。然后，我使用下面的代码检查所有类别并删除它们。请记住，这些类别很少或没有重要性，并进行太多稀疏计数，它是有效的。

```
df= pd.read_csv('molecule_subsample.csv')for col in df.select_dtypes(include=[object]):
    print(df[col].value_counts(dropna=False), "\n\n")df= df.drop(columns=['molecule_name', 'atom_atom1_structure', 'type', 'type_scc', 'atom'])
```

现在我有了一个包含我想要的特性的合适的数据框架。我还决定将它保存为 csv 文件，以便将来可以更快地导入它:

> df.to_csv('subsample_nocat.csv '，index=False)

然后我创建特征和目标变量，然后进行训练测试分割。我选择了较小的训练规模，因为即使这个子样本也包含了将近 50，000 个数据集。

```
feature= df.drop(columns=['scalar_coupling_constant'])
target= df[['scalar_coupling_constant']]feature_train, feature_test, target_train, target_test= train_test_split(feature, target, test_size=0.1)total feature training features:  419233
total feature testing features:  46582
total target training features:  419233
total target testing features:  46582
```

之后，就相对简单了。我加载了决策树回归器，并用我想要的标准填充它。知道我有一个相对较大的数据集，我选择了一个相当大的最大深度

```
DTR= tree.DecisionTreeRegressor(max_depth=75, min_samples_split=3, min_samples_leaf=5, random_state=1)DR= DTR.fit(feature_train, target_train)DR:
DecisionTreeRegressor(criterion='mse', max_depth=75, max_features=None,
                      max_leaf_nodes=None, min_impurity_decrease=0.0,
                      min_impurity_split=None, min_samples_leaf=5,
                      min_samples_split=3, min_weight_fraction_leaf=0.0,
                      presort=False, random_state=1, splitter='best')
```

我试图将它可视化，但我的电脑一直在崩溃，所以我决定采用另一种方法，对我的决策树进行交叉验证。我基本上重做了上面的代码，但考虑到了交叉验证:

```
drcv= DR.fit(feature_train, target_train)drcv_scores =cv(drcv, feature_train, target_train, cv = 10)drcv_scores:
{'fit_time': array([20.96234488, 20.751436  , 20.6993022 , 20.62980795, 20.80624795,
        20.72991371, 20.73874903, 20.65243793, 20.55556297, 20.36065102]),
 'score_time': array([0.03302193, 0.0274229 , 0.02751803, 0.02114892, 0.02561307,
        0.02700615, 0.02410102, 0.02259707, 0.02510405, 0.02420998]),
 'test_score': array([0.99999431, 0.99998765, 0.99999402, 0.99999096, 0.99999444,
        0.99999466, 0.99998819, 0.99999362, 0.99999481, 0.99998841])}print("regression score: {}".format(drcv.score(feature_train, target_train)))regression score: 0.9999964614281138
```

看看结果。它有 99%的预测是正确的。这只能意味着一件事:数据被过度拟合。不管怎样，我都愿意看透这一切。我知道我可以通过降低我的最大深度来调整这种过度拟合，并在稍后更改我的树最小节点和树叶，以及实现随机森林集合方法。

我再次运行了上面的代码，但是做了一点小小的改动:

> cv= cross_validate(DR，feature_train，target_train，n_jobs=-1，return_train_score=True)

我还想查看我的训练成绩状态:

```
{'fit_time': array([19.88309979, 19.68618298, 19.56496   ]),
 'score_time': array([0.06965423, 0.08991718, 0.07562518]),
 'test_score': array([0.99999126, 0.99998605, 0.99999297]),
 'train_score': array([0.99999497, 0.99999486, 0.99999842])}
```

这表明我的训练和测试几乎没有任何区别。即使在过度配合的情况下，也不应该发生这种情况。我认为这可能是由于发生了一些代码泄漏，或者我在编写变量时犯了某种形式的错误。所以我决定再看看几个结果:

```
DTR.score(feature_test, target_test)
0.9999870784300612DTR.score(feature_train,target_train)
0.9999964614281138
```

这显示了两个不同的数字，虽然非常接近。因此，虽然我在名字上没有犯任何错误，但似乎因为我如何建立我的树，一切都被过度拟合了。然后我决定检查最后两个指标。现在，由于它们是两种不同的数据格式，我必须先将其中一种更改为 numpy 数组，然后绘制:

```
predict=DTR.predict(feature_test)
type(predict):
numpy.ndarraytt_np= target_test.to_numpy()
type(tt_np):
numpy.ndarrayplt.rcParams["figure.figsize"] = (8, 8)
fig, ax = plt.subplots()
ax.scatter(predict, tt_np)
ax.set(title="Predict vs Actual")
ax.set(xlabel="Actual", ylabel="Predict");
```

![](img/52a5ea8760617aad437ad7315e1d161b.png)

Look at how beautiful it is!

哇，我没想到会这样。虽然我的预测错过了一些，但看起来它几乎把所有事情都做对了。这让我得出结论，这个机器学习模型也是不正确的，但我会在尝试随机森林后获得更好的洞察力。