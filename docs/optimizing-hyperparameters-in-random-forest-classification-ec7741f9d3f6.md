# 随机森林分类中超参数的优化

> 原文：<https://towardsdatascience.com/optimizing-hyperparameters-in-random-forest-classification-ec7741f9d3f6?source=collection_archive---------0----------------------->

## 什么是超参数，如何选择超参数值，以及它们是否值得您花费时间

在本文中，我将使用 scikit-learn 的几个分类和模型选择包，深入研究随机森林分类模型的超参数调优。我将分析来自 [UCI 机器学习库](https://archive.ics.uci.edu/ml/datasets/wine+quality)的葡萄酒质量数据集。出于本文的目的，我将红葡萄酒和白葡萄酒的单个数据集进行了合并，并为两者分配了一个额外的列来区分葡萄酒的颜色，其中 0 代表红葡萄酒，1 代表白葡萄酒。这种分类模式的目的是确定葡萄酒是红葡萄酒还是白葡萄酒。为了优化这个模型以创建最准确的预测，我将只关注超参数调整和选择。

## **什么是超参数？**

通常，超参数是在学习过程开始之前设置的模型参数。不同的模型有不同的可以设置的超参数。对于随机森林分类器，有几个不同的超参数可以调整。在这篇文章中，我将研究以下四个参数:

1.  *n _ estimators*:n _ estimators 参数指定了模型森林中的树的数量。该参数的默认值是 10，这意味着将在随机森林中构建 10 个不同的决策树。

2.*max _ depth*:max _ depth 参数指定每棵树的最大深度。max_depth 的默认值是 None，这意味着每棵树都将扩展，直到每片叶子都是纯的。纯叶子是叶子上的所有数据都来自同一个类。

3.*min _ samples _ split*:min _ samples _ split 参数指定分割内部叶节点所需的最小样本数。此参数的默认值为 2，这意味着内部节点必须至少有两个样本，才能被拆分为更具体的分类。

4.*min _ samples _ leaf:*min _ samples _ leaf 参数指定在叶节点上所需的最小样本数。这个参数的缺省值是 1，这意味着每个叶子必须至少有 1 个它要分类的样本。

关于 RandomForestClassifier()的超参数的更多文档可以在[这里](https://scikit-learn.org/stable/modules/generated/sklearn.ensemble.RandomForestClassifier.html)找到。

## **如何调整超参数？**

当您调用创建模型的函数时，可以手动调整超参数。

```
forest = RandomForestClassifier(random_state = 1, n_estimators = 10, min_samples_split = 1)
```

## **您如何选择要调整的超参数？**

在开始调整超参数之前，我对我的数据进行了 80/20 的训练/测试分割。将在训练集上测试不同的超参数，并且一旦选择了优化的参数值，将使用所选择的参数和测试集来构建模型，然后将在训练集上测试，以查看该模型能够多准确地对酒的类型进行分类。

```
forest = RandomForestClassifier(random_state = 1)
modelF = forest.fit(x_train, y_train)
y_predF = modelF.predict(x_test)
```

当使用超参数的默认值对训练集进行测试时，预测测试集的值的准确度为 0.991538461538。

**验证曲线**

有几种不同的方法可以为您的模型选择要调整的超参数。直观检查模型超参数的潜在优化值的一个好方法是使用验证曲线。可以在图表上绘制验证曲线，以显示模型在单个超参数的不同值下的表现。运行下面的代码来创建这里看到的四条验证曲线，其中 *param_name* 和 *param_range* 的值针对我们正在研究的四个参数中的每一个进行了相应的调整。

```
train_scoreNum, test_scoreNum = validation_curve(
                                RandomForestClassifier(),
                                X = x_train, y = y_train, 
                                param_name = 'n_estimators', 
                                param_range = num_est, cv = 3)
```

![](img/22cca3d689c3666cde758e2eef26bda4.png)

该验证曲线是使用[100，300，500，750，800，1200]值创建的，这些值是要对 n_estimators 进行测试的不同值。在此图中，我们看到，在测试这些值时，最佳值似乎是 750。值得注意的是，尽管训练和交叉验证得分之间似乎存在很大的差异，但训练集对三个交叉验证中的每一个都具有 100%的平均准确性，而交叉验证集对 n_estimators 的所有值都具有 99.5%至 99.6%的准确性，这表明无论使用多少个估计值，该模型都非常准确。

![](img/3dd47ba1ad7505ba0e939f4b1cfe0bc5.png)

在此图中，我们看到当 max_depth 设置为 15 时，交叉验证的最高准确度值接近 99.3%，这是我们将放入模型中的值。总的来说，选择 max _ depth 30 似乎更好，因为该值对于训练分数具有最高的准确性，我们选择不选择它，以防止我们的模型过度拟合训练数据。

![](img/d658fb7dc5009edc14d875cbe29ac31a.png)

在该图中，我们看到，在 min_samples_split 的值较高时，训练集和交叉验证集的准确性实际上都下降了，因此我们将选择 5 作为 min_samples_split 的值。在这种情况下，我们希望 min_samples_split 有一个较低的值是有意义的，因为这个参数的默认值是 2。由于在分割内部节点之前，我们为所需的最小样本数选择了较高的值，因此我们将拥有更多通用叶节点，这将对我们模型的整体准确性产生负面影响。

![](img/8de21a085800cd4919da694df620ccff.png)

在此图中，我们看到，min_samples_leaf 值每增加一次，训练集和交叉验证集的准确性都会下降，因此我们将选择 1 作为参数值，考虑到此参数的默认值为 1，这也是有意义的。

值得注意的是，在构建验证曲线时，其他参数保持默认值。出于本文的目的，我们将在一个模型中一起使用所有的优化值。构建了新的随机森林分类器，如下所示:

```
forestVC = RandomForestClassifier(random_state = 1,
                                  n_estimators = 750,
                                  max_depth = 15, 
                                  min_samples_split = 5,  min_samples_leaf = 1) modelVC = forestVC.fit(x_train, y_train) 
y_predVC = modelVC.predict(x_test)
```

该模型的精度为 0.993076923077，比我们的第一个模型更精确，但只差 0.0015。

**穷举网格搜索**

选择要调整的超参数的另一种方法是进行彻底的网格搜索或随机搜索。随机搜索将不会在这篇文章中讨论，但是可以在[这里](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.RandomizedSearchCV.html)找到关于其实现的更多文档。

详尽的网格搜索会尽可能多地接收超参数，并尝试超参数的每一种可能组合以及尽可能多的交叉验证。彻底的网格搜索是确定要使用的最佳超参数值的好方法，但是随着每个额外的参数值和您添加的交叉验证，它会很快变得非常耗时。

```
n_estimators = [100, 300, 500, 800, 1200]
max_depth = [5, 8, 15, 25, 30]
min_samples_split = [2, 5, 10, 15, 100]
min_samples_leaf = [1, 2, 5, 10] 

hyperF = dict(n_estimators = n_estimators, max_depth = max_depth,  
              min_samples_split = min_samples_split, 
             min_samples_leaf = min_samples_leaf)

gridF = GridSearchCV(forest, hyperF, cv = 3, verbose = 1, 
                      n_jobs = -1)
bestF = gridF.fit(x_train, y_train)
```

这里显示的代码运行了 25 分钟，但是选择的超参数在预测训练模型时具有 100%的准确性。得到的“最佳”超参数如下: *max_depth* = 15， *min_samples_leaf* = 1， *min_samples_split* = 2， *n_estimators* = 500。

再次，使用这些值作为超参数输入来运行新的随机森林分类器。

```
forestOpt = RandomForestClassifier(random_state = 1, max_depth = 15,     n_estimators = 500, min_samples_split = 2, min_samples_leaf = 1)

modelOpt = forestOpt.fit(x_train, y_train)
y_pred = modelOpt.predict(x_test)
```

当使用测试集进行测试时，该模型还产生了 0.993076923077 的准确度。

**调整超参数值得吗？**

仔细而有条理地调整超参数可能是有利的。它可以使您的分类模型更加准确，这将导致整体预测更加准确。然而，这并不总是值得你去做。让我们来看看不同测试的结果:

![](img/64be3e94dab4cd2382e5cb1a24d581b9.png)

最值得注意的是准确性的整体提高。当模型应用于我们的测试集时，基于网格搜索和验证曲线的结果选择的超参数产生了相同的精度:0.995386386386 这将我们的原始模型在测试集上的准确度提高了 0.0015。考虑到在我们需要的 4 个超参数上进行彻底的网格搜索花费了 25 分钟，在这种情况下可能不值得花时间。此外，我们的网格搜索给出的两个“优化”超参数值与 scikit-learn 的随机森林分类器的这些参数的默认值相同。当查看两个优化模型的混淆矩阵时，我们看到两个模型对红葡萄酒和白葡萄酒的错误预测数量相同，如下所示:

![](img/99395483327c267ac9a9310376bf9929.png)![](img/12864fcc54ed89440d9ae9558f9e1f19.png)

# 结论

超参数调整有利于创建更擅长分类的模型。在随机森林的情况下，可能没有必要，因为随机森林已经非常擅长分类。使用穷举网格搜索来选择超参数值也非常耗时。但是，在超参数只有几个潜在值的情况下，或者当初始分类模型不太准确时，最好至少调查一下更改模型中某些超参数值的影响。

关键术语/概念:超参数、验证曲线、穷举网格搜索、交叉验证