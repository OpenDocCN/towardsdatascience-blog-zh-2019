# 超参数调整和模型选择，就像电影明星一样

> 原文：<https://towardsdatascience.com/hyper-parameter-tuning-and-model-selection-like-a-movie-star-a884b8ee8d68?source=collection_archive---------13----------------------->

![](img/9cc598f0cb2f90de3e6886f4cb7f1c65.png)

Photo by [Markus Spiske](https://unsplash.com/@markusspiske?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

## 像你一样编码、分析、选择和调整*真的*知道你在做什么。

“针对随机森林分类器优化的超参数调整”是那些在电影场景中听起来很轻松的短语之一，在电影场景中，黑客正在积极地输入以“获得对大型机的访问”，就像在关于数据科学的媒体文章中一样。然而，事实是，这样的短语是将数学和计算概念结合在一个领域的不幸结果，更糟糕的是，是一个名字。虽然本文中的概念将受益于对使用 scikit-learn 的基本 python 建模以及其中一些模型如何工作的扎实理解，但我将尝试自下而上地解释一切，以便所有级别的读者都可以享受和学习这些概念；你也可以听起来(和编码)像好莱坞黑客。

在本文中，我将尝试解决以下问题:

*   什么是超参数，它与参数有何不同？
*   什么时候应该使用超参数？
*   超参数实际上是做什么的？
*   如何调整超参数？
*   什么是网格搜索？
*   什么是流水线？
*   如何定义单个超参数？

跳到最后，查看所有这些主题的摘要。

# 什么是超参数？

超参数这个术语是由于机器学习在编程和大数据中日益流行而产生的。许多作为数据科学家或程序员开始其旅程的人都知道参数这个词被定义为一个值，该值被传递到一个函数中，使得该函数对这些值执行操作和/或被这些值通知。然而，在机器学习和建模中，参数**不是由程序员**输入的**，而是由机器学习模型**开发的。这是由于机器学习和传统编程的根本区别；在传统编程中，规则和数据由程序员输入以便输出结果，而在机器学习中，输入数据和结果以便输出规则(在这种情况下通常称为参数)。这个 [Google I/O 2019 演讲](https://youtu.be/VwVg9jCtqaU?t=111)在最初几分钟非常简洁地解决了这个翻转。

如果模型本身生成参数，那么将我们(程序员、数据科学家等)输入的内容也称为模型参数会非常混乱。这就是超参数这个术语的诞生。超参数被输入到生成其自身参数的任何机器学习模型中，以便影响所述生成的参数的值，希望使模型更加精确。在本文的稍后部分，我将展示具体的例子，以及定义什么是单个的超参数。

# 这些单独的超参数是如何定义的，它们有什么影响？

让我们快速浏览一下 [scikit-learn 关于逻辑回归的文档](https://scikit-learn.org/stable/modules/generated/sklearn.linear_model.LogisticRegression.html)，以便更好地理解这个问题的真正含义。

```
**LogisticRegression**(*penalty=’l2’*, *dual=False*, *tol=0.0001*, *C=1.0*, *fit_intercept=True*, *intercept_scaling=1*, *class_weight=None*, *random_state=None*, *solver=’warn’*, *max_iter=100*, *multi_class=’warn’*, *verbose=0*, *warm_start=False*, *n_jobs=None*, *l1_ratio=None*)
```

正如我们在这里看到的，`LogisticRegression()`接受了 15 个不同的值，我们现在知道这些值被称为超参数。然而，这 15 个值中的每一个都定义了一个默认值，这意味着在没有指定任何超参数的情况下创建一个`LogisticRegression()`对象是非常可能的，甚至是常见的。这是 scikit-learn 中所有模型的情况。因此，我将只花时间来定义和解释四种常见建模方法的一些更相关和通常修改的超参数。

# 逻辑回归:

*   **惩罚**:用于指定对非贡献变量系数的惩罚方式。
*   Lasso (L1)执行要素选择，因为它将不太重要的要素的系数缩小到零。
*   里奇(L2)所有的变量都包括在模型中，尽管有些被缩小了。计算强度低于 lasso。
*   两个罚值都限制解算器的选择，如这里的[所示](https://scikit-learn.org/stable/modules/linear_model.html#logistic-regression)。
*   **C** :是正则项的逆(1/λ)。它告诉模型有多大的参数被惩罚，较小的值导致较大的惩罚；必须是正浮点数。
*   常用值:[0.001，0.1 …10..100]
*   **class_weight** :允许你更强调一个类。例如，如果类别 1 和类别 2 之间的分布严重不平衡，则模型可以适当地处理这两种分布。
*   默认值是所有权重= 1。类别权重可以在字典中指定。
*   “平衡”将创建与类别频率成反比的类别权重，给予较小类别的个别事件更多权重。

# 线性回归:

*   **拟合截距**:指定是否计算模型截距或设置为零。
*   如果为假，回归线的截距将为 0。
*   如果为真，模型将计算截距。
*   **规格化:**指定是否使用 L2 范数规格化模型的数据。

# SVM

*   **C** :是正则项的逆(1/λ)。它告诉模型有多大的参数被惩罚，较小的值导致较大的惩罚；必须是正浮点数。
*   较高的 C 将导致模型的错误分类较少，但更有可能导致过度拟合。
*   良好的值范围:[0.001，0.01，10，100，1000…]
*   **class_weight** :将 class i 的参数设置为 class_weight[i] *C。
*   这可以让你更加重视一门课。例如，如果类别 1 和类别 2 之间的分布严重不平衡，则模型可以适当地处理这两种分布。
*   默认值是所有权重= 1。类别权重可以在字典中指定。
*   “平衡”将创建与类别频率成反比的类别权重，给予较小类别的个别事件更多权重。

# k-最近邻

*   **n_neighbors** :确定计算最近邻算法时使用的邻居数量。
*   良好的值范围:[2，4，8，16]
*   **p** :计算闵可夫斯基度规时的功率度规，这是一个数学上相当复杂的话题。在评估模型时，简单地尝试这里的 1 和 2 通常就足够了。
*   使用值 1 计算曼哈顿距离
*   使用值 2 计算欧几里德距离(默认)

# 随机森林

*   **n_estimators** :设置要在林中使用的决策树的数量。
*   默认值为 100
*   良好的值范围:[100，120，300，500，800，1200]
*   **max_depth** :设置树的最大深度。
*   如果未设置，则没有上限。这棵树会一直生长，直到所有的叶子都变得纯净。
*   限制深度有利于修剪树，以防止对噪声数据的过度拟合。
*   良好的值范围:[5，8，15，25，30，无]
*   **min_samples_split** :在内部节点进行分割(微分)之前所需的最小样本数
*   默认值为 2
*   良好的值范围:[1，2，5，10，15，100]
*   **min_samples_leaf** :创建叶(决策)节点所需的最小样本数。
*   默认值为 1。这意味着，仅当每条路径至少有 1 个样本时，才允许在任何深度的分割点。
*   良好的值范围:[1，2，5，10]
*   **max_features** :设置考虑最佳节点分割的特征数量
*   默认为“自动”，这意味着特征数量的平方根用于树中的每次分割。
*   “无”表示所有特征都用于每次分割。
*   随机森林中的每个决策树通常使用随机的特征子集进行分割。
*   良好的值范围:[log2，sqrt，auto，None]

# 如何调整超参数，它们实际上有什么作用？

为了弄清楚这两个问题，让我们用经典的加州大学欧文分校虹膜数据集来解决一个例子。

首先，我们将加载数据集并导入我们将使用的一些包:

```
# import packages
import numpy as np
from sklearn import linear_model, datasets
from sklearn.model_selection import GridSearchCV
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline# Loading dataset
iris = datasets.load_iris()
features = iris.data
target = iris.target
```

现在，让我们创建一个快速模型，不使用额外的超参数，并获得分数供以后评估。

```
logistic.fit(features, target)
print(logistic.score(features, target))
```

输出:

```
0.96
```

现在，让我们尝试一些超参数调整的方法，看看我们是否可以提高我们的模型的准确性。

## 什么是网格搜索？

网格搜索是一种方法，通过这种方法，我们为每个超参数创建可能的超参数值集，然后在“网格”中相互测试它们例如，如果我想用值`[L1, L2]`和值 C 作为`[1,2]`来测试一个逻辑回归，`GridSearchCV()`方法会用`C=1`测试`L1`，然后用`C=2`测试`L1`，然后用两个值`C`测试`L2`，创建一个 2x2 的网格和总共四个组合。让我们看一个没有当前数据集的例子。verbose 参数指示函数运行时是否打印信息，cv 参数指的是[交叉验证](/cross-validation-a-beginners-guide-5b8ca04962cd)折叠。关于`GridSearchCV()`的完整文档可以在这里找到[。](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.GridSearchCV.html)

```
# Create range of candidate penalty hyperparameter values
penalty = ['l1', 'l2']# Create range of candidate regularization hyperparameter values C
# Choose 10 values, between 0 and 4
C = np.logspace(0, 4, 10)# Create dictionary hyperparameter candidates
hyperparameters = dict(C=C, penalty=penalty)# Create grid search, and pass in all defined values
gridsearch = GridSearchCV(logistic, hyperparameters, cv=5, verbose=1) 
# the verbose parameter above will give output updates as the calculations are complete. # select the best model and create a fit
best_model = gridsearch.fit(features, target)
```

既然我们的模型是基于更大的输入空间创建的，我们可以希望看到改进。让我们检查一下:

```
print('Best Penalty:', best_model.best_estimator_.get_params(['penalty']) 
print('Best C:', best_model.best_estimator_.get_params()['C'])
print("The mean accuracy of the model is:",best_model.score(features, target))
```

输出:

```
Best Penalty: l1 
Best C: 7.742636826811269
The mean accuracy of the model is: 0.98
```

使用相同的模型并增加超参数的小变化，精度提高了 0.02。尝试用不同的超参数集进行试验，并将它们添加到超参数字典中，然后再次运行`GridSearchCV()`。请注意添加许多超参数如何快速增加计算时间。

## 什么是流水线？

如果我们想要用多个超参数测试多个算法，以便找到可能的最佳模型，该怎么办？流水线允许我们以一种代码高效的方式做到这一点。让我们看一个 Iris 数据集的例子，看看我们是否可以改进我们的逻辑回归模型。

```
# Create a pipeline
pipe = Pipeline([("classifier", RandomForestClassifier())])# Create dictionary with candidate learning algorithms and their hyperparameters
search_space = [
                {"classifier": [LogisticRegression()],
                 "classifier__penalty": ['l2','l1'],
                 "classifier__C": np.logspace(0, 4, 10)
                 },
                {"classifier": [LogisticRegression()],
                 "classifier__penalty": ['l2'],
                 "classifier__C": np.logspace(0, 4, 10),
                 "classifier__solver":['newton-cg','saga','sag','liblinear'] ##This solvers don't allow L1 penalty
                 },
                {"classifier": [RandomForestClassifier()],
                 "classifier__n_estimators": [10, 100, 1000],
                 "classifier__max_depth":[5,8,15,25,30,None],
                 "classifier__min_samples_leaf":[1,2,5,10,15,100],
                 "classifier__max_leaf_nodes": [2, 5,10]}]# create a gridsearch of the pipeline, the fit the best model
gridsearch = GridSearchCV(pipe, search_space, cv=5, verbose=0,n_jobs=-1) # Fit grid search
best_model = gridsearch.fit(features, target)
```

注意这个函数运行需要多长时间。在另一篇文章中，我将讨论如何减少运行时间和挑选有效的超参数，以及如何将一个`RandomizedSearchCV()`和一个`GridSearchCV`结合起来。运行该方法后，让我们检查结果。

```
print(best_model.best_estimator_)
print("The mean accuracy of the model is:",best_model.score(features, target))
```

输出:

```
Pipeline(memory=None, steps=[('classifier', LogisticRegression(C=7.742636826811269, class_weight=None, dual=False, fit_intercept=True,intercept_scaling=1, l1_ratio=None,                                     max_iter=100, multi_class='warn', n_jobs=None, penalty='l1',                                     random_state=None, solver='warn', tol=0.0001, verbose=0, warm_start=False))], verbose=False) The mean accuracy of the model is: 0.98
```

根据我们的管道搜索，具有指定超参数的`LogisticRegression()`比具有任何给定超参数的`RandomForestClassifier()`执行得更好。有意思！

好了，我们已经使用了一个管道方法来实现这一切，但是它实际上做什么，为什么我们要传入一个`RandomForestClassifier()`？

pipeline 方法允许我们传入预处理方法以及我们想要用来创建数据模型的算法。在这个简单的例子中，我们跳过了预处理步骤，但是我们仍然输入了一个模型。我们输入的算法只是用于实例化管道对象的算法，但是将被我们创建的`search_space`变量的内容所替代，该变量稍后将被传递到我们的`GridSearchCV()`中。这里可以找到一个简化的帖子，只关注管道[。](/a-simple-example-of-pipeline-in-machine-learning-with-scikit-learn-e726ffbb6976)

我们的原始基线模型和用我们的超参数调整生成的模型之间的精度差异显示了超参数调整的效果。通过指导我们的机器学习模型的创建，我们可以提高它们的性能，并创建更好、更可靠的模型。

# 摘要

## 什么是超参数，它与参数有何不同？

在机器学习模型中使用超参数来更好地指导模型用来生成数据预测的参数的创建。超参数由程序员设置，而参数由模型生成。

## 什么时候应该使用超参数？

永远！模型通常有内置的默认超参数，可用于大多数目的。然而，在许多情况下，使用超参数调优会挤出模型的额外性能。了解不同超参数的限制和影响有助于限制过度拟合等负面影响，同时提高性能。

## 超参数实际上是做什么的？

简单地说，它们改变了模型寻找模型参数的方式。个别定义可以在上面的文章中找到。

## 如何调整超参数？

网格搜索、随机搜索和流水线是常用的方法。这篇文章没有提到随机搜索，但是你可以在这里阅读更多的。

## 什么是网格搜索？

网格搜索是对传递给`GridSearchCV()`函数的所有超参数的元素测试。网格搜索在大搜索空间上的计算开销很大，它的测试也很详尽。

## 什么是流水线？

流水线允许搜索多个算法，每个算法都有许多超参数。这是一种非常高效的测试许多模型的方法，以便选择最好的一个。此外，它还可以处理再加工方法，允许进一步控制过程。

最后，下面是一些函数，它们可以通过传入参数来执行一些不同类型的超参数调整。包含本文中使用的所有代码的 Google Colab 笔记本也可以在这里找到。使用这些函数，您可以在一行代码中高效地执行超参数调整！

```
# # # Hyperparameter tuning and model selection
import numpy as np
from sklearn import linear_model
from sklearn import datasets
from sklearn.linear_model import LogisticRegression 
from sklearn.ensemble import RandomForestClassifier 
from sklearn.model_selection import GridSearchCV 
from sklearn.pipeline import Pipeline
from sklearn.model_selection import RandomizedSearchCV
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressordef perform_gridsearch_log(features, labels,
                       log_params = {'penalty': ['l1', 'l2'], 'C': np.logspace(0, 4, 10)},
                       cv=5, verbose = 1):
  import numpy as np
  from sklearn import linear_model, datasets
  from sklearn.model_selection import GridSearchCV

  global best_model
  logistic = linear_model.LogisticRegression()
  penalty = log_params['penalty']
  C = log_params['C']
  hyperparameters = dict(C=C, penalty=penalty)gridsearch = GridSearchCV(logistic, hyperparameters, cv=cv, verbose=verbose) # Fit grid search
  best_model = gridsearch.fit(features, target)

  print(best_model.best_estimator_)
  print("The mean accuracy of the model is:",best_model.score(features, labels))def rand_forest_rand_grid(features, labels, n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                                           max_features = ['auto', 'sqrt'],
                                           max_depth = [int(x) for x in np.linspace(10, 110, num = 11)],
                                           min_samples_split = [2, 5, 10],
                                           min_samples_leaf = [1, 2, 4], bootstrap = [True, False]):

  max_depth.append(None)
  global best_model

  random_grid = {'n_estimators': n_estimators,
                 'max_features': max_features,
                 'max_depth': max_depth,
                 'min_samples_split': min_samples_split,
                 'min_samples_leaf': min_samples_leaf,
                 'bootstrap': bootstrap}

  rf = RandomForestRegressor()

  rf_random = RandomizedSearchCV(estimator = rf, param_distributions = random_grid, n_iter = 100, cv = 3, verbose=1, random_state=42, n_jobs = -1)

  best_model = rf_random.fit(features, labels)
  print(best_model.best_estimator_)
  print("The mean accuracy of the model is:",best_model.score(features, labels))def rand_forest_grid_search(features, labels, n_estimators = [int(x) for x in np.linspace(start = 200, stop = 2000, num = 10)],
                                           max_features = ['auto', 'sqrt'],
                                           max_depth = [int(x) for x in np.linspace(10, 110, num = 11)],
                                           min_samples_split = [2, 5, 10],
                                           min_samples_leaf = [1, 2, 4], bootstrap = [True, False]):
  param_grid = {'n_estimators': n_estimators,
                 'max_features': max_features,
                 'max_depth': max_depth,
                 'min_samples_split': min_samples_split,
                 'min_samples_leaf': min_samples_leaf,
                 'bootstrap': bootstrap}

  global best_model
  rf = RandomForestRegressor()

  grid_search = GridSearchCV(estimator = rf, param_grid = param_grid, 
                          cv = 3, n_jobs = -1, verbose = 1)best_model = grid_search.fit(train_features, train_labels)
  print(best_model.best_estimator_)
  print("The mean accuracy of the model is:",best_model.score(features, labels))def execute_pipeline(features,labels, search_space=[
                {"classifier": [LogisticRegression()],
                 "classifier__penalty": ['l2','l1'],
                 "classifier__C": np.logspace(0, 4, 10)
                 },
                {"classifier": [LogisticRegression()],
                 "classifier__penalty": ['l2'],
                 "classifier__C": np.logspace(0, 4, 10),
                 "classifier__solver":['newton-cg','saga','sag','liblinear'] ##This solvers don't allow L1 penalty
                 },
                {"classifier": [RandomForestClassifier()],
                 "classifier__n_estimators": [10, 100, 1000],
                 "classifier__max_depth":[5,8,15,25,30,None],
                 "classifier__min_samples_leaf":[1,2,5,10,15,100],
                 "classifier__max_leaf_nodes": [2, 5,10]}], cv=5, verbose=0, n_jobs=-1):global best_model

  pipe = Pipeline([("classifier", RandomForestClassifier())])

  gridsearch = GridSearchCV(pipe, search_space, cv=cv, verbose=verbose,n_jobs=n_jobs) # Fit grid search
  best_model = gridsearch.fit(features, labels)
  print(best_model.best_estimator_)
  print("The mean accuracy of the model is:",best_model.score(features, labels))
```

感谢阅读！