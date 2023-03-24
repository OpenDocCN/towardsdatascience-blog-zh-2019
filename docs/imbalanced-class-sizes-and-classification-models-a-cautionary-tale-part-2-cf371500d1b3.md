# 不平衡的班级:第 2 部分

> 原文：<https://towardsdatascience.com/imbalanced-class-sizes-and-classification-models-a-cautionary-tale-part-2-cf371500d1b3?source=collection_archive---------12----------------------->

![](img/f0102f52b46b2ae68830e82d40de28d7.png)

## 避免分类中的不平衡分类陷阱

R 最近，我写了[这篇文章](https://medium.com/@rriso88/imbalanced-class-sizes-and-classification-models-a-cautionary-tale-3648b8586e03?source=your_stories_page---------------------------)关于分类模型中不平衡的班级规模可能会导致分类模型的性能被高估。这篇文章讨论了我正在使用来自 Kaggle 的 [Airbnb 首次用户预订数据](https://www.kaggle.com/c/airbnb-recruiting-new-user-bookings)开发的一个分类项目。该项目的目标是预测首次使用 Airbnb 的用户是否会在美国/加拿大或国际上的某个地方预订度假屋。然而，美国/加拿大境内的预订量约占数据的 75%，因此很难准确估计国际预订量。

考虑到几乎 100%的观察值被预测为主导类(在我的例子中，是去美国/加拿大的旅行者)，我使用自适应合成(ADASYN)对测试集中的少数类进行过采样。由于对使用默认参数的开箱即用逻辑回归的结果不满意，我决定使用 scikit-learn 中的 GridSearchCV 进行一些模型选择和强力参数调整。我的特性是工程化的，我的类是平衡的，我闪亮的新 AWS 实例是计算优化的。什么会出错？

**数据预处理**

我急切地想拟合一些模型，看看我能根据已获得的特征对用户位置进行多好的分类。我已经将数据框架分成了目标变量(y)和特征矩阵(X)。我对数据集执行了训练-测试-分割(80%训练，20%测试),并进行分层，以确保少数类在测试集中得到代表。我在特征集(X_train 和 X_test)上使用了标准的标量转换。最后，我使用了自适应合成(ADASYN)方法对训练数据中的少数类进行过采样(详见[不平衡类第一部分](https://medium.com/@rriso88/imbalanced-class-sizes-and-classification-models-a-cautionary-tale-3648b8586e03))。

```
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from imblearn.over_sampling import ADASYNX_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, 
                                                stratify=y,random_state = 88)
std_scale = StandardScaler()
X_train_scaled = std_scale.fit_transform(X_train)
X_test_scaled = std_scale.transform(X_test)adasyn = ADASYN(random_state=88)
X_adasyn, y_adasyn = adasyn.fit_resample(X_train_scaled, y_train)
```

**GridSearchCV**

我循环了五个分类器:逻辑回归、K 近邻、决策树、随机森林和支持向量分类器。我将“模型”定义为带有分类器对象的每个分类器的字典列表(为了再现性，随机状态总是设置为 88，你能猜出我最喜欢的数字吗？)，以及要调整的特定于模型的超参数网格。

```
from sklearn.linear_model import LogisticRegression
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVCmodels = [{'name': 'logreg','label': 'Logistic Regression',
           'classifier': LogisticRegression(random_state=88),
           'grid': {"C":np.logspace(-3,3,7), "penalty":["l1","l2"]}},

          {'name': 'knn','label':'K Nearest Neighbors',
           'classifier':KNeighborsClassifier(),
           'grid': {"n_neighbors":np.arange(8)+1}},

          {'name': 'dsc','label': 'Descision Tree', 
           'classifier': DecisionTreeClassifier(random_state=88),
           'grid': {"max_depth":np.arange(8)+1}},

          {'name': 'rf', 'label': 'Random Forest',
           'classifier': RandomForestClassifier(random_state=88),
           'grid': {'n_estimators': [200, 500],'max_features': ['auto', 'sqrt', 'log2'],
                    'max_depth' : [4,5,6,7,8],'criterion' :['gini', 'entropy']}},

          {'name': 'svm_rbf', 'label': 'SVC (RBF)',
           'classifier':SVC(random_state=88),
           'grid': {'C': [1, 10, 100, 1000], 'gamma': [0.001, 0.0001], 'kernel': ['rbf']}}]
```

我在下面定义了 model_selection 函数来执行网格搜索，以优化给定模型的五个交叉验证集中的 hypterparameters。我决定使用受试者操作特征曲线(ROC AUC)分数下的计算面积来确定模型性能，以便尝试最大化真阳性，同时最小化模型预测中的假阳性。该函数返回一个字典，其中包括分类器、GridSearch 中的最佳参数以及验证集中的最佳平均 ROC_AUC 分数。

```
from sklearn.metrics import roc_auc_score
def model_selection(classifier, name, grid, X_train, y_train, scoring):

    gridsearch_cv=GridSearchCV(classifier, 
                               grid,
                               cv=5, 
                               scoring = scoring)

    gridsearch_cv.fit(X_adasyn, y_adasyn)

    results_dict = {}

    results_dict['classifier_name'] = name    
    results_dict['classifier'] = gridsearch_cv.best_estimator_
    results_dict['best_params'] = gridsearch_cv.best_params_
    results_dict['ROC_AUC'] = gridsearch_cv.best_score_

    return(results_dict)results = []
for m in models:    
    print(m['name'])    
    results.append(fit_first_model(m['classifier'], 
                                   m['name'],
                                   m['grid'],
                                   X_adasyn, 
                                   y_adasyn, 
                                   'roc_auc'))      
    print('completed')
```

最后，我将每个分类器的结果进行了比较！这个过程花费的时间并不多，所以要准备好消磨一些时间(这个过程结束时，我的厨房一尘不染！).

我将结果放入数据框中，以便更好地查看:

```
results_df = pd.DataFrame(results).sort_values(by='ROC_AUC', ascending = False)
```

![](img/f311c4441ad8d073578eff0ff30f9f65.png)

GridSearch ross-validation results across several classifiers with optimized parameters

嗯，我对第一行非常满意，随机森林分类器获得了 0.85 的 ROC_AUC 分数。这让我想到了 XGBoost 树分类器的一些宏伟计划。我将得到一个预测模型来与 Kaggle 竞赛的获胜者竞争…或者我是吗？

为了确认随机森林分类器的性能，我在测试集上对模型的预测性能进行了评分。当我看到测试集的 ROC_AUC 分数只有 0.525 时，我惊呆了。根据我的经验，测试分数通常低于交叉验证分数，我知道随机森林可能容易过度拟合，但这是 38%的性能下降！

为了更好地衡量，我对测试集上其余四个分类器的性能进行了评分。果然，ROC_AUC 测试分数明显低于交叉验证 ROC_AUC 平均值。逻辑回归是个例外，在验证集和测试集上的表现都比随机猜测略好。

![](img/c56fd30d12b1fb96e3b1099489766884.png)

Average cross-validation ROC_AUC scores are well above test scores

**订单事项**

这里发生了什么事？嗯，请记住，在分割我的训练数据进行交叉验证之前，我用 ADASYN *进行了过采样。所以，我的五重验证集并不代表现实世界中的分布。相反，它们包含代表少数类中难以分类的观察的“合成”数据点。因此，当我在测试集上对模型性能进行评分时(目标类比例代表真实世界)，分数显著下降。*

幸运的是，[不平衡学习有一个管道类](https://imbalanced-learn.readthedocs.io/en/stable/generated/imblearn.pipeline.Pipeline.html)，它将只在分类器拟合期间应用 ADASYN 重采样，从而允许我避免一些笨拙的 for 循环和手动 GridSearchCV。

下面是在交叉验证拟合期间使用过采样在随机森林分类器上构建 GridSearchCV 超参数调整管道的代码。(注意网格字典里的 class__ 前缀！)

```
from imblearn.pipeline import make_pipeline, Pipelinerf = RandomForestClassifier(random_state=88)
adasyn = ADASYN(random_state=88)grid = {'class__n_estimators': [200, 500],
        'class__max_features': ['auto', 'sqrt', 'log2'],
        'class__max_depth' : [4,5,6,7,8],
        'class__criterion' :['gini', 'entropy']}pipeline = Pipeline([('sampling', adasyn), ('class', rf)])grid_cv = GridSearchCV(pipeline, grid, scoring = 'roc_auc', cv = 5)

grid_cv.fit(X_train_scaled, y_train)grid_cv.best_score_
```

你瞧，交叉验证的 ROC_AUC 只有 0.578，更能说明模型在应用于测试集时的实际表现。我又一次遍历了所有模型，保存了最佳参数、平均验证分数和测试分数，以确保完整性。

![](img/28be5d83f1b75326273fe6219a3f7a11.png)

Average cross-validation roc_auc scores with pipeline oversampling are far more similar to test set scores

**关键要点**

在验证集中使用合成数据的分类算法的 ROC-AUC 曲线(左)表明，与在具有代表性不平衡类别的验证集中评分的 ROC-AUC 曲线(右)相比，明显高估了拟合度。(注意:这些算法是在单个验证集上评分的，而不是三次的平均值，因此得分与上表不同)。

![](img/efe746feec212b80bad27c67160f0219.png)

ROC-AUC Curves

虽然我无法自信地预测 Airbnb 用户的度假目的地，但这个项目说明了关注高级机器学习算法“幕后”发生的事情的重要性。特别是，我在之前关于[不平衡的类规模如何导致分类器性能被高估的文章](https://medium.com/@rriso88/imbalanced-class-sizes-and-classification-models-a-cautionary-tale-3648b8586e03)的基础上，讨论了为什么在使用交叉验证和超参数调整对重新采样的训练数据拟合分类器时，顺序很重要。

我希望这个系列对您有用。请随时在下面的评论中提供任何意见或额外的见解！