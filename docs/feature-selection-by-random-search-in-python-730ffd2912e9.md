# Python 中随机搜索的特征选择

> 原文：<https://towardsdatascience.com/feature-selection-by-random-search-in-python-730ffd2912e9?source=collection_archive---------18----------------------->

## 如何在 Python 中使用随机搜索进行要素选择

![](img/14a9b91c9a4c90db4b662131bd9f027e.png)

Photo by [Ian Gonzalez](https://unsplash.com/@ian_gonz?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

**特征选择**一直是机器学习的一大任务。根据我的经验，我可以肯定地说，特征选择**比型号选择本身更重要**。

# 特征选择和共线性

我已经写了一篇关于功能选择的[文章](https://medium.com/data-science-journal/how-to-measure-feature-importance-in-a-binary-classification-model-d284b8c9a301)。这是一种在**二元分类**模型中测量特征重要性的**无监督**方式，使用皮尔逊卡方检验和相关系数。

一般来说，对于简单的特征选择，无监督的方法通常是足够的。然而，每个模型都有自己的方式来“思考”这些特征，并处理它们与目标变量的相关性。而且，还有不太在意**共线性**(即特征之间的相关性)的模型，以及其他出现**非常大问题**的模型(例如线性模型)。

尽管可以通过模型引入的某种**相关性**度量标准(例如，对线性回归系数执行的 t-test 的 p 值)对**特性进行排序，但仅采用最相关的变量是不够的。想想一个等于另一个的特征，只是乘以 2。这些特征之间的线性相关性如果 1 和这个简单的乘法不影响与目标变量的相关性，所以如果我们只取最相关的变量，我们就取原始特征和相乘的那个。这导致了**共线性**，这对我们的模型来说是相当危险的。**

这就是为什么我们必须引入一些方法来更好地选择我们的功能。

# 随机搜索

随机搜索是数据科学家工具箱中非常有用的工具。这是一个非常简单的技术，经常使用，例如，在交叉验证和**超参数优化**中。

很简单。如果你有一个多维网格，并想在这个网格上寻找使某个**目标函数**最大化(或最小化)的点，随机搜索工作如下:

1.  在网格上随机取一点，测量目标函数值
2.  如果该值比迄今为止达到的最佳值更好，则将该点保存在内存中。
3.  重复预定的次数

就是这样。只是产生随机点，并寻找最好的一个。

这是寻找全局最小值(或最大值)的好方法吗？当然不是。我们寻找的点在一个非常大的空间中只有一个(如果我们幸运的话)，并且我们只有有限的迭代次数。在一个 *N-* 点网格中得到那个单点的概率是 *1/N* 。

那么，为什么随机搜索会被这么多使用呢？因为我们**从来没有真正想要**最大化我们的绩效评估；我们想要一个好的，**合理的高值**，它不是可能的最高值，以避免过度拟合。

这就是为什么随机搜索是可行的，并且可以用于特征选择。

# 如何使用随机搜索进行特征选择

随机搜索可用于特征选择，结果相当**好**。类似于随机搜索的程序的一个例子是**随机森林**模型，它为每棵树执行特征的随机选择。

这个想法非常简单:随机选择特性**，通过 **k 倍交叉验证**测量模型性能，并重复多次。提供最佳性能的功能组合正是我们所寻求的。**

**更准确地说，以下是要遵循的步骤:**

1.  **生成一个介于 1 和特征数之间的随机整数 *N* 。**
2.  **生成一个 0 到 *N-1* 之间的 *N* 整数随机序列，不重复。这个序列代表了我们的特征阵列。记住 Python 数组是从 0 开始的。**
3.  **在这些特征上训练模型，并用 k-fold 交叉验证对其进行交叉验证，保存一些性能测量的平均值**。****
4.  ****从第 1 点开始重复，重复次数不限。****
5.  ****最后，根据所选择的性能度量，获得给出最佳性能的特征阵列。****

# ****Python 中的一个实际例子****

****对于这个例子，我将使用包含在 **sklearn** 模块中的**乳腺癌数据集**。我们的模型将是一个**逻辑回归**，我们将使用**准确性**作为性能测量来执行 5 重交叉验证。****

****首先要导入必要的模块。****

```
**import sklearn.datasets
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score
import numpy as np**
```

****然后我们可以导入乳腺癌数据，并将其分解为输入和目标。****

```
**dataset= sklearn.datasets.load_breast_cancer()
data = dataset.data
target = dataset.target**
```

****我们现在可以创建一个逻辑回归对象。****

```
**lr = LogisticRegression()**
```

****然后，我们可以测量所有特征在 k 倍 CV 中的平均准确度。****

```
**# Model accuracy using all the features
np.mean(cross_val_score(lr,data,target,cv=5,scoring="accuracy"))
# 0.9509041939207385**
```

****是 95%。让我们记住这一点。****

****现在，我们可以实现一个随机搜索，例如，300 次迭代。****

```
**result = []# Number of iterations
N_search = 300# Random seed initialization
np.random.seed(1)for i in range(N_search):
    # Generate a random number of features
    N_columns =  list(np.random.choice(range(data.shape[1]),1)+1)

    # Given the number of features, generate features without replacement
    columns = list(np.random.choice(range(data.shape[1]), N_columns, replace=False))

    # Perform k-fold cross validation
    scores = cross_val_score(lr,data[:,columns], target, cv=5, scoring="accuracy")

    # Store the result
    result.append({'columns':columns,'performance':np.mean(scores)})# Sort the result array in descending order for performance measure
result.sort(key=lambda x : -x['performance'])**
```

****在循环和排序函数结束时，*结果*列表的第一个元素就是我们要寻找的对象。****

****我们可以使用这个值来计算这个特性子集的新性能度量。****

```
**np.mean(cross_val_score(lr, data[:,result[0][‘columns’]], target, cv=5, scoring=”accuracy”))
# 0.9526741054251634**
```

****如您所见，精确度提高了。****

# ****结论****

****随机搜索可以是执行特征选择的强大工具。它并不意味着给出为什么一些特性比其他特性更有用的原因(相对于其他特性选择过程，如递归特性消除)，但它可以是一个有用的工具，可以在更短的时间内达到**好的结果**。****