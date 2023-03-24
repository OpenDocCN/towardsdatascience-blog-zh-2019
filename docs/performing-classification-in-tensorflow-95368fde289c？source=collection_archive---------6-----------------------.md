# 在 TensorFlow 中执行分类

> 原文：<https://towardsdatascience.com/performing-classification-in-tensorflow-95368fde289c?source=collection_archive---------6----------------------->

![](img/0dfff5a2593d28c41add417703c44908.png)

在本文中，我将解释如何使用 Python 中的 TensorFlow 库执行分类。我们将使用加州人口普查数据，并尝试使用个人的各种特征来预测他们属于哪个收入阶层(> 50k 或<=50k). The data can be accessed at my GitHub profile in the TensorFlow repository. Here is the [链接](https://github.com/HarshSingh16/Tensorflow)来访问数据。我的代码和 Jupyter 笔记本可以在下面访问:

[](https://github.com/HarshSingh16/Tensorflow/blob/master/Classification_Tensorflow.ipynb) [## HarshSingh16/Tensorflow

### 我目前正在做的 Tensorflow 项目。通过创建帐户，为 harshsing 16/tensor flow 的发展做出贡献…

github.com](https://github.com/HarshSingh16/Tensorflow/blob/master/Classification_Tensorflow.ipynb) 

## 导入库和数据集

让我们从导入必要的库和数据集到我们的 Jupyter 笔记本开始。

![](img/61961be60c895e06b7da3044dd3b6f53.png)

让我们看看我们的数据集。所以，有 15 列。在这 15 列中，有 6 列是数值型的，其余 9 列是分类型的。下图提供了有关列类型和相应描述的信息。请注意，在这个例子中，我们不会使用变量“fnlwgt”。

![](img/b22a0001681471eada83e4510e9017e9.png)

## 查看我们的目标列:

我们现在来看看我们的目标栏*“收入”*。如前所述，我们正试图对个人的收入等级进行分类。所以，基本上有两类——“≤50K”和“> 50K”。

![](img/f298dc23fa8485496fa4db9c28d2a1f9.png)

然而，我们不能让我们的目标标签保持当前的字符串格式。这是因为 TensorFlow 不把字符串理解为标签。我们必须将这些字符串转换成 0 和 1。如果收入等级大于 50K，则为“1 ”,如果收入等级小于或等于 50K，则为“0”。我们可以通过创建一个 for 循环，然后将标签附加到一个列表中来实现。我还用我们刚刚创建的新列表直接更新了现有的*“收入”*列。下面是执行转换的代码:

![](img/ddc1142030afe24c2ee78b995a728778.png)

## 标准化我们的数字特征:

我们现在想要规范化我们的数字特征。**归一化**是将数值
特征可采用的实际值范围转换为标准值范围的过程，通常在区间[1，1]或[0，1]内。规范化数据并不是一个严格的要求。然而，在实践中，它可以提高学习速度。此外，确保我们的输入大致在同一个相对较小的
范围内是有用的，这样可以避免计算机在处理非常小或非常大的数字时出现的问题(称为数字溢出)。我们将使用 lambda 函数来做到这一点。代码如下:

![](img/52dc1121544af079cdb9ed146b6f9182.png)

## 创建连续和分类特征:

下一步是为数值和分类数据创建特征列。将**特性列**视为原始数据和估计值之间的中介。特性列非常丰富，使您能够将各种各样的原始数据转换成评估人员可以使用的格式，从而允许进行简单的实验。这里有一个来自 TensorFlow 网站的例子，说明了功能栏是如何工作的。这里讨论的数据是著名的 Iris 数据集。如下图所示，通过估计器的`feature_columns`参数(Iris 的`DNNClassifier`)指定模型的输入。特征列将输入数据(由`input_fn`返回)与模型连接起来。

![](img/2db5a37b4c424176daeabaa78c7bc890.png)

为了创建特性列，我们必须从`[tf.feature_column](https://www.tensorflow.org/api_docs/python/tf/feature_column)`模块调用函数。这张来自 TensorFlow 网站的图片解释了该模块中的九个功能。如下图所示，所有九个函数都返回分类列或密集列对象，除了`bucketized_column`，它从这两个类继承:

![](img/292210fc82868296ff0e5d233bb8247a.png)

现在是时候为数据集创建要素列了。我们将首先处理数字列，并通过使用`[tf.feature_column.numeric_column](https://www.tensorflow.org/api_docs/python/tf/feature_column/numeric_column)`将它们转换成特性

![](img/248c8f96bd16da71e526670720dba9dd.png)

接下来，我们将处理分类特征。这里我们有两个选择-

1.  `[tf.feature_column.categorical_column_with_hash_bucket](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_hash_bucket) :`如果您事先不知道分类列的一组可能值，并且有太多可能值，请使用此选项
2.  `[tf.feature_column.categorical_column_with_vocabulary_list](https://www.tensorflow.org/api_docs/python/tf/feature_column/categorical_column_with_vocabulary_list) :` 如果您知道一列的所有可能的特征值的集合，并且只有少数几个，则使用此选项

因为在我们的例子中，每个分类列中有太多的特征值，所以我们将使用散列函数。请确保指定的哈希值大于列的类别总数，以避免将两个不同的类别分配给同一个哈希值。

![](img/45148869c7df11d414cfdd72a1c7c7d4.png)

接下来，我们希望将所有这些变量放入一个名为 *feat_columns* 的列表中。

![](img/7180fa5a500ea4665eaebf36bed0a29d.png)

## 执行培训和测试分割

我们将使用`sklearn`库来执行我们的训练测试分割。因此，我们必须将我们的标签与特征分开。这是因为来自`sklearn` 的模块 *train_test_split* 模块要求您显式指定特性及其目标列。

![](img/734e9e76a35fbd69eb729e3c27d1c64a.png)

我们现在将导入我们的 *train_test_split* 模块。我们将在测试集中保留 33%的数据。这将为我们提供足够数量的观察结果，以准确评估我们模型的性能。

![](img/36342361761bb6afe4d384f9210da957.png)

## 定义输入函数和线性分类器:

我们现在创建一个输入函数，将熊猫数据帧输入到我们的分类器模型中。模块 *tf.estimator.inputs* 提供了一种非常简单的方法。它要求您指定功能、标签和批量大小。它还有一个名为`**shuffle,**`的特殊参数，允许模型以随机顺序读取记录，从而提高模型性能。

![](img/a409b3a91b302689308df54ec5103b9d.png)

接下来，我们将定义我们的线性分类器。我们的线性分类器将训练一个线性模型来将实例分类为两个可能的类别之一——即 0 代表收入小于或等于 50K，1 代表收入大于 50K。同样，*TF . estimator . linear classifier*允许我们只用一行代码就能做到这一点。作为参数的一部分，我们必须指定我们的特性列和类的数量。

![](img/bfa06c26e2f8053ac3f0bc8fa68d921d.png)

## 训练模型:

最后，激动人心的部分！让我们开始训练我们的模型。显而易见，我们必须指定输入函数。`**steps**` 参数指定训练模型的步数。

![](img/b77da5eb40eff4e30a01a098c358d77f.png)![](img/3b426e4a778836cf5f20888a84f91aa0.png)

## **预测**

现在是时候进行预测了。首先，我们需要重新定义我们的输入函数。虽然定型模型需要您指定目标标注和要素，但在生成预测时，您不需要指定目标标注。预测结果将在稍后与测试数据上的实际标签进行比较，以评估模型。所以让我们开始吧！

![](img/9d1cb5f150ae4ce06de4613c1c660e51.png)

现在让我们将输入函数输入到 *model.predict 中。请注意，我已经在 model.predict* 函数周围调用了 list 对象，这样我就可以在下一步轻松访问预测的类。

![](img/2869905b303db5639ad1611dd0022abf.png)

万岁！我们现在有了我们的预测。让我们看看对测试数据中第一个观察值的预测。在下图中，我们可以看到我们的模型预测它属于 0 类(参考 *class_ids* )。我们还有一堆其他的预测，比如类的概率，逻辑等等。然而，为了进行我们的模型评估，我们只需要*class _ id*。下一步，我们将尝试创建一个我们的*class _ id*的列表。

![](img/ab07977ad5cfff21b8c46cbcf2b02a1f.png)

如上所述，我们现在将从字典的预测列表中创建一个仅包含*class _ id*键值的列表，这些预测将用于与真实的 *y_test* 值进行比较。

![](img/78d478d36323994930b7c96f17295918.png)

查看前 10 个预测的类别。

![](img/b816c51177500211a830edcc77f45ca8.png)

## **车型评价**

我们现在已经到了项目的最后阶段。我们现在将尝试评估我们模型的预测，并通过使用 *sklearn* 库将它们与实际标签进行比较。

![](img/b2c41d37da05a1bff91bd045a9ca24e1.png)

这是我们的分类报告:

![](img/1cc175a5f9dfb59115832d132a355aa8.png)

我还打印了一些其他评估指标，这些指标将让我们非常清楚地了解我们模型的性能。我们的模型具有 82.5%的总体准确性和 86.5%的 AUC。好的分类器在曲线下有更大的面积。显然，我们的模型已经取得了一些非常好的结果。

![](img/86eee94aee940d3fea2f07336b071215.png)

最后备注:

我希望这篇文章能让您很好地理解在 TensorFlow 中执行分类任务。期待听到大家的想法和评论。请随时通过 LinkedIn 联系我。