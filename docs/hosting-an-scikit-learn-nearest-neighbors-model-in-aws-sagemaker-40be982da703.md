# 在 AWS SageMaker 中托管 Scikit-Learn 最近邻模型

> 原文：<https://towardsdatascience.com/hosting-an-scikit-learn-nearest-neighbors-model-in-aws-sagemaker-40be982da703?source=collection_archive---------12----------------------->

除了一系列内置算法，AWS SageMaker 还提供了训练和托管 Scikit-Learn 模型的能力。在本帖中，我们将展示如何在 SageMaker 中训练和托管 Scikit-Learn 最近邻模型。

![](img/1aed18e22b81b05fa17d6ce027da6ff4.png)

我们的用例将是本文中[描述的葡萄酒推荐模型。在本练习中，我们将假设葡萄酒推荐器的输入数据已经准备就绪:一组 180，000 个 300 维向量，每个向量描述一种特定葡萄酒的风味特征。根据前面提到的文章，我们将这些向量称为 wine 嵌入。我们的目标是有一个模型，可以返回与给定输入向量最相似的葡萄酒嵌入。](/robosomm-chapter-3-wine-embeddings-and-a-wine-recommender-9fc678f1041e)

正如官方[文件](https://sagemaker.readthedocs.io/en/stable/using_sklearn.html)所述，我们将在整个过程中处理两个主要步骤:

1.  准备一个 Scikit-Learn 脚本在 SageMaker 上运行
2.  通过 Scikit-Learn 估算器在 SageMaker 上运行这个脚本

**准备在 SageMaker 上运行的 Scikit-Learn 脚本**

在我们的 Scikit-Learn 脚本中，我们将从我们的输入通道(S3 桶和我们完全准备好的葡萄酒嵌入集)加载数据，并配置我们将如何训练我们的模型。模型的输出位置也在这里指定。这一功能已被置于主要保护之下。

我们还将安装 s3fs，这是一个允许我们与 S3 交互的包。这个包使我们能够识别特定的 S3 目录(输入数据、输出数据、模型),以便脚本与之交互。另一种方法是使用特定于 SageMaker 的环境变量，这些变量指定要与之交互的标准 S3 目录。为了说明这两个选项，我们将使用环境变量 SM_MODEL_DIR 来存储模型，以及输入和输出数据的特定目录地址。

到目前为止一切顺利！通常，我们可以在 SageMaker 上运行这个脚本，首先训练模型，然后通过调用“predict”方法返回预测。然而，我们的 Scikit-Learn 最近邻模型没有“预测”方法。实际上，我们的模型正在计算各种葡萄酒嵌入之间的余弦距离。对于任何给定的输入向量，它将返回最接近该点的葡萄酒嵌入。这与其说是一种预测，不如说是一种计算哪些点彼此距离最近的方法。

幸运的是,“模型服务”功能允许我们配置 Scikit-Learn 脚本来实现这种类型的定制。模型服务由三个功能组成:

i) **input_fn** :这个函数将输入数据反序列化为一个对象，该对象被传递给 prediction_fn 函数

ii) **predict_fn** :该函数获取 input_fn 函数的输出，并将其传递给加载的模型

iii) **output_fn** :该函数获取 predict_fn 的结果并将其序列化

这些函数中的每一个都有一个运行的[默认实现](https://github.com/aws/sagemaker-scikit-learn-container/blob/master/src/sagemaker_sklearn_container/serving.py)，除非在 Scikit-Learn 脚本中另有说明。在我们的例子中，我们可以依赖 input_fn 的默认实现。我们传递到最近邻模型中进行预测的 wine 嵌入是一个 Numpy 数组，这是默认 input_fn 可接受的内容类型之一。

对于 predict_fn，我们会做一些定制。我们不是在模型对象上运行“预测”方法，而是返回前 10 个最近邻居的索引列表，以及输入数据和每个相应建议之间的余弦距离。我们将让函数返回一个 Numpy 数组，该数组由一个包含这些信息的列表组成。

函数 output_fn 也需要一些小的定制。我们希望这个函数返回一个序列化的 Numpy 数组。

Scikit-Learn 脚本还有一个组件:加载模型的函数。必须指定函数 model_fn，因为这里没有提供默认值。该函数从保存模型的目录中加载模型，以便 predict_fn 可以访问它。

具有上述所有功能的脚本应该保存在一个源文件中，该文件独立于您用来向 SageMaker 提交脚本的笔记本。在我们的例子中，我们将这个脚本保存为 sklearn_nearest_neighbors.py。

**通过 Scikit-Learn 评估器在 SageMaker 上运行这个脚本**

从这里开始就一帆风顺了:我们需要做的就是运行 Scikit-Learn 脚本来适应我们的模型，将它部署到一个端点，然后我们就可以开始使用它来返回最近邻葡萄酒嵌入。

在 SageMaker 笔记本中，我们运行以下代码:

现在，我们的近邻模型已经准备好行动了！现在，我们可以使用。predict 方法，返回样本输入向量的葡萄酒推荐列表。正如预期的那样，这将返回一个嵌套的 Numpy 数组，该数组由输入向量与其最近邻居之间的余弦距离以及这些最近邻居的索引组成。

```
[[1.37459606e-01 1.42040288e-01 1.46988100e-01 1.54312524e-01
  1.56549391e-01 1.62581288e-01 1.62581288e-01 1.62931791e-01
  1.63314825e-01 1.65550581e-01]
 [91913 24923 74096 26492 77196 96871 113695 874654
  100823 14478]]
```

我们走吧！我们在 AWS SageMaker 中培训并主持了一个 Scikit-Learn 最近邻模型。