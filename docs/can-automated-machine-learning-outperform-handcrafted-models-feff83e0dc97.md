# 自动化机器学习能胜过手工模型吗？

> 原文：<https://towardsdatascience.com/can-automated-machine-learning-outperform-handcrafted-models-feff83e0dc97?source=collection_archive---------16----------------------->

![](img/dcf158b6e9e6d1a55db05602c8619263.png)

Can Auto-Keras really find better models than you? — Photo by [Annie Theby](https://unsplash.com/@annietheby?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

## 在真实数据集上测试 Auto-Keras 模型

自动机器学习(AutoML)可用于自动寻找和训练机器学习模型。您不再需要自己创建模型，AutoMl 算法将分析您的数据并自动选择最佳模型。

但是那些模型真的有多好呢？它们能与定制模型相比吗？或者它们更好吗？我们再也不需要挑选另一个模特了吗？让我们来了解一下！

# 简介:Auto-Keras

像谷歌这样的公司已经提供了 AutoML 产品，但是有了 Auto-Keras，还有一个开源的解决方案。在[官方入门示例](https://autokeras.com/start/)中，Auto-Keras 用于为 MNIST 数据集寻找最佳神经架构。当我尝试这个例子时，得到的模型达到了大约 98%的分数。这令人印象深刻，所以我决定使用 Auto-Keras 尝试在 [**Kaggle Titanic 数据集**](https://www.kaggle.com/francksylla/titanic-machine-learning-from-disaster) 上击败自己。

我在这次比赛中的最好成绩是大约 80%的精确度，这使我目前在所有参赛者中排名前 900。这是一个相当不错的分数，所以让我们看看 Auto-Keras 是否能超过我！

> 你可以在这里找到笔记本的完整代码

## 设置

我用谷歌 Colab 做这个项目。要在 Google Colab 中安装 Auto-Keras，只需运行:

```
!pip install autokeras
```

如果您想在本地运行它，您可以使用 pip 从命令行安装 Auto-Keras。

## 数据

对于这个例子，我使用了可以从 [Kaggle 竞赛](https://www.kaggle.com/c/titanic/data)下载的数据集。要使用 Auto-Keras 模型中的数据，需要将其作为 numpy 数组导入。因为 Titanic 数据包含文本数据，所以我们需要先做一些预处理:

这是我为自己的泰坦尼克号解决方案所做的同样的预处理和特征工程。当然，每个项目的预处理是不同的，但是如果您想要使用 Auto-Keras，您将需要提供 numpy 数组。

预处理后，您可以将数据作为训练和测试数据进行加载:

## 寻找合适的模型

当你的数据集有了 *x_train* 、 *y_train* 、 *x_test* 和 *y_test* 之后，你可以使用 *fit* 方法找到最佳模型并训练它:

就是这样——两行代码就是您所需要的全部。其他一切都是自动发生的。

# 结果

训练完模型后，我用它来生成预测，并将预测上传到 Kaggle。我的分数是 0.79904。这比我的手工模型略差，但我们需要考虑到 Auto-Keras 仍然很新。TabularClassifier 目前仅支持 LGBMClassifier，这对于该数据集来说并不理想。如果这个项目将来继续下去，我们可能会有更好的结果。此外，找到模型非常快速和容易，所以我认为这是一个非常好的和有希望的结果！

## 利弊

虽然使用生成的模型来生成预测非常容易，但是您会丢失许多关于模型的知识。如果你不主动搜索，你就不知道结果模型的架构和参数值——毕竟它们是自动选择的，所以你永远不会真正需要它们。虽然这对初学者和没有经验的机器学习工程师来说可能是一件好事，但如果你后来试图适应或扩展你的模型，这可能会变得危险。

“黑盒”模型已经在机器学习项目中进行了大量讨论，但当使用 AutoML 时,*整个过程*甚至变得更像是一个黑盒——你只需扔进一些数据，然后期待最好的结果。

在你的机器学习项目中实现 AutoML 时，这些是你需要注意的一些风险。

## 对未来工作的意义

那么，AutoML 是否消除了对定制模型的需求？不，可能不会。
它能帮助你创造更好的模型吗？大概吧。
例如，您可以使用它快速找到并训练好的基础模型，稍后您可以自己对其进行改进。您还可以快速尝试不同的特性工程策略，以获得一种最佳效果的感觉。试一试，看看如何在自己的项目中实现。

我认为 AutoML 可以在未来对许多机器学习项目产生重大影响——我期待着这一领域的进一步发展和改进。