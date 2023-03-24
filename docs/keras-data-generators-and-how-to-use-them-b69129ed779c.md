# Keras 数据生成器及其使用方法

> 原文：<https://towardsdatascience.com/keras-data-generators-and-how-to-use-them-b69129ed779c?source=collection_archive---------0----------------------->

![](img/4c1ce85c46c5bf46dd2a18ea95689957.png)

您可能遇到过这样的情况:您试图加载一个数据集，但是您的计算机没有足够的内存。随着机器学习领域的进步，这个问题变得越来越普遍。今天，这已经是视觉领域中的挑战之一，在视觉领域中，要处理大数据集的图像和视频文件。

在这里，我们将重点讨论如何构建数据生成器，以便在 Keras 中加载和处理图像。

# 数据生成器的功能是什么

在 Keras 模型类中，有三个方法让我们感兴趣:fit_generator、evaluate_generator 和 predict_generator。这三者都需要数据生成器，但并非所有生成器都是同等创建的。

让我们看看每种方法需要哪种生成器:

## 拟合 _ 生成器

需要两个生成器，一个用于训练数据，另一个用于验证。幸运的是，它们都应该返回一个元组(输入，目标),并且它们都可以是 Sequence 类的实例。

## 评估 _ 生成器

这里的数据生成器与 fit_generator 中的要求相同，并且可以与训练生成器相同。

## 预测生成器

这里的发电机有点不同。它应该只返回输入。

记住这一点，让我们构建一些数据生成器。由于 fit_generator 和 evaluate_generator 中的生成器相似，我们将重点构建 fit_generator 和 predict_generator 的数据生成器。

# ImageDataGenerator 类

ImageDataGenerator 类在图像分类中非常有用。有几种方法可以使用这个生成器，取决于我们使用的方法，这里我们将重点介绍 flow_from_directory 取一个路径到包含在子目录中排序的图像和图像增强参数的目录。

让我们看一个例子:

我们将使用可从[https://www.kaggle.com/c/dogs-vs-cats/data](https://www.kaggle.com/c/dogs-vs-cats/data)下载的数据集，其结构如下:

```
data/
    train/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
    validation/
        dogs/
            dog001.jpg
            dog002.jpg
            ...
        cats/
            cat001.jpg
            cat002.jpg
            ...
```

首先，让我们导入所有必要的库，并创建一个带有一些图像增强的数据生成器。

最后，创建一个模型并运行 fit_generator 方法。

ImageDataGenerator 是一种为图像分类任务批量加载和扩充图像的简单方法。但是！如果你有一个分段任务呢？为此，我们需要构建一个定制的数据生成器。

# 灵活的数据生成器

要构建自定义数据生成器，我们需要从 Sequence 类继承。让我们这样做，并添加我们需要的参数。

Sequence 类迫使我们实现两个方法；__len__ 和 __getitem__。如果我们想让生成器在每个纪元后做一些事情，我们也可以在 _epoch_end 上实现这个方法。

__len__ 方法应该返回每个时期的批数。一种可能的实现如下所示。

如果 shuffle=True，此示例中的 on_epoch_end 可以对训练的索引进行混洗。但是这里可以有任何我们想要在每个纪元后运行的逻辑。

我们必须实现的第二个方法是 __getitem__ 它完全符合您的预期。如果我们进行预测，它应该会返回一批图像和遮罩。这可以通过将 to_fit 设置为 True 或 False 来控制。

整个数据生成器应该如下所示:

假设我们有两个目录，一个保存图像，另一个保存蒙版图像，并且每个图像都有一个同名的相应蒙版，下面的代码将使用自定义数据生成器为模型定型。

最后，如果我们想使用数据生成器进行预测，应该将 to_fit 设置为 False，并调用 predict_generator。

# 结论

虽然 Keras 提供了数据生成器，但是它们的能力有限。原因之一是每个任务都需要不同的数据加载器。有时每个图像有一个遮罩，有时有几个，有时遮罩被保存为图像，有时被编码，等等…

对于每个任务，我们可能需要调整我们的数据生成器，但结构将保持不变。

# 参考链接

## 斯坦福大学的 Keras 如何使用数据生成器的详细示例

[](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly) [## 使用 Keras 的数据生成器的详细示例

### python keras 2 fit _ generator Afshine Amidi 和 Shervine Amidi 的大型数据集多重处理您是否曾经不得不…

stanford.edu](https://stanford.edu/~shervine/blog/keras-how-to-generate-data-on-the-fly) 

## Keras 模型类

[](https://keras.io/models/model/) [## 模型(功能 API) - Keras 文档

### 在函数式 API 中，给定一些输入张量和输出张量，您可以通过以下方式实例化一个模型:from…

keras.io](https://keras.io/models/model/) 

## Keras ImageDataGenerator 类

[](https://keras.io/preprocessing/image/) [## 图像预处理

### keras . preprocessing . image . image data generator(feature wise _ center = False，samplewise_center=False…

keras.io](https://keras.io/preprocessing/image/)