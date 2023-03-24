# TensorFlow 2.0 中的实用编码

> 原文：<https://towardsdatascience.com/practical-coding-in-tensorflow-2-0-1aab32bcfde1?source=collection_archive---------15----------------------->

## tf.function、TensorArray 和高级控制流

![](img/43cdcf7dc81d063503a2ec4a8a52203c.png)

An [image](https://pixabay.com/illustrations/instructions-user-manual-76729/) by [Gerd Altmann](https://pixabay.com/users/geralt-9301/)

## 摘要

与 PyTorch 的激烈竞争给我们带来了 TensorFlow (TF)的新版本。包经历了很多变化，但最关键的是`session.run()`的退役。默认情况下，TF 2 使用的是 eager 模式，而不是我们熟悉的构建和执行静态图的模式。这种代码可以用 pythonic 的方式编写，并转换成计算图。为了以静态图的形式执行代码，开发人员必须用`@tf.function`修饰想要的函数。

在这篇文章中，我将用例子来解释这些概念。我假设读者了解 Python 和机器学习(ML)的基础知识。如果你是这个领域的新手，欢迎！这个[课程](https://www.coursera.org/specializations/deep-learning)将会是一个很好的开始。在这里，你可以找到这篇文章的 colab 版本。

## 装置

要安装 TF 2.x，请访问这个[页面](https://www.tensorflow.org/install)。

要检查您的当前版本:

```
import tensorflow as tf
print(tf.__version__)
```

## 急切的执行

代码在 TF 2 中急切地执行。这意味着您可以在不调用`session.run()`或使用占位符的情况下向计算图提供数据。计算图是定义一系列操作的结构。这种结构允许我们通过沿着图形向后移动来自动计算导数。观看此[视频](https://www.youtube.com/watch?v=hCP1vGoCdYU)了解更多细节。在 TF 1 中，开发人员必须创建一个图形，然后执行它。现在图形是动态构建的，执行类似于函数调用。

为了了解它的工作原理，让我们做一个简单的模型。例如，我们有一个包含 3 个训练示例的数据集，其中每个示例都是一个二维向量。

```
import numpy as npnp.random.seed(0)
data = np.random.randn(3, 2)
```

首先，我们必须初始化变量。TF 2 中没有变量范围。因此，保持所有变量在一个计数中的最佳方式是使用 Keras 层。

```
inputer = tf.keras.layers.InputLayer(input_shape=(2))denser1 = tf.keras.layers.Dense(4, activation='relu')denser2 = tf.keras.layers.Dense(1, activation='sigmoid')
```

然后我们可以定义一个简单的模型。我们可以仅仅通过调用函数来运行这个模型。在这里，数据进入具有 4 个隐藏单元的密集层，然后进入具有一个单元的最终层。

```
def model_1(data):x = inputer(data)
  x = denser1(x)
  print('After the first layer:', x)
  out = denser2(x)
  print('After the second layer:', out)return outprint(‘Model\’s output:’, model(data))...
After the first layer: tf.Tensor( 
[[0.9548421  0\.         0\.         1.4861959 ]  
 [1.3276602  0.18780036 0.50857764 0\.        ]  
 [0.45720425 0\.         0\.         2.5268495 ]], shape=(3, 4), dtype=float32) 
After the second layer: tf.Tensor( 
[[0.27915245]  
 [0.31461754]  
 [0.39550844]], shape=(3, 1), dtype=float32) 
Model's output: tf.Tensor( 
[[0.27915245]  
 [0.31461754] 
 [0.39550844]], shape=(3, 1), dtype=float32)
```

要从张量中获取 numpy 数组:

```
print('Model\'s output:', model_1(data).numpy())...
Model's output: [[0.27915245]  [0.31461754]  [0.39550844]]
```

然而，急切的执行可能会很慢。该图是动态计算的。也就是说，让我们看看它是如何在我们的模型中构建的。输入数据进入第一层，这是第一个节点。当添加第二个节点时，第一个节点的输出进入第二个节点，然后计算第二个节点的输出，以此类推。它允许我们打印模型的中间状态(就像我们在上面的例子中所做的那样)，但是会使计算变慢。

## 静态图

幸运的是，我们仍然可以通过用`@tf.function`修饰模型来构建一个静态图。与动态图相反，静态图首先连接所有节点进行一个大的计算操作，然后执行它。因此，我们不能看到模型的中间状态，也不能动态地添加任何节点。

```
@tf.function
def model_2(data):
  x = inputer(data)
  x = denser1(x)
  print('After the first layer:', x)
  out = denser2(x)
  print('After the second layer:', out)

  return outprint('Model\'s output:', model_2(data))...
After the first layer: Tensor("dense_12/Relu:0", shape=(3, 4), dtype=float32)
After the second layer: Tensor("dense_13/Sigmoid:0", shape=(3, 1), dtype=float32)
Model's output: tf.Tensor(
[[0.27915245] 
 [0.31461754] 
 [0.39550844]], shape=(3, 1), dtype=float32)
```

第二个优点是静态图只构建一次，而动态图在每次模型调用后都要重新构建。当您重复使用同一个图形时，它会降低计算速度。例如，当您在培训期间重新计算批次损失时。

```
for i, d in enumerate(data):
  print('batch:', i)
  model_1(d[np.newaxis, :])  # eager modelfor i, d in enumerate(data):
  print('batch:', i)
  model_2(d[np.newaxis, :])  # static model...
batch: 0
After the first layer: tf.Tensor(
[[0.9548421 0\.        0\.        1.486196 ]], shape=(1, 4), dtype=float32)
After the second layer: tf.Tensor(
[[0.27915245]], shape=(1, 1), dtype=float32)batch: 1
After the first layer: tf.Tensor(
[[1.3276603  0.18780035 0.50857764 0\.        ]], shape=(1, 4), dtype=float32)
After the second layer: tf.Tensor(
[[0.3146175]], shape=(1, 1), dtype=float32)batch: 2
After the first layer: tf.Tensor(
[[0.45720425 0\.         0\.         2.5268495 ]], shape=(1, 4), dtype=float32)
After the second layer: tf.Tensor(
[[0.39550844]], shape=(1, 1), dtype=float32)batch: 0
After the first layer: Tensor("dense_12/Relu:0", shape=(1, 4), dtype=float32)
After the second layer: Tensor("dense_13/Sigmoid:0", shape=(1, 1), dtype=float32)batch: 1batch: 2
```

内部打印只在图形构建期间调用，在第二种情况下，图形只构建一次，然后重用。对于大型数据集，时间上的差异可能是巨大的。

## 高级控制流程

AutoGraph 简化了 if/else 语句和 for/while 循环的使用。与 TF 1 相反，现在它们可以用 python 语法来编写。例如:

```
a = np.array([1, 2, 3], np.int32)@tf.function
def foo(a):
  b = tf.TensorArray(tf.string, 4)
  b = b.write(0, "test") for i in tf.range(3):
    if a[i] == 2:
      b = b.write(i, "fuzz")
    elif a[i] == 3:
      b = b.write(i, "buzz")return b.stack()...
tf.Tensor([b'test' b'fuzz' b'buzz' b''], shape=(4,), dtype=string)
```

现在数组的使用类似于 Java 中的数组。首先，用所需的数据类型和长度声明一个数组:

```
tf.TensorArray(data_type, length)
```

常见的数据类型有`tf.int32, tf.float32, tf.string`。要将数组`b`转换回张量，使用`b.stack()`。