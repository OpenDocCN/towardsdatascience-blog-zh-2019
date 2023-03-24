# 用于图像识别的深度学习:具有 Tensorflow 和 Keras 的卷积神经网络

> 原文：<https://towardsdatascience.com/deep-learning-for-image-recognition-convolutional-neural-network-with-tensorflow-de6349c31c07?source=collection_archive---------14----------------------->

![](img/485828c0cbf403b42a0c9fd2e991a6fa.png)

## 使用 Python 从头开始构建您的神经网络

深度学习是机器学习的一个子集，其算法基于人工神经网络中使用的层。它有各种各样的应用，其中包括图像识别，这就是我们在这篇文章中要讨论的。

为了展示如何构建、训练和预测你的神经网络，我将使用 Tensorflow，你可以通过 *pip* 在你的 Jupyter 笔记本上轻松运行它。

```
import tensorflow as tf
from tensorflow import keras
import matplotlib.pyplot as plt
import numpy as np
import pydot_ng as pydot
```

然后，我将使用 Keras(一个可以从 Tensorflow 导入的开源包)库中可用的数据集之一，其中包含一组我们的算法将被训练的图像。让我们下载并先看看我们的数据:

```
from keras.datasets import cifar10

*#I'm dividing my data into training and test set*

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

x_train.shape, x_test.shape, y_train.shape, y_test.shape
```

![](img/eab7415e725d83d3605b3195076c3d7d.png)

正如你所看到的，在我们的 *x* 训练集中，我们有 50000 幅图像，每幅 32×32 像素，有 3 个通道(与 *x* 测试集相同，但只有 10000 个观察值)。另一方面，我们的 *y* 集合是从 0 到 9 的数字数组，对应于我们的类。所以我们可以从创建一个相应类别的向量开始，然后分配给我们的预测。此外，我们还可以再设置两个变量:

*   **时期**:训练时我们的神经网络的迭代次数
*   **batch_size** :我们希望每个时期使用的样本数量

```
batch_size=32
epochs=3
class_names = ["airplane","automobile","bird","cat","deer","dog","frog","horse","ship","truck"]
```

因为我首先想以一种非常直观的方式展示这个过程，所以我将对我的图像进行调整，从 3 通道图像调整到 1 通道图像(也就是说，从彩色到黑白)。通过这样做，我们将能够可视化卷积、汇集和完全连接的整个过程(稍后我将解释这三个步骤)。

我们将在这项任务中使用的算法是卷积神经网络。我不打算在它背后的数学上花太多时间，但是我想提供一个关于它如何工作的直观想法。

所以，让我们一起来看看这张图片和下面的脚本(我将一步一步地解释)。在图中，我检查了一个简单的任务:我们有一个由四个字母组成的字母表——A、B、C 和 D——我们的算法被要求识别我们的输入字母(在我们的例子中，是“C”)。另一方面，脚本中构建的算法引用了我们的数据集，因此输出向量将有十个而不是四个条目。然而，底层过程是相同的。

![](img/81f2cf1b0cdc7487ec4db25fcdaaae8b.png)

```
model=tf.keras.Sequential() model.add(tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(32,32,1))) model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2))) model.add(tf.keras.layers.Flatten()) model.add(tf.keras.layers.Dense(1024,activation='relu')) model.add(tf.keras.layers.Dense(10,activation='softmax'))
```

让我们解释每一段:

*   **卷积**:图像由一个过滤器检查，它能够将输入图像分割成更小的部分，返回一个所谓的特征图(更准确地说，它返回的特征图与使用的过滤器数量一样多)。为了得到这个过滤过程的输出，我们需要一个所谓的激活函数。激活函数映射结果值，即在 0 到 1 或-1 到 1 之间，依此类推(取决于函数)。在我们的例子中，ReLU 函数简单地将任何负值归零；
*   **池化**:这一步的主要目标是通过一个函数减少我们的特征图的大小(在这种情况下，我们使用了一个‘max’函数:它返回被检查的那些中最高的像素值)；
*   **全连接**:全连接层的目的是使用那些特征，根据训练数据集将输入图像分类成各种类别。在这一步中，在将我们的矩阵形状的图像转换成数字数组之后，我们再次应用一个激活函数，然后获得一个概率向量作为最终输出，该向量与类的向量一样长。事实上，我们使用的激活函数，称为“softmax”，将输入转换成一个概率范围。范围从 0 到 1，当然，所有概率的总和等于 1。在我们的例子中，由于它是一个多分类任务，这个函数返回每个类的概率，目标类将具有最高的概率。

我们 CNN 的主要目标是产生一个尽可能接近真实的预测结果。此外，该算法一旦被评估，就能够通过重新加权一些参数并最小化误差项来从其过去的路径中学习。这种操作称为反向传播。

为了实施，反向传播需要另外两个要素:

*   **损失函数**:衡量模型的一致性。当拟合值远离实际值时，它返回较大的值。在线性回归模型中广泛使用的典型损失函数是均方误差(MSE)。在我们的例子中，我们将使用分类交叉熵函数。
*   **优化器**:为了最小化误差，需要修改权重，我们可以通过使用一类称为优化函数的函数来实现。优化函数通常计算损失函数相对于权重的偏导数(称为梯度)，并且权重在计算的梯度的相反方向上被修改。重复这个循环，直到我们达到损失函数的最小值。在我们的例子中，我们将使用 Adam 优化器。

```
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001,decay=1e-6), metrics=['accuracy'])
```

请注意，参数“度量”作为损失函数，是衡量模型性能的一种方式。但是，它不会影响训练过程(而损失函数会影响)。

现在，我们可以训练我们的模型，并在我们的测试集上验证它(总是将其重塑为单通道数据集):

```
*#I'm dividing my train and test set by 255, since I want to normalize the value of each pixel (ranging from 0 to 255)*
model.fit(np.resize(x_train, (50000,32,32,1))/255.0,tf.keras.utils.to_categorical(y_train),
         batch_size=batch_size,
         shuffle=True,
         epochs=epochs,
         validation_data=(np.resize(x_test, (10000,32,32,1))/255.0,tf.keras.utils.to_categorical(y_test))
         )
```

请原谅这个模型这么差(准确率达到了 10.74%的尴尬值)，但是我要求只迭代 3 次，减少通道。事实上，第一种方法的目的只是可视化这个过程。

```
keras.utils.plot_model(model,show_shapes=True)
```

![](img/4d1467ca8f2f432a56dcf30bf816812f.png)

首先，我们将 3×3 滤波器应用于输入图像。该操作返回 32 个特征地图，每个 30×30 像素。然后，我们使用大小为 2×2 的池过滤器将图像的尺寸从 30×30 减小到 15×15(这意味着，我们的池过滤器一次将检查四个像素，仅返回等于其评估的最大值的一个像素)。

![](img/096d7a81c3302d2d533e1a73a841a1e7.png)

现在让我们用 3 通道图像训练模型。首先，我们可以看看我们将要分析的图像类型:

```
plt.imshow(x_train[12]) 
plt.title(class*_names[y_*train[12:13][0][0]])
```

![](img/1a858b704e35a18f428ad5bf4966e100.png)

现在，让我们再次在 3 通道图像上训练我们的神经网络(添加一些修改):

```
model.add(tf.keras.layers.Conv2D(32,kernel_size=(3,3),activation='relu',input_shape=(32,32,3))) model.add(tf.keras.layers.MaxPooling2D(pool_size=(2,2))) 
*#I'm adding two Dropout layers to prevent overfitting* model.add(tf.keras.layers.Dropout(0.25)) model.add(tf.keras.layers.Flatten()) model.add(tf.keras.layers.Dense(1024,activation='relu')) model.add(tf.keras.layers.Dropout(0.5)) model.add(tf.keras.layers.Dense(10,activation='softmax'))
```

让我们编译它，记住上面关于损失函数和优化器的所有考虑:

```
model.compile(loss='categorical_crossentropy', optimizer=tf.keras.optimizers.Adam(lr=0.001,decay=1e-6), metrics=['accuracy']) 
model.fit(x_train/255.0,tf.keras.utils.to_categorical(y_train), batch_size=batch_size, shuffle=True, epochs=epochs, validation_data=(x_test/255.0,tf.keras.utils.to_categorical(y_test)) )
```

![](img/b7fca9c9fda1553f0c0eb817d7332130.png)

我们现在可以在验证测试中对其进行评估(并对其进行预测):

```
predictions=model.predict(x_test) 
scores = model.evaluate(x_test / 255.0, tf.keras.utils.to_categorical(y_test))
```

![](img/e761c76678952af3fb6d9cb59d404a1a.png)

不错，准确率提高到 63.87%。让我们比较一些预测:

```
*#I'm defining a function that plot my predicted image, with true #label as title* 
def plot_pred(i,predictions_array,true_label,img):
    predictions_array,true_label,img=predictions_array[i],true_label[i:i+1],img[i]
    plt.grid(False)
    plt.title(class_names[true_label[0][0]])
    plt.xticks([])
    plt.yticks([])

    plt.imshow(img)

*#I'm defining a function that plot my prediction vector, showing #whether my*
*#predicted value is correct (in blue) or incorrect (in red)*

def plot_bar(i,predictions_array,true_label):
    predictions_array, true_label = predictions_array[i], true_label[i]
    plt.grid(False)
    plt.yticks([])
    plt.xticks(np.arange(10),class_names,rotation=40)

    thisplot=plt.bar(range(10),predictions_array, color='grey')
    plt.ylim([0,1])
    predicted_label=np.argmax(predictions_array)

    if predicted_label==true_label:
        color='blue'
    else:
        color='red'

    thisplot[predicted_label].set_color(color)

*#plotting both the images*

plt.figure(figsize=(15,6))
plt.subplot(1,2,1)
plot_pred(10, predictions, y_test, x_test)
plt.subplot(1,2,2)
plot_bar(10, predictions,  y_test)
plt.show()
plt.imshow(img)
```

![](img/0e6cc80838812f95b9de1b70f88d89ef.png)

现在让我们看看当预测错误时会发生什么:

```
plt.figure(figsize=(15,6)) 
plt.subplot(1,2,1) 
plot_pred(20, predictions, y_test, x_test) 
plt.subplot(1,2,2) 
plot_bar(20, predictions, y_test) 
plt.show()
```

![](img/c079efcf01f34d280ebc471324c03c9a.png)

Tensorflow 和 Keras 的好处是，你可以建立自己的“自制”CNN，根据可用数据改变层数/类型。然后，您的模型将自己进行反向传播，以调整其参数并最小化误差项。

嗯，我们在这里建立的 CNN 很简单，数据量适中，然而我们能够建立一个很好的分类器，我们可能希望用新的层来实现它。

我想通过分享一种非常直观的方式来了解 CNN 实际上是如何工作的，通过自己喂它，并将学习的整个过程可视化(如果你想尝试，请单击下面的链接)，来结束这篇文章。

![](img/4827ac9df3352632b6669b779f4d8650.png)

来源:http://scs.ryerson.ca/~aharley/vis/conv/flat.html

**参考文献:**

*   【https://www.tensorflow.org/ 

*原载于 2019 年 7 月 10 日*[*http://datasciencechalktalk.com*](https://datasciencechalktalk.com/2019/07/10/deep-learning-for-image-recognition-convolutional-neural-network-with-tensorflow/)*。*