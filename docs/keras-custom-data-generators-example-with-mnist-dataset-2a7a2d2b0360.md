# MNIST 数据集的 Keras 自定义数据生成器示例

> 原文：<https://towardsdatascience.com/keras-custom-data-generators-example-with-mnist-dataset-2a7a2d2b0360?source=collection_archive---------14----------------------->

![](img/27f18382048e56cae4074b29515cd98b.png)

Photo by [Carlos Muza](https://unsplash.com/@kmuza?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/data-generator?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

通常，在现实世界的问题中，用于训练我们模型的数据集占用的内存比我们在 RAM 中的要多得多。问题是我们不能将整个数据集加载到内存中，并使用标准的 keras *fit* 方法来训练我们的模型。

解决这个问题的一种方法是只把一批数据装入内存，然后把它输入网络。重复这个过程，直到我们用所有数据集训练了网络。然后我们打乱所有的数据集，重新开始。

为了制作自定义生成器，keras 为我们提供了一个序列类。这个类是抽象的，我们可以创建继承它的类。

我们将编码一个自定义数据生成器，该生成器将用于生成 MNIST 数据集的批量样本。

首先，我们将导入 python 库:

```
**import** tensorflow **as** tf
**import** os
**import** tensorflow.keras **as** keras
**from** tensorflow.keras.models **import** Sequential
**from** tensorflow.keras.layers **import** Dense, Dropout, Flatten
**from** tensorflow.keras.layers **import** Conv2D, MaxPooling2D
**import** numpy **as** np
**import** math
```

然后，我们将 MNIST 数据集加载到 RAM 内存中:

```
mnist = tf.keras.datasets.mnist(x_train, y_train), (x_test, y_test) = mnist.load_data()
```

MNIST 数据集由 60000 幅手写数字训练图像和 10000 幅测试图像组成。

每个图像尺寸为 28×28 像素。你应该考虑到，为了训练模型，我们必须将 *uint8* 数据转换为 *float32。*float 32 中的每个像素需要 4 *字节*的内存。

因此，整个数据集需要:

每像素 4 字节* (28 * 28)每图像像素* 70000 个图像+ (70000*10)个标签。

总共 220 Mb 的内存完全可以放在 RAM 内存中，但在现实世界的问题中，我们可能需要更多的内存。

我们的生成器模拟生成器将从 RAM 中加载图像，但在实际问题中，它们将从硬盘中加载。

```
class **DataGenerator**(tf.compat.v2.keras.utils.Sequence):

    def **__init__**(self, X_data , y_data, batch_size, dim, n_classes,
                 to_fit, shuffle = True): self.batch_size = batch_size
        self.X_data = X_data
        self.labels = y_data
        self.y_data = y_data
        self.to_fit = to_fit
        self.n_classes = n_classes
        self.dim = dim
        self.shuffle = shuffle
        self.n = 0
        self.list_IDs = np.arange(len(self.X_data))
        self.on_epoch_end() def **__next__**(self):
        # Get one batch of data
        data = self.__getitem__(self.n)
        # Batch index
        self.n += 1

        # If we have processed the entire dataset then
        if self.n >= self.__len__():
            self.on_epoch_end
            self.n = 0

        **return** data def **__len__**(self):
        # Return the number of batches of the dataset
        **return** math.ceil(len(self.indexes)/self.batch_size) def **__getitem__**(self, index):
        # Generate indexes of the batch
        indexes = self.indexes[index*self.batch_size:
            (index+1)*self.batch_size] # Find list of IDs
        list_IDs_temp = [self.list_IDs[k] for k in indexes]

        X = self._generate_x(list_IDs_temp)

        if self.to_fit:
            y = self._generate_y(list_IDs_temp)
            **return** X, y
        else:
            **return** X def **on_epoch_end**(self):

        self.indexes = np.arange(len(self.X_data))

        if self.shuffle: 
            np.random.shuffle(self.indexes) def **_generate_x**(self, list_IDs_temp):

        X = np.empty((self.batch_size, *self.dim))

        for i, ID in enumerate(list_IDs_temp):

            X[i,] = self.X_data[ID]

            # Normalize data
            X = (X/255).astype('float32')

        **return** X[:,:,:, np.newaxis] def **_generate_y**(self, list_IDs_temp):

        y = np.empty(self.batch_size)

        for i, ID in enumerate(list_IDs_temp):

            y[i] = self.y_data[ID]

        **return** keras.utils.to_categorical(
                y,num_classes=self.n_classes)
```

然后我们要建立分类网络:

```
n_classes = 10
input_shape = (28, 28)model = Sequential()
model.add(Conv2D(32, kernel_size=(3, 3),
                 activation='relu',
                 input_shape=(28, 28 , 1)))
model.add(Conv2D(64, (3, 3), activation='relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))
model.add(Dropout(0.25))
model.add(Flatten())
model.add(Dense(128, activation='relu'))
model.add(Dropout(0.5))
model.add(Dense(n_classes, activation='softmax'))model.compile(loss=keras.losses.categorical_crossentropy,
              optimizer=keras.optimizers.Adadelta(),
              metrics=['accuracy'])
```

下一步是制作我们的生成器的一个实例:

```
train_generator = DataGenerator(x_train, y_train, batch_size = 64,
                                dim = input_shape,
                                n_classes=10, 
                                to_fit=True, shuffle=True)val_generator =  DataGenerator(x_test, y_test, batch_size=64, 
                               dim = input_shape, 
                               n_classes= n_classes, 
                               to_fit=True, shuffle=True)
```

如果我们想检查生成器是否正常工作，我们可以调用产生一批样本和标签的 *next()* 方法。然后检查图像和标签的数据类型是否正确，检查批次的尺寸等…

```
images, labels = next(train_generator)
print(images.shape)
print(labels.shape)
```

如果我们希望在一个时期内将整个数据集输入网络:

```
steps_per_epoch = len(train_generator)
validation_steps = len(val_generator)
```

最后，我们将使用 keras 函数 *fit_generator()来训练网络。*

```
model.fit_generator(
        train_generator,
        steps_per_epoch=steps_per_epoch,
        epochs=10,
        validation_data=val_generator,
        validation_steps=validation_steps)
```

感谢阅读这篇文章。希望你觉得有用。