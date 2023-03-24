# 使用 NumPy 的人工神经网络实现和水果 360 图像数据集的分类

> 原文：<https://towardsdatascience.com/artificial-neural-network-implementation-using-numpy-and-classification-of-the-fruits360-image-3c56affa4491?source=collection_archive---------9----------------------->

本教程使用 NumPy 从头开始在 Python 中构建人工神经网络，以便对 Fruits360 数据集进行图像分类应用。本教程中使用的所有内容(即图像和源代码)，而不是 color Fruits360 图像，都是我的书的专有权利，该书被称为“**Ahmed Fawzy Gad‘使用深度学习的实用计算机视觉应用与 CNN’。2018 年 12 月，新闻，978–1–4842–4167–7**。这本书在斯普林格书店有售，链接:【https://springer.com/us/book/9781484241660[。](https://springer.com/us/book/9781484241660)

![](img/b37b3ac062c3bf516185778bce9941d4.png)

本教程中使用的源代码可以在我的 GitHub 页面上找到:https://github.com/ahmedfgad/NumPyANN

书中使用的示例是关于使用人工神经网络(ANN)对 Fruits360 图像数据集进行分类。该示例没有假设读者既没有提取特征也没有实现 ANN，因为它讨论了什么是适合使用的特征集以及如何从头开始在 NumPy 中实现 ANN。Fruits360 数据集包含 60 类水果，如苹果、番石榴、鳄梨、香蕉、樱桃、枣、猕猴桃、桃子等。为了使事情更简单，它只对 4 个选定的类有效，它们是苹果，柠檬，芒果和覆盆子。每个类有大约 491 幅图像用于训练，另外 162 幅用于测试。图像大小为 100x100 像素。

**特征提取**

这本书从选择合适的特征集开始，以达到最高的分类精度。根据下面显示的 4 个选定类别的样本图像，它们的颜色似乎有所不同。这就是为什么颜色特征适合在这个任务中使用。

![](img/f3666ffe4b11b22078a30a860901f4d9.png)

RGB 颜色空间不会将颜色信息与其他类型的信息(如照明)隔离开来。因此，如果 RGB 用于表示图像，则 3 个通道将参与计算。因此，最好使用将颜色信息隔离到单一通道的颜色空间，如 HSV。这种情况下的颜色通道是色调通道(H)。下图显示了之前提供的 4 个样本的色调通道。我们可以注意到每个图像的色调值与其他图像的不同。

![](img/e947becf986f40babb68aeedb932ee01.png)

色调通道尺寸仍然是 100x100。如果将整个通道应用于 ANN，那么输入层将有 10，000 个神经元。网络还是很庞大的。为了减少使用的数据量，我们可以使用直方图来表示色调通道。直方图将具有 360 个面元，反映色调值的可能值的数量。以下是 4 幅样本图像的直方图。使用色调通道的 360 格直方图，似乎每个水果都投票给直方图的一些特定格。与使用 RGB 颜色空间中的任何通道相比，不同类别之间的重叠更少。例如，苹果直方图的区间范围是 0 到 10，而芒果直方图的区间范围是 90 到 110。每个类别之间的余量使得更容易减少分类中的模糊性，从而提高预测精度。

![](img/9bbd17777f8cde817a3f7d95a6b6523a.png)

下面是从 4 幅图像中计算色调通道直方图的代码。

```
**import** numpy
 **import** skimage.io, skimage.color
 **import** matplotlib.pyplot

 raspberry = skimage.io.imread(fname=**"raspberry.jpg"**, as_grey=**False**)
 apple = skimage.io.imread(fname=**"apple.jpg"**, as_grey=**False**)
 mango = skimage.io.imread(fname=**"mango.jpg"**, as_grey=**False**)
 lemon = skimage.io.imread(fname=**"lemon.jpg"**, as_grey=**False**)

 apple_hsv = skimage.color.rgb2hsv(rgb=apple)
 mango_hsv = skimage.color.rgb2hsv(rgb=mango)
 raspberry_hsv = skimage.color.rgb2hsv(rgb=raspberry)
 lemon_hsv = skimage.color.rgb2hsv(rgb=lemon)

 fruits = [**"apple"**, **"raspberry"**, **"mango"**, **"lemon"**]
 hsv_fruits_data = [apple_hsv, raspberry_hsv, mango_hsv, lemon_hsv]
 idx = 0
 **for** hsv_fruit_data **in** hsv_fruits_data:
     fruit = fruits[idx]
     hist = numpy.histogram(a=hsv_fruit_data[:, :, 0], bins=360)
     matplotlib.pyplot.bar(left=numpy.arange(360), height=hist[0])
     matplotlib.pyplot.savefig(fruit+**"-hue-histogram.jpg"**, bbox_inches=**"tight"**)
     matplotlib.pyplot.close(**"all"**)
     idx = idx + 1
```

通过循环使用 4 个图像类中的所有图像，我们可以从所有图像中提取特征。接下来的代码会这样做。根据 4 个类中的图像数量(1962)和从每个图像中提取的特征向量长度(360)，创建一个 NumPy 个零数组，并保存在 **dataset_features** 变量中。为了存储每个图像的类标签，创建了另一个名为**输出**的 NumPy 数组。苹果的分类标签是 0，柠檬是 1，芒果是 2，覆盆子是 3。代码期望它运行在一个根目录下，这个根目录下有 4 个文件夹，它们是根据名为**水果**的列表中列出的水果名称命名的。它遍历所有文件夹中的所有图像，从每个图像中提取色调直方图，为每个图像分配一个类别标签，最后使用 pickle 库保存提取的特征和类别标签。您还可以使用 NumPy 来保存生成的 NumPy 数组，而不是 pickle。

```
**import** numpy
**import** skimage.io, skimage.color, skimage.feature
**import** os
**import** pickle

fruits = [**"apple"**, **"raspberry"**, **"mango"**, **"lemon"**] *#492+490+490+490=1,962* dataset_features = numpy.zeros(shape=(1962, 360))
outputs = numpy.zeros(shape=(1962))
idx = 0
class_label = 0

**for** fruit_dir **in** fruits:
    curr_dir = os.path.join(os.path.sep, fruit_dir)
    all_imgs = os.listdir(os.getcwd()+curr_dir)
    **for** img_file **in** all_imgs:
        fruit_data = skimage.io.imread(fname=os.getcwd()+curr_dir+img_file, as_grey=**False**)
        fruit_data_hsv = skimage.color.rgb2hsv(rgb=fruit_data)
        hist = numpy.histogram(a=fruit_data_hsv[:, :, 0], bins=360)
        dataset_features[idx, :] = hist[0]
        outputs[idx] = class_label
        idx = idx + 1
    class_label = class_label + 1

**with** open(**"dataset_features.pkl"**, **"wb"**) **as** f:
    pickle.dump(**"dataset_features.pkl"**, f)

**with** open(**"outputs.pkl"**, **"wb"**) **as** f:
    pickle.dump(outputs, f)
```

目前，每个图像使用 360 个元素的特征向量来表示。这样的元素被过滤，以便仅保留用于区分 4 个类别的最相关的元素。减少的特征向量长度是 102 而不是 360。使用更少的元素有助于比以前做更快的训练。 **dataset_features** 变量形状将是 **1962x102** 。你可以在书中读到更多关于减少特征向量长度的内容。

至此，训练数据(特征和类标签)都准备好了。接下来是使用 NumPy 实现 ANN。

**安实施**

下图显示了目标人工神经网络结构。有一个输入层有 102 个输入，2 个隐藏层有 150 和 60 个神经元，一个输出层有 4 个输出(每个水果类一个)。

![](img/a2fb2fece8a7483990fe2296e37c1802.png)

任一层的输入向量乘以(矩阵乘法)连接到下一层的权重矩阵，以产生输出向量。这种输出向量再次乘以连接其层和下一层的权重矩阵。该过程一直持续到到达输出层。下图是矩阵乘法的总结。

![](img/d14d4c7b5c3702efa383a9e2656cb54f.png)

大小为 1×102 的输入向量要乘以大小为 102×150 的第一个隐藏层的权重矩阵。记住这是矩阵乘法。因此，输出数组形状为 1x150。这样的输出然后被用作第二隐藏层的输入，在那里它被乘以大小为 150x60 的权重矩阵。结果大小为 1x60。最后，这样的输出乘以第二隐藏层和大小为 60×4 的输出层之间的权重。结果最后大小是 1x4。这种结果向量中的每个元素都引用一个输出类。根据具有最高分数的类来标记输入样本。

下面列出了实现这种乘法的 Python 代码。

```
**import** numpy
**import** pickle

**def** sigmoid(inpt):
    **return** 1.0 / (1 + numpy.exp(-1 * inpt))

f = open(**"dataset_features.pkl"**, **"rb"**)
data_inputs2 = pickle.load(f)
f.close()

features_STDs = numpy.std(a=data_inputs2, axis=0)
data_inputs = data_inputs2[:, features_STDs > 50]

f = open(**"outputs.pkl"**, **"rb"**)
data_outputs = pickle.load(f)
f.close()

HL1_neurons = 150
input_HL1_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(data_inputs.shape[1], HL1_neurons))HL2_neurons = 60
HL1_HL2_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(HL1_neurons, HL2_neurons))output_neurons = 4
HL2_output_weights = numpy.random.uniform(low=-0.1, high=0.1, size=(HL2_neurons, output_neurons))H1_outputs = numpy.matmul(a=data_inputs[0, :], b=input_HL1_weights)
H1_outputs = sigmoid(H1_outputs)
H2_outputs = numpy.matmul(a=H1_outputs, b=HL1_HL2_weights)
H2_outputs = sigmoid(H2_outputs)
out_otuputs = numpy.matmul(a=H2_outputs, b=HL2_output_weights)

predicted_label = numpy.where(out_otuputs == numpy.max(out_otuputs))[0][0]
print(**"Predicted class : "**, predicted_label)
```

读取之前保存的要素及其输出标注并过滤要素后，定义图层的权重矩阵。它们被随机赋予从-0.1 到 0.1 的值。例如，变量“ **input_HL1_weights** ”保存输入层和第一个隐藏层之间的权重矩阵。这种矩阵的大小根据特征元素的数量和隐藏层中神经元的数量来定义。

创建权重矩阵后，下一步是应用矩阵乘法。例如，变量“**H1 _ 输出**”保存将给定样本的特征向量乘以输入层和第一个隐藏层之间的权重矩阵的输出。

通常，激活函数被应用于每个隐藏层的输出，以创建输入和输出之间的非线性关系。例如，矩阵乘法的输出被应用于 sigmoid 激活函数。

生成输出层输出后，进行预测。预测的类标签被保存到变量“**预测 _ 标签**”中。对每个输入样本重复这些步骤。下面给出了适用于所有示例的完整代码。

```
**import** numpy
**import** pickle

**def** sigmoid(inpt):
    **return** 1.0 / (1 + numpy.exp(-1 * inpt))

**def** relu(inpt):
    result = inpt
    result[inpt < 0] = 0
    **return** result

**def** update_weights(weights, learning_rate):
    new_weights = weights - learning_rate * weights
    **return** new_weights

**def** train_network(num_iterations, weights, data_inputs, data_outputs, learning_rate, activation=**"relu"**):

    **for** iteration **in** range(num_iterations):
        print(**"Itreation "**, iteration)
        **for** sample_idx **in** range(data_inputs.shape[0]):
            r1 = data_inputs[sample_idx, :]
            **for** idx **in** range(len(weights) - 1):
                curr_weights = weights[idx]
                r1 = numpy.matmul(a=r1, b=curr_weights)
                **if** activation == **"relu"**:
                    r1 = relu(r1)
                **elif** activation == **"sigmoid"**:
                    r1 = sigmoid(r1) curr_weights = weights[-1]
            r1 = numpy.matmul(a=r1, b=curr_weights)
            predicted_label = numpy.where(r1 == numpy.max(r1))[0][0]
            desired_label = data_outputs[sample_idx]
            **if** predicted_label != desired_label:
                weights = update_weights(weights, learning_rate=0.001)
    **return** weights

**def** predict_outputs(weights, data_inputs, activation=**"relu"**):
    predictions = numpy.zeros(shape=(data_inputs.shape[0]))
    **for** sample_idx **in** range(data_inputs.shape[0]):
        r1 = data_inputs[sample_idx, :]
        **for** curr_weights **in** weights:
            r1 = numpy.matmul(a=r1, b=curr_weights)
            **if** activation == **"relu"**:
                r1 = relu(r1)
            **elif** activation == **"sigmoid"**:
                r1 = sigmoid(r1)
        predicted_label = numpy.where(r1 == numpy.max(r1))[0][0]
        predictions[sample_idx] = predicted_label
    **return** predictions

f = open(**"dataset_features.pkl"**, **"rb"**)
data_inputs2 = pickle.load(f)
f.close()features_STDs = numpy.std(a=data_inputs2, axis=0)
data_inputs = data_inputs2[:, features_STDs > 50]

f = open(**"outputs.pkl"**, **"rb"**)
data_outputs = pickle.load(f)
f.close()HL1_neurons = 150
input_HL1_weights = numpy.random.uniform(low=-0.1, high=0.1,size=(data_inputs.shape[1], HL1_neurons))
HL2_neurons = 60
HL1_HL2_weights = numpy.random.uniform(low=-0.1, high=0.1,size=(HL1_neurons,HL2_neurons))
output_neurons = 4
HL2_output_weights = numpy.random.uniform(low=-0.1, high=0.1,size=(HL2_neurons,output_neurons))

weights = numpy.array([input_HL1_weights,
                       HL1_HL2_weights,
                       HL2_output_weights])

weights = train_network(num_iterations=10,
                        weights=weights,
                        data_inputs=data_inputs,
                        data_outputs=data_outputs,
                        learning_rate=0.01,
                        activation=**"relu"**)

predictions = predict_outputs(weights, data_inputs)
num_flase = numpy.where(predictions != data_outputs)[0]
print(**"num_flase "**, num_flase.size)
```

“**权重**”变量包含整个网络的所有权重。基于每个权重矩阵的大小，网络结构被动态地指定。比如“ **input_HL1_weights** ”变量的大小是 102x80，那么我们可以推导出第一个隐层有 80 个神经元。

“ **train_network** ”是核心函数，它通过循环所有样本来训练网络。对于每个示例，都应用了清单 3–6 中讨论的步骤。它接受训练迭代次数、特征、输出标签、权重、学习率和激活函数。激活功能有两个选项，ReLU 或 sigmoid。ReLU 是一个阈值函数，只要它大于零，就返回相同的输入。否则，它返回零。

如果网络对给定的样本做出了错误的预测，则使用“ **update_weights** 函数更新权重。不使用优化算法来更新权重。简单地根据学习率更新权重。准确率不超过 45%。为了获得更好的精度，使用优化算法来更新权重。例如，您可以在 scikit-learn 库的 ANN 实现中找到梯度下降技术。

在我的书中，你可以找到一个使用遗传算法(GA)优化技术优化人工神经网络权重的指南，这种技术可以提高分类的准确性。您可以从我准备的以下资源中了解有关 GA 的更多信息:

**遗传算法优化简介**

[https://www . LinkedIn . com/pulse/introduction-优化-遗传-算法-ahmed-gad/](https://www.linkedin.com/pulse/introduction-optimization-genetic-algorithm-ahmed-gad/)

[https://www . kdnugges . com/2018/03/introduction-optimization-with-genetic-algorithm . html](https://www.kdnuggets.com/2018/03/introduction-optimization-with-genetic-algorithm.html)

[https://towards data science . com/introduction-to-optimization-with-genetic algorithm-2f 5001d 9964 b](/introduction-to-optimization-with-genetic-algorithm-2f5001d9964b)

[https://www.springer.com/us/book/9781484241660](https://www.springer.com/us/book/9781484241660)

**遗传算法(GA)优化—分步示例**

[https://www . slide share . net/AhmedGadFCIT/genetic-algorithm-ga-optimization-step by step-example](https://www.slideshare.net/AhmedGadFCIT/genetic-algorithm-ga-optimization-stepbystep-example)

**遗传算法在 Python 中的实现**

[https://www . LinkedIn . com/pulse/genetic-algorithm-implementation-python-Ahmed-gad/](https://www.linkedin.com/pulse/genetic-algorithm-implementation-python-ahmed-gad/)

[https://www . kdnugges . com/2018/07/genetic-algorithm-implementation-python . html](https://www.kdnuggets.com/2018/07/genetic-algorithm-implementation-python.html)

[https://towardsdatascience . com/genetic-algorithm-implementation-in-python-5ab 67 bb 124 a 6](/genetic-algorithm-implementation-in-python-5ab67bb124a6)

[](https://github.com/ahmedfgad/GeneticAlgorithmPython) [## ahmedfgad/遗传算法 Python

### 遗传算法在 Python 中的实现。通过创建一个……

github.com](https://github.com/ahmedfgad/GeneticAlgorithmPython) 

# 联系作者

电子邮件:[ahmed.f.gad@gmail.com](mailto:ahmed.f.gad@gmail.com)

**领英**:[https://linkedin.com/in/ahmedfgad/](https://linkedin.com/in/ahmedfgad/)

**KD nuggets**:[https://kdnuggets.com/author/ahmed-gad](https://kdnuggets.com/author/ahmed-gad)

**YouTube**:[https://youtube.com/AhmedGadFCIT](https://youtube.com/AhmedGadFCIT)

**走向 https://towardsdatascience.com/@ahmedfgad**: