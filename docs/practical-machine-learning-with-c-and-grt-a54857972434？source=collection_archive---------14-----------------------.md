# 用 C++和 GRT 实现实用的机器学习

> 原文：<https://towardsdatascience.com/practical-machine-learning-with-c-and-grt-a54857972434?source=collection_archive---------14----------------------->

![](img/fd836ce1881c571023b73763462bbba9.png)

Photo by [Franck V.](https://unsplash.com/@franckinjapan?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

这将是从程序员的角度解释机器学习基础知识的系列教程中的第一篇。在第 1 部分中，我将展示如何使用 [GRT](https://github.com/nickgillian/grt) 库将基本的机器学习整合到 C++项目中。

## 什么是机器学习？

机器学习是一种计算方法，它使程序能够基于给定的输入生成可预测的输出，而无需使用显式定义的逻辑。

例如，使用传统的基于逻辑的编程，我们可以编写一个对水果进行分类的函数，它以颜色和尺寸作为输入，输出水果的名称。大概是这样的:

```
string classifyFruit(Colour c, Dimensions d)
{
    if (c.similar({255, 165, 0}))   // green
    {
        if (d.similar({10, 9, 11})) // round-ish
        {
             return "Apple";
        }
        else
        {
             return "Pear";
        }
    }
    if (c.similar({255, 255, 0}))   // yellow
    {
         return "Banana";
    }
    if (c.similar({255, 165, 0}))   // orange
    {
         return "Orange";
    }

    return "Unknown";
}
```

可以看出，这种方法存在各种各样的问题。我们的函数只知道四种水果，所以如果我们想扩展它来对 Clementines 进行分类，我们需要额外的语句来区分它们和橙子。根据水果的确切形状和我们的`similar()`方法的定义，我们还会将梨和苹果混在一起。要创建一个函数来对各种各样的水果进行高精度分类，事情会变得非常复杂。

机器学习通过将输入和输出之间的关系表示为[状态](https://en.wikipedia.org/wiki/State_(computer_science)#Program_state)而不是通过逻辑规则来解决这个问题。这意味着我们可以使用`given / then`例子来表达我们的意图，而不是使用`if / then / else`语句来编写我们的分类器。所以决定我们函数行为的代码看起来更像这样:

```
ml.given({0,   255, 0, 10, 9,  11}) .then("Apple");
ml.given({0,   255, 0, 15, 7,  8})  .then("Pear");
ml.given({255, 255, 0, 20, 4,  4})  .then("Banana");
ml.given({255, 165, 0, 10, 10, 10}) .then("Orange");
```

这里，`ml`表示我们的机器学习对象，`given()`的列表参数表示不同类型水果的颜色和尺寸元组，`then()`的字符串参数表示给定输入时我们期望从分类器得到的输出。

如果我们程序的下一步是类似于`x = ml.classify({255, 165, 0, 10, 10, 10})`的东西，我们期望`x`的值是“橙色”。

这种方法的优点是，由于我们已经将*状态*从*逻辑*中分离，指定输入/输出关系的过程可以自动化。

类似于:

```
auto rows = csv.read();for (auto& r : rows)
{
    ml.given(r.colour, r.dimension).then(r.name);
}
```

现在，我们可以添加尽可能多的不同类型的水果，或者在不修改代码的情况下给出同一类型水果的许多不同示例！我们唯一需要改变的是输入数据的大小和内容，在本例中是来自一个 [CSV](https://en.wikipedia.org/wiki/Comma-separated_values) 文件。

敏锐的读者现在会想:我们不是刚刚创建了一个大的查找表吗？答案是“否”,因为这只会对*与示例中的颜色和尺寸完全匹配的水果进行分类。相反，我们希望系统能够对新的水果项目进行分类，这些项目与我们的一个示例具有相同的名称，但是具有不同的尺寸和颜色。因此对于`x = c.classify({245, 145, 0, 11, 9, 10})`,我们期望输出为“橙色”,即使颜色和尺寸与我们在`given / then`语句中提供的任何示例都不完全匹配。*

为了实现这一点，机器学习系统使用一组统计参数来定义基于现有输入的给定输出的*可能性*。每次提供新的示例时，这些参数都会更新，以使系统更加精确。这就是监督机器学习的本质。

因此，让我们回顾并建立一些术语:

*   在*监督机器学习*中，我们有一组定义系统输入和输出之间预期关系的例子。这被称为[数据集](https://en.wikipedia.org/wiki/Training,_validation,_and_test_sets)。
*   数据集中的每一行都由确定输入的*类别*的*标签*和确定其属性的*特征向量*组成(例如`{245, 145, 0, 11, 9, 10}`包含颜色和尺寸特征)。
*   我们的目标是创建输入(特征向量)和输出(类别标签)之间的*统计* *关系*的表示。这叫做[型号](https://en.wikipedia.org/wiki/Predictive_modelling)。
*   为了实现这一点，我们使用一个系统来从*训练数据集*生成*模型*，并使用该模型执行*分类*。这叫做机器学习[算法](https://en.wikipedia.org/wiki/Algorithm)。

因此，机器学习算法被称为从数据集“学习”，以便生成可用于对原始数据集中不存在的输入特征向量进行分类的模型。

## 有什么用？

![](img/b61f43f8237c1835d54037ef9694ceef.png)

机器学习有很多种。在这篇文章中，我主要关注监督分类。当我们想要建立一个可以自动分类许多不同类别的对象的系统时，监督分类是有用的。“独特的”这个词很重要，因为当物体的类别具有将它们彼此分开的独特特征(颜色、形状、大小、重量等)时，分类算法工作得最好。如果我们的对象非常相似，并且只能通过特征的微小变化来区分，那么分类的效果就不太好。因此，它可以很好地用于水果或人脸(不同的眼睛颜色、头发颜色、面部毛发、脸型等)，但会很难根据气温和天空颜色来识别某人的位置。这种区分不同物体的能力被称为[可分性](https://en.wikipedia.org/wiki/Linear_separability)，是机器学习中的一个重要概念。一个很好的经验法则是，如果人类难以区分不同类别的物体，机器也会如此。

## 我如何将机器学习整合到我的 C++项目中？

现在我们对机器学习有了基本的了解，我们如何将它融入到我们的 C++项目中呢？一个很好的库是[手势识别工具包](http://www.nickgillian.com/wiki/pmwiki.php/GRT/GestureRecognitionToolkit)或 GRT。GRT 是为实时手势识别而开发的，适用于一系列机器学习任务。在麻省理工学院的许可下，它可以在 Windows、Mac 和 Linux 上使用，因此可以用于封闭或开源项目。GRT 的完整类文档可以在[这里](http://nickgillian.com/grt/api/0.2.0/index.html)找到。

GRT 由许多 C++类组成，每个类都实现一个特定的机器学习算法。几乎所有 GRT 类都使用以下约定:

*   `train(...)`使用输入数据`(...)`来训练新的模型，该模型然后可以用于分类。
*   `predict(...)`使用输入向量`(...)`和*训练模型*进行分类。
*   `getPredictedClassLabel()`返回从输入向量预测的类别标签
*   `save(...)`将模型或数据集保存到文件中。
*   `load(...)`从文件中加载预训练模型或数据集。
*   `clear()`清除 ML 对象，删除所有预训练模型

## 建筑 GRT

以下说明是针对 Mac 和 Linux 平台的，Windows 用户应该参考官方的[构建指南](https://github.com/jamiebullock/grt/tree/master/build#windows-build-instructions)。

要构建 GRT，需要 CMake。CMake 可以从[项目页面](https://cmake.org)安装。在 macOS 上，我推荐使用[自制软件](https://brew.sh)来安装 CMake。

安装 CMake 后，从 git 存储库下载 GRT:

```
$ git clone [https://github.com/jamiebullock/grt](https://github.com/jamiebullock/grt)
```

然后:

```
$ cd grt/build
$ mkdir tmp && cd tmp
$ cmake .. -DBUILD_PYTHON_BINDING=OFF
$ make
```

如果发生构建错误，可以在项目[问题跟踪器](https://github.com/nickgillian/grt/issues)中报告。否则，可以按如下方式测试构建:

```
$ ./KNNExample ../../data/IrisData.grt
```

这应该会输出类似如下的内容:

```
[TRAINING KNN] Training set accuracy: 97.5TestSample: 0 ClassLabel: 3 PredictedClassLabel: 3TestSample: 1 ClassLabel: 2 PredictedClassLabel: 2...TestSample: 29 ClassLabel: 2 PredictedClassLabel: 2Test Accuracy: 93.3333%
```

这将执行在`examples/ClassificationModulesExamples/KNNExample/KNNExample.cpp`中找到的代码——基于来自[虹膜数据集](https://en.wikipedia.org/wiki/Iris_flower_data_set)的数据的虹膜花分类器。更多解释见[此处](http://www.nickgillian.com/wiki/pmwiki.php/GRT/MachineLearning101)。

## 机器学习 Hello World！

我们现在准备使用 GRT 构建一个简单的水果分类器！

首先我们需要创建一个新的源文件，(姑且称之为 fruit.cpp)并包含 GRT 头文件。这是使用 GRT 库中大部分功能所需的唯一头文件。

```
#include "GRT.h"
using namespace GRT;typedef std::vector<double> v_;int main (int argc, const char * argv[])
{
}
```

接下来，我们将从训练数据集中添加一些数据。为了这个例子的目的，我们将使用同样的“水果数据”。为此，我们使用 GRT [ClassificationData](http://nickgillian.com/grt/api/0.2.0/class_classification_data.html) 类来创建一个简单的数据集。代替字符串，在机器学习中使用数字标签，这里我们假设一个映射:1 =苹果，2 =梨，3 =香蕉，4 =橘子。所以我们增加了我们的`main()`函数:

```
ClassificationData dataset;
dataset.setNumDimensions(6);// Add 3 examples for each item to give our classifier enough data
for (int i = 0; i < 3; ++i)
{
    dataset.addSample(1, v_{0,   255, 0, 10, 9,  11});   // Apple
    dataset.addSample(2, v_{0,   255, 0, 15, 7,  8});    // Pear
    dataset.addSample(3, v_{255, 255, 0, 20, 4,  4});    // Banana
    dataset.addSample(4, v_{255, 165, 0, 10, 10, 10});   // Orange
}
```

在实际代码中，我们将添加更多不同的训练示例，以便我们的分类器可以很好地推广到各种输入。我们还将使用`loadDatasetFromCSVFile()`方法从文件中加载数据集，这样数据就可以从我们的代码中分离出来。这方面的文档可以在[这里](http://nickgillian.com/grt/api/0.2.0/class_classification_data.html#ad082098a4f995a74c626fb151cfcf979)找到。

接下来，我们加载数据集并训练分类器。这里我们使用一个`KNN`分类器，它实现了 [k-NN 算法](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)，但是 GRT 中的任何其他分类器都可以工作。作为一个练习，鼓励读者尝试从 GRT 中替换出各种不同的分类器。

```
// The classification class. Try also SVM!
KNN classifier;// Train our classifier
classifier.train(dataset); 
```

就是这样！我们现在有了一个训练好的模型，使用基于我们提供的输入数据集的 k-NN 算法。现在测试分类器…

```
VectorFloat testVector = v_{0,   255, 0, 10, 9,  11};// Predict the output based on testVector
classifier.predict(testVector);// Get the label
auto classLabel = classifier.getPredictedClassLabel();std::cout << "Class label: " << classLabel << std::endl;
```

最后，为了证明我们的分类器可以推广到原始数据集中不存在的输入，我们将要求它对与我们的训练示例之一相似但不相同的特征向量进行分类。

```
VectorFloat differentVector = v_{10, 240, 40, 8, 10, 9};classifier.predict(differentVector);
auto otherLabel = classifier.getPredictedClassLabel();std::cout << "Other label: " << otherLabel << std::endl;
```

因为我们的`differentVector`与苹果的`dataset`向量最相似，所以我们希望程序输出`Other label: 1`。让我们看看它是否有效！

## 完整的代码列表

```
#include "GRT.h"
using namespace GRT;typedef std::vector<double> v_;int main (int argc, const char * argv[])
{
    ClassificationData dataset;
    dataset.setNumDimensions(6); // Add 3 examples each to give our classifier enough data
    for (int i = 0; i < 3; ++i)
    {
        // Apple
        dataset.addSample(1, v_{0,   255, 0, 10, 9,  11});
        // Pear
        dataset.addSample(2, v_{0,   255, 0, 15, 7,  8});
        // Banana
        dataset.addSample(3, v_{255, 255, 0, 20, 4,  4});
        // Orange
        dataset.addSample(4, v_{255, 165, 0, 10, 10, 10});   
    } // The classification class. Try also SVM
    KNN classifier; // Train our classifier
    classifier.train(dataset); // Create a test vector
    VectorFloat testVector = v_{0,   255, 0, 10, 9,  11}; // Predict the output based on testVector
    classifier.predict(testVector); // Get the label
    auto classLabel = classifier.getPredictedClassLabel(); std::cout << "Class label: " << classLabel << std::endl; // Try an input vector not in the original dataset
    VectorFloat differentVector = v_{10, 240, 40, 8, 10, 9}; classifier.predict(differentVector);
    auto otherLabel = classifier.getPredictedClassLabel(); std::cout << "Other label: " << otherLabel << std::endl; return 0;}
```

要在 macOS 或 Linux 上编译代码，键入以下内容(来自我们在本教程开始时创建的同一个`tmp`目录)。

```
g++ -std=c++14 fruit.cpp -ofruit -lgrt -L. -I../../GRT
```

这告诉编译器链接到当前目录中的`libgrt.so`，头文件在`../../GRT`中，如果将 GRT 移动到其他地方，参数`-L`和`-I`将需要调整。

最后，我们运行我们的程序:

```
./fruit
```

输出应该是:

```
Class label: 1
Other label: 1
```

恭喜你！您刚刚用 26 行 C++代码编写了一个通用分类器🤩。如果我们从一个 CSV 文件加载数据集，它会少得多。

我希望这个教程是有用的。敬请关注未来更多内容！