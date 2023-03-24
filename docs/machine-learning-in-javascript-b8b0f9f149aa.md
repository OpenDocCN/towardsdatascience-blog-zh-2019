# JavaScript 中的机器学习

> 原文：<https://towardsdatascience.com/machine-learning-in-javascript-b8b0f9f149aa?source=collection_archive---------7----------------------->

## 是不是更容易？难？还是单纯的好玩？

![](img/ded61265006a9e9b14b40265b507d2e5.png)

Picture on Unsplash by Luca Bravo

如果你以前尝试过机器学习，你可能会认为这篇文章的标题中有一个巨大的错别字，我打算用 Python 或 R 来代替 JavaScript。

如果你是一名 JavaScript 开发人员，你可能知道自从 NodeJS 创建以来，在 JavaScript 中几乎任何事情都是可能的。您可以使用 React 和 Vue 构建用户界面，使用 Node/Express 构建所有“服务器端”内容，使用 D3 构建数据可视化(Python 和 R 主导的另一个领域)。

在这篇文章中，我将向你展示如何用 JavaScript 执行机器学习！我们将从定义什么是机器学习开始，快速介绍 TensorFlow 和 TensorFlow.js，然后使用 React 和 ML5.js 构建一个非常简单的图像分类应用程序！

# 机器学习？那是什么？

除非你一直生活在一块石头下面，否则你很可能听说过机器学习(ML)和人工智能(AI)之类的词。即使你不是一个非常注重科学的人，你也可能在电视和互联网上看到过那些微软的广告，在那里人们谈论微软正在做的所有令人惊奇的事情。

事实是，几乎每个人在一生中的某个时刻都使用过机器学习和人工智能。划掉那个，每个人在生活中每天都在使用 ML 和 AI。从要求 Siri 和 Alexa 播放一些歌曲到使用手机上的导航应用程序来获得最快的上班路线，这都是 ML 和 AI。

但是我们如何定义这两个术语呢？让我们集中讨论 ML，因为它是本文的主题。用最简单的话说，机器学习就是:

> 一个研究领域，它允许计算机系统在没有给它任何具体指令的情况下做某事。

作为一名开发人员，你要以一种特殊的方式编写代码。您的客户或经理告诉您他们想要的输出是什么，您尝试编写一些代码来获得该输出。但是在机器学习中，你只知道需要解决的问题！你“教”你的计算机一些东西，然后坐下来，看看你从系统中得到什么惊人的结果！

要回答的问题是:我们如何进行机器学习？Python 程序员使用类似`scikit-learn`和谷歌令人惊叹的`TensorFlow`这样的包来执行机器学习。但是去年(2018 年)，谷歌发布了 JavaScript 版本的 TensorFlow，优雅地叫做`TensorFlow.js`！

但是为什么要用 JavaScript 做机器学习呢？首先，Python 的机器学习方式要求开发人员将机器学习代码保存在服务器上，然后使用 JavaScript 允许用户在客户端访问模型。这里我们遇到了一个潜在的问题。如果你的机器学习模型太受欢迎，很多用户都想访问它，那么你的服务器很有可能会崩溃！

但是如果我们使用机器学习，我们不仅为机器学习代码和用户界面代码保留了 JavaScript 环境，模型也将保留在客户端本身！还有，机器学习模型大多被金融公司使用。所以客户端 ML 模型意味着你的数据是私有的。

# 我们写点代码吧！

现在，您已经对 ML 有了一些基本的了解，并且知道了为什么用 JavaScript 来实现 ML 是一个好主意。但是 ML 是那些你可以通过尝试更好理解的东西之一。如果你想了解更多关于机器学习的内容，可以看看我不久前写的另一篇文章:

[](https://medium.com/@geekyants/deep-learning-with-react-native-65fae456839d) [## React Native 深度学习

### 围绕着人工智能(AI)这样的话题，人们总是非常兴奋。当有人提到…

medium.com](https://medium.com/@geekyants/deep-learning-with-react-native-65fae456839d) 

在本节中，我们将使用 React 构建一个机器学习应用程序，它可以执行一些非常好的图像分类。

**侧栏:**机器学习过程包括两个步骤:训练和测试。训练包括向模型提供大量数据，然后模型将处理这些数据并识别不同的模式，然后模型将使用这些数据进行未来预测。

由于我们正在构建一个图像分类模型，因此在我们能够做出任何预测之前，我们需要将数以千计的图像发送到该模型进行处理。图像需要以某种方式相互关联，老实说，我没有那么多图片(我是一个害羞的人)。还有，JavaScript 的机器学习对我来说还是新鲜的。所以作为一个捷径，我将使用一个预先训练好的模型。

在我们开始编码之前，请确保您的系统上安装了以下内容:

1.  [节点](https://nodejs.org/) —我们将需要它来安装不同的软件包
2.  代码编辑器——任何好的编辑器都可以。我个人喜欢用 [VSCode](https://code.visualstudio.com/download)

接下来是构建一个样板 React 应用程序。为此，请打开命令终端并运行以下命令:

```
$ npx create-react-app ml-in-js
```

该命令将创建一个名为`ml-in-js`的文件夹，并在您的系统中构建一个启动应用程序。接下来，回到命令终端，运行以下命令:

```
$ cd ml-in-js
$ npm run start
```

第一个命令非常简单。真正的奇迹发生在第二部。`npm run start`命令创建系统的本地开发级别，并在浏览器上自动打开，如下所示:

![](img/428c35ccdfdc635c7e5a3be7e521e2b1.png)

这个入门应用根本不知道什么是机器学习或者 Tensorflow。要解决这个问题，我们需要安装 Tensorflow.js 包。对于 Python 开发人员来说，您需要在您的系统上执行一次`pip install tensorflow`，并且可以在任何地方和项目中自由使用这个包。但是对于 JavaScript，你需要为每个项目运行`npm install`命令。

但我们不会在应用程序中安装 Tensorflow.js (TFJS)库，而是安装另一个名为 ML5.js 的库。这个库就像是 TFJS 的一个更好的版本，使我们在客户端进行机器学习变得更加容易。所以让我们像这样安装这个库:

```
$ npm install --save ml5
```

如果您想确保库安装成功，请转到`src`文件夹中的`App.js`文件，并编写以下代码:

```
import React, {Component} from 'react';
import * as ml5 from 'ml5';export default class App extends Component {
  render() {
    return (
      <div>
        <h1>{ml5.version}</h1>
      </div>
    )
  }
}
```

如果你回到浏览器，你会看到一个大大的 **0.4.1** 印在屏幕上。根据 ML5.js 的最新版本，这个数字可能会有所不同。只要您看到屏幕上打印的数字，您就可以放心，您的 ML5 库已成功安装。

至此，我们完成了安装部分。现在，我们需要创建一个函数，它可以接收图像，并使用预先训练好的模型对其进行分类。为此，在`App`组件中编写以下代码:

```
classifyImage = async () => {
  const classifier = await ml5.imageClassifier('MobileNet')
  this.setState({ready: true})
  const image = document.getElementById("image")
  classifier.predict(image, 1, (err, results) => {
    this.setState({
      predictionLabel: results[0].label,
      predictionConfidence: results[0].confidence,
      predicted: true
    })
  })
}
```

这里我们有一个名为`classifyImage`的异步函数。在这个函数中，我们首先通过加载`MobileNet`数据作为训练集来定义我们的图像分类器。一旦这个分类器准备好了，我们将`ready`状态设置为`true`。然后，我们选择用户插入的`image`并将其传递给分类器，运行`predict`函数。我们将最高预测的标签和置信度保存在`state`对象中。

我们的整个 App.js 文件将如下所示:

测试模型的时间到了。为此，我将给出下面的应用程序图像:

![](img/cd6d888e06b5b85cb200d0d5e7573ec5.png)

当我按下`Classify`按钮时，应用程序运行`classifyImage`功能，一段时间后您会得到以下结果:

```
The app is 63.99456858634949% sure that this is bucket
```

不是我所期待的。造成这种不正确结果的原因有很多。其中最重要的是，MobileNet 不是对该图像进行分类的合适数据集。

让我们试试其他图像。也许是一种动物:

![](img/9afd4f00c23d31e501e0e2cbe869406a.png)

Pic on Unsplash by Joséphine Menge

再次点击`Classify`按钮，我得到🤞：

```
The app is 90.23570418357849% sure that this is Border collie
```

哇！90%的信心，图像有边境牧羊犬，这是一种狗！

![](img/a3bd718292e838e11c82026bb76beceb.png)

# 包扎

如果你一直坚持到现在，那么你已经用 JavaScript 做了一些机器学习！来吧，拍拍自己的背！

但是你和我离成为 ML 专家还有很长的路要走。在本文中，我们做了机器学习的测试部分，并使用了预训练的模型。真正的乐趣始于你获取自己的原始数据，并试图用它来训练你的数据。

这就是我现在要努力去做的。所以祝我好运吧！如果我成功了，我会写下一篇关于它的文章。

一如既往的感谢大家阅读我的长文。JavaScript 的机器学习我还是新手。因此，如果你认为我在这篇文章中犯了什么错误，或者我本可以做些不同的事情，那么请对此发表评论。

你可以在这里查看 React 应用的完整源代码:

 [## rajatk16/ml-in-js

### 这个项目是用 Create React App 引导的。在项目目录中，您可以运行:在…中运行应用程序

github.com](https://github.com/rajatk16/ml-in-js)