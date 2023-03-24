# 浏览器中的简易机器学习:实时图像分类

> 原文：<https://towardsdatascience.com/easy-machine-learning-in-the-browser-31b225d6cee0?source=collection_archive---------24----------------------->

![](img/31a52a825ce117f1cf23dc975433cb8d.png)

Photo by [Kasya Shahovskaya](https://unsplash.com/@kasya?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

> 利用 Tensorflow.js KNN 模块和 Angular 在浏览器中训练实时图像识别模型

将机器学习应用部署到生产环境中曾经是一个令人望而生畏的过程，因为它需要在许多不同的环境中处理复杂的操作。随着 Tensorflow.js 的引入，使用 javascript 在 web 浏览器中开发、训练和部署机器学习应用程序变得超级容易。

为了演示这一点，我将使用 Angular 创建一个简化版本的[可示教机器](https://teachablemachine.withgoogle.com)演示应用程序。这个演示应用程序教你的计算机使用你的网络摄像头在网络浏览器中实时识别图像。

为了实现这一点，我将使用 Tensorflow.js 提供的 KNN 模块。该模块使用[的 K-最近邻](https://en.wikipedia.org/wiki/K-nearest_neighbors_algorithm)算法创建一个分类器。它不是为模型提供权重，而是使用另一个模型的激活来创建 KNN 模型。

为此，mobilenet 图像分类模型是一个非常好的选择，因为它是轻量级的，可以在 Tensorflow 中使用。机器学习中的这一过程被称为迁移学习，因为我们使用的是另一种机器学习模型的表示。

![](img/8d79fc33a229b0079ecf2c040d93f6a2.png)

Photo by [Andrii Podilnyk](https://unsplash.com/@yirage?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

在深入细节之前，让我们首先更好地了解应用程序中遵循的步骤是什么；

***Step1:***

用户应提供并标记用于训练的 3 组输入图像。网络摄像头和浏览器上的按钮可用于此目的。

***第二步*** :

一旦为所有输入图像提供了标签，将通过将输入图像转储到 mobilenet 模型中来预测激活张量。然后，这些激活张量将被用作 KNN 分类器的输入，以创建具有分配给所提供的每个激活张量的标签的数据集。培训过程将在这一步完成。

***步骤三:***

对于预测，从网络摄像头捕捉的图像将被实时输入 mobilenet 模型，以获得激活张量。这些将被输入到训练好的 KNN 模型中，以识别图像的类别。

一开始看起来有点复杂，但是一旦用例子一步一步地解释，就很容易理解了。

![](img/5c792e441ca062279c2d52288a10e391.png)

Photo by [Robert Baker](https://unsplash.com/@vegasphotog?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

在开始编码之前，让我们进一步了解 K-最近邻算法和 Tensorflow.js KNN 模块是如何工作的。

# K 近邻算法是如何工作的？

k 近邻(KNN)是一种简单、易于实现的机器学习算法，在推荐系统和基于相似性的分类任务中有许多实际用途。

它将示例存储为带标签的类。当你需要预测一个新例子的类别时，它会计算这个新例子和所有其他已经被标记和已知的例子之间的距离。k 是最近邻居的数量。此时，通过邻居的多数投票来完成分类。

你可以点击[这个链接](https://www.datacamp.com/community/tutorials/k-nearest-neighbor-classification-scikit-learn)获得更多关于 KNN 算法的信息。

![](img/dd6c3511e11da2ec67b643ba7283b747.png)

Photo by [Nina Strehl](https://unsplash.com/@ninastrehl?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# Tensorflow.js KNN 模块

要使用 Tensorflow.js KNN 模块，必须将已知示例添加到分类器中。这是通过 **addExample** 方法完成的。在我们的例子中，已知的例子将通过身体姿态图像生成，这些图像是从实时网络摄像头获取的。这将是 KNN 算法的训练阶段。一旦完成，我们将能够用**预测类**方法预测未知的身体姿势。

```
***//importing tensorflow.js KNN module***
import * as knnClassifier from ‘@tensorflow-models/knn-classifier’;***//instantiating the classifier***
classifier = knnClassifier.**create()*****//adding examples to the KNN model for training***
classifier.**addExample**(example: tf.Tensor,label: number|string): void;***//predicting unknown examples with the trained KNN classifier***
classifier.**predictClass**(input: tf.Tensor,k = 3): Promise<{label: string, classIndex: number, confidences: {[classId: number]: number}}>;
```

这是足够的理论！

让我们把这个理论付诸实践吧！

![](img/6501660d5cffca517db4fea16dc7bf68.png)

Photo by [Jakob Owens](https://unsplash.com/@jakobowens1?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

此时，我们必须生成我们的 Angular WebApp。下面介绍的所有打字稿代码都必须在 **app.component.ts.** 中

如果你想了解更多关于如何生成 Angular WebApp，你可以查看我以前的帖子。

[](https://medium.com/@erdemisbilen/building-realtime-object-detection-webapp-with-tensorflow-js-and-angular-a4ff5062bdf1) [## 使用 Tensorflow.js 和 Angular 构建实时对象检测 WebApp

### 张量流。射流研究…

medium.com](https://medium.com/@erdemisbilen/building-realtime-object-detection-webapp-with-tensorflow-js-and-angular-a4ff5062bdf1) 

让我们首先启动网络摄像头和模型。

```
init_webcam()
{
***// Get the HTMLVideoElement*** this.video = <HTMLVideoElement> document.getElementById("vid");***// Start webcam stream***
navigator.mediaDevices.getUserMedia({audio: false, video: {facingMode: "user"}}).then(*stream* *=>*{
this.video.srcObject = stream;
this.video.onloadedmetadata = () *=>* {this.video.play();};});
}async init_models()
{
***// Initiate the KNN classifier***
this.classifier = await knnClassifier.create();
*console*.log('KNN classifier is loaded')***// Load mobilenet model***
*const* mobilenet =  await tf.loadLayersModel('https://storage.googleapis.com/tfjs-models/tfjs/mobilenet_v1_0.25_224/model.json');
*console*.log('Mobilenet model is loaded')***// Modify the mobilenet model so that we can get the activations from 'conv_preds' layer*** *const* outputLayer=mobilenet.getLayer('conv_preds');
this.mobilenetModified = tf.model({inputs: mobilenet.inputs, outputs: outputLayer.output, name: 'modelModified' });
*console*.log('MobilenetModified model is generated')
}
```

下一步是创建一个函数，将示例添加到我们刚刚启动的 KNN 分类器中。这个函数接受定义我们添加的示例类的参数 **className** 。

```
addExample(*className*)
{
***// Get the image from the video feed and convert it to tensor*** *let* img=tf.browser.fromPixels(<HTMLVideoElement> document.getElementById("vid"));***// Get the activations from the mobilenet model*** *let* logits = <tf.Tensor> this.mobilenetModified.predict(img.expandDims());***// Add the activations as an example into the KNN model with a class name assigned*** this.classifier.addExample(logits, className);***// Show the stored image on the browser*** *let* canvas = <HTMLCanvasElement> document.getElementById("canvas");
*let* ctx = canvas.getContext("2d");
ctx.drawImage(<HTMLVideoElement> document.getElementById("vid"),0,0,224,224)
*console*.log('KNN example added')
}
```

然后我们需要创建一个函数来预测未知的例子。

```
***// Calls detectFrame with video, mobilenet and KNN arguments***
predict()
{this.detectFrame(this.video,this.mobilenetModified, this.classifier);
}***// Performs real-time predictions continuously by generating the activations of images produced from video feed. Then it predicts the class of the image based on the similarity with stored examples.***detectFrame = (*video*, *mobileNetModel*, *KNNModel*) *=>* {*const* predictions= <tf.Tensor> mobileNetModel.predict(tf.browser.fromPixels(video).expandDims());KNNModel.predictClass(predictions).then(*result=>* {this.renderPredictions(result);requestAnimationFrame(() *=>* {
this.detectFrame(video, mobileNetModel, KNNModel);});});
}***//Writes KNNClassifier results to the console***
renderPredictions = *result* *=>* {*console*.log(result)};
```

最后一步是修改**app.component.html**，将必要的UI 元素放到应用程序中。

```
<h1>Easy Machine Learning in the Browser: Real-time Image Classification</h1><video  id="vid" width="224" height="224"></video>
<canvas id="canvas" width="224" height="224"></canvas><button mat-button (click)="addExample('0')">Add Example Class1</button>
<button mat-button (click)="addExample('1')">Add Example Class2</button>
<button mat-button (click)="addExample('2')">Add Example Class3</button><button mat-button (click)="predict()">Predict</button>
```

干得好！

您已经做了这么多，并且创建了一个 web 应用程序来对浏览器中的图像进行分类。