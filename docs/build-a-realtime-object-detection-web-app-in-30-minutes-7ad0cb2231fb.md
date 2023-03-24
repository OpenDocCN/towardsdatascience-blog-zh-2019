# 在 30 分钟内构建一个实时对象检测 Web 应用程序

> 原文：<https://towardsdatascience.com/build-a-realtime-object-detection-web-app-in-30-minutes-7ad0cb2231fb?source=collection_archive---------7----------------------->

![](img/28a827bff1e160256668b32fa6ddb538.png)

Image Credit: [https://github.com/tensorflow/models/tree/master/research/object_detection](https://github.com/tensorflow/models/tree/master/research/object_detection)

# 张量流。射流研究…

Tensorflow.js 是一个开源库，使我们能够使用 Javascript 在浏览器中定义、训练和运行机器学习模型。我将使用 Angular 中的 [Tensorflow.js 框架](https://www.tensorflow.org/js/)来构建一个 Web 应用程序，该应用程序可以检测网络摄像头视频馈送上的多个对象。

# COCO-SSD 型号

首先，我们必须选择我们将用于对象检测的预训练模型。Tensorflow.js 提供了几个预训练模型，用于分类、姿态估计、语音识别和对象检测目的。查看所有 [Tensoflow.js 预训练模型](https://github.com/tensorflow/tfjs-models)了解更多信息。

COCO-SSD 模型是一种预训练的对象检测模型，旨在定位和识别图像中的多个对象，是我们将用于对象检测的模型。

![](img/8bc826b321d6feeed17ba1ffc3cf21a8.png)

Photo by [Brooke Cagle](https://unsplash.com/@brookecagle?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

原 [ssd_mobilenet_v2_coco](http://download.tensorflow.org/models/object_detection/ssd_mobilenet_v2_coco_2018_03_29.tar.gz) 模型大小为 187.8 MB，可从 TensorFlow 模型动物园下载。与原始模型相比，Tensorflow.js 版本的模型非常轻量级，并针对浏览器执行进行了优化。

Tensorflow.js COCO-SSD 默认的物体检测模型是**‘lite _ mobilenet _ v2’**体积非常小，1MB 以下，推理速度最快。如果您想要更好的分类准确性，您可以使用**‘mobilenet _ v2’**，在这种情况下，模型的大小增加到 75 MB，这不适合 web 浏览器体验。

**“model . detect”**直接从 HTML 中获取图像或视频输入，因此在使用之前，您无需将输入转换为张量。它返回检测到的对象的类、概率分数以及边界框坐标的数组。

```
model.detect(
 img: tf.Tensor3D | ImageData | HTMLImageElement |
   HTMLCanvasElement | HTMLVideoElement, maxDetectionSize: number
)
[{
 bbox: [x, y, width, height],
 class: "person",
 score: 0.8380282521247864
}, {
 bbox: [x, y, width, height],
 class: "kite",
 score: 0.74644153267145157
}]
```

![](img/7c5d8de4e0b25cf1d659e996aea02c36.png)

Photo by [Marc Kleen](https://unsplash.com/@marckleen?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# ANGULAR WEB 应用正在初始化

在我们清楚了模型之后，是时候使用 Angular 命令行界面来初始化 [Angular](https://angular.io/) web 应用程序了。

```
npm install -g @angular/cli
ng new TFJS-ObjectDetection
cd TFJS-ObjectDetection
```

然后我将使用 NMP 包管理器加载 Tensorflow.js 和 COCO-SSD 库。

```
TFJS-ObjectDetection **npm install @tensorflow/tfjs --save** TFJS-ObjectDetection **npm install @tensorflow-models/coco-ssd --save**
```

现在都准备好了。所以我们可以开始编码了。我先从**‘app . component . ts’**导入 COCO-SSD 模型开始。

```
import { Component, OnInit } from '@angular/core';**//import COCO-SSD model as cocoSSD**
import * as cocoSSD from '@tensorflow-models/coco-ssd';
```

# 开始网络摄像机馈送

然后，我将使用以下代码启动网络摄像头。

```
webcam_init()
{
  this.video = <HTMLVideoElement> document.getElementById("vid");
  navigator.mediaDevices
  .getUserMedia({
  audio: false,
  video: {facingMode: "user",}
  })
  .then(stream => {
  this.video.srcObject = stream;
  this.video.onloadedmetadata = () => {
  this.video.play();};
  });
}
```

# 物体检测功能

我们需要另一个函数来加载 COCO-SSD 模型，该模型也调用**‘detect frame’**函数来使用来自网络摄像头馈送的图像进行预测。

```
public async predictWithCocoModel()
{
  const model = await cocoSSD.load('lite_mobilenet_v2');
  this.detectFrame(this.video,model);

}
```

**‘detect frame’**函数使用 **requestAnimationFrame** 通过确保视频馈送尽可能平滑来反复循环预测。

```
detectFrame = (video, model) => {
  model.detect(video).then(predictions => {
  this.renderPredictions(predictions);
  requestAnimationFrame(() => {
  this.detectFrame(video, model);});
  });
}
```

![](img/e32ac9ad22239beec025eb1ff21f44de.png)

Photo by [Mikhail Vasilyev](https://unsplash.com/@miklevasilyev?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 渲染预测

同时，**‘render predictions’**函数将检测到的对象的边界框和类名绘制到屏幕上。

```
renderPredictions = predictions => {const canvas = <HTMLCanvasElement> document.getElementById     ("canvas");

  const ctx = canvas.getContext("2d");  
  canvas.width  = 300;
  canvas.height = 300;
  ctx.clearRect(0, 0, ctx.canvas.width, ctx.canvas.height);  

  **// Fonts**
  const font = "16px sans-serif";
  ctx.font = font;
  ctx.textBaseline = "top";
  ctx.drawImage(this.video,0,0,300,300);

  predictions.forEach(prediction => {  

   **// Bounding boxes's coordinates and sizes**
   const x = prediction.bbox[0];
   const y = prediction.bbox[1];
   const width = prediction.bbox[2];
   const height = prediction.bbox[3];**// Bounding box style**
   ctx.strokeStyle = "#00FFFF";
   ctx.lineWidth = 2;**// Draw the bounding
**   ctx.strokeRect(x, y, width, height);  

   **// Label background**
   ctx.fillStyle = "#00FFFF";
   const textWidth = ctx.measureText(prediction.class).width;
   const textHeight = parseInt(font, 10); // base 10
   ctx.fillRect(x, y, textWidth + 4, textHeight + 4);
   });

   predictions.forEach(prediction => {
    **// Write prediction class names**
    const x = prediction.bbox[0];
    const y = prediction.bbox[1];  
    ctx.fillStyle = "#000000";
    ctx.fillText(prediction.class, x, y);});   
   };
```

现在，在浏览器上执行对象检测所需的所有功能都已准备就绪。

我们只需要在**【ngOnInit】**上调用**‘web cam _ init’**和**‘predictwithcocomoodel’**就可以在启动时初始化 app。

```
ngOnInit()
{
  this.webcam_init();
  this.predictWithCocoModel();
}
```

![](img/ed291037b2ce1ed4b22fdec8c0b2a5a6.png)

Photo by [Andrew Umansky](https://unsplash.com/@angur?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 用于对象检测的 HTML 元素

剩下的最后一步是修改 **'app.component.html'** ，以包含上述功能工作所需的 *< video >* 和 *< canvas >* HTML 元素。

```
<div style="text-align:center">
  <h1>Tensorflow.js Real Time Object Detection</h1>
  <video hidden id="vid" width="300" height="300"></video>
  <canvas id="canvas"></canvas>
</div>
```

# 完整代码

访问我的 [GitHub 资源库](https://github.com/eisbilen/TFJS-ObjectDetection)获取该项目的完整代码。

# 演示 WEB 应用程序

访问[现场演示](https://tfjs-objectdetection.firebaseapp.com/)应用程序，查看运行中的代码。该应用程序在 Google Chrome 浏览器中运行没有任何问题。如果您使用任何其他浏览器，请确保您使用的浏览器支持' **requestAnimationFrame'** 。