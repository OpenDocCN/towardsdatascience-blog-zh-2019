# 使用 React 和 face-api.js 的 BNK48 偶像团体面部识别 SPA

> 原文：<https://towardsdatascience.com/facial-recognition-spa-for-bnk48-idol-group-using-react-and-face-api-js-ad62b43ec5b6?source=collection_archive---------6----------------------->

## 如何零后端代码部署自己的偶像识别系统

如今，人脸检测和识别已经不是什么新鲜事了。一年前，我曾经尝试在 Python 上使用 TensorFlow 和 [facenet](https://github.com/davidsandberg/facenet) 制作自己的面部识别系统。该项目旨在从 AKB48 成员的照片中进行人脸检测和识别。

通过看他们的脸，你可能会同意我的观点，任何人都很难记住他们的名字，更不用说识别和区分他们的脸了。(声明一下，我不是粉丝什么的)

![](img/42adb8dc96410180206b0999f0bff126.png)

Cr: [http://akb48.wikia.com](http://akb48.wikia.com)

我的 Python 项目进行得非常好。我可以在 Jupyter-Notebook 中编写代码，从任何输入图像中检测和识别成员。然而，该系统是在 Python 环境下运行的，这对于大多数人脸识别系统来说是很正常的。但这意味着，如果我想从其他设备输入图像，例如智能手机，我需要创建前端来连接和发送图像到 Python 后端，以处理面部检测和识别。(大量工作要做)

由于缺乏动力，项目被搁置，一年过去了。直到我找到了[文森特·穆勒](https://medium.com/u/ffd42e31db07?source=post_page-----ad62b43ec5b6--------------------------------)的 [face-api.js](https://github.com/justadudewhohacks/face-api.js) ，一个用 [TensorFlow.js](https://js.tensorflow.org/) 进行人脸检测和识别的 JavaScript API。现在可以在浏览器上运行深度学习的所有过程，而不需要后端代码。(声音好听！)

是的，不需要任何后端编码或环境设置！！你需要的只是一个静态的虚拟主机。它可以在任何设备或浏览器上运行。(如果你的浏览器可以运行 TensorFlow.js，就这些)在本文的最后，我会向你展示如何将这个 React app 部署到 Github 页面。

## 又一个偶像团体:BNK48

这次回到我的项目，我决定使用另一个偶像团体的照片， [BNK48](https://www.bnk48.com/) ，AKB48 的姐妹乐队，总部设在曼谷。

我来告诉你为什么偶像组合是实验面部识别 app 的好选择。(再次声明，我不是粉丝。)这是因为我们将需要许多已知的面孔和名字来测试我们的系统，对吗？使用 5-10 张照片来测试系统可能很容易，但在现实世界中，你不会为仅仅 10 个人制作人脸识别系统，对吗？这就是为什么偶像团体的 30-50 名成员是一个很好的测试数字。(不算少，也不算多)我们可以很容易地从网上找到他们各种视角的肖像照片，尤其是他们的 facebook 粉丝页面。

## 我们要做什么？

在这里，在这个项目中，我们将使用 React 和 face-api.js 库制作单页 App 来检测和识别偶像人脸。由于 Vincent 在他的 API 中为我们完成了所有困难的部分，该 API 带有预先训练的人脸检测、人脸标志、人脸对齐和人脸识别模型，所以我们不必自己训练模型。我们甚至也不需要用 TensorFlow 写 DL 模型。事实上，你真的不需要知道深度学习或 CNN 如何制作这个应用程序。你所需要知道的至少是 JavaScript 和 React 的基本概念。

如果你迫不及待想看看它的样子，请访问我的演示页面[这里](https://supachaic.github.io/bnk48-face-recognition)。而我的 App 完整回购就是[这里](https://github.com/supachaic/bnk48-face-recognition)。我们将在本教程中制作的代码将会更简单，但是不要担心，我也会在另一个 repo 中分享它。

# 人脸识别系统简介

如果您已经知道它是如何工作，或者不是很关心，您可以直接进入编码部分。

现在，让我们想象一下，当你去某个政府办公室索要一份你的个人文件时。柜台后面的工作人员通常会要求你证明你是谁。你给他们看你的身份证。她看着你的名字和照片，然后检查你的脸，确保你就是你声称的那个人。

同样，面部识别系统应该已经存储了您的姓名以及您的参考面部信息。然后，当你输入另一张照片进行识别时，系统将首先尝试检测图像上是否存在任何人脸，在这一步**人脸检测网络**将完成这项工作。我在这个项目中使用的模型是**微型人脸检测器**，因为它体积小，便于移动。(API 还为面部检测器提供了 **SSD mobileNet** 和 **MTCNN** ，但现在让我们忘记它们。)

回到我们的系统。一旦检测到人脸，人脸检测器模型将返回每个人脸的**边界框**，告诉我们人脸在图像中的位置。然后，我们使用**人脸标志网络**来标记 68 个点的人脸标志，并在馈送到**人脸识别网络**之前使用对齐模型来确保人脸居中。

**人脸识别网络**是另一个神经网络(准确地说， **RestNet-34 类似于**神经网络)返回一个**人脸描述符**(特征向量包含 128 个值)，我们可以用它来比较和识别图像中的人。

就像指纹一样，**人脸描述符**是每张人脸的唯一值。当我们比较来自不同图像源的同一个人的面部描述符时，它们应该非常接近。在这个项目中我们用**欧几里德距离**来比较。如果距离小于我们设置的阈值，我们确定他们很可能是同一个人。(距离越小，自信越高)

通常情况下，系统会将每个人的**脸部描述符**作为参考，同时将他或她的名字作为标签保存。当我们输入查询图像时，系统会将新图像的人脸描述符与所有参考描述符进行比较，并识别出最低的人。如果比较结果都不低于阈值，此人将被识别为**未知**。

# 开始编码吧！

有 2 个功能，我们希望在这个应用程序中实现。一种是从输入图像文件中识别偶像，另一种是使用实况视频作为输入。

先从`create-react-app`开始，安装`react-router-dom`，启动 App。

```
npx create-react-app react-face-recognition
cd react-face-recognition
npm i react-router-domnpm start
```

打开浏览器，进入 [http://localhost:3000/](http://localhost:3000/) 如果你看到带有 React 标志的起始页，那么你就可以继续了。现在用你喜欢的任何代码编辑器打开项目文件夹。您应该会看到这样的文件夹结构。

```
react-face-recognition 
├── README.md 
├── node_modules 
├── package.json 
├── .gitignore 
├── public 
│   ├── favicon.ico 
│   ├── index.html 
│   └── manifest.json 
└── src 
    ├── App.css 
    ├── App.js 
    ├── App.test.js 
    ├── index.css 
    ├── index.js 
    ├── logo.svg 
    └── serviceWorker.js
```

现在转到`src/App.js`，用下面的代码替换代码。

src/App.js

我们这里只有导入`Home`组件并创建一条到`"/"`的路径作为我们的登陆页面。我们将很快创建这个组件。

让我们从创建新文件夹`src/views`开始，在这个文件夹中创建新文件`Home.js`。然后将下面的代码放到文件中并保存。

src/views/Home.js

我们只创建 2 个链接，分别是`Photo Input`链接到`"localhost:3000/photo"`和`Video Camera`链接到`"localhost:3000/camera`。如果一切顺利，我们应该在登录页面看到如下内容。

![](img/850fa7799b2f20024e3a43bbf459822a.png)

Landing Page

## Face API

在我们继续创建新页面之前，我们希望安装 face-api.js 并创建我们的 api 文件来连接与 API 的反应。现在回到控制台并安装库。

```
npm i face-api.js
```

该库附带 TensorFlow.js 和我们想要的所有组件，除了**模型权重**。如果你不知道它们是什么，模型权重是已经用大型数据集训练过的神经网络权重，在这种情况下，是成千上万的人脸图像。

由于许多聪明人已经为我们训练了模型，我们需要做的只是掌握我们想要使用的必要权重，并手动输入我们的项目。

你会在这里找到这个 API [的所有权重](https://github.com/justadudewhohacks/face-api.js/tree/master/weights)。现在让我们新建一个文件夹`public/models`来放置所有的模型权重。然后下载所有必要的重量到下面的文件夹。(正如我告诉你的，我们将在这个项目中使用**微型人脸检测器**型号，所以我们不需要 **SSD MobileNet** 和 **MTCNN** 型号。)

![](img/3b8186e998982a31207aa680eb5d02de.png)

Necessary Models

确保您将所有重量放在下面的`public/models`文件夹中，否则没有合适的重量，我们的模型将无法工作。

```
react-face-recognition 
├── README.md 
├── node_modules 
├── package.json 
├── .gitignore 
├── public 
│   ├── models
│   │   ├── face_landmark_68_tiny_model-shard1
│   │   ├── face_landmark_68_tiny_model-weights_manifest.json
│   │   ├── face_recognition_model-shard1
│   │   ├── face_recognition_model-shard2
│   │   ├── face_recognition_model-weights_manifest.json
│   │   ├── tiny_face_detector_model-shard1
│   │   └── tiny_face_detector_model-weights_manifest.json
│   ├── favicon.ico 
│   ├── index.html 
│   └── manifest.json
```

现在返回并为 API 创建新文件夹`src/api`，并在文件夹内创建新文件`face.js`。我们要做的是加载模型并创建函数来将图像提供给 API 并返回所有的人脸描述，还可以比较描述符来识别人脸。稍后我们将导出这些函数并在 React 组件中使用。

src/api/face.js

这个 API 文件有两个重要的部分。第一个是用函数`loadModels()`加载模型和权重。我们在这一步只加载微小人脸检测器模型、人脸标志微小模型和人脸识别模型。

另一部分是函数`getFullFaceDescription()`，其接收图像斑点作为输入，并返回全脸描述。该函数使用 API 函数`faceapi.fetchImage()`将图像 blob 提取到 API。然后`faceapi.detectAllFaces()`将获取该图像并找到图像中的所有人脸，然后`.withFaceLandmarks()`将绘制 68 个人脸标志，然后使用`.withFaceDescriptors()`返回 128 个值的人脸特征作为`Float32Array`。

值得一提的是，我使用 image `inputSize` 512 像素进行图像输入，稍后将使用 160 像素进行视频输入。这是 API 推荐的。

现在我要你把下面的图片保存到新文件夹`src/img`中，并命名为`test.jpg`。这将是我们的测试图像来测试我们的应用程序。(以防你不知道，她是 Cherprang，顺便说一下，是 BNK48 的成员。)

![](img/6e91c839316ff7b597b19f5f75948b14.png)

Save this image as src/img/test.jpg

让我们创建新文件`src/views/ImageInput.js`。这将是视图组件输入和显示我们的图像文件。

src/views/ImageInput.js

此时，该组件将只显示测试图像`src/img/test.jpg`，并开始将 API 模型加载到您的浏览器中，这将花费几秒钟的时间。之后，图像将被输入 API 以获得完整的面部描述。我们可以将返回的`fullDesc`存储在`state`中以备后用，也可以在 console.log 中看到它的详细信息

但是在此之前，我们必须将`ImageInput`组件导入到我们的`src/App.js`文件中。并为`/photo`创建新的`Route`。开始了。

src/App.js with new Route and Component

现在，如果您转到登录页面`[http://localhost:3000](http://localhost:3000)`并点击`Photo Input`，您应该会看到照片显示。如果你检查你的浏览器控制台，你应该看到这个图片的**全脸描述**如下。

![](img/98c90711f69e203e635983b9337d8e60.png)

## 面部检测盒

如你所见，描述包含了我们在这个项目中需要的所有人脸信息，包括`descriptor`和`detection`。`detection`内有坐标`x``y``top``bottom``left``right``height``width`等方框信息。

face-api.js 库自带函数用 html 画布画人脸检测框，真的很好看。但是既然我们用的是 React，为什么不用 CSS 来画人脸检测框呢，这样我们就可以用 React 的方式来管理框和识别显示了。

我们想要做的是使用检测的`box`信息在图像上叠加人脸框。我们还可以稍后显示应用程序识别的每张脸的名称。这就是我如何将`drawBox`添加到`ImageInput`组件中。

让我们一起添加`input`标签，这样我们就可以改变输入图像。

src/views/ImageInput.js

在 React 中使用内联 CSS，我们可以像这样放置所有的面部框来覆盖图像。如果您尝试用更多面孔来更改照片，您也将能够看到更多框。

![](img/2f53913cffe998128955b2c14308d69f.png)

# 面部识别

现在有趣的部分来了。为了识别一个人，我们需要至少一个参考图像来从图像中提取 128 个特征向量值或`descriptor`。

API 具有函数`LabeledFaceDescriptors`来为我们想要识别每个人创建描述符和名字的标签。该标签将与查询的`descriptor`一起提供给 API 以匹配人员。但在此之前，我们需要准备一个名称和描述符的配置文件。

## 平面轮廓

我们已经有一个 Cherprang 的图像参考。因此，让我们使用它的`descriptor`来制作一个配置文件。

我们现在要做的是创建新的 JSON 文件和文件夹`src/descriptors/bnk48.json`。该文件将包含成员姓名和参考照片中的描述符。这是第一个只有一个`descriptor`的样本文件。

Sample face profile

如果我们有所有成员的照片，我们可以添加`descriptor`并逐一命名来完成我们的面部轮廓。你知道吗？我已经做了一个。我用每个成员的 5-10 张照片来创建这个完整的面部轮廓。所以，你可以下载这个[文件](https://raw.githubusercontent.com/supachaic/bnk48-face-recognition/master/src/descriptors/bnk48.json)并替换`src/descriptors/bnk48.json`，很简单。(抱歉，我用泰语和平假名作为显示名称)

整个成员的文件大小在 1MB 左右，对于我们的测试 App 来说还不错。但是在现实世界中，您可能需要将所有面部轮廓存储在数据库中，这样您就不必再担心文件大小，但是您将需要使用服务器端来运行面部识别过程。

## 面部匹配器

下一步我们要为人脸识别任务创建`labeledDescriptors`和`faceMatcher`。现在回到`src/api/face.js`，然后将下面的函数添加到你的文件中。

src/api/face.js add function createMatcher

该函数将接收面部轮廓(JSON 文件)作为输入，并创建每个成员的描述符的`labeledDescriptors`，以其名称作为标签。然后我们可以创建并导出带标签的`faceMatcher`。

你可能会注意到我们配置了`maxDescriptorDistance` 0.5。这是**欧氏距离**的阈值，用来确定引用描述符和查询描述符是否足够接近，可以说点什么。API 默认值为 0.6，对于一般情况来说已经足够了。但我发现 0.5 对我来说更精确，误差更小，因为一些偶像的脸非常相似。如何调整这个参数取决于您。

既然我们的函数已经准备好了，让我们回到`src/views/ImageInput.js`来完成我们的代码。这是我们的最后一个。

Final code for ImageInput.js

在这个最终代码中，我们从`face.js`导入`createMatcher`函数，并用我们准备好的面部轮廓创建`faceMatcher`。内部函数`handleImage()`，我们从图像中得到`fullDesc`后，绘制出`descriptors`，找到每张脸的最佳`match`。

然后，我们使用`p`标签和 CSS 在每个人脸检测框下显示我们的最佳匹配。就像这样。

![](img/035d1be3279c0e5a256869ca5633ece8.png)

Face detect and recognize correctly

如果您已经下载了完整的[面部轮廓](https://raw.githubusercontent.com/supachaic/bnk48-face-recognition/master/src/descriptors/bnk48.json)。你可以试着用这个改变形象。我希望你能看到正确匹配检测到的所有人脸！！

![](img/da20cd56ac4130e269ad8cfadcaf0d5d.png)

Try this image

# 实时视频输入

本节将指导您使用 React-webcam 将实时视频作为 face-api.js 的输入。让我们从安装库开始。

```
npm i react-webcam
```

同样，在制作新的视图组件之前，我们还需要在`/src/App.js`中添加一个用于视频输入的`Route`。我们将很快创建`VideoInput`组件。

Add VideoInput Component and Route

## 视频输入组件

让我们创建新文件`src/views/VideoInput.js`并将下面的所有代码放入文件并保存。这是这个组件的完整代码。(不再按部就班。解释如下。)

人脸检测和识别的所有机制与`ImageInput`组件相同，除了输入是每隔 1500 毫秒从网络摄像头捕获的屏幕截图。

我将屏幕尺寸设置为 420x420 像素，但您可以尝试更小或更大的尺寸。(尺寸越大，处理人脸检测所需的时间越长)

内部功能`setInputDevice`我只是检查设备是否有 1 个或 2 个摄像头(或更多)。如果只有一个摄像头，我们的应用程序将假设它是一台 PC，然后我们将从网络摄像头`facingMode: user`捕捉，但如果有两个或更多，那么它可能是智能手机，然后我们将从背面用摄像头捕捉`facingMode: { exact: ‘environment’ }`

我使用与组件`ImageInput`相同的函数来绘制人脸检测框。其实我们可以把它做成另一个组件，这样就不用重复两遍了。

现在我们的应用程序完成了。你可以用你的脸测试 VideoInput，但它很可能会把你识别为`unknown`或者有时会错误地把你识别为某个偶像。这是因为如果欧几里德距离小于 0.5，系统将尝试识别所有的脸。

# 结论和经验教训

该应用程序可以相当准确地检测和识别偶像的脸，但仍有一些错误时有发生。这是因为拍摄对象可能没有直接面对相机，他们的脸可能会倾斜，或者照片被其他一些应用程序编辑过。

一些偶像可能长得很像，这让 App 有时会混淆。我发现，当来自不同的来源或光线设置时，偶像的脸会有所不同。戴眼镜或浓妆艳抹的偶像也会让我们的应用程序感到困惑。

我不得不承认这个系统并不完美，但仍有改进的余地。

我在 Chrome 和 Safari 上进行了测试，在 PC 上运行良好。我认为它应该也能在 IE 或 Firefox 上运行。用 Android 智能手机测试图像输入和视频输入都很好，但 react-webcam 由于安全问题不能与 iPhone 一起工作，我仍在寻找解决方法。老式手机往往无法与 TensorFlow 一起正常工作，因为它需要足够的计算能力来运行神经网络。

# 部署到 Github 页面

您可以将这个应用程序部署到任何静态主机，但是本节将指导您使用一些技巧将这个 React 应用程序部署到 Github 页面。你需要有 Github 帐户。如果你没有，去做一个。它是免费的。

首先，让我们安装`gh-pages`库。

```
npm i gh-pages
```

然后我们需要像这样在`src/App.js`的`createHistory()`里面加上`{ basename: process.env.PUBLIC_URL }`。

现在转到您的 [Github](https://www.github.com) 并创建一个名为 App name 的新存储库，在我们的例子中是`react-face-recognition`，然后复制 git URL，稍后添加到我们的项目中。接下来，打开`package.json`，像这样用你的 Github 账号和 App 名添加`"homepage"`。

```
"homepage": "http://YOUR_GITHUB_ACCOUNT.github.io/react-face-recognition"
```

暂时不要关闭`package.json`文件，因为我们会像这样在`"scripts"`下添加`predeploy`和`deploy`命令行。

```
"scripts": {
  "start": "react-scripts start",
  "build": "react-scripts build",
  "test": "react-scripts test",
  "eject": "react-scripts eject",
  "predeploy": "npm run build",
  "deploy": "gh-pages -d build"
}
```

现在您可以保存文件并返回到您的控制台终端，然后运行 git 命令将代码上传到您的 Github 存储库，并运行`npm run deploy`部署到 Github 页面。该页面应使用您设置为`http://YOUR_GITHUB_ACCOUNT.github.io/react-face-recognition`的 URL 发布

```
git add .
git commit -m "make something good"
git remote add origin https://github.com/YOUR_GITHUB_ACCOUNT/react-face-recognition.git
git push -u origin master

npm run deploy
```

你可以在这里查看本教程[的 Github 页面，也可以完成](https://supachaic.github.io/react-face-recognition/)[回购](https://github.com/supachaic/react-face-recognition)。

我希望你喜欢我的教程，并尝试让你自己的反应面部识别。如果你觉得这个教程很简单，想看更完整的版本，请访问我的演示页面[这里](https://supachaic.github.io/bnk48-face-recognition/)，还有[回购](https://github.com/supachaic/bnk48-face-recognition)。