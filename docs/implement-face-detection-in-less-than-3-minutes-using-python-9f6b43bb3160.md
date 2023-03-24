# 使用 Python 在不到 3 分钟的时间内实现人脸检测

> 原文：<https://towardsdatascience.com/implement-face-detection-in-less-than-3-minutes-using-python-9f6b43bb3160?source=collection_archive---------5----------------------->

## 使用这个简单的代码将人脸检测功能添加到您的应用程序中

![](img/bb519a5ba2d9eafeac99765fb6c96419.png)

Face detection (Image by [teguhjati pras](https://pixabay.com/users/teguhjatipras-8450603/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=3252983) from [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=3252983))

人脸检测是人工智能最常见的应用之一。从智能手机中的相机应用到脸书的标签建议，人脸检测在应用程序中的使用每天都在增加。

> 人脸检测是计算机程序在数字图像中识别和定位人脸的能力。

随着应用程序中对人脸检测功能的需求不断增加，每个人都希望在自己的应用程序中使用人脸检测，这样他们就不会在竞争中落后。

在这篇文章中，我将教你如何在不到 3 分钟的时间内为自己建立一个人脸检测程序。

如果尚未安装以下 python 库，则需要进行安装:

```
opencv-python
cvlib
```

下面是导入所需 python 库、从存储中读取图像并显示它的代码。

```
# import libraries
import cv2
import matplotlib.pyplot as plt
import cvlib as cvimage_path = 'couple-4445670_640.jpg'
im = cv2.imread(image_path)
plt.imshow(im)
plt.show()
```

![](img/54843fb65676b9c556865a2e54aa48b3.png)

Couple Photo (Image by [Sonam Prajapati](https://pixabay.com/users/sonamabcd-7296816/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=4445670) from [Pixabay](https://pixabay.com/?utm_source=link-attribution&utm_medium=referral&utm_campaign=image&utm_content=4445670))

在加载的图像中检测人脸，在检测到的人脸周围绘制一个边界框，并显示带有检测到的人脸的最终图像的代码如下。

```
faces, confidences = cv.detect_face(im)# loop through detected faces and add bounding box
for face in faces: (startX,startY) = face[0],face[1]
    (endX,endY) = face[2],face[3] # draw rectangle over face
    cv2.rectangle(im, (startX,startY), (endX,endY), (0,255,0), 2)# display output        
plt.imshow(im)
plt.show()
```

![](img/2611d3e6db94d834285f1adffbb81ac8.png)

Result of Face Detection on couple image

你已经准备好了面部检测程序。就这么简单！

***觉得这个帖子有帮助？*** *在下面留下你的想法作为评论。*

[**点击这里**](https://medium.com/@sabinaa.pokhrel) 阅读我其他关于 AI/机器学习的帖子。

*要了解更多关于 cvlib 库的信息，可以访问下面的链接。*

 [## cvlib

### 用于 Python 的高级易用开源计算机视觉库。它的开发重点是实现简单的…

www.cvlib.net](https://www.cvlib.net/) 

*为了理解人脸检测是如何工作的，这里有一些进一步的阅读:*

 [## FaceNet:人脸识别和聚类的统一嵌入

### 尽管最近在人脸识别领域取得了重大进展，但实现人脸验证和识别…

arxiv.org](https://arxiv.org/abs/1503.03832) [](https://www.coursera.org/learn/convolutional-neural-networks) [## 卷积神经网络| Coursera

### 从 deeplearning.ai 学习卷积神经网络。本课程将教你如何构建卷积神经网络…

www.coursera.org](https://www.coursera.org/learn/convolutional-neural-networks)  [## 深度学习计算机视觉 CNN、OpenCV、YOLO、SSD 和 GANs

### 深度学习计算机视觉使用 Python & Keras 实现 CNN、YOLO、TFOD、R-CNN、SSD & GANs+A 免费…

www.udemy.com](https://www.udemy.com/master-deep-learning-computer-visiontm-cnn-ssd-yolo-gans/)