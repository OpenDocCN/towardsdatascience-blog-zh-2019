# 墙上的魔镜魔镜谁是他们当中最喜欢猫的？

> 原文：<https://towardsdatascience.com/mirror-mirror-on-the-wall-who-is-the-most-cat-of-them-all-b078a0b7f632?source=collection_archive---------32----------------------->

## 询问训练过的神经网络对世界的看法。

![](img/ecbad1142fc0e035b64517bff431352c.png)![](img/df166206b6a473a98d3b25e06ba58cef.png)

Riddle me this: How does a cat look like to you?

# 训练神经网络

用神经网络进行图像分类现在已经不是什么新鲜事了。通常，训练神经网络的过程是这样的:

![](img/2e8f7960b1998ecc72a90b5fdb7fb224.png)

为了训练这样一个神经网络，我们需要大量的训练数据。训练数据中的每一项都由一幅图像和一个标签组成(标签告诉我们图像上显示的是什么对象)。

网络显示图像并产生一些输出(红框)。将此输出与预期结果(绿框)进行比较。考虑到网络的“错误程度”,下次它会做得更好。

神经网络的输出是一系列数字。列表的长度等于网络可以检测到的不同对象的数量(上图中为 2)。列表中的每个数字对应于该对象出现在当前图像中的概率。

# 逆转训练

我们已经看到神经网络是如何被训练用于图像分类的。但它到底学到了什么？

有一篇很好的博文描述了一种可视化神经网络各个部分所学内容的方法:

[](/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030) [## 如何用 40 行代码可视化卷积特征

### 发展解释网络的技术是一个重要的研究领域。本文解释了您可以如何…

towardsdatascience.com](/how-to-visualize-convolutional-features-in-40-lines-of-code-70b7d87b0030) 

扩展本文的思想，我们可以使用网络来生成在特定类别中产生最高置信度的图像。例如:“生成一张你最确信是一只猫的图片。”

下图显示了实现这一目标所需的培训过程中的细微变化:

![](img/a7d1f2b3e0f33c7d124666cd85f32db2.png)

我们不是从随机网络开始，而是从随机图像开始。预期的结果是对我们想要可视化的类赋予 100%的置信度。

[](https://www.datadriveninvestor.com/2019/03/03/editors-pick-5-machine-learning-books/) [## DDI 编辑推荐:5 本让你从新手变成专家的机器学习书籍|数据驱动…

### 机器学习行业的蓬勃发展重新引起了人们对人工智能的兴趣

www.datadriveninvestor.com](https://www.datadriveninvestor.com/2019/03/03/editors-pick-5-machine-learning-books/) 

然后更新最初的随机图像，使其更符合预期的结果。运行这个程序几次(希望如此)会揭示网络认为猫长什么样。

## ImageNet

对于这个实验，我使用了一个在 [ImageNet](http://www.image-net.org/) 数据集上预先训练的神经网络。这个数据集包含超过 1400 万张图片，每张图片属于 1000 个类别中的一个。

我在谷歌实验室上运行了可视化程序。如果你需要帮助在 Colab 上做你自己的深度学习实验，这里有一篇博客文章，详细介绍了如何设置你的环境:

[](https://medium.com/datadriveninvestor/setting-up-google-colab-for-deeplearning-experiments-53de394ae688) [## 为深度学习实验设置 Google Colab

### 不是每个人都有带 GPU 的本地机。幸运的是，你可以免费得到一个。

medium.com](https://medium.com/datadriveninvestor/setting-up-google-colab-for-deeplearning-experiments-53de394ae688) [](https://medium.com/datadriveninvestor/speed-up-your-image-training-on-google-colab-dc95ea1491cf) [## 在 Google Colab 上加速你的图像训练

### 获得一个因素 20 加速训练猫对狗分类器免费！

medium.com](https://medium.com/datadriveninvestor/speed-up-your-image-training-on-google-colab-dc95ea1491cf) 

我的实现严重依赖于这个笔记本:

[](https://nbviewer.jupyter.org/github/anhquan0412/animation-classification/blob/master/convolutional-feature-visualization.ipynb) [## nbviewer 笔记本

### 想法如下:我们从包含随机像素的图片开始。我们将评估模式下的网络应用于…

nbviewer.jupyter.org](https://nbviewer.jupyter.org/github/anhquan0412/animation-classification/blob/master/convolutional-feature-visualization.ipynb) 

## 1000 节课

既然一切都准备好了，我就开始实验。为 imagenet 中 1000 个不同的类中的每一个生成一个图像需要几个小时，但谢天谢地 GoogleColab 坚持下来了，没有崩溃。

生成的图像大多不是很有趣。我筛选了整个堆栈(是的，通过所有 1000 张图片…)并挑选了最好的。

以下是我的首选，大致分为几类。尽情享受吧！

(如果没有时间看完所有图片，可以滚动到底部。最好的在那里)

# 随机对象

这些图像通常显示一个对象的多次出现，相互堆叠。

## 衣服

![](img/e075727082793cd95361363473b7ff78.png)

## 水果

![](img/f06f68e01914a9047cd3a4c23b5db19f.png)

## 混搭程式

![](img/af93d54611d1b2c8023e1e6dffc1d07b.png)

## 军事保护

![](img/da1a8eefe0485199e01213d15421c60f.png)

## 军事武器

![](img/9f9f4a70065b71d22f61d4c11b59b762.png)

## 数字

![](img/a6cf72d7a0d5320c4e51d36cedc437ef.png)

## 涂鸦

![](img/4e312a10c0962af3776e1aab0f338129.png)

## 技术的

![](img/3ff0a83ea31918f1a183fcb66e2122ea.png)![](img/f84a6d6cd9393c8f5cb5a6364d2aa144.png)

## 工具

![](img/0da1018fbdd711c05112f99a34f93fac.png)

# 景色

这些图像通常不仅显示生成它们的物体，还显示某种周围的场景。

## 衣服

![](img/da2d6f1904d380b9ca87eb478c72a7a4.png)

## 食物

![](img/0bdcb57941fa24a3a3125c419a871d67.png)

## 运动

![](img/95a278d5a5b18e364dd4a5131fec7f16.png)![](img/a752c2b9a2cafb57a2d1691e485b795f.png)

## 商店

![](img/73e214948d0a2b543d1f446ee133c50b.png)

# 动物

啊，是的，动物。他们不可爱吗？嗯，除了蟑螂可能…

## 国产+熊

![](img/6aa22455a1d49922fa16e24b81603066.png)![](img/2cdf32c1d924c3bcaeb5840782cc8324.png)

## 昆虫

![](img/731191da13f776aa06341ac93fbad1b6.png)![](img/b06420b8852fc9f268cdf7476e4eefe5.png)

## 海洋的

![](img/65b4a6106a0c4063e8f171a714396042.png)![](img/b9119500de02b92df2b3b6c46e3ba29c.png)

## 厚皮

![](img/051de2513aaa7b518fd83434cfbf6ce5.png)

## 热带的

![](img/c7497e01d33853fe378976eae89ae5c9.png)![](img/2104d1312a4ebaf8d77542717d1de8b0.png)

# 最好的

下面的图片是我个人最喜欢的，拼图玩具以很大优势赢得了第一名。

## 个人最爱

![](img/c728cde233b971e105fbf4f9571c836b.png)![](img/694d1d875d18340a5aef87961466ed4e.png)![](img/5133b4c6cac70161dd50c014e793c936.png)![](img/90fba861bf02e5263a1f621e0cc8113a.png)![](img/8186a5047f73381eec92f574cb5b9c01.png)

## 自然的

![](img/b9a5c203f3f5d9035c210d22cd0774a6.png)![](img/c7337d112152b30f4dd588da8ef50e91.png)![](img/62325162027b9446f4ee6dfbfe44d0b7.png)

## 运动

![](img/971f35b88fec31048f7f660edbab71cb.png)![](img/6404badd13e2fb4455a67da39a1968b8.png)

## 技术的

![](img/55337882afabc789a9e236d42cd9d310.png)![](img/2040c8315671fb36a96baf0e63ed1754.png)![](img/c9860934073c701802aea38ae71f60f6.png)

就是这样。我希望你喜欢这场画展。让我知道哪个是你最喜欢的！