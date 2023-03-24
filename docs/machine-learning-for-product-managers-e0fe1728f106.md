# 面向产品经理的机器学习简介

> 原文：<https://towardsdatascience.com/machine-learning-for-product-managers-e0fe1728f106?source=collection_archive---------25----------------------->

## 理解机器学习如何在引擎盖下工作，而不要过于技术化。

想象一个石头剪子布游戏…

![](img/a69618ae21945fe7dbd77e4c1814841c.png)

而且，你想写代码，这样当你用石头、布或剪刀移动你的手时，计算机会识别它并与你对弈。

想想写代码实现这样一个游戏。你必须从相机中提取图像，查看这些图像的内容，等等。

这对你来说会有很多代码要写。不出所料，给游戏编程最终会变得极其复杂和不可行。

# 传统编程

在传统编程中——多年来这一直是我们的谋生之道——我们考虑用编程语言来表达规则。

![](img/0b61bcb4f1f7d4c0669fa2b6ed431e7f.png)

> 传统编程中，程序员定义的规则一般作用于数据，给我们答案。

例如，在石头剪子布中，数据将是一幅图像，规则将是所有查看该图像中的像素以尝试确定对象的`if-then`语句。

传统编程的问题是，它将涵盖所有可预测的基础。但是，如果我们不知道某个规则的存在呢？

# 机器学习

机器学习扭转了传统编程的局面。它通过设置某些参数来产生期望的输出，从而得出预测。

这是一种复杂的说法，模型从错误中学习，就像我们一样。

参数是函数中的那些值，当模型试图学习如何将这些输入与这些输出相匹配时，这些值将被设置和更改。

把这些函数想象成你生活中的时刻，把这些参数想象成你从这些时刻中学到的东西。你希望永远不要重蹈覆辙，就像你希望你的模特那样。

![](img/6650d9bd3a67fdba3ad3e928618e27df.png)

> 在机器学习中，我们不是用代码编写和表达规则，而是向**提供大量**答案，标记这些答案，然后让机器推断出将一个答案映射到另一个答案的规则。

例如，在我们的石头剪刀布游戏中，我们可以说明石头的像素是多少。也就是说，我们告诉电脑，“嘿，这就是石头的样子。”我们可以这样做几千次，以获得不同的手，肤色等。

如果机器能够找出这些之间的模式，我们现在就有了机器学习；我们有一台为我们决定这些事情的计算机！

# 我如何运行一个看起来像这样的应用程序？

![](img/1cc8567bb5e4223e0b2c6bb6c94a6dda.png)

在上面显示的培训阶段，我们已经培训了这方面的模型；这个模型本质上是一个神经网络。

![](img/8d2e24908fd3791a4fd71954281ed410.png)

在运行时，我们会传入我们的数据，我们的模型会给出一个叫做预测的东西。

例如，假设您已经在许多石头、纸和剪刀上训练了您的模型。

![](img/213dfe0deefd5e7403b1455279ca786c.png)

当你对着摄像头举起拳头时，你的模型会抓取你拳头的数据，并给出我们所说的预测。

我们的预测是，有 80%的可能性这是石头，有 10%的可能性这是剪刀或布。

机器学习中的许多术语与传统编程中的术语略有不同。我们称之为训练，而不是编码和编译；我们称之为推理，从推理中得到预测。

您可以使用 [TensorFlow](https://www.tensorflow.org/) 轻松实现这个模型。下面是创建这种神经网络的一段非常简单的代码:

```
model  = tf.keras.models.Sequential([
  tf.keras.layers.Flatten(input_shape=(150, 150, 3)),
  tf.keras.layers.Dense(512, activation='relu'),
  tf.keras.layers.Dense(3, activation='softmax')
])model.compile(loss='categorical_crossentropy', optimizer='rmsprop')model.fit(..., epochs=100)
```

# 潜得更深

以下是我在过去几个月里收集的一些很棒的资源。我希望他们能帮你找到正确的方向。

## MOOCs

*   经典:[吴恩达的 Coursera 机器学习](https://www.coursera.org/learn/machine-learning)
*   [Udacity 机器学习简介](https://www.udacity.com/course/intro-to-machine-learning--ud120)
*   [谷歌的 Udacity 深度学习](https://www.udacity.com/course/deep-learning--ud730)
*   [EdX 对 AI 的介绍](https://www.edx.org/course/artificial-intelligence-ai-columbiax-csmm-101x-0)
*   [Fast.ai](http://Fast.ai) 的 7 周[实用深度学习](http://course.fast.ai/)，老师杰瑞米·霍华德(Kaggle 前任总裁，Enlitic 和 [fast.ai](http://fast.ai) 创始人)

## 教程

*   [韦尔奇实验室神经网络简介](https://www.youtube.com/watch?v=bxe2T-V8XRs)，几个 4-5 分钟的精彩视频
*   [谷歌的 Tensorflow 教程](https://www.tensorflow.org/tutorials/)
*   谷歌 3 小时课程[学习 TensorFlow 和深度学习，没有博士](https://cloud.google.com/blog/big-data/2017/01/learn-tensorflow-and-deep-learning-without-a-phd)
*   斯坦福的[深度学习教程](http://ufldl.stanford.edu/tutorial/)

## 讲座和播客

*   [AI 与深度学习](http://a16z.com/2016/06/10/ai-deep-learning-machines/)。从机器智能的类型到算法之旅，16z 交易和研究团队负责人 Frank Chen 在这张幻灯片演示中向我们介绍了人工智能和深度学习的基础知识(及其他)。
*   [a16z 播客:机器学习初创公司中的产品优势](http://a16z.com/2017/03/17/machine-learning-startups-data-saas/)。很多机器学习初创公司最初在与大公司竞争时会感到有点“冒名顶替综合症”，因为(论点是这样的)，那些公司拥有所有的数据；当然我们不能打败它！然而，初创公司有很多方法可以、也确实成功地与大公司竞争。本播客的嘉宾认为，如果你在此基础上构建了正确的产品，即使只有相对较小的数据集，你也可以在许多领域取得巨大的成果。
*   [当人类遇上 AI](http://a16z.com/2016/06/29/feifei-li-a16z-professor-in-residence/) 。安德森·霍洛维茨杰出的计算机科学客座教授费-李非(以的名义发表文章)是斯坦福大学的副教授，他认为我们需要注入更强的人文主义思维元素来设计和开发能够与人和在社交(包括拥挤的)空间中共处的算法和人工智能。
*   [“面向智能计算机系统的大规模深度学习”](https://www.youtube.com/watch?v=QSaZGT4-6EY)，2016 年 3 月，谷歌科技与杰夫·迪恩在首尔校园的对话。