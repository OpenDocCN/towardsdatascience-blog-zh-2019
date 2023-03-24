# 从 Tensorflow 1.0 到 PyTorch &回到 Tensorflow 2.0

> 原文：<https://towardsdatascience.com/from-tensorflow-1-0-to-pytorch-back-to-tensorflow-2-0-f2f8a4c716b7?source=collection_archive---------20----------------------->

为了开发深度学习项目，我们必须在 Tensorflow 和 Pytorch 这样的库之间来回移动，这是一种常见的情况。

![](img/bb8424933499d3a66b9dae4d59c42c70.png)

Photo by [Simon Matzinger](https://unsplash.com/@8moments?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

我在 2015 年左右开始了我的机器学习之旅，当时我快十几岁了。由于对这个领域没有清晰的认识，我读了很多文章，看了大量 YouTube 视频。我不知道这个领域是什么，也不知道它是如何运作的。那是谷歌流行的机器学习库 Tensorflow 发布的时间。

## 步入张量流

Tensorflow 于 2015 年 11 月作为“机器智能开源软件库”发布。它一发布，人们就开始投身其中。GitHub 中的分叉数量呈指数增长。但是它有一个根本性的缺陷:'*静态图*'。

在幕后，Tensorflow 使用静态图实现来管理数据流和操作。这意味着从程序员的角度来看，您必须首先创建一个完整的模型架构，将其存储到一个图中，然后使用“会话”启动该图。虽然这对于创建生产模型来说是一个优势，但是它缺乏*的 pythonic 性质*。

社区开始抱怨这个问题，Tensorflow 团队创建了“渴望执行”来解决这个问题。但是，它仍然没有被添加为默认模式。Tensorflow 也有很多(有点太多)API，以至于令人困惑。同样的函数定义也有很多冗余。

## 转移到 PyTorch

这段时间(2017 年左右)，我要做一个项目，需要同时训练两个神经网络，两者之间有数据流。我用 Tensorflow 开始我的工作，因为它是一个烂摊子。多个图，图与图之间切换，手动选择参数进行更新……纯粹一塌糊涂。最后我放弃了 Tensorflow，开始用 PyTorch。

使用 PyTorch 要容易得多…具体的定义，稳定的设计，通用的设计结构。我非常喜欢它。尽管如此，我没有完全放弃 Tensorflow，我使用 Keras 和 Tensorflow 和 Tensorboard 参加了研讨会和讲座。后来，我完成了我的项目，在 PyTorch 上又做了一些，但是我觉得这个社区还不够强大。

## 返回张量流

在此期间，Tensorflow 变得越来越强大，与软件包紧密集成，通过 GPU 轻松训练，并作为 Tensorflow 服务为生产提供强大支持。感觉 TensorFlow 正在向一个新的方向发展。

2019 年 10 月 1 日，TF2.0 首个稳定版发布。新特性包括与 Keras 的紧密集成、默认的急切执行(最后😅)、**功能和非会话**、多 GPU 支持等等。另一个漂亮的特性是 *tf.function* decorator，它将代码块转换成优化的图形，从而提供更快的执行速度。

此外，像 *tf.probability、tensorflow_model_optimization、tf.agents* 这样的 API 将深度学习的能力推向了其他领域。作为一个建议，我希望看到不同的强化学习包，像 Tensorflow 代理，TFRL(松露)，多巴胺合并成一个单一的 API。

谢谢大家！！