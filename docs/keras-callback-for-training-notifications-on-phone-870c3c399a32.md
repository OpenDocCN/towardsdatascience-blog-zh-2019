# 电话培训通知的 Keras 回拨

> 原文：<https://towardsdatascience.com/keras-callback-for-training-notifications-on-phone-870c3c399a32?source=collection_archive---------32----------------------->

## 创建一个简单的 Keras 回调，将培训进度作为通知发送到您的手机。

![](img/4e0475de452e138a097a4e68bd05795a.png)

Photo by [Jonah Pettrich](https://unsplash.com/@jonah_jpg?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

训练神经网络通常需要几个小时甚至几天才能完成，当您无法监控训练进度时，这可能会令人沮丧。这个想法很简单，很容易实现一个回调来发送通知到你想要的应用程序。

为了发送通知，我使用了一个名为 [Notifiers](https://pypi.org/project/notifiers/) 的 python 库。它支持大多数流行的应用程序，如:

*   容易做的事情
*   推杆
*   简单推送
*   松弛的
*   谷歌邮箱
*   电报
*   特维利奥

还有很多。看文档[这里](https://notifiers.readthedocs.io/en/latest/)。

在这个介绍中，我将使用 PushBullet。所有的服务都差不多。从所需的服务获取一个 API 令牌，并在身份验证中使用它。

你完了！！

谢谢你。