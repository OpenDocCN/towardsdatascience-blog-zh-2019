# 用可变自动编码器和张量流生成假的 FIFA 19 足球运动员

> 原文：<https://towardsdatascience.com/generating-fake-fifa-19-football-players-with-variational-autoencoders-and-tensorflow-aff6c10016ae?source=collection_archive---------13----------------------->

更新 25/01/2021:我会很快更新这些帖子，在 Twitter 上关注我以获得更多信息[https://twitter.com/mmeendez8](https://twitter.com/mmeendez8)

这是我的第三篇关于可变自动编码器的文章。如果你想赶上数学，我推荐你查看[我的第一篇文章](https://mmeendez8.github.io/projects/2019_vae_tensorflow.html)。如果你想跳过这一部分，直接用 VAEs 做一些简单的实验，那就来看看我在[的第二篇文章](https://mmeendez8.github.io/projects/2019_vae_tensorflow.html)，在那里我展示了这些网络有多有用。如果你只是想看看神经网络如何创造足球运动员的假脸，那么你来对地方了！[阅读更多](https://mmeendez8.github.io/2019/02/06/vae-fifa.html) …