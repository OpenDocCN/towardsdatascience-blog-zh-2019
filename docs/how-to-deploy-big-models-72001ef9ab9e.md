# 如何部署大模型

> 原文：<https://towardsdatascience.com/how-to-deploy-big-models-72001ef9ab9e?source=collection_archive---------18----------------------->

## AppEngine 和 Kubernetes 上的 500 MB py torch

![](img/36ab6a59637c568de45d7e0ec5103979.png)

我最近[部署了](https://duet.li)一个 500MB Pytorch 模型。出乎意料的难！在这篇文章中，我记录了我犯的错误和权衡。

对于非批处理，在 CPU 上运行几乎和 GPU 一样快，所以如果可以的话，我建议从 GPU 开始。

## 简单的方法会失败

[Tensorflow Serving](https://www.tensorflow.org/tfx/guide/serving) 看起来还可以，但是将我们的模型从 Pytorch 转换成 ONNX 可能会有[的困难](https://medium.com/styria-data-science-tech-blog/running-pytorch-models-in-production-fa09bebca622)。我们还想让本地代码尽可能简单，以便于开发。为了确保服务器快速启动，我将模型复制到带有. gitignore 条目的代码库中。我将 pytorch-pretrained-bert 添加到我的 requirements.txt 中，这增加了超过 300MB 的依赖项。

首先我尝试了**谷歌 AppEngine (GAE)标准环境**。当我尝试部署时，远程构建失败，并显示一条神秘的错误消息。我最终发现这是因为我的应用程序已经超出了远程构建环境的最大值。根据支持论坛，没有办法增加内存限制。

我想也许 **Heroku** 会支持更大的有效载荷，但它原来有同样的 500MB 限制和一个更合理的错误信息:“编译的 slug 大小:634M 太大了(最大是 500M)。”

然后我设置了一个 [docker 容器](https://github.com/JasonBenn/duet/blob/master/Dockerfile)来部署在 **GAE 灵活环境**上。这也失败了，因为机器在运行时耗尽了内存，并被无声地终止了。我通过查看内存使用量发现了这一点，并看到它超过了默认限制。我[提高了](https://github.com/JasonBenn/duet/blob/master/app.yaml)的内存需求，它开始服务了！但是它在服务器上仍然非常慢(仅加载静态页面就需要 30 秒)，但是在本地却很快。

我发现主要的区别是 **Gunicorn vs Flask** 开发服务器。由于某种原因，Gunicorn 没有对健康检查做出反应。我试图在 Flask 中手动实现这些检查，但是没有用。然后我尝试用 Flask dev 服务器进行部署，我知道这是不允许的，因为它不提供像请求队列这样的东西。每个请求都很快，但是 flask dev 服务器不是按比例构建的。我在本地用 Gunicorn 测试了 Docker 容器，发现它和 AppEngine 上的一样慢。我尝试了许多神奇的配置选项，包括详细日志记录，但没有任何帮助或揭示问题。在评估了许多备选方案后，我最终选定了[女服务员](https://docs.pylonsproject.org/projects/waitress/en/stable/#)。这是非常糟糕的记录，但我最终找到了神奇的调用。成功了！

出于某种原因，部署仍然需要一个半永恒的时间(大约 15-30 分钟)，所以为了加快速度，我在本地执行我的`docker build`，然后再执行`docker push`已经构建好的映像。

## 添加 GPU 支持

AppEngine 不支持 GPU，所以我用了**Google Kubernetes Engine**(GKE)。事实证明，仅仅给你的节点添加 GPU 是不够的。您还需要安装驱动程序。创建集群和设置服务应该是这样的:

```
gcloud config set compute/zone us-west1-b
cloud container clusters create duet-gpu --num-nodes=2 --accelerator type=nvidia-tesla-k80,count=1 --image-type=UBUNTU
gcloud container clusters get-credentials duet-gpu
kubectl apply -f [https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/ubuntu/daemonset-preloaded.yaml](https://raw.githubusercontent.com/GoogleCloudPlatform/container-engine-accelerators/master/nvidia-driver-installer/ubuntu/daemonset-preloaded.yaml)
kubectl apply -f [cloud.yaml](https://github.com/JasonBenn/duet/blob/master/cloud.yaml)
```

然后使用`kubectl get ingress`找到你的公共 IP。这需要一分钟来调配。

像 Heroku 和 AppEngine 这样的托管服务试图将开发人员与 DevOps 隔离开来，但这种承诺似乎并没有扩展到大型模型，尤其是 Pytorch 模型。在这篇文章中，我展示了如何使用 Flask 和服务员部署到 GAE 灵活环境和 GKE。现在我有了一个有效的食谱，以后就可以轻松地主持更多的节目了。好的一面是，我们仍然可以从 GAE 免费获得自动缩放、日志记录等功能。对 GKE 来说，这只是一个配置。

你找到更简单的方法了吗？留个条！