# 用 Docker 和 Golang 设置 GitHub 包注册表

> 原文：<https://towardsdatascience.com/setting-up-github-package-registry-with-docker-and-golang-7a75a2533139?source=collection_archive---------15----------------------->

*注:此文最初发布于*[*martinheinz . dev*](https://martinheinz.dev/blog/6)

通常，对于任何编程语言，要运行您的应用程序，您都需要创建某种包(`npm`表示 *JavaScript* ，`NuGet`表示 *C#* ，...)然后存放在某个地方。在 *Docker* 的情况下，人们通常只是将他们的图像扔进 *Docker Hub* 中，但是我们现在有了新的选择...

另一个选择是*GitHub Package Registry*——它已经测试了一段时间，似乎越来越多的人开始使用它，所以感觉是时候探索它的功能了，这里专门针对 *Docker* 和 *Go* 项目。

这篇文章是系列文章*“你的下一个 Golang 项目所需要的一切”*的一部分，如果这听起来很有趣，请点击这里查看上一部分[。](https://gist.github.com/MartinHeinz/134729d0d26ca88b5afb23961759de6e)

*注意:本文适用于任何使用 docker 图像的项目，而不仅仅是 Golang。*

![](img/85106e33384b61a5b12587815be32d8a.png)

# 为什么使用 GitHub 注册表

首先，你为什么要考虑从，比方说， *Docker Hub* 或任何其他注册表，切换到 *GitHub 包注册表*:

*   如果你已经在使用 *GitHub* 作为你的 SCM，那么使用 *GitHub Package Registry* 是有意义的，因为它允许你把所有东西放在一个地方，而不是把你的包推到别处。
*   *GitHub* — *动作*还有另一个闪亮的新( *beta* )特性，你可以结合 *GitHub 包注册表*来利用它(在另一个帖子中有更多关于这个的内容……)。
*   即使我认为 *Docker* 图像优于例如`npm`包，如果你更喜欢*非 Docker* 工件，你也可以将它们推送到 *GitHub 包注册表*。

# 我们开始吧！

那么，现在，让我们来看看如何使用它。让我们从构建和标记您的图像开始:

*注意:如果你正在使用 Go，那么你可能想在这里查看* [*我的库*](https://github.com/MartinHeinz/go-project-blueprint) *，这里所有的 GitHub 包注册表功能都已经绑定到 Makefile 目标中。*

为了能够将图像推送到 *GitHub 包注册表*，您需要使用上面显示的格式来命名它——这实际上只是带有您的 *GitHub* 用户名和存储库名称的注册表的 URL。

接下来，我们如何访问它？

首先，为了能够用 *GitHub 包注册表*认证我们自己，我们需要创建个人访问令牌。这个访问令牌必须有`read:packages`和`write:packages`作用域，此外，如果您有一个私有存储库，您还必须包括`repo`作用域。

在 *GitHub* 帮助网站上已经有非常好的关于如何创建个人令牌的指南，所以我不打算在这里复制和粘贴步骤。你可以在这里阅读

现在，我们有了个人令牌，让我们登录:

最后，是时候推广我们的形象了:

很明显，你也可以把图像拉出来:

# 在 CI/CD 渠道中使用它

对于 GitHub 包注册表，我们最不想做的事情就是将它与 CI/CD 工具集成在一起。我们来看看如何用 *Travis* 来完成(*全* `*.travis.yml*` *可在我的资源库* [*这里*](https://github.com/MartinHeinz/go-project-blueprint/blob/master/.travis.yml) ):

正如你在上面看到的，你可以像在你的机器上一样运行`build`和`push`，唯一的区别是`docker login`。这里，我们使用在 *Travis* UI 中指定的环境变量。用户名通过`-u`参数传递给登录命令，密码使用`echo`传递给`stdin`，这是必需的，这样我们就不会在 *Travis* 日志中打印出我们个人的 *GitHub* 令牌。

那么，我们如何设置这些环境变量呢？这些是步骤:

*   导航到存储库的 *Travis* 作业设置，例如[https://Travis-ci . com/Martin Heinz/go-project-blue print/settings](https://travis-ci.com/MartinHeinz/go-project-blueprint/settings)
*   向下滚动到*环境变量*部分
*   将变量名分别设置为`DOCKER_USERNAME`和`DOCKER_PASSWORD`。在密码( *GitHub* 令牌)情况下，确保构建日志中的*显示值设置为假。*
*   点击*添加*并触发构建

如果你没有使用 *Travis* 并且想要使用 *GitHub Webhook* 来触发构建，那么你可以使用`[RegistryPackageEvent](https://developer.github.com/v3/activity/events/types/#registrypackageevent)`。如果您正在使用 *Jenkins* 、 *OpenShift* 或 *Kubernetes* ，并且希望每次在 *GitHub Package Registry* 中发布或更新您的包时触发部署，这可能会很有用。

# 告诫的话

使用 *GitHub 包注册表*时，你要记住的一件事是，你不能删除你推送到注册表的包。这样你就不会破坏依赖于你的包的项目。你可以从 *GitHub Support* 请求删除包，但是你不应该指望他们真的删除任何东西。

# 结论

希望看完这个你给 *GitHub 包注册表*打个预防针。如果你还没有 *beta* 权限，可以在这里注册[。如果你有任何问题，不要犹豫，联系我，或者你也可以看看我的](https://github.com/features/package-registry/signup)[库](https://github.com/MartinHeinz/go-project-blueprint)，在那里你可以找到 *GitHub 包注册表*用法的例子。

# 资源

*   [https://github.com/features/package-registry](https://github.com/features/package-registry)
*   [https://help . github . com/en/articles/about-github-package-registry](https://help.github.com/en/articles/about-github-package-registry)