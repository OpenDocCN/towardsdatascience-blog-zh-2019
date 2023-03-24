# 使用 Docker 在本地部署您的 RShiny 应用程序

> 原文：<https://towardsdatascience.com/deploy-your-rshiny-app-locally-with-docker-427386010eba?source=collection_archive---------10----------------------->

![](img/39ced853d94a75105d8d8f649ede9e4f.png)

我最喜欢在本地部署 [RShiny](https://shiny.rstudio.com/) 的方式是简单地将它打包成一个 [docker](https://www.docker.com/) 映像并运行它。在 docker 中运行任何应用程序都使其易于移植，并且是 2019 年普遍接受的分发应用程序的方式。

这个解决方案不需要 rshiny-server 或 shinyapps.io，这并不是因为我反对这些解决方案。我只是倾向于坚持一些最喜欢的部署方法，以防止我的头从身体上直接旋转下来。；-)

如果你不熟悉 docker，那么我有一个[免费课程](https://www.dabbleofdevops.com/offers/zWv4nh3e/checkout)可供选择。第一个模块足以让你在一两个小时内启动并运行。在大多数情况下，如果你能使用命令行，你可以使用 docker。

# 在 Docker 中打包你的闪亮应用

现在我们已经介绍了一些日常工作，让我们开始建立你的 docker 形象。像任何项目一样，您希望将所有相关代码放在一个目录中。这样 docker 构建过程就可以访问它。

# 项目目录结构

```
➜ 2019-11-rshiny-app-docker tree
.
├── Dockerfile
├── README.md
├── app.R
└── environment.yml

0 directories, 4 files
```

这是一个非常简单的例子，一个 R 文件服务于我们的 RShiny 应用程序，*应用程序。R* 。 *Dockerfile* 有我们的构建指令， *environment.yml* 列出了我们的 conda 包，即 r-shiny 和 r-devtools。

我个人喜欢用康达安装我所有的科学软件。如果你喜欢用另一种方式安装你的软件，那就试试吧。重要的是，您的依赖项已经安装好，可以运行您的 RShiny 应用程序了。

# RShiny 应用程序

这是一个非常简单的例子，直接取自 RShiny examples github repo，只做了一些修改。

# Dockerfile 文件

docker 文件有我们的 docker 构建指令。它的语法相当简单。使用 FROM 选择基础映像，使用 run 在构建期间运行命令，使用 copy 复制文件，使用 CMD 定义启动命令。

*确保您已经制作了应用程序。chmod 777 应用程序的可执行文件。否则下一步就没用了。*

# Conda 环境定义文件

同样，用 conda 安装您的软件是可选的。简直是我最喜欢的安装方式。；-)

```
name: r-shiny
channels:
  - conda-forge
  - defaults
dependencies:
  - python=3.6
  - r-devtools
  - r-shiny
```

# 构建 Docker 容器

现在我们已经准备好了所有的文件，让我们构建并运行 docker 容器！

```
docker build -t r-shiny-app .
```

# 运行我们的 RShiny 应用程序

现在我们已经构建了我们的应用程序，我们可以运行它了！

```
docker run -it -p 8080:8080 r-shiny-app
```

你应该看到一条消息说*监听*[*http://0 . 0 . 0 . 0:8080*](http://0.0.0.0:8080)*。一旦你看到一切都准备好了！在 localhost:8080 上打开您的浏览器，查看您的应用程序的运行情况！*

*最初发表于*[*【https://www.dabbleofdevops.com】*](https://www.dabbleofdevops.com/blog/deploy-your-rshiny-app-locally-with-docker)*。*