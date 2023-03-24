# 精简你的 Docker 图片

> 原文：<https://towardsdatascience.com/slimming-down-your-docker-images-275f0ca9337e?source=collection_archive---------1----------------------->

## 学习足够有用的 Docker 的第 4 部分

在本文中，您将了解如何加快 Docker 构建周期并创建轻量级映像。按照我们的食物比喻，我们要吃沙拉🥗当我们精简 Docker 图片时——不再有披萨、甜甜圈和百吉饼。

![](img/d5dd7c241673c3da81d81c3334316275.png)

在本系列的第 3 部分中，我们介绍了十几个需要了解的 Dockerfile 指令。如果你错过了，请点击这里查看这篇文章:

[](/learn-enough-docker-to-be-useful-b0b44222eef5) [## 学习足够的码头工人是有用的

### 第 3 部分:一打漂亮的 Dozen 文件指令

towardsdatascience.com](/learn-enough-docker-to-be-useful-b0b44222eef5) 

这是备忘单。

`FROM` —指定基础(父)图像。
`LABEL`—提供元数据。包含维护者信息的好地方。
`ENV` —设置持久的环境变量。
`RUN`—运行命令并创建一个图像层。用于将包安装到容器中。
`COPY` —将文件和目录复制到容器中。
`ADD` —将文件和目录复制到容器。可以备份本地。焦油文件。
`CMD` —为正在执行的容器提供命令和参数。参数可以被覆盖。只能有一个 CMD。
`WORKDIR` —设置后续指令的工作目录。
`ARG` —定义在构建时传递给 Docker 的变量。
`ENTRYPOINT` —为正在执行的容器提供命令和参数。争论一直存在。
`EXPOSE` —暴露一个端口。
`VOLUME` —创建目录挂载点以访问和存储持久数据。

现在，让我们看看如何在开发图像和提取容器时设计 docker 文件以节省时间。

# 贮藏

Docker 的优势之一是它提供了缓存来帮助您更快地迭代您的映像构建。

构建映像时，Docker 会逐步执行 Docker 文件中的指令，按顺序执行每个指令。在检查每条指令时，Docker 会在其缓存中查找可以重用的现有中间映像，而不是创建新的(副本)中间映像。

如果缓存无效，使其无效的指令和所有后续 Dockerfile 指令都会生成新的中间映像。一旦缓存失效，docker 文件中的其余指令也就失效了。

因此，从 docker 文件的顶部开始，如果基础映像已经在缓存中，它将被重用。那是成功的。否则，缓存将失效。

![](img/d4549d2b9ae125434ed6b95ad31d976d.png)

Also a hit

然后，将下一条指令与缓存中从该基础映像派生的所有子映像进行比较。比较每个缓存的中间图像，以查看指令是否找到缓存命中。如果是缓存未命中，则缓存无效。重复相同的过程，直到到达 Dockerfile 文件的末尾。

大多数新指令只是简单地与中间图像中的指令进行比较。如果匹配，则使用缓存的副本。

例如，当在 Docker 文件中找到一个`RUN pip install -r requirements.txt`指令时，Docker 会在其本地缓存的中间映像中搜索相同的指令。新旧 *requirements.txt* 文件的内容不做对比。

如果您用新的包更新您的 *requirements.txt* 文件，并使用`RUN pip install`并希望用新的包名重新运行包安装，这种行为可能会有问题。一会儿我会展示几个解决方案。

与其他 Docker 指令不同，添加和复制指令确实需要 Docker 查看文件的内容，以确定是否存在高速缓存命中。参考文件的校验和与现有中间映像中的校验和进行比较。如果文件内容或元数据已更改，则缓存会失效。

这里有一些有效使用缓存的技巧。

*   通过用`docker build`传递`--no-cache=True`可以关闭缓存。
*   如果你要对指令进行修改，那么接下来的每一层都要频繁地重新构建。为了利用缓存，将可能发生变化的指令放在 docker 文件中。
*   链接`RUN apt-get update`和`apt-get install`命令以避免缓存未命中问题。
*   如果你使用一个包安装程序，比如带有 *requirements.txt* 文件的 pip，那么遵循下面的模型，确保你不会收到一个陈旧的中间映像，其中包含了 *requirements.txt* 中列出的旧包。

```
COPY requirements.txt /tmp/
RUN pip install -r /tmp/requirements.txt
COPY . /tmp/
```

这些是有效使用 Docker 构建缓存的建议。如果你有其他的，请在评论或推特上分享。

# 粉碎

Docker 图像可能会变大。你想让它们小一些，这样它们就可以快速拉动，使用较少的资源。让我们瘦下来你的形象！

![](img/2521021f325283dc5979407a6e8b5989.png)

Go for a salad instead of a bagel

Alpine 基础映像是一个完整的 Linux 发行版，没有其他内容。下载它通常不到 5 MB，但它需要你花更多的时间来编写构建一个工作应用程序所需的依赖关系代码。

![](img/358d41e7f43ab8711c9ab23b206237e2.png)

Alpine comes from Alps

如果您的容器中需要 Python，Python Alpine 构建是一个不错的折衷。它包含 Linux 和 Python，你提供几乎所有其他的东西。

我用最新的 Python Alpine build 和*print(“hello world”)*脚本构建的一个图像重 78.5 MB。这是 Dockerfile 文件:

```
FROM python:3.7.2-alpine3.8
COPY . /app
ENTRYPOINT [“python”, “./app/my_script.py”, “my_var”]
```

在 Docker Hub 网站上，基本映像被列为 29 MB。当构建子映像时，它下载并安装 Python，使它变得更大。

除了使用 Alpine 基本图像，另一种减小图像大小的方法是使用多级构建。这种技术也增加了 docker 文件的复杂性。

# 多阶段构建

![](img/fb6e1fc17293f90c2fd2634e9d6701d0.png)![](img/68bf459c511ebcf2f3f81597df73953a.png)

One stage + another stage = multistage

多阶段构建使用多个 FROM 指令。您可以有选择地将文件(称为构建工件)从一个阶段复制到另一个阶段。您可以在最终图像中留下任何不想要的内容。这种方法可以减少您的整体图像大小。

每个来自指令

*   开始构建的新阶段。
*   留下在先前阶段创建的任何状态。
*   可以使用不同的碱基。

这里有一个来自 [Docker 文档](https://docs.docker.com/develop/develop-images/multistage-build/)的多阶段构建的修改示例:

```
FROM golang:1.7.3 AS build
WORKDIR /go/src/github.com/alexellis/href-counter/
RUN go get -d -v golang.org/x/net/html  
COPY app.go .
RUN CGO_ENABLED=0 GOOS=linux go build -a -installsuffix cgo -o app .FROM alpine:latest  
RUN apk --no-cache add ca-certificates
WORKDIR /root/
COPY --from=build /go/src/github.com/alexellis/href-counter/app .
CMD ["./app"]
```

请注意，我们通过在 FROM 指令 to name 后面附加一个名称来命名第一个阶段。然后在 docker 文件后面的`COPY --from=`指令中引用命名的阶段。

在某些情况下，多阶段构建是有意义的，因为您将在生产中制作大量的容器。多阶段构建可以帮助你从图像尺寸中挤出最后一盎司(如果你用公制的话是克)。然而，有时多阶段构建会增加更多的复杂性，使映像更难维护，所以您可能不会在大多数构建中使用它们。在这里和这里[进一步讨论折衷](https://medium.com/@tonistiigi/advanced-multi-stage-build-patterns-6f741b852fae)[和高级模式](https://blog.realkinetic.com/building-minimal-docker-containers-for-python-applications-37d0272c52f3)。

相比之下，每个人都应该使用一个. dockerignore 文件来帮助他们的 Docker 图像保持苗条。

# 。dockerignore

*。作为一个对 Docker 有足够了解的人，你应该知道一些对 d̶a̶n̶g̶e̶r̶o̶u̶s̶有用的文件。*

。dockerignore 类似于. gitignore，是一个文件，里面有一个 Docker 用来匹配文件名的模式列表，在制作图像时排除。

![](img/099b285339470ac82f360bff510c84ab.png)

Just .dockerignore it

把你的。docker 文件与您的 docker 文件和其余的构建上下文放在同一个文件夹中。

当您运行`docker build`来创建一个图像时，Docker 会检查一个. dockerignore 文件。如果找到了，它就一行一行地检查文件，并使用 Go 的[文件路径。匹配规则](https://golang.org/pkg/path/filepath/#Match) —以及一些 [Docker 自己的规则](https://docs.docker.com/v17.09/engine/reference/builder/#dockerignore-file) —匹配要排除的文件名。考虑 Unix 风格的 glob 模式，而不是正则表达式。

所以`*.jpg`会排除带有*的文件。jpg* 扩展名。而`videos`将排除视频文件夹及其内容。

你可以解释你在做什么？使用以`#`开头的注释。

使用。从 Docker 映像中排除不需要的文件是个好主意。。dockerignore 可以:

*   帮你保守秘密。没有人希望他们的图像中有密码。
*   缩小图像尺寸。更少的文件意味着更小、更快的图像。
*   减少构建缓存失效。如果日志或其他文件发生变化，并且您的映像的缓存因此而失效，这将会减慢您的构建周期。

这些就是使用. dockerignore 文件的原因。查看[文档](https://docs.docker.com/v17.09/engine/reference/builder/#dockerignore-file)了解更多详情。

# 尺寸检验

让我们看看如何从命令行找到 Docker 图像和容器的大小。

*   要查看正在运行的容器的大概大小，可以使用命令`docker container ls -s`。
*   运行`docker image ls`显示图像的尺寸。
*   使用`docker image history my_image:my_tag`查看组成图像的中间图像的大小。
*   运行`docker image inspect my_image:tag`将显示你的图像的许多信息，包括每层的大小。层与组成整个图像的图像有细微的不同。但是在大多数情况下，你可以认为它们是一样的。如果你想深入研究图层和中间图像的错综复杂，可以看看奈杰尔·布朗的这篇伟大的文章。
*   安装和使用 [dive](https://github.com/wagoodman/dive) 包可以很容易地看到你的图层内容。

我在 2019 年 2 月 8 日更新了上述部分，以使用管理命令名称。在本系列的下一部分，我们将深入探讨常见的 Docker 命令。跟着我，确保你不会错过。

现在，让我们来看看一些最佳实践来减肥。

# 减少映像大小和构建时间的八个最佳实践

1.尽可能使用官方基础图像。官方图像会定期更新，比非官方图像更安全。
2。尽可能使用阿尔卑斯山图片的变体来保持你的图片轻巧。
3。如果使用 apt，在同一个指令中结合运行 apt-get 更新和 apt-get 安装。然后在该指令中链接多个包。用`\`字符在多行上按字母顺序列出包装。例如:

```
RUN apt-get update && apt-get install -y \
    package-one \
    package-two 
 && rm -rf /var/lib/apt/lists/*
```

这种方法减少了需要建造的层数，并且保持了整洁。
4。在 RUN 指令的末尾包含`&& rm -rf /var/lib/apt/lists/*`以清理 apt 缓存，使其不存储在层中。查看更多关于 [Docker Docks](https://docs.docker.com/develop/develop-images/dockerfile_best-practices/) 。感谢 Vijay Raghavan Aravamudhan 的建议。更新于 2019 年 2 月 4 日。
5。明智地使用缓存，将可能发生变化的指令放在 lower 文件的较低位置。
6。使用. dockerignore 文件将不想要的和不必要的文件从您的映像中删除。
7。看看[dive](https://github.com/wagoodman/dive)——一个非常酷的工具，用于检查你的 Docker 图像层，并帮助你修剪脂肪。
8。不要安装你不需要的软件包。咄！但很常见。

# 包装

现在你知道如何制作 Docker 图像，它可以快速构建，快速下载，并且不占用太多空间。就像健康饮食一样，知道是成功的一半。享受你的蔬菜！🥗

![](img/26b601feace374c2a2400365f6606894.png)

Healthy and yummy

在本系列的下一篇文章中，我将深入探讨基本的 Docker 命令。跟着 [me](https://medium.com/@jeffhale) 确保不要错过。

如果你觉得这篇文章有帮助，请分享到你最喜欢的社交媒体上，帮助其他人找到它。👍

我帮助人们了解云计算、数据科学和其他技术主题。如果你对这些感兴趣，可以看看我的其他文章。

[![](img/ba32af1aa267917812a85c401d1f7d29.png)](http://eepurl.com/gjfLAz)

码头快乐！🛥