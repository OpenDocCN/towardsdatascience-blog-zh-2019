# 用 AI 用 Cloudsight + Python 为图片写标题

> 原文：<https://towardsdatascience.com/use-ai-to-write-captions-for-images-with-cloudsight-python-a82115f16ab8?source=collection_archive---------27----------------------->

## Pythonic API 允许您自动为图像编写人类可读的标题

![](img/4acbdcc41d3b65b4c101e055d49f4ac2.png)

How would you caption this? Credit: Gado Images.

如今市场上有很多使用人工智能和机器学习来标记图像的解决方案。IBM Watson、Imagga、Cloudsight、Google Vision 和微软 Azure 的解决方案都表现良好，通过 Algorithmia 等服务，你可以轻松地启动和训练自己的图像标记网络。

然而，为图像编写人类可读的句子长度的标题是一项更困难的任务。您生成的句子不仅需要准确描述场景中的对象，还需要捕捉它们彼此之间的关系、上下文等。它需要知道什么才是真正重要的——没有人想要一个对桌子上的每一个物体或背景中的每一株植物都进行详细描述的图像。当人工智能知道得太多时，它可能比知道得太少更糟糕。

总部位于洛杉矶的初创公司 Cloudsight 正致力于解决使用人工智能和机器学习来自动为图像编写句子长度、人类可读的字幕的挑战。

他们为特定应用提供完全定制的人在回路系统，但他们也有一个通用模型，你可以很快开始测试，并返回一些令人印象深刻的结果。这是他们唯一的人工智能系统，但同样，你也可以将其插入人类团队，并获得更好的输出(以更高的价格和更多的训练时间)。

## 入门指南

要开始使用 Cloudsight API 进行测试，您不能像在 IBM 或 Google Vision 中那样注册一个帐户。Cloudsight 是一家新兴公司，他们喜欢与潜在的新客户进行合作，尤其是在早期测试阶段。

好消息是他们反应很快。通过他们网站上的联系方式[联系他们，他们通常会在几个小时后回来。你甚至可能会听到他们的首席执行官或团队的高级成员的消息。Cloudsight 通常可以提供 API 访问和一定数量的免费 API 调用来进行测试。我得到了 1000 个免费的 API 调用来开始。](https://cloudsight.ai/contact)

一旦你联系并设置好了，他们会给你一个 API 密匙。

## 准备 Python

方便的是，Cloudsight 有一个随时可用的 Python 库。

用 pip 安装。

```
pip install cloudsight
```

如果你还没有，我也喜欢通过 Pillow 安装 PIL，这是一个图像处理库。

```
pip install pillow
```

## 预处理

首先，选择要处理的图像。我将使用本文顶部的那张照片，它是由我公司的一位新西兰摄影师拍摄的。

设置基本导入。

```
from PIL import Imageimport cloudsight
```

我喜欢先通过 PIL 的缩略图功能将图像缩小到标准尺寸。这使得它更小，更容易上传。如果你不这样做，Cloudsight 会帮你做，但是发送更少的字节更容易，所以你也可以在你这边精简一下。

这是标准的东西——你接收图像，使用缩略图功能使其大小一致，然后再次保存。

```
im = Image.open('YOURIMAGE.jpg')im.thumbnail((600,600))im.save('cloudsight.jpg')
```

## 打电话

现在您已经有了一个经过适当处理的图像，您可以进行实际的 API 调用了。首先，使用 API 密钥进行身份验证。

```
auth = cloudsight.SimpleAuth('YOUR KEY')
```

接下来，打开您的图像文件，并发出请求:

```
with open('cloudsight.jpg', 'rb') as f:
    response = api.image_request(f, 'cloudsight.jpg',  {'image_request[locale]': 'en-US',})
```

接下来的一点有点出乎意料。因为 Cloudsight 有时会有人参与，所以您需要给他们几秒钟或更长的时间来完成他们的请求。这与 API 对 Watson 等标签服务的请求略有不同，后者往往会立即完成。

使用 Cloudsight，您可以调用 wait 函数，并定义最大等待时间。30 秒通常足够了。

```
status = api.wait(response['token'], timeout=30)
```

最后，打印出响应！

```
print status
```

## 回应

Cloudsight 将返回一个包含响应数据的对象。这是我得到的上面的图像。

```
{u'status': u'completed', u'name': u'man in gray crew neck t-shirt sitting on brown wooden chair', u'url': u'https://assets.cloudsight.ai/uploads/image_request/image/734/734487/734487841/cloudsight.jpg', u'similar_objects': [u'chair', u't-shirt', u'crew neck'], u'token': u'fgo4F5ufJsQUCLrtwGJkxQ', u'nsfw': False, u'ttl': 54.0, u'structured_output': {u'color': [u'gray', u'brown'], u'gender': [u'man'], u'material': [u'wooden']}, u'categories': [u'fashion-and-jewelry', u'furniture']}
```

这是非常酷的东西！正如你所看到的，系统给图片加了标题**“一个穿着灰色圆领 t 恤的男人坐在棕色的木椅上。”**说得好极了！

除了非结构化的句子，API 还给我关于图像的结构化数据，比如图像中的颜色(灰色、棕色)、人的性别(男人)、类别和呈现的材料。

在一次调用中结合结构化和非结构化数据非常有帮助——您可以进行标记或语义映射，同时还可以获得人类可读的句子来描述图像。

这是完整的节目单。

```
from PIL import Imageimport cloudsight im = Image.open('YOUR IMAGE')im.thumbnail((600,600))im.save('cloudsight.jpg')auth = cloudsight.SimpleAuth('YOUR API KEY')api = cloudsight.API(auth)with open('cloudsight.jpg', 'rb') as f:response = api.image_request(f, 'cloudsight.jpg', {'image_request[locale]': 'en-US',})status = api.wait(response['token'], timeout=30)print status
```

这是一个句子长度的图像字幕，通过人工智能，大约 10 行代码！

## 从这里去哪里

如果您想在测试后扩大您对 Cloudsight 的使用，该团队可以帮助您为生产做好准备。他们不报一般价格，这完全取决于您的预计数量、使用情形和所需的任何额外培训。

如果通用模型的输出还不错，但并不完美，您还可以与团队合作，根据您的特定用例对其进行定制。例如，我正与他们合作，将人类部分引入到处理历史图像的循环中。定价可能会高于纯人工智能解决方案，但它也将更具体地针对我的用例。这是与新兴公司合作的最大好处，因为他们能更好地满足你的特殊需求。

如果你对自动图像字幕感兴趣，可以看看 Cloudsight！同样，使用 Python，您可以在几分钟内开始为您的图像添加字幕——或者只是使用这一新功能——而且只需很少的代码！