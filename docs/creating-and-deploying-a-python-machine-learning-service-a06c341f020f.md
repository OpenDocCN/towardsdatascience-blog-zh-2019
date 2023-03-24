# 创建和部署 Python 机器学习服务

> 原文：<https://towardsdatascience.com/creating-and-deploying-a-python-machine-learning-service-a06c341f020f?source=collection_archive---------15----------------------->

## 用 scikit 构建一个仇恨言论检测系统-通过 Heroku 上的 Docker 学习和部署它。

![](img/8fdea00edd0f652382bc8b8b1eb28a2b.png)

Photo by [Jon Tyson](https://unsplash.com/@jontyson?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

# 介绍

假设你是留言板或评论区的版主。你不想阅读你的用户在网上写的所有东西，但你想在讨论变得不愉快或人们开始到处散布种族诽谤时得到提醒。所以，你决定为自己建立一个自动检测仇恨言论的系统。

通过机器学习进行文本分类是一种明显的技术选择。然而，将模型原型转化为工作服务被证明是一个普遍的挑战。为了帮助弥合这一差距，本四步教程展示了仇恨言论检测应用程序的示例性部署工作流程:

1.  用 **scikit-learn** 训练并保持一个预测模型
2.  用 **firefly** 创建一个 API 端点
3.  为这个服务创建一个 **Docker** 容器
4.  在 **Heroku** 上部署容器

该项目的代码可在[这里](https://github.com/dhaitz/python-sklearn-firefly-docker-heroku)获得。

# 1.创建预测模型

## 资料组

该方法基于戴维森、瓦姆斯利、梅西和韦伯的论文 [*自动仇恨言论检测和攻击性语言问题*](https://arxiv.org/abs/1703.04009) 。他们的结果是基于超过 20 000 条有标签的推文，这些推文可以在相应的 Github 页面上找到。

的。csv 文件作为数据帧加载:

```
import pandas as pd
import re**df** = pd.read_csv('labeled_data.csv', usecols=['class', 'tweet'])**df**['tweet'] = **df**['tweet'].apply(lambda tweet: re.sub('[^A-Za-z]+', ' ', tweet.lower()))
```

最后一行通过将所有文本转换为小写并删除非字母字符来清理 tweet 列。

结果:

class 属性可以假设三个类别值:`0`表示仇恨言论，`1`表示攻击性语言，`2`两者都不表示。

## 模特培训

在训练机器学习分类器之前，我们必须将我们的预测器，即推文文本，转换为数字表示。我们可以使用 scikit-learn 的 **TfidfVectorizer** 来完成这项任务，它将文本转换为适合机器学习的术语频率乘以逆文档频率(tf-idf)值的矩阵。此外，我们可以从处理中删除**停用词**(常用词如*、*、*为*、…)。

对于文本分类，支持向量机( **SVMs** )是一个可靠的选择。由于它们是二元分类器，我们将使用**一对其余**策略，其中对于每个类别，训练一个 SVM 来将该类别与所有其他类别分开。

通过使用 scikit-learn 的**管道**功能并定义相应的步骤，可以在一个命令中执行文本矢量化和 SVM 训练:

```
from sklearn.pipeline import make_pipeline
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.multiclass import OneVsRestClassifier
from sklearn.svm import SVC
from stop_words import get_stop_words**clf** = make_pipeline(
    TfidfVectorizer(stop_words=get_stop_words('en')),
    OneVsRestClassifier(SVC(kernel='linear', probability=True))
)**clf** = **clf**.fit(X=**df**['tweet'], y=**df**['class'])
```

现在，应该评估模型的性能，例如，使用交叉验证方法来计算分类度量。然而，由于本教程关注的是模型部署，我们将跳过这一步(**在实际项目中从不这样做**)。这同样适用于参数调整或自然语言处理的附加技术，在[原始论文](https://arxiv.org/abs/1703.04009)中有所描述。

## 测试模型

我们现在可以尝试一个测试文本，让模型预测概率:

```
**text** = "I hate you, please die!" **clf**.predict_proba([**text**.lower()])# Output:
array([0.64, 0.14, 0.22])
```

数组中的数字对应于三个类别的概率(仇恨言论、攻击性语言，都不是)。

## 模型持久性

使用 joblib 模块，我们可以将模型作为二进制对象保存到磁盘。这将允许我们在应用程序中加载和使用模型。

```
from sklearn import externals**model_filename** = 'hatespeech.joblib.z'
externals.joblib.dump(**clf**, **model_filename**)
```

# 2.创建 REST API

## 创建 API 端点

python 文件`app.py`加载模型并定义一个简单的模块级函数，该函数封装了对模型的 predict_proba 函数的调用:

```
from sklearn import externals**model_filename** = 'hatespeech.joblib.z'
**clf** = externals.joblib.load(model_filename)def predict(**text**):
    probas = **clf**.predict_proba([**text**.lower()])[0]
    return {'hate speech': probas[0],
           'offensive language': probas[1],
           'neither': probas[2]}
```

现在，我们使用 [firefly](https://github.com/rorodata/firefly) ，这是一个轻量级 python 模块，用于将*功能作为服务*。对于高级配置或在生产环境中的使用，Flask 或 Falcon 可能是更好的选择，因为它们已经在大型社区中建立了良好的声誉。对于快速原型，我们对 firefly 很满意。

我们将在命令行中使用 firefly 将 predict 函数绑定到本地主机上的端口 5000:

```
**$** firefly app.predict --bind 127.0.0.1:5000
```

## 本地测试 API

通过`curl`，我们可以向创建的端点发出 POST 请求，并获得一个预测:

```
**$** curl -d '{"text": "Please respect each other."}' \ http://127.0.0.1:5000/predict# Output:
{"hate speech": 0.04, "offensive language": 0.31, "neither": 0.65}
```

当然，在一个成熟的实际应用程序中，会有更多的附加功能(日志记录、输入和输出验证、异常处理等等)和工作步骤(文档、版本控制、测试、监控等等)，但是这里我们只是部署一个简单的原型。

# 3.创建 Docker 容器

为什么是 Docker？Docker 容器在一个隔离的环境中运行应用程序，包括所有的依赖项，并且可以作为映像提供，从而简化服务设置和扩展。

## 构建图像

我们必须在一个名为`Dockerfile`的文件中配置容器的内容和开始动作:

```
FROM python:3.6
RUN pip install scikit-learn==0.20.2  firefly-python==0.1.15
COPY app.py hatespeech.joblib.z ./CMD firefly app.predict --bind 0.0.0.0:5000
EXPOSE 5000
```

前三行是关于将`python:3.6`作为基础映像，另外安装 scikit-learn 和 firefly(与开发环境中的版本相同)并复制里面的 app 和模型文件。后两行告诉 Docker 启动容器时执行的命令以及应该暴露的端口 5000。

创建图像`hatespeechdetect`的构建过程通过以下方式开始:

```
**$** docker build . -t hatespeechdetect
```

## 运行容器

`run`命令启动一个容器，从一个图像派生。此外，我们通过`-p`选项将容器的端口 5000 绑定到主机的端口 3000:

```
**$** docker run -p 3000:5000 -d hatespeechdetect
```

## 使用预测服务

现在，我们可以发送一个请求并获得一个预测:

```
**$** curl -d '{"text": "You are fake news media! Crooked!"}' \ http://127.0.0.1:3000/predict# Output:
{"hate speech": 0.08, "offensive language": 0.76, "neither": 0.16}
```

在本例中，容器在本地运行。当然，实际的目的是让它在一个永久的位置运行，并且可能通过在一个企业集群中启动多个容器来扩展服务。

# 4.部署为 Heroku 应用程序

让其他人可以公开使用该应用的一种方式是使用平台即服务，如 **Heroku** ，它支持 Docker 并提供免费的基本会员资格。要使用它，我们必须注册一个帐户并安装 Heroku CLI。

Heroku 的应用程序容器公开了一个动态端口，这需要在我们的`Dockerfile`中进行编辑:我们必须将端口 5000 更改为环境变量`PORT`:

```
CMD firefly app.predict --bind 0.0.0.0:$PORT
```

在此变更之后，我们就可以开始部署了。在命令行上，我们登录 heroku(它会在浏览器中提示我们输入凭证)并创建一个名为`hate-speech-detector`的应用程序:

```
**$** heroku login**$** heroku create hate-speech-detector
```

然后我们登录到容器注册中心。`heroku container:push`将基于当前目录中的 Dockerfile 构建一个映像，并将其发送到 Heroku 容器注册表。之后，我们可以将图像发布到应用程序:

```
**$** heroku container:login**$** heroku container:push web --app hate-speech-detector**$** heroku container:release web --app hate-speech-detector
```

和以前一样，API 可以通过 curl 来处理。但是，这一次，服务不是在本地运行，而是面向全球！

```
**$** curl -d ‘{“text”: “You dumb idiot!”}’ https://hate-speech-detector.herokuapp.com/predict# Output:
{"hate speech": 0.26, "offensive language": 0.68, "neither": 0.06}
```

现在，只需点击几下鼠标或输入几个命令，就可以扩展应用程序。此外，该服务需要连接到留言板，需要设置触发阈值并实现警报。

希望这篇教程能帮助你部署你自己的机器学习模型和应用。有其他想法吗？请在评论中分享你的观点！