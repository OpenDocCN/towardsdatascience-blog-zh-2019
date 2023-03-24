# 使用 TensorFlow 服务部署用于文本分类的 Keras 模型(第 1 部分，共 2 部分)

> 原文：<https://towardsdatascience.com/deploying-kaggle-solution-with-tensorflow-serving-part-1-of-2-803391c9648?source=collection_archive---------10----------------------->

![](img/9f8cd44a2837dd6eaae7de6d0804235d.png)

***注:*** *这是在 TensorFlow 1.0 中完成的*

**更新:**抱歉，不会有第二部了。这里使用的许多函数在 2.X 中不再受支持。相反，我写了[这篇文章，介绍如何在 TensorFlow 2.0](/deploying-a-text-classifier-with-tensorflow-serving-docker-in-2-0-cba6851e46ed?source=post_stats_page-------------------------------------) 中提供类似的分类器。

下面是附带的 [Github 回购](https://github.com/happilyeverafter95/toxic-comment-classifer)。

大约两年前，我使用 Keras 库为 [Kaggle 的有毒评论分类挑战构建了一个解决方案。](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/)该解决方案集成了多个深度学习分类器，实现了 98.6%的平均 ROC。

像我提交的大多数 Kaggle 文件一样，这份文件是 Jupyter 笔记本中的一堆杂乱的代码，除了生成一个非常随意的 csv 文件之外，几乎没有其他用途。为了使我的提交更有用，我选择了 ensemble 解决方案中使用的一个模型，对其进行了清理，并使用 TensorFlow 为模型推理公开了一个 HTTP 端点。

# 比赛背景

> 在这场比赛中，你面临的挑战是建立一个多头模型，能够比 Perspective 的[当前模型](https://github.com/conversationai/unintended-ml-bias-analysis)更好地检测不同类型的毒性，如威胁、淫秽、侮辱和基于身份的仇恨。你将使用维基百科谈话页面编辑的评论数据集。对当前模式的改进将有望帮助在线讨论变得更有成效和尊重。

该数据集包含大量被评估为有毒行为的维基百科评论。每个注释被分配一个二进制指示符，指示它们是否:

*   有毒的
*   剧毒
*   猥亵的
*   威胁
*   侮辱
*   身份仇恨

# 问题陈述

为简单起见，我们将只关注基本的有毒类。给定一条文本评论，我们的分类器能确定这条评论是否有毒吗？

# 模型部署

模型部署是将机器学习模型与生产环境相集成的过程，通常是为了使推理可用于其他业务系统。数据被发送到模型服务器，在那里进行预处理并用于生成推理结果。推断结果随后被返回给消费应用程序。

# 真实生活应用

想象一下，我们正在运行一个聊天平台，它促进了文本消息的交换。根据设计，我们的系统会过滤掉所有有害信息。

![](img/47539399656e5c1f34e4a5c668b66f49.png)

我们系统中的数据流可能如下所示:

*   用户写了一条消息
*   我们的应用程序服务接收消息。应用程序服务通过 POST 请求将消息(以及任何相关的元数据)发送到我们的模型服务器

```
message_payload = {
    'message': 'got plans today?',
    'region': 'CA'
}
```

*   模型服务器使用消息和元数据作为预测器返回推理输出

```
inference_payload = {
    'toxic': False,
    'probability': 0.93,
    'version': 'toxicity-detector-3.0'
}
```

*   app 系统接收推理有效载荷，并决定是否应该发送消息。app 系统还可以将推断结果写入数据库。

# 数据获取和管理

数据采集是每个建模过程中最重要的步骤之一，也是最具挑战性的步骤。[冷启动问题](https://www.kdnuggets.com/2019/01/data-scientist-dilemma-cold-start-machine-learning.html)是每个数据科学家都会遇到的头疼问题。这是 Kaggle 竞赛不总是代表真实生活数据科学项目的最大原因之一。

在生产环境中工作还会给数据获取和数据管理带来额外的注意事项。为了让我们的模型可以通过本地机器访问，我们需要构建管道来检索我们的数据，而不是在机器之间传递平面文件。

# Kaggle API

记住最后一点，我们转向 [Kaggle API](https://www.kaggle.com/docs/api) 。Kaggle API 允许以编程方式/通过命令行界面下载 Kaggle 数据集。

1.  如果您还没有 Kaggle 帐户，请创建一个
2.  在`kaggle.com/USERNAME/accounts.`生成新令牌这将提示您下载一个`kaggle.json`文件，其中包含访问 API 的凭证
3.  确保您接受[竞赛规则](https://www.kaggle.com/c/jigsaw-toxic-comment-classification-challenge/rules)(这是下载数据的要求)

4.将凭证从 json 文件导出为环境变量:

`export KAGGLE_USERNAME = [kaggle username]`

`export KAGGLE_KEY = [generated key]`

# 数据管道

导出 Kaggle 凭证后，我们可以构建一个简单的数据管道来从 Kaggle API 获取数据，并执行任何功能工程/数据清理步骤。

数据管道包含在类结构中。训练数据和测试数据将作为类变量存储。入口点是`preprocess`方法，它用预处理的训练和测试返回一个实例化的`DataPipeline`类。

预处理步骤序列包含在`preprocessing_steps`变量中。在`preprocess()`调用期间，每个步骤都按照它们出现的顺序应用于两个数据集。

这个模型使用的预处理步骤非常简单:缺失的注释用`UNKNOWN`文本进行估算，所有注释都转换成小写。

# 建模过程

文本分类器是使用 Keras 库构建的。

## 对文本进行标记

标记化是提取每个唯一标记(这里，我们基于空格分隔来确定标记)并将其映射到唯一数字/向量的过程。

这个模型使用的标记化非常简单。每个令牌被任意分配给一个整数。虽然保留了令牌的序列，但是没有映射提供关于令牌的显式信息。使用预先训练的 GloVe/word2vec 嵌入可能会给我们带来性能优势，但它会增加这些嵌入存储位置的复杂性。

在本例中，记号赋予器也被限制为前 10，000 个最常用的记号(这是由`max_features`参数设置的)。

考虑一个玩具例子:

```
tokenizer.word_index = {
    'the': 1,
    'brown': 2,
    'fox': 3,
    'jumped': 4,
    'over': 5,
    'lazy': 6,
    'dog': 7,
    'quick': 8}
```

短语*快速的棕色狐狸跳过懒惰的狗*将被映射到`[1, 8, 2, 3, 4, 5, 1, 6, 7]`

## 填充标记化的文本

可变长度序列必须转换成相同的长度。我们通常选择`0`来填充每个标记化的向量，使得每个输入都是相同的长度。

在我们的模型中，我们设置了`max_len = 100`，它将每个观察限制为 100 个令牌。少于 100 个标记的观察值将被填充，多于 100 个标记的观察值将被截断。

默认是从前面(注释的开头)截断/填充。

如果`max_len = 10.`，我们之前的标记化短语将被填充为`[0, 1, 8, 2, 3, 4, 5, 1, 6, 7]`

## 用于分类的递归神经网络

让我们一次看完每一层。

**嵌入层**

```
model.add(Embedding(self.max_features, self.embed_size))
```

嵌入层只能用作第一层。标记化的文本被传递到这一层，并作为密集向量输出。我们将输入的形状定义为(词汇特征的数量，每个嵌入的大小)。除非另有说明，否则初始嵌入权重作为训练过程的一部分被随机化和细化。

**轮回层**

*注意:我减少了原始解决方案中的单元数量，以便在没有 GPU 的情况下更容易训练这个模型*

```
model.add(Bidirectional(LSTM(100, return_sequences=True)))
```

来自嵌入层的密集向量被传递到递归层。该循环图层使用 LSTM 单位。LSTM 单元使用三个门来调节来自新观测的信息:*输入门*调节传递到单元中的信息的范围，*存储门*调节保留多少信息，*输出门*调节从单元输出的信息。

顺序模型在每个时间步消耗一个令牌。第一个令牌(或本例中的单词)在第一个时间步长消耗，第二个令牌在第二个时间步长消耗，依此类推。

如果我们设置`return_sequences=True`，那么将返回每个时间步长的输出。当此参数设置为 false 时，将仅返回最后一个时间步长的输出(在整个观察被处理后)。

**全局最大池层**

```
model.add(GlobalMaxPooling1D())
```

池是一种降采样技术，通过只保留最重要的信息来降低维数。全局最大池常用于自然语言处理的神经网络中。

在这里，全局最大池层吸收每个时间步长的输出，并选择每个步长的最大值来组合维数减少的输出。

**密集层(输出)**

```
model.add(Dense(2, activation='softmax'))
```

密集层使用 softmax 激活函数来输出类别概率。

# 在本地部署模型

TensorFlow 服务是部署 TensorFlow/Keras 模型的最简单方法之一。服务器使用`SavedModel` API 来公开模型预测的推理端点。要了解更多关于 TensorFlow 服务的信息，我建议阅读以下资源:

*   [https://medium . com/tensor flow/serving-ml-quickly-with-tensor flow-serving-and-docker-7df 7094 aa 008](https://medium.com/tensorflow/serving-ml-quickly-with-tensorflow-serving-and-docker-7df7094aa008)
*   [https://github.com/tensorflow/serving](https://github.com/tensorflow/serving)

## SavedModel API

`[SavedModel](https://github.com/tensorflow/tensorflow/blob/master/tensorflow/python/saved_model/README.md)` API 将整个 TensorFlow 会话导出为包含在单个目录中的语言不可知格式。

该目录具有以下结构:

```
assets/
assets.extra/
variables/
    variables.data-?????-of-?????
    variables.index
saved_model.pb
```

*   **saved_model.pb** 包含了`MetaGraphDef`类，定义了数据流结构
*   **资产**子目录包含所有辅助模型文件
*   **assets.extra** 子目录包含其他库生成的资产
*   **变量**子目录包含用于恢复推理模型的权重

## 构建张量流图

完整的代码可以在[这里](https://github.com/happilyeverafter95/toxic-comment-classifer/blob/master/profanity_detector/model.py#L52)找到。

张量流图包含训练模型所需的所有计算。这包括用于预处理输入的任何步骤。通过在图形定义中定义标记化和填充步骤，客户机将能够向模型输入原始文本。

虽然将标记化步骤合并到图中非常方便，但是在其他预处理步骤中这样做有几个缺点:

*   由于选项非常有限，并不是所有的预处理步骤都是可行的(我们实现的那些步骤经常会变得杂乱无章)
*   预处理变得难以并行化
*   代码变得非常混乱

## 部署我们的 Keras 模型

在使用 Keras 库训练模型之后，我们可以提取模型权重并将它们加载到 TensorFlow 会话中定义的非常相似的模型中。可以使用`get_weights()`方法提取重量。

我们需要创建一个查找表来标记原始文本。当令牌不存在于我们的词汇表中时，我们将它映射到 0。

然后，我们为客户端输入定义一个占位符。输入被标记化(由单个空格分隔的标记)、整形并评估填充/截断。

用`InputLayer.`重新定义模型

我们定义了代表输入和模型输出的`x_info, y_info`。

```
x_info = tf.saved_model.utils.build_tensor_info(x_input)        y_info = tf.saved_model.utils.build_tensor_info(serving_model.output)
```

可选地，我们可以将元数据作为模型有效负载的一部分。在这里，我们将模型版本包装成一个张量。

```
model_version_tensor = tf.saved_model.utils.build_tensor_info(tf.constant(version))
```

我们用它来创建预测签名:

## 要在本地设置 TensorFlow 服务器:

1.  安装[对接器](https://docs.docker.com/install/)
2.  使用`docker pull tensorflow/serving:latest`获取 TensorFlow 服务 docker 映像的最新版本
3.  导出包含`SavedModel`文件的目录。这应该是`/model`目录:`ModelPath="$(pwd)/model"`
4.  通过公开 REST API 端点的端口 8501 来启动服务器

```
docker run -t --rm -p 8501:8501 \
    -v "$ModelPath/:/models/toxic_comment_classifier" \
    -e MODEL_NAME=toxic_comment_classifier \
    tensorflow/serving
```

会出现一堆日志。每个日志中的第一个字符将指示进程的状态。

*   **E =错误**
*   **W =警告**
*   **I =信息**

```
docker run -t --rm -p 8501:8501    -v "$ModelPath/:/models/toxic_comment_classifier"    -e MODEL_NAME=toxic_comment_classifier    tensorflow/servingI tensorflow_serving/model_servers/server.cc:82] Building single TensorFlow model file config:  model_name: toxic_comment_classifier model_base_path: /models/toxic_comment_classifierI tensorflow_serving/model_servers/server_core.cc:461] Adding/updating models.
```

如果每个日志都以 **I、**开头，那么恭喜您——该模型已经成功提供了！

## **版本控制**

TensorFlow 服务基于子目录名称管理模型版本。子目录名称必须是整数值。默认情况下，它将总是获取最新版本。

## 有用的 Docker 命令

TensorFlow Serving 拥有自己的 Docker 映像，用于打包模型及其依赖项。这里有两个有用的 Docker 命令，在使用服务器时可能会派上用场:

*   `docker ps` -这显示哪些 Docker 容器当前正在运行；这对于获取容器 id 以便进一步操作非常有用
*   `docker kill [container id]` -如果您构建了错误的模型，您可以终止当前容器来释放端口并重启服务器

## 发布请求

既然我们的模型服务器已经在本地机器上启动并运行，我们就可以发送一个示例 POST 请求了。发布请求可以通过 [curl](https://curl.haxx.se/docs/manpage.html) 发送，这是一个用于在服务器之间传输数据的简单工具，也可以通过 Python 中的`request`库发送。

**样本卷曲命令:**

```
curl -d '{"signature_name": "predict","inputs":{"input": "raw text goes here"}}' \
  -X POST [http://localhost:8501/v1/models/toxic_comment_classifier:predict](http://localhost:8501/v1/models/division_inference:predict)
```

输出将如下所示。预测输出是一个概率列表。第一个索引将始终对应于第一类，第二个索引对应于第二类，依此类推。在这种情况下，第一个指数 0.996 表示评论为 0 级/无毒的概率。

```
{
    "outputs": {
        "prediction": [
            [
                0.996692061,
                0.00330786966
            ]
        ],
        "model_version": "20191005180917"
    }
```

# 后续步骤(第 2 部分的路线图)

## 预处理输入

我们当前实现的一个最大缺陷是，客户端输入的处理方式不同于训练数据。没有特征工程步骤应用于客户端输入。

我们可以使用辅助服务器来应用这些处理步骤。辅助服务器将接收输入，对其进行处理，调用张量流服务器进行预测，并返回处理后的张量流输出。

## TensorFlow 服务配置

我们可以选择包含一个`yml`文件来配置我们的服务器。这个配置文件可以用来指定模型名称、模型路径、版本策略(我们是想使用特定的版本还是总是最新的版本？)，记录配置以及服务器检查新型号版本的频率。这一步在每个生产环境中都是绝对重要的。

## 模型改进

第 1 部分为部署构建了一个快速模型，但没有关注性能改进。在第 2 部分中，我们将探索早期停止、预训练向量的有效性以及模型改进的其他潜在领域。

## 张量板

TensorBoard 是一套用于可视化和理解机器学习实验的 web 应用程序。它非常容易设置，但是解释起来有点棘手。

*敬请关注第二部分*

# 感谢您的阅读！

如果你喜欢这篇文章，可以看看我关于数据科学、数学和编程的其他文章。[通过 Medium](https://medium.com/@mandygu) 关注我的最新动态。😃

作为一个业余爱好项目，我还在 www.dscrashcourse.com[建立了一套全面的**免费**数据科学课程和练习题。](http://www.dscrashcourse.com/)

如果你想支持我的写作，下次你报名参加 Coursera 课程时，可以考虑使用我的[会员链接](https://click.linksynergy.com/fs-bin/click?id=J2RDo*Rlzkk&offerid=759505.198&type=3&subid=0)。完全公开—我从每一次注册中获得佣金，但不会对您产生额外费用。

再次感谢您的阅读！📕