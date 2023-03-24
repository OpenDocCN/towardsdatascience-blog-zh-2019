# 使用卷积神经网络和 Flask 的垃圾邮件预测器

> 原文：<https://towardsdatascience.com/spam-predictor-using-convolutional-neural-networks-and-flask-bb94f5c54a35?source=collection_archive---------19----------------------->

## 了解如何将机器学习集成到 web 界面中。

由[大卫·洛伦兹](https://www.linkedin.com/in/d-a-lorenz/)、[克里斯蒂扣篮](https://www.linkedin.com/in/kristi-dunks/)和[塞丽娜·帕特尔](https://www.linkedin.com/in/serena-patel-95462949/)

希望为您的公司制作一个易于使用的内部预测工具，开发一个原型来向潜在的投资者推销机器学习产品，或者向朋友炫耀您的机器学习模型？

由于 Python 的 Flask，将机器学习模型与用户友好的 HTML 界面集成起来很简单。该框架可以应用于用户提供数据并从机器学习模型接收预测的任何示例。例如，X 射线技术人员可以上传患者的 X 射线，并通过图像识别立即接收自动诊断。Flask 是一个服务于应用程序后端模型的解决方案。

# 我们的用例示例

让我们想象一下，有成千上万的手机用户很难区分垃圾短信和朋友发来的短信。作为一名创新者，你想建立一个原型网站，用户可以输入文本，以接收自动垃圾邮件或火腿的区别。你来对地方了！您将学习(a)如何构建一个卷积神经网络来将文本分类为火腿或垃圾邮件，以及(b)如何将这种深度学习模型与使用 Flask 的前端应用程序集成。

是的，你很有可能相当擅长判断一条短信是否是垃圾短信，并且不需要机器学习。因为数据和问题的简单性，我们选择它作为我们的用例。您可以花更少的时间了解问题，花更多的时间了解工具！此外，您将能够自己轻松地运行整个示例，甚至不需要 GPU！

# 设计深度学习模型

卷积神经网络(CNN)在图像识别之外还有许多应用。例如，CNN 对于时间序列预测和自然语言处理(NLP)具有预测能力。CNN 的输入是一个矩阵。在图像识别中，每个图像的像素被编码为代表每个像素颜色强度的数值。

我们将重点讨论 CNN 的 NLP 应用，并训练一个单词 CNN。一个单词 CNN 的输入矩阵包括代表一个句子中单词的行和代表 n 个维度的单词嵌入的列。我们将回到单词嵌入，但是现在，考虑句子“哟，我们正在看一部关于网飞的电影”这句话 10 维的矩阵表示如下。

![](img/a2045156c5a5fa284f99c15084b9f58d.png)

Matrix representation of sentence with 10 dimensions (note: padding not pictured)

你可能想知道上面这些数字代表什么。让我们打开包装。单词嵌入是单词的广义矢量表示。具有相似上下文的单词共享相似的向量。如下图所示，“我”和“我们”有相似的向量表示，但“网飞”是不同的。

![](img/6847350f7ea40000b8c3901540cf9320.png)

Vector representations of “I”, “we”, and “netflix”

为什么以这种方式概括单词？用单词嵌入来概括单词有助于防止过度拟合。您可能想知道向量中的每个维度代表什么。技术上的解释是，它们是神经网络中隐藏层的权重，该神经网络在周围单词的上下文中预测给定单词。实际上，将这些维度视为单词的属性是很有帮助的。例如，一个维度可以代表一个单词有多快乐或悲伤。

有两种常见的方法来生成单词嵌入:(1)预训练的单词嵌入，如 Word2vec 或 GloVe，以及(2)从训练样本生成的单词嵌入。预先训练的单词嵌入是从大量文本语料库中产生的嵌入，并且概括得很好。在训练数据中生成的单词嵌入会导致特定于语料库的嵌入。

关于单词嵌入的更多信息，这篇文章很有帮助。

# 数据集

我们使用垃圾短信收集，可下载[这里](https://archive.ics.uci.edu/ml/datasets/SMS+Spam+Collection)。

该数据集由被分类为垃圾邮件(好的)或垃圾邮件(坏的)的文本消息组成。数据的摘录如下所示。

![](img/d364c10771ca0da3deda24aa4ff8ec01.png)

SMS Spam data excerpt

![](img/b29d57e048f3b1a4e69f2afa5fc0bb91.png)

That’s a lot of ham!

# 带 Keras 的 Word CNNs

Keras 只需几行代码就可以轻松创建一个单词 CNN。对于这个模型，我们使用 Keras“嵌入”层在我们的语料库中生成嵌入。注意，嵌入层的输出是一个矩阵，它是卷积层的必要输入。

在不到一个小时的时间里，通过处理这些数据，我们能够在测试集中达到 98%的准确率。CNN 的力量！

现在我们有了一个训练好的模型，我们保存预先训练好的权重、结构和记号赋予器。我们的 Flask 应用程序利用这些文件，这样我们就不需要在每次启动应用程序时都重新运行模型训练过程。这节省了时间和计算。我们的代码如下所示。

该模型的源代码和短信数据可以在 GitHub [这里](https://github.com/DaveLorenz/FlaskDeepLearningHamSpam)找到。

# 用烧瓶给模型上菜

我们现在创建一个界面，使用户能够与我们的 CNN 互动。为此，我们使用 Python Flask 创建了一个 REST(表述性状态转移)API(应用程序编程接口)。

RESTful API 在用户和托管预训练模型的服务器之间创建了一个链接。你可以把它想象成一个总是在“倾听”等待用户输入数据、生成预测并向用户提供响应的模型。我们使用 Python Flask 与用户在网页上输入和接收的内容进行交互。您可以将 Python Flask 视为预训练模型和 HTML 页面之间的桥梁。

首先，我们加载预训练的模型和分词器进行预处理。

接下来，我们创建一个应用预处理的助手函数。为什么我们需要加载记号赋予器？否则，附加到由用户输入的单词的标识符与在训练过程中分配的标识符不对齐。因此，单词 CNN 会将输入解释为与用户输入完全不同的句子。加载经过酸洗的标记化器确保了与模型训练的一致性。

接下来，我们编译我们的模型，并通过示例确认它的工作情况:

既然我们已经加载了预处理和预训练的模型，我们就可以让 Flask 与我们的 HTML 页面交互了。我们构建了两个 HTML 页面:(1) search_page.html 和(2)prediction.html。当用户第一次访问网页时，下面的 if 条件不成立，将加载 search_page.html。该页面包含一个名为“text _ entered”的 HTML id，它是表单的一部分。

当用户输入文本并单击表单的提交按钮时，“request.method”变成了“POST”因此，用户在 HTML 表单中输入的 text _ entered 变成了 textData，它被转换成一个名为 Features 的数组，其中包含每个单词的数字标识符。这个数组通过预先训练的单词 CNN 生成一个代表垃圾邮件概率的预测。

然后，render_template 将 prediction.html 页面发送给用户。在此页面上，来自单词 CNN 的预测被插入到 HTML 代码的“{{ prediction }}”中。

我们的最后一步是定义我们将在本地运行它的位置:

为了进行测试，在执行上述代码行之后，您在 web 浏览器上访问“0.0.0.0:5000”以加载 search_page.html。在执行此操作之前，请确保(HTML 文件保存在名为“templates”的文件夹中，并且(2)model . H5、model.json 和 tokenizer.pickle 保存在运行 Python 或 Jupyter 笔记本的同一目录中。我们现在可以在本地测试应用程序了。

1.  执行 Flask 应用程序中的所有代码行(等待查看上面的“*运行于…”消息)
2.  打开谷歌 Chrome 之类的浏览器，访问“0.0.0.0:5000”
3.  输入下面的例子，然后点击“获得垃圾邮件预测！”

![](img/10df3f39075064391f438d46b792cf7c.png)

HTML interface to enter in email text

4.单击“获取垃圾邮件预测！”，这将返回下面的页面。

![](img/b8f920faf4f2be305bbbae6630f0a29a.png)

HTML output of prediction

5.观察 Jupyter 笔记本中的输出，它打印了预测函数中的每一步

![](img/456f9da7a5f650b9b8660d4ce3ad04aa.png)

Jupyter notebook with application running locally and user entering example

成功！我们输入了一个 ham 示例，应用程序返回了 2%的垃圾邮件。我们的应用程序在本地运行。

# 部署到云

当您的本地应用程序就位并正常工作后，您现在可能希望将它启动到云中，以便任何地方的用户都可以与它进行交互。然而，在云中调试要比在本地调试困难得多。因此，我们建议在迁移到云之前在本地进行测试。

启动到云很容易！

如果你使用的是 Jupyter 笔记本，你需要创建一个 main.py 文件。

您需要将以下内容上传到 Google Cloud:

1.  main.py
2.  模板文件夹
3.  一个 yaml 文件
4.  带权重的 h5 文件
5.  带有模型框架的 json 文件
6.  腌制的记号赋予器
7.  requirement.txt 文件

请注意，您需要在 requirements.txt 中指定 gunicorn 版本，以便 Google Cloud 可以连接到 Python web 服务器来下载 requirements.txt 文件中指定的适当库。requirements.txt 中不需要 pickle(Python 标配)，如果包含它，Google Cloud 会返回错误。

接下来，我们在云中键入以下命令来运行它:

1.  CD[文件夹名]
2.  gcloud 应用部署

有关启动 Google Cloud 的更多详细信息，请参见 GitHub 上的分步指南[此处](https://github.com/DaveLorenz/FlaskDeepLearningHamSpam/blob/master/setting_up_google_cloud.pdf)。此外，这个[视频](https://www.youtube.com/watch?v=RbejfDTHhhg)为谷歌云部署步骤提供了很好的参考。

# 垃圾邮件预测器诞生了

现在，您已经创建了一个可用的应用程序，它允许您确定一条消息是垃圾邮件还是 ham。

Flask 应用程序的源代码可以在 GitHub [这里](https://github.com/DaveLorenz/FlaskDeepLearningHamSpam)找到。

有兴趣看另一个 Flask + Word CNN 用例吗？点击观看我们关于药物不良反应的视频[。](https://www.youtube.com/watch?v=76G3Wf91JR0&t=2s)

请注意，Flask 非常适合用户数量有限的原型和应用程序(例如，一个公司的小用户群的内部工具)。Flask 不适用于为成千上万用户提供服务的生产级模型。

# 烧瓶和图像识别

你可能会想:这很棒，但我想让 Flask 与图像识别的深度学习进行交互。你很幸运！虽然我们在这个例子中关注于单词 CNN，但是对于图像，方法是类似的，并且我们有一个图像的例子。下面链接的 GitHub 库通过 CNN 交互用户上传的图像，以进行图像识别。

卷积神经网络使用 UCSD 的标记图像来诊断儿童肺炎可以在这里找到。

Python Flask 接受 X 射线技师上传的图像并提供诊断(健康或肺炎？)可以在这里找到[。](https://github.com/DaveLorenz/FlaskXRAYpneumonia)