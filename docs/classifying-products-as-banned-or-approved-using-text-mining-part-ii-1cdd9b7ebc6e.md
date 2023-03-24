# 使用文本挖掘将产品分类为禁止或批准-第二部分

> 原文：<https://towardsdatascience.com/classifying-products-as-banned-or-approved-using-text-mining-part-ii-1cdd9b7ebc6e?source=collection_archive---------20----------------------->

## 在这一部分，我们将解释如何优化 [Part I](/classifying-products-as-banned-or-approved-using-text-mining-5b48d2eb1544) 中已有的机器学习模型，以及使用 Flask 部署这个 ML 模型。

![](img/a29562fda23a3d14650fe03c934aff30.png)

Connecting the dots -moving from M to L in Machine Learning

在本系列的前一篇文章中，我们已经讨论了这个业务问题，展示了如何使用 fastText 训练模型，以及如何根据诸如(*产品名称、产品描述和规格*)等信息对禁止或批准的产品进行分类。**要在**[**Indiamart**](https://dir.indiamart.com/)**的背景下理解一个产品，请参考这篇** [**文章**](/classifying-products-as-banned-or-approved-using-text-mining-5b48d2eb1544) **。**

# 从 85%提高到 98%,目标更高…..

当我们将结果与人类审计员进行匹配时，上一篇文章中提到的模型的准确性是 85%。

为了提高该模型的准确性，我们在训练该模型时加入了一些附加数据，并进行超参数调整。在机器学习中，更精确的训练数据给出更好的结果*(我们可以说对于机器来说，垃圾进就是垃圾出，反之亦然)。*

# **超参数调谐:**

在 fastText 中，有一些调优参数*(像学习率(LR)、wordN-grams、character _ n grams(Minn)、迭代次数(epoch)等。)*可以在训练模型的同时对其进行优化以提高分类器的性能。这些调整参数因情况而异。

在我们的案例中，我们在训练模型时对这些超参数进行了多种排列和组合，然后从中选择了最佳的。使用下面的命令获得具有多种调整参数组合的模型。

```
while read p; do while read q; do while read r; do while read s; do fasttext  supervised -input ~/Desktop/Banned_Model/train.txt -output ~/Desktop/Banned_Model/hypertuned/$p.$q.$r.$s -lr $p -minn $q -epoch $r -wordNgrams $s -thread 4 -loss hs -lrUpdateRate 100; done ; done
```

在这个阶段，我们已经创建了超过 2k 个独特的模型。下一步是从中选出最好的一个。根据精确度、召回率和准确度选择最佳模型的测试命令如下:

```
for b in ~/Desktop/Banned_Model/hypertuned/*.bin ;do ( echo Test results with $b && fasttext test $b /home/Desktop/test\val.txt ); done >> /home/Desktop/banned_hyper_parameters_test.txt
```

使用这种方法，我们能够根据最合适的超参数组合选择最佳模型。这些超参数可能因不同的用例而异。

**模型精度:**

在这个阶段，当我们将结果与人类审计员进行匹配时，该模型能够以 98%以上的准确率进行正确预测，并且我们已经完成了我们的机器学习模型，并将其保存为。“bin”文件。下一个目标是将这个机器学习模型投入生产。

# **模型部署:让事情活起来**

这里的简单方法是调用 REST API 并从模型中获取预测。正如我们所知，有许多 web 开发框架是用 javascript、ASP.net、PHP 等编写的。但是在我们的例子中，我们已经使用 python 创建了我们的机器学习模型，并且我们正在 python 本身中寻找一个基于 web 的接口。在这里[烧瓶](http://flask.pocoo.org/)进入画面。

![](img/2bdab8d0cacfa604dd99b2131c9bde05.png)

ML Model deployment using Flask

> **什么是烧瓶？**

Flask 是一个基于 python 的**微框架**，可以用来开发 web 应用、网站，部署 ML 模型相当容易。**微框架**烧瓶基于 Pocoo 项目 Werkzeug 和 Jinja2。Werkzeug 是一个用于 Web 服务器网关接口(WSGI)应用程序的工具包，并获得了 [BSD 许可](http://flask.pocoo.org/docs/1.0/license/)。在进行部署之前，我们需要下载 flask 和其他一些库。

```
**pip install flask**
```

让我们创建一个文件夹 **Flask_Deploy** 并将机器学习模型复制到其中。

```
**mkdir Flask_Deploy**
```

下一步是创建两个 python 脚本，一个用于处理数据预处理/清理，另一个用于获取基于文本输入预测结果的请求。

用 python 导入 flask。

> **app =烧瓶。Flask(__name__) :** 创建 Flask 的实例
> 
> **@ app . route("/predict "):***用于指定 flask app 在 web 上的路由。例如:http://0.0.0.0:5000/predict？msg =输入字符串。* ***【预测与消息】这里的关键是*** *。*

要访问 URL(？msg =输入字符串)我们使用以下属性:

```
**Input = flask.request.args.get("msg")**
```

> **flask . jasonify(data):***用于返回 JSON 格式的 python 字典。*
> 
> **app . run(host = 0 . 0 . 0 . 0)***:用于在主机地址上运行 flask app。在本例中，我们在本地主机上运行它。*

创建 web 应用程序后，我们将获得一个指向 flask 端点的 URL。感谢 Shantanu Aggarwal @阿洛克·库马尔在这么短的时间内帮助编写代码。为了处理数据预处理，我们创建了一个脚本 preprocess.py，并保存在保存 ML 模型文件的同一目录下(Flask_Deploy ),以便我们可以在需要时轻松调用它。

script to get requests and predict results.

在我们的例子中，我们得到了许多不必要的数据，如特殊字符、超链接、一些停用词、重复词等。为了处理这些问题，我们根据我们的用例创建了下面的脚本，并将其保存为 preprocess.py

**处理数据清理和预处理的脚本:**

preprocess.py script

# **最后一轮:**

```
**cd Flask_Deploy
python3 flask_banned_model.py**
```

部署模型后，我们能够检查预测、模型准确性、延迟以及模型的负载处理时间。我们还可以记录结果、数据输入、延迟等。按照我们的要求做成表格。

# **结论:**

任何机器学习模型的最终目标都是不需要太多麻烦就能投入使用。在投入了大量团队努力和辛勤工作后，最终部署了上述简单但有效的模型。此外，我们很高兴地提到，我们能够为上述模型实现大约 4 毫秒的显著延迟。我们正在对该模型进行全面测试，以发现任何问题，本系列的下一篇文章将分享该模型的性能，以及该模型在时间和金钱方面为我们带来的优化。

# 当团队工作创造奇迹的时候！！

这个模型是整个团队不断努力的成果。感谢@ [维克拉姆·瓦什尼](https://medium.com/u/47a096395dd5?source=post_page-----1cdd9b7ebc6e--------------------------------)、[苏尼尔·帕罗莉亚](https://medium.com/u/f3022f1424ae?source=post_page-----1cdd9b7ebc6e--------------------------------)发起项目、[普拉奇·贾因](https://medium.com/u/a9b6c476fde?source=post_page-----1cdd9b7ebc6e--------------------------------)进行模型训练和超参数调优、[帕拉德普·乔普拉](https://medium.com/u/7012b8be20c2?source=post_page-----1cdd9b7ebc6e--------------------------------)、@里蒂卡·阿加瓦尔确保对模型进行全面测试、@梅达·塔亚吉@安基塔·萨拉斯瓦特提供所有产品方面的知识@阿洛克·库马尔、@普内特·阿加瓦尔、[尚塔努·阿加瓦尔](https://medium.com/u/ba9d2ef2623c?source=post_page-----1cdd9b7ebc6e--------------------------------)、@希普拉·古普塔、@阿布