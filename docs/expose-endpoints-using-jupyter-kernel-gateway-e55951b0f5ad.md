# 使用 Jupyter 内核网关公开端点

> 原文：<https://towardsdatascience.com/expose-endpoints-using-jupyter-kernel-gateway-e55951b0f5ad?source=collection_archive---------14----------------------->

![](img/631a8084b2016e09dcb81870dc9a7e20.png)

Photo by [Nikola Knezevic](https://unsplash.com/@nknezevic?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

在浏览中型文章时，我偶然发现了 Jupyter 内核网关，它对我来说是全新的。所以，我也探索了一样。我发现了它的文档，并决定实施一个项目来更好地理解它。

> Jupyter 内核网关是一个 web 服务器，提供对 Jupyter 内核的无头访问。— [Jupyter 内核网关回购](https://github.com/jupyter/kernel_gateway)

基本上，该服务允许我们与任何给定的 Jupyter 笔记本的 Jupyter 细胞进行交互，然后相应地使用这些信息。支持`GET`、`POST`、`PUT`和`DELETE`。让我们一起开始这段旅程吧。你可以在这里查看我的知识库[。](https://github.com/kb22/House-Price-Predictions)

# 关于项目

![](img/012133eddae00117ef19803041f3c59a.png)

Photo by [Breno Assis](https://unsplash.com/@brenoassis?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

在这个项目中，我们将使用来自`sklearn`的加州住房数据集，在其上训练一个`Random Forest Regressor`模型，然后预测房价。我们将设置一个`GET`调用来获取数据集的统计数据，并设置一个`POST`端点来获取具有给定特征集的房屋的预测价格。

由于这篇文章是关于 Kernel Gateway 的，而不是关于加州住房的机器学习项目本身，我将跳过它的细节，但将解释笔记本的相关单元。笔记本位于[这里](https://github.com/kb22/House-Price-Predictions/blob/master/House%20Price%20Prediction.ipynb)。

# 基础

![](img/5b0fe4a9548565c8f1bcf664fb9538eb.png)

Photo by [Mimi Thian](https://unsplash.com/@mimithian?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

我们先来看一下 Jupyter 内核网关的基础知识。

## 定义端点

我们需要用注释开始我们想要创建为端点的单元格。如果我们想要创建一个路径为`/housing_stats`的`GET`端点，它被定义为:

```
# GET /housing_stats
```

然后，我们定义 Python 代码并处理数据。在单元中完成所有工作后，我们定义端点的响应。它被定义为一个带有 JSON 值的`print()`命令。因此，如果我们必须在参数`total_houses`中返回数据集中房屋的总数，我们将其定义为:

```
print(json.dumps({
  'total_houses': 20640
}))
```

就这么简单。如果需要，我们可以将功能扩展到更复杂的解决方案。因此，每个端点 Jupyter 单元将类似于下面的 Github 要点:

## 启动服务器

要启动服务器，有一个非常简单的命令。我们将使用名为`House Price Prediction`的 Jupyter 笔记本。同样的将在`0.0.0.0:9090`上市。代码如下:

```
jupyter kernelgateway --api='kernel_gateway.notebook_http' --seed_uri='House Price Prediction.ipynb' --port 9090
```

只要您计划在不同的项目上工作，只需更改端口号和/或 Jupyter 笔记本名称。

# 获取和发布端点

![](img/c005e3d06d6e4bf4adaf7045d5b58d11.png)

Photo by [Mathyas Kurmann](https://unsplash.com/@mathyaskurmann?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

让我们在笔记本中设置我们的端点。

## 获取端点

第一行包含单词`GET`来定义它是一个 GET 端点，然后是路径`/housing_stats`。我们提取数据集中的所有房屋，所有房屋的最大值和所有房屋的最小值。此外，根据我在笔记本中的分析，我发现最重要的特征是区块的*中值收入。因此，我将所有这些值转储到一个`JSON`中，并将其放在打印命令中。print 命令定义了该端点的响应。*

## 终点后

现在，我想使用我训练过的机器学习模型来预测任何具有给定功能集的新房子的价格。因此，我使用这个端点来发布我的功能，并获得作为预测价格的响应。

我通过将单元格定义为路径为`/get_prices`的`POST`端点来开始该单元格。请求数据包含在对象`REQUEST`内，键`body`内。因此，我首先加载请求，然后从`body`标签中读取所有值，并将其转换成一个 Numpy 数组。然而，形状是不正确的，因此，我使用 Numpy 的函数`reshape`来纠正它。然后我预测价格。它返回一个预测价格的数组，该数组只有一个值，所以我将该值读入变量`predicted_price`。我将其重新格式化为两位小数，并乘以`100000`，因为值的单位是 100，000。最后，我通过将该值附加到一个字符串并放入 print 命令中来返回值。

# 提出请求

![](img/06aff05b2083024d39460beb66b2d266.png)

Photo by [Dan Gold](https://unsplash.com/@danielcgold?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

让我们按照步骤与我们的端点进行交互:

1.  启动你的 Jupyter 笔记本。我把我的笔记本命名为`House Price Prediction`，所以我开始也一样。
2.  启动服务器。使用我在上面定义的命令，您的服务器将在`http://0.0.0.0:9090`开始运行。
3.  最后，决定从哪里调用端点。您可以使用名为`Postman`的工具，或者创建一个网页来进行这些呼叫，或者您可以简单地创建另一个笔记本来呼叫这些端点。

这里，我们将创建一个笔记本`Requests`并使用 Python 中的`requests`包来调用这些端点并获得结果。

## 建立

让我们创建一个新的 Jupyter 笔记本`Requests`。我们将把`requests`包导入到笔记本中，它用于调用 Python 中的端点。然后，我们将变量`URL`中的基本 url 指定为`[http://0.0.0.0:9090](http://0.0.0.0:9090.)`。

## 发出 GET 请求

我们使用`request.get()`发出 get 请求，并指定完整的 URL，即`[http://0.0.0.0:9090](http://0.0.0.0:9090.)/housing_stats`，并将其保存在变量`stats`中。然后，我们从该变量加载`JSON`。对于每个键值对，我们打印相同的内容。

`stats`有应答对象。为了获得编码的内容，我们使用`content`后跟`decode('UTF-8)`。然后迭代结果。

我已经在上面的 Github Gist 中添加了结果作为注释。端点用所有房屋、它们的最高和最低价格以及预测任何房屋价格的最重要因素来响应。

## 提出发布请求

这里，我们将使用`requests.post()`来发出 POST 请求。我们首先指定完整的 URL，即`[http://0.0.0.0:9090](http://0.0.0.0:9090.)/get_price`。我们以 JSON 的形式发送特性。您可以随意更改这些值，并查看对预测价格的影响。然后，我们从端点加载结果，并打印我们得到的结果。

`expected_price`有响应对象。为了获得编码的内容，我们使用`content`后跟`decode('UTF-8)`。然后读取结果，其`result`字段具有我们的实际响应。

从上面的评论回复中，您可以看到对于给定的一组特性，房子的预测价格是 210，424 美元。

# 结论

在本文中，我们讨论了 Jupyter 内核网关，它允许我们将 Jupyter 笔记本单元转换为 REST 端点，我们可以调用这些端点并从中获得响应。然后，我们通过一个示例项目探索了相同的用法。要了解更多信息，您应该查看 Jupyter 内核网关的文档。

请随时分享你的想法，想法和建议。我们随时欢迎您的反馈。