# 用 Flask 构建 HTTP 数据管道

> 原文：<https://towardsdatascience.com/constructing-http-data-pipelines-with-flask-27fba04fbeed?source=collection_archive---------16----------------------->

![](img/cc4849de251df6f33aaed5584c975018.png)

许多数据科学工作倾向于关注那些非常明显的属性，如训练模型、验证、统计……然而，如果没有可用的数据，所有这些技能最终都像是在泥沼中冲刺。如今用于在网络上传输数据的最大工具之一是 Flask。Flask 是一个非常棒的工具，通常用于在 Python 中创建数据驱动的 web 应用程序和管道。Flask 有大量的内置功能，可以与 SQL、Json 和 HTML 一起使用，这使得它非常适合 web 上的后端使用。

> 请求和返回

为了拥有一个完全基于云的管道，以及相应的回报，我们需要了解 Flask 的请求和返回系统。对于计算型或信息型返回管道，通常会保留默认路线，或者:

```
[@app](http://twitter.com/app).route('/')
```

这意味着 Flask-app 的目录没有被更改，并允许我们在访问路由时执行一个函数。这可以比作“Index.html”

当然，我们知道我们在哪里，但是我们如何将超文本传输协议应用到 Flask 以接收数据和传输结果呢？为此，我们需要向我们的项目添加请求:

```
from flask import Flask, render_template, request
```

对于这个特殊的例子，我和我的一些同事一起做一个项目；全栈开发者、UI/UX 开发者、前端开发者。和往常一样，我有一个 Github 回购的源代码链接。我的任务是创建一个算法，该算法将获取房屋的某些属性，并返回一个预测，用户可以将该预测保存到他们的帐户中。最大的挑战是以一种特定的方式设置事情，以便他们的请求和我的回报可以正确地排列。如果你想看整个项目的 Github，这里是[。](https://github.com/FTBW-AppraisersBFF)

有了这个方法，我构建了我的 Sklearn 管道，并把它分配出去。让这变得困难的是数据中包含的大量观察结果…这很难处理，尤其是首先很难阅读。如果你不确定这样的事情是如何做的，你可以在这里查看[笔记本。](https://github.com/emmettgb/Emmetts-DS-NoteBooks/blob/master/Python3/Exploratory_zillow.ipynb)

我选择的模型是 XGBoost 回归器，对于这种情况，它有自己的优点和缺点。最大的缺点是一个不重要的特性会扭曲结果。我只用了几个特征，就像我们数据科学家所做的那样，我可以用统计数据证明这些特征对模型的结果有重大影响，然后用标准标量把它放入管道，以提高一点准确性。

# 最后请求！

你曾经看过地址栏，并在谷歌搜索中读出网址吗？您可能会看到类似这样的内容:

```
search=hello%where%are%we?
```

这是因为 Google 的整个网站都是建立在 HTTP 请求之上的，当然这和返回管道有一点不同，但是不管怎样，这就是这个方法的价值范围。

我们可以使用函数从 flask 请求传输协议 URL 信息

```
request.args
```

所以在我的项目中，我做了这样的事情:

```
try:        
  bathrooms = request.args['bathrooms']        
  bedrooms = request.args['bedrooms']        
  squarefeet = request.args['squarefeet']        
  yearbuilt = request.args['yearbuilt']    
except KeyError as e:        
  return ('Some Values are missing')
```

这只是在当前的 http 请求中请求参数，以便将参数引入 Python。因此，如果我们请求应用程序的路线，比如:

```
ourapp.com/?[bedrooms=5&bathrooms=2&squarefeet=1500&yearbuilt=1988](https://predictorman.herokuapp.com/?bedrooms=5&bathrooms=2&squarefeet=1500&yearbuilt=1988)
```

我们的请求会带来:

```
bedrooms = 5
bathrooms = 2
squarefeet = 1500
yearbuilt = 1988
```

很酷，对吧？你也可以在这里测试一下。但现在我们进入了利用这条管道的有趣部分，即返回所请求的信息。还记得我们基于工作的管道吗？我们现在可以把它带进我们的 Flask 应用程序

```
from joblib import load
pipeline = load('alg.sav')
```

我最终将所有的值都放入一个观察数据框中，因为这似乎是最容易处理的。最后但同样重要的是，我们可以使用管道进行预测，并返回结果:

```
estimate = pipeline.predict(dcry)        
return str(int(estimate))
```

因此，我们在应用程序的路径目录下的最后一个功能是:

```
def template():
    try:
        bathrooms = request.args['bathrooms']
        bedrooms = request.args['bedrooms']
        squarefeet = request.args['squarefeet']
        yearbuilt = request.args['yearbuilt']
    except KeyError as e:
        return ('Some Values are missing')
    try:
        bathrooms = float(bathrooms)
        bedrooms = float(bedrooms)
        squarefeet = float(squarefeet)
        yearbuilt = int(yearbuilt)
    except ValueError as e:
        return ('That aint a number, Cowboy.')
    else:
        dcry = pd.DataFrame({"YearBuilt": [yearbuilt],        "LotSize": [squarefeet],"Bedrooms": [bedrooms],        "Bathrooms": [bathrooms]})
        pipeline = load('alg.sav')
        estimate = pipeline.predict(dcry)
        return str(int(estimate))
```

# 最后

这些管道绝对可以成为任何数据驱动的应用程序的基础。拥有一个独立的实体的好处是显而易见的，尽管这也可以在本地应用程序中实现。无论你是想返回 k-best 结果，还是想预测一栋房子的价格，数据管道都是一种非常好的方式，可以完全独立地在幕后完成。