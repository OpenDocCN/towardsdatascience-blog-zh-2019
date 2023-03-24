# Flask 数据科学应用程序

> 原文：<https://towardsdatascience.com/flask-app-for-data-science-cf5651c8a859?source=collection_archive---------12----------------------->

数据科学家一般在后端工作。构建一个交互式应用程序不是我的强项，但它可以是展示你所能做的工作的一个很好的方式。在这篇文章中，我将讨论一些创建 Flask 应用程序的技巧，这些技巧是我在为我的[旅游推荐系统建立网站时学到的。](http://www.triprecommender.online/)

![](img/15b3d708592d430fdb8d17ec9003f4a9.png)

Photo by [Goran Ivos](https://unsplash.com/@goran_ivos?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/programming?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

## 入门指南

在我们开始考虑美化我们的应用程序之前，你应该把你所有的功能放在一个单独的 py 文件中，而不是放在一个 Jupyter 笔记本中。这一步是必要的，因为 Flask 不能与 Jupyter 通信，但它可以与 py 文件通信。

## 知识库组织

存储库的组织对于部署到网站非常重要。在进入存储库时，应该有一个与存储库同名的文件夹。这个文件夹里应该是你的应用程序需要的所有东西。例如，一个*推荐. py* 对我的应用程序是必要的，但一个*刮擦. py* 不是，因为它只用于收集数据。

该文件夹中需要的另一个文件是 *__init__.py.* 这可以是一个空文件，但这允许在导入包时运行代码。最后，您将需要一个 web_app 文件夹。此文件夹将包含运行您的应用程序所需的文件，包括引导静态文件夹和模板文件夹。

```
travel_destination_recommendation
    travel_destination_recommendation
         __init__.py
         recommend.py
         web_app
              __init__.py
              static
              templates
              app.py
```

当您想要运行/测试您的应用程序时，您将在最高目录中的命令行中使用以下代码:

```
FLASK_APP = travel_destination_recommendation.web_app.app.py
```

在 app.py 中，您可以通过以下方式导入自定义函数:

```
from ..recommend import custom_functions
```

## 创建烧瓶应用程序

py 文件是我们从 flask 导入的地方。您想要导入一些工具:

*   瓶
*   请求(允许与 Javascript 通信)
*   render_template(从 HTML 创建模板)
*   jsonify(所有内容都需要以 JSON 格式返回)

```
from flask import Flask, request, render_template, jsonifyapp = Flask(__name__, static_url_path="")
```

你的应用需要有一些东西来展示。这是你的第一个函数。任何与 HTML 通信的函数都以 decorator *@app.route('/')开始。括号内是我们想要与之交互的 HTML 的标签。/本身代表主页。*

```
[@app](http://twitter.com/app).route('/')
def index():
    """Return the main page."""
    return render_template('theme.html')
```

我建议，尤其是对于第一个应用程序，开始时不要做任何设计。我从这个函数开始。它从文本框中获取文本并返回一个预测。

```
with open('model.pkl', 'rb') as f:    
     model = model.load(f)@app.route('/predict', methods=['GET','POST'])
def predict():    
     """Return a random prediction."""    
     data = request.json
     prediction = model.predict_proba([data['user_input']])         return jsonify({'probability': prediction[0][1]})
```

## 连接到 HTML

上面的这段代码连接到下面的 HTML，它创建了一个文本框和一个标记为“预测”的按钮

```
<html>  
  <head>    
    <title>Predict</title>    
    <script type="text/javascript" src="brython.js"></script>    
    <script type="text/javascript" src="brython_stdlib.js"></script>     
  </head>    

  <body onload="brython(1)">    
    <script type="text/python3">      
      from browser import document, ajax      
      import json      
      import warnings  

      def show_results(response):          
        if response.status==200:               
          document["result"].html = response.text          
        else:              
          warnings.warn(response.text)

      def get_prediction(ev):          
        """Get the predicted probability."""          
        req = ajax.ajax()          
        req.bind('complete', show_results)          
        req.open('POST', '/predict', True)      
        req.set_header('content-type','application/json')        
        data = json.dumps({'user_input': document['user_input'].value})
        req.send(data)      

      document["predict_button"].bind("click", get_prediction)      
      </script>
      <textarea id="user_input"></textarea>    
      <button id="predict_button">Predict!</button>    
      <div id="result"></div> 
  </body>
</html>
```

这是相当多的代码。我们来分解一下。我们要确保做的第一件事是将脚本设置为 brython。Brython 代表浏览器 python，允许在 Javascript 环境中使用 python。在脚本和正文中都设置这个。

我们希望我们的代码在脚本的主体中。我们需要两个独立的函数来收集数据和显示结果。在 *show_result* s 中，我们只想要一个结果，如果存在的话。在 if 语句中，我们引用 HTML 文档的 div id*“result”*，因为这是我们希望显示结果的地方。

在*get _ prediction*中，我们有参数 *ev* 表示事件。这意味着该函数将在特定事件发生时被调用(我们稍后将回到这一点)。

*req.bind('complete '，show_results)* 行将该函数连接到 *show_results* 函数。

下一行 *req.open('POST '，'/predict '，True)* 通过装饰器将这个函数连接到我们在 app.py 文件中创建的函数。

数据是我们可以传递给 app.py 文件的内容。注意，结果应该被转储为 JSON 格式。这一行有两个“用户输入”。字典中的那个是键，它在 app.py 函数中作为数据的一部分被引用。该数据通过 *req.send(data)* 线发送至 app.py 函数。文档["user_input"]是文本框的 id。

既然函数已经完成，我们需要一个触发函数的事件。我们希望在点击预测按钮时进行预测。脚本中的最后一行用特定的 id 连接按钮，并将“click”绑定到函数 *get_prediction。*

## 结论

这是一个非常基础的 Flask app。随着时间的推移，通过小的改变，我们可以创建一个漂亮的应用程序。作为一名数据科学家，你不需要成为 web 开发的大师，但它可以成为一个有趣的工具来展示你的模型可以做什么。

感谢我的教练为我的烧瓶起点，检查他的垃圾邮件预测[在这里](https://github.com/mileserickson/flask-brython-example)。要查看一个详细的 Flask 应用程序，看看我的[库](https://github.com/kayschulz/travel_destination_recommendation)。