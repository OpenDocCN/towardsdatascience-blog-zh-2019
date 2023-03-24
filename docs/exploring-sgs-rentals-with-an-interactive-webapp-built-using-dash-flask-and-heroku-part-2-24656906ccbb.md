# 使用 Dash、Flask 和 Heroku 构建的交互式 Webapp 探索 SG 的租赁——第 2 部分

> 原文：<https://towardsdatascience.com/exploring-sgs-rentals-with-an-interactive-webapp-built-using-dash-flask-and-heroku-part-2-24656906ccbb?source=collection_archive---------18----------------------->

## 通过构建 DS/ML Webapp 并将其部署到世界各地来升级

** *这是 2 篇系列文章的第 2 部分。查看* [*第一部*](https://medium.com/p/cccd9e8dd1b8/edit) *这里！*

在上一篇文章中，我们讨论了如何构建不同的组件来形成 Flaskapp。现在是时候将它们集成在一起了

## 4.把所有东西放在主程序中

这是万物融为一体的地方。主程序配置后端 Flaskapp，并定义它如何响应来自前端的请求。所有子模块都在这里调用

********* *主程序的* ***骨架*** *如下图*

```
***import ___                       //whatever packages you need
***import ___                       //whatever helper funcs needed
from flask import Flask, request, render_template
from dash import Dash
from dash_core_components import ____
from dash_html_components import ____
from wtforms import Form
from wtforms.fields import _____    //choose necessary fields**app** = Flask(__name__)
**app_dash** = Dash(__name__, server=app, url_base_pathname = '/dash/')
app_dash.layout = ____              //set up your dash layoutform1 = Form(request.form)
form2 = Form(request.form)
//Form can be customized using extented class and wtforms.fields**@app.route**('/', methods=___)        //main view, choose methods
def some_function():
    if request.form = form1
        //Do something
        **return** ___ 
    elif request.form = form2
        //Do something
        **return** ___
//something to the server OR render_template('somepage.html')**@app.route**('/route1', methods=___)  //new view, choose methods  
def some_function():
    //Do something
    **return** ___ 
//something to the server OR render_template('somepage.html')**@app.route**('/route2', methods=___)  //new view, choose methods   
def some_function():
    global app_dash
    //Do something
    app_dash.layout = ___           //change your dash layout  
    **return** flask.redicrect('/dash/')
//direct the flask app to display the Dash dashboardif __name__ == '__main__':
 **app**.run(port=8000, debug=True)
```

主程序由**一个或多个视图**组成。每个路线/视图都有一个代码部分。每个路线/视图**最终都必须返回**一些东西

一旦你理解了上面的框架，你就可以开始阅读我的复杂得多的主程序，而不会被它包含的代码行数所淹没。此程序的更多解释如下

** * *我的主程序的一些说明*

*   **应用** = Flask app 的名称。每个 **@application.route()** 代码块定义一个路径/视图
*   主程序有 **3 个视图** : A)主视图'/'，B)第二视图'/哈哈哈'和 C)第三视图'/seaborn '
*   类 **ReusableForm(Form)** 是一个将基本表单类扩展为具有可定制输入字段的新表单类的例子
*   如果没有来自前端网络浏览器的请求，**主视图**提供由 **5 个表格**组成的**index.html**页面。**第一节**
*   如果有来自**表单 1** 的**发布**请求，应用程序修改数据集，重新显示带有一些通知文本的**index.html**页面
*   如果有一个来自 **form2** 的 **POST** 请求，app 会创建并训练一个 **ML 模型**，然后用它来预测所提议房产的**租金**，并返回 **prediction_result.html** 页面显示预测**。**
*   如果有一个来自 **form3、**的 **POST** 请求，应用程序会搜索一些**的与提议的酒店相似的房源**，并返回 **filtering_result.html** 显示这些房源的详细信息
*   如果有一个来自 **form4 的 **POST** 请求，**应用程序触发第二个视图**@ application . route('/哈哈哈')**，这将改变 **Dash 应用程序的布局**，然后将用户重定向到其**仪表板**
*   如果有一个来自 **form5、**的 **POST** 请求，应用程序会触发第三个视图**@ application . route('/seaborn ')**，返回一个显示图表的页面
*   **方法**:指定路由/视图可以处理来自前端网页的哪种 **HTTP 请求(' POST '或' GET')**
*   **request.form** 用于定义一个表单。此外，它还用于接收**后端程序所需的** **用户输入**。
*   **request.method** 用于在每个视图中构造 **if 子句**，分别处理不同类型的 HTTP 请求。
*   **render_template** :该函数用于在触发路线/视图时返回一个 Html 页面。模板内部调用的表单和变量需要作为参数解析到 render_template()函数中
*   **全局**用于每个视图内的变量，因此每个视图内的动作导致变量的永久变化，这些变化可以在整个程序中共享
*   **port=8080** 将允许您通过浏览器中的 localhost:8080 在本地部署应用程序

— — — — — — — — — — — — — — — — — — — — — — — — — — — — — — —

## 5.在 HEROKU 上部署完成的应用程序

在你完成主程序并确保程序中不再有 bug 之后(重要！—在本地运行，直到你对此非常有信心)，然后你就可以把它部署到 Heroku 上，这是一个面向业余开发者的流行的免费应用托管网站

![](img/9d43fe2630395dcdc41e868c6171f9b8.png)

## 创建帐户

但是，在您做任何事情之前，您需要首先通过访问该网站创建一个帐户，除非您已经有了一个帐户。

创建您的帐户并登录后，您将看到一个可以管理您的应用程序的仪表板。您可以选择从那里配置您的应用程序，或者下载适用于您的操作系统的 [Heroku Cli](https://devcenter.heroku.com/articles/heroku-cli) 并使用客户端配置您的应用程序。我将在这里展示第二种方法

安装 CLI 客户端后，在您的终端中运行以下命令，登录到您的 Heroku 帐户

```
$ heroku login
```

出现提示时，输入您的凭据(电子邮件和密码)。

## 创建 Heroku 应用程序

部署过程的下一步是您需要创建一个 Heroku 应用程序。为此，请在您的终端中运行以下命令；

```
$ heroku apps:create unique-name-of-your-app
```

此操作将在 Heroku 上创建一个应用程序，并为您上传文件创建一个存储库

## 准备 Git 存储库

在这个阶段，我假设您应该已经建立了一个本地 git 存储库。您需要通过运行命令向现有的本地目录添加一个新的远程上游

```
$ git remote add heroku YOUR_HEROKU_REPO_ADDRESS
```

另一方面，如果您还没有设置存储库，请将创建的 Heroku 存储库克隆到您的本地目录中，并将您的工作文件复制到该文件夹中。

## 选择合适的 web 服务器

Flask 对于开发阶段来说是一个很好的 web 服务器，但对于生产环境来说不是最佳选择。建议使用更好的 web 服务器。最受欢迎的两个是 Gunicorn 和 waste。你可以在这里查看这些服务器的列表。我在我的应用程序中使用 waste，因为我发现它有助于应用程序更好地运行，因为它能够缓冲请求和响应。另一方面，Gunicorn 导致我的应用程序给出不一致的响应，可能是因为它没有缓冲能力，无法处理我的应用程序的缓慢进程。此外，这里的一篇文章[解释了 Gunicorn 不是 Heroku 应用程序最佳选择的一些原因](https://blog.etianen.com/blog/2014/01/19/gunicorn-heroku-django/)

您可以用这个命令下载这两个文件

```
$ pip install waitress
OR
$ pip install gunicorn
```

## 创建 Procfile

procfile 告诉 Heroku 如何运行你的应用程序。该文件没有文件类型扩展名。用简单的命令在存储库中创建一个

```
$ touch Procfile
```

将下面一行添加到配置文件中。我为 Guniconr 和 waste 都举了例子。此外，“文件名”是您的主程序文件的名称，而“应用程序名”是您的程序中配置的 Flask 应用程序的名称。

```
$ web:gunicorn -w:? file_name:app_name
OR
$ web:waitres-serve --port=$PORT -w:? file_name:app_name
\\? is the number of threads/workers to be started by the server.
```

## 准备 Requirements.txt 文件

这个文件告诉 Heroku 哪些库/包要和你的应用一起安装来运行它。它包含包的名称和每个包的版本号。如果您不确定您当前使用的是什么版本，您可以使用“$ pip list”命令进行检查。记得把新下载的网络服务器放在其中

这是我自己的 requirements.txt 文件的一个例子

```
gunicorn==19.9.0
waitress==1.2.1
numpy==1.16.2
pandas==0.23.0
matplotlib==2.2.2
seaborn==0.9.0
pyspark==2.3.0
scikit-learn==0.20.3
plotly==3.3.0
flask==1.0.2
flask_sqlalchemy==2.3.2
wtforms==2.2.1
psycopg2==2.7.7
dash==0.38.0
dash_core_components==0.43.1
dash_html_components==0.13.5
```

但是，如果您一直在使用虚拟环境开发您的应用程序，那么您只需运行一个简单的命令，就可以帮助您在一瞬间创建这个文件

```
$ pip freeze > requirements.txt
```

## 指定合适的构建包

在部署到 Heroku 之前，您需要指定您的应用程序将使用哪个(哪些)构建包。您可以通过运行以下程序来添加它们(根据您的应用程序的语言来更改构建包)

```
$ heroku buildpacks:set heroku/python
```

## * *添加数据库(可选)

如果您的应用程序需要数据库服务器，Heroku 提供免费或付费的 Postgresql。自由层版本最多只允许 10，000 行和 20 个并发连接。

实际上，为应用程序设置数据库并配置它与数据库进行交互需要几个步骤。

**步骤 1** :要创建一个数据库，运行以下命令

```
$ heroku addons: add heroku-postgresql:hobby-dev
\\hobby-dev is the free tier mentioned
```

您可以从应用仪表板查看数据库的连接详细信息。只需转到新创建的数据库部分>点击设置>查看凭证。在这里你可以看到数据库的 URI 被配置为一个名为`$DATABASE_URL` 的环境变量。数据导入过程和程序设置都需要这个变量。

**步骤 2** :通过添加下面的代码片段来修改您的 flask 应用程序，以便它可以连接到数据库。

```
import os
app.config['SQLALCHEMY_DATABASE_URI'] = os.environ['DATABASE_URL']
\\app is your Flask app's name
```

**第三步**:将您的数据导入 Heroku

我将假设您的数据以 CSV 文件的形式出现。如果您已经有了本地 PostgreSQL 数据库中的数据，或者 SQL 文件形式的数据，那么您实际上可以跳过下面的一些步骤

```
**STEP 0\.** Download and install PostgreSQL on your computer **STEP 1**. Create a local Postgres Database & Table (with name=Tablename) with PGAdmin**STEP 2**. Run the following command
$ psql -U user_name -d database_name -c "\copy Tablename FROM '\file directory\file.csv' WITH CSV;"
\\ Transfer the csv file to local database\\If you have many columns, table creation can be a headache. For such case, you can use a tool called pgfutter to automate table creation with some datatype identification. Check out this [answer](https://stackoverflow.com/questions/21018256/can-i-automatically-create-a-table-in-postgresql-from-a-csv-file-with-headers) on stackoverflow**STEP 3**. Push from local database to remote Heroku database
Method1:
A. pg_dump -U user_name -d database_name --no-owner --no-acl -f backup.sql
B. heroku pg:psql -app app_name < backup.sqlMethod2:
A. heroku pg:reset //make sure remote database is empty
B. heroku pg:push local_database_name --app app_name
```

**步骤 4** :创建 SQLAlchemy ORM 对象，以便从 Flask 应用程序内部与数据库进行通信。您首先需要安装软件包

```
$ pip install flask_sqlalchemy
```

然后你把这些代码添加到你的程序中

```
from flask_sqlalchemy import SQLAlchemy
db = SQLAlchemy(app)
\\app is your Flask app's name
```

下面是代码要点，展示了我如何使用我的 ORM 与数据库通信，查询它，用一些用户的输入条件过滤查询，并返回一个结果数据帧。我在这里使用反射从步骤 3 中上传的现有数据库中加载信息

或者，您可以通过用 **db 声明 table 类来创建一个全新的表。在 Flask 应用程序中建模**，并使用 **db.session.add()** 和 **db.session.commit()** 将数据添加到表中。当你的应用有一些允许用户操作/更新数据库的特性时，这是很有用的。你可以看看这篇文章了解更多关于如何做的细节

## * *为 Heroku 定制

在将应用程序部署到 Heroku 时，有两件事需要注意

1.  Heroku 自由层(和一些低级计划)只有**低内存分配** (512MB)。确保您的进程/应用程序在此限制内使用资源。如有必要，自定义您的应用程序
2.  Heroku 有一个 **30 秒超时策略**。这意味着对于 web 客户端请求，您的后端服务器有 30 秒的“期限”来返回响应。如果你的服务器需要更长的时间，应用程序将崩溃

我用来绕过第二条规则的一个技巧(我的一些过程，比如模型训练或交互式情节准备需要超过 30 秒)是让应用程序**以某种形式的状态**返回给服务器，关于正在运行的过程，在**指定的间隔**小于 30 秒

```
<h3 class=”heading-3-impt-red”>Click <button style=”font-size:20px” **onclick=”myfunc()**”>Here</button> to Start Preparing Interactive Scatter Plot
</h3><script src="[**https://ajax.googleapis.com/ajax/libs/jquery/2.2.0/jquery.min.j**s](https://ajax.googleapis.com/ajax/libs/jquery/2.2.0/jquery.min.js)"></script>
<script>

  **function myfunc(){** alert("Preparing Plot Now...");
  count = 0;
  var interval = setInterval(function(){**wowwah()**; count++; if (count>=4)clearInterval(interval);},8000); // call every 8 seconds
  };

  **function wowwah(){** $.ajax({
  url:"/hehehe", //the page containing python script
  method:"GET",
  dataType:"json",
  success:function(result){ alert(result); }
  });}
</script>
```

我在我的 HTML 模板中使用了这个脚本。每当调用 wowwah()函数时(每 8 秒一次)，视图“/hehehe”以 jsonify()对象的形式向浏览器返回一个状态文本。

## 部署到 Heroku

最后，当所有工作完成后，你可以将应用程序部署到 Heroku，就像你将你的东西部署到 Github 一样。确保您准备好所有内容，并更新您的。git 如果需要，忽略文件

```
$ git add *
$ git commit
$ git push heroku master
// change heroku to origin if you previously clone the heroku repo
```

在***your _ app _ name . heroku . com***访问您的应用程序，看看它是否有效。如果它不像预期的那样工作，您可以在控制台中键入“$ heroku logs-tails”来找出错误是什么。如果是的话，那么恭喜你成功了！您已经构建了您的第一个 DS webapp

## 6.结论

通过 Flask 应用程序将您的 DS/ML 项目部署到 Heroku，使您能够将您的想法和发现转化为工作产品，并向全世界展示。对于那些新手来说，这应该让你体会到当公司聘请数据科学家/软件工程师团队来交付数据科学项目时，他们真正关心的是什么:一个成品生产级模型/软件，可以在需要时为他们提供可操作的见解，而不仅仅是一份报告或一个代码笔记本。

尽管我自己在这方面还是个新手，但我希望我的文章能够对您的学习之旅有所帮助，并成为数据科学艺术的更好实践者。如果是的话，我很乐意看到你的一两条评论来解释这是怎么回事。如果您能给我一些反馈，告诉我如何更好地改进我的帖子/文章，我将不胜感激

感谢大家一直关注到这个阶段！不要忘记关注我的下一篇文章。

Huy，