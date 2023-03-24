# 如何使用 Heroku 将您的网站部署到自定义域

> 原文：<https://towardsdatascience.com/how-to-deploy-your-website-to-a-custom-domain-8cb23063c1ff?source=collection_archive---------12----------------------->

这篇博客记录了使用 [Heroku](https://heroku.com) 和 [NameCheap](https://namecheap.com) 将一个用 Python 和 [Flask](http://flask.pocoo.org/) 框架编写的网站部署到一个定制域所需的步骤。Flask 是一个微框架，它允许我们在后端使用 Python 与 HTML/CSS 或 Javascript 中的前端代码进行交互，以构建网站。人们也为此使用其他框架，比如 Django，但是我最喜欢的是 Flask，因为它很容易上手。

![](img/184b3a92cec44a609c400c053da76875.png)

Having your website can help you stand out

一个小 Flask 网站可以通过创建一个新的 repo 来创建，通过创建一个 Python 文件，如 [this](http://flask.pocoo.org/) :

```
# app.py 
from flask import Flask
app = Flask(__name__)

@app.route("/")
def hello():
    return "Hello World!"**if** __name__ == **"__main__"**:
    app.run(port=5000)
```

一个典型的 Flask 应用程序有文件夹`static`存放所有的 CSS 和 JS 文件以及图片等。和一个用于 HTML 文件的`templates`文件夹。我不会在这里说太多细节，但是在米格尔·格林伯格的[教程中可以找到关于这方面的精彩信息。](https://blog.miguelgrinberg.com/post/the-flask-mega-tutorial-part-i-hello-world)

当运行上面的文件`python app.py`时，它会在本地打开网站，显示“Hello World！”用 HTML 写的。

我们如何将这个网站部署到一个自定义域，让每个人都可以看到？

## 1.创建一个 Heroku 帐户

去 Heroku [网站](https://www.heroku.com/)注册一个账号。Heroku 将允许我们将网站部署到以`herokuapp.com`结尾的域中。我们将使用这个网站，并部署到自定义域。我们本可以使用亚马逊网络服务来部署网站，但是我发现这比使用像 Heroku 这样的解决方案要复杂得多。AWS 是基础设施即服务(IaaS)，Heroku 是平台即服务(PaaS)。因此，我们不需要担心 Heroku 的具体基础设施细节！这使得它更容易使用。

现在，到这个[链接](https://devcenter.heroku.com/articles/getting-started-with-python#set-up)并下载命令行的`heroku`。如果您的机器上有自制软件，您可以使用`brew install heroku/brew/heroku`将其安装到您的机器上。下次运行`heroku login`输入您的 Heroku 凭证。

![](img/593a961ee737c8a9755e82cec703a48c.png)

Heroku is used to deploy our local website to cloud

## 2.将您的应用部署到 Heroku

**a)制作一个 Procfile:**
要将你的 app 部署到 Heroku，首先创建一个`Procfile`，保存在与`app.py`相同的文件夹中。请注意，Procfile 的**而**没有任何扩展名，如`txt`等。该 Procfile 将包含内容`web: gunicorn app:app`该语句将允许 Heroku 运行 gunicorn 作为该应用程序的 web 服务器。确保你的虚拟环境中安装了`gunicorn`。

**b)制作 requirements.txt:**
还有，用`pip freeze > requirements.txt`把这个项目的所有需求写成一个文件。Heroku 将使用它在云中安装所有必要的软件包来运行您的应用程序。

**c)【可选】Make runtime.txt:**
如果你想指定 Heroku 应该在云端使用哪个 Python 版本来运行你的 app，在这个文件里就提为`python-3.6.6`就可以了。

**d)部署到 Heroku:** 运行`heroku create <app-name>`其中`<app-name>`是你喜欢的名字。如果你的应用程序名称是`calm-river`，那么 Heroku 会将你的网站部署到`calm-river.herokuapp.com`。确保`git`安装在你的命令行上。
运行`git init`将当前文件夹初始化为 git 存储库。接下来，运行`git add .`，然后运行`git commit -m "first commit"`，将所有文件提交给`git`。最后运行`git push heroku master`将你的应用部署到 Heroku~
如果一切顺利，你可以在`<app-name>.herokuapp.com`访问你的网站

## **3。链接到 Heroku 上的自定义域名:**

接下来，我们需要从 NameCheap 或 GoDaddy 购买一个自定义域名。假设您购买了域名，`example.com`
接下来，运行`heroku domains:add [w](http://www.calm-river.com)ww.example.com`将这个自定义域名添加到您的 Heroku 应用程序中。它会给你一个 DNS 目标类型和 DNS 目标名称。这个目标类型应该是`CNAME`。我们需要将这个`CNAME`记录链接到你域名的设置控制台。

## 4.将 CNAME 记录添加到您的自定义域:

进入 Namecheap 设置控制台，然后进入`Advanced DNS`。将`CNAME`记录添加到主机目标，其中`Type`为`CNAME`,`Host`为`www`,`Value`为您将域添加到 Heroku 时收到的 DNS 目标名称。

## 5.将 URL 重定向记录添加到您的自定义域:

添加 CNAME 记录后，添加`URL Redirect record`，主机为`@`，设置`Unmasked`为`[http://www.example.com](http://www.example.com)`。

允许 30 分钟来传播这些设置，您的网站应该在您的自定义域上运行！希望这些步骤对你寻求与世界分享你的工作有用。[更多细节](https://devcenter.heroku.com/articles/custom-domains)可以在 Heroku 网站上找到这些步骤。
希望能看到你的网站转转！

**附注:我正在创建一个名为
“查询理解技术概述”**的新课程。
本课程将涵盖信息检索技术、文本挖掘、查询理解技巧和诀窍、提高精确度、召回率和用户点击量的方法。有兴趣的请在这里报名[！这种兴趣展示将允许我优先考虑和建立课程。](https://sanketgupta.teachable.com/p/query-understanding-techniques/)

![](img/11a0a9419ef0c239a2529bb5b76d4b1d.png)

[My new course: Overview of Query Understanding Techniques](https://sanketgupta.teachable.com/p/query-understanding-techniques)

如果你有任何问题，请给我的 [LinkedIn 个人资料](https://www.linkedin.com/in/sanketgupta107/)留言，或者给我发电子邮件到 sanket@omnilence.com。感谢阅读！