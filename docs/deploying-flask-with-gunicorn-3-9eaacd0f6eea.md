# 用 Gunicorn 3 展开烧瓶

> 原文：<https://towardsdatascience.com/deploying-flask-with-gunicorn-3-9eaacd0f6eea?source=collection_archive---------5----------------------->

![](img/af0889d46998ae75c857c30b9858809d.png)

应用程序部署经常会带来意想不到的后果和错误。部署可能是决定最终期限的步骤，也可能是决定应用程序成败的步骤。许多数据科学家害怕这种部署，因为它可能非常乏味，并且需要我们许多人不一定精通的技能(开发-运营。)在我的[“数据科学的软技能”文章](/the-unspoken-data-science-soft-skills-cc836d51b73d?source=your_stories_page---------------------------)中，我提到了数据科学家在某些方面是如何实践开发-运营技能的。这是有充分理由的，因为我发现自己对*“系统管理员活动”越来越感兴趣了*

我真正喜欢的一项服务是基于 Linux 的 VPS 主机服务。Linode 非常棒，因为它拥有我想要的极简主义，以及我喜欢的安全性和功能。因此，我们首先假设您有一个想要托管的域(或 IP)，并且有一个 Linode 或类似的虚拟专用服务器正在运行，准备好通过 SSH 连接。

# 进去吧。

第一步当然是知道你的主机和根密码。您的主机将是 Linode 提供给您的 IPV4 地址，例如:

```
48.87.54.01
```

为了 SSH 到我们的服务器，我们可以使用一个 SSH 工具，比如 PuTTy(主要针对 Windows 用户)，或者只是通过我们的终端使用 SSH。然后就像对你的根用户一样简单了。万一这个命令返回“ssh: command not found ”,您可能需要安装 openssh。

```
ssh root@your.ip.address
```

太好了！现在我们进来了，我们必须创建一个新用户！当然，这取决于你的发行版…

![](img/cef84c44ec8f9e5c704b19714a2c76b6.png)

我要戴上红帽:

```
useradd *username*
```

但是如果你选择了 Ubuntu:

```
adduser username
```

现在，如果您想让您的新帐户能够使用超级用户的密码访问他们的命令:

```
usermod -aG sudo username
```

# 安装目录(NGINX)

设置您的用户帐户后，您可以使用 root 的“login”命令登录，或者您可以使用您的用户名替换 root，通过 SSH 登录您的用户帐户:

```
ssh username@your.ip.address
```

第一步是获取 NGINX。你可以把 NGINX 想象成一个巨大的路由器，NGINX 使得从一个网站服务多个域和监听端口变得非常容易。所以打开你的终端，通过你的包管理器安装它。

```
sudo dnf install nginx
```

您的操作系统的软件包管理器(如果您不知道)也在下面。(只需运行 sudo {包管理器}安装 nginx)

```
Distro      |     Package Manager
---------------------------------
Ubuntu      |        apt-get
RedHat      |      dnf / yum
Opensuse    |          man
Arch        |         pacman
```

现在我们需要在 HTTP 协议中为 www 建立一个域或位置。无论是购买的域名，还是你的服务器的 IP 地址，都没有那么重要。接下来，我们必须编辑 NGINX 的配置文件:

```
sudo nano /etc/nginx/sites-enabled/flask_app
```

这将打开一个文本编辑器，文件有可能是旧的，但很可能是新的。结果应该是这样的:

```
**server** {
    **listen** 80;
    **server_name** your.ip.here;

    **location** / {
        **proxy_pass** http://127.0.0.1:8000;
        **proxy_set_header** Host $host;
        **proxy_set_header** X-Forwarded-For $proxy_add_x_forwarded_for;
    }
}
```

现在我们只需要取消默认配置文件的链接:

```
sudo unlink /etc/nginx/sites-enabled/default
```

现在向上移动到根目录(不是您的用户主目录。)

```
cd ..
```

现在我们需要在 var/www/your.ip.here 中创建一个目录。

```
sudo mkdir var/www/your.ip.here
```

然后给我们自己超级用户特权，这样我们就不需要使用 sudo 来访问我们的文件系统。

```
sudo chown var/www/your.ip.here
```

基本上就是这样了！恭喜你，你的静态文件将正式在这个目录中工作！但是弗拉斯克呢？我们只需要一个快速的服务器设置！

# 设置服务器(Gunicorn)

将您的文件克隆、SSH 或安全复制到我们在文件系统设置中创建的 var/www 目录中。现在我们需要重新安装 Gunicorn 3。

```
sudo dnf install gunicorn3
```

并且和 supervisor 做同样的事情(如果需要可以像上次一样参考包管理器图表！)

```
sudo dnf install supervisor
```

现在我们必须创建一个主管，所以准备在 Nano 中打开更多的文本吧！

```
sudo nano /etc/supervisor/conf.d/flask_app.conf
```

并用您的设置进行配置！：

```
[program:flask_app]
directory=/var/
command=gunicorn3 -workers=3 flask_app:app
autostart=true
autorestart=true
stopasgroup=true
killasgroup=true
stderr_logfile=/var/log/appname/lognameerr.log
stdout_logfile=/var/log/appname/lognamestdout.log
```

然后为你的日志创建目录(当然这不是完全必要的，但是绝对是一个好的实践。)

```
sudo mkdir /path/to/logs (/var/log for me)
sudo touch outputpath
sudo touch stderrpath
```

# 恭喜你！

你已经用“传统的方式”部署了你的 Flask 应用程序这允许更多的扩展能力和多功能性，特别是与 Heroku 等其他替代方案相比。我认为这是一种非常有趣的方式，但有些人可能会说这有点乏味。当然，这些通常不是 DS 的职责，但是如果你发现你的团队陷入困境，这肯定是你很高兴学到的东西！