# 使用 Python 和 Gmail 帐户为您的刮板设置电子邮件更新

> 原文：<https://towardsdatascience.com/setting-up-email-updates-for-your-scraper-using-python-and-a-gmail-account-4e678be99ed5?source=collection_archive---------13----------------------->

![](img/0010288f80e66494c3bcec97c24d75c6.png)

Photo by [Jamie Street](https://unsplash.com/photos/33oxtOMk6Ac?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/notification?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

通常，当构建 web 抓取器来收集数据时，您会遇到以下情况之一:

*   您想将程序的结果发送给其他人
*   您正在远程服务器上运行该脚本，并且希望获得自动、实时的结果报告(例如，来自在线零售商的价格信息更新，表明竞争公司对其职位空缺网站进行了更改的更新)

一个简单而有效的解决方案是让你的网络抓取脚本自动将结果通过电子邮件发送给你(或者其他感兴趣的人)。

事实证明这在 Python 中非常容易做到。你只需要一个 Gmail 账户，就可以搭载谷歌的简单邮件传输协议(SMTP)服务器。我发现这个技巧真的很有用，特别是对[来说，我最近创建了一个项目](https://github.com/marknagelberg/mint-finance-monitor)，通过一个程序给我和我的家人发送每月财务更新，这个程序对我们的 [Mint](https://www.mint.com/) 账户数据进行一些定制计算。

第一步是导入内置的 Python 包，它将为我们完成大部分工作:

```
import smtplib
from email.mime.text import MIMEText
```

`smtplib`是内置的 Python SMTP 协议客户端，允许我们连接到我们的电子邮件帐户并通过 SMTP 发送邮件。

`MIMEText`类用于定义电子邮件的内容。MIME(多用途互联网邮件扩展)是一种标准，用于格式化要通过互联网发送的文件，以便可以在浏览器或电子邮件应用程序中查看。它已经存在很久了，它基本上允许你通过电子邮件发送除 ASCII 文本之外的东西，比如音频、视频、图像和其他好东西。以下示例用于发送包含 HTML 的电子邮件。

以下是构建 MIME 电子邮件的示例代码:

```
sender = ‘your_email@email.com’
receivers = [‘recipient1@recipient.com’, ‘recipient2@recipient.com’]
body_of_email = ‘String of html to display in the email’
msg = MIMEText(body_of_email, ‘html’)
msg[‘Subject’] = ‘Subject line goes here’
msg[‘From’] = sender
msg[‘To’] = ‘,’.join(receivers)
```

`MIMEText`对象将电子邮件消息作为一个字符串接收，并指定该消息有一个 html“子类型”。参见[本网站](https://www.sitepoint.com/web-foundations/mime-types-complete-list/)获取 MIME 媒体类型和相应子类型的有用列表。查看 [Python email.mime docs](https://docs.python.org/2/library/email.mime.html) 中可用于发送其他类型 mime 消息的其他类(例如 MIMEAudio、MIMEImage)。

接下来，我们使用主机`‘smtp.gmail.com’`和端口 465 连接到 Gmail SMTP 服务器，使用您的 Gmail 帐户凭据登录，然后发送:

```
s = smtplib.SMTP_SSL(host = ‘smtp.gmail.com’, port = 465)
s.login(user = ‘your_username’, password = ‘your_password’)
s.sendmail(sender, receivers, msg.as_string())
s.quit()
```

*注意:*注意，电子邮件收件人的列表需要在`msg[‘From’]`的赋值中表示为一个字符串(每封电子邮件用逗号分隔)，在`smtplib`对象`s.sendmail(sender, receivers, msg.as_string()`中指定时表示为一个 Python 列表。(在相当长的一段时间里，我都在用头撞墙，试图找出为什么消息只发送给第一个收件人，或者根本没有发送，这就是错误的根源。终于碰到[这个 StackExchange 帖子](http://stackoverflow.com/questions/8856117/how-to-send-email-to-multiple-recipients-using-python-smtplib)解决了问题。)

作为最后一步，你需要更改你的 Gmail 帐户设置，以允许访问“不太安全的应用程序”，这样你的 Python 脚本就可以访问你的帐户并从它发送电子邮件(参见此处的说明)。在您的计算机或其他机器上运行的 scraper 被认为是“不太安全的”,因为您的应用程序被视为第三方，它会将您的凭据直接发送到 Gmail 以获得访问权限。相反，第三方应用程序应该使用像 OAuth 这样的授权机制来访问你的账户(见这里的讨论)。

当然，您不必担心您自己的应用程序访问您的帐户，因为您知道它不是恶意的。然而，如果其他不受信任的应用程序可以做到这一点，他们可能会在不告诉您或做其他讨厌的事情的情况下存储您的登录凭据。因此，允许不太安全的应用程序访问会使您的 Gmail 帐户不太安全。

如果你不愿意在你的个人 Gmail 帐户上打开对不太安全的应用程序的访问，一个选择是创建第二个 Gmail 帐户，专门用于从你的应用程序发送电子邮件。这样，如果由于开启了不太安全的应用程序访问，该帐户由于某种原因而受到威胁，攻击者将只能看到来自 scraper 的已发送邮件。

*最初发表于*[*【www.marknagelberg.com】*](http://www.marknagelberg.com/setting-up-email-updates-for-your-scraper-using-python-and-a-gmail-account/)*。你可以在推特上关注我* [*这里*](https://twitter.com/MarkNagelberg) *。要访问我共享的 Anki deck 和 Roam Research notes 知识库，以及关于间隔重复和提高学习效率的技巧和想法的定期更新，* [*加入“下载马克的大脑”。*](http://downloadmarksbrain.marknagelberg.com/auth)