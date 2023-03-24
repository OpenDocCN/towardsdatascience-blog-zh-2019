# 如何使用 Python 发送带附件的电子邮件

> 原文：<https://towardsdatascience.com/how-to-send-email-with-attachments-by-using-python-41a9d1a3860b?source=collection_archive---------11----------------------->

## 使用 Python 电子邮件库实现报告自动化的几个简单步骤。

![](img/06376798963cd5b8a460b1575202b63e.png)

Photo by [Web Hosting](https://unsplash.com/@webhost?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/email?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

作为一名数据分析师，我经常会收到这样的请求:“你能每周给我发一份报告吗？”或者“你能每个月通过邮件把这些数据发给我吗？”。发送报告很容易，但如果你每周都要做同样的事情，那就太烦人了。这就是为什么您应该学习如何使用 python 来发送电子邮件/报告以及在您的服务器上安排脚本。

在本文中，我将向您展示如何从 **Google BigQuery** 中提取数据，并将其作为报告发送。如果你只想知道如何使用 Python 发送电子邮件，你可以跳到**电子邮件**部分。

# 导入库

像往常一样，我们必须在进入编码部分之前导入库。我们将使用 [SMTP 协议客户端](https://docs.python.org/3/library/smtplib.html)发送电子邮件。 **ConfigParser** 用于读取存储 SQL 查询的配置文件。我将在下一部分解释细节。

# 从 BigQuery 提取数据

首先，您必须创建一个配置文件来存储所有的 SQL 查询。将 python 脚本与 SQL 查询分开是一种很好的做法，尤其是当您的 SQL 查询很长(超过 20 行)时。这样，您的主脚本就不会被冗长的 SQL 查询弄得杂乱无章。

## **配置文件示例**

*【报告 1】*是您的子配置。`filename`、`dialect`和`query`是子配置的属性。

您可以使用以下代码读取配置文件。

## 写一个函数来读取子配置的属性

这个自定义函数将读取您传入的子配置文件的属性，并输出一个 CSV 文件。

`yourProjectID`将是您的 BigQuery 项目 ID，而`credential.json`将是您的 BigQuery 凭证 JSON 文件。如果您希望使用 Web Auth 登录，您可以删除它。

现在，您只需要一个循环来提取您在配置文件中定义的所有文件。`config.sections()`将返回您的配置文件中的子配置文件列表。

# 通过电子邮件发送您的报告和附件

## 定义您的电子邮件内容

以上是你如何定义你的电子邮件属性，如发件人，收件人，抄送和主题。`htmlEmail`将是你的邮件正文。您可以使用纯文本或 html 作为您的电子邮件正文。我更喜欢使用 html，因为我可以做更多的格式化工作，比如粗体、斜体和改变字体的颜色。

同样的事情，我们将使用循环来附加所有的文件。

## 发送电子邮件

出于演示目的，密码被硬编码在脚本中。这不是一个好的做法。为了更安全，您应该将密码保存在其他文件中。

现在你已经学会了如何使用 Python 发送带有附件的电子邮件。如果我犯了任何错误或打字错误，请给我留言。

您可以在我的 [**Github**](https://github.com/chingjunetao/google-service-with-python/tree/master/email-with-python) 中查看完整的脚本。如果你觉得这篇文章对你有用，请鼓掌支持我。干杯！