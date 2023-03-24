# 如何使用 Python 获取 SalesForce 数据

> 原文：<https://towardsdatascience.com/using-python-to-get-salesforce-data-97bb5a7ef2cf?source=collection_archive---------3----------------------->

![](img/33d0f0dbf3935f9f842d47edf666960e.png)

Photo by [Denys Nevozhai](https://unsplash.com/@dnevozhai?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

我在一家大量使用 SalesForce 的初创公司工作。

当我第一次开始时，我们必须通过 Salesforce 网站登录。转到“报告”选项卡，创建一个包含必要字段的报告。下载逗号分隔值电子表格。在这里和那里做一些数据清理。主要是过滤字段，看看有没有空值或细微差别。再次导出 CSV，并在 Excel 中做一个 Vlookup，看看我们有什么数据。

我心想，一定有一种更简单的方法，我只需运行一个 Python 脚本就可以完成这项工作。做了一些谷歌搜索，发现了`[simple_salesforce](https://github.com/simple-salesforce/simple-salesforce/blob/master/README.rst)`。

> *来自他们的网站* — Simple Salesforce 是一个为 Python 2.6、2.7、3.3、3.4、3.5 和 3.6 构建的基本 Salesforce.com REST API 客户端。目标是为 REST 资源和 APEX API 提供一个非常底层的接口，返回 API JSON 响应的字典。

**使用 Python 拉 Salesforce 数据需要的东西:
1。Python 模块*simple _ sales force
2。具有 API 访问权限的 SalesForce 凭据***

假设你已经掌握了 Python 的基本知识，那么继续在你的机器上安装`simple_salesforce` 。

```
pip install simple_salesforce
```

完成后，我们可以继续创建我们的 Python 文件，并进行必要的导入。我们还需要我们的 SalesForce 凭据。如果不能联系您的 SalesForce 开发人员，帐户必须具有 API 访问权限。

```
from simple_salesforce import Salesforcesf = Salesforce(
username='myemail@example.com', 
password='password', 
security_token='token')
```

如果您还没有 SalesForce 安全令牌，请登录到 SalesForce 网站。导航到设置。然后到我的个人信息，在那个下拉菜单下应该重置我的安全令牌。这将以带有字母数字代码的电子邮件形式发送给您。

**使用 SOQL 查询 SalesForce 数据**

我们现在可以使用 Python 登录到 SalesForce。为了查询数据，`simple_salesforce`有一个叫做`query_all`的方法，这使得获取数据变得非常容易。SalesForce 有自己的方式来编写查询，称为 SalesForce 对象查询语言。

以下是使用 Python 和自定义字段进行查询的示例:

```
"SELECT Owner.Name, store_id__c, account_number__c, username__c, password__c, program_status__c, FROM Account WHERE program_status__c IN ('Live','Test')"
```

我们现在可以将这个 SOQL 代码插入到方法中，并将其提取到一个变量中:

```
sf_data = sf.query_all("SELECT Owner.Name, store_id__c, account_number__c, username__c, password__c, program_status__c, FROM Account WHERE program_status__c IN ('Live','Test')")
```

输出将是 JSON 格式，但是我们可以使用`[pandas](https://pandas.pydata.org/pandas-docs/stable/index.html)`很容易地将其转换成数据帧。JSON 返回了一些我认为不必要的属性，所以我放弃了它们。

```
sf_df = pd.DataFrame(sf_data['records']).drop(columns='attributes')
```

我们现在有了一个数据框架，可以进行数据分析和数据清理。

访问我的代码[这里](https://www.patreon.com/melvfnz)！

我在[这里也有家教和职业指导](https://square.site/book/8M6SR06V2RQRB/mentor-melv)！

如果你们有任何问题、评论或担忧，别忘了在 [LinkedIn](https://www.linkedin.com/in/melvfernandez/) 上联系我！