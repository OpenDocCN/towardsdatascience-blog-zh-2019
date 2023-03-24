# 如何使用 Python 从 S3 的 pickle 文件加载数据

> 原文：<https://towardsdatascience.com/how-to-load-data-from-a-pickle-file-in-s3-using-python-ffe2866b7eba?source=collection_archive---------10----------------------->

我不知道你怎么想，但我喜欢尽可能高效地钻研我的数据。从 S3 提取不同的文件格式是我每次都要查找的事情，所以这里我展示了如何将数据从存储在 S3 的 pickle 文件加载到我本地的 Jupyter 笔记本上。

![](img/e6b9546cfc9ebd0f840759b134f6d41c.png)

This has got to be the ugliest picture I’ve ever used for one of my blogs. Thx Google Search and Print Screen!

## 一些注意事项:

*   永远不要硬编码您的凭证！如果你这样做了，一定不要把代码上传到仓库，尤其是 Github。有网络爬虫寻找意外上传的密钥，你的 AWS 帐户将被危及。相反，使用`boto3.Session().get_credentials()`
*   在 python 的老版本中(Python 3 之前)，你将使用一个名为`cPickle`的包而不是`pickle`，正如[这个 StackOverflow](https://stackoverflow.com/questions/49579282/cant-find-module-cpickle-using-python-3-5-and-anaconda) 所验证的。

维奥拉。而从那里看，`data`应该是一个熊猫的数据帧。

我发现消除数据帧中字段和列名的空白很有帮助。我不确定这是一个 pickle 文件，还是针对我的数据。如果这个问题影响到你，下面是我使用的:

要消除列名中的空白:

```
col_list = []
for col in list(data.columns):
    col_list.append(col.strip())data.columns = col_list
```

要消除数据中的空白:

```
data = data.applymap(lambda x: x.strip() if type(x)==str else x)
```

## 资源:

 [## 凭证- Boto 3 文档 1.9.123 文档

### boto3 中有两种类型的配置数据:凭证和非凭证。凭据包括以下项目……

boto3.amazonaws.com](https://boto3.amazonaws.com/v1/documentation/api/latest/guide/configuration.html)