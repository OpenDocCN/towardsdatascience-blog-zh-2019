# 如何用 Python 管理 Google Drive 中的文件

> 原文：<https://towardsdatascience.com/how-to-manage-files-in-google-drive-with-python-d26471d91ecd?source=collection_archive---------3----------------------->

![](img/e5293d191f877b975b4d3bcca9f9dfc5.png)

Photo on [unsplash](https://unsplash.com/photos/RIb4BDwiakQ)

作为一名数据分析师，大多数时候我需要与我的产品经理/利益相关者分享我提取的数据，Google Drive 始终是我的首选。这里的一个主要问题是我必须每周甚至每天都做，这很无聊。我们所有人都讨厌重复的任务，包括我。

幸运的是，谷歌为其大部分服务提供了 API。我们将使用 [**Google Drive API**](https://developers.google.com/drive/) 和 [**PyDrive**](https://pythonhosted.org/PyDrive/) 来管理我们在 Google Drive 中的文件。

# 使用 Google Drive API

在开始编码之前，您应该准备好 Google Drive API 访问。我写了一篇关于如何通过客户端 ID 获得你的 [Google 服务访问的文章。你应该可以得到包含密钥的 JSON 文件来访问你的 Google Drive。](https://medium.com/@chingjunetao/simple-way-to-access-to-google-service-api-a22f4251bb52)

# PyDrive 入门

## 安装 PyDrive

我们将使用 python 包管理器来安装 PyDrive

```
pip install pydrive
```

## 正在连接到 Google Drive

PyDrive 只用 2 行代码就使认证变得非常容易。

> 你必须**将**JSON 文件重命名为“**client _ secrets . JSON”**，并将它与你的脚本放在同一个目录下。

将启动浏览器并要求您进行身份验证。选择您想要访问的 google 帐户并授权应用程序。

`drive = GoogleDrive(gauth)`创建一个 Google Drive 对象来处理文件。您将使用该对象来列出和创建文件。

## 在 Google Drive 中列出和上传文件

第 1 行到第 4 行将显示 Google Drive 中的文件/文件夹列表。它还会给你这些文件/文件夹的详细信息。我们获取您想要上传文件的文件夹的**文件 ID** 。在这种情况下，`To Share`是我上传文件的文件夹。

> **文件 ID** 很重要，因为 Google Drive 使用文件 ID 来指定位置，而不是使用文件路径。

`drive.CreateFile()`接受**元数据** ( *dict* )。)作为输入来初始化一个 GoogleDriveFile。我用`"mimeType" : "text/csv"`和`"id" : fileID`初始化了一个文件。这个`id`将指定文件上传到哪里。在这种情况下，文件将被上传到文件夹`To Share`。

`file1.SetContentFile("small_file.csv")`将打开指定的文件名并将文件的内容设置为 GoogleDriveFile 对象。此时，文件仍未上传。您将需要`file1.Upload()`来完成上传过程。

## 访问文件夹中的文件

如果你想上传文件到一个文件夹里的文件夹里呢？是的，你还需要文件 ID ！你可以使用`ListFile`来获取文件，但是这次把`root`改为`file ID`。

```
file_list = drive.ListFile({'q': "'<folder ID>' in parents and trashed=false"}).GetList()
```

现在我们可以进入文件夹`To Share`中的文件夹`picture`。

除了上传文件到 Google Drive，我们还可以删除它们。首先，用指定的文件 ID 创建一个 GoogleDriveFile。使用`Trash()`将文件移至垃圾箱。您也可以使用`Delete()`永久删除文件。

现在你已经学会了如何用 Python 管理你的 Google Drive 文件。希望这篇文章对你有用。如果我犯了任何错误或打字错误，请给我留言。

可以在我的 [**Github**](https://github.com/chingjunetao/google-service-with-python/tree/master/google-drive-with-python) 中查看完整的脚本。干杯！

**如果你喜欢读这篇文章，你可能也会喜欢这些:**

[](/how-to-rewrite-your-sql-queries-in-python-with-pandas-8d5b01ab8e31) [## 如何用熊猫用 Python 重写 SQL 查询

### 在 Python 中再现相同的 SQL 查询结果

towardsdatascience.com](/how-to-rewrite-your-sql-queries-in-python-with-pandas-8d5b01ab8e31) [](/how-to-master-python-command-line-arguments-5d5ad4bcf985) [## 如何掌握 Python 命令行参数

### 使用命令行参数创建自己的 Python 脚本的简单指南

towardsdatascience.com](/how-to-master-python-command-line-arguments-5d5ad4bcf985) 

**你可以在 Medium 上找到我其他作品的链接，关注我** [**这里**](https://medium.com/@chingjunetao) **。感谢阅读！**