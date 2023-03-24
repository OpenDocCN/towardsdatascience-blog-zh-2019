# 使用 Python 将您的熊猫数据框推送到 Google Sheets

> 原文：<https://towardsdatascience.com/using-python-to-push-your-pandas-dataframe-to-google-sheets-de69422508f?source=collection_archive---------4----------------------->

![](img/3587f704e5f6e879868b4fc12cda2f32.png)

Photo by [Christian Kielberg](https://unsplash.com/@chilis?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

这也可以作为像使用数据库一样使用 Google Sheets 的简单方法。

假设您已经安装了最新版本的 Python。为此需要的 Python 模块有:

1.  `pandas`(获取和读取数据)
2.  `gspread`(连接到谷歌工作表)
3.  `df2gspread`(与谷歌工作表互动)

仔细安装这些模块后，我们现在可以创建一个 Python 文件，并开始导入。

```
import pandas as pd
import gspread
import df2gspread as d2g
```

现在我们需要任何类型的数据，我们可以从 CSV 或其他来源获取。利用熊猫，我们可以把它组织成一个数据框架。任何类型的数据框架都可以。如果你还没有，让我们用熊猫做一个。

```
d = {'col1': [1, 2], 'col2': [3, 4]}
df = pd.DataFrame(data=d)
```

现在棘手的部分，获得凭证访问。为了使用 Python，我们需要从 Google 获得某种形式的 ***密钥*** 。这个 ***键*** 主要是出于安全目的，将采用 JSON 文件的格式。

让我们登录我们希望使用的谷歌账户，并按照这里的说明[进行操作](https://gspread.readthedocs.io/en/latest/oauth2.html)。如果一切顺利，JSON 文件应该看起来像这样:

```
{
    "private_key_id": "2cd … ba4",
    "private_key": "-----BEGIN PRIVATE KEY-----**\n**NrDyLw … jINQh/9**\n**-----END PRIVATE KEY-----**\n**",
    "client_email": "473000000000-yoursisdifferent@developer.gserviceaccount.com",
    "client_id": "473 … hd.apps.googleusercontent.com",
    "type": "service_account"
}
```

这个 JSON 文件需要与 Python 脚本在同一个文件夹中。对于这个例子，我将其命名为`jsonFileFromGoogle.json`。我们可以在代码中调用 JSON 文件，并将其设置为连接的凭证。我们还可以设置*范围，*我们希望凭证被发送到哪里以便进行连接。

```
scope = ['[https://spreadsheets.google.com/feeds'](https://spreadsheets.google.com/feeds'),
         '[https://www.googleapis.com/auth/drive'](https://www.googleapis.com/auth/drive')]credentials = ServiceAccountCredentials.from_json_keyfile_name(
    'jsonFileFromGoogle.json', scope)gc = gspread.authorize(credentials)
```

您现在应该能够以这种方式连接到 Google。是时候把数据帧发送到谷歌电子表格了。对于我的目的，我发现创建一个新的工作表并获取电子表格键更容易。电子表格密钥可以在这里用红色标记的 URL 中找到。

![](img/91cf23c3e1dc44ef7ea92b7523bac911.png)

复制它，并将其设置为一个名为`spreadsheet_key`的变量。如果您的工作表有多个工作表，也要设置工作表名称。我在我的代码里设置为`wks_name`，默认情况下会叫“Master”。

```
spreadsheet_key = 'red_url_code_goes_here'
wks_name = 'Master'
d2g.upload(df, spreadsheet_key, wks_name, credentials=credentials, row_names=True)
```

现在，当您最终运行这个脚本时，您设置的 DataFrame 将把它上传到 Google Sheets。这有许多其他的可能性，但是你可以用它作为一个更新的小数据库，你可以和你的同伴分享。

我在这里也有家教和职业指导！

如果你喜欢这些内容，请随时在 [Patreon](https://www.patreon.com/join/melvfnz) 上支持我！

如果你们有任何问题、意见或担忧，请不要忘记通过 LinkedIn 与我联系！