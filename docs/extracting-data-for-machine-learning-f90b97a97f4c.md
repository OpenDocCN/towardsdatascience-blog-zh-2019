# 为机器学习提取数据

> 原文：<https://towardsdatascience.com/extracting-data-for-machine-learning-f90b97a97f4c?source=collection_archive---------13----------------------->

## 为下一个机器学习项目获取数据的三种方法

![](img/63f6e707dcc00c4a47c00f11e0101fac.png)

Photo by [Matthieu Oger](https://unsplash.com/@matthieuoger?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/crete?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

*任何机器学习项目最重要的第一步就是获得优质数据。作为一名数据科学家，您经常需要使用各种不同的方法来提取数据集。您可能使用公开可用的数据、通过 API 可用的数据、在数据库中找到的数据，或者在许多情况下这些方法的组合。*

在下面的帖子中，我将简要介绍 python 中提取数据的三种不同方法。在这篇文章中，我将讲述如何在 Jupyter 笔记本中提取数据。我以前在早先的[文章](/five-command-line-tools-for-data-science-29f04e5b9c16)中报道过如何从命令行使用这些方法。

## 结构化查询语言

如果您需要从关系数据库中获取数据，很可能需要使用 SQL。您可以使用名为 SQLAlchemy 的库将 Jupyter 笔记本连接到最常见的数据库类型。这个[链接](https://docs.sqlalchemy.org/en/13/core/engines.html#postgresql)提供了支持哪些数据库以及如何连接到每种类型的描述。

您可以直接使用 SQLAlchemy 来查看和查询表，也可以编写原始查询。要连接到您的数据库，您需要一个包含您的凭据的 URL。然后，您可以使用`create_engine`命令来创建连接。

```
from sqlalchemy import create_engine
engine = create_engine('dialect+driver://username:password@host:port/database')
```

现在，您可以编写数据库查询并返回结果。

```
connection = engine.connect()
result = connection.execute("select * from my_table")
```

## 擦

Web 抓取用于从网站下载数据，并从这些页面中提取所需的信息。有很多 python 库可以用来做这件事，但是最简单的一个是 [Beautiful Soup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) 。

你可以通过 pip 安装这个软件包。

```
pip install BeautifulSoup4
```

让我们通过一个简单的例子来说明如何使用它。我们将使用 Beautiful Soup 和 urllib 库从猫途鹰网站获取酒店名称和价格。

让我们导入将要使用的所有库。

```
from bs4 import BeautifulSoup
import urllib.request
```

接下来，我们要下载我们要抓取的页面内容。我将收集希腊克里特岛酒店的价格，所以我使用一个包含该目的地酒店列表的 URL。

![](img/4d94b0537accbc492f90889be3f2b9dc.png)

[TripAdvisor](https://www.tripadvisor.co.uk/Hotels-g189413-Crete-Hotels.html)

下面的代码将 URL 定义为一个变量，使用 urllib 库打开页面，使用 Beautiful Soup 读取页面，并以易于阅读的格式返回结果。代码下面显示了部分输出。

```
URL = '[https://www.tripadvisor.co.uk/Hotels-g189413-Crete-Hotels.html'](https://www.tripadvisor.co.uk/Hotels-g189413-Crete-Hotels.html')
page = urllib.request.urlopen(URL)
soup = BeautifulSoup(page, 'html.parser')
print(soup.prettify())
```

![](img/b9aa19cab33f2930c588532d748b8bfb.png)

接下来，让我们获取页面上的酒店名称列表。我们将使用`find_all`函数，它允许您提取文档中您感兴趣的部分。您可以使用`find_all`以多种方式过滤文档。通过传入字符串、正则表达式或列表。您还可以过滤标签的一个属性，这就是我们将在这里使用的方法。如果你不熟悉 HTML 标签和属性，这篇文章[给出了一个很好的概述。](https://en.wikipedia.org/wiki/HTML_attribute)

要了解如何最好地访问数据点，您需要检查 web 页面上该元素的代码。要查找酒店名称的代码，我们右键单击列表中的名称，如下图所示。

![](img/f4644a8fd66dd5583718b20786cbc90d.png)

当您点击`inspect`时，代码将出现，包含酒店名称的部分将高亮显示，如下所示。

![](img/d8cf8ee557cb68ad57777f7d654923d1.png)

我们可以看到酒店名称是名为`listing_title`的类中唯一的一段文本。下面的代码将这个属性的类和名称传递给`find_all`函数，以及`div`标签。

```
content_name = soup.find_all('div', attrs={'class': 'listing_title'})
print(content_name)
```

这会以列表形式返回包含酒店名称的每一段代码。

![](img/0e597dfc2bedfd0085aab27c028f12c6.png)

为了从代码中提取酒店名称，我们可以使用 Beautiful Soup 的`getText`函数。

```
content_name_list = []
for div in content_name:
    content_name_list.append(div.getText().split('\n')[0])
print(content_name_list)
```

这会以列表形式返回酒店名称。

![](img/c0373097c9bd90dd95ab4487aebbdd41.png)

我们可以用类似的方法得到价格。检查价格代码，我们可以看到它有以下结构。

![](img/e34c80dc4d88329d1fa72d6ddd175893.png)

所以我们可以用非常相似的代码提取这一段。

```
content_price = soup.find_all('div', attrs={'class': 'price-wrap'})
print(content_price)
```

价格有点复杂，如果我们运行下面的代码，我们会看到它。

```
content_price_list = []
for div in content_price:
    content_price_list.append(div.getText().split('\n')[0])
print(content_price_list)
```

输出如下所示。当酒店列表显示降价时，除了一些文本之外，还会返回原始价格和销售价格。为了使这个有用，我们只想返回酒店的实际价格，如果我们今天预订的话。

![](img/d839343499dca1c0d94047d0b623a784.png)

我们可以使用一些简单的逻辑来获得文本中显示的最后价格。

```
content_price_list = []
for a in content_price:
        a_split = a.getText().split('\n')[0]
        if len(a_split) > 5:
            content_price_list.append(a_split[-4:])
        else:
            content_price_list.append(a_split)  

print(content_price_list)
```

这给出了以下输出。

![](img/48e911cae24c52cb5dcb076258dbdb55.png)

## 应用程序接口

API 代表应用程序编程接口，就数据提取而言，它是一个基于 web 的系统，为数据提供一个端点，您可以通过一些编程连接到该端点。通常，数据将以 JSON 或 XML 格式返回。

在机器学习中，你可能需要用这种方法获取数据。我将给出一个简单的例子，说明如何从一个公开可用的名为 [Dark Sky](https://darksky.net/dev) 的 API 中获取天气数据。要访问这个 API，你需要注册，每天免费提供 1000 个电话，这应该足够尝试了。

为了访问来自黑暗天空的数据，我将使用`requests`库。首先，我需要获得正确的 URL 来请求数据。“黑暗天空”提供预报和历史天气数据。对于这个例子，我将使用历史数据，我可以从[文档](https://darksky.net/dev/docs#time-machine-request)中获得正确的 URL。

该 URL 具有以下结构。

```
https://api.darksky.net/forecast/[key]/[latitude],[longitude],[time]
```

我们将使用`requests`库来获取特定纬度和经度以及日期和时间的结果。让我们想象一下，在获得克里特岛酒店的每日价格后，我们想要找出价格是否以某种方式与天气相关。例如，让我们为列表中的一家酒店选择坐标。

![](img/da49b03a304818b423570187d0baed79.png)

[Google.com](https://www.google.com/search?q=mitsis+laguna+resort+%26+spa+latitude+and+longitude&oq=mitsis+laguna+resort+%26+spa+latit&aqs=chrome.2.69i57j33l3.7096j0j7&sourceid=chrome&ie=UTF-8)

首先，我们用我们需要的正确坐标和日期时间构造 URL。使用`requests`库，我们可以访问 JSON 格式的数据。

```
import requestsrequest_url = '[https://api.darksky.net/forecast/fd82a22de40c6dca7d1ae392ad83eeb3/35.3378,-25.3741,2019-07-01T12:00:00'](https://api.darksky.net/forecast/fd82a22de40c6dca7d1ae392ad83eeb3/35.3378,-25.3741,2019-07-01T12:00:00')
result = requests.get(request_url).json()
result
```

我们可以将结果标准化到一个数据框架中，以便于阅读和分析。

```
import pandas as pddf = pd.DataFrame.from_dict(json_normalize(result), orient='columns')
df.head()
```

![](img/b10c8fe7e147d785f9e8d30267a0b13a.png)

Part of the resulting data frame

使用这些方法，您可以做更多的事情来自动提取这些数据。对于 web 抓取和 API 方法，可以编写函数来自动执行该过程，以便轻松提取大量日期和/或位置的数据。在这篇文章中，我想简单地给出一个概述，用足够的代码来探索这些方法。在以后的文章中，我将会写一些更深入的文章，介绍如何构建完整的数据集并使用这些方法进行分析。

感谢阅读！