# 数据科学家的工具包—如何从不同来源收集数据

> 原文：<https://towardsdatascience.com/data-scientists-toolkit-how-to-gather-data-from-different-sources-1b92067556b3?source=collection_archive---------17----------------------->

## Master all — csv、tsv、zip、txt、api、json、sql …

![](img/2f403c08d875f4a6d013d051f86a23cb.png)

Photo by [Jakob Owens](https://unsplash.com/@jakobowens1?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

不久前！

您还记得将外部硬盘中的数据发送给您进行分析或建模的时间吗？

现在，作为一名数据科学家，你不局限于这些方法。存储数据、共享数据以及获取数据、扩充数据的不同来源有多种方式。

下面，我列出了几种收集数据的方法供你分析

## 目录:

1.  CSV 文件
2.  平面文件(制表符、空格或任何其他分隔符)
3.  文本文件(在单个文件中—一次读取所有数据)
4.  压缩文件
5.  多个文本文件(数据被分割到多个文本文件中)
6.  从互联网下载文件(服务器上托管的文件)
7.  网页(抓取)
8.  API(JSON)
9.  文本文件(逐行读取数据)
10.  关系数据库管理系统(SQL 表)

在 Python 中，文件被描述为文本**或二进制**文件**，两者之间的区别很重要**

**文本文件**由一系列行组成。每一行都以一个称为 EOL 或行尾字符的特殊字符结束。有几种类型，但最常见的是`\n`或`,`

**二进制文件**类型基本上是除文本文件之外的任何类型的文件。由于其性质，二进制文件只能由知道或理解文件结构的应用程序来处理

> ***1。CSV 文件***

存储和共享数据集最常见的格式是逗号分隔格式或 csv 文件。`pandas.read_csv()`是最有用和最强大的方法，我强烈推荐你阅读它的[文档](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.read_csv.html)。通过使用适当类型的`sep`,您可以在 dataframe 中加载多种类型的数据

```
import pandasdf = pd.read_csv('data.csv', sep =',')
```

> **2。平锉平锉**

但有时您可能会收到制表符分隔、固定宽度格式或分号分隔等文件。`Pandas`提供多种方法来恰当地读取此类数据。但是，当您指定正确的分隔符时，甚至`read_csv`也能很好地工作

```
import pandas# Tab separated file
df = pd.read_csv('data.tsv', sep='\t')
OR 
# columns are separated by space 
df = pd.read_csv('data.txt', sep = ' ')
```

> ***3。文本文件***

我们看到使用`pandas.read_csv()`方法可以读取 txt 文件，但是让我们看看读取 txt 文件的另一种方式是**上下文管理器**。它们在分配和管理资源、关闭所有打开的文件方面非常有效

上下文管理器最广泛使用的例子是`with`语句

```
file_name = 'data.txt'with open(file_name, mode = 'r') as file:
    df = file
```

> ***4。Zip 文件***

有时，您可能会得到一个 csv 文件，其中可能包含您需要的 csv 文件，以节省大小。提取。你需要使用' zip file '库。如果你使用的是 windows，这样做比从 windows 中提取要好得多

`zipfile`也是一个上下文管理器，将支持我们前面看到的`with`语句

```
from zipfile import ZipFilefile_name = 'my_zip_file.zip'
with open(file_name, mode = 'r') as f:
    f.extractall()
# Extractall method will extract all the contents of zipfile in the same folder# now we can load the extracted csv file in our dataframe
import pandasdf = pd.read_csv('my_csv_file.csv')
```

> ***5。多个文本文件***

您可能会遇到这样的情况，您的数据以多个文本文件的形式提供给您。例如，你被提供了 1000 个电影标题的评论，而不是把所有的放入一个，他们把每个单独的评论作为一个文件，并且提供了 1000 个这样的文本文件

我们将使用一个库`glob`——它使得打开具有相似路径结构的文件变得简单

```
import pandas
import globfolder_name = 'my_folder'df_list = []
for review in glob.glob('my_folder/*.txt'):
    with open(review, mode = 'r') as file:
        movie = {}
        movie['title'] = file.readline()
        movie['review'] = file.read()
df_list.append(movie)df = pd.DataFrame(df_list)
```

*   是 glob 语句中的通配符。它允许 python 扫描所有的。给定路径中的 txt 文件

> ***6。从网上下载文件***

如果你必须下载保存在服务器上的文件。你必须使用一个库— `requests`

```
import requests
import ossample_url = '192.145.232.xx/2019/2/class_notes/gather_data.txt'folder_name = 'my_folder'if not os.path.exists(folder_name):
    os.makedirs(folder_name)response = requests.get(url)
file_name = 'gather_data.txt'
file_loc = os.path.join(folder_name, file_name)
with open(file_loc, mode='wb') as outfile:
    outfile.write(response.content)
```

> ***7。*网页(网页抓取)**

网络抓取是一种使用代码从网页中提取数据的奇特方式。存储在网页上的数据被称为 HTML，即超文本标记语言。它是由这些叫做标签的东西组成的，这些东西赋予了网页结构。`<title>,` `<div>,` `<h1>` …..等等

因为 HTML 代码只是文本，所以可以使用解析器提取其中的标签和内容。为此，我们将使用一个库— `BeautifulSoup`

假设我们想从 IMDb 和烂番茄两者中提取关于《复仇者联盟》最终结局(2019)的信息

注意:Inspect element 是您寻找相关标签和属性以提取数据的最好朋友

```
from bs4 import BeautifulSoup as bs
import requests
import pandasimdb_url = '[https://www.imdb.com/title/tt4154796/](https://www.imdb.com/title/tt4154796/)'
response = requests.get(imdb_url)
soup = bs(response.content, features = b'lxml)
movie = {}
movie['title'] = soup.find('div', attrs = {'class': 'title_wrapper'}).find('h1').text
movie['imdb_rating'] = soup.find('span', attrs = {'itemprop': 'ratingValue'}).textrt_url = '[https://www.rottentomatoes.com/m/avengers_endgame](https://www.rottentomatoes.com/m/avengers_endgame)'
response = requests.get(rt_url)
soup = bs(response.content, features = 'lxml')
movie['tomatometer'] = soup.find('span', attrs = {'class' : 'mop-ratings-wrap__percentage'}).text
movie['audience_score'] = soup.find('div', attrs = {'class' : 'audience-score'}).find('span', attrs={'class' : 'mop-ratings-wrap__percentage'}).textdf = pd.DataFrame(movie)
```

通过这种方式，我们从两个不同的来源收集了关于《复仇者联盟》结局的信息

> ***8。API***(应用程序编程接口)

你当然可以从网页中提取信息，但是更好的获取信息的方式是通过 API。

维基百科有几个公开开放的 API，其中流行的一个是 Mediawiki。我们将使用 python 库`wptools`

```
import wptools
import pandas# For a wikipedia URL '[https://en.wikipedia.org/wiki/Avengers:_Endgame](https://en.wikipedia.org/wiki/Avengers:_Endgame)' we only need to pass the string after /wiki/wiki_page = wptools.page('[Avengers:_Endgame](https://en.wikipedia.org/wiki/Avengers:_Endgame)).get()# Now this wiki_page has fetched extracts, images, infobox data, wiki data etc
# By using wikipage.data() method we can extract all the informationwiki_page.data['image'] # this will return 3 images backfirst_image = wiki_page.data['image'][0]
print (first_image['url'])
'https://upload.wikimedia.org/wikipedia/en/0/0d/Avengers_Endgame_poster.jpg'# Now you can save this poster link or use the requests method as seen above to save poster of the movie
```

> ***9。文本文件*** (逐行读取数据)

假设您已经使用 twitter api — tweepy 下载了 tweepy 数据

你已经将所有的推文保存在一个文本文件中——tweets _ JSON . txt

你现在的目标是从这些推文中提取有用的信息，将其保存在数据框架中，并用于进一步的分析。这是您在第 3 步和第 8 步中学到的技能的组合

第 1 部分—让我向您展示如何使用 tweepy 获取 twitter 数据。

```
import tweepy
import jsonconsumer_key = 'xxx'
consumer_secret = 'xxx'
access_token = 'xxx'
access_secret = 'xxx'auth = OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_secret)api = tweepy.API(auth, wait_on_rate_limit = True)tweet_ids = [] #This will be the list of tweets for which you need the information forfile_name = 'tweets_json.txt'
with open(file_name, mode='w') as file:
    for tweet_id in tweet_id:
        tweet = api.get_status(tweet_id, tweet_mode = 'extended')
        json.dump(tweet._json, file)
        file.write('/n')
```

第 2 部分—阅读 tweets_json.txt 以提取有用的信息

```
import pandas
import jsondf_list = []
with open('tweets_json.txt', mode = 'r') as file:
    for line in file:
        tweet = {}
        json_data = json.loads(line)
        tweet['tweet_id'] = json_data['id']
        tweet['retweet_count'] = json_data['retweet_count']
        tweet['favorite_count'] = json_data['favorite_count']
        df_list.append(tweet)
tweet_df = pd.DataFrame(df_list)
```

> ***10。RDBMS*T3【SQL 数据库】**

数据库是“保存在计算机中的一组结构化数据，尤其是可以通过各种方式访问的数据。”它便于数据的存储、检索、修改和删除

在数据争论的背景下，数据库和 SQL 开始用于存储数据或收集数据:

*   **连接数据库，将数据**导入熊猫数据框架
*   **连接数据库并将数据从熊猫数据帧存储到数据库**

1.  用 python 连接到数据库—

我们将使用 SQLAlchemy 连接到 SQLite 数据库，这是一个用于 python 的数据库工具包

```
import sqlalchemy
import pandasengine = sqlalchemy.create_engine('sqlite:///movies.db')
```

*movies.db 不会显示在 jupyter 笔记本仪表盘上*

2.在数据库中存储熊猫数据帧

```
# Store dataframe 'df' in a table called movie_tbl in the databasedf.to_sql('movie_tbl', engine, index=False)
```

*movies.db 现在将显示在 jupyter 笔记本仪表盘中*

3.读取熊猫数据框架中的数据库表

```
df = pd.read_sql('SELECT * FROM movie_tbl', engine)
```

## 结束语:

*   我试图给出各种事物的基本概念，避免写下太多的细节
*   请随意写下您的想法/建议/反馈