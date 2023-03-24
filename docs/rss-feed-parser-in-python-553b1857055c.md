# 清除编程面试任务:Python 中的 RSS 提要解析器

> 原文：<https://towardsdatascience.com/rss-feed-parser-in-python-553b1857055c?source=collection_archive---------17----------------------->

## *如何使用 Python 完成数据科学和数据工程面试编程任务*

最近一直在面试数据科学家的职位。其中一家公司给了我一个任务，用 python 创建一个 RSS 提要解析器。它以增量方式获取提要条目，并将它们存储在数据库中。
完整代码可以在[***github***](https://github.com/vintageplayer/RSS-Parser)***上找到。***

# **问题阐述**

因此，与其深入技术细节，我想回顾一下问题的背景。
对于那些不知道的人来说，RSS 是一种基于网络的内容(或提要)共享格式。([https://en.wikipedia.org/wiki/RSS](https://en.wikipedia.org/wiki/RSS))。以下是手头问题的细节:

1.  要解析的 RSS 提要是 [**印**](https://audioboom.com/channels/4930693.rss)**(**[https://audioboom.com/channels/4930693.rss](https://audioboom.com/channels/4930693.rss))
2.  需要设计一个合适的数据模型来表示每个帖子
3.  任何支持 python 连接的关系数据库都可以用来存储 post 数据
4.  禁止使用能够进行 RSS 解析的库(如 feedparser)
5.  负载应该是递增的，即只有先前没有处理的记录应该被处理
6.  应该安排脚本每天运行一次以获取更新

# 方法和系统设计

显然我决定用 ***python 3.7*** ，用生命终结支持来换 ***python 2.7*** 。以上基本上意味着我必须自己实现 RSS 解析器。考虑到 1 天的时间限制，所有其他决定都非常简单，为这项工作留出 3-4 个小时(假设每天工作 8-9 个小时)。下面是我最终使用的方法:

> **1。**RSS 解析器库的限制，基本上指望我自己写解析器。由于它最终是基于 xml 的内容，我决定使用一直可靠的 [**BeautifulSoup 库**](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)**(**[【https://www.crummy.com/software/BeautifulSoup/bs4/doc/】](https://www.crummy.com/software/BeautifulSoup/bs4/doc/)) **2。**我选择的关系数据库是[***postgres***](https://www.postgresql.org/)***(***[https://www.postgresql.org/](https://www.postgresql.org/)***)***。没有特别的原因，除了易用性和熟悉度，显然还有疯狂流行的开源支持。
> **3。**库 [***自动调度器***](https://apscheduler.readthedocs.io/en/latest/index.html)***(***【https://apscheduler.readthedocs.io/en/latest/index.html】***)***每天调度一次任务(简单的库开始， [***气流***](https://airflow.apache.org/)***(***[【https://airflow.apache.org/】](https://airflow.apache.org/)**库[此外，由于没有给出环境规范，我决定使用两个](https://pypi.org/project/psycopg2/)[***docker***](https://www.docker.com/)***(***[https://www.docker.com/](https://www.docker.com/)***)***容器来构建所有这些，一个用于*，一个用于***

*****有了上面分享的方法，让我们开始实施吧！*****

# *******设置事物*******

*****首先设置 docker 环境和容器:*****

*****1.为容器创建一个单独的网络进行交互:*****

```
***docker network create rss***
```

*****2.使用数据库名称、密码的环境变量创建 postgres 容器。公开端口，设置绑定挂载和工作目录:*****

```
***docker run -d --name rss-postgres \
 --net=rss \
 -e POSTGRES_DB=audioboom \
 -e POSTGRES_PASSWORD=parserssfeed \
 -p 5432:5432 \
 -v $(PWD):/home \
 -w /home  \
 postgres***
```

*****3.创建 python 容器，绑定挂载，设置工作目录并运行它:*****

```
***docker run -dt --name rss-python \
 --net=rss \
 -v $(PWD)/src:/home/src \
 -w /home/src  \
 conda/miniconda3-centos6 bash***
```

*****4.为 python 安装必要的库:*****

```
***docker exec rss-python conda update -c base -c defaults conda
docker exec rss-python conda install beautifulsoup4 lxml psycopg2
docker exec rss-python conda install -c conda-forge apscheduler***
```

*****让我们现在开始构建我们的数据模型…*****

# *****创建数据模型*****

*****看了一些帖子后，我意识到需要三个实体:*****

> *******1。** **帖子:**保存 feed 上发布的每个帖子的条目
> **2。itunes_data:** 一篇文章可以有选择地包含它的 itunes 列表的链接。媒体:一篇文章可以呈现 0 个或多个媒体对象*****

*****跟踪这三个实体几乎包含了整篇文章。
请使用此 [***github 链接***](https://github.com/vintageplayer/RSS-Parser/blob/master/create_db.sql) 进行精确的创建表查询(因为它们非常简单)
可以通过运行以下命令一次性创建表:*****

```
***docker exec -it rss-postgres psql -U postgres -d audioboom -f create_db.sql***
```

# *****编码开始…*****

*******助手模块** 除了主脚本之外，还创建了三个脚本来抽象一些带有函数调用的底层功能:*****

> *******1。**[**content _ fetcher . py**](https://github.com/vintageplayer/RSS-Parser/blob/master/src/content_fetcher.py)**:**用于半健壮地处理网页请求，并返回其内容
> **2。**[**data _ parser . py**](https://github.com/vintageplayer/RSS-Parser/blob/master/src/data_parser.py)**:**将网页内容转换为 BeautifulSoup 对象进行解析，解析 feed 中给定的 RSS post 记录，并返回相同的字典。
> **3。**[**DB _ connect . py**](https://github.com/vintageplayer/RSS-Parser/blob/master/src/db_connect.py)**:**包含 DB helper 函数，用于获取连接、获取已有记录的计数(用于增量加载)以及执行给定的查询。*****

# *****Main.py！！！*****

*****最后，让我们构建将所有部分连接在一起的脚本…
**1。** **导入:**下面几行将导入所需的模块和对象*****

```
***from data_parser import get_soup, parse_record, store_tags
from db_connect import get_connection, get_max_records,execute_query
from apscheduler.schedulers.blocking import BlockingScheduler***
```

*******2。**我们还将定义一些全局变量(我知道这不是一个推荐的做法，但考虑到时间限制，这是我们都必须采取的一个折衷方案)*****

```
***# Query to find the max record processed so far
get_max_query = 'SELECT COALESCE(max(itunes_episode),0) FROM tasteofindia.posts;'# Query template to insert values in any table
query_string = 'INSERT INTO tasteofindia.{0} ({1}) VALUES ({2}{3});'# List of columns present in the table
col_list  = {
 'posts'  : [<check the github script for the actual names>]
 ,'itunes_data' : [<check the github script for the actual names>]
 ,'media'  : ['itunes_episode','url','type','duration','lang','medium']
}# Creating insert queries for all the tables from template
query_strings = {k: query_string.format(k , ','.join(col_list[k]),('%s,'*(len(col_list[k])-1) ),'%s' ) for k in col_list}***
```

> *****未显示 ***帖子& itunes_data*** 的列列表。同样请参考 [***github 链接***](https://github.com/vintageplayer/RSS-Parser/blob/master/src/main.py) 。放给***itunes _ episode***看，了解一下大意。*****

*******3。begin(***feed _ URL***，***db _ credential _ file***)**:获取凭据文件上的的连接，并开始解析来自 feed url 的 feed:*****

```
***def begin(feed_url,db_credential_file):
 try:
  connection = get_connection(db_credential_file)
  update_feed_data(feed_url,connection)
 except Exception as e:
  print('Error Received...')
  print(e)
 finally:
  print('Closing connection')
  connection.close()***
```

*****一个简单的功能，开始所有的骚动，让事情动起来。
**4。update _ feed _ data(***feed***，** *conn* **):** 请求给定 url 的 BeautifulSoup 对象并尝试处理其中的任何记录:*****

```
***def update_feed_data(feed,conn):
 content   = get_soup(feed)
 print(f"Processing Records for : {feed}")
 records   = content.find_all('item')
 process_records(records,conn)
 return***
```

*****同样，按照函数范式，它通过检索 BeautifulSoup 对象来完成工作，并传递内容以供进一步处理。*****

*******5。process_records(*******

```
***def process_records(content,conn):
  record_count = len(content)
  current_max  = get_max_records(conn,get_max_query)
  print('Current Max : ',current_max)
  records = {} if record_count == current_max:
    print("No new records found!!")
    return records print(f"Total Records Found: {record_count}. Currently present: {current_max}") # List comprehension on the result of map operation on records
  [persist_taste_of_india_record(conn,record) for record in map(parse_record, content[record_count-current_max-1::-1])] return records***
```

*****这是最大的功能。它检查是否找到新记录。如果是，它首先为每个记录调用***parse _ record()***，然后继续保存记录。*****

*******6。persist _ tastse _ of _ India _ record(***conn***，** *data* **):** 它尝试分别持久化帖子的每个组成部分(基于定义的实体)*****

```
***def persist_taste_of_india_record(conn,data):
  persist_record(conn,data,'posts')
  persist_record(conn,data['itunes'],'itunes_data')
  for media in data['media']:
    persist_record(conn,media,'media')
  conn.commit()
 return True***
```

*****conn.commit() 是必需的，否则数据库中的更改不是永久的，并且会在会话过期后丢失。*****

*******7。persist _ record(***conn***，** *data* **，***TB _ name***):**根据对象类型执行插入查询:*****

```
***def persist_record(conn,data,tb_name):
 query_param  = tuple(
                list(map(lambda k : data[k],col_list[tb_name]))) execute_query(conn,query_strings[tb_name],query_param)
 return***
```

********query_param*** 只是将列顺序中的值存储在一个元组中。
***execute _ query()***最后将数据插入数据库*****

*****8.**执行并调度它:**脚本通过调用 begin 函数并调度它每天执行一次来完成，如下所示:*****

```
***if __name__ == '__main__':
  feed_url  = 'https://audioboom.com/channels/4930693.rss'
  db_credentials = 'connection.json' print('Main Script Running...')
  begin(feed_url,db_credentials)
  scheduler = BlockingScheduler()
  scheduler.add_job(begin, 'interval',[feed_url,db_credentials], hours=24) try:
    scheduler.start()
  except Exception as e:
    print('Stopping Schedule!!') print('Main Script Exiting!!')***
```

*****我在这里使用了阻塞调度程序，因此 python 线程总是活跃的。如果您想要停止执行，那么 **try…catch** 块将干净地退出。现在，您只需使用以下命令执行主脚本，立即运行一次，并安排在每天同一时间运行一次:*****

```
***docker exec -d rss-python python main.py***
```

# *****瞧啊。*****

*****就是这样。您已经准备好一个 RSS 解析器，每天运行并更新数据库。*****

# *****提高*****

*****显然，许多升级和增强是可能。许多最佳实践没有被遵循。坚实的原则，后台调度程序，使用 docker 编写文件等可能是第一步。在构建这个系统的时候，我首先关注的是得到一个功能性的系统，并在重构和设计上花费最少的精力。
尽管如此，请留下您的评论，并随时通过 github 联系我。*****