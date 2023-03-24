# 生成包含感兴趣的主题/关键词的公共“tweets”语料库

> 原文：<https://towardsdatascience.com/generate-a-corpus-of-public-tweets-containing-a-subject-keyword-of-interest-da86bc90259f?source=collection_archive---------22----------------------->

![](img/ed0c8dc6a8a6d79742eeb498c5f1bdea.png)

通常很难对人们的兴趣进行研究，民意测验和调查结果经常是有偏差的，或者有无意中加载的问题。因此，这意味着理解人们兴趣的最真实的 T2 方法在于他们想要分享或谈论什么，而不受任何外部影响。最简单的方法是在网上搜索包含某些特征的 twitter 帖子。

## Twitter API

最初，最直观的方法是查看 twitter API。要使用它，你首先必须在 https://developer.twitter.com/en/apply-for-access 注册一个开发者账户(免费)

这里为您分配了一组凭证，您可以将它们保存为以下格式的 JSON 文件:

```
{
 “twitter”: {
 “auth”: {
     “consumer_key”: “xxxx”,
     “consumer_secret”: “xxx”,
     “access_token_key”: “xxx”,
     “access_token_secret”: “xxx”
 },
  “token”: “xxx”,
  “secret”: “xxx”
 }
}
```

接下来，我们需要安装 Twython 库`pip install twython`(不幸的是 python 3)，之后我们可以根据需要定制下面的脚本。

```
# Import the Twython class
from twython import Twython
import json# Load credentials from json file
with open("twitter_credentials.json", "r") as file:
    creds = json.load(file)['twitter']['auth']# Instantiate an object
python_tweets = Twython(creds['consumer_key'], creds['consumer_secret'])
# Create our query
query = {'q': 'air quality',
        'result_type': 'popular',  # other options 'mixed'
        'count': 100,   # max 100
         # 'until':"2019-02-01",
        }import pandas as pd
# Search tweets
dict_ = {'user': [], 'date': [], 'text': [], 'favorite_count': []}
for status in python_tweets.search(**query)['statuses']:
    dict_['user'].append(status['user']['screen_name'])
    dict_['date'].append(status['created_at'])
    dict_['text'].append(status['text'])
    dict_['favorite_count'].append(status['favorite_count'])# Structure data in a pandas DataFrame for easier manipulation
df = pd.DataFrame(dict_)
```

这种方法的缺点是结果仅限于过去 7 天，并且最多只能返回 100 条推文。

## 访问两个日期之间的所有推文

相反，如果我们希望获得更久以前日期之间的所有推文，我们需要依靠其他方法。要做到这一点，我们可以依赖 GetTweetsFrom 库；[https://github.com/Jefferson-Henrique/GetOldTweets-python](https://github.com/Jefferson-Henrique/GetOldTweets-python)

这是在 **python 2** 上设计的，尽管有一个 python 3 的实验接口，它可能工作也可能不工作。

因为我希望只提取每条 tweet 的文本，并将其保存在新的一行上进行文本处理(word2vec)，所以对库提供的原始示例代码进行了轻微的调整。这里，我对查询、开始和结束日期以及输出文件名进行了硬编码。

下面的脚本遍历两个日期之间的每一天，并下载它找到的与查询匹配的前 1000 条 tweets。这可以适应用户的需求。

```
# -*- coding: utf-8 -*-
import sys,getopt,datetime,codecs
from datetime import date, timedeltaedate = date(2019, 10, 1)   # start date
sdate = date(2018, 10, 1)   # end datequery = 'atmospheric chemistry'
maxtweets = 1000
toptweet=False
outputFileName = 'myoutput.csv'delta = edate - sdate       # as timedelta
dates = [(sdate + timedelta(days=i)) for i in range(delta.days + 1)]if sys.version_info[0] < 3:
    import got
else:
    import got3 as gotdef main():try:tweetCriteria = got.manager.TweetCriteria()tweetCriteria.querySearch = query
        tweetCriteria.topTweets = toptweet
        tweetCriteria.maxTweets = maxtweetsoutputFile = codecs.open(outputFileName, "w+", "utf-8")def receiveBuffer(tweets):
            for t in tweets:
                outputFile.write('\n%s'% t.text)
            outputFile.flush()for dt in dates:
            print (dt)
            tweetCriteria.since = str(dt)
            tweetCriteria.until = str(dt + timedelta(days=1))got.manager.TweetManager.getTweets(tweetCriteria, receiveBuffer)except arg:
        print('Arguments parser error, try -h' + arg)
   finally:
        outputFile.close()
        print('Done. Output file generated "%s".' % outputFileName)if __name__ == '__main__':
   main()
```

## 结论

*   Twitter API 仅用于收集过去 7 天的推文。
*   如果您需要历史数据，GetOldTweets 库非常有用
*   然后，生成的输出文件可以用作基于 NLP 的机器学习和预测的语料库，或者关于该主题的公众观点的样本。
*   这两种方法都非常容易使用，而且几乎是即插即用的(一旦您使用了正确版本的 python)。