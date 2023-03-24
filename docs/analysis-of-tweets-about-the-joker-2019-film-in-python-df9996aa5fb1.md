# Python 中关于小丑(2019 年电影)的推文分析

> 原文：<https://towardsdatascience.com/analysis-of-tweets-about-the-joker-2019-film-in-python-df9996aa5fb1?source=collection_archive---------28----------------------->

![](img/092e9cc1ddc1212753d668ea3faeee38.png)

Photo by [Pixabay](https://www.pexels.com/@pixabay) on [Pexels](https://www.pexels.com/photo/apps-blur-button-close-up-267350/)

在本帖中，我们将分析与*小丑(2019 年电影)*相关的推特推文。首先，你需要申请一个 Twitter 开发者账户:

![](img/8de478e493cd200d2ddb35292f06e27d.png)

[Source](https://projects.raspberrypi.org/en/projects/getting-started-with-the-twitter-api/3)

您的开发人员帐户获得批准后，您需要创建一个 Twitter 应用程序:

![](img/535749c04db1726dafff300aac41628c.png)

[Source](https://projects.raspberrypi.org/en/projects/getting-started-with-the-twitter-api/4)

申请 Twitter 开发者账户和创建 Twitter 应用的步骤在[这里](https://projects.raspberrypi.org/en/projects/getting-started-with-the-twitter-api/4)有所概述。

为了访问 Twitter API，我们将使用免费的 python 库 tweepy。tweepy 的文档可以在[这里](https://tweepy.readthedocs.io/en/latest/getting_started.html)找到。

1.  **安装**

首先，确保您已经安装了 tweepy。打开命令行并键入:

```
pip install tweepy 
```

2.**导入库**

接下来，打开您最喜欢的编辑器，导入 tweepy 和 pandas 库:

```
import tweepy
import pandas as pd**AUTHENTICATION**
```

接下来，我们需要我们的消费者密钥和访问令牌:

![](img/6e8d1ea132e104bb6c262ddad9d50495.png)

[Source](https://projects.raspberrypi.org/en/projects/getting-started-with-the-twitter-api/4)

请注意，该网站建议您保持您的密钥和令牌私有！这里我们定义了一个假的密钥和令牌，但是在创建 Twitter 应用程序时，您应该使用真正的密钥和令牌，如上所示:

```
consumer_key = '5GBi0dCerYpy2jJtkkU3UwqYtgJpRd' 
consumer_secret = 'Q88B4BDDAX0dCerYy2jJtkkU3UpwqY'
access_token = 'X0dCerYpwi0dCerYpwy2jJtkkU3U'
access_token_secret = 'kly2pwi0dCerYpjJtdCerYkkU3Um'
```

下一步是创建 OAuthHandler 实例。我们传递上面定义的消费者密钥和访问令牌:

```
auth = tweepy.OAuthHandler(consumer_key, consumer_secret)
auth.set_access_token(access_token, access_token_secret)
```

接下来，我们将 OAuthHandler 实例传递给 API 方法:

```
api = tweepy.API(auth)
```

3 **。TWITTER API 请求**

接下来，我们为我们有兴趣分析的字段初始化列表。现在，我们可以查看推文字符串、用户和推文时间。接下来，我们在一个 tweepy“Cursor”对象上编写一个 for 循环。在“Cursor”对象中，我们传递“api.search”方法，为我们想要搜索的内容设置查询字符串(q =“Joker”)，并设置“count”= 1000，这样我们就不会超过 twitter 的速率限制。我们还使用“item()”方法将“Cursor”对象转换为 iterable。

为了简化查询，我们可以删除转发，只包含英文推文。为了了解该请求返回的内容，我们还可以打印附加到每个列表的值:

```
twitter_users = []
tweet_time = []
tweet_string = []
for tweet in tweepy.Cursor(api.search,q="Joker", count=1000).items(1000):
        if (not tweet.retweeted) and ('RT @' not in tweet.text):
            if tweet.lang == "en":
                twitter_users.append(tweet.user.name)
                tweet_time.append(tweet.created_at)
                tweet_string.append(tweet.text)
                print([tweet.user.name,tweet.created_at,tweet.text])
```

![](img/80c8a9672011a3f3d7e0994986e337cc.png)

Tweets related to “Joker”

我们也可以使用查询字符串。让我们把它从“小丑”改成“华金”，这是*小丑*中男主角的名字:

```
for tweet in tweepy.Cursor(api.search,q="Joaquin", count=1000).items(1000):
        if (not tweet.retweeted) and ('RT @' not in tweet.text):
            if tweet.lang == "en":
                twitter_users.append(tweet.user.name)
                tweet_time.append(tweet.created_at)
                tweet_string.append(tweet.text)
                print([tweet.user.name,tweet.created_at,tweet.text])
```

![](img/43fe70cce1a7c7cfa102bd95f49fad1d.png)

Tweets related to “Joaquin”

接下来，我们可以将查询结果存储在数据帧中。为此，让我们定义一个函数，该函数将一个关键字作为参数，并返回一个包含与该关键字相关的 1000 条推文的数据帧:

```
def get_related_tweets(key_word):
    twitter_users = []
    tweet_time = []
    tweet_string = [] 
    for tweet in tweepy.Cursor(api.search,q=key_word, count=1000).items(1000):
            if (not tweet.retweeted) and ('RT @' not in tweet.text):
                if tweet.lang == "en":
                    twitter_users.append(tweet.user.name)
                    tweet_time.append(tweet.created_at)
                    tweet_string.append(tweet.text)
                    #print([tweet.user.name,tweet.created_at,tweet.text])
    df = pd.DataFrame({'name':twitter_users, 'time': tweet_time, 'tweet': tweet_string})
    df.to_csv(f"{key_word}.csv")
    return df
```

当我们用“Joker”调用函数时，定义一个 dataframe 作为函数的返回值，并打印它的前五行，我们得到:

```
df_joker = get_related_tweets("Joker")
print(df_joker.head(5))
```

![](img/2fc31a515eaf65ea216bd7b808486b89.png)

First five rows of dataframe of tweets for “Joker”

如果我们对“华金”做同样的事情:

```
df_joaquin = get_related_tweets("Joaquin")
print(df_joaquin.head(5))
```

![](img/3fd4b875e7adbdd99d1b5a8ac0f6efb2.png)

First five rows of dataframe of tweets for “Joaquin”

我们还可以搜索带有“小丑”和“糟糕电影”的推文。csv '行):

```
def get_related_tweets(key_word):
    twitter_users = []
    tweet_time = []
    tweet_string = [] 
    for tweet in tweepy.Cursor(api.search,q=key_word, count=1000).items(1000):
            if (not tweet.retweeted) and ('RT @' not in tweet.text):
                if tweet.lang == "en":
                    twitter_users.append(tweet.user.name)
                    tweet_time.append(tweet.created_at)
                    tweet_string.append(tweet.text)
                    #print([tweet.user.name,tweet.created_at,tweet.text])
    df = pd.DataFrame({'name':twitter_users, 'time': tweet_time, 'tweet': tweet_string})
    return df
df_bad = get_related_tweets("Joker bad movie")
print(df_bad.head(5))
```

![](img/2ddf50261a85c2334e1b56885dbb7f07.png)

让我们通过遍历 dataframe 索引并从 tweet 列中选择值来仔细查看几行:

![](img/28447fceda4daff0ff78168111be7d0f.png)

Expanded strings of ”Joker bad movie” tweets

对于带有“小丑”和“好电影”的推文:

```
df_good = get_related_tweets("Joker good movie")
print(df_good.head(5))
```

![](img/c47652ff2a7e8fea46d835a2fbcaa42e.png)![](img/b47a853ca9901aad5b0121f2771255ed.png)

Expanded strings of ”Joker good movie” tweets

在下一篇文章中，我们将使用一个名为 TextBlob 的 python 库对其中一些推文进行情感分析。在此基础上，我们可以构建一个情绪分析器，将一条推文分类为负面情绪或正面情绪。这篇文章的代码可以在 Github 上找到。感谢您的阅读！祝好运，机器学习快乐！