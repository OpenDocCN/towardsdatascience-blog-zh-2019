# Twitter 如何看待新款特斯拉 Cybertruck？Python 中的情感分析

> 原文：<https://towardsdatascience.com/what-does-twitter-think-of-the-new-tesla-cybertruck-sentiment-analysis-in-python-afa22a9aefce?source=collection_archive---------30----------------------->

![](img/c4073e18c27941566b53e3d5e23a5cad.png)

Photo by [Pixabay](https://www.pexels.com/@pixabay) on [Pexels](https://www.pexels.com/photo/close-up-of-illuminated-text-against-black-background-258083/)

最近，埃隆·马斯克(Elon Musk)推出了特斯拉 Cybertruck，这是一款由特斯拉公司(Tesla Inc .)开发的全电动电池驱动的商用车。Cybertruck 是每天销售的数千辆化石燃料驱动卡车的可持续能源替代品。最近，在一次电脑卡车的演示中，埃隆告诉卡车的主要设计师之一，向窗户扔一个小钢珠，以展示防弹玻璃的耐用性。两次投掷后，司机和乘客座位上的玻璃都碎了。

此次发布会在社交媒体上引起了大量的关注。尽管许多人对卡车的锐利轮廓、不寻常的形状以及揭幕期间的失败测试表示不满，但特斯拉收到了超过 25 万辆卡车的预订。鉴于这种差异，看看社交媒体上对使用机器学习的网络卡车的普遍看法将是有趣的。在本帖中，我们将对关于特斯拉赛博卡车的推文进行情感分析。

首先，你需要申请一个 Twitter 开发者账户:

![](img/8de478e493cd200d2ddb35292f06e27d.png)

[Source](https://projects.raspberrypi.org/en/projects/getting-started-with-the-twitter-api/3)

您的开发人员帐户获得批准后，您需要创建一个 Twitter 应用程序:

![](img/535749c04db1726dafff300aac41628c.png)

[Source](https://projects.raspberrypi.org/en/projects/getting-started-with-the-twitter-api/4)

申请 Twitter 开发者账户和创建 Twitter 应用程序的步骤在[这里](https://projects.raspberrypi.org/en/projects/getting-started-with-the-twitter-api/4)列出。

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
import pandas as pd
```

3.**认证**

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

4. **TWITTER API 请求**

接下来，我们为我们有兴趣分析的字段初始化列表。现在，我们可以查看推文字符串、用户和推文时间。接下来，我们在一个 tweepy“Cursor”对象上编写一个 for 循环。在“Cursor”对象中，我们传递“api.search”方法，为我们想要搜索的内容设置查询字符串，并设置“count”= 1000，这样我们就不会超过 twitter 的速率限制。在这里，我们将搜索关于“特斯拉 Cybertruck”的推文。我们还使用“item()”方法将“Cursor”对象转换为 iterable。

为了简化查询，我们可以删除转发，只包含英文推文。为了了解该请求返回的内容，我们还可以打印附加到每个列表的值:

```
twitter_users = []
tweet_time = []
tweet_string = []
for tweet in tweepy.Cursor(api.search,q='Tesla Cybertruck', count=1000).items(1000):
            if (not tweet.retweeted) and ('RT @' not in tweet.text):
                if tweet.lang == "en":
                    twitter_users.append(tweet.user.name)
                    tweet_time.append(tweet.created_at)
                    tweet_string.append(tweet.text)
                    print([tweet.user.name,tweet.created_at,tweet.text])
```

![](img/01f5d6d0366ec51e6137b0a803dd2080.png)

为了实现可重用性，我们可以把它打包成一个函数，把关键字作为输入。我们还可以将结果存储在数据帧中并返回值:

```
def get_related_tweets(key_word):twitter_users = []
    tweet_time = []
    tweet_string = [] 
    for tweet in tweepy.Cursor(api.search,q=key_word, count=1000).items(1000):
            if (not tweet.retweeted) and ('RT @' not in tweet.text):
                if tweet.lang == "en":
                    twitter_users.append(tweet.user.name)
                    tweet_time.append(tweet.created_at)
                    tweet_string.append(tweet.text)
                    print([tweet.user.name,tweet.created_at,tweet.text])
    df = pd.DataFrame({'name':twitter_users, 'time': tweet_time, 'tweet': tweet_string})

    return df
```

当我们用关键词“特斯拉 Cybertruck”调用函数时，我们得到:

```
get_related_tweets('Tesla Cybertruck')
```

![](img/db176b986aa646d1cc28a95343abb599.png)

为了获得情感分数，我们需要导入一个名为 textblob 的 python 包。textblob 的文档可以在这里找到[。要安装 textblob，请打开命令行并键入:](https://textblob.readthedocs.io/en/dev/)

```
pip install textblob
```

下次导入 textblob:

```
from textblob import TextBlob
```

我们将使用极性得分作为积极或消极感觉的衡量标准。极性得分是一个从-1 到+1 的浮点数。

例如，如果我们定义一个 textblob 对象并传入句子“我爱特斯拉 Cybertruck ”,我们应该得到一个正值的极性得分:

```
sentiment_score = TextBlob(“I love the Tesla Cybertruck”).sentiment.polarity
print("Sentiment Polarity Score:", sentiment_score)
```

![](img/43703cb93427dca4d0ee5dd5f6bee937.png)

让我们来看看关于“特斯拉赛博卡车”的推文的情绪极性得分:

```
df = get_related_tweets("Tesla Cybertruck")
df['sentiment'] = df['tweet'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
print(df.head())
```

![](img/d77c8007e638d8c31f6ad3487b419015.png)

我们也可以计算积极和消极情绪的数量:

```
df_pos = df[df['sentiment'] > 0.0]
df_neg = df[df['sentiment'] < 0.0]
print("Number of Positive Tweets", len(df_pos))
print("Number of Negative Tweets", len(df_neg))
```

![](img/fa0e95cc5f9a8376cd84438cfe1153a3.png)

同样，对于代码重用，我们可以将其全部封装在一个函数中:

```
def get_sentiment(key_word):
    df = get_related_tweets(key_word)
    df['sentiment'] = df['tweet'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
    df_pos = df[df['sentiment'] > 0.0]
    df_neg = df[df['sentiment'] < 0.0]
    print("Number of Positive Tweets about {}".format(key_word), len(df_pos))
    print("Number of Negative Tweets about {}".format(key_word), len(df_neg))
```

如果我们用“Tesla Cybertruck”调用这个函数，我们得到:

```
get_sentiment(“Tesla Cybertruck”)
```

![](img/ff702c72af6025f7285e968a8304832b.png)

如果我们能以编程方式可视化这些结果，那将会很方便。让我们导入 seaborn 和 matplotlib 并修改我们的 get _ 情操函数:

```
import seaborn as sns
import matplotlib.pyplot as pltdef get_sentiment(key_word):
    df = get_related_tweets(key_word)
    df['sentiment'] = df['tweet'].apply(lambda tweet: TextBlob(tweet).sentiment.polarity)
    df_pos = df[df['sentiment'] > 0.0]
    df_neg = df[df['sentiment'] < 0.0]
    print("Number of Positive Tweets about {}".format(key_word), len(df_pos))
    print("Number of Negative Tweets about {}".format(key_word), len(df_neg))
    sns.set()
    labels = ['Postive', 'Negative']
    heights = [len(df_pos), len(df_neg)]
    plt.bar(labels, heights, color = 'navy')
    plt.title(key_word)get_sentiment(“Tesla Cybertruck”)
```

![](img/890008b46f6b325dea17acccf82bef63.png)

如你所见，关于特斯拉电动卡车的推文正面情绪多于负面情绪。收集几天的数据来观察情绪如何随时间变化会很有趣。也许我会把它留到以后的文章里。

感谢您的阅读。这篇文章的代码可以在 [GitHub](https://github.com/spierre91/medium_code) 上找到。祝好运，机器学习快乐！