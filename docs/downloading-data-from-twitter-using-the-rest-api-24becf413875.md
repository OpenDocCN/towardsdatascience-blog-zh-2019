# 使用 REST API 从 Twitter 下载数据

> 原文：<https://towardsdatascience.com/downloading-data-from-twitter-using-the-rest-api-24becf413875?source=collection_archive---------13----------------------->

![](img/2b33eef80e48f5e2006ef17bf3ba92bd.png)

嘿大家好！这是关于从 Twitter 上获取数据并使用它来获得某些见解的出版物列表中的第二篇文章，比如某个趋势上最有影响力的用户、主题建模等等。

如果你没有读过第一篇文章，你可以在这里看看:

[](https://medium.com/@jaimezornoza/downloading-data-from-twitter-using-the-streaming-api-3ac6766ba96c) [## 使用流式 API 从 Twitter 下载数据

### 在这篇文章中，我们将介绍如何使用流媒体 API 来获取包含特定单词或标签的推文，以及如何…

medium.com](https://medium.com/@jaimezornoza/downloading-data-from-twitter-using-the-streaming-api-3ac6766ba96c) 

虽然上一篇文章讨论了如何从 Twitter 上收集实时生成的数据，但这篇新文章将涵盖如何收集历史数据，如某个用户、他的追随者或他的朋友以前的推文。

我们开始吧！

# 使用 REST API 收集历史数据

当使用流 Twitter API 时，我们收集实时产生的数据，REST API 服务于相反的目的:收集收集时间之前产生的数据，即。历史数据。

使用这个 API，我们可以收集包含某些关键字的旧推文，类似于以前的做法，但我们也可以收集与平台相关的其他信息，如不同用户帐户的朋友和追随者，某个帐户的转发，或某个推文的转发。

Twitter APIs 中的用户由两个不同的变量来标识:

*   用户 ***screen_name*** ，也就是我们都习以为常的带@的 Twitter 名称。比如“@jaimezorno”。
*   ***user_id*** ，是每个 Twitter 用户的唯一数字标识符，是一个很长的数字串，比如 747807250819981312。

在数据收集过程中，当我们想要指定我们想要从中收集数据的用户时，我们可以使用该用户的 *screen_name* 或 *user_id* 来完成，因此在深入研究 REST API 提供的更复杂的函数之前，我们将了解如何获取某个用户的 Twitter Id，我们知道该用户的用户名，反之亦然。

# 用用户名获取 Twitter 的 Id，反之亦然

从 Twitter Id 到用户屏幕名称是需要的，因为我们稍后将描述的一些函数返回 Twitter 标识符而不是用户屏幕名称，所以如果我们想知道谁是与相应 Id 相关联的实际用户，我们需要这个功能。

和往常一样，第一步是收集到 Twitter API。

```
import tweepy  
import time

access_token = "ENTER YOUR ACCESS TOKEN"  
access_token_secret = "ENTER YOUR ACCESS TOKEN SECRET"  
consumer_key = "ENTER YOUR CONSUMER KEY"  
consumer_secret = "ENTER YOUR CONSUMER SECRET"  

auth = tweepy.OAuthHandler(consumer_key, consumer_secret)  
auth.set_access_token(access_token, access_token_secret)  
api = tweepy.API(auth)
```

在代码中，将“输入您的……”然后运行最后三行来创建到 Twitter REST API 的连接。

注意，这次我们没有像使用流式 API 那样创建一个**流**对象，而是创建了一个 **api** 对象。一旦我们做到了这一点，从屏幕名称到 id 以及从 id 到屏幕名称的转换就非常简单了，这是通过运行以下代码块中的行来完成的:

```
user = api.get_user(screen_name = 'theresa_may')  
print(user.id)
```

这个块在 REST API 中查询 Theresa May 的官方 Twitter 帐户的 user_id，返回: *747807250819981312，*这是与该帐户相关联的 id。这里需要注意的是，screen_name 不包含@。

要以相反的方向完成这项工作，并收集我们知道其 id 的帐户的屏幕名称，非常简单:

```
user = api.get_user(747807250819981312)  
print(user.screen_name)
```

它将打印: *theresa_may* 。正如我们所见，Id 和屏幕名称都是 API 返回的 ***用户*** 对象的属性，其中包含许多有价值的信息，如用户关注者计数、出版物数量、帐户创建日期等等。这些参数将在另一篇文章中探讨。

从用户名到 id 以及从 id 到用户名的转换就这么简单。现在让我们探索 REST API 更复杂和有用的功能。

# 从特定用户的时间线收集推文

某个用户的时间线是他或她过去发布或转发的推文。收集这些信息有助于了解社交网络中某个账户的先前活动。

然而，我们必须知道，将要使用的方法只能返回特定用户最近 3200 条推文，因此，如果我们正在收集一个非常活跃的帐户的帖子，并且想要很久以前的推文，我们将无法获得它们。

这是 Twitter API 的一个已知限制，目前还没有修复，因为通过这样做，Twitter 不必存储每个 Twitter 帐户产生的所有 tweets。

在如上所述创建了到 Twitter REST API 的连接之后，为了收集用户的时间表，我们必须使用类似于以下代码块中所示的代码结构:

```
try:  
    for tweet in tweepy.Cursor(api.user_timeline, screen_name="theresa_may", exclude_replies=True).items():                      
                    print(tweet)  
except tweepy.TweepError:  
    time.sleep(60)
```

正如我们所见，这段代码引入了 Twitter API 固有的新概念: ***光标对象。*** 尽管看起来令人生畏，但这只不过是 API 必须处理分页并能够以高效有序的方式交付内容的方式。

在这种情况下，我们将从用户 *@theresa_may* 收集历史推文，排除其他用户对推文的回复。可以添加一个类似的参数 *include_rts* 来从这个用户时间表中消除转发。

此外， ***try-except*** duo 被添加来处理我们可能发现的任何错误，如请求率超出或保护用户。当操作这种类型的 API 时，这是非常常见的。

这段代码的输出是一个非常难看的对象，叫做 ***状态对象*** ，用于每条推文，看起来像这样:

```
Status(_api=<tweepy.api.API object at 0x000001C52728A710>, _json={'created_at': 'Sun May 12 11:55:41 +0000 2019', 'id': 1127542860520329216, 'id_str': '1127542860520329216', 'text': 'Congratulations to @SPendarovski on your inauguration as President of North Macedonia. I witnessed the strong relat…………
```

另一篇文章，如用户对象案例，将详细解释这些对象的性质及其属性，然而现在我们将只描述如何从中收集一些最有趣的字段。

让我们看看我们如何能做到这一点。

我们将保持与前一个块相同的代码结构，但是添加了一些额外的行，我们将使用这些行来获取我们认为最相关的 status 对象的部分。

```
try:  
    for tweet in tweepy.Cursor(api.user_timeline, screen_name="theresa_may", exclude_replies=True, count = 10).items():  
                    tweet_text = tweet.text  
                    time = tweet.created_at  
                    tweeter = tweet.user.screen_name  
                    print("Text:" + tweet_text + ", Timestamp:" + str(time) + ", user:" +  tweeter)  
except tweepy.TweepError:  
    time.sleep(60)
```

这一次，执行这段代码应该会显示如下内容:

```
Text:We’re driving the biggest transformation in mental health services for more than a generation. [https://t.co/qOss2jOh4c,](https://t.co/qOss2jOh4c,) Timestamp:2019-06-17 07:19:59, user:theresa_may
Text:RT @10DowningStreet: PM @Theresa_May hosted a reception at Downing Street to celebrate:
✅ 22 new free schools approved to open 
✅ 19,000 ad…, Timestamp:2019-06-15 13:53:34, user:theresa_may
Text:Two years on from the devastating fire at Grenfell Tower, my thoughts remain with the bereaved, the survivors and t… [https://t.co/Pij3z3ZUJB,](https://t.co/Pij3z3ZUJB,) Timestamp:2019-06-14 10:31:59, user:theresa_may
```

考虑到您将获得的 tweet 取决于您正在搜索的用户在执行代码之前发布的 tweet，因此如果您以目标用户 *theresa_may* 运行这些块，您很可能不会获得与我相同的 tweet。

尽管从前面的代码块返回的结果可能看起来更好，但我们可能希望数据的格式便于以后存储和处理，比如 JSON。

我们将对代码做最后一次修改，以便打印出每条 tweet，以及我们想要的 tweet 中的字段，作为 JSON 对象。为此，我们需要导入 json 库，并对代码做进一步的修改，如下所示:

```
import json  

try:  
    for tweet in tweepy.Cursor(api.user_timeline, screen_name="theresa_may", exclude_replies=True, count = 10).items():  
                    tweet_text = tweet.text  
                    time = tweet.created_at  
                    tweeter = tweet.user.screen_name  
                    tweet_dict = {"tweet_text" : tweet_text.strip(), "timestamp" : str(time), "user" :tweeter}  
                    tweet_json = json.dumps(tweet_dict)  
                    print(tweet_json)  
except tweepy.TweepError:  
    time.sleep(60)
```

这一次，我们将输出与以前相同的字段，但采用 JSON 格式，这样便于其他人处理和理解。在这种情况下，相同 tweet 的输出将是:

```
{"tweet_text": "We\u2019re driving the biggest transformation in mental health services for more than a generation. [https://t.co/qOss2jOh4c",](https://t.co/qOss2jOh4c%22,) "timestamp": "2019-06-17 07:19:59", "user": "theresa_may"}
{"tweet_text": "RT @10DowningStreet: PM @Theresa_May hosted a reception at Downing Street to celebrate:\n\u2705 22 new free schools approved to open \n\u2705 19,000 ad\u2026", "timestamp": "2019-06-15 13:53:34", "user": "theresa_may"}
{"tweet_text": "Two years on from the devastating fire at Grenfell Tower, my thoughts remain with the bereaved, the survivors and t\u2026 [https://t.co/Pij3z3ZUJB",](https://t.co/Pij3z3ZUJB%22,) "timestamp": "2019-06-14 10:31:59", "user": "theresa_may"}
```

在了解了如何有效地收集和处理某个用户的时间表之后，我们将看看如何收集他们的朋友和追随者。

# 收集某个用户的追随者。

获取一组用户的关注者是 Twitter 研究中最常见的行为之一，因为创建关注者/被关注者网络可以提供一些关于某个主题或标签的特定用户群的非常有趣的见解。

要获得某个用户的关注者，只需像以前一样使用我们的凭据连接到 API，然后运行以下代码即可:

```
try:   
    followers = api.followers_ids(screen_name="theresa_may")  
except tweepy.TweepError:  
    time.sleep(20)
```

通过在 *api = tweepy 中设置参数***wait _ on _ rate _ limit***为真。API(auth，wait_on_rate_limit=True)* 当我们连接到 API 时，下载任何类型的数据时超过速率限制的错误都可以避免，因此尽管在本文的前几部分中没有使用过它，我还是建议在您打算从 Twitter REST API 下载大量数据时使用它。

这里有一个列表，上面有账户 *@theresa_may 的所有关注者的 id。*这些 id 然后可以使用我们之前描述的 *api.get_user* 方法翻译成用户名。

如果我们想要收集某一组用户的关注者，我们只需要在前面的代码块中添加几行代码，如下所示:

```
user_list = ["AaltoUniversity", "helsinkiuni","HAAGAHELIAamk", "AaltoENG"]follower_list = []  
for user in user_list:  
    try:   
        followers = api.followers_ids(screen_name=user)  
    except tweepy.TweepError:  
        time.sleep(20)  
        continue  
    follower_list.append(followers)
```

在这种情况下，我们将收集与芬兰大学相关的用户帐户的追随者。该代码的输出将是一个列表( *follower_list* )，在每个索引中有一个列表，其中包含来自 *user_list* 的具有相同索引的帐户的追随者。

使用*枚举*函数可以很容易地关联这两个列表(用户和关注者列表):

```
for index, user in enumerate(user_list):  
    print("User: " + user + "\t Number of followers: " + str(len(follower_list[index])))
```

这个模块的输出将是:

```
User: AaltoUniversity  Number of followers: 5000User: helsinkiuni      Number of followers: 5000User: HAAGAHELIAamk    Number of followers: 4927User: AaltoENG  Number of followers: 144
```

这可能会让你感到疑惑:账号*@阿尔托大学*和 *@helsinkiuni* 的粉丝数量是否完全相同，都是 5000 人？

最明显的答案是否定的。如果你查看这两所大学的 Twitter 账户，你会发现它们的粉丝都在万分之几的范围内。

那为什么我们只得到 5000 呢？

嗯，这是因为对于涉及分页的问题，Twitter API 将它们的响应分解在不同的*页面*中，我们可以认为这些页面是具有某个最大大小的所请求信息的*【块】*，要从一个页面转到下一个页面，我们需要使用一种特殊的对象，称为*光标*对象，这在上面已经提到过。

以下代码使用了相同的功能，但这次使用了一个*光标*对象，以便能够抓取每个用户的所有关注者:

```
user_list = ["AaltoUniversity", "helsinkiuni","HAAGAHELIAamk", "AaltoENG"]    
follower_list = []  
for user in user_list: 
    followers = []  
    try:         
        for page in tweepy.Cursor(api.followers_ids, screen_name=user).pages():  
            followers.extend(page)  
    except tweepy.TweepError:  
        time.sleep(20)  
        continue  
    follower_list.append(followers)
```

这一次，如果我们使用*枚举*循环来打印每个用户和他们的追随者数量，输出将是:

```
User: AaltoUniversity  Number of followers: 35695User: helsinkiuni      Number of followers: 31966User: HAAGAHELIAamk    Number of followers: 4927User: AaltoENG  Number of followers: 144
```

也就是每个账户的真实粉丝数。

# 收集某个用户的朋友。

类似于我们如何收集某个用户的关注者，我们也可以收集他的*【朋友】*，也就是某个用户关注的人群。为此，我们将一如既往地使用我们的凭据连接到 API，然后运行以下代码:

```
friends = []  
try:   
    for page in tweepy.Cursor(api.friends_ids, screen_name="theresa_may").pages():  
        friends.extend(page)  
except tweepy.TweepError:  
    time.sleep(20)
```

该代码块中的变量 friends 将是一个列表，其中包含我们选择的 screen_name 用户的所有朋友(在本例中为 theresa_may)

# 查看某个用户的关注者/朋友的数量。

如果我们对某个账户的追随者/朋友是谁不感兴趣，而只对他们的数量感兴趣，Twitter API 允许我们收集这些信息，而不必收集所需账户的所有追随者/朋友。

要做到这一点而不实际收集所有的关注者(考虑到下载速率限制，如果用户有很多关注者，这可能需要一段时间)，我们可以使用我们之前使用的从 *user.screen_name* 到 *user.id* 的 *api.get_user* 方法*，反之亦然。下面的代码块显示了如何操作:*

```
user = api.get_user(screen_name = 'theresa_may')  

print(user.followers_count)  
print(user.friends_count)
```

它将输出:

```
83939129
```

我们也可以使用 Twitter *user.id* 来做这件事，如果我们知道它的话，就像之前看到的那样，就像这样:

```
user = api.get_user(747807250819981312)print(user.followers_count)  
print(user.friends_count)
```

它会再次输出:

```
83939129
```

我们可以从 Theresa 的官方账户中看到，这是正确的追随者和朋友数量。

![](img/4986c4d8d5170370acf1d8d26eef827e.png)

# 结论:

我们已经描述了 Twitter REST API 的主要功能，并解决了从它那里收集数据时可能会发现的一些问题。

这些数据可以用于很多目的:从使用复杂的机器学习算法检测趋势或假新闻，到推断某个品牌的积极程度的情感分析，图形构建，信息扩散模型等等。

如需进一步研究或澄清此处的信息，请参考本指南中的链接或:

Twitter 开发者页面:[https://developer.twitter.com/en/docs](https://developer.twitter.com/en/docs)

Tweepy 的 github 页面:[https://github.com/tweepy/tweepy](https://github.com/tweepy/tweepy)

Tweepy 官方页面:[https://www.tweepy.org/](https://www.tweepy.org/)

推特的高级搜索:[https://twitter.com/search-advanced](https://twitter.com/search-advanced)

敬请关注《社交网络分析》的更多帖子！