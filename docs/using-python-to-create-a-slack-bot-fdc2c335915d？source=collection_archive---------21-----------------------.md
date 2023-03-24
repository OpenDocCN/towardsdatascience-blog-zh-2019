# 使用 Python 创建一个 Slack Bot

> 原文：<https://towardsdatascience.com/using-python-to-create-a-slack-bot-fdc2c335915d?source=collection_archive---------21----------------------->

![](img/59bab0be776ac8697525c7d4882e6eb8.png)

Photo by [Lenin Estrada](https://unsplash.com/@lenin33?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

在一家初创公司工作时，我们需要自动处理消息，以便获得某些事件和触发的通知。例如，我工作的公司处理与某些商店的联系。如果连接中断，Python 将读取我们数据库中的信息。我们现在可以将该数据发送到一个 Slack 通道，专门用于重新连接该存储。

我们可以为这个 Slack Bot 创造许多可能性，我的同事创造的另一个例子是将 Python 错误消息输出到通道。不仅创建通知，还创建持续的错误日志。

我们现在可以做些简单的事情。对于这个项目，我们将构建一个 Slack Bot，如果它检测到脚本运行的日期是美国假日，它将输出一条消息。

**这个项目需要什么:**
**1。 *Python
2。松弛工作空间/帐户***

**所需 Python 模块:**
1。 ***datetime*** (告知脚本运行的日期并标准化日期时间)
2。 ***熊猫*** (主要用于将数据组织成 dataframe)
3 . ***请求*** (连接到我们从网站获取的数据，并将数据发送到 slack API)
4 . ***bs4*** (我们正在从网站获取的数据的数据解析)
5。 ***json*** (对数据进行编码，以便 slack API 可以使用)

首先，让我们创建一个新的 Python 文件并导入所需的模块。

```
from datetime import date, datetime, timedelta
import pandas as pd
import requestsfrom bs4 
import BeautifulSoup
import json
```

我们可以使用`datetime`模块获取今天的日期时间，并将其存储为年-月-日这样的字符串格式。

```
today = datetime.now().strftime(‘%Y-%m-%d’)
#today = '2019-10-29'
```

使用模块`requests`，我们现在可以连接到我们想要获取数据的网站。为此，我连接到这个 [*网站*](https://www.timeanddate.com/holidays/us/) 获取假期。当我们从一个网站获取数据时，它会发送我们在该页面上看到的所有内容。我们只想要这个数据中的假期。这就是 [BeautifulSoup](https://www.crummy.com/software/BeautifulSoup/bs4/doc/) 发挥作用的地方，使用模块`bs4`我们可以很容易地解析这些数据。我们现在可以把它放到一个数据框架中。

这不是将数据传输到数据帧的最佳方式，但我们只需要完成这项工作。

```
regionalHolidaysList = []
for result in results:    
     date = result.contents[0].text + ', ' + datetime.today().strftime('%Y')    
     weekday = result.contents[1].text    
     holidayName = result.contents[2].text    
     observance = result.contents[3].text     
     stateObserved = result.contents[4].text
     regionalHolidaysList.append((date, weekday, holidayName, observance, stateObserved))regionalHolidayDf = pd.DataFrame(regionalHolidaysList, columns = ['date', 'weekday', 'holidayName', 'observance', 'stateObserved'])regionalHolidayDf['date'] = regionalHolidayDf['date'].apply(lambda x: (datetime.strptime(x, '%b %d, %Y').strftime('%Y-%m-%d')))
```

我们现在可以从这个数据帧创建一个日期时间列表

```
dateList = regionalHolidayDf['date'].tolist()
```

如果`today`在这个`dateList`中，我们可以告诉它打印到请求的空闲通道。为了让 Python 使用 Slack 发送东西，我们需要创建一个传入的 webhook。Slack 有一些关于如何做到这一点的文档，在 https://api.slack.com/messaging/webhooks。

如果一切都做得正确，那么我们应该有这样的东西:

```
https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX
```

最后一部分变得有点复杂，所以我将在代码中用标签来发布我的评论。

```
#So we get the date if it is in this list it will send a message in slackif today in dateList:
     todayHoliday = regionalHolidayDf[regionalHolidayDf['date'] ==          today]    
     info=[]    
     for holiday in todayHoliday.itertuples():
          info.append(':umbrella_on_ground:' +                       holiday.holidayName +  ' in ' +                      holiday.stateObserved.replace('*', ''))    
     if len(info) >1:        
          infoFinal = '\n'.join(info)    
     else:        
       infoFinal = info[0]#Here is where we can format the slack message, it will output any holiday with todays

     message = f'@here Fun fact! \n Today({today}) is: \n {infoFinal}'        
     print('Sending message to the happyholiday channel...')
     slackmsg = {'text': message} #Using the module json it formats it where the slack API can accept it
#we can store the slack link into a variable called webhook webhook='https://hooks.slack.com/services/T00000000/B00000000/XXXXXXXXXXXXXXXXXXXXXXXX'
response = requests.post(webhook,  data=json.dumps(slackmsg), headers={'Content-Type': 'application/json'})    
     if response.status_code != 200:        
          raise ValueError('Request to slack returned an error %s, the response is:\n%s'            % (response.status_code, response.text)        )    
     print('Request completed!')
else:    
     print('No regional holiday today! See you next time!')
```

现在我们终于可以运行这个脚本了，如果成功，它将根据脚本运行的日期输出一个假日。

为了进一步自动化，我们可以将它放在 EC2 上，让它每天运行！但这需要更多的证书和设置，让我知道如果这是另一个话题，你们有兴趣！

访问我的代码[这里](https://www.patreon.com/melvfnz)！

我在这里也有家教和职业指导。

如果你们有任何问题、评论或顾虑，请不要忘记通过 LinkedIn[与我联系！](https://www.linkedin.com/in/melvfernandez/)