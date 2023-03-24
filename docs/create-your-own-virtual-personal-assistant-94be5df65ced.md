# 创建您自己的虚拟个人助理

> 原文：<https://towardsdatascience.com/create-your-own-virtual-personal-assistant-94be5df65ced?source=collection_archive---------7----------------------->

![](img/13a7ba0b0c30a43647f10a465e1cb97d.png)

Courtesy: [Pixabay](https://pixabay.com/photos/robot-mech-machine-technology-2301646/#)

你知道 Cortana，Siri，Google Assistant 吧？你有没有想象过你可以制作自己的虚拟个人助理，随心所欲的定制？今天，我们就在这里做。我们将用 python 从头开始构建一个个人助理。哦，在开始之前，让我告诉你，这绝不是一个人工智能，而只是一个人工智能可以做什么的有力例子，以及 python 是多么多才多艺和令人惊叹。此外，为了开始使用 python，您需要有一些使用 python 的经验。那么，让我们开始吧:

首先，我们需要安装一些重要的软件包:

*   SpeechRecognition:用于执行语音识别的库，支持多种引擎和 API，在线和离线。
*   Pyttsx3 : Pyttsx 是 python 中一个很好的文本到语音转换库。
*   Wikipedia : Wikipedia 是一个 Python 库，使得访问和解析来自 Wikipedia 的数据变得容易。
*   Wolframalpha:针对 [Wolfram|Alpha](http://wolframalpha.com/) v2.0 API 构建的 Python 客户端。
*   py audio:PortAudio 的 Python 绑定。

确保你已经安装了所有这些软件包，否则你可能会遇到一些错误，这是你如何安装它:

```
pip install PackageName
```

**PyAudio 安装:**

你可能会在安装 Pyaudio 时遇到一些错误，我也遇到过同样的问题。您可以利用这些步骤来避免安装错误:

*   通过`python --version`找到你的 Python 版本比如我的是`3.7.3`
*   从 [**这里**](https://www.lfd.uci.edu/~gohlke/pythonlibs/#pyaudio) 找到合适的`.whl`文件，比如我的是`PyAudio‑0.2.11‑cp37‑cp37m‑win_amd64.whl`，下载。
*   转到下载它的文件夹，例如`cd C:\Users\foobar\Downloads`
*   以我的例子为例，用`pip`安装`.whl`文件:

```
pip install PyAudio-0.2.11-cp37-cp37m-win_amd64.whl
```

此外，安装它可以避免不必要的错误:

```
pip install pypiwin32
```

如果您完成了这些包的安装，那么我们可以导入它们并返回代码:

```
import os
import sys
import datetime
import pyttsx3
import speech_recognition as sr
import wikipedia
import wolframalpha
import webbrowser
import smtplib
import random
```

现在，我们将使用“SAPI5”作为 pyttsx3 的 TTS 引擎，获取 wolframaplha 的密钥，并定义客户端。

```
engine = pyttsx3.init(‘sapi5’) 
client = wolframalpha.Client(‘Get your own key’)
```

*您可以从 wolframalpha.com->Apps->Key 获得自己的密钥。*

现在，我们将初始化一个变量，并获得我们需要的必要的声音参数。女声可以在第二行设置为-1，男声设置为-2。接下来，我们将创建一个函数 **talk** ，将音频作为输入参数。

```
voices = engine.getProperty(‘voices’)
engine.setProperty(‘voice’, voices[len(voices) — 2].id)def talk(audio): 
    print(‘KryptoKnite: ‘ + audio) 
    engine.say(audio) 
    engine.runAndWait()
```

接下来，让我们创建另一个函数 **greetMe** ，它将用于在用户运行程序时问候用户。 *datetime.datetime.now()。小时*用于以小时为单位获取当前时间，并根据时间和以下条件给出输出。 **Talk** fn 将用于给出语音方面的输出。

```
def greetMe():
    CurrentHour = int(datetime.datetime.now().hour)
    if CurrentHour >= 0 and CurrentHour < 12:
        talk('Good Morning!') elif CurrentHour >= 12 and CurrentHour < 18:
        talk('Good Afternoon!') elif CurrentHour >= 18 and CurrentHour != 0:
        talk('Good Evening!') greetMe()talk('Hey Buddy, It\'s  your assistant KryptoKnite!')
talk('tell me about today?')
```

接下来，我们将创建另一个函数 **GivenCommand** ，用于识别用户输入，它将定义麦克风用作输入源，我们将暂停阈值设置为 1。尝试使用 except 块，要识别的语言将被设置为英语-印度，如果语音未被识别或听不到，我们将发送文本输入作为一种错误消息。

```
def GivenCommand():
    k = sr.Recognizer()
    with sr.Microphone() as source:
        print("Listening...")
        k.pause_threshold = 1
        audio = k.listen(source)
    try:
        Input = k.recognize_google(audio, language='en-in')
        print('Kunal Dhariwal: ' + Input + '\n') except sr.UnknownValueError:
        talk('Sorry! I didn\'t get that! Try typing it here!')
        Input = str(input('Command: ')) return Input
```

现在，让我们开始主要功能:

在这里，我们将声明一些重要的函数和条件，这些函数和条件将增强我们的个人助理的功能，并帮助他提供输出和接收来自用户的输入。

```
if __name__ == '__main__': while True: Input = GivenCommand()
        Input = Input.lower() if 'open google' in Input:
            talk('sure')
            webbrowser.open('www.google.co.in') elif 'open youtube' in Input:
            talk('sure')
            webbrowser.open('www.youtube.com') elif "what\'s up" in Input or 'how are you' in Input:
            setReplies = ['Just doing some stuff!', 'I am good!',                                     
                          'Nice!', 'I am amazing and full of power']
            talk(random.choice(setReplies))
```

同样，你可以添加更多的 ***elif*** 和其他功能，比如我添加了一个发送电子邮件的功能。

```
elif 'email' in Input:
    talk('Who is the recipient? ')
    recipient = GivenCommand() if 'me' in recipient:
        try:
            talk('What should I say? ')
            content = GivenCommand() server = smtplib.SMTP('smtp.gmail.com', 587)
            server.ehlo()
            server.starttls()
            server.login("Your_Username", 'Your_Password')
            server.sendmail('Your_Username', "Recipient_Username", content)
            server.close()
            talk('Email sent!') except:
            talk('Sorry ! I am unable to send your message at this moment!')
```

或者可能正在播放一些音乐？

```
elif 'play music' in Input:
    music_folder = 'Path
    music = ['song']
    random_music = music_folder + random.choice(music) + '.mp3'
    os.system(random_music) talk('Okay, here is your music! Enjoy!')
```

接下来，我们将添加一些功能，使用这些输入在维基百科、谷歌上进行搜索，并使用 wolframalpha。

```
else:
    Input = Input
    talk('Searching...')
    try:
        try:
            res = client.Input(Input)
            outputs = next(res.outputs).text
            talk('Alpha says')
            talk('Gotcha')
            talk(outputs) except:
            outputs = wikipedia.summary(Input, sentences=3)
            talk('Gotcha')
            talk('Wikipedia says')
            talk(outputs) except:
        talk("searching on google for " + Input)
        say = Input.replace(' ', '+')
        webbrowser.open('https://www.google.co.in/search?q=' + Input)talk('Next Command! Please!')
```

当这一切完成后，程序退出是非常重要的。让我们在这里为它写一个条件:

```
elif 'nothing' in Input or 'abort' in Input or 'stop' in Input:
    talk('okay')
    talk('Bye, have a good day.')
    sys.exit()elif 'bye' in Input:
    talk('Bye, have a great day.')
    sys.exit()
```

就是这样！您已经创建了自己的虚拟个人助理。

你现在可以自定义它，并为它设置任何条件，还可以添加 N 个功能，使它更加神奇。

> **完整代码:**[https://bit.ly/2VaBsEU](https://bit.ly/2VaBsEU)
> 
> 你可以在我的 LinkedIn 帖子[https://bit.ly/2DW8qU0](https://bit.ly/2DW8qU0)这里获得 ***视频演示***
> 
> 如果您遇到任何错误或需要任何帮助，您可以随时在 LinkedIn 上发表评论或 ping 我。
> 
> **领英**:[https://bit.ly/2u4YPoF](https://bit.ly/2u4YPoF)
> 
> **Github**:[https://bit.ly/2SQV7ss](https://bit.ly/2SQV7ss)

> 我希望这有助于增强您的知识库:)
> 
> 关注我了解更多！
> 
> 感谢您的阅读和宝贵的时间！