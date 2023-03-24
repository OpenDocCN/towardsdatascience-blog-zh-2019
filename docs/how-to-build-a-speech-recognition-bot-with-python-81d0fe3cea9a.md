# 如何用 Python 构建语音识别机器人

> 原文：<https://towardsdatascience.com/how-to-build-a-speech-recognition-bot-with-python-81d0fe3cea9a?source=collection_archive---------2----------------------->

## 即使你对语音识别一无所知

![](img/27138bbf33e14c53f63babdc86e10591.png)

你现在可能意识到了什么。

像亚马逊 Alexa 这样的语音产品的巨大成功已经证明，在可预见的未来，某种程度的语音支持将是家用技术的一个重要方面。

换句话说，支持语音的产品将会改变游戏规则，因为它提供的交互性和可访问性是很少有技术可以比拟的。

不需要 GUI。

不需要发短信。

不需要表情符号。

都是关于 [***速度***](https://developer.amazon.com/blogs/alexa/post/26c76734-3445-4d5f-8c78-aa1eae4100cf/gary-vaynerchuk-voice-will-explode-and-drive-companies-the-size-of-facebook-instagram) *。*

速度是语音成为下一个主要用户界面的一个重要原因。每十年，我们都拥抱一种与技术互动的新方式。我们已经从字符模式发展到图形用户界面、网络和移动设备。

> **现在，语音提供了一种比移动应用更快捷、更简单的沟通和完成任务的方式。**

我们可以告诉 Alexa 我们需要什么(关灯、调节恒温器、设置闹钟——或者使用像“Alexa，晚安”这样的单一话语来完成以上所有事情)，或者你可以拔出你的手机，解锁它，打开正确的应用程序，并执行一项或多项任务。

当您考虑习惯性用例时，即那些让客户不断回头的用例，通过语音获得的效率会随着时间的推移而增加。

> “由于 Alexa，短信在未来将会下降”
> 
> —加里·维纳查克

[Gary Vaynerchuk](https://medium.com/u/c4ec9163657c?source=post_page-----81d0fe3cea9a--------------------------------): Voice Lets Us Say More Faster

因此，这让我对着手一个新项目非常感兴趣，用 Python 构建一个简单的语音识别。

当然，我不会从头开始构建代码，因为这需要大量的训练数据和计算资源来使语音识别模型以一种体面的方式准确无误。

相反，我使用了 **Google 语音识别 API 来执行 Python** 的语音转文本任务(查看下面的演示，我向您展示了语音识别是如何工作的——现场！).

在本文结束时，我希望您能更好地理解语音识别的一般工作原理，最重要的是，如何使用 Google 语音识别 API 和 Python 来实现它。

相信我。就这么简单。

如果你感兴趣的话，可以在这里查看源代码。

我们开始吧！

# 为什么使用谷歌语音识别 API？

您可能想知道，“鉴于语音识别日益增长的需求和流行，这是唯一可用的 API 吗？”

答案是，还有其他免费或付费的 API，如下所示:

*   `recognize_bing()` : [微软必应演讲](https://azure.microsoft.com/en-us/services/cognitive-services/speech/)
*   `recognize_google()` : [谷歌网络语音 API](https://w3c.github.io/speech-api/speechapi.html)
*   `recognize_google_cloud()` : [谷歌云语音](https://cloud.google.com/speech/)——需要安装谷歌云语音包
*   `recognize_houndify()` : [用 SoundHound 显示](https://www.houndify.com/)
*   `recognize_ibm()` : [IBM 语音转文字](https://www.ibm.com/watson/services/speech-to-text/)
*   `recognize_sphinx()` : [CMU 斯芬克斯](https://cmusphinx.github.io/) -需要安装 PocketSphinx
*   `recognize_wit()` : [机智.爱](https://wit.ai/)

最后，我从[**speecher recognition**](https://pypi.org/project/SpeechRecognition/)**库中选择了 [Google Web Speech API](https://w3c.github.io/speech-api/speechapi.html) ，因为它有一个默认的 API 密匙，这个密匙是硬编码到 speecher recognition 库中的。**

**这意味着您可以立即开始，而不必使用 API 密钥或其他 API 的用户名/密码组合进行身份验证。**

**但是 [Google Web Speech API](https://w3c.github.io/speech-api/speechapi.html) 的便利性也带有一定的局限性:**自己键的 API 配额是每天 50 个请求，目前没有办法提高这个限制。****

**如果我们只是想将这个 API 用于实验目的，这符合我们的用例。请注意，如果您运行的应用程序或网站一直在调用 API，那么您可能需要考虑从上面的任何一个 API 获取付费服务。**

# **使用 Google 语音识别 API 用 Python 构建语音识别**

**![](img/03ba4a5f579c5975038020273adc3cf9.png)**

**[(Source)](https://unsplash.com/photos/npxXWgQ33ZQ)**

**为了避免让您对语音识别如何工作的技术细节感到厌烦，您可以阅读这篇很棒的[文章，它讨论了一般的机制以及如何实现 API](https://realpython.com/python-speech-recognition/) 。**

**在接下来的文章中，我将向您展示我是如何按照本文一步一步地实现这个 API 的。**

**但是首先你需要使用`pip install SpeechRecognition`安装[**speecher recognition**](https://pypi.org/project/SpeechRecognition/)**库。****

****我们可以使用来自这个库本身的 Google Web Speech API。****

****在这个实现中，我使用自己的麦克风录制了我的语音，SpeechRecognizer 访问了麦克风**(安装** [**PyAudio 包**](https://people.csail.mit.edu/hubert/pyaudio/) **以访问麦克风)**并相应地识别了我的语音。****

****查看下面的代码片段来理解完整的实现，因为它们相对来说是不言自明的。****

****Function to recognize speech from microphone****

****为了处理环境噪声，您需要使用`Recognizer`类的`adjust_for_ambient_noise()`方法，以便库能够识别您的声音。****

****运行`adjust_for_ambient_noise()`方法后，等待一秒钟，让它分析收集的音频源，以处理环境噪声并捕捉正确的语音。****

****最后，我们需要实现`try and except`块来处理错误，比如在发送请求后 API 不可达或没有响应，或者我们的语音无法识别。****

****要使用上面的函数，您只需实现下面的块，然后…瞧！你做到了！😃****

# ****使用谷歌语音识别 API 的简单演示****

****既然我们已经准备好了完整的实现代码。是时候看看这东西是怎么运作的了。****

****我录制了一个简短的视频，向您展示 API 是如何从录制我的声音到以文本格式返回它的。****

****虽然这可能看起来不像我们预期的那样准确，但这绝对值得花时间来研究代码和 API！****

# ****最后的想法****

****![](img/ad65e36ac9ae7157b776fc812322ade1.png)****

****[(Source)](https://unsplash.com/photos/Nl-GCtizDHg)****

****感谢您的阅读。****

****我希望您现在对语音识别的一般工作原理有了更好的理解，最重要的是，如何通过 Python 使用 Google 语音识别 API 来实现它。****

****如果你有兴趣，可以在这里查看源代码。****

****我还建议您尝试其他 API 来比较语音到文本的准确性。****

****尽管现阶段支持语音的产品尚未在企业和我们的日常生活中广泛使用，但我真的相信这项技术迟早会颠覆许多企业以及消费者使用语音识别功能产品的方式。****

****一如既往，如果您有任何问题或意见，请随时在下面留下您的反馈，或者您可以随时通过 [LinkedIn](https://www.linkedin.com/in/admond1994/) 联系我。在那之前，下一篇文章再见！😄****

## ****关于作者****

****[**阿德蒙德·李**](https://www.linkedin.com/in/admond1994/) 目前是东南亚排名第一的商业银行 API 平台 [**Staq**](https://www.trystaq.com) **—** 的联合创始人/首席技术官。****

****想要获得免费的每周数据科学和创业见解吗？****

****你可以在 [LinkedIn](https://www.linkedin.com/in/admond1994/) 、 [Medium](https://medium.com/@admond1994) 、 [Twitter](https://twitter.com/admond1994) 、[脸书](https://www.facebook.com/admond1994)上和他联系。****

****[](https://www.admondlee.com/) [## 阿德蒙德·李

### 让每个人都能接触到数据科学。Admond 正在通过先进的社交分析和机器学习，利用可操作的见解帮助公司和数字营销机构实现营销投资回报。

www.admondlee.com](https://www.admondlee.com/)****