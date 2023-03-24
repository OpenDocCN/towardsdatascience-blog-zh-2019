# 对话式人工智能:设计和构建一个上下文人工智能助手

> 原文：<https://towardsdatascience.com/conversational-ai-design-build-a-contextual-ai-assistant-61c73780d10?source=collection_archive---------3----------------------->

![](img/779403613fc1d8f28cabd1f44383ea4c.png)

Photo by [Alex Knight](https://unsplash.com/@agkdesign?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

虽然对话式人工智能自 20 世纪 60 年代以来就已经存在，但近年来它正经历一场新的焦点。虽然我们仍然处于智能对话式人工智能的设计和开发的早期，但谷歌非常正确地宣布，我们正在从移动优先转向人工智能优先的世界，在这个世界中，我们希望技术能够自然对话，考虑上下文，并具有进化能力。换句话说，我们期望技术能够学习和进化。

今天，大多数聊天机器人可以处理简单的问题，并根据基于规则的对话处理，以预先构建的响应进行响应。例如，如果用户说 X，用 Y 来响应；如果用户说 Z，调用 REST API，依此类推。然而，在这个节骨眼上，我们对对话有更多的期待。我们想要超越回答简单问题或发送推送通知的上下文助手。在这个系列中，我将带你经历一个设计、开发和部署情境化人工智能助手的过程，这个助手设计策划旅行体验。

首先，让我们来谈谈上下文助手的成熟度水平，从它们的能力来解释:

# 对话式人工智能成熟度级别

一级成熟度:在这个级别，聊天机器人本质上是一个传统的通知助手；它可以用预先构建的响应来回答问题。它可以向你发送某些事件的通知，或者提醒你明确表示感兴趣的事情。例如，一个 1 级旅行机器人可以为你提供一个预订旅行的链接。

![](img/ddb6d85e1053e8b02f2cf798358cefc7.png)

2 级成熟度:在这个级别，聊天机器人可以回答常见问题，但也能够处理简单的后续问题。

![](img/c145851ea447beac20af76e973eec064.png)

第三级成熟度:在这一级，上下文助手可以与你进行灵活的互动，并提供比预先构建的答案更多的答案，因为它知道如何响应意外的用户话语。此时，助手也开始理解上下文。例如，旅行机器人将能够带你走过几个受欢迎的目的地，并做出必要的旅行安排。

![](img/b82ac9f55ea98e50ff01a4b36ccbf7d9.png)

第 4 级成熟度:在这一级，上下文助手已经更了解你了。它会记住您的偏好，并可以提供个性化、情境化的建议或“轻推”,以更加主动地提供护理。例如，在你着陆后，助理会主动伸出手帮你订一辆车。

![](img/8ebb71b4fc6bc373748824a50ad6164f.png)

第 5 级及以上:在这一级，上下文助手能够监控和管理许多其他助手，以便运行企业运营的某些方面。他们能够针对特定的旅游体验开展促销活动，根据历史趋势更有效地锁定特定的客户群，提高转化率和采用率，等等。

# 对话式人工智能源于自然语言处理…

自然语言处理(NLP)是人工智能的一种应用，它使计算机能够处理和理解人类语言。机器学习的最新进展，更具体地说，它的子集深度学习，使计算机更好地理解自然语言成为可能。这些深度学习模型可以分析大量文本，并提供文本摘要、语言翻译、上下文建模和情感分析等功能。

自然语言理解(NLU)是将自然语言转换成结构化数据的自然语言处理的一个子集。NLU 能够做两件事——意图分类和实体提取。

当我们阅读一个句子时，我们会立即理解该句子背后的含义或意图。意图是用户试图传达或实现的东西。意图分类是一个两步过程。首先，我们向 NLU 模型输入带标签的数据，这些数据提供了已知意图的列表以及与这些意图相对应的例句。一旦经过训练，该模型能够将它看到的新句子分类为预定义的意图之一。实体抽取是识别给定文本中关键信息的过程。时间、地点和人名等都提供了与意图相关的附加上下文和信息。意图分类和实体提取是对话式人工智能的主要驱动因素。

出于本文的目的，我们将使用 Rasa，这是一个开源堆栈，提供了构建上下文 AI 助手的工具。Rasa 堆栈中有两个主要组件将帮助我们构建旅行助手——Rasa NLU 和 Rasa core。

Rasa NLU 提供意图分类和实体提取服务。Rasa 核心是堆栈的主要框架，提供由机器学习支持的对话或对话管理。假设已经训练了 NLU 和核心组件，让我们看看 Rasa 堆栈是如何工作的。

让我们用这个对话的例子:

![](img/4b9ab7c85e3930daff44634c5c939169.png)

NLU 组件标识用户打算参与基于假期的旅行(意图分类)，并且他或她是唯一一个进行此旅行的人(实体提取)。

核心组件负责控制对话流。基于来自 NLU 的输入、对话的当前状态及其训练的模型，核心组件决定下一个最佳的动作过程，该动作过程可以是向用户发送回复或者采取动作。Rasa 基于 ML 的对话管理是上下文感知的，不依赖于硬编码规则来处理对话。

# 安装和设置

现在，让我们安装 Rasa，并开始为我们的旅行助理创建初始训练数据集。

Rasa 有两种设置方式。您可以使用 python/pip 在本地机器上安装 Rasa 堆栈，也可以使用 docker 通过预配置的 docker 映像来设置 Rasa 堆栈。我们将使用 python 和 pip 安装 Rasa 堆栈。

如果您的机器上没有安装 python，您可以使用 [Anaconda](https://www.anaconda.com/distribution/) 来设置它。请注意，您需要 python 3.6.x 版本来运行 Rasa 堆栈。python 的最新版本(本文发布时是 3.7.x)并不完全兼容。

运行以下命令安装 Rasa 核心:

```
pip install -U rasa_core
```

通过运行以下命令安装 Rasa NLU:

```
pip install rasa_nlu[tensorflow]
```

现在让我们通过克隆 Rasa 提供的[启动包](https://github.com/RasaHQ/starter-pack-rasa-stack)来实现我们的应用程序:

```
git clone [https://github.com/RasaHQ/starter-pack-rasa-stack.git](https://github.com/RasaHQ/starter-pack-rasa-stack.git) travel-bot
```

克隆完成后，运行这些命令来安装实体提取所需的包和 spaCy 英语语言模型。

```
pip install -r requirements.txt \
  && python -m spacy download en
```

在这一点上，我们有一切我们需要开始发展我们的旅行助理。让我们看看在搭建过程中创建的文件夹结构和文件。

“domain.yml”文件描述了旅行助手的域。它指定了助手理解和操作的意图、实体、插槽和响应模板的列表。让我们更新文件，添加一组与我们的旅游领域相对应的初始意图。这里有一个片段:

```
intents:
  - greet
  - request_vacation
  - affirm
  - inform
...entities:
  - location
  - people
  - startdate
  - enddate
...slots:
  people:
    type: unfeaturized
  location:
    type: unfeaturized
...actions:
  - utter_greet
  - utter_ask_who
  - utter_ask_where
  - utter_confirm_booking
...templates:
  utter_ask_who:
    - text: "Fun! Let's do it. Who's going?"
  utter_ask_where:
    - text: "Perfect. Where would you like to go?"
...
```

“data/nlu_data.md”文件用一组例子描述了每一个意图，然后将这些例子提供给 nlu 拉萨进行培训。这里有一个片段:

```
## intent:request_vacation
- I want to go on vacation
- I want to book a trip
- Help me plan my vacation
- Can you book a trip for me?
...## intent:inform
- just me
- we're [2](people)
- anywhere in the [amazon](location)
- [Paris](location)
- somewhere [warm](weather_attribute)
- somewhere [tropical](weather_attribute)
- going by myself
...
```

“data/stories.md”文件为 Rasa 提供了用户和旅行助手之间的示例对话，可以用来训练其对话管理模型。这里有一个片段:

```
## vacation happy path 1
* request_vacation
    - utter_ask_who
* inform{"people": "1"}
    - utter_ask_where
* inform{"location": "paris"}
    - utter_ask_duration
* inform{"startdate": "2019-10-03T00:00:00", "enddate": "2019-10-13T00:00:00"}
    - utter_confirm_booking
* affirm
    - goodbye
...
```

Rasa 在配置 NLU 和核心组件方面提供了很大的灵活性。现在，我们将为 nlu 使用默认的“nlu_config.yml ”,为核心模型使用“policies.yml”。

运行以下命令来训练拉莎·NLU:

```
make train-nlu
```

运行以下命令来训练 Rasa 核心:

```
make train-core
```

我们现在可以通过命令行运行服务器来测试 Rasa:

```
make cmdline
```

Rasa stack 提供挂钩，将我们的助手连接到各种前端通道，如 Slack 和脸书。让我们将我们的旅行助手配置和部署到 Slack。

# 配置时差

让我们从在 [slack](https://api.slack.com/apps?new_app=1) 中创建一个新应用开始

*   在功能下，进入“OAuth & Permissions”，添加权限范围，如“chat:write:bot ”,并保存您的更改
*   转到“机器人用户”，添加一个机器人用户并保存您的更改
*   返回“OAuth & Permissions”并将应用程序安装到您的工作区
*   将“Bot 用户 OAuth 访问令牌”复制到您的工作区的令牌下
*   回到本地机器上的“travel-bot”文件夹，创建一个新的“credentials.yml”文件。将令牌粘贴到文件中，如下所示:

```
slack:
  slack_token: "xoxb-XXXXXXXXXXX"
```

我们需要把这些凭证交给 Rasa。为了让我们的生活更轻松，让我们更新“travel-bot”文件夹下的“Makefile ”,添加一个名为“start”的新命令

```
...
start:
    python -m rasa_core.run \
    --enable_api \
    -d models/current/dialogue \
    -u models/current/nlu \
    -c rest --cors "*" \
    --endpoints endpoints.yml \
    --credentials credentials.yml \
    --connector slack
```

> 警告:根据您的要求，在将您的 bot 部署到生产环境之前更新“cors”设置。

让我们通过调用以下命令来启动服务器:

```
make start
```

我们的旅行助手现在运行在本地端口 5005 上，它被配置为通过 REST APIs 与 slack 对话。我们需要使这些端点对外部世界可达。让我们使用 [ngrok](https://ngrok.com/) 让我们的服务器对 Slack 可用。Ngrok 有助于建立一个到我们本地服务器的安全通道，以便快速开发和测试。安装 ngrok 后，在新的命令行终端中运行以下命令:

```
ngrok http 5005
```

记下 ngrok 提供的“https”URL。回到 Slack 应用程序管理用户界面:

*   启用“事件订阅”。在“请求 URL”文本框中，粘贴您的 ngrok URL 并将“/webhooks/slack/webhook”添加到该 URL 的末尾。通过订阅类似“message.im”的 bot 事件来完成事件订阅，并保存您的更改
*   或者，启用“交互式组件”并传递您在上一步中使用的相同请求 URL。请务必保存您的更改

至此，我们已经完全配置好了我们的 bot 助手来与 Slack 交互。为了测试旅行助手，放松你新创建的旅行机器人。

![](img/cf862c50ca13ffb79d73ab5dbdd444ca.png)

# 下一步是什么？

在本系列的下一部分中，我们将深入研究我们的 NLU 管道、自定义组件，如 Google 的 BERT 和递归嵌入对话策略(REDP)，以及上下文、注意力和非线性对话等概念。