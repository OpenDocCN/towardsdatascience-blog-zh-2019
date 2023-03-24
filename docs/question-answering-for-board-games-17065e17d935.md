# 棋盘游戏问答

> 原文：<https://towardsdatascience.com/question-answering-for-board-games-17065e17d935?source=collection_archive---------17----------------------->

## 建立一个问答系统来使用棋盘游戏规则手册

![](img/3172d26ccd742ad127c50af14d056a2d.png)

Photo by [Christopher Paul High](https://unsplash.com/@christopherphigh?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/board-games?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

## 什么是问答系统？

在自然语言处理中，我们不仅试图解释单词之间的关系，而且希望建立对文本的机器理解。问答是一门特殊的学科，旨在建立自动回答问题的系统，这些问题可以从文本中的信息得到回答。目标是回答一系列问题(事实、列表、定义等)。)通过返回相应的文本部分。可以想象，拥有一个分析问题和文本语义的模型是相当复杂的。最近的发展语言建模(如 BERT)有助于使这样的任务更加精确和健壮。

当考虑问答系统时，需要理解的一个更广泛的范例是开放域和封闭域。顾名思义，开放领域寻求解决任何问题，而封闭领域专注于某一特定领域。想想维基百科上的信息差异吧，它涵盖了所有的内容，而伍基培百科上的信息都是关于星球大战的。各有各的效用，只是看你的目标是什么。

因此，问题回答系统听起来相当惊人，但请注意，像许多其他自然语言处理领域一样，它仍在发展，也有不足之处。我不会在这里详细说明，但是如果你感兴趣，我鼓励你阅读文章 [*关于语义网中问答挑战的调查。*](http://www.semantic-web-journal.net/system/files/swj1375.pdf) 它很好地概述了这些系统仍然面临的一些挑战。

## 有哪些应用？

创建和支持一个问答系统可能是一项严肃的工作，所以首先花点时间思考为什么要创建它们是很重要的。最实际的用途之一是创建聊天机器人。我知道，我知道，他们不是每个人都喜欢的机械助手。不过，它们确实有潜力帮助客户或处理特定领域的问题，否则就像许多人担心的那样，需要人们实际阅读一些东西。

## 制作你自己的

您可以想到许多场景，其中您有一些文本形式的信息，并希望使其易于搜索，所以让我们尝试一个想法。自然地，我的大脑去了最实际的用途，棋盘游戏规则！

我经常玩游戏，偶尔会玩一些更复杂的策略游戏。有时候像这样的游戏的问题是所有的工作都投入到实际上知道如何玩它们。人们很容易忘记不同规则的复杂性，忘记发牌的数量，甚至忘记如何知道游戏何时结束。还有一个个人的烦恼是，当你想做一个战略性的举动，但想确保它是合法的，所以你必须查找它，并给别人一个机会来判断你在做什么。

如果你有一个工具，任何人都可以问一个问题，找到特定的规则，而不必手动挖掘文本，会怎么样？听起来像一个问题回答系统的情况！

为此，我们将使用一个名为 [cdQa](https://github.com/cdqa-suite/cdQA) 的 Python 包。它有一套强大的功能来构建、注释和查询你自己的封闭领域问答系统。首先，您导入 cdQa 和一些不同的工具:

```
import pandas as pd
from cdqa.utils.converters import pdf_converter
from cdqa.utils.filters import filter_paragraphs
from cdqa.utils.download import download_model
from cdqa.pipeline.cdqa_sklearn import QAPipeline
```

`pdf_converter`:从 pdf 中提取文本(这个特定的转换器需要一个文件夹)。在本例中，我有一个名为“rules”的几个游戏的 rulebook pdfs 文件夹，将按如下方式实现:

```
boardgames_df = pdf_converter(directory_path=’rules/’)
```

`filter_paragraphs`:缩小范围，只在我们的文本中找到段落(本质上是在文本中寻找某个长度或其他参数)。要使用，您需要给它创建的整个数据帧:

```
boardgames_df = filter_paragraphs(boardgames_df)
```

`download_model`:找一个预先训练好的模特。我们将使用的模型是一个预训练的 BERT 模型，在 [SQuAD 1.1、](https://rajpurkar.github.io/SQuAD-explorer/)上进行微调，这是一个阅读理解数据集，通常用于测试和基准测试问答系统的表现。要下载名为“models”的文件夹中的商店:

```
download_model(model=’bert-squad_1.1', dir=’./models’)
```

`QAPipeline`:创建我们的 QA 系统。这将使用预先训练的模型使管道适合您的语料库。

```
cdqa_pipeline = QAPipeline(reader='models/bert_qa_vCPU-sklearn.joblib')
cdqa_pipeline.fit_retriever(boardgames_df)
```

厉害！我们现在准备测试我们的 QA 系统。为了询问和查看我们的结果，我们使用管道实例的`predict`方法，该方法返回一个包含答案、文档标题、段落和分数的元组。以下是一个搜索示例:

```
query = 'How do you win at Villianous?'
prediction = cdqa_pipeline.predict(query)
print('Query: {}\n'.format(query))
print('Answer: {}\n'.format(prediction[0]))
print('Paragraph: {}\n'.format(prediction[2]))*Query: How do you win at Villianous?

Answer: you must explore your character’s unique abilities and discover how to achieve your own story-based objective

Paragraph: To win, you must explore your character’s unique abilities and discover how to achieve your own story-based objective. Each Villain Guide will inspire you with strategies and tips. Once you’ve figured out the best way to play as one Villain, try to solve another. There are six different Villains, and each one achieves victory in a different way!™*
```

看起来不错！对于我的进一步实现，我实际上更喜欢只返回段落，因为对于某些规则来说，拥有额外的上下文是有帮助的。cdQa 还有一些其他非常好的特性，比如能够手动注释文本，以及轻松导出管道以在网站上使用，所以我鼓励进一步探索它。

我将继续自己的工作，但是要查看这里显示的代码，你可以[访问我的 GitHub 页面](https://github.com/jnawjux/qa_gamekit)。