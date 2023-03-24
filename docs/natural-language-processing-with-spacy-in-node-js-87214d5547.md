# Node.js 中带空间的自然语言处理

> 原文：<https://towardsdatascience.com/natural-language-processing-with-spacy-in-node-js-87214d5547?source=collection_archive---------4----------------------->

![](img/1193e3a30234021e2b1c4b23e4b96ce7.png)

在浏览器中使用 JavaScript 的自然语言处理有很多用例，但不幸的是，在这个领域没有多少工具。为了超越正则表达式和字符串操作，您需要利用一个好的 NLP 框架。SpaCy 自称为工业级自然语言处理，是一个流行的工具包，但它是一个 Python 库。这对那些只使用 JavaScript 的人来说是一个障碍。

SpaCy 的特点是神经网络模型、集成单词向量、多语言支持、标记化、词性标注、句子分割、依存解析和实体识别。SpaCy 拥有 92.6%的准确率，并声称是世界上最快的语法分析器。

一个名为 spacy-nlp 的 npm 包旨在通过向 node.js 公开 spacy 的自然语言处理能力来跨越这一障碍。

*   句法剖析
*   名词解析
*   动词解析
*   形容词解析
*   命名实体解析
*   日期解析
*   时间解析

一些可用的助手方法有:

*   拆分大文本
*   删除重复的字符串
*   字符串中的前 n 个单词

这个包最初是由[华龙径](https://github.com/kengz)创建的，后来我用它来扩展功能。这里可以找到分叉版[。](https://github.com/jeffreyflynt/spacy-nlp) [Textalytic](https://www.textalytic.com) 也使用这个包与 Spacy 和其他 Python 自然语言处理框架接口。

# 装置

首先，推荐使用 Python3，但也可以使用 Python2。您需要首先通过 Python 包索引安装 SpaCy:

```
**python3 -m pip install spacy**
```

接下来，您需要通过 Python 安装 socketIO-client。该软件包用于促进内部跨语言交流。

```
**python3 -m pip install -U socketIO-client**
```

默认情况下，这个包使用 SpaCy 的 en_core_web_md 语言模型。要下载它，您需要运行以下命令:

```
**python3 -m spacy download en_core_web_md**
```

接下来，通过 NPM 安装软件包:

```
**npm install git+**[**https://github.com/jeffreyflynt/spacy-nlp.git**](https://github.com/jeffreyflynt/spacy-nlp.git)
```

在您的 JavaScript 文件中，您将导入包并使用公开 spacyIO 的 Python 客户端启动服务器。(默认端口为 6466)。

```
**const nlp = require("spacy-nlp");
const serverPromise = spacyNLP.server({ port: process.env.IOPORT });**
```

# 一些需要注意的事情

*   将空间加载到内存中可能需要 15 秒。
*   在应用程序的生命周期中，语言模型将留在内存中。(en_core_web_md —大小:91 MB)
*   如果你对应用程序的多个实例进行负载平衡，你会希望为每个`spacyNLP.server()`提供一个唯一的端口。

# 给我看一些例子

## 提取日期

假设您想从该文本中提取所有日期:

在这个共产主义国家已经饱受经济停滞之苦的时候，美国增加了对苏联的外交、军事和经济压力。1982 年 6 月 12 日，100 万抗议者聚集在纽约中央公园，呼吁结束冷战军备竞赛，特别是核武器。20 世纪 80 年代中期，新的苏联领导人米哈伊尔·戈尔巴乔夫(Mikhail Gorbachev)引入了改革(“重组”，1987 年)和开放(“开放”，约 1985 年)的自由化改革，并结束了苏联在阿富汗的参与。

输入您的文本并选择返回单词或只返回总数。返回的结果是:

*   1982 年 6 月 12 日
*   二十世纪八十年代中期
*   1987
*   1985

## 提取命名实体

或者您可能希望从该文本中提取所有命名实体:

*冷战的第一阶段始于 1945 年第二次世界大战结束后的头两年。苏联巩固了对东方集团国家的控制，而美国开始了挑战苏联权力的全球遏制战略，向西欧国家提供军事和财政援助(例如，支持希腊内战中的反共一方)并创建了北约联盟。*

这将返回:

*   冷战
*   第二次世界大战
*   苏维埃社会主义共和国联盟
*   东方集团
*   美国
*   苏维埃
*   西欧
*   希腊内战
*   北大西洋公约组织（North Atlantic Treaty Organization）

## 提取名词

你可以得到更细的粒度，提取一个文本中的所有名词。

【1941 年 6 月 22 日，欧洲轴心国发动了对苏联的入侵，开启了历史上最大的陆地战场，这使得轴心国，最关键的是德国国防军陷入了消耗战。第二次世界大战(通常缩写为 WWII 或 WW2)，又称第二次世界大战，是一场从 1939 年持续到 1945 年的全球性战争。

这将返回:

*   能力
*   侵略
*   陆地
*   戏院
*   战争
*   历史
*   战争
*   消耗
*   战争

更多的例子可以在 Github 上找到，更多的方法将很快出现。

# 结论

上述方法是 NLP 管道的起点。人们可以发挥创意，将日期/时间解析与命名实体提取结合起来，构建演员/玩家事件的时间线。或者用一个 web scraper 来聚合提及，这种可能性是无穷无尽的。

我计划增加更多的功能，如训练模型，文本分类，句子分割，等等。欢迎任何反馈。

感谢阅读这篇文章。如果您有任何问题或意见，可以通过 Twitter [@jeffrey_flynt](https://twitter.com/jeffrey_flynt) 联系我。