# 使文本上升

> 原文：<https://towardsdatascience.com/making-text-ascent-1ee428b7a13d?source=collection_archive---------27----------------------->

## 我如何构建和部署机器学习 web 应用程序

我经常发现自己在阅读一篇文章，比如说关于数据科学的文章，并且想知道，在哪里可以读到关于这个主题的更简单的文章？当一个朋友在 LinkedIn 上发布类似问题时，我意识到我不是唯一一个。她问如何在最简单和最复杂之间的特定范围内找到文章。我意识到我们没有一个简单的系统来进行这种类型的搜索，除了手动阅读来寻找合适的信息。

## 商业理解

基于我对网络搜索的兴趣，我创建了 Text Ascent，这是一个网络应用程序，它使用无监督的 ML 来帮助用户基于文本复杂性发现内容。我希望 Text Ascent 可以成为一种工具，用于在我们学习旅程的所有阶段搜索内容。我为 Text Ascent 设定的核心目标是让人们之间更容易接触到感兴趣的小众话题。

![](img/feb3632808592cfb97e468eb5c74c72f.png)

Photo By Ted Bryan Yu from Unsplash

## 数据理解

我使用了 [Wikipedia-API](https://pypi.org/project/Wikipedia-API/) ，这是一个为 [Wikipedia](https://www.wikipedia.org/) 的 API 设计的 python 包装器，用来收集从艺术到科学等主题的文章标题。然后，我运行了一个数据收集函数(scrape_to_mongodb.py ),该函数将这些标题和 11k+文章的摘要、全文和 URL 收集到一个 mongodb 数据库中。我排除了全文少于 300 个单词的文章，因为维基百科中有像“音乐文件”这样的条目不符合我的模型的目的。
参见[数据采集笔记本](https://github.com/sherzyang/text-ascent/blob/master/collect_data.ipynb) & [数据探索笔记本](https://github.com/sherzyang/text-ascent/blob/master/data_exploration.ipynb)。

## 数据准备

从 Wikipedia-API 包装器返回的内容不需要进一步清理。我确实需要确保当内容显示在 web 应用程序上时，html 被读取为 JSON，以避免向用户显示回车。我使用 [textstat 包](https://pypi.org/project/textstat/)的 Flesch-Kincaid 等级给每个文档的全文打分。

这些文件保存在 AWS S3 存储桶中，以允许访问 web 应用程序。参见[数据准备笔记本](https://github.com/sherzyang/text-ascent/blob/master/data_preparation.ipynb)。

## 建模

当前模型使用语料库向量和用户输入向量中的前 20 个重要特征之间的余弦距离来将相似内容从库中返回给用户输入。使用 TF-IDF 矢量器创建模型特征。TF-IDF 矢量器拆分语料库文档中的单词，删除停用词，并计算每个文档中每个单词的词频，根据单词在语料库中出现的频率进行调整。换句话说，不常用的词比常用的词更重要。

## 复制这个模型

*   获取感兴趣的文档列表，并格式化成类似于`clean_df`的数据帧。使用 TextStat 获得文本难度分数。*我在 AWS S3 上的例子:* [*clean_df*](https://text-ascent.s3-us-west-2.amazonaws.com/clean_df.pkl)
*   使你的语料库适合你的矢量器(从训练集中学习词汇和 idf)，它是你的 df 中的文本系列*我在 AWS S3 上的例子:* [*矢量器*](https://text-ascent.s3-us-west-2.amazonaws.com/vectorizer.pkl)
*   使用矢量器转换功能(将文档转换为文档术语矩阵)来创建您的语料库向量*我在 AWS S3 上的示例:* [*语料库向量*](https://text-ascent.s3-us-west-2.amazonaws.com/corpus_vectors.pkl)
*   克隆此存储库
*   在`traverse_flask`目录中，创建一个名为`data`的空子目录。
*   用`$ export FLASK_APP=app $ flask run` 在终端运行`traverse_flask`中的 flask，实现 flask app。这个瓶子`app.py`接受`functions.py`的功能。调整函数以改变后端的数据管道。调整`static/templates/index.html`中的 brython，改变数据反映给用户的方式。

参见[模型功能](https://github.com/sherzyang/text-ascent/blob/master/traverse_flask/functions.py)。

## 估价

如果用户能够发现与他们已经阅读的内容相关的不同阅读难度的内容，那么这个产品就是成功的。用户满意度、重复使用、网络应用流量和应用分享是我用来评估 Text Ascent 成功的指标。在使用部署在 web 应用程序上的模型之前，我评估了 4 个模型:

*   模型 1:使用 TextStat、Gensim 和 Spacy。
*   模型 2:使用具有 10 个主题的潜在狄利克雷分配(LDA)主题建模，然后将用户内容分类到一个主题中。
*   模型 3:使用 2000 维的 TextStat 和 TF-IDF 矢量器。
*   模型 4:使用具有前 20 个特性的 TextStat 和 TF-IDF 矢量器。

每一次迭代都是为了使结果内容更类似于用户输入的内容。

## 未来建模

我还想将预训练的神经网络与我当前的 TFIDF 矢量化进行比较，看看返回内容的质量是否有所提高。改进将通过一个简单的手动评分系统添加到网络应用程序的用户反馈来衡量。参见[评测笔记本](https://github.com/sherzyang/text-ascent/blob/master/evaluation_notebook.ipynb)。

## 部署

Text Ascent 已经作为一个支持 flask 的 web 应用【traverse.sherzyang.com 部署在 EC2 实例上(目前没有运行)。该应用程序使用 brython 在 python 函数和 html 之间进行交互。下面是来自网络应用的两张图片。给定任何用户输入文本，该模型将从库中输出相关文章，标题中有链接到完整长度的文章。用户可以从较简单的内容滚动或遍历到较复杂的内容，表格会相应地更新。

![](img/9ff9ad9634721d84c9e2e5bdf74d4356.png)![](img/d36b534a44586040af314b13a09af741.png)

## 未来迭代

作为我对搜索和我们的[一次性答案新世界](https://www.wired.com/story/amazon-alexa-search-for-the-one-perfect-answer/)——谢谢 Alexa、Siri 和 Google Home——的兴趣的一部分，我计划将文本提升部署为亚马逊 Alexa 的一项技能。这项技能将允许用户“滚动”或“遍历”某个主题从简单到复杂的摘要，就像告诉 Alexa 将歌曲播放得更大声或更小声一样。我相信在内容上创造选择会以积极的方式让我们超越一次性答案的世界。

此外，我渴望扩大语料库，以包括来自古滕贝格项目和其他项目的书籍。如果你想看到一些内容被添加到当前的维基百科文章库中，请在 LinkedIn 上给我发消息。我在亚马逊或 Goodreads 上见过几个给一本书的阅读难度打分的网络扩展( [Read Up](http://www.arialvetica.com/readup/) 就是一个很好的例子)。这些产品激励我为将来的文本提升开发一个无语料库的搜索功能。我认为当 Text Ascent 可以返回 Google 或 Bing web search API 支持的内容时，它会变得更加有用。

## 信用

*   [Werlindo Mangrobang，可视化和 Web 应用部署](/plotly-express-yourself-98366e35ad0f)
*   [凯莉·梁，布里森](https://github.com/kayschulz/travel_destination_recommendation/blob/master/travel_destination_recommendation/recommend.py)