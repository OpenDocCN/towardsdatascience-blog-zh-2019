# 用 5 行 Python 代码抓取并总结新闻文章

> 原文：<https://towardsdatascience.com/scrape-and-summarize-news-articles-in-5-lines-of-python-code-175f0e5c7dfc?source=collection_archive---------6----------------------->

## *好程序员写的代码，伟大的先搜 github。*

![](img/85474ebd517df827b0a8649e99942959.png)

Photo by [Thomas Charters](https://unsplash.com/@lifeofteej?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

想从一群只做机器学习和可视化的数据科学家中脱颖而出？然后，您可以通过收集自己的数据集，而不是使用 Kaggle 中过时的 CSV 文件，提前一步开始。

在这篇文章中，我将向你展示如何以一种统一的方式从许多来源收集大量的新闻数据。因此，你将使用 [newspaper3k](https://github.com/codelucas/newspaper) 自动提取结构化信息，而不是花费数月时间为每个新闻网站编写脚本。

安装软件包:

```
$ pip install newspaper3k
```

现在让 newspaper3k 把文章刮下来，摘抄信息，给我们总结一下。

```
>>> from newspaper import Article>>> article = Article('https://www.npr.org/2019/07/10/740387601/university-of-texas-austin-promises-free-tuition-for-low-income-students-in-2020')>>> article.download()>>> article.parse()>>> article.nlp()
```

这是所有的乡亲。5 行代码，包括包导入。

如果您执行了之前的所有步骤，并且没有出现错误，您应该可以访问以下信息:

```
>>> article.authors['Vanessa Romo', 'Claire Mcinerny']>>> article.publish_datedatetime.datetime(2019, 7, 10, 0, 0)>>> article.keywords['free', 'program', '2020', 'muñoz', 'offering', 'loans', 'university', 'texas', 'texasaustin', 'promises', 'families', 'lowincome', 'students', 'endowment', 'tuition']
```

关于文本本身，您可以选择访问全文:

```
>>> print(article.text)University of Texas-Austin Promises Free Tuition For Low-Income Students In 2020toggle caption Jon Herskovitz/ReutersFour year colleges and universities have difficulty recruiting...
```

除此之外，您还可以获得内置摘要:

```
>>> print(article.summary)University of Texas-Austin Promises Free Tuition For Low-Income Students In 2020toggle caption Jon Herskovitz/ReutersFour year colleges and universities have difficulty recruiting talented students from the lower end of the economic spectrum who can't afford to attend such institutions without taking on massive debt.To remedy that — at least in part — the University of Texas-Austin announced it is offering full tuition scholarships to in-state undergraduates whose families make $65,000 or less per year.The endowment — which includes money from oil and gas royalties earned on state-owned land in West Texas — more than doubles an existing program offering free tuition to students whose families make less than $30,000.It also expands financial assistance to middle class students whose families earn up to $125,000 a year, compared to the current $100,000.In 2008, Texas A&M began offering free tuition to students whose families' income was under $60,000.
```

对于一个内置功能来说还不错。

要从所有功能中获益，包括自动订阅杂志和访问热门话题，请参考[官方文档](https://newspaper.readthedocs.io/en/latest/)。

使用 newspaper3k，您可以收集独特的数据集来训练您的模型。更重要的是，在模型准备好之后，您将有一个真实的数据馈送，因此您也将能够看到真实的性能。

首先定义一个问题，然后才搜索数据，而不是相反。试着成为一个真正的问题解决者，思考你的模型如何解决真正的商业问题，因为这是你将会得到报酬的。

如果你喜欢这篇文章，我想强调的是，你应该读一下启发我的那篇文章。