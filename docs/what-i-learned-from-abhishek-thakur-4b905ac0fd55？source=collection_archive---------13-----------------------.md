# 我从(两届)卡格大师阿布舍克·塔库尔那里学到了什么

> 原文：<https://towardsdatascience.com/what-i-learned-from-abhishek-thakur-4b905ac0fd55?source=collection_archive---------13----------------------->

## 从 Abhishek Thakur 的 NLP 内核中汲取灵感

![](img/4fd1ef8d62670db6e9098924ef619122.png)

Photo by [Georgie Cobbs](https://unsplash.com/@georgie_cobbs?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

## **快速简历**

Abhishek Thakur 曾在遍布德国的多家公司担任数据科学家，他在 NIT Surat 获得了电子工程学士学位，在波恩大学获得了计算机科学硕士学位。目前，他拥有挪威 boost.ai 首席数据科学家的头衔，这是一家“专门从事对话式人工智能(ai)的软件公司”。但我印象最深的是阿布舍克的影响力。

你可以点击这里访问他的简介[。以下是他所获荣誉的快照:](https://www.kaggle.com/abhishek)

*   比赛特级大师(17 枚金牌和空前的世界排名第三)
*   内核专家(他属于 Kagglers 的前 1%之列)
*   讨论特级大师(65 枚金牌和空前的世界排名第二)

我想看看 Abhishek 的教程，[在 Kaggle](https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle) 上接近(几乎)任何 NLP 问题。我选择 Abhishek 的这个内核是因为我自己一直在试图学习更多关于自然语言处理的知识，我怎么能拒绝学习 Kaggle 的万圣节主题[幽灵作者](https://www.kaggle.com/c/spooky-author-identification/data)数据集呢？

## **Abhishek 的自然语言处理方法**

我强烈建议您阅读这篇关于内核的文章。如果你真的想更牢固地掌握 NLP 或一般的数据科学，确保你理解 Abhishek 的每一行代码，在你浏览他的内核时自己写。

只是我们不要忘记——我们的任务是确定**作者**(EAP——埃德加·爱伦·坡；HPL—h . p . love craft；MWS——玛莉·渥斯顿克雷福特·雪莱)。

**1。探索数据和理解问题**

在导入必要的 Python 模块和数据之后，Abhishek 对数据调用 head()方法来查看前五行是什么样子的。由于 Abhishek 是专家，这是一个 NLP 问题，与涉及数字数据的问题相比，**探索性数据分析**(你最常看到的是 EDA)是肤浅的。数据科学新手可能会从更彻底的 EDA 中受益。对数据的深入研究可以发现任何缺失的值，让您知道需要清理多少数据，并帮助您在问题的后期做出建模决策。

Abhishek 还提醒我们，我们正在处理一个多类文本分类问题。不要忘记我们正在努力实现的目标，这总是一个好主意！他注意到 Kaggle 将使用什么评估标准来对提交的内容进行评分。对于这场比赛，Kaggle 使用**多类对数损失**来衡量提交模型的性能。理想情况下，我们的多类分类模型的对数损失为 0。如果你感兴趣，这里有更多关于[原木损耗](http://wiki.fast.ai/index.php/Log_Loss)的内容。

**2。预处理**

接下来，Abhishek 使用 scikit-learn 中的 LabelEncoder()方法为每个作者分配一个整数值。通过用整数值(0，1，2)对 **author** 列中的值的文本标签进行编码，Abhishek 使数据更易于他的分类模型理解。

对**作者**标签进行编码后，Abhishek 使用 scikit-learn 中的 train_test_split 将数据分成训练集和验证集。他选择 90:10 的训练/验证分割(Python 数据科学中最常用的分割通常在 70:30 到 80:20 之间)。所以他打算用数据集中 90%的句子来训练模型，然后他会用剩下的 10%的数据来评估他的模型的准确性。

**3。建立模型**

在创建他的第一个模型之前，Abhishek 对数据使用了 TF-IDF(术语频率——逆文档频率)。TF-IDF 将对出现在**文本**列的句子中的单词进行加权。因此，TF-IDF 将帮助我们理解当我们试图确定哪个作者写了一个特定的句子时，哪些词是重要的——诸如“the”之类的词对于分类任何作者来说都不重要，因为“the”出现频率很高，不会透露太多信息，但是，例如，“Cthulhu”之类的词在分类 H.P. Lovecraft 所写的句子时将非常重要。更多关于 TF-IDF 的信息可以在[这里](https://en.m.wikipedia.org/wiki/Tf%E2%80%93idf)和[这里](https://www.quora.com/How-does-TfidfVectorizer-work-in-laymans-terms)找到。

对数据运行这个 TF-IDF 是**特征提取**的一种形式。在这里，我们需要推导出某种重要的预测因子或数据特征，来帮助我们找出哪个作者写了一个特定的句子。使用 TF-IDF，我们有了一个单词重要性的统计度量，可以帮助我们预测句子的作者。

在对训练集和验证集拟合 TF-IDF 后，Abhishek 准备了一个逻辑回归模型。如果这种类型的分类模型对您来说是新的，请在继续之前阅读[此](/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8)。在拟合了逻辑回归模型之后，Abhishek 计算了他的逻辑回归模型的对数损失(回想一下，他在内核的开头附近编写了多类对数损失函数)。多类对数损失函数返回逻辑回归模型的对数损失值 *0.626* 。虽然拟合 TF-IDF 和逻辑回归模型给了我们一个良好的开端，但我们可以改善这一对数损失评分。

**4。模型调整**

因此，我们对 0.626*的对数损失分数不满意，希望优化这一评估指标。从这里，我们可以采取多条路线，这正是 Abhishek 所做的。在我们探索和预处理了我们的数据之后，我们剩下了许多不同的特征提取和模型拟合的组合。例如，Abhishek 使用字数来进行特征提取，而不是 TF-IDF。使用这种特征提取技术，他的逻辑回归模型的对数损失分数从 *0.626* 提高到*0.528*——这是一个巨大的 *0.098* 的提高！*

# **总结**

由于 Abhishek 的内核从这一点开始变得越来越详细，我将让他来解释其他分类模型。

我们讨论的内容如下:

*   **EDA** :如果我们想要理解数据集，探索性数据分析是至关重要的，当我们开始构建模型时，EDA 可以节省我们的时间
*   **多类分类问题**:这类问题要求我们预测哪些观察值属于哪个类，其中每个观察值可能属于三个或更多类中的任何一类
*   预处理:在建立任何模型之前，我们必须对数据进行预处理。在这个例子中，为了我们的模型，我们需要使用 LabelEndcoder()将文本标签转换成整数值
*   **特征提取**:每当我们有一个原始数据的数据集(在我们的例子中是句子摘录)时，我们将需要导出一些预测器，帮助我们确定如何对我们的观察进行分类。Abhishek 向我们展示了如何使用 TF-IDF 和字数统计

从这里开始，由我们来提取具有高预测能力的特征，选择与问题匹配的模型，并优化我们关心的指标。不要害怕弄脏自己的手，尝试几种模型——你很可能会找到一种模型，通过更多的实验来优化你的评估指标。我希望读完这篇文章后，你能更好地理解如何处理 NLP 问题，并且你也能欣赏 Abhishek 的工作。

## **附录**

[阿布舍克的卡格尔简介](https://www.kaggle.com/abhishek)

[Abhishek 的 NLP 内核](https://www.kaggle.com/abhishek/approaching-almost-any-nlp-problem-on-kaggle)

[怪异作者数据集](https://www.kaggle.com/c/spooky-author-identification/data)

[什么是原木损耗？](http://wiki.fast.ai/index.php/Log_Loss)

[什么是 TF-IDF？](https://en.m.wikipedia.org/wiki/Tf%E2%80%93idf)

[TF-IDF 通俗地说](https://www.quora.com/How-does-TfidfVectorizer-work-in-laymans-terms)

[什么是逻辑回归？](/building-a-logistic-regression-in-python-step-by-step-becd4d56c9c8)

这里是本文中提到的 Abhishek 的所有代码。我想重申一下，这是我自己的工作——这个要点旨在帮助初学者跟随 Abhishek 的 NLP 教程。

Credit to Abhishek Thakur for this NLP tutorial