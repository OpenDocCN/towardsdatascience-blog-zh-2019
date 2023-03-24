# 从在线评论中提取热门话题

> 原文：<https://towardsdatascience.com/extract-trending-topics-from-user-reviews-9d6c896451d7?source=collection_archive---------27----------------------->

## 文本中频繁项集挖掘的实验

![](img/a4a90a1af26b0a8f6c3f0cd2821543f5.png)

Photo by [Rock Staar](https://unsplash.com/@rockstaar_?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/restaurant-trends?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

顾客在选择餐馆时依靠在线用户评论。当一群评论者开始发布关于一个共同话题的评论时，评论可以保持趋势。这个话题可以是一家餐馆新增加的特色，一家新餐馆在城里开张，或者是关于菜单上一个受欢迎的食物。这个共同话题可以解释为趋势性话题。然后顾客开始追随新的潮流。我的方法是通过使用频繁项集挖掘技术来获得评论者每月的趋势主题。我要使用的数据集是来自 [Yelp 挑战赛 2019 的 Yelp 数据集。](https://www.yelp.com/dataset/challenge)

# 频繁项集挖掘与关联规则挖掘

频繁模式在数据集中反复出现。频繁项集由这些模式之一组成。频繁项集挖掘发现超过最小阈值的有趣的频繁项集。此阈值是一个度量值，它指示项集中包含的最小项数。频繁模式挖掘的概念是随着购物篮分析而发展起来的。购物篮分析可以作为当今最流行的例子。这里对一起销售的产品进行分析。简单地说，频繁项集挖掘显示哪些项在一个事务中一起出现。

关联规则挖掘就是寻找这些项目之间的关系。鉴于购物篮分析，它计算购买一个或多个产品后购买某个产品的概率。然而，当应用于问题时，这两个概念是一起的。正在使用的流行算法有 ***先验*、 *FP-Growth*** 和 ***Eclat*** 。

# 支持和信心

对于频繁项集挖掘，有两种主要的度量方法。他们是支持和信心。这两个度量评估模式有多有趣。支持度是绝对频率的量度，信心是相对频率。支持指示项目在所有事务中一起出现的次数。置信度表示有多少交易遵循某一规则。基本上，支持度是不考虑项目之间关系的频繁项目的度量，而置信度是项目之间关联规则的度量。以下是注释。

> Support(A-> B)= Support _ count(A**∩**B)
> 
> 置信度(A -> B) =支持计数(A**∩**B)/支持计数(A)

# 文本中的频繁项集挖掘

文本通常被称为非结构化数据。大多数时间频繁项集挖掘算法都可以应用于结构化数据。然而，文本可以转化为结构化数据，数据挖掘技术可以应用。为了输出数据的结构化版本，需要遵循几个预处理步骤。这种结构化数据可以分为两类:单个单词和多个单词。一个单词叫做单词包。包词和多重词的主要区别在于多重词保留了词与词之间的关系，从而保持了句子的语义。

另一方面，一个句子可以被看作是单词的集合。多个句子之间可以有常用词。类似地，段落由单词组成，并且多个段落具有频繁的单词集。这些常用词可以看作是一个频繁项集。Word 将成为项目集中的一个项目。所以频繁模式可以由多个单词组成。挖掘这些模式将给出句子或段落中常见单词的基本概念，而不是从文本中提取单个关键词。说到用户评论，这些常用词可以认为是趋势话题。该实验的主要目的是使用频繁项集挖掘来导出用户评论中的趋势主题。

# 履行

为了实现上述想法，选择来自特定餐馆的最新评论，并提取每个月的趋势主题。以下是筛选出最方便解决问题的餐厅的标准。

数据集有各种类型的业务。为了过滤掉餐馆，我在类别字段中选择了包含“餐馆”的企业。对于最新的评论，我选择了 2018 年发布的评论。当比较美国各州的餐馆数量时(**图 1** )，我们看到亚利桑那州的餐馆数量最多。然后通过计算和比较 2018 年每个餐厅的评论数，选出了“亚利桑那州”最受好评的餐厅。

![](img/38ca58a7f465450bf110a57901e77d46.png)

**Figure 1 : Number of restaurants in each state in the USA according to Yelp Data set (Yelp Challenge 2019)**

一开始，数据是从 Yelp 提供的 JSON 文件中提取的，并存储在关系数据库中以便于访问。我遵循的预处理步骤是转换成小写，删除标点符号，删除空格，标记单词和删除停用词。以下是代码片段。

```
**import** string**import** pandas **as** pd
**from** nltk.tokenize **import** word_tokenize
**from** sklearn.feature_extraction.stop_words **import** ENGLISH_STOP_WORDS**def** preprocess_reviews(reviews: pd.DataFrame):
    word_token_list = []
    **for** _, r **in** reviews.iterrows():
        formatted_review = r[**'review_text'**].lower()
        formatted_review = formatted_review.translate(str.maketrans(**""**, **""**, string.punctuation))
        formatted_review = formatted_review.strip()
        tokens = word_tokenize(formatted_review)
        result = [i **for** i **in** tokens **if not** i **in** ENGLISH_STOP_WORDS]
        word_token_list.append(result)

    **return** word_token_list;
```

用于检索趋势主题的算法是 ***先验*** ，支持度是 0.3。下面是代码片段。(我提到的文章位于本文末尾的参考资料部分，以了解更多详细信息)

```
**import** pandas **as** pd
**from** mlxtend.frequent_patterns **import** apriori
**from** mlxtend.preprocessing **import** TransactionEncoder**def** get_trending_words_from_reviews(preprocess_reviews: []):
    transaction_encoder = TransactionEncoder()
    transaction_encoder_array = transaction_encoder.fit(preprocess_reviews).transform(preprocess_reviews)
    df = pd.DataFrame(transaction_encoder_array, columns=transaction_encoder.columns_)

    frequent_itemset = apriori(df, min_support=0.3, use_colnames=**True**)

    **return** frequent_itemset
```

请注意，在这个实验中没有推导出关联规则。下面的代码解释了如何调用上述函数以及要传递的审核列表的格式。

```
reviews = pd.DataFrame([{**'review_text'**: **'good food'**}, {**'review_text'**: **'great food'**}, ])reviews = preprocess_reviews(reviews)
frequent_itemset = get_trending_words_from_reviews(reviews)
print(frequent_itemset)
```

# 结果和讨论

下表(**表 1** )显示了上述实验的结果。

![](img/11300d202a433f15856dd258f52bf711.png)

**Table 1 : Frequent words derived from frequent itemset mining for each month using reviews of most reviewed restaurant in Arizona by year 2018**

所选餐厅在类别字段中描述自己为'*三明治、意大利早餐&早午餐、美式(传统)'*。我们可以看到“肉丸”、“酱”和“意大利面”这些词在一月份很流行。在二月，人们只谈论这三个术语中的两个。它们是“肉丸”和“酱”。当谈到三月时,“酱”和“意大利面”都消失了，而“肉丸”这个词幸存了下来。尽管这三个项目都是今年年初的流行趋势，但只有一个项目可以继续流行。此外，调查结果显示，术语“肉丸”已经成为全年的热门话题。“意大利面”在八月又成了一个热门话题。我们可以观察到的另一件事是，年底的复习次数一直在减少(**图 2** )。

![](img/1779b4c8a1fa7e6e30aacfc4fd01e658.png)

Figure 2 : Monthly review count of most reviewed restaurant in Arizona by year 2018

这可以解释为餐馆正随着时间的推移而失去它的名声。原因可能是，尽管餐馆的服务和位置得到了好评，但对于常客来说，没有什么新鲜和令人兴奋的东西可以分享。如果一家餐馆能推出新的食品，并提高逐渐消失的食品的质量，它就能挽回声誉。

每个餐厅的结果都不一样，应该据此做出解释。可以通过增加更多的预处理步骤来改善结果。在这种情况下，词干、词汇化和词性标注等概念会有所帮助。此外，可以通过将该方法分别应用于好评和差评来改进预测。

参考

[](https://iopscience.iop.org/article/10.1088/1757-899X/434/1/012043) [## 文本频繁项集挖掘的概念

### 频繁项集挖掘是一种流行的具有频繁模式或频繁项集的数据挖掘技术。

iopscience.iop.org](https://iopscience.iop.org/article/10.1088/1757-899X/434/1/012043) [](https://medium.com/analytics-vidhya/association-analysis-in-python-2b955d0180c) [## Python 中的关联分析

### Python 中基于 Apriori 算法的频繁项集挖掘

medium.com](https://medium.com/analytics-vidhya/association-analysis-in-python-2b955d0180c) [](https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/) [## Python 中基于 Apriori 算法的关联规则挖掘

### 关联规则挖掘是一种识别不同项目之间潜在关系的技术。举一个例子…

stackabuse.com](https://stackabuse.com/association-rule-mining-via-apriori-algorithm-in-python/)