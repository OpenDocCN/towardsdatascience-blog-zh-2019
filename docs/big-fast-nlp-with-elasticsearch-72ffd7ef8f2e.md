# 具有弹性搜索的大型快速人在回路 NLP

> 原文：<https://towardsdatascience.com/big-fast-nlp-with-elasticsearch-72ffd7ef8f2e?source=collection_archive---------2----------------------->

> **第一部分:关键词工厂**TL；dr:如果你 1)在 Elasticsearch 中存储你的数据 2)使用`[clio-lite](https://github.com/nestauk/clio-lite#keywords-getting-under-the-hood)`包中的`clio_keywords`函数，指向你的 Elasticsearch 端点 3)在 Flask 应用中托管它，[比如这个](https://github.com/nestauk/arxlive/blob/b76fda906901d1f3afab1cdb9b24cddfb717d9d2/arxlive/views.py#L10)。
> 
> **第二部分:上下文搜索引擎**TL；dr:如果你 1)将你的数据存储在 Elasticsearch 中 2)使用`[*clio-lite*](https://github.com/nestauk/clio-lite)`托管一个 lambda API 网关 3)用你自己的前端询问它，或者使用像 [searchkit](http://www.searchkit.co/) 这样的开箱即用的东西，你可以使[成为其中之一](https://arxlive.org/hierarxy)

许多 NLP 数据科学家目前的日常工作范式是打开笔记本电脑，启动 Python 或 R，制作一些模型，总结一些结论。这对于探索性分析来说非常有效，但是如果你需要把一个人(比如一个专家，你的老板，甚至你自己)放入循环中，这可能会变得非常慢。在这篇由两部分组成的博客中，我将试图让你相信有更大、更快的方法来实现 NLP。

![](img/813802a4e6db0c9c4ae30163407c1f0a.png)

Big fast human-in-the-loop NLP (Photo by [Stephen Hateley](https://unsplash.com/@shateley?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/rollercoaster?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText))

# 人在回路 NLP 和我

在我[Nesta](https://www.nesta.org.uk/team/joel-klinger/)[的日常工作](https://www.nesta.org.uk/)中，我开发工具和基础设施，让人们能够做出更好的决策，让人们能够利用最新的数据做出这些决策。我们为地方、国家和国际决策者和资助者提供工具，他们依赖于与科学、技术和社会的最新创新保持同步。由于这些人对他们的决定负责，这通常排除了采用黑盒程序的工具。“人在回路中的自然语言处理”是我们解决这些需求的方式，尤其是因为非结构化文本数据是最丰富、最可用和最新的数据形式之一。

# 弹性搜索

数据工程师、数据库管理员和 devops 工程师都应该熟悉“elastic stack”(elastic search 是其核心)，它是存储和分析日志文件或构建搜索引擎的首选技术，尽管他们中的许多人可能不太了解数据科学研究的巨大潜力。与此同时，许多数据科学家充其量对作为数据存储技术的 Elasticsearch 有一个基本的了解。

**简而言之** : Elasticsearch 是一个搜索引擎的数据库，因为 ***数据的存储方式*** 而能够进行闪电般的搜索。

在 Elasticsearch 中，文档被存储为**词频向量**(一个被称为“倒排索引”的过程)，并且**文档频率**是为每个词预先计算的。这意味着几件事:

1.  一个术语接一个术语的同现以令人难以置信的速度快速从**中提取***。*
2.  *重要术语可通过标准数据科学‘TF-IDF’程序*即时识别*。*

*从数据科学家的角度来看，Elasticsearch 数据库是一个非常基本(但功能强大)的预训练模型，用于提取关键字、同义词、相似文档和离群值。在这个由两部分组成的博客中，我将使用开箱即用的功能来触及所有这些内容(尽管当然也可以采用更复杂的方法)。*

# *第一部分:关键词工厂*

## *最简单的情况:允许非专家识别他们自己的关键字和同义词*

*生成关键字(或同义词)列表是数据科学家的一项常见 NLP 任务。它可以有从维度减少到主题建模的应用，并且还可以通过向人类分析师提供可以用于更费力的任务的一组数据驱动的术语来用于人在回路中的分析。*

*许多数据科学家会使用适当的 python(或 R)包来处理这个问题，以便产生相当静态的输出，这对于报告或论文来说是很好的。大多数时候，我们都希望我们的结果能够被非专家访问，但是实际上，人们最终得到的是他们所得到的:静态输出。我们有哪些可供非专家使用的可扩展且灵活的工具？*

## *Python(或 R)包:错误工作的正确工具*

*有大量基于 python 的方法可以解决这个问题，如主题建模(如潜在的狄利克雷分配或相关性解释)、单词向量聚类(如单词嵌入或香草计数向量)或使用共现矩阵或网络。*

*所有这些方法都可以给出非常合理的结果，老实说，我并不想在感知准确性上击败这些方法。如果我的任务是进行一次性的特别分析，我可以考虑以上任何一种方法。*

*然而，我喜欢可扩展、可共享和灵活的问题解决方案:*

*   ***可伸缩性**:我建议的所有 python 解决方案都要求数据和模型驻留在内存中。对于大量的文档或大量的词汇，内存消耗会很大。对此的一个解决方案是以牺牲模型的“深度”为代价对数据进行采样。*
*   *可共享性:如果你没有在非专家的笔记本电脑上安装 python 包，同时又希望你的设置与他们的笔记本电脑兼容，你如何与他们共享结果？两种可能是在远程服务器上托管你的机器学习模型(可以是从集群模型到共生矩阵的任何东西)(但是要小心那巨大的内存开销！)，或者您可以预先生成静态的关键字集(这非常简单)。*
*   ***灵活性:**想象一下，你想用多一个文档来更新你的模型，或者决定就地过滤你的数据——用常规的机器学习模型来做这件事并不简单。您的最佳方法可能是为每个预定义的过滤器预生成一个模型，这在计算上是昂贵的。*

## *弹性搜索:适合工作的合适工具*

*请记住，Elasticsearch 实际上是一个预先训练好的词共现模型，可以根据词的重要性进行过滤，这就很清楚为什么它可以动态地生成关键字列表。此外，我们应用于 Elasticsearch 的任何方法都具有内在的可扩展性、可共享性和灵活性:*

*   ***可伸缩性** : Elasticsearch 的性能高达 Pb 级。*
*   ***可共享性** : Elasticsearch 通过一个简单的 REST API 公开数据上的方法。要完成复杂的任务，您可以简单地将一些托管在远程服务器上的轻量级 python 代码串在一起。*
*   ***灵活性**:更新您的“模型”就像向服务器添加新文档一样简单。按任何字段过滤数据都是一项基本操作。*

## *“重要文本”集合*

*原则上，我们可以从零开始实现我们自己的提取关键词的程序，但是有一些快捷方式，你可以通过使用 Elasticsearch 的开箱即用功能来使用。*

*下面的 python 代码包装了一个对 Elasticsearch API 的查询(用您自己的端点替换`URL`,用您想要查询的数据库中的字段名称替换`FIELD_NAME`):*

```
*import requests
import json

def make_query(url, q, alg, field, shard_size=1000, size=25):
    """See [this gist](https://gist.github.com/jaklinger/6a644956f32e3e8b0d5e41c543ee49e1) for docs"""
    query = {"query" : { "match" : {field : q } },
             "size": 0,
             "aggregations" : {
                 "my_sample" : {
                     "sampler" : {"shard_size" : shard_size},
                     "aggregations": {
                        "keywords" : {
                            "significant_text" : {
                                "size": size,
                                "field" : field,
                                alg:{}
                             }
                        }
                    }
                }
            }
        }
    return [row['key'] 
            for row in requests.post(f'{url}/_search',
                                     data=json.dumps(query),
                                     headers={'Content-Type':'application/json'}).json()['aggregations']['my_sample']['keywords']['buckets']]*
```

*在幕后，该查询执行以下操作:*

1.  *在字段`field`中查找包含文本`query`的所有文档。*
2.  *从`field`中提取`size`最高有效项，根据`jlh`算法计算。*

*最重要的是，增加`shard_size`的大小将增加你的“模型”的稳定性(和深度),代价是计算性能。实际上，您只会期望您的模型在极少数情况下变得不太稳定——在这种情况下，您可以构建一个解决方法。*

## *tweeks 之前的性能:arXiv 数据*

*我的 Elasticsearch 数据库中有 arXiv 的所有科学出版物，下面是它对以下查询的抽象文本的表现:*

```
*python pandas['pandas', 'numpy', 'package', 'scipy', 'scikit', 'library', 'pypi', 'cython', 'github']-----------------------------elasticsearch['kibana', 'lucene', 'hadoop', 'retrieving', 'apache', 'engine', 'textual', 'documents', 'ranking']-----------------------------machine learning['learning', 'training', 'algorithms', 'neural', 'supervised', 'automl', 'intelligence', 'deep', 'tasks']----------------------------------------------------------drones and robots['robot', 'drones', 'robotics', 'robotic', 'humanoid', "robot's", 'drone', 'autonomous', 'mobile']-----------------------------*
```

*…这是开箱即用的功能！一些简单的批评是:*

1.  *无论是在查询中还是在结果中，n 元语法都没有被利用。例如，`machine learning`被视为`{machine, learning}`，而不是`{machine learning, machine, learning}`。*
2.  *搜索结果中出现停用词并非不可能。*
3.  *样本外的拼写错误根本不会被处理。*
4.  *名词的复数和所有格形式以及所有动词的变化都单独列出。*

*我不打算在这里解决后两点，但是处理它们是相当琐碎的。例如，**拼写错误**至少可以用两种方式处理，例如使用:*

*   *[Elasticsearch 的 n-gram 标记器](https://www.elastic.co/guide/en/elasticsearch/reference/current/analysis-ngram-tokenizer.html)(注意，Elasticsearch 在字符级定义 n-gram，不要与数据科学家的术语级 n-gram 混淆)*
*   *或者使用[注音符号化插件](https://www.elastic.co/guide/en/elasticsearch/plugins/7.3/analysis-phonetic.html)。*

***为了处理 n-gram 查询**(比如`machine learning`)我在我的 Elasticsearch 数据库中创建了一个包含预标记摘要的字段，在其中我已经识别了 n-gram。请注意，该字段的“模式”可在此处找到[。如果你想知道，我通过使用基于 wiki tionary](https://github.com/nestauk/nesta/blob/arxiv-ngrams/nesta/core/orms/arxiv_es_config.json#L93)的 [n-grams 的查找表来处理我的 n-grams(但是更数据驱动的方法也可以)。](https://github.com/nestauk/nesta/blob/dev/nesta/packages/nlp_utils/ngrammer.py)*

*虽然我自己没有实现这一点，但是可以对**复数/所有格/变形**进行类似的预处理，有效地将所有非简单形式的术语替换为它们的简单形式。*

*最后，为了避免**返回停用词**的潜在尴尬，我使用我的`make_query`函数从数据中生成它们:*

```
*and of but yes with however[‘however’, ‘but’, ‘not’, ‘answer’, ‘with’, ‘the’, ‘is’, ‘of’, ‘to’, ‘a’, ‘in’, ‘and’, ‘that’, ‘no’, ‘this’, ‘we’, ‘only’, ‘for’, ‘are’, ‘be’, ‘it’, ‘can’, ‘by’, ‘on’, ‘an’, ‘question’, ‘also’, ‘have’, ‘has’, ‘which’, ‘there’, ‘as’, ‘or’, ‘such’, ‘if’, ‘whether’, ‘does’, ‘more’, ‘from’, ‘one’, ‘been’, ‘these’, ‘show’, ‘at’, ‘do’]*
```

*我只是将这些从返回的结果中排除。*

## *把所有的放在一起*

*在 arXlive 网站上查看正在运行的[关键字工厂，其特色是 n-grams 和停用词移除。你可以使用](https://arxlive.org/keywords)`[clio-lite](https://github.com/nestauk/clio-lite#keywords-getting-under-the-hood)`包中的`clio_keywords`函数制作自己的 Flask 应用程序。玩得开心！*

# *第二部分:上下文搜索引擎*

*考虑一个技术性很强的数据集，比如来自全球最大的物理、定量和计算科学预印本库 [arXiv](https://arxiv.org/search/) 的数据集。假设您不是博学的学者，您会采取什么策略来查找与 ***大数据和安全*** 相关的 arXiv 最新研究？如果你在 arXiv 上[进行精确的搜索，你会发现自己有一组不错的结果，但问题是，当你搜索 ***大数据*** 时，你可能没有意识到你还想在查询中包括一些不太相关的术语，如 *{hadoop、spark、云计算}* 。如果在 ***云计算和安全*** 领域有一些你一直错过的重大突破，会怎么样？(TL；博士](https://arxiv.org/search/?query=big+data+security&searchtype=all&abstracts=show&order=-announced_date_first&size=50) [***这是相同的搜索与一个‘上下文’搜索引擎***](https://arxlive.org/hierarxy/?q=big%20data%20security&metric_novelty_article[min]=113&metric_novelty_article[max]=250) )*

*我将把这个问题分成两部分，通过使用 Python 中的一些 Elasticsearch 功能来解决它:*

*   *首先，你如何在不是天才的情况下做出一个像样的搜索查询？*
*   *其次，如何定义新奇？*

## *做一个像样的查询，而不是一个天才*

*回到制作一个像样的搜索查询的问题。天才可能会采取哪些方法？嗯，他们可以沿着'**关键词扩展**的路线走下去，例如通过考虑所有的 *{hadoop、spark、云计算}* 以及 ***大数据*** ，以及所有的*{攻击、加密、认证}* 以及 ***安全*** *。*这可能是一条很有前途的道路，我们已经在之前的博客中编写了工具来帮助实现这一点。然而，“关键字扩展”方法的主要问题是它缺少 ***上下文*** 。对此的一个自然扩展是'**文档扩展**'，谢天谢地，Elasticsearch 内置了这个特性。*

## *更像这样*

*好吧，老实说，Elasticsearch 的 [*更像是——这个*](https://www.elastic.co/guide/en/elasticsearch/reference/current/query-dsl-mlt-query.html) 查询实际上是‘关键词扩展**++**’，而不是‘文档扩展’，就像你在向量空间中想象的那样。在引擎盖下，从您想要“扩展”的输入文档中选择有代表性的术语(根据高度可配置的过程)。与纯粹的“关键词扩展”方法相比，这样做的优点在于，与所有输入项共现的项被认为比那些仅与输入项的子集共现的项更重要。结果是，可以假设用于播种“文档扩展”的扩展的关键字集具有高度的上下文相关性。*

*所以我的策略是:*

*   *对 Elasticsearch 进行常规查询，检索 10-25 个最相关的文档。这些将是我们的“种子”文档。*
*   *使用种子文档，用一个 *more-like-this* 查询跟进。*

*这种策略看起来有点像这样(实际上代码要多一点，所以实际上[看起来像这个](https://github.com/nestauk/clio-lite)):*

```
***# Make the initial vanilla query** r = simple_query(url, old_query, event, fields)
data, docs = extract_docs(r)**# Formulate the MLT query** total = data['hits']['total']
max_doc_freq = int(max_doc_frac*total)
min_doc_freq = int(min_doc_freq*total)
mlt_query = {"query":
             {"more_like_this":
              {"fields": fields,  **# the fields to consider**
               "like": docs,  **# the seed docs**
               "min_term_freq": min_term_freq,
               "max_query_terms": max_query_terms,
               "min_doc_freq": min_doc_freq,
               "max_doc_freq": max_doc_freq,
               "boost_terms": 1.,
               "minimum_should_match": minimum_should_match,
               "include": True  **# include the seed docs**
              }
             }
            }**# Make the MLT query** query = json.dumps(dict(**query, **mlt_query))
params = {"search_type":"dfs_query_then_fetch"}
r_mlt = requests.post(url, data=query,
                      headers=headers,
                      params=params)**# Extract the results** _data, docs = extract_docs(r_mlt)*
```

**注意，我通过 AWS API Gateway 在 Lambda 函数中提供此功能。部署上述功能的代码也可以在同一个 repo 中找到。**

## *定义新颖性*

*新奇没有特别狭窄的定义，我承认我对这个博客的定义会相当狭窄…*

*新奇通常可以被定义为任何(或更多)的{新的、原创的、不寻常的}，我的定义将跨越{原创的、不寻常的}概念。更正式一点(但不是很正式)我是问 Elasticsearch 里每个文档的以下问题:*

> *你和你最近的邻居有多大的不同？*

*只是为了理清这里的逻辑:如果文档的总样本是不平衡的，那么一个属于小众话题的文档会和一般的文档有很大的不同。我们可以通过比较最近的邻居来避免这种情况。还有什么比再次使用*more-like-this*([完整代码在此](https://github.com/nestauk/nesta/blob/dev/nesta/packages/novelty/lolvelty.py))更好的获取最近邻居的方法呢:*

```
*mlt_query = {
    "query": {
        "more_like_this": {
            "fields": fields,  **# field you want to query**
            "like": [{'_id':doc_id,     **# the doc we're analysing**
                      '_index':index}], 
            "min_term_freq": 1,
            "max_query_terms": max_query_terms, 
            "min_doc_freq": 1,
            "max_doc_freq": max_doc_freq, 
            "boost_terms": 1., 
            "minimum_should_match": minimum_should_match,
            "include": True
        }
    },
    "size":1000,  **# the number of nearest neighbours**
    "_source":["_score"]
}**# Make the search and normalise the scores** r = es.search(index=index, body=mlt_query)
scores = [h['_score']/r['hits']['max_score'] 
          for h in r['hits']['hits']]**# Calculate novelty as the distance to a low percentile** delta = np.percentile(scores, similar_perc)
return 1 - delta*
```

*自然，任何包含“不寻常”概念的像样的新颖性定义都应该包含坏数据，因为人们希望这些数据是不寻常的。我发现，通过对 arXiv 数据应用上面的新颖性评分器，我能够找出一大堆糟糕的数据，比如[抄袭的](https://arxiv.org/abs/hep-ph/0304042)和“[评论](https://arxiv.org/abs/1804.08090)的文章。我继续给这些标上 0 的新奇度，但是我相信你可以找到自己的方法！*

*这种新颖性评分方法的一个缺点是实现起来相对较慢(它必须在逐个文档的基础上计算)，因此我对我的 Elasticsearch 数据库中的所有文档进行了预处理。*

## *把所有的放在一起*

*因此，通过过度使用 Elasticsearch 的 *more-like-this* 查询，我们能够进行广泛的搜索，同时衍生出一种非常轻量级的新奇度。查看`[clio-lite](https://github.com/nestauk/clio-lite)`以更好地理解代码，并且**如果您想看到这在行动中**，请查看 arXiv 数据的[***hierar xy***搜索引擎](https://arxlive.org/hierarxy)。请注意，我还使用了与第一部分中描述的相同的预处理，以及本博客中描述的数据清理。感谢阅读！*