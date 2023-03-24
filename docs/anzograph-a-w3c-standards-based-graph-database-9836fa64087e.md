# AnzoGraph:一个基于 W3C 标准的图形数据库

> 原文：<https://towardsdatascience.com/anzograph-a-w3c-standards-based-graph-database-9836fa64087e?source=collection_archive---------19----------------------->

## 剑桥语义学的巴里·赞访谈

![](img/eedab9fc05b078d7434f5f05f79708ad.png)

[Image by Kyle McDonald on Flickr CC BY 2.0](https://www.flickr.com/photos/kylemcdonald/13229639943/in/photolist-ma4n3p-7h3S7E-3tr8pq-u4WtrM-WmvePL-9xNzNj-2aYztZ9-2cn12vH-bQWs6K-cmZ4XL-WxYpnh-UTd9rq-8NY5sg-RBjFZQ-arYriz-6PgEQi-6PgERR-2cXuRwT-2cQTjcK-2btdkME-23imjzY-p7Cq4W-24jff5S-2emqKxB-9Bda7P-2cNkr7H-p9oSR4-5tiUhQ-24Hyw2X-QAc8gm-9FeQ5W-c6cfGW-27DwQUE-Jqijim-9FbUzD-HGmMUY-NXLbQ8-2cmpRrT-gEXeVS-WpJ4TD-7w5KCG-JwRc1W-2cDcJ59-RBgz2C-2baGXvw-WpWabi-q1SgQk-HenVvy-2dEyf8E-pYLnBf)

# 介绍

在这次采访中，我正赶上[剑桥语义学](https://www.cambridgesemantics.com/)的副总裁[巴里·赞](https://www.linkedin.com/in/barry-zane-b701732/?lipi=urn%3Ali%3Apage%3Ad_flagship3_search_srp_people%3BYAsFqmxiR%2FmsZEoamawtVA%3D%3D&licu=urn%3Ali%3Acontrol%3Ad_flagship3_search_srp_people-search_srp_result&lici=HcCAFvW3SdGuwmX8OApREA%3D%3D)。巴里是 [AnzoGraph](https://www.cambridgesemantics.com/product/anzograph/) 的创造者，这是一个本地的大规模并行处理(MPP)分布式图形数据库。Barry 在数据库领域经历了一段漫长的旅程。他在 2000 年至 2005 年期间担任 Netezza Corporation 的技术副总裁，负责指导软件架构和实施的所有方面，从最初的原型到批量出货，再到领先的电信、零售和互联网客户。Netezza 最终被出售给了 IBM，但在此之前，Barry 已经将注意力转向了其他地方，成立了另一家公司 ParAccel，该公司最终成为了 AWS Redshift 的核心技术。市场上开始出现对基于图形的在线分析处理(图形 OLAP)数据库的需求，基于这一市场需求，Barry 于 2013 年创立了 SPARQL City。

Barry 友好地同意本周与我交谈，此前[最近宣布](https://www.prweb.com/releases/cambridge_semantics_announces_anzograph_open_standards_graph_database_for_analytics_available_for_standalone_use/prweb15833988.htm)AnzoGraph 数据库现已可供下载，用于独立评估和在客户应用程序中使用，无论是在内部还是在云上。虽然还没有宣布，Barry 还透露 AnzoGraph 已经得到了增强，可以使用 RDF*/SPARQL*来提供完整的属性图功能。所以和他交谈并了解更多关于图形分析和 W3C 标准是如何结合在一起的是令人兴奋的。

## 首先，巴里，你能告诉我们一些关于剑桥语义学的事情吗？

剑桥语义学大约从 2007 年开始出现。我们多年来构建的解决方案之一是名为 Anzo 的语义层产品。Anzo 用于许多大型企业，如制药、金融服务、零售、石油和天然气、医疗保健公司和政府部门。这些企业都有一个共同的趋势，即拥有多样化的数据源，并且真正需要发现和分析数据。Anzo 提供的语义层将原始数据与业务含义和上下文结合起来并呈现出来。恰好图形数据库是这个解决方案的关键基础设施元素。

Cambridge Semantics 很早就看到了图形分析的价值，并且是 SPARQL City 的首批客户之一。他们在 2016 年收购了我们。2018 年末，我们将图形引擎置于 Anzo 之下，并将其分拆为自己的产品，名为 AnzoGraph。

## 请解释一下 AnzoGraph 的主要使用案例好吗？

在 OLTP 数据库方面，图形数据库市场得到了很好的覆盖。我们决定构建一个 OLAP 风格的图形数据库，而不是像 Neo4J 和最近的 AWS Neptune 那样的 OLTP 图形数据库。市场上确实需要执行数据仓库风格的分析，并获得处理结构化和非结构化数据的额外好处。借助 AnzoGraph，我们可以提供报告、BI 分析和聚合、图形算法(如页面排名和最短路径)、推理以及更多市场上缺失的数据仓库式分析。

客户使用 AnzoGraph 发现大规模多样化数据的新见解，包括历史和最近的数据。它非常适合在非常大的数据集上运行算法和分析，以找到相关的实体、关系和见解。我们将用户使用基于 W3C 标准的 RDF 数据库获得的价值与他们使用属性图获得的价值结合起来。

我们对将 AnzoGraph 用于多种用途很感兴趣。想一想，在所有需要执行分析的时候，连接数据的信息与数据本身同样重要。例如，[知识图](https://hackernoon.com/wtf-is-a-knowledge-graph-a16603a1a25f)在许多试图将不同数据源连接在一起的公司中很流行，我们在 Anzo 中的经验对此有所帮助。各公司都在努力理解买家的意图，并建立推荐引擎。图表可以帮助解决“喜欢产品 A 的人可能也会喜欢产品 B”的问题。在金融服务领域，银行正在使用图表来“跟踪资金”。图表提供了跟踪衍生品和其他资产转移的能力，因此可以帮助银行管理风险。甚至 IT 组织也在关注复杂的网络，并试图更好地了解 IP 流量如何在设备之间流动。

有几个新出现的用例让我感到非常兴奋。首先，当与自然语言处理引擎或解析器配对时，AnzoGraph 非常擅长处理链接的结构化/非结构化数据和基于图形的基础设施，用于人工智能和机器学习中基于图形的算法。其次，关注图表分析如何对基因组研究产生影响是很有趣的。科学家们没有采用带来遗传学中许多分析驱动创新的蛮力技术，而是通过图形分析开发新的分析技术，允许用户找到新的见解，而无需像在关系数据库中那样为这些见解显式编程。

## AnzoGraph 有什么不同于其他数据库仓库解决方案的地方？

这可能是您没有预料到的，并且与传统 RDBMS 数据仓库世界中共享模式的不灵活性有关，在传统 RDBMS 数据仓库世界中，我们的任务是创建表和固定模式。为了得到答案，我们可能需要创建复杂的连接来查询表。然而，在图形数据库世界中，由于一切都用三元组表示，我们用一个动词和一个描述来描述一个人、一个地方或一件事，所以很容易添加更多的三元组来进一步描述它，而不需要改变模式。标准的[本体](https://blog.grakn.ai/what-is-an-ontology-c5baac4a2f6c)帮助我们描述关系，这在我们想要共享数据时尤其有用。数据库模式通常不那么灵活，因为它们通常从一开始就是固定的和定制的。

> 图形数据库中的本体非常灵活，可以更好地与你的伙伴共享数据。

当然，对分析的支持也是一个巨大的差异。虽然 AnzoGraph 提供了传统数据仓库的所有分析功能，但它还提供了图形算法、推理和其他功能。这使得处理我上面提到的用例变得非常容易。图形数据库更适合某些类型的机器学习算法，并提供基于机器的推理，这在机器学习中非常有价值。

与传统的数据仓库不同，AnzoGraph 非常适合部署灵活性和可伸缩性。由于可伸缩性因素，市场对用 Docker 和 Kubernetes 这样的容器构建的应用程序做出了响应。当您可以随意旋转多个容器并将其旋转回来时，这是一个非常经济的可扩展解决方案。在基准测试中，我们实现了比其他数据库快 100 倍的性能，前途无量。当然，AnzoGraph 可以部署在裸机、虚拟机或任何云中，但容器最受关注。

## 2018 年，机器学习的一系列技术领域都有了巨大的发展，而深度学习正在等待时机。图形数据库有什么可以提供给那些想要加入人工智能淘金热的海量数据的人吗？

我们正在看到机器学习和人工智能的更广泛采用，图形数据库将发挥作用。我们都知道机器学习最大的挑战是数据准备。然而，通过直接导入原始数据，然后在图形数据库本身而不是复杂的 ETL 管道中进行管理，这种准备和管理得到了简化。数据模型的简单性使得管理比关系数据库中的管理更加简单和快速。当复杂的模式消失时，用户将能够更容易地挖掘非结构化数据，并且他们可以利用容器的可伸缩性。

> 图形数据库使用户能够自由地“旋转”他们的分析，提出新的、*特别的*问题，而不受关系技术的限制。图形数据库可以为机器学习和人工智能提供很多东西。

## 图形数据库已经存在了一段时间，但现在才开始成熟。你对这个领域未来两年的预测是什么，AnzoGraph 将如何领导下一代图形数据库？

我期待在未来几年里，人们对执行大数据分析的一般类别有更深入的了解，而不是运营查询。AnzoGraph 非常关注跨图形空间聚合的大数据分析。我们可以超越狭窄的查询“*告诉我关于史蒂夫的事情*”来涵盖更广泛的分析，例如“*告诉我关于人类的事情*”。

> 我认为来年将会看到下一代标准查询语言的定义。

W3C 标准是目前唯一的正式标准，但是 Cypher 显然是标签属性图的事实标准。有一个组织已经成立来创建下一代正式标准，看看它是如何形成的会很有趣。在剑桥语义学，我们非常支持这一过程，拥有一种强大的图形语言是一件好事。因此，我对图形空间未来几年的预测是，专有模型即将过时。

市场将决定确切的标准，我们将调整我们的解决方案以符合标准，因为我们坚定地致力于标准。我不认为这种演变是一种威胁，而是一个巨大的机会，因为它符合我们的心态，只会增加图形技术的吸收。

# 最后

我要感谢 Barry 和剑桥语义的团队给我机会去了解更多关于 AnzoGraph 的知识。我与该公司没有任何关系，我应该指出，我没有从他们那里得到这次面试的报酬。

如果你想了解关于 AnzoGraph 的更多细节，从 2018 年 10 月开始，Slideshare 上有一个很棒的[技术演示，或者查看一下](https://www.slideshare.net/camsemantics/large-scale-graph-analytics-with-rdf-and-lpg-parallel-processing)[网站](https://www.cambridgesemantics.com/product/anzograph/)。请在下面的评论中留下任何问题！