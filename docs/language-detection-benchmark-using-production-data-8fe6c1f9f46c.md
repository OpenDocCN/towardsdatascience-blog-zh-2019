# 使用生产数据的语言检测基准

> 原文：<https://towardsdatascience.com/language-detection-benchmark-using-production-data-8fe6c1f9f46c?source=collection_archive---------35----------------------->

## *这是多语言语言检测算法的真实社交媒体数据基准。*

![](img/c2b5cf9d9837047084c6427f90a5c042.png)

[*The Tower of Babel*](https://en.wikipedia.org/wiki/The_Tower_of_Babel_(Bruegel)) by [Pieter Bruegel the Elder](https://en.wikipedia.org/wiki/Pieter_Bruegel_the_Elder) (1563)

作为数据科学家，我们习惯于处理许多不同类型的数据。但是对于基于文本的数据，了解数据的语言是重中之重。在为英语、西班牙语、法语和塞尔维亚语开发四种基于语言的算法时，我亲身经历了这一挑战。

一般来说，处理多语言 NLP 算法的第一步是:**检测文本的语言**。然而，在生产过程中，数据在需要处理的不同语言之间通常是不平衡的，这意味着在将数据发送给算法之前，必须定义每个数据点*的语言。对于任何处理数据中语言不平衡的产品来说，显然不“丢失”数据是产品成功的关键。因此，为了确保产品不会丢失数据，我们必须确保数据不会因为 LD 故障而被发送到错误的算法。了解数据分布对于为任何研究项目找到一个好的**关键绩效指标(KPI)** 至关重要，尤其是对于这个基准测试项目。*

## 测试集的采样数据:

对于这个基准测试，我希望测试集尽可能地代表我在生产中通常拥有的东西。因此，我从 Twitter 上选择了有语言参数的 10 万条帖子，包括 95780 条英语推文、1093 条西班牙语推文、2500 条法语推文和 627 条塞尔维亚语推文。这个分布代表了生产中大约两周的数据。

此外，为了获得最好的测试集，一些注释者手工检查并纠正了 Twitter 语言参数中的潜在错误。这意味着我将测试过的模型与这个事实进行了比较。我还可以通过正确标记数据来消除 Twitter LD 偏见。([这里有个好办法](/the-definite-guide-for-creating-an-academic-level-dataset-with-industry-requirements-and-6db446a26cb2))。

## 测试算法:

对于这个语言检测(LD)基准，我对比了 [**多语种 LD**](https://polyglot.readthedocs.io/en/latest/) **，**[**Azure Text Analytics LD**](https://docs.microsoft.com/en-us/azure/cognitive-services/text-analytics/how-tos/text-analytics-how-to-language-detection)**，**[**Google Cloud LD**](https://cloud.google.com/translate/docs/reference/rest/v2/detect)**，****fast Text LD****。**

为了重现下面的代码，您需要 pip 安装所有导入的包。此外，您需要下载 [FastText 算法](https://fasttext.cc/docs/en/language-identification.html)，以创建 [Azure API 信用](https://westcentralus.dev.cognitive.microsoft.com/docs/services/TextAnalytics-v2-1/operations/56f30ceeeda5650db055a3c7)和 [Google Cloud API JSON 信用](https://cloud.google.com/docs/authentication/)。

## 结果分析:

我的基准测试项目的结果表明，不同的模型服务于不同的度量。

**准确性:**如果准确性是最重要的指标，那么所有模型的表现几乎都是一样的(图 1)。

![](img/d4641f80a58b3a8cd60f2b04a589713c.png)

Figure 1: Global accuracy per model

然而，在我的项目中，准确的语言检测并不是唯一重要的 KPI。此外，防止数据丢失也很重要，因为如果数据丢失，产品将会失败。

**召回:**召回图(图 2)显示，尽管每个模型都非常精确，但总体而言，谷歌的模型在召回方面优于其他模型。

![](img/af4330ba14975686f6624bd7f46ee5c9.png)

Figure 2: Recall by model for all languages

## **结论:**

虽然第一直觉是只看准确性指标，这将导致有人假设 Azure 是语言检测的最佳表现者，但当其他指标发挥作用时，其他模型可能更好。在这种情况下，Google Cloud LD 在召回率方面优于 Azure(和所有其他测试的模型)，特别是当正在处理的语言之间的数据不平衡时，其中有一个较大的数据集(英语)，然后是其他语言(西班牙语、法语和塞尔维亚语)的明显较小的数据集。在我的特定项目中，回忆是主要的衡量标准，Google Cloud LD 最终是我的 LD 模型选择。

我要感谢来自 Zencity 的同事，他们在这个项目中发挥了重要作用:Inbal Naveh Safir、Ori Cohen 和 Yoav Talmi。

Samuel Jefroykin 是 [Zencity.io](https://zencity.io/) 的一名数据科学家，他试图积极影响城市的生活质量。他还共同创立了 Data For Good Israel，这是一个致力于利用数据的力量解决社会问题的社区。