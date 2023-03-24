# 用于 ETL 的 Python 数据转换工具

> 原文：<https://towardsdatascience.com/python-data-transformation-tools-for-etl-2cb20d76fcd0?source=collection_archive---------7----------------------->

**免责声明:**我不是 ETL 专家，我欢迎在这个领域更有经验的人的任何评论、建议或批评。

前几天，我在 [Reddit](https://www.reddit.com/r/ETL/comments/cnbl1w/using_python_for_etlelt_transformations/) 上询问我是否应该使用 Python 进行 ETL 相关的转换，压倒性的回应是**是的**。

![](img/6145cc6a6c019ab6676ba71a01a12e4e.png)

source: [Pinclipart](https://www.pinclipart.com/pindetail/iioxbJw_python-logo-clipart-easy-pandas-python-logo-png/)

然而，虽然我的 Redditors 同事热情地支持使用 Python，但他们建议看看 Pandas 以外的库——引用了对 Pandas 处理大型数据集的性能的担忧。

在做了一些研究之后，我发现了大量为数据转换而构建的 Python 库:一些提高了 Pandas 的性能，而另一些提供了自己的解决方案。

我找不到这些工具的完整列表，所以我想我应该利用我所做的研究来编译一个——如果我遗漏了什么或弄错了什么，请让我知道！

# 熊猫

**网址:**[https://pandas.pydata.org/](https://pandas.pydata.org/)

**概述**

熊猫当然不需要介绍，但我还是要给它介绍一下。

Pandas 将数据帧的概念添加到 Python 中，并在数据科学社区中广泛用于分析和清理数据集。作为 ETL 转换工具，它非常有用，因为它使得操作数据变得非常容易和直观。

**优点**

*   广泛用于数据操作
*   简单、直观的语法
*   与包括可视化库在内的其他 Python 工具集成良好
*   支持通用数据格式(从 SQL 数据库、CSV 文件等读取。)

**弊端**

*   因为它将所有数据加载到内存中，所以它是不可伸缩的，并且对于非常大(大于内存)的数据集来说可能是一个糟糕的选择

**延伸阅读**

*   [距离熊猫还有 10 分钟](https://pandas.pydata.org/pandas-docs/stable/getting_started/10min.html)
*   [用熊猫进行机器学习的数据操作](/data-manipulation-for-machine-learning-with-pandas-ab23e79ba5de)

# 达斯克

![](img/ac4d8186c23b0d81fef8b62de415e1b3.png)

**网址:**[https://dask.org/](https://dask.org/)

**概述**

根据他们的网站，“Dask 是一个灵活的 Python 并行计算库。”

本质上，Dask 扩展了公共接口，如 panda，用于分布式环境中——例如，Dask DataFrame 模仿 panda。

**优点**

*   可扩展性— Dask 可以在您的本地机器上运行*和*扩展到一个集群
*   能够处理不适合内存的数据集
*   即使在相同的硬件上，也能以相同的功能提高性能(得益于并行计算)
*   从熊猫切换的最小代码更改
*   旨在与其他 Python 库集成

**弊端**

*   除了并行性，还有其他方法可以提高 Pandas 的性能(通常更显著)
*   如果你正在做的计算量很小，那就没什么好处
*   有些功能没有在 Dask 数据帧中实现

**延伸阅读**

*   [Dask 文档](https://docs.dask.org/en/latest/)
*   [为什么每个数据科学家都应该使用 Dask](/why-every-data-scientist-should-use-dask-81b2b850e15b)

# 摩丁

![](img/07bc07842b455ccadf30c8d7af515102.png)

**网址:**[https://github.com/modin-project/modin](https://github.com/modin-project/modin)

**概述**

Modin 与 Dask 相似，它试图通过使用并行性和启用分布式数据帧来提高 Pandas 的效率。与 Dask 不同，Modin 基于任务并行执行框架 [Ray](https://github.com/ray-project/ray/) 。

与 Dask 相比，Modin 的主要优势在于，Modin 自动处理跨机器内核的数据分发(无需配置)。

**优点**

*   可伸缩性 Ray 比 Modin 提供的更多
*   即使在相同的硬件上，也能以完全相同的功能提高性能
*   从 Pandas 切换的最小代码更改(更改导入语句)
*   提供 Pandas 的所有功能——比 Dask 更像是一个“嵌入式”解决方案

**弊端**

*   除了并行性，还有其他方法可以提高 Pandas 的性能(通常更显著)
*   如果你正在做的计算量很小，那就没什么好处

**延伸阅读**

*   [摩丁文档](https://modin.readthedocs.io/en/latest/)
*   [达斯克和摩丁有什么区别？](https://github.com/modin-project/modin/issues/515)

# petl

**网址:**[https://petl.readthedocs.io/en/stable/](https://petl.readthedocs.io/en/stable/)

**概述**

petl 包含了 pandas 拥有的许多特性，但是它是专门为 etl 设计的，因此缺少额外的特性，比如那些用于分析的特性。petl 拥有 etl 所有三个部分的工具，但是这篇文章只关注数据转换。

虽然 petl 提供了转换表的能力，但是其他工具(比如 pandas)似乎更广泛地用于转换，并且有很好的文档记录，这使得 petl 在这方面不太受欢迎。

**优点**

*   最大限度地减少系统内存的使用，使其能够扩展到数百万行
*   对于 SQL 数据库之间的迁移非常有用
*   轻巧高效

**弊端**

*   通过最大限度地减少系统内存的使用，petl 的执行速度较慢——对于性能非常重要的应用程序，不推荐使用它
*   与列表中的其他数据操作解决方案相比，较少使用

**延伸阅读**

*   [使用 petl 快速了解数据转换和迁移](http://www.sentia.com.au/blog/a-quick-dive-into-data-transformation-and-migration-with-petl)
*   [petl 转换文档](https://petl.readthedocs.io/en/stable/transform.html)

# PySpark

![](img/0587c4822f860fabeece928d69f657ee.png)

**网址:**[http://spark.apache.org/](http://spark.apache.org/)

**概述**

Spark 旨在处理和分析大数据，并提供多种语言的 API。使用 Spark 的主要优势是 Spark 数据帧使用分布式内存并利用延迟执行，因此它们可以使用集群处理更大的数据集——这是 Pandas 等工具所无法实现的。

如果您正在处理的数据非常大，并且数据操作的速度和大小非常大，那么 Spark 是 ETL 的一个好选择。

**优点**

*   更大数据集的可扩展性和支持
*   Spark 数据帧在语法方面与 Pandas 非常相似
*   通过 Spark SQL 使用 SQL 语法进行查询
*   与其他流行的 ETL 工具兼容，包括 Pandas(您实际上可以将 Spark 数据帧转换成 Pandas 数据帧，使您能够使用所有其他类型的库)
*   兼容 Jupyter 笔记本电脑
*   对 SQL、流和图形处理的内置支持

**弊端**

*   需要分布式文件系统，如 S3
*   使用像 CSV 这样的数据格式限制了延迟执行，需要将数据转换成其他格式，比如 T21 的拼花地板
*   缺乏对 Matplotlib 和 Seaborn 等数据可视化工具的直接支持，这两个工具都得到 Pandas 的良好支持

**延伸阅读**

*   [Python 中的 Apache Spark:初学者指南](https://www.datacamp.com/community/tutorials/apache-spark-python)
*   [py spark 简介](/a-brief-introduction-to-pyspark-ff4284701873)
*   [PySpark 文档](https://spark.apache.org/docs/latest/api/python/pyspark.sql.html#pyspark.sql.DataFrame)(特别要看语法)

# 值得注意的提及

虽然我想这是一个全面的列表，但我不想这篇文章变得太长！

确实有很多很多用于数据转换的 Python 工具，所以我在这一部分至少提到了我错过的其他项目(我可能会在本文的第二部分进一步探讨这些项目)。

*   **倭** https://www.bonobo-project.org/
*   **气泡** [http://bubbles.databrewery.org/](http://bubbles.databrewery.org/)
*   **pygrametl** [http://chrthomsen.github.io/pygrametl/](http://chrthomsen.github.io/pygrametl/)
*   **阿帕奇光束**
    [https://beam.apache.org/](https://beam.apache.org/)

# 结论

我希望这个列表至少能帮助您了解 Python 为数据转换提供了哪些工具。在做了这项研究之后，我确信 Python 是 ETL 的一个很好的选择——这些工具和它们的开发者已经使它成为一个令人惊奇的使用平台。

正如我在这篇文章的开头所说，我不是这方面的专家——如果你有什么要补充的，请随时评论！

感谢阅读！