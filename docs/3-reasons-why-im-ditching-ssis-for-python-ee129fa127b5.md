# 我放弃 SSIS 转投 Python 的 3 个原因

> 原文：<https://towardsdatascience.com/3-reasons-why-im-ditching-ssis-for-python-ee129fa127b5?source=collection_archive---------1----------------------->

![](img/19567a058960521d09515667ff1fb35a.png)

Photo by [Chris Ried](https://unsplash.com/@cdr6934?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

十多年来，我一直在使用微软的 SQL Server 技术堆栈，虽然我一直非常看好它，但最近我改变了对它的一个关键组件的看法，即[SQL Server Integration Services](https://docs.microsoft.com/en-us/sql/integration-services/sql-server-integration-services?view=sql-server-ver15)，简称 SSIS。SSIS 是一个非常强大的工具，可以对数据执行提取、转换和加载(ETL)工作流，并且可以与几乎任何格式进行交互。虽然我主要是在将数据加载到 SQL Server 或从 SQL Server 中加载数据的上下文中看到它，但这肯定不是它的唯一用途。

这些年来，我已经编写了很多 SSIS 软件包，虽然我仍然觉得它是您的武库中的一个巨大的工具(在许多情况下，它可能是对技术使用有严格标准的大型企业中唯一可用的工具)，但我现在已经决定，出于我将在下面概述的原因，我更喜欢使用 Python 来满足大多数(如果不是全部)ETL 需求。尤其是当 Python 与两个专门为大规模操作和分析数据而设计的模块结合时，即 [Dask](https://dask.org) 和 [Pandas](https://pandas.pydata.org/) 。

# Python 是免费和开源的

Python 是一种完全开源的语言，由 [Python 软件基金会](https://www.python.org/psf/)维护。它和它的大量软件包都是完全免费的，如果你发现一个错误或者需要一个特性，你可以很容易地贡献出底层的源代码。例如，Dask 和 Pandas 在 GitHub 上总共有超过 25，000 次提交和 9，000 次分叉。两者都是非常活跃的项目，背后都有大型的、分布式的、活跃的社区。此外，Python 可以使用其他开源包与几乎任何数据源对话；从 [CSV 文件](https://examples.dask.org/dataframes/01-data-access.html#Read-CSV-files)，到[卡夫卡](https://github.com/Parsely/pykafka)，到[抓取网站](/how-to-web-scrape-with-python-in-4-minutes-bc49186a8460)。Python 作为一个整体非常受欢迎并且不断增长，在 Stack Overflow 的 [2019 开发者调查](https://insights.stackoverflow.com/survey/2019#most-popular-technologies)中从第七位跃升至第四位。

另一方面，SSIS 要求您许可任何运行它的机器，就像您许可任何其他运行 SQL Server 完整实例的机器一样。因此，如果您想遵循良好的实践，将 ETL 处理卸载到不同于 SQL Server 实例的机器上，那么您必须全额支付该机器的许可费用。假设您能够成功地完成 SQL Server 的复杂许可，请考虑一下:许多 ETL 工作负载都是*批处理*操作，这意味着它们往往在一天中的预定时间运行，否则就会闲置。你真的想为一天用一次或有时用得更少的东西支付(有时是一大笔)钱吗？虽然 SSIS 是可扩展的，但我还没有看到你在 Python 上看到的那种广泛的开源工具集。

# 使用像 Dask 这样的工具，Python 天生具有水平可伸缩性

Dask 是专门为处理数据集太大而无法在单个节点的内存中容纳的问题而设计的，并且可以在许多节点上[扩展](https://docs.dask.org/en/latest/why.html#dask-scales-out-to-clusters)。因此，您可以使用您的组织可能已经使用的工具(如 [Kubernetes](https://kubernetes.dask.org/en/latest/) )根据您的需求轻松扩展您的数据处理环境的规模，而无需编写复杂的代码来跨节点分发数据。我个人用它在我的笔记本电脑上处理了数十亿字节的数据，只使用了内置的[本地分布式模型](https://docs.dask.org/en/latest/scheduling.html#dask-distributed-local)，并且我没有改变我编写数据处理代码的方式。

至少据我所知，SSIS 没有内在的方法在多台计算机上分配处理，至少没有复杂的解决方案，就像在[这个 ServerFault 线程](https://serverfault.com/questions/361318/distributed-and-or-parallel-ssis-processing)上提出的。虽然自 SQL Server 2017 起，SSIS 确实具有[横向扩展](https://docs.microsoft.com/en-us/sql/integration-services/scale-out/integration-services-ssis-scale-out?view=sql-server-ver15)功能，但这更多地是为了在*包*和*任务*级别分配工作(例如，以分布式方式运行包的各个部分)，因此，如果这些单个任务中的任何一个处理大量数据，您仍然会受到限制。

# Python 代码本质上是可测试的

正如我的好朋友 Nick Hodges 最近写的那样，[对你的代码进行单元测试是很重要的](https://medium.com/better-programming/unit-testing-and-why-you-should-be-doing-it-ab61407c53ce)。在处理 ETL 工作流时也是如此；在没有某种人工观察的情况下，您如何确保您的 ETL 过程在给定预期输入的情况下产生正确的输出(或者同样重要的是，当它获得意外数据时如何处理事情)？

Python 有很多有用的单元测试框架，比如 [unittest](https://docs.python.org/3/library/unittest.html) 或者 [PyTest](https://docs.pytest.org/en/latest/) 。通过将您的 ETL 过程分解成可消耗的代码单元，您可以轻松地确保预期的行为并进行更改，而不必担心无意中破坏了某些东西。

相比之下，SSIS 没有任何简单的方法来编写单元测试，最常被提及的两个框架要么是不活跃的，要么似乎已经很大程度上转变为专有产品。

底线是:在这个狂热的 SQL Server 粉丝看来，如果你正在开发新的 ETL 工作流，更好的选择是 Python 而不是 SSIS。请注意，SSIS 仍然非常强大，如果您有一个强大的现有团队支持它，并且在 SQL Server 许可方面投入了大量资金，那么当然没有理由对所有这些进行改造。正如我在开始所说的，在许多大型组织中，SSIS(或者 Informatica，举另一个非常流行的专有 ETL 工具的例子)是事实上的标准。尽管如此，鉴于 Python 的爆炸性增长，它显然应该获得一席之地。