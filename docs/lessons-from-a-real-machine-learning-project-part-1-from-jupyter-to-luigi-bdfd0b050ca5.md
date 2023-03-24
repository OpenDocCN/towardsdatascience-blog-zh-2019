# 真实机器学习项目的经验，第 1 部分:从 Jupyter 到 Luigi

> 原文：<https://towardsdatascience.com/lessons-from-a-real-machine-learning-project-part-1-from-jupyter-to-luigi-bdfd0b050ca5?source=collection_archive---------12----------------------->

## 如何组织机器学习代码

![](img/45c45564f10675682c825ac2adb18123.png)

在过去的 6 个月里，我和我的 awesome 团队一直在从事一个具有挑战性的企业级机器学习项目:从零开始重建一家主要能源提供商的短期电力负荷预测模型。

这是一个艰难但令人满意的旅程:我们学到了很多，主要是通过试图从幼稚的错误中恢复过来。是时候尝试分享学到的主要教训了，希望能有所帮助——为什么不呢？—得到帮助，进一步提高。

一篇文章太长了，而且很无聊:我会试着贴一堆，每一篇都关注一个主题。这一条涵盖了一个非常棘手的问题:如何组织、构建和管理机器学习项目的代码。

# 开始:一个基于 Jupyter 的项目

我在最近的学术项目中使用过 Jupyter 我喜欢它的强大和灵活性。一个快速原型工具也是最终的成果共享平台是无价的，不是吗？因此，Jupyter 是我们第一次分析的选择。

它只进行了数据探索。那时，我们的项目库看起来像这样。

![](img/a84e589cbbb742ac604681a022009356.png)

你发现问题了，对吗？它不可扩展。一点也不。

我们过去常常将部分代码从一个笔记本复制粘贴到另一个笔记本上——这是任何软件工程师都无法忍受的。此外，我们实际上被阻止在相同的代码基础上合作，因为基于 json 的笔记本结构使得用 git 进行版本控制非常痛苦。

# 突破:ML 中的软件工程

在越过不归路之前，我们寻找一个更具可伸缩性的结构。我们发现以下资源非常有用:

*   [Mateusz Bednarski，机器学习项目的结构和自动化工作流，2017 年](/structure-and-automated-workflow-for-a-machine-learning-project-2fa30d661c1e)
*   [Kaggle 社区，如何管理一个机器学习项目，2013](https://www.kaggle.com/general/4815)
*   [丹·弗兰克，可重复研究:Stripe 的机器学习方法，2016 年](https://stripe.com/blog/reproducible-research)
*   [DrivenData，数据科学 cookiecutter，2019](https://github.com/drivendata/cookiecutter-data-science) (最后更新)

他们都在重复同样的信息。

*机器学习项目的代码与任何其他项目没有不同，应遵循软件工程的最佳实践。*

因此，我们重构了整个代码库:

1.  我们将笔记本中所有可重用的代码提取到几个实用模块中，从而消除了笔记本中的所有重复
2.  我们将模块分成 4 个包，对应于我们工作流程的 4 个主要步骤:数据准备、特征提取、建模和可视化
3.  我们将最关键的功能置于单元测试之下，从而防止危险的回归

操作一完成，Python，非笔记本代码就成了事实的唯一来源:按照团队惯例，它是我们最先进的数据准备、特征提取、建模和评分版本。每个人都可以进行试验，但是只有当一个经过验证的改进模型可用时，Python 代码才会被修改。

实验呢？而笔记本呢？

两个问题，同一个答案。

当然，笔记本电脑并没有从我们的开发流程中消失。它们仍然是一个令人敬畏的原型开发平台和一个无价的结果共享工具，不是吗？

我们只是开始将它们用于它们最初被创造的目的。

笔记本变得个性化，从而避免了任何 git 的痛苦，并遵循严格的命名规则:*author _ incremental number _ title . ipynb*，以便于搜索。它们仍然是所有分析的起点:模型是用笔记本做原型的。如果有一个碰巧超过了我们最先进的，它被集成到了*生产* Python 代码中。超越的概念在这里被很好地定义了，因为评分程序是在实用模块中实现的，并由团队的所有成员共享。笔记本也构成了大部分文档。

我们作为一个团队只花了几天时间就完成了转型。差异令人难以置信。几乎在一夜之间，我们释放了集体代码所有权、单元测试、代码可重用性以及过去 20 年软件工程的所有遗产的力量。显而易见的结果是，生产率和对新请求的响应能力大大提高。

最明显的证据是当我们意识到我们在所有的拟合优度图表中缺少测量单位和标签时。因为它们都是由一个函数实现的，所以修复它们既快又容易。如果同样的图表仍然被复制并粘贴在许多笔记本上，会发生什么？

在转换的最后，我们的存储库看起来像这样。

```
├── LICENSE
├── README.md          <- The top-level README for developers
│
├── data
│   ├── interim        <- Intermediate data
│   ├── output         <- Model results and scoring
│   ├── processed      <- The final data sets for modeling
│   └── raw            <- The original, immutable data dump
│
├── models             <- Trained and serialized models
│
├── notebooks          <- Jupyter notebooks
│
├── references         <- Data explanatory materials
│
├── reports            <- Generated analysis as HTML, PDF etc.
│   └── figures        <- Generated charts and figures for reporting
│
├── requirements.yml   <- Requirements file for conda environment
│
├── src                <- Source code for use in this project.
    │
    ├── tests          <- Automated tests to check source code
    │    
    ├── data           <- Source code to generate data
    │
    ├── features       <- Source code to extract and create features
    │
    ├── models         <- Source code to train and score models
    │
    └── visualization  <- Source code to create visualizations
```

双赢的局面。

# 结束:框架的价值

我们对 Jupyter 原型和 Python 产品代码的分离感到满意，但是我们知道我们仍然缺少一些东西。尽管尝试应用干净编码的所有原则，但是随着越来越多的步骤加入，我们用于训练和评分的端到端脚本变得有点混乱。

再一次，我们发现我们处理问题的方式有缺陷，于是我们寻找更好的解决方案。宝贵的资源再次出手相救:

*   [Norm Niemer，你的机器学习代码可能不好的 4 个原因，2019](/4-reasons-why-your-machine-learning-code-is-probably-bad-c291752e4953)
*   [Lorenzo Peppoloni，数据管道，Luigi，气流:你需要知道的一切，2018](/data-pipelines-luigi-airflow-everything-you-need-to-know-18dc741449b7)

我们研究了气流、Luigi 和 d6tflow，最终选择了 Luigi/d6tflow 管道，后者用于更简单的任务，而前者用于更高级的用例。

这一次只花了一天时间来实现整个管道:我们保存了所有的函数和类，封装了预处理、特性工程、训练和评分的逻辑，并且用管道替换了脚本。易读性和灵活性方面的改进是显著的:当我们必须改变训练集和测试集的划分方式时，我们可以只修改两个任务，保留输入和输出签名，而不用担心其他任何事情。

# 包扎

总结一下，我们得到了关于机器学习项目中代码的三个重要教训:

1.  一个机器学习项目*是*一个软件项目:我们应该关心我们代码的质量。不得不处理统计和数学不是编写糟糕代码的借口
2.  Jupyter 笔记本是很好的原型制作和共享工具，但不能取代由模块、包和脚本组成的传统的 T4 代码库
3.  有向无环图(DAG)结构非常适合数据科学和机器学习管道。当有非常好的框架可以帮助时，尝试从头创建这样的结构是没有意义的

我的读者，感谢你来到这里！

这是我第一篇关于媒介的文章，但我绝不是一个好作家。如果你有任何意见、建议或批评，我恳求你与我分享。

*还有，如果你对本帖的话题有任何疑问或疑问，欢迎随时联系！*