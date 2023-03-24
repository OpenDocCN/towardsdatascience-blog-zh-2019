# 用 Mlflow 为非傻瓜完成数据科学项目模板。

> 原文：<https://towardsdatascience.com/complete-data-science-project-template-with-mlflow-for-non-dummies-d082165559eb?source=collection_archive---------4----------------------->

## 从刚起步的忍者到大型企业团队，在本地或云中工作的每个人的最佳实践。

数据科学作为一个领域和业务职能部门已经走过了漫长的道路。现在有跨职能团队在研究算法，一直到全栈数据科学产品。

![](img/18818718c21cd37092fa8d9c52731464.png)

[Kenneth Jensen](https://commons.wikimedia.org/w/index.php?title=User:Kennethajensen&action=edit&redlink=1) [CC BY-SA 3.0]

随着数据科学的日益成熟，出现了最佳实践、平台和工具包的新兴标准，大大降低了数据科学团队的准入门槛和价格。这使得公司和从业者更容易接触到数据科学。对于大多数商业应用团队来说，数据科学家可以站在高质量开源社区的肩膀上进行日常工作。

但是成功的故事仍然被许多未能获得业务适应性的数据科学项目所掩盖。据普遍报道，超过 80%的数据科学项目仍然无法产生业务影响。有了所有高质量的开源工具包，为什么数据科学难以产生业务影响？这不是一道数学题！

数据科学的成功受到了通常称为“**最后一英里问题**”的困扰:

> **“模型的生产化是数据科学中最棘手的问题”(Schutt，R & O'Neill C .从第一线直接做数据科学，O'Reilly 出版社。加利福尼亚州，2014 年)**

在这篇博文中，我讨论了建立一个**数据科学项目**，模型开发和实验的最佳实践。数据科学项目模板的源代码可以在 GitLab 上找到:

[](https://gitlab.com/jan-teichmann/ml-flow-ds-project) [## Jan Teichmann 的基于 Mlflow 的数据科学项目模板

### GitLab.com 项目源代码

gitlab.com](https://gitlab.com/jan-teichmann/ml-flow-ds-project) 

你可以在我之前的一篇博文中读到 Rendezvous Architecture，它是在生产中操作模型的一个很好的模式:

[](/rendezvous-architecture-for-data-science-in-production-79c4d48f12b) [## 生产中数据科学的会合体系结构

### 如何构建一个前沿的数据科学平台来解决数据科学中的真正挑战:生产化。

towardsdatascience.com](/rendezvous-architecture-for-data-science-in-production-79c4d48f12b) 

# 像专家一样研究数据科学

当工程师构建生产平台、DevOps 和 DataOps 模式以在生产中运行模型时，许多数据科学家的工作方式仍然在数据科学和生产环境之间造成了**巨大的差距**。许多数据科学家(没有任何指责)

*   在本地工作**而不太考虑可能在生产中托管他们的模型的云环境**
*   **使用 python 和 pandas 在内存中处理**小型数据集**,而没有过多考虑如何将工作流扩展到生产中的大数据量**
*   **使用 **Jupyter 笔记本**工作，无需过多考虑可重复实验**
*   **很少将项目分解成独立的任务，为 ETL 步骤、模型训练和评分构建分离的管道**
*   **软件工程的最佳实践很少应用于数据科学项目，例如抽象和可重用代码、单元测试、文档、版本控制等。**

**不幸的是，我们心爱的灵活的 Jupyter 笔记本电脑在这方面发挥了重要作用。**

**![](img/d99446c377b3e9a29b7e2a396ade8a73.png)**

**From “Managing Messes in Computational Notebooks” by Andrew Head et al., doi>10.1145/3290605.3300500**

**请务必记住，数据科学是一个正在经历快速创新的领域和业务职能。如果你在我发表文章一个月后读到这篇文章，可能已经有新的工具和更好的方式来组织数据科学项目了。**成为数据科学家的激动人心的时刻**！让我们看看数据科学项目模板的详细信息:**

# **数据科学环境**

**![](img/0e934c99e0fade693f3c71c82448b47e.png)**

**Icons made by [Freepi](https://www.flaticon.com/authors/freepik)k, [phatplus](https://www.flaticon.com/authors/phatplus), [Becris](https://www.flaticon.com/authors/becris) from [www.flaticon.com](http://www.flaticon.com)**

**一个数据科学项目由许多可移动的部分组成，实际的模型可能是项目中最少的代码行。数据是您项目的燃料和**基础**，首先，我们应该为我们的项目建立坚实、高质量和可移植的基础。数据管道是大多数数据科学项目中隐藏的技术债务，您可能听说过臭名昭著的 80/20 法则:**

> ****80%的数据科学是寻找、清理和准备数据****

**我将在下面解释 Mlflow 和 Spark 如何帮助我们更高效地处理数据。**

**![](img/560e18d79839dc6f7413fda55afacc0e.png)**

**Google NIPS 2015, fair usage**

**创建您的数据科学模型本身是在**实验**和扩展项目代码库以捕获有效的代码和逻辑之间的持续往复。这可能会变得混乱，Mlflow 在这里使实验和**模型管理**对我们来说明显更容易。这也将为我们简化**模型部署**。稍后将详细介绍这一点！**

# ****数据****

**![](img/d7adc2aefd06c68341ab5335d1c1fb21.png)**

**[W.Rebel](https://commons.wikimedia.org/wiki/User:W.Rebel) [CC BY 3.0]**

**你可能听说过数据科学的 **80/20 法则**:许多数据科学工作都是关于创建数据管道来消费原始数据、清洁数据和工程特征，以最终提供给我们的模型。**

**我们希望如此**

*   **我们的数据是**不可变的:**我们只能创建新的数据集，而不能就地改变现有的数据。因此，**
*   **我们的管道是 DAG 中一组**分离的步骤**,逐步提高我们数据的质量和聚合到特性和分数中，以便在生产中轻松地将我们的 ETL 与 Airflow 或 Luigi 等工具相结合。**
*   **每个数据集都有一个定义的数据模式**T21，可以安全地读取数据，不会出现任何意外，这些数据可以记录到一个中央数据目录中，以便更好地管理和发现数据。****

**数据科学项目模板有一个保存项目数据和相关模式的数据文件夹:**

**![](img/c622c9b2f70ecf76c8c71694502490df.png)**

**[OC]**

**数据科学家通常不仅处理大数据集，还处理非结构化数据。构建数据仓库的模式需要设计工作和对业务需求的良好理解。在数据科学中，当创建 DWH 的模式时，许多问题或问题陈述是未知的。这就是为什么 Spark 已经发展成为该领域的黄金标准**

*   **使用非结构化数据**
*   **分布式企业成熟的大数据 ETL 管道**
*   **数据湖部署**
*   **统一批处理和微批处理流平台**
*   **能够在许多不同的系统之间使用和写入数据**

**拥有成熟基础设施的公司的项目使用先进的**数据湖**，其中包括用于数据/模式发现和管理的**数据目录**，以及带有气流的调度任务等。虽然数据科学家不一定要理解生产基础设施的这些部分，但是最好在创建项目和工件时牢记这一点。**

**这个数据科学项目模板使用 Spark，不管我们是在本地数据样本上运行它，还是在云中针对数据湖运行它。一方面，Spark 在本地处理小数据样本时会感觉有些矫枉过正。但另一方面，它允许一个可转移的堆栈为云部署做好准备，而无需任何不必要的重新设计工作。这也是一种重复的模式，可以很好地自动化，例如通过 Makefile。**

**虽然模型的第 1 版可能会使用来自 DWH 的结构化数据，但最好还是使用 Spark，并减少项目中的技术债务，因为模型的第 2 版将使用更广泛的数据源。**

**对于本地项目开发，我使用一个简单的 Makefile 来自动执行数据管道和项目目标。然而，这可以很容易地转化为气流或 Luigi 管道，用于云中的生产部署。**

**![](img/c2d576e15805e9568ec3fc04a192c4b5.png)**

**[OC]**

**该项目的端到端数据流由三个步骤组成:**

*   ****原始数据**:从机器学习数据库档案中下载虹膜原始数据**
*   ****中间数据**:执行特征工程流水线，用 Spark 批量实现虹膜特征**
*   ****处理数据**:执行分类模型，使用 Spark 批量实现预测**

**您可以使用以下 make 命令，通过 Spark 管道将 iris 原始数据转换为特征:**

```
make interim-data
```

**它将压缩当前项目代码库，并将**project/data/features . py**脚本提交给我们 docker 容器中的 Spark 以供执行。我们使用 Mlflow runid 来识别和加载所需的 Spark 功能管道模型，但稍后将详细介绍 Mlflow 的使用:**

**![](img/3ccb421a13aceb460883dee4a484daeb.png)**

**[OC]**

**在执行我们的 Spark 特征管道之后，我们在我们的小型本地数据湖中实现了临时特征数据:**

**![](img/de5b094fd838bf17673fdbf661f49373.png)**

**[OC]**

**如您所见，我们保存了我们的特征数据集及其相应的模式。Spark 使得保存和读取模式变得非常容易:**

```
some_sparksql_dataframe.schema.json()
T.StructType.fromJson(json.loads(some_sparksql_schema_json_string))
```

> **总是用相应的模式来具体化和读取数据！强化模式是打破数据科学中 80/20 法则的关键。**

**我认为为 csv 和 json 文件编写模式是强制性的，但我也会为任何自动保留其模式的 parquet 或 avro 文件编写模式。我真的推荐你多读一些关于[三角洲湖](https://delta.io/)、[阿帕奇胡迪](https://hudi.apache.org/)、数据目录和特色商店的内容。如果有兴趣的话，我会就这些话题发表一篇独立的博文。**

# **实验用 Jupyter 笔记本**

**Jupyter 笔记本对于实验来说非常方便，数据科学家不太可能会停止使用它们，这让许多被要求从 Jupyter 笔记本“生产”模型的工程师非常沮丧。折中的办法是利用工具的优势。在笔记本上进行实验是高效的，而且效果很好，因为在实验中证明有价值的长代码会被添加到遵循软件工程最佳实践的代码库中。**

**我个人的**工作流程**是这样的:**

1.  **实验:首先在 Jupyter 笔记本上测试代码和想法**
2.  **将有价值的代码整合到 Python 项目代码库中**
3.  **删除 Jupyter 笔记本中的代码单元格**
4.  **用从项目代码库导入的代码替换它**
5.  ****重启你的内核**和**按顺序执行所有步骤**，然后继续下一个任务。**

**![](img/cbaaa8983ee87cebbd4d31b598d84aa8.png)**

**[OC]**

**隔离我们的数据科学项目环境并管理我们的 Python 项目的需求和依赖性是很重要的。没有比通过 Docker 容器更好的方法了。我的项目模板使用来自 [DockerHub](https://hub.docker.com/r/jupyter/all-spark-notebook/) 的**jupyter all-spark-notebook Docker image**作为方便的实验室设置，包括所有电池。项目模板包含一个 docker-compose.yml 文件，Makefile 通过一个简单的**

```
make jupyter
```

**该命令将启动我们的项目所需的所有服务(Jupyter with Spark、Mlflow、Minio ),并在我们的实验环境中安装 pip 的所有项目需求。我使用 [Pipenv](https://realpython.com/pipenv-guide/) 为我的项目管理我的虚拟 Python 环境，使用 [pipenv_to_requirements](https://github.com/gsemet/pipenv-to-requirements) 为 DevOps 管道和基于 Anaconda 的容器映像创建 requirements.txt 文件。考虑到 Python 作为一种编程语言的流行程度，Python 工具有时会让人觉得笨重和复杂。😒**

**下面的屏幕截图显示了示例笔记本环境。我使用片段来设置个人笔记本，使用 [**%load magic**](https://ipython.readthedocs.io/en/stable/interactive/magics.html#magic-load) 。它还展示了我如何使用项目代码库的代码来导入原始 iris 数据:**

**![](img/0f14919f792f2b2843d16b9722c8c29b.png)**

**Jupyter 笔记本展示了我的开发工作流程，扩展了项目代码库和模型实验，并添加了一些注释。[https://git lab . com/Jan-teichmann/ml-flow-ds-project/blob/master/notebooks/Iris % 20 features . ipynb](https://gitlab.com/jan-teichmann/ml-flow-ds-project/blob/master/notebooks/Iris%20Features.ipynb)**

# **笔记本电脑的 Git 版本控制**

**![](img/10b8af3feb54fa5b305380cb766b1e3f.png)**

**public domain image**

**Git 中 Jupyter 笔记本的版本控制并不像我希望的那样用户友好。的。ipynb 文件格式不是很友好。至少，GitHub 和 GitLab 现在可以在他们的 web 界面中呈现 Jupyter 笔记本，这非常有用。**

**为了使本地计算机上的版本控制更容易，该模板还安装了 [nbdime](https://github.com/jupyter/nbdime) 工具，它使 Jupyter 笔记本的 git 差异和合并变得清晰而有意义。您可以在项目中使用以下命令:**

*   **以终端友好的方式比较笔记本电脑**
*   ****nbmerge** 三方合并笔记本，自动解决冲突**
*   ****nbdiff-web** 向您展示丰富的笔记本渲染差异**
*   **nbmerge-web 给你一个基于 web 的三路笔记本合并工具**
*   **以终端友好的方式呈现单个笔记本**

**![](img/b03ab90cc4d53e71d19217216a34e2d8.png)**

**[https://github.com/jupyter/nbdime](https://github.com/jupyter/nbdime)**

**最后但同样重要的是，项目模板使用 [IPython 钩子](https://ipython.readthedocs.io/en/stable/api/generated/IPython.core.hooks.html)来扩展 Jupyter 笔记本保存按钮，并额外调用 [nbconvert](https://nbconvert.readthedocs.io/en/latest/) 来创建一个额外的。py 脚本和。html 版本的 Jupyter 笔记本，每次您将笔记本保存在子文件夹中。一方面，这使得其他人更容易通过检查纯 Python 脚本版本的 diff 来检查 Git 中的代码更改。另一方面，html 版本允许任何人查看渲染的笔记本输出，而无需启动 Jupyter 笔记本服务器。这使得将笔记本输出的一部分通过电子邮件发送给不使用 Jupyter 的同事变得非常容易。👏**

**要在每次保存时自动转换笔记本，只需放置一个空的**。ipynb _ saveprocess**文件放在您笔记本的当前工作目录下。**

**![](img/f4950994f652bd80af26f3262f35c62e.png)**

**[OC]**

# **Mlflow 跟踪实验和日志模型**

**![](img/b4ef799c9e18d097212bdfcd70c33d93.png)**

**fair usage**

**作为我们在 Jupyter 实验的一部分，我们需要跟踪我们创建的参数、度量和工件。Mlflow 是创建可重复且可靠的数据科学项目的绝佳工具。它为中央跟踪服务器提供了一个简单的用户界面来浏览实验，并提供了强大的工具来打包、管理和部署模型。**

**在我们的数据科学项目模板中，我们使用专用跟踪服务器和存储在 S3 上的工件来模拟生产 Mlflow 部署。我们在本地使用 [Min.io](https://min.io/) 作为开源的 S3 兼容替身。无论您是在本地使用 Mlflow，还是在云中托管或完全在数据块上管理，使用 ml flow 的行为和体验都不应有任何差异。**

**docker-compose.yml 文件为我们的项目提供了所需的服务。您可以访问[http://localhost:9000/minio](http://localhost:9000/minio)上的 blob 存储 UI 和 [http://localhost:5000/](http://localhost:5000/) 上的 Mlflow 跟踪 UI**

**在 Mlflow 中，我们命名了可以运行任意次数的实验。每次运行都可以跟踪参数、指标和工件，并且有一个唯一的运行标识符。Mlflow 1.4 还刚刚发布了一个[模型注册表](https://www.mlflow.org/docs/latest/model-registry.html)，以便更容易地围绕模型生命周期组织运行和模型，例如不同版本的生产和试运行部署。**

**![](img/4c74ad54dd22aa115f0ce9c0f42a1033.png)**

**[OC]**

**这个示例项目训练两个模型:**

1.  **一种火花特征流水线**
2.  **Sklearn 分类器**

**我们的 Spark 特性管道使用 Spark ML [StandardScaler](https://spark.apache.org/docs/latest/mllib-feature-extraction.html#standardscaler) 使管道有状态。因此，我们对待特征工程的方式与对待任何其他数据科学模型的方式完全相同。本例的目的是在批评分和实时评分中同时使用使用不同框架的两个模型，而无需对模型本身进行任何重新设计。这将有望展示使用 Mlflow 简化数据科学模型的管理和部署的强大功能！**

**![](img/9a058d8b358c0b53095d1ed8dfeb1659.png)**

**Icons made by [Freepik](https://www.flaticon.com/authors/freepik), [phatplus](https://www.flaticon.com/authors/phatplus), [Smashicons](https://www.flaticon.com/authors/smashicons) from [www.flaticon.com](http://www.flaticon.com)**

**我们简单地遵循 Mlflow 惯例，将训练好的模型记录到中央跟踪服务器。**

**![](img/5fb16f25c090d3bc89c6528ba4183abc.png)**

**public domain image**

**唯一的问题是，当你试图用 boto3 上传空文件到 minio 时，当前的 boto3 客户端和 Minio 不能很好地协同工作。Spark 用 empty _SUCCESS 文件序列化模型，导致标准的 **mlflow.spark.log_model()** 调用超时。我们通过将序列化的模型保存到本地磁盘，并使用 minio 客户端记录它们，而不是使用**project . utility . ml flow . log _ artifacts _ minio()**函数，来解决这个问题。下面的代码训练一个新的 spark 特性管道，并以两种方式记录管道:Spark 和 Mleap。稍后将详细介绍 Mleap。**

**![](img/e0d67877b43c638844d159a105220173.png)**

**[OC]**

**在我们的项目中，模型被保存并记录在 models 文件夹中，不同的 docker 服务将它们的数据保存在这里。我们可以在 **models/mlruns** 子文件夹中找到来自 **Mlflow tracking server** 的数据，并在 **models/s3** 子文件夹中找到保存的工件。**

**![](img/cbfba3afc516925c81dbb3d86a73f7c3.png)**

**[OC]**

**示例项目中的 Jupyter 笔记本有望给出如何使用 Mlflow 来跟踪参数和度量以及日志模型的好主意。Mlflow 使序列化和加载模型成为一个梦想，并从我以前的数据科学项目中删除了许多样板代码。**

**![](img/3282526e689f5acd142d1ffee9f48dd1.png)**

**[OC]**

# **Mlflow 模型的批量评分**

**我们的目标是使用具有不同技术和风格的完全相同的模型对我们的数据进行批量和实时评分，而无需任何更改、重新设计或代码复制。**

**我们**批量评分**的目标是**火花**。虽然我们的特征管道已经是 Spark 管道，但是我们的分类器是一个 **Python** sklearn 模型。但不用担心，Mlflow 使处理模型变得非常容易，并且有一个方便的函数可以将 Python 模型打包到 Spark SQL UDF 中，以便在 Spark 集群中分发我们的分类器。就像魔法一样！**

1.  **加载序列化的 Spark 功能管道**
2.  **将序列化的 Sklearn 模型加载为火花 UDF**
3.  **使用特征管线转换原始数据**
4.  **将火花特征向量转化为默认火花阵列**
5.  **用展开的数组项调用 UDF**

**![](img/0effb8453e6e3e4e2b75be6186c9940f.png)**

**[OC]**

**有一些关于正确版本的 **PyArrow** 的问题，UDF 不能与火花矢量一起工作。**

**![](img/5fb16f25c090d3bc89c6528ba4183abc.png)**

**public domain image**

**但是从一个例子来看，这很容易实现。我希望这能为您省去无尽的 Spark Python Java 回溯的麻烦，也许未来的版本会进一步简化集成。现在，在 Spark 2.4 中使用 PyArrow 0.14，将 Spark 向量转换成 numpy 数组，然后转换成 python 列表，因为 Spark 还不能处理 numpy 类型。您在使用 PySpark 之前可能遇到的问题。**

**所有详细的代码都在 Git 存储库中。[https://git lab . com/Jan-teichmann/ml-flow-ds-project/blob/master/notebooks/Iris % 20 batch % 20 scoring . ipynb](https://gitlab.com/jan-teichmann/ml-flow-ds-project/blob/master/notebooks/Iris%20Batch%20Scoring.ipynb)**

# **Mlflow 模型的实时评分**

**我们希望使用我们的模型**使用 API** 实时地对请求进行交互式评分，而不是依赖 Spark 来托管我们的模型。我们希望我们的评分服务快如闪电，由**集装箱化的微服务**组成。同样，Mlflow 为我们提供了实现这一目标所需的大部分东西。**

**我们的 sklearn 分类器是一个简单的 Python 模型，将它与 API 结合并打包到容器映像中非常简单。这是一种非常常见的模式，Mlflow 为此提供了一个命令:**

```
mlflow models build-docker
```

**这就是用 Mlflow 打包 Python 风格的模型所需的全部内容。不需要用 Flask 编写 API 的重复代码来封装数据科学模型。🙌**

**不幸的是，我们的功能管道是一个火花模型。然而，我们在 **Mleap** 版本中序列化了管道，这是一个托管 Spark 管道的项目，不需要任何 Spark 上下文。**

**![](img/bb3db7b35cf5259661681d2f6c72d8b2.png)**

**fair usage**

**[Mleap](https://mleap-docs.combust.ml/) 非常适合数据量较小且我们不需要任何分布式计算且速度最为重要的使用情形。在 Makefile 中，Mleap 模型的打包是自动进行的，但包括以下步骤:**

1.  **使用 Mlflow 将模型工件下载到一个临时文件夹中**
2.  **压缩工件以进行部署**
3.  **运行**combustml/mleap-spring-boot**docker 映像，并将我们的模型工件挂载为一个卷**
4.  **使用 mleap 服务器 API 部署用于服务的模型工件**
5.  **将 JSON 数据传递给我们服务的特性管道的转换 API 端点**

**运行**make deploy-real time-model**命令，您将获得两个微服务:一个用于使用 Mleap 创建要素，另一个用于使用 Sklearn 进行分类。 **project/model/score.py** 中的 Python 脚本将对这两个微服务的调用包装成一个方便的函数，便于使用。运行**make score-real-time-model**以获得对评分服务的示例调用。**

**您也可以从 Jupyter 笔记本中调用微服务。下面的代码显示了我们的交互式评分服务有多快:**对两个模型 API 的调用加起来不到 20 毫秒**。**

**![](img/4572f37c189fd29de1f4c6574ab16b0f.png)**

**[OC] [https://gitlab.com/jan-teichmann/ml-flow-ds-project/blob/master/notebooks/Iris%20Realtime%20Scoring.ipynb](https://gitlab.com/jan-teichmann/ml-flow-ds-project/blob/master/notebooks/Iris%20Realtime%20Scoring.ipynb)**

# **测试和文件**

**示例项目使用 [Sphinx](https://www.sphinx-doc.org/en/master/) 来创建文档。进入文档目录，运行 **make html** 为你的项目生成 html 文档。只需将斯芬克斯 RST 格式的文档添加到 python 文档字符串中，并在 **docs/source/index.rst** 文件中包含要包含在生成的文档中的模块。**

**![](img/4a5956b7bc47ea90b27b9470a7db1885.png)**

**[OC]**

**单元测试放在测试文件夹中，python 单元测试可以用**

```
make test
```

**在写这篇博文的时候，数据科学项目模板和大多数数据科学项目一样，还没有测试😱我希望有一些额外的时间和反馈，这将改变！**

# ****数据块****

**![](img/80e870d48697165d12a3ea750c3c270e.png)**

**fair usage**

**没有对 Databricks 平台的讨论，任何关于 Mlflow 的博客文章都是不完整的。如果没有 Databricks 的持续开源贡献，数据科学社区将不会是现在这样，我们必须感谢他们对 Mlflow 的最初开发。❤️**

**因此，运行 MLFlow 的另一种方法是利用 Databricks 提供的 Apache Spark 的平台即服务版本。Databricks 使您能够以非常少的配置运行 MLFlow，通常称为“托管 MLFlow”。这里可以找到一个特征对比:[https://databricks.com/product/managed-mlflow](https://databricks.com/product/managed-mlflow)**

**只需用 pip 将 MLFlow 包安装到您的项目环境中，您就拥有了所需的一切。值得注意的是，Databricks 中的 MLFlow 版本不是已经描述过的完整版本。这是一个在 Databricks 生态系统内部工作的优化版本。**

**![](img/a0e1cfe08d58a855881518ca30a2dde4.png)**

**[OC]**

**当你创建一个新的“MLFlow 实验”时，系统会提示你输入一个项目名称和一个要用作人工制品存储的人工制品位置。该位置表示您将捕获在 MLFlow 实验期间生成的数据和模型的位置。您需要输入的位置需要遵循以下约定“dbfs:/ <location in="" dbfs="">”。DBFS 中的位置可以是 DBFS(数据块文件系统)，也可以是使用外部装载点映射的位置，例如 S3 存储桶。</location>**

**![](img/0ce482b8dd222e44df59259444b6eecb.png)**

**[OC]**

**配置 MLFlow 实验后，您将看到实验跟踪屏幕。在这里，您可以看到模型在训练时的输出。您可以根据参数或指标灵活地过滤多次运行。创建实验后，您需要记下实验 ID。在每个笔记本中配置 MLFlow 的过程中，您需要使用它来指回这个单独的位置。**

**![](img/2eeb60302eb4926eceff435fa5cced40.png)**

**[OC]**

**要将实验跟踪器连接到您的模型开发笔记本，您需要告诉 MLFlow 您正在使用哪个实验:**

```
with mlflow.start_run(experiment_id="238439083735002")
```

**一旦 MLFlow 被配置为指向实验 ID，每次执行将开始记录和捕获您需要的任何指标。除了度量标准，您还可以捕获参数和数据。数据将与创建的模型存储在一起，这为前面讨论的可重用性提供了一个很好的管道。要开始记录参数，只需添加以下内容:**

```
mlflow.log_param("numTrees", numTrees)
```

**要记录损失指标，您可以执行以下操作:**

```
mlflow.log_metric("rmse", evaluator.evaluate(predictionsDF))
```

**![](img/5185ea6176bbc3e371ffdea5f16ea82b.png)**

**[OC]**

**一旦度量被捕获，就很容易看到每个参数是如何对模型的整体有效性做出贡献的。有各种可视化来帮助你探索不同的参数组合，以决定哪种模型和方法适合你正在解决的问题。最近 MLFlow 实现了一个自动记录功能，目前只支持 Keras。当这个被打开时，所有的参数和度量将被自动捕获，这真的很有帮助，它大大减少了你需要添加的模板代码的数量。**

**![](img/550b170a1133a4007299ce1571b8cae9.png)**

**[OC]**

**如前所述，可以使用以下工具记录模型:**

```
mlflow.mleap.log_model(spark_model=model, sample_input=df, artifact_path="model")
```

**如果您已经在使用数据块，托管 MLflow 是一个很好的选择。实验捕捉只是提供的众多功能之一。在 Spark & AI 峰会上，宣布了支持模型版本控制的 MLFlows 功能。当与 MLFlow 的跟踪功能结合使用时，使用新的模型注册中心只需点击几下鼠标，就可以将模型从开发转移到生产中。**

# **摘要**

**在这篇博文中，我记录了我的[固执己见的]数据科学项目模板，该模板在本地开发时考虑了云中的生产部署。这个模板的全部目的是应用最佳实践，减少技术债务和避免重新设计。**

**它演示了如何使用 Spark 创建数据管道和带有 Mlflow 的日志模型，以便于管理实验和部署模型。**

**我希望这能为你节省数据科学的时间。❤️**

# **密码**

**![](img/2904a165ae6b484c9cb8d59c2041523c.png)**

**public domain image**

**数据科学项目模板可在 GitLab 上找到:**

**[](https://gitlab.com/jan-teichmann/ml-flow-ds-project) [## Jan Teichmann 的基于 Mlflow 的数据科学项目模板

### GitLab.com 项目源代码

gitlab.com](https://gitlab.com/jan-teichmann/ml-flow-ds-project) 

# 信用

我并不总是一个优秀的工程师和解决方案架构师。我正站在巨人的肩膀上，我要特别感谢我的朋友们[来自 www.advancinganalytics.co.uk](https://www.linkedin.com/in/tpmccann/)[的特里·麦肯](http://www.advancinganalytics.co.uk)和[西蒙·怀特利](https://www.linkedin.com/in/simon-whiteley-uk/)

![](img/be63dedc941ade24aacf866f5a513135.png)****![](img/b40fa03f9762d1ec3c427365a4c45786.png)**

**Jan 是公司数据转型方面的成功思想领袖和顾问，拥有将数据科学大规模应用于商业生产的记录。他最近被 dataIQ 评为英国 100 位最具影响力的数据和分析从业者之一。**

****在 LinkedIn 上连接:**[**https://www.linkedin.com/in/janteichmann/**](https://www.linkedin.com/in/janteichmann/)**

****阅读其他文章:**[**https://medium.com/@jan.teichmann**](https://medium.com/@jan.teichmann)**