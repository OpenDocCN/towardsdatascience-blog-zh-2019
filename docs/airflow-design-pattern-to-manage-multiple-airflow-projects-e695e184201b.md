# 管理多个气流项目的气流设计模式

> 原文：<https://towardsdatascience.com/airflow-design-pattern-to-manage-multiple-airflow-projects-e695e184201b?source=collection_archive---------13----------------------->

## 探索数据工程技术和代码，以在 Airflow 实例上或作为 ECS 服务连续部署多个项目

![](img/71e97aef19fcf437ec2dbf49e1bfd6d0.png)

Photo by I[an Dooley](https://unsplash.com/@sadswim?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

Airflow 是调度、可视化和执行数据管道的一个很好的工具。但是如果你像我一样，必须管理大约 100 多个不同的管道，你会很快意识到开发和管理这些管道需要一点工程技术。

我们将探索设计模式的 3 个方面，这将帮助我们使开发过程简单和易于管理。

1.  每个项目 DAG 的单独存储库。
2.  使用 CI/CD 部署代码。
3.  使用容器执行代码。

# 1.项目分离:如何维护每条管道的单独存储库？

Airflow 将自动初始化配置中指定的 DAG 主目录下的所有 DAG。但是要从单独的项目初始化 DAG.py，我们必须安装 *DAGFactory* 。

**Dag factory**:*Dag factory 将从特定文件夹下的单个项目中收集并初始化 Dag，在我的例子中是@ airflow/projects。*

```
*airflow
├── READEME.md
├── airflow-scheduler.pid
├── airflow.cfg
├── dags
│   └─ DAGFactory.py
├── global_operators
├── logs
├── projects
│     └── test_project
│            └── DAG.py
├── requirements.txt
├── scripts
├── tests
└── unittests.cfg*
```

*   *将其安装在`airflow/dags/DAGFactory.py`下*

*气流将初始化气流/dags 目录下的所有 Dag。所以我们将在这里安装 DAGFactory。*

```
*airflow
├── READEME.md
├── airflow-scheduler.pid
├── airflow.cfg
└── dags
    └─ DAGFactory.py*
```

*`DAGFactory.py`的代码:遍历 airflow/projects 下的所有项目，并将`DAG.py`中的`DAGS`变量加载到 airflow 的全局名称空间中。*

*这将允许我们将您的代码放在一个单独的存储库中。我们要做的就是*

1.  *在你的回购协议的根层有一个`DAG.py`文件*
2.  *有一个名为`DAGS`的列表，里面有所有的*主*Dag。(可以，可以传多个 Dag。每个 dag 将在用户界面上显示为单独的 DAG。并且所有 Dag 都可以从相同的代码库中维护。)*

# *2.如何将项目的代码应用到生产气流服务中*

1.  *如果你在虚拟机上运行气流。你可以在`airflow/projects`下`git checkout` 这个项目*
2.  *您可以使用类似`chef`的配置管理工具在 airflow VM 上部署新项目。*
3.  *您可以在 AWS 上使用代码构建或代码管道来构建和部署您的 Kubernetes 服务。如果你在 EKS 运营 airflow 服务。(或 GCP 的 Kubernetes 发动机)*

*我将很快就这些写一篇文章。*

# *3.创建复杂的 Dag v/s 编写复杂的代码并在容器中执行。*

*随着 Dag 数量的增加及其复杂性的增加。执行它们需要更多的资源，这意味着如果您的生产气流环境在虚拟机上，您将不得不进行扩展以满足不断增长的需求。(如果您使用 Kubernetes 来执行或任何其他执行人，这将不适用于您)。*

*现在，为了克服这个问题，我建议为每个项目创建单独的容器，并通过气流在 ECS 中执行它们，而不是在单个 m/c 上执行所有的 Dag。*

*这种方法让我将我的 airflow 实例大小保持在最小，容器在 ECS 中的 FARGATE 上执行，我只为它们运行所需的时间付费。*

*由于我可以通过使用 DAGFactory 将我的代码库分开，所以我可以将`DAG.py`和`dockerfile` 放在同一个存储库中。我可以将 DAG.py 放在 ECR 中的 airflow 实例和 Docker 容器上。*

*这种方法的一个缺点是，您不能在内部使用气流操作符进行 ETL 或其他处理。但是您仍然可以构建您的模块库，并在不同的管道中使用它们。*

*那么为什么要保持气流呢？*

*好吧，Airflow 附带了一个更好的 UI 和一个调度程序。这使得跟踪作业失败、启动作业(本例中是一个容器)以及在一个地方查看日志变得很容易。*

*简而言之，不需要登录到实例来调试问题。*