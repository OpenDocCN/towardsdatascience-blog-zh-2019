# 如何使用气流而不头疼

> 原文：<https://towardsdatascience.com/how-to-use-airflow-without-headaches-4e6e37e6c2bc?source=collection_archive---------3----------------------->

![](img/edbd9e71b6de773aaf955f2ef5854c5b.png)

Photo by [Sebastian Herrmann](https://unsplash.com/@officestock?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

按计划处理和移动数据的数据管道和/或批处理作业为所有美国数据人员所熟知。事实上的标准工具来编排所有的是[阿帕奇气流](https://airflow.apache.org/)。它是一个以编程方式创作、调度和监控工作流的平台。工作流是表示为直接非循环图(DAG)的一系列任务。例如，将提取、转换、加载(ETL)作业视为一个工作流/DAG，其中 E、T 和 L 步骤是其任务。您可以使用 Python 在代码中配置工作流。这允许您在 Git 这样的源代码控制系统中对工作流进行版本控制，这非常方便。

总而言之，气流是一个很棒的工具，我喜欢它。但是，我最初用错了方法，可能其他人也是这样。这种误用会导致令人头疼的问题，尤其是在工作流部署方面。为什么？简而言之，我们将它用于在同一个气流实例上协调工作流*和运行任务的* **。在这篇文章中，我将告诉你为什么这是一个问题。当然，我还将向您展示如何轻松地解决这个问题。我希望这能减少你未来的阿司匹林摄入量，就像我一样:)**

# 气流流向不好的方向

我以一个关于我自己和气流的小故事开始这篇文章。

创建工作流时，您需要实现和组合各种任务。在 Airflow 中，您使用运算符来实现任务。Airflow 提供了一组开箱即用的操作符，比如 [BashOperator](https://airflow.apache.org/howto/operator/bash.html) 和 [PythonOperator](https://airflow.apache.org/howto/operator/python.html) 等等。显然，我在工作中大量使用了 PythonOperator，因为我是一名数据科学家和 Python 爱好者。这开始很好，但是过了一会儿，我想

> “嗯，我如何将我的工作流部署到我们的生产实例中？我的包和其他依赖项是如何安装在那里的？”

一种方法是为每个工作流添加一个 requirements.txt 文件，该文件在部署时安装在所有 Airflow workers 上。我们试过了，但是我的任务和同事的任务需要不同的熊猫版本。由于这个**包依赖问题**，工作流不能在同一个 Python 实例上运行。不仅如此，我使用了需要 Python 3.7 的 [Dataclasses](https://docs.python.org/3/library/dataclasses.html) 包，但是在生产实例上，我们只安装了 Python 3.6。这可不好。

所以我继续谷歌了一下，找到了另一个由 Airflow 提出的解决方案，名为 Packaged-DAGs。上面写着:“将所有 Dag 和外部依赖项打包成一个 zip 文件，然后部署它。”您可以点击此[链接](https://airflow.apache.org/concepts.html#packaged-dags)了解更多详情。对我来说，这听起来不是一个很好的解决方案。它也没有解决 Python **版本问题**。

作为操作符提供的另一种可能性是将您的任务包装在一个 [PythonVirtualEnvOperator](https://airflow.apache.org/_api/airflow/operators/python_operator/index.html?highlight=pythonvirtualenvoperator#airflow.operators.python_operator.PythonVirtualenvOperator) 中。我不得不说，我没有试过，因为它仍然没有解决 Python 版本的问题。但是，我想如果你每次执行一个任务都要创建一个虚拟环境的话，可能会很慢。除此之外，文档中还提到了其他一些注意事项。

Damm，三次尝试仍然没有令人满意的解决方案来部署用 Python 编写的任务。

最后，我问自己如何使用 Python 之外的语言编写任务？那一步会不会变成**语言不可知**？甚至有人能执行一个**不知道具体气流**的任务吗？我如何在气流中部署和集成这样的任务？我是否必须编译它，并在每个 worker 上安装带有所有依赖项的结果，以便最终通过 BashOperator 调用它？这听起来像是一次痛苦的部署和开发经历。此外，这可能再次导致依赖性冲突。总之，这听起来不太令人满意。

> *但是我要用气流！*

那么，我们能修好它吗？

![](img/e4270a884ef4b31e4560a9c3ab373e0e.png)

Taken from [https://knowyourmeme.com/photos/292809-obama-hope-posters](https://knowyourmeme.com/photos/292809-obama-hope-posters)

# 良好的气流方式

每当我听到“*依赖冲突*”、“*版本问题*”或“*我想成为语言不可知者*”之类的话，我马上会想到容器和 **Docker** 。幸运的是，Airflow 提供了一个开箱即用的[**docker operator**](https://airflow.apache.org/_api/airflow/operators/docker_operator/index.html?highlight=dockeroperator#airflow.operators.docker_operator.DockerOperator)(还有一个用于 [Kubernetes](https://airflow.apache.org/_api/airflow/contrib/operators/kubernetes_pod_operator/index.html?highlight=kubernetes#module-airflow.contrib.operators.kubernetes_pod_operator) )。这允许我们从气流中调用隔离的 docker 容器作为任务。

好快啊:)

但是现在，更详细一点。在下文中，我将向您展示从开发基于 Docker 的任务和 Dag 到部署它们的端到端过程。对于每一步，我都突出了各自解决的问题。

## 任务开发和部署

1.  用你想要的任何语言和语言版本来开发和测试你的任务。*这允许你在与气流细节和其他任务隔离的情况下测试和开发任务。它降低了新开发人员的门槛，因为他们可以选择自己最熟悉的语言。*
2.  将工件和所有依赖项打包到 Docker 映像中。*这解决了依赖和版本冲突的问题。它大大有助于减轻你的头痛。*
3.  使用 DockerOperator 从容器中公开一个入口点来调用和参数化任务。*这使你能够从气流中使用你的图像。*
4.  构建您的图像，标记它，并将其推送到一个中心 Docker 注册表。理想情况下，这是自动化的，并且是您的 CI/CD 渠道的一部分。气流将图像从注册表中拉到执行任务的机器上。这有助于部署任务。又是一个减少头痛的步骤。此外，它允许您拥有同一任务的多个版本。最后，拥有一个中央注册中心使您能够在整个组织中共享和重用任务。

## DAG 开发和部署

1.  使用 DockerOperator 作为唯一的操作符来构建 DAG。使用它来调用 Docker 注册表中的各种任务。对我来说，这使我的 DAG 定义变得小巧、清晰、易读。除了 DockerOperator 之外，我也不必学习任何特定的气流操作符。
2.  将您的 DAG 放入版本控制系统。*这样，部署您的 DAG 只是一个简单的推拉动作。同样，这应该是自动化的，并成为您的 CI/CD 渠道的一部分。*

我希望这听起来合理，你会得到它提供的许多好处。你所要做的就是[学习 Docker](https://medium.com/@simon.hawe/how-to-build-slim-docker-images-fast-ecc246d7f4a7) 如果你还不知道并且你已经准备好了。

现在，我们知道**该做什么了。下一段，我们通过一个例子来看看**是怎么做的**是怎么做的。**

# 例子

让我们快速浏览一个示例 DAG，其中我只使用了 DockerOperator。使用这个 DAG，我模拟了一个 ETL 作业。请注意，DAG 定义文件并没有显示它只是一个模拟的 ETL 作业。它可能是一个超级复杂的或者只是一个虚拟的。实际的复杂性从 DAG 定义中去除，并转移到各自的任务实现中。生成的 DAG 定义文件简洁易读。你可以在我的 [Github 库中找到所有代码。](https://github.com/Shawe82/airflow-tutorial)您不仅可以在这里找到 DAG 定义，还可以了解如何使用 Docker 构建和运行相应的 Airflow 实例。您还可以找到模拟的 ETL 任务实现以及 Dockerfile 文件。但是现在，让我们具体点。

## 代码

首先，让我们导入必要的模块，并为 DAG 定义默认参数。

```
**from** datetime **import** datetime, timedelta
**from** airflow **import** DAG
**from** airflow.operators.docker_operator **import** DockerOperator
d_args = {
    "start_date": datetime(2019, 11, 14),
    "owner": "shawe",
    "depends_on_past": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
}
```

这里没有什么特别的，但是您可以看到我只导入了 DAG，这是必需的，还有 DockerOperator。在这个阶段，我们不需要气流以外的东西。

在我们构建 DAG 之前，关于 DockerOperator 和您将看到的内容，还有一些话要说。我的所有任务都使用同一个 Docker 映像，名为 **etl-dummy** ，标记为 **latest** 。该映像提供了一个名为 **etl** 的 CLI。该 CLI 有 3 个子 CLI，分别是**提取**、**转换**和**加载**。子 CLI 有不同的参数。例如，要运行转换任务，您需要调用

```
etl --out-dir /some/path transform --upper
```

要从 DockerOperator 中调用 Docker 图像，只需将图像名称指定为 name:tag，并指定要调用的命令。请注意，我的图像存储在本地 Docker 注册表中。这样，我的 etl dag 定义看起来像

```
si = "@hourly"
**with** DAG("etl", default_args=d_args, schedule_interval=si) **as** dag:
    **def** etl_operator(task_id: str, sub_cli_cmd: str):
        out_dir = "/usr/local/share"
        **cmd = f"'etl --out-dir {out_dir} {**sub_cli_cmd**}'"**
        **return** DockerOperator(
            command=cmd,
            task_id=task_id,
            image=f"**etl-dummy:latest**",
            volumes=[f"/usr/local/share:{out_dir}"],
        )
 extract = etl_operator("e-step", **"extract --url http://etl.de"**)
    transform = etl_operator("t-step", **"transform --lower"**)
    load = etl_operator("l-step", **"load --db 9000://fake-db"**)
    **# Combine the tasks to a DAG**
    extract >> transform >> load
```

我只是添加了一个小的助手函数来创建 DockerOperators。对我来说，这看起来又好又干净。

## 最后的笔记和提示

如果您在一个可远程访问的 Docker 注册表中托管您的图像，您必须以*registry-name/image-name:image-tag*的形式传递图像名称。此外，您必须提供一个 *docker_conn_id* 来使气流能够访问注册表。这个 docker_conn_id 引用了一个由 Airflow 管理的秘密。

您可以添加的另一个功能是将图像名称和标签作为变量存储在 Airflow 中。如果你想更新你的 DAG，你所要做的就是用一个新的标签将另一个图像推送到你的注册表中，并改变 Airflow 中变量的值。

最后，我想重复一遍，你可以在我的 [**Github 资源库中找到包括 Docker 上的 Airflow 和示例 Docker 镜像在内的所有代码。**](https://github.com/Shawe82/airflow-tutorial)

# 包裹

我希望这篇文章对你有用，如果你过去有过头痛，我希望它们将来会消失。感谢您关注这篇文章。一如既往，如有任何问题、意见或建议，请随时联系我。