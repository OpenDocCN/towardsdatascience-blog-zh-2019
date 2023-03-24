# 使用 MLRun 跟踪数据科学实验的最简单方法

> 原文：<https://towardsdatascience.com/the-easiest-way-to-track-data-science-experiments-with-mlrun-d2df0ac147d9?source=collection_archive---------21----------------------->

![](img/98347d7b909c5126278801e8890603fc.png)

我遇到的几乎每个客户都处于开发基于 ML 的应用程序的某个阶段。有些才刚刚起步，有些已经投入巨资。看到数据科学这个曾经常用的时髦词汇如何成为几乎所有公司的真正战略，真是令人着迷。

在下面的文章中，我将解决客户一再提出的挑战之一——运行和调优实验跟踪。通过一步一步的教程，我将涵盖复杂性问题，并展示如何用 MLRun 解决它们，ml run 是一个新的开源框架，它优化了机器学习操作的管理。

MLRun 是一个开源框架，它为数据科学家和开发人员/工程师提供了一个通用且易于使用的机制，用于描述和跟踪机器学习相关任务(执行)的代码、元数据、输入和输出。

MLRun 跟踪各种元素，将它们存储在数据库中，并在单个报告中显示所有正在运行的作业以及历史作业。

数据库位置是可配置的，用户可以根据标准运行查询来搜索特定的作业。用户在本地 IDE 或笔记本上运行 MLRun，然后使用横向扩展容器或函数在更大的集群上运行相同的代码。

**安装 MLRun**

安装 MLRun 库，运行 import 命令并设置 MLRun 数据库的路径:

```
!pip install git+https://github.com/mlrun/mlrun.gitfrom mlrun import new_function, RunTemplate, NewRun, get_run_db
import yaml
import pandas as pd
# set mlrun db path (can also be specified in run_start command)
%env MLRUN_DBPATH=./
```

**关键要素**

*   任务(运行)-使您能够定义运行所需的参数、输入、输出和跟踪。可以从模板创建运行，并在不同的运行时或函数上运行。
*   功能—特定于运行时的软件包和属性(例如，映像、命令、参数、环境等)。).一个函数可以运行一个或多个运行/任务，并且可以从模板中创建。
*   运行时——一个计算框架。MLRun 支持多种运行时，如本地 Kubernetes 作业、DASK、Nuclio、Spark 和 mpijob (Horovod)。运行时可以支持并行和集群(即在进程/容器之间分配工作)。

**在代码中添加 MLRun 钩子**

为了捕获上面提到的数据，在作业本身中添加钩子。您可以跟踪作业的结果、模型文件或 csv 文件之类的工件、源代码、标签等等。

MLRun 引入了 ML 上下文的概念。检测代码以从上下文中获取参数和输入，以及日志输出、工件、标签和时序度量。

```
from mlrun.artifacts import ChartArtifact, TableArtifact, PlotArtifact
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd# define a function with spec as parameter
import time
def handler(context, p1=1, p2='xx'):
    # access input metadata, values, and inputs
    print(f'Run: {context.name} (uid={context.uid})')
    print(f'Params: p1={p1}, p2={p2}')

    time.sleep(1)

    # log the run results (scalar values)
    context.log_result('accuracy', p1 * 2)
    context.log_result('loss', p1 * 3)

    # add a lable/tag to this run 
    context.set_label('category', 'tests')

    # log a simple artifact + label the artifact 
    context.log_artifact('model.txt', body=b'abc is 123', labels={'framework': 'xgboost'})

    # create a matplot figure and store as artifact 
    fig, ax = plt.subplots()
    np.random.seed(0)
    x, y = np.random.normal(size=(2, 200))
    color, size = np.random.random((2, 200))
    ax.scatter(x, y, c=color, s=500 * size, alpha=0.3)
    ax.grid(color='lightgray', alpha=0.7)

    context.log_artifact(PlotArtifact('myfig', body=fig))

    # create a dataframe artifact 
    df = pd.DataFrame([{'A':10, 'B':100}, {'A':11,'B':110}, {'A':12,'B':120}])
    context.log_artifact(TableArtifact('mydf.csv', df=df, visible=True)) return 'my resp'
```

**运行作业**

具体到这个例子，我们在 Jupyter 中使用了一个内联代码。但是请注意，您也可以将其保存为 python 文件并运行。

接下来，定义任务及其参数，例如任务名称、输入参数、秘密文件(如果需要)和标签。

创建一个函数，指定命令(即我们的 Python 脚本)并将其分配给之前定义的任务。

作业完成后，它将显示作业运行时在 MLRun 跟踪数据库中捕获的信息。

```
task = NewRun(handler=handler,name='demo', params={'p1': 5}).with_secrets('file', 'secrets.txt').set_label('type', 'demo')
run = new_function().run
```

作业被捕获到 MLRun 数据库中。

![](img/e05309f7a4affdd37ac98584c1bf12bd.png)

**使用超级参数运行作业**

超参数非常重要，因为它们直接控制训练算法的行为，并对被训练模型的性能有重大影响。数据科学家通常希望用不同的参数运行同一个模型，以找出最适合的配置。

首先，创建一个模板(见下文)，然后用 hyper_params 标志运行它。

在这种情况下，MLRun 跟踪每个正在运行的实例的信息，从而便于在不同的运行之间进行比较。

基于下面的示例，运行训练作业的三次迭代，每次使用不同的参数，并查看作业如何确定损失最低的迭代。

```
task = NewRun(handler=handler).with_hyper_params({'p1': [5, 2, 3]}, 'min.loss')
run = new_function().run(task)
```

**查看作业结果**

一旦工作完成，结果就会显示出来。单击“iteration_results”工件，打开一个包含详细信息的窗口。在我们的例子中，第二次迭代具有最低的损失分数，这被定义为我们的最佳拟合函数，因此显示为“最佳迭代”。

![](img/a7ce44357206d75c7ecbb8d3d73c930c.png)

显示迭代结果

![](img/db85e37d20eb60740cf3772cc289e68d.png)

**使用分布式框架(即 DASK)运行作业**

现在，让我们在集群上运行相同的作业，作为大规模运行的分布式进程。你需要做的只是改变框架，而不需要改变任何代码。

在下面的例子中，我使用了 DASK 框架，它与 Iguazio 数据科学平台相集成，因此不需要 DevOps。

```
%%timeit -n 1 -r 1
run = new_function(command='dask://').run(task, handler=handler)
```

**总结**

如果您试图独自完成所有工作，运行和跟踪数据科学工作可能需要大量开发工作和劳动密集型任务。MLRun 提供了在您的笔记本电脑上本地运行任何作业的灵活性，或者以更简单的方式在大规模的分布式集群上运行任何作业。用户可以轻松运行报告来获得每个作业的详细信息，如果需要，可以使用所有工件和元数据重新生成和运行任何以前的作业。

欲了解更多关于 MLRun 的详情，请访问 MLRun 回购[https://github.com/mlrun/mlrun](https://github.com/mlrun/mlrun)
[https://github.com/mlrun/demos](https://github.com/mlrun/demos)

**接下来的步骤**

在 Iguazio 的数据科学平台上运行，可以挑选各种内置框架并大规模运行作业，从而在整个机器学习管道中提高性能并消除基础设施开销。