# 如何将气流式 Dag 用于高效的数据科学工作流

> 原文：<https://towardsdatascience.com/how-to-use-airflow-style-dags-for-highly-effective-data-science-workflows-b52053dedd32?source=collection_archive---------10----------------------->

Airflow 和 Luigi 对于数据工程来说很棒，但是对于数据科学来说并没有优化。 [d6tflow](https://github.com/d6t/d6tflow) 为数据科学带来了气流式的 Dag。

# 数据科学工作流是 Dag

数据科学工作流通常如下所示。

![](img/db2a93224bcc4c179e8a5669d6703328.png)

此工作流类似于数据工程工作流。它包括将参数化的任务链接在一起，这些任务在彼此之间传递多个输入和输出。参见[为什么你的机器学习代码可能不好的 4 个原因](/4-reasons-why-your-machine-learning-code-is-probably-bad-c291752e4953)为什么在函数之间传递数据或硬编码文件/数据库名称而没有明确定义任务依赖关系不是编写数据科学代码的好方法。

```
# bad data science code
def process_data(data, do_preprocess):
    data = do_stuff(data, do_preprocess)
    data.to_pickle('data.pkl')data = pd.read_csv('data.csv')
process_data(data, True)
df_train = pd.read_pickle(df_train)
model = sklearn.svm.SVC()
model.fit(df_train.iloc[:,:-1], df_train['y'])
```

# R&D 与生产数据工作流

使用[气流](https://airflow.apache.org/)和 [luigi](https://github.com/spotify/luigi) 比编写用于管理数据工作流的功能代码前进了一大步。虽然数据工程和数据科学工作流相似，但它们的注意事项是相同的。这两个库都是为数据工程师在生产环境中使用而设计的，其重点是:

*   确保一切按时顺利进行
*   时间安排和协调
*   从故障中恢复
*   数据质量

相比之下，研发工作流程的重点是:

*   产生洞察力
*   原型速度
*   评估不同模型和参数的预测能力
*   可视化输出

因此，研发工作流程:

*   不太明确
*   涉及反复试验
*   当模型、参数和数据发生变化时，需要频繁重置任务和输出
*   接受数据工程师的输出

# R&D 环境中的气流/luigi 问题

由于这两个库都针对数据工程生产环境进行了优化，因此数据科学研发环境的 UX 并不是很大:

*   用于读/写数据的湿代码
*   手动跟踪保存数据的文件名或数据库表名
*   当模型、参数和数据改变时，不方便重新设置任务
*   不方便跟踪不同参数设置的模型结果

在复杂的数据工作流中手动跟踪文件名…不可扩展。

```
# example of manually caching dataframes and keep track of files
cfg_fpath_cc_base = cfg_fpath_base + 'vendor/'
cfg_fpath_cc_raw = cfg_fpath_cc_base + 'df_cc_raw.pkl'
cfg_fpath_cc_raw_recent2 = cfg_fpath_cc_base + 'df_cc_raw_recent2.pkl'
cfg_fpath_cc_yoy = cfg_fpath_cc_base + 'df_cc_yoy.pkl'
cfg_fpath_cc_yoy_bbg = cfg_fpath_cc_base + 'df_cc_yoy_bbg.pkl'
cfg_fpath_cc_yoy_fds = cfg_fpath_cc_base + 'df_cc_yoy_fds.pkl'
cfg_fpath_cc_var_fds = cfg_fpath_cc_base + 'df_cc_var_fds.pkl'
cfg_fpath_cc_yoy_recent2 = cfg_fpath_cc_base + 'df_cc_yoy_recent2.pkl'
cfg_fpath_cc_actual = cfg_fpath_cc_base + 'df_cc_sales_actual.pkl'
cfg_fpath_cc_monthly = cfg_fpath_cc_base + 'df_cc_monthly.pkl'
cfg_fpath_cc_yoy_cs2 = 'data/processed/df_cc_yoy_cs2.pq' # consistent shopper data for new methodology from 2018
```

# d6tflow 与 airflow/luigi 有何不同

d6tflow 针对数据科学研究和开发工作流进行了优化。以下是在数据科学中使用 d6tflow 的好处。

# 好处:任务有输入和输出数据

不再需要手动加载和保存数据，而是外包给图书馆。这可以更好地伸缩，并减少维护，因为输入/输出数据的位置可以改变，而不必重写代码。这也使得数据工程师更容易将数据交给数据科学家。

```
class TaskProcess(d6tflow.tasks.TaskPqPandas): # define output format def requires(self):
        return TaskGetData() # define dependency def run(self):
        data = self.input().load() # load input data
        data = do_stuff(data) # process data
        self.save(data) # save output data
```

# 好处:轻松使任务无效

实现了常见的失效场景。当您在试错过程中更改代码和数据时，这将提高原型开发速度。

```
# force execution including downstream tasks
d6tflow.run(TaskTrain(), force=TaskGetData())# reset single task
TaskGetData().invalidate()# reset all downstream tasks
d6tflow.invalidate_downstream(TaskGetData(), TaskTrain())# reset all upstream tasks
d6tflow.invalidate_upstream(TaskTrain())
```

# 优点:使用不同的参数轻松训练模型

更改参数后，您可以智能地重新运行工作流。参数从目标任务传递到相关的下游任务。因此，您不再需要手动跟踪要更新的任务，从而提高了原型开发速度并减少了错误。

```
d6tflow.preview(TaskTrain(do_preprocess=False))'''
└─--[TaskTrain-{'do_preprocess': 'False'} (PENDING)]
   └─--[TaskPreprocess-{'do_preprocess': 'False'} (PENDING)]
      └─--[TaskGetData-{} (COMPLETE)] => this doesn't change and doesn't need to rerun
'''
```

# 好处:轻松比较模型

用不同参数训练的不同模型可以容易地加载和比较。

```
df_train1 = TaskPreprocess().output().load()
model1 = TaskTrain().output().load()
print(sklearn.metrics.accuracy_score(df_train1['y'],model1.predict(df_train1.iloc[:,:-1])))df_train2 = TaskPreprocess(do_preprocess=False).output().load()
model2 = TaskTrain(do_preprocess=False).output().load()
print(sklearn.metrics.accuracy_score(df_train2['y'],model2.predict(df_train2.iloc[:,:-1])))
```

# 加速数据工程师向数据科学家的移交

要快速共享工作流输出文件，可以使用 [d6tpipe](https://github.com/d6t/d6tpipe) 。参见[共享工作流和输出](https://d6tflow.readthedocs.io/en/latest/collaborate.html)。

```
import d6tflow.piped6tflow.pipe.init(api, 'pipe-name') # save flow output 
pipe = d6tflow.pipes.get_pipe()
pipe.pull()class Task2(d6tflow.tasks.TaskPqPandas): def requires(self):
        return Task1() # define dependency def run(self):
        data = self.input().load() # load data from data engineer
```

或者，您可以使用 [d6tflow premium](https://pipe.databolt.tech/gui/request-premium/) 将输出保存在数据库中。

```
d6tflow2.db.init('postgresql+psycopg2://usr:pwd@localhost/db', 'schema_name')class Task1(d6tflow2.tasks.TaskSQLPandas): def run(self):
        df = pd.DataFrame()
        self.save(df)
```

最后，数据科学家可以继承数据工程师编写的任务来快速加载源数据。

```
import tasks_factors # import tasks written by data engineer
import utilsclass Task1(tasks_factors.Task1):
    external = True # rely on data engineer to run def run(self):
        data = self.input().load() # load data from data engineer
```

# d6tflow 快速入门

这是一个完整的例子，展示了如何使用 d6tflow 进行 ML 工作流[https://github.com/d6t/d6tflow#example-output](https://github.com/d6t/d6tflow#example-output)

# 可扩展 ML 项目的模板

在[https://github.com/d6t/d6tflow-template](https://github.com/d6t/d6tflow-template)有一个用于真实项目的 d6t 流代码模板

*   多任务输入和输出
*   参数继承
*   模块化任务，运行和