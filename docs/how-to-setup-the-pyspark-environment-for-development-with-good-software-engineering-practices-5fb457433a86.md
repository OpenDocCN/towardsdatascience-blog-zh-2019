# 如何使用良好的软件工程实践来设置 Python 和 Spark 开发环境

> 原文：<https://towardsdatascience.com/how-to-setup-the-pyspark-environment-for-development-with-good-software-engineering-practices-5fb457433a86?source=collection_archive---------7----------------------->

![](img/810058a93097118f26c91f8ceef0146f.png)

在本文中，我们将讨论如何设置我们的开发环境以创建高质量的 python 代码，以及如何自动化一些繁琐的任务以加快部署。

我们将回顾以下步骤:

*   使用 [**pipenv**](https://pipenv.readthedocs.io/en/latest/) 在隔离的虚拟环境中设置我们的依赖关系
*   如何为多项任务设置项目结构
*   如何运行 pyspark 作业
*   如何使用 [**Makefile**](https://opensource.com/article/18/8/what-how-makefile) 来自动化开发步骤
*   如何使用 [**flake8**](http://flake8.pycqa.org/en/latest/) 测试我们代码的质量
*   如何使用 [**pytest-spark**](https://pypi.org/project/pytest-spark/) 为 PySpark 应用运行单元测试
*   运行测试覆盖，看看我们是否已经使用 [**pytest-cov**](https://pypi.org/project/pytest-cov/) 创建了足够的单元测试

# 步骤 1:设置虚拟环境

虚拟环境有助于我们将特定应用程序的依赖关系与系统的整体依赖关系隔离开来。这很好，因为我们不会陷入现有库的依赖性问题，并且在单独的系统上安装或卸载它们更容易，比如 docker 容器或服务器。对于这个任务，我们将使用 **pipenv。**

例如，要在 mac os 系统上安装它，请运行:

```
brew install pipenv
```

为了声明应用程序的依赖项(库),我们需要在项目的路径路径中创建一个 **Pipfile** :

```
[[source]]
url = '[https://pypi.python.org/simple'](https://pypi.python.org/simple')
verify_ssl = true
name = 'pypi'[requires]
python_version = "3.6"[packages]
flake8 = "*"
pytest-spark = ">=0.4.4"
pyspark = ">=2.4.0"
pytest-cov = "*"
```

这里有三个组成部分。在 **[[source]]** 标签中，我们声明了下载所有包的 **url** ，在**【requires】**中，我们定义了 python 版本，最后在**【packages】**中定义了我们需要的依赖项。我们可以将一个依赖项绑定到某个版本，或者使用**" ***符号获取最新的版本。

要创建虚拟环境并激活它，我们需要在终端中运行两个命令:

```
pipenv --three install
pipenv shell
```

一旦这样做了一次，您应该看到您在一个新的 venv 中，项目的名称出现在命令行的终端中(默认情况下，env 使用项目的名称):

```
(pyspark-project-template) host:project$ 
```

现在，您可以使用两个命令移入和移出。

停用环境并移回标准环境:

```
deactivate
```

再次激活虚拟环境(您需要在项目的根目录下):

```
source `pipenv --venv`/bin/activate
```

# 步骤 2:项目结构

项目可以有以下结构:

```
pyspark-project-template
    src/
        jobs/   
            pi/
                __init__.py
                resources/
                    args.json
            word_count/
                __init__.py
                resources/
                    args.json
                    word_count.csv
        main.py
    test/
        jobs/
            pi/
                test_pi.py
            word_count/
                test_word_count.py
```

某 __init__。为了使事情更简单，py 文件被排除在外，但是您可以在 github 上找到教程末尾的完整项目的链接。

我们基本上有源代码和测试。每个作业都放在一个文件夹中，每个作业都有一个资源文件夹，我们可以在其中添加该作业所需的额外文件和配置。

在本教程中，我使用了两个经典的例子——**pi**，生成圆周率数字直到小数位数，以及**字数统计**，计算 csv 文件中的字数。

# 步骤 3:使用 spark-submit 运行作业

让我们先看看 **main.py** 文件是什么样子的:

当我们运行我们的作业时，我们需要两个命令行参数: **—作业**，是我们想要运行的作业的名称(在 out case pi 或 word_count 中)和 **— res-path** ，是作业的相对路径。我们需要第二个参数，因为 spark 需要知道资源的完整路径。在生产环境中，当我们在集群上部署代码时，我们会将资源转移到 HDFS 或 S3，我们会使用那条路径。

在进一步解释代码之前，我们需要提到，我们必须压缩**作业**文件夹，并将其传递给 **spark-submit** 语句**。**假设我们在项目的根中:

```
cd src/ 
zip -r ../jobs.zip jobs/
```

这将使代码在我们的应用程序中作为一个模块可用。基本上在第 16 行的 **main.py 中，**我们正在编程导入 **job** 模块。

我们的两个作业， **pi** 和 **word_count，**都有一个 **run** 函数，所以我们只需要运行这个函数，来启动作业(main.py 中的 **line 17)。我们也在那里传递作业的配置。**

让我们看一下我们的 **word_count** 工作来进一步理解这个例子:

这个代码在 **__init__ 中定义。py** 文件在 **word_count** 文件夹中。我们可以看到，我们使用两个配置参数来读取 csv 文件:相对路径和 csv 文件在 resources 文件夹中的位置。剩下的代码只是计算字数，所以我们在这里不再赘述。值得一提的是，每个作业在 resources 文件夹中都有一个 **args.json** 文件。这里我们实际上定义了传递给作业的配置。这是**字数**作业的配置文件:

```
{
  "words_file_path": "/word_count/resources/word_count.csv"
}
```

现在我们已经有了运行我们的 **spark-submit** 命令的所有细节:

```
spark-submit --py-files jobs.zip src/main.py --job word_count --res-path /your/path/pyspark-project-template/src/jobs
```

要运行另一个作业， **pi，**，我们只需更改 **—作业**标志的参数。

# 步骤 4:编写单元测试，并在覆盖范围内运行它们

为了编写 pyspark 应用程序的测试，我们使用了 **pytest-spark** ，一个非常容易使用的模块。

**word_count** 作业单元测试:

我们需要从 **src** 模块中导入我们想要测试的函数。这里更有趣的部分是我们如何进行 **test_word_count_run。**我们可以看到没有初始化 spark 会话，我们只是在测试中将它作为参数接收。这要感谢 **pytest-spark** 模块，所以我们可以专注于编写测试，而不是编写样板代码。

接下来让我们讨论一下代码覆盖率。我们如何知道我们是否写了足够多的单元测试？很简单，我们运行一个测试覆盖工具，它会告诉我们哪些代码还没有被测试。对于 python，我们可以使用 **pytest-cov** 模块。要使用代码覆盖运行所有测试，我们必须运行:

```
pytest --cov=src test/jobs/
```

where **— cov** 标志告诉 pytest 在哪里检查覆盖率。

测试覆盖率结果:

```
---------- coverage: platform darwin, python 3.7.2-final-0 -----------
Name                              Stmts   Miss  Cover
-----------------------------------------------------
src/__init__.py                       0      0   100%
src/jobs/__init__.py                  0      0   100%
src/jobs/pi/__init__.py              11      0   100%
src/jobs/word_count/__init__.py       9      0   100%
-----------------------------------------------------
TOTAL                                20      0   100%
```

我们的测试覆盖率是 100%，但是等一下，少了一个文件！为什么 **main.py** 没有列在那里？

如果我们认为我们有不需要测试的 python 代码，我们可以将其从报告中排除。为此，我们需要创建一个**。coveragerc** 文件放在我们项目的根目录下。对于这个例子，它看起来像这样:

```
[run]
omit = src/main.py
```

# 步骤 5:运行静态代码分析

太好了，我们有一些代码，我们可以运行它，我们有覆盖率良好的单元测试。我们做得对吗？还没有！我们还需要确保按照 python 的最佳实践编写易读的代码。为此，我们必须用名为 **flake8 的 python 模块来检查我们的代码。**

要运行它:

```
flake8 ./src
```

它将分析 **src** 文件夹。如果我们有干净的代码，我们应该不会得到警告。但是没有，我们有几个问题:

```
flake8 ./src
./src/jobs/pi/__init__.py:13:1: E302 expected 2 blank lines, found 1
./src/jobs/pi/__init__.py:15:73: E231 missing whitespace after ','
./src/jobs/pi/__init__.py:15:80: E501 line too long (113 > 79 characters)
```

让我们看看代码:

我们可以看到在第 **13 行有一个 **E302** 警告。这意味着我们需要在这两个方法之间多加一行。然后在 **15 线上安装 **E231** 和 **E501** 。**这一行的第一个警告，告诉我们在`**range(1, number_of_steps +1),**` 和`**config[**` 之间需要一个额外的空格，第二个警告通知我们这一行太长了，很难读懂(我们甚至不能完整的看到大意！).**

在我们解决了所有的警告之后，代码看起来更容易阅读了:

# 步骤 6:用一个 Makefile 把它们放在一起

因为我们已经在终端中运行了许多命令，所以在最后一步中，我们将研究如何简化和自动化这项任务。

我们可以在项目的根目录下创建一个 **Makefile** ，如下图所示:

```
.DEFAULT_GOAL := runinit:
 pipenv --three install
 pipenv shellanalyze:
 flake8 ./srcrun_tests:
 pytest --cov=src test/jobs/run:
 find . -name '__pycache__' | xargs rm -rf
 rm -f jobs.zip cd src/ && zip -r ../jobs.zip jobs/ spark-submit --py-files jobs.zip src/main.py --job $(JOB_NAME) --res-path $(CONF_PATH)
```

如果我们想要运行覆盖率测试，我们可以简单地输入:

```
make run_tests
```

如果我们想要运行 **pi** 作业:

```
make run JOB_NAME=pi CONF_PATH=/your/path/pyspark-project-template/src/jobs
```

那都是乡亲们！我希望你觉得这是有用的。

一如既往，代码存储在 [github](https://github.com/BogdanCojocar/medium-articles/tree/master/pyspark-project-template) 上。