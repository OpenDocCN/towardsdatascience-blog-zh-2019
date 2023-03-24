# Jupyter 笔记本——忘记 CSV，用 Python 从 DB 获取数据

> 原文：<https://towardsdatascience.com/jupyter-notebook-forget-csv-fetch-data-from-db-with-python-5ef1a56402f2?source=collection_archive---------14----------------------->

## 从数据库加载训练数据的简单方法

![](img/51d9e9f76ee3278a69dc11a56a71c966.png)

Source: Pixabay

如果你阅读关于机器学习的书籍、文章或博客，它很可能会使用 CSV 文件中的训练数据。CSV 没什么不好，但是大家想想是不是真的实用。直接从数据库中读取数据不是更好吗？通常你不能直接将业务数据输入到 ML 训练中，它需要预处理——改变分类数据，计算新的数据特征，等等。在获取原始业务数据时，使用 SQL 可以非常容易地完成数据准备/转换步骤。直接从数据库中读取数据的另一个优点是，当数据发生变化时，更容易自动化 ML 模型重训练过程。

在这篇文章中，我描述了如何从 Jupyter 笔记本 Python 代码调用 Oracle DB。

**第一步**

安装 [cx_Oracle](https://oracle.github.io/python-cx_Oracle/) Python 模块:

*python -m pip 安装 cx_Oracle*

该模块有助于从 Python 连接到 Oracle DB。

**第二步**

cx_Oracle 支持从 Python 代码执行 SQL 调用。但是为了能够从 Python 脚本调用远程数据库，我们需要在运行 Python 的机器上安装和配置 [Oracle Instant Client](https://www.oracle.com/technetwork/database/database-technologies/instant-client/overview/index.html) 。

如果你用的是 Ubuntu，安装*外星人*:

*sudo apt-get 更新*
*sudo apt-get 安装外星人*

下载 Oracle 即时客户端的 RPM 文件并使用 *alien* 安装:

*alien-I Oracle-instant client 18.3-basiclite-18 . 3 . 0 . 0 . 0–1 . x86 _ 64 . rpm
alien-I Oracle-instant client 18.3-sqlplus-18 . 3 . 0 . 0 . 0–1 . x86 _ 64 . rpm
alien-I Oracle-instant client 18.3-0 . 0 . 0 . 0–1 . x86 _ 64 . rpm*

添加环境变量:

*导出 ORACLE _ HOME =/usr/lib/ORACLE/18.3/client 64
导出路径=$PATH:$ORACLE_HOME/bin*

在这里阅读更多。

**第三步**

安装神奇的 SQL Python 模块:

*pip 安装 jupyter-sql
pip 安装 ipython-sql*

安装和配置完成。

对于今天的样本，我使用的是[皮马印第安人糖尿病数据库](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.names)。CSV 数据可以从[这里](https://raw.githubusercontent.com/jbrownlee/Datasets/master/pima-indians-diabetes.data.csv)下载。我将 CSV 数据上传到数据库表中，并将通过 SQL 直接在 Jupyter notebook 中获取它。

首先，建立到数据库的连接，然后执行 SQL 查询。查询结果集存储在一个名为 *result* 的变量中。您看到%%sql 了吗——这个神奇的 sql:

Connecting to Oracle DB and fetching data through SQL query

建立连接时必须指定用户名和密码。为了避免共享密码，请确保从外部源读取密码值(可以是本例中的简单 JSON 文件，也可以是来自 keyring 的更高级的编码令牌)。

这种方法的美妙之处在于，通过 SQL 查询获取的数据是数据框中现成可用的。机器学习工程师可以像通过 CSV 加载数据一样处理数据:

在 [GitHub](https://github.com/abaranovskis-redsamurai/automation-repo/blob/master/diabetes_redsamurai_db.ipynb) 上可以找到 Jupyter 笔记本的样本。样本凭证 JSON [文件](https://github.com/abaranovskis-redsamurai/automation-repo/blob/master/credentials.json)。