# 在生产中设置 MLflow

> 原文：<https://towardsdatascience.com/setup-mlflow-in-production-d72aecde7fef?source=collection_archive---------3----------------------->

## 使用 Postgres DB 存储元数据和 systemd 单元保持 MLflow 运行的分步指南。

![](img/b3b4593ca9a7ec19626adced49c04532.png)

Source: [https://mlflow.org](https://mlflow.org)

这是我的 MLflow 教程系列的第一篇文章:

1.  [在生产中设置 ml flow](https://medium.com/@gyani91/setup-mlflow-in-production-d72aecde7fef)(你在这里！)
2.  [MLflow:基本测井功能](https://medium.com/@gyani91/mlflow-basic-logging-functions-e16cdea047b)
3.  [张量流的 MLflow 测井](https://medium.com/@gyani91/mlflow-logging-for-tensorflow-37b6a6a53e3c)
4.  [MLflow 项目](https://medium.com/@gyani91/mlflow-projects-24c41b00854)
5.  [使用 Python API 为 MLflow 检索最佳模型](https://medium.com/@gyani91/retrieving-the-best-model-using-python-api-for-mlflow-7f76bf503692)
6.  [使用 MLflow 服务模型](https://medium.com/@gyani91/serving-a-model-using-mlflow-8ba5db0a26c0)

MLflow 是一个用于机器学习生命周期管理的开源平台。最近，我在产品中设置了 MLflow，使用 Postgres 数据库作为跟踪服务器，使用 SFTP 通过网络传输工件。我花了大约 2 周的时间才把所有的组件都弄好，但是这篇文章可以帮助你在 10 分钟内把 MLflow 安装到生产环境中。

# 要求

*   [蟒蛇](https://www.anaconda.com/distribution/)

# 跟踪服务器设置(远程服务器)

跟踪服务器存储您在 MLflow 用户界面中看到的元数据。首先，让我们创建一个新的 Conda 环境:

```
conda create -n mlflow_env
conda activate mlflow_env
```

安装 MLflow 和 PySFTP 库:

```
conda install python
pip install mlflow
pip install pysftp
```

我们的跟踪服务器使用 Postgres 数据库作为存储元数据的后端。所以让我们安装 PostgreSQL:

```
apt-get install postgresql postgresql-contrib postgresql-server-dev-all
```

接下来，我们将为跟踪服务器创建管理员用户和数据库

```
sudo -u postgres psql
```

在 psql 控制台中:

```
CREATE DATABASE mlflow_db;
CREATE USER mlflow_user WITH ENCRYPTED PASSWORD 'mlflow';
GRANT ALL PRIVILEGES ON DATABASE mlflow_db TO mlflow_user;
```

因为我们需要与 Python 中的 Postgres 进行交互，所以需要安装 psycopg2 库。但是，为了确保成功安装，我们需要在以下操作之前安装 GCC Linux 软件包:

```
sudo apt install gcc
pip install psycopg2-binary
```

如果您希望远程连接到 PostgreSQL 服务器，或者希望将其访问权授予用户。你可以

```
cd /var/lib/pgsql/data
```

然后在 **postgresql.conf** 文件的末尾添加下面一行。

```
listen_addresses = '*'
```

然后，您可以通过在 **pg_hba.conf** 文件的末尾添加以下行来指定您希望允许连接到 PostgreSQL 服务器的远程 IP

```
host    all    all    10.10.10.187/32    trust
```

其中 **10.10.10.187/32** 为远程 IP。要允许来自任何 IP 的连接，请使用 **0.0.0.0/0** 来代替。然后重新启动 PostgreSQL 服务器以应用更改。

```
service postgresql restart
```

下一步是为我们的跟踪服务器创建一个目录，以记录机器学习模型和其他工件。请记住，Postgres 数据库仅用于存储关于这些模型的元数据。这个目录叫做神器 URI。

```
mkdir ~/mlflow/mlruns
```

创建日志目录。

```
mkdir ~/mlflow/mllogs
```

您可以使用以下命令运行跟踪服务器。但是一旦你按 Ctrl-C 或者退出终端，服务器就会停止。

```
mlflow server --backend-store-uri postgresql://mlflow_user:mlflow@localhost/mlflow_db --default-artifact-root sftp://mlflow_user@**<hostname_of_server>**:~/mlflow/mlruns -h 0.0.0.0 -p 8000
```

如果您希望跟踪服务器在重新启动后启动并运行，并且对故障具有弹性，那么将它作为 systemd 服务运行是非常有用的。

您需要进入 **/etc/systemd/system** 目录，创建一个名为 **mlflow-tracking.service** 的新文件，内容如下:

```
[Unit]
Description=MLflow Tracking Server
After=network.target[Service]
Restart=on-failure
RestartSec=30
StandardOutput=file:/path_to_your_logging_folder/stdout.log
StandardError=file:/path_to_your_logging_folder/stderr.log
User=root
ExecStart=/bin/bash -c 'PATH=/path_to_your_conda_installation/envs/mlflow_env/bin/:$PATH exec mlflow server --backend-store-uri postgresql://mlflow_user:mlflow@localhost/mlflow_db --default-artifact-root sftp://mlflow_user@**<hostname_of_server>**:~/mlflow/mlruns -h 0.0.0.0 -p 8000'[Install]
WantedBy=multi-user.target
```

使用以下命令激活并启用上述服务:

```
sudo systemctl daemon-reload
sudo systemctl enable mlflow-tracking
sudo systemctl start mlflow-tracking
```

使用以下命令检查是否一切正常:

```
sudo systemctl status mlflow-tracking
```

您应该会看到类似如下的输出:

![](img/65d7f82b02378fa49eb8d848d6997633.png)

Systemd unit running

为名为 *mlflow_user* 的服务器创建用户，并将 mlflow 目录作为该用户的工作目录。然后在**中创建一个 ssh-key 对。*ml flow _ user*(**/ml flow/)的 ssh** 目录。在我们的例子中是 ssh** )。将公钥放在 **authorized_keys** 文件中，与用户共享私钥。

此外，为了让 MLflow UI 能够读取工件，将私钥复制到 **/root/。宋承宪/** 也是。

接下来，我们需要使用以下命令为服务器手动创建主机密钥:

```
cd /root/.ssh
ssh-keyscan -H **<hostname_of_server>** >> known_hosts
```

现在，您可以重新启动计算机，MLflow Tracking Server 将在重新启动后启动并运行。

# 在客户端机器上(本地)

为了开始跟踪生产跟踪服务器下的一切，有必要在您的**中设置以下环境变量。bashrc** 。

```
export MLFLOW_TRACKING_URI='http://**<hostname_of_server>**:8000'
```

不要忘记为您的**寻找来源。bashrc** 文件！

```
. ~/.bashrc
```

确保在您的环境中为 **mlflow** 和 **pysftp** 安装 pip 包(需要 **pysftp** 来促进工件到生产服务器的传输)。

```
pip install mlflow
pip install pysftp
```

为了能够认证 **pysftp** 传输，将生产服务器上生成的私钥放在**中。本地机器的 ssh** 目录。那就做吧

```
ssh **<hostname_of_server>**
```

当提示将**<hostname _ of _ server>**保存为已知主机时，回答 **yes** 。

您可以在[http://**<hostname _ of _ server>**:8000](http://nmkzh-testing-001.nomoko.lan:7000)访问 **MLflow UI**

![](img/b7a4009760978ad7e0b7259d3d07baa5.png)

The Mlflow UI

从互联网上运行一个样本机器学习模型，以检查 MLflow 是否可以跟踪运行。

```
mlflow run git@github.com:databricks/mlflow-example.git -P alpha=0.5
```

在下一篇文章中，我将讲述基本的 MLflow 日志记录功能

# 参考资料:

[1]物流，[安装物流](https://www.mlflow.org/docs/latest/quickstart.html#installing-mlflow) (2019)，物流文件