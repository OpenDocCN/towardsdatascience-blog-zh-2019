# 如何从 EC2 实例运行 Spark 应用程序

> 原文：<https://towardsdatascience.com/how-to-run-a-spark-application-from-an-ec2-instance-a4584d4d490d?source=collection_archive---------12----------------------->

为什么你会这样做，而不是使用电子病历？好吧，问得好。在某些情况下，使用 EC2 可能比使用 EMR 更便宜，但在其他情况下，EMR 可能是可取的。无论如何，下面是如何从 EC2 实例运行 Spark 应用程序:

![](img/4e17b890d11dfb66dba552e5028424d2.png)

Photo by [Steve Richey](https://unsplash.com/@steverichey?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/web?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

我使用了一个深度学习 AMI (Ubuntu 16.04)版本 25.3，带有一个 p3 实例，[用于加速计算。](https://aws.amazon.com/ec2/instance-types/#instance-type-matrix)

## SSH 到您的 EC2 实例。

```
ssh -i pem_key.pem ubuntu@public_dns_key
```

## 从一开始，你就安装了一些东西。

您在 EC2 终端中键入:

```
java -version
```

它返回:

```
openjdk version “1.8.0_222”OpenJDK Runtime Environment (build 1.8.0_222–8u222-b10–1ubuntu1~16.04.1-b10)OpenJDK 64-Bit Server VM (build 25.222-b10, mixed mode)
```

Java 8 是我们希望 Spark 运行的，所以这很好。

我的应用程序是使用 python 编写的，所以我想检查它是否已安装。

```
python --version
```

它返回:

```
Python 3.6.6 :: Anaconda, Inc.
```

太好了！

## 现在你需要安装 Hadoop。

我使用了以下准则:

[https://data wookie . netlify . com/blog/2017/07/installing-Hadoop-on-Ubuntu/](https://datawookie.netlify.com/blog/2017/07/installing-hadoop-on-ubuntu/)

1.  去 [Spark 下载网站](http://Spark.apache.org/downloads.html)看看它用的是哪个版本的 Hadoop:

![](img/7a41dd4b9f16ea7e63ceada1707da324.png)

It uses Hadoop 2.7, as of November 2019, this may be different for you.

2.它使用 Hadoop 2.7。好了，现在转到 [Hadoop 镜像站点](http://apache.mirrors.ionfish.org/hadoop/common/hadoop-2.7.7/)并使用 wget

*   右键单击，将链接复制到 Hadoop-2.7.7.tar.gz
*   输入你的 ubuntu 终端(粘贴你刚刚复制的，我加粗是为了让你知道你的可能不一样):

```
wget [**http://apache.mirrors.ionfish.org/hadoop/common/hadoop-2.7.7/hadoop-2.7.7.tar.gz**](http://apache.mirrors.ionfish.org/hadoop/common/hadoop-2.7.7/hadoop-2.7.7.tar.gz)
```

3.打开压缩的内容

```
tar -xvf hadoop-2.7.7.tar.gz
```

4.找到 java 的位置

```
type -p javac|xargs readlink -f|xargs dirname|xargs dirname
```

它返回:

```
/usr/lib/jvm/java-8-openjdk-amd64
```

好的，复制你的输出^

5.好了，现在编辑 Hadoop 配置文件，这样它就可以与 java 交互了

```
vi hadoop-2.7.7/etc/hadoop/hadoop-env.sh
```

键入`i`通过粘贴您复制的输出来插入和更新 JAVA_HOME 变量

![](img/cb3e0755a0e50b711883b7e5a6d79df8.png)

要退出 vim，使用 ESC + :wq！(wq 代表 write 和 quit，解释点就是强制)

6.设置 HADOOP_HOME 和 JAVA_HOME 环境变量。[JAVA _ HOME 环境变量指向计算机上安装 JAVA 运行时环境(JRE)的目录。目的是指向 Java 安装的位置。](https://stackoverflow.com/questions/5102022/what-is-the-reason-for-the-existence-of-the-java-home-environment-variable)

您可以通过使用

```
export JAVA_HOME=/usr/lib/jvm/java-8-openjdk-amd64export HADOOP_HOME=/home/ubuntu/hadoop-2.7.7
```

并更新您的路径:

```
export PATH=$PATH:$HADOOP_HOME/bin/
```

但是如果关闭 EC2 实例，这可能无法保存。所以我使用 vim 并把这些导出添加到我的。bashrc 文件。

不及物动词 bashrc，I 表示插入，然后:wq 表示完成。

PS:这是你如何删除路径中的副本，你不需要这样做，它们不会伤害任何东西，但如果你想知道:

```
if [ -n "$PATH" ]; then
    old_PATH=$PATH:; PATH=
    while [ -n "$old_PATH" ]; do
        x=${old_PATH%%:*}       # the first remaining entry
        case $PATH: in
            *:"$x":*) ;;          # already there
            *) PATH=$PATH:$x;;    # not there yet
        esac
        old_PATH=${old_PATH#*:}
    done
    PATH=${PATH#:}
    unset old_PATH x
fi
```

来源:[https://UNIX . stack exchange . com/questions/40749/remove-duplicate-path-entries-with-awk-command](https://unix.stackexchange.com/questions/40749/remove-duplicate-path-entries-with-awk-command)

7.找到你的。bashrc 文件，在您的主目录中键入`source .bashrc`

8.现在要检查版本，您可以回到您的主目录并键入
hadoop 版本，它应该会告诉您

9.您可以从您的主目录中删除压缩的 tar 文件:`rm hadoop-2.7.7.tar.gz`

## 现在你需要安装 Spark。我遵循这些准则:

[https://data wookie . netlify . com/blog/2017/07/installing-spark-on-Ubuntu/](https://datawookie.netlify.com/blog/2017/07/installing-spark-on-ubuntu/)

1.转到[该站点](https://www.apache.org/dyn/closer.lua/spark/spark-2.4.4/spark-2.4.4-bin-hadoop2.7.tgz)并复制第一个镜像站点的链接地址

2.在你的 ubuntu 终端中输入 wget 并粘贴复制的链接

```
wget [http://mirrors.sonic.net/apache/spark/spark-2.4.4/spark-2.4.4-bin-hadoop2.7.tgz](http://mirrors.sonic.net/apache/spark/spark-2.4.4/spark-2.4.4-bin-hadoop2.7.tgz)
```

3.现在解压压缩的 tar 文件

```
tar -xvf spark-2.4.4-bin-hadoop2.7.tgz
```

4.给你的环境变量添加火花`export SPARK_HOME=/home/ubuntu/spark-2.4.4-bin-hadoop2.7`

5.安装 scala。为什么我们在这里用简单的方法？因为我们不需要查看 jar 文件或任何东西。

```
sudo apt install scala
```

## 现在你必须确保 Spark、Hadoop 和 Java 能够协同工作。在我们的情况下，这也包括能够从 S3 读和写。

1.配置 Spark
[https://stack overflow . com/questions/58415928/Spark-S3-error-Java-lang-classnotfoundexception-class-org-Apache-Hadoop-f](https://stackoverflow.com/questions/58415928/spark-s3-error-java-lang-classnotfoundexception-class-org-apache-hadoop-f)

a.导航到 EC2 终端中的文件夹`~/spark-2.4.4-bin-hadoop2.7/conf`

运行代码:

```
touch spark_defaults.confvi spark_defaults.conf
```

在这里，您需要添加下面几行:
如果 2FA 适用于您:确保您的访问密钥和秘密密钥是用于服务帐户的(不与用户名/密码相关联)

```
spark.hadoop.fs.s3a.access.key ***your_access_key***spark.hadoop.fs.s3a.secret.key ***your_secret_key***spark.hadoop.fs.s3a.impl org.apache.hadoop.fs.s3a.S3AFileSystemspark.driver.extraClassPath /home/ubuntu/spark-2.4.4-bin-hadoop2.7/jars/**hadoop-aws-2.7.3**.**jar**:/home/ubuntu/spark-2.4.4-bin-hadoop2.7/jars/**aws-java-sdk-1.7.4.jar**
```

b.如何确保这些 jar 文件是正确的呢？？我把它们加粗是因为它们对你来说可能不一样。所以要检查，去你的 Hadoop jars。

使用 cd 转到以下文件夹:`cd ~/hadoop-2.7.7/share/hadoop/tools/lib`并检查哪些罐子用于`aws-java-sdk`和`hadoop-aws`，确保这些。jar 文件与您刚刚放入`spark_defaults.conf`的内容相匹配。

c.将这些文件复制到 spark jars 文件夹:

为了让下面的代码工作，请放在`~/hadoop-2.7.7/share/hadoop/tools/lib`文件夹中

```
cp **hadoop-aws-2.7.3.jar** ~/spark-2.4.4-bin-hadoop2.7/jars/cp **aws-java-sdk-1.7.4.jar** ~/spark-2.4.4-bin-hadoop2.7/jars/
```

## 配置 Hadoop

a.让你的每个工作节点都可以访问 S3(因为我们的代码从 S3 读写)
[https://github.com/CoorpAcademy/docker-pyspark/issues/13](https://github.com/CoorpAcademy/docker-pyspark/issues/13)

编辑文件`hadoop-2.7.7/etc/hadoop/core-site.xml`以包含以下行:

```
<configuration>
<property>
<name>fs.s3.awsAccessKeyId</name>
<value>*****</value>
</property><property>
<name>fs.s3.awsSecretAccessKey</name>
<value>*****</value>
</property>
</configuration>
```

b.将下面的 jar 文件复制到`hadoop-2.7.7/share/hadoop/common/lib`目录

```
sudo cp hadoop-2.7.7/share/hadoop/tools/lib/aws-java-sdk-1.7.4.jar hadoop-2.7.7/share/hadoop/common/lib/sudo cp hadoop-2.7.7/share/hadoop/tools/lib/hadoop-aws-2.7.5.jar Hadoop-2.7.7/share/hadoop/common/lib/
```

## 好了，现在您已经准备好克隆您的存储库了

git 克隆路径/到/your/repo.git

确保将下面一行添加到您的。bashrc 文件:

```
export PYTHONPATH=$PYTHONPATH:/home/ubuntu/repo
```

PYTHONPATH 是一个环境变量，您可以设置它来添加额外的目录，python 将在这些目录中查找模块和包。对于大多数安装，您不应该设置这些变量，因为 Python 运行不需要它们。Python 知道在哪里可以找到它的标准库。

设置 PYTHONPATH 的唯一原因是维护您不想安装在全局默认位置(即 site-packages 目录)的自定义 Python 库的目录。

来源:[https://www . tutorialspoint . com/What-is-Python path-environment-variable-in-Python](https://www.tutorialspoint.com/What-is-PYTHONPATH-environment-variable-in-Python)

## 现在运行您的代码！

快乐火花！

## 有关在 EC2 中使用 Spark 的更多信息

查看 [Snipe.gg](https://medium.com/snipe-gg) 的博客:

[](https://medium.com/snipe-gg/running-apache-spark-on-aws-without-busting-the-bank-5566dad18ea3) [## 在 AWS 上运行 Apache Spark，而不会让银行破产

### 在 Snipe，我们处理大量的数据。考虑到有超过 1 亿的活跃联盟…

medium.com](https://medium.com/snipe-gg/running-apache-spark-on-aws-without-busting-the-bank-5566dad18ea3) 

在他们的案例中，他们发现

> [……]在检查了 [EMR 定价](https://aws.amazon.com/emr/pricing/)后，我们发现 EMR 在其使用的 EC2 实例的价格上增加了高达 **25%** 的定价开销

这个博客也很好地突出了工具和生产我刚刚概述的努力的考虑。