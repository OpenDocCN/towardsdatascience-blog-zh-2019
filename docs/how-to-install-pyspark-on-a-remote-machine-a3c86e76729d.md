# 如何在远程机器上安装 PySpark

> 原文：<https://towardsdatascience.com/how-to-install-pyspark-on-a-remote-machine-a3c86e76729d?source=collection_archive---------18----------------------->

简单的方法。

![](img/93f6c4964725301cce2e0e522a059eef.png)

Spark, [Wikipedia](https://en.wikipedia.org/wiki/Apache_Spark#/media/File:Apache_Spark_Logo.svg).

假设您想在使用 python 的机器上运行 Spark。要做到这一点，您需要在机器上运行 PySpark，并在 Jupyter 或 python 中使用它，这需要在您的 shell 中进行一些安装和调试。我使用下面的过程解决了这个问题，成功地安装了 PySpark 并在 Jupyter 实验室中运行了演示代码。此程序旨在使用股票张量流 1.14 引导映像在基于 Gloud 的机器上执行，但是，这些指令也应该在本地机器上工作。

## 安装 Java 1.8

1.  网上说很多版本都有兼容性问题，所以请安装 java 1.8:

> sudo apt 更新

2.检查您的 java 版本

> java 版本

3.如果没有安装 Java，请执行以下操作

> apt 安装 openjdk-9-jre-headless

4.验证 java 的版本

> java 版本

5.您应该会得到与此类似的结果:

> openjdk 版本" 1.8.0_222"
> OpenJDK 运行时环境(build 1 . 8 . 0 _ 222–8u 222-b10–1 ~ deb9u 1-b10)
> open JDK 64 位服务器 VM (build 25.222-b10，混合模式)

6.(可选)如果显示另一个版本的 java，请执行以下操作并选择指向 java-8 的版本。

> sudo 更新-备选方案-配置 java

输出:

```
There are 2 choices for the alternative java (providing /usr/lib/java).Selection    Path                                            Priority   Status
------------------------------------------------------------
* 0            /usr/lib/java/java-11-openjdk-amd64/bin/java      
  1            /usr/lib/java/java-11-openjdk-amd64/bin/java      
  2            /usr/lib/java/java-8-openjdk-amd64/jre/bin/java Press <enter> to keep the current choice[*], or type selection number:
```

7.找到 java.home 在哪里

> Java-XshowSettings:properties-version 2 > & 1 >/dev/null | grep ' Java . home '

8.(或者)您可以设置 JAVA_HOME

> 导出 JAVA _ HOME = " $(/myJavaHome/JAVA _ HOME-v 1.8)"
> 或者创建并添加到~/。bash _ profile

然后

> 来源。bashrc

此时，您应该已经有了 java_home 目录，您可以开始安装 PySpark，过程是类似的，因此，我们还需要找到 Spark 的安装位置。

## 安装 PySpark

1.  pip 安装以下项目:

> pip3 安装 findspark
> pip3 安装 pyspark

2.找到 pyspark 在哪里

> pip3 显示 pyspark

输出:

> 名称:pyspark
> 版本:2.3.1
> 摘要:Apache spark Python API
> 主页:[https://github.com/apache/spark/tree/master/python](https://github.com/apache/spark/tree/master/python)
> 作者:Spark 开发者
> 作者-邮箱:dev@spark.apache.org
> 许可:[http://www.apache.org/licenses/LICENSE-2.0](http://www.apache.org/licenses/LICENSE-2.0)
> **位置:/usr/local/lib/Python 3.5/site-packages** 需求:py4j

至此，我们已经拥有了所需的一切，只需在下面的代码中替换主目录指针并运行演示程序。

## 初始化:

下面是在您的代码中配置一切，使用之前为 PySpark & Java 找到的主目录信息。

> 导入 findspark
> 导入 OS
> spark _ HOME = ' mySparkLocation '
> JAVA _ HOME = ' myJavaHome '
> OS . environ[' JAVA _ HOME ']= JAVA _ HOME find spark . init(spark _ HOME = spark _ dir)

## 演示:

如果一切配置正确，这个演示将工作，祝你好运:)

> 从 pyspark.sql 导入 pyspark
> 导入 spark session
> spark = spark session . builder . getor create()
> df = spark . SQL(" select ' spark ' as hello ")
> df . show()
> spark . stop()

## 参考

1.  [类似中帖](https://medium.com/@ashok.tankala/run-your-first-spark-program-using-pyspark-and-jupyter-notebook-3b1281765169)
2.  [Ubuntu 上的 Java](https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-on-ubuntu-18-04)
3.  [finspark](https://bigdata-madesimple.com/guide-to-install-spark-and-use-pyspark-from-jupyter-in-windows/)
4.  [查找 java_home](https://www.baeldung.com/find-java-home)
5.  [寻找火花位置](https://stackoverflow.com/questions/53583199/pyspark-error-unsupported-class-file-major-version-55)
6.  [pyspark 演示](https://bigdata-madesimple.com/guide-to-install-spark-and-use-pyspark-from-jupyter-in-windows/)
7.  [选择备用 java](https://www.digitalocean.com/community/tutorials/how-to-install-java-with-apt-on-ubuntu-18-04)

Ori Cohen 博士拥有计算机科学博士学位，主要研究机器学习。他是 TLV 新遗迹公司的首席数据科学家，从事 AIOps 领域的机器和深度学习研究。