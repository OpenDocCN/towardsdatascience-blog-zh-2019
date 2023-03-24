# HiveQL/SQL 脚本完成的自动电子邮件通知

> 原文：<https://towardsdatascience.com/execute-hiveql-script-with-email-notifications-86bc0967a454?source=collection_archive---------15----------------------->

## 自动将产量提高 2-3 倍

![](img/7cfdbce20ebfab8882de43d3a6ceb537.png)

Stock image from RawPixel.com.

> 如果你觉得这篇文章有任何帮助，请评论或点击左边的掌声按钮给我免费的虚拟鼓励！

# 动机

这个脚本对于执行 HiveQL 文件`(.hql)`或者 SQL 文件`(.sql)`非常有用。例如，这个脚本提供了电子邮件通知以及所涉及的 Hive/SQL 操作所需的时钟时间记录，而不是必须定期检查`CREATE TABLE`或`JOIN`何时结束。这对于连夜完成一系列 Hive/SQL 连接以便第二天进行分析特别有用。

# 剧本

答作为一名实践中的数据科学家，当我只是为了自己的目的而寻找代码进行回收和修改时，我不喜欢过度解释的琐碎片段，所以请找到下面的脚本。如果除了代码注释之外，你还想了解更多关于脚本的解释，请参见最后的附录。对`bash`的充分解释、它在位级的转换、HiveQL/SQL 和数据库引擎超出了本文的范围。正如所写的那样，下面的脚本仅在复制粘贴时适用于 HiveQL。您必须对它做一些小的修改，以便与您的 SQL 发行版一起工作。例如，如果使用 MySQL，将`hive -f`改为`mysql source`。也可以让脚本向多个电子邮件地址发送电子邮件。

```
#!/bin/bash
## By Andrew Young 
## Contact: andrew.wong@team.neustar
## Last modified date: 13 Dec 2019################
## DIRECTIONS ##
################
## Run the following <commands> without the "<" or ">".
## <chmod u+x scriptname> to make the script executable. (where scriptname is the name of your script)
## To run this script, use the following command:
## ./<yourfilename>.sh#################################
## Ask user for Hive file name ##
#################################
echo Hello. Please enter the HiveQL filename with the file extension. For example, 'script.hql'. To cancel, press 'Control+C'. For your convenience, here is a list of files in the present directory\:
ls -l
read -p 'Enter the HiveQL filename: ' HIVE_FILE_NAME
#read HIVE_FILE_NAME
echo You specified: $HIVE_FILE_NAME
echo Executing...######################
## Define variables ##
######################
start="$(date)"
starttime=$(date +%s)#####################
## Run Hive script ##
#####################
hive -f "$HIVE_FILE_NAME"#################################
## Human readable elapsed time ##
#################################
secs_to_human() {
    if [[ -z ${1} || ${1} -lt 60 ]] ;then
        min=0 ; secs="${1}"
    else
        time_mins=$(echo "scale=2; ${1}/60" | bc)
        min=$(echo ${time_mins} | cut -d'.' -f1)
        secs="0.$(echo ${time_mins} | cut -d'.' -f2)"
        secs=$(echo ${secs}*60|bc|awk '{print int($1+0.5)}')
    fi
    timeelapsed="Clock Time Elapsed : ${min} minutes and ${secs} seconds."
}################
## Send email ##
################
end="$(date)"
endtime=$(date +%s)
secs_to_human $(($(date +%s) - ${starttime}))subject="$HIVE_FILE_NAME started @ $start || finished @ $end"
## message="Note that $start and $end use server time. \n $timeelapsed"## working version:
mail -s "$subject" youremail@wherever.com <<< "$(printf "
Server start time: $start \n
Server   end time: $end   \n
$timeelapsed")"#################
## FUTURE WORK ##
#################
# 1\. Add a diagnostic report. The report will calculate:
#    a. number of rows,
#    b. number of distinct observations for each column,
#    c. an option for a cutoff threshold for observations that ensures the diagnostic report is only produced for tables of a certain size. this can help prevent computationally expensive diagnostics for a large table
#
# 2\. Option to save output to a text file for inclusion in email notification.
#################
```

# 脚本解释

在此，我将解释脚本每一部分的代码。

```
################
## DIRECTIONS ##
################
## Run the following <commands> without the "<" or ">".
## <chmod u+x scriptname> to make the script executable. (where scriptname is the name of your script)
## To run this script, use the following command:
## ./<yourfilename>.sh
```

`chmod`是代表“改变模式”的`bash`命令我用它来改变文件的权限。它确保您(所有者)被允许在您正在使用的节点/机器上执行您自己的文件。

```
#################################
## Ask user for Hive file name ##
#################################
echo Hello. Please enter the HiveQL filename with the file extension. For example, 'script.hql'. To cancel, press 'Control+C'. For your convenience, here is a list of files in the present directory\:
ls -l
read -p 'Enter the HiveQL filename: ' HIVE_FILE_NAME
#read HIVE_FILE_NAME
echo You specified: $HIVE_FILE_NAME
echo Executing...
```

在本节中，我使用命令`ls -l`列出与保存我的脚本的`.sh`文件相同的目录(即“文件夹”)中的所有文件。`read -p`用于保存用户输入，即您想要运行的`.hql`或`.sql`文件的名称，保存到一个变量中，稍后在脚本中使用。

```
######################
## Define variables ##
######################
start="$(date)"
starttime=$(date +%s)
```

这是记录系统时钟时间的地方。我们将两个版本保存到两个不同的变量:`start`和`starttime`。

`start="$(date)"`保存系统时间和日期。
`starttime=$(date +%s)`保存系统时间和日期，以数字秒为单位，从 1900 开始偏移。第一个版本稍后在电子邮件中使用，以显示 HiveQL/SQL 脚本开始执行的时间戳和日期。第二个用于计算经过的秒数和分钟数。这为在您提供的 HiveQL/SQL 脚本中构建表所花费的时间提供了一个方便的记录。

```
#################################
## Human readable elapsed time ##
#################################
secs_to_human() {
    if [[ -z ${1} || ${1} -lt 60 ]] ;then
        min=0 ; secs="${1}"
    else
        time_mins=$(echo "scale=2; ${1}/60" | bc)
        min=$(echo ${time_mins} | cut -d'.' -f1)
        secs="0.$(echo ${time_mins} | cut -d'.' -f2)"
        secs=$(echo ${secs}*60|bc|awk '{print int($1+0.5)}')
    fi
    timeelapsed="Clock Time Elapsed : ${min} minutes and ${secs} seconds."
}
```

这是一个复杂的解码部分。基本上，有很多代码做一些非常简单的事情:找出开始和结束时间之间的差异，都是以秒为单位，然后将经过的时间以秒为单位转换为分和秒。例如，如果工作需要 605 秒，我们将它转换为 10 分 5 秒，并保存到一个名为`timeelapsed`的变量中，以便在我们发送给自己和/或各种利益相关者的电子邮件中使用。

```
################
## Send email ##
################
end="$(date)"
endtime=$(date +%s)
secs_to_human $(($(date +%s) - ${starttime}))subject="$HIVE_FILE_NAME started @ $start || finished @ $end"
## message="Note that $start and $end use server time. \n $timeelapsed"## working version:
mail -s "$subject" youremail@wherever.com <<< "$(printf "
Server start time: $start \n
Server   end time: $end   \n
$timeelapsed")"
```

在本节中，我再次以两种不同的格式将系统时间记录为两个不同的变量。我实际上只用了其中一个变量。我决定保留未使用的变量，`endtime`,以备将来对这个脚本的扩展。电子邮件通知中使用变量`$end`来报告 HiveQL/SQL 文件完成时的系统时间和日期。

我定义了一个变量`subject`，不出所料，它成为了电子邮件的主题行。我注释掉了`message`变量，因为我无法使用`printf`命令在邮件正文中正确地用新行替换`\n`。我把它留在原处，因为我想让您可以根据自己的目的编辑它。

`mail`是一个发送邮件的程序。你可能会有它或类似的东西。`mail`还有其他选项，如`mailx`、`sendmail`、`smtp-cli`、`ssmtp`、S [waks](https://www.jetmore.org/john/code/swaks/) 等多个选项。这些程序中的一些是彼此的子集或衍生物。

# 未来的工作

本着合作和进一步努力的精神，以下是对未来工作的一些想法:

1.  允许用户指定 HiveQL/SQL 文件的目录，在其中查找要运行的目标文件。一个使用案例是，用户可能根据项目/时间将他们的 HiveQL/SQL 代码组织到多个目录中。
2.  改进电子邮件通知的格式。
3.  通过自动检测是否输入了 HiveQL 或 SQL 脚本，以及如果是后者，是否有指定分布的选项(即 OracleDB、MySQL 等),增加脚本的健壮性。).
4.  为用户添加选项，以指定是否应将查询的输出发送到指定的电子邮件。用例:为探索性数据分析(EDA)运行大量 SELECT 语句。例如，计算表中每个字段的不同值的数量，查找分类字段的不同级别，按组查找计数，数字汇总统计，如平均值、中值、标准偏差等。
5.  添加选项，让用户在命令行指定一个或多个电子邮件地址。
6.  添加一个选项，让用户指定多个`.hql`和/或`.sql`文件按顺序或并行运行。
7.  添加并行化选项。用例:你的截止日期很紧，不需要关心资源利用或同事礼仪。我需要我的结果！
8.  添加计划代码执行的选项。可能以用户指定的`sleep`的形式。用例:您希望在生产代码运行期间以及在工作日集群使用率较高时保持礼貌并避免运行。
9.  通过标志使上述一个或多个[选项可用。这将是非常酷的，但也有很多工作只是为了添加顶部的粉末。](https://stackoverflow.com/questions/7069682/how-to-get-arguments-with-flags-in-bash)
10.  更多。

# 新手教程

1.  打开您喜欢的任何命令行终端界面。例子包括 MacOS 上的**终端**和 Windows 上的 **Cygwin** 。MacOS 已经安装了**终端**。在 Windows 上，你可以在这里 下载 **Cygwin** [*。*](https://cygwin.com/install.html)
2.  导航到包含您的`.hql`文件的目录。例如，输入`cd anyoung/hqlscripts/`来改变你当前的工作目录到那个文件夹。这在概念上相当于双击一个文件夹来打开它，只是这里我们使用的是基于文本的命令。
3.  `nano <whateverfilename>.sh`
    `nano`是一个命令，它打开一个同名的程序，并要求它创建一个名为`<whateverfilename>.sh`
    的新文件。要修改这个文件，使用相同的命令。
4.  复制我的脚本并粘贴到这个新的`.sh`文件中。将我的脚本中的收件人电子邮件地址从`youremail@wherever.com`更改为您的。
5.  `bash <whateverfilename>.sh`
    Bash 使用扩展名为`.sh`的文件。这个命令要求程序`bash`运行扩展名为`.sh`的“shell”文件。运行时，该脚本将输出与 shell 脚本位于同一目录中的所有文件的列表。这个列表输出是我为了方便而编写的。
6.  漫威在你的生产力增益！

# 关于作者

安德鲁·杨是 Neustar 的 R&D 数据科学家经理。例如，Neustar 是一家信息服务公司，从航空、银行、政府、营销、社交媒体和电信等领域的数百家公司获取结构化和非结构化的文本和图片数据。Neustar 将这些数据成分结合起来，然后向企业客户出售具有附加值的成品，用于咨询、网络安全、欺诈检测和营销等目的。在这种情况下，Young 先生是 R&D 一个小型数据科学团队的实践型首席架构师，该团队构建为所有产品和服务提供信息的系统，为 Neustar 带来超过 10 亿美元的年收入。

# 附录

## 创建 HiveQL 脚本

运行上述 shell 脚本的先决条件是要有一个 HiveQL 或 SQL 脚本。下面是一个 HiveQL 脚本示例，`example.hql`:

```
CREATE TABLE db.tb1 STORED AS ORC AS
SELECT *
FROM db.tb2
WHERE a >= 5;CREATE TABLE db.tb3 STORED AS ORC AS
SELECT *
FROM db.tb4 a
INNER JOIN db.tb5 b
ON a.col2 = b.col5
WHERE dt >= 20191201 AND DOB != 01-01-1900;
```

注意，在一个 HiveQL/SQL 脚本中可以有一个或多个`CREATE TABLE`命令。它们将按顺序执行。如果您可以访问一个节点集群，您还可以从查询的并行化中获益。我可能会在另一篇文章中解释。