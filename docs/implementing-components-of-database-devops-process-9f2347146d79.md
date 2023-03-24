# 实施数据库开发运维流程的组件

> 原文：<https://towardsdatascience.com/implementing-components-of-database-devops-process-9f2347146d79?source=collection_archive---------30----------------------->

## 让我们以潜在雇员数据库为例，看看如何实现数据库开发运维流程的一些组件。

![](img/63e6634748bd5d50382e869e97cc71cd.png)

近年来，数据库 DevOps 已经成为需求最大的流程之一。没有它，即使是小公司也无法真正有效地工作，大多数 it 公司已经实施了 DevOps 数据库自动化，并将其作为工作流程的重要部分。

# **那么，数据库 DevOps 到底是什么？**

数据库开发运维流程的主要作用是不断地将开发和测试环境中的变更交付到生产环境中，监控生产环境的状态，并在需要时提供回滚变更的能力。在此之后，更改被传输到生产环境管理服务。基本上，DevOps 过程将开发和测试与各种类型的管理(系统、数据库等)联系起来。)等生产环境和系统管理流程。DevOps 数据库自动化特别包括所有环境中的自动化，并使数据库的开发和管理任务更简单、更安全。

通常，数据库 DevOps 过程分为*两部分*:

**1。生产前** —在将变更实际提交到生产环境之前发生的所有事情。这包括以下阶段:

1.1 编译和记录代码

1.2 用测试数据填充数据库并运行自动测试

1.3 将变更从开发环境应用到测试环境

1.4 用测试数据填充数据库

1.5 决定变更是否可以部署到生产环境中

流程的第一部分主要由开发人员和测试人员完成，第 1.3 阶段由 DevOps 工程师实施。阶段 1.2 到 1.4 可以根据任务规格以及 DevOps 流程本身的建立方式以任何顺序执行。这个想法是成功地通过从编译代码到决定是否准备好部署到生产的所有步骤。

在阶段 1.3 之后，对解决方案执行各种测试(压力、功能、集成测试等)。如果检测到与工作标准的任何重大偏差(例如，如果发现错误或性能问题)，通常在阶段 1.5 确定代码还没有准备好部署到生产中，因此应该送回开发。此后，该过程从阶段 1.1 再次开始。事情与阶段 1.2 相似。

第 1.5 阶段也可能由于与第 1.2 或 1.3 阶段无关的原因而失败。只有在阶段 1.5 通过后，DevOps 流程才会进入第二个(也是最重要的)部分:

**2。实施** —将解决方案投入生产。这包括以下阶段:

2.1 部署

2.2 初级监控

2.3 回滚部署或将更改转移到生产环境管理服务。

阶段 2.1 通常由开发运维专家和生产环境管理服务专家共同手动启动，以避免生产中出现问题。最好提到阶段 2.1 包括阶段 1.3——唯一的区别是变更被部署到生产中。如果部署失败(阶段 2.1)，则问题被定位，然后重复阶段 2.1 或者开发运维流程返回到第 1 部分的开头。

另一方面，如果部署成功但主要监控失败，通常决定部署应该回滚(阶段 2.3)，开发运维流程返回到其第一部分。

只有当根据所有必要的标准进行评估时，阶段 2.1 和 2.2 完全成功，并且没有检测到任何问题，才会将更改提交给生产环境管理服务(阶段 2.3)，以便由管理部门进行持续维护。这是 DevOps 流程迭代的终点。

重要的是要注意，这个过程是连续发生的，一次又一次，因此最大限度地减少了将代码从开发环境交付到测试环境的时间，反之亦然，或者在开发或测试环境之间交付。DevOps 流程还缩短了交付到生产环境的时间，并且能够监控和回滚更改。

# **创建 PowerShell 脚本来组织数据库开发操作流程**

让我们看看如何实现数据库开发运维流程的一些组件。在这个例子中，我们将使用潜在雇员的数据库。

我们将研究以下几个方面，以及如何为它们生成 PowerShell 脚本来组织数据库开发运维流程。[dbForge devo PS Automation for SQL Server](https://www.devart.com/dbforge/sql/database-devops/)和各种 db forge 工具将在这方面帮助我们:

## 1.生产前:

1.1 代码编译和文档( [SQL 完成](https://www.devart.com/dbforge/sql/database-devops/sqlcomplete.html)、[源代码控制](https://www.devart.com/dbforge/sql/database-devops/source-control.html)、[文档](https://www.devart.com/dbforge/sql/documenter/))

1.2 用测试数据填充数据库并运行自动测试([数据生成器](https://www.devart.com/dbforge/sql/database-devops/test-data-management.html)、[单元测试](https://www.devart.com/dbforge/sql/database-devops/unit-testing.html))

1.3 将解决方案从开发环境部署到测试环境([模式比较](https://www.devart.com/dbforge/sql/database-devops/database-schema-changes.html)、[数据比较](https://www.devart.com/en/dbforge/sql/datacompare/))

1.4 用测试数据填充数据库

1.5 决定代码是否准备好部署到生产中(这是一项管理职责，所以我们在这里不讨论它)

## 2.实施:

2.1 部署(数据库模式比较、数据比较、 [dbForge 数据泵](https://www.devart.com/dbforge/sql/database-devops/continuous-intergation-export-import-data.html))

2.2 主监控( [dbForge 事件探查器](https://www.devart.com/dbforge/sql/event-profiler/)、 [dbForge 监控器](https://www.devart.com/dbforge/sql/studio/monitor.html)、dbForge 数据泵)

2.3 回滚部署或将更改提交给生产环境管理服务(这可以通过多种方式完成，因此我们也将省略其讨论，但事务回滚可以使用 dbForge 事务日志来执行)

在本文中，我们不会深入研究 1.1 阶段和第二部分。然而，应该提到 [SQL Complete](https://www.devart.com/dbforge/sql/sqlcomplete/) 的一个重要特性。具体来说，要使用脚本格式化文件夹，我们可以使用以下 PowerShell 脚本:

```
$result = Invoke-DevartFormatScript -Source $scriptFolder
```

这里，$scriptFolder 是包含需要格式化的脚本的必要文件夹的完整路径。此外，Invoke-DevartFormatScript cmdlet 用于调用 SQL Complete 工具的格式化功能。正如我们从脚本中看到的，cmdlet 只有一个参数—包含数据库模式创建或更新脚本的文件夹的路径。

现在，我们将详细了解阶段 1.2–1.4。由于在数据库开发运维过程中，这些阶段可以按任何顺序执行，我们将按如下方式安排它们:

1.  用测试数据填充数据库(dbForge 数据生成器)
2.  运行自动测试(dbForge 单元测试)
3.  将解决方案从开发环境部署到测试环境(数据库模式比较、数据比较)

我们需要保存一个数据生成项目，或者创建一个包含必要参数的新项目，并将其保存为 JobEmplDB.dgen。

现在，我们可以使用 Invoke-DevartDatabaseTests cmdlet 根据 JobEmplDB.dgen 项目调用数据生成过程:

```
Invoke-DevartDatabaseTests -InputObject $ConnectionString -DataGeneratorProject $JobEmplDBFolder -InstalltSQLtFramework -RewriteReport -UnInstalltSQLtFramework -IncludeTestData
```

在这里，将$ConnectionString 替换为连接字符串，将$JobEmplDBFolder 替换为 JobEmplDB.dgen 文件的全名，即该文件的完整路径，包括实际的文件名和扩展名。

现在，让我们看看如何使用 PowerShell 脚本来运行自动测试。

为此，我们将使用 Invoke-DevartExecuteScript 和 Invoke-DevartDatabaseTests cmdlet:

```
Invoke-DevartExecuteScript -Input $UnitTestsFolder -Connection $ConnectionStringInvoke-DevartDatabaseTests -InputObject $ConnectionString -InstalltSQLtFramework -OutReportFileName $TestReportOutputFileName -ReportFormat JUnit
```

**第一个 cmdlet 从文件夹中为数据库创建测试，其中:**

1.  $UnitTestsFolder —包含单元测试的文件夹的路径；
2.  $ConnectionString —将为其创建测试的数据库的连接对象(在我们的示例中，它是 JobEmplDB 数据库)

**第二个 cmdlet 对数据库运行单元测试:**

1.  InputObject $ConnectionString 是我们需要对其运行测试的数据库的连接字符串。
2.  InstalltSQLtFramework 参数指定应该在运行测试之前安装 tSQLt 框架。

OutReportFileName $ TestReportOutputFileName 参数是输出测试报告文件的路径，ReportFormat JUnit 指定该文件的格式

将来，Invoke-DevartDatabaseTests cmdlet 将分为两个独立的部分，一部分用于运行自动测试，另一部分用于生成测试数据。

要传输架构更改，我们可以使用 Invoke-DevartSyncDatabaseSchema cmdlet 来使用以下 PowerShell 脚本:

```
$syncResult = Invoke-DevartSyncDatabaseSchema -Source $sourceConnection -Target $targetConnection
```

这里，$sourceConnection 是源数据库的连接字符串，$targetConnection 是目标数据库的连接字符串。$syncResult 变量将包含数据库模式同步的结果。

目前，没有用于在数据库之间同步数据的 cmdlet，但将来会有。

现在，我们可以使用启动 dbForge 数据泵的 Invoke-DevartDataImport cmdlet 来传输数据:

```
Invoke-DevartDataImport -Connection $connection -TemplateFile
```

这里，$connection 是 JobEmplDB 数据库的连接字符串，$importDataFileName 是用于创建导入数据的文件的全名。Devart 还有一系列其他有用的 cmdlets，它们将帮助使用 PowerShell 脚本实现各种 DevOps 流程组件。你可以在这里了解他们[。](https://docs.devart.com/devops-automation-for-sql-server/powershell-cmdlets/export-devartdbproject.html)

正如我们所看到的，在 dbForge 模式比较和 dbForge 数据比较(dbForge 数据比较目前只能在手动模式下工作)的帮助下，数据库更改(模式和数据中的更改)可以在环境之间自动转移。我们还可以使用 dbForge 数据泵组件 cmdlets 来传输数据。

在 PowerShell 脚本的帮助下，我们可以自动化整个开发运维流程，从代码编译到将代码部署到生产和主要监控。