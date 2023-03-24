# 使用 Azure 数据工厂将数据从本地 SQL Server 移动到 Azure Blob 存储

> 原文：<https://towardsdatascience.com/move-data-from-on-premise-sql-server-to-azure-blob-storage-using-azure-data-factory-bbf67e4e5fde?source=collection_archive---------12----------------------->

![](img/2b9d21a56f1aba08e34b31d2d3b0e959.png)

虽然有一个将数据从本地 SQL Server 复制到 Azure Blob 存储的官方教程([https://docs . Microsoft . com/en-us/Azure/data-factory/tutorial-hybrid-copy-portal](https://docs.microsoft.com/en-us/azure/data-factory/tutorial-hybrid-copy-portal))，但本文将重点讨论该教程中没有涉及的一些细节。举个例子，

*   如何手动设置自托管集成运行时(SIR)？
*   SQL Server 安全配置提示
*   安全气囊系统连接绳配置提示
*   将 Azure Blob 存储添加为接收器的提示

本教程不会从创建 Azure 数据工厂(ADF)实例开始。如果你还没有，并希望从那里开始，使用上面的官方教程就足够了。

现在，我假设您已经准备好了本地 SQL Server 和 ADF 实例。现在我们应该开始了。

# 准备数据源

让我们创建一个测试数据库，将本地 SQL Server 中的测试应用表作为数据源。因此，首先转到您的 SQL Server Management Studio (SSMS)来创建数据库和表。

![](img/1320188f4b36a00c4582767ab260c0b4.png)

Right-click “Databases” and select “New Database…”

![](img/1733fa67dd4e760d65a6062af0f47eb2.png)

Input a database name

![](img/53e29a637fa6d5b4979c16d7ae297a48.png)

Right-click the database select “New Query”

复制粘贴下面的 SQL 脚本来创建表并插入一些测试条目。这段代码来自上面的官方教程

```
CREATE TABLE dbo.emp
(
     ID int IDENTITY(1,1) NOT NULL,
     FirstName varchar(50),
     LastName varchar(50)
)
GOINSERT INTO emp (FirstName, LastName) VALUES ('John', 'Doe')
INSERT INTO emp (FirstName, LastName) VALUES ('Jane', 'Doe')
GO
```

# 创建 SQL Server 用户

在实践中，我们应该让 SIR 使用 root 用户或任何具有 admin 权限的用户。相反，出于安全原因，我们希望为目标数据库创建一个只读用户。让我们创建这个用户。

![](img/2c77ade0eceaeb3fd3dbdd3c4ee3cfa9.png)

Right-click “Security” -> “Logins” under the SQL Server and select “New Login”

![](img/b924fcf1f002ca7de72d12a0a77b23e3.png)

Create the user called “adfuser”, input password

请注意，我们不想为此用户选中“用户必须在下次登录时更改密码”，因为它不应该提供给任何“用户”，而只能由 ADF 服务使用。

然后，让我们将这个用户添加到我们刚刚创建的数据库中。

![](img/947d106ffee45935db96a2126bf79772.png)

Right-click “Security” -> “Users” under the new database and select “New User”

![](img/6179cbbe62f8745984ddbb9c5f102013.png)

Input user name and default schema

![](img/45e695ab7f73788c5be8ee0d74d3cc8b.png)

In the “Membership” tab, enable “db_datareader”

之后，很重要的一步就是检查你的 SQL Server 是否已经启用了“SQL Server 身份验证模式”。否则，您创建的用户将永远无法访问。

![](img/62351bcc9673274f2d6706493e674a17.png)

Right-click the SQL Server Instance and go to “Properties”

![](img/9f906953ab06bba1ff25bf5eef1c635f.png)

Go to the “Security” tab and make sure “SQL Server and Windows Authentication mode” is enabled

# 创建 ADF 管道

转到 ADF Author 选项卡，单击`Pipelines`旁边的“…”按钮，并在下拉列表中选择`New pipeline`。

![](img/d0c909dea7ffd0f6be2e604d3c96c97a.png)

然后，在右窗格的主视图中，为管道命名。姑且称之为`OnPremSQLEmpToBlob_Pipeline`。

![](img/bac600086e223fd5932b7057322f0bb7.png)

展开`Move & transform`部分，将`Copy data`拖动到右侧主区域，重命名为`SQLEmpToBlobEmp_Activity`。

![](img/f7f8eeb8cfbaed958eaafcd95518a8ba.png)

之后，转到`Source`选项卡，点击`New`创建一个新的数据集作为源。

![](img/d308e690b588d711d186a325df079510.png)

搜索“sql server”以获取 SQL Server 连接。选择它并点按“继续”。

![](img/940106e1194cb8b47402bd631547d9c0.png)

让我们称这个数据集为`OnPremSQLServerEmp`，然后我们需要为这个数据集创建一个新的链接服务。

![](img/9bbf3c0fadc0fd3206fbd9f9432cb486.png)

让我们将新的链接服务命名为`OnPremSQLServer_LinkedService`，然后为这个本地资源创建一个集成时间。

![](img/6e8bb7eccea18b7d811bf857735a640b.png)

# 手动设置自承载集成运行时

在上一步之后，我们为集成运行时选择“New”。然后，我们选择“自托管”并单击继续。

![](img/a07d44c3a4326b163cfef8e26911d1b6.png)

输入 SIR 的名称。我想将其命名为`DevVmIntegrationRuntime`,因为我将在我的开发虚拟机上设置它。

![](img/744672d0ea212af2e206949279018c1d.png)

在这里，我们来到了重要的一步。在官方教程中，建议使用快速设置，这样肯定简单很多。然而，这在某些情况下可能是不可行的，例如，系统安全策略不允许这个自动脚本在这台机器上运行，当您想要使用快速设置时，经常会看到这个错误。

![](img/d674a7faeffcb0df7310c411e2f9ba43.png)

因此，我们需要使用选项 2:手动设置。因此，请下载 SIR 安装程序，并在您想要托管 SIR 的机器上运行它。

![](img/3364702af381e4a1efa7bde7604489d5.png)

只需等待 SIR 安装完成，然后复制密钥并粘贴到已安装的 SIR 中，将它们连接在一起。

![](img/0277d60eaeaaa563da4059fd41975073.png)![](img/39d8d08cecadc469169ffe05c03ff2e4.png)

它应该会自动相互连接，并且运行 SIR 的机器名称会自动填充为节点名称。你可以重命名它，但对我来说，这没关系。

单击“完成”并等待它自己注册，然后单击“启动配置管理器”，您应该能够看到如下屏幕。

![](img/e86fce71f91a6474651b9f9ea4282674.png)

然后，我们可以在`Diagnostics`选项卡中测试 SQL Server 连接，如下所示。请注意，我在本地 SQL Server 的同一台机器上运行 SIR，因此我使用`localhost`作为服务器名称。然而，出于安全考虑，在实践中不建议这样做。如果您将一台独立的机器与 SQL Server 实例连接，只需用机器名和实例名替换`localhost`。

另一个重要的提示是，在这个连接测试中，我们需要使用双斜杠`\\`来分隔服务器名和实例名。

![](img/ea6875a175463a9d3b1d63f54c27cfdd.png)

然后，我们回到 ADF。应用并关闭安全气囊系统配置面板。现在，我们也想测试这里的连接。请看下面的截图。

![](img/65c42e6198fc107bc0f30870524e4267.png)

另一个非常重要的提示是，我在其他地方没有找到任何详细说明。也就是说，这里的服务器名不应该是双斜杠。这将导致您测试连接时出错。相反，您应该使用单斜线`\`来分割服务器名和实例名。我相信这是因为 SIR 依靠反斜杠来转义斜杠字符，但 ADF 不需要转义，但它会将双斜杠转换为实际上与服务器名和实例名不匹配的两个斜杠符号。因此，您可能会得到以下错误:

> 连接失败
> 
> 无法连接到 SQL Server:“localhost \ \ SQLEXPRESS”，数据库:“testdb”，用户:“adfuser”。实例失败。活动 ID: xxxxxx-xxxxx-xxxxx。

现在，我们已经有了一个到本地 SQL Server 的有效链接服务器。让我们选择要从中提取数据的表，即`dbo.emp`，然后单击 OK。

![](img/0b8fb5d0aa05122bf742a67cd4e5a83f.png)

# 在 Azure Blob 存储中创建接收器数据集

转到管道配置面板上的接收器选项卡，单击“新建”以添加新数据集。

![](img/aa73e25eb0bb2ea293f7b4186963ea95.png)

然后，选择 Azure Blob 存储并单击继续。

![](img/72f2db6e465bf11b362398d6820d7d90.png)

在这里，我将使用`DelimitedText`作为一个例子，因为它是最可读的。

![](img/930886e5f5b390753cc321eb6d97d7f9.png)

然后，按照说明创建新的链接服务。这次我们将使用`AutoResolveIntegrationRuntime`，它将是 Azure Integration Runtime (AIR)，因为 Azure Blob 存储是 Azure Cloud 中的原生服务。

![](img/d1be73c61f3c22e063097edc183cca2f.png)

在下一步中，将自动选择创建的链接服务。我们需要输入接收器数据集的文件路径。这里我已经创建了一个名为`test-container`的 blob 容器，但是文件夹`output`和文件`emp.csv`目前并不存在。ADF 会自动为我们创建它们。

![](img/8194a0d4ac930e196c49f41262c5e171.png)

提示 1:如果您想在 CSV 文件中包含字段标题，请选择“首行作为标题”。

提示 2:我们需要为`Import schema`选择“None ”,否则，由于模式导入失败错误，您将无法进入下一步:

> 架构导入失败:缺少所需的 Blob。container name:[https://xxxxx.blob.core.windows.net/test-container,](https://ctstoragetest.blob.core.windows.net/test-container,)container exist:True，BlobPrefix: emp.csv，BlobCount: 0。\r\n .活动 ID: xxxxx-xxxxx-xxxxx

# 执行管道

现在，我们应该准备好运行这个数据移动管道了。让发布一切，然后到管道面板顶部工具栏，点击“添加触发器”->“立即触发”。

![](img/c802529354f4e4d84dc76bfdd0c556e6.png)

# 核实结果

等到通知说管道成功，然后转到 Blob 存储检查输出文件。我更喜欢使用 Azure Storage Explorer，因为我可以直接下载并轻松打开 CSV 文件，但如果你更喜欢 Azure Portal 中的 web 视图，这也没问题。

![](img/3c214c7575dd94482b6e241d0b8e168b.png)[](https://medium.com/@qiuyujx/membership) [## 通过我的推荐链接加入灵媒-陶

### 作为一个媒体会员，你的会员费的一部分会给你阅读的作家，你可以完全接触到每一个故事…

medium.com](https://medium.com/@qiuyujx/membership) 

如果你觉得我的文章有帮助，请考虑加入 Medium 会员来支持我和成千上万的其他作者！(点击上面的链接)

# 资源:

Azure 文档，将数据从本地 SQL Server 数据库复制到 Azure Blob 存储:[https://docs . Microsoft . com/en-us/Azure/data-factory/tutorial-hybrid-Copy-portal](https://docs.microsoft.com/en-us/azure/data-factory/tutorial-hybrid-copy-portal)

SQL Server 2017:[https://www . Microsoft . com/en-in/SQL-Server/SQL-Server-downloads](https://www.microsoft.com/en-in/sql-server/sql-server-downloads)

Azure 存储浏览器:[https://azure . Microsoft . com/en-GB/features/Storage-Explorer/](https://azure.microsoft.com/en-gb/features/storage-explorer/)