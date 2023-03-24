# 使用 Python 从 Blackbaud CRM (BBEC)发送自动电子邮件

> 原文：<https://towardsdatascience.com/sending-automated-emails-from-blackbaud-crm-bbec-using-python-part-1-ba4955e91ca2?source=collection_archive---------20----------------------->

## [Blackbaud CRM Python 教程](http://towardsdatascience.com/tagged/python-and-blackbaud)

## 连接到 BBEC 的基本设置

我的新工作使用 Blackbaud CRM 作为他们数据基础设施的主干，我的经验主要是 Python。令我惊讶的是，关于使用 Python 连接到 Blackbaud Enterprise CRM (BBEC)的资源并不多。最近，我需要拉一个选民名单，并给他们发送一封定制的电子邮件，其中包含他们的帐户信息。这比我预想的要困难得多，我想分享一下我的技巧和工具，以备其他人想要处理类似的问题时使用。我发现的主要方法是使用 API，特别是 AppFxWebService 端点。

要实现这一点，需要几个主要步骤。API 是基于 SOAP 的，这意味着我们必须使用 HTTP 和 XML 来与之通信。我发现通过创建 Jinja2 模板可以很容易地将这些合并到 Python 脚本中。之后，我们将使用`pandas`和`xmltodict`将 XML 响应转换成数据帧。一旦数据变得更易于管理，我们将向选出的选民发送一些个性化的电子邮件。这篇文章将重点介绍如何使用 Python 建立到 Blackbaud CRM 的连接。我将在以后的文章中更详细地介绍这些其他主题，但这里是从 CRM 中提取信息并使用 Python 发送电子邮件所需的整个过程的概述:

*   使用 Python 建立与 Blackbaud CRM 的连接
*   使用 Jinja2 模板与 API 通信
*   将 XML 结果解析成熊猫数据帧
*   使用我们从 CRM 收到的数据发送电子邮件

在这个过程中， [Blackbaud CRM API 文档](https://www.blackbaud.com/files/support/guides/infinitydevguide/infsdk-developer-help.htm#infinitywebapi/coapioverview.htm)非常有用。根据您对这些概念的熟悉程度，您可能需要阅读入门指南。但是要注意，当我搜索这些文档时，我找不到任何 Python 示例。希望这篇指南能让你在下一个涉及 Blackbaud CRM 的项目中使用 Python。

另外，应该注意的是，这只是一个人解决这个问题的方法。如果你有任何建议或不同的方法来解决这些问题，我很乐意听到你的想法。

本指南假设您有一些 Python 经验，一些使用`requests`包的经验，也许还有一些使用`pandas`和数据分析的经验。如果你认为你能处理，我们将通过打开 IDE 或编辑器直接进入。我更喜欢 PyCharm，这也是本指南中的演示将使用的。然而，我会尽可能地包含使用替代方法完成这些步骤的链接。

第一步是在 PyCharm 中创建新项目。如果您选择另一条路线，主要步骤是为您的项目创建一个新目录，[创建并激活一个虚拟环境](https://docs.python.org/3/tutorial/venv.html)，并安装一些必需的包。如果你像我一样使用 PyCharm，你需要从欢迎界面选择“创建一个新项目”。然后在接下来的屏幕上从列表顶部选择“纯 Python”。默认情况下，我的配置是用 conda 创建一个新环境。通过将“位置”栏中的“无标题”更改为项目的描述性名称来更改项目名称。对于这个例子，我选择了`bbec_demo`。PyCharm 很好地自动创建了一个同名的新 conda 环境，并将其指定为这个项目的项目解释器。

接下来，我们想要在我们的项目中创建一个名为`config.py`的新 Python 文件。如果您非常喜欢 PyCharm 快捷方式，那么当您选择了项目根目录时，可以使用 Alt + Insert +'pf '创建一个新的 python 文件。然后，我们要将以下代码添加到该文件中:

```
from requests import Session
from requests.auth import HTTPBasicAuth
```

为了向 API 发送请求，我们将结合使用`requests`包和其他一些工具。如果您使用 PyCharm，它可能会为您建立一个虚拟环境，您只需安装 requests 包。在 PyCharm 中，最简单的方法是将光标放在标有红色下划线的包(`requests`)上，然后按 Alt + Shift + Enter。这将自动选择最合适的上下文菜单项，即“安装包请求”

接下来，在开始与 Blackbaud CRM 通信之前，我们需要收集一些不同的 URL。您可能会发现关于制作 HTTP 请求的文档页面[和](https://www.blackbaud.com/files/support/guides/infinitydevguide/content/infinitywebapi/cocraftinghttprequests.htm) [Fiddler 演示](https://www.blackbaud.com/files/support/guides/infinitydevguide/infsdk-developer-help.htm#infinitywebapi/cousingfiddlercreatehttprequest.htm)有助于找到这些请求。如果你登录了你公司的客户关系管理系统，你可以通过这个网址找到你需要的信息。例如，您可能有一个如下所示的 URL:

```
[https://bbisec02pro.blackbaudhosting.com/1234ABC_](https://bbisec04pro.blackbaudhosting.com/4249COL_fa341b46-12a4-4119-a334-8379e2e59d29/webui/WebShellLogin.aspx?databaseName=4249COL&url=https%3A%2F%2Fbbisec04pro.blackbaudhosting.com%2F4249COL_fa341b46-12a4-4119-a334-8379e2e59d29%2Fwebui%2FWebShellPage.aspx%3FdatabaseName%3D4249COL)fbd2546c-1d4d-4508-812c-5d4d915d856a[/webui/WebShellLogin.aspx](https://bbisec04pro.blackbaudhosting.com/4249COL_fa341b46-12a4-4119-a334-8379e2e59d29/webui/WebShellLogin.aspx?databaseName=4249COL&url=https%3A%2F%2Fbbisec04pro.blackbaudhosting.com%2F4249COL_fa341b46-12a4-4119-a334-8379e2e59d29%2Fwebui%2FWebShellPage.aspx%3FdatabaseName%3D4249COL)
```

可能有一个以“？”开头的查询字符串紧跟在`[WebShellLogin.aspx](https://bbisec04pro.blackbaudhosting.com/4249COL_fa341b46-12a4-4119-a334-8379e2e59d29/webui/WebShellLogin.aspx?databaseName=4249COL&url=https%3A%2F%2Fbbisec04pro.blackbaudhosting.com%2F4249COL_fa341b46-12a4-4119-a334-8379e2e59d29%2Fwebui%2FWebShellPage.aspx%3FdatabaseName%3D4249COL)`之后，但是我们可以忽略该信息。在本例中，我们希望所有内容都达到`/webui`。我们将把它记录在配置文件中，现在完整的文件应该是这样的:

```
from requests import Session
from requests.auth import HTTPBasicAuthbase_url = 'https://bbisec02pro.blackbaudhosting.com/'
database_name = '1234ABC_fbd2546c-1d4d-4508-812c-5d4d915d856a'
```

我把 URL 分成两部分，这样我们的代码可读性更好一些。我们需要的最后一部分是 AppFxWebService 端点，即`/appfxwebservice.asmx`。我们将使用 f 弦把所有这些放在一起。请注意，f 字符串仅在 Python 3.6 及更高版本中可用，如果您使用的是 Python 的旧版本，则必须使用`.format()`字符串方法。我们的配置文件现在应该如下所示:

```
from requests import Session
from requests.auth import HTTPBasicAuth

# URLs used for various API calls
base_url = 'https://bbisec02pro.blackbaudhosting.com/'
database_name = '1234ABC_fbd2546c-1d4d-4508-812c-5d4d915d856a'
appfx = '/AppFxWebService.asmx'
api = f'{base_url}{database_name}{appfx}'
```

我们现在已经精心制作了将用于与 API 通信的 URL。我们需要的下一部分是登录数据库的凭证。我选择使用 AWS Secret Management 来存储我的凭据。这相对容易，但是设置它可能超出了本文的范围。你需要按照`[boto3](https://pypi.org/project/boto3/)`和[的设置指南添加一个新的秘密](https://console.aws.amazon.com/secretsmanager/home?region=us-east-1#/newSecret?step=selectSecret)。与此相关的成本很小，但它非常安全，只需几分钱。如果你以前没有用过，甚至可以免费试用 30 天。

如果 AWS 方法不适合您，一个快速的替代方法是创建一个名为`secret.py`的新 Python 文件，但是要确保这个文件不受源代码控制(将其添加到您的`.gitignore`文件中)。这个文件应该和`config.py`在同一个目录下。它可以简单到:

```
username = 'BLACKBAUDHOST\your_username1234ABC'
password = 'your password'
```

我在这里遇到的一个问题是你用户名前后的前缀和后缀。如果您的环境由 Blackbaud 托管(如果您正在阅读本文，很可能就是这样)，那么合适的前缀是“BLACKBAUDHOST\”。请注意，它只适用于反斜杠，不适用于正斜杠。后缀通常作为用户名的一部分显示在 Blackbaud 中，但它应该是数据库名称。登录 Blackbaud CRM 后，您可以通过查看 URL 中的查询字符串来确认这一点:

```
.../WebShellPage.aspx?databaseName=1234ABC
```

`1234ABC`是需要添加到您的用户名后面的内容。确认您的用户名格式正确的另一种方法是在 Blackbaud CRM 中导航至管理>安全>应用程序用户，并查看您的姓名旁边列出的“登录名”内容。这将为您提供我们进行身份验证所需的完整登录名。我还想包含 AWS 的这个代码片段，展示如何访问存储在秘密管理器中的秘密，以防您决定走这条路:

```
import boto3
import base64
from botocore.exceptions import ClientError

def get_secret():

    secret_name = "credentials"
    region_name = "us-east-1"

    # Create a Secrets Manager client
    session = boto3.session.Session()
    client = session.client(
        service_name='secretsmanager',
        region_name=region_name
    )

    # In this sample we only handle the specific exceptions for the 'GetSecretValue' API.
    # See https://docs.aws.amazon.com/secretsmanager/latest/apireference/API_GetSecretValue.html
    # We rethrow the exception by default.

    try:
        get_secret_value_response = client.get_secret_value(
            SecretId=secret_name
        )
    except ClientError as e:
        if e.response['Error']['Code'] == 'DecryptionFailureException':
            # Secrets Manager can't decrypt the protected secret text using the provided KMS key.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InternalServiceErrorException':
            # An error occurred on the server side.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidParameterException':
            # You provided an invalid value for a parameter.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'InvalidRequestException':
            # You provided a parameter value that is not valid for the current state of the resource.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
        elif e.response['Error']['Code'] == 'ResourceNotFoundException':
            # We can't find the resource that you asked for.
            # Deal with the exception here, and/or rethrow at your discretion.
            raise e
    else:
        # Decrypts secret using the associated KMS CMK.
        # Depending on whether the secret is a string or binary, one of these fields will be populated.
        if 'SecretString' in get_secret_value_response:
            secret = get_secret_value_response['SecretString']
            secret = secret.strip('{}').replace('"', '')
            username, password = secret.split(':')
        else:
            decoded_binary_secret = base64.b64decode(get_secret_value_response['SecretBinary'])

    return username, password
```

它包括一些基本的错误处理，我只需要为自己的目的添加三行代码。

```
secret = secret.strip('{}').replace('"', '')
username, password = secret.split(':')
...
return username, password
```

这只是一些文本解析，将响应格式化为用户名和密码对。一旦完成，我就在函数的末尾添加`return username, password`。我要说的是，由于“\”字符，在我的用户名中包含“BLACKBAUDHOST”前缀时，我得到了一些奇怪的结果。我无法让它正确地转义，所以我选择从 AWS 中存储的秘密中删除它，并在检索到秘密后添加它。因为它不是用户名的唯一部分，我不认为这里有任何安全问题。

如果您走第一条路，将用户名和密码作为明文存储在源代码控制之外的文件中，那么您会想要将这个 import 语句添加到`config.py`:

```
from secret import username, password
```

如果您走 AWS Secrets Manager 路线，您的导入看起来会稍有不同，您必须分配用户名和密码:

```
import secretusername, password = secret.get_secret()
username = 'BLACKBAUDHOST\\' + username
```

由于`get_secret()`函数返回一个带有用户名密码对的元组，我们可以编写`username, password`来告诉 Python 解包元组并分别分配变量。然后我们会加上我之前提到的前缀。这里需要一个双反斜杠，因为单反斜杠是转义序列的开始，您需要第二个反斜杠来表示被转义的字符(" \ ")。

现在凭证已经处理好了，并且可以安全地访问它们，我们只需要做最后一点设置。我们必须用`requests`包创建一个会话，并使用我们的凭证授权该会话。

```
session = Session()
session.auth = HTTPBasicAuth(username, password)
```

我将在这里包括一个其他的位，这是我们所有的 API 调用将使用的头。在发送请求之前，我们将添加一点点信息，但是我喜欢设置一个默认的信息。一旦我们包含了这些，完整的`config.py`文件应该是这样的:

```
from requests import Session
from requests.auth import HTTPBasicAuth
import secret

# URLs used for various API calls
base_url = 'https://bbisec02pro.blackbaudhosting.com/'
database_name = '1234ABC_fbd2546c-1d4d-4508-812c-5d4d915d856a'
appfx = '/AppFxWebService.asmx'
api = f'{base_url}{database_name}{appfx}'
base_endpoint = 'Blackbaud.AppFx.WebService.API.1/'# Credentials
username, password = secret.get_secret()
username = 'BLACKBAUDHOST\\' + username# Authorization
session = Session()
session.auth = HTTPBasicAuth(username, password)# Headers
headers = {'Content-Type': 'text/xml; charset=utf-8',
           'Host': 'bbisec02pro.blackbaudhosting.com',
           'SOAPAction': ''}
```

我还偷偷添加了一行代码来定义`base_endpoint`。这是我们将用来配置我们的 SOAP 请求的东西。不是超级关键，但是为了方便可能要加上。`Content-Type`头将根据您使用的 SOAP 版本(1.1 或 1.2)而变化。到目前为止，我只使用了 1.1，所以这就是这里所反映的。您也可以通过自己导航到端点来找到此信息。登录你的 Blackbaud CRM，用`.../appfxwebservice.asmx`替换网址的`.../webui/WebShellPage.aspx?`部分和其后的所有内容。在这里，您将能够看到所有可用的操作以及使用它们所需的 SOAP 模板。模板显示了填充了你的数据库信息的标题，如果对你来说更容易的话，你可以窃取这些并把它们放在如上图所示的`config.py`中。

我们就要连接到数据库了，但是我们需要再创建一个名为`api.py`的文件。在这里，让我们多写一点代码来测试我们的连接:

```
import config

action = 'AdHocQueryGetIDByName'
body = '''<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xmlns:xsd="http://www.w3.org/2001/XMLSchema"
               xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
              <soap:Body>
                <AdHocQueryGetIDByNameRequest >
                  <ClientAppInfo REDatabaseToUse="1234ABC" 
                                 ClientAppName="bbec_demo" 
                                 TimeOutSeconds="5"/>
                  <Name>Interaction Data List Query</Name>
                </AdHocQueryGetIDByNameRequest>
              </soap:Body>
            </soap:Envelope>'''
```

现在，我从`/appfxwebservice.asmx` 端点借用了一个模板，具体来说就是名为“AdHocQueryGetIDByName”的模板，这就是为什么`action = ‘AdHocQueryGetIDByName'`。然后，我将文本直接粘贴到我们的文件中。我去掉了所有我不会使用的不重要的标签，然后我必须输入一些其他的信息。你可能会注意到`ClientAppInfo`标签。这不会出现在`/appfxwebservice.asmx`端点的模板中，但是我们的请求必须有效。他们只是在文档中简单地提到了它，当我第一次开始做这个的时候，它对我来说绝对是一个陷阱。不过，你所要做的就是将`REDatabaseToUse`设置为你的数据库名称(在你的 CRM 的 URL 的查询字符串中找到)并为`ClientAppName`输入一些名称。按照惯例，我喜欢将它设置为我们的项目名称，但它可以是您想要的任何名称。我们需要做的最后一件事是在`Name`标签中输入我们的信息库中存在的查询的名称。我选择“交互数据列表查询”是因为它是一个现成的报告，应该默认包含在您的 CRM 中。

接下来，让我们使用给定的操作创建一个函数来设置我们的标题:

```
def set_head(endpoint):
    headers = config.headers.copy()
    headers.update({'SOAPAction': f'{config.base_endpoint}{endpoint}'})
    return headers
```

这将更新我们在`config.py`文件中定义的头的副本，并返回给`SOAPAction`合适的头。

最后一步是使用我们在`config.py`文件中设置的`requests`和`session`发送 POST 请求。我们将添加我们的头部和身体，我们应该准备好了。

```
res = config.session.post(config.api, headers=set_head(action), data=body)
print(res)
```

我们的`api.py`文件作为一个整体现在应该看起来像这样:

```
import config

action = 'AdHocQueryGetIDByName'
body = '''<soap:Envelope xmlns:xsi="http://www.w3.org/2001/XMLSchema-instance"
               xmlns:xsd="http://www.w3.org/2001/XMLSchema"
               xmlns:soap="http://schemas.xmlsoap.org/soap/envelope/">
              <soap:Body>
                <AdHocQueryGetIDByNameRequest >
                  <ClientAppInfo REDatabaseToUse="1234ABC" 
                                 ClientAppName="bbec_demo" 
                                 TimeOutSeconds="5"/>
                  <Name>AdHocQueryGetIDByName</Name>
                </AdHocQueryGetIDByNameRequest>
              </soap:Body>
            </soap:Envelope>'''

def set_head(endpoint):
    headers = config.headers.copy()
    headers.update({'SOAPAction': f'{config.base_endpoint}{endpoint}'})
    return headers

res = config.session.post(config.api, headers=set_head(action), data=body)
print(res)
```

运气好的话，当您运行它时，它会打印出`<Response [200]>`。这意味着我们已经成功地建立了到数据库的连接，我们可以开始从中提取信息了！

在下一节中，我们将去掉 body 的那个难看的三重引号字符串，并用一些漂亮的 Jinja2 模板替换它。欢迎在下面提出建议或问题。你可以在我的 [GitHub](https://github.com/smidem/bbec_demo/) 上找到本节涉及的所有代码。