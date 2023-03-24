# 使用 PaaS 加速数据科学

> 原文：<https://towardsdatascience.com/paas-to-accelerate-data-science-dc-os-data-science-engine-656842dd7ba8?source=collection_archive---------17----------------------->

## 试用 DC/操作系统数据科学引擎

作为一名专注于向市场提供新产品的全栈机器学习工程师，我经常发现自己处于数据科学、数据工程和开发运营的交叉点。因此，我一直怀着极大的兴趣关注着数据科学平台即服务(PaaS)的兴起。在这一系列帖子中，我将评估不同的平台即服务(PaaS)及其自动化数据科学运营的潜力。我将探索他们的能力，然后自动设置和执行我即将出版的书*中的代码* (O'Reilly，2020 年)，为这本书的读者找到阅读这本书的最佳方式。

在我的上一本书[敏捷数据科学 2.0](https://www.amazon.com/Agile-Data-Science-2-0-Applications/dp/1491960116) (4.5 星:D)中，我为读者构建了自己的平台，使用 bash 脚本、AWS CLI、 [jq](https://stedolan.github.io/jq/) 、vagger 和 EC2 运行代码。虽然这使得这本书对那些运行代码有困难的初学者来说更有价值，但是维护和保持运行是非常困难的。旧软件从互联网上脱落，平台腐烂。这个项目已经出现了 85 个问题，虽然许多问题已经通过读者的贡献得到了解决，但它仍然占据了我在开源软件上花费的大量时间。这次会有所不同。

*注:本岗位代码可在*[*github.com/rjurney/paas_blog*](https://github.com/rjurney/paas_blog)获得

# DC/操作系统数据科学引擎

![](img/49dea0e9091342a615315ead6552eed1.png)

DC/OS is a highly scalable PaaS

我评估的第一个数据科学 PaaS 是新推出的 DC/OS 数据科学引擎。在这篇文章中，我将通过 GUI 向您介绍我对 DC/OS(注意:我过去曾使用过它)及其数据科学引擎的初步实验，然后我们将讲述如何用几行代码来自动化相同的过程。事实证明，这实际上比使用 AWS CLI 创建等效资源更简单，这给我留下了深刻的印象。我们将设置我们的环境和软件先决条件，使用 Terraform 和[通用安装程序](https://docs.d2iq.com/mesosphere/dcos/1.13/installing/evaluation/aws/)初始化 DC/操作系统集群，安装[数据科学引擎包](https://universe.dcos.io/#/package/data-science-engine/version/latest)，然后通过运行标记堆栈溢出帖子的模型来评估环境。

一些警告:我们将一步一步地介绍如何引导安装了 JupyterLab 的 DC/OS 集群，但是如果您遇到麻烦，您可以随时参考[通用安装程序文档](https://docs.d2iq.com/mesosphere/dcos/1.13/installing/evaluation/aws/)来解决 DC/OS 平台问题，或者参考[数据科学引擎文档](https://docs.d2iq.com/mesosphere/dcos/services/data-science-engine/1.0.0/quick-start/)来解决部署该服务的问题。

# 为什么选择 DC/操作系统数据科学引擎？

在任何给定的云环境中，如亚马逊网络服务(AWS)、谷歌云平台(GCP)和 Azure，为个人数据科学家工作设置一个 Jupyter 笔记本已经变得相当容易。对于创业公司和小型数据科学团队来说，这是一个很好的解决方案。没有什么需要维护的，笔记本可以保存在 Github 中以便持久化和共享。

对于大企业来说，事情没有这么简单。在这种规模下，跨多个云的临时资产上的临时环境可能会造成混乱而不是秩序，因为环境和建模变得不可复制。企业跨多个云和在本地工作，具有特定的访问控制和身份验证要求，并且需要提供对内部资源的访问以获得数据、源控制、流和其他服务。

*对于这些组织，DC/操作系统数据科学引擎提供了一个统一的系统，该系统提供 Python ML 堆栈、Spark、Tensorflow 和其他 DL 框架，包括*[*TensorFlowOnSpark*](https://github.com/yahoo/TensorFlowOnSpark)*以支持分布式多节点、多 GPU 模型训练。*这是一个非常引人注目的设置，开箱即用，可以结束大型数据科学团队和公司的许多挫折和复杂性。

# 软件和平台先决条件

DC/操作系统通用安装程序可以在 Linux、Mac OS X 和 Windows 上运行。对于所有平台，你都需要 [Python](https://www.python.org/about/gettingstarted/) 、 [pip](https://pip.pypa.io/en/stable/) 、 [AWS 命令行界面(CLI)](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-welcome.html) 和 [Terraform](https://www.terraform.io/intro/index.html) 。如果您正在测试数据科学引擎，您可能已经安装了 Python，但是如果没有，我们将安装 [Anaconda Python](https://www.anaconda.com/distribution/) 。

*您还需要在运行 DC/操作系统数据科学引擎的地区授权 5 个 GPU 实例(在本文中，我们使用“us-west-2”)。让我们一步一步来。*

# 安装 Anaconda Python

![](img/bd19aa31d0cad937f362921f57e1d59d.png)

Anaconda Python

如果您的机器上已经安装了 Python，就没有必要安装 Anaconda Python。如果您不知道，我推荐 Anaconda Python，因为它易于安装，位于您的主目录中，并且具有优秀的 conda 包管理器。

下载蟒蛇 Python 3。x 为您的平台[这里](https://www.anaconda.com/distribution/#download-section)，然后按照说明安装 [Linux](https://docs.anaconda.com/anaconda/install/linux/) 、 [Mac OS X](https://docs.anaconda.com/anaconda/install/mac-os/) 和 [Windows](https://docs.anaconda.com/anaconda/install/windows/) 。

# 通过安装和配置 Amazon Web Services CLI 进行 AWS 身份验证

![](img/e60352c3a3f47cceea3c73a0f1513fba.png)

为 [Terraform AWS 提供者](https://www.terraform.io/docs/providers/aws/index.html)设置 AWS 访问有两种方式:通过 [AWS CLI](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html#cli-quick-configuration) 或通过编辑(来自 Github 项目根的路径)*PAAs _ blog/dcos/terra form/terra form . TF*。

首先，我们将为 Terraform 设置要使用的 AWS 凭证。要使用 PyPI [awscli 包](https://pypi.org/project/awscli/):

```
# Install using conda
conda install -c conda-forge awscli# OR…# Install using pip
pip install awscli --upgrade --user
```

现在[使用您的访问密钥 ID 和秘密访问密钥设置](https://docs.aws.amazon.com/cli/latest/userguide/cli-chap-configure.html)您的 AWS 凭证，您可以在 [IAM 控制台](https://console.aws.amazon.com/iam/home#/security_credentials)中找到它们。您可以将区域更改为您自己的默认值。Terraform 的 AWS 模块将默认使用这些凭证。它们存放在 *~/中。aws/* 目录。

```
aws configureAWS Access Key ID [None]: **AKIAIOSAODNN7EXAMPLE**
AWS Secret Access Key [None]: **wJalrXUtnFEMI/K7MBENG/bPxRfiCYEXAMPLEKEY**
Default region name [None]: **us-west-2**
Default output format [None]: **json**
```

现在验证设置是否有效:

```
aws sts get-caller-identity
```

您应该会看到类似这样的内容:

```
{
  “UserId”: “123456789101”,
  “Account”: “109876543210”,
  “Arn”: “arn:aws:iam::123456789101:root”
}
```

# 通过编辑 terraform.tf 进行 AWS 身份验证

您可以通过编辑(从项目根目录)*PAAs _ blog/dcos/terra form/terra form . TF*来显式设置 AWS 身份验证，以便在调用 AWS 提供者时包含您的凭证。只需添加您的访问密钥/秘密密钥和默认区域。

```
provider “aws” {
  access_key = “foo”
  secret_key = “bar”
  region = “us-west-2”
}
```

# 安装 Terraform

![](img/3a543272b73eea43fba006d82988c010.png)

> Terraform 使用户能够使用一种称为 Hashicorp 配置语言(HCL)的高级配置语言来定义和配置数据中心基础架构。
> 
> —维基百科，Terraform(软件)

DC/OS 通用安装程序需要安装 Terraform 11.x. Ubuntu 用户可以像这样安装 Terraform 11.x:

```
wget [https://releases.hashicorp.com/terraform/0.11.14/terraform_0.11.14_linux_386.zip](https://releases.hashicorp.com/terraform/0.11.14/terraform_0.11.14_linux_386.zip)
sudo apt-get install unzip
unzip terraform_0.11.14_linux_386.zip
sudo mv terraform /usr/local/bin/
```

在 Mac 上，使用[自制软件](https://brew.sh/):

```
brew unlink terraform || true
brew install tfenv
tfenv install 0.11.14
```

Windows 用户可以使用 [Chocolatey](https://chocolatey.org/) :

```
choco install terraform --version 0.11.14 -y
```

验证 terraform 是否有效:

```
terraform --version
```

您应该看到:

```
Terraform v0.11.14
```

现在我们可以走了！忽略任何关于升级的消息。既然已经安装了 Terraform，我们就可以在 AWS 上配置和启动我们的 DC/操作系统集群了。

# 为 GPU 实例授权 AWS 服务限制

![](img/50455a39b4f8a0123daf797aec45a8a6.png)

[AWS 服务限制](https://docs.aws.amazon.com/general/latest/gr/aws_service_limits.html)定义您在任何给定地区可以使用多少 AWS 资源。我们将启动 5 个 p 3.2x 大型 GPU 实例，作为运行笔记本电脑的 DC/操作系统集群的私有代理。**p 3.2x 大型实例类型的默认服务配额为 0。为了运行本教程，您需要请求 AWS 将这个值增加到 5 或更多。**

您可以通过登录 AWS Web 控制台并访问此处的服务配额控制台来完成此操作:[https://us-west-2 . Console . AWS . Amazon . com/Service Quotas/home？区域=美国-西部-2#！/services/ec2/quotas](https://us-west-2.console.aws.amazon.com/servicequotas/home?region=us-west-2#!/services/ec2/quotas) (您可以在 url 中替换您首选的 [AWS 区域](https://docs.aws.amazon.com/AWSEC2/latest/UserGuide/using-regions-availability-zones.html#concepts-available-regions)，只要确保将其替换在 url 中出现的 _both_ places 中即可)。在搜索框中搜索 *p3.2xlarge* ，点击右侧橙色*请求配额增加*按钮。

![](img/528bc748bf9d8d3aed8ee5b3839f385b.png)

AWS Service Limit Quota Increase for p3.2xlarge instances

在*更改配额值*字段中输入 *5* 。然后点击右下角的橙色*请求*按钮。

![](img/dc849f8e64b841c5ecd3bd2e90e2eb8b.png)

Quota Increase Submit Form

现在，您必须等待 12-48 小时，请求才能获得批准。我有一个基本的 AWS 帐户，当我在下午请求增加时，它在第二天早上被批准。如果您需要加快速度，您可以前往 AWS [支持中心](https://console.aws.amazon.com/support/home?region=us-west-2)请求代理致电。他们通常可以加速很多事情。

# 带通用安装程序的 AWS 上的 DC/操作系统

![](img/49dea0e9091342a615315ead6552eed1.png)

在[d2iq.com](http://docs.d2iq.com)上有很好的[文档](https://docs.d2iq.com/mesosphere/dcos/1.13/installing/evaluation/aws/)用于 DC/操作系统通用安装程序，但是我将在这篇文章中提供“正常工作”的代码。我们将使用 Terraform 来配置和启动我们的集群，然后我们将安装[数据科学引擎包](https://universe.dcos.io/#/package/data-science-engine/version/latest)，并开始使用 JupyterLab！

**注:将目录改为** [***Github 项目***](https://github.com/rjurney/paas_blog) **子目录**[***PAAs _ blog/dcos/terraform***](https://github.com/rjurney/paas_blog/tree/master/dcos/terraform)**以了解教程的剩余内容。**

# 配置 DC/操作系统

配置 DC/操作系统集群的第一步是创建一个特定于集群的 ssh 密钥。我们把这个键叫做 *my_key.pub* 。

```
ssh-keygen -t rsa -f my_key
```

按两次回车键创建没有密码的密钥。

```
Generating public/private rsa key pair.
Enter passphrase (empty for no passphrase):
Enter same passphrase again:
Your identification has been saved in my_key.
Your public key has been saved in my_key.pub.
The key fingerprint is:
SHA256:OCOfrIzrTznoMqKOJ2VD2r6YPQp8jFFKIbe/BDyhWuI rjurney@Hostname.local
The key’s randomart image is:
+ — -[RSA 2048] — — +
|..o                |
|.+.o               |
|o.*.               |
|+o++ .             |
|.E + + S           |
|o Oo * +           |
|.*.+= +            |
|B==+ o             |
|XOB=+              |
+ — — [SHA256] — — -+
```

我们需要把 *my_key* 的权限改成只对我们的用户 *0600* 可读，不然以后 ssh 会抱怨的。

```
chmod 0600 ./my_key
```

现在运行`ssh-agent`(如果它没有运行)并将密钥添加到代理中。

*注意:如果没有这个步骤，在创建集群时会出现 ssh 错误。请参见下面的常见错误部分。*

```
eval "$(ssh-agent -s)"
ssh-add ./my_key
```

现在，验证是否已经添加了密钥:

```
ssh-add -l
```

这应该显示:

```
2048 SHA256:1234567891234567891234567891234567891234567 ./my_key (RSA)
```

# 通过脚本创建超级用户密码哈希

现在我们将使用 Python 的 [hashlib](https://docs.python.org/3/library/hashlib.html) 模块为集群创建一个超级用户密码散列文件。我们将调用我们的*dcos _ super user _ password _ hash*。

我已经创建了一个命令行脚本，它将生成、打印一个名为*PAAs _ blog/dcos/terra form/generate _ password _ hash . py*的密码散列并写入磁盘。

```
usage: generate_password_hash.py [-h] PASSWORDGenerate an SHA512 password hash for the given passwordpositional arguments:
 PASSWORD A password to generate a SHA512 hash ofoptional arguments:
 -h, — help show this help message and exit
```

要运行它，只需运行:

```
python ./generate_password_hash.py my_password
```

您应该会看到类似这样的输出:

```
Wrote SHA512 digest: '1234567891011121314151617181920212223242526…' for password 'my_password' to ./dcos_superuser_password_hash
```

验证写入文件*dcos _ super user _ password _ hash*是否成功:

```
cat dcos_superuser_password_hash
```

在 Python 中创建超级用户密码散列

要手动创建散列，请打开 Python shell(考虑使用 [ipython](https://ipython.org/) ):

```
python # or ipython
```

现在运行以下代码:

```
import hashlibwith open('dcos_superuser_password_hash', 'w') as f:
  m = hashlib.sha512('my_password'.encode())
  f.write( m.hexdigest() )
```

将密码散列的权限更改为仅对您的用户可读， *0600* ，否则 DC/操作系统 CLI 将报错:

```
chmod 0600 ./dcos_superuser_password_hash
```

验证密码已成功保存:

```
cat dcos_superuser_password_hash
```

您应该会看到类似这样的内容:

```
dda8c3468860dcb2aa228ab8ee44208d43eb2f5fd2a3a538bafbd8860dcb24e228ab8ee44208d43eb2f5a2f8b143eb2f5a2f8b17eccc87ffe79d8459e2df294fb01
```

# 创建许可证文件—(仅限企业版)

由于我们将使用开放版本的 DC 操作系统，我通过`touch license.txt`创建了一个空的 *license.txt* 文件。它需要存在，但可以是空的。它已经致力于 Github，所以对于 DC/OS 的开放版本，一旦你签出了 [GitHub 项目](https://github.com/rjurney/paas_blog)，你就不需要创建它了。如果您使用的是企业版，您需要将您的实际许可证放在 *license.txt* 中。

# 配置我们的 DC/操作系统集群

我已经编辑了文件*PAAs _ blog/dcos/terra form/desired _ cluster _ profile . TF vars*以个性化超级用户帐户名称，从我们上面创建的密码哈希文件中加载超级用户密码哈希，以指定空的许可证字符串并指向空的 *license.txt* 文件。DC/操作系统版本设置为 *open* ，DC/操作系统版本设置为 *1.13.3* ，我们将使用一个 *m5.xlarge* 作为我们的引导实例类型，我们将使用 *p3.2xlarge* 实例来运行 JupyterLab。我们将 GPU 代理的数量设置为 5，这足以运行 Spark 和 JupyterLab。最后，我们指定我们之前生成的公钥， *my_key.pub* 。记住使用公钥，而不是私钥。

注:你可以在[发布页面](https://dcos.io/releases/)找到最新版本的 DC/OS，点击其 [Github repo](https://github.com/dcos-terraform/terraform-aws-dcos) 上的*标签*可以找到最新版本的通用安装程序。如果遇到问题，通过编辑 *terraform.tf* 中的*版本*键，使用最高的 *0.2.x* 标签。

```
cluster_owner = "rjurney"
dcos_superuser_password_hash = "${file("dcos_superuser_password_hash")}"
dcos_superuser_username = "rjurney"
dcos_license_key_contents = ""
dcos_license_key_file = "./license.txt"
dcos_version = "1.13.4"
dcos_variant = "open"
bootstrap_instance_type = "m5.xlarge"
gpu_agent_instance_type = "p3.2xlarge"
num_gpu_agents = "5"
ssh_public_key_file = "./my_key.pub"
```

注意:如果您使用的是 DC 操作系统的**企业版**，您需要填写***dcos _ license _ key _ contents***，对于开放版本，我们会将其留空。您还需要更改配置，使`dcos_variant =` “ee”。

# 初始化地形

首先，我们需要[用我们将要使用的所有模块初始化](https://www.terraform.io/docs/commands/init.html) Terraform:

```
terraform init -upgrade
```

您应该看到:

```
Upgrading modules...<list of modules>
Initializing provider plugins…
<list of plugins>Terraform has been successfully initialized!
```

# 生成地形图

现在我们需要使用我们在*PAAs _ blog/dcos/terraform/desired _ cluster _ profile . TF vars*中定义的变量来生成一个[行动计划](https://www.terraform.io/docs/commands/plan.html)供 terra form 执行。我们将计划保存到*PAAs _ blog/dcos/terra form/plan . out*。

```
terraform plan -var-file desired_cluster_profile.tfvars -out plan.out
```

您应该会看到很多没有错误的输出，结尾是:

```
Plan: 75 to add, 3 to change, 0 to destroy. — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — — This plan was saved to: plan.outTo perform exactly these actions, run the following command to apply:
 terraform apply "plan.out"
```

现在我们有一个计划，让 Terraform 在 *plan.out* 中创建我们的 DC/操作系统集群，它是二进制的，不太容易检查。

# 启动我们的 DC/操作系统集群

现在我们已经有了一个包含自定义变量的计划，我们不需要在 apply 命令中再次包含它们。我们可以按照*plan*命令输出末尾的指示去做。注意，我们没有将`--var-file`与 apply 一起使用，因为 plan 已经将我们的变量插入到计划中。

```
terraform apply plan.out
```

该命令可能需要长达 15 分钟的时间来执行，因为在初始化 AWS 资源序列，然后执行命令来初始化 EC2 实例上的服务时会有延迟。您应该会看到很多输出，开始是:

```
module.dcos.module.dcos-infrastructure.module.dcos-lb.module.dcos-lb-masters.module.masters.tls_private_key.selfsigned: Creating...<lots more stuff>
```

如果您看到任何错误，最好是销毁集群并重试。AWS 资源延迟初始化中的定时问题有时会导致引导问题。

# 摧毁我们的 DC/操作系统集群

当您完成集群时，如果有错误，或者如果您需要重新创建它，您可以使用下面的命令销毁所有相关的资源。注意，我们确实需要将`--var-file`与`destroy`一起使用。

```
terraform destroy --auto-approve --var-file desired_cluster_profile.tfvars
```

这可能需要一点时间，因为有许多资源要删除。一旦销毁完成，你可以自由地计划和再次申请。

# 登录我们的 DC/操作系统集群

“terraform apply”命令的最终输出为我们提供了要连接的主节点的地址，应该如下所示:

```
Apply complete! Resources: 75 added, 0 changed, 0 destroyed.Outputs:masters-ips = [
  54.XX.XXX.XXX
]
public-agents-loadbalancer = ext-jupyter-gpu-12a9-abcde6g1234.elb.us-west-2.amazonaws.com
```

在浏览器中打开主 ip 地址，您应该会看到一个登录窗口。请注意，唯一授权连接到这台机器的 IP 是您的源 IP。

![](img/cd74951ff5388a99dbef7bce7ffa548e.png)

DC/OS Cluster Authentication Page

选择你的认证方式——我用谷歌。一旦您通过验证，它会将您返回到主页。

# 安装数据科学引擎服务

![](img/315ef78a57e473928f9b12b8b41a37f1.png)

DC/OS Dashboard

在左侧菜单中，从顶部往下数第四个，在此图中用橙色圈出的是*Catalog*项目。点击它，DC/操作系统服务菜单将出现。当我这样做的时候，数据科学引擎服务就出现在第二行，如果没有的话，可以使用左上角的搜索框来找到它。

![](img/d15183d3a8456593736c87ea92e5c612.png)

The DC/OS Service Catalog — Every Package a One Click Instasll

点击*数据-科学-引擎*服务，其服务页面就会出现。点击*查看&运行*安装服务。

![](img/120fd1018c51b41fedeb05335b297550.png)

The Data Science Engine Catalog Page

这将打开一个窗口，您可以在其中编辑数据科学引擎的配置。您需要使用字母和破折号来命名集群，但是您如何命名并不重要，除非名称是唯一的。因为我们使用的是 *p3.2xlarge* 实例，所以将服务配置为使用 **58GB 内存和 8 个 CPU**。选中 Nvidia GPU 分配配置已启用复选框，并为 GPU 数量键入 1。

![](img/f48c70bec8797dd3f29d323778f17dfa.png)

Data Science Engine Service Configuration

点击右上角的紫色*审查&运行*按钮。这将带您进入最终审查屏幕。点击右上角的紫色*运行服务*按钮。

*请注意，您可以将服务配置作为 JSON 下载，以便稍后使用 DC/OS CLI 运行，从而使您能够自动部署服务，例如作为您的持续集成系统的一部分。为此，请单击下载配置。*

![](img/2052fa8af2250cc0f91ec8929a6a9321.png)

Data Science Engine Run Confirmation Page

您应该会看到一个弹出窗口，宣布系统成功启动。点击*打开服务*。

![](img/cc327bf8206959896af4c727790648d7.png)

Service Confirmation Popup

这将带您进入数据科学引擎服务页面。起初，页面的状态会显示服务正在加载，但很快健康样本会变成绿色，状态会显示*正在运行*。

![](img/9d178a6febe6acc94ebe51788216595c.png)

Data Science Engine Service Page

# 使用数据科学引擎

现在点击文本*服务*，在屏幕白色区域的顶部显示*服务>数据科学引擎*。这将带您进入服务列表。您应该会看到列出了*数据科学引擎*。单击下图中用橙色圈出的启动图标。这将打开与 JupyterLab 的连接。默认的 Jupyter 密码是 *jupyter* ，但是您可以使用我们用来启动服务的服务配置窗口来设置它。

![](img/008ee80714206a11702999c220bc6a76.png)

Running Services Page — Click on the open icon to open JupyterLab

一旦输入 *jupyter* 的默认 Jupyter 密码(可以在*desired _ cluster _ profile . TF vars*中更改)，就会看到 JupyterLab 的主页。第一页加载可能需要一点时间。Tada！

![](img/97019f3b9b554f32a3d6ec8b059fe574.png)

JupyterLab Dashboard

# 用一些自然语言处理来锻炼我们的 GPU

现在我们已经启动了集群和服务，让我们通过训练一个神经网络来标记堆栈溢出问题来练习它。我们将此视为一个多类别、多标签的问题。通过对至少有一个答案、一个投票并且至少有一个标签出现超过 2000 次的问题的完整转储进行上采样，来平衡训练数据。大约 600MB。这个数据集是[之前计算的](https://github.com/rjurney/deep_products/blob/master/code/stackoverflow/get_questions.spark.py)，文件可以在 Github repo 的 *paas_blog/data* 目录中找到。

你可以在 github.com/rjurney/paas_blog/DCOS_Data_Science_Engine.ipynb[的 Github 上查看 Jupyter 笔记本的代码。我们将使用 JupyterLab Github 接口打开它，但是如果你愿意，你可以将它的内容逐块粘贴到新的 Python 3 笔记本中。](https://github.com/rjurney/paas_blog/blob/master/DCOS_Data_Science_Engine.ipynb)

# 加载教程笔记本

JupyterLab 的 Github 模块超级棒，让加载教程笔记本变得很容易。点击屏幕最左边的 Github 图标，在文件和跑男图标之间。在显示 *<的地方输入 *rjurney* 编辑用户>* 。

![](img/9a6c4050d1e8ddf1ebfde114dab33537.png)

JupyterLab’s Github Browser

我的公共 Github 项目将会出现。选择 *paas_blog* 然后双击*DCOS _ 数据 _ 科学 _ 引擎. ipynb* Jupyter 笔记本打开。它使用 S3 的数据，所以你不必下载任何数据。

![](img/19df3d3e640d69a857b475cd41073287.png)

Example Notebook from Github Loaded in JupyterLab

# 验证 GPU 支持

要做的第一件事是验证我们的数据科学引擎 EC2 实例上的 JupyterLab Python 环境是否正确配置为与其板载 GPU 一起工作。我们使用`tensorflow.test.is_gpu_available`和`tensorflow.compat.v2.config.experimental.list_physical_devices`来验证 GPU 是否与 Tensorflow 一起工作。

```
gpu_avail = tf.test.is_gpu_available(
  cuda_only=False,
  min_cuda_compute_capability=None
)
print(f'1 or more GPUs is available: {gpu_avail}')from tensorflow.python.client import device_lib
local_devices = device_lib.list_local_devices()
gpu = local_devices[3]
print(f"{gpu.name} is a {gpu.device_type} with {gpu.memory_limit / 1024 / 1024 / 1024:.2f}GB RAM")
```

您应该会看到类似这样的内容:

```
1 or more GPUs is available: True
/device:GPU:0 is a GPU with 10.22GB RAM
```

# 从 S3 加载数据

您可以使用 *pandas.read_parquet* 加载本教程的数据。

```
# Load the Stack Overflow questions right from S3
s3_parquet_path = f's3://{BUCKET}/08–05–2019/Questions.Stratified.Final.2000.parquet'
s3_fs = s3fs.S3FileSystem()# Use pyarrow.parquet.ParquetDataset and convert to pandas.DataFrame
posts_df = ParquetDataset(
  s3_parquet_path,
  filesystem=s3_fs,
).read().to_pandas()posts_df.head(3)
```

现在我们加载索引，在标签索引和文本标记之间来回转换。我们将使用这些来查看教程最后预测的实际结果标签。

```
# Get the tag indexes
s3_client = boto3.resource('s3')def json_from_s3(bucket, key):
  """Given a bucket and key for a JSON object, return the parsed object"""
  obj = s3_client.Object(bucket, key)
  obj.get()['Body'].read().decode('utf-8')
  json_obj = json.loads(obj.get()[‘Body’].read().decode('utf-8'))
  return json_objtag_index = json_from_s3(BUCKET, '08–05–2019/tag_index.2000.json')
index_tag = json_from_s3(BUCKET, '08–05–2019/index_tag.2000.json')list(tag_index.items())[0:5], list(index_tag.items())[0:5]
```

然后，我们验证加载的记录数量:

```
print(
  ‘{:,} Stackoverflow questions with a tag having at least 2,000 occurrences’.format(
    len(posts_df.index)
  )
)
```

您应该看到:

```
1,554,788 Stackoverflow questions with a tag having at least 2,000 occurrences
```

# 准备数据

我们需要将之前标记化的文本连接回字符串，以便在标记化器中使用，标记化器提供了有用的属性。此外，Tensorflow/Keras 需要将文档数量设为批量的倍数，以便在多个 GPU 之间分配工作，并使用某些模型，如 Elmo。

```
import mathBATCH_SIZE = 64
MAX_LEN = 200
TOKEN_COUNT = 10000
EMBED_SIZE = 50
TEST_SPLIT = 0.2# Convert label columns to numpy array
labels = posts_df[list(posts_df.columns)[1:]].to_numpy()# Training_count must be a multiple of the BATCH_SIZE times the MAX_LEN
highest_factor = math.floor(len(posts_df.index) / (BATCH_SIZE * MAX_LEN))
training_count = highest_factor * BATCH_SIZE * MAX_LEN
print(f'Highest Factor: {highest_factor:,} Training Count: {training_count:,}')# Join the previously tokenized data for tf.keras.preprocessing.text.Tokenizer to work with
documents = []
for body in posts_df[0:training_count]['_Body'].values.tolist():
  words = body.tolist()
  documents.append(' '.join(words))labels = labels[0:training_count]# Conserve RAM
del posts_df
gc.collect()# Lengths for x and y match
assert( len(documents) == training_count == labels.shape[0] )
```

您应该看到:

```
Highest Factor: 121 Training Count: 1,548,800
```

# 填充序列

数据已经被删减到每篇 200 字，但在一些文档中，使用 10K 顶级词汇的标记化将这一数据缩减到 200 字以下。如果任何文件超过 200 字，这些数据就不能正确地转换成下面的*数字矩阵。*

除了用一个键将文本转换成数字序列，Keras' *Tokenizer* 类对于通过`[keras.preprocessing.text.Tokenizer.sequences_to_texts](https://keras.io/preprocessing/text/#tokenizer)`方法生成模型的最终结果也很方便。然后，我们使用 Keras' `[keras.preprocessing.sequence.pad_sequences](https://keras.io/preprocessing/sequence/#pad_sequences)`方法并检查输出，以确保序列的长度都是 200 项，否则它们不会正确地转换为矩阵。字符串`__PAD__`以前已经被用来填充文档，所以我们在这里重用它。

```
from tf.keras.preprocessing.text import Tokenizer
from tf.keras.preprocessing.sequence import pad_sequencestokenizer = Tokenizer(
  num_words=TOKEN_COUNT,
  oov_token='__PAD__'
)
tokenizer.fit_on_texts(documents)
sequences = tokenizer.texts_to_sequences(documents)padded_sequences = pad_sequences(
  sequences,
  maxlen=MAX_LEN,
  dtype='int32',
  padding='post',
  truncating='post',
  value=1
)# Conserve RAM
del documents
del sequences
gc.collect()print( max([len(x) for x in padded_sequences]), min([len(x) for x in padded_sequences]) )
assert( min([len(x) for x in padded_sequences]) == MAX_LEN == max([len(x) for x in padded_sequences]) )padded_sequences.shape
```

# 分成测试/训练数据集

我们需要一个数据集来训练，一个单独的数据集来测试和验证我们的模型。经常使用的`[sklearn.model_selection.train_test_split](https://scikit-learn.org/stable/modules/generated/sklearn.model_selection.train_test_split.html)`使它如此。

```
from sklearn.model_selection import train_test_splitX_train, X_test, y_train, y_test = train_test_split(
  padded_sequences,
  labels,
  test_size=TEST_SPLIT,
  random_state=1337
)# Conserve RAM
del padded_sequences
del labels
gc.collect()assert(X_train.shape[0] == y_train.shape[0])
assert(X_train.shape[1] == MAX_LEN)
assert(X_test.shape[0] == y_test.shape[0]) 
assert(X_test.shape[1] == MAX_LEN)
```

# 计算类别权重

虽然已经对数据进行了过滤和上采样，以将其限制在至少有一个标签出现超过 2000 次的问题样本中，但常见标签和不常见标签之间的比率仍然不均衡。如果没有类权重，最常见的标签比最不常见的标签更有可能被预测到。类别权重将使损失函数更多地考虑不常用的类别，而不是常用的类别。

```
train_weight_vec = list(np.max(np.sum(y_train, axis=0))/np.sum(y_train, axis=0))
train_class_weights = {i: train_weight_vec[i] for i in range(y_train.shape[1])}sorted(list(train_class_weights.items()), key=lambda x: x[1])[0:10]
```

# 训练分类器模型来标记堆栈溢出帖子

现在，我们准备训练一个模型，用标签类别对问题进行分类/标记。该模型基于 [Kim-CNN](https://arxiv.org/abs/1408.5882) ，一种常用于句子和文档分类的卷积神经网络。我们使用函数式 API，并对代码进行了大量参数化处理，以便于实验。

```
from tensorflow.keras.initializers import RandomUniform
from tensorflow.keras.callbacks import ReduceLROnPlateau, EarlyStopping, ModelCheckpoint
from tensorflow.keras.layers import (
  Dense, Activation, Embedding, Flatten, MaxPool1D, GlobalMaxPool1D, Dropout, Conv1D, Input, concatenate
)
from tensorflow.keras.losses import binary_crossentropy
from tensorflow.keras.models import Model
from tensorflow.keras.optimizers import AdamFILTER_LENGTH = 300
FILTER_COUNT = 128
FILTER_SIZES = [3, 4, 5]
EPOCHS = 4
ACTIVATION = 'selu'
CONV_PADDING = 'same'
EMBED_SIZE = 50
EMBED_DROPOUT_RATIO = 0.1
CONV_DROPOUT_RATIO = 0.1
LOSS = 'binary_crossentropy'
OPTIMIZER = 'adam'
```

在 Kim-CNN 中，我们首先使用*嵌入*对序列进行编码，然后使用*去除*层以减少过度拟合。接下来，我们将图形分割成多个宽度不同的 *Conv1D* 层，每个层后面跟有 *MaxPool1D* 。它们通过串联连接在一起，旨在表征文档中不同大小序列长度的模式。接下来是另一个*Conv1D*/*globalmaxpool 1d*层来总结这些模式中最重要的模式。接下来是展平成一个*密集*层，然后到最后的*s 形*输出层。否则我们通篇使用*卢瑟*。

```
padded_input = Input(
  shape=(X_train.shape[1],), 
  dtype=’int32'
)# Create an embedding with RandomUniform initialization
emb = Embedding(
  TOKEN_COUNT, 
  EMBED_SIZE, 
  input_length=X_train.shape[1],
  embeddings_initializer=RandomUniform()
)(padded_input)
drp = Dropout(EMBED_DROPOUT_RATIO)(emb)# Create convlutions of different kernel sizes
convs = []
for filter_size in FILTER_SIZES:
  f_conv = Conv1D(
    filters=FILTER_COUNT,
    kernel_size=filter_size,
    padding=CONV_PADDING,
    activation=ACTIVATION
  )(drp)
  f_pool = MaxPool1D()(f_conv)
  convs.append(f_pool)l_merge = concatenate(convs, axis=1)
l_conv = Conv1D(
  128,
  5,
  activation=ACTIVATION
)(l_merge)
l_pool = GlobalMaxPool1D()(l_conv)
l_flat = Flatten()(l_pool)
l_drop = Dropout(CONV_DROPOUT_RATIO)(l_flat)
l_dense = Dense(
  128,
  activation=ACTIVATION
)(l_drop)
out_dense = Dense(
  y_train.shape[1],
  activation='sigmoid'
)(l_dense)model = Model(inputs=padded_input, outputs=out_dense)
```

接下来我们编译我们的模型。我们使用各种各样的度量标准，因为没有一个度量标准可以概括模型性能，我们需要深入研究真的和假的肯定和否定。我们还使用 *ReduceLROnPlateau* 、 *EarlyStopping* 和 *ModelCheckpoint* 回调来提高性能，一旦达到稳定状态，我们就提前停止，并且只保留在验证分类准确性方面最好的模型。

分类准确性是衡量我们模型性能的最佳方式，因为它为我们分类的每个类的每一行分别给出了分数。这意味着，如果我们错过了一个，但得到了其他的权利，这是一个伟大的结果。对于二进制精度，整行都被标记为不正确。

然后就是拟合模型的时候了。我们给它我们之前计算的类权重。

```
model.compile(
  optimizer=OPTIMIZER,
  loss=LOSS,
  metrics=[
    tf.keras.metrics.CategoricalAccuracy(),
    tf.keras.metrics.Precision(),
    tf.keras.metrics.Recall(),
    tf.keras.metrics.Accuracy(),
    tf.keras.metrics.TruePositives(),
    tf.keras.metrics.FalsePositives(),
    tf.keras.metrics.TrueNegatives(),
    tf.keras.metrics.FalseNegatives(),
  ]
)
model.summary()callbacks = [
  ReduceLROnPlateau(
    monitor='val_categorical_accuracy',
    factor=0.1,
    patience=1,
  ), 
  EarlyStopping(
    monitor='val_categorical_accuracy',
    patience=2
  ), 
  ModelCheckpoint(
    filepath='kim_cnn_tagger.weights.hdf5',
    monitor='val_categorical_accuracy',
    save_best_only=True
  ),
]history = model.fit(X_train, y_train,
  class_weight=train_class_weights,
  epochs=EPOCHS,
  batch_size=BATCH_SIZE,
  validation_data=(X_test, y_test),
  callbacks=callbacks
)
```

# 从训练时期加载最佳模型

因为我们使用了`ModelCheckpoint(save_only_best-True)`，就`CategoricalAccuracy`而言，最好的纪元是被保存的。我们想用它来代替上一个纪元的模型，它存储在上面的`model`中。所以我们在评估模型之前加载文件。

```
model = tf.keras.models.load_model('kim_cnn_tagger.weights.hdf5')
metrics = model.evaluate(X_test, y_test)
```

# 解析并打印最终指标

度量包括像 *precision_66* 这样的名字，它们在运行之间是不一致的。我们修复这些来清理我们关于训练模型的报告。我们还添加了一个 f1 分数，然后制作一个数据帧来显示日志。这可以在重复实验中推广。

```
def fix_metric_name(name):
  """Remove the trailing _NN, ex. precision_86"""
  if name[-1].isdigit():
    repeat_name = '_'.join(name.split('_')[:-1])
  else:
    repeat_name = name
    return repeat_namedef fix_value(val):
  """Convert from numpy to float"""
  return val.item() if isinstance(val, np.float32) else valdef fix_metric(name, val):
  repeat_name = fix_metric_name(name)
  py_val = fix_value(val)
  return repeat_name, py_vallog = {}
for name, val in zip(model.metrics_names, metrics):
  repeat_name, py_val = fix_metric(name, val)
  log[repeat_name] = py_val
log.update({'f1': (log['precision'] * log['recall']) / (log['precision'] + log['recall'])})pd.DataFrame([log])
```

# 绘制历元精度

我们想知道每个时期的性能，这样我们就不会不必要地训练大量的时期。

```
%matplotlib inlinenew_history = {}
for key, metrics in history.history.items():
  new_history[fix_metric_name(key)] = metricsimport matplotlib.pyplot as pltviz_keys = ['val_categorical_accuracy', 'val_precision', 'val_recall']# summarize history for accuracy
for key in viz_keys:
  plt.plot(new_history[key])
plt.title('model accuracy')
plt.ylabel('metric')
plt.xlabel('epoch')
plt.legend(viz_keys, loc='upper left')
plt.show()# summarize history for loss
plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()
```

![](img/7179bef5da3cffe17b5e364928f1c85c.png)

Categorical Accuracy, Precision and Recall Across Epochs. Note that recall climbs even on the 8th Epoch

![](img/a5f8cc58786cd9895063df0612b90487.png)

Test / Train Loss

# 检查实际预测输出

光知道理论性能是不够的。我们需要看到标记者在不同置信度阈值下的实际输出。

```
TEST_COUNT = 1000X_test_text = tokenizer.sequences_to_texts(X_test[:TEST_COUNT])y_test_tags = []
for row in y_test[:TEST_COUNT].tolist():
   tags = [index_tag[str(i)] for i, col in enumerate(row) if col == 1]
   y_test_tags.append(tags)CLASSIFY_THRESHOLD = 0.5y_pred = model.predict(X_test[:TEST_COUNT])
y_pred = (y_pred > CLASSIFY_THRESHOLD) * 1y_pred_tags = []
for row in y_pred.tolist():
  tags = [index_tag[str(i)] for i, col in enumerate(row) if col > CLASSIFY_THRESHOLD]
  y_pred_tags.append(tags)
```

让我们来看看在*数据帧*中带有实际标签和预测标签的句子:

```
prediction_tests = []
for x, y, z in zip(X_test_text, y_pred_tags, y_test_tags):
  prediction_tests.append({
    'Question': x,
    'Predictions': ' '.join(sorted(y)),
    'Actual Tags': ' '.join(sorted(z)),
 })pd.DataFrame(prediction_tests)
```

我们可以从这三个记录中看出，该模型做得相当不错。这讲述了一个不同于单独的性能指标的故事。奇怪的是，大多数机器学习的例子只是计算性能，并没有真正使用“predict()”方法！归根结底，统计性能是无关紧要的，重要的是真实世界的性能——它不包含在简单的汇总统计中！

![](img/30af2e74680e31a1cca8fe700c4a39d6.png)

Questions along with Actual Labels and Predicted Labels

# 自动化 DC/操作系统数据科学引擎设置

这涵盖了如何手动使用平台，但这是关于 PaaS 自动化。那么我们如何加快速度呢？

DC/操作系统的图形用户界面和 CLI 共同支持各类用户通过数据科学引擎轻松访问 JupyterLab:试图在笔记本上查看报告的非技术管理人员和希望自动化流程的开发运营/数据工程师。如果手动 GUI 过程似乎很复杂，我们可以在将服务配置作为 JSON 文件后，通过几行代码将其自动化，方法是通过 Terraform 命令启动 DC/操作系统集群，从 Terraform 获取集群地址，然后使用 DC/操作系统 CLI 验证集群并运行服务。

*注:查看 DC/OS CLI 的* [*Github 页面*](https://github.com/dcos/dcos-cli) *了解更多关于它如何工作的信息。*

# 安装 DC/操作系统命令行界面

使用 Terraform 启动集群并在用户界面中手动安装 Data Science 引擎时，不需要 DC/操作系统 CLI，但需要它来自动执行该过程。如果您遇到了麻烦，请查看 [CLI 安装文档](https://docs.d2iq.com/mesosphere/dcos/1.12/cli/install/)。你将需要[卷发](https://curl.haxx.se/)。

```
# Optional: make a /usr/local/bin if it isn’t there. Otherwise change the install path.
[ -d usr/local/bin ] || sudo mkdir -p /usr/local/bin# Download the executable
curl [https://downloads.dcos.io/binaries/cli/linux/x86-64/dcos-1.13/dcos](https://downloads.dcos.io/binaries/cli/linux/x86-64/dcos-1.13/dcos) -o dcos
```

请注意，您也可以使用打印的命令下载 CLI，方法是单击右上方标题为 *jupyter-gpu-xxxx* (或您为集群命名的任何名称)的下拉菜单，然后单击下方橙色框中的 Install CLI 按钮。

![](img/ef7b698bc22804f577613924522538b9.png)

Note the **Install CLI** button in the dropdown at top right

将出现一个弹出窗口，显示 Windows、OS X 和 Linux 的安装代码。将代码复制/粘贴到终端以完成安装。

![](img/57f1383ae71853e3a01b13612bc1e5c7.png)

Install Commands for DC/OS CLI

现在，如果您运行`dcos`，您应该会看到以下内容:

```
Usage:
  dcos [command]Commands:
  auth
  Authenticate to DC/OS cluster
  cluster
  Manage your DC/OS clusters
  config
  Manage the DC/OS configuration file
  help
  Help about any command
  plugin
  Manage CLI pluginsOptions:
  — version
  Print version information
  -v, -vv
  Output verbosity (verbose or very verbose)
  -h, — help
  Show usage helpUse "dcos [command] — help" for more information about a command.
```

# 使用 GUI 手动导出软件包配置

为了自动安装数据科学引擎包，我们首先需要以 JSON 文件的形式获取它的配置。这可以通过两种方式完成:通过 GUI 或通过 DC/操作系统 CLI 或通过 Terraform CLI。我们将介绍这两种方法。

![](img/66e7b301d3b799e77a00742e034f9fff.png)

Note the **Download Config** button below the dividing line at the top right

使用 GUI，我们可以通过从*数据-科学-引擎*服务页面的*配置*选项卡下载 JSON 来获得包配置。点击该行右上方紫色的*下载配置*。

顺便说一下，您可以通过点击右上角的紫色*编辑*按钮来编辑数据科学引擎服务。您可以使用菜单或 JSON 编辑器来更改服务——例如增加 RAM 或 GPUs 它会根据该配置自动重新部署。

![](img/7b81e8bac55fee3f9adb8be7de2e0a35.png)

Editing the Data Science Engine Service Config

# 向群集验证 DC/操作系统 CLI

要使用 CLI，我们需要向群集进行身份验证。这可以通过 Google、用户名或其他适合您组织的方法来完成。为此，首先我们需要来自 Terraform 的服务器地址。我们可以通过`terraform output -json`和 [*jq*](https://stedolan.github.io/jq/) json 实用程序提取集群主机 IP 地址。一旦我们有了这些，我们可以使用任何我们喜欢的方法，包括谷歌，来进行认证。这也可以是便于自动化的用户名。

```
# Get the cluster address from Terraform’s JSON output and Authenticate CLI to Cluster
export CLUSTER_ADDRESS=`terraform output -json | jq -r '.["masters-ips"].value[0]'`
dcos cluster setup [http://$CLUSTER_ADDRESS](/$CLUSTER_ADDRESS) # add whatever arguments you need for automated authentication
```

查看`dcos cluster setup`的[文档](https://docs.d2iq.com/mesosphere/dcos/1.13/cli/command-reference/dcos-cluster/dcos-cluster-setup/)，了解不同认证方法的信息。

# 使用 CLI 导出程序包配置

一旦我们通过了身份验证，我们就可以使用 CLI 来生成包配置文件，供以后重用。

对于麦克·OS X 来说，指挥旗`base64`是一个大写字母`-D`:

```
dcos marathon app show data-science-engine | jq -r .labels.DCOS_PACKAGE_OPTIONS | base64 -D | jq > my-data-science-engine.json
```

对于 Linux，`base64`的标志是小写的`-d`:

```
dcos marathon app show data-science-engine | jq -r .labels.DCOS_PACKAGE_OPTIONS | base64 -d | jq > my-data-science-engine.json
```

请注意，导出的选项涵盖了每一个选项，这并不理想，因为选项可能会在平台的不同版本之间发生变化，如果您不更改值，最好依赖系统默认值。例如，我将该导出编辑为:

```
{
  "service": {
    "name": "data-science-engine",
    "cpus": 8,
    "mem": 51200,
    "gpu": {
      "enabled": true,
      "gpus": 1
      }
   }
}
```

您可以编辑该文件以满足您的需要，或者使用 GUI 来完成，并下载和编辑配置。

# 使用 CLI 安装数据科学引擎

现在，我们可以使用 CLI 安装数据科学引擎包:

```
dcos package install data-science-engine --options=./data-science-engine.json --yes 
```

# 完整的集群/服务自动化示例

总之，这使得整个过程使用导出的 JSON 包配置:

```
# Boot DC/OS Cluster
terraform init -upgrade
terraform plan -var-file desired_cluster_profile.tfvars -out plan.out
terraform apply plan.out# Get the cluster address from Terraform’s JSON output
export CLUSTER_ADDRESS = `terraform output -json | jq -r '.["masters-ips"].value[0]'`# Authenticate CLI to Cluster using its address and Install the Data Science Engine Package
dcos cluster setup [http://$CLUSTER_ADDRESS](/$CLUSTER_ADDRESS) # add whatever arguments you need for automated authentication
dcos package install data-science-engine --options=data-science-engine-options.json
```

# 结论

在这篇文章中，我们启动了一个 DC/操作系统集群，并以可重复的方式部署了数据科学引擎，然后执行了一个测试笔记本来创建一个堆栈溢出标记。这展示了如何使用 PaaS 来提高数据科学团队的生产力。请继续关注该系列的下一篇文章！

*注意:教程中的每一步都经过严格测试，所以如果你遇到任何问题，请在评论中告诉我。*

# 附录:常见错误

我在写这篇文章的时候遇到了一些错误，你可能会在这篇文章中的例子中遇到，所以我已经包括了这些错误以及如何修复它们！:)

# `ssh-agent`问题

如果您忘记通过`eval "$(ssh-agent -s)"`和`ssh-add`您的密钥运行 ssh-agent，您将看到下面的错误。如果您打开一个新的 shell 并从中运行 terraform 命令，这将非常容易。

```
Error: Error applying plan:1 error occurred:
  * module.dcos.module.dcos-install.module.dcos-install.null_resource.run_ansible_from_bootstrap_node_to_install_dcos: timeout — last error: ssh: handshake failed: ssh: unable to authenticate, attempted methods [none publickey], no supported methods remain
```
```

解决方案是来自 *paas_blog/dcos/terraform* 目录的`ssh-add ./my_key`。为了避免将来出现这种情况，您可以编辑*desired _ cluster _ profile . TF vars*中的密钥字段，以便在您的 *~ /中使用公钥。ssh/* 目录，当 shell 启动时，使用 *~/自动添加到 ssh-agent 中。简介*、 *~/。bash_profile* 或者 *~/。bashrc* 。

现在，验证是否已经添加了密钥:

```
ssh-add -l
```

这应该显示:

```
2048 SHA256:1234567891234567891234567891234567891234567 ./my_key (RSA)
```

# 超时问题

有时，当 DC/操作系统启动时，事情不会正确同步，您会得到一个关于 Ansible bootstrap 或负载平衡器超时的错误。要做的事情就是破坏，然后计划/应用再创造。第二次会成功的。

如果您看到以下任一错误，这就是正在发生的情况。

```
Error: Error applying plan:1 error occurred:
 * module.dcos.module.dcos-infrastructure.module.dcos-lb.module.dcos-lb-public-agents.module.public-agents.aws_lb.loadbalancer: 1 error occurred:
 * aws_lb.loadbalancer: timeout while waiting for state to become 'active' (last state: ‘provisioning’, timeout: 10m0s)
```

或者:

```
Error: Error applying plan:1 error occurred:
 * module.dcos.module.dcos-install.module.dcos-install.null_resource.run_ansible_from_bootstrap_node_to_install_dcos: timeout — last error: dial tcp 34.221.233.243:22: i/o timeout
```

# 数据综合症

我叫拉塞尔·朱尼。我是 [Data Syndrome](http://datasyndrome.com) 的首席数据科学家，在那里我构建端到端的机器学习和可视化产品、推荐系统、线索生成系统、数据工程，并专注于[弱监督学习](https://www.snorkel.org/blog/weak-supervision)，或用更少的数据做更多的事情。我在 http://blog.datasyndrome.com[写博客](http://blog.datasyndrome.com)

我正在写一本将于明年出版的新书，书名是《T21:弱监督学习》。敬请关注更多内容！