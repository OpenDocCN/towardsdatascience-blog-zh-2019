# 在 Kubernetes 上提供 Vowpal Wabbit 模型

> 原文：<https://towardsdatascience.com/serving-vowpal-wabbit-models-on-kubernetes-de0ed968f3b0?source=collection_archive---------16----------------------->

![](img/dda6a8dd9266de69e58598dfbdfe4d58.png)

Vowpal Wabbit Logo

在这篇文章中，我将展示一个构建 Kubernetes 服务的例子，将使用没有“本地”服务 API 的库训练的机器学习模型投入生产。我选择了 Vowpal Wabbit，因为虽然它不像其他机器学习框架那样受欢迎，但它通常会提供非常好的开箱即用的结果，并且在数据准备、优化器选择和超参数调整方面需要最少的努力。我个人认为，学习 vow pal Wabbit T1(VW)是一个(初级)数据科学家的良好起点，因为初学者可以轻松、快速、迭代地在相当大的数据集上构建原型。另一方面，从机器学习工程师的角度来看，服务于用 VW 训练的模型是一项有趣的任务，因为该库提供的接口很少，并且没有官方的服务 API。

在这篇文章中，我将带你了解这项服务的发展。有几个 GitHub gists，你可以在这里找到完整的库。

## **推出 VowpalWabbit**

VW 是微软研究院赞助的快速核外机器学习系统。第一个公开版本出现在 2007 年，多年来已经增加了许多功能和改进。大众的设计是学术研究和软件开发之间良性循环的一个例子，论文中出现的算法被纳入库中，框架的速度和性能可以催生进一步的学术研究。

为了更具体地说明 VW 如何工作，让我们考虑一个预测任务，其中我们想要建立一个线性模型 *y* = X𝛃，其中 *y* 是要预测的变量， *X* 是特征向量，𝛃是模型参数的向量。

第一个关键想法是大众如何使用所谓的[散列技巧](http://hunch.net/~jl/projects/hash_reps/index.html)来表示特征。为了简单起见，首先考虑分类特征 **𝓕** 的情况，其可以取许多可能的值，例如在 *(LAX，AMS)* 中的对(出发机场，到达机场)。大众不需要用户预先指定所有可能的级别，或者让它知道𝓕 可以假设的不同值的数量；用户需要做的是指定特征的总最大数量𝓓，即𝛃.的行数在幕后，大众将使用一个[散列函数](https://en.wikipedia.org/wiki/Hash_function)将所有特征存储到𝓓值中。具体来说，在 **𝓕** = *(LAX，AMS)* 的情况下，VW 会将字符串“ **𝓕** = *(LHR，AMS)”*哈希成一个介于 0 和𝓓 *-1* 之间的整数 *i* ，并设置𝛃( *i* ) = 1 ( [一热编码](https://hackernoon.com/what-is-one-hot-encoding-why-and-when-do-you-have-to-use-it-e3c6186d008f))。在数字特征的情况下，比如说 **𝓕** = 2.5，整数 *i* 将通过散列字符串“ **𝓕** ”来计算，然后大众将设置𝛃( *i* ) = 2.5。

第二个关键思想是大众使用在线梯度下降(一种在线学习的形式)来优化模型权重𝛃；具体来说，**不需要将数据保存在存储器**中，可以一次处理一个例子；这使得 VW 能够在非常大的数据集上很好地扩展，因为内存需求现在基本上被绑定到为𝛃.分配内存例如，您可以通过在网络上传输数据，使用 Hadoop 集群中存储的万亿字节数据在您的笔记本电脑上训练一个大众模型。此外，小心实施梯度下降以处理不同重要性的[特征，并降低算法对学习率(学习率衰减)精确设置的敏感性。](https://arxiv.org/abs/1011.1576)

在进入下一步之前，让我们先看一下大众要求的数据格式。我首先以我们将用于服务的 JSON 格式呈现它:

Training data: JSON format

这里我们处理一个分类任务，因此*目标*可以是 *0* 或*1；*标签*为示例的**可选**唯一标识；特性被组织在**名称空间**中，其中每个名称空间都有自己的哈希函数，减少了哈希冲突的可能性；例如，在名称空间 *a* 中，我们发现一个分类特征 *x1* 取值 *a* 和一个数值特征 *x2* 取值 *0.814* 。在 VW 中接收这个示例的方法是将所有内容放在同一个字符串中，使用“*|”*字符分隔名称空间:*

Training data: VW native format

注意分类特征如何使用“*=”*字符，数字特征如何使用“*:*字符；进一步的细节可以在这里找到[。](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Input-format)

## **分解问题和设计考虑**

让我们首先集思广益，为大众服务。我很快想到了两个解决方案:

1.  *哈希反转* : VW 提供了以人类可读格式存储模型权重𝛃的可能性；通过额外的工作，VW 还可以对哈希进行反转，从而生成训练示例中出现的原始特征的𝛃索引，例如上面 gists 中的 *x2* 、 *x1* 、…。通过这种方式，人们可以获得模型权重的文本文件，并使用任何可以对线性模型进行评分的函数(例如，参见此处[采用的为大众模型构建 TensorFlow 包装器的方法](https://github.com/salayatana66/vw_utilities/tree/master/tensorflow_wrapper))。
2.  *Java 原生接口* ( [JNI](https://en.wikipedia.org/wiki/Java_Native_Interface) ): [这个项目](https://github.com/indeedeng/vowpal-wabbit-java/tree/master/build-jni)旨在提供一个 API 从 Java 调用 VW；因此，这可以用来将 VW 集成到任何基于 Java 的服务库中。

稍加思考，我们就可以很容易地发现这两种方法都有重大缺陷，并且本质上都不令人满意:

1.  哈希反转是一种开销很大的操作；它需要进一步遍历所有的训练数据集，并且不能很好地扩展大量的特征。此外，它占用了数据科学家的操作工作时间，而这些时间本可以更好地用于创建和改进算法。此外，在提供权重时总是可能存在翻译错误，这可能导致生产中的模型不同于经过训练的模型。
2.  在撰写本文时，大众 Java 项目似乎并不活跃，而且落后于大众的最新版本。此外，使用 JNI [可能会带来一些风险](https://www.ibm.com/developerworks/library/j-jni/index.html),即引入一个可能导致产品中的模型崩溃的错误。

因此，我们的目标是 1)以尽可能接近原生大众环境的方式提供模型，2)易于扩展。

[码头工人](https://www.docker.com/)将允许满足 1)；Docker 是一个在操作系统级别执行虚拟化的软件。这意味着我们可以在操作系统上构建应用程序，同时隔离它所需的特定库和依赖项，也就是说，我们不需要在操作系统级别安装它们。该应用程序将在它自己的**容器**中运行，这将给人一种独立虚拟机的错觉。具体来说，我们将围绕 VW shell 命令构建一个 **Python 包装器**，它将模型加载到容器的内存中，并在预测请求到来时对其进行评分。请注意，这种方法可以很容易地推广到其他机器学习库*加以必要的修改*。

[Kubernetes](https://kubernetes.io/) 将允许通过扩展集群上的容器来满足 2)并将模型公开为服务。Kubernetes 是容器编排系统的一个例子，它在过去几年中获得了相当大的流行。顺便提一下，我一直认为这个名字意味着类似“立方体”的东西，因为容器通常被描绘成在机器内部运行的小立方体；然而，事实证明它来源于希腊语，意思是船长。

以下是后续步骤的分类:

*   创建 Docker 映像的原型，以在“本地”环境中隔离大众
*   构建一个操场测试模型，以更熟悉大众，并对最终结果进行健全性检查
*   创建一个 Python 包装器来为模型服务
*   开发一个 Flask 应用程序来处理评分请求
*   在 Kubernetes 上将模型部署为服务

这里值得注意的遗漏是:

*   监控培训和生产环境之间的功能对等性。
*   监控需要在生产中加速运行的请求和实例的数量。

理想情况下，最后两个问题的解决方案应该是独立的服务，这些服务可以使用公共 API 与每个单独的模型服务进行对话。

## 在 Docker 上开发

第一个概念验证是构建一个能够运行大众的 docker 映像。如果我们从一个 Ubuntu 镜像开始，这很容易，因为 VW 已经作为一个包被维护了:

Prototype Docker Image

使用 *apt-get* 我们安装 Python 3.6 和 VW；然后我们在 [*requirements.txt*](https://github.com/salayatana66/vw-serving-flask/blob/master/docker/requirements.txt) 中安装附加包；然后，当我们将在 Kubernetes 上部署时，我们创建了一个用户 *vwserver* ，由于安全限制，我们可能无法在 Kubernetes 集群上以 **root** 身份运行容器。最后，我们创建一个挂载点目录 *vw_models* 来 1)存储生产中的模型，2)允许我们在开发时在笔记本电脑和容器之间交换脚本和代码。

您可以从标签为***andrejschioppa/VW _ serving _ flask:tryubuntu***的 [DockerHub](https://hub.docker.com) 中提取此图像；如果你更喜欢 CentOS 图像，我也提供***andrejschioppa/VW _ serving _ flask:trycentos***。我想指出的是，对于 CentOS，我必须从其原始库编译 VW，由于过时的 GCC / g++编译器， [Dockerfile](https://github.com/salayatana66/vw-serving-flask/blob/master/docker/DockerfileCentos_tryvw) 变得更加棘手。

## 创建操场测试模型

下一步，我们将更加熟悉 VW，并创建一个模型，用于测试我们的服务应用程序的正确性。让我们初始化一个本地卷目录并启动容器；然后我们将 *numpy* 安装到容器中:

Running the prototype container

我们现在创建一个测试模型；我们有这个文件[*test _ model . py*](https://github.com/salayatana66/vw-serving-flask/blob/master/fake_model/test_model.py)*坐在 *localVolume* :*

*Training data generator*

*这个模型是一个简单的线性模型，有一个数字特征 *x2* 和三个分类特征 *x1，x2，x3*；模型是完全确定的:如果分数是 *> 0* 我们预测正标签 *1* ，否则负标签 *0* 。*

*我们现在生成训练示例，并将模型训练到完美拟合(毕竟这里的一切都是确定的，因此我们应该期待一个完美的分类器；返回的概率可能不是[精确校准的](https://scikit-learn.org/stable/modules/calibration.html)):*

*Training a VW model*

*我们看到，正面例子的概率为 73.11% ，负面例子的概率为 50%*。在这些 *10* 的例子中，模型看起来像一个**完美的分类器**，尽管概率可能**没有被校准**。我们将保留文件 *test_data.json、test_data.vw* 和 *test_model.model* 以备将来测试之用。**

## *用于提供预测的 Python 模块*

*我写了一个类 *VWModel* 来服务 Python 中的 VW 模型。让我们来看看构造函数:*

*constructor of VWModel*

*关键思想是启动一个 shell 进程，运行 VW 并通过 Python 控制[标准流](https://en.wikipedia.org/wiki/Standard_streams) : *标准输入*，*标准输出*和*标准错误*。第一步是获得正确的 *vw* 命令(第 26 行)；注意，我们首先停止训练更新(选项*仅测试*)，即我们不再更新权重𝛃；其次，我们通过选项 *initial_regressor* 设置用于评分的模型，并将预测重定向到 *stdout* 。选项*链接*(第 32 行)允许在广义线性模型(如使用逻辑链接的二元分类器)的情况下指定一个链接函数，如果在训练期间使用了特征[交互](https://github.com/VowpalWabbit/vowpal_wabbit/wiki/Feature-interactions)，选项 *sort_features* 强制 VW 对特征进行分类。*

*现在让我们来看看模型实际上是如何启动的:*

*How VWModel starts a model*

*这里，我们只是用上面构建的命令启动一个 shell 子进程，并通过*子进程控制标准流。管道*。关于处理标准流和管道的精彩阅读请看[这个精彩的系列](https://lyceum-allotments.github.io/2017/03/python-and-pipes/)。*

*最后让我们看看实际的评分函数:*

*How VWModel handles a prediction request*

*我们使用一个静态方法， *parse_example* ，将输入 json 转换成 VW 输入格式的字符串。然后，我们将字符串刷新到正在运行的进程的 *stdin* 中，并收集 *stdout* 来提取预测。*

*完整的 Python 模块请看这里的；请注意，在*测试*目录中，您可以找到一个基于示例和我们上面构建的模型的单元测试；注意这个单元测试意味着在将 *vw_model* 模块添加到 *PYTHONPATH* 之后在容器内部运行。*

## *烧瓶应用程序*

*为了处理模型服务的预测请求，我们将开发一个简单的 [Flask](http://flask.pocoo.org/) 应用程序。Flask 是一个非常流行的用于开发 web 应用程序的 Python 微框架。服务应用程序变得非常简单:*

*The base Flask application*

*一个服务请求将通过对 */serve/* 的 POST 请求使用[REST API](https://en.wikipedia.org/wiki/Representational_state_transfer)；具体地说，这相当于将 JSON 格式的特性发送到 */serve/* URL。上面的代码所做的只是将预测请求转发给一个名为 *FlaskServer* 的对象，我们可以在这里检查它:*

*The interface between Flask application and VWModel*

*这个 FlaskServer 类只是提供了一个到 *VWModel* 的接口；有两个关键的环境变量与容器相关联:*

*   **模型文件*，它指向我们想要服务的模型文件*
*   **模型 _CONF* ，它指向一个配置文件，该文件目前支持链接规范和选项来分类特征。*

*让我们把所有东西放在一起，在一个容器中测试 Flask 应用程序。您可以在这里找到最终的 docker 文件:*

*   *CentOS:tag*T25[andrejschioppa/VW _ serving _ flask:CentOS _ v1](https://hub.docker.com/r/andrejschioppa/vw_serving_flask/tags)**
*   *Ubuntu:tag[*andrejschioppa/VW _ serving _ flask:Ubuntu _ v1*](https://hub.docker.com/r/andrejschioppa/vw_serving_flask/tags)*

*让我们提取映像并将测试模型和数据复制到*本地卷*:*

*preparing to run a test*

*让我们从一个具有相关配置的容器开始:*

*starting a container for testing*

*现在，我们在笔记本电脑中打开另一个 shell，检查预测与测试数据是否一致:*

*the test*

*一切看起来都很好，我们开始在 Kubernetes 上部署。*

## *在 Kubernetes 部署*

*最后一步，我们将在 Kubernetes 上部署该模型。这里的基本概念是:*

*   *[**持久卷**](https://kubernetes.io/docs/concepts/storage/persistent-volumes/) :这是一个我们存储模型的卷；这可以在服务于相同模型的容器之间共享；这样，我们只需要上传模型一次。在 Kubernetes 的术语中，具有共享网络、存储和如何运行它们的指令的一个或多个容器的组被称为 [**pod**](https://kubernetes.io/docs/concepts/workloads/pods/pod/) 。在我们的讨论中，pod 与容器是一样的，因为我们只需要一个容器来为模型服务。*
*   *[**部署**](https://kubernetes.io/docs/concepts/workloads/controllers/deployment/) :这控制给定种类的多少个 pod 正在运行；它确保达到目标吊舱数量，如果某个吊舱崩溃，它会被一个新的吊舱所取代。通过这种方式，可以以可靠的方式扩展服务模型，因为我们可以确保始终有足够的 pod 来处理预测请求。*
*   *[**服务**](https://kubernetes.io/docs/concepts/services-networking/service/) :我们将使用它使我们部署中的 pod 共享相同的 IP 地址；通过这种方式，预测请求只发送到一个 IP 地址，而不必担心负载如何在 pod 之间平衡。*

*Kubernetes 可以在配置了不同配置的集群上运行。对于这篇文章，我将使用一个本地集群，即我的笔记本电脑上的集群。这可以通过使用 [**minikube**](https://github.com/kubernetes/minikube) 来实现，这是一个允许在本地运行 Kubernetes 的工具。安装 minikube 并不总是一个简单的过程，请参考[文档](https://kubernetes.io/docs/setup/minikube/)。在我的例子中，我使用的是 Linux 发行版，所以我能够运行 minikube 而不需要安装[管理程序](https://en.wikipedia.org/wiki/Hypervisor)，但是如果你使用的是 MacOS，你就需要安装一个。[这里](https://github.com/salayatana66/vw-serving-flask/tree/master/kubernetes)你可以用*。我将使用的 yaml* 配置文件。*

*让我们启动 Kubernetes 集群并创建一个目录 */mnt/data* ，它将作为我们的持久卷。*

*starting minikube*

*然后，我们为永久卷和永久卷声明创建配置文件:*

*creating a persistent volume*

*claiming a persistent volume*

*我们现在可以创建体积和体积索赔*

*persistent volumes on the cluster*

*为了复制模型和配置文件，我们将使用一个“假部署”,它只是挂载持久卷。*

*a fake deployment to access the persistent volume*

*下一步是识别运行假部署的 pod，并复制模型和配置文件；我们终于可以登录到 pod 中检查一切是否正常:*

*copying the model in the persistent volume*

*部署只是运行具有所需环境变量的模型服务容器，这些环境变量指定了模型文件、模型配置和服务端口(在本例中为*6025*)；副本的数量指定了要保持运行多少个容器来分配负载:*

*the application deployment*

*该服务将 pod 包装在同一个 IP 地址下；这样，我们就不需要担心如何在副本之间平衡请求:*

*the service*

*我们最终可以部署豆荚和服务；然后定位服务的 IP，并通过提交预测请求来运行测试:*

*deploying and testing*

## *结论*

*通过一点点努力和设计考虑，我们已经将在没有本地服务 API 的库中训练的生产模型。*

*如果你面临一个类似的任务，我建议不要急于寻找快速的解决方案，不要在纸上探索设计思想及其结果；给予一些思考和反思通常可以避免走进死胡同。在这种情况下，理想的解决方案非常简单:只需使用一个 shell 进程，并通过 Python 控制它。*

*我非常感谢 [Antonio Castelli](https://medium.com/@antonio.castelli) 让我注意到 JNI 的问题，并测试了这个库的早期原型。*

*我将非常感谢对 GitHub 代码的改进/扩展的评论、建议和批评以及合并请求。*