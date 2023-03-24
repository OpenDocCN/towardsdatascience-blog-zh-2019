# 在 AWS 上自动化机器学习模型

> 原文：<https://towardsdatascience.com/automating-machine-learning-models-on-aws-bfa183fe4065?source=collection_archive---------13----------------------->

## 使用自动气象站 Lambda、S3 和 EC2

![](img/1f8c7f36deb333852937c5e6abfcec41.png)

Photo by [Samuel Zeller](https://unsplash.com/@samuelzeller?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

作为马里兰大学 Gordon Gao 教授手下的一名研究助理，我有机会将我的数据工程和科学兴趣结合起来，在云中实现机器学习模型的自动化。在协助高教授的一位博士生时，我的任务是提供一种基于 AWS 的解决方案，这将减少在为一家即将成立的健康初创公司运行深度学习模型时的人工干预。

# 问题定义

我们的客户将问题定义为

> “通过仅运行 EC2 实例进行计算，降低 EC2 实例的成本。只要客户将数据上传到 S3 存储桶，这些计算就会发生，这可能发生在一天中的任何时间。”

提到的计算包括机器学习模型和处理上传到 S3 的数据。这意味着，EC2 应该仅在执行 ML 模型时运行，而在其他时间应该关闭，而且这些作业没有固定的时间，它们拥有的唯一固定属性是它们必须在数据上传到 S3 数据桶后立即运行。我用下面的算法形成了一个粗略的工作流程，这个流程后来帮助我简化了整个自动化过程。

> *1。将数据上传到 S3 数据桶，触发 Lambda 函数。*
> 
> *2。触发器 Lambda 函数打开 EC2 实例并调用 Worker Lambda 函数。*
> 
> *3。Worker Lambda 函数从 S3 数据桶中复制数据，并开始执行机器学习模型。*
> 
> *3。EC2 实例执行 ML 模型并将输出发送到 S3 输出桶。*
> 
> *4。输出文件被放入 S3 输出桶，该桶执行 Stop EC2 Lambda 函数。*
> 
> *5。停止 EC2 Lambda 功能关闭 EC2 实例。*

# 可视化管道

在我担任高教授的助教期间，我曾参与过 EC2 和的课堂作业。然而，Lambda 没有，这些是学术设置，我没有太多的限制。现在，我正在与一个客户打交道，这带来了安全的重要方面，并增加了在截止日期前工作的元素。这一点，再加上我不得不在项目的大部分繁重工作中使用 Lambda 函数，对我来说更加困难。

![](img/38d9dc5c1a3a3ea8a4d5cdbfb7c4575b.png)

Automation Workflow Visualized

这种自动化确保了 EC2 实例仅在处理数据时被实例化，并在输出被写入 S3 存储桶时被关闭。

我从设置我的弹性云计算实例开始。我用 Python 创建了一个虚拟环境来加载库，这是运行 worker.py 和 trigger.py 脚本以及 ML 模型所需的包所必需的。我还向 EC2 实例添加了一个标记，因为这将帮助我在配置 Lambda 函数时识别实例。一旦 EC2 实例被设置好，我就开始研究 Lambda 函数。触发器函数负责两项任务:当数据上传到 S3 数据桶时启动 EC2 实例，然后在 EC2 启动后调用 Worker Lambda 函数。

Trigger Lambda Function

我使用 boto3 库来调用各种 AWS APIs，例如启动 EC2 实例，调用 Worker Lambda 函数，并将 EC2 的 IP 传递给它。

接下来，我开始开发 Worker Lambda 函数，并将它编程为将数据从 S3 数据桶复制到 EC2，然后开始执行机器学习模型。

Worker Lambda Function

从上面的函数可以看出，我用了我的私钥(。pem)登录到 EC2，然后执行命令以启动从数据桶到 EC2 的 S3 复制，然后对数据调用机器学习 Python 脚本，并将生成的结果复制到 S3 输出桶。

此外，我编写了 Stop EC2 Lambda 函数，该函数将在机器学习模型的输出被复制到 S3 输出桶时启动，该 Lambda 函数将调用 EC2 上的 stop API，确保 EC2 仅在机器学习模型的持续时间内运行。

Stop EC2 Lambda Function

# 将所有东西整合在一起

![](img/7db290c735dbbf301842d11c3c6029b3.png)

Photo by [Perry Grone](https://unsplash.com/@perrygrone?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

我已经准备好了单独的 Lambda 函数，但是让它们一起工作，当一个函数完成时启动另一个函数是需要完成的不同任务。最初，我开始使用 AWS CloudWatch，并试图用它来触发不同的 Lambda 函数，设置 cron 类计时并使用警报来执行函数。虽然这些看起来可行，但我们的客户觉得我增加了整个工作流程的复杂性，希望我把事情变得简单。这些限制促使我使用现有服务寻找解决方案，这就是我遇到 S3 事件的原因。

S3 事件是可以在 S3 存储桶的属性选项卡下找到的功能，它允许您在 S3 存储桶中执行某些操作时安排事件。经过进一步研究，我能够为我的两个 Lambda 函数设置事件，允许我链接管道并完成自动化。

# 决赛成绩

![](img/4bdfc2190beb4b7b3b10e34767d3b7b5.png)

Photo by [NeONBRAND](https://unsplash.com/@neonbrand?utm_source=medium&utm_medium=referral) on [Unsplash](https://unsplash.com?utm_source=medium&utm_medium=referral)

工作流现在更容易理解和设置，S3 数据桶中的初始 PUT 请求将触发第一个 S3 事件，该事件旨在调用 Trigger Lambda 函数。Trigger 函数打开 EC2，然后调用 Worker Lambda 函数。如前所述，Worker 函数将从数据桶中复制上传的数据，并对数据运行深度学习模型。一旦计算完成，Worker 函数就会将结果复制到 S3 输出桶中，触发下一个调用 Stop EC2 Lambda 函数并停止 EC2 的事件。

这确保了深度学习模型现在是自动化的，并且 EC2 将仅在需要时运行，而在所有其他时间被关闭。

> 因此，我已经成功地减少了 EC2 实例的运行时间，以前，我的客户机必须从上午 9 点到下午 5 点运行 EC2，用于一天运行三四次并且每次运行不超过 20 分钟的作业。在我实现流程自动化后，客户的运行时间从 8 小时减少到了 1 小时，从而使客户在 AWS 服务上减少了大约 90%的成本。

编辑[卡纳克亚达夫](https://medium.com/u/e4a60d4b1fc1?source=post_page-----bfa183fe4065--------------------------------)

# 参考链接

[](https://aws.amazon.com/premiumsupport/knowledge-center/start-stop-lambda-cloudwatch/) [## 使用 Lambda 和 CloudWatch 以预定的时间间隔停止和启动 EC2 实例

### 我想通过停止和启动我的 EC2 实例来减少我的 Amazon 弹性计算云(Amazon EC2)的使用…

aws.amazon.com](https://aws.amazon.com/premiumsupport/knowledge-center/start-stop-lambda-cloudwatch/)