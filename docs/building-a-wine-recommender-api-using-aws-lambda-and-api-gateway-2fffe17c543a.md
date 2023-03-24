# 使用 AWS Lambda 和 API Gateway 构建葡萄酒推荐 API

> 原文：<https://towardsdatascience.com/building-a-wine-recommender-api-using-aws-lambda-and-api-gateway-2fffe17c543a?source=collection_archive---------31----------------------->

在 RoboSomm 系列的一个章节中，我们构建了一个[葡萄酒推荐模式](/robosomm-chapter-3-wine-embeddings-and-a-wine-recommender-9fc678f1041e?source=friends_link&sk=2823c32030f1f56594b8128167973130) l。在下面的文章中，我们将探索如何将它转化为一个 API，该 API 可以从 180，000 种葡萄酒的存储库中返回葡萄酒推荐。我们将关注一个特定的用例:返回给定描述符列表的葡萄酒推荐列表。

事实上，让我们说，我的心情是一个大胆的红葡萄酒与黑莓，蓝莓，巧克力和甘草的香味。我希望这款酒单宁含量高，酒体饱满。现在告诉我:什么葡萄酒最符合这个描述？

我们将使用 AWS 服务套件来确保返回葡萄酒推荐的过程完全是无服务器的。

**勾画出解决方案**

构建 API 的第一步是勾画出提交带有葡萄酒描述符的 POST 请求和返回葡萄酒建议列表之间所需的所有步骤。

虽然在本文中我们将重点关注 Lambda 和 API Gateway 的使用，但我们也将利用 S3 和 SageMaker 中的一个模型端点。概括地说，该流程如下所示:

![](img/549133c4497cf4e83970091e6d9383d9.png)

首先，让我们把注意力集中在 Lambda 函数上，我们将使用它来运行上面概述的过程。当设置 Lambda 函数时，我们需要小心选择一个执行角色，它将允许我们与 S3 和 SageMaker 进行交互。我们需要添加到 IAM 角色的 JSON 如下:

```
{
 “Sid”: “VisualEditor1”,
 “Effect”: “Allow”,
 “Action”: “sagemaker:InvokeEndpoint”,
 “Resource”: “*”
 },
 {
 “Sid”: “VisualEditor2”,
 “Effect”: “Allow”,
 “Action”: “s3:*”,
 “Resource”: “*”
 },
```

现在来看函数本身。因为我们将从函数内部的查找表中检索值，所以我们希望使用 Pandas 库。我们还将处理单词嵌入形式的向量，为此我们希望使用 Numpy。不幸的是，这些库不是 AWS Lambda 的原生库。为了利用这个功能，我们必须创建一个包含 Lambda 函数以及 Pandas 和 Numpy 的 Linux 二进制文件的 zip 文件。[本文](https://medium.com/@korniichuk/lambda-with-pandas-fd81aa2ff25e)更详细地概述了如何做到这一点。

Lambda 函数的第一部分如下所示:

除了导入必要的包，我们还定义了 lambda_handler 函数。这是每当调用 Lambda 函数时运行的函数。它处理的事件是 json 文件，其中包含我们的示例葡萄酒描述符列表:

{ "实例":["大胆"、"黑莓"、"蓝莓"、"巧克力"、"甘草"、"高单宁"、"醇厚"]}

为了安全起见，这个 json 文件被解码和编码，以确保它具有适当的(json)格式。

接下来，我们希望从 S3 的查找表中检索这些描述符的 IDF 加权单词嵌入。该查找表(S3 的 CSV 文件)的生成过程在本文的[中有详细描述。与本文中描述的过程的唯一区别是，我们将嵌入到查找文件中的每个单词乘以它的逆文档频率(IDF)分数。我们希望 Lambda 函数的这一部分获取以下单词嵌入，并将它们存储为一个列表。](/training-word-embeddings-on-aws-sagemaker-using-blazingtext-93d0a0838212?source=friends_link&sk=70d69339566eb57500ecd1589d11cadd)

![](img/7e4647c6dd90d1a9199880ac2db88f6c.png)

代码如下:

将单词嵌入转换成单个葡萄酒嵌入的过程相当简单。通过取它们的平均值，我们最终得到一个单一的词频率逆文档频率(TF-IDF)加权向量。

现在我们已经嵌入了我们的葡萄酒，我们准备检索我们的推荐。为此，我们将调用一个 AWS SageMaker 端点，该端点托管一个 Scikit-Learn Nearest Neighbors 模型，该模型专门为此进行了培训。[本文](/hosting-an-scikit-learn-nearest-neighbors-model-in-aws-sagemaker-40be982da703?source=friends_link&sk=fa7fbcb957b0e521c5f457d4af5932ce)更详细地解释了这样做的过程。

我们的最近邻模型端点返回我们的葡萄酒推荐的索引，以及这些推荐和我们的输入葡萄酒嵌入之间的余弦距离。最终，我们希望我们的 Lambda 函数返回这些葡萄酒的名称和风味特征。这些附加信息存储在 S3 的 CSV 文件中，需要检索:

现在我们已经设置了 Lambda 函数，我们可以将注意力转向创建 API 了。这里详细描述了建立我们的 API 的步骤[。我们创建了一个 POST 方法，它将接受一个带有我们的 wine 描述符的 JSON 文件，调用我们的 Lambda 函数，并返回一个推荐列表。](https://aws.amazon.com/blogs/machine-learning/call-an-amazon-sagemaker-model-endpoint-using-amazon-api-gateway-and-aws-lambda/)

我们可以通过使用 Postman 运行测试来确保我们的 API 正常工作。

![](img/2b250489a5639b586ef98b4d007f54a0.png)![](img/16699327d81d332dd155bfd95f7411f5.png)

推荐的葡萄酒似乎主要包括坦普拉尼洛和赤霞珠。该去购物了！