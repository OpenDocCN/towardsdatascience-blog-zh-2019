# TDD 不应该是 TDDious

> 原文：<https://towardsdatascience.com/tdd-shouldnt-be-tddious-e8d6f34cb9d8?source=collection_archive---------24----------------------->

## [现实世界中的数据科学](https://towardsdatascience.com/data-science-in-the-real-world/home)

## 我仍然会遇到古老的“如何测试”的争论，但是我们能让测试变得有趣吗？

![](img/a0a952a23eb9d217b785cbc9477493ec.png)

*(Image by author)*

我作为一名工程师已经工作了十多年，仍然会遇到“如何测试”的争论。我是首席工程师，这意味着我要和我的团队一起研究如何建造东西，所以我总是乐于接受不同的观点。但是我喜欢的一个话题是我们是否应该使用测试驱动开发或者 TDD。我认为这可以归结为一个根本性的误解，即为什么你应该享受 TDD 而不是憎恨它…

> TDD 以一种清晰的方式让我明白我应该写什么代码。

对我来说，TDD 很有趣。我喜欢用测试驱动的方式编码，我无法想象为什么有人会想用其他方式编码。当我不知道如何为某个东西写一个简洁的测试时，我会很难过。这就把我们带到了通常对 TDD 的第一个抱怨——“我不能只编码”。这很有趣，因为 TDD 以一种清晰的方式让我明白我应该写什么代码。如果没有它，我可能会迷失在边缘案例中，或者在我并不打算进行的重构中。另一件事是 TDD 实际上让我“只写代码”，因为我可以写任何我喜欢的可怕的代码，只要测试通过。那种自由太神奇了！我写了一些可怕的代码来通过测试，测试我的假设，证明我的方法，或者看看我是否能过度优化。以连接到数据库为例:

```
from mocks import mock_databasedef test_query_database():
  expected_customer = {
    'id': 32,
    'name': 'woof'
  } mock_database()
  customer = get_customer('woof')
  assert customer == expected_customer
```

这可以通过多种方式实现。我很快就把这些放在一起:

```
import databasedef get_customer(user):
  return database.connect().query(f'SELECT id,name FROM customers WHERE name="{user}"').fetch()
```

TDD 允许我专注于我希望我的代码实现的事情。遵循红色->绿色->重构的循环，我可以在任何时候返回以使代码更整洁:

```
import databasecustomer_query = 'SELECT id,name FROM customers WHERE name="{}"'def get_customer(user):
  conn = database.connect()
  cursor = conn.query(customer_query.format(user))
  return cursor.fetch()
```

> 测试给你一个安全网，你之前写的功能仍然在运行和工作。

这证明了在测试通过后重写代码的想法。对此，我经常采用删除的方法；如果我可以删除代码并且测试仍然通过，那么我可以很高兴地重构它。我也将这种技术作为教学工具，向人们展示他们编写了多少额外的代码。删除代码有助于证明它不是不需要的，就是覆盖度量是误导的。带着“最大化未完成的工作”的敏捷思维，你应该为测试编写尽可能少的代码。这种代码的不断减少是有益的，它本身就是一个游戏，增加了乐趣！请注意，测试代码的成本可能过高；例如，我们在 AWS Lambda 处理函数之外初始化框架。这个初始化比处理程序更难测试，因为它在导入时运行。所以删除代码是有一定的背景的，但是一般来说一些测试层可以覆盖所有的东西。

令人兴奋的是，如果我不立即重构，在我有几个测试用例之后，我的代码仍然会很糟糕。重构并不总是发生在每次测试之后。测试给你一个安全网，你之前写的功能仍然在运行和工作。您可以继续沿着一条通往其自然结论的道路前进，而不用担心其他东西被破坏或者您稍后需要做的深入重构。通过评估代码，尽可能多地删除代码，您可以在测试保护您的情况下积极地进行大量重构。

另一个常见的抱怨是在做出改变时“我必须改变所有的测试”。在测试领域也有一些争论，关于你应该在什么水平上测试多少。我个人的哲学是，你应该测试你想要达到的目标。在数据管道的情况下，这可能是:从 S3 拉一个文件，对它做一些工作，然后把它放回 S3 的另一个位置。因为我重视 TDD 中的乐趣，这类测试比成百上千的说着同样事情的单元测试更让我高兴。这种数据处理测试的一个例子可能是:

```
import pandas
import moto 
from pandas.testing import assert_frame_equalfrom functions.cake_mixer import mix_ingredients@pytest.fixture
def bucket_fixture():    
  with moto.mock_s3():        
    s3_resource = boto3.resource("s3")        
    bucket = s3_resource.Bucket("test")
    bucket.create()
    yield bucketdef test_processing_cake(bucket_fixture):
  input_fixture = pandas.read_csv('fixtures/mixture.csv')
  output_fixture = pandas.read_csv('fixtures/cake.csv') input_fixture.to_parquet('s3://test/ingredients/mixture.parquet')
  mix_ingredients({
    'bucket': 'test_bucket',
    'mixture_key': 'ingredients/mixture.parquet',
    'cake_key': 'cake/chocolate.parquet'
  })
  output = pandas.read_parquet('s3://test/cake/chocolate.parquet')
  assert_frame_equal(output, output_fixture)
```

这做了很多:

*   为 S3 设置固定装置
*   上传测试示例文件
*   运行一些代码来处理这个例子
*   读出结果文件
*   断言您的预期输出与处理后的输出相匹配

这个例子说明了在这种测试中你可能采取的步骤。要采取更小的步骤，您可以在涉及 S3 之前就开始处理数据:

```
import pandas
from pandas.testing import assert_frame_equalfrom functions.cake_mixer import mix_ingredientsdef test_processing_cake():
  input_fixture = pandas.read_csv('fixtures/mixture.csv')
  output_fixture = pandas.read_csv('fixtures/cake.csv') output = mix_ingredients(input_fixture)
  assert_frame_equal(output, output_fixture)
```

当您编写一个不断发展的测试时，您可以感觉到代码在一起。希望这能解决“我必须改变所有的测试”的问题，因为您只完成了实际的需求。紧密耦合的单元不容易改变。一个警告是，当一个单元有一组复杂的输入和输出时，单元测试可以帮助定义它们。

重要的是，不要试图一次实现太多，要朝着更大的结果努力。有时你想要使用多个测试循环来帮助构建一个更大的目标。对于数据管道，您可以为管道编写一个测试，然后为每个步骤编写一组更小的测试。也就是说，您可以测试管道的输出和概念证明管道是否匹配。然而，管道中的各个阶段都有针对具体的、可解释的转换的测试。从 pandas 概念验证到大规模 Spark 数据管道可以使用相同的端到端测试数据。

> 测试本质上是一个有趣的游戏，你可以设计自己的挑战，然后找到最好、最令人愉快的方式来完成它们。

最后，人们发现考试很单调，对此我无能为力。测试本质上是一个有趣的游戏，你可以设计自己的挑战，然后找到最好、最令人愉快的方式来完成它们。让这一点变得更清楚的最好方法是给自己找一个搭档，用乒乓球的方式进行练习。在乒乓球比赛中，一个人写一个测试，另一个人让它通过。竞相编写更好的测试和代码。有趣的是找到让代码做意想不到的事情的方法，或者找到让你的伙伴多思考一点来解决的边缘情况。例如，如果有一个函数要返回传递的值，您可以很容易地编写以下代码:

```
def test_identity():
  assert identity(1) == 1
```

你可以很容易地从中获得乐趣，如下所示:

```
def identity(x):
  return 1
```

这意味着您的合作伙伴现在必须实现预期的功能。下一次，技巧可能是检查 x 的两个不同值，以防止您玩得太开心:

```
def test_identity():
  assert identity(1) == 1
  assert identity(404) == 404
```

您已经介绍了几个额外的案例，并设法确保代码完全按照您的意图运行。

我真的希望这篇文章向你展示，TDD 给你的不是负担，而是:

*   理解你要写的代码和它应该做什么
*   这是一张继续编写糟糕代码的安全网，直到你想重构为止
*   工作生活的游戏化，包括多人游戏！

所以，就算你以前试过，再试试 TDD，因为大家应该会玩得更开心:-)。