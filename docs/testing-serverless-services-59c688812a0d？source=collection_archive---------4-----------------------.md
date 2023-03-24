# 测试无服务器服务

> 原文：<https://towardsdatascience.com/testing-serverless-services-59c688812a0d?source=collection_archive---------4----------------------->

## 使用 moto 对 AWS lambda 函数进行 python 单元测试。

避免测试对于任何软件项目的成功都是至关重要的。如果您正在开发一个应用程序，您希望编写测试来检查您的应用程序的功能。通过测试，您可以添加新的特性或修复错误，并轻松地部署更改后的代码。没有测试，你将生活在一个充满不安全感和挫败感的世界里。

我们将使用 python 默认测试框架 **unittest** 和一个名为 **moto** 的 AWS 服务模拟工具来测试我们的 AWS lambda 的 python 功能。此外，我们将解释如何使用 **AAA 模式** (Arrange，Act，Assert)来安排和格式化你的测试代码。

![](img/52fc24291657099564701c82b43c03c8.png)

Photo by [Battlecreek Coffee Roasters](https://unsplash.com/@battlecreekcoffeeroasters?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/search/photos/quality-control?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

# Lambda 处理器逻辑

下面的代码是原始代码的一部分，它是一个名为“无服务器数据管道示例”的项目的一部分，是我之前写的[教程](/build-a-serverless-data-pipeline-on-aws-7c7d498d9707)的结果。这个处理程序的原始代码可以在我的 Github 上找到。

lambda 处理程序由 s3 事件触发。该事件是一个. eml 文件(电子邮件文件),它是在我们的 s3 bucket 的 unzip/文件夹中创建的。lambda 的配置和触发 lambda 的 s3 事件可以在项目的 serverless.yml 中的[这里](https://github.com/vincentclaes/serverless_data_pipeline_example/blob/master/serverless.yml#L43)找到。

该事件以字典的形式作为第一个参数传递给我们的处理函数。我们从事件中获取桶名和 s3 键，它指向我们的电子邮件文件的位置。我们读取电子邮件文件，并将内容提取为 JSON 对象。JSON 对象被转储回 s3。从环境变量中检索目标桶和关键字。环境变量在 serverless.yml 文件中配置[这里是](https://github.com/vincentclaes/serverless_data_pipeline_example/blob/master/serverless.yml#L52)。

# 测试 Lambda 处理器逻辑

从为项目中的测试创建一个结构开始。最简单的方法是模仿 tests 文件夹中的应用程序结构。你可以在这里找到一个例子:

```
.
├── serverless_data_pipeline
│   ├── lambda_function
│   │   ├── extract.py
└── serverless_data_pipeline_tests
    └── lambda_function
        └── test_extract.py
```

在 test_extract.py 模块中，我们有以下代码:

# 单元测试

为了测试我们的 lambda 处理程序的逻辑，我们将使用 [unittest](https://docs.python.org/3/library/unittest.html#module-unittest) 。unittest 是 python 标准库的一部分，帮助您创建和运行单元测试。

## **测试用例**

我们首先在 test_extract.py 文件中创建一个名为`TestExtract`的类。我们的类继承自 unittest.TestCase。通过从 TestCase 继承，您可以访问有助于您的测试的函数。在一个测试用例中，以关键字“test”开头的方法被 unittest 测试运行器视为一个测试。您将在我们的示例中找到一个测试函数:

```
test_extract_the_contents_of_an_email_successfully(self):
```

> 提示:以你的测试函数的名义描述你正在测试的东西，让你自己和他人容易理解你正在测试的东西。

## **安装()和拆卸()**

unittest 提供了一些函数，这些函数可以设置和拆除您的测试环境，而不需要在每次测试之前和之后显式调用它们。在我们的例子中，为了让我们的测试成功运行，我们需要必要的 s3 存储桶和一些环境变量。我们将此逻辑添加到 [*设置*](https://gist.github.com/vincentclaes/659f940445b482ae74aad92ac5d056bf#file-test_extract-py-L20) 功能中，并在执行测试之前调用设置功能。

```
def setUp(self):
    os.environ[TestExtract.ENV_DEST_KEY] = TestExtract.DEST_KEY
    os.environ[TestExtract.ENV_DEST_BUCKET] = TestExtract.DEST_BUCKET

    conn = boto3.resource('s3')
    conn.create_bucket(Bucket=TestExtract.SOURCE_BUCKET)
    conn.create_bucket(Bucket=TestExtract.DEST_BUCKET)
```

测试运行后，我们要删除 s3 存储桶和环境变量。我们在 [*拆卸*](https://gist.github.com/vincentclaes/659f940445b482ae74aad92ac5d056bf#file-test_extract-py-L28) 函数中添加了该逻辑，并且每次测试完成时，都会执行拆卸函数。

```
def tearDown(self):
    del os.environ[TestExtract.ENVIRON_DEST_KEY]
    del os.environ[TestExtract.ENVIRON_DEST_BUCKET]

    self.remove_bucket(TestExtract.SOURCE_BUCKET)
    self.remove_bucket(TestExtract.DESTINATION_BUCKET)
```

# **摩托**

我们希望在本地执行我们的功能，而不想与“真实的”AWS 环境交互。最好我们想与一个模拟的 AWS 环境互动。为了创建一个模拟的 AWS 环境，我们将使用 [moto](https://github.com/spulec/moto) 。moto 是一个库，允许您的测试轻松模拟 AWS 服务。我们从 moto 继承了 mock_s3，并将其用作测试类的装饰器:

```
@mock_s3
class TestExtract(unittest.TestCase):
```

一旦将 mock_s3 设置为装饰器，通过 boto3 与 s3 的每一次交互都会被模仿。像这样，我们的测试可以离线运行，就像它们与 AWS 服务交互一样。

# 安排动作断言(AAA)

在测试函数中，我们测试内容是否从。eml 文件，并且内容作为 JSON 对象成功地转储到 s3 上。每个测试函数都是按照 [**安排、动作、断言(AAA)** 原则](http://wiki.c2.com/?ArrangeActAssert)构造的。

## **安排你的模拟 AWS 环境**

首先，我们 [*安排*](https://github.com/vincentclaes/serverless_data_pipeline_example/blob/master/serverless_data_pipeline_tests/lambda_function/test_extract.py#L106) 我们的环境以便执行我们的测试。如上所述，一般的设置和拆卸功能可以在设置和拆卸功能中完成。特定于测试的配置可以在测试函数本身中完成。

一旦创建了 s3 存储桶并在 setUp 函数中设置了环境变量，我们就通过读取一个. eml 文件并将该文件上传到 s3 来在我们的测试函数中安排特定于测试的环境。读取和上传发生在函数中

```
def put_email_to_s3(self, test_email_path, email_name):
```

## **执行您想要测试的功能**

下一步是 [*act*](https://github.com/vincentclaes/serverless_data_pipeline_example/blob/master/serverless_data_pipeline_tests/lambda_function/test_extract.py#L113) 通过执行我们想要测试的 lambda 处理函数。

```
s3_key_extracted_message = extract.handler(event, None)
```

我们看到我们的处理函数返回了一个 s3 键。在生产中，这个 s3 密钥没有被使用，也没有任何价值。我们返回这个值只是为了验证我们的测试是否如我们所期望的那样运行。

## **断言我们得到了预期的结果**

一旦我们的处理函数运行，我们 [*断言*](https://github.com/vincentclaes/serverless_data_pipeline_example/blob/master/serverless_data_pipeline_tests/lambda_function/test_extract.py#L116) 预期的结果已经出现。我们通过获取 lambda 处理程序返回的 s3 键并检查被模仿的 s3 桶是否包含预期的 JSON 对象来做到这一点。

我们从 s3 中读取对象，并断言 JSON 对象等于我们期望的对象。不要开始写自己的断言函数，unittest 有一堆高级断言方法，可以在[这里](https://docs.python.org/2/library/unittest.html#assert-methods)找到。我们使用函数

```
self.assertDictEqual(json.loads(email_as_json), expected_json)
```

比较字典是否相等。

# 运行测试

要自己运行上面的示例，从克隆原始项目并安装依赖项开始:

```
git clone [git@github.com](mailto:git@github.com):vincentclaes/serverless_data_pipeline_example.gitcd serverless_data_pipeline_example
pipenv install
pipenv shell
```

要执行项目中的所有测试，请运行:

```
python -m unittest
```

它将运行测试用例的所有测试功能。测试应该通过，您应该看到输出“OK”。

```
bash-3.2$  . /Users/vincent/.local/share/virtualenvs/serverless_data_pipeline-27feO5HC/bin/activate
(serverless_data_pipeline) bash-3.2$ python -m unittest serverless_data_pipeline_tests/lambda_function/test_extract.py
email object {'id': 'test_extract_the_contents_of_an_email_successfully.eml', 'from': '[vclaes1986@gmail.com](mailto:vclaes1986@gmail.com)', 'to': '[vincent.v.claes@gmail.com](mailto:vincent.v.claes@gmail.com)', 'cc': '', 'subject': 'Hey how are you doing', 'date': '2019-07-09 13:42:54+02:00', 'body': '\nCash Me Outside How Bout Dah'}
.
----------------------------------------------------------------------
Ran 1 test in 0.185sOK
```

要运行一个单独的测试用例，您可以运行:

```
python -m unittest serverless_data_pipeline_tests.lambda_function.test_extract.TestExtract
```

要运行单个测试函数，您可以运行:

```
python -m unittest serverless_data_pipeline_tests.lambda_function.test_extract.TestExtract.test_extract_the_contents_of_an_email_successfully
```

如果您的所有测试都通过了，您就可以放心地将项目部署到 AWS 了。

祝你好运！