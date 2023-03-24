# 如何像专业人士一样编写 Python 命令行界面

> 原文：<https://towardsdatascience.com/how-to-write-python-command-line-interfaces-like-a-pro-f782450caf0d?source=collection_archive---------2----------------------->

![](img/4750f6e9b3a4c28146472c5d69ad0d1f.png)

Photo by [Kelly Sikkema](https://unsplash.com/@kellysikkema?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText) on [Unsplash](https://unsplash.com/s/photos/interface?utm_source=unsplash&utm_medium=referral&utm_content=creditCopyText)

作为数据科学家，我们面临着许多重复和相似的任务。这包括创建每周报告，执行提取、转换、加载( [ETL)](https://en.wikipedia.org/wiki/Extract,_transform,_load) 作业，或者使用不同的参数集训练模型。通常，我们最终会有一堆 Python 脚本，每次运行它们时，我们都会更改代码中的参数。我讨厌这样做！这就是为什么我养成了将脚本转换成可重用的命令行界面(CLI)工具的习惯。这提高了我的效率，让我在日常生活中更有效率。我开始用 [Argparse](https://docs.python.org/3/howto/argparse.html) 做这件事，但是这并不愉快，因为我不得不产生很多难看的代码。所以我想，我不需要一遍又一遍地写很多代码就能实现吗？我能享受编写 CLI 工具的乐趣吗？

> [点击](https://click.palletsprojects.com/en/7.x/)是你的朋友！

那么什么是点击呢？从网页上:

> (点击)它旨在使编写命令行工具的过程变得快速而有趣，同时防止因无法实施预期的 CLI API 而导致的任何挫折。

对我来说，这听起来很棒，不是吗？

在本文中，我将向您提供如何使用 Click 构建 Python CLIs 的实践指南。我一步一步地构建了一个示例，向您展示 Click 提供的基本功能和好处。学完本教程后，您应该能够愉快地编写下一个 CLI 工具，而且只需一眨眼的功夫:)所以，让我们动手吧！

# 教程

在本教程中，我们使用 Click 逐步构建一个 Python CLI。我从基础开始，每一步都引入 Click 提供的新概念。除了 Click，我还使用[poems](https://poetry.eustace.io/)来管理依赖项和包。

## 准备

首先，我们来装诗。有多种方法可以做到这一点，参见我的文章，但是这里我们使用 pip

```
pip install poetry==0.12.7
```

接下来，我们使用诗歌创建一个名为 *cli-tutorial 的项目，*添加 click 和 funcy 作为依赖项，并创建一个文件 cli.py，稍后我们将在其中填充代码

```
poetry new cli-tutorial
cd cli-tutorial
poetry add click funcy
**# Create the file we will put in all our code**
touch cli_tutorial/cli.py
```

我已经添加了 funcy，因为我稍后会用到它。要了解这个模块有什么用处，我建议感兴趣的读者参考这篇文章。现在，我们已经准备好实施我们的第一个 CLI。顺便说一下，所有示例代码都可以在我的 [GitHub 账户上找到。](https://github.com/Shawe82/cli-tutorial)

## 我们的第一个 Click CLI

我们的初始 CLI 从磁盘读取一个 CSV 文件，对其进行处理(如何处理对本教程并不重要)，并将结果存储在 Excel 文件中。输入文件和输出文件的路径都应该由用户配置。用户*必须*指定输入文件路径。指定输出文件路径是可选的，默认为 *output.xlsx* 。使用 Click，这样做的代码读作

```
**import** click**@click.command()
@click.option("--in", "-i", "in_file", required=True,
    help="Path to csv file to be processed.",
)
@click.option("--out-file", "-o", default="./output.xlsx",
    help="Path to excel file to store the result.")**
**def** process(**in_file**, **out_file**):
 **""" Processes the input file IN and stores the result to 
    output file OUT.
    """**
    input = read_csv(in_file)
    output = process_csv(input)
    write_excel(output, out_file)**if** __name__ =="__main__":
    process()
```

我们在这里做什么？

1.  我们用 *click.command.* 修饰我们想要从命令行调用的方法 *process*
2.  我们使用 *click.option* 装饰器定义命令行参数。现在，您必须小心在修饰函数中使用正确的参数名。如果我们将不带破折号的字符串添加到 *click.option* 中，参数必须与该字符串匹配。这是- in 和 in_file 的情况。如果所有名称都包含前导破折号，单击将使用最长的名称生成参数名称，并将所有非前导破折号转换为下划线。该名称被转换为小写。出文件和出文件就是这种情况。更多详情，请参考[点击文档](https://click.palletsprojects.com/en/7.x/parameters/#parameter-names) n。
3.  我们使用 *click.option* 的相应参数来配置所需的先决条件，如默认值或必需参数。
4.  我们将帮助文本添加到我们的参数中，当使用- help 调用我们的函数时会显示该文本。来自我们函数的 docstring 也将显示在那里。

现在，您可以通过多种方式调用这个 CLI

```
**# Prints help**
python -m cli_tutorial.cli --help
**# Use single char -i for loading the file**
python -m cli_tutorial.cli -i path/to/some/file.csv
**# Specify both file with long name**
python -m cli_tutorial.cli --in path/to/file.csv --out-file out.xlsx
```

太棒了，我们已经使用 Click 创建了第一个 CLI！

注意，我没有实现 *read_csv* 、 *process_csv* 和 *write_excel* ，而是假设它们存在并做它们应该做的事情。

CLIs 的一个问题是我们将参数作为通用字符串传递。为什么这是一个问题？因为这些字符串必须被解析为实际类型，这可能会由于用户输入的格式不正确而失败。看看我们的例子，我们使用路径并尝试加载一个 CSV 文件。用户可以提供一个根本不代表路径的字符串。即使字符串格式正确，相应的文件也可能不存在，或者您没有访问它的正确权限。自动验证输入，如果可能的话解析它，或者在失败的早期给出有用的错误消息，这难道不是一件好事吗？理想情况下，所有这些都不需要编写大量代码？Click 通过为我们的参数指定类型来支持我们。

## 类型规范

在我们的示例 CLI 中，我们希望用户传入一个**有效路径**到一个**现有文件**，我们对该文件拥有**读取权限**。如果满足了这些要求，我们就可以加载输入文件了。此外，如果用户指定输出文件路径，这应该是一个有效的路径。我们可以通过传递一个*点击来执行所有这些操作。路径*对象到 *click.option* decorator 的*类型*参数

```
@click.command()
@click.option("--in", "-i", "in_file", required=True,
    help="Path to csv file to be processed.",
    **type=click.Path(exists=True, dir_okay=False, readable=True)**,
)
@click.option("--out-file", "-o", default="./output.csv",
    help="Path to csv file to store the result."**,**
    **type=click.Path(dir_okay=False)**,
)
**def** process(in_file, out_file):
    """ Processes the input file IN and stores the result to output     
    file OUT.
    """
    input = read_csv(in_file)
    output = process_csv(input)
    *write_excel*(output, out_file)
...
```

*点击。路径*是 Click out of the box 提供的各种类型之一。您也可以实现自定义类型，但这超出了本教程的范围。有关更多细节，我建议感兴趣的读者参考 Click [文档](https://click.palletsprojects.com/en/7.x/parameters/#implementing-custom-types)。

## 布尔标志

Click 提供的另一个有用的特性是布尔标志。可能，最著名的布尔标志是*详细*标志。如果设置为 true，您的工具将向终端打印出大量信息。如果设置为 false，则只打印少量内容。通过点击，我们可以实现为

```
**from** funcy **import** identity
...
**@click.option('--verbose', is_flag=True, help="Verbose output")**
**def** process(in_file, out_file, **verbose**):
    **""" Processes the input file IN and stores the result to
    output file OUT.
    """**
    print_func = print **if** verbose **else** identity    print_func("We will start with the input")
    input = read_csv(in_file)
    print_func("Next we procees the data")
    output = process_csv(input)
    print_func("Finally, we dump it")
    write_excel(output, out_file)
```

你所要做的就是再添加一个*click . option*decoration 并设置 *is_flag=True* 。现在，要获得详细的输出，您需要调用 CLI 作为

```
python -m cli_tutorial.cli -i path/to/some/file.cs --verbose
```

## 特征开关

假设我们不仅想在本地存储 *process_csv* 的结果，还想将它上传到服务器。此外，不仅有一个目标服务器，还有一个开发实例、一个测试实例和一个生产实例。您可以通过不同的 URL 访问这三个实例。用户选择服务器的一个选项是将完整的 URL 作为参数传递，她必须键入该参数。但是，这不仅容易出错，而且是一项繁琐的工作。在这种情况下，我使用*功能开关*来简化用户的生活。他们所做的最好通过代码来解释

```
...
@click.option(
    **"--dev", "server_url"**, help="Upload to dev server",
    **flag_value**='https://dev.server.org/api/v2/upload',
)
@click.option(
    **"--test", "server_url"**, help="Upload to test server",
    **flag_value**='https://test.server.com/api/v2/upload',
)
@click.option(
    **"--prod", "server_url"**, help="Upload to prod server",
    **flag_value**='https://real.server.com/api/v2/upload',
    **default=True**
)
**def** process(in_file, out_file, verbose, **server_url**):
    """ Processes the input file IN and stores the result to output
    file OUT.
    """
    print_func = print **if** verbose **else** identity
    print_func("We will start with the input")
    input = read_csv(in_file)
    print_func("Next we procees the data")
    output = process_csv(input)
    print_func("Finally, we dump it")
    write_excel(output, out_file)
    print_func("Upload it to the server")
    upload_to(server_url, output)
...
```

这里，我为三个可能的服务器 URL 添加了三个*click . option*decorator。重要的一点是，所有三个选项都有相同的目标变量 *server_url* 。根据您选择的选项，**server _ URL*的值等于 *flag_value* 中定义的相应值。您通过添加- *dev* 、- *test* 或- *prod* 作为参数来选择其中之一。所以当你执行的时候*

```
*python -m cli_tutorial.cli -i path/to/some/file.csv **--test***
```

*server_url 等于“https://test . server . com/API/v2/upload”。如果我们不指定这三个标志中的任何一个，Click 将取- prod 的值，因为我设置了 *default=True* 。*

## *用户名和密码提示*

*不幸的是，或者说幸运的是:)，我们的服务器有密码保护。所以要上传我们的文件，我们需要一个用户名和密码。当然，您可以提供标准的 click.option 参数。但是，您的密码会以纯文本的形式出现在您的命令历史记录中。这可能会成为一个安全问题。*

*我们喜欢提示用户输入密码，而不将密码回显到终端，也不将密码存储在命令历史中。对于用户名，我们也希望一个简单的提示*与*回显。当你知道点击时，没有比这更容易的了。这是代码。*

```
*import os
...
@click.option('--user', **prompt=True**,
              **default=lambda: os.environ.get('USER', '')**)
@click.**password_option**()
def process(in_file, out_file, verbose, server_url, **user**, **password**):
    ...
    upload_to(server_url, output, user, password)*
```

*要为一个参数添加提示，你必须*设置提示=真*。每当用户没有指定- *user* 参数时，这将*添加*一个提示，但是她仍然可以这样指定。当您在提示符下点击 enter 键时，将使用默认的值。默认值由函数决定，这是 Click 提供的另一个便利特性。*

*提示密码而不回显到终端并要求确认是如此普遍，以至于 Click 提供了一个名为 *password_option* 的专用装饰器。重要的注释；这不会阻止用户通过-password my cretpassword 传递密码。这只能让她不那么做。*

*仅此而已。我们已经构建了完整的 CLI。在今天结束之前，我想在下一部分给你一个最后的提示。*

## *诗歌脚本*

*我想给你的最后一个技巧是创建[诗歌脚本](https://poetry.eustace.io/docs/pyproject/#scripts)，它与点击无关，但与 CLI 主题完全匹配。使用诗歌脚本，您可以创建可执行文件来从命令行调用您的 Python 函数，就像您使用 [Setuptools 脚本](https://python-packaging.readthedocs.io/en/latest/command-line-scripts.html#the-scripts-keyword-argument)一样。那看起来怎么样？首先，需要将以下内容添加到 pyproject.toml 文件中*

```
*[tool.poetry.scripts]
your-wanted-name = 'cli_tutorial.cli:process'*
```

**your-want-name*是 *cli_tutorial.cli* 模块中定义的函数*进程*的别名。现在，你可以通过*

```
*poetry run your-wanted-name -i ./dummy.csv --verbose --dev*
```

*例如，这允许您向同一个文件添加多个 CLI 函数，定义别名，并且您不必添加 *if __name__ == "__main__"* 块。*

# *包裹*

*在本文中，我向您展示了如何使用点击和诗歌来轻松构建 CLI 工具，并提高工作效率。这只是 Click 提供的一小部分功能。还有很多其他的，比如[回调](https://click.palletsprojects.com/en/7.x/options/#callbacks-and-eager-options)、[嵌套命令](https://click.palletsprojects.com/en/7.x/commands/#nested-handling-and-contexts)，或者[选择选项](https://click.palletsprojects.com/en/7.x/options/#choice-options)等等。现在，我建议感兴趣的读者参考 Click 文档，但我可能会写一篇后续文章来讨论这些高级主题。敬请关注，感谢您关注这篇文章。如有任何问题、意见或建议，请随时联系我。*