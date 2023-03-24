# Python 教程:政府社交媒体分析器类构建

> 原文：<https://towardsdatascience.com/python-tutorial-government-social-media-analyser-class-build-out-12858190b284?source=collection_archive---------21----------------------->

## [Python 统计教程系列](https://towardsdatascience.com/tagged/python-stat-tutorial)

## 增强您的程序，使其适用于任何政治人物名单

点击标题上方的链接，查看所有教程文章的列表。

在这个练习中，我们将把我们的初始程序([参考第一个教程)](https://dev.cloudburo.net/2019/01/27/python-tutorial-retrieve-a-list-of-swiss-government-members-from-twitter.html)转换成一个通用程序，用于获取一个国家的议会成员的 twitter 信息。这意味着我们对代码进行第一次迭代的概括和重构。

我们的目标是有一个程序可以分析任何国家的政治家的 Twitter 账户，而不仅仅是瑞士。

![](img/c81ceb3d2c215944a71a403619c0cefd.png)

因此，我们的计划应该以一种我们也可以在其他国家使用(即配置)的方式进行推广。

# Yaml 配置文件

我们的第一个 [lesson1.py](https://github.com/talfco/clb-sentiment/blob/master/src/lesson1/lesson1.py) Python 程序是直接在程序中硬编码 twitter 账户及其列表名。

现在我们要做的是从配置文件中读出这些信息，类似于我们在第一课中为传递 Twitter API 秘密所做的工作([参考教程](https://dev.cloudburo.net/2019/01/27/python-tutorial-retrieve-a-list-of-swiss-government-members-from-twitter.html))

这个信息不是什么秘密，所以我们创建了一个新的公共 yaml 配置文件，它将用于我们程序的任何国家配置参数。当您一般化程序时，将所有这些参数卸载到一个配置文件中是一个关键点。

正如你在下面的截图中看到的，我们将文件命名为 *config-CH.yaml* 。

config-CH.yaml

那么为什么要使用 CH 后置呢？CH 后缀是瑞士国际 ISO 国家代码标准的 Alpha-2 代码。代码值(或参考值)是任何信息编码程序的组成部分。最佳实践是尽可能使用标准化的(不要在这里重复发明)。也就是说，对于国家，我们决定使用标准 ISO 3166-1 的 Alpha-2 代码。正如您将在后面看到的，对于编码语言，我们采用了类似的方法。顺便说一下，ISO 的意思是“[国际标准化组织](https://en.wikipedia.org/wiki/International_Organization_for_Standardization)”

![](img/17cd61fdff729c2455491af8ea580372.png)

# 重构和增强

## GovernmentSocialMediaAnalyzer 类

对于我们的一般化程序，我们做第一个[重构](https://en.wikipedia.org/wiki/Code_refactoring)步骤。代码重构是在不改变现有计算机代码外部行为的情况下对其进行重构的过程。所以我们将 sample1.py 中的类重命名为*government social media analyzer*，并通过参数 *country_code* 增强其类构造函数 __ *init__* 方法。我们做出了第一个设计决定:

> 设计决策 1:我们的类 GovernmentSocialMediaAnalyzer 的一个实例将封装一个专用国家的数据和行为。

代码增强如下所示:

*   创建类时传入的 *country_code* 参数(如 *CH*
*   将作为私有实例变量 __ *country_code* 存储并使用
*   要创建 yaml 配置*文件名*，
*   从这里我们将加载配置数据并将数据存储在私有变量 *__cfg* 中

因此，现在我们准备通过从存储在 *self 中的配置数据中读取 twitter 帐户和列表名称来推广我们的 *get_government_members* 方法。__cfg* 实例变量。

但是让我们先完成对我们的 *__init__* 类的重构和增强。我们采取另一个设计决策

> 设计决策 2:*init*方法应该封装所有 Twitter 帐户从 Twitter 的加载，以及到 attributes 列(=字符串数组)中相关属性的转换。该列应该可以作为私有类实例变量使用

这意味着当我们为一个专门的国家创建一个 *GovernmentAnalyzer* 实例时，案例的初始化阶段将包括将来自 Twitter 的数据放入我们的 intern 数据结构(如列所示)的所有必要步骤

我们将在一个专用帮助器方法中完成这个步骤，这个方法将被称为 __ *extract_columns* 。我们将它定义为私有方法(__ prefix)，因为它不应该被这个类之外的任何人使用。

第一课中重构的类现在看起来像这样。

我们使列属性更具描述性，并将它们定义为类实例变量，以便这些列可以被我们的类中的任何方法使用。

所以我们已经完成并重构了类实例创建类

*   5–12:加载特定于国家的配置文件的代码块
*   16–21:从秘密配置文件中读取 twitter 安全令牌和密钥的代码块，然后连接到 Twitter API
*   39:调用 *_extract_columns* 方法检索数据并转换成列。

教程一中的 *check_for_party* 算法是在代码中硬编码 party 缩写。

好了，让我们重构代码，并将参与方信息移动到我们的配置文件中。由于 YAML 文件的灵活性，这可以很容易地完成。

> 设计决策 3:我们希望为每个政党使用几个政党缩写(可能使用多种语言)和关键字(例如，政党 Twitter 昵称)，以尝试识别政治家的政党所有权。

所以我们的配置 *config-CH.yaml* 将需要每一方的配置信息。在瑞士的 [parlament.ch](https://www.parlament.ch/de/organe/fraktionen/im-parlament-vertretene-parteien) 网站上可以找到四种语言的政党及其缩写。

在 YAML，您可以快速建立一个配置项目列表(例如，聚会配置项目)。列表成员由前导连字符(-)表示，每一到多行一个成员，或者用方括号([ ])括起来，并用逗号分隔(，)。

*   当事人名单成员用连字符表示。一个政党名单成员有一个 twitter 和*缩写*属性。*缩写*(缩写)属性本身是一个由方括号符号表示的字符串列表。

config-CH.yaml (with parties list)

如果我们检查加载的配置文件(存储在 *self 中。_cfg* 变量)在 Python 调试器中，使用 Python 列表和字典应该很清楚数据结构是什么样子。

![](img/e5a7b5628175fff8635fe057adbf3862.png)

作为对*abbres*属性的补充说明，我们介绍了他们的政党缩写列表，即拥有多种国家语言也意味着政党有多种缩写(例如，德语和法语)。在我们上面的例子中是“FDP”和“PLR”我们想检查所有的。在其他国家，可能只有一个缩写，但有了这个决定，我们就是未来的证明。

我们改进的 *check_for_party* 方法现在看起来如下。它将遍历所有 twitter 帐户和每一方的配置记录，并检查 twitter 帐户的描述或昵称是否与某一方匹配。

*   在第 6、10 和 19 行，我们从配置结构中获取数据。
*   根据属性类型，我们必须遍历值列表(6，10)或直接获取数据(19)
*   如果我们有一个匹配，第一个缩写将被返回，作为我们识别当事人的代码值:*RES = party[' abbs '][0]*

# 微调算法

## 引入第二个 Plotly 表:按交易方对账户进行分组

为了微调我们的算法，我们必须检查它在寻找 twitter 帐户的一方时的有效性。为此，我们必须引入第二个表，它将根据我们的 twitter 帐户的聚会分配对它们进行分组。

强大的熊猫包将为我们提供必要的工具。你可以参考 panda API 描述后面的[，了解如何分组数据的所有细节。](https://pandas.pydata.org/pandas-docs/stable/reference/api/pandas.DataFrame.groupby.html)

代码片段的一些注释:

*   4–8:我们在这里创建一个包含 4 列的 panda_data 记录。 *__col_party* ， *__col_followers_count* ， *__col_friends_count* 。 *__col_party* 使用了两次，第一列用于标记每一行(如您在第 11 行看到的，我们按参与方进行分组),在第二列中，我们汇总了具有相同参与方的行。
*   9:我们创建了这个表的第一个 panda 数据框，有四列
*   11:这里我们通过使用 *groupby* 函数来转换创建的数据帧。我们还为第二、第三和第四行定义了聚合 *agg* 操作。
*   15–19:创建一个漂亮的 plotly 表的基本材料。

让我们运行程序并检查我们的 party assignment 分配算法的准确性。作为程序执行的结果，您现在必须在您的 plotly 帐户中创建表格(还将创建一些网格表，这些表格目前并不相关)。

![](img/64168c21e403d2df3a7f378abf4b11d8.png)

# 用关键字属性增强我们的配置文件

我们的首次调查显示，大多数政客(65 人)不会在他们的账户描述/昵称中提及他们的党派缩写或党派 twitter 昵称。

![](img/2d42ce77ac391906c1db0dfa4a187d93.png)

所以让我们试着微调我们的算法。让我们再次浏览这个列表，并检查其他可以帮助我们识别他们的聚会关系的关键字。

![](img/4c8aaedaef95a82838d0757d6960fd96.png)![](img/603ae543ca4ecae442e28113c2957192.png)

我们找到了以下关键词:

*   社会主义党
*   glp (GLP)、
*   格鲁内(GLP)
*   勒加

所以让我们用一个新属性将它添加到我们的配置文件中: *keywords* 。这就是 YAML 的魅力所在，你可以很容易地用附加属性来扩展它。在我们的例子中是另一个列表属性。

我们在我们的 *check_for_party* 方法(23–28)中添加了额外的检查

瞧，我们可以找到另外 13 个拥有超过 20，000 名粉丝的推特账户。尽管如此，52 个帐户不能映射到一个聚会，但为此，我们必须连接另一个数据源，这将在后面的教程中完成。

![](img/b470ddaa1737ad9f285c8cf3aba0ef18.png)

作为今天的最后一步，我们重构了 *create_politican_table* 方法。我们主要通过在文件名中使用国家代码来规范 plotly 中使用的文件名。这使我们能够为不同的国家生成表格，并确保它们不会在我们的 plotly 帐户中相互覆盖(20)。

现在，我们已经概括和重构了我们的整个应用程序，并为进一步的构建打下了良好的基础。

我们现在可以为一个特定的国家实例化一个*government social media analyzer*(假设我们已经提供了必要的配置文件),并将 twitter 相关数据提取到一个 plotly 表中以供进一步处理。

作为一个可视化的 UML 序列图，我们的类的交互流可以表示如下:

![](img/572d2515f6e4b6d1055346254993f35c.png)

如果你想了解更多关于 UML 序列图的细节，请参考下面的[教程](https://www.geeksforgeeks.org/unified-modeling-language-uml-sequence-diagrams/)。这是一种很好的可视化程序各个方面的技术。在上图中，消息的返回调用用蓝色表示。

例如:panda 包创建数据帧的消息行用红色表示(createDataFrame)，它返回的 data frame 对象的消息行用蓝色表示。

# 锻炼

使用通过政府推特账户(【https://twitter.com/TwitterGov】[)提供的列表之一，例如，英国议会成员列表(](https://twitter.com/TwitterGov/lists/us-cabinet?lang=de)[【https://twitter.com/TwitterGov/lists/uk-mps】](https://twitter.com/TwitterGov/lists/uk-mps?lang=de))。

*   制作相应的 yaml 配置文件
*   看看什么样的信息可以用来识别政治家的政党。
*   用你的发现强化关键词
*   用一个用户输入问题增强主程序，类似于“你想分析哪个政府”。提供可用配置列表，然后根据用户选择运行程序。
*   想想分析每个国家的多个政治家名单所需要的改变。也就是说，我们希望区分每个国家的不同政府机构，并在配置文件中对其进行概括。

源代码可以在这里找到:[https://github.com/talfco/clb-sentiment](https://github.com/talfco/clb-sentiment)

*原载于 2019 年 2 月 3 日 dev.cloudburo.net*[](https://dev.cloudburo.net/2019/02/03/python-tutorial-government-social-media-analyser-class-build-out.html)**。**