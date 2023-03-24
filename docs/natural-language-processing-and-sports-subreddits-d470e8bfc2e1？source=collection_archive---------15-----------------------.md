# 自然语言处理和体育子编辑

> 原文：<https://towardsdatascience.com/natural-language-processing-and-sports-subreddits-d470e8bfc2e1?source=collection_archive---------15----------------------->

![](img/05cfa50310e4a7bc2ca632cffc8fce14.png)

如果你必须知道关于我的两件事，那就是我热爱运动和在 Reddit 上浪费我的空闲时间。如果你必须知道关于我的三件事，那就是我也喜欢发现新的应用自然语言处理(NLP)技术。因此，当被分配一个项目任务，通过从子数据集获取真实数据来探索我新发现的 NLP 技能时，我欣然接受了这个机会。

一段时间以来，Reddit 上关于体育的讨论让我特别感兴趣。俗话说，“凡事皆有子编辑”，所以自然 reddit 上的每个体育联盟都会有一个社区。从 NFL 和英格兰超级联赛，到 Overwatch 联盟和冰壶大满贯，几乎每个体育迷都有一个社区。他们每个人都有自己讨论体育的语言，更不用说他们自己的笑话和每周的话题讨论了。基于这些兴趣，我想进行一个项目来比较两个最大的体育子街道， [r/NFL](https://www.reddit.com/r/nfl/) 和 [r/NBA](https://www.reddit.com/r/nba) ，看看语言有什么不同，并使用 NLP 来预测一个帖子的来源，仅仅基于它的语言。

我将带您浏览数据、数据清理过程、矢量化、建模、预测，并评估模型在数据的一些有趣变化上的表现。我分析不同子主题中的语言的目的是确定哪些主题或趋势最能区分这些联盟的讨论。对每个模型的最重要的词特征的观察和每个模型的分类性能将用于得出我的结论。

# 数据是从哪里来的？

实际分析 Reddit 数据的第一步是…从 Reddit 获取数据。然而，这并不是一个非常困难的任务！幸运的是，Reddit 内置的 API 功能允许任何用户在网站上查询给定页面上热门帖子的. json 文件。例如，你可以现在就去 https://www.reddit.com/r/nba.json，你会看到。r/NBA 上前 25 个帖子的 json。然而，出于我的目的，我只需要提取每个帖子的标题和“自我文本”。

因此，我们需要多次遍历 subreddit 的页面来提取这些帖子。Python 的`requests`模块可以下载。json 文件直接放入代码中，准备好进行操作。然而，为了获得有用的数据，我需要多次这样做。此外，我想只提取这个的特征。在我的建模过程中很有价值的 json。我使用以下函数来完成这项任务:

使用函数的`num`参数，我告诉函数在页面中循环指定的次数。每个页面都有一个“after”参数，您可以从。json 文件，它引用页面上第一篇文章的文章 ID。您可以将这个参数传递到 URL 中，API 将为您提供在“after”ID 之后索引的所有帖子。

这个函数将返回一个相当大的熊猫 dataframe 对象。我们能够检索 Reddit 帖子的几乎所有功能，包括发布帖子的用户名、其 upvote 分数、任何附加图片的 URL、唯一的帖子 ID，当然还有帖子的标题。这是一个很大的数据量，但它现在能够被清理和建模！

# 计数矢量化与 TF-IDF

我们已经有了 reddit 帖子，现在我们该做什么呢？我将探索的 NLP 模型不能简单地解释出现在网站上的英语，我们需要在将其用于机器学习模型之前将其转换为数字数据。当**文件用数字表示时，计算机想要读取**文件。文档是 NLP 模型要评估的文本主体，在我们的例子中，是标题文本和“自身文本”主体的组合。

当文档可以用数字来解释时，计算机和程序员都更容易理解文档。为了对文档进行分类，每个文档都使用一个“*输入*”，并且类标签是“*输出”、*，在本例中是 r/NBA 或 r/NFL。大多数分类机器学习算法以数字向量作为输入，因此我们需要将 post 文档转换为固定长度的数字向量。

NLP 中的计数向量化允许我们将文本分割成句子和单词，收集文本文档并建立一个已知单词的词汇表。这是通过评估文档的整个**语料库**来完成的，获取在语料库的词汇表中找到的每个单词，并基于在单个文档中找到的词汇表为文档分配一个向量。

![](img/53722988218d289dde2420c4177015e3.png)

Word vectors, visualized. Source: [https://towardsdatascience.com/natural-language-processing-count-vectorization-with-scikit-learn-e7804269bb5e](/natural-language-processing-count-vectorization-with-scikit-learn-e7804269bb5e)

本质上，计数矢量器计算每个单词在文档中出现的次数，并根据语料库中已建立的词汇在矢量中为文档分配适当的值。

类似于计数矢量化，术语频率—逆文档频率，或 **TF-IDF，**查看整个语料库以形成其词汇。TF-IDF 的不同之处在于，它根据单词在文档中出现的频率对单词进行加权。这背后的逻辑是，如果一个单词在文档中多次出现*，我们应该提高它的相关性，因为它应该比出现次数较少的其他单词更有意义(TF)。然而，如果一个单词在一个文档中多次出现，而且*还出现在许多其他文档中*，则模型会降低该单词的权重，就好像它只是一个频繁出现的单词，而不是因为它是相关的或有意义的(IDF)。*

因此，现在我已经确定了如何预处理文档以便在模型中使用，我需要清理实际的文档本身。我使用正则表达式，或者说 **RegEx** ，在每个文档中找到要删除或替换的单词和字符的特定模式。计数矢量器和 TF-IDF 都将把互联网帖子中的杂乱条目(如 URL 和表情符号)解释为令牌，因此只向它们提供字母数字文本非常重要。例如，我用来删除 URL 的代码行如下所示:

```
sub['text'] = sub['text'].map(lambda x: re.sub(r"((http|ftp|https):\/\/)?[\w\-_]+(\.[\w\-_]+)+([\w\-\.,@?^=%&amp;:/~\+#]*[\w\-\@?^=%&amp;/~\+#])?", 
' ', 
x)
```

正如您所看到的，解释正则表达式并不简单，但它是清理文本的一个非常有效的工具。然而，RegEx 是非常通用的，因为它也被用在这个项目中来删除某些扭曲我的结果的单词、短语和字符序列。在这个 URL 清理器的例子中，它也在字符串中寻找独特的模式。

既然我们已经组装了矢量器，并清除了文本中不需要的特征，现在是建模的时候了！

# 建模！

在我的建模过程中，我的主要兴趣之一是确定在预测源的子编辑时，哪些单词具有更大的重要性，而不是简单地对某些预测单词进行计数。因此，虽然我确实利用了计数矢量化模型，但我发现使用 Scikit-Learn 的 TF-IDF 功能生成的加权矢量化结果`TfidfVectorizer`在评估每个模型时会产生更多信息。然而，为了便于理解，我将分享两个矢量器产生的结果。

我评估的第一个模型是多项式朴素贝叶斯(NB)模型，同时具有计数矢量器和 TF-IDF 矢量器。计数矢量化 NB 模型在 97%的情况下正确预测了帖子是否在 r/NFL，在 93%的情况下正确预测了帖子是否在 r/NBA。

```
# run the Count Vectorized model with the optimal hyperparameters
cvec = CountVectorizer(stop_words='english', ngram_range=(1, 2), max_features=20000, max_df=0.60)
train_raw = cvec.fit_transform(X_train)
train_df = pd.SparseDataFrame(train_raw, columns=cvec.get_feature_names())test_raw = cvec.transform(X_test)
test_df = pd.SparseDataFrame(test_raw, columns=cvec.get_feature_names())train_df.fillna(0, inplace=True)
test_df.fillna(0, inplace=True)# fit the multinomial NB model
cvec_nb = MultinomialNB()
cvec_nb.fit(train_df, y_train)
```

![](img/599d96aba8a2b550daa598aebbb90a0f.png)

Feature importances for the Count Vectorized NB Model, ranked by class probability.

在对来自 r/NBA 的帖子进行分类时，模型中最重要的词引用了自由球员的举动，如“威斯布鲁克”、“科怀”、“勒布朗”，但“交易”和“挑选”等加权术语也很重要。对来自 r/NFL 的帖子进行分类的最重要的词采取了不同的方向，其中“day”具有最高的计数，在关于实际足球比赛的讨论中发现的术语，如“round”、“yards”和“pass”，也非常重要。

TF-IDF 模型在 98%的时间里正确预测了一个帖子是否在 r/NFL，在 91%的时间里正确预测了 r/NBA。在对 r/NBA 的帖子进行分类时，模型中最重要的词再次引用了自由球员的举动，如“威斯布鲁克”、“科怀”和“勒布朗”。我还看到，在对来自 r/NFL 的帖子进行分类时，最重要的词是在关于实际足球比赛的讨论中发现的术语，如“highlight”、“yard”和“td”。

讨论我收集数据的时间很重要，因为很明显，尽管两个数据集都是在各自运动的休赛期发布的，但两个子数据集的侧重点是不同的。大多数帖子都是在 2019 年 7 月 8 日至 2019 年 7 月 12 日这一周被刮下来的，正如 NBA 球迷可能记得的那样，这是相当疯狂的一周。当 NBA 超级巨星拉塞尔·维斯特布鲁克、凯文·杜兰特、凯里·欧文和科怀·伦纳德决定加入新球队时，他们都在 NBA 世界产生了冲击波，这反映在我正在分析的数据中。这并不是说 r/NFL 或 NFL 媒体不讨论自由球员的转会，但至少 r/NBA 的文化是主要关注这项运动中的人物，而不是这项运动本身。

知道了这一点，我想看看不同的分类算法如何衡量每个文档的特征，以及我是否可以提高我的准确性指标。因此，我决定通过一个**支持向量分类器(SVC)** 运行这个相同的数据集。SVC 是非常强大的分类模型，它试图使用一个“分离超平面”将一个数据集分类到有标签的类中，例如 origin 的 subreddit。换句话说，给定一个二维空间，这个超平面是一条将平面分成两部分的线，其中每一类位于两侧。在支持向量机之下还有很多事情要做，但是出于我的目的，使用了 SVC，因为它是一个全面而强大的建模工具。

因此，我对从 r/NBA 和 r/NFL 收集的数据运行了 TF-IDF 矢量器和 SVC。该模型的得分不如多项式 NB 模型高，但仍然表现得非常好，发布了 93.14%的准确度得分和 93%的加权精度。带有我选择的超参数的模型如下:

```
# fit tf-idf svc model for best hyperparameters
tfidf = TfidfVectorizer(max_features=20000, ngram_range=(1, 1), stop_words='english', max_df=.6)
train_raw = tfidf.fit_transform(X_train)
test_raw = tfidf.transform(X_test)svc = SVC(C = 1, kernel='linear', gamma = 'auto', probability=True)
svc.fit(train_raw, y_train)
```

![](img/353ca074db301feaf13ef824f3aa24af.png)

The weighted coefficients from the SVC model.

该模型选择的重要特征类似于 NB 分类器，但倾向于将专有名词如球员姓名、球队名称甚至记者的权重更高。然而，与主题相关的更一般的术语，如“篮球”和“足球”，也被赋予了更大的权重。

# 大学体育呢？

在预测涵盖不同运动的两个子主题的帖子来源时，我没有看到模型有太大的差异，所以我想给我的建模过程一个新的挑战。我想确定我是否可以创建一个模型来确定一个帖子的来源，这些帖子来自覆盖同一项运动，但在不同的联盟。

如果你需要知道关于我的四件事，那就是我热爱运动，在 Reddit 上浪费我的空闲时间，发现 NLP 的新应用，为 UVA 篮球队加油。因此，很自然地，我想以某种方式将[r/学院篮球](https://www.reddit.com/r/collegebasketball/)纳入这个分析*。*

使用与我收集 r/NFL 和 r/NBA 帖子相同的收集技术，我从 r/CollegeBasketball 收集帖子，清理文本，并将这些帖子与 r/NBA 帖子一起组合在 Pandas 数据框架中。我通过 SVC 模型运行了这些帖子，结果让我有些吃惊。

![](img/d83e2b2935bcb7ecdf5c26b09422f40e.png)

The SVC model, this time weighting the words from r/collegebasketball posts alongside r/nba.

这个模型表现很好，发布了 91.39%的准确度分数和 91%的加权准确度，但它显然没有根据 r/NFL 的内容对 r/NBA 的帖子进行分类时那么准确。这个模型倾向于给专有名词，比如球员名字和球队名字赋予更高的权重。“锦标赛”、“转会”和“提交”等术语在 r/CollegeBasktball 中非常重要，因为它们在关于 NCAA 篮球的讨论中比在 r/NBA 中使用得更广泛。然而，我们发现像“卡利帕里”这样的教练的名字比球员的名字更重要。一些有趣的区别是“basketball”和“gif ”,这两个词在任一子编辑中都很常见，但 r/CollegeBasketball 的权重更大。

# 结论

对所有迭代的子网格数据进行分类的所有模型得分都很高，但计数矢量化多项式朴素贝叶斯在训练和测试数据中表现最佳且最一致。支持向量机模型的表现不如 NB 模型，但让我更好地理解了哪些词/术语在分类帖子时是至关重要的。

在这个项目中分析的所有模型中，他们能够让我理解在分类子编辑之间的帖子时什么类型的内容是重要的。对于 r/NBA 来说，很明显，球员姓名和交易是帖子中非常频繁的话题，尤其是考虑到数据拍摄的时间。对于 r/NFL 来说，主题更倾向于面向团队，但是像“td”和“yard”这样的游戏专用术语在所有模型中都很重要。对于 r/CollegeBasketball 来说，很明显学校的骄傲和竞争是帖子主题中出现最多的，因为某些学校的名字权重最大，而教练往往比个人球员更经常进行讨论。

作为一个经常光顾这三个子街道的人，我非常好奇这个项目的结果会告诉我什么。不仅看到每个模型如何解释来自每个子编辑的帖子特别有趣，而且我能够结合我自己对每个社区中讨论的主题语言的知识以及模型来解释结果。虽然像单词云这样的东西是可视化 subreddit 语言选择的一种简洁方式，但对帖子进行矢量化和分类让我对每个社区如何讨论这项运动有了新的认识。

如果你想探索这个项目和收集的数据，Temple 已经上传了所有相关的 Jupyter 笔记本和。csv 文件在他的公共 [*GitHub*](https://github.com/templecm4y/project-reddit-nlp) *上。*