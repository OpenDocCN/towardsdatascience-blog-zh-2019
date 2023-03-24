# 用脸书的快速文本(自然语言处理)介绍文本分类

> 原文：<https://towardsdatascience.com/natural-language-processing-with-fasttext-part-1-an-intro-to-text-classification-with-fasttext-11b9771722d8?source=collection_archive---------12----------------------->

## [快速文本系列](https://towardsdatascience.com/tagged/the-fasttext-series)

## 一个初学者友好的文本分类库。

![](img/2ea68b18ae9ab01591045bcc4c6d14a1.png)

> 最初发表于[我的博客](https://blog.contactsunny.com/data-science/an-intro-to-text-classification-with-facebooks-fasttext-natural-language-processing)。

文本分类是机器学习的一个非常常见的应用。在这样的应用中，机器学习被用于将一段文本分类成两个或多个类别。文本分类有监督学习模型和非监督学习模型。在这篇文章中，我们将看到如何使用脸书的快速文本库进行一些简单的文本分类。

由脸书开发的 fastText 是一个流行的文本分类库。该库是 GitHub 上的一个开源项目，非常活跃。该库还提供了用于文本分类的预建模型，包括监督和非监督的。在这篇文章中，我们将了解如何在库中训练监督模型进行快速文本分类。该库可以用作命令行工具，也可以用作 Python 包。为了让事情变得简单，在这篇文章中我们将只看几个 CLI 命令。

# 安装 fastText

为命令行安装 fastText 就像克隆 Git repo 并在目录中运行 *make* 命令一样简单:

```
git clone [https://github.com/facebookresearch/fastText.git](https://github.com/facebookresearch/fastText.git)
cd fastText
make
```

一旦你这样做了，你就已经安装了 fastText CLI，只要你没有得到任何错误。您也可以通过从同一目录运行以下命令来安装 Python 库:

```
pip install .
```

您可以通过运行以下命令来验证安装:

```
./fasttext
```

您应该在终端中看到类似这样的内容:

```
usage: fasttext <command> <args>The commands supported by fasttext are:supervised train a supervised classifier
 quantize quantize a model to reduce the memory usage
 test evaluate a supervised classifier
 test-label print labels with precision and recall scores
 predict predict most likely labels
 predict-prob predict most likely labels with probabilities
 skipgram train a skipgram model
 cbow train a cbow model
 print-word-vectors print word vectors given a trained model
 print-sentence-vectors print sentence vectors given a trained model
 print-ngrams print ngrams given a trained model and word
 nn query for nearest neighbors
 analogies query for analogies
 dump dump arguments,dictionary,input/output vectors
```

这表明您已经安装了该工具。下一步是获取我们的数据集。

# 获取数据

脸书开发者已经包含了一个测试这个库的数据集。所以我们会用同样的数据。这是一个关于烹饪的 stackexchange 问题集。这里的目的是对问题进行自动分类。因为我们使用监督学习，我们必须确保我们在数据中标记问题的类别。幸运的是，这些数据都带有已经标记的类别。所以先下载数据吧。数据在这里以压缩文件的形式提供:[https://dl . fbaipublicfiles . com/fast text/data/cooking . stack exchange . tar . gz](https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz)。我们可以下载数据并手动解压缩，也可以从 CLI 运行以下命令:

```
wget [https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz](https://dl.fbaipublicfiles.com/fasttext/data/cooking.stackexchange.tar.gz) && tar xvzf cooking.stackexchange.tar.gz
```

解压缩后，目录中会有一些文件。但是我们的数据在一个名为*cooking . stack exchange . txt*的文件里。如果你打开文件或者*头*它，你会看到这样的东西:

```
__label__sauce __label__cheese How much does potato starch affect a cheese sauce recipe?
__label__food-safety __label__acidity Dangerous pathogens capable of growing in acidic environments
__label__cast-iron __label__stove How do I cover up the white spots on my cast iron stove?
__label__restaurant Michelin Three Star Restaurant; but if the chef is not there
__label__knife-skills __label__dicing Without knife skills, how can I quickly and accurately dice vegetables?
__label__storage-method __label__equipment __label__bread What’s the purpose of a bread box?
__label__baking __label__food-safety __label__substitutions __label__peanuts how to seperate peanut oil from roasted peanuts at home?
__label__chocolate American equivalent for British chocolate terms
__label__baking __label__oven __label__convection Fan bake vs bake
__label__sauce __label__storage-lifetime __label__acidity __label__mayonnaise Regulation and balancing of readymade packed mayonnaise and other sauces
```

如你所见，这篇文章有些不寻常。在每一行中，我们都有*_ _ 标签 __* 文本。这其实就是问题的范畴。因此，在训练数据中指定类别的方法是包含文本 *__label__* ，后跟类别。我们也可以为一个问题指定多个类别，从上面的例子可以看出。现在我们已经准备好了数据，让我们把它分成训练和测试数据。

# 将数据分为训练数据和测试数据

在开始训练我们的模型之前，我们必须分割数据，以便我们有一个数据集用于训练模型，一个数据集用于测试模型的准确性。如果你想知道为什么这是必要的或者如何用 Python 来做，你可以在这里阅读我关于它的帖子[。通常，我们将数据分成 80-20 份，80%的数据用于训练，20%用于测试。要做到这一点，我们首先需要看看我们有多少问题。每行有一个问题，所以获取文件中的行数就可以了:](https://medium.com/@contactsunny/how-to-split-your-dataset-to-train-and-test-datasets-using-scikit-learn-e7cf6eb5e0d)

```
$ wc -l cooking.stackexchange.txt
15404 cooking.stackexchange.txt
```

正如我们从输出中看到的，文件中有 15404 行。其中 80%是 12323.2，所以我们将前 12324 行作为训练数据集。剩下的 3080 行将是我们的测试数据。为此，我们将运行以下命令:

```
$ head -n 12324 cooking.stackexchange.txt > training_data.txt
$ tail -n 3080 cooking.stackexchange.txt > testing_data.txt
```

我们现在将有两个新文件，一个用于培训，一个用于测试。接下来，我们将使用训练数据训练模型。

# 训练模型

这实际上是这个库的一个非常简单的命令。我们只需使用*监督的*命令运行 fastText CLI 工具，并提供输入文件(这是我们的训练数据文件)，以及将要生成的模型的名称。该命令如下所示:

```
./fasttext supervised -input training_data.txt -output cooking_question_classification_model
```

如您所见，这是一个非常容易理解的命令。*-输入*选项指定输入文件，而*-输出*选项指定将要生成的模型的名称。运行该命令后，您应该会得到类似如下的输出:

```
$ ./fasttext supervised -input training_data.txt -output cooking_question_classification_model
Read 0M words
Number of words: 14492
Number of labels: 735
Progress: 100.0% words/sec/thread: 47404 lr: 0.000000 avg.loss: 10.243105 ETA: 0h 0m 0s
```

我们的模型现在已经训练好了，可以回答一些问题进行分类了。让我们现在试试。

# 用一些问题测试我们的模型

当我们在上一步中训练我们的模型时，该命令生成了几个新文件:*cooking _ question _ class ification _ model . bin*和*cooking _ question _ class ification _ model . vec*。*。bin* 文件，或者模型的二进制文件，就是我们现在要用的。我们可以通过运行以下命令开始测试模型:

```
./fasttext predict cooking_question_classification_model.bin -
```

如您所见，我们使用 *predict* 命令告诉我们的模型，我们现在要做一些预测。命令末尾的破折号(-)表示我们将在命令行中键入问题。我们也可以给命令一个包含多个问题的文件，但是我们将在下一篇文章中讨论这个问题。一旦你运行这个命令，模型将开始监听问题，你可以输入一个问题，然后点击*回车*或*回车键*得到一个预测。现在让我们试试:

```
how to bake a cake
__label__baking
how to cook some rice
__label__food-safety
Does milk make cakes lighter or tougher?
__label__baking
How to prepare dried tortellini?
__label__food-safety
What exactly is a chowder?
__label__baking
How bad is it to freeze vegetables in glass for a maximum of 4 weeks?
__label__baking
```

我的每个问题都用文本 *__label__* 回答，后跟模型认为该问题所属的类别。正如你已经知道的，这个模型没有得到所有的答案或者正确的分类。这是意料之中的，因为我们还没有真正清理我们的数据或调整模型。我们可以通过对数据进行一些预处理来对模型进行微调，使其更加清晰，便于模型理解。

这个例子是您第一次尝试 fastText 时会遇到的。你可以登录 [fastText 网站](https://fasttext.cc/)获取更多关于这个库和这个教程的信息。因为我想把这篇文章限制在基础知识上，所以我不打算谈论验证模型或预处理我们的数据以获得更好的结果。

我们可以获得更好结果的另一种方法是使用 n 元语法，并将句子视为单词序列。一旦我们开始以这种方式看待单词，我们将能够更多地了解我们训练的数据中的模式，并更好地预测。我会写另一篇关于[什么是 n-gram](https://blog.contactsunny.com/data-science/understanding-word-n-grams-and-n-gram-probability-in-natural-language-processing) (因为它值得有一篇自己的帖子)以及我们如何使用概率来更好地理解我们的数据。

> 在 [Twitter](https://twitter.com/contactsunny) 上关注我，了解更多[数据科学](https://blog.contactsunny.com/tag/data-science)、[机器学习](https://blog.contactsunny.com/tag/machine-learning)，以及一般[技术更新](https://blog.contactsunny.com/category/tech)。另外，你可以关注我的个人博客。

如果你喜欢我在 Medium 或我的个人博客上的帖子，并希望我继续做这项工作，请考虑在 Patreon 上支持我。