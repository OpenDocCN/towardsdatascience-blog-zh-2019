# 优化快速文本模型以提高准确性

> 原文：<https://towardsdatascience.com/optimising-a-fasttext-model-for-better-accuracy-739ca134100a?source=collection_archive---------16----------------------->

## [快速文本系列](https://towardsdatascience.com/tagged/the-fasttext-series)

## 理解精确度和召回率。

![](img/4e702866519ed8a6d1e0a19fd614ef73.png)

> 最初发表于[我的博客](https://blog.contactsunny.com/data-science/optimising-a-fasttext-model-for-better-accuracy)。

在我们之前的帖子中，我们看到了[什么是 n-grams 以及它们如何有用](https://blog.contactsunny.com/data-science/understanding-word-n-grams-and-n-gram-probability-in-natural-language-processing)。在那篇文章之前，我们使用脸书的 fastText 库构建了一个[简单的文本分类器。在本帖中，我们将看到如何优化模型以获得更好的准确性。](https://blog.contactsunny.com/data-science/an-intro-to-text-classification-with-facebooks-fasttext-natural-language-processing)

# 精确度和召回率

为了更好地理解模型的准确性，我们需要知道精确度和召回率这两件事。而且这两件事也不是很难理解。Precision 是由 fastText 模型预测的正确标签数，recall 是正确标签中成功预测的标签数。这可能有点令人困惑，所以让我们看一个例子来更好地理解它。

假设一句话，我们给模型分类，从我们的栈交换烹饪样本，当然，模型预测标签食品安全，烘焙，设备，替代品和面包。堆栈交换中的实际标签是设备、清洁和刀具。这里，在模型预测的前五个标签中，只有一个是正确的。所以精度变成了 1 / 5，或者 0.20。另外，在三个正确的标签中，模型只正确预测了一个标签(设备)，因此召回率为 1 / 3 或 0.33。这就是精确和回忆的含义。

有一种方法可以让我们在 fastText 中使用一个简单的命令来测试模型的精度和召回率。在这一点上，确保你已经阅读了我之前写的[快速文本介绍文章](https://blog.contactsunny.com/data-science/an-intro-to-text-classification-with-facebooks-fasttext-natural-language-processing)，因为我将在这里使用那篇文章中的同一个例子。假设您已经这样做了，并且希望您已经从那篇文章中获得了数据，我们将在根目录中运行以下命令来获得我们的精度并召回数据:

```
./fasttext test cooking_question_classification_model.bin testing_data.txt
```

运行此命令后，您应该会得到类似于以下内容的输出:

```
N 3080
P@1 0.139
R@1 0.0602
```

正如您从输出中看到的，我们得到了 *P@1* 和 *R@1* 的结果，结果是精度为 1，召回为 1。我们将在这篇文章中看到如何改进这些。

# 清理数据

如果你看我们的数据文件，你可以看到有一些大写字母。这些对于我们的模型来说并不重要，我们可以去掉它们来在一定程度上提高性能。但是我们不能检查所有的数据并清理它们。所以我们将使用一个简单的命令将所有大写字母转换成小写字母。为此，请运行以下命令:

```
cat cooking.stackexchange.txt | sed -e “s/\([.\!?,’/()]\)/ \1 /g” | tr “[:upper:]” “[:lower:]” > cooking.preprocessed.txt
```

在这个命令中，我们使用 *cat* 将数据打印到标准输出，使用管道将数据重定向到 *sed* 命令，对输入数据运行正则表达式，然后使用另一个管道将这个新输出运行到 translate 命令，将所有大写字母转换为小写字母。我们将这个最终输出重定向到一个名为“cooking.preprocessed.txt”的文件中，这也是官方 fastText 网站上提供的一个简单示例。在真实的生产场景中，这可能不是一个简单的任务。无论如何，一旦我们有了这个新的预处理文件，让我们看看它有什么。

```
➜ head cooking.preprocessed.txt
__label__sauce __label__cheese how much does potato starch affect a cheese sauce recipe ? 
__label__food-safety __label__acidity dangerous pathogens capable of growing in acidic environments
__label__cast-iron __label__stove how do i cover up the white spots on my cast iron stove ? 
__label__restaurant michelin three star restaurant; but if the chef is not there
__label__knife-skills __label__dicing without knife skills , how can i quickly and accurately dice vegetables ? 
__label__storage-method __label__equipment __label__bread what ‘ s the purpose of a bread box ? 
__label__baking __label__food-safety __label__substitutions __label__peanuts how to seperate peanut oil from roasted peanuts at home ? 
__label__chocolate american equivalent for british chocolate terms
__label__baking __label__oven __label__convection fan bake vs bake
__label__sauce __label__storage-lifetime __label__acidity __label__mayonnaise regulation and balancing of readymade packed mayonnaise and other sauces
```

如你所见，数据现在清晰多了。现在，我们不得不再次[分割它来测试和训练数据集](https://blog.contactsunny.com/data-science/how-to-split-your-dataset-to-train-and-test-datasets)。为此，我们将运行以下两个命令:

```
➜ head -n 12324 cooking.preprocessed.txt > preprocessed_training_data.txt
➜ tail -n 3080 cooking.preprocessed.txt > preprocessed_testing_data.txt
```

我们必须根据这些新数据再次训练我们的模型，因为我们已经更改了数据。为此，我们将运行以下命令，输出应该类似于您在这里看到的内容:

```
➜ ./fasttext supervised -input preprocessed_training_data.txt -output cooking_question_classification_model
Read 0M words
Number of words: 8921
Number of labels: 735
Progress: 100.0% words/sec/thread: 47747 lr: 0.000000 avg.loss: 10.379300 ETA: 0h 0m 0s
```

为了检查精确度和召回率，我们将在新的测试数据上测试这个模型:

```
➜ ./fasttext test cooking_question_classification_model.bin preprocessed_testing_data.txt
N 3080
P@1 0.171
R@1 0.0743
```

如你所见，准确率和召回率都有所提高。这里要注意的另一件事是，当我们用新数据训练模型时，我们只看到 8921 个单词，而上次，我们看到了 14492 个单词。因此，由于大写和小写的差异，该模型具有相同单词的多个变体，这在一定程度上降低了精确度。

# 纪元

如果你有软件开发背景，你就知道 epoch 和时间有关系。你是对的。在这个上下文中，epoch 是模型看到一个短语或一个示例输入的次数。默认情况下，模型会将一个示例查看五次，即 epoch = 5。因为我们的数据集只有大约 12k 的样本，少了 5 个时期。我们可以使用- *ecpoch* 选项将它增加到 25 次，使模型“看到”一个例句 25 次，这可以帮助模型更好地学习。现在让我们试试:

```
➜ ./fasttext supervised -input preprocessed_training_data.txt -output cooking_question_classification_model -epoch 25
Read 0M words
Number of words: 8921
Number of labels: 735
Progress: 100.0% words/sec/thread: 43007 lr: 0.000000 avg.loss: 7.383627 ETA: 0h 0m 0s
```

你可能已经注意到，现在完成这个过程花了一点时间，这是我们增加纪元时所期望的。无论如何，现在让我们测试我们的模型的精度:

```
➜ ./fasttext test cooking_question_classification_model.bin preprocessed_testing_data.txt
N 3080
P@1 0.518
R@1 0.225
```

如你所见，我们在精确度和召回率上有了显著的提高。那很好。

# 算法的学习速率

算法的学习率表示在每个例句被处理后模型改变了多少。我们既可以提高也可以降低算法的学习速度。学习率为 0 表示学习没有变化，或者变化率正好为 0，所以模型根本没有变化。通常的学习率是 0.1 比 1。对于我们这里的例子，我们将保持学习率为 1，并重新训练我们的模型。为此，我们将使用 *-lr* 选项:

```
➜ ./fasttext supervised -input preprocessed_training_data.txt -output cooking_question_classification_model -lr 1.0 
Read 0M words
Number of words: 8921
Number of labels: 735
Progress: 100.0% words/sec/thread: 47903 lr: 0.000000 avg.loss: 6.398750 ETA: 0h 0m 0s
```

我们将再次测试模型，看看改变学习率后是否有任何改进:

```
➜ ./fasttext test cooking_question_classification_model.bin preprocessed_testing_data.txt
N 3080
P@1 0.572
R@1 0.248
```

绝对有进步。但是，如果我们一起增加纪元和学习率，会发生什么呢？

# 一起增加纪元和学习率

现在，我们将保持纪元为 25，学习速率为 1。让我们看看精确度和召回率会发生什么变化:

```
➜ ./fasttext supervised -input preprocessed_training_data.txt -output cooking_question_classification_model -epoch 25 -lr 1.0 
Read 0M words
Number of words: 8921
Number of labels: 735
Progress: 100.0% words/sec/thread: 41933 lr: 0.000000 avg.loss: 4.297409 ETA: 0h 0m 0s
```

现在让我们测试模型:

```
➜ ./fasttext test cooking_question_classification_model.bin preprocessed_testing_data.txt
N 3080
P@1 0.583
R@1 0.253
```

我们可以很容易地看到这里的改进。

所以，我们在这个帖子中学到了很多(我希望)。不过还有更多事情要做(比如 [n-grams](https://blog.contactsunny.com/data-science/understanding-word-n-grams-and-n-gram-probability-in-natural-language-processing) )。我们将在以后的帖子中看到这一点。如果你对此有任何补充，请在下面留下评论。

> 在[推特](https://twitter.com/contactsunny)上关注我，了解更多[数据科学](https://blog.contactsunny.com/tag/data-science)、[机器学习](https://blog.contactsunny.com/tag/machine-learning)，以及通用[技术更新](https://blog.contactsunny.com/category/tech)。另外，你可以[关注我的个人博客](https://blog.contactsunny.com/)。