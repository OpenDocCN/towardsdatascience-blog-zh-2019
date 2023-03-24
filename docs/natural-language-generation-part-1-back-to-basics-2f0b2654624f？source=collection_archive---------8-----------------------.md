# 自然语言生成第 1 部分:回归基础

> 原文：<https://towardsdatascience.com/natural-language-generation-part-1-back-to-basics-2f0b2654624f?source=collection_archive---------8----------------------->

![](img/52d891a73dbdd49a36308a416cea0d2e.png)

你有没有遇到过那些脸书或推特上的帖子，显示一个人工智能被“强迫”看电视或看书的输出，它得出的新输出与它看到或读到的相似？它们通常非常有趣，并不完全遵循某人实际上是如何说或写的，但它们是自然语言生成的例子。NLG 是 ML 的一个非常有趣的地区，在这里玩玩，想出你自己的模型会很有趣。也许你想做一个瑞克和莫蒂星际迷航穿越脚本，或者只是创建听起来像另一个人的推文的推文。

随着研究和硬件的进步，使用 ML 生成文本、图像和视频变得越来越普遍。随着基于深度学习的系统的最新进展，例如 OpenAI 的 GPT-2 模型，我们现在看到语言模型可以用于从大量其他示例中生成非常真实的声音文本。我对构建一个以另一种风格或人物的风格生成假文本的系统很感兴趣，所以我决定专注于学习不同的 ML 方法，并概述我使用这些不同技术所学到的东西。

**回归基础**

不要直接跳到花哨的深度学习技术，让我们看看一种非常容易理解和容易实现的技术作为起点。多年来，语言生成最常用的方法之一是 ***马尔可夫链*** ，这是一种非常简单的技术，但却惊人地强大。马尔可夫链是一种随机过程，用于在只给定前一事件的情况下描述序列中的下一事件。这很酷，因为这意味着我们真的不需要跟踪序列中所有先前的状态来推断下一个可能的状态。

在我们的例子中，状态将是前一个单词(一元)或 2 个单词(二元)或 3 个单词(三元)。这些通常被称为 ngrams，因为我们将使用最后 n 个单词来生成序列中的下一个可能的单词。马尔可夫链通常通过概率加权来选择下一个状态，但在我们的例子中，这只会创建在结构和单词选择上过于确定的文本。你可以考虑概率的权重，但是真正的随机选择有助于让生成的文本看起来更有创意。

**建立语言模型**

创建语言模型相当简单。首先，我们需要一些示例文本作为我们的语料库来构建我们的语言模型。它可以是任何类型的文本，如书籍段落，推文，reddit 帖子，你能想到的。在我的语料库中，我使用了多位总统的总统演讲。

一旦我们收集了文本，我们需要做以下步骤。注意，我将标记“#END#”添加到我的语言模型中，以便于确定任何示例语音中的结束状态。

*   对文本进行标记
*   对令牌做任何预处理(在我的例子中没有)
*   从令牌生成 ngrams
*   创建从 ngram 到文本中找到的下一个可能单词的映射

为了更好地理解这个模型是如何构建的，让我们看一个超级简单的例子。假设我们有一个简单的文本“狗跳过月亮。狗很有趣。”。使用上面的过程，我们将生成下面的语言模型。

```
(The, dog)     -> [jumped, is]
(dog, jumped)  -> [over]
(jumped, over) -> [the]
(over, the)    -> [moon.]
(dog, is)      -> [funny.]
(is, funny)    -> [#END#]
```

一旦我们完成了 ngram 映射，这个模型就可以用来生成一些新的文本了。现在，通过传入一个示例 ngram 的种子(如“The dog ”),或者让系统从提取的 ngram 密钥中随机选择一个起始点，就可以相当容易地做到这一点。一旦我们选择了一个种子 ngram，我们就从可能的单词列表中随机选择下一个单词，然后从目前生成的文本中选择下一个 ngram，并选择下一个状态等等…

**简单的 Python 马尔可夫链**

既然我们已经从概念上了解了它是如何工作的，那么让我们来看看训练和生成文本的完整代码。下面是我从网上其他例子中拼凑的 python 脚本，它用 python 构建了一个基本的马尔可夫模型。

让我们更深入地研究一些代码。它主要只有两个功能。学习功能和生成功能。让我们首先看一下 learn 函数，它从一系列大小为 n 的记号和 n 元语法中构建模型。

```
def learn(self,tokens,n=2):
        model = {}

        for i in range(0,len(tokens)-n):
            gram = tuple(tokens[i:i+n])
            token = tokens[i+n]

            if gram in model:
                model[gram].append(token)
            else:
                model[gram] = [token]

        final_gram = tuple(tokens[len(tokens) - n:])
        if final_gram in model:
            model[final_gram].append("#END#")
        else:
            model[final_gram] = ["#END#"]
        self.model = model
        return model
```

我们从从第一个标记到列表长度减 n 的循环开始。随着我们的进行，我们建立了在标记化文本中找到的相邻单词的 ngrams 字典。在循环之后，我们在输入文本的最后 n 个单词之前停止，并创建最终的令牌变量，然后用“#END#”将其添加到模型中，以表示我们已经到达了文档的末尾。

我要指出的这种方法的一个局限性是，我将所有文本放在一个列表中，所以我们实际上只有一个结束状态。进一步的改进是我们处理的每个文档都有结束状态，或者可以更进一步，在句子的末尾添加结束状态，这样我们可以更好地知道什么时候开始一个新的句子等等。接下来我们有生成函数。

```
def generate(self,n=2,seed=None, max_tokens=100):
        if seed is None:
            seed = random.choice(list(self.model.keys()))

        output = list(seed)
        output[0] = output[0].capitalize()
        current = seed

        for i in range(n, max_tokens):
            # get next possible set of words from the seed word
            if current in self.model:
                possible_transitions = self.model[current]
                choice = random.choice(possible_transitions)
                if choice is "#END#" : break

                if choice == '.':
                    output[-1] = output[-1] + choice
                else:
                    output.append(choice)
                current = tuple(output[-n:])
            else:
                # should return ending punctuation of some sort
                if current not in string.punctuation:
                    output.append('.')
        return output
```

生成新文本的代码接受我们训练的 ngrams 的大小，以及我们希望生成的文本有多长。它还接受一个可选的种子参数，如果没有设置，它将从模型中学习的可能的 ngrams 中随机选取一个起始种子。在循环的每次迭代中，我们查看前一个 ngram，并随机选择下一个可能的过渡词，直到我们到达结束状态之一或到达文本的最大长度。

下面是一个使用二元模型作为语言模型的脚本输出示例。

```
Us from carrying out even the dishonest media report the facts! my hit was on the 1st of december, 1847, being the great reviews &amp; will win on the front lines of freedom. we are now saying you will never forget the rigged system that is what we do at a 15 year high. i can perceive no good reason why the civil and religious freedom we enjoy and by the secretary of war would be 0.0 ratings if not.
```

如你所见，这很有道理，但听起来也有点随意。如果我们尝试使用三元语法语言模型呢？

```
Was $7,842,306.90, and during the seven months under the act of the 3d of march last i caused an order to be issued to our military occupation during the war, and may have calculated to gain much by protracting it, and, indeed, that we might ultimately abandon it altogether without insisting on any indemnity, territorial or otherwise. whatever may be the least trusted name in news if they continue to cover me inaccurately and with a group, it’s going to be open, and the land will be left in better shape than it is right now. is that right? better shape. (applause.)  we declined to certify the terrible one-sided iran nuclear deal. that was a horrible deal. (applause.) whoever heard you give $150 billion to a nation
```

嗯，这听起来好多了…但是等一下，当我深入研究样本语料库时，我注意到它从语料库中提取了大量的文本。在现实中，除非你有大量的数据来构建，否则一旦你开始使用三元模型或更高的模型，大多数模型都会表现出这种行为。这是马尔可夫模型方法的一个缺点。虽然 bigram 模型听起来更随机，但似乎每次运行都会生成相当独特的输出，并且不会从语料库中提取文本部分。

**火花分布马尔可夫链**

比方说，我们创建了一个更大的数据集，从整个子数据集或多年的推文中提取数据。如果我们想让 python 脚本运行足够长的时间，并在我们的机器上有足够的内存，那么它可能可以处理数据集，但最终它可能不容易扩展。让我们做同样的事情，但是使用 Apache Spark 并使用它的分布式计算能力来构建和存储模型。下面是基于 spark 的模型的完整代码，我们也将深入挖掘它的操作。

最大的障碍是试图找出如何在 Spark 中生成 ngram 模型，创建类似字典的结构并对其进行查询。幸运的是，Sparks mllib 已经在框架中内置了 ngram 特征提取功能，因此 park 已经被处理好了。它只是接收一个 Spark dataframe 对象，即我们的标记化文档行，然后在另一列中将 ngrams 输出到一个新的 dataframe 对象。

```
ngram = NGram(n=self.n, inputCol='tokenized_text',outputCol='ngram')
ngram_df = ngram.transform(text_df)
```

使用 Sparks ngram 模块，让我创建一个函数来映射数据帧中的每一行，并处理文本以生成每个 ngram 的相邻单词。这是在 Spark 进程的第一个 map 调用中调用的函数。它的目标是只获取文档的 ngram 列表，并以[(ngram，adjacent term)]的形式循环生成每个文档的元组列表。现在，列表中可能会有重复的(ngram，相邻项)元组。

```
def generate_adjacent_terms(ngrams):
    adjacent_list = []
    for i in range(0, len(ngrams)):
        if(i == len(ngrams) - 1):
            adjacent_tuple = (ngrams[i], "#END#")
            adjacent_list.append(adjacent_tuple)
        else:
            adjacent_tuple = (ngrams[i], ngrams[i+1].split(" ")[-1])
            adjacent_list.append(adjacent_tuple)
    return adjacent_list
```

平面映射将所有元组列表放入一个平面 rdd 中，而不是每个 rdd 元素都是来自每个文档的列表。下一个映射是设置 reduceByKey，因此我们获取每个元素并将其修改为一个(ngram，list object)元组，然后可以使用该元组将 ngram 键组合在一起，最终以(ngram，[相邻术语列表])的形式创建模型。重要的是要注意邻接表中会有重复的术语。我保留了这些副本，作为我们算法随机选择特定下一个状态的可能性的加权。

```
self.ngram_model = ngram_df.rdd.map(lambda x: PreProcess
    .generate_adjacent_terms(x.asDict()['ngram'])) \
    .flatMap(lambda xs: [x for x in xs]) \
    .map(lambda y: (y[0], [y[1]])) \
    .reduceByKey(lambda a, b: a + b)
```

下面是我用来创建基于 Spark 的马尔可夫链代码的完整代码。文本生成逻辑与其他脚本非常相似，只是我们不是查询字典，而是查询 rdd 来获取序列中的下一个术语。实际上，这很可能是在 api 调用之后，但是现在我们可以直接调用 rdd。

Spark 代码将生成与第一个 python 脚本类似的输出，但是理论上，当在集群上运行大型数据集时，应该可以更好地扩展。

您可以随意查看代码和脚本，并让我知道您的想法/我应该研究哪些改进！[https://github.com/GeorgeDittmar/MarkovTextGenerator](https://github.com/GeorgeDittmar/MarkovTextGenerator)。下一轮工作是能够保存模型以用新数据扩展它们，以及添加更多的示例脚本供人们使用。

感谢阅读！