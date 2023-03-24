# 查询分段和拼写纠正

> 原文：<https://towardsdatascience.com/query-segmentation-and-spelling-correction-483173008981?source=collection_archive---------17----------------------->

在英语中，人们通常输入由空格分隔的查询，但有时不知何故，这个空格被无意中遗漏了。找到单词边界的方法被称为**查询分割**。例如，我们可以很容易地破译

> 坚果巧克力

如同

> 无坚果巧克力

但是，机器不能，除非我们教他们。这里有一个证据:

![](img/294f56337e8b87d65d07355f842f22f1.png)

Showing results of an unsegmented search query

因为我们没有失去我们的能力，我们的大脑不知何故直觉地和无意识地这样做。但是，在搜索领域，它会影响没有正确分割的特定查询的结果的精确度和召回率。让我们看看如何:

1.  **电子商务中的搜索引擎**特别是基于倒排索引的，应该有索引的查询或者[分析器](https://lucene.apache.org/solr/guide/6_6/analyzers.html#analyzers)【比如子字符串等等。]必须在索引链接到特定产品列表的产品属性时使用。但是，问题依然存在，这主要与查准率和查全率有关。如果我们将上述查询分为三部分，那么我们可以对三个查询词进行布尔搜索[使用 AND],即“nut”、“free”和“chocolates ”,结果是三个结果交集的产品列表，即只返回三个列表中包含的产品。因此提高了精度。
2.  该查询也可能拼写错误

> 例如“nutfreechacolatas”或“skommedmilk”

在这里，我们可以看到巧克力的拼写是错误的。但是，为了纠正这一点，前提条件是识别分段，即单词边界，以便首先为**拼写纠正**算法提供输入。

![](img/c0eab3e41e37521d27ac08b951841489.png)

No Results for an unsegmented query having misspelled words

3.**在基于机器学习 NLP 的** [**查询理解**](https://medium.com/@sonusharma.mnnit/understanding-the-search-query-part-i-632d1b323b50) **，**我们需要将正确分段和正确拼写的查询馈送到模型中，以便在找到搜索查询的意图时获得更好的准确性。我实现这些算法主要就是为了这个目的。

这里，我们看到拼写校正算法依赖于分词。所以，让我们先看看如何进行细分:

通过以几种方式分割字符串，可以实现查询分割。但是，等等，通过使用这种天真的方法，可以存在多少种可能的不同组合。实际上，它是指数的，对于长度为 n 的单词，它是{ **2^(n−1)}** 可能的组合，因为对于给定的字符串将有(n-1)个边界。查询的有效和真实组合的最终列表将基于该组合是否由有效单词组成来决定，并且基于查询中最可能的单词组合(即具有最高概率的组合)来排序或分类。稍后将解释如何计算每个组合的概率。现在，使用这种简单的方法，查询“nutfreechocolates”(长度，n = 17)有 2 个⁶ = 65，536 个组合。随着查询长度的增加，它将无法扩展，并且需要大量的计算能力。为了更高效，我分析了两种算法:

# **动态编程方法:**

在朴素方法中，我们做了一些类似的计算，一次又一次地寻找查询的前缀的最可能的段，该段可以被记忆以改进从指数(O(2^n)到多项式函数(O(n))的复杂性。我们通过一个例子来了解一下:

```
For query: **“isit” [Total Segment = 2^3 = 8]** It can be either segmented in 2 parts or 3 parts or 4 partsStart with 4: [Possible composition = 3C3 = 1]
"i" "s" "i" "t = P(i|'')*P(s|i)*P(i|t)*P(t|i)Start with 3: [Possible composition = 3C2 = 3]
"i" "s" "it" = P(i|'')*P(s|i)*P(it|s)
"is" "i" "t" = P(is|'')*P(i|is)*P(t|i)
"i" "si" "t" = P(i|'')*P(si|i)(P(t|si)"Start with 2: [Possible composition = 3C1 = 3] 
"i" "sit" = P(i|'')*P(sit|i)
"is" "it" = P(is|'')*P(it|is)
"isi" "t" = P(isi|'')*P(t|isi)where P(A|B) is conditional probability of happening of A given that B has already occured. The bigram probability is introduced for more context. Unigram probability can also be used to find the probability of composition.
```

这里，通过使用蛮力方法计算了 8 个组成。为了使用 DP 避免这种情况，每次在我们的程序进行递归调用来分割剩余的字符串之前，我们都要检查这个精确的子串以前是否被分割过。如果是这样，我们只需消除递归调用，并从缓存中检索该子串的最佳分割结果(在特定子串第一次被分割时存储在缓存中)。

在计算每个单词的概率得分时，我们假设这个单词是有效的，并且存在于字典中(如果这个单词无效，那么我们需要在这里应用拼写纠正)。一个单词的分数将是:

> p(A)= freq(A)/freq 之和(所有字)
> 
> 而 P(composition)= log(P(word 1))+log(P(word 2))+…其中组合由单词 1、单词 2 等组成。
> 
> 因为，合成的朴素贝叶斯概率将是独立单词概率的乘积，即 P(AB) = P(A) * P(B)

如果这个词在字典中不存在，那么按照[http://norvig.com/ngrams/ch14.pdf](http://norvig.com/ngrams/ch14.pdf)

> P(A) = 10 /(频率(所有字的总和)* 10^len(A))

代码可以在这里找到:[https://github.com/grantjenks/python-wordsegment](https://github.com/grantjenks/python-wordsegment)

# 三角矩阵法；

代替记忆查询的前缀的所有组成，这里只有具有最高概率的前缀组成将被较低概率覆盖。这种方法在这里得到了更好的解释:[https://towardsdatascience . com/fast-word-segmentation-for-noise-text-2c 2c 41 f 9 E8 da](/fast-word-segmentation-for-noisy-text-2c2c41f9e8da)

S pelling 校正也可以使用许多方法来完成。但是，它们可能非常昂贵。在这里，我将讨论除简单方法之外的两种优化方法:

1.  [Peter Norvig 方法:](https://norvig.com/spell-correct.html)它包括通过对每次迭代中提供的查询使用插入、删除、替换和转置操作来生成 n 个编辑距离之外的候选项。然后，语言模型，其中如果在字典中找到，则计算在第一步中生成的可能候选的得分/概率，以及错误模型，其中字典中没有找到的单词将经历插入、删除、替换和换位操作的又一次迭代，并且将再次应用语言模型。但是，这是非常昂贵的，因为每次编辑生成的候选列表对于每个单词都会增加(54 *n* +25)倍。对于我们示例中的查询“nutfreechacolatas ”,长度为 17。因此，在计算编辑的第一次迭代中，将有 54*17+25 = 943 个候选项。同样，在每个候选的第二次迭代中，可能的候选将是 54*943+25 = 50，947。因此，它不会随着编辑距离的增加和查询时间的延长而扩展。
2.  [使用对称删除的拼写校正【sym spell】](/symspellcompound-10ec8f467c9b):这里，通过在在线处理查询之前使用预处理，已经优化了复杂度。在生产场景中，预处理任务需要时间，并且依赖于字典的大小，因此我们可以创建拼写纠正类的单例类，该类在第一次尝试创建字典的对象时加载字典。主要思想是通过**仅从字典中删除查询中的字符**以及要被纠正的给定查询并在中间相遇来生成候选。这里的四个操作将按以下方式处理:

a)删除和插入:从字典中删除一个字符相当于在给定的查询中添加一个字符，反之亦然。

> 删除(字典条目，编辑距离)=输入条目
> 
> 字典条目=删除(输入条目，编辑 _ 距离)

b)替换和换位:从字典和给定查询中删除一个字符相当于替换，从字典和给定查询中删除两个连续字符相当于换位操作。

> 删除(字典条目，编辑距离)=删除(输入条目，编辑距离)

在这种方法中，我们需要计算由给定查询的删除操作生成的候选词和也由删除操作形成的词典词之间的**编辑距离**，因为编辑距离可能在两侧同时删除期间改变。比如对于单词:bank，经过预处理后，在字典中，bank = ban[编辑距离= 1]并且如果给定查询=“xban”。关于给定查询的删除，xban = ban[编辑距离= 1]，但是 bank！=xban，因为两者之间的编辑距离为 2。

有几种计算编辑距离的方法:

1.  汉明距离:它只允许替换，因此，它只适用于相同长度的字符串。
2.  [最长公共子序列(LCS)](https://en.wikipedia.org/wiki/Longest_common_subsequence_problem) :只允许插入和删除。
3.  [Levenshtein 距离:](https://en.wikipedia.org/wiki/Levenshtein_distance)允许插入、删除、替换，但不允许换位。即转换 AC- > CA 将被替换计为 2 编辑距离。
4.  [Damerau Levenshtein 距离:](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)它允许我们正在进行的所有四种操作，以便生成候选。实际上，这种算法有两种变体:a)受限 Damerau-Levenshtein 距离(最优字符串对齐算法):相邻换位计为 1 次编辑，但子字符串不能编辑多次:ed("CA "，" ABC") =3，有 2 次替换(CA- > AC)和 1 次插入(AC- > ABC)。b)真实 Damerau-Levenshtein 距离:相邻的转座算作 1 次编辑，子串可以编辑多次:ed("CA "，" ABC") =2 带 1 次转座(CA- > AC)和 1 次插入 AC- > ABC。后者正在实施中。

使用真实 Damerau-Levenshtein 距离计算编辑距离的代码:可在此处找到:[https://github . com/mammothb/symspellpy/blob/master/symspellpy/edit distance . py](https://github.com/mammothb/symspellpy/blob/master/symspellpy/editdistance.py)

这里，不是使用 2-d DP 来计算编辑距离，而是使用两个大小等于查询长度的 1-d 数组。在空间复杂度方面，这里进行了另一层优化。详细概述可以在这里找到:[https://www . code project . com/Articles/13525/Fast-memory-efficient-Levenshtein-algorithm-2](https://www.codeproject.com/Articles/13525/Fast-memory-efficient-Levenshtein-algorithm-2)

该算法经过优化，具有恒定的时间复杂度。字典中所有可能删除的单词都被散列，并用它们的原始单词列表进行索引，以使搜索更快。该词典是通过使用三个来源创建的:1 .产品的品牌名称，2。产品的显示名称，3。先前搜索的前 n 千个关键字。

> 对于我们的用例，字典中包含大约 350，000 个单词，前缀长度 7 用于候选生成，对于平均长度为 8 个字符的查询，平均花费 5 ms。

感谢阅读。请务必阅读以前的故事，以了解更多信息，并继续关注空间的更多更新。

参考资料:

1.  [https://en . Wikipedia . org/wiki/damer au % E2 % 80% 93 levenshtein _ distance](https://en.wikipedia.org/wiki/Damerau%E2%80%93Levenshtein_distance)
2.  【https://en.wikipedia.org/wiki/Levenshtein_distance 
3.  [https://github.com/mammothb/symspellpy](https://github.com/mammothb/symspellpy)
4.  [https://towards data science . com/symspellcompound-10 EC 8 f 467 c 9 b](/symspellcompound-10ec8f467c9b)
5.  [https://towards data science . com/fast-word-segmentation-for-noise-text-2c 2c 41 F9 E8 da](/fast-word-segmentation-for-noisy-text-2c2c41f9e8da)
6.  [https://towards data science . com/sym spell-vs-bk-tree-100 x-faster-fuzzy-string-search-spell-checking-C4 F10 d 80 a 078](/symspell-vs-bk-tree-100x-faster-fuzzy-string-search-spell-checking-c4f10d80a078)
7.  [https://medium . com/@ wolfgarbe/1000 x-faster-spelling-correction-algorithm-2012-8701 fcd 87 a5f](https://medium.com/@wolfgarbe/1000x-faster-spelling-correction-algorithm-2012-8701fcd87a5f)
8.  [http://norvig.com/ngrams/ch14.pdf](http://norvig.com/ngrams/ch14.pdf)