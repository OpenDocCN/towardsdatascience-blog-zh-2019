# 数据科学能帮你挑宝宝名字吗？

> 原文：<https://towardsdatascience.com/can-data-science-help-you-pick-a-baby-name-b7e98a98268e?source=collection_archive---------23----------------------->

## 探索各种数据科学技术如何用于缩小婴儿名字的想法

![](img/a72cf893b4a1670deed2dacaf00edb64.png)

Source: Christopher Doughty

为你的宝宝找到一个完美的名字可能是一个挑战。很难找到一个不仅你和你的伴侣都喜欢的名字，而且还符合你的姓氏。此外，名字的受欢迎程度也很复杂，比如，你选择的是普通的还是独特的名字。当试图涉水通过所有这些方面时，数据科学能使名字选择过程更容易吗？

为了研究这个话题，我收集了一些关于婴儿名字的数据。名和姓的数据来自国家统计局和苏格兰国家记录[1]。我的最终数据集包含 31，697 个名和 6，453 个姓。第一个名字数据是根据 2000 年至 2017 年出生的婴儿数量按年份划分的。

**TLDR:** [为这个项目创建的在线应用程序可以在这里找到。](https://penguinstrikes.github.io/content/baby_names/index.html)

# 发音预测

人们倾向于避免名字和姓氏押韵。节奏一致的名和姓在你大声说出来时感觉不自然，例如汉克·班克斯。影响名字发音的另一个因素是名字和姓氏以同一个音结尾，例如 Emilie Kelly。在许多情况下，一种算法不能单独使用字母来识别一个单词的结尾音；它需要知道发音。

为了根据相似的节奏给单词打分，我需要获得如何发音的训练数据。这些数据来自 CMU 发音词典[2]，经过清理，该词典包含了 117，414 个独特的单词。CMU 词典包含单词及其阿帕贝特语音组成。对于字典中没有的单词，我预测了它们的发音。这可以使用 **seq2seq** 模型来实现，也称为**编码器-解码器**模型。

我们现有字典的预处理需要标记一组独特的字母表和 ARPAbet 码。这是使用标签编码器实现的，并创建了一组 81 个独特的令牌。下面是一个 LabelEncoder 将字符转换成数字的例子，使用的是我的名字:

```
Source: ['C', 'H', 'R', 'I', 'S', 'T', 'O', 'P', 'H', 'E', 'R']
Target: ['K', 'R', 'IH1', 'S', 'T', 'AH0', 'F', 'ER0']#### TOKENISING ####Source: [21 37 62 39 63 65 53 60 37 25 62]
Target: [48 62 41 63 65  8 35 29]
```

评估输入的 CMU 数据，我发现最长的源向量是 34 个元素，最长的目标向量是 32 个元素。这些最大值似乎是合理的，因此使用填充元素 **' < PAD > '** 来标准化源向量和目标向量的长度。

seq2seq 模型是使用两个 LSTM(长短期记忆)网络用 Keras 构建的。该模型训练了 35 个历元，最终准确率为 95.6%，损失为 0.147。用一些原始字典中没有的名字示例来测试最终模型产生了令人满意的结果。

```
LEXI        L EH1 K S IY
AYLA        EY1 L AH0
MYA         M IY1 AH0
```

然后将该模型应用于名和姓数据[1]。它被用来计算 31697 个名字中的 25036 个和 6452 个姓氏中的 739 个的发音。

# 押韵相似性

拥有所有名字和姓氏的语音意味着我现在可以对名字之间的相似性进行评分。我通过计算每个单词之间的距离来达到这个目的。此方法描述将一个字符串转换为另一个字符串所需的最少插入、删除或替换次数。

Levenshtein 距离越大，两个字符串之间的相似性越小，例如，距离为 0 意味着两个字符串相同。下面列出了创建使两个字符串相同所需的更改矩阵的代码。

```
# Get the length of each string
lenx = len(str1) + 1
leny = len(str2) + 1# Create a matrix of zeros
matrx = np.zeros((lenx, leny))# Index the first row and column
matrx[:,0] = [x for x in range(lenx)]
matrx[0] = [y for y in range(leny)]# Loop through each value in the matrix
for x in range(1, lenx):
    for y in range(1, leny):
        # If the two string characters are the same
        if str1[x-1] == str2[y-1]:
            matrx[x,y] = matrx[x-1,y-1]
        else:
            matrx[x,y] = min(matrx[x-1, y]+1, matrx[x-1,y-1]+1, matrx[x,y-1]+1)
```

测试代码与单词 CONTAIN 和 TODAY 的相似性，产生以下输出矩阵。使用下面的等式可以使用操作的数量来计算相似性得分:*(1-(操作/最大字符串长度)* 100)。*该技术计算出 CONTAIN 与 OBTAIN 的相似度为 71%，而 TODAY 的相似度仅为 17%。

![](img/de4b9bc3d40c1bfefa967691f57c13db.png)

Levenshtein distance calculated between different words. The similarity between the words is higher on the left (two alterations) compared to the right (five alterations)

这在我们的数据中的使用方式略有不同。这种方法不是直接应用于姓氏并根据名字的相似性来评分，而是应用于语音来评分与姓氏具有相似发音或节奏的名字。例如，如果我们使用姓 SMITH，并搜索发音最相似和最不相似的男孩的名字，分析将生成以下输出:

```
# TOP SCORING SIMILARITY
SETH    75.0
KIT     62.5
SZYMON  50.0# LEAST SCORING SIMILARITY
ZACHARY 8.33
HUSSAIN 9.09
PARKER  10.0
```

# 流行趋势

在这一点上，我们有了一个比较语音相似性的工作方法，去除发音相似的成分，并对名字和姓氏进行评分。我们希望考虑数据中名字的当前趋势。我们会选择一个受欢迎的、受欢迎程度增加的、受欢迎程度减少的、稳定的或在数据中相当罕见的名字吗？

我拥有的数据让我可以根据 2000 年至 2017 年的同比变化，查看每个名字的趋势变化。下图显示了 2012 年至 2017 年间英国前 10 名男孩和女孩名字的变化。这表明有大量的运动。

![](img/eb1eb842292f4e34adadf32546d2575f.png)![](img/a33b0125b1f6e549b38979db2ab624e9.png)

Top 10 boys and girls names between the years 2012 and 2017; lines show rank changes within the top 10

为了给每个名字贴上“受欢迎程度”的标签，我对每个名字的 4 年数据进行了线性回归。将每个部分的回归系数合并，以创建一个趋势数组。看起来像这样的数组[+，+，+，+，+，+]表示名字在随时间增长，像这样的数组[-，-，-，-，-]表示名字在衰落。这允许为每个名称创建与其阵列相关的配置文件。下面是我整理的七个简介:

```
'CONTINUED DECLINE'    :   Most recently declining, +1 previous 
'DECLINE STABILISING'  :   Most recently stable, +2 decline
'RECENT DECLINE'       :   Most recently declining, 0 previous
'CURRENTLY STABLE'     :   Most recently stable, +1 previous
'GROWTH STABILISING'   :   Most recently stable, +2 growth
'RECENT BOOM'          :   Most recently growing, 0 previous
'GROWING BOOM'         :   Most recently growing, +1 previous
```

为了将简介与名字匹配，我绘制了 2017 年前 100 名女性婴儿的名字，并根据他们的简介给他们涂上颜色(如下图)。

![](img/d2699ac85430391f510f8310a85e0368.png)

Top 100 ranked female names of 2017 and their historical profiles

从这个数字中一个有趣的发现是，前 100 名中的许多名字都在下降。这可以通过观察每年独特名字的数量来解释(下图)，因为在 2000 年至 2012 年间，名字选择的多样性增加了。

![](img/b9a4cd419ddcf8f1dfe48fa2c085a7e0.png)

Count of unique male and female baby names by year

# 一些更有趣的发现…

对数据的深入研究揭示了一些更有趣的见解。其中之一是基于名字长度的婴儿频率计数。它显示大多数名字的长度在四到八个字母之间。

![](img/d47b7e94c42f6217b8a294013db8b9bb.png)

Frequency plot of the length of first names

另一个是在名字中使用连字符。我们可以观察到，在过去的 17 年里，这变得越来越流行。

![](img/94dec872a48918d9179df62ef6aa12c2.png)

Number of babies with hyphenated first names by year

# 最终想法和应用

当将发音相似性模型与姓氏和流行度数据相结合时，输出是将数千个名字减少到数百个的非常强大的方法。这是此类工具的主要好处，它减少了因过大而无法通读的数据集，并提供了对特定名称的历史和地理背景的了解。

我已经将评分模型和趋势概要数据推送到 GitHub，供任何有兴趣测试模型和趋势数据的轻量级版本的人使用:[https://penguinstrikes . GitHub . io/content/baby _ names/index . html](https://penguinstrikes.github.io/content/baby_names/index.html)。

![](img/a8f0f32cc96a826a12c573c43463c9d7.png)

Source: Christopher Doughty

总之，像这样一个问题的模型不可能，也不应该提供一个“完美”的名字。像这样的任务涉及太多个人属性，无法提供一个明确的解决方案。然而，在我看来，与一些帮助准父母挑选婴儿名字的网站和文献相比，数据驱动的方法要有用得多。

**参考文献**

[1]根据[开放政府许可 v3.0](http://www.nationalarchives.gov.uk/doc/open-government-licence/version/3/) 提供的国家统计局数据

[2][http://SVN . code . SF . net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b](http://svn.code.sf.net/p/cmusphinx/code/trunk/cmudict/cmudict-0.7b)