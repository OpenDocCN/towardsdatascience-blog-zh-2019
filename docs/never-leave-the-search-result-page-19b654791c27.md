# 使用深度学习预测下一个搜索关键词

> 原文：<https://towardsdatascience.com/never-leave-the-search-result-page-19b654791c27?source=collection_archive---------17----------------------->

# 第一部分:盗梦空间

下一个单词预测或语言建模的任务是预测下一个单词是什么。当你写短信或电子邮件时，你可能每天都在使用它而没有意识到这一点。电子商务，尤其是基于杂货的电子商务，可以广泛受益于这些功能。

![](img/aab49b321c2bdb0028d9c348da2205b4.png)![](img/afa22bd6597ac33338141fe0b0228220.png)

Examples of next word prediction

**搜索**在电子商务购物中起着举足轻重的作用。特别是对于杂货购物，大多数电子商务网站报告 40-50%的转化率来自搜索结果页面。因此，对于任何杂货电子商务网站来说，在搜索功能方面保持领先是至关重要的。

在寻找食品杂货时，我们发现自己每次都在寻找或多或少相同的东西。任何一般的家庭购物清单都会有一套日常用品，也会有一个购物模式。此外，搜索特定食谱的项目，我们会发现一个购物模式。

因此，对于客户搜索的每个会话，将形成一种以搜索词为节点的 [DAG](https://en.wikipedia.org/wiki/Directed_acyclic_graph) ，从一个搜索词到另一个搜索词的过渡是边。举个例子，

```
milk → bread → butter → cheese → eggs → chicken
```

> 所有这些使我们能够根据会话中之前的搜索来预测客户接下来会搜索什么，就像电子邮件或手机键盘的情况一样。

![](img/347e2bcf95b460793bb3c5f781139b8e.png)

## 赞成的意见

1.  我们的搜索将变得**客户友好**和**互动**，因为客户不可能键入下一个搜索词。
2.  我们可以“引导”我们的客户选择高性能的搜索词，而不是长尾搜索词、不正确的拼写等等。
3.  我们的百分比转换从搜索结果页面相比，总转换可能会增加。
4.  客户可能会添加比他们预期更多的产品来代替建议。

## 面向连接的网络服务(Connection Oriented Network Service)

1.  我们的下一个搜索词预测可能会限制客户可能搜索的产品范围。例如，一个全新的项目出现在我们的网站上，但在搜索时不会出现在现有的搜索列表中。由于我们的领域是杂货，产品的数量及其各自的上下文是有限的，我们可以忽略这个问题(根据您的目录可能不存在)。此外，如果该项目和搜索词变得足够流行，它会在下一次迭代中被添加到我们的集合中。
2.  预测不受欢迎的或/和重复的术语可能会妨碍客户体验。就像我们在键盘和电子邮件中看到的这个特性一样，我们通过一个非侵入式的 UI 来解决这个问题，这是一个更好的辅助特性。

## 对自然语言处理的需求

由于我们有一个可接受术语的大字典，运行日常分析是不可能的，因为所有数据组合都不可用，并且所有组合都会产生一个相当大的数字(对于 1000 个术语和预测第 5 个术语，我们需要计算 10 个 Dag)。我们需要根据前一个序列的上下文来预测下一个搜索词，这是我们在运行纯分析时无法捕捉到的。因此，为了预测下一个关键字，我们需要探索 NLP 技术。

# 第 2 部分:设计和实现

## 客户行动流程和高级设计

![](img/501631f4453cc43fdca48ba2d8974fed.png)

Customer action flow diagram

![](img/26632ff60a83178df2a05636c76c1d95.png)

所以我们的预测有两个动作。

一个是客户没有为搜索词添加任何项目，我们预测一些与他/她可能正在寻找的东西相关的搜索词。例如，假设客户搜索牛奶，但他可能正在寻找蓝色牛奶，这可能不会出现在我们的 SRP(搜索结果页面)的顶部。因此，我们给我们的客户更具体的搜索条件。

第二个例子是顾客在他的购物车中添加了一些牛奶商品，并寻找更多的东西，如面包、奶酪、鸡蛋等。这里，我们根据会话中之前的搜索给出下一个搜索词的预测。

为了预测下一个可能的搜索关键字:

1.  前端服务传递最后 N 个搜索关键字。
2.  后端服务验证所有 N 个关键字都存在于模型的关键字字典中。
3.  对于字典中不存在的关键词，后端调用相似词模型，得到与前端传递的词相似且存在于字典中的词。
4.  后端将修改后的最后 N 个术语传递给下一个术语模型，得到结果并将其提供给前端。

## 下一期模型的详细设计

为了找到和设计下一个术语模型，我们称之为 NLSRP 模型，(模型名称工作进行中，欢迎任何建议:) )我们遵循:

![](img/e4f5740f5cf69d3f805195622ced5101.png)

## 收集原始数据和预处理

为了建模，我们需要数据。在我们的例子中，它应该是具有连续术语的会话的形式，导致添加到 cart，dag 的列表，类似于:

```
milk → bread → eggs → ...
```

这里，

1.  我们需要过滤掉不在我们的首选列表中的术语。
2.  我们需要短时间内的连续术语(我们花了 20 分钟)。

如果您有一个包含会话 id、时间、添加到 SRP 上的购物车的术语的表，我们可以运行一个 spark 作业来获取所需的数据。

```
mainDataset = mainDataset.groupBy(
*col*(**"session_id"**),
*window*(*col*(**"datetime"**), **"20 minutes"**))
.agg(*collect_set*(
*col*(**"term"**))
.as(**"termlist"**));
```

## 数据建模

我们的第一个挑战来了，如何对我们的搜索词进行建模，以将它们提供给学习模型。

![](img/b07c05aee3d7cdba08ee7cc109bb5e04.png)

为我们的搜索词生成 [*word2vec*](/introduction-to-word-embedding-and-word2vec-652d0c2060fa) 嵌入，用于 [CBOW 模型](http://mccormickml.com/assets/word2vec/Alex_Minnaar_Word2Vec_Tutorial_Part_II_The_Continuous_Bag-of-Words_Model.pdf)(给定相邻词给出当前词)，以及 [skip-gram 模型](http://mccormickml.com/2016/04/19/word2vec-tutorial-the-skip-gram-model/)(给定前一个词预测下一个词)。word2vec 的超参数:

**大小:** [每个搜索项嵌入的大小](https://datascience.stackexchange.com/questions/31109/ratio-between-embedded-vector-dimensions-and-vocabulary-size)。100–300 是推荐的大小。由于我们的 vocab 大小限制在 100 个或最多 1000-2000 个搜索词，我们可以放心地认为是 100 个。

**窗口:**在我们的用例中，7 似乎是一个理想的值，因为我们序列的最大长度是 15。

**hs:** 不需要分层 softmax，因为我们的 vocab 大小很小。(它是 softmax 的近似值)

**iter:** 经过 25+1 次[迭代后](https://stats.stackexchange.com/questions/215637/word2vec-vector-quality-vs-number-of-training-iterations)我们在模型中看不到任何变化。我们可以使用 5 为耐心的 EarlyStopping 回调来获得迭代次数。

我们使用 gensim 的 cython 优化实现来获得我们的定制嵌入。

我们得到这样的结果:

```
[('semi skimmed milk', 0.652964174747467),
 ('cravendale', 0.4939464330673218),
 ('skimmed milk', 0.35889583826065063),
 ('long life milk', 0.3340461850166321),
 ('sandwich fillers', 0.321510910987854),
 ('sandwich meat', 0.3212273120880127),
 ('flavoured water', 0.30973631143569946),
 ('cheese spread', 0.3049270212650299),
 ('crisp', 0.3014737665653229),
 ('cereal bars', 0.3007051646709442)]
```

我们可以看到类似的术语出现，只是通过学习(应用 word 2 vec)Dag。这也证实了我们关于顾客购物模式的假设。

从我们的 Dag 列表中，我们创建输入，输出对应于固定的大小。我们将固定窗口 N 取为 5。因此，我们传递 4 个搜索项作为输入，得到序列中的第 5 个项作为输出。

**基线我们的期望**

Word2Vec 在输出层训练的是下一个单词的概率。因此，线性神经网络的输出层用于获得单词嵌入，从而为我们提供未来的单词预测。

Gensim 提供了一个 API 来实现这一点

我们得到的输出如下:

```
[('fruits', 0.0055794665),
 ('pork', 0.0042083673),
 ('potato', 0.0032076887),
 ('choc waffle', 0.0028934616),
 ('drink', 0.0026189752),
 ('chocolate pudding', 0.0023179431),
 ('rice', 0.002288311),
 ('yogurt', 0.0022665549),
 ('breakfast. bars', 0.0021832779),
 ('freddio', 0.0021441795),
 ('flavour water', 0.0021338842),
 ('beef', 0.0021277776),
 ('dog', 0.0021113018),
 ('angel cajes', 0.002109515),
 ('yogurts', 0.0021087618),
 ('apple', 0.0020760428),
 ('figita', 0.002075238),
 ('schwepp', 0.00204034),
 ('breakfast', 0.0020321317)]
```

我们以此为基础对 CBOW 和 skip gram 进行了基线预测，得到了一个相当好的模型。

## 更加努力

我们的任务是通过查看之前的关键字来预测每个搜索关键字，而 [RNNs](/recurrent-neural-networks-and-lstm-4b601dd822a5) 可以保持一个隐藏状态，可以将信息从一个时间步转移到下一个时间步。

![](img/d6856fd755a43a2b058e51534a9df024.png)

Sample RNN example

我们尝试用 [RNNs](/building-a-next-word-predictor-in-tensorflow-e7e681d4f03f) 实现 [seq2seq](https://medium.com/@curiousily/making-a-predictive-keyboard-using-recurrent-neural-networks-tensorflow-for-hackers-part-v-3f238d824218) 模型。我们在 keras 和 tensorflow 后端的帮助下做到了这一点。

我们在 Keras 设置了一个多层 LSTM/GRU，每层有 N 个隐藏单元，共 2 层。RNN 的输入是最后 4 个搜索项，目标是下一个搜索项。

该模型可以根据复杂程度在 LSTM 或 GRU 进行训练。它可以有隐藏的细胞。这里你需要使用 RNN 类、[批量](https://datascience.stackexchange.com/questions/12532/does-batch-size-in-keras-have-any-effects-in-results-quality)、隐藏层输入、嵌入矩阵类型(cbow 或 skip gram)、退出、层(深度模型为 2)来获得最佳模型。

回调将在每个时代和我们引入的自定义记录器之后保存每个模型。

这个模型将产生比 word2vec 的输出层更好的模型。

**在生产中部署**

我们可以把 keras 模型转换成 tensorflow，用 tensorflow-serving 的 docker 来服务它。这将把. hd5 模型保存到. pb 文件中，该文件通过 tensorflow serving 和 docker 部署。

## 相似性搜索

每个电子商务网站都有一个基本的东西，叫做目录。该目录分为类别、子类别和进一步的子类。对于 SRP 上点击的每个项目，我们将获得每个搜索项的类别级别、子类别级别和其他子类级别的贡献。

该贡献作为一个整体将成为搜索项意图树。意图树将是类和它们各自的值(贡献)的向量。我们可以通过许多技术找到给定向量和向量列表之间的 [L2 距离](https://en.wikipedia.org/wiki/Euclidean_distance)相似性，即给定的搜索项和可接受的搜索项列表。如果你的列表很小，你可以穷举迭代。另外，我们可以使用脸书的 [*FAISS*](https://github.com/facebookresearch/faiss) 来寻找 Voronoi 单元中的相似向量。

我以此结束这篇文章，如果你呆了这么久，请评论你的疑问和建议:)。