# 伯特分类器:只是另一个 Pytorch 模型

> 原文：<https://towardsdatascience.com/bert-classifier-just-another-pytorch-model-881b3cf05784?source=collection_archive---------9----------------------->

![](img/f0a56f8d4410608bb0d4e67de6c3940b.png)

Valencia, Spain. Whenever I don’t do projects with image outputs I just use parts of my photo portfolio…. Per usual FRIEND LINK [here](/bert-classifier-just-another-pytorch-model-881b3cf05784?source=friends_link&sk=6600cc7ff8762bff146ebbda1dfab54d)

2018 年底，谷歌发布了 [BERT](https://ai.googleblog.com/2018/11/open-sourcing-bert-state-of-art-pre.html) ，它本质上是一个基于所有维基百科训练的 12 层网络。训练协议很有趣，因为与最近的其他语言模型不同，BERT 被训练为从两个方向考虑语言上下文，而不仅仅是单词左侧的内容。在[的预训练](https://arxiv.org/abs/1810.04805)中，BERT 屏蔽掉给定句子中的随机单词，并使用句子的其余部分来预测丢失的单词。Google 还通过在与其他语言模型规模相当的数据集上训练 BERT 来对其进行基准测试，并显示出更强的性能。

NLP 是一个我有点熟悉的领域，但是看到 NLP 领域拥有它的“ImageNet”时刻是很酷的，该领域的从业者现在可以相当容易地将最先进的模型应用于他们自己的问题。简单回顾一下，ImageNet 是一个大型开源数据集，在其上训练的模型通常可以在 Tensorflow、Pytorch 等库中找到。这些熟练的预训练模型让数据科学家花更多时间解决有趣的问题，而不是重新发明轮子，专注于数据集的监管(尽管数据集监管仍然非常重要)。你现在需要数千而不是数百万的数据集来开始深度学习。

为了工作，我在有限的能力范围内使用过几次 BERT，主要是基于我找到的其他教程。然而，我一直在推迟深入剖析管道并以我更熟悉的方式重建它…在这篇文章中，我只想更好地理解如何以我习惯的方式创建 BERT 管道，以便我可以开始在更复杂的用例中使用 BERT。主要是我对将 BERT 集成到各种网络的多任务集合中感兴趣。

通过这个学习过程，我的希望是展示虽然 BERT 是一个推动 NLP 边界的艺术模型，但它就像任何其他 Pytorch 模型一样，并且通过理解它的不同组件，我们可以使用它来创建其他有趣的东西。我真正想要的是克服我对使用 BERT 的恐惧/恐吓，并像使用其他预训练模型一样自由地使用 BERT。

# 资料组

所以在这篇文章中，我使用了经典的 IMDB 电影评论数据集。这个数据集有 50，000 条电影评论，每条评论都标有“正面”或“负面”的情绪。与我的其他帖子不同，我没有构建自定义数据集，部分原因是我不知道快速构建文本数据集的方法，我不想在这上面花太多时间，这一个很容易在互联网上找到。

总的来说，我同意这不是我能做的最有趣的事情，但是在这篇文章中，我更关注如何使用 BERT 构建一个管道。一旦管道到位，我们就可以根据自己的选择交换出数据集，用于更多样/有趣的任务。

# 分类架构

在这篇文章中，我将使用一个名为[拥抱脸](https://github.com/huggingface/pytorch-pretrained-BERT)的团体的伯特 Pytorch port(很酷的团体，奇怪的名字…让我想到半条命拥抱脸)。通常最好是使用任何内置的网络来避免新移植实现的准确性损失…但谷歌对拥抱脸的移植竖起了大拇指，这很酷。

无论如何…继续…

我要做的第一件事是建立一个模型架构。为此我主要从拥抱脸的例子中拿出一个例子，叫做*。目前，这个类在文档中看起来已经过时了，但是它是如何构建 BERT 分类器的一个很好的例子。基本上，您可以使用 BertModel 类初始化一个 BERT 预训练模型。然后，您可以根据需要添加额外的层作为分类器头。这与创建其他定制 Pytorch 架构的方式相同。*

*像其他 Pytorch 模型一样，您有两个主要部分。首先，您有 init，其中您定义了架构的各个部分，在这种情况下，它是 Bert 模型核心(在这种情况下，它是较小的小写模型，大约 110M 参数和 12 层)，要应用的 dropout，以及分类器层。第二部分是前面的部分，在这里我们定义了如何将架构的各个部分整合到一个完整的管道中。*

```
***class** **BertForSequenceClassification**(nn.Module):

    **def** __init__(self, num_labels=2):
        super(BertForSequenceClassification, self).__init__() self.num_labels = num_labels self.bert = BertModel.from_pretrained('bert-base-uncased') self.dropout = nn.Dropout(config.hidden_dropout_prob) self.classifier = nn.Linear(config.hidden_size, num_labels) nn.init.xavier_normal_(self.classifier.weight) **def** forward(self, input_ids, token_type_ids=**None**, attention_mask=**None**, labels=**None**): _, pooled_output = self.bert(input_ids, token_type_ids, attention_mask, output_all_encoded_layers=**False**) pooled_output = self.dropout(pooled_output) logits = self.classifier(pooled_output)

        **return** logits*
```

*既然已经定义了模型，我们只需要弄清楚如何组织我们的数据，这样我们就可以输入数据并优化权重。在图像的情况下，这通常只是计算出我们需要应用什么样的转换，并确保我们得到正确的格式。对于 BERT，我们需要能够将字符串标记化，并将它们转换成映射到 BERT 词汇表中单词的 id。*

*![](img/85321531e5d48019d353eef02b19ab3c.png)*

*Mendoza, Argentina. Lots of good wine!*

# *BERT 数据预处理*

*使用 BERT 进行数据准备时，我们需要的主要功能是如何标记输入，并将其转换为 BERT 词汇表中相应的 id。拥抱脸为 BertModel 和 BertTokenizer 类增加了非常好的功能，你可以输入你想要使用的模型的名字，在这篇文章中，它是“bert-base-uncased”模型。*

```
*tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')*
```

*要对文本进行标记，您只需调用标记化器类的标记化函数。见下文。*

```
*tokenized_text = tokenizer.tokenize(some_text)*
```

*然后，一旦你把一个字符串转换成一个记号列表，你就必须把它转换成一个匹配 BERT 词汇表中单词的 id 列表。这一次，您只需对之前标记化的文本调用 convert_tokens_to_ids 函数。*

```
*tokenizer.convert_tokens_to_ids(tokenized_text)*
```

*有了这些基础知识，我们就可以组装数据集生成器，它总是像流水线中的无名英雄一样，这样我们就可以避免将整个东西加载到内存中，这是一种痛苦，会使在大型数据集上学习变得不合理。*

# *自定义 BERT 数据集类*

*一般来说，Pytorch 数据集类是基本数据集类的扩展，您可以在其中指定如何获取下一个项目以及该项目的返回内容，在本例中，它是一个长度为 256 的 id 张量和一个热编码目标值。从技术上来说，你可以做长度为 512 的序列，但我需要一个更大的显卡。我目前正在一台配有 11GB GPU RAM 的 GTX 2080ti 上进行训练。在我之前的 1080 卡上，我只能舒服地使用 128 的序列。*

```
*max_seq_length = 256
**class** **text_dataset**(Dataset):
    **def** __init__(self,x_y_list):self.x_y_list = x_y_list

    **def** __getitem__(self,index):

        tokenized_review = tokenizer.tokenize(self.x_y_list[0][index])

        **if** len(tokenized_review) > max_seq_length:
            tokenized_review = tokenized_review[:max_seq_length]

        ids_review  = tokenizer.convert_tokens_to_ids(tokenized_review)

        padding = [0] * (max_seq_length - len(ids_review))

        ids_review += padding

        **assert** len(ids_review) == max_seq_length

        *#print(ids_review)*
        ids_review = torch.tensor(ids_review)

        sentiment = self.x_y_list[1][index] *# color* 
        list_of_labels = [torch.from_numpy(np.array(sentiment))]

        **return** ids_review, list_of_labels[0]

    **def** __len__(self):
        **return** len(self.x_y_list[0])*
```

*因为这是一段相当不错的未加注释的代码…让我们把它分解一下！*

*对于变量 x_y_list 的第一位。这是我在建立数据集时经常做的事情…它基本上只是一个 x 和 y 的列表，不管它们有多少。然后，我对特定的列表进行索引，根据需要检索特定的 x 或 y 元素。*

*如果有人看过我的其他图像管道，我基本上总是有这个，它通常是对应于测试或训练集的图像 URL 列表。在这种情况下，它是训练电影评论文本的测试，第二个元素是这些电影评论文本的标签。*

```
***class** **text_dataset**(Dataset):
    **def** __init__(self,x_y_list):
        self.x_y_list = x_y_list*
```

*所以不要再说了！其中最重要的部分是数据集类如何定义给定样本的预处理。*

1.  *对于这个 BERT 用例，我们在“self.x_y_list[0][index]”检索给定的评论*
2.  *然后如上所述用“tokenizer.tokenize”对该评论进行标记。*
3.  *所有的序列需要长度一致，因此，如果序列长于最大长度 256，它将被截短为 256。*
4.  *然后通过“tokenizer . convert _ tokens _ to _ IDs”将标记化和截断的序列转换成 BERT 词汇表 id*
5.  *在序列短于 256 的情况下，现在用 0 填充直到 256。*
6.  *该评论被转换成 torch 张量。*
7.  *然后，该函数返回评论的张量及其一个热编码的正或负标签。*

```
 ***def** __getitem__(self,index):

        tokenized_review = tokenizer.tokenize(self.x_y_list[0][index])

        **if** len(tokenized_review) > max_seq_length:
            tokenized_review = tokenized_review[:max_seq_length]

        ids_review  = tokenizer.convert_tokens_to_ids(tokenized_review)

        padding = [0] * (max_seq_length - len(ids_review))

        ids_review += padding

        **assert** len(ids_review) == max_seq_length

        *#print(ids_review)*
        ids_review = torch.tensor(ids_review)

        sentiment = self.x_y_list[1][index]
        list_of_labels = [torch.from_numpy(np.array(sentiment))]

        **return** ids_review, list_of_labels[0]

    **def** __len__(self):
        **return** len(self.x_y_list[0])*
```

# *培养*

*在这一点上，培训管道相当标准(现在 BERT 只是另一个 Pytorch 模型)。如果你想检查[笔记本](https://github.com/sugi-chan/custom_bert_pipeline/blob/master/bert_pipeline.ipynb)的第 21 块，我可以使用普通的循环训练。这款笔记本和我的其他笔记本之间唯一真正的区别是风格上的，我在网络本身之外使用了最终分类器层的 softmax。*

```
*outputs = F.softmax(outputs,dim=1)*
```

*最后一个有趣的部分是，我给网络的不同部分分配了特定的学习率。几个月前，当我浏览 fastai 视频时，我对这样做产生了兴趣，并发现它很有用。*

*这个部分做的第一件事是分配两个学习率值，称为 *lrlast* 和 *lrmain。* lrlast 相当标准，为 0.001，而 lrmain 则低得多，为 0.00001。其思想是，当网络的部分被随机初始化，而其他部分已经被训练时，你不需要对预训练的部分应用激进的学习速率而没有破坏速率的风险，然而，新的随机初始化的部分可能不会覆盖，如果它们处于超低的学习速率…因此，对网络的不同部分应用更高或更低的学习速率有助于使每个部分适当地学习。下一部分可以是积极的，而预训练部分可以进行逐步调整。*

*应用这一点的机制出现在字典列表中，您可以在其中指定应用于优化器(在本例中为 Adam 优化器)中网络不同部分的学习率。*

```
*lrlast = .001
lrmain = .00001
optim1 = optim.Adam(
    [
        {"params":model.bert.parameters(),"lr": lrmain},
        {"params":model.classifier.parameters(), "lr": lrlast},       
   ])

optimizer_ft = optim1*
```

*设置好学习速率后，我让它运行 10 个周期，每 3 个周期降低学习速率。网络从一个非常强的点开始…*

```
*Epoch 0/9
----------
train total loss: 0.4340 
train sentiment_acc: 0.8728
val total loss: 0.4089 
val sentiment_acc: 0.8992*
```

*基本上，用 Bert 预先训练的权重初始化网络意味着它已经对语言有了很好的理解。*

```
*Epoch 9/9
----------
train total loss: 0.3629 
train sentiment_acc: 0.9493
val total loss: 0.3953 
val sentiment_acc: 0.9160*
```

*在这个过程结束时，精确度提高了几个点，损失略有减少…我还没有真正看到模型通常如何在这个数据集上得分，但我认为这是合理的，现在足以表明网络正在进行一些学习。*

*在我的新 2080ti 卡上，这个数据集上的 10 个时代花了 243m 48s 才完成。顺便提一下，让卡和 Pytorch 一起工作有很多麻烦…主要是更新各种版本的东西。*

*![](img/1f066760e55d502d681f3f820869e0f8.png)*

*Buenos Aires Metropolitan Cathedral*

# *结束语*

*对我来说，这样做很重要，可以向自己表明，虽然 BERT 是最先进的，但当我试图将它应用于我自己的问题时，我不应该被吓倒。由于人们付出了大量努力将 BERT 移植到 Pytorch 上，以至于谷歌对其性能表示赞赏，这意味着 BERT 现在只是数据科学家在 NLP 盒子中的另一个工具，就像 Inception 或 Resnet 对于计算机视觉一样。*

*就性能而言，我认为我可以通过在最终分类器之前添加额外的层来挤出几个额外的百分点。这将允许在这个特定的任务中有更多的层。我也不太精通如何在 NLP 领域进行数据扩充，所以这将是其他需要检查的东西，可能使用其他经过训练的语言模型来生成合成文本...但是现在我有了一个 BERT 管道，并且知道我可以像对任何其他模型那样在其上构建自定义分类器…谁知道呢…这里有很多令人兴奋的可能性。*

> *像往常一样，你可以在这里随意查看笔记本[。为了简单起见，数据集也在 repo 中，所以如果你安装了 pytorch 和 pytorch-pretrained-bert 库，你应该可以使用了。](https://github.com/sugi-chan/custom_bert_pipeline)*